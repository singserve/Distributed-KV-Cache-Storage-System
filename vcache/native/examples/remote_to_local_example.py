"""Example: cross-process remote->local GPU transfer using cuda IPC handle.

Producer process:
 - allocates a small tensor on src GPU
 - fills it with known data
 - calls cudaIpcGetMemHandle and sends the 64-byte handle to the main process
 - waits for an acknowledgement before freeing memory and exiting

Main process (consumer):
 - receives IPC handle bytes and metadata
 - allocates a destination tensor on dst GPU
 - invokes DistributedNVLINKTransferEngine.transfer_sync(...) with source_address=0 and ipc_handle=handle_bytes
 - validates that the transferred data matches the expected values

Requirements:
 - Linux, >1 GPUs with peer access (NVLINK/peer enabled)
 - Build and place libnvlink_transfer.so in LD_LIBRARY_PATH or the current working directory

Run:
  python vcache/native/examples/remote_to_local_example.py
"""

import ctypes
import os
import time
from multiprocessing import Process, Pipe

import torch

from lmcache.vcache.nvlink_transfer_engine import DistributedNVLINKTransferEngine


LIB_CUDA_NAMES = ["libcudart.so.12", "libcudart.so.11", "libcudart.so.10", "libcudart.so"]
IPC_HANDLE_SIZE = 64  # typical size for cudaIpcMemHandle_t


def load_libcudart():
    for name in LIB_CUDA_NAMES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError("Could not load libcudart (tried {})".format(LIB_CUDA_NAMES))


def producer(conn, src_gpu: int, n: int):
    """Producer runs in a separate process and exports a cudaIpcMemHandle."""
    try:
        torch.cuda.set_device(src_gpu)

        # Allocate a large buffer and write known data at an offset
        total_size = 1024 * 1024  # 1MB buffer
        t = torch.zeros(total_size, dtype=torch.uint8, device=f'cuda:{src_gpu}')

        # Write a pattern at offset
        offset = 64 * 1024  # 64KB offset
        pattern = torch.arange(256, dtype=torch.uint8, device=f'cuda:{src_gpu}')
        t[offset:offset+pattern.numel()] = pattern

        expected = pattern.cpu().numpy().tobytes()
        size_bytes = pattern.numel() * pattern.element_size()
        dev_ptr = t.data_ptr()

        # Get IPC handle for the base buffer
        libcudart = load_libcudart()
        cudaIpcGetMemHandle = libcudart.cudaIpcGetMemHandle
        cudaIpcGetMemHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cudaIpcGetMemHandle.restype = ctypes.c_int

        handle_buf = ctypes.create_string_buffer(IPC_HANDLE_SIZE)
        rc = cudaIpcGetMemHandle(ctypes.cast(handle_buf, ctypes.c_void_p), ctypes.c_void_p(dev_ptr))
        if rc != 0:
            conn.send({'error': f'cudaIpcGetMemHandle failed with code {rc}'})
            conn.close()
            return

        handle_bytes = handle_buf.raw

        # Send metadata (handle bytes, size, src_gpu, expected data, offset) to consumer
        conn.send({'handle': handle_bytes, 'size': size_bytes, 'src_gpu': src_gpu, 'expected': expected, 'offset': offset})
        # Wait for confirmation before exiting (so buffer stays valid)
        msg = conn.recv()
        if msg == 'done':
            conn.close()
            # Optionally sleep briefly to ensure consumer has time to finish
            time.sleep(0.1)
            return
        else:
            conn.close()
            return

    except Exception as e:
        conn.send({'error': str(e)})
        conn.close()
        return


class DummyConfig:
    def get_extra_config_value(self, k, default):
        return default


def consumer_and_verify(conn, dst_gpu: int):
    msg = conn.recv()
    if 'error' in msg:
        raise RuntimeError(f"Producer error: {msg['error']}")

    handle_bytes = msg['handle']
    size_bytes = msg['size']
    src_gpu = msg['src_gpu']
    expected = msg['expected']

    # Create transfer engine on destination GPU
    engine = DistributedNVLINKTransferEngine(DummyConfig(), gpu_id=dst_gpu)

    # Allocate destination tensor
    torch.cuda.set_device(dst_gpu)
    dst = torch.empty(size_bytes, dtype=torch.uint8, device=f'cuda:{dst_gpu}')
    dst_ptr = dst.data_ptr()

    # Perform synchronous transfer from offset: provide source ipc_handle and src_offset
    src_offset = msg.get('offset', 0)
    success = engine.transfer_sync(
        source_gpu=src_gpu,
        target_gpu=dst_gpu,
        source_address=0,
        target_address=dst_ptr,
        size=size_bytes,
        ipc_handle=handle_bytes,
        src_offset=src_offset,
        dst_offset=0
    )

    if not success:
        # Signal producer to finish and then raise
        try:
            conn.send('done')
        except Exception:
            pass
        raise RuntimeError('transfer_sync failed')

    # Verify content
    out = dst.cpu().numpy().tobytes()
    ok = out == expected

    # Signal producer to finish and return result
    try:
        conn.send('done')
    except Exception:
        pass

    return 'ok' if ok else 'mismatch'


if __name__ == '__main__':
    # Basic parameters
    src_gpu = 0
    dst_gpu = 1
    n = 1024  # bytes

    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for this example")
        raise SystemExit(1)

    parent_conn, child_conn = Pipe()

    p = Process(target=producer, args=(child_conn, src_gpu, n), daemon=True)
    p.start()

    try:
        result = consumer_and_verify(parent_conn, dst_gpu)
        print("Transfer result:", {'result': result})
    finally:
        # Ensure producer exits
        if p.is_alive():
            try:
                parent_conn.send('done')
            except Exception:
                pass
        p.join(timeout=5)

    print("Example finished")
