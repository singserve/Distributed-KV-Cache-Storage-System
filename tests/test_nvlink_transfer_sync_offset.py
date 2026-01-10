#!/usr/bin/env python3
"""Test nvlink transfer engine's transfer_sync function with offset between two processes using different GPUs."""

import ctypes
import sys
import time
import multiprocessing as mp

import torch

from lmcache.vcache.nvlink_transfer_engine import DistributedNVLINKTransferEngine

IPC_HANDLE_SIZE = 64


def producer_process(conn, src_gpu: int, total_size: int, offset: int, pattern_len: int):
    """Producer process: allocate memory on source GPU and create IPC handle with offset."""
    try:
        torch.cuda.set_device(src_gpu)

        # Allocate base buffer and write pattern at offset
        t = torch.zeros(total_size, dtype=torch.uint8, device=f'cuda:{src_gpu}')
        pattern = torch.arange(pattern_len, dtype=torch.uint8, device=f'cuda:{src_gpu}')
        t[offset:offset + pattern.numel()] = pattern

        expected = pattern.cpu().numpy().tobytes()
        size_bytes = pattern.numel() * pattern.element_size()
        dev_ptr = t.data_ptr()

        # Get IPC handle
        libcudart = ctypes.CDLL("libcudart.so")
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

        # Send metadata
        conn.send({
            'handle': handle_bytes,
            'size': size_bytes,
            'src_gpu': src_gpu,
            'expected': expected,
            'offset': offset,
            'src_address': dev_ptr,
            'total_size': total_size,
        })

        # Wait for done signal before exiting
        msg = conn.recv()
        if msg == 'done':
            conn.close()
            return
        else:
            conn.close()
            return

    except Exception as e:
        try:
            conn.send({'error': str(e)})
        except Exception:
            pass
        conn.close()
        return


def consumer_process(res_conn, handle_bytes, size_bytes, expected, src_gpu, src_offset, dst_gpu):
    """Consumer process: use IPC handle and call transfer_sync with offset."""
    try:
        # Create a simple config object
        class SimpleConfig:
            def __init__(self, gpu_id):
                self.gpu_id = gpu_id
            
            def get_extra_config_value(self, key, default):
                if key == "max_concurrent_transfers":
                    return 2
                elif key == "transfer_timeout_sec":
                    return 30.0
                elif key == "gpu_id":
                    return self.gpu_id
                elif key == "local_hostname_TE":
                    return "localhost"
                elif key == "metadata_server":
                    return "http://localhost:8080/metadata"
                elif key == "protocol_TE":
                    return "nvlink"
                elif key == "device_name":
                    return ""
                return default

        # Create transfer engine on destination GPU
        config = SimpleConfig(gpu_id=dst_gpu)
        engine = DistributedNVLINKTransferEngine(config, gpu_id=dst_gpu)

        if not engine.cuda_available:
            res_conn.send({'status': 'failed', 'reason': 'CUDA not available in transfer engine'})
            res_conn.close()
            return

        print(f"  [Consumer] NVLINK available: {engine.nvlink_available}")
        print(f"  [Consumer] Transfer engine initialized on GPU {dst_gpu}")

        # Allocate memory on destination GPU
        torch.cuda.set_device(dst_gpu)
        dst = torch.empty(size_bytes, dtype=torch.uint8, device=f'cuda:{dst_gpu}')
        dst_ptr = dst.data_ptr()
        print(f"  [Consumer] Allocated {size_bytes} bytes on GPU {dst_gpu}")
        print(f"  [Consumer] Target address: 0x{dst_ptr:X}")

        # Test transfer_sync with IPC handle and offset
        print(f"  [Consumer] Calling transfer_sync with IPC handle and offset {src_offset}...")
        start_time = time.time()

        success = engine.transfer_sync(
            source_gpu=src_gpu,
            target_gpu=dst_gpu,
            source_address=0,  # Will be calculated from handle + offset
            target_address=dst_ptr,
            size=size_bytes,
            ipc_handle=handle_bytes,
            src_offset=src_offset,
            dst_offset=0,
        )

        transfer_time = time.time() - start_time

        if not success:
            res_conn.send({'status': 'failed', 'reason': 'transfer_sync returned False'})
            res_conn.close()
            return

        print(f"  [Consumer] transfer_sync completed in {transfer_time:.3f}s")

        # Verify data on destination GPU
        out = dst.cpu().numpy().tobytes()
        if out != expected:
            # Try to compare values
            dst_values = dst.cpu().numpy()
            expected_values = torch.frombuffer(torch.tensor(list(expected), dtype=torch.uint8), 
                                              dtype=torch.uint8).numpy()
            
            mismatch_count = 0
            for i in range(min(10, len(dst_values))):
                if dst_values[i] != expected_values[i]:
                    mismatch_count += 1
                    print(f"    Mismatch at index {i}: dst={dst_values[i]}, expected={expected_values[i]}")
            
            res_conn.send({'status': 'failed', 'reason': f'data mismatch ({mismatch_count} mismatches in first 10 elements)'})
            res_conn.close()
            return

        res_conn.send({'status': 'ok', 'transfer_time': transfer_time})
        res_conn.close()
        return

    except Exception as e:
        try:
            res_conn.send({'status': 'failed', 'reason': f'exception: {e}'})
        except Exception:
            pass
        res_conn.close()
        return


def main():
    # Fixed parameters
    src_gpu_id = 0
    dst_gpu_id = 1
    total_size = 1024 * 1024  # 1MB
    offset = 64 * 1024  # 64KB
    pattern_len = 1024  # 1KB

    print("=" * 60)
    print("Testing nvlink transfer engine's transfer_sync function with offset")
    print("=" * 60)
    print(f"Source GPU: {src_gpu_id}")
    print(f"Destination GPU: {dst_gpu_id}")
    print(f"Total buffer size: {total_size} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"Offset: {offset} bytes ({offset/1024:.2f} KB)")
    print(f"Transfer size: {pattern_len} bytes ({pattern_len/1024:.2f} KB)")

    # Check environment
    if not torch.cuda.is_available():
        print("❌ CUDA not available", file=sys.stderr)
        return 2

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"❌ Only {gpu_count} GPU(s) available, need at least 2", file=sys.stderr)
        return 2

    print(f"✓ CUDA available with {gpu_count} GPU(s)")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")

    # Check native library
    try:
        ctypes.CDLL("libnvlink_transfer.so")
        print(f"✓ Native library found")
    except OSError:
        print("❌ libnvlink_transfer.so not found; build it before running this test", file=sys.stderr)
        return 3

    # Create producer process
    ctx = mp.get_context('spawn')
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(target=producer_process, args=(child_conn, src_gpu_id, total_size, offset, pattern_len), daemon=True)
    p.start()

    try:
        # Receive metadata from producer
        print(f"\n[Main] Waiting for producer (GPU {src_gpu_id}) to create IPC handle...")
        msg = parent_conn.recv()
        if 'error' in msg:
            print(f"❌ Producer error: {msg['error']}", file=sys.stderr)
            return 1

        handle_bytes = msg['handle']
        size_bytes = msg['size']
        expected = msg['expected']
        src_gpu = msg['src_gpu']
        src_offset = msg['offset']
        src_address = msg['src_address']
        total_size_received = msg['total_size']

        print(f"✓ Received IPC handle: {len(handle_bytes)} bytes")
        print(f"  Source GPU: {src_gpu}")
        print(f"  Source address: 0x{src_address:X}")
        print(f"  Total buffer size: {total_size_received} bytes ({total_size_received/1024/1024:.2f} MB)")
        print(f"  Transfer size: {size_bytes} bytes ({size_bytes/1024:.2f} KB)")
        print(f"  Offset: {src_offset} bytes ({src_offset/1024:.2f} KB)")

        # Create consumer process
        print(f"\n[Main] Starting consumer (GPU {dst_gpu_id})...")
        cctx = mp.get_context('spawn')
        cres_parent, cres_child = cctx.Pipe()
        consumer_p = cctx.Process(target=consumer_process, 
                                 args=(cres_child, handle_bytes, size_bytes, expected, 
                                       src_gpu, src_offset, dst_gpu_id),
                                 daemon=True)
        consumer_p.start()

        # Wait for consumer result with longer timeout
        try:
            if cres_parent.poll(timeout=60):  # Increased timeout to 60 seconds
                cres_msg = cres_parent.recv()
            else:
                print(f"❌ Consumer timeout after 60 seconds")
                cres_msg = {'status': 'failed', 'reason': 'consumer_timeout'}
        except Exception as e:
            print(f"❌ Pipe error: {e}")
            cres_msg = {'status': 'failed', 'reason': f'pipe_error: {e}'}

        # Clean up consumer process
        consumer_p.join(timeout=10)
        if consumer_p.is_alive():
            print(f"❌ Consumer process still alive, terminating...")
            consumer_p.terminate()
            consumer_p.join(timeout=5)

        if cres_msg.get('status') != 'ok':
            print(f"❌ Consumer failed: {cres_msg.get('reason', 'unknown error')}", file=sys.stderr)
            parent_conn.send('done')
            return 1

        transfer_time = cres_msg.get('transfer_time', 0)
        bandwidth = (size_bytes / (1024**3)) / transfer_time if transfer_time > 0 else 0
        print(f"\n✓ Success: transfer_sync with offset verified")
        print(f"  Transfer time: {transfer_time:.3f}s")
        print(f"  Bandwidth: {bandwidth:.2f} GB/s")
        print(f"  Offset: {src_offset} bytes")
        print(f"  Transfer size: {size_bytes} bytes")
        
        parent_conn.send('done')
        return 0

    finally:
        try:
            parent_conn.send('done')
        except Exception:
            pass
        p.join(timeout=5)


if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
