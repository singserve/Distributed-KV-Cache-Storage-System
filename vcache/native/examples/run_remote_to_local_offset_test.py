#!/usr/bin/env python3
"""Standalone remote->local offset transfer test (no pytest).

Usage:
  python vcache/native/examples/run_remote_to_local_offset_test.py [--src 0] [--dst 1] [--total 1048576] [--offset 65536] [--len 1024]

Exit codes:
  0 - success (data matched)
  1 - failure (transfer failed or data mismatch)
  2 - environment not suitable (no CUDA or <2 GPUs)
  3 - native library not found
"""

import argparse
import ctypes
import sys
import time
import multiprocessing as mp  # use mp.get_context('spawn') for CUDA safety

import torch

from vcache.nvlink_transfer_engine import DistributedNVLINKTransferEngine

IPC_HANDLE_SIZE = 64


def load_libnvlink():
    try:
        return ctypes.CDLL("libnvlink_transfer.so")
    except OSError:
        return None


def producer(conn, src_gpu: int, total_size: int, offset: int, pattern_len: int):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=int, default=0)
    parser.add_argument('--dst', type=int, default=1)
    parser.add_argument('--total', type=int, default=1024 * 1024)  # 1MB
    parser.add_argument('--offset', type=int, default=64 * 1024)   # 64KB
    parser.add_argument('--len', dest='plen', type=int, default=1024)  # 1KB

    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Environment requires CUDA and at least 2 GPUs", file=sys.stderr)
        return 2

    if load_libnvlink() is None:
        print("libnvlink_transfer.so not found; build it before running this test", file=sys.stderr)
        return 3

    ctx = mp.get_context('spawn')
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(target=producer, args=(child_conn, args.src, args.total, args.offset, args.plen), daemon=True)
    p.start()

    try:
        msg = parent_conn.recv()
        if 'error' in msg:
            print(f"Producer error: {msg['error']}", file=sys.stderr)
            return 1

        handle_bytes = msg['handle']
        size_bytes = msg['size']
        expected = msg['expected']
        src_gpu = msg['src_gpu']
        src_offset = msg['offset']

        # Run the consumer work in a separate spawned process to isolate possible native crashes
        def consumer_worker(res_conn):
            try:
                engine = DistributedNVLINKTransferEngine(type('C', (), {'get_extra_config_value': lambda *_: 2})(), gpu_id=args.dst)

                torch.cuda.set_device(args.dst)
                dst = torch.empty(size_bytes, dtype=torch.uint8, device=f'cuda:{args.dst}')
                dst_ptr = dst.data_ptr()

                success = engine.transfer_sync(
                    source_gpu=src_gpu,
                    target_gpu=args.dst,
                    source_address=0,
                    target_address=dst_ptr,
                    size=size_bytes,
                    ipc_handle=handle_bytes,
                    src_offset=src_offset,
                    dst_offset=0,
                )

                if not success:
                    res_conn.send({'status': 'failed', 'reason': 'transfer_sync returned False'})
                    res_conn.close()
                    return

                out = dst.cpu().numpy().tobytes()
                if out != expected:
                    res_conn.send({'status': 'failed', 'reason': 'data_mismatch'})
                    res_conn.close()
                    return

                res_conn.send({'status': 'ok'})
                res_conn.close()
                return
            except Exception as e:
                try:
                    res_conn.send({'status': 'failed', 'reason': f'exception: {e}'})
                except Exception:
                    pass
                res_conn.close()
                return

        # spawn consumer
        cctx = mp.get_context('spawn')
        cres_parent, cres_child = cctx.Pipe()
        consumer_p = cctx.Process(target=consumer_worker, args=(cres_child,), daemon=True)
        consumer_p.start()

        # Wait for consumer result (with timeout)
        try:
            if cres_parent.poll(timeout=30):
                cres_msg = cres_parent.recv()
            else:
                cres_msg = {'status': 'failed', 'reason': 'consumer_timeout'}
        except Exception as e:
            cres_msg = {'status': 'failed', 'reason': f'pipe_error: {e}'}

        # Ensure consumer process cleaned up
        consumer_p.join(timeout=5)
        if consumer_p.is_alive():
            consumer_p.terminate()

        if cres_msg.get('status') != 'ok':
            print(f"Consumer failed: {cres_msg}", file=sys.stderr)
            parent_conn.send('done')
            return 1

        print("Success: remote->local offset transfer verified")
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
