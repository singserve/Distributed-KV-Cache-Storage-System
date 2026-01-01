# NVLink Transfer native helper

This directory contains a small C++ CUDA helper to perform cross-GPU and cross-process transfers using CUDA IPC and `cudaMemcpyPeerAsync`.

Build (Linux/macOS):

1. mkdir build && cd build
2. cmake ..
3. cmake --build . --config Release

Build (Windows with Visual Studio + CUDA):

1. mkdir build && cd build
2. cmake -G "Visual Studio 17 2022" .. -A x64
3. cmake --build . --config Release

This produces a shared library `libnvlink_transfer.so` (Linux) or `nvlink_transfer.dll` (Windows).

Usage:
- The library exports `perform_direct_transfer(...)` returning 0 on success or a CUDA error code.
- The Python `nvlink_transfer_engine` will attempt to load this library via `ctypes` and call it when available.

Examples:
- See `vcache/native/examples/remote_to_local_example.py` for a two-process remote->local transfer demo.

Notes:
- The function opens provided IPC handles (if any), launches a peer copy on the source device context, and then synchronizes the provided CUDA stream before automatically closing any opened IPC handles. This simplifies caller logic: you no longer need to explicitly close the source IPC handle after calling the helper.
- Destination IPC handles are optional: for remote->local pulls you only need to provide the source `ipc_handle` bytes. The helper supports an optional destination handle if both source and destination buffers are remote, but this is not required for the typical remote->local use case.
- New: `perform_direct_transfer` accepts `src_offset_bytes` and `dst_offset_bytes` to copy from/to offsets inside the mapped buffers. Use `src_offset_bytes` to pull from an offset inside a remote buffer without remapping or slicing on the remote side.

Run the standalone offset test (no pytest):

  python vcache/native/examples/run_remote_to_local_offset_test.py --src 0 --dst 1 --total 1048576 --offset 65536 --len 1024

Note: On Linux use the 'spawn' start method for multiprocessing to avoid CUDA re-initialization errors in forked subprocesses. The example scripts use spawn internally (`mp.get_context('spawn')`).

Return codes: 0 = success, 1 = failure (transfer/data mismatch), 2 = no suitable CUDA env, 3 = native lib missing.
- This is a minimal helper intended to be improved and hardened for production (error translation, richer logging, tests, etc.).
