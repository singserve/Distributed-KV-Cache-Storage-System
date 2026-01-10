# NVLink Transfer native helper

This directory contains a small C++ CUDA helper to perform cross-GPU and cross-process transfers using CUDA IPC and `cudaMemcpyPeerAsync`.

Build:

1. mkdir build && cd build
2. cmake ..
3. cmake --build . --config Release
