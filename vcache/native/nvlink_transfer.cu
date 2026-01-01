#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint> // for uintptr_t

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {

// Return 0 on success, non-zero CUDA error code on failure.
EXPORT int perform_direct_transfer(
    int source_gpu,
    unsigned long long source_ptr, // device pointer value or 0
    int target_gpu,
    unsigned long long target_ptr, // device pointer value or 0
    unsigned long long size_bytes,
    const void* src_ipc_handle, 
    unsigned int src_ipc_handle_size,
    const void* dst_ipc_handle, 
    unsigned int dst_ipc_handle_size,
    unsigned long long src_offset_bytes, 
    unsigned long long dst_offset_bytes,
    void* stream_ptr // cudaStream_t as void*
) {
    if (size_bytes == 0) return 0;

    int rc = cudaSuccess;
    void* mapped_src = nullptr;
    void* mapped_dst = nullptr;
    bool opened_src = false;
    bool opened_dst = false;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Validate devices
    int device_count = 0;
    rc = cudaGetDeviceCount(&device_count);
    if (rc != cudaSuccess) return rc;
    if (source_gpu < 0 || source_gpu >= device_count) return cudaErrorInvalidDevice;
    if (target_gpu < 0 || target_gpu >= device_count) return cudaErrorInvalidDevice;

    // If source IPC handle provided, open it on current process
    if (src_ipc_handle != nullptr && src_ipc_handle_size > 0) {
        // cudaIpcMemHandle_t is opaque 64 bytes (commonly)
        cudaIpcMemHandle_t handle;
        if (src_ipc_handle_size > sizeof(handle)) {
            // invalid size
            return cudaErrorInvalidValue;
        }
        memset(&handle, 0, sizeof(handle));
        memcpy(&handle, src_ipc_handle, src_ipc_handle_size);

        // It's recommended to set the device to the source device when opening, but
        // the runtime allows opening from any device. We'll set the device to target_gpu
        // only if needed later. Here set to source_gpu to be consistent.
        rc = cudaSetDevice(source_gpu);
        if (rc != cudaSuccess) return rc;

        rc = cudaIpcOpenMemHandle(&mapped_src, handle, cudaIpcMemLazyEnablePeerAccess);
        if (rc != cudaSuccess) {
            // try without flag
            rc = cudaIpcOpenMemHandle(&mapped_src, handle, 0);
            if (rc != cudaSuccess) return rc;
        }
        opened_src = true;
    }

    // If dst IPC handle provided, open it on this process
    if (dst_ipc_handle != nullptr && dst_ipc_handle_size > 0) {
        cudaIpcMemHandle_t handle;
        if (dst_ipc_handle_size > sizeof(handle)) return cudaErrorInvalidValue;
        memset(&handle, 0, sizeof(handle));
        memcpy(&handle, dst_ipc_handle, dst_ipc_handle_size);

        // For safety set device to target device while opening dst mapping
        rc = cudaSetDevice(target_gpu);
        if (rc != cudaSuccess) return rc;

        rc = cudaIpcOpenMemHandle(&mapped_dst, handle, cudaIpcMemLazyEnablePeerAccess);
        if (rc != cudaSuccess) {
            rc = cudaIpcOpenMemHandle(&mapped_dst, handle, 0);
            if (rc != cudaSuccess) return rc;
        }
        opened_dst = true;
    }

    // Determine actual pointers to use (apply offsets if provided)
    void* src = nullptr;
    void* dst = nullptr;

    if (opened_src) {
        // mapped_src is the base pointer for the IPC mapping; add offset
        src = (void*)((uintptr_t)mapped_src + (uintptr_t)src_offset_bytes);
    } else {
        if (source_ptr == 0) return cudaErrorInvalidDevicePointer;
        src = (void*)((uintptr_t)source_ptr + (uintptr_t)src_offset_bytes);
    }

    if (opened_dst) {
        dst = (void*)((uintptr_t)mapped_dst + (uintptr_t)dst_offset_bytes);
    } else {
        if (target_ptr == 0) return cudaErrorInvalidDevicePointer;
        dst = (void*)((uintptr_t)target_ptr + (uintptr_t)dst_offset_bytes);
    }

    // Launch the peer copy from source device context
    rc = cudaSetDevice(source_gpu);
    if (rc != cudaSuccess) goto cleanup;

    // cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)
    rc = cudaMemcpyPeerAsync(dst, target_gpu, src, source_gpu, (size_t)size_bytes, stream);
    if (rc != cudaSuccess) goto cleanup;

    // Synchronize the provided stream (if any) to ensure the async copy completes
    if (stream != nullptr) {
        rc = cudaStreamSynchronize(stream);
        if (rc != cudaSuccess) goto cleanup;
    } else {
        // As a fallback, synchronize the device
        rc = cudaDeviceSynchronize();
        if (rc != cudaSuccess) goto cleanup;
    }

    // Close any opened IPC handles now that the copy has completed
    if (opened_src) {
        cudaError_t close_rc = cudaIpcCloseMemHandle(mapped_src);
        if (close_rc != cudaSuccess) {
            // prefer returning close error so caller is aware
            rc = close_rc;
        }
    }

    if (opened_dst) {
        cudaError_t close_rc = cudaIpcCloseMemHandle(mapped_dst);
        if (close_rc != cudaSuccess) {
            rc = close_rc;
        }
    }

    return rc;

cleanup:
    // Attempt to close any mappings if they were opened before returning
    if (opened_src) {
        cudaIpcCloseMemHandle(mapped_src);
    }
    if (opened_dst) {
        cudaIpcCloseMemHandle(mapped_dst);
    }

    return rc;
}

} // extern C
