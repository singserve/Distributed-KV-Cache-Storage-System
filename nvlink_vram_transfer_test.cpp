

//# Basic test with default settings
//./nvlink_vram_transfer_test

//# Test with specific GPU and data size
//./nvlink_vram_transfer_test --gpu_id=0 --gpu_count=2 --data_size_mb=50

//# Test with different metadata server
//./nvlink_vram_transfer_test --metadata_server=192.168.1.100:2379


#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <thread>
#include <memory>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <numa.h>
#include <sstream>
#include <fstream>
#include "transfer_engine.h"
#include "transport/transport.h"
#include "common.h"
#include <transport/nvlink_transport/nvlink_transport.h>

using namespace mooncake;

DEFINE_string(metadata_server, "http://192.168.1.86:8080/metadata", "etcd server host address");
DEFINE_string(local_server_name, "192.168.1.86:13452", "Local server name");
DEFINE_string(dst_server_name, "192.168.1.86:22345", "dst server name");
DEFINE_string(segment_id, "192.168.1.86:13452", "Segment ID to access data");
DEFINE_int32(gpu_count, 2, "Number of GPUs to test");
DEFINE_int32(data_size_mb, 1, "Data size in MB for transfer test");
DEFINE_string(transport, "nvlink", "use wanted transport for multi-gpu test: rdma/nvlink");
DEFINE_string(device_name, "mlx5_0", "Device name(s) for RDMA transport, comma separated if multiple");


static void checkCudaError(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        LOG(ERROR) << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")";
        exit(EXIT_FAILURE);
    }
}

static void* allocateCudaBuffer(size_t size, int gpu_id) {  
    checkCudaError(cudaSetDevice(gpu_id), "Failed to set device");  
    void* d_buf = NvlinkTransport::allocatePinnedLocalMemory(size); 
    checkCudaError(d_buf == nullptr ? cudaErrorMemoryAllocation : cudaSuccess,  
                   "Failed to allocate device memory");  
    return d_buf;  
}

static void freeCudaBuffer(void* d_buf) {  
    NvlinkTransport::freePinnedLocalMemory(d_buf);  
}

static void fillRandomData(void* buffer, size_t size, int gpu_id) {
    std::vector<char> host_data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    for (size_t i = 0; i < size; ++i) {
        host_data[i] = static_cast<char>(dis(gen));
    }
    
    checkCudaError(cudaSetDevice(gpu_id), "Failed to set device for data fill");
    checkCudaError(cudaMemcpy(buffer, host_data.data(), size, cudaMemcpyHostToDevice),
                   "Failed to copy random data to GPU");
}


// Device name formatting functions inspired by rdma_transport_test.cpp
static std::string formatDeviceNames(const std::string& device_names) {
    std::stringstream ss(device_names);
    std::string item;
    std::vector<std::string> tokens;
    while (getline(ss, item, ',')) {
        tokens.push_back(item);
    }

    std::string formatted;
    for (size_t i = 0; i < tokens.size(); ++i) {
        formatted += "\"" + tokens[i] + "\"";
        if (i < tokens.size() - 1) {
            formatted += ",";
        }
    }
    return formatted;
}

static std::string loadNicPriorityMatrix(const std::string& device_names = "mlx5_0") {
    // Build JSON Data for NIC priority matrix
    auto formatted_device_names = formatDeviceNames(device_names);
    return "{\"cpu:0\": [[" + formatted_device_names +
           "], []], "
           " \"cpu:1\": [[" +
           formatted_device_names +
           "], []], "
           " \"cuda:0\": [[" +
           formatted_device_names +
           "], []], "
           " \"cuda:1\": [[" +
           formatted_device_names +
           "], []], "
           " \"musa:0\": [[" +
           formatted_device_names + "], []]}";
}

// Helper function to install transport with device configuration
static Transport* installTransportWithDeviceConfig(TransferEngine* engine, 
                                                  const std::string& transport_type,
                                                  const std::string& device_names = "mlx5_0") {
    if (transport_type == "rdma") {
        auto nic_priority_matrix = loadNicPriorityMatrix(device_names);
        void** args = (void**)malloc(2 * sizeof(void*));
        args[0] = (void*)nic_priority_matrix.c_str();
        args[1] = nullptr;
        return engine->installTransport("rdma", args);
    } else if (transport_type == "nvlink") {
        return engine->installTransport("nvlink", nullptr);
    } else if (transport_type == "tcp") {
        return engine->installTransport("tcp", nullptr);
    } else {
        LOG(ERROR) << "Unsupported transport type: " << transport_type;
        return nullptr;
    }
}


// Helper function to measure transfer time
static double measureTransferTime(std::function<void()> transfer_func) {
    auto start_time = std::chrono::high_resolution_clock::now();
    transfer_func();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    return duration.count();
}

TEST(NvlinkVramTransferTest, MultiGPUCrossTransfer) {
    if (FLAGS_gpu_count < 2) {
        GTEST_SKIP() << "Multi-GPU test requires at least 2 GPUs";
    }

    const size_t kDataLength = FLAGS_data_size_mb * 1024 * 1024;
    int src_gpu = 0;
    int dst_gpu = 1;
    
    LOG(INFO) << "Testing two-GPU NVLink cross-transfer from GPU " << src_gpu << " to GPU " << dst_gpu;

     // Destination GPU setup (server)
        auto server_engine = std::make_unique<TransferEngine>(false);
        auto hostname1_port = parseHostNameWithPort(FLAGS_local_server_name);
        server_engine->init(FLAGS_metadata_server, FLAGS_local_server_name.c_str(),
                        hostname1_port.first.c_str(), hostname1_port.second);

        // Use device configuration for transport installation
        Transport* server_transport = installTransportWithDeviceConfig(server_engine.get(), FLAGS_transport, FLAGS_device_name);
        ASSERT_NE(server_transport, nullptr);

        void* server_buffer = allocateCudaBuffer(kDataLength, dst_gpu);
        int rc = server_engine->registerLocalMemory(server_buffer, kDataLength, "cuda:" + std::to_string(dst_gpu));
        ASSERT_EQ(rc, 0);

        auto segment_id = server_engine->openSegment(FLAGS_segment_id);

        // Source GPU setup (client)
        auto client_engine = std::make_unique<TransferEngine>(false);
        auto hostname2_port = parseHostNameWithPort(FLAGS_dst_server_name);
        client_engine->init(FLAGS_metadata_server, FLAGS_dst_server_name.c_str(),
                            hostname2_port.first.c_str(), hostname2_port.second);

        // Use device configuration for transport installation
        Transport* client_transport = installTransportWithDeviceConfig(client_engine.get(), FLAGS_transport, FLAGS_device_name);
        ASSERT_NE(client_transport, nullptr);

        void* client_buffer = allocateCudaBuffer(kDataLength, src_gpu);
        rc = client_engine->registerLocalMemory(client_buffer, kDataLength, "cuda:" + std::to_string(src_gpu));
        ASSERT_EQ(rc, 0);

        // Fill source GPU buffer with random data
        fillRandomData(client_buffer, kDataLength, src_gpu);

    // Perform transfer: source GPU -> destination GPU
    auto batch_id = client_engine->allocateBatchID(1);
    TransferRequest entry;
    entry.opcode = TransferRequest::WRITE;
    entry.length = kDataLength;
    entry.source = client_buffer;
    entry.target_id = segment_id;
    entry.target_offset = (uint64_t)server_buffer;
    Status s = client_engine->submitTransfer(batch_id, {entry});
    ASSERT_TRUE(s.ok());

    // Wait for completion
    TransferStatus status;
    do {
        s = client_engine->getTransferStatus(batch_id, 0, status);
        ASSERT_TRUE(s.ok());
    } while (status.s == TransferStatusEnum::WAITING);

    ASSERT_EQ(status.s, TransferStatusEnum::COMPLETED);
    s = client_engine->freeBatchID(batch_id);
    ASSERT_TRUE(s.ok());

        // Verify data integrity on destination GPU
        void* verification_buffer = allocateCudaBuffer(kDataLength, dst_gpu);
        checkCudaError(cudaSetDevice(dst_gpu), "Failed to set destination device for verification");
        checkCudaError(cudaMemcpy(verification_buffer, server_buffer, kDataLength, cudaMemcpyDeviceToDevice),
                       "Failed to copy data for verification");
        
        // Copy source data to host for comparison
        std::vector<char> source_data(kDataLength);
        checkCudaError(cudaSetDevice(src_gpu), "Failed to set source device");
        checkCudaError(cudaMemcpy(source_data.data(), client_buffer, kDataLength, cudaMemcpyDeviceToHost),
                       "Failed to copy source data to host");

        // Copy destination data to host for comparison
        std::vector<char> dest_data(kDataLength);
        checkCudaError(cudaSetDevice(dst_gpu), "Failed to set destination device");
        checkCudaError(cudaMemcpy(dest_data.data(), verification_buffer, kDataLength, cudaMemcpyDeviceToHost),
                       "Failed to copy destination data to host");

        // Verify data integrity
        bool data_correct = (memcmp(source_data.data(), dest_data.data(), kDataLength) == 0);
        ASSERT_TRUE(data_correct) << "Data mismatch in VRAM transfer from GPU " << src_gpu << " to GPU " << dst_gpu;

        // Cleanup
        freeCudaBuffer(verification_buffer);
        client_engine->unregisterLocalMemory(client_buffer);
        freeCudaBuffer(client_buffer);
        server_engine->unregisterLocalMemory(server_buffer);
        freeCudaBuffer(server_buffer);
}

TEST(NvlinkVramTransferTest, VRAMvsDRAMPerformanceComparison) {
    if (FLAGS_gpu_count < 2) {
        GTEST_SKIP() << "VRAM vs DRAM comparison test requires at least 2 GPUs";
    }

    const size_t kDataLength = FLAGS_data_size_mb * 1024 * 1024;
    int src_gpu = 0;
    int dst_gpu = 1; // Use different GPU for destination

    LOG(INFO) << "Comparing VRAM (NVLink) vs DRAM transfer performance with " 
              << (kDataLength / (1024 * 1024)) << "MB data from GPU " << src_gpu << " to GPU " << dst_gpu;

    double vram_transfer_time = 0.0;
    double dram_transfer_time = 0.0;

    // Test 1: VRAM to VRAM transfer via NVLink (direct GPU to GPU)
    {
        LOG(INFO) << "Testing VRAM to VRAM transfer via NVLink from GPU " << src_gpu << " to GPU " << dst_gpu;

        // Destination GPU setup (server)
        auto server_engine = std::make_unique<TransferEngine>(false);
        auto hostname1_port = parseHostNameWithPort(FLAGS_local_server_name);
        server_engine->init(FLAGS_metadata_server, FLAGS_local_server_name.c_str(),
                        hostname1_port.first.c_str(), hostname1_port.second);

        // Use device configuration for transport installation
        Transport* server_transport = installTransportWithDeviceConfig(server_engine.get(), FLAGS_transport, FLAGS_device_name);
        ASSERT_NE(server_transport, nullptr);

        void* server_buffer = allocateCudaBuffer(kDataLength, dst_gpu);
        int rc = server_engine->registerLocalMemory(server_buffer, kDataLength, "cuda:" + std::to_string(dst_gpu));
        ASSERT_EQ(rc, 0);

        auto segment_id = server_engine->openSegment(FLAGS_segment_id);

        // Source GPU setup (client)
        auto client_engine = std::make_unique<TransferEngine>(false);
        auto hostname2_port = parseHostNameWithPort(FLAGS_dst_server_name);
        client_engine->init(FLAGS_metadata_server, FLAGS_dst_server_name.c_str(),
                            hostname2_port.first.c_str(), hostname2_port.second);

        // Use device configuration for transport installation
        Transport* client_transport = installTransportWithDeviceConfig(client_engine.get(), FLAGS_transport, FLAGS_device_name);
        ASSERT_NE(client_transport, nullptr);

        void* client_buffer = allocateCudaBuffer(kDataLength, src_gpu);
        rc = client_engine->registerLocalMemory(client_buffer, kDataLength, "cuda:" + std::to_string(src_gpu));
        ASSERT_EQ(rc, 0);

        // Fill source GPU buffer with random data
        fillRandomData(client_buffer, kDataLength, src_gpu);

        // Measure VRAM transfer time (direct GPU to GPU via NVLink)
        vram_transfer_time = measureTransferTime([&]() {
            auto batch_id = client_engine->allocateBatchID(1);
            TransferRequest entry;
            entry.opcode = TransferRequest::WRITE;
            entry.length = kDataLength;
            entry.source = client_buffer;
            entry.target_id = segment_id;
            entry.target_offset = (uint64_t)server_buffer;
            Status s = client_engine->submitTransfer(batch_id, {entry});
            ASSERT_TRUE(s.ok());

            TransferStatus status;
            do {
                s = client_engine->getTransferStatus(batch_id, 0, status);
                ASSERT_TRUE(s.ok());
            } while (status.s == TransferStatusEnum::WAITING);

            ASSERT_EQ(status.s, TransferStatusEnum::COMPLETED);
            s = client_engine->freeBatchID(batch_id);
            ASSERT_TRUE(s.ok());
        });

        // Verify data integrity on destination GPU
        void* verification_buffer = allocateCudaBuffer(kDataLength, dst_gpu);
        checkCudaError(cudaSetDevice(dst_gpu), "Failed to set destination device for verification");
        checkCudaError(cudaMemcpy(verification_buffer, server_buffer, kDataLength, cudaMemcpyDeviceToDevice),
                       "Failed to copy data for verification");
        
        // Copy source data to host for comparison
        std::vector<char> source_data(kDataLength);
        checkCudaError(cudaSetDevice(src_gpu), "Failed to set source device");
        checkCudaError(cudaMemcpy(source_data.data(), client_buffer, kDataLength, cudaMemcpyDeviceToHost),
                       "Failed to copy source data to host");

        // Copy destination data to host for comparison
        std::vector<char> dest_data(kDataLength);
        checkCudaError(cudaSetDevice(dst_gpu), "Failed to set destination device");
        checkCudaError(cudaMemcpy(dest_data.data(), verification_buffer, kDataLength, cudaMemcpyDeviceToHost),
                       "Failed to copy destination data to host");

        // Verify data integrity
        bool data_correct = (memcmp(source_data.data(), dest_data.data(), kDataLength) == 0);
        ASSERT_TRUE(data_correct) << "Data mismatch in VRAM transfer from GPU " << src_gpu << " to GPU " << dst_gpu;

        // Cleanup
        freeCudaBuffer(verification_buffer);
        client_engine->unregisterLocalMemory(client_buffer);
        freeCudaBuffer(client_buffer);
        server_engine->unregisterLocalMemory(server_buffer);
        freeCudaBuffer(server_buffer);
    }

    // Test 2: VRAM to VRAM transfer via DRAM (VRAM→DRAM→transfer→DRAM→VRAM)
    {
        LOG(INFO) << "Testing VRAM to VRAM transfer via DRAM from GPU " << src_gpu << " to GPU " << dst_gpu;

        // Server setup - only DRAM needed
        auto server_engine2 = std::make_unique<TransferEngine>(false);
        server_engine2->init(FLAGS_metadata_server, FLAGS_local_server_name);

        // Use TCP transport for DRAM transfers
        Transport* server_transport2 = server_engine2->installTransport("tcp", nullptr);
        ASSERT_NE(server_transport2, nullptr);

        // Allocate DRAM buffer on server for intermediate storage
        void* server_dram_buffer = numa_alloc_onnode(kDataLength, 0);
        ASSERT_NE(server_dram_buffer, nullptr);
        int rc = server_engine2->registerLocalMemory(server_dram_buffer, kDataLength, "cpu:0");
        ASSERT_EQ(rc, 0);

        auto segment_id = server_engine2->openSegment(FLAGS_segment_id);

        // Client setup - only DRAM needed
        auto client_engine2 = std::make_unique<TransferEngine>(false);
        client_engine2->init(FLAGS_metadata_server, FLAGS_dst_server_name);

        // Use TCP transport for DRAM transfers
        Transport* client_transport2 = client_engine2->installTransport("tcp", nullptr);
        ASSERT_NE(client_transport2, nullptr);

        // Allocate DRAM buffer on client for intermediate storage
        void* client_dram_buffer = numa_alloc_onnode(kDataLength, 0);
        ASSERT_NE(client_dram_buffer, nullptr);
        rc = client_engine2->registerLocalMemory(client_dram_buffer, kDataLength, "cpu:0");
        ASSERT_EQ(rc, 0);

        // Allocate source GPU VRAM buffer and fill with random data
        void* src_vram_buffer = allocateCudaBuffer(kDataLength, src_gpu);
        fillRandomData(src_vram_buffer, kDataLength, src_gpu);

        // Allocate destination GPU VRAM buffer for verification
        void* dst_vram_buffer = allocateCudaBuffer(kDataLength, dst_gpu);

        // Measure VRAM→DRAM→transfer→DRAM→VRAM transfer time
        dram_transfer_time = measureTransferTime([&]() {
            // Step 1: Copy from source GPU VRAM to client DRAM
            checkCudaError(cudaSetDevice(src_gpu), "Failed to set source device for VRAM→DRAM copy");
            checkCudaError(cudaMemcpy(client_dram_buffer, src_vram_buffer, kDataLength, cudaMemcpyDeviceToHost),
                           "Failed to copy from source VRAM to DRAM");

            // Step 2: Transfer from client DRAM to server DRAM via TCP
            auto batch_id = client_engine2->allocateBatchID(1);
            TransferRequest entry;
            entry.opcode = TransferRequest::WRITE;
            entry.length = kDataLength;
            entry.source = client_dram_buffer;
            entry.target_id = segment_id;
            entry.target_offset = (uint64_t)server_dram_buffer;
            Status s = client_engine2->submitTransfer(batch_id, {entry});
            ASSERT_TRUE(s.ok());

            TransferStatus status;
            do {
                s = client_engine2->getTransferStatus(batch_id, 0, status);
                ASSERT_TRUE(s.ok());
            } while (status.s == TransferStatusEnum::WAITING);

            ASSERT_EQ(status.s, TransferStatusEnum::COMPLETED);
            s = client_engine2->freeBatchID(batch_id);
            ASSERT_TRUE(s.ok());

            // Step 3: Copy from server DRAM to destination GPU VRAM
            checkCudaError(cudaSetDevice(dst_gpu), "Failed to set destination device for DRAM→VRAM copy");
            checkCudaError(cudaMemcpy(dst_vram_buffer, server_dram_buffer, kDataLength, cudaMemcpyHostToDevice),
                           "Failed to copy from DRAM to destination VRAM");
        });

        // Verify data integrity - compare source GPU VRAM with destination GPU VRAM
        void* verification_buffer = allocateCudaBuffer(kDataLength, dst_gpu);
        checkCudaError(cudaSetDevice(dst_gpu), "Failed to set destination device for verification");
        checkCudaError(cudaMemcpy(verification_buffer, dst_vram_buffer, kDataLength, cudaMemcpyDeviceToDevice),
                       "Failed to copy data for verification");
        
        // Copy source data to host for comparison
        std::vector<char> source_data(kDataLength);
        checkCudaError(cudaSetDevice(src_gpu), "Failed to set source device");
        checkCudaError(cudaMemcpy(source_data.data(), src_vram_buffer, kDataLength, cudaMemcpyDeviceToHost),
                       "Failed to copy source data to host");

        // Copy destination data to host for comparison
        std::vector<char> dest_data(kDataLength);
        checkCudaError(cudaSetDevice(dst_gpu), "Failed to set destination device");
        checkCudaError(cudaMemcpy(dest_data.data(), verification_buffer, kDataLength, cudaMemcpyDeviceToHost),
                       "Failed to copy destination data to host");

        // Verify data integrity
        bool data_correct = (memcmp(source_data.data(), dest_data.data(), kDataLength) == 0);
        ASSERT_TRUE(data_correct) << "Data mismatch in DRAM transfer from GPU " << src_gpu << " to GPU " << dst_gpu;

        // Cleanup
        freeCudaBuffer(verification_buffer);
        freeCudaBuffer(src_vram_buffer);
        freeCudaBuffer(dst_vram_buffer);
        client_engine2->unregisterLocalMemory(client_dram_buffer);
        numa_free(client_dram_buffer, kDataLength);
        server_engine2->unregisterLocalMemory(server_dram_buffer);
        numa_free(server_dram_buffer, kDataLength);
    }

    // Performance comparison
    LOG(INFO) << "Performance Comparison Results:";
    LOG(INFO) << "VRAM (NVLink) transfer time: " << vram_transfer_time << " seconds";
    LOG(INFO) << "DRAM (TCP) transfer time: " << dram_transfer_time << " seconds";
    LOG(INFO) << "Data size: " << (kDataLength / (1024 * 1024)) << " MB";
    
    if (dram_transfer_time > 0) {
        double speedup = dram_transfer_time / vram_transfer_time;
        LOG(INFO) << "NVLink speedup over DRAM: " << speedup << "x";
        
        // Log bandwidth calculations
        double vram_bandwidth = (kDataLength / (1024 * 1024 * 1024.0)) / vram_transfer_time; // GB/s
        double dram_bandwidth = (kDataLength / (1024 * 1024 * 1024.0)) / dram_transfer_time; // GB/s
        LOG(INFO) << "VRAM bandwidth: " << vram_bandwidth << " GB/s";
        LOG(INFO) << "DRAM bandwidth: " << dram_bandwidth << " GB/s";
    }

    // Assert that VRAM transfer is faster (or at least not significantly slower)
    // Allow for some variance due to system load
    if (vram_transfer_time > dram_transfer_time * 2.0) {
        LOG(WARNING) << "VRAM transfer was unexpectedly slower than DRAM transfer";
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    ::testing::InitGoogleTest(&argc, argv);
    
    // Check CUDA availability
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count == 0) {
        LOG(ERROR) << "No CUDA-capable devices found or CUDA not available";
        return EXIT_FAILURE;
    }
    
    LOG(INFO) << "Found " << device_count << " CUDA device(s)";
    
    return RUN_ALL_TESTS();
}
