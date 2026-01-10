"""
Simple test script for NVLINK Transfer Engine with 2 GPUs.

This script demonstrates:
1. Initializing transfer engines on two GPUs
2. Allocating memory on both GPUs
3. Transferring data between GPUs
4. Verifying the transferred data
5. Monitoring transfer statistics
"""

import torch
import time
import numpy as np
from vcache.transfer_engine.nvlink_transfer_engine import DistributedNVLINKTransferEngine

class SimpleConfig:
    """Simple config object for the transfer engine"""
    def __init__(self):
        self.max_concurrent_transfers = 2
        self.transfer_timeout_sec = 30.0
    
    def get_extra_config_value(self, key, default):
        """Get configuration values"""
        if key == "max_concurrent_transfers":
            return self.max_concurrent_transfers
        elif key == "transfer_timeout_sec":
            return self.transfer_timeout_sec
        return default


def check_requirements():
    """Check if we have at least 2 GPUs"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✓ CUDA available with {num_gpus} GPU(s)")
    
    if num_gpus < 2:
        print(f"❌ Need at least 2 GPUs, but only have {num_gpus}")
        return False
    
    print(f"✓ {num_gpus} GPUs available, ready for testing")
    
    # Print GPU info
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
    
    return True


def test_basic_transfer():
    """Test basic GPU-to-GPU transfer"""
    print("\n" + "="*60)
    print("TEST 1: Basic GPU-to-GPU Transfer")
    print("="*60)
    
    # Create config
    config = SimpleConfig()
    
    # Create transfer engine on GPU 0
    engine = DistributedNVLINKTransferEngine(config, gpu_id=0)
    
    try:
        # Allocate memory on GPU 0 (target)
        print("\n[Step 1] Allocating memory on GPU 0 (target)...")
        torch.cuda.set_device(0)
        target_size = 1024 * 256  # 1MB (256K float32s)
        target_tensor = torch.zeros(target_size, dtype=torch.float32, device='cuda:0')
        target_addr = target_tensor.data_ptr()
        print(f"  Allocated {target_size*4/1024:.1f}KB on GPU 0")
        print(f"  Address: 0x{target_addr:x}")
        
        # Allocate memory on GPU 1 (source)
        print("\n[Step 2] Allocating memory on GPU 1 (source)...")
        torch.cuda.set_device(1)
        source_tensor = torch.ones(target_size, dtype=torch.float32, device='cuda:1')
        source_addr = source_tensor.data_ptr()
        print(f"  Allocated {target_size*4/1024:.1f}KB on GPU 1")
        print(f"  Address: 0x{source_addr:x}")
        
        # Verify data before transfer
        torch.cuda.set_device(0)
        before_sum = target_tensor.sum().item()
        print(f"\n[Step 3] Verifying data before transfer...")
        print(f"  GPU 0 target tensor sum: {before_sum:.2f} (should be ~0)")
        
        torch.cuda.set_device(1)
        source_sum = source_tensor.sum().item()
        print(f"  GPU 1 source tensor sum: {source_sum:.2f} (should be ~{target_size})")
        
        # Perform transfer
        print(f"\n[Step 4] Submitting transfer request...")
        transfer_size = target_size * 4  # 4 bytes per float32
        
        transfer_id = engine.submit_transfer(
            source_gpu=1,
            target_gpu=0,
            source_address=source_addr,
            target_address=target_addr,
            size=transfer_size
        )
        print(f"  Transfer ID: {transfer_id}")
        print(f"  Transfer size: {transfer_size/1024:.1f}KB")
        
        # Wait for transfer to complete
        print(f"\n[Step 5] Waiting for transfer to complete...")
        start_time = time.time()
        
        success = engine.wait_for_transfer(transfer_id, timeout=30.0)
        elapsed = time.time() - start_time
        
        if success:
            print(f"  ✓ Transfer completed successfully in {elapsed*1000:.3f}ms")
        else:
            print(f"  ✗ Transfer failed or timed out!")
            # Get more diagnostic information
            status = engine.get_transfer_status(transfer_id)
            if status:
                print(f"    Transfer status: {status['status']}")
                if status['error_message']:
                    print(f"    Error: {status['error_message']}")
            else:
                print(f"    Warning: Could not retrieve transfer status")
            return False
        
        # Verify data after transfer
        print(f"\n[Step 6] Verifying data after transfer...")
        torch.cuda.set_device(0)
        after_sum = target_tensor.sum().item()
        print(f"  GPU 0 target tensor sum: {after_sum:.2f}")
        
        # Check if data matches
        if torch.allclose(target_tensor, source_tensor.to('cuda:0')):
            print(f"  ✓ Data verification PASSED! Data matches perfectly.")
            success = True
        else:
            print(f"  ✗ Data verification FAILED! Data mismatch.")
            print(f"    Expected sum: {source_sum:.2f}, Got: {after_sum:.2f}")
            success = False
        
        # Show engine status
        print(f"\n[Step 7] Engine status:")
        status = engine.get_status()
        print(f"  Engine type: {status['engine_type']}")
        print(f"  Initialized: {status['initialized']}")
        print(f"  CUDA available: {status['cuda_available']}")
        print(f"  NVLINK available: {status['nvlink_available']}")
        print(f"  GPU ID: {status['gpu_id']}")
        print(f"  Worker thread running: {status['worker_thread_running']}")
        print(f"  Worker thread alive: {status['worker_thread_alive']}")
        print(f"  Pending transfers: {status['pending_transfers']}")
        print(f"  Active transfers: {status['active_transfers']}")
        print(f"  Completed transfers count: {status['completed_transfers_count']}")
        
        return success
        
    finally:
        # Cleanup
        print(f"\n[Cleanup] Shutting down engine...")
        engine.shutdown()
        print(f"  ✓ Engine shutdown complete")


def test_multiple_transfers():
    """Test multiple sequential transfers"""
    print("\n" + "="*60)
    print("TEST 2: Multiple Sequential Transfers")
    print("="*60)
    
    config = SimpleConfig()
    engine = DistributedNVLINKTransferEngine(config, gpu_id=0)
    
    try:
        chunk_size = 1024 * 64  # 256KB per chunk
        num_chunks = 4
        total_size = chunk_size * num_chunks
        
        print(f"\nTransferring {num_chunks} chunks of {chunk_size*4/1024:.0f}KB each...")
        
        # Allocate memory
        torch.cuda.set_device(0)
        target_tensor = torch.zeros(total_size, dtype=torch.float32, device='cuda:0')
        target_base_addr = target_tensor.data_ptr()
        
        torch.cuda.set_device(1)
        source_tensor = torch.ones(total_size, dtype=torch.float32, device='cuda:1')
        source_base_addr = source_tensor.data_ptr()
        
        # Transfer each chunk
        print("\nSubmitting transfers...")
        start_time = time.time()
        transfer_ids = []
        
        for i in range(num_chunks):
            offset = i * chunk_size * 4  # bytes
            src_addr = source_base_addr + offset
            tgt_addr = target_base_addr + offset
            
            tid = engine.submit_transfer(
                source_gpu=1,
                target_gpu=0,
                source_address=src_addr,
                target_address=tgt_addr,
                size=chunk_size * 4
            )
            transfer_ids.append(tid)
            print(f"  Chunk {i}: {tid}")
        
        # Wait for all transfers
        print("\nWaiting for all transfers to complete...")
        all_success = True
        for i, tid in enumerate(transfer_ids):
            if engine.wait_for_transfer(tid, timeout=30.0):
                print(f"  ✓ Chunk {i} completed")
            else:
                print(f"  ✗ Chunk {i} failed")
                all_success = False
        
        elapsed = time.time() - start_time
        
        # Verify
        torch.cuda.set_device(0)
        if torch.allclose(target_tensor, source_tensor.to('cuda:0')):
            print(f"\n✓ All transfers PASSED!")
        else:
            print(f"\n✗ Data verification FAILED!")
            all_success = False
        
        # Statistics
        print(f"\nStatistics for {num_chunks} chunks:")
        print(f"  Total time: {elapsed*1000:.1f}ms")
        print(f"  Total data: {num_chunks * chunk_size * 4 / 1024 / 1024:.1f}MB")
        print(f"  Throughput: {num_chunks * chunk_size * 4 / 1024 / 1024 / elapsed:.1f}MB/s")
        print(f"  Avg latency per transfer: {elapsed / num_chunks * 1000:.3f}ms")
        
        return all_success
        
    finally:
        engine.shutdown()


def test_sync_transfer():
    """Test synchronous transfer"""
    print("\n" + "="*60)
    print("TEST 3: Synchronous Transfer")
    print("="*60)
    
    config = SimpleConfig()
    engine = DistributedNVLINKTransferEngine(config, gpu_id=0)
    
    try:
        size = 1024 * 256  # 1MB
        
        print(f"\nAllocating memory for {size*4/1024:.0f}KB transfer...")
        
        # Allocate
        torch.cuda.set_device(0)
        target = torch.zeros(size, dtype=torch.float32, device='cuda:0')
        target_addr = target.data_ptr()
        
        torch.cuda.set_device(1)
        source = torch.ones(size, dtype=torch.float32, device='cuda:1')
        source_addr = source.data_ptr()
        
        # Synchronous transfer
        print(f"Performing synchronous transfer...")
        start_time = time.time()
        
        success = engine.transfer_sync(
            source_gpu=1,
            target_gpu=0,
            source_address=source_addr,
            target_address=target_addr,
            size=size * 4
        )
        
        elapsed = time.time() - start_time
        
        if success:
            print(f"✓ Transfer completed in {elapsed*1000:.3f}ms")
            
            # Verify
            torch.cuda.set_device(0)
            if torch.allclose(target, source.to('cuda:0')):
                print(f"✓ Data verification PASSED!")
                return True
            else:
                print(f"✗ Data verification FAILED!")
                return False
        else:
            print(f"✗ Transfer failed!")
            return False
        
    finally:
        engine.shutdown()


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("NVLINK Transfer Engine Test Suite")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Test environment check failed!")
        return False
    
    # Run tests
    test_results = {}
    
    try:
        test_results["Basic Transfer"] = test_basic_transfer()
    except Exception as e:
        print(f"\n❌ Basic Transfer test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        test_results["Basic Transfer"] = False
    
    try:
        test_results["Multiple Transfers"] = test_multiple_transfers()
    except Exception as e:
        print(f"\n❌ Multiple Transfers test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        test_results["Multiple Transfers"] = False
    
    try:
        test_results["Sync Transfer"] = test_sync_transfer()
    except Exception as e:
        print(f"\n❌ Sync Transfer test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        test_results["Sync Transfer"] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(test_results.values())
    print("\n" + ("="*60))
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
