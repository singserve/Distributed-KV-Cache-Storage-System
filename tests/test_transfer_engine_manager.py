"""
Test script for TransferEngineManager with NVLINK transfer engine.

This script tests:
1. Initializing TransferEngineManager
2. Transferring data between GPUs using transfer_gpu_to_gpu method
3. Verifying the transferred data
4. Testing shutdown functionality
"""

import torch
import time
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcache.transfer_engine_manager import TransferEngineManager

class SimpleConfig:
    """Simple config object for the transfer engine manager"""
    def __init__(self):
        self.max_concurrent_transfers = 2
        self.transfer_timeout_sec = 30.0
        self.gpu_id = 0  # Default GPU ID
    
    def get_extra_config_value(self, key, default):
        """Get configuration values"""
        if key == "max_concurrent_transfers":
            return self.max_concurrent_transfers
        elif key == "transfer_timeout_sec":
            return self.transfer_timeout_sec
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


def test_transfer_engine_manager_initialization():
    """Test TransferEngineManager initialization"""
    print("\n" + "="*60)
    print("TEST 1: TransferEngineManager Initialization")
    print("="*60)
    
    config = SimpleConfig()
    
    try:
        print("\n[Step 1] Creating TransferEngineManager...")
        manager = TransferEngineManager(config)
        
        if manager.initialized and manager.engine is not None:
            print(f"✓ TransferEngineManager initialized successfully")
            print(f"  Engine type: {type(manager.engine).__name__}")
            print(f"  Initialized: {manager.initialized}")
            return True
        else:
            print(f"✗ TransferEngineManager initialization failed")
            print(f"  Initialized: {manager.initialized}")
            print(f"  Engine: {manager.engine}")
            return False
            
    except Exception as e:
        print(f"✗ Exception during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'manager' in locals():
            manager.shutdown()


def test_transfer_gpu_to_gpu():
    """Test transfer_gpu_to_gpu method"""
    print("\n" + "="*60)
    print("TEST 2: transfer_gpu_to_gpu Method")
    print("="*60)
    
    config = SimpleConfig()
    manager = None
    
    try:
        print("\n[Step 1] Initializing TransferEngineManager...")
        manager = TransferEngineManager(config)
        
        if not manager.initialized or not manager.engine:
            print("✗ TransferEngineManager not initialized properly")
            return False
        
        print("✓ TransferEngineManager initialized")
        
        # Allocate memory on GPU 0 (target)
        print("\n[Step 2] Allocating memory on GPU 0 (target)...")
        torch.cuda.set_device(0)
        target_size = 1024 * 256  # 1MB (256K float32s)
        target_tensor = torch.zeros(target_size, dtype=torch.float32, device='cuda:0')
        target_buffer = target_tensor.data_ptr()
        print(f"  Allocated {target_size*4/1024:.1f}KB on GPU 0")
        print(f"  Buffer address: 0x{target_buffer:x}")
        
        # Allocate memory on GPU 1 (source)
        print("\n[Step 3] Allocating memory on GPU 1 (source)...")
        torch.cuda.set_device(1)
        source_tensor = torch.ones(target_size, dtype=torch.float32, device='cuda:1')
        source_buffer = source_tensor.data_ptr()
        print(f"  Allocated {target_size*4/1024:.1f}KB on GPU 1")
        print(f"  Buffer address: 0x{source_buffer:x}")
        
        # Verify data before transfer
        print("\n[Step 4] Verifying data before transfer...")
        torch.cuda.set_device(0)
        before_sum = target_tensor.sum().item()
        print(f"  GPU 0 target tensor sum: {before_sum:.2f} (should be ~0)")
        
        torch.cuda.set_device(1)
        source_sum = source_tensor.sum().item()
        print(f"  GPU 1 source tensor sum: {source_sum:.2f} (should be ~{target_size})")
        
        # Perform transfer using transfer_gpu_to_gpu method
        print("\n[Step 5] Calling transfer_gpu_to_gpu...")
        transfer_size = target_size * 4  # 4 bytes per float32
        
        # Note: target_hostname parameter is kept for API compatibility but not used by NVLINK engine
        success = manager.transfer_gpu_to_gpu(
            target_hostname="localhost",  # Not used by NVLINK engine
            source_gpu=1,
            target_gpu=0,
            source_buffer=source_buffer,
            target_buffer=target_buffer,
            size=transfer_size
        )
        
        if success:
            print("✓ transfer_gpu_to_gpu returned success")
        else:
            print("✗ transfer_gpu_to_gpu returned failure")
            return False
        
        # Verify data after transfer
        print("\n[Step 6] Verifying data after transfer...")
        torch.cuda.set_device(0)
        after_sum = target_tensor.sum().item()
        print(f"  GPU 0 target tensor sum: {after_sum:.2f}")
        
        # Check if data matches
        if torch.allclose(target_tensor, source_tensor.to('cuda:0')):
            print("✓ Data verification PASSED! Data matches perfectly.")
            return True
        else:
            print("✗ Data verification FAILED! Data mismatch.")
            print(f"  Expected sum: {source_sum:.2f}, Got: {after_sum:.2f}")
            return False
            
    except Exception as e:
        print(f"✗ Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if manager:
            manager.shutdown()


def test_shutdown():
    """Test shutdown method"""
    print("\n" + "="*60)
    print("TEST 3: Shutdown Method")
    print("="*60)
    
    config = SimpleConfig()
    
    try:
        print("\n[Step 1] Creating TransferEngineManager...")
        manager = TransferEngineManager(config)
        
        if not manager.initialized:
            print("✗ Manager not initialized, skipping shutdown test")
            return False
        
        print("✓ Manager initialized")
        print(f"  Initialized: {manager.initialized}")
        print(f"  Engine: {manager.engine is not None}")
        
        print("\n[Step 2] Calling shutdown...")
        shutdown_success = manager.shutdown()
        
        if shutdown_success:
            print("✓ shutdown returned success")
        else:
            print("✗ shutdown returned failure")
            return False
        
        print("\n[Step 3] Verifying shutdown state...")
        print(f"  Initialized after shutdown: {manager.initialized} (should be False)")
        print(f"  Engine after shutdown: {manager.engine} (should be None)")
        
        if not manager.initialized and manager.engine is None:
            print("✓ Shutdown state verification PASSED!")
            return True
        else:
            print("✗ Shutdown state verification FAILED!")
            return False
            
    except Exception as e:
        print(f"✗ Exception during shutdown test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_transfers():
    """Test multiple transfers"""
    print("\n" + "="*60)
    print("TEST 4: Multiple Transfers")
    print("="*60)
    
    config = SimpleConfig()
    manager = None
    
    try:
        print("\n[Step 1] Initializing TransferEngineManager...")
        manager = TransferEngineManager(config)
        
        if not manager.initialized:
            print("✗ Manager not initialized")
            return False
        
        print("✓ Manager initialized")
        
        # Test parameters
        num_transfers = 3
        chunk_size = 1024 * 64  # 256KB per chunk
        transfer_size = chunk_size * 4  # bytes
        
        print(f"\n[Step 2] Testing {num_transfers} transfers...")
        
        results = []
        for i in range(num_transfers):
            print(f"\n  Transfer {i+1}/{num_transfers}:")
            
            # Allocate memory
            torch.cuda.set_device(0)
            target_tensor = torch.zeros(chunk_size, dtype=torch.float32, device='cuda:0')
            target_buffer = target_tensor.data_ptr()
            
            torch.cuda.set_device(1)
            source_tensor = torch.full((chunk_size,), float(i+1), dtype=torch.float32, device='cuda:1')
            source_buffer = source_tensor.data_ptr()
            
            # Perform transfer
            success = manager.transfer_gpu_to_gpu(
                target_hostname="localhost",
                source_gpu=1,
                target_gpu=0,
                source_buffer=source_buffer,
                target_buffer=target_buffer,
                size=transfer_size
            )
            
            # Verify
            torch.cuda.set_device(0)
            expected_value = float(i+1)
            actual_value = target_tensor[0].item()
            
            if success and abs(actual_value - expected_value) < 0.001:
                print(f"    ✓ Transfer successful, data verified")
                results.append(True)
            else:
                print(f"    ✗ Transfer failed or data mismatch")
                print(f"      Expected: {expected_value}, Got: {actual_value}")
                results.append(False)
        
        all_passed = all(results)
        if all_passed:
            print(f"\n✓ All {num_transfers} transfers PASSED!")
        else:
            print(f"\n✗ Some transfers FAILED!")
            print(f"  Passed: {sum(results)}/{num_transfers}")
        
        return all_passed
            
    except Exception as e:
        print(f"✗ Exception during multiple transfers test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if manager:
            manager.shutdown()


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TransferEngineManager Test Suite")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Test environment check failed!")
        return False
    
    # Run tests
    test_results = {}
    
    try:
        test_results["Initialization"] = test_transfer_engine_manager_initialization()
    except Exception as e:
        print(f"\n❌ Initialization test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        test_results["Initialization"] = False
    
    try:
        test_results["transfer_gpu_to_gpu"] = test_transfer_gpu_to_gpu()
    except Exception as e:
        print(f"\n❌ transfer_gpu_to_gpu test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        test_results["transfer_gpu_to_gpu"] = False
    
    try:
        test_results["Shutdown"] = test_shutdown()
    except Exception as e:
        print(f"\n❌ Shutdown test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        test_results["Shutdown"] = False
    
    try:
        test_results["Multiple Transfers"] = test_multiple_transfers()
    except Exception as e:
        print(f"\n❌ Multiple Transfers test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        test_results["Multiple Transfers"] = False
    
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
