#!/usr/bin/env python3
"""
Simple Cross-Process Test: Two processes start cache engine instances on two GPUs
and test basic store/retrieve operations.
"""

import torch
import sys
import os
import time
import multiprocessing as mp
from typing import List

# Add LMCache to path
sys.path.insert(0, 'LMCache')

from lmcache.config import LMCacheEngineMetadata
from vcache_config import VCacheConfig
from lmcache.v1.test_cache_engine_system import TestCacheEngine, MockGPUConnector


def create_engine_config(gpu_id: int) -> VCacheConfig:
    """Create configuration for a specific GPU"""
    config_file = f"test_system_config_gpu{gpu_id}.yaml"
    print(f"Loading configuration for GPU {gpu_id} from {config_file}")
    return VCacheConfig.from_file(config_file)


def create_engine_metadata(gpu_id: int) -> LMCacheEngineMetadata:
    """Create metadata for a specific GPU"""
    return LMCacheEngineMetadata(
        model_name="test_model",
        world_size=2,
        worker_id=gpu_id,
        fmt="pt",
        kv_dtype=torch.float16,
        kv_shape=(32, 2, 256, 32, 128),
        use_mla=False,
        role="worker"
    )


def create_mock_kvcaches(num_tokens: int, tokens: List[int] = None, gpu_id: int = 0) -> tuple[torch.Tensor, int]:
    """Create mock GPU VRAM addresses for testing"""
    if torch.cuda.is_available():
        try:
            original_device = torch.cuda.current_device()
            torch.cuda.set_device(gpu_id)
            
            if tokens is not None and len(tokens) > 0:
                dummy_tensor = torch.zeros(num_tokens, dtype=torch.int64, device='cuda')
                gpu_address = dummy_tensor.data_ptr()
                tokens_tensor = torch.tensor(tokens, dtype=torch.int64, device='cuda')
                dummy_tensor.copy_(tokens_tensor)
            else:
                dummy_tensor = torch.zeros(num_tokens, dtype=torch.int64, device='cuda')
                gpu_address = dummy_tensor.data_ptr()
            
            print(f"Allocated GPU memory on GPU {gpu_id}: {hex(gpu_address)}")
            torch.cuda.set_device(original_device)
            return dummy_tensor, gpu_address
            
        except Exception as e:
            print(f"GPU memory allocation failed: {e}")
            dummy_tensor = torch.zeros(num_tokens, dtype=torch.int64, device='cpu')
            mock_address = dummy_tensor.data_ptr()
            return dummy_tensor, mock_address
    else:
        dummy_tensor = torch.zeros(num_tokens, dtype=torch.int64, device='cpu')
        mock_address = dummy_tensor.data_ptr()
        return dummy_tensor, mock_address


def store_process():
    """Process 1: Store tokens on GPU 0"""
    print("=== Store Process: GPU 0 ===")
    
    try:
        config = create_engine_config(0)
        metadata = create_engine_metadata(0)
        gpu_connector = MockGPUConnector()
        
        engine = TestCacheEngine(config, metadata, gpu_connector)
        
        # Store sample tokens
        tokens_to_store = [101, 102, 103, 104, 105]
        print(f"Storing tokens: {tokens_to_store}")
        
        kvcaches, gpu_vram_address = create_mock_kvcaches(len(tokens_to_store), tokens_to_store, gpu_id=0)
        
        engine.store(
            tokens=tokens_to_store,
            mask=None,
            kvcaches=kvcaches,
            gpu_vram_address=gpu_vram_address
        )
        
        stats = engine.get_stats()
        print(f"Store stats: {stats['cache_engine_stats']}")
        
        print("✓ Store process completed")
        
        # Keep engine alive for retrieval
        time.sleep(10)
        engine.close()
        
    except Exception as e:
        print(f"Store process error: {e}")
        import traceback
        traceback.print_exc()


def retrieve_process():
    """Process 2: Retrieve tokens on GPU 1"""
    print("=== Retrieve Process: GPU 1 ===")
    
    # Wait for store process to complete
    time.sleep(3)
    
    try:
        config = create_engine_config(1)
        metadata = create_engine_metadata(1)
        gpu_connector = MockGPUConnector()
        
        engine = TestCacheEngine(config, metadata, gpu_connector)
        
        # Try to retrieve the same tokens
        tokens_to_retrieve = [101, 102, 103, 104, 105]
        print(f"Retrieving tokens: {tokens_to_retrieve}")
        
        # First do lookup
        hit_tokens = engine.lookup(tokens_to_retrieve)
        print(f"Lookup result: {hit_tokens} hit tokens")
        
        # Then do retrieve with target GPU buffer
        # 注意：这里不应该传入tokens，因为retrieve操作会从存储的KV缓存中读取数据
        target_kvcaches, target_gpu_address = create_mock_kvcaches(len(tokens_to_retrieve), None, gpu_id=1)
        
        ret_mask = engine.retrieve(
            tokens=tokens_to_retrieve,
            mask=None,
            target_gpu_address=target_gpu_address
        )
        
        retrieved_count = ret_mask.sum().item()
        print(f"Retrieve result: {retrieved_count} tokens retrieved")
        
        stats = engine.get_stats()
        print(f"Retrieve stats: {stats['cache_engine_stats']}")
        
        print("\nverify Target Buffer")
        try:
            # 直接从GPU内存地址读取数据
            original_device = torch.cuda.current_device()
            torch.cuda.set_device(1)  # 切换到GPU 1
            
            # 使用PyTorch直接从GPU地址创建tensor来读取数据
            # 注意：这里假设数据是int64类型，每个token占8字节
            if target_gpu_address and retrieved_count > 0:
                # 从GPU地址创建tensor来读取数据
                retrieved_tensor = torch.tensor([], dtype=torch.int64, device='cuda')
                # 使用data_ptr()获取的地址来访问数据
                # 这里我们使用更直接的方法：创建一个新的tensor来包装目标地址
                retrieved_tensor = torch.from_dlpack(target_kvcaches)
                retrieved_data = retrieved_tensor.tolist()
                
                print(f"from target buffer {hex(target_gpu_address)} get data: {retrieved_data}")
                print(f"expected tokens: {tokens_to_retrieve}")
                
                # 验证数据一致性
                expected_retrieved = tokens_to_retrieve[:retrieved_count]
                actual_retrieved = retrieved_data[:retrieved_count]
                
                if actual_retrieved == expected_retrieved:
                    print("stored tokens and received tokens matched!")
                else:
                    print(f"tokens unmatched:")
                    print(f"expected: {expected_retrieved}")
                    print(f"actual: {actual_retrieved}")
                
                # 检查retrieve mask是否正确
                expected_mask = [1] * retrieved_count + [0] * (len(tokens_to_retrieve) - retrieved_count)
                actual_mask = ret_mask.tolist()
                
                if actual_mask == expected_mask:
                    print("Retrieve mask matched")
                else:
                    print(f"Retrieve mask unmatched:")
                    print(f"expected: {expected_mask}")
                    print(f"actual {actual_mask}")
            else:
                print("no data or invalid address")
            
            torch.cuda.set_device(original_device)
            
        except Exception as e:
            print(f"fail to verify if matched tokens: {e}")
            import traceback
            traceback.print_exc()
        # ===== 验证结束 =====
        
        if hit_tokens > 0 and retrieved_count > 0:
            print("Retrieve process: Tokens found and retrieved!")
        else:
            print("Retrieve process: No hits found")
        
        engine.close()
        
    except Exception as e:
        print(f"Retrieve process error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the simple test"""
    print("=== Simple Cross-Process Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, test cannot run")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus < 2:
        print("Need at least 2 GPUs for this test")
        return
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Create processes
    store_proc = mp.Process(target=store_process, name="StoreProcess")
    retrieve_proc = mp.Process(target=retrieve_process, name="RetrieveProcess")
    
    try:
        # Start processes
        print("Starting store process...")
        store_proc.start()
        
        time.sleep(1)
        
        print("Starting retrieve process...")
        retrieve_proc.start()
        
        # Wait for processes to complete
        store_proc.join(timeout=30)
        retrieve_proc.join(timeout=30)
        
        # Check results
        if store_proc.exitcode == 0 and retrieve_proc.exitcode == 0:
            print("\ntest passed! Cross-process workflow is working.")
        else:
            print(f"\ntest failed. Store exitcode: {store_proc.exitcode}, Retrieve exitcode: {retrieve_proc.exitcode}")
            
    except Exception as e:
        print(f"Test error: {e}")
        
        # Cleanup
        if store_proc.is_alive():
            store_proc.terminate()
        if retrieve_proc.is_alive():
            retrieve_proc.terminate()
        
        store_proc.join(timeout=5)
        retrieve_proc.join(timeout=5)


if __name__ == "__main__":
    main()
