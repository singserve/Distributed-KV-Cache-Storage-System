#!/usr/bin/env python3
"""
Cross-GPU Store and Retrieve Test
Store logic and retrieve logic are placed in two processes, using GPU0 and GPU1 respectively
Verify data integrity after cross-GPU transfer
"""

import sys
import os

# Save original sys.path
original_sys_path = sys.path.copy()

# Temporarily remove LMCache directory to avoid circular imports
lmcache_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if lmcache_path in sys.path:
    sys.path.remove(lmcache_path)

# Now import torch and other standard library modules
import torch
import random
import time
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event

# Restore sys.path
sys.path = original_sys_path

# Set multiprocessing start method to 'spawn' to support CUDA
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcache.vcache_engine_system import VCacheEngine
from lmcache.config import LMCacheEngineMetadata
from lmcache.vcache.vcache_config import VCacheConfig



def generate_kv_cache_paged_list_tensors(
    num_blocks, device, block_size=16, dtype=torch.float16, use_mla=False
):
    """Generate paged KV cache data"""
    ret = []
    num_layers = 4
    num_heads = 1 if use_mla else 8
    head_size = 64
    
    if use_mla:
        shape = [num_blocks, block_size, head_size]
    else:
        shape = [2, num_blocks, block_size, num_heads, head_size]
    
    for i in range(num_layers):
        kv = torch.rand(shape, dtype=dtype, device=device)
        ret.append(kv)
    
    return ret


def store_process(gpu_id, config, metadata, tokens, slot_mapping_data, kv_cache_template, ready_event, done_event, result_queue, sample_queue, num_samples_to_check=5):
    """
    Store process: Execute store operation on specified GPU
    
    Args:
        gpu_id: GPU ID (0 or 1)
        config: VCacheConfig configuration
        metadata: LMCacheEngineMetadata
        tokens: Token list
        slot_mapping_data: Slot mapping data (list on CPU)
        kv_cache_template: KV cache template (for reference shape and data type)
        ready_event: Ready event
        done_event: Done event
        result_queue: Result queue
        sample_queue: Queue for passing store samples to parent process for comparison
        num_samples_to_check: Number of samples to check (must match retrieve process)
    """
    try:
        print(f"[Store Process GPU{gpu_id}] Starting store process...")
        
        # 设置当前进程的GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        
        # 在GPU上创建slot mapping副本
        slot_mapping_gpu = torch.tensor(
            slot_mapping_data,
            dtype=torch.int64,
            device=device
        )
        
        # 使用generate_kv_cache_paged_list_tensors创建store用的KV cache
        # 从模板获取参数
        num_blocks = kv_cache_template[0].shape[1]  # 获取block数量
        block_size = kv_cache_template[0].shape[2]  # 获取block大小
        
        store_kv_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, device, block_size, kv_cache_template[0].dtype
        )
        
        print(f"[Store Process GPU{gpu_id}] Store KV cache created using generate_kv_cache_paged_list_tensors on GPU{gpu_id}")
        print(f"[Store Process GPU{gpu_id}] Store KV cache[0] address: {hex(store_kv_cache[0].data_ptr())}")
        print(f"[Store Process GPU{gpu_id}] Store KV cache shape: {store_kv_cache[0].shape}")
        
        print(f"[Store Process GPU{gpu_id}] Creating VCacheEngine...")
        
        # 创建VCacheEngine
        engine = VCacheEngine(config, metadata, gpu_connector=None)
        
        print(f"[Store Process GPU{gpu_id}] Performing store operation...")
        
        # 执行store操作
        start_time = time.time()
        engine.store(
            tokens=tokens,
            mask=None,
            kvcaches=store_kv_cache,
            slot_mapping=slot_mapping_gpu,
            offset=0
        )
        store_time = time.time() - start_time
        
        print(f"[Store Process GPU{gpu_id}] Store completed in {store_time:.4f} seconds")
        
        # 根据slot mapping保存一些数据样本到队列中，用于后续比较
        # 我们保存前num_samples_to_check个token对应的slot位置的数据
        num_samples = min(num_samples_to_check, len(slot_mapping_gpu))
        store_samples = {}
        sample_slots = []  # 记录检查了哪些slot
        
        for i in range(num_samples):
            slot = slot_mapping_gpu[i].item()
            block_idx = slot // block_size
            block_offset = slot % block_size
            
            # 保存这个slot位置的数据
            # 我们检查第一个layer的第一个head的第一个head_size
            data = store_kv_cache[0][0, block_idx, block_offset, 0, 0].item()
            store_samples[f"token_{i}_slot_{slot}"] = data
            sample_slots.append(slot)
        
        # 将样本数据通过队列传递给父进程，用于后续与retrieve数据比较
        # 同时传递slot信息，确保retrieve检查相同的slot位置
        sample_queue.put({
            'store_samples': store_samples,
            'gpu_id': gpu_id,
            'sample_slots': sample_slots,  # 传递检查的slot列表
            'num_samples': num_samples
        })
        
        # 通知retrieve进程可以开始
        ready_event.set()
        print(f"[Store Process GPU{gpu_id}] Signaled retrieve process to start")
        
        # 等待retrieve进程完成
        done_event.wait()
        print(f"[Store Process GPU{gpu_id}] Retrieve process completed")
        
        stats_sum = engine.get_stats_summary()
        print(f"[Retrieve Process GPU{gpu_id}] Engine stats summary: {stats_sum}")
        
        # 清理资源
        engine.close()
        
        # 将结果放入队列
        result_queue.put({
            'gpu_id': gpu_id,
            'status': 'success',
            'store_time': store_time,
            'store_kv_cache_ptr': hex(store_kv_cache[0].data_ptr()) if store_kv_cache else 'N/A',
            'store_data_sample': store_kv_cache[0][0, 0, 0, 0, 0].item() if len(store_kv_cache) > 0 else 0.0,
        })
        
        print(f"[Store Process GPU{gpu_id}] Store process completed successfully")
        
    except Exception as e:
        print(f"[Store Process GPU{gpu_id}] Error in store process: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({
            'gpu_id': gpu_id,
            'status': 'error',
            'error': str(e)
        })


def retrieve_process(gpu_id, config, metadata, tokens, slot_mapping_data, kv_cache_template, 
                     ready_event, done_event, result_queue, store_gpu_id=0, sample_slots=None):
    """
    Retrieve进程：在指定GPU上执行retrieve操作
    
    Args:
        gpu_id: GPU ID (0或1)
        config: VCacheConfig配置
        metadata: LMCacheEngineMetadata
        tokens: token列表
        slot_mapping_data: slot mapping数据（CPU上的列表）
        kv_cache_template: KV cache模板（用于参考形状和数据类型）
        ready_event: 准备就绪事件
        done_event: 完成事件
        result_queue: 结果队列
        store_gpu_id: store操作所在的GPU ID
        sample_slots: 要检查的slot列表（与store进程检查相同的slot）
    """
    try:
        print(f"[Retrieve Process GPU{gpu_id}] Starting retrieve process...")
        
        # 设置当前进程的GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        
        # 等待store进程完成
        print(f"[Retrieve Process GPU{gpu_id}] Waiting for store process on GPU{store_gpu_id} to complete...")
        ready_event.wait()
        print(f"[Retrieve Process GPU{gpu_id}] Store process completed, waiting for 1 second before starting retrieve...")
        
        # 等待一段时间，确保store操作完全完成
        time.sleep(1.0)
        print(f"[Retrieve Process GPU{gpu_id}] Starting retrieve after waiting...")
        
        # 在GPU上创建slot mapping副本
        slot_mapping_gpu = torch.tensor(
            slot_mapping_data,
            dtype=torch.int64,
            device=device
        )
        
        # 使用generate_kv_cache_paged_list_tensors创建retrieve用的KV cache
        # 从模板获取参数
        num_blocks = kv_cache_template[0].shape[1]  # 获取block数量
        block_size = kv_cache_template[0].shape[2]  # 获取block大小
        
        retrieve_kv_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, device, block_size, kv_cache_template[0].dtype
        )
        
        print(f"[Retrieve Process GPU{gpu_id}] Retrieve KV cache created using generate_kv_cache_paged_list_tensors on GPU{gpu_id}")
        print(f"[Retrieve Process GPU{gpu_id}] Retrieve KV cache[0] address: {hex(retrieve_kv_cache[0].data_ptr())}")
        print(f"[Retrieve Process GPU{gpu_id}] Retrieve KV cache shape: {retrieve_kv_cache[0].shape}")
        
        print(f"[Retrieve Process GPU{gpu_id}] Creating VCacheEngine...")
        
        # 创建VCacheEngine
        engine = VCacheEngine(config, metadata, gpu_connector=None)
        
        print(f"[Retrieve Process GPU{gpu_id}] Performing retrieve operation...")
        
        # 执行retrieve操作
        start_time = time.time()
        ret_mask = engine.retrieve(
            tokens=tokens,
            mask=None,
            kvcaches=retrieve_kv_cache,
            slot_mapping=slot_mapping_gpu
        )
        retrieve_time = time.time() - start_time
        
        retrieved_tokens = ret_mask.sum().item() if ret_mask is not None else 0
        
        print(f"[Retrieve Process GPU{gpu_id}] Retrieve completed in {retrieve_time:.4f} seconds")
        print(f"[Retrieve Process GPU{gpu_id}] Retrieved tokens: {retrieved_tokens}/{len(tokens)}")
        
        # 验证数据：根据sample_slots检查特定的slot位置
        # 如果没有提供sample_slots，则检查前5个token对应的slot
        if sample_slots is None:
            sample_slots = []
            num_samples = min(5, len(slot_mapping_gpu))
            for i in range(num_samples):
                sample_slots.append(slot_mapping_gpu[i].item())
        
        retrieve_samples = {}
        
        for i, slot in enumerate(sample_slots):
            block_idx = slot // block_size
            block_offset = slot % block_size
            
            # 检查这个slot位置的数据
            # retrieve_kv_cache[0]的形状是 [2, num_blocks, block_size, num_heads, head_size]
            # 我们检查第一个layer的第一个head的第一个head_size
            data = retrieve_kv_cache[0][0, block_idx, block_offset, 0, 0].item()
            retrieve_samples[f"token_{i}_slot_{slot}"] = data
                # 清理资源
        
        stats_sum = engine.get_stats_summary()
        print(f"[Retrieve Process GPU{gpu_id}] Engine stats summary: {stats_sum}")
        engine.close()
        
        # 通知store进程已完成
        done_event.set()
        
        # 将结果放入队列
        result_queue.put({
            'gpu_id': gpu_id,
            'status': 'success',
            'retrieve_time': retrieve_time,
            'retrieved_tokens': retrieved_tokens,
            'retrieve_kv_cache_ptr': hex(retrieve_kv_cache[0].data_ptr()) if retrieve_kv_cache else 'N/A',
            'retrieve_data_sample': retrieve_kv_cache[0][0, 0, 0, 0, 0].item() if len(retrieve_kv_cache) > 0 else 0.0,
            'retrieve_samples': retrieve_samples,
            'sample_slots': sample_slots  # 传递检查的slot列表，用于验证
        })
        
        print(f"[Retrieve Process GPU{gpu_id}] Retrieve process completed successfully")
        
    except Exception as e:
        print(f"[Retrieve Process GPU{gpu_id}] Error in retrieve process: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({
            'gpu_id': gpu_id,
            'status': 'error',
            'error': str(e)
        })


def test_cross_gpu_store_retrieve():
    """测试跨GPU存储和检索"""
    print("=" * 80)
    print("Cross-GPU Store and Retrieve Test")
    print("Store on GPU0, Retrieve on GPU1")
    print("=" * 80)
    
    try:
        # 检查可用的GPU数量
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        
        if num_gpus < 2:
            print("ERROR: Need at least 2 GPUs for cross-GPU test")
            return False
        
        # 1. 创建配置 - GPU0配置
        print("\n1. Creating configuration for GPU0...")
        config_path_gpu0 = "../vcache_config_gpu0.yaml"
        try:
            config_gpu0 = VCacheConfig.from_file(config_path_gpu0)
            config_gpu0.connector_role = "worker"  
            print(f"✓ Successfully loaded config from file: {config_path_gpu0}")
        except Exception as e:
            print(f"⚠ Falling back to default config for GPU0 due to: {e}")
            config_gpu0 = VCacheConfig()
            config_gpu0.enable_gpu_vram_pool = True  # 启用GPU VRAM池以支持跨GPU传输
            config_gpu0.enable_gpu_vram_segments = True
            config_gpu0.gpu_vram_segment_size_mb = 256
            config_gpu0.connector_role = "worker"    #worker
        
        # 2. 创建配置 - GPU1配置
        print("\n2. Creating configuration for GPU1...")
        config_path_gpu1 = "../vcache_config_gpu1.yaml"
        try:
            config_gpu1 = VCacheConfig.from_file(config_path_gpu1)
            config_gpu1.connector_role = "worker"  
            print(f"✓ Successfully loaded config from file: {config_path_gpu1}")
        except Exception as e:
            print(f"⚠ Falling back to default config for GPU1 due to: {e}")
            config_gpu1 = VCacheConfig()
            config_gpu1.enable_gpu_vram_pool = True  # 启用GPU VRAM池以支持跨GPU传输
            config_gpu1.enable_gpu_vram_segments = True
            config_gpu1.gpu_vram_segment_size_mb = 256
            config_gpu1.connector_role = "worker"  
        
        # 3. 创建metadata
        print("\n3. Creating metadata...")
        num_layers = 4
        block_size = 16
        num_kv_heads = 8
        head_size = 64
        
        metadata_gpu0 = LMCacheEngineMetadata(
            model_name="test_model",
            world_size=2,  # 总共有2个worker
            worker_id=0,  # GPU0的worker_id
            fmt="vllm",
            kv_dtype=torch.float16,
            kv_shape=(num_layers, 2, block_size, num_kv_heads, head_size)
        )
        
        metadata_gpu1 = LMCacheEngineMetadata(
            model_name="test_model",
            world_size=2,  # 总共有2个worker
            worker_id=1,  # GPU1的worker_id
            fmt="vllm",
            kv_dtype=torch.float16,
            kv_shape=(num_layers, 2, block_size, num_kv_heads, head_size)
        )
        
        # 4. 创建测试数据
        print("\n4. Creating test data...")
        num_blocks = 32  # 512 tokens / 16 block_size = 32 blocks
        num_tokens = num_blocks * block_size  # 512
        
        # 生成token列表
        tokens = list(range(num_tokens))
        
        # 生成slot mapping数据（CPU上的列表）
        # 使用随机映射
        slot_mapping_data = random.sample(range(0, num_blocks * block_size), num_tokens)
        
        print(f"Number of tokens: {num_tokens}")
        print(f"Number of blocks: {num_blocks}")
        print(f"Slot mapping data length: {len(slot_mapping_data)}")
        print(f"Slot mapping sample: {slot_mapping_data[:5]}... (random mapping)")
        
        # 决定检查哪些slot（前5个token对应的slot）
        num_samples_to_check = 5
        sample_slots = slot_mapping_data[:num_samples_to_check]
        print(f"Sample slots to check: {sample_slots}")
        
        # 生成KV cache模板（在GPU0上）
        kv_cache_template = generate_kv_cache_paged_list_tensors(
            num_blocks, "cuda:0", block_size
        )
        
        print(f"KV cache template created on GPU0")
        print(f"KV cache[0] address: {hex(kv_cache_template[0].data_ptr())}")
        
        # 5. 创建进程间通信对象
        print("\n5. Setting up inter-process communication...")
        ready_event = Event()
        done_event = Event()
        result_queue = Queue()
        sample_queue = Queue()  # 用于传递store样本数据的队列
        
        # 6. 创建并启动进程
        print("\n6. Starting processes...")
        
        # Store进程在GPU0上，使用GPU0的配置和metadata
        store_proc = Process(
            target=store_process,
            args=(0, config_gpu0, metadata_gpu0, tokens, slot_mapping_data, kv_cache_template, 
                  ready_event, done_event, result_queue, sample_queue, num_samples_to_check)
        )
        
        # Retrieve进程在GPU1上，使用GPU1的配置和metadata
        retrieve_proc = Process(
            target=retrieve_process,
            args=(1, config_gpu1, metadata_gpu1, tokens, slot_mapping_data, kv_cache_template,
                  ready_event, done_event, result_queue, 0, sample_slots)
        )
        
        # 启动进程
        store_proc.start()
        retrieve_proc.start()
        
        # 7. 等待进程完成并收集结果
        print("\n7. Waiting for processes to complete...")
        store_proc.join()
        retrieve_proc.join()
        
        # 8. 收集结果
        print("\n8. Collecting results...")
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # 从sample_queue获取store的样本数据
        store_samples_data = None
        if not sample_queue.empty():
            store_samples_data = sample_queue.get()
        
        # 9. 分析结果
        print("\n9. Test Results:")
        print("-" * 40)
        
        store_result = None
        retrieve_result = None
        
        for result in results:
            if result['gpu_id'] == 0:
                store_result = result
            elif result['gpu_id'] == 1:
                retrieve_result = result
        
        success = True
        
        if store_result and store_result['status'] == 'success':
            print(f"Store Process (GPU0): SUCCESS")
            print(f"  Store time: {store_result['store_time']:.4f} seconds")
            print(f"  Store KV cache address: {store_result.get('store_kv_cache_ptr', 'N/A')}")
        else:
            print(f"Store Process (GPU0): FAILED")
            if store_result and 'error' in store_result:
                print(f"  Error: {store_result['error']}")
            success = False
        
        if retrieve_result and retrieve_result['status'] == 'success':
            print(f"Retrieve Process (GPU1): SUCCESS")
            print(f"  Retrieve time: {retrieve_result['retrieve_time']:.4f} seconds")
            print(f"  Retrieve KV cache address: {retrieve_result.get('retrieve_kv_cache_ptr', 'N/A')}")
            print(f"  Retrieved tokens: {retrieve_result.get('retrieved_tokens', 0)}/{num_tokens}")
            
            # 如果有store样本数据，与retrieve样本数据进行比较
            if store_samples_data and 'retrieve_samples' in retrieve_result:
                print(f"\n  Data Comparison (based on slot_mapping):")
                print(f"  {'Token/Slot':<20} {'Store Value':<15} {'Retrieve Value':<15} {'Match':<10}")
                print(f"  {'-'*20:<20} {'-'*15:<15} {'-'*15:<15} {'-'*10:<10}")
                
                store_samples = store_samples_data['store_samples']
                retrieve_samples = retrieve_result['retrieve_samples']
                
                all_match = True
                mismatch_count = 0
                
                for key in store_samples:
                    if key in retrieve_samples:
                        store_val = store_samples[key]
                        retrieve_val = retrieve_samples[key]
                        match = abs(store_val - retrieve_val) < 1e-5
                        
                        if not match:
                            all_match = False
                            mismatch_count += 1
                        
                        match_str = "✓" if match else "✗"
                        print(f"  {key:<20} {store_val:<15.8f} {retrieve_val:<15.8f} {match_str:<10}")
                
                print(f"\n  Data Match Summary:")
                print(f"    All samples match: {'Yes' if all_match else 'No'}")
                print(f"    Mismatch count: {mismatch_count}")
                
                if not all_match:
                    print(f"  ⚠ Data mismatch detected!")
                    success = False
                else:
                    print(f"  ✓ All data samples match perfectly!")
        else:
            print(f"Retrieve Process (GPU1): FAILED")
            if retrieve_result and 'error' in retrieve_result:
                print(f"  Error: {retrieve_result['error']}")
            success = False
        
        # 10. 总体结论
        print("\n10. Overall Test Result:")
        print("-" * 40)
        if success:
            print("✓ CROSS-GPU STORE/RETRIEVE TEST PASSED")
        else:
            print("✗ CROSS-GPU STORE/RETRIEVE TEST FAILED")
        
        return success
        
    except Exception as e:
        print(f"\nERROR in test_cross_gpu_store_retrieve: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Cross-GPU Store and Retrieve Test")
    print("Store on GPU0, Retrieve on GPU1")
    print("=" * 60)
    
    # 运行跨GPU测试
    cross_gpu_success = test_cross_gpu_store_retrieve()
    
    if cross_gpu_success:
        print("\n✓ CROSS-GPU TEST PASSED")
        sys.exit(0)
    else:
        print("\n✗ CROSS-GPU TEST FAILED")
        sys.exit(1)
