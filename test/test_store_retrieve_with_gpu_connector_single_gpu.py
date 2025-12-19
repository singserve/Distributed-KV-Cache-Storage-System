#!/usr/bin/env python3
"""
简单的store和retrieve测试，使用相同的slot mapping
"""

import torch
import sys
import os
import random

# 添加路径以便导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmcache.test.test_cache_engine_system import TestCacheEngine
from lmcache.config import LMCacheEngineMetadata
from lmcache.test.test_config import TestConfig

def generate_kv_cache_paged_list_tensors(
    num_blocks, device, block_size=16, dtype=torch.float16, use_mla=False
):
    """生成分页KV cache数据"""
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

def check_paged_kv_cache_equal(kvcache1, kvcache2, slot_mapping):
    """
    检查两个kvcache在slot_mapping指定的位置是否相等
    
    Args:
        kvcache1: 第一个kvcache列表
        kvcache2: 第二个kvcache列表  
        slot_mapping: slot mapping tensor
    """
    num_layers = len(kvcache1)
    block_size = kvcache1[0].shape[2]  # block_size维度
    num_blocks = kvcache1[0].shape[1]  # num_blocks维度
    
    all_match = True
    
    for token_idx in range(len(slot_mapping)):
        slot = slot_mapping[token_idx].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        
        for layer_idx in range(num_layers):
            # 提取两个kvcache中相同位置的数据
            data1 = kvcache1[layer_idx][:, block_idx, block_offset, :, :]
            data2 = kvcache2[layer_idx][:, block_idx, block_offset, :, :]
            
            # 计算差异
            diff = torch.abs(data1 - data2)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            # 打印检验结果的数值
            print(f"  Token {token_idx}, Layer {layer_idx}:")
            print(f"    Slot={slot}, Block={block_idx}, Offset={block_offset}")
            print(f"    Max diff={max_diff:.6e}, Mean diff={mean_diff:.6e}")
            
            if max_diff > 1e-5:
                print(f"    NOT MATCHED!")
                # 打印更多样本数值用于调试
                print(f"    Sample comparisons:")
                # 打印前3个head的前3个位置
                for k in range(min(2, data1.shape[0])):  # k维度 (0,1)
                    for h in range(min(3, data1.shape[1])):  # head维度
                        for d in range(min(3, data1.shape[2])):  # head_size维度
                            val1 = data1[k, h, d].item()
                            val2 = data2[k, h, d].item()
                            diff_val = abs(val1 - val2)
                            print(f"      k={k}, h={h}, d={d}: data1={val1:.6f}, data2={val2:.6f}, diff={diff_val:.6e}")
                all_match = False
            else:
                print(f"    MATCHED")
                # 即使匹配也打印一些样本数值
                if token_idx < 2 and layer_idx < 2:  # 限制打印数量，避免输出太多
                    print(f"    Sample (matched): data1[0,0,0]={data1[0,0,0].item():.6f}, data2[0,0,0]={data2[0,0,0].item():.6f}")
    
    return all_match

def test_with_simulated_kv_cache():
    """使用模拟KV cache进行测试"""
    print("=" * 60)
    print("test_with_simulated_kv_cache")
    print("=" * 60)
    
    try:
        # 1. 创建配置
        print("\n1. config...")
        config_path = "./test_system_config_gpu0.yaml"
        try:
            config = TestConfig.from_file(config_path)
            print(f"✓ successfully build config from file: {config_path}")
        except Exception as e:
            print(f"fallback to default config due to: {e}")
            config = TestConfig()
            config.enable_gpu_vram_pool = False
            config.enable_gpu_vram_segments = True
            config.gpu_vram_segment_size_mb = 256
        
        # 2. 创建metadata
        print("\n2. create metadata...")
        num_layers = 4
        block_size = 16
        num_kv_heads = 8
        head_size = 64
        
        metadata = LMCacheEngineMetadata(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            fmt="vllm",
            kv_dtype=torch.float16,
            kv_shape=(num_layers, 2, block_size, num_kv_heads, head_size)
        )
        
        # 3. 创建模拟KV cache
        print("\n3. create simulated KV cache...")
        num_blocks = 2
        num_tokens = num_blocks * block_size  # 32
        
        # store kvcache
        store_kv_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, "cuda:0", block_size
        )
        
        # retrieve kvcache（不同地址）
        retrieve_kv_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, "cuda:0", block_size
        )
        
        print(f"Store kvcache[0] addr: {hex(store_kv_cache[0].data_ptr())}")
        print(f"Retrieve kvcache[0] addr: {hex(retrieve_kv_cache[0].data_ptr())}")
        print(f"地址是否相同: {store_kv_cache[0].data_ptr() == retrieve_kv_cache[0].data_ptr()}")
        
        # 4. 生成slot mapping（使用int64类型，因为C++内核期望int64_t）
        print("\n4. generate slot mapping（int64）...")
        slot_mapping = torch.tensor(
            random.sample(range(0, num_blocks * block_size), num_tokens),
            dtype=torch.int64,  # 改为int64，因为C++内核期望int64_t
            device="cuda:0"
        )
        
        print(f"Slot mapping shape: {slot_mapping.shape}")
        print(f"Slot mapping value: {slot_mapping.tolist()}")
        
        # 5. 创建test tokens
        print("\n5. create test tokens...")
        tokens = list(range(num_tokens))
        
        # 6. 创建TestCacheEngine
        print("\n6. create TestCacheEngine...")
        engine = TestCacheEngine(config, metadata, gpu_connector=None)
        
        # 7. 测试store
        print("\n7. test store...")
        try:
            engine.store(
                tokens=tokens,
                mask=None,
                kvcaches=store_kv_cache,
                slot_mapping=slot_mapping,
                offset=0
            )
            print("store successful")
        except Exception as e:
            print(f"store failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 8. 测试retrieve
        print("\n8. test retrieve...")
        try:
            ret_mask = engine.retrieve(
                tokens=tokens,
                mask=None,
                kvcaches=retrieve_kv_cache,
                slot_mapping=slot_mapping
            )
            
            retrieved_tokens = ret_mask.sum().item()
            print(f"retrieve successful")
            print(f"retrieved tokens: {retrieved_tokens}/{len(tokens)}")
            
        except Exception as e:
            print(f"retrieve failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 9. 验证数据完整性
        print("\n9. check integrity...")
        print("  - check if store_kv_cache and retrieve_kv_cache equals with slot_mapping")
        
        all_match = check_paged_kv_cache_equal(store_kv_cache, retrieve_kv_cache, slot_mapping)
        
        if all_match:
            print("  - data all match")
        else:
            print("  - data not match")
        
        # 10. 清理资源
        print("\n10. clear...")
        engine.close()
        print("engine closed")
        
        return all_match
        
    except Exception as e:
        print(f"\ntest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_simulated_kv_cache()
    if success:
        print("\npassed")
    else:
        print("\nfailed")

