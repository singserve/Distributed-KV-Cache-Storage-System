"""
Simple store and retrieve test using the same slot mapping
"""

import torch
import sys
import os
import random

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcache.vcache_engine_system import VCacheEngine
from lmcache.config import LMCacheEngineMetadata
from vcache.vcache_config import VCacheConfig

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

def check_paged_kv_cache_equal(kvcache1, kvcache2, slot_mapping):
    """
    Check if two kvcaches are equal at positions specified by slot_mapping
    
    Args:
        kvcache1: First kvcache list
        kvcache2: Second kvcache list  
        slot_mapping: Slot mapping tensor
    """
    num_layers = len(kvcache1)
    block_size = kvcache1[0].shape[2]  # block_size dimension
    num_blocks = kvcache1[0].shape[1]  # num_blocks dimension
    
    all_match = True
    
    for token_idx in range(len(slot_mapping)):
        slot = slot_mapping[token_idx].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        
        for layer_idx in range(num_layers):
            # Extract data from same position in both kvcaches
            data1 = kvcache1[layer_idx][:, block_idx, block_offset, :, :]
            data2 = kvcache2[layer_idx][:, block_idx, block_offset, :, :]
            
            # Calculate difference
            diff = torch.abs(data1 - data2)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            # Print verification results
            print(f"  Token {token_idx}, Layer {layer_idx}:")
            print(f"    Slot={slot}, Block={block_idx}, Offset={block_offset}")
            print(f"    Max diff={max_diff:.6e}, Mean diff={mean_diff:.6e}")
            
            if max_diff > 1e-5:
                print(f"    NOT MATCHED!")
                # Print more sample values for debugging
                print(f"    Sample comparisons:")
                # Print first 3 heads and first 3 positions
                for k in range(min(2, data1.shape[0])):  # k dimension (0,1)
                    for h in range(min(3, data1.shape[1])):  # head dimension
                        for d in range(min(3, data1.shape[2])):  # head_size dimension
                            val1 = data1[k, h, d].item()
                            val2 = data2[k, h, d].item()
                            diff_val = abs(val1 - val2)
                            print(f"      k={k}, h={h}, d={d}: data1={val1:.6f}, data2={val2:.6f}, diff={diff_val:.6e}")
                all_match = False
            else:
                print(f"    MATCHED")
                # Print some sample values even when matched
                if token_idx < 2 and layer_idx < 2:  # Limit output to avoid too much printing
                    print(f"    Sample (matched): data1[0,0,0]={data1[0,0,0].item():.6f}, data2[0,0,0]={data2[0,0,0].item():.6f}")
    
    return all_match

def test_with_simulated_kv_cache():
    """Test with simulated KV cache"""
    print("=" * 60)
    print("test_with_simulated_kv_cache")
    print("=" * 60)
    
    try:
        # 1. Create configuration
        print("\n1. Creating configuration...")
        config_path = "./vcache_config_gpu0.yaml"
        try:
            config = VCacheConfig.from_file(config_path)
            print(f"✓ Successfully loaded config from file: {config_path}")
        except Exception as e:
            print(f"⚠ Falling back to default config due to: {e}")
            config = VCacheConfig()
            config.enable_gpu_vram_pool = False
            config.enable_gpu_vram_segments = True
            config.gpu_vram_segment_size_mb = 256
        
        # 2. Create metadata
        print("\n2. Creating metadata...")
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
        
        # 3. Create simulated KV cache
        print("\n3. Creating simulated KV cache...")
        num_blocks = 2
        num_tokens = num_blocks * block_size  # 32
        
        # Store kvcache
        store_kv_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, "cuda:0", block_size
        )
        
        # Retrieve kvcache (different address)
        retrieve_kv_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, "cuda:0", block_size
        )
        
        print(f"Store kvcache[0] address: {hex(store_kv_cache[0].data_ptr())}")
        print(f"Retrieve kvcache[0] address: {hex(retrieve_kv_cache[0].data_ptr())}")
        print(f"Addresses are same: {store_kv_cache[0].data_ptr() == retrieve_kv_cache[0].data_ptr()}")
        
        # 4. Generate slot mapping (using int64 type because C++ kernel expects int64_t)
        print("\n4. Generating slot mapping (int64)...")
        slot_mapping = torch.tensor(
            random.sample(range(0, num_blocks * block_size), num_tokens),
            dtype=torch.int64,  # Changed to int64 because C++ kernel expects int64_t
            device="cuda:0"
        )
        
        print(f"Slot mapping shape: {slot_mapping.shape}")
        print(f"Slot mapping values: {slot_mapping.tolist()}")
        
        # 5. Create test tokens
        print("\n5. Creating test tokens...")
        tokens = list(range(num_tokens))
        
        # 6. Create VCacheEngine
        print("\n6. Creating VCacheEngine...")
        engine = VCacheEngine(config, metadata, gpu_connector=None)
        
        # 7. Test store
        print("\n7. Testing store...")
        try:
            engine.store(
                tokens=tokens,
                mask=None,
                kvcaches=store_kv_cache,
                slot_mapping=slot_mapping,
                offset=0
            )
            print("Store operation successful")
        except Exception as e:
            print(f"Store operation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 8. Test retrieve
        print("\n8. Testing retrieve...")
        try:
            ret_mask = engine.retrieve(
                tokens=tokens,
                mask=None,
                kvcaches=retrieve_kv_cache,
                slot_mapping=slot_mapping
            )
            
            retrieved_tokens = ret_mask.sum().item()
            print(f"Retrieve operation successful")
            print(f"Retrieved tokens: {retrieved_tokens}/{len(tokens)}")
            
        except Exception as e:
            print(f"Retrieve operation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 9. Verify data integrity
        print("\n9. Checking integrity...")
        print("  - Checking if store_kvcache and retrieve_kvcache are equal with slot_mapping")
        
        all_match = check_paged_kv_cache_equal(store_kv_cache, retrieve_kv_cache, slot_mapping)
        
        if all_match:
            print("  - All data matches")
        else:
            print("  - Data does not match")
        
        # 10. Clean up resources
        print("\n10. Cleaning up...")
        engine.close()
        print("Engine closed")
        
        return all_match
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_simulated_kv_cache()
    if success:
        print("\n✅ TEST PASSED")
    else:
        print("\n❌ TEST FAILED")
