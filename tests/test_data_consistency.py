#!/usr/bin/env python3
"""
Simple test to verify data consistency in cross-GPU store/retrieve
"""

import sys
import os
import torch
import random
import time

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcache.vcache_engine_system import VCacheEngine
from lmcache.config import LMCacheEngineMetadata
from vcache.vcache_config import VCacheConfig

def generate_kv_cache_paged_list_tensors(num_blocks, device, block_size=16, dtype=torch.float16):
    """Generate paged KV cache data with predictable values"""
    ret = []
    num_layers = 4
    num_heads = 8
    head_size = 64
    
    shape = [2, num_blocks, block_size, num_heads, head_size]
    
    for i in range(num_layers):
        # Generate predictable data: layer index + position-based values
        kv = torch.zeros(shape, dtype=dtype, device=device)
        for b in range(num_blocks):
            for t in range(block_size):
                for h in range(num_heads):
                    for hs in range(head_size):
                        # Create predictable values
                        val = (i * 1000 + b * 100 + t * 10 + h * 1 + hs * 0.1)
                        kv[0, b, t, h, hs] = val  # Key
                        kv[1, b, t, h, hs] = val + 0.5  # Value
        ret.append(kv)
    
    return ret

def test_simple_data_consistency():
    """Test data consistency in a single process"""
    print("Testing data consistency in cross-GPU store/retrieve")
    print("=" * 60)
    
    # Create config
    config = VCacheConfig()
    config.enable_gpu_vram_pool = True
    config.enable_gpu_vram_segments = True
    config.gpu_vram_segment_size_mb = 256
    config.connector_role = "worker"
    
    # Create metadata
    metadata = LMCacheEngineMetadata(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        fmt="vllm",
        kv_dtype=torch.float16,
        kv_shape=(4, 2, 16, 8, 64)  # num_layers, 2, block_size, num_heads, head_size
    )
    
    # Create test data
    num_blocks = 2
    block_size = 16
    num_tokens = num_blocks * block_size  # 32 tokens
    
    tokens = list(range(num_tokens))
    slot_mapping_data = list(range(num_tokens))  # Simple sequential mapping
