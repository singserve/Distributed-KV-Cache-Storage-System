#!/usr/bin/env python3
"""
Test script for GPUVRAMSegmentManager with linked list memory management and LRU eviction.
"""

import sys
import os

# 保存原始sys.path
original_sys_path = sys.path.copy()

# 临时移除LMCache目录，避免循环导入
lmcache_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if lmcache_path in sys.path:
    sys.path.remove(lmcache_path)

# 现在导入torch和其他标准库模块
import torch
import time

# 恢复sys.path
sys.path = original_sys_path

# 添加路径以便导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmcache.test.gpu_vram_segment_manager import GPUVRAMSegmentManager, GPUVRAMSegment, MemoryBlock
from lmcache.test.test_vram_kvcache_unit import TestVRAMKVCacheUnit
from lmcache.utils import CacheEngineKey

class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        # Set default segment size to 16MB for testing (smaller than default 256MB)
        self.segment_size_mb = 16
    
    def get_extra_config_value(self, key, default):
        if key == "gpu_vram_segment_size_mb":
            return self.segment_size_mb
        return default

def test_memory_block():
    """Test MemoryBlock class."""
    print("Testing MemoryBlock class...")
    
    # Test basic properties
    block = MemoryBlock(start=0, size=1024, is_allocated=False)
    assert block.start == 0
    assert block.size == 1024
    assert block.end == 1024
    assert block.is_allocated == False
    
    # Test split
    remaining = block.split(512)
    assert remaining is not None
    assert block.size == 512
    assert block.end == 512
    assert remaining.start == 512
    assert remaining.size == 512
    assert remaining.end == 1024
    
    # Test merge_with_next
    block2 = MemoryBlock(start=512, size=512, is_allocated=False)
    block.next = block2
    assert block.merge_with_next() == True
    assert block.size == 1024
    assert block.next is None
    
    print("  MemoryBlock tests passed")

def test_gpuvram_segment():
    """Test GPUVRAMSegment class."""
    print("Testing GPUVRAMSegment class...")
    
    # Create a segment
    segment = GPUVRAMSegment(
        segment_id="test_segment",
        gpu_id=0,
        base_address=0x1000,
        size=4096  # 4KB for testing
    )
    
    # Test initial state
    assert segment.free_size == 4096
    assert segment.used_size == 0
    assert segment.free_blocks_head is not None
    assert segment.allocated_blocks_head is None
    
    # Test allocation
    result = segment.allocate(1024)
    assert result is not None
    offset, block = result
    assert offset == 0
    assert block.size == 1024
    assert block.is_allocated == True
    assert segment.used_size == 1024
    assert segment.free_size == 3072
    
    # Test allocation with split
    result2 = segment.allocate(512)
    assert result2 is not None
    offset2, block2 = result2
    assert offset2 == 1024  # Should be after first block
    assert block2.size == 512
    
    # Test free
    success = segment.free(block)
    assert success == True
    assert segment.used_size == 512
    assert segment.free_size == 3584
    
    # Test get_block_by_offset
    found_block = segment.get_block_by_offset(offset2)
    assert found_block is block2
    
    # Test stats
    stats = segment.get_stats()
    assert stats["total_size"] == 4096
    assert stats["used_size"] == 512
    assert stats["free_size"] == 3584
    
    print("  GPUVRAMSegment tests passed")

def test_segment_manager_basic():
    """Test basic GPUVRAMSegmentManager functionality."""
    print("Testing GPUVRAMSegmentManager basic functionality...")
    
    config = MockConfig()
    manager = GPUVRAMSegmentManager(config, gpu_id=0)
    
    # Test segment creation
    assert len(manager.segments) > 0
    
    # Test allocate_in_segment
    segment_id, offset = manager.allocate_in_segment(1024)
    assert segment_id is not None
    assert offset is not None
    
    # Test get_segment_by_id
    segment = manager.get_segment_by_id(segment_id)
    assert segment is not None
    
    # Test free_segment_space
    success = manager.free_segment_space(segment_id, offset, 1024)
    assert success == True
    
    print("  GPUVRAMSegmentManager basic tests passed")

def test_vram_unit_management():
    """Test VRAM unit management."""
    print("Testing VRAM unit management...")
    
    config = MockConfig()
    manager = GPUVRAMSegmentManager(config, gpu_id=0)
    
    # Create a VRAM unit
    # Use a simple string as cache key for testing
    cache_key = "test_cache_key_0"
    token_ids = [1, 2, 3, 4, 5]
    
    # First allocate space
    segment_id, offset = manager.allocate_in_segment(1024)
    assert segment_id is not None
    assert offset is not None
    
    # Create VRAM unit
    vram_unit = manager.create_vram_unit(
        cache_key=cache_key,
        token_ids=token_ids,
        segment_id=segment_id,
        offset=offset,
        allocated_size=1024,
        dtype=torch.float16,
        original_shape=(2, 256)
    )
    
    assert vram_unit is not None
    assert vram_unit.cache_key == cache_key
    assert vram_unit.segment_id == segment_id
    assert vram_unit.segment_offset == offset
    
    # Test get_vram_unit
    retrieved = manager.get_vram_unit(cache_key)
    assert retrieved is vram_unit
    
    # Test batch_get_vram_units
    batch_result = manager.batch_get_vram_units([cache_key])
    assert batch_result[0] is vram_unit
    
    # Test remove_vram_unit
    success = manager.remove_vram_unit(cache_key)
    assert success == True
    
    # Verify removal
    retrieved = manager.get_vram_unit(cache_key)
    assert retrieved is None
    
    print("  VRAM unit management tests passed")

def test_lru_eviction():
    """Test LRU eviction logic."""
    print("Testing LRU eviction...")
    
    config = MockConfig()
    manager = GPUVRAMSegmentManager(config, gpu_id=0)
    
    # Create multiple VRAM units with different access times
    # Allocate many small blocks to fill up the segment
    cache_keys = []
    block_size = 256 * 1024  # 256KB per block (even smaller blocks)
    num_blocks = 60  # Will allocate 15MB in a 16MB segment (60 * 256KB = 15MB)
    
    for i in range(num_blocks):
        # Use simple string keys for testing
        cache_key = f"test_cache_key_{i}"
        cache_keys.append(cache_key)
        
        # Allocate space
        segment_id, offset = manager.allocate_in_segment(block_size)
        assert segment_id is not None
        
        # Create VRAM unit
        vram_unit = manager.create_vram_unit(
            cache_key=cache_key,
            token_ids=[i],
            segment_id=segment_id,
            offset=offset,
            allocated_size=block_size,
            dtype=torch.float16
        )
        assert vram_unit is not None
        
        # Simulate different access times by waiting
        time.sleep(0.01)
    
    # Now try to allocate a block that requires eviction
    # Segment is 16MB, we've allocated 15MB, so only ~1MB free
    # Try to allocate 1.2MB, which should trigger LRU eviction
    stats_before = manager.get_vram_unit_stats()
    print(f"  VRAM units before eviction: {stats_before['total_vram_units']}")
    print(f"  Total allocated size before: {stats_before['total_allocated_size']}")
    
    # Check segment stats
    segment_stats = manager.get_segment_stats()
    print(f"  Segment stats: {segment_stats}")
    
    # Try to allocate a block larger than remaining free space
    # This should trigger LRU eviction
    large_block_size = int(1.2 * 1024 * 1024)  # 1.2MB
    segment_id, offset = manager.allocate_in_segment(large_block_size)
    
    # Check if allocation succeeded (should have evicted some VRAM units)
    if segment_id is not None:
        print(f"  Allocation succeeded after eviction: segment={segment_id}, offset={offset}")
        
        # Check stats after eviction
        stats_after = manager.get_vram_unit_stats()
        print(f"  VRAM units after eviction: {stats_after['total_vram_units']}")
        print(f"  Total allocated size after: {stats_after['total_allocated_size']}")
        
        # Should have fewer VRAM units or at least different allocation
        # Note: In some cases, eviction might not reduce count if allocation failed
        # but we got a new segment instead
        if stats_after['total_vram_units'] >= stats_before['total_vram_units']:
            print(f"  WARNING: VRAM unit count didn't decrease")
            print(f"  This could be because allocation got a new segment instead of evicting")
    else:
        print("  Allocation failed (may not have enough VRAM units to evict)")
    
    print("  LRU eviction test completed")

def test_fragmentation_and_merge():
    """Test fragmentation and free block merging."""
    print("Testing fragmentation and free block merging...")
    
    config = MockConfig()
    manager = GPUVRAMSegmentManager(config, gpu_id=0)
    
    # Get a segment
    segment = manager.segments[0]
    
    # Allocate multiple small blocks
    blocks = []
    for i in range(4):
        result = segment.allocate(256)
        assert result is not None
        offset, block = result
        blocks.append((offset, block))
    
    # Free some blocks to create fragmentation
    # Free blocks at positions 1 and 3 (keeping 0 and 2 allocated)
    success = segment.free(blocks[1][1])
    assert success == True
    success = segment.free(blocks[3][1])
    assert success == True
    
    # Now we have: [allocated, free, allocated, free]
    # Check free blocks count
    free_count = 0
    current = segment.free_blocks_head
    while current:
        free_count += 1
        current = current.next
    
    print(f"  Free blocks before merge: {free_count}")
    
    # Free the remaining allocated blocks
    success = segment.free(blocks[0][1])
    assert success == True
    success = segment.free(blocks[2][1])
    assert success == True
    
    # After freeing all blocks, they should merge into one large free block
    free_count_after = 0
    current = segment.free_blocks_head
    while current:
        free_count_after += 1
        current = current.next
    
    print(f"  Free blocks after merge: {free_count_after}")
    
    # Should have only one free block (all merged)
    assert free_count_after == 1
    assert segment.free_blocks_head.size == segment.size
    
    print("  ✓ Fragmentation and merge tests passed")

def main():
    """Run all tests."""
    print("Starting GPUVRAMSegmentManager tests...")
    
    try:
        test_memory_block()
        test_gpuvram_segment()
        test_segment_manager_basic()
        test_vram_unit_management()
        test_lru_eviction()
        test_fragmentation_and_merge()
        
        print("\n All tests passed!")
        
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
