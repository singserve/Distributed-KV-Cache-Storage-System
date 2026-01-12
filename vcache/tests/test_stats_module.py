#!/usr/bin/env python3
"""
Test script for stats collector module.
This script tests the integration of stats collector with VCache engine components.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any
import torch
import time

from lmcache.vcache.vcache.vcache_engine_system import VCacheEngine
from lmcache.vcache.vcache_config import VCacheConfig
from lmcache.config import LMCacheEngineMetadata
from lmcache.vcache.vcache.stats_collector import get_stats_collector, StatsCollector
from lmcache.vcache.vcache.gpu_vram_segment_manager import GPUVRAMSegmentManager
from lmcache.vcache.transfer_engine.transfer_engine_manager import TransferEngineManager
from lmcache.vcache.vcache.mooncake_storage_backend import MooncakeStorageBackend
from lmcache.vcache.vcache.token_database import TokenDatabase
from lmcache.vcache.vcache.vram_metadata_ipc_client import get_vram_metadata_ipc_client


def test_stats_collector_basic():
    """Test basic stats collector functionality."""
    print("=== Testing Stats Collector Basic Functionality ===")
    
    # Create a stats collector
    stats_collector = StatsCollector()
    print(f"Stats collector created: {stats_collector}")
    print(f"Stats collector start time: {stats_collector.start_time}")
    
    # Test that we can get the global stats collector
    global_collector = get_stats_collector()
    print(f"Global stats collector: {global_collector}")
    assert global_collector is not None, "Global stats collector should not be None"
    
    print("✓ Basic stats collector test passed\n")
    return True


def test_stats_data_structures():
    """Test stats data structures."""
    print("=== Testing Stats Data Structures ===")
    
    from lmcache.vcache.vcache.stats_collector import (
        SegmentStats, GPUVRAMStats, TransferEngineStats,
        VRAMMetadataStats, StorageBackendStats, TokenDatabaseStats,
        VCacheEngineStats
    )
    
    # Test SegmentStats
    segment_stats = SegmentStats(
        segment_id="test_segment_1",
        total_size_bytes=1024 * 1024 * 256,  # 256MB
        used_bytes=1024 * 1024 * 128,        # 128MB
        free_bytes=1024 * 1024 * 128,        # 128MB
        utilization_percent=50.0,
        allocated_blocks_count=10,
        free_blocks_count=5,
        largest_free_block=1024 * 1024 * 64,  # 64MB
        vram_unit_count=3,
        total_allocated_size=1024 * 1024 * 96  # 96MB
    )
    print(f"SegmentStats created: {segment_stats}")
    assert segment_stats.segment_id == "test_segment_1"
    assert segment_stats.utilization_percent == 50.0
    
    # Test GPUVRAMStats
    gpu_vram_stats = GPUVRAMStats(
        gpu_id=0,
        total_segments=2,
        total_segment_size_bytes=1024 * 1024 * 512,  # 512MB
        total_used_segment_bytes=1024 * 1024 * 256,  # 256MB
        segments=[segment_stats]
    )
    print(f"GPUVRAMStats created: {gpu_vram_stats}")
    assert gpu_vram_stats.gpu_id == 0
    assert gpu_vram_stats.total_segments == 2
    
    # Test TransferEngineStats
    transfer_stats = TransferEngineStats(
        engine_type="nvlink",
        gpu_id=0,
        initialized=True,
        total_transfers=100,
        successful_transfers=95,
        failed_transfers=5,
        total_transfer_bytes=1024 * 1024 * 1024,  # 1GB
        uptime_seconds=3600.0,
        engine_status={"status": "active"}
    )
    print(f"TransferEngineStats created: {transfer_stats}")
    assert transfer_stats.engine_type == "nvlink"
    assert transfer_stats.total_transfers == 100
    
    # Test VRAMMetadataStats
    vram_metadata_stats = VRAMMetadataStats(
        total_requests=1000,
        failed_requests=50,
        is_connected=True,
        server_address="127.0.0.1:8080",
        server_stats={"connections": 10}
    )
    print(f"VRAMMetadataStats created: {vram_metadata_stats}")
    assert vram_metadata_stats.total_requests == 1000
    assert vram_metadata_stats.is_connected == True
    
    # Test StorageBackendStats
    storage_stats = StorageBackendStats(
        retrieves=50,
        stores=30,
        lookups=200,
        total_hit_tokens=5000,
        mooncake_operations=280,
        zero_copy_operations=20,
        total_entries=30,
        total_size_bytes=1024 * 1024 * 100,  # 100MB
        mooncake_available=True
    )
    print(f"StorageBackendStats created: {storage_stats}")
    assert storage_stats.retrieves == 50
    assert storage_stats.mooncake_available == True
    
    # Test TokenDatabaseStats
    token_db_stats = TokenDatabaseStats(
        total_tokens_processed=10000,
        total_chunks_generated=40,
        total_cache_keys_created=40,
        total_skipped_chunks=5,
        total_hash_calls=45,
        chunk_size=256,
        save_unfull_chunk=True,
        uptime_seconds=7200.0,
        hash_function="sha256_cbor"
    )
    print(f"TokenDatabaseStats created: {token_db_stats}")
    assert token_db_stats.total_tokens_processed == 10000
    assert token_db_stats.chunk_size == 256
    
    # Test VCacheEngineStats
    vcache_stats = VCacheEngineStats(
        worker_id=0,
        connector_role="worker",
        start_time=time.time(),
        uptime_seconds=3600.0,
        hits=800,
        misses=200,
        total_lookups=1000,
        total_stores=50,
        total_retrieves=30,
        gpu_vram_hits=600,
        gpu_vram_misses=150,
        cross_gpu_transfers=10,
        gpu_vram_stats=gpu_vram_stats,
        transfer_engine_stats=transfer_stats,
        vram_metadata_stats=vram_metadata_stats,
        storage_backend_stats=storage_stats,
        token_database_stats=token_db_stats
    )
    print(f"VCacheEngineStats created: {vcache_stats}")
    assert vcache_stats.worker_id == 0
    assert vcache_stats.hits == 800
    
    print("✓ Stats data structures test passed\n")
    return True


def test_stats_collector_methods():
    """Test stats collector methods."""
    print("=== Testing Stats Collector Methods ===")
    
    stats_collector = StatsCollector()
    
    # Test format_stats
    from lmcache.vcache.vcache.stats_collector import VCacheEngineStats
    
    # Create a minimal VCacheEngineStats for testing
    test_stats = VCacheEngineStats(
        worker_id=0,
        connector_role="worker",
        start_time=time.time(),
        uptime_seconds=100.0,
        hits=100,
        misses=50,
        total_lookups=150,
        total_stores=20,
        total_retrieves=15,
        gpu_vram_hits=80,
        gpu_vram_misses=20,
        cross_gpu_transfers=5
    )
    
    formatted_stats = stats_collector.format_stats(test_stats)
    print(f"Formatted stats: {formatted_stats}")
    assert "vcache_engine" in formatted_stats
    assert formatted_stats["vcache_engine"]["worker_id"] == 0
    assert "operation_counts" in formatted_stats["vcache_engine"]
    
    # Test get_summary
    summary = stats_collector.get_summary(test_stats)
    print(f"Stats summary:\n{summary}")
    assert "VCache Engine Stats" in summary
    assert "GPU 0" in summary
    assert "worker" in summary
    
    print("✓ Stats collector methods test passed\n")
    return True


def test_module_stats_methods():
    """Test that all modules have get_stats() method."""
    print("=== Testing Module Stats Methods ===")
    
    # Test GPUVRAMSegmentManager stats
    print("Testing GPUVRAMSegmentManager stats...")
    try:
        # We'll create a mock config for testing
        class MockConfig:
            def get_extra_config_value(self, key, default):
                if key == "gpu_vram_segment_size_mb":
                    return 256
                return default
        
        config = MockConfig()
        segment_manager = GPUVRAMSegmentManager(config, gpu_id=0, transfer_engine_manager=None)
        stats = segment_manager.get_stats()
        print(f"GPUVRAMSegmentManager stats: {stats}")
        assert isinstance(stats, dict)
        assert "gpu_id" in stats
        assert "total_segments" in stats
        assert "segments" in stats
        print("✓ GPUVRAMSegmentManager stats OK")
    except Exception as e:
        print(f"✗ GPUVRAMSegmentManager stats test failed: {e}")
        return False
    
    # Test TokenDatabase stats
    print("Testing TokenDatabase stats...")
    try:
        token_db = TokenDatabase(chunk_size=256, save_unfull_chunk=True)
        stats = token_db.get_stats()
        print(f"TokenDatabase stats: {stats}")
        assert isinstance(stats, dict)
        assert "chunk_size" in stats
        assert "total_tokens_processed" in stats
        print("✓ TokenDatabase stats OK")
    except Exception as e:
        print(f"✗ TokenDatabase stats test failed: {e}")
        return False
    
    print("✓ All module stats methods test passed\n")
    return True


def main():
    """Main test function."""
    print("Starting stats module tests...\n")
    
    tests = [
        test_stats_collector_basic,
        test_stats_data_structures,
        test_stats_collector_methods,
        test_module_stats_methods,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append((test.__name__, False))
    
    print("\n=== Test Summary ===")
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
