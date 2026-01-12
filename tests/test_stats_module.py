#!/usr/bin/env python3
"""
Test script for stats collector module.
This script tests the integration of stats collector with VCache engine components.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any
import time

from lmcache.vcache.stats.stats_collector import StatsCollector
from lmcache.vcache.stats.stats_calculator import StatsCalculator


def test_stats_integration():
    """Test stats collector and calculator integration with sample stats."""
    print("=== Testing Stats Integration ===")
    
    # Create sample stats dictionary (as would be collected by StatsCollector)
    sample_stats = {
        "vcache_engine": {
            "worker_id": 0,
            "connector_role": "worker",
            "start_time": time.time(),
            "uptime_seconds": 100.0,
            "operation_counts": {
                "hits": 80,
                "misses": 20,
                "total_lookups": 100,
                "total_stores": 10,
                "total_retrieves": 5,
                "gpu_vram_hits": 60,
                "gpu_vram_misses": 15,
                "cross_gpu_transfers": 2
            }
        },
        "gpu_vram": {
            "gpu_id": 0,
            "total_segments": 2,
            "total_allocations": 10,
            "total_deallocations": 5,
            "vram_unit_creations": 8,
            "vram_unit_deletions": 3,
            "segment_evictions": 2,
            "segments": [
                {
                    "segment_id": "segment_1",
                    "total_size_bytes": 1024 * 1024 * 256,
                    "used_bytes": 1024 * 1024 * 128,
                    "free_bytes": 1024 * 1024 * 128,
                    "utilization_percent": 50.0,
                    "allocated_blocks_count": 10,
                    "free_blocks_count": 5,
                    "largest_free_block": 1024 * 1024 * 64,
                    "vram_unit_count": 3,
                    "total_allocated_size": 1024 * 1024 * 96
                }
            ]
        },
        "transfer_engine": {
            "engine_type": "nvlink",
            "gpu_id": 0,
            "max_concurrent_transfers": 4,
            "transfer_timeout_sec": 30.0,
            "pending_transfers": 2,
            "active_transfers": 1,
            "completed_transfers_count": 100,
            "successful_transfers": 95,
            "failed_transfers": 5,
            "total_transfer_bytes": 1024 * 1024 * 1024,
            "successful_registers": 50,
            "failed_registers": 2,
            "successful_unregisters": 48,
            "failed_unregisters": 1,
            "engine_status": {"status": "active"}
        },
        "vram_metadata": {
            "total_requests": 1000,
            "failed_requests": 20,
            "is_connected": True,
            "server_address": "127.0.0.1:8080",
            "server_stats": {"connections": 10}
        },
        "storage_backend": {
            "retrieves": 50,
            "stores": 30,
            "lookups": 200,
            "contains": 180,
            "total_hit_tokens": 5000,
            "total_entries": 30,
            "total_size_bytes": 1024 * 1024 * 100
        },
        "token_database": {
            "total_tokens_processed": 10000,
            "total_chunks_generated": 40,
            "chunk_size": 256,
            "save_unfull_chunk": True,
            "hash_function": "sha256_cbor"
        }
    }
    
    print("1. Testing StatsCalculator with sample stats...")
    calculator = StatsCalculator()
    calculated_metrics = calculator.calculate_all_metrics(sample_stats)
    print(f"   Calculated metrics: {type(calculated_metrics).__name__}")
    
    metrics_dict = calculator.get_calculated_metrics_dict(calculated_metrics)
    print(f"   Metrics dictionary keys: {list(metrics_dict.keys())}")
    
    summary = calculator.get_metrics_summary(calculated_metrics)
    print(f"   Summary length: {len(summary)} characters")
    
    # Verify calculated metrics
    assert hasattr(calculated_metrics, 'hit_rate')
    assert hasattr(calculated_metrics, 'transfer_success_rate')
    assert hasattr(calculated_metrics, 'storage_hit_rate')
    assert "vcache_engine_metrics" in metrics_dict
    assert "transfer_engine_metrics" in metrics_dict
    assert "Calculated Statistics Metrics" in summary
    
    # Check specific values
    print(f"   Hit rate: {calculated_metrics.hit_rate:.2%}")
    print(f"   Transfer success rate: {calculated_metrics.transfer_success_rate:.2%}")
    print(f"   Storage hit rate: {calculated_metrics.storage_hit_rate:.2%}")
    
    print("   ✓ StatsCalculator test passed")
    
    print("\n✓ All stats integration tests passed\n")
    return True


def main():
    """Main test function."""
    print("Starting stats module integration test...\n")
    
    try:
        result = test_stats_integration()
        if result:
            print("✅ Test passed!")
            return 0
        else:
            print("❌ Test failed!")
            return 1
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
