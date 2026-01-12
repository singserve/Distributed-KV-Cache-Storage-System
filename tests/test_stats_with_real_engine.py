#!/usr/bin/env python3
"""
Test script for stats manager with real VCache engine.
This script creates a real VCacheEngine using gpu0 yaml config and tests stats collection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any
import time
import yaml
import torch

from lmcache.vcache.vcache.vcache_engine_system import VCacheEngine
from lmcache.vcache.vcache_config import VCacheConfig
from lmcache.config import LMCacheEngineMetadata


def create_vcache_engine_from_yaml(yaml_path: str) -> VCacheEngine:
    """Create a VCacheEngine instance from YAML config file."""
    print(f"Loading config from: {yaml_path}")
    
    # Load YAML config
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create VCacheConfig
    config = VCacheConfig()
    
    # Update config with YAML values
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # For extra config values
            config.set_extra_config_value(key, value)
    
    # Create metadata with all required parameters
    metadata = LMCacheEngineMetadata(
        worker_id=0,
        model_name="test_model",
        world_size=1,  # Single GPU
        fmt="",
        kv_dtype=torch.float16,
        kv_shape=(32, 2, 16, 32, 128)  # (num_layers, 2, block_size, num_kv_heads, head_size)
    )
    
    # Create VCacheEngine
    print("Creating VCacheEngine...")
    engine = VCacheEngine(config, metadata)
    
    return engine


def test_stats_with_real_engine():
    """Test stats manager with real VCacheEngine."""
    print("=== Testing Stats Manager with Real VCacheEngine ===")
    
    # Create VCacheEngine from gpu0 yaml
    yaml_path = "vcache/vcache_config_gpu0.yaml"
    if not os.path.exists(yaml_path):
        print(f"YAML config file not found: {yaml_path}")
        print("Using default config instead...")
        # Create with default config
        config = VCacheConfig()
        metadata = LMCacheEngineMetadata(
            worker_id=0,
            model_name="test_model",
            world_size=1,
            fmt="",
            kv_dtype=torch.float16,
            kv_shape=(32, 2, 16, 32, 128)
        )
        engine = VCacheEngine(config, metadata)
    else:
        engine = create_vcache_engine_from_yaml(yaml_path)
    
    print(f"VCacheEngine created: {engine}")
    print(f"Engine config: {engine.config.__dict__}")
    
    # Test get_stats() method of VCacheEngine
    print("\nTesting VCacheEngine.get_stats() method...")
    try:
        # Call get_stats() method on the engine
        result = engine.get_stats()
        print(f"Result keys: {list(result.keys())}")
        
        # Check result structure
        assert "raw_statistics" in result
        assert "calculated_metrics" in result
        assert "summary" in result
        
        raw_stats = result["raw_statistics"]
        print(f"Raw stats keys: {list(raw_stats.keys())}")
        
        # Check that we have expected components
        expected_components = ["vcache_engine", "gpu_vram", "transfer_engine", 
                              "vram_metadata", "storage_backend", "token_database"]
        
        for component in expected_components:
            if component in raw_stats:
                print(f"  ✓ {component} stats collected")
            else:
                print(f"  ⚠ {component} stats not collected (may not be initialized)")
        
        # Check calculated metrics
        calc_metrics = result["calculated_metrics"]
        print(f"Calculated metrics keys: {list(calc_metrics.keys())}")
        
        # Check summary
        summary = result["summary"]
        print(f"Raw summary preview:\n{summary['raw'][:200]}...")
        print(f"Calculated summary preview:\n{summary['calculated'][:200]}...")
        
        print("\n✓ VCacheEngine.get_stats() test passed")
        
        # Print some key metrics
        print("\n=== Key Metrics ===")
        if "vcache_engine_metrics" in calc_metrics:
            vcache_metrics = calc_metrics["vcache_engine_metrics"]
            for key, value in vcache_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2%}")
                else:
                    print(f"  {key}: {value}")
        
        if "transfer_engine_metrics" in calc_metrics:
            transfer_metrics = calc_metrics["transfer_engine_metrics"]
            for key, value in transfer_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2%}")
                else:
                    print(f"  {key}: {value}")
        
        # Test get_stats_summary() method
        print("\nTesting VCacheEngine.get_stats_summary() method...")
        stats_summary = engine.get_stats_summary()
        print(f"Stats summary length: {len(stats_summary)} characters")
        print(f"Stats summary preview:\n{stats_summary[:300]}...")
        
        print("\n✓ VCacheEngine.get_stats_summary() test passed")
        
        return True
        
    except Exception as e:
        print(f"Error getting stats: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        print("\nCleaning up...")
        try:
            engine.shutdown()
            print("Engine shutdown complete")
        except:
            pass


def main():
    """Main test function."""
    print("Starting stats manager test with real VCacheEngine...\n")
    
    try:
        result = test_stats_with_real_engine()
        if result:
            print("\n✅ Test passed!")
            return 0
        else:
            print("\n❌ Test failed!")
            return 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
