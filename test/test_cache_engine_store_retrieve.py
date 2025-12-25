#!/usr/bin/env python3
"""
Test Cache Engine Store and Retrieve Test

This test directly instantiates TestCacheEngine and tests:
1. Store KV cache data with simulated slot mapping
2. Retrieve KV cache data with simulated slot mapping
3. Verify data consistency between store and retrieve

Simplified version: Uses single TestCacheEngine instance, mask test is optional.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from vcache_config import VCacheConfig
from lmcache.test.test_cache_engine_system import TestCacheEngine, MockGPUConnector

logger = init_logger(__name__)

class TestCacheEngineStoreRetrieveTest:
    """Test class for TestCacheEngine store and retrieve operations."""
    
    def __init__(self, run_mask_test: bool = False):
        """Initialize the test.
        
        Args:
            run_mask_test: Whether to run the mask test (optional)
        """
        # Test parameters (must be defined before _create_test_metadata)
        self.num_layers = 2  # Reduced for testing
        self.num_heads = 4   # Reduced for testing
        self.head_size = 64  # Reduced for testing
        self.block_size = 16  # vLLM block size
        self.chunk_size = 256  # Must match TestTokenDatabase chunk_size in TestCacheEngine
        self.run_mask_test = run_mask_test
        
        self.config = self._create_test_config()
        self.metadata = self._create_test_metadata()
        
        # 不传递gpu_connector，让TestCacheEngine自动创建BlockedKVGPUConnector
        # Initialize TestCacheEngine (single instance for all tests)
        self.test_engine = TestCacheEngine(
            config=self.config,
            metadata=self.metadata,
            gpu_connector=None  # 让TestCacheEngine自动创建BlockedKVGPUConnector
        )
        
        # 获取实际的GPU connector
        self.gpu_connector = self.test_engine.gpu_connector
        
        # Statistics
        self.stats = {
            "store_success": 0,
            "store_failed": 0,
            "retrieve_success": 0,
            "retrieve_failed": 0,
            "data_matches": 0,
            "data_mismatches": 0
        }
        
        logger.info(f"TestCacheEngineStoreRetrieveTest initialized (mask_test={run_mask_test})")
    
    def _create_test_config(self) -> VCacheConfig:
        """Create test configuration from YAML file."""
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "test_system_config_gpu0.yaml")
        
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            # Fall back to defaults
            config = VCacheConfig.from_defaults()
            logger.warning("Using default configuration instead of YAML file")
        else:
            logger.info(f"Loading configuration from: {config_file}")
            config = VCacheConfig.from_file(config_file)
        
        return config
    
    def _create_test_metadata(self) -> LMCacheEngineMetadata:
        """Create test metadata."""
        return LMCacheEngineMetadata(
            model_name="test_model",
            world_size=1,
            worker_id=0,  # Single GPU
            fmt="vllm",
            kv_dtype=torch.float16,
            kv_shape=(self.num_layers, 2, self.chunk_size, self.num_heads, self.head_size),
            use_mla=False,
            role="worker"
        )
    
    def create_mock_kvcaches(self, num_tokens: int) -> List[torch.Tensor]:
        """
        Create mock vLLM KV caches for testing.
        
        Args:
            num_tokens: Number of tokens to create KV cache for
            
        Returns:
            List of KV cache tensors (one per layer)
        """
        kvcaches = []
        
        # Calculate number of blocks needed
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        
        for layer_idx in range(self.num_layers):
            # Create KV cache tensor with vLLM blocked layout
            # vLLM格式: [2, num_blocks, block_size, num_heads, head_size] (KV维度在前)
            kv_cache = torch.randn(
                2, num_blocks, self.block_size, self.num_heads, self.head_size,
                dtype=torch.float16,
                device=f"cuda:{self.metadata.worker_id}"
            )
            
            # Fill with some recognizable pattern for verification
            # Use simpler pattern for easier debugging
            # Key: fixed pattern based on token index (0.1 per token)
            # Value: fixed pattern based on token index (0.2 per token, plus 0.5 base)
            for block_idx in range(num_blocks):
                for slot_idx in range(self.block_size):
                    token_idx = block_idx * self.block_size + slot_idx
                    if token_idx < num_tokens:
                        # Key: simple pattern - 0.1 per token
                        kv_cache[0, block_idx, slot_idx, :, :] = token_idx * 0.1
                        # Value: different simple pattern - 0.2 per token, plus 0.5 base
                        # This makes it easy to distinguish from Key
                        kv_cache[1, block_idx, slot_idx, :, :] = 0.5 + token_idx * 0.2
            
            kvcaches.append(kv_cache)
        
        logger.info(f"Created {len(kvcaches)} mock KV caches for {num_tokens} tokens, shape: {kvcaches[0].shape}")
        return kvcaches
    
    def create_slot_mapping(self, num_tokens: int) -> torch.Tensor:
        """
        Create slot mapping for tokens.
        
        Args:
            num_tokens: Number of tokens
            
        Returns:
            Slot mapping tensor
        """
        # Simple slot mapping: tokens are assigned to slots in order
        slot_mapping = torch.zeros(num_tokens, dtype=torch.int32)
        
        for i in range(num_tokens):
            block_id = i // self.block_size
            slot_in_block = i % self.block_size
            slot_mapping[i] = block_id * self.block_size + slot_in_block
        
        logger.info(f"Created slot mapping for {num_tokens} tokens")
        return slot_mapping
    
    def verify_kvcache_data(self, original_kvcaches: List[torch.Tensor], 
                           retrieved_kvcaches: List[torch.Tensor],
                           token_indices: List[int]) -> bool:
        """
        Verify that retrieved KV cache data matches original data.
        
        Args:
            original_kvcaches: Original KV caches (vLLM格式: [2, num_blocks, block_size, num_heads, head_size])
            retrieved_kvcaches: Retrieved KV caches (vLLM格式: [2, num_blocks, block_size, num_heads, head_size])
            token_indices: Token indices to verify
            
        Returns:
            True if data matches, False otherwise
        """
        if len(original_kvcaches) != len(retrieved_kvcaches):
            logger.error(f"Layer count mismatch: {len(original_kvcaches)} vs {len(retrieved_kvcaches)}")
            return False
        
        for layer_idx in range(len(original_kvcaches)):
            original_kv = original_kvcaches[layer_idx]
            retrieved_kv = retrieved_kvcaches[layer_idx]
            
            # 检查形状是否匹配
            if original_kv.shape != retrieved_kv.shape:
                logger.error(f"Layer {layer_idx}: Shape mismatch: {original_kv.shape} vs {retrieved_kv.shape}")
                return False
            
            for token_idx in token_indices:
                block_id = token_idx // self.block_size
                slot_in_block = token_idx % self.block_size
                
                # Check Key data (索引0是Key)
                original_key = original_kv[0, block_id, slot_in_block, :, :]
                retrieved_key = retrieved_kv[0, block_id, slot_in_block, :, :]
                
                # 添加调试信息
                if token_idx == 0 and layer_idx == 0:
                    logger.info(f"Debug verify - Layer {layer_idx}, Token {token_idx}:")
                    logger.info(f"  Original key shape: {original_key.shape}")
                    logger.info(f"  Original key mean: {original_key.mean().item():.6f}")
                    logger.info(f"  Original key[0,0]: {original_key[0,0].item():.6f}")
                    logger.info(f"  Retrieved key mean: {retrieved_key.mean().item():.6f}")
                    logger.info(f"  Retrieved key[0,0]: {retrieved_key[0,0].item():.6f}")
                
                if not torch.allclose(original_key, retrieved_key, rtol=1e-2, atol=1e-2):  # 放宽tolerance
                    logger.error(f"Layer {layer_idx}, Token {token_idx}: Key mismatch")
                    logger.error(f"Original: {original_key.mean().item():.4f}, Retrieved: {retrieved_key.mean().item():.4f}")
                    # 打印更多调试信息
                    logger.error(f"Original key min/max: {original_key.min().item():.4f}/{original_key.max().item():.4f}")
                    logger.error(f"Retrieved key min/max: {retrieved_key.min().item():.4f}/{retrieved_key.max().item():.4f}")
                    return False
                
                # Check Value data (索引1是Value)
                original_value = original_kv[1, block_id, slot_in_block, :, :]
                retrieved_value = retrieved_kv[1, block_id, slot_in_block, :, :]
                
                # 添加调试信息
                if token_idx == 0 and layer_idx == 0:
                    logger.info(f"  Original value shape: {original_value.shape}")
                    logger.info(f"  Original value mean: {original_value.mean().item():.6f}")
                    logger.info(f"  Original value[0,0]: {original_value[0,0].item():.6f}")
                    logger.info(f"  Retrieved value mean: {retrieved_value.mean().item():.6f}")
                    logger.info(f"  Retrieved value[0,0]: {retrieved_value[0,0].item():.6f}")
                
                if not torch.allclose(original_value, retrieved_value, rtol=1e-2, atol=1e-2):  # 放宽tolerance
                    logger.error(f"Layer {layer_idx}, Token {token_idx}: Value mismatch")
                    logger.error(f"Original: {original_value.mean().item():.4f}, Retrieved: {retrieved_value.mean().item():.4f}")
                    # 打印更多调试信息
                    logger.error(f"Original value min/max: {original_value.min().item():.4f}/{original_value.max().item():.4f}")
                    logger.error(f"Retrieved value min/max: {retrieved_value.min().item():.4f}/{retrieved_value.max().item():.4f}")
                    return False
        
        logger.info(f"Verified {len(token_indices)} tokens across {len(original_kvcaches)} layers")
        return True
    
    def test_store_retrieve_multiple_chunks(self):
        """Test store and retrieve for multiple chunks using single TestCacheEngine instance."""
        logger.info("=" * 60)
        logger.info("Testing store and retrieve for multiple chunks")
        logger.info("=" * 60)
        
        # Create test data - multiple chunks
        num_tokens = self.chunk_size * 3  # Three chunks
        tokens = list(range(num_tokens))
        
        # Create mock KV caches
        original_kvcaches = self.create_mock_kvcaches(num_tokens)
        
        # Create slot mapping
        slot_mapping = self.create_slot_mapping(num_tokens)
        
        # Step 1: Store KV cache (all chunks)
        logger.info(f"Step 1: Storing {num_tokens} tokens ({num_tokens//self.chunk_size} chunks)")
        try:
            self.test_engine.store(
                tokens=tokens,
                kvcaches=original_kvcaches,
                slot_mapping=slot_mapping
            )
            logger.info("Store operation completed successfully")
            self.stats["store_success"] += 1
        except Exception as e:
            logger.error(f"Store operation failed: {e}")
            self.stats["store_failed"] += 1
            return False
        
        # Step 2: Lookup to verify cache hit
        logger.info(f"Step 2: Looking up {num_tokens} tokens")
        hit_tokens = self.test_engine.lookup(tokens)
        
        if hit_tokens != num_tokens:
            logger.error(f"Lookup failed: Expected {num_tokens} hits, got {hit_tokens}")
            return False
        
        logger.info(f"Lookup successful: {hit_tokens}/{num_tokens} tokens hit")
        
        # Step 3: Create new KV caches for retrieval (empty)
        retrieved_kvcaches = []
        for layer_idx in range(self.num_layers):
            num_blocks = (num_tokens + self.block_size - 1) // self.block_size
            retrieved_kv = torch.zeros(
                2, num_blocks, self.block_size, self.num_heads, self.head_size,
                dtype=torch.float16,
                device=f"cuda:{self.metadata.worker_id}"
            )
            retrieved_kvcaches.append(retrieved_kv)
        
        # Step 4: Retrieve KV cache
        logger.info(f"Step 4: Retrieving {num_tokens} tokens")
        try:
            ret_mask = self.test_engine.retrieve(
                tokens=tokens,
                kvcaches=retrieved_kvcaches,
                slot_mapping=slot_mapping
            )
            
            if ret_mask is None:
                logger.error("Retrieve operation returned None mask")
                self.stats["retrieve_failed"] += 1
                return False
            
            retrieved_count = ret_mask.sum().item()
            logger.info(f"Retrieve operation completed: {retrieved_count}/{num_tokens} tokens retrieved")
            
            if retrieved_count == num_tokens:
                self.stats["retrieve_success"] += 1
            else:
                logger.error(f"Retrieve incomplete: {retrieved_count}/{num_tokens} tokens")
                self.stats["retrieve_failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Retrieve operation failed: {e}")
            self.stats["retrieve_failed"] += 1
            return False
        
        # Step 5: Verify data consistency for sample tokens
        logger.info("Step 5: Verifying data consistency for sample tokens")
        
        # Test tokens from each chunk
        test_tokens = [
            0, 10, 20, 30,  # First chunk
            self.chunk_size, self.chunk_size + 10,  # Second chunk
            self.chunk_size * 2, self.chunk_size * 2 + 15  # Third chunk
        ]
        
        if self.verify_kvcache_data(original_kvcaches, retrieved_kvcaches, test_tokens):
            logger.info("✅ Data verification PASSED for sample tokens")
            self.stats["data_matches"] += 1
            return True
        else:
            logger.error("❌ Data verification FAILED for sample tokens")
            self.stats["data_mismatches"] += 1
            return False
    
    def test_partial_retrieve_with_mask(self):
        """Test partial retrieve with mask (optional test)."""
        if not self.run_mask_test:
            logger.info("Skipping mask test as requested")
            return True
        
        logger.info("=" * 60)
        logger.info("Testing partial retrieve with mask")
        logger.info("=" * 60)
        
        # Create test data
        num_tokens = self.chunk_size * 2  # Two chunks (512 tokens)
        tokens = list(range(num_tokens))
        
        # Create mock KV caches
        original_kvcaches = self.create_mock_kvcaches(num_tokens)
        
        # Create slot mapping
        slot_mapping = self.create_slot_mapping(num_tokens)
        
        # Step 1: Store KV cache
        logger.info(f"Step 1: Storing {num_tokens} tokens")
        try:
            self.test_engine.store(
                tokens=tokens,
                kvcaches=original_kvcaches,
                slot_mapping=slot_mapping
            )
            logger.info("Store operation completed successfully")
            self.stats["store_success"] += 1
        except Exception as e:
            logger.error(f"Store operation failed: {e}")
            self.stats["store_failed"] += 1
            return False
        
        # Step 2: Create mask for partial retrieval
        # With chunk_size=256, the number of False values must be a multiple of 256
        # Create mask with 256 True values and 256 False values
        mask = torch.zeros(num_tokens, dtype=torch.bool)
        # Set first 256 tokens as True (to be retrieved), last 256 as False
        mask[:256] = True
        
        masked_tokens = mask.sum().item()
        logger.info(f"Step 2: Created mask for {masked_tokens}/{num_tokens} tokens")
        logger.info(f"  Number of False values: {num_tokens - masked_tokens} (must be multiple of {self.chunk_size})")
        
        # Step 3: Create new KV caches for retrieval (empty)
        retrieved_kvcaches = []
        for layer_idx in range(self.num_layers):
            num_blocks = (num_tokens + self.block_size - 1) // self.block_size
            retrieved_kv = torch.zeros(
                2, num_blocks, self.block_size, self.num_heads, self.head_size,
                dtype=torch.float16,
                device=f"cuda:{self.metadata.worker_id}"
            )
            retrieved_kvcaches.append(retrieved_kv)
        
        # Step 4: Retrieve KV cache with mask
        logger.info(f"Step 4: Retrieving {masked_tokens} tokens with mask")
        try:
            ret_mask = self.test_engine.retrieve(
                tokens=tokens,
                mask=mask,
                kvcaches=retrieved_kvcaches,
                slot_mapping=slot_mapping
            )
            
            if ret_mask is None:
                logger.error("Retrieve operation returned None mask")
                self.stats["retrieve_failed"] += 1
                return False
            
            retrieved_count = ret_mask.sum().item()
            logger.info(f"Retrieve operation completed: {retrieved_count}/{masked_tokens} tokens retrieved")
            
            if retrieved_count == masked_tokens:
                self.stats["retrieve_success"] += 1
            else:
                logger.error(f"Retrieve incomplete: {retrieved_count}/{masked_tokens} tokens")
                self.stats["retrieve_failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Retrieve operation failed: {e}")
            self.stats["retrieve_failed"] += 1
            return False
        
        # Step 5: Verify data consistency for masked tokens
        logger.info("Step 5: Verifying data consistency for masked tokens")
        
        # Get indices of masked tokens
        masked_indices = torch.where(mask)[0].tolist()
        
        if self.verify_kvcache_data(original_kvcaches, retrieved_kvcaches, masked_indices):
            logger.info("✅ Data verification PASSED for masked tokens")
            self.stats["data_matches"] += 1
            return True
        else:
            logger.error("❌ Data verification FAILED for masked tokens")
            self.stats["data_mismatches"] += 1
            return False
    
    def run_all_tests(self):
        """Run tests - only multiple chunks and optional mask test."""
        logger.info("=" * 60)
        logger.info("Running TestCacheEngine store/retrieve tests")
        logger.info("=" * 60)
        
        test_results = []
        
        # Test 1: Multiple chunks (main test)
        logger.info("\nTest 1: Multiple chunks store/retrieve")
        result1 = self.test_store_retrieve_multiple_chunks()
        test_results.append(("Multiple chunks", result1))
        
        # Test 2: Partial retrieve with mask (optional)
        logger.info("\nTest 2: Partial retrieve with mask")
        result2 = self.test_partial_retrieve_with_mask()
        test_results.append(("Partial retrieve with mask", result2))
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        all_passed = True
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
            if not result:
                all_passed = False
        
        logger.info("\n" + "=" * 60)
        logger.info("STATISTICS")
        logger.info("=" * 60)
        for key, value in self.stats.items():
            logger.info(f"{key}: {value}")
        
        logger.info("=" * 60)
        if all_passed:
            logger.info("🎉 ALL TESTS PASSED!")
        else:
            logger.error("💥 SOME TESTS FAILED!")
        
        return all_passed
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up TestCacheEngine resources")
        try:
            self.test_engine.close()
            logger.info("TestCacheEngine closed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description="Test Cache Engine Store and Retrieve")
    parser.add_argument("--test", choices=["multiple-chunks", "mask-test", "all"], 
                       default="multiple-chunks",
                       help="Test to run: multiple-chunks (default), mask-test, or all")
    args = parser.parse_args()
    
    try:
        # Set PYTHONHASHSEED for reproducible hashing
        import os
        os.environ['PYTHONHASHSEED'] = '0'
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. Test requires GPU.")
            return False
        
        # Create test instance
        run_mask_test = args.test in ["mask-test", "all"]
        test = TestCacheEngineStoreRetrieveTest(run_mask_test=run_mask_test)
        
        try:
            if args.test == "multiple-chunks":
                logger.info("Running only multiple chunks test")
                success = test.test_store_retrieve_multiple_chunks()
            elif args.test == "mask-test":
                logger.info("Running only mask test")
                success = test.test_partial_retrieve_with_mask()
            else:  # all
                logger.info("Running all tests")
                success = test.run_all_tests()
            
            return success
        finally:
            test.cleanup()
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
