# SPDX-License-Identifier: Apache-2.0
"""
Test Mooncake Lookup Client for scheduler role.

This client provides lookup functionality compatible with MooncakeStorageBackend's lookup method,
allowing the scheduler to query Mooncake for cache hits without instantiating the full storage backend.
"""

# Standard
from typing import Union, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.test.test_token_database import TestTokenDatabase

logger = init_logger(__name__)


class TestMooncakeLookupClient:
    """
    Test Mooncake Lookup Client for scheduler role.
    
    This client provides lookup functionality compatible with MooncakeStorageBackend's lookup method,
    allowing the scheduler to query Mooncake for cache hits without instantiating the full storage backend.
    """
    
    def __init__(self, vllm_config: "VllmConfig", master_addr: str):
        """Initialize TestMooncakeLookupClient."""
        # Third Party
        from mooncake.store import MooncakeDistributedStore
        
        self.store = MooncakeDistributedStore()
        self.store.setup(
            "localhost",
            "P2PHANDSHAKE",
            0,
            16 * 1024 * 1024,
            "tcp",
            "",
            master_addr,
        )
        
        # Initialize test token database for processing tokens
        # Use TestTokenDatabase instead of ChunkedTokenDatabase
        from lmcache.test.test_token_database import TestTokenDatabase
        
        # Create test metadata for token database
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        
        # Get KV cache dtype
        kv_dtype = torch.float16  # Default
        if hasattr(vllm_config.cache_config, 'cache_dtype'):
            kv_dtype = vllm_config.cache_config.cache_dtype
        
        # Convert torch dtype to string format expected by TestTokenDatabase
        kv_dtype_str = "float16"  # Default
        if kv_dtype == torch.float16:
            kv_dtype_str = "float16"
        elif kv_dtype == torch.float32:
            kv_dtype_str = "float32"
        elif kv_dtype == torch.bfloat16:
            kv_dtype_str = "bfloat16"
        else:
            kv_dtype_str = str(kv_dtype).replace("torch.", "")
        
        # Calculate KV shape
        num_layer = model_config.get_num_layers(parallel_config)
        num_kv_head = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        chunk_size = 256  # Default chunk size
        
        kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)
        
        # Initialize TestTokenDatabase with appropriate parameters
        self.token_database = TestTokenDatabase(chunk_size=chunk_size, save_unfull_chunk=True)
        
        # Store metadata for cache key generation
        self.metadata = {
            'model_name': model_config.model,
            'worker_id': parallel_config.rank,
            'world_size': parallel_config.world_size,
            'kv_dtype': kv_dtype_str,
            'kv_shape': kv_shape
        }
        
        logger.info(f"TestMooncakeLookupClient initialized with master address: {master_addr}")
    
    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: Optional[str] = None
    ) -> int:
        """
        Lookup tokens in Mooncake store using simplified MooncakeStorageBackend lookup logic.
        
        Args:
            token_ids: List of token IDs or torch.Tensor
            lookup_id: Optional lookup ID for tracking
            
        Returns:
            Number of hit tokens (continuous from the beginning)
        """
        # Convert token_ids to list if it's a tensor
        if isinstance(token_ids, torch.Tensor):
            token_list = token_ids.tolist()
        else:
            token_list = token_ids
        
        # Process token_ids to get all chunks using TestTokenDatabase
        # IMPORTANT: Use the same parameters as TestCacheEngine.lookup() to ensure consistent cache key generation
        # TestCacheEngine.lookup() calls process_tokens with mask=None and model_name="test_model"
        all_chunks = []
        for start, end, cache_key in self.token_database.process_tokens(
            tokens=token_list,
            mask=None,  # Match TestCacheEngine.lookup() parameter
            make_key=True,
            model_name="test_model"  # Match TestCacheEngine.lookup() parameter
        ):
            # Debug logging for cache key generation
            if cache_key is not None:
                logger.debug(f"TestMooncakeLookupClient: Generated cache_key for chunk [{start}, {end}): type={type(cache_key)}, "
                           f"chunk_hash={getattr(cache_key, 'chunk_hash', 'N/A') if hasattr(cache_key, 'chunk_hash') else 'N/A'}")
            all_chunks.append((start, end, cache_key))
        
        if not all_chunks:
            logger.debug(f"TestMooncakeLookupClient: No chunks generated for {len(token_list)} tokens")
            return 0
        
        # Sort chunks by start position
        sorted_chunks = sorted(all_chunks, key=lambda x: x[0])
        
        # Only check continuous chunks from the beginning (start=0)
        continuous_hit_tokens = 0
        expected_start = 0
        
        for start, end, cache_key in sorted_chunks:
            # Check if this chunk is continuous from the beginning
            if start != expected_start:
                # Found a gap, stop checking
                logger.debug(f"TestMooncakeLookupClient: Gap found at start={start}, expected={expected_start}, stopping lookup")
                break
            
            # Skip chunks with None cache_key (masked chunks)
            if cache_key is None:
                logger.debug(f"TestMooncakeLookupClient: Chunk [{start}, {end}) has None cache_key (masked), skipping")
                # Masked chunks are considered as hits for continuity
                chunk_tokens = token_list[start:end]
                continuous_hit_tokens += len(chunk_tokens)
                expected_start = end
                continue
            
            # Ensure cache_key is a CacheEngineKey object
            if not isinstance(cache_key, CacheEngineKey):
                logger.warning(f"TestMooncakeLookupClient: cache_key is not a CacheEngineKey object, type={type(cache_key)}")
                # Try to convert to string anyway
                key_str = str(cache_key)
            else:
                # Convert cache_key to string for Mooncake store
                # Follow MooncakeStorageBackend.lookup logic: use chunk_hash if available
                if hasattr(cache_key, 'chunk_hash'):
                    # Use chunk_hash for LayerCacheEngineKey objects
                    key_str = str(cache_key.chunk_hash)
                else:
                    key_str = str(cache_key)
            
            logger.debug(f"TestMooncakeLookupClient: Checking key {key_str} for chunk [{start}, {end})")
            
            # Check if key exists in Mooncake store (simplified version of MooncakeStorageBackend.lookup)
            exists_result = self.store.is_exist(key_str)
            if exists_result == 1:
                # Key exists, add to continuous hit tokens
                chunk_tokens = token_list[start:end]
                continuous_hit_tokens += len(chunk_tokens)
                expected_start = end
                logger.debug(f"TestMooncakeLookupClient: Found hit for chunk [{start}, {end}): {len(chunk_tokens)} tokens")
            else:
                # Key does not exist, stop checking
                logger.debug(f"TestMooncakeLookupClient: Key {key_str} does not exist in Mooncake store, stopping lookup")
                break
        
        logger.info(f"TestMooncakeLookupClient lookup: {continuous_hit_tokens} continuous hit tokens from {len(token_list)} total tokens")
        return continuous_hit_tokens
    
    def close(self):
        """Close the lookup client."""
        # nothing here
        pass
