"""
Mooncake Lookup Client for scheduler role.

This client provides lookup functionality compatible with MooncakeStorageBackend's lookup method,
allowing the scheduler to query Mooncake for cache hits without instantiating the full storage backend.
"""


from typing import Union, Optional
import torch


from lmcache.vcache.vcache_logging import init_logger
from lmcache.vcache.utils import VCacheKey
from lmcache.vcache.vcache.token_database import TokenDatabase

logger = init_logger(__name__)


class MooncakeLookupClient:
    """
    Mooncake Lookup Client for scheduler role.
    
    This client provides lookup functionality compatible with MooncakeStorageBackend's lookup method,
    allowing the scheduler to query Mooncake for cache hits without instantiating the full storage backend.
    """
    
    def __init__(self, vllm_config: "VllmConfig", master_addr: str, vcache_config: "VCacheConfig"):
        """Initialize MooncakeLookupClient.
        
        Args:
            vllm_config: vLLM configuration
            master_addr: Master server address
            vcache_config: VCache configuration containing chunk_size
        """
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
        
        # Create test metadata for token database
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        
        # Get KV cache dtype
        kv_dtype = torch.float16  # Default
        if hasattr(vllm_config.cache_config, 'cache_dtype'):
            kv_dtype = vllm_config.cache_config.cache_dtype
        
        # Convert torch dtype to string format expected by TokenDatabase
        from lmcache.vcache.utils import dtype_to_str
        kv_dtype_str = dtype_to_str(kv_dtype)
        
        # Get chunk size from vcache_config
        chunk_size = vcache_config.chunk_size
        
        # Calculate KV shape
        num_layer = model_config.get_num_layers(parallel_config)
        num_kv_head = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        
        kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)
        
        # Initialize TokenDatabase with appropriate parameters
        self.token_database = TokenDatabase(chunk_size=chunk_size, save_unfull_chunk=True)
        
        # Store metadata for cache key generation
        self.metadata = {
            'model_name': model_config.model,
            'worker_id': parallel_config.rank,
            'world_size': parallel_config.world_size,
            'kv_dtype': kv_dtype_str,
            'kv_shape': kv_shape,
            'chunk_size': chunk_size
        }
        
        logger.info(f"MooncakeLookupClient initialized with master address: {master_addr}, chunk_size={chunk_size}")
    
    def lookup(
        self,
        tokens: list[int]
    ) -> int:
        """
        Lookup tokens in Mooncake store using simplified MooncakeStorageBackend lookup logic.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Number of hit tokens (continuous from the beginning)
        """
        
        # Process tokens to get all chunks using TokenDatabase
        all_chunks = []
        for start, end, cache_key in self.token_database.process_tokens(
            tokens=tokens,
            mask=None, 
            make_key=True,
            model_name="test_model" 
        ):
            if cache_key is not None:
                logger.debug(f"MooncakeLookupClient: Generated cache_key for chunk [{start}, {end}): "
                             f"type={type(cache_key)}, "
                            f"chunk_hash={getattr(cache_key, 'chunk_hash', 'N/A') if hasattr(cache_key, 'chunk_hash') else 'N/A'}")
            all_chunks.append((start, end, cache_key))
        
        if not all_chunks:
            logger.debug(f"MooncakeLookupClient: No chunks generated for {len(tokens)} tokens")
            return 0
        
        
        # Only check continuous chunks from the beginning (start=0)
        continuous_hit_tokens = 0
        expected_start = 0
        
        for start, end, cache_key in all_chunks:
            if start != expected_start:
                # Found a gap, stop checking
                logger.debug(f"MooncakeLookupClient: Gap found at start={start}," 
                             f"expected={expected_start}, stopping lookup")
                break
            
            # Skip chunks with None cache_key (masked chunks)
            if cache_key is None:
                logger.debug(f"MooncakeLookupClient: Chunk [{start}, {end}) has None cache_key (masked), skipping")
                # Masked chunks are considered as hits for continuity
                chunk_tokens = tokens[start:end]
                continuous_hit_tokens += len(chunk_tokens)
                expected_start = end
                continue
            
            if hasattr(cache_key, 'chunk_hash'):
                key_str = str(cache_key.chunk_hash)
            else:
                key_str = str(cache_key)
            
            logger.debug(f"MooncakeLookupClient: Checking key {key_str} for chunk [{start}, {end})")
            
            # Check if key exists in Mooncake store
            exists_result = self.store.is_exist(key_str)
            if exists_result == 1:
                # Key exists, add to continuous hit tokens
                chunk_tokens = tokens[start:end]
                continuous_hit_tokens += len(chunk_tokens)
                expected_start = end
                logger.debug(f"MooncakeLookupClient: Found hit for chunk [{start}, {end}): "
                             f"{len(chunk_tokens)} tokens")
            else:
                # Key does not exist, stop checking
                logger.debug(f"MooncakeLookupClient: Key {key_str} does not exist "
                             f"in Mooncake store, stopping lookup")
                break
        
        logger.info(f"MooncakeLookupClient lookup: {continuous_hit_tokens} continuous "
                    f"hit tokens from {len(tokens)} total tokens")
        return continuous_hit_tokens
