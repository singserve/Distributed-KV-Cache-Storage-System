# SPDX-License-Identifier: Apache-2.0
"""
Mooncake Storage Backend

This module provides a storage backend implementation using Mooncake distributed store
for KV cache storage and retrieval.
"""

# Standard
from typing import Dict, List, Optional, Tuple, Union
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from test_config import TestConfig

logger = init_logger(__name__)

# Mooncake imports
try:
    from mooncake.store import MooncakeDistributedStore
    MOONCAKE_AVAILABLE = True
    logger.info("Mooncake store is available")
except ImportError:
    MOONCAKE_AVAILABLE = False
    logger.warning("Mooncake store not available, falling back to mock storage")


class MooncakeStorageBackend:
    """Mooncake storage backend using mooncake-store distributed store."""
    
    def __init__(self, gpu_id: int, config: Optional[TestConfig] = None):
        self.gpu_id = gpu_id
        self.config = config
        self.stats = {
            "retrieves": 0,
            "stores": 0,
            "lookups": 0,
            "total_hit_tokens": 0,
            "mooncake_operations": 0,
            "zero_copy_operations": 0,
            "total_entries": 0,
            "total_size_bytes": 0
        }
        
        # Initialize Mooncake store client
        self.store_client = None
        self._initialize_mooncake()
    
    def _initialize_mooncake(self):
        """Initialize Mooncake store using config and transfer engine."""
        try:
            # Create Mooncake store client using the correct API from test files
            self.store_client = MooncakeDistributedStore()
            
            # Get configuration from config or use defaults
            local_hostname = self._get_config_value("local_hostname", f"gpu_{self.gpu_id}")
            metadata_server = self._get_config_value("metadata_server", "http://127.0.0.1:8080/metadata")
            global_segment_size = self._get_config_value("global_segment_size", 3200 * 1024 * 1024)  # 3200 MB
            local_buffer_size = self._get_config_value("local_buffer_size", 512 * 1024 * 1024)  # 512 MB
            protocol = self._get_config_value("protocol", "tcp")
            device_name = self._get_config_value("device_name", "")
            master_server_address = self._get_config_value("master_server_address", "127.0.0.1:50051")
            
            # Support for different metadata server formats
            if metadata_server.startswith("http://"):
                # HTTP metadata server format
                logger.info(f"Using HTTP metadata server: {metadata_server}")
            elif ":" in metadata_server and not metadata_server.startswith("http"):
                # TCP metadata server format (e.g., "127.0.0.1:2379")
                logger.info(f"Using TCP metadata server: {metadata_server}")
            else:
                logger.warning(f"Unrecognized metadata server format: {metadata_server}")
            
            # Enhanced protocol configuration
            if protocol == "rdma":
                # For RDMA, use appropriate device name
                device_name = self._get_config_value("device_name", "")
                logger.info(f"Using RDMA protocol with device: {device_name}")
            else:
                # For TCP, use default device
                device_name = self._get_config_value("device_name", "")
                logger.info(f"Using TCP protocol with device: {device_name}")
            
            logger.info(f"Initializing Mooncake store with enhanced metadata config:")
            logger.info(f"  local_hostname: {local_hostname}")
            logger.info(f"  metadata_server: {metadata_server}")
            logger.info(f"  protocol: {protocol}")
            logger.info(f"  device_name: {device_name}")
            logger.info(f"  global_segment_size: {global_segment_size} bytes")
            logger.info(f"  local_buffer_size: {local_buffer_size} bytes")
            logger.info(f"  master_server_address: {master_server_address}")
            
            # Check if we have a transfer engine to pass to setup
            logger.info("Mooncake store setup")
            # Setup Mooncake store with transfer engine
            result = self.store_client.setup(
                local_hostname,
                metadata_server,
                global_segment_size,
                local_buffer_size,
                protocol,
                device_name,
                master_server_address
            )
            
            if result != 0:
                logger.error(f"Failed to setup Mooncake store client: {result}")
                logger.error("Please check metadata server configuration and network connectivity")
                self.store_client = None
                return
            
            logger.info("Mooncake store client setup successful")
            logger.info(f"Mooncake storage backend initialized for GPU {self.gpu_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Mooncake: {e}")
            logger.error("Falling back to mock storage backend")
            self.store_client = None
    
    
    def _get_config_value(self, key: str, default: any) -> any:
        """Get configuration value from config or return default."""
        if self.config is None:
            return default
        
        # Try to get from extra_config first
        value = self.config.get_extra_config_value(key, None)
        if value is not None:
            return value
        
        # Try to get from config attributes
        if hasattr(self.config, key):
            return getattr(self.config, key)
        
        return default
    
    
    def _generate_key(self, tokens: List[int]) -> int:
        """Generate a unique key for the token sequence using chain hash."""
        # Use chain hash similar to GPU VRAM pool for consistency
        if not tokens:
            return 0
        
        # Generate chain hash: hash of all tokens and their prefixes
        chain_hash = 0
        for i in range(len(tokens)):
            # Hash the prefix up to position i
            prefix = tuple(tokens[:i+1])
            prefix_hash = hash(prefix)
            # Combine with previous hash using XOR for chain effect
            chain_hash ^= prefix_hash
        
        return chain_hash
    
    def retrieve(self, cache_key: Union[int, str], tokens: Union[torch.Tensor, List[int]], **kwargs) -> Tuple[List[int], torch.Tensor, Dict]:
        """
        Retrieve actual KV cache data from Mooncake store using the provided cache key and tokens.
        Enhanced to support TestCacheEngine integration with actual vLLM KV cache data retrieval.
        
        Workflow:
        1. Use the provided cache key to directly retrieve stored data from Mooncake
        2. Parse enhanced storage data with actual KV cache data
        3. Reconstruct KV cache tensor from stored bytes
        4. Return tokens, KV cache tensor, and KV cache structure
        
        Args:
            cache_key: Cache key (int, str, or LayerCacheEngineKey) used to retrieve the data
            tokens: List of token IDs or torch.Tensor for verification
            **kwargs: Additional arguments for future extension
            
        Returns:
            Tuple of (retrieved_tokens, kv_cache_tensor, kv_cache_structure) where:
            - retrieved_tokens: List of token IDs that were stored
            - kv_cache_tensor: Reconstructed KV cache tensor data
            - kv_cache_structure: KV cache structure metadata
        """
        self.stats["retrieves"] += 1
        self.stats["mooncake_operations"] += 1
        
        if not self.store_client:
            logger.warning("Mooncake store client not available, falling back to mock behavior")
            # Mock: return empty data
            if isinstance(tokens, torch.Tensor):
                token_list = tokens.tolist()
            else:
                token_list = tokens
            return token_list[:int(len(token_list) * 0.5)], torch.tensor([]), {}
        
        try:
            import pickle
            import numpy as np
            
            # Convert tokens to list format if it's a tensor
            if isinstance(tokens, torch.Tensor):
                token_list = tokens.tolist()
            else:
                token_list = tokens
            
            # Convert cache key to string for Mooncake store
            # Handle LayerCacheEngineKey objects and other key types
            if hasattr(cache_key, 'chunk_hash'):
                # Use chunk_hash for LayerCacheEngineKey objects to ensure consistency
                key_str = str(cache_key.chunk_hash)
                logger.debug(f"Using LayerCacheEngineKey chunk_hash for retrieval: {key_str}")
            else:
                key_str = str(cache_key)
            
            # Check if the key exists in Mooncake store
            exists_result = self.store_client.is_exist(key_str)
            if exists_result != 1:
                logger.warning(f"Mooncake retrieve: key {key_str} does not exist in store")
                return [], torch.tensor([]), {}
            
            # Get the stored data
            retrieved_data = self.store_client.get(key_str)
            if not retrieved_data or retrieved_data == b"":
                logger.warning(f"Mooncake retrieve: key {key_str} exists but has no data")
                return [], torch.tensor([]), {}
            
            # Parse the enhanced storage data
            try:
                storage_data = pickle.loads(retrieved_data)
                
                if isinstance(storage_data, dict) and "token_ids" in storage_data and "kv_cache_data" in storage_data:
                    # Extract token IDs, KV cache data, and structure from enhanced storage data
                    retrieved_tokens = storage_data["token_ids"]
                    kv_cache_data = storage_data["kv_cache_data"]
                    kv_cache_structure = storage_data.get("kv_cache_structure", {})
                    
                    # Verify that retrieved tokens match the input tokens (for the matching prefix)
                    if not self._verify_token_match(token_list, retrieved_tokens):
                        logger.warning(f"Mooncake retrieve: token mismatch for key {key_str}")
                        return [], torch.tensor([]), {}
                    
                    # Reconstruct KV cache tensor from stored bytes
                    kv_cache_tensor = None
                    
                    if "kv_cache_bytes" in kv_cache_data:
                        kv_cache_bytes = kv_cache_data["kv_cache_bytes"]
                        kv_cache_shape = kv_cache_data.get("kv_cache_shape")
                        kv_cache_dtype_str = kv_cache_data.get("kv_cache_dtype", "float16")
                        
                        # Convert bytes back to numpy array and then to tensor
                        if kv_cache_dtype_str == "float16":
                            np_array = np.frombuffer(kv_cache_bytes, dtype=np.float16)
                        elif kv_cache_dtype_str == "float32":
                            np_array = np.frombuffer(kv_cache_bytes, dtype=np.float32)
                        else:
                            # Default to float16
                            np_array = np.frombuffer(kv_cache_bytes, dtype=np.float16)
                        
                        if kv_cache_shape:
                            np_array = np_array.reshape(kv_cache_shape)
                        
                        # Make numpy array writable to avoid PyTorch warning
                        # np.frombuffer creates a read-only array, we need to copy it to make it writable
                        np_array = np_array.copy()
                        
                        # Convert to torch tensor
                        if kv_cache_dtype_str == "float16":
                            kv_cache_tensor = torch.from_numpy(np_array).to(torch.float16)
                        elif kv_cache_dtype_str == "float32":
                            kv_cache_tensor = torch.from_numpy(np_array).to(torch.float32)
                        else:
                            kv_cache_tensor = torch.from_numpy(np_array).to(torch.float16)
                        
                        logger.debug(f"Reconstructed KV cache tensor: shape={kv_cache_tensor.shape}, dtype={kv_cache_tensor.dtype}")
                    else:
                        logger.warning(f"Mooncake retrieve: no kv_cache_bytes found for key {key_str}")
                        return [], torch.tensor([]), {}
                    
                    self.stats["total_hit_tokens"] += len(retrieved_tokens)
                    logger.info(f"Mooncake retrieve: {len(retrieved_tokens)} tokens retrieved for key {key_str}, "
                               f"tensor_shape={kv_cache_tensor.shape if kv_cache_tensor is not None else 'None'}, "
                               f"data_size={storage_data.get('data_size', 0)} bytes")
                    
                    return retrieved_tokens, kv_cache_tensor, kv_cache_structure
                
                else:
                    logger.warning(f"Mooncake retrieve: parsed data is not a valid format for key {key_str}")
                    return [], torch.tensor([]), {}
                    
            except Exception as parse_error:
                logger.error(f"Failed to parse stored token data: {parse_error}")
                return [], torch.tensor([]), {}
            
        except Exception as e:
            logger.error(f"Mooncake retrieve failed: {e}")
            return [], torch.tensor([]), {}
    
    def _verify_token_match(self, input_tokens: List[int], stored_tokens: List[int]) -> bool:
        """
        Verify that the stored tokens match the input tokens (for the matching prefix).
        
        Args:
            input_tokens: Input token list
            stored_tokens: Stored token list from Mooncake
            
        Returns:
            True if stored tokens match the prefix of input tokens, False otherwise
        """
        if len(stored_tokens) > len(input_tokens):
            logger.warning(f"Stored tokens length ({len(stored_tokens)}) exceeds input tokens length ({len(input_tokens)})")
            return False
        
        # Check if stored tokens match the prefix of input tokens
        for i in range(len(stored_tokens)):
            if stored_tokens[i] != input_tokens[i]:
                logger.warning(f"Token mismatch at position {i}: stored={stored_tokens[i]}, input={input_tokens[i]}")
                return False
        
        return True
    
    def store(self, tokens: Union[torch.Tensor, List[int]], **kwargs) -> bool:
        """
        Store actual KV cache data to Mooncake store for a single chunk.
        Enhanced to support TestCacheEngine integration with actual vLLM KV cache data storage.
        
        Args:
            tokens: List of token IDs or torch.Tensor (chunk tokens)
            **kwargs: Additional arguments including:
                - kvcaches: Single CPU tensor containing chunk KV cache data (from TestCacheEngine store)
                - kv_cache_structure: KV cache structure metadata
                - cache_key: CacheEngineKey for consistent key generation
                
        Returns:
            True if store successful, False otherwise
        """
        self.stats["stores"] += 1
        self.stats["mooncake_operations"] += 1
        
        if not self.store_client:
            logger.warning("Mooncake store client not available, falling back to mock behavior")
            return True  # Mock success
        
        try:
            # Convert tokens to list format if it's a tensor
            if isinstance(tokens, torch.Tensor):
                token_list = tokens.tolist()
            else:
                token_list = tokens
            
            # Extract KV cache data and metadata from kwargs
            kvcaches = kwargs.get("kvcaches")
            kv_cache_structure = kwargs.get("kv_cache_structure", {})
            cache_key = kwargs.get("cache_key")
            
            if kvcaches is None:
                logger.error("No kvcaches provided for actual KV cache data storage")
                return False
            
            # Use CacheEngineKey if provided, otherwise fallback to token hash
            if cache_key is not None:
                # Use chunk_hash as the key for consistency with contains() method
                # This ensures that store() and contains() use the same key format
                if hasattr(cache_key, 'chunk_hash'):
                    key = cache_key.chunk_hash
                    logger.debug(f"Using CacheEngineKey chunk_hash for storage: {key}")
                else:
                    key = cache_key
                    logger.debug(f"Using CacheEngineKey for storage: {key}")
            else:
                # Fallback to token hash
                key = hash(tuple(token_list))
                logger.debug(f"Using token hash for storage: {key}")
            
            # Prepare actual KV cache data for storage
            # kvcaches is now a single CPU tensor containing chunk KV cache data for one layer
            kv_cache_data = {}
            kv_cache_bytes = None
            
            if isinstance(kvcaches, torch.Tensor):
                # Single tensor case - vLLM KV cache (one layer)
                kv_cache_bytes = kvcaches.cpu().numpy().tobytes()
                kv_cache_data["kv_cache_bytes"] = kv_cache_bytes
                kv_cache_data["kv_cache_shape"] = kvcaches.shape
                kv_cache_data["kv_cache_dtype"] = str(kvcaches.dtype)
            elif isinstance(kvcaches, list) and len(kvcaches) > 0:
                # List of tensors case (backward compatibility)
                # For vLLM, we only expect one tensor, so take the first one
                kv_cache = kvcaches[0]
                kv_cache_bytes = kv_cache.cpu().numpy().tobytes()
                kv_cache_data["kv_cache_bytes"] = kv_cache_bytes
                kv_cache_data["kv_cache_shape"] = kv_cache.shape
                kv_cache_data["kv_cache_dtype"] = str(kv_cache.dtype)
            else:
                logger.error(f"Unsupported kvcaches type: {type(kvcaches)}")
                return False
            
            # Enhanced storage data: include token sequence, KV cache metadata, and actual KV cache data
            storage_data = {
                "token_ids": token_list,  # Store only the chunk tokens, not prefixes
                "kv_cache_structure": kv_cache_structure,
                "kv_cache_data": kv_cache_data,
                "timestamp": time.time(),
                "gpu_id": self.gpu_id,
                "data_size": len(kv_cache_bytes) if kv_cache_bytes is not None else 0,
                "cache_key_info": {
                    "chunk_hash": cache_key.chunk_hash if cache_key else None,
                    "model_name": cache_key.model_name if cache_key else "unknown",
                    "worker_id": cache_key.worker_id if cache_key else self.gpu_id,
                    "world_size": cache_key.world_size if cache_key else 1
                } if cache_key else {}
            }
            
            # Convert storage data to bytes for storage
            import pickle
            storage_bytes = pickle.dumps(storage_data)
            
            # Store the enhanced data with actual KV cache in Mooncake store
            result = self.store_client.put(str(key), storage_bytes)
            
            if result == 0:
                logger.info(f"Mooncake store: {len(token_list)} chunk tokens stored with key {key}, data_size={storage_data['data_size']} bytes")
                # Update statistics
                self.stats["total_entries"] += 1
                self.stats["total_size_bytes"] += storage_data['data_size']
                return True
            else:
                logger.warning(f"Mooncake store failed for {len(token_list)} chunk tokens with error code: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Mooncake store failed: {e}")
            return False
    
    def lookup(self, tokens: List[int], all_chunks: Optional[List[Tuple[int, int, any]]] = None) -> Tuple[int, Optional[List[Tuple[Tuple[int, int], int, bool]]]]:
        """
        Enhanced lookup KV cache in Mooncake store using all_chunks information.
        Similar to vram manager lookup - uses cache_key for direct lookup.
        Only returns continuous hits from the beginning (start=0).
        
        Returns (hit_tokens, chunk_info_list) where:
        - hit_tokens: number of matching tokens (only continuous from the beginning)
        - chunk_info_list: list of tuples for each matching chunk, each tuple contains:
            - (start, end): chunk range
            - gpu_id: GPU ID where the chunk is stored (from storage metadata)
            - needs_transfer: True if this chunk needs cross-GPU transfer (always False for Mooncake as it's remote storage)
          Returns None if no match
        """
        self.stats["lookups"] += 1
        self.stats["mooncake_operations"] += 1
        
        if not self.store_client:
            logger.warning("Mooncake store client not available, falling back to mock behavior")
            # Mock lookup logic - return mock continuous hits
            if all_chunks is not None and len(all_chunks) > 0:
                # Sort chunks by start position
                sorted_chunks = sorted(all_chunks, key=lambda x: x[0])
                
                # Only check continuous chunks from the beginning (start=0)
                continuous_hit_tokens = 0
                expected_start = 0
                chunk_info_list = []
                
                for start, end, cache_key in sorted_chunks:
                    if start != expected_start:
                        break
                    
                    chunk_tokens = tokens[start:end]
                    # Mock: assume first 2 chunks are hit
                    if len(chunk_info_list) < 2:
                        continuous_hit_tokens += len(chunk_tokens)
                        expected_start = end
                        # Mock GPU ID (always 0 for mock)
                        chunk_info = ((start, end), 0, False)
                        chunk_info_list.append(chunk_info)
                        logger.debug(f"Mock Mooncake hit for chunk [{start}, {end}): {len(chunk_tokens)} tokens")
                    else:
                        break
                
                if continuous_hit_tokens > 0:
                    logger.info(f"Mock Mooncake lookup: {continuous_hit_tokens} tokens from {len(chunk_info_list)} chunks")
                    return continuous_hit_tokens, chunk_info_list
            
            return 0, None
        
        try:
            # all_chunks is required for enhanced lookup
            if all_chunks is None:
                logger.warning("Mooncake lookup: all_chunks is None, returning 0")
                return 0, None
            
            logger.debug(f"Mooncake lookup with {len(all_chunks)} chunks")
            
            # Sort chunks by start position
            sorted_chunks = sorted(all_chunks, key=lambda x: x[0])
            
            # Only check continuous chunks from the beginning (start=0)
            continuous_hit_tokens = 0
            expected_start = 0
            chunk_info_list = []
            
            for start, end, cache_key in sorted_chunks:
                # Check if this chunk is continuous from the beginning
                if start != expected_start:
                    # Found a gap, stop checking
                    logger.info(f"Breaking at gap: expected start={expected_start}, got start={start}, stopping Mooncake lookup")
                    break
                
                chunk_tokens = tokens[start:end]
                logger.debug(f"Checking Mooncake storage for continuous chunk [{start}, {end}): {len(chunk_tokens)} tokens")
                
                # Use cache_key to check if chunk exists in Mooncake store
                # Convert cache_key to string for Mooncake store
                if hasattr(cache_key, 'chunk_hash'):
                    # Use chunk_hash for LayerCacheEngineKey objects
                    key_str = str(cache_key.chunk_hash)
                else:
                    key_str = str(cache_key)
                
                exists_result = self.store_client.is_exist(key_str)
                if exists_result == 1:
                    # Key exists, retrieve the actual stored data to get the real length
                    retrieved_data = self.store_client.get(key_str)
                    if retrieved_data and retrieved_data != b"":
                        try:
                            import pickle
                            storage_data = pickle.loads(retrieved_data)
                            if isinstance(storage_data, dict) and "token_ids" in storage_data:
                                stored_tokens = storage_data["token_ids"]
                                actual_stored_length = len(stored_tokens)
                                
                                # Verify that stored tokens match the chunk tokens
                                if self._verify_token_match(chunk_tokens, stored_tokens):
                                    # Check if the entire chunk is hit
                                    if actual_stored_length == len(chunk_tokens):
                                        # Entire chunk is hit, add to continuous hit tokens
                                        continuous_hit_tokens += actual_stored_length
                                        expected_start = end
                                        
                                        # Get GPU ID from storage metadata (default to self.gpu_id if not available)
                                        gpu_id = storage_data.get("gpu_id", self.gpu_id)
                                        # Mooncake is remote storage, so always needs transfer
                                        needs_transfer = True
                                        
                                        # Add chunk info to the list
                                        chunk_info = ((start, end), gpu_id, needs_transfer)
                                        chunk_info_list.append(chunk_info)
                                        
                                        logger.debug(f"Found Mooncake storage hit for continuous chunk [{start}, {end}): {actual_stored_length} tokens, GPU={gpu_id}, needs_transfer={needs_transfer}")
                                    else:
                                        # Chunk not fully hit, stop checking
                                        logger.info(f"Breaking at chunk [{start}, {end}): expected {len(chunk_tokens)} tokens, got {actual_stored_length} tokens, stopping Mooncake lookup")
                                        break
                                else:
                                    # Token mismatch, stop checking
                                    logger.warning(f"Token mismatch for chunk [{start}, {end}): stored {len(stored_tokens)} tokens but expected {len(chunk_tokens)}, stopping Mooncake lookup")
                                    break
                            else:
                                logger.warning(f"Mooncake lookup: unexpected storage data format for key {key_str}, stopping lookup")
                                break
                        except Exception as parse_error:
                            logger.warning(f"Failed to parse stored token data: {parse_error}, stopping Mooncake lookup")
                            break
                    else:
                        # Key exists but has no data, stop checking
                        logger.warning(f"Mooncake lookup: key {key_str} exists but has no data, stopping lookup")
                        break
                else:
                    # Key does not exist, stop checking
                    logger.debug(f"Mooncake lookup: key {key_str} does not exist in store, stopping lookup")
                    break
            
            if continuous_hit_tokens > 0:
                self.stats["total_hit_tokens"] += continuous_hit_tokens
                logger.info(f"Mooncake lookup: found {continuous_hit_tokens} tokens from {len(chunk_info_list)} continuous chunks")
                for i, ((start, end), gpu_id, needs_transfer) in enumerate(chunk_info_list):
                    logger.info(f"  Chunk {i}: [{start}, {end}) -> GPU {gpu_id}, needs_transfer={needs_transfer}")
                return continuous_hit_tokens, chunk_info_list
            else:
                logger.debug(f"Mooncake lookup: no continuous match found from the beginning")
                return 0, None
            
        except Exception as e:
            logger.error(f"Mooncake lookup failed: {e}")
            return 0, None
    
    def contains(self, cache_key) -> bool:
        """
        Check if the cache key exists in Mooncake store.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.store_client:
            logger.warning("Mooncake store client not available, returning False")
            return False
        
        try:
            # Convert cache key to string
            if hasattr(cache_key, 'chunk_hash'):
                key_str = str(cache_key.chunk_hash)
            else:
                key_str = str(cache_key)
            
            # Check if key exists in Mooncake store
            exists_result = self.store_client.is_exist(key_str)
            return exists_result == 1
            
        except Exception as e:
            logger.error(f"Failed to check if key exists: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get storage backend statistics."""
        stats = self.stats.copy()
        stats["mooncake_available"] = self.store_client is not None
        return stats
    
    def close(self):
        """Close Mooncake connections and release all resources."""
        logger.info("Closing Mooncake storage backend and releasing all resources")
        
        try:
            if self.store_client:
                # Close the store client
                self.store_client.close()
                logger.info("Mooncake store client closed successfully")
            
            # Reset statistics
            self.stats = {
                "retrieves": 0,
                "stores": 0,
                "lookups": 0,
                "total_hit_tokens": 0,
                "mooncake_operations": 0,
                "zero_copy_operations": 0
            }
            
            # Reset client reference
            self.store_client = None
            self.transfer_engine = None
            
            logger.info("Mooncake storage backend shutdown completed")
            
        except Exception as e:
            logger.error(f"Error closing Mooncake connections: {e}")

