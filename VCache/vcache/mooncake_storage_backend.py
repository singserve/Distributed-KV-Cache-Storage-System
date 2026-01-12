"""
Mooncake Storage Backend

This module provides a storage backend implementation using Mooncake distributed store
for KV cache storage and retrieval.
"""


from typing import Dict, List, Optional, Tuple, Union
import pickle
import numpy as np
import torch
from lmcache.vcache.logging.vcache_logging import init_logger
from lmcache.vcache.vcache_config import VCacheConfig

logger = init_logger(__name__)

try:
    from mooncake.store import MooncakeDistributedStore
    MOONCAKE_AVAILABLE = True
    logger.info("Mooncake store is available")
except ImportError:
    MOONCAKE_AVAILABLE = False
    logger.error("Mooncake store not available")


class MooncakeStorageBackend:
    """Mooncake storage backend using mooncake-store distributed store."""
    
    def __init__(self, gpu_id: int, config: Optional[VCacheConfig] = None):
        self.gpu_id = gpu_id
        self.config = config
        self.stats = {
            "retrieves": 0,
            "stores": 0,
            "lookups": 0,
            "contains": 0,
            "total_hit_tokens": 0,
            "total_entries": 0,
            "total_size_bytes": 0,
        }
        
        # Initialize Mooncake store client
        self.store_client = None
        self._initialize_mooncake()
    
    def _initialize_mooncake(self):
        """Initialize Mooncake store using config and transfer engine."""
        if not MOONCAKE_AVAILABLE:
            logger.error("Mooncake store library is not available. "
                         "Please install mooncake-store package to use this backend.")
            return
        self.store_client = MooncakeDistributedStore()
        
        local_hostname = self._get_config_value("local_hostname", f"gpu_{self.gpu_id}")
        metadata_server = self._get_config_value("metadata_server", "http://127.0.0.1:8080/metadata")
        global_segment_size = self._get_config_value("global_segment_size", 3200 * 1024 * 1024)  # 3200 MB
        local_buffer_size = self._get_config_value("local_buffer_size", 512 * 1024 * 1024)  # 512 MB
        protocol = self._get_config_value("protocol", "tcp")
        device_name = self._get_config_value("device_name", "")
        master_server_address = self._get_config_value("master_server_address", "127.0.0.1:50051")
        
        if metadata_server.startswith("http://"):
            logger.info(f"Using HTTP metadata server: {metadata_server}")
        else:
            logger.warning(f"Unrecognized metadata server format: {metadata_server}")
        
        # Enhanced protocol configuration
        if protocol == "rdma":
            # For RDMA, use appropriate device name
            device_name = self._get_config_value("device_name", "")
            logger.info(f"mooncake storage backend using RDMA protocol with device: {device_name}")
        else:
            # For TCP, use default device
            device_name = self._get_config_value("device_name", "")
            logger.info(f"mooncake storage backend using TCP protocol with device: {device_name}")
        
        logger.info(f"Initializing Mooncake store with config:")
        logger.info(f"  local_hostname: {local_hostname}")
        logger.info(f"  metadata_server: {metadata_server}")
        logger.info(f"  protocol: {protocol}")
        logger.info(f"  device_name: {device_name}")
        logger.info(f"  global_segment_size: {global_segment_size} bytes")
        logger.info(f"  local_buffer_size: {local_buffer_size} bytes")
        logger.info(f"  master_server_address: {master_server_address}")
        
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
            self.store_client = None
            return
        
        logger.info("Mooncake store client setup successful")
        logger.info(f"Mooncake storage backend initialized for GPU {self.gpu_id}")
        
    
    
    def _get_config_value(self, key: str, default: any) -> any:
        """Get configuration value from config or return default."""
        if self.config is None:
            return default
        
        value = self.config.get_extra_config_value(key, None)
        if value is not None:
            return value
        
        if hasattr(self.config, key):
            return getattr(self.config, key)
        
        return default
    
    def retrieve(
        self, 
        cache_key, 
        tokens: List[int], 
        **kwargs
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Retrieve KV cache data from Mooncake store using the provided cache key and tokens.
        
        Workflow:
        1. Use the provided cache key to directly retrieve stored data from Mooncake
        2. Parse enhanced storage data with actual KV cache data
        3. Reconstruct KV cache tensor from stored bytes
        4. Return tokens and KV cache tensor
        
        Args:
            cache_key: Cache key (VCacheKey, int, or str) used to retrieve the data
            tokens: List of token IDs or torch.Tensor for verification
            **kwargs: Additional arguments for future extension
            
        Returns:
            Tuple of (retrieved_tokens, kv_cache_tensor) where:
            - retrieved_tokens: List of token IDs that were stored
            - kv_cache_tensor: Reconstructed KV cache tensor data
        """
        assert self.store_client is not None, "Mooncake store client is not initialized"
          
        # Convert cache key to appropriate format for Mooncake store
        if hasattr(cache_key, 'chunk_hash'):
            key = cache_key.chunk_hash
        else:
            key = cache_key
        
        key_str = str(key)
        
        # Check if the key exists in Mooncake store
        exists_result = self.store_client.is_exist(key_str)
        if exists_result != 1:
            logger.warning(f"Mooncake retrieve: key {key_str} does not exist in store")
            return [], torch.tensor([])
        
        # Get the stored data
        retrieved_data = self.store_client.get(key_str)
        if not retrieved_data or retrieved_data == b"":
            logger.warning(f"Mooncake retrieve: key {key_str} exists but has no data")
            return [], torch.tensor([])
        
        try:
            storage_data = pickle.loads(retrieved_data)
            
            if isinstance(storage_data, dict) and "token_ids" in storage_data and "kv_cache_data" in storage_data:
                # Extract token IDs and KV cache data from enhanced storage data
                retrieved_tokens = storage_data["token_ids"]
                kv_cache_data = storage_data["kv_cache_data"]
                
                # Verify that retrieved tokens match the input tokens (for the matching prefix)
                if not self._verify_token_match(tokens, retrieved_tokens):
                    logger.warning(f"Mooncake retrieve: token mismatch for key {key_str}")
                    return [], torch.tensor([])
                
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
                    
                    # np.frombuffer creates a read-only array, we need to copy it to make it writable
                    np_array = np_array.copy()
                    
                    # Convert to torch tensor
                    if kv_cache_dtype_str == "float16":
                        kv_cache_tensor = torch.from_numpy(np_array).to(torch.float16)
                    elif kv_cache_dtype_str == "float32":
                        kv_cache_tensor = torch.from_numpy(np_array).to(torch.float32)
                    else:
                        kv_cache_tensor = torch.from_numpy(np_array).to(torch.float16)
                else:
                    logger.warning(f"Mooncake retrieve: no kv_cache_bytes found for key {key_str}")
                    return [], torch.tensor([])
                
                # Update statistics
                self.stats["retrieves"] += 1
                self.stats["total_hit_tokens"] += len(retrieved_tokens)
                
                logger.info(f"Mooncake retrieve: {len(retrieved_tokens)} tokens retrieved for key {key_str}, "
                            f"tensor_shape={kv_cache_tensor.shape if kv_cache_tensor is not None else 'None'}, "
                            f"data_size={storage_data.get('data_size', 0)} bytes")
                
                return retrieved_tokens, kv_cache_tensor
            
            else:
                logger.error(f"Mooncake retrieve: parsed data is not a valid format for key {key_str}")
                return [], torch.tensor([])
                
        except Exception as parse_error:
            logger.error(f"Failed to parse stored token data: {parse_error}")
            return [], torch.tensor([])
    
    def _verify_token_match(
        self, 
        input_tokens: List[int], 
        stored_tokens: List[int]
    ) -> bool:
        """
        Verify that the stored tokens match the input tokens (for the matching prefix).
        
        Args:
            input_tokens: Input token list
            stored_tokens: Stored token list from Mooncake
        Returns:
            True if stored tokens match the prefix of input tokens, False otherwise
        """
        for i in range(len(stored_tokens)):
            if stored_tokens[i] != input_tokens[i]:
                return False
        
        return True
    
    def store(
        self, 
        tokens: List[int], 
        kvcaches, 
        cache_key
    ) -> bool:
        """
        Store actual KV cache data to Mooncake store for a single chunk.
        
        Args:
            tokens: List of token IDs
            kvcaches: tensor containing chunk KV cache data
            cache_key: VCacheKey for consistent key generation (optional)
                
        Returns:
            True if store successful, False otherwise
        """   
        assert self.store_client is not None, "Mooncake store client is not initialized" 
      
        if kvcaches is None:
            logger.error("No kvcaches provided for actual KV cache data storage")
            return False
        
        if hasattr(cache_key, 'chunk_hash'):
            key = cache_key.chunk_hash
        else:
            key = cache_key
        key_str = str(key)
        
        kv_cache_data = {}
        kv_cache_bytes = None
        
        if isinstance(kvcaches, torch.Tensor):
            kv_cache_bytes = kvcaches.cpu().numpy().tobytes()
            kv_cache_data["kv_cache_bytes"] = kv_cache_bytes
            kv_cache_data["kv_cache_shape"] = kvcaches.shape
            kv_cache_data["kv_cache_dtype"] = str(kvcaches.dtype)
        else:
            logger.error(f"Unsupported kvcaches type: {type(kvcaches)}")
            return False
        
        storage_data = {
            "token_ids": tokens, 
            "kv_cache_data": kv_cache_data,
            "data_size": len(kv_cache_bytes)
        }
               
        storage_bytes = pickle.dumps(storage_data)
        
        # Store the data with KV cache in Mooncake store
        result = self.store_client.put(key_str, storage_bytes)
        
        if result == 0:
            # Update statistics
            self.stats["stores"] += 1
            self.stats["total_entries"] += 1
            self.stats["total_size_bytes"] += storage_data['data_size']
            
            logger.info(f"Mooncake store: {len(tokens)} chunk tokens "
                        f"stored with key {key}, "
                        f"data_size={storage_data['data_size']} bytes")
            return True
        else:
            logger.warning(f"Mooncake store failed for {len(tokens)} chunk tokens with error code: {result}")
            return False

    
    def lookup(
        self, 
        tokens: List[int], 
        all_chunks: Optional[List[Tuple[int, int, any]]]
    ) -> Tuple[int, Optional[List[Tuple[int, int, any]]]]:
        """
        lookup KV cache in Mooncake store.
        Only returns continuous hits from the beginning (start=0).
        
        Returns (hit_tokens, chunk_info_list) where:
        - hit_tokens: number of matching tokens (only continuous from the beginning)
        - chunk_info_list: list of tuples for each matching chunk, each tuple contains:
            - start: chunk start position
            - end: chunk end position
            - cache_key: cache key for the chunk
          Returns None if no match
        """
        assert self.store_client is not None, "Mooncake store client is not initialized"    

        if all_chunks is None:
            logger.error("Mooncake lookup: all_chunks is None, returning 0")
            return 0, None
        
        logger.info(f"Mooncake lookup with {len(all_chunks)} chunks")
        
        # Only check continuous chunks from the beginning (start=0)
        continuous_hit_tokens = 0
        expected_start = 0
        chunk_info_list = []
        
        for start, end, cache_key in all_chunks:
            chunk_tokens = tokens[start:end]
            logger.debug(f"Checking Mooncake storage for continuous chunk [{start}, {end}): "
                         f"{len(chunk_tokens)} tokens")
            
            # Check if this chunk is continuous from the beginning
            if start != expected_start:
                # Found a gap, stop checking
                logger.debug(f"Breaking at gap: "
                             f"expected start={expected_start}, "
                             f"got start={start}, "
                             f"stopping Mooncake lookup")
                break
            

            
            # Use cache_key to check if chunk exists in Mooncake store
            if hasattr(cache_key, 'chunk_hash'):
                key_str = str(cache_key.chunk_hash)
            else:
                key_str = str(cache_key)
            
            exists_result = self.store_client.is_exist(key_str)
            if exists_result != 1:
                # Key does not exist, stop checking
                logger.debug(f"Mooncake lookup: key {key_str} does not exist in store, stopping lookup")
                break

            retrieved_data = self.store_client.get(key_str)
            if retrieved_data is None or retrieved_data == b"":
                # Key exists but has no data, stop checking
                logger.warning(f"Mooncake lookup: key {key_str} exists but has no data, stopping lookup")
                break

            try:
                storage_data = pickle.loads(retrieved_data)
                if isinstance(storage_data, dict) and "token_ids" in storage_data:
                    stored_tokens = storage_data["token_ids"]
                    actual_stored_length = len(stored_tokens)
                    
                    # Verify that stored tokens match the chunk tokens
                    if self._verify_token_match(chunk_tokens, stored_tokens):
                        # Entire chunk is hit, add to continuous hit tokens
                        continuous_hit_tokens += actual_stored_length
                        expected_start = end
                        
                        # Add chunk info to the list
                        chunk_info_list.append((start, end, cache_key))                              
                        logger.debug(f"Found Mooncake storage hit"
                                    f"for continuous chunk [{start}, {end}): "
                                    f"{actual_stored_length} tokens")

                    else:
                        # Token mismatch, stop checking
                        logger.warning(f"Token mismatch for chunk [{start}, {end}): "
                                        f"stored {len(stored_tokens)} tokens "
                                        f"but expected {len(chunk_tokens)}, "
                                        f"stopping Mooncake lookup")
                        break
                else:
                    logger.warning(f"Mooncake lookup: unexpected storage data format for key {key_str}, "
                                    f"stopping lookup")
                    break
            except Exception as parse_error:
                logger.warning(f"Failed to parse stored token data: "
                               f"{parse_error}, stopping Mooncake lookup")
                break
               
        
        if continuous_hit_tokens > 0:
            # Update statistics
            self.stats["lookups"] += 1
            self.stats["total_hit_tokens"] += continuous_hit_tokens
            
            logger.info(f"Mooncake lookup: found {continuous_hit_tokens} tokens "
                        f"from {len(chunk_info_list)} continuous chunks")
            
            return continuous_hit_tokens, chunk_info_list
        else:
            # Update statistics (lookup still happened)
            self.stats["lookups"] += 1
            
            logger.info(f"Mooncake lookup: no continuous match found from the beginning")
            return 0, None
        
    
    def contains(
        self, 
        cache_key
    ) -> bool:
        """
        Check if the cache key exists in Mooncake store.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        assert self.store_client is not None, "Mooncake store client is not initialized"
        
        if hasattr(cache_key, 'chunk_hash'):
            key = cache_key.chunk_hash
        else:
            key = cache_key
        key_str = str(key)
        
        # Check if key exists in Mooncake store
        exists_result = self.store_client.is_exist(key_str)
        
        # Update statistics
        self.stats["contains"] += 1
        
        return exists_result == 1

    
    def get_stats(self) -> Dict:
        """Get storage backend statistics."""
        stats = self.stats.copy()
        return stats
    
    def close(self):
        """Close Mooncake connections and release all resources."""
        logger.info("Closing Mooncake storage backend and releasing all resources")
        
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
        }
        
        # Reset client reference
        self.store_client = None
        self.transfer_engine = None
        
        logger.info("Mooncake storage backend shutdown completed")
