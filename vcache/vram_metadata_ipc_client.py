"""
GPU VRAM Metadata IPC Client

A client for cache engine instances to communicate with the centralized
VRAM metadata server using IPC (Inter-Process Communication).
"""

# Standard
from typing import Dict, List, Optional, Tuple, Any
import threading
import time
from multiprocessing.managers import BaseManager
import torch

from lmcache.vcache.gpu_vram_pool_manager import GPUVRAMEntry
from lmcache.vcache.vcache_logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class VRAMMetadataIPCClient:
    """
    IPC client for communicating with the centralized VRAM metadata server.
    Provides the same interface as GPUVRAMPoolManager but delegates to IPC server.
    """
    
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        
        # Server configuration
        self.server_address = config.get_extra_config_value(
            "vram_metadata_ipc_address", "localhost"
        )
        self.server_port = config.get_extra_config_value(
            "vram_metadata_ipc_port", 9091
        )
        self.authkey = b'vram_metadata'
        
        # IPC connection
        self.manager = None
        self.server_proxy = None
        self.is_connected = False
        
        # Connection timeout settings
        self.connection_timeout = 10.0
        self.retry_count = 3
        
        # Statistics
        self.total_requests = 0
        self.failed_requests = 0
        
        # Connect to server
        self._connect_to_server()
        
        logger.info(f"VRAM Metadata IPC Client initialized for server: {self.server_address}:{self.server_port}")

    def _connect_to_server(self):
        """Connect to the IPC metadata server."""
        for attempt in range(self.retry_count):
            try:
                # Create a custom manager class
                class VRAMManager(BaseManager):
                    pass
                
                # Register methods with the manager
                VRAMManager.register('lookup_prefix')
                VRAMManager.register('register_kvcache')
                VRAMManager.register('batch_register_kvcache')
                VRAMManager.register('get_entry')
                VRAMManager.register('batch_get_entry')
                VRAMManager.register('get_ipc_mem_handle')
                VRAMManager.register('register_segment_ipc_handle')
                VRAMManager.register('remove')
                VRAMManager.register('get_stats')
                VRAMManager.register('health_check')
                VRAMManager.register('shutdown_server')
                
                # Connect to the server
                self.manager = VRAMManager(
                    address=(self.server_address, self.server_port),
                    authkey=self.authkey
                )
                self.manager.connect()
                # Get the actual proxy object for the registered methods
                # The manager itself becomes the proxy for registered methods
                self.server_proxy = self.manager
                self.is_connected = True
                
                # Test the connection by calling a simple method
                try:
                    health_info = self.server_proxy.health_check()
                    logger.info(f"Connection test successful: {health_info}")
                except Exception as e:
                    logger.error(f"Connection test failed: {e}")
                    raise ConnectionError("Not connected to VRAM metadata IPC server")
                
                logger.info(f"Successfully connected to IPC server on attempt {attempt + 1}")
                return
                
            except Exception as e:
                logger.warning(f"Failed to connect to IPC server (attempt {attempt + 1}): {e}")
                
                # Wait before retry
                if attempt < self.retry_count - 1:
                    time.sleep(0.5 * (attempt + 1))
        
        logger.error(f"All {self.retry_count} connection attempts failed")
        self.is_connected = False

    def _ensure_connection(self):
        """Ensure we have a connection to the server."""
        if not self.is_connected or self.server_proxy is None:
            self._connect_to_server()
        
        if not self.is_connected:
            raise ConnectionError("Not connected to VRAM metadata IPC server")

    def lookup_prefix(
        self,
        token_ids: List[int],
        all_chunks: Optional[List[Tuple[int, int, CacheEngineKey]]] = None,
        current_gpu_id: Optional[int] = None,
    ) -> Tuple[int, Optional[List[Tuple[Tuple[int, int], int, bool]]]]:
        """
        Lookup prefix match by delegating to VRAM metadata IPC server.
        """
        with self.lock:
            self.total_requests += 1
            
            try:
                self._ensure_connection()
                
                # Serialize all_chunks for IPC transmission if provided
                serialized_chunks = None
                if all_chunks is not None:
                    serialized_chunks = []
                    for start, end, cache_key in all_chunks:
                        serialized_chunks.append((start, end, self._serialize_key(cache_key)))
                
                # Make IPC call - use the registered method from the manager
                result = self.server_proxy.lookup_prefix(
                    token_ids, serialized_chunks, current_gpu_id
                )
                
                if hasattr(result, '_getvalue'):
                    result = result._getvalue()
                
                if isinstance(result, (tuple, list)) and len(result) == 2:
                    hit_tokens, chunk_info_list = result
                    logger.info(f"IPC lookup successful: "
                                f"{hit_tokens} hits, "
                                f"Chunks: {len(chunk_info_list) if chunk_info_list else 0}")
                    
                    logger.debug(f"IPC lookup result - "
                                 f"Hits: {hit_tokens}, "
                                 f"Chunk info list: {chunk_info_list}")
                    return hit_tokens, chunk_info_list
                else:
                    logger.error(f"Unexpected result format: {type(result)}, "
                                 f"length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                    return 0, None
                
            except Exception as e:
                self.failed_requests += 1
                logger.error(f"IPC lookup failed: {e}")
                return 0, None

    def register_kvcache(
        self,
        cache_key: CacheEngineKey,
        token_ids: List[int],
        gpu_id: int,
        tensor_shape: tuple,
        tensor_dtype: torch.dtype,
        tensor_size: int,
        buffer_pointer: Optional[int] = None,
        segment_id: Optional[str] = None,
        resident_hostname: Optional[str] = None,
        segment_offset: int = 0,
    ) -> bool:
        """
        Register KV cache by delegating to VRAM metadata IPC server.
        """
        with self.lock:
            self.total_requests += 1
            
            try:
                self._ensure_connection()
                
                # Serialize key and dtype for IPC transmission
                key_dict = self._serialize_key(cache_key)
                tensor_dtype_str = str(tensor_dtype)
                
                # Make IPC call
                success = self.server_proxy.register_kvcache(
                    key_dict, 
                    token_ids,
                    gpu_id,
                    tensor_shape,
                    tensor_dtype_str,
                    tensor_size,
                    buffer_pointer,
                    segment_id,
                    resident_hostname,
                    segment_offset
                )
                
                if success:
                    logger.info(f"Successfully registered KV cache via IPC - "
                                f"Key: {cache_key.chunk_hash}, "
                                f"GPU: {gpu_id}, "
                                f"Segment offset: {segment_offset}")
                    return True
                else:
                    logger.error("Registration failed via IPC")
                    self.failed_requests += 1
                    return False
                    
            except Exception as e:
                self.failed_requests += 1
                logger.error(f"IPC registration failed: {e}")
                return False

    def batch_register_kvcache(
        self,
        entries_data: List[Tuple[CacheEngineKey, List[int], int, tuple, torch.dtype, int, Optional[int], Optional[str], Optional[str], int]]
    ) -> List[bool]:
        """
        batch register KV cache to GPU VRAM pool metadata
        
        Args:
            entries_data: list of entry data tuples:
                (cache_key, token_ids, gpu_id, tensor_shape, tensor_dtype, tensor_size, 
                 buffer_pointer, segment_id, resident_hostname, segment_offset)
            
        Returns:
            list of registration results for each entry, True indicates success, False indicates failure
        """
        with self.lock:
            self.total_requests += 1
            
            try:
                self._ensure_connection()
                
                batch_entries = []
                
                for entry_data in entries_data:
                    try:
                        (cache_key, token_ids, gpu_id, tensor_shape, tensor_dtype, 
                         tensor_size, buffer_pointer, segment_id, resident_hostname, segment_offset) = entry_data
                        
                        # Serialize key and dtype for IPC transmission
                        key_dict = self._serialize_key(cache_key)
                        tensor_dtype_str = str(tensor_dtype)
                        
                        batch_entries.append((
                            key_dict, token_ids, gpu_id, tensor_shape, tensor_dtype_str, 
                            tensor_size, buffer_pointer, segment_id, resident_hostname, segment_offset
                        ))
                        
                    except Exception as e:
                        logger.error(f"Failed to serialize entry data: {e}")
                        continue
                
                if batch_entries:
                    results = self.server_proxy.batch_register_kvcache(batch_entries)
                    
                    success_count = sum(results)
                    logger.info(f"IPC batch_register_kvcache processed {len(batch_entries)} entries, "
                                f"{success_count} successful")
                    
                    result_list = []
                    entry_idx = 0
                    for i in range(len(entries_data)):
                        try:
                            if entry_idx < len(results):
                                result_list.append(results[entry_idx])
                                entry_idx += 1
                            else:
                                result_list.append(False)
                        except Exception as e:
                            logger.error(f"Error processing result for entry {i}: {e}")
                            result_list.append(False)
                    
                    return result_list
                else:
                    logger.warning("No valid entries to batch register")
                    return [False] * len(entries_data)
                    
            except Exception as e:
                self.failed_requests += 1
                logger.error(f"IPC batch_register_kvcache failed: {e}")
                return [False] * len(entries_data)

    def get_entry(self, key: CacheEngineKey) -> Optional[object]:
        """
        Get metadata entry by delegating to VRAM metadata IPC server.
        Returns a GPUVRAMEntry object.
        """
        with self.lock:
            self.total_requests += 1
            
            try:
                self._ensure_connection()
                
                # Serialize key for IPC transmission
                key_dict = self._serialize_key(key)
                
                # Make IPC call
                entry_data_proxy = self.server_proxy.get_entry(key_dict)
                
                # Handle AutoProxy object - call _getvalue() to get actual result
                if hasattr(entry_data_proxy, '_getvalue'):
                    entry_data = entry_data_proxy._getvalue()
                else:
                    entry_data = entry_data_proxy
                
                if entry_data:
                    # Create GPUVRAMEntry object from the dictionary data
                                  
                    # Deserialize the key
                    deserialized_key = self._deserialize_key(entry_data.get('key', {}))
                    
                    # Deserialize the dtype
                    tensor_dtype = self._deserialize_dtype(entry_data.get('tensor_dtype', 'torch.float16'))
                    
                    # Create GPUVRAMEntry object
                    entry = GPUVRAMEntry(
                        key=deserialized_key,
                        token_ids=entry_data.get('token_ids', []),
                        gpu_id=entry_data.get('gpu_id', 0),
                        tensor_shape=entry_data.get('tensor_shape', ()),
                        tensor_dtype=tensor_dtype,
                        tensor_size=entry_data.get('tensor_size', 0),
                        created_time=entry_data.get('created_time', 0.0),
                        last_access_time=entry_data.get('last_access_time', 0.0),
                        access_count=entry_data.get('access_count', 0),
                        is_pinned=entry_data.get('is_pinned', False),
                        buffer_pointer=entry_data.get('buffer_pointer'),
                        resident_hostname=entry_data.get('resident_hostname'),
                        transfer_in_progress=entry_data.get('transfer_in_progress', False),
                        transfer_target_gpu=entry_data.get('transfer_target_gpu'),
                        prefetch_priority=entry_data.get('prefetch_priority', 0),
                        segment_id=entry_data.get('segment_id')
                    )
                    
                    return entry
                else:
                    return None
                
            except Exception as e:
                self.failed_requests += 1
                logger.error(f"IPC get_entry failed: {e}")
                return None

    def get_ipc_mem_handle(self, address: int, gpu_id: Optional[int] = None) -> Optional[Tuple[bytes, int, int, int]]:
        """Request cudaIpcMemHandle bytes for GPU memory address from IPC server.

        Args:
            address: GPU memory address
            gpu_id: GPU ID to specify which GPU the address belongs to.
        
        Returns (handle_bytes, gpu_id, base_pointer, size) or None on failure.
        """
        with self.lock:
            self.total_requests += 1
            try:
                self._ensure_connection()
                address_dict = {'address': address}
                if gpu_id is not None:
                    address_dict['gpu_id'] = gpu_id
                res_proxy = self.server_proxy.get_ipc_mem_handle(address_dict)
                if hasattr(res_proxy, '_getvalue'):
                    res = res_proxy._getvalue()
                else:
                    res = res_proxy
                if res:
                    return res
                else:
                    return None
            except Exception as e:
                self.failed_requests += 1
                logger.error(f"IPC get_ipc_mem_handle failed: {e}")
                return None


    def register_segment_ipc_handle(self, 
                                    segment_id: str, 
                                    buffer_pointer: int, 
                                    handle_bytes: bytes, 
                                    gpu_id: int, 
                                    size: int) -> bool:
        """Register an IPC mem handle for a pre-allocated segment on this host.

        Returns True on success, False otherwise.
        """
        with self.lock:
            self.total_requests += 1
            try:
                self._ensure_connection()
                res_proxy = self.server_proxy.register_segment_ipc_handle(segment_id, 
                                                                          buffer_pointer, 
                                                                          handle_bytes, 
                                                                          gpu_id, 
                                                                          size)
                if hasattr(res_proxy, '_getvalue'):
                    res = res_proxy._getvalue()
                else:
                    res = res_proxy
                if res:
                    logger.info(f"Registered segment IPC handle via IPC: {segment_id} "
                                f"@ {hex(buffer_pointer)}")
                    return True
                else:
                    self.failed_requests += 1
                    logger.error("Server failed to register segment IPC handle")
                    return False
            except Exception as e:
                self.failed_requests += 1
                logger.error(f"IPC register_segment_ipc_handle failed: {e}")
                return False
    
    def unregister_segment_ipc_handle(self, segment_id: str, buffer_pointer: int, gpu_id: int) -> bool:
        """
        Unregister a previously registered segment IPC handle.
        
        Args:
            segment_id: Unique segment identifier
            buffer_pointer: Base GPU memory address of the segment
            gpu_id: GPU device ID
            
        Returns:
            True if unregistration successful, False otherwise
        """
        with self.lock:
            self.total_requests += 1
            try:
                self._ensure_connection()
                
                # First, we need to register the unregister method with the manager
                # Since we haven't registered it yet, we need to check if it's available
                # We'll try to call it and handle the case where it doesn't exist
                try:
                    res_proxy = self.server_proxy.unregister_segment_ipc_handle(segment_id, buffer_pointer, gpu_id)
                    if hasattr(res_proxy, '_getvalue'):
                        res = res_proxy._getvalue()
                    else:
                        res = res_proxy
                    
                    if res:
                        logger.info(f"Unregistered segment IPC handle via IPC: {segment_id} "
                                   f"@ GPU{gpu_id}:{hex(buffer_pointer)}")
                        return True
                    else:
                        self.failed_requests += 1
                        logger.error("Server failed to unregister segment IPC handle")
                        return False
                except AttributeError:
                    # Method not registered on server, log warning and return False
                    logger.warning(f"unregister_segment_ipc_handle method not available on server. "
                                  f"Please update the server to support this method.")
                    self.failed_requests += 1
                    return False
                    
            except Exception as e:
                self.failed_requests += 1
                logger.error(f"IPC unregister_segment_ipc_handle failed: {e}")
                return False

    def batch_get_entry(self, keys: List[CacheEngineKey]) -> List[Optional[object]]:
        """
        batch get metadata entries by delegating to VRAM metadata IPC server.
        Returns a list of GPUVRAMEntry objects or None.
        
        Args:
            keys: cache key list
            
        Returns:
            list of GPUVRAMEntry objects or None for each key
        """
        with self.lock:
            self.total_requests += 1
            
            try:
                self._ensure_connection()
                
                # Serialize keys for IPC transmission
                key_dicts = [self._serialize_key(key) for key in keys]
                
                # Make IPC call
                entries_data_proxy = self.server_proxy.batch_get_entry(key_dicts)
                
                # Handle AutoProxy object - call _getvalue() to get actual result
                if hasattr(entries_data_proxy, '_getvalue'):
                    entries_data = entries_data_proxy._getvalue()
                else:
                    entries_data = entries_data_proxy
                
                if not entries_data:
                    return [None] * len(keys)
                
                result = []
                for entry_data in entries_data:
                    if entry_data:
                        # Deserialize the key
                        deserialized_key = self._deserialize_key(entry_data.get('key', {}))
                        
                        # Deserialize the dtype
                        tensor_dtype = self._deserialize_dtype(entry_data.get('tensor_dtype', 'torch.float16'))
                        
                        # Create GPUVRAMEntry object
                        entry = GPUVRAMEntry(
                            key=deserialized_key,
                            token_ids=entry_data.get('token_ids', []),
                            gpu_id=entry_data.get('gpu_id', 0),
                            tensor_shape=entry_data.get('tensor_shape', ()),
                            tensor_dtype=tensor_dtype,
                            tensor_size=entry_data.get('tensor_size', 0),
                            created_time=entry_data.get('created_time', 0.0),
                            last_access_time=entry_data.get('last_access_time', 0.0),
                            access_count=entry_data.get('access_count', 0),
                            is_pinned=entry_data.get('is_pinned', False),
                            buffer_pointer=entry_data.get('buffer_pointer'),
                            resident_hostname=entry_data.get('resident_hostname'),
                            transfer_in_progress=entry_data.get('transfer_in_progress', False),
                            transfer_target_gpu=entry_data.get('transfer_target_gpu'),
                            prefetch_priority=entry_data.get('prefetch_priority', 0),
                            segment_id=entry_data.get('segment_id')
                        )
                        result.append(entry)
                    else:
                        result.append(None)
                
                return result
                
            except Exception as e:
                self.failed_requests += 1
                logger.error(f"IPC batch_get_entry failed: {e}")
                return [None] * len(keys)

    def remove(self, key: CacheEngineKey) -> bool:
        """Remove metadata entry by delegating to VRAM metadata IPC server."""
        with self.lock:
            self.total_requests += 1
            
            try:
                self._ensure_connection()
                
                # Serialize key for IPC transmission
                key_dict = self._serialize_key(key)
                
                # Make IPC call
                success = self.server_proxy.remove(key_dict)
                
                return success
                
            except Exception as e:
                self.failed_requests += 1
                logger.error(f"IPC remove failed: {e}")
                return False

    def get_stats(self) -> dict:
        """Get statistics from VRAM metadata IPC server."""
        with self.lock:
            self.total_requests += 1
            
            try:
                self._ensure_connection()
                
                # Get server stats
                server_stats = self.server_proxy.get_stats()
                
                # Combine server stats with client stats
                stats = {
                    "client_stats": {
                        "total_requests": self.total_requests,
                        "failed_requests": self.failed_requests,
                        "success_rate": (self.total_requests - self.failed_requests) / max(self.total_requests, 1),
                        "is_connected": self.is_connected,
                        "server_address": f"{self.server_address}:{self.server_port}"
                    },
                    "server_stats": server_stats
                }
                
                return stats
                
            except Exception as e:
                self.failed_requests += 1
                logger.error(f"IPC get_stats failed: {e}")
                return {
                    "client_stats": {
                        "total_requests": self.total_requests,
                        "failed_requests": self.failed_requests,
                        "success_rate": 0,
                        "is_connected": False,
                        "server_address": f"{self.server_address}:{self.server_port}"
                    },
                    "server_stats": {}
                }

    def health_check(self) -> bool:
        """Check if the VRAM metadata IPC server is healthy."""
        try:
            self._ensure_connection()
            health_info = self.server_proxy.health_check()
            return health_info.get("status") == "healthy" and health_info.get("server_running", False)
        except:
            return False

    def shutdown_server(self) -> bool:
        """Request server shutdown (for testing purposes)."""
        try:
            self._ensure_connection()
            return self.server_proxy.shutdown_server()
        except Exception as e:
            logger.error(f"IPC shutdown_server failed: {e}")
            return False

    def _serialize_key(self, key: CacheEngineKey) -> Dict:
        """Serialize CacheEngineKey to dictionary for IPC transmission."""
        return {
            'fmt': key.fmt,
            'model_name': key.model_name,
            'world_size': key.world_size,
            'worker_id': key.worker_id,
            'chunk_hash': key.chunk_hash,
            'dtype': str(key.dtype)
        }

    def _deserialize_key(self, key_dict: Dict) -> CacheEngineKey:
        """Deserialize dictionary to CacheEngineKey."""
        dtype = self._deserialize_dtype(key_dict.get('dtype', 'torch.float16'))
        
        return CacheEngineKey(
            fmt=key_dict.get('fmt', 'pt'),
            model_name=key_dict.get('model_name', 'test_model'),
            world_size=key_dict.get('world_size', 2),
            worker_id=key_dict.get('worker_id', 0),
            chunk_hash=key_dict.get('chunk_hash', 0),
            dtype=dtype
        )

    def _deserialize_dtype(self, dtype_str: str) -> torch.dtype:
        """Deserialize dtype string to torch.dtype."""
        dtype_map = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.int64": torch.int64,
        }
        return dtype_map.get(dtype_str, torch.float16)

    def contains(self, key: CacheEngineKey) -> bool:
        """Check if key exists in VRAM metadata IPC server."""
        with self.lock:
            self.total_requests += 1
            
            try:
                self._ensure_connection()
                
                # Serialize key for IPC transmission
                key_dict = self._serialize_key(key)
                
                # Make IPC call to check if key exists
                # We can use get_entry and check if it returns None
                entry_data_proxy = self.server_proxy.get_entry(key_dict)
                
                # Handle AutoProxy object - call _getvalue() to get actual result
                if hasattr(entry_data_proxy, '_getvalue'):
                    entry_data = entry_data_proxy._getvalue()
                else:
                    entry_data = entry_data_proxy
                
                return entry_data is not None
                
            except Exception as e:
                self.failed_requests += 1
                logger.error(f"IPC contains check failed: {e}")
                return False

    def shutdown(self) -> bool:
        """Shutdown the VRAM metadata IPC client."""
        logger.info("Shutting down VRAM Metadata IPC Client")
        
        # Close connection
        if self.manager:
            try:
                self.manager.shutdown()
            except:
                pass
        
        self.is_connected = False
        self.server_proxy = None
        self.manager = None
        
        logger.info("VRAM Metadata IPC Client shutdown completed")
        return True


# Factory function for easy instantiation
def get_vram_metadata_ipc_client(config):
    """Factory function to get VRAM metadata IPC client instance."""
    return VRAMMetadataIPCClient(config)
