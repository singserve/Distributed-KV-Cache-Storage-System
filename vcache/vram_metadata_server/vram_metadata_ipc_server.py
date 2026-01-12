"""
GPU VRAM Metadata IPC Server

A centralized metadata server using IPC (Inter-Process Communication)
"""


from typing import Dict, List, Optional, Tuple, Any
import threading
import time
from multiprocessing.managers import BaseManager
import torch

from lmcache.vcache.logging.vcache_logging import init_logger
from lmcache.vcache.utils import VCacheKey
from lmcache.vcache.vram_metadata_server.gpu_vram_pool_manager import GPUVRAMPoolManager
logger = init_logger(__name__)


class VRAMMetadataIPCServer:
    """
    IPC-based VRAM metadata server that:
    - Instantiates a GPU VRAM pool manager internally
    - Provides IPC interface for vcache engine instances
    """
    
    def __init__(self, config, authkey: bytes = b'vram_metadata'):
        self.config = config
        self.address = config.get_extra_config_value(
            "vram_metadata_ipc_address", "192.168.1.86"
        )
        self.port = config.get_extra_config_value(
            "vram_metadata_ipc_port", 9091
        )
        self.authkey = authkey
        
        # Import and instantiate the GPU VRAM pool manager using the config       
        self.vram_pool_manager = GPUVRAMPoolManager.get_instance(config)
        
        # IPC server
        self.manager = None
        self.server = None
        self.is_running = False
        
        # Lock for thread safety - use threading.RLock for reentrant locks
        self.lock = threading.RLock()
        
        # Client connection tracking
        self.total_requests = 0
        self.request_stats = {
            'lookup_prefix': 0,
            'register_kvcache': 0,
            'get_entry': 0,
            'remove': 0,
            'get_stats': 0,
            'health_check': 0
        }
        # Mapping for registered segment IPC handles: base_ptr -> {segment_id, handle_bytes, gpu_id, size, time}
        self.segment_ipc_handles: Dict[int, Dict] = {}
        
        logger.info(f"GPU VRAM Metadata IPC Server initialized on {self.address}:{self.port}")

    def start(self):
        """Start the IPC metadata server."""
        # Create a custom manager class
        class VRAMManager(BaseManager):
            pass
        
        # Register methods with the manager
        VRAMManager.register('lookup_prefix', self.lookup_prefix)
        VRAMManager.register('register_kvcache', self.register_kvcache)
        VRAMManager.register('batch_register_kvcache', self.batch_register_kvcache)
        VRAMManager.register('get_entry', self.get_entry)
        VRAMManager.register('batch_get_entry', self.batch_get_entry)
        VRAMManager.register('get_ipc_mem_handle', self.get_ipc_mem_handle)
        VRAMManager.register('register_segment_ipc_handle', self.register_segment_ipc_handle)
        VRAMManager.register('unregister_segment_ipc_handle', self.unregister_segment_ipc_handle)
        VRAMManager.register('remove', self.remove)
        VRAMManager.register('get_stats', self.get_stats)
        VRAMManager.register('health_check', self.health_check)
        VRAMManager.register('shutdown_server', self.shutdown_server)
        
        # Create and start the manager
        self.manager = VRAMManager(address=(self.address, self.port), authkey=self.authkey)
        self.server = self.manager.get_server()
        self.is_running = True
        
        logger.info(f"GPU VRAM Metadata IPC Server started on {self.address}:{self.port}")
        
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"IPC server error: {e}")
            self.stop()

    def stop(self):
        """Stop the IPC metadata server."""
        if self.manager:
            try:
                self.manager.shutdown()
            except:
                pass
        self.is_running = False
        logger.info("GPU VRAM Metadata IPC Server stopped")

    def lookup_prefix(
        self,
        token_ids: List[int],
        all_chunks: Optional[List[Tuple[int, int, Dict]]] = None,
        current_gpu_id: Optional[int] = None,
    ) -> Tuple[int, Optional[List[Tuple[Tuple[int, int], int, bool]]]]:
        """
        Lookup prefix match by delegating to internal pool manager.
        Returns serialized data for IPC transmission.
        """
        with self.lock:
            try:
                # Track request
                self.total_requests += 1
                self.request_stats['lookup_prefix'] += 1
                
                # Log request details
                logger.info(f"IPC lookup_prefix request - "
                            f"Tokens: {len(token_ids)}, "
                            f"GPU: {current_gpu_id}")
                
                # Deserialize all_chunks if provided
                deserialized_chunks = None
                if all_chunks is not None:
                    deserialized_chunks = []
                    for start, end, key_dict in all_chunks:
                        cache_key = self._deserialize_key(key_dict)
                        deserialized_chunks.append((start, end, cache_key))
                
                # Call the internal pool manager's lookup_prefix method
                hit_tokens, chunk_info_list = self.vram_pool_manager.lookup_prefix(
                    token_ids=token_ids,
                    all_chunks=deserialized_chunks,
                    current_gpu_id=current_gpu_id,
                )
                
                # Log detailed results
                logger.info(f"IPC lookup result - Hits: {hit_tokens}, "
                            f"Chunks: {len(chunk_info_list) if chunk_info_list else 0}")
                if hit_tokens > 0 and chunk_info_list:
                    for i, ((start, end), gpu_id, needs_transfer) in enumerate(chunk_info_list):
                        logger.debug(f"Chunk {i}: [{start}, {end}) -> GPU {gpu_id}, needs_transfer={needs_transfer}")
                
                # Ensure we return a proper tuple for IPC transmission
                result_tuple = (hit_tokens, chunk_info_list)
                logger.debug(f"Returning result tuple: {result_tuple}")
                return result_tuple
                
            except Exception as e:
                logger.error(f"Lookup error: {e}")
                return 0, None

    def register_kvcache(
        self,
        key_dict: Dict, 
        token_ids: List[int],
        gpu_id: int,
        tensor_shape: tuple,
        tensor_dtype_str: str,
        tensor_size: int,
        buffer_pointer: Optional[int] = None,
        segment_id: Optional[str] = None,
        resident_hostname: Optional[str] = None,
        segment_offset: int = 0,
    ) -> bool:
        """
        Register KV cache by delegating to internal pool manager.
        Accepts serialized data for IPC transmission.
        """
        with self.lock:
            try:
                # Track request
                self.total_requests += 1
                self.request_stats['register_kvcache'] += 1
                
                # Deserialize key and dtype
                cache_key = self._deserialize_key(key_dict)
                dtype = self._deserialize_dtype(tensor_dtype_str)
                
                # Log request details
                logger.info(f"IPC register_kvcache request - "
                            f"Key: {cache_key.chunk_hash}, "
                            f"GPU: {gpu_id}, "
                            f"Tokens: {len(token_ids)}")
                
                # Call the internal pool manager's register_kvcache method
                success = self.vram_pool_manager.register_kvcache(
                    cache_key=cache_key,
                    token_ids=token_ids,
                    gpu_id=gpu_id,
                    tensor_shape=tensor_shape,
                    tensor_dtype=dtype,
                    tensor_size=tensor_size,
                    buffer_pointer=buffer_pointer,
                    segment_id=segment_id,
                    resident_hostname=resident_hostname,
                    segment_offset=segment_offset
                )
                
                if success:
                    logger.info(f"Successfully registered KV cache via IPC -"
                                f"Key: {cache_key.chunk_hash}, "
                                f"GPU: {gpu_id}, "
                                f"Tokens: {len(token_ids)}, "
                                f"Segment offset: {segment_offset} bytes")
                    return True
                else:
                    logger.error("Registration failed in pool manager")
                    return False
                    
            except Exception as e:
                logger.error(f"Registration error: {e}")
                return False

    def batch_register_kvcache(
        self,
        entries_data: List[Tuple[Dict, List[int], int, tuple, str, int, Optional[int], Optional[str], Optional[str], int]]
    ) -> List[bool]:
        """
        batch register KV cache to GPU VRAM pool metadata
        
        Args:
            entries_data: list of serialized entry data tuples
                (key_dict, token_ids, gpu_id, tensor_shape, tensor_dtype_str, tensor_size, 
                 buffer_pointer, segment_id, resident_hostname, segment_offset)
            
        Returns:
            list of bool indicating success for each entry
        """
        with self.lock:
            try:
                # Track request
                self.total_requests += 1
                self.request_stats['register_kvcache'] += 1
                
                batch_entries = []
                
                for entry_data in entries_data:
                    try:
                        # unpack entry data - new format with segment_offset
                        (key_dict, token_ids, gpu_id, tensor_shape, tensor_dtype_str, 
                         tensor_size, buffer_pointer, segment_id, resident_hostname, segment_offset) = entry_data
                        
                        # Deserialize key and dtype
                        cache_key = self._deserialize_key(key_dict)
                        dtype = self._deserialize_dtype(tensor_dtype_str)
                        
                        batch_entries.append((
                            cache_key, token_ids, gpu_id, tensor_shape, dtype, 
                            tensor_size, buffer_pointer, segment_id, resident_hostname, segment_offset
                        ))
                        
                    except Exception as e:
                        logger.error(f"Failed to deserialize entry data: {e}")
                        continue
                
                # Call the internal pool manager's batch_register_kvcache method
                if batch_entries:
                    results = self.vram_pool_manager.batch_register_kvcache(batch_entries)
                    
                    success_count = sum(results)
                    logger.info(f"IPC batch_register_kvcache processed {len(batch_entries)} entries, "
                                f"{success_count} successful")
                    
                    # list to hold final results
                    result_list = []
                    entry_idx = 0
                    for i in range(len(entries_data)):
                        try:
                            # check if this entry was deserialized succcessfully
                            if entry_idx < len(results):
                                result_list.append(results[entry_idx])
                                entry_idx += 1
                            else:
                                # false
                                result_list.append(False)
                        except Exception as e:
                            logger.error(f"Error processing result for entry {i}: {e}")
                            result_list.append(False)
                    
                    return result_list
                else:
                    logger.warning("No valid entries to batch register")
                    return [False] * len(entries_data)
                    
            except Exception as e:
                logger.error(f"Batch registration error: {e}")
                return [False] * len(entries_data)

    def get_entry(self, key_dict: Dict) -> Optional[Dict]:
        """
        Get metadata entry by delegating to internal pool manager.
        Accepts and returns serialized data for IPC transmission.
        """
        with self.lock:
            try:
                # Deserialize key
                key = self._deserialize_key(key_dict)
                
                # Call the internal pool manager's get_entry method
                entry = self.vram_pool_manager.get_entry(key)
                
                if entry:
                    # Serialize entry for IPC transmission - match client expectations
                    entry_data = {
                        "key": {
                            'fmt': entry.key.fmt,
                            'model_name': entry.key.model_name,
                            'chunk_hash': entry.key.chunk_hash,
                            'dtype': str(entry.key.dtype)
                        },
                        "token_ids": entry.token_ids,
                        "gpu_id": entry.gpu_id,
                        "tensor_shape": entry.tensor_shape,
                        "tensor_dtype": str(entry.tensor_dtype),
                        "tensor_size": entry.tensor_size,
                        "buffer_pointer": entry.buffer_pointer,
                        "segment_id": entry.segment_id,
                        "segment_offset": entry.segment_offset,
                        "resident_hostname": entry.resident_hostname,
                        "created_time": entry.created_time,
                        "last_access_time": entry.last_access_time,
                        "access_count": entry.access_count,
                        "is_pinned": entry.is_pinned,
                        "transfer_in_progress": entry.transfer_in_progress,
                        "transfer_target_gpu": entry.transfer_target_gpu,
                        "prefetch_priority": entry.prefetch_priority
                    }
                    return entry_data
                else:
                    return None
                    
            except Exception as e:
                logger.error(f"Get entry error: {e}")
                return None

    def batch_get_entry(self, key_dicts: List[Dict]) -> List[Optional[Dict]]:
        """
        batch get metadata entries by delegating to internal pool manager.
        Accepts and returns serialized data for IPC transmission.
        
        Args:
            key_dicts: list of serialized key dict
            
        Returns:
            list of serialized entry dicts or None for each key
        """
        with self.lock:
            try:
                # Track request
                self.total_requests += 1
                self.request_stats['get_entry'] += 1
                
                # Deserialize keys
                keys = [self._deserialize_key(key_dict) for key_dict in key_dicts]
                
                # Call the internal pool manager's batch_get_entry method
                entries = self.vram_pool_manager.batch_get_entry(keys)
                
                # Serialize entries for IPC transmission
                result = []
                for entry in entries:
                    if entry:
                        # Serialize entry for IPC transmission
                        entry_data = {
                            "key": {
                                'fmt': entry.key.fmt,
                                'model_name': entry.key.model_name,
                                'chunk_hash': entry.key.chunk_hash,
                                'dtype': str(entry.key.dtype)
                            },
                            "token_ids": entry.token_ids,
                            "gpu_id": entry.gpu_id,
                            "tensor_shape": entry.tensor_shape,
                            "tensor_dtype": str(entry.tensor_dtype),
                            "tensor_size": entry.tensor_size,
                            "buffer_pointer": entry.buffer_pointer,
                            "segment_id": entry.segment_id,
                            "segment_offset": entry.segment_offset,
                            "resident_hostname": entry.resident_hostname,
                            "created_time": entry.created_time,
                            "last_access_time": entry.last_access_time,
                            "access_count": entry.access_count,
                            "is_pinned": entry.is_pinned,
                            "transfer_in_progress": entry.transfer_in_progress,
                            "transfer_target_gpu": entry.transfer_target_gpu,
                            "prefetch_priority": entry.prefetch_priority
                        }
                        result.append(entry_data)
                    else:
                        result.append(None)
                
                logger.info(f"IPC batch_get_entry processed {len(keys)} keys, "
                            f"found {len([e for e in entries if e])} entries")
                return result
                    
            except Exception as e:
                logger.error(f"Batch get entry error: {e}")
                return [None] * len(key_dicts)

    def get_ipc_mem_handle(self, address_dict: Dict):
        """Return pre-registered serialized cudaIpcMemHandle for a GPU memory address.

        This method will only return a handle that was previously registered using
        `register_segment_ipc_handle`. 

        Args:
            address_dict: Dictionary with 'address' key containing the GPU memory address
                and optionally 'gpu_id' key for the GPU ID.
                Example: {'address': 1234567890, 'gpu_id': 0}
        
        Returns tuple (handle_bytes, gpu_id, base_pointer, size) or None on failure.
        """
        with self.lock:
            try:
                # Direct address lookup
                if 'address' not in address_dict:
                    logger.error("Missing 'address' key in request")
                    return None
                    
                buffer_ptr = int(address_dict['address'])
                requested_gpu_id = address_dict.get('gpu_id')
                
                # Look for a registered segment IPC handle that covers this buffer pointer
                for composite_key, info in self.segment_ipc_handles.items():
                    # composite_key is (gpu_id, base_ptr)
                    gpu_id, base_ptr = composite_key
                    seg_base = int(base_ptr)
                    seg_size = int(info.get('size', 0))
                    
                    # Check if this handle matches the requested GPU (if specified)
                    if requested_gpu_id is not None and gpu_id != requested_gpu_id:
                        continue
                    
                    # Check if the address is within this segment
                    if buffer_ptr >= seg_base and buffer_ptr < seg_base + seg_size:
                        logger.info(f"Found pre-registered segment IPC handle "
                                     f"for address {hex(buffer_ptr)} "
                                     f"on GPU {gpu_id}: {info.get('segment_id')} "
                                     f"@ {hex(seg_base)}")
                        return (info.get('handle_bytes'), info.get('gpu_id'), seg_base, seg_size)
                
                logger.error(f"No pre-registered IPC handle found for address {hex(buffer_ptr)}" + 
                             (f" on GPU {requested_gpu_id}" if requested_gpu_id is not None else "") + 
                             "; returning None")
                return None

            except Exception as e:
                logger.error(f"Failed to get IPC mem handle: {e}")
                return None



    def remove(self, key_dict: Dict) -> bool:
        """Remove metadata entry by delegating to internal pool manager."""
        with self.lock:
            try:
                key = self._deserialize_key(key_dict)
                success = self.vram_pool_manager.remove(key)
                return success
                
            except Exception as e:
                logger.error(f"Remove error: {e}")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from internal pool manager."""
        with self.lock:
            try:
                # Track request
                self.total_requests += 1
                self.request_stats['get_stats'] += 1
                
                logger.info("IPC get_stats request")
                
                manager_stats = self.vram_pool_manager.get_stats()
                # Add server info and request statistics
                stats = {
                    'address': self.address,
                    'port': self.port,
                    'is_running': self.is_running,
                    'total_requests': self.total_requests,
                    'request_stats': self.request_stats.copy(),
                    'manager_stats': manager_stats
                }
                
                logger.debug(f"Stats response: {stats}")
                return stats
            except Exception as e:
                logger.error(f"Get stats error: {e}")
                return {}

    def health_check(self) -> Dict[str, Any]:
        """Health check for IPC server."""
        return {
            "status": "healthy",
            "server_running": self.is_running,
            "timestamp": time.time(),
            "address": self.address,
            "port": self.port
        }

    def shutdown_server(self) -> bool:
        """Shutdown the IPC server (called by client)."""
        logger.info("Shutdown requested by client")
        self.stop()
        return True

    def register_segment_ipc_handle(self, 
                                    segment_id: str, 
                                    buffer_pointer: int, 
                                    handle_bytes: bytes, 
                                    gpu_id: int, 
                                    size: int) -> bool:
        """Register an IPC mem handle for a pre-allocated segment.

        Stores the handle bytes keyed by a composite key (gpu_id, buffer_pointer) 
        to avoid conflicts between different GPUs with the same address.
        """
        with self.lock:
            try:
                if not buffer_pointer or buffer_pointer == 0:
                    logger.error("Invalid buffer_pointer for segment IPC registration")
                    return False
                # Create a composite key with GPU ID to avoid conflicts
                composite_key = (gpu_id, int(buffer_pointer))
                # Overwrite or set the mapping
                self.segment_ipc_handles[composite_key] = {
                    'segment_id': segment_id,
                    'handle_bytes': handle_bytes,
                    'gpu_id': gpu_id,
                    'size': size,
                    'registered_time': time.time()
                }
                logger.info(f"Registered segment IPC handle: {segment_id} "
                            f"@ GPU{gpu_id}:{hex(buffer_pointer)} "
                            f"(size={size})")
                return True
            except Exception as e:
                logger.error(f"Failed to register segment IPC handle: {e}")
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
            try:
                # Create composite key for lookup
                composite_key = (gpu_id, int(buffer_pointer))
                
                # Check if the segment exists
                if composite_key not in self.segment_ipc_handles:
                    logger.warning(f"No registered segment IPC handle found for "
                                  f"segment {segment_id} @ GPU{gpu_id}:{hex(buffer_pointer)}")
                    return False
                
                # Verify segment_id matches
                stored_info = self.segment_ipc_handles[composite_key]
                if stored_info.get('segment_id') != segment_id:
                    logger.warning(f"Segment ID mismatch for unregistration: "
                                  f"expected {stored_info.get('segment_id')}, got {segment_id}")
                    return False
                
                # Remove the segment from registry
                del self.segment_ipc_handles[composite_key]
                
                logger.info(f"Unregistered segment IPC handle: {segment_id} "
                           f"@ GPU{gpu_id}:{hex(buffer_pointer)}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unregister segment IPC handle: {e}")
                return False
            
    def _deserialize_key(self, key_dict: Dict) -> VCacheKey:
        """Deserialize key dictionary to VCacheKey."""
        dtype = self._deserialize_dtype(key_dict.get('dtype', 'torch.float16'))
        
        return VCacheKey(
            fmt=key_dict.get('fmt', 'pt'),
            model_name=key_dict.get('model_name', 'test_model'),
            chunk_hash=key_dict.get('chunk_hash', 0),
            dtype=dtype
        )

    def _deserialize_dtype(self, dtype_str: str) -> torch.dtype:
        """Deserialize dtype string to torch.dtype."""
        from lmcache.vcache.utils import str_to_dtype
        return str_to_dtype(dtype_str)


def start_vram_metadata_ipc_server(config=None):
    """Convenience function to start the IPC metadata server."""
    server = VRAMMetadataIPCServer(config)
    server.start()
    return server
