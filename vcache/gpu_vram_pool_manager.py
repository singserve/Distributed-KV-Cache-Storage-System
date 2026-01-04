# Standard
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import threading
import time

# Third Party
import torch

# First Party
from lmcache.vcache.vcache_logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)

@dataclass
class GPUVRAMEntry:
    """Metadata entry for GPU VRAM pool."""
    key: CacheEngineKey
    token_ids: List[int]
    gpu_id: int
    tensor_shape: tuple
    tensor_dtype: torch.dtype
    tensor_size: int  # in bytes
    created_time: float
    last_access_time: float
    access_count: int
    is_pinned: bool = False
    # GPU buffer pointer - actual GPU memory address
    buffer_pointer: Optional[int] = None
    # Cross-GPU transfer tracking
    resident_hostname: Optional[str] = None
    transfer_in_progress: bool = False
    transfer_target_gpu: Optional[int] = None
    prefetch_priority: int = 0  # Higher value = higher prefetch priority
    # Segment tracking
    segment_id: Optional[str] = None  # Associated GPU VRAM segment
    segment_offset: int = 0  # Offset within the segment (in bytes)


class GPUVRAMPoolManager:
    """
    GPU VRAM pool manager.
    Manages metadata for KV cache chunks stored in the entire GPU VRAM pool.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls, config):
        """Get instance of GPU VRAM pool manager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance
    
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        
        # Metadata storage: key -> GPUVRAMEntry
        self.metadata: Dict[CacheEngineKey, GPUVRAMEntry] = {}
        
        # GPU-specific metadata tracking
        self.gpu_metadata: Dict[int, Set[CacheEngineKey]] = {}  # gpu_id -> set of keys
        
        # Statistics
        self.total_entries = 0
        self.total_size_bytes = 0
        self.max_metadata_size = config.get_extra_config_value(
            "max_gpu_vram_metadata_size", 50000  # Maximum number of metadata entries
        )
        
        # Initialize GPU tracking
        self._initialize_gpu_tracking()
        
        logger.info(f"GPU VRAM Pool Manager initialized for {len(self.gpu_metadata)} GPUs")

    def _initialize_gpu_tracking(self):
        """Initialize tracking for all available GPUs."""
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            self.gpu_metadata[gpu_id] = set()
        logger.info(f"Tracking {num_gpus} GPUs in VRAM pool")

    def lookup_prefix(
        self,
        token_ids: List[int],
        all_chunks: List[Tuple[int, int, CacheEngineKey]],
        current_gpu_id: Optional[int] = None,
    ) -> Tuple[int, Optional[List[Tuple[Tuple[int, int], int, bool]]]]:
        """
        Lookup prefix match in GPU VRAM pool metadata.
        Requires all_chunks parameter - finds continuous matching chunks from the beginning.
        Only returns continuous hits from the beginning (start=0).
        
        Returns (hit_tokens, chunk_info_list) where:
        - hit_tokens: number of matching tokens (only continuous from the beginning)
        - chunk_info_list: list of tuples for each matching chunk, each tuple contains:
            - (start, end): chunk range
            - gpu_id: GPU ID where the chunk is stored
            - needs_transfer: True if this chunk needs cross-GPU transfer
          Returns None if no match
        """
        with self.lock:
            # Handle edge cases - all_chunks is required
            if not all_chunks:
                logger.warning("No all_chunks provided, returning 0 (no match)")
                return 0, None
            
            # Handle edge cases for token_ids
            if not token_ids:
                return 0, None
            
            # find continuous matching chunks from the beginning
            if len(all_chunks) > 0:
                
                # Only check continuous chunks from the beginning (start=0)
                continuous_hit_tokens = 0
                expected_start = 0
                chunk_info_list = []
                
                for start, end, chunk_cache_key in all_chunks:
                    # Check if this chunk is continuous from the beginning
                    if start != expected_start:
                        # Found a gap, stop checking
                        logger.info(f"Breaking at gap: expected start={expected_start}, "
                                    f"got start={start}, stopping GPU VRAM lookup")
                        break
                    
                    # Get chunk tokens from token_ids
                    chunk_tokens = token_ids[start:end]
                    chunk_length = end - start
                    logger.debug(f"Checking GPU VRAM for continuous chunk [{start}, {end}): "
                                 f"{chunk_length} tokens,"
                                 f"key: {chunk_cache_key}, "
                                 f"chunk_hash: {chunk_cache_key.chunk_hash if hasattr(chunk_cache_key, 'chunk_hash') else 'N/A'}")
                    
                    # Check if this chunk exists in metadata
                    if chunk_cache_key in self.metadata:
                        entry = self.metadata[chunk_cache_key]
                        
                        # Check if cross-GPU transfer is needed for this chunk
                        chunk_needs_transfer = False
                        if (current_gpu_id is not None and 
                            entry.gpu_id != current_gpu_id and
                            not entry.transfer_in_progress):
                            chunk_needs_transfer = True
                        
                        # Update access time
                        entry.last_access_time = time.time()
                        entry.access_count += 1
                        
                        chunk_hit_tokens = min(len(entry.token_ids), len(chunk_tokens))
                        
                        # Check if the entire chunk is hit
                        if chunk_hit_tokens == len(chunk_tokens):
                            # Entire chunk is hit, add to continuous hit tokens
                            continuous_hit_tokens += chunk_hit_tokens
                            expected_start = end
                            
                            # Add chunk info to the list
                            chunk_info = ((start, end), entry.gpu_id, chunk_needs_transfer)
                            chunk_info_list.append(chunk_info)
                            
                            logger.debug(f"Found GPU VRAM hit for continuous chunk [{start}, {end}): "
                                         f"{chunk_hit_tokens} tokens,"
                                         f"GPU={entry.gpu_id}, "
                                         f"needs_transfer={chunk_needs_transfer}")
                        else:
                            # Chunk not fully hit, stop checking
                            logger.debug(f"Breaking at chunk [{start}, {end}):"
                                         f"expected {len(chunk_tokens)} tokens,"
                                         f"got {chunk_hit_tokens} tokens, "
                                         f"stopping GPU VRAM lookup")
                            break
                    else:
                        # Chunk not found in metadata, stop checking
                        logger.debug(f"Breaking at chunk [{start}, {end}): chunk not found in GPU VRAM metadata, stopping lookup")
                        break
                
                if continuous_hit_tokens > 0:
                    logger.info(f"GPU VRAM pool hit from continuous chunks: {continuous_hit_tokens} tokens,"
                                f"{len(chunk_info_list)} chunks matched")
                    return continuous_hit_tokens, chunk_info_list
                else:
                    logger.info("No continuous GPU VRAM hits found from the beginning")
                    return 0, None
            
            # return 0 directly when all_chunks is not provided
            logger.warning("No all_chunks provided, returning 0 (no match)")
            return 0, None

    def get_stats(self) -> dict:
        """Get comprehensive GPU VRAM pool statistics."""
        with self.lock:
            stats = {
                "total_entries": self.total_entries,
                "total_size_bytes": self.total_size_bytes,
                "max_metadata_size": self.max_metadata_size,
                "gpu_distribution": {
                    gpu_id: len(keys) 
                    for gpu_id, keys in self.gpu_metadata.items()
                },
                "transfer_stats": {},
            }
            
            # Calculate hit rate if we have access tracking
            total_accesses = sum(entry.access_count for entry in self.metadata.values())
            stats["total_accesses"] = total_accesses
            
            return stats

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
        Register a new KV cache in GPU VRAM pool metadata.
        Called when vLLM generates new KV cache on any GPU.

        Args:
            cache_key: CacheEngineKey generated by TokenDatabase in store function
            token_ids: List of token IDs (already processed as chunk by TokenDatabase)
            gpu_id: GPU ID where the KV cache is stored
            tensor_shape: Shape of the KV cache tensor
            tensor_dtype: Data type of the KV cache tensor
            tensor_size: Size of the KV cache tensor in bytes
            buffer_pointer: Optional GPU buffer pointer address
            segment_id: Optional segment ID if the cache is stored in a segment
            segment_offset: Offset within the segment (in bytes)

        Returns:
            True if registration successful, False otherwise
        """
        with self.lock:
            # Check if we've reached the metadata limit
            if self.total_entries >= self.max_metadata_size:
                if not self._evict_oldest_entry():
                    logger.warning("GPU VRAM pool metadata full, cannot register new entry")
                    return False

            current_time = time.time()
            
            entry = GPUVRAMEntry(
                key=cache_key,
                token_ids=token_ids.copy(), 
                gpu_id=gpu_id,
                tensor_shape=tensor_shape, 
                tensor_dtype=tensor_dtype,
                tensor_size=tensor_size,
                created_time=current_time,
                last_access_time=current_time,
                access_count=1,
                buffer_pointer=buffer_pointer,
                resident_hostname=resident_hostname,
                segment_id=segment_id,
                segment_offset=segment_offset
            )
            
            # store metadata
            self.metadata[cache_key] = entry
            self.total_entries += 1
            self.total_size_bytes += tensor_size
            
            # Track GPU-specific metadata
            if gpu_id not in self.gpu_metadata:
                self.gpu_metadata[gpu_id] = set()
            self.gpu_metadata[gpu_id].add(cache_key)
            
            # Log detailed information
            logger.info(
                f"Registered GPU VRAM pool entry - "
                f"Key chunk_hash: {cache_key.chunk_hash}, "
                f"Token IDs: {token_ids}, "
                f"Token length: {len(token_ids)}, "
                f"GPU: {gpu_id}, "
                f"Segment: {segment_id if segment_id else 'external'}, "
                f"Segment offset: {segment_offset} bytes, "
                f"Size: {tensor_size} bytes"
            )
            return True

    def batch_register_kvcache(
        self,
        entries_data: List[Tuple[CacheEngineKey, 
                                 List[int], 
                                 int, 
                                 tuple, 
                                 torch.dtype, 
                                 int, 
                                 Optional[int], 
                                 Optional[str], 
                                 Optional[str],
                                 int]]  # New format with segment_offset as required parameter
    ) -> List[bool]:
        """
        batch register KV cache to GPU VRAM pool metadata
        
        Args:
            entries_data: each entry contains:
                (cache_key, token_ids, gpu_id, tensor_shape, tensor_dtype, tensor_size, 
                 buffer_pointer, segment_id, resident_hostname, segment_offset)
            
        Returns:
            list of bool indicating success/failure for each entry
        """
        with self.lock:
            results = []
            current_time = time.time()
            
            for entry_data in entries_data:
                try:
                    # unpack entry data - new format with segment_offset
                    (cache_key, token_ids, gpu_id, tensor_shape, tensor_dtype, 
                     tensor_size, buffer_pointer, segment_id, resident_hostname, segment_offset) = entry_data
                    
                    # check metadata limit
                    if self.total_entries >= self.max_metadata_size:
                        if not self._evict_oldest_entry():
                            logger.warning("GPU VRAM pool metadata full, cannot register new entry")
                            results.append(False)
                            continue
                    
                    entry = GPUVRAMEntry(
                        key=cache_key,
                        token_ids=token_ids.copy(),
                        gpu_id=gpu_id,
                        tensor_shape=tensor_shape,
                        tensor_dtype=tensor_dtype,
                        tensor_size=tensor_size,
                        created_time=current_time,
                        last_access_time=current_time,
                        access_count=1,
                        buffer_pointer=buffer_pointer,
                        resident_hostname=resident_hostname,
                        segment_id=segment_id,
                        segment_offset=segment_offset
                    )
                    
                    # store metadata
                    self.metadata[cache_key] = entry
                    self.total_entries += 1
                    self.total_size_bytes += tensor_size
                    
                    # Track GPU-specific metadata
                    if gpu_id not in self.gpu_metadata:
                        self.gpu_metadata[gpu_id] = set()
                    self.gpu_metadata[gpu_id].add(cache_key)
                    
                    logger.debug(
                        f"Batch registered GPU VRAM pool entry - "
                        f"Key chunk_hash: {cache_key.chunk_hash}, "
                        f"Token length: {len(token_ids)}, "
                        f"GPU: {gpu_id}, "
                        f"Segment offset: {segment_offset} bytes, "
                        f"Size: {tensor_size} bytes"
                    )
                    
                    results.append(True)
                    
                except Exception as e:
                    logger.error(f"Failed to batch register entry: {e}")
                    results.append(False)
            
            logger.info(f"Batch registered {sum(results)} out of {len(entries_data)} entries")
            return results

    def contains(self, key: CacheEngineKey) -> bool:
        """Check if key exists in GPU VRAM pool metadata."""
        with self.lock:
            return key in self.metadata

    def get_entry(self, key: CacheEngineKey) -> Optional[GPUVRAMEntry]:
        """Get metadata entry and update access time."""
        with self.lock:
            if key in self.metadata:
                entry = self.metadata[key]
                entry.last_access_time = time.time()
                entry.access_count += 1
                return entry
            return None

    def batch_get_entry(self, keys: List[CacheEngineKey]) -> List[Optional[GPUVRAMEntry]]:
        """
        batch get metadata entries
        
        Args:
            keys: cache key list
            
        Returns:
            corresponding metadata entry list, if a key does not exist return None
        """
        with self.lock:
            result = []
            for key in keys:
                if key in self.metadata:
                    entry = self.metadata[key]
                    entry.last_access_time = time.time()
                    entry.access_count += 1
                    result.append(entry)
                else:
                    result.append(None)
            return result

    def remove(self, key: CacheEngineKey) -> bool:
        """Remove metadata entry."""
        with self.lock:
            if key in self.metadata:
                entry = self.metadata[key]
                # Remove from GPU tracking
                if entry.gpu_id in self.gpu_metadata:
                    self.gpu_metadata[entry.gpu_id].discard(key)
                # Remove metadata
                del self.metadata[key]
                self.total_entries -= 1
                self.total_size_bytes -= entry.tensor_size
                logger.info(f"Removed metadata entry for key: {key},"
                            f"chunk_hash: {key.chunk_hash}, "
                            f"GPU: {entry.gpu_id}")
                return True
            logger.warning(f"Key not found in metadata: {key}, "
                           f"chunk_hash: {key.chunk_hash if hasattr(key, 'chunk_hash') else 'N/A'}")
            return False

    def _evict_oldest_entry(self) -> bool:
        """Evict the oldest entry to make space for new entries."""
        if not self.metadata:
            return False
        
        # Find the oldest entry
        oldest_key = None
        oldest_time = float('inf')
        
        for key, entry in self.metadata.items():
            if entry.last_access_time < oldest_time:
                oldest_time = entry.last_access_time
                oldest_key = key
        
        if oldest_key:
            logger.debug(f"Evicting oldest entry: {oldest_key}")
            return self.remove(oldest_key)
        else:
            return False

    def shutdown(self) -> bool:
        """
        Shutdown the GPU VRAM pool manager and release all resources.
        This should be called when the program is exiting.
        
        Returns:
            True if shutdown successful, False otherwise
        """
        logger.info("Shutting down GPU VRAM pool manager")
        
        # Clear all metadata
        self.metadata.clear()
        self.gpu_metadata.clear()
        
        # Reset statistics
        self.total_entries = 0
        self.total_size_bytes = 0
        
        # Reset singleton instance
        GPUVRAMPoolManager._instance = None
        
        logger.info("GPU VRAM pool manager shutdown completed")
