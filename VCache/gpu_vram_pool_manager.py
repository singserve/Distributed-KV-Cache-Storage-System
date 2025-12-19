# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
import asyncio
import threading
import time
import ctypes
import uuid
import os
import json
import requests
from enum import Enum

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class MetadataOperation(Enum):
    """Operations for metadata management"""
    REGISTER = "register"
    LOOKUP = "lookup"
    GET = "get"
    REMOVE = "remove"
    UPDATE = "update"
    LIST_GPU = "list_gpu"


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
    # KV cache structure information for retrieve operations
    kv_cache_structure: Optional[Dict] = None  # KV cache structure information


class GPUVRAMPoolManager:
    """
    Enhanced GPU VRAM pool manager with cross-GPU transfer support.
    Manages metadata for KV cache chunks stored in the entire machine's GPU VRAM pool.
    Singleton pattern ensures one instance per machine.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls, config):
        """Get the singleton instance of GPU VRAM pool manager."""
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
        
        logger.info(f"Enhanced GPU VRAM Pool Manager initialized for {len(self.gpu_metadata)} GPUs")

    def _initialize_gpu_tracking(self):
        """Initialize tracking for all available GPUs."""
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            self.gpu_metadata[gpu_id] = set()
        logger.info(f"Tracking {num_gpus} GPUs in VRAM pool")

    def lookup_prefix(
        self,
        token_ids: List[int],
        max_tokens: Optional[int] = None,
        current_gpu_id: Optional[int] = None,
        cache_key: Optional[CacheEngineKey] = None,
        all_chunks: Optional[List[Tuple[int, int, CacheEngineKey]]] = None,
    ) -> Tuple[int, Optional[List[Tuple[Tuple[int, int], int, bool]]]]:
        """
        Lookup prefix match in GPU VRAM pool metadata.
        If all_chunks is provided, find continuous matching chunks from the beginning.
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
            # Handle edge cases
            if not token_ids or (max_tokens is not None and max_tokens <= 0):
                return 0, None
            
            if max_tokens is None:
                max_tokens = len(token_ids)
            
            # 不再直接使用cache_key进行查找，而是通过all_chunks在vram manager中逐一匹配查找
            # 这样可以确保使用全部prefix chunks信息进行最准确的匹配
            
            # If all_chunks is provided, find continuous matching chunks from the beginning
            if all_chunks is not None and len(all_chunks) > 0:
                # Sort chunks by start position
                sorted_chunks = sorted(all_chunks, key=lambda x: x[0])
                
                # Only check continuous chunks from the beginning (start=0)
                continuous_hit_tokens = 0
                expected_start = 0
                chunk_info_list = []
                
                for start, end, chunk_cache_key in sorted_chunks:
                    # Check if this chunk is continuous from the beginning
                    if start != expected_start:
                        # Found a gap, stop checking
                        logger.info(f"Breaking at gap: expected start={expected_start}, got start={start}, stopping GPU VRAM lookup")
                        break
                    
                    chunk_tokens = token_ids[start:end]
                    logger.info(f"Checking GPU VRAM for continuous chunk [{start}, {end}): {len(chunk_tokens)} tokens, key: {chunk_cache_key}, chunk_hash: {chunk_cache_key.chunk_hash if hasattr(chunk_cache_key, 'chunk_hash') else 'N/A'}")
                    
                    # Debug: list all keys in metadata
                    logger.info(f"Debug: Total metadata entries: {len(self.metadata)}")
                    for key in list(self.metadata.keys())[:5]:  # Show first 5 keys
                        logger.info(f"Debug: Metadata key: {key}, chunk_hash: {key.chunk_hash if hasattr(key, 'chunk_hash') else 'N/A'}")
                    
                    # Check if this chunk exists in metadata
                    if chunk_cache_key in self.metadata:
                        entry = self.metadata[chunk_cache_key]
                        
                        # No need to verify token match - cache key uniquely identifies the chunk
                        # System stores data in prefix chunks granularity
                        
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
                            
                            logger.info(f"Found GPU VRAM hit for continuous chunk [{start}, {end}): {chunk_hit_tokens} tokens, GPU={entry.gpu_id}, needs_transfer={chunk_needs_transfer}")
                        else:
                            # Chunk not fully hit, stop checking
                            logger.info(f"Breaking at chunk [{start}, {end}): expected {len(chunk_tokens)} tokens, got {chunk_hit_tokens} tokens, stopping GPU VRAM lookup")
                            break
                    else:
                        # Chunk not found in metadata, stop checking
                        logger.info(f"Breaking at chunk [{start}, {end}): chunk not found in GPU VRAM metadata, stopping lookup")
                        # Debug: check if any key has similar chunk_hash
                        target_hash = chunk_cache_key.chunk_hash if hasattr(chunk_cache_key, 'chunk_hash') else None
                        if target_hash:
                            for key, entry in self.metadata.items():
                                if hasattr(key, 'chunk_hash') and key.chunk_hash == target_hash:
                                    logger.info(f"Debug: Found key with matching chunk_hash: {key}, but cache_key mismatch")
                                    logger.info(f"Debug: Expected cache_key: {chunk_cache_key}")
                                    logger.info(f"Debug: Found cache_key: {key}")
                                    logger.info(f"Debug: Are they equal? {chunk_cache_key == key}")
                        break
                
                if continuous_hit_tokens > 0:
                    logger.info(f"GPU VRAM pool hit from continuous chunks: {continuous_hit_tokens} tokens, {len(chunk_info_list)} chunks matched")
                    for i, ((start, end), gpu_id, needs_transfer) in enumerate(chunk_info_list):
                        logger.info(f"  Chunk {i}: [{start}, {end}) -> GPU {gpu_id}, needs_transfer={needs_transfer}")
                    return continuous_hit_tokens, chunk_info_list
                else:
                    logger.info("No continuous GPU VRAM hits found from the beginning")
                    return 0, None
            
            # No fallback logic - return 0 directly when all_chunks is not provided
            # This avoids inefficient token-by-token searching
            logger.debug("No all_chunks provided, returning 0 (no match)")
            return 0, None

    def _verify_token_match(self, cached_tokens: List[int], query_tokens: List[int]) -> bool:
        """
        Verify if cached tokens match the query tokens.
        Handles cases where cached sequence might be longer than query.
        """
        if len(cached_tokens) < len(query_tokens):
            return False
        
        # Check if the beginning of cached tokens matches the query tokens
        return cached_tokens[:len(query_tokens)] == query_tokens

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

    def enable_monitoring(self, enabled: bool = True):
        """Enable or disable monitoring features."""
        # This is a placeholder for future monitoring features
        logger.info(f"GPU VRAM pool monitoring {'enabled' if enabled else 'disabled'}")

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
        kv_cache_structure: Optional[Dict] = None,
    ) -> bool:
        """
        Register a new KV cache in GPU VRAM pool metadata.
        Called when vLLM generates new KV cache on any GPU.
        This method only manages metadata registration, not memory allocation.

        Args:
            cache_key: CacheEngineKey generated by TestTokenDatabase in store function
            token_ids: List of token IDs (already processed as chunk by TestTokenDatabase)
            gpu_id: GPU ID where the KV cache is stored
            tensor_shape: Shape of the KV cache tensor
            tensor_dtype: Data type of the KV cache tensor
            tensor_size: Size of the KV cache tensor in bytes
            buffer_pointer: Optional GPU buffer pointer address
            segment_id: Optional segment ID if the cache is stored in a segment
            kv_cache_structure: Optional KV cache structure information

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
            
            # 直接使用TestTokenDatabase生成的CacheEngineKey
            # 不需要重新创建key，确保key的一致性
            
            # 创建metadata entry
            entry = GPUVRAMEntry(
                key=cache_key,
                token_ids=token_ids.copy(),  # 存储完整的chunk tokens
                gpu_id=gpu_id,
                tensor_shape=tensor_shape,  # 使用传入的tensor shape
                tensor_dtype=tensor_dtype,
                tensor_size=tensor_size,  # 使用传入的KV cache大小
                created_time=current_time,
                last_access_time=current_time,
                access_count=1,
                buffer_pointer=buffer_pointer,
                resident_hostname=resident_hostname,
                segment_id=segment_id,  # 关联segment ID
                kv_cache_structure=kv_cache_structure  # 存储KV cache结构信息
            )
            
            # 存储metadata
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
                f"Size: {tensor_size} bytes"
            )

            logger.debug(
                f"Registered GPU VRAM pool entry: tokens={len(token_ids)}, "
                f"GPU={gpu_id}, size={tensor_size} bytes, "
                f"buffer_pointer={hex(buffer_pointer) if buffer_pointer else 'None'}, "
                f"segment={segment_id if segment_id else 'external'}"
            )
            return True

    def batch_register_kvcache(
        self,
        entries_data: List[Tuple[CacheEngineKey, List[int], int, tuple, torch.dtype, int, Optional[int], Optional[str], Optional[str], Optional[Dict]]]
    ) -> List[bool]:
        """
        批量注册KV cache到GPU VRAM pool metadata
        
        Args:
            entries_data: 每个entry的数据元组列表，格式为:
                (cache_key, token_ids, gpu_id, tensor_shape, tensor_dtype, tensor_size, 
                 buffer_pointer, segment_id, resident_hostname, kv_cache_structure)
            
        Returns:
            每个entry的注册结果列表，True表示成功，False表示失败
        """
        with self.lock:
            results = []
            current_time = time.time()
            
            for entry_data in entries_data:
                try:
                    # 解包entry数据
                    (cache_key, token_ids, gpu_id, tensor_shape, tensor_dtype, 
                     tensor_size, buffer_pointer, segment_id, resident_hostname, 
                     kv_cache_structure) = entry_data
                    
                    # 检查是否达到metadata限制
                    if self.total_entries >= self.max_metadata_size:
                        if not self._evict_oldest_entry():
                            logger.warning("GPU VRAM pool metadata full, cannot register new entry")
                            results.append(False)
                            continue
                    
                    # 创建metadata entry
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
                        kv_cache_structure=kv_cache_structure
                    )
                    
                    # 存储metadata
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
        批量获取metadata entries
        
        Args:
            keys: cache key列表
            
        Returns:
            对应的metadata entry列表，如果某个key不存在则返回None
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
                logger.info(f"Removed metadata entry for key: {key}, chunk_hash: {key.chunk_hash}, GPU: {entry.gpu_id}")
                return True
            logger.warning(f"Key not found in metadata: {key}, chunk_hash: {key.chunk_hash if hasattr(key, 'chunk_hash') else 'N/A'}")
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

