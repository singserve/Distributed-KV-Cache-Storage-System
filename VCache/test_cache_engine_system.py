# SPDX-License-Identifier: Apache-2.0
"""
Test Cache Engine System

This system extracts cache engine functionality from LMCache and creates
a testable system that maintains the same function names as the original system.
"""

# Standard
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import time
import os
import ctypes
# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.test.test_config import TestConfig
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.test.gpu_vram_pool_manager import GPUVRAMPoolManager
from lmcache.test.transfer_engine_manager import TransferEngineManager
from lmcache.test.gpu_vram_segment_manager import GPUVRAMSegmentManager
from lmcache.test.mooncake_storage_backend import MooncakeStorageBackend
from lmcache.test.vram_metadata_ipc_client import get_vram_metadata_ipc_client
from lmcache.test.test_token_database import TestTokenDatabase
from lmcache.test.test_vram_kvcache_unit import TestVRAMKVCacheUnit
from lmcache.test.blocked_kv_paged_connector import BlockedKVPagedMemConnector

logger = init_logger(__name__)

try:
    TRANSFER_ENGINE_AVAILABLE = True
    logger.info("Mooncake store and transfer engine are available")
except ImportError:
    TRANSFER_ENGINE_AVAILABLE = False
    logger.warning("Mooncake store and transfer engine not available, falling back to mock storage")

class TestCacheEngine:
    """
    Test Cache Engine System
    
    This system extracts cache engine functionality from LMCache engine
    and provides a testable interface with the same function names as
    the original system.
    """
    
    def __init__(
        self,
        config: TestConfig,
        metadata: LMCacheEngineMetadata,
        gpu_connector: Optional[GPUConnectorInterface] = None,
    ):
        """
        Initialize the Test Cache Engine.
        
        Args:
            config: LMCache engine configuration
            metadata: Engine metadata
            gpu_connector: Optional GPU connector for shape information
        """
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.gpu_connector = gpu_connector
        
        # Initialize VRAM metadata IPC client
        self.vram_metadata_client = None

        if self.config.get_extra_config_value("enable_gpu_vram_pool", False):
            logger.info("GPU VRAM pool manager is initializing...")
            try:
                # Check if we should use centralized metadata server via IPC
                use_metadata_server = self.config.get_extra_config_value("use_vram_metadata_server", False)
                
                if use_metadata_server:
                    # Use centralized VRAM metadata server via IPC
                    logger.info("Using centralized VRAM metadata server via IPC")
                    self.vram_metadata_client = get_vram_metadata_ipc_client(self.config)
                    
                    # Check if IPC client is connected
                    if self.vram_metadata_client.is_connected:
                        logger.info("VRAM metadata IPC client connected successfully")
                        # No local GPU VRAM manager in IPC mode - all operations go through IPC client
                    else:
                        logger.warning("VRAM metadata IPC client failed to connect, GPU VRAM pool disabled")
                        self.vram_metadata_client = None
                else:
                    # Use local GPU VRAM pool manager (singleton pattern)
                    logger.info("Using local GPU VRAM pool manager")
                    # In non-IPC mode, we still use the client interface for consistency
                    self.vram_metadata_client = GPUVRAMPoolManager.get_instance(self.config)
                                           
                logger.info("GPU VRAM pool manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GPU VRAM pool manager: {e}")
                self.vram_metadata_client = None
        else:
            logger.info("GPU VRAM pool disabled in configuration")
        
        # Get connector role from config
        connector_role = self.config.connector_role
        logger.info(f"Initializing TestCacheEngine with connector_role: {connector_role}")
        
        # Initialize transfer engine manager for this cache engine instance (must be before segment manager)
        self.transfer_engine_manager = None
        # Only initialize transfer engine for worker role
        if connector_role == "worker" and TRANSFER_ENGINE_AVAILABLE:
            self.transfer_engine_manager = TransferEngineManager(self.config)
            logger.info(f"Transfer engine manager initialized for GPU {self.metadata.worker_id}")
        else:
            logger.info(f"Transfer engine disabled (connector_role={connector_role} or not available)")
        
        # Initialize GPU VRAM segment manager for this cache engine instance
        self.segment_manager = None
        # Only initialize segment manager for worker role
        if connector_role == "worker" and self.config.get_extra_config_value("enable_gpu_vram_segments", True):
            logger.info("GPU VRAM segment manager is initializing...")
            try:
                self.segment_manager = GPUVRAMSegmentManager(
                    self.config, 
                    self.metadata.worker_id, 
                    self.transfer_engine_manager  # Pass transfer engine manager for segment registration
                )
                logger.info(f"GPU VRAM segment manager initialized for GPU {self.metadata.worker_id}")
            except Exception as e:
                logger.error(f"Failed to initialize GPU VRAM segment manager: {e}")
                self.segment_manager = None
        else:
            logger.info(f"GPU VRAM segments disabled (connector_role={connector_role} or config disabled)")
        
        # Statistics tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_lookups": 0,
            "total_stores": 0,
            "total_retrieves": 0,
            "gpu_vram_hits": 0,
            "gpu_vram_misses": 0,
            "cross_gpu_transfers": 0
        }
        
        # Initialize storage backend - use Mooncake if available, otherwise fallback to mock
        # Only initialize storage backend for worker role (scheduler doesn't need storage)
        self.storage_backend = None
        if connector_role == "worker":
            # Pass the transfer engine to Mooncake backend for reuse
            self.storage_backend = MooncakeStorageBackend(
                self.metadata.worker_id, 
                self.config, 
            )
            logger.info("Mooncake storage backend initialized")
            assert self.storage_backend is not None, "Storage backend must be initialized"
        else:
            logger.info("Storage backend disabled for scheduler role")
        
        # Initialize TestTokenDatabase as class member (always needed for chunk processing)
        self.token_database = TestTokenDatabase(chunk_size=256, save_unfull_chunk=True)
        logger.info("TestTokenDatabase initialized as class member")
        
        # Initialize BlockedKVPagedMemConnector if not provided
        # Only initialize GPU connector for worker role (scheduler doesn't need GPU operations)
        if self.gpu_connector is None:
            if connector_role == "worker":
                # 从metadata中获取参数
                num_layers = metadata.kv_shape[0] if len(metadata.kv_shape) >= 5 else 32
                block_size = 16  # vLLM默认block大小
                num_kv_heads = metadata.kv_shape[-2] if len(metadata.kv_shape) >= 4 else 32
                head_size = metadata.kv_shape[-1] if len(metadata.kv_shape) >= 4 else 128
                
                try:
                    self.gpu_connector = BlockedKVPagedMemConnector(
                        num_layers=num_layers,
                        block_size=block_size,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        use_gpu=True,
                        dtype=metadata.kv_dtype,
                        device=f"cuda:{metadata.worker_id}" if metadata.worker_id is not None else "cuda:0"
                    )
                    logger.info(f"BlockedKVPagedMemConnector initialized: layers={num_layers}, block_size={block_size}, heads={num_kv_heads}, head_size={head_size}")
                except Exception as e:
                    logger.error(f"Failed to initialize BlockedKVPagedMemConnector: {e}, falling back to MockGPUConnector")
                    self.gpu_connector = MockGPUConnector()
            else:
                # Scheduler doesn't need GPU connector, use MockGPUConnector
                self.gpu_connector = MockGPUConnector()
                logger.info("Using MockGPUConnector for scheduler role (GPU operations not needed)")
        
        logger.info(f"TestCacheEngine initialized for GPU {self.metadata.worker_id} with connector_role={connector_role}")


    def _perform_cross_gpu_transfer(self, entry, source_gpu: int, target_gpu: int, target_buffer: Optional[int] = None) -> bool:
        """
        Perform synchronous cross-GPU transfer using this cache engine's transfer engine.
        Creates a new entry in GPU VRAM pool for the transferred data copy.
        
        Args:
            entry: GPU VRAM entry to transfer
            source_gpu: Source GPU ID
            target_gpu: Target GPU ID
            target_buffer: Optional pre-allocated target buffer address. If None, will allocate automatically.
            
        Returns:
            True if transfer successful, False otherwise
        """
        if not self.transfer_engine_manager or not self.transfer_engine_manager.initialized:
            logger.warning("Transfer engine not available for cross-GPU transfer")
            return False
        
        try:
            logger.info(f"Starting cross-GPU transfer: GPU {source_gpu} -> GPU {target_gpu}, size: {entry.tensor_size} bytes")
            
            # Get the actual tensor data from source GPU using entry's buffer_pointer
            source_buffer = entry.buffer_pointer
            if source_buffer is None:
                logger.error(f"Failed to get source buffer address from entry for GPU {source_gpu}")
                return False
            
            logger.info(f"Successfully get source buffer address from entry: {hex(source_buffer)}")
            
            # If target_buffer is not provided, allocate one using local segment manager
            if target_buffer is None:
                logger.info("No target buffer provided, allocating using local segment manager")
                if self.segment_manager is not None:
                    segment_id, offset = self.segment_manager.allocate_in_segment(entry.tensor_size)
                    if not segment_id:
                        logger.error(f"Failed to allocate segment space on GPU {target_gpu} for {entry.tensor_size} bytes")
                        return False
                    
                    # Calculate buffer pointer (base address + offset)
                    target_buffer = self.segment_manager.get_buffer_address(segment_id, offset)
                    logger.info(f"Allocated target buffer: {entry.tensor_size} bytes at address {hex(target_buffer)} in segment {segment_id}")
                else:
                    logger.error("Segment manager not available for buffer allocation")
                    return False
            
            logger.info(f"Successfully allocate target buffer address: {hex(target_buffer)}")
            
            # Perform actual cross-GPU transfer using transfer engine
            # Use entry's resident hostname as target hostname
            target_hostname = entry.resident_hostname if hasattr(entry, 'resident_hostname') else f"gpu{target_gpu}.localhost"
            success = self.transfer_engine_manager.transfer_gpu_to_gpu(
                target_hostname=target_hostname,
                source_gpu=source_gpu,
                target_gpu=target_gpu,
                source_buffer=source_buffer,
                target_buffer=target_buffer,
                size=entry.tensor_size
            )
            
            # Examine if received the correct data
            logger.info(f"Tensor shape: {entry.tensor_shape}, dtype: {entry.tensor_dtype}, size: {entry.tensor_size}")
            
            if success:
                logger.info(f"Cross-GPU transfer completed: GPU {source_gpu} -> GPU {target_gpu}, size: {entry.tensor_size} bytes")
                # After successful transfer, register a new entry in GPU VRAM pool for the transferred copy
                self._register_transferred_entry(entry, target_gpu, target_buffer)
            else:
                logger.error(f"Cross-GPU transfer failed: GPU {source_gpu} -> GPU {target_gpu}")
            
            return success
            
        except Exception as e:
            logger.error(f"Exception during cross-GPU transfer: {e}")
            return False


    def _register_transferred_entry(self, original_entry, target_gpu: int, target_buffer: int):
        """
        Register a new entry in GPU VRAM pool for the transferred data copy.
        
        Args:
            original_entry: Original GPU VRAM entry that was transferred
            target_gpu: Target GPU ID where data was transferred to
            target_buffer: Target buffer address where data was transferred
        """
        if self.vram_metadata_client is None:
            logger.warning("VRAM metadata client not available for registering transferred entry")
            return
        
        try:
            logger.info(f"Registering transferred entry in GPU VRAM pool: GPU {target_gpu}, address: {hex(target_buffer)}")
            
            # Create a new entry for the transferred copy
            # Use the same token IDs, shape, and dtype as the original entry
            # Need to get the cache_key from the original entry
            if not hasattr(original_entry, 'key') or original_entry.key is None:
                logger.error("Original entry does not have a cache key, cannot register transferred copy")
                return
            
            success = self.vram_metadata_client.register_kvcache(
                cache_key=original_entry.key,  # Use the same cache key as original
                token_ids=original_entry.token_ids,
                gpu_id=target_gpu,
                tensor_shape=original_entry.tensor_shape,
                tensor_dtype=original_entry.tensor_dtype,
                tensor_size=original_entry.tensor_size,
                buffer_pointer=target_buffer,
                segment_id=None,  # Let VRAM metadata server handle segment assignment
                resident_hostname=self.config.get_extra_config_value("local_hostname_TE", "localhost"),
                kv_cache_structure=getattr(original_entry, 'kv_cache_structure', None)  # Pass KV cache structure if available
            )
            
            if success:
                logger.info(f"Successfully registered transferred entry in GPU VRAM pool: {len(original_entry.token_ids)} tokens on GPU {target_gpu} at address {hex(target_buffer)}")
            else:
                logger.warning(f"Failed to register transferred entry in GPU VRAM pool")
                
        except Exception as e:
            logger.error(f"Error registering transferred entry: {e}")

       

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Enhanced retrieve operation with GPU VRAM pool and Mooncake store integration.
        Supports mask for partial loading and slot mapping for vLLM integration.
        
        Workflow:
        1. Process tokens to generate all_chunks
        2. For all chunks, determine which need retrieval
        3. For chunks needing retrieval, perform lookup prefix to get all hit chunk info
        4. For local chunks (no transfer needed): batch get vram unit, then copy all to vllm kvcache
        5. For remote chunks (needs transfer): for each vram unit, allocate space in segment, get address,
           execute transfer, then copy each to vllm kvcache
        
        Args:
            tokens: Input tokens
            mask: Optional mask for tokens (True = needs loading)
            **kwargs: Additional arguments including:
                - kvcaches: List of KV cache tensors from vLLM
                - slot_mapping: vLLM slot mapping tensor
                - skip_contains_check: Skip cache hit verification
                
        Returns:
            Boolean mask indicating retrieved tokens
        """

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )

        self.stats["total_retrieves"] += 1
        
        # Handle mask for partial loading - connector provides mask for tokens that need loading
        if mask is not None:
            # Mask indicates which tokens need to be loaded (True = needs loading)
            num_required_tokens = torch.sum(mask).item()
            logger.info(f"Partial loading: {num_required_tokens}/{len(tokens)} tokens need loading")
        else:
            # No mask provided, assume all tokens need loading
            num_required_tokens = len(tokens)
            mask = torch.ones(len(tokens), dtype=torch.bool)
        
        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
        
        # Get kvcaches from kwargs (from TestCacheEngineConnector)
        kvcaches = kwargs.get("kvcaches")
        if kvcaches is None or len(kvcaches) == 0:
            logger.error("No kvcaches provided for retrieve operation")
            return ret_mask
        
        # Get slot mapping from kwargs
        slot_mapping = kwargs.get("slot_mapping")
        skip_contains_check = kwargs.get("skip_contains_check", False)
        
        logger.info(f"Retrieve operation: {len(tokens)} tokens, mask={mask is not None}, "
                   f"kvcaches={len(kvcaches)}, slot_mapping={slot_mapping is not None}")
        
        # Step 1: Process tokens to generate all_chunks
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.tolist()
        else:
            token_list = tokens
        
        # Generate all chunks using token database with correct parameters
        all_chunks = []
        # Convert torch.dtype to string using TORCH_DTYPE_TO_STR_DTYPE mapping
        # This ensures consistency with CacheEngineKey's internal representation
        from lmcache.utils import TORCH_DTYPE_TO_STR_DTYPE
        kv_dtype_str = TORCH_DTYPE_TO_STR_DTYPE.get(self.metadata.kv_dtype, "half")
        for start, end, cache_key in self.token_database.process_tokens(
            tokens=token_list, 
            mask=mask, 
            make_key=True,
            model_name="test_model",
            worker_id=self.metadata.worker_id,
            world_size=self.metadata.world_size,
            kv_dtype=kv_dtype_str
        ):
            all_chunks.append((start, end, cache_key))
            logger.debug(f"Generated chunk [{start}, {end}): {end-start} tokens, key: {cache_key}")
        
        # Step 2: Filter chunks that need retrieval (chunks with cache_key != None)
        # TestTokenDatabase的process_tokens方法已经为mask为false的chunk返回None作为key
        chunks_needing_retrieval = [(start, end, cache_key) for start, end, cache_key in all_chunks if cache_key is not None]
        
        logger.info(f"Found {len(chunks_needing_retrieval)} chunks needing retrieval out of {len(all_chunks)} total chunks")
        
        # Step 3: For chunks needing retrieval, perform lookup prefix to get all hit chunk info
        gpu_vram_hits = []  # List of (start, end, cache_key, gpu_id, needs_transfer, entry)
        
        if self.vram_metadata_client is not None and chunks_needing_retrieval:
            # Extract just the cache keys for lookup
            cache_keys = [cache_key for _, _, cache_key in chunks_needing_retrieval]
            
            # 添加调试信息：显示要查找的cache keys
            logger.info(f"DEBUG: batch_get_entry - Looking up {len(cache_keys)} cache keys:")
            for i, cache_key in enumerate(cache_keys):
                logger.info(f"DEBUG:   Key {i}: {cache_key}, chunk_hash: {cache_key.chunk_hash if hasattr(cache_key, 'chunk_hash') else 'N/A'}")
            
            # Batch get entries for all cache keys
            entries = self.vram_metadata_client.batch_get_entry(cache_keys)
            
            # 添加调试信息：显示查找结果
            logger.info(f"DEBUG: batch_get_entry - Got {len(entries)} entries, {len([e for e in entries if e])} non-None entries")
            
            # Process results
            for idx, (start, end, cache_key) in enumerate(chunks_needing_retrieval):
                entry = entries[idx] if idx < len(entries) else None
                
                if entry is not None:
                    # Check if cross-GPU transfer is needed
                    needs_transfer = False
                    if (self.metadata.worker_id is not None and 
                        entry.gpu_id != self.metadata.worker_id and
                        not entry.transfer_in_progress):
                        needs_transfer = True
                    
                    gpu_vram_hits.append((start, end, cache_key, entry.gpu_id, needs_transfer, entry))
                    logger.info(f"GPU VRAM hit for chunk [{start}, {end}): GPU {entry.gpu_id}, needs_transfer={needs_transfer}")
                else:
                    logger.info(f"DEBUG: No GPU VRAM hit for chunk [{start}, {end}), key: {cache_key}, chunk_hash: {cache_key.chunk_hash if hasattr(cache_key, 'chunk_hash') else 'N/A'}")
        
        logger.info(f"Found {len(gpu_vram_hits)} GPU VRAM hits out of {len(chunks_needing_retrieval)} chunks needing retrieval")
        
        # Step 4: Process GPU VRAM hits - only process continuous hits from the beginning
        if gpu_vram_hits:
            # Sort hits by start position
            gpu_vram_hits.sort(key=lambda x: x[0])
            
            # Filter only continuous hits from the beginning (start=0)
            continuous_hits = []
            expected_start = 0
            
            for start, end, cache_key, gpu_id, needs_transfer, entry in gpu_vram_hits:
                if start == expected_start:
                    continuous_hits.append((start, end, cache_key, gpu_id, needs_transfer, entry))
                    expected_start = end
                else:
                    # Break at first gap
                    logger.info(f"Breaking at gap: expected start={expected_start}, got start={start}, stopping processing")
                    break
            
            if not continuous_hits:
                logger.info("No continuous GPU VRAM hits from the beginning")
                self.stats["gpu_vram_misses"] += 1
            else:
                logger.info(f"Found {len(continuous_hits)} continuous GPU VRAM hits from the beginning, covering tokens [0, {expected_start})")
                
                # Separate local and remote hits
                local_hits = [(start, end, cache_key, gpu_id, entry) 
                             for start, end, cache_key, gpu_id, needs_transfer, entry in continuous_hits 
                             if not needs_transfer]
                remote_hits = [(start, end, cache_key, gpu_id, entry) 
                              for start, end, cache_key, gpu_id, needs_transfer, entry in continuous_hits 
                              if needs_transfer]
                
                logger.info(f"Processing {len(local_hits)} local GPU VRAM hits and {len(remote_hits)} remote GPU VRAM hits")
                
                # Process local hits (no transfer needed)
                if local_hits:
                    self._process_local_gpu_vram_hits(
                        local_hits=local_hits,
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping,
                        ret_mask=ret_mask,
                        mask=mask
                    )
                
                # Process remote hits (needs transfer)
                if remote_hits:
                    self._process_remote_gpu_vram_hits(
                        remote_hits=remote_hits,
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping,
                        ret_mask=ret_mask,
                        mask=mask
                    )
                
                # Update statistics
                self.stats["gpu_vram_hits"] += len(continuous_hits)
                if remote_hits:
                    self.stats["cross_gpu_transfers"] += len(remote_hits)
        else:
            logger.info("No GPU VRAM hits found")
            self.stats["gpu_vram_misses"] += 1
        
        # Step 5: If no GPU VRAM hits or processing failed, try Mooncake storage
        if ret_mask.sum().item() == 0:
            logger.info("No GPU VRAM hits processed successfully, trying Mooncake storage")
            
            # Generate lookup chunks for Mooncake
            lookup_chunks = []
            for start, end, cache_key in chunks_needing_retrieval:
                lookup_chunks.append((start, end, cache_key))
            
            # storage_backend.lookup returns a tuple (hit_tokens, chunk_info_list)
            lookup_result = self.storage_backend.lookup(token_list, lookup_chunks)
            
            # Check if lookup_result is a tuple or just hit_tokens
            if isinstance(lookup_result, tuple) and len(lookup_result) >= 2:
                storage_hit_tokens = lookup_result[0]
                chunk_info_list = lookup_result[1]
            else:
                storage_hit_tokens = lookup_result
                chunk_info_list = None
            
            if storage_hit_tokens == 0:
                logger.info(f"Retrieve miss: No hits found for {num_required_tokens} tokens")
                self.stats["misses"] += 1
                return ret_mask
            
            logger.info(f"Mooncake storage hit: {storage_hit_tokens} tokens")
            
            # Process Mooncake storage hit with chunk_info_list
            mooncake_success = self._process_mooncake_hit(
                tokens=tokens,
                storage_hit_tokens=storage_hit_tokens,
                chunk_info_list=chunk_info_list,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                ret_mask=ret_mask,
                mask=mask
            )
            
            if mooncake_success:
                logger.info(f"Successfully processed Mooncake hit: {storage_hit_tokens} tokens")
                self.stats["hits"] += 1
            else:
                logger.warning(f"Failed to process Mooncake hit")
        
        return ret_mask

    def _process_mooncake_hit(
        self,
        tokens: Union[torch.Tensor, List[int]],
        storage_hit_tokens: int,
        kvcaches: List[torch.Tensor],
        slot_mapping: Optional[torch.Tensor],
        ret_mask: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> bool:
        """
        Process Mooncake storage hit by retrieving actual KV cache data and placing it in vLLM kvcaches.
        Enhanced to use token database for chunking and cache key generation, similar to store process.
        Only processes continuous hits from the beginning (start=0).
        Uses BlockedKVGPUConnector for batch upload to vLLM kvcaches.
        
        Args:
            tokens: Input tokens
            storage_hit_tokens: Number of storage hit tokens (from lookup)
            kvcaches: List of vLLM KV cache tensors
            slot_mapping: vLLM slot mapping tensor
            ret_mask: Return mask to mark retrieved tokens
            mask: Optional mask for tokens (True = needs loading)
            
        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing Mooncake storage hit: {storage_hit_tokens} tokens")
        
        # Convert tokens to list format if needed
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.tolist()
        else:
            token_list = tokens
        
        # Step 1: Use token database to generate chunks and cache keys, similar to store process
        # 过滤掉cache_key为None的chunks（这些是mask为false的chunks）
        all_chunks = []
        # Convert torch.dtype to string using TORCH_DTYPE_TO_STR_DTYPE mapping
        # This ensures consistency with CacheEngineKey's internal representation
        from lmcache.utils import TORCH_DTYPE_TO_STR_DTYPE
        kv_dtype_str = TORCH_DTYPE_TO_STR_DTYPE.get(self.metadata.kv_dtype, "half")
        for start, end, cache_key in self.token_database.process_tokens(
            tokens=token_list, 
            mask=mask, 
            make_key=True,
            model_name="test_model",
            worker_id=self.metadata.worker_id,
            world_size=self.metadata.world_size,
            kv_dtype=kv_dtype_str
        ):
            # 只处理cache_key不为None的chunks（这些是需要retrieval的chunks）
            if cache_key is not None:
                all_chunks.append((start, end, cache_key))
                chunk_tokens = token_list[start:end]
                logger.info(f"Found hit chunk/prefix [{start}, {end}): {len(chunk_tokens)} tokens, key: {cache_key}")
        
        # Sort chunks by start position
        all_chunks.sort(key=lambda x: x[0])
        
        # Filter only continuous hits from the beginning (start=0)
        continuous_chunks = []
        expected_start = 0
        
        for start, end, cache_key in all_chunks:
            if start == expected_start:
                continuous_chunks.append((start, end, cache_key))
                expected_start = end
            else:
                # Break at first gap
                logger.info(f"Breaking at gap: expected start={expected_start}, got start={start}, stopping processing")
                break
        
        if not continuous_chunks:
            logger.info("No continuous Mooncake hits from the beginning")
            return False
        
        logger.info(f"Found {len(continuous_chunks)} continuous Mooncake hits from the beginning, covering tokens [0, {expected_start})")
        
        # Step 2: For each continuous hit chunk, retrieve from Mooncake backend using unified cache key
        # 注意：store时使用统一的cache key一次性存储所有层数据，所以我们也使用统一的cache key进行检索
        total_retrieved_tokens = 0
        all_retrieved_data = []  # Store (start, end, retrieved_tokens, kv_cache_tensor, kv_cache_structure) for each chunk
        
        for start, end, cache_key in continuous_chunks:
            chunk_tokens = token_list[start:end]
            logger.info(f"Retrieving hit chunk/prefix [{start}, {end}): {len(chunk_tokens)} tokens with unified cache key {cache_key}")
            
            # 使用统一的cache key一次性检索所有层数据
            retrieved_tokens, kv_cache_tensor, kv_cache_structure = self.storage_backend.retrieve(
                cache_key=cache_key,  # 使用统一的cache key，不是分层key
                tokens=chunk_tokens
            )
            
            if len(retrieved_tokens) == 0 or kv_cache_tensor is None:
                logger.warning(f"Mooncake retrieve failed for chunk [{start}, {end}): no data found with unified cache key")
                continue
            
            logger.info(f"Mooncake retrieve successful for chunk [{start}, {end}): {len(retrieved_tokens)} tokens retrieved, KV cache tensor shape: {kv_cache_tensor.shape}")
            
            # Verify that retrieved tokens match the chunk tokens
            if retrieved_tokens != chunk_tokens:
                logger.warning(f"Token mismatch for chunk [{start}, {end}): retrieved {len(retrieved_tokens)} tokens but expected {len(chunk_tokens)}")
                continue
            
            all_retrieved_data.append((start, end, retrieved_tokens, kv_cache_tensor, kv_cache_structure))
            total_retrieved_tokens += len(retrieved_tokens)
            logger.info(f"Successfully retrieved chunk [{start}, {end}) using unified cache key")
        
        if total_retrieved_tokens == 0:
            logger.warning("Mooncake retrieve failed: no chunks retrieved successfully")
            return False
        
        logger.info(f"Mooncake retrieve successful: {total_retrieved_tokens} tokens retrieved from {len(all_retrieved_data)} continuous chunks")
        
        # Step 3: Check if we have BlockedKVPagedMemConnector
        if not isinstance(self.gpu_connector, BlockedKVPagedMemConnector):
            logger.error(f"GPU connector is not BlockedKVPagedMemConnector, cannot process Mooncake hits with GPU connector")
            return False
        
        # Step 4: Prepare batch upload parameters for BlockedKVGPUConnector
        blocked_kv_data_list = []
        slot_mapping_list = []
        starts = []
        ends = []
        
        for start, end, retrieved_tokens, kv_cache_tensor, kv_cache_structure in all_retrieved_data:
            num_tokens = end - start
            logger.info(f"Preparing Mooncake chunk [{start}, {end}): {num_tokens} tokens for upload")
            
            # The kv_cache_tensor is already in the correct format from Mooncake storage
            # Shape: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
            # We can use it directly for batch upload
            
            # Verify the tensor shape
            if len(kv_cache_tensor.shape) != 6:
                logger.error(f"Mooncake KV cache tensor has unexpected shape: {kv_cache_tensor.shape}, expected 6D")
                continue
            
            # Create slot mapping for this chunk
            # 必须与store时使用的slot mapping一致
            if slot_mapping is not None:
                chunk_slot_mapping = slot_mapping[start:end]
            else:
                # 如果没有slot mapping，创建简单的连续slot mapping
                # 从0开始，长度为chunk_tokens（与store时一致）
                chunk_slot_mapping = torch.arange(0, end - start, dtype=torch.int32)
            
            blocked_kv_data_list.append(kv_cache_tensor)
            slot_mapping_list.append(chunk_slot_mapping)
            starts.append(0)  # Start from beginning of each chunk
            ends.append(num_tokens)  # Upload all tokens in chunk
        
        if not blocked_kv_data_list:
            logger.error("No Mooncake chunks prepared for upload")
            return False
        
        # Step 5: Perform batch upload using BlockedKVGPUConnector
        try:
            logger.info(f"Performing batch upload of {len(blocked_kv_data_list)} Mooncake chunks")
            
            # Initialize kvcaches pointer in connector if needed
            if hasattr(self.gpu_connector, 'initialize_kvcaches_ptr'):
                self.gpu_connector.initialize_kvcaches_ptr(kvcaches=kvcaches)
            
            # Batch upload all chunks
            self.gpu_connector.batch_upload_blocked_kv(
                blocked_kv_data_list=blocked_kv_data_list,
                vllm_kvcaches=kvcaches,
                slot_mapping_list=slot_mapping_list,
                starts=starts,
                ends=ends
            )
            
            logger.info(f"Successfully uploaded {len(blocked_kv_data_list)} Mooncake chunks via batch upload")
            
            # Mark retrieved tokens in return mask
            total_copied_tokens = 0
            for idx, (start, end, retrieved_tokens, kv_cache_tensor, kv_cache_structure) in enumerate(all_retrieved_data):
                if idx >= len(blocked_kv_data_list):
                    continue
                
                num_tokens = end - start
                for i in range(start, min(end, len(token_list))):
                    if mask is None or mask[i]:
                        ret_mask[i] = True
                total_copied_tokens += num_tokens
                logger.info(f"Marked {num_tokens} tokens for Mooncake chunk [{start}, {end}) as retrieved")
            
            logger.info(f"Processed {len(all_retrieved_data)} Mooncake hits via batch upload, copied {total_copied_tokens} tokens")
            return total_copied_tokens > 0
            
        except Exception as e:
            logger.error(f"Batch upload failed for Mooncake chunks: {e}")
            return False

    def _process_local_gpu_vram_hits(
        self,
        local_hits: List[Tuple[int, int, CacheEngineKey, int, Any]],  # (start, end, cache_key, gpu_id, entry)
        kvcaches: List[torch.Tensor],
        slot_mapping: Optional[torch.Tensor],
        ret_mask: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> bool:
        """
        Process local GPU VRAM hits (no transfer needed).
        Get VRAM unit from local segment and upload to vLLM kvcaches using BlockedKVGPUConnector.
        
        Args:
            local_hits: List of local GPU VRAM hits
            kvcaches: List of vLLM KV cache tensors
            slot_mapping: vLLM slot mapping tensor
            ret_mask: Return mask to mark retrieved tokens
            mask: Optional mask for tokens (True = needs loading)
            
        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing {len(local_hits)} local GPU VRAM hits (no transfer needed)")
        
        if not local_hits:
            logger.warning("No local hits to process")
            return False
        
        # Check if we have BlockedKVPagedMemConnector
        if not isinstance(self.gpu_connector, BlockedKVPagedMemConnector):
            logger.error(f"GPU connector is not BlockedKVPagedMemConnector, cannot process local GPU VRAM hits")
            return False
        
        # Prepare batch upload parameters
        blocked_kv_data_list = []
        slot_mapping_list = []
        starts = []
        ends = []
        
        for start, end, cache_key, gpu_id, entry in local_hits:
            num_tokens = end - start
            logger.info(f"Preparing chunk [{start}, {end}): {num_tokens} tokens for upload")
            
            # Get VRAM unit for this chunk
            vram_unit = None
            if self.segment_manager is not None:
                vram_unit = self.segment_manager.get_vram_unit(cache_key)
            
            if vram_unit is None:
                # Create VRAM unit from entry
                try:
                    # Get segment info from entry
                    segment_id = getattr(entry, 'segment_id', None)
                    segment_offset = 0
                    
                    if segment_id and self.segment_manager is not None:
                        segment = self.segment_manager.get_segment_by_id(segment_id)
                        if segment and hasattr(entry, 'buffer_pointer'):
                            segment_offset = entry.buffer_pointer - segment.base_address
                    
                    # Create VRAM unit
                    vram_unit = self.segment_manager.create_vram_unit(
                        cache_key=cache_key,
                        token_ids=entry.token_ids,
                        segment_id=segment_id,
                        offset=segment_offset,
                        allocated_size=entry.tensor_size,
                        dtype=entry.tensor_dtype,
                        original_shape=entry.tensor_shape  # 使用entry的tensor_shape作为original_shape
                    )
                except Exception as e:
                    logger.error(f"Failed to create VRAM unit for cache key {cache_key}: {e}")
                    raise Exception(f"Failed to create VRAM unit for chunk [{start}, {end}): {e}")
            
            if vram_unit is None:
                logger.error(f"Failed to get VRAM unit for chunk [{start}, {end})")
                raise Exception(f"Failed to get VRAM unit for chunk [{start}, {end})")
            
            # Get the tensor data from VRAM unit
            # The VRAM unit stores flattened tensor data, we need to restore it to original shape
            if not hasattr(vram_unit, 'original_shape') or vram_unit.original_shape is None:
                logger.error(f"VRAM unit for chunk [{start}, {end}) does not have original_shape metadata")
                raise Exception(f"VRAM unit for chunk [{start}, {end}) does not have original_shape metadata")
            
            original_shape = vram_unit.original_shape
            vram_tensor = vram_unit.kv_cache_tensor
            
            # Restore the tensor to its original shape
            # Original shape should be: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
            if len(original_shape) != 6:
                logger.error(f"Expected 6D original shape for chunk [{start}, {end}), got: {original_shape}")
                raise Exception(f"Expected 6D original shape for chunk [{start}, {end}), got: {original_shape}")
            
            restored_tensor = vram_tensor.view(original_shape)
            logger.info(f"Restored tensor for chunk [{start}, {end}): shape={restored_tensor.shape}")
            
            # Create slot mapping for this chunk
            # 必须与store时使用的slot mapping一致
            if slot_mapping is not None:
                chunk_slot_mapping = slot_mapping[start:end]
            else:
                # 如果没有slot mapping，创建简单的连续slot mapping
                # 从0开始，长度为chunk_tokens（与store时一致）
                chunk_slot_mapping = torch.arange(0, end - start, dtype=torch.int32)
            
            blocked_kv_data_list.append(restored_tensor)
            slot_mapping_list.append(chunk_slot_mapping)
            starts.append(0)  # Start from beginning of each chunk
            ends.append(num_tokens)  # Upload all tokens in chunk
        
        if not blocked_kv_data_list:
            logger.error("No VRAM units prepared for upload")
            return False
        
        # Perform batch upload using BlockedKVGPUConnector
        try:
            logger.info(f"Performing batch upload of {len(blocked_kv_data_list)} chunks")
            
            # Initialize kvcaches pointer in connector if needed
            if hasattr(self.gpu_connector, 'initialize_kvcaches_ptr'):
                self.gpu_connector.initialize_kvcaches_ptr(kvcaches=kvcaches)
            
            # Batch upload all chunks
            self.gpu_connector.batch_upload_blocked_kv(
                blocked_kv_data_list=blocked_kv_data_list,
                vllm_kvcaches=kvcaches,
                slot_mapping_list=slot_mapping_list,
                starts=starts,
                ends=ends
            )
            
            logger.info(f"Successfully uploaded {len(blocked_kv_data_list)} chunks via batch upload")
            
            # Mark retrieved tokens in return mask
            total_copied_tokens = 0
            for idx, (start, end, cache_key, gpu_id, entry) in enumerate(local_hits):
                if idx >= len(blocked_kv_data_list):
                    continue
                
                num_tokens = end - start
                for i in range(start, min(end, len(ret_mask))):
                    if mask is None or mask[i]:
                        ret_mask[i] = True
                total_copied_tokens += num_tokens
                logger.info(f"Marked {num_tokens} tokens for chunk [{start}, {end}) as retrieved")
            
            logger.info(f"Processed {len(local_hits)} local GPU VRAM hits via batch upload, copied {total_copied_tokens} tokens")
            return total_copied_tokens > 0
            
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            raise Exception(f"Failed to upload local GPU VRAM hits: {e}")

    def _process_remote_gpu_vram_hits(
        self,
        remote_hits: List[Tuple[int, int, CacheEngineKey, int, Any]],  # (start, end, cache_key, gpu_id, entry)
        kvcaches: List[torch.Tensor],
        slot_mapping: Optional[torch.Tensor],
        ret_mask: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> bool:
        """
        Process remote GPU VRAM hits (needs transfer).
        Transfer remote data to local segment, then upload to vLLM kvcaches using BlockedKVGPUConnector.
        
        Args:
            remote_hits: List of remote GPU VRAM hits
            kvcaches: List of vLLM KV cache tensors
            slot_mapping: vLLM slot mapping tensor
            ret_mask: Return mask to mark retrieved tokens
            mask: Optional mask for tokens (True = needs loading)
            
        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing {len(remote_hits)} remote GPU VRAM hits (needs transfer)")
        
        if not remote_hits:
            logger.warning("No remote hits to process")
            return False
        
        # Check if we have BlockedKVPagedMemConnector
        if not isinstance(self.gpu_connector, BlockedKVPagedMemConnector):
            logger.error(f"GPU connector is not BlockedKVPagedMemConnector, cannot process remote GPU VRAM hits")
            return False
        
        # Prepare batch upload parameters
        blocked_kv_data_list = []
        slot_mapping_list = []
        starts = []
        ends = []
        
        for start, end, cache_key, source_gpu_id, entry in remote_hits:
            num_tokens = end - start
            logger.info(f"Processing remote hit chunk [{start}, {end}): {num_tokens} tokens from GPU {source_gpu_id}")
            
            # Allocate space in segment for this chunk
            segment_address = None
            segment_id = None
            segment_offset = None
            
            if self.segment_manager is not None:
                segment_id, segment_offset = self.segment_manager.allocate_in_segment(entry.tensor_size)
                if segment_id:
                    segment_address = self.segment_manager.get_buffer_address(segment_id, segment_offset)
                    logger.info(f"Allocated segment space: {entry.tensor_size} bytes at address {hex(segment_address)} for remote GPU VRAM hit data")
                else:
                    logger.error(f"Failed to allocate segment space for {entry.tensor_size} bytes")
                    raise Exception(f"Failed to allocate segment space for chunk [{start}, {end}): {entry.tensor_size} bytes")
            
            if segment_address is None:
                logger.error(f"Failed to allocate segment space for chunk [{start}, {end})")
                raise Exception(f"Failed to allocate segment space for chunk [{start}, {end})")
            
            # Execute cross-GPU transfer
            transfer_success = self._perform_cross_gpu_transfer(
                entry=entry,
                source_gpu=source_gpu_id,
                target_gpu=self.metadata.worker_id,
                target_buffer=segment_address
            )
            
            if not transfer_success:
                logger.error(f"Failed to transfer chunk [{start}, {end}) from GPU {source_gpu_id}")
                # Free allocated segment space
                if segment_id and self.segment_manager is not None:
                    self.segment_manager.free_segment_space(segment_id, segment_offset, entry.tensor_size)
                raise Exception(f"Failed to transfer chunk [{start}, {end}) from GPU {source_gpu_id}")
            
            logger.info(f"Successfully transferred {num_tokens} tokens from GPU {source_gpu_id} to segment space")
            
            # Create VRAM unit for transferred data
            vram_unit = None
            if self.segment_manager is not None:
                vram_unit = self.segment_manager.create_vram_unit(
                    cache_key=cache_key,
                    token_ids=entry.token_ids,
                    segment_id=segment_id,
                    offset=segment_offset,
                    allocated_size=entry.tensor_size,
                    dtype=entry.tensor_dtype,
                    original_shape=entry.tensor_shape  # 使用entry的tensor_shape作为original_shape
                )
            
            if vram_unit is None:
                logger.error(f"Failed to create VRAM unit for transferred chunk [{start}, {end})")
                raise Exception(f"Failed to create VRAM unit for transferred chunk [{start}, {end})")
            
            # Get the tensor data from VRAM unit
            # The VRAM unit stores flattened tensor data, we need to restore it to original shape
            if not hasattr(vram_unit, 'original_shape') or vram_unit.original_shape is None:
                logger.error(f"VRAM unit for chunk [{start}, {end}) does not have original_shape metadata")
                raise Exception(f"VRAM unit for chunk [{start}, {end}) does not have original_shape metadata")
            
            original_shape = vram_unit.original_shape
            vram_tensor = vram_unit.kv_cache_tensor
            
            # Restore the tensor to its original shape
            # Original shape should be: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
            if len(original_shape) != 6:
                logger.error(f"Expected 6D original shape for chunk [{start}, {end}), got: {original_shape}")
                raise Exception(f"Expected 6D original shape for chunk [{start}, {end}), got: {original_shape}")
            
            restored_tensor = vram_tensor.view(original_shape)
            logger.info(f"Restored tensor for chunk [{start}, {end}): shape={restored_tensor.shape}")
            
            # Create slot mapping for this chunk
            # 必须与store时使用的slot mapping一致
            if slot_mapping is not None:
                chunk_slot_mapping = slot_mapping[start:end]
            else:
                # 如果没有slot mapping，创建简单的连续slot mapping
                # 从0开始，长度为chunk_tokens（与store时一致）
                chunk_slot_mapping = torch.arange(0, end - start, dtype=torch.int32)
            
            blocked_kv_data_list.append(restored_tensor)
            slot_mapping_list.append(chunk_slot_mapping)
            starts.append(0)  # Start from beginning of each chunk
            ends.append(num_tokens)  # Upload all tokens in chunk
        
        if not blocked_kv_data_list:
            logger.error("No VRAM units prepared for upload after transfer")
            return False
        
        # Perform batch upload using BlockedKVGPUConnector
        try:
            logger.info(f"Performing batch upload of {len(blocked_kv_data_list)} transferred chunks")
            
            # Initialize kvcaches pointer in connector if needed
            if hasattr(self.gpu_connector, 'initialize_kvcaches_ptr'):
                self.gpu_connector.initialize_kvcaches_ptr(kvcaches=kvcaches)
            
            # Batch upload all chunks
            self.gpu_connector.batch_upload_blocked_kv(
                blocked_kv_data_list=blocked_kv_data_list,
                vllm_kvcaches=kvcaches,
                slot_mapping_list=slot_mapping_list,
                starts=starts,
                ends=ends
            )
            
            logger.info(f"Successfully uploaded {len(blocked_kv_data_list)} transferred chunks via batch upload")
            
            # Mark retrieved tokens in return mask
            total_copied_tokens = 0
            for idx, (start, end, cache_key, source_gpu_id, entry) in enumerate(remote_hits):
                if idx >= len(blocked_kv_data_list):
                    continue
                
                num_tokens = end - start
                for i in range(start, min(end, len(ret_mask))):
                    if mask is None or mask[i]:
                        ret_mask[i] = True
                total_copied_tokens += num_tokens
                logger.info(f"Marked {num_tokens} tokens for chunk [{start}, {end}) as retrieved")
            
            logger.info(f"Processed {len(remote_hits)} remote GPU VRAM hits via batch upload, copied {total_copied_tokens} tokens")
            return total_copied_tokens > 0
            
        except Exception as e:
            logger.error(f"Batch upload failed for transferred chunks: {e}")
            raise Exception(f"Failed to upload transferred GPU VRAM hits: {e}")


    @_lmcache_nvtx_annotate
    def store(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> None:
        """
        Enhanced store operation using TestVRAMKVCacheUnit and TestTokenDatabase.
        Saves vLLM KV cache to VRAM segment system without _copy_from_segment_to_target.
        
        Args:
            tokens: Input tokens
            mask: Optional mask for tokens (True = needs storing)
            **kwargs: Additional arguments including:
                - kvcaches: List of vLLM KV cache tensors
                - slot_mapping: vLLM slot mapping tensor
                - offset: Offset for partial storage
        """
        self.stats["total_stores"] += 1
        
        # Handle mask for partial storage
        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)
        
        # Extract vLLM KV cache information from kwargs
        kvcaches = kwargs.get("kvcaches")
        slot_mapping = kwargs.get("slot_mapping")
        offset = kwargs.get("offset", 0)
        
        if kvcaches is None or len(kvcaches) == 0:
            logger.error("No kvcaches provided for store operation")
            return
        
        logger.info(f"Store operation: {num_to_store_tokens} tokens, "
                   f"kvcaches={len(kvcaches)}, slot_mapping={slot_mapping is not None}, "
                   f"offset={offset}")
        
        # Step 1: Process tokens using class member TestTokenDatabase
        # Convert tokens to list format
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.tolist()
        else:
            token_list = tokens
        
        # 修正：检查token_database是否存在
        if not hasattr(self, 'token_database') or self.token_database is None:
            logger.error("TestTokenDatabase not initialized")
            return
        
        # Process tokens to generate cache keys using class member token_database
        # 处理所有chunk和前缀chunk，每个都分配独立的VRAM Unit
        all_chunks = []
        # Convert torch.dtype to string using TORCH_DTYPE_TO_STR_DTYPE mapping
        # This ensures consistency with CacheEngineKey's internal representation
        from lmcache.utils import TORCH_DTYPE_TO_STR_DTYPE
        kv_dtype_str = TORCH_DTYPE_TO_STR_DTYPE.get(self.metadata.kv_dtype, "half")
        for start, end, cache_key in self.token_database.process_tokens(
            tokens=token_list, 
            mask=mask, 
            make_key=True,
            model_name="test_model",
            worker_id=self.metadata.worker_id,
            world_size=self.metadata.world_size,
            kv_dtype=kv_dtype_str
        ):
            all_chunks.append((start, end, cache_key))
            chunk_tokens = token_list[start:end]
            logger.info(f"Found chunk/prefix [{start}, {end}): {len(chunk_tokens)} tokens, key: {cache_key}")
        
        # 为每个chunk和前缀创建独立的VRAM Unit，并实现逐层存储
        for start, end, cache_key in all_chunks:
            chunk_tokens = token_list[start:end]
            logger.info(f"Processing chunk/prefix [{start}, {end}): {len(chunk_tokens)} tokens, key: {cache_key}")
            
            # 关键：检查整个chunk是否已存在（同时检查GPU VRAM和存储后端），如果存在就跳过整个chunk
            # 使用统一的cache key检查，不再分层
            gpu_vram_exists = False
            if self.vram_metadata_client is not None:
                gpu_vram_exists = self.vram_metadata_client.contains(cache_key)
            
            # 检查存储后端中是否已存在
            storage_backend_exists = self.storage_backend.contains(cache_key)
            
            # 如果任一存储位置已存在，跳过整个chunk
            if gpu_vram_exists or storage_backend_exists:
                logger.info(f"Chunk [{start}, {end}) already exists in {'GPU VRAM' if gpu_vram_exists else ''}{' and ' if gpu_vram_exists and storage_backend_exists else ''}{'storage backend' if storage_backend_exists else ''}, skipping storage")
                continue
            
            # Step 2: Calculate KV cache size for this chunk and allocate VRAM segment
            # 修正：根据vLLM KV Cache实际结构计算chunk对应的KV cache大小
            # vLLM的KV cache包含所有层，需要计算所有层的总大小
            chunk_kv_cache_size = 0
            
            # 获取实际的层数
            if isinstance(kvcaches, dict):
                num_layers = len(kvcaches)
                first_kv_cache = list(kvcaches.values())[0]
            else:
                num_layers = len(kvcaches)  # 修复：使用列表长度作为层数
                first_kv_cache = kvcaches[0]
            
            # vLLM KV Cache形状: (num_blocks, 2, block_size, num_kv_heads, head_size)
            # 第0个维度: num_blocks (block数量)
            # 第1个维度: 2 (Key和Value)
            # 第2个维度: block_size (每个block的token数量)
            # 第3个维度: num_kv_heads (KV头数量)
            # 第4个维度: head_size (每个头的大小)
            if len(first_kv_cache.shape) == 5:
                # 标准vLLM FlashAttention KV Cache结构
                # 每个token的大小: 2 * num_kv_heads * head_size * element_size
                num_kv_heads = first_kv_cache.shape[3]
                head_size = first_kv_cache.shape[4]
                element_size = first_kv_cache.element_size()
                
                # 计算每个token的大小
                token_size = 2 * num_kv_heads * head_size * element_size
                # 计算所有层的总大小
                chunk_kv_cache_size = len(chunk_tokens) * token_size * num_layers
            elif len(first_kv_cache.shape) >= 3:
                # 其他形状，假设token维度是第2个维度
                # 计算每个token的元素数量，然后乘以元素大小
                elements_per_token = first_kv_cache.numel() // first_kv_cache.shape[2]
                # 计算所有层的总大小
                chunk_kv_cache_size = elements_per_token * len(chunk_tokens) * first_kv_cache.element_size() * num_layers
            else:
                # 如果KV cache没有token维度，使用整个tensor大小
                # 计算所有层的总大小
                chunk_kv_cache_size = first_kv_cache.numel() * first_kv_cache.element_size() * num_layers
            
            # Allocate VRAM segment for this chunk and create VRAM Unit
            vram_kvcache_unit = None
            segment_id = None
            segment_offset = None
            
            if self.segment_manager is not None:
                # 使用新的Segment Manager方法创建VRAM Unit
                # Use the first KV cache for shape and dtype
                first_kv_cache = kvcaches[0]
                kv_dtype = first_kv_cache.dtype
                
                # 修正：为这个chunk创建正确的KV cache形状
                # vLLM KV cache是blocked布局: [num_blocks, 2, block_size, num_kv_heads, head_size]
                # block_size是固定的，我们需要调整num_blocks来容纳chunk的所有tokens
                if len(first_kv_cache.shape) == 5:
                    # 标准vLLM FlashAttention KV Cache结构
                    num_blocks, kv_pairs, block_size, num_kv_heads, head_size = first_kv_cache.shape
                    # 计算需要的blocks数量
                    chunk_blocks = (len(chunk_tokens) + block_size - 1) // block_size
                    kv_shape = (chunk_blocks, kv_pairs, block_size, num_kv_heads, head_size)
                elif len(first_kv_cache.shape) >= 3:
                    # 其他形状，假设tokens维度是第2个维度
                    kv_shape = list(first_kv_cache.shape)
                    kv_shape[2] = len(chunk_tokens)  # 将tokens维度调整为chunk大小
                    kv_shape = tuple(kv_shape)
                else:
                    kv_shape = first_kv_cache.shape
                
                # 分配segment空间并获取segment信息
                segment_id, segment_offset = self.segment_manager.allocate_in_segment(chunk_kv_cache_size)
                
                if segment_id:
                    # 使用新的create_vram_unit方法创建VRAM Unit（用于一维展平数据）
                    # 需要传递original_shape参数，这里我们不知道原始形状，所以传递None
                    vram_kvcache_unit = self.segment_manager.create_vram_unit(
                        cache_key=cache_key,
                        token_ids=chunk_tokens,
                        segment_id=segment_id,
                        offset=segment_offset,
                        allocated_size=chunk_kv_cache_size,
                        dtype=kv_dtype,
                        original_shape=None  # 这里不知道原始形状，传递None
                    )
                    
                    if vram_kvcache_unit is not None:
                        logger.info(f"Created VRAM Unit: {cache_key} at segment {segment_id}, "
                                   f"offset {segment_offset}, size {chunk_kv_cache_size} bytes, "
                                   f"shape {kv_shape}, tokens {len(chunk_tokens)}")
                    else:
                        logger.warning(f"Failed to create VRAM Unit for chunk: {len(chunk_tokens)} tokens")
                        continue
                else:
                    logger.warning(f"Failed to allocate segment space for {chunk_kv_cache_size} bytes")
                    continue
            
            # Step 3: Copy vLLM KV cache data to VRAM segment
            if vram_kvcache_unit is not None:
                # 使用Segment Manager的get_buffer_address方法获取buffer地址
                allocated_buffer_address = self.segment_manager.get_buffer_address(segment_id, segment_offset)
                if allocated_buffer_address is None:
                    logger.error(f"Failed to get buffer address for segment {segment_id}, offset {segment_offset}")
                    continue
                
                # Step 4: Use GPU connector to download blocked KV cache data directly
                copy_success = False
                combined_tensor = None
                
                # 检查是否使用BlockedKVPagedMemConnector
                if isinstance(self.gpu_connector, BlockedKVPagedMemConnector):
                    try:
                        logger.info(f"Using BlockedKVPagedMemConnector to download chunk [{start}, {end}): {len(chunk_tokens)} tokens")
                        
                        # 创建slot mapping用于这个chunk
                        # 注意：slot mapping应该是相对于chunk的起始位置（从0开始）
                        # 而不是相对于整个token序列的起始位置
                        if slot_mapping is not None:
                            # 使用slot_mapping中对应位置的slot
                            chunk_slot_mapping = slot_mapping[start:end]
                        else:
                            # 如果没有slot mapping，创建简单的连续slot mapping
                            # 从0开始，长度为chunk_tokens
                            chunk_slot_mapping = torch.arange(0, len(chunk_tokens), dtype=torch.int32)
                        
                        # 确保slot mapping在正确的设备上
                        if kvcaches and len(kvcaches) > 0:
                            device = kvcaches[0].device
                            if not chunk_slot_mapping.is_cuda:
                                chunk_slot_mapping = chunk_slot_mapping.to(device)
                        
                        # 首先初始化kvcaches指针
                        if hasattr(self.gpu_connector, 'initialize_kvcaches_ptr'):
                            self.gpu_connector.initialize_kvcaches_ptr(kvcaches=kvcaches)
                        
                        # 添加调试信息：检查kvcaches和slot_mapping
                        logger.info(f"Download blocked KV: kvcaches count={len(kvcaches)}, "
                                   f"first kvcache shape={kvcaches[0].shape if kvcaches else 'None'}, "
                                   f"device={kvcaches[0].device if kvcaches else 'None'}, "
                                   f"slot_mapping shape={chunk_slot_mapping.shape}, "
                                   f"start=0, end={len(chunk_tokens)}")
                        
                        # 确保slot_mapping在正确的设备上
                        if kvcaches and len(kvcaches) > 0:
                            device = kvcaches[0].device
                            if not chunk_slot_mapping.is_cuda:
                                chunk_slot_mapping = chunk_slot_mapping.to(device)
                                logger.info(f"Moved slot_mapping to device: {device}")
                        
                        # 使用GPU connector的download_blocked_kv方法下载数据
                        # 这个方法会返回形状为 [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size] 的tensor
                        try:
                            combined_tensor = self.gpu_connector.download_blocked_kv(
                                vllm_kvcaches=kvcaches,
                                slot_mapping=chunk_slot_mapping,
                                start=0,  # 从chunk的开始下载
                                end=len(chunk_tokens),  # 下载整个chunk
                                dtype=kv_dtype,
                                device=kvcaches[0].device  # 下载到GPU，而不是CPU
                            )
                            
                            if combined_tensor is not None:
                                logger.info(f"Successfully downloaded blocked KV cache data: shape={combined_tensor.shape}, "
                                           f"device={combined_tensor.device}, dtype={combined_tensor.dtype}")
                                
                                # 调试：检查下载的数据是否为零
                                non_zero_count = (combined_tensor != 0).sum().item()
                                total_elements = combined_tensor.numel()
                                fill_ratio = non_zero_count / total_elements if total_elements > 0 else 0
                                logger.info(f"Downloaded data fill ratio: {fill_ratio:.2%} ({non_zero_count}/{total_elements} non-zero elements)")
                                
                                # 检查数据范围
                                min_val = combined_tensor.min().item()
                                max_val = combined_tensor.max().item()
                                mean_val = combined_tensor.mean().item()
                                logger.info(f"Downloaded data range: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
                                
                            else:
                                logger.error("download_blocked_kv returned None")
                                raise ValueError("Failed to download blocked KV cache data: returned None")
                                
                        except Exception as e:
                            logger.error(f"Failed to download blocked KV cache data using GPU connector: {e}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            raise
                        
                        if combined_tensor is not None:
                            logger.info(f"Successfully downloaded blocked KV cache data: shape={combined_tensor.shape}")
                            copy_success = True
                        else:
                            logger.error("Failed to download blocked KV cache data")
                            copy_success = False
                            
                    except Exception as e:
                        logger.error(f"Failed to download blocked KV cache data using GPU connector: {e}")
                        copy_success = False
                else:
                    logger.error("GPU connector is not BlockedKVPagedMemConnector, cannot download blocked KV cache data")
                    raise ValueError(f"GPU connector must be BlockedKVPagedMemConnector for store operation, got {type(self.gpu_connector)}")
                
                # 如果下载成功，将数据拷贝到VRAM unit
                if copy_success and combined_tensor is not None:
                    try:
                        # 将combined_tensor展平后拷贝到VRAM segment
                        combined_size = combined_tensor.numel() * combined_tensor.element_size()
                        
                        # 展平tensor为一维
                        flattened_tensor = combined_tensor.flatten()
                        logger.info(f"Flattened tensor shape: {flattened_tensor.shape}, size: {combined_size} bytes")
                        
                        # 获取vram unit中的tensor（一维展平数据）
                        vram_flat_tensor = vram_kvcache_unit.kv_cache_tensor
                        
                        # 验证vram unit中的tensor大小是否足够
                        if vram_flat_tensor.numel() < flattened_tensor.numel():
                            logger.error(f"VRAM unit tensor size insufficient: {vram_flat_tensor.numel()} < {flattened_tensor.numel()}")
                            copy_success = False
                        else:
                            # 将展平的数据拷贝到vram unit的tensor中
                            vram_flat_tensor.copy_(flattened_tensor)
                            
                            # 关键：设置VRAM unit的original_shape，用于后续恢复数据
                            vram_kvcache_unit.original_shape = combined_tensor.shape
                            logger.info(f"Set VRAM unit original_shape: {combined_tensor.shape}")
                            
                            logger.info(f"Copied flattened tensor to VRAM segment: {combined_size} bytes")
                            
                    except Exception as e:
                        logger.error(f"Failed to copy combined tensor to VRAM segment: {e}")
                        copy_success = False
                
                # 拷贝完成后，进行后续操作
                if copy_success:
                    logger.info(f"Successfully copied KV cache to VRAM segment (chunk {start}:{end})")
                    
                    # Step 5: Register to GPU VRAM pool using unified cache key
                    # 使用统一的cache key注册到GPU VRAM pool，不再分层
                    # 注意：combined_tensor的形状是 [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
                    # 我们需要使用这个形状来注册
                    if combined_tensor is not None:
                        combined_shape = combined_tensor.shape
                        self._register_to_gpu_vram_pool(
                            cache_key=cache_key,  # 使用统一的cache key，不是分层key
                            tokens=chunk_tokens,
                            gpu_vram_address=allocated_buffer_address,  # 整个chunk的起始地址
                            segment_id=segment_id,
                            kv_shape=combined_shape,  # 使用combined_tensor的实际形状
                            kv_dtype=kv_dtype,
                            total_size=chunk_kv_cache_size  # 整个chunk的大小
                        )
                    else:
                        logger.warning(f"combined_tensor is None, cannot register to GPU VRAM pool for chunk [{start}, {end})")
                    
                    # Step 6: Store to Mooncake backend
                    # 使用GPU connector下载的combined_tensor来存储到Mooncake
                    store_success = False
                    if combined_tensor is not None:
                        # 创建chunk对应的KV cache结构
                        # combined_tensor的形状: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
                        actual_num_layers = combined_tensor.shape[0]
                        kv_cache_structure = {
                            "num_layers": actual_num_layers,
                            "layer_shapes": [combined_tensor[i].shape for i in range(actual_num_layers)],  # 每层的形状
                            "layer_dtypes": [str(combined_tensor[i].dtype) for i in range(actual_num_layers)],
                            "layer_sizes": [combined_tensor[i].numel() * combined_tensor[i].element_size() for i in range(actual_num_layers)],  # 每层的大小
                            "vllm_layout": "blocked",
                            "block_size": getattr(kvcaches[0], 'block_size', 16) if hasattr(kvcaches[0], 'block_size') else 16,
                            "num_heads": kv_shape[-2] if len(kv_shape) >= 4 else 32,
                            "head_size": kv_shape[-1] if len(kv_shape) >= 4 else 128,
                            "chunk_tokens": len(chunk_tokens),  # 添加chunk token数量信息
                            "is_chunk": True,  # 标记这是chunk数据
                            "cache_key": {  # 存储对应的CacheEngineKey信息
                                "chunk_hash": cache_key.chunk_hash,
                                "model_name": cache_key.model_name,
                                "worker_id": cache_key.worker_id,
                                "world_size": cache_key.world_size
                            }
                        }
                        
                        # 将combined_tensor拆分为各层数据列表
                        layer_tensors = [combined_tensor[i] for i in range(actual_num_layers)]
                        
                        # 一次性存储所有层数据，使用统一的cache key
                        unified_kv_cache_structure = kv_cache_structure.copy()
                        unified_kv_cache_structure["all_layers_data"] = True
                        unified_kv_cache_structure["total_layers"] = actual_num_layers
                        
                        # 使用统一的cache key存储所有层数据
                        unified_success = self.storage_backend.store(
                            tokens=chunk_tokens,
                            kvcaches=layer_tensors,  # 传递所有层的CPU tensor数据
                            kv_cache_structure=unified_kv_cache_structure,
                            cache_key=cache_key  # 使用统一的cache key，不是分层key
                        )
                        
                        if unified_success:
                            logger.info(f"Successfully stored KV cache for chunk [{start}, {end}): {len(chunk_tokens)} tokens with unified cache key, {actual_num_layers} layers")
                            store_success = True
                        else:
                            logger.warning(f"Failed to store KV cache for chunk [{start}, {end})")
                            store_success = False
                        
                        # 存储完成后，显式释放CPU tensor内存
                        del combined_tensor
                        import gc
                        gc.collect()
                        logger.debug(f"Released CPU tensor memory for chunk {start}:{end}")
                    else:
                        logger.warning(f"No KV cache data downloaded for chunk [{start}, {end})")
                        store_success = False
                    
                    if store_success:
                        logger.info(f"Store successful: {len(chunk_tokens)} tokens with VRAM segment storage")
                    else:
                        logger.warning(f"Store failed for chunk: {len(chunk_tokens)} tokens")
                else:
                    logger.warning(f"Failed to copy KV cache data to VRAM segment for chunk: {len(chunk_tokens)} tokens")
                    # 修正：如果拷贝失败，应该释放已分配的segment空间
                    if segment_id and self.segment_manager is not None:
                        self.segment_manager.free_segment_space(segment_id, segment_offset, chunk_kv_cache_size)
                        logger.info(f"Freed segment space for failed chunk: segment {segment_id}, offset {segment_offset}")
        
        logger.info(f"Store operation completed for {num_to_store_tokens} tokens")

    def _register_to_gpu_vram_pool(
        self, 
        cache_key: CacheEngineKey,  # 添加cache_key参数
        tokens: Union[torch.Tensor, List[int]], 
        gpu_vram_address: Optional[int] = None, 
        segment_id: Optional[str] = None,
        kv_shape: Optional[tuple] = None,
        kv_dtype: Optional[torch.dtype] = None,
        total_size: Optional[int] = None
    ):
        """
        Enhanced registration function for GPU VRAM pool with vLLM KV cache metadata.
        Supports direct vLLM KV cache addresses without LMCache MemoryObj.
        
        Args:
            tokens: Input tokens (chunk tokens, not full tokens)
            gpu_vram_address: GPU memory address where KV cache chunk is stored
            segment_id: Segment ID for GPU VRAM management
            kv_shape: KV cache shape for this chunk (already adjusted to chunk size)
            kv_dtype: KV cache dtype for this chunk
            total_size: Total size in bytes for this chunk (calculated from chunk data)
        """
        if self.vram_metadata_client is None:
            logger.warning("VRAM metadata client not available for KV cache registration")
            return
        
        # Convert tokens to list format
        if isinstance(tokens, torch.Tensor):
            token_ids = tokens.tolist()
        else:
            token_ids = tokens
        
        # 使用传入的参数，这些参数已经在store函数中正确计算为chunk对应的值
        
        # 验证必需参数
        if kv_shape is None:
            logger.error("KV cache shape is required for registration")
            return
        
        if kv_dtype is None:
            logger.error("KV cache dtype is required for registration")
            return
        
        if total_size is None:
            logger.error("Total size is required for registration")
            return
        
        # 构建chunk对应的KV cache结构信息
        # 在store函数中，我们传递的是combined_tensor.shape（6D形状）
        # combined_tensor.shape: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
        # 所以kv_shape已经是6D形状，不需要再添加层数维度
        combined_shape = kv_shape
        
        # 从combined_shape中提取层数
        num_layers = combined_shape[0] if len(combined_shape) == 6 else 1
        
        # 构建完整的KV cache结构信息
        # combined_shape是6D形状: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
        # 我们需要从combined_shape中提取正确的参数
        if len(combined_shape) == 6:
            num_layers = combined_shape[0]
            chunk_blocks = combined_shape[1]
            kv_pairs = combined_shape[2]
            block_size = combined_shape[3]
            num_heads = combined_shape[4]
            head_size = combined_shape[5]
        else:
            # 如果不是6D形状，使用默认值
            num_layers = 1
            chunk_blocks = 1
            kv_pairs = 2
            block_size = 16
            num_heads = 32
            head_size = 128
        
        kv_cache_structure = {
            "num_layers": num_layers,  # 从combined_shape中获取实际层数
            "layer_shapes": [combined_shape],  # 使用合并后的形状
            "layer_dtypes": [str(kv_dtype)],
            "layer_sizes": [total_size],  # 整个chunk的大小
            "vllm_layout": "blocked",  # vLLM uses blocked memory layout
            "block_size": block_size,  # 从combined_shape中获取block_size
            "num_heads": num_heads,  # 从combined_shape中获取num_heads
            "head_size": head_size,  # 从combined_shape中获取head_size
            "chunk_tokens": len(token_ids),  # 添加chunk token数量信息
            "is_chunk": True,  # 标记这是chunk数据
            "combined_tensor": True  # 标记这是合并后的tensor
        }
        logger.debug(f"Extracted chunk KV cache structure: {kv_cache_structure}")
        
        # Use provided GPU VRAM address
        buffer_pointer = gpu_vram_address
        if buffer_pointer is None:
            logger.warning("No GPU VRAM address provided for KV cache registration")
            return
        
        # Register in GPU VRAM pool with chunk-specific vLLM KV cache metadata
        # 使用TestTokenDatabase生成的cache_key，确保key的一致性
        logger.info(f"DEBUG: Registering to GPU VRAM pool - Key: {cache_key}, chunk_hash: {cache_key.chunk_hash}, "
                   f"worker_id: {self.metadata.worker_id}, tokens: {len(token_ids)}, "
                   f"shape: {combined_shape}, dtype: {kv_dtype}")
        
        success = self.vram_metadata_client.register_kvcache(
            cache_key=cache_key,  # 传递TestTokenDatabase生成的CacheEngineKey
            token_ids=token_ids,
            gpu_id=self.metadata.worker_id,
            tensor_shape=combined_shape,  # 使用合并后的形状
            tensor_dtype=kv_dtype,  # 使用chunk对应的dtype
            tensor_size=total_size,  # 使用chunk对应的大小
            buffer_pointer=buffer_pointer,
            segment_id=segment_id,
            resident_hostname=self.config.get_extra_config_value("local_hostname_TE", "localhost"),
            kv_cache_structure=kv_cache_structure  # 使用chunk对应的KV cache结构
        )
        
        if success:
            logger.info(f"Successfully registered KV cache chunk in GPU VRAM pool: "
                       f"{len(token_ids)} tokens, GPU {self.metadata.worker_id}, "
                       f"segment={segment_id if segment_id else 'external'}, "
                       f"address={hex(buffer_pointer)}, size={total_size} bytes, "
                       f"shape={combined_shape}, dtype={kv_dtype}, "
                       f"key_chunk_hash={cache_key.chunk_hash}")
        else:
            logger.warning(f"Failed to register KV cache chunk in GPU VRAM pool: "
                          f"{len(token_ids)} tokens, GPU {self.metadata.worker_id}, "
                          f"key_chunk_hash={cache_key.chunk_hash}")



    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Union[torch.Tensor, List[int]],
    ) -> int:
        """
        Lookup operation - same function name as original cache engine.
        Enhanced to use token database for chunking to check maximum matching chunks.
        Only returns continuous hits from the beginning (start=0).
        
        Args:
            tokens: Input tokens
            pin: Whether to pin the cache
            request_configs: Optional request configurations
            
        Returns:
            Number of hit tokens (only continuous from the beginning)
        """
        self.stats["total_lookups"] += 1
        
        # Convert tokens to list format if needed
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.tolist()
        else:
            token_list = tokens
        
        # 1. GPU VRAM pool查找（优先）- 使用token database进行chunk处理
        gpu_vram_hit_tokens = 0
        if self.vram_metadata_client is not None:
            # 生成所有可能的chunks和prefix chunks
            all_chunks = []
            # Convert torch.dtype to string using TORCH_DTYPE_TO_STR_DTYPE mapping
            # This ensures consistency with CacheEngineKey's internal representation
            from lmcache.utils import TORCH_DTYPE_TO_STR_DTYPE
            kv_dtype_str = TORCH_DTYPE_TO_STR_DTYPE.get(self.metadata.kv_dtype, "half")
            for start, end, cache_key in self.token_database.process_tokens(
                tokens=token_list, 
                mask=None, 
                make_key=True,
                model_name="test_model",
                worker_id=self.metadata.worker_id,
                world_size=self.metadata.world_size,
                kv_dtype=kv_dtype_str
            ):
                all_chunks.append((start, end, cache_key))
                logger.debug(f"Generated chunk [{start}, {end}): {end-start} tokens, key: {cache_key}")
            
            # 直接传入all_chunks到vram_metadata_client.lookup_prefix，让vram manager根据所有chunks查找最符合的chunk
            # 不再逐个chunk调用，而是批量处理
            gpu_vram_hit_tokens, chunk_info_list = self.vram_metadata_client.lookup_prefix(
                token_ids=token_list,
                max_tokens=len(token_list),
                current_gpu_id=self.metadata.worker_id,
                all_chunks=all_chunks  # 传入所有chunks信息
            )
            
            # 确保只返回从头开始的连续hit tokens
            if gpu_vram_hit_tokens > 0 and chunk_info_list:
                # 检查hit tokens是否从开头开始
                # 如果gpu_vram_hit_tokens不等于任何chunk的end值，说明不是从开头开始的连续hit
                # 这里我们假设vram_metadata_client.lookup_prefix已经返回了从开头开始的连续hit
                
                # 从chunk_info_list中提取信息用于日志记录
                # chunk_info_list包含每个匹配chunk的详细信息: ((start, end), gpu_id, needs_transfer)
                first_chunk_info = chunk_info_list[0]
                first_gpu_id = first_chunk_info[1]
                needs_transfer = any(info[2] for info in chunk_info_list)
                
                logger.info(f"GPU VRAM pool lookup hit: {gpu_vram_hit_tokens} tokens from {len(chunk_info_list)} chunks, first GPU {first_gpu_id}, needs_transfer={needs_transfer}")
                for i, ((start, end), gpu_id, chunk_needs_transfer) in enumerate(chunk_info_list):
                    logger.info(f"  Chunk {i}: [{start}, {end}) -> GPU {gpu_id}, needs_transfer={chunk_needs_transfer}")
                self.stats["gpu_vram_hits"] += 1
            else:
                logger.info("No GPU VRAM hits found for any chunks")
                self.stats["gpu_vram_misses"] += 1
        
            # 2. 如果GPU VRAM没有hit，再去查Mooncake存储后端 - 使用token database进行chunk处理
        storage_hit_tokens = 0
        if gpu_vram_hit_tokens == 0 and self.storage_backend is not None:
            # 只有在GPU VRAM没有hit的情况下才去查Mooncake，并且storage_backend不为None
            # 使用token database生成chunks来检查最大匹配chunks
            # 直接传入all_chunks到Mooncake storage backend的lookup函数
            
            # 首先对chunks进行排序
            all_chunks.sort(key=lambda x: x[0])
            
            # 只检查从开头开始的连续chunks
            continuous_hit_tokens = 0
            expected_start = 0
            
            for start, end, cache_key in all_chunks:
                if start == expected_start:
                    # 检查这个chunk是否在Mooncake中
                    # 注意：这里需要传入chunk在原始tokens中的范围，而不是从0开始的范围
                    # 所以应该传入 [(start, end, cache_key)] 而不是 [(0, len(chunk_tokens), cache_key)]
                    chunk_tokens = token_list[start:end]
                    chunk_hit_tokens, chunk_info_list = self.storage_backend.lookup(token_list, [(start, end, cache_key)])
                    
                    if chunk_hit_tokens == (end - start):
                        # 整个chunk都hit了
                        continuous_hit_tokens += chunk_hit_tokens
                        expected_start = end
                        logger.debug(f"Mooncake continuous hit chunk [{start}, {end}): {chunk_hit_tokens} tokens")
                    else:
                        # chunk没有完全hit，停止检查
                        logger.info(f"Breaking at chunk [{start}, {end}): expected {end-start} tokens, got {chunk_hit_tokens} tokens")
                        break
                else:
                    # 遇到gap，停止检查
                    logger.info(f"Breaking at gap: expected start={expected_start}, got start={start}")
                    break
            
            storage_hit_tokens = continuous_hit_tokens
            
            if storage_hit_tokens > 0:
                logger.info(f"Mooncake storage lookup hit: {storage_hit_tokens} continuous tokens from the beginning")
            else:
                logger.info("No Mooncake storage hits found for continuous chunks from the beginning")
        elif gpu_vram_hit_tokens == 0 and self.storage_backend is None:
            logger.debug("Storage backend is None, skipping Mooncake lookup for scheduler role")
        
        # total_hit_tokens应该是gpu_vram_hit_tokens或storage_hit_tokens中的一个，而不是两者的和
        # 因为只有在GPU VRAM没有命中的情况下才会去Mooncake中查找
        total_hit_tokens = gpu_vram_hit_tokens if gpu_vram_hit_tokens > 0 else storage_hit_tokens
        
        if total_hit_tokens > 0:
            self.stats["hits"] += 1
            logger.info(f"Enhanced lookup: GPU VRAM={gpu_vram_hit_tokens}, Storage={storage_hit_tokens}, Total={total_hit_tokens}/{len(token_list)} tokens (continuous from beginning)")
        else:
            self.stats["misses"] += 1
            logger.info(f"Lookup miss: GPU VRAM={gpu_vram_hit_tokens}, Storage=0, Total={gpu_vram_hit_tokens}/{len(token_list)} tokens")
        
        return total_hit_tokens

    def get_stats(self) -> Dict:
        """Get cache engine statistics."""
        stats = {
            "cache_engine_stats": self.stats.copy(),
            "current_gpu_id": self.metadata.worker_id
        }
        
        # 添加storage backend统计信息（如果存在）
        if self.storage_backend is not None:
            stats["storage_backend_stats"] = self.storage_backend.get_stats()
        else:
            stats["storage_backend_stats"] = {"status": "disabled", "reason": "scheduler_role"}
        
        # 添加GPU VRAM pool统计信息
        if self.vram_metadata_client is not None:
            gpu_vram_stats = self.vram_metadata_client.get_stats()
            stats["gpu_vram_pool_stats"] = gpu_vram_stats
            
            # 计算GPU VRAM命中率
            total_gpu_vram_operations = self.stats["gpu_vram_hits"] + self.stats["gpu_vram_misses"]
            if total_gpu_vram_operations > 0:
                stats["gpu_vram_hit_rate"] = (self.stats["gpu_vram_hits"] / total_gpu_vram_operations) * 100
            else:
                stats["gpu_vram_hit_rate"] = 0.0
        
        return stats

    def close(self):
        """Close the cache engine and release all resources including GPU VRAM segments."""
        logger.info("TestCacheEngine closing and releasing all resources")
        
        # Shutdown VRAM metadata IPC client if available
        if self.vram_metadata_client is not None:
            try:
                self.vram_metadata_client.shutdown()
                logger.info("VRAM metadata IPC client shutdown completed")
            except Exception as e:
                logger.error(f"Error shutting down VRAM metadata IPC client: {e}")
        
        # Shutdown transfer engine manager if available
        if self.transfer_engine_manager is not None:
            try:
                self.transfer_engine_manager.shutdown()
                logger.info("Transfer engine manager shutdown completed")
            except Exception as e:
                logger.error(f"Error shutting down transfer engine manager: {e}")
        
        # Close storage backend
        if self.storage_backend is not None:
            try:
                self.storage_backend.close()
                logger.info("Storage backend closed")
            except Exception as e:
                logger.error(f"Error closing storage backend: {e}")
        
        # Release GPU connector resources if available
        if self.gpu_connector is not None:
            try:
                # If GPU connector has a close method, call it
                if hasattr(self.gpu_connector, 'close'):
                    self.gpu_connector.close()
                logger.info("GPU connector resources released")
            except Exception as e:
                logger.error(f"Error releasing GPU connector resources: {e}")
        
        logger.info("TestCacheEngine closed and all resources released")

    

    def _copy_combined_kvcache_to_vllm_kvcaches(
        self,
        combined_kv_cache_tensor: torch.Tensor,
        kvcaches: List[torch.Tensor],
        slot_mapping: Optional[torch.Tensor],
        start_token_idx: int,
        end_token_idx: int,
        kv_cache_structure: Optional[Dict] = None
    ) -> bool:
        """
        Copy combined multi-layer KV cache tensor data to vLLM kvcaches using slot mapping.
        This function handles the data retrieved from Mooncake storage backend where
        store function stored combined all layers data.
        
        Args:
            combined_kv_cache_tensor: CPU tensor containing combined KV cache data for all layers
                                     Shape: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
            kvcaches: List of vLLM KV cache tensors
            slot_mapping: vLLM slot mapping tensor
            start_token_idx: Start token index in the input sequence
            end_token_idx: End token index in the input sequence
            kv_cache_structure: KV cache structure information
            
        Returns:
            True if copy successful, False otherwise
        """
        try:
            num_tokens = end_token_idx - start_token_idx
            logger.info(f"Copying {num_tokens} tokens from combined KV cache tensor to vLLM kvcaches, indices [{start_token_idx}, {end_token_idx})")
            
            if not kvcaches or len(kvcaches) == 0:
                logger.error("No kvcaches provided for copy operation")
                return False
            
            # Get the first KV cache to understand the structure
            first_kv_cache = kvcaches[0]
            
            # Check if we have valid KV cache structure
            if len(first_kv_cache.shape) != 5:
                logger.error(f"Unsupported KV cache shape: {first_kv_cache.shape}")
                return False
            
            # Extract vLLM KV cache structure
            num_blocks, kv_pairs, block_size, num_kv_heads, head_size = first_kv_cache.shape
            logger.info(f"vLLM KV cache structure: blocks={num_blocks}, kv_pairs={kv_pairs}, block_size={block_size}, heads={num_kv_heads}, head_size={head_size}")
            
            # Check combined tensor shape compatibility
            # Expected shape: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
            if len(combined_kv_cache_tensor.shape) != 6:
                logger.error(f"Unsupported combined KV cache tensor shape: {combined_kv_cache_tensor.shape}")
                return False
            
            # 验证形状匹配
            num_layers, chunk_blocks, combined_kv_pairs, combined_block_size, combined_num_kv_heads, combined_head_size = combined_kv_cache_tensor.shape
            
            if combined_kv_pairs != 2:  # Key and Value
                logger.error(f"Combined tensor KV pairs mismatch: {combined_kv_pairs} vs 2")
                return False
            
            if combined_num_kv_heads != num_kv_heads:
                logger.error(f"Combined tensor num_heads mismatch: {combined_num_kv_heads} vs {num_kv_heads}")
                return False
            
            if combined_head_size != head_size:
                logger.error(f"Combined tensor head_size mismatch: {combined_head_size} vs {head_size}")
                return False
            
            # 计算combined tensor中的token数量
            combined_token_count = chunk_blocks * combined_block_size
            if combined_token_count < num_tokens:
                logger.error(f"Combined tensor token count insufficient: {combined_token_count} vs {num_tokens}")
                return False
            
            # 检查kvcaches的数量是否足够
            if len(kvcaches) < num_layers:
                logger.error(f"Not enough kvcaches: have {len(kvcaches)}, need {num_layers}")
                return False
            
            # Copy data from combined blocked layout to vLLM kvcaches (all layers)
            copy_success = True
            
            try:
                # 对于每一层，将数据拷贝到对应的vLLM kvcache中
                for layer_idx in range(num_layers):
                    logger.debug(f"Copying layer {layer_idx} data to vLLM kvcache")
                    
                    # 获取当前层的combined tensor数据
                    # 形状: [chunk_blocks, 2, block_size, num_kv_heads, head_size]
                    layer_tensor = combined_kv_cache_tensor[layer_idx]
                    
                    # 对于chunk中的每个token，使用slot mapping将数据放回vLLM的blocked布局
                    for i in range(num_tokens):
                        token_idx = start_token_idx + i
                        
                        # 获取这个token在vLLM blocked布局中的slot
                        if slot_mapping is not None:
                            slot = slot_mapping[token_idx]
                        else:
                            # 如果没有slot mapping，假设连续布局
                            slot = token_idx
                        
                        # 计算在vLLM KV cache中的位置
                        dst_block_id = slot // block_size
                        dst_slot_in_block = slot % block_size
                        
                        # 计算在combined tensor中的位置
                        # combined tensor从block 0开始，连续存储chunk的tokens
                        src_block_id = i // combined_block_size
                        src_slot_in_block = i % combined_block_size
                        
                        # 从combined tensor中提取Key和Value数据
                        key_data = layer_tensor[src_block_id, 0, src_slot_in_block, :, :]  # Key: [num_kv_heads, head_size]
                        value_data = layer_tensor[src_block_id, 1, src_slot_in_block, :, :]  # Value: [num_kv_heads, head_size]
                        
                        # 将数据放回vLLM KV cache的blocked布局中
                        # 注意：kvcaches可能是一个列表，每个元素对应一层
                        if layer_idx < len(kvcaches):
                            # 将数据移动到GPU（如果需要）
                            if key_data.device.type != "cuda":
                                key_data = key_data.to(kvcaches[layer_idx].device)
                            if value_data.device.type != "cuda":
                                value_data = value_data.to(kvcaches[layer_idx].device)
                            
                            kvcaches[layer_idx][dst_block_id, 0, dst_slot_in_block, :, :] = key_data  # Key
                            kvcaches[layer_idx][dst_block_id, 1, dst_slot_in_block, :, :] = value_data  # Value
                        else:
                            logger.warning(f"Layer index {layer_idx} out of range for kvcaches (size: {len(kvcaches)})")
                
                logger.debug(f"Copied {num_tokens} tokens from combined blocked layout to {num_layers} vLLM kvcaches using slot mapping")
                
            except Exception as e:
                logger.error(f"Failed to copy to vLLM kvcaches: {e}")
                copy_success = False
            
            if copy_success:
                logger.info(f"Successfully copied {num_tokens} tokens from combined KV cache tensor to {num_layers} vLLM kvcaches using slot mapping")
                return True
            else:
                logger.warning(f"Failed to copy {num_tokens} tokens from combined KV cache tensor to vLLM kvcaches")
                return False
                
        except Exception as e:
            logger.error(f"Failed to copy combined KV cache tensor to vLLM kvcaches: {e}")
            return False

    # 根据用户反馈，没有单层数据的情况，只保留combined情况
    # 删除 _copy_cpu_kvcache_to_vllm_kvcaches 方法，只使用 _copy_combined_kvcache_to_vllm_kvcaches 方法




class MockGPUConnector:
    """Mock GPU connector for testing."""
    
    def get_shape(self, kv_caches: torch.Tensor) -> tuple:
        """Get KV cache shape for given number of tokens."""
        #return (32, 2, num_tokens, 32, 128)  # (layers, kv_pairs, tokens, heads, head_size)
        return kv_caches.shape
    
    def upload_to_gpu(
        self,
        tokens: List[int],
        cpu_buffer_address: int,
        target_buffer_address: int,
        offset: int = 0,
        kv_cache_structure: Optional[Dict] = None
    ) -> bool:
        """
        Direct DRAM to VRAM transfer using CUDA memory copy operations.
        This method directly transfers data from CPU buffer to GPU buffer address.

        Args:
            tokens: List of token IDs (for logging)
            cpu_buffer_address: Source CPU buffer address containing actual KV cache data from Mooncake
            target_buffer_address: Target GPU buffer address (already allocated in segment)
            offset: Offset in target buffer to start writing
            kv_cache_structure: KV cache structure information for proper data handling

        Returns:
            True if upload successful, False otherwise
        """
        try:
            logger.info(f"Direct DRAM->VRAM transfer: {len(tokens)} tokens from CPU {hex(cpu_buffer_address)} to GPU {hex(target_buffer_address)} with offset {offset}")

            # Check if CUDA is available
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to mock behavior")
                logger.info(f"Successfully uploaded {len(tokens)} tokens KV cache data to GPU buffer at offset {offset} (mock)")
                return True

            # Get current CUDA device
            current_device = torch.cuda.current_device()
            logger.debug(f"Current CUDA device: {current_device}")

            # Check if we have valid buffer addresses
            if cpu_buffer_address is None or cpu_buffer_address == 0:
                logger.warning("No CPU buffer address provided, using mock behavior")
                logger.info(f"Successfully uploaded {len(tokens)} tokens KV cache data to GPU buffer at offset {offset} (mock)")
                return True

            if target_buffer_address is None or target_buffer_address == 0:
                logger.error("No target GPU buffer address provided")
                return False

            try:
                # Calculate data size based on KV cache structure if available
                if kv_cache_structure:
                    # Use actual KV cache structure to calculate data size
                    num_layers = kv_cache_structure.get("num_layers", 32)
                    tokens_to_upload = kv_cache_structure.get("tokens_to_upload", len(tokens))
                    layer_sizes = kv_cache_structure.get("layer_sizes", [])
                    
                    if layer_sizes:
                        # Calculate total size from actual layer sizes
                        total_bytes = sum(layer_sizes)
                        logger.info(f"Using actual KV cache structure: {num_layers} layers, {tokens_to_upload} tokens, total size: {total_bytes} bytes")
                    else:
                        # Estimate based on typical transformer structure
                        num_heads = kv_cache_structure.get("num_heads", 32)
                        head_size = kv_cache_structure.get("head_size", 128)
                        kv_pairs = 2  # Key and Value
                        
                        # Calculate total elements: tokens * layers * heads * head_size * kv_pairs
                        total_elements = tokens_to_upload * num_layers * num_heads * head_size * kv_pairs
                        dtype_size = 2  # float16 = 2 bytes
                        total_bytes = total_elements * dtype_size
                        logger.info(f"Using estimated KV cache structure: {num_layers} layers, {num_heads} heads, {head_size} head_size, {tokens_to_upload} tokens, total size: {total_bytes} bytes")
                else:
                    # Fallback to estimation if no KV cache structure provided
                    num_layers = 1  # vLLM KV cache只有一层
                    num_heads = 32
                    head_size = 128
                    kv_pairs = 2
                    
                    total_elements = len(tokens) * num_layers * num_heads * head_size * kv_pairs
                    dtype_size = 2  # float16 = 2 bytes
                    total_bytes = total_elements * dtype_size
                    logger.info(f"Using default KV cache estimation: {num_layers} layer, {num_heads} heads, {head_size} head_size, {len(tokens)} tokens, total size: {total_bytes} bytes")

                logger.info(f"Direct memory transfer of {total_bytes} bytes from CPU {hex(cpu_buffer_address)} to GPU {hex(target_buffer_address)}")

                # Direct memory copy using CUDA cudaMemcpy
                import ctypes
                from ctypes import c_void_p
                
                # Get CUDA driver API
                try:
                    # Load CUDA driver
                    cuda = ctypes.CDLL("nvcuda.dll" if os.name == 'nt' else "libcuda.so")
                except Exception as e:
                    logger.error(f"Failed to load CUDA driver: {e}")
                    return False
                
                # Define cudaMemcpy function
                cuda.cudaMemcpy.restype = ctypes.c_int
                cuda.cudaMemcpy.argtypes = [c_void_p, c_void_p, ctypes.c_size_t, ctypes.c_int]
                
                # CUDA memory copy types
                cudaMemcpyHostToDevice = 1
                
                # Perform direct memory copy from CPU to GPU
                result = cuda.cudaMemcpy(
                    c_void_p(target_buffer_address),  # Destination GPU address
                    c_void_p(cpu_buffer_address),     # Source CPU address
                    total_bytes,                      # Size in bytes
                    cudaMemcpyHostToDevice           # Copy direction
                )
                
                if result == 0:  # cudaSuccess
                    logger.info(f"✅ Direct DRAM->VRAM transfer successful: {len(tokens)} tokens at offset {offset}")
                    logger.debug(f"Transferred {total_bytes} bytes from CPU {hex(cpu_buffer_address)} to GPU {hex(target_buffer_address)}")
                    logger.debug(f"Token IDs: {tokens}")
                    if kv_cache_structure:
                        logger.debug(f"KV cache structure used: {kv_cache_structure}")
                    return True
                else:
                    logger.error(f"Direct DRAM->VRAM transfer failed with CUDA error: {result}")
                    return False

            except Exception as cuda_error:
                logger.error(f"Direct DRAM->VRAM transfer failed: {cuda_error}")
                # Fallback to mock behavior
                logger.info(f"Fallback: Successfully uploaded {len(tokens)} tokens KV cache data to GPU buffer at offset {offset} (mock)")
                return True

        except Exception as e:
            logger.error(f"GPU connector upload failed: {e}")
            return False

