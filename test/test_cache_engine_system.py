"""
VCache Engine System

"""

# Standard
from typing import Any, Dict, List, Optional, Tuple, Union
import os
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
    logger.warning("Mooncake store and transfer engine not available")

class TestCacheEngine:
    """
    VCache Engine System

    """
    
    def __init__(
        self,
        config: TestConfig,
        metadata: LMCacheEngineMetadata,
        gpu_connector: Optional[GPUConnectorInterface] = None,
    ):
        """
        Initialize the VCache Engine.
        
        Args:
            config: VCache engine configuration
            metadata: Engine metadata
            gpu_connector: GPU connector
        """
        logger.info(f"Creating VCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.gpu_connector = gpu_connector
        # Get connector role from config
        connector_role = self.config.connector_role

        # Initialize VRAM metadata IPC client
        self.vram_metadata_client = None

        if self.config.get_extra_config_value("enable_gpu_vram_pool", True):
            logger.info("GPU VRAM pool manager is initializing...")
            # Check if we should use centralized metadata server via IPC
            use_metadata_server = self.config.get_extra_config_value("use_vram_metadata_server", True)

            assert use_metadata_server is True, "VRAM metadata server must be enabled"
            
            # Use centralized VRAM metadata server via IPC
            logger.info("Using centralized VRAM metadata server via IPC")
            self.vram_metadata_client = get_vram_metadata_ipc_client(self.config)
            
            # Check if IPC client is connected
            if self.vram_metadata_client.is_connected:
                logger.info("VRAM metadata IPC client connected successfully")
                # No local GPU VRAM manager in IPC mode - all operations go through IPC client
            else:
                self.vram_metadata_client = None
                assert self.vram_metadata_client.is_connected, "VRAM metadata IPC client must be connected"                         
        else:
            logger.warning("GPU VRAM pool disabled in configuration")
        
        
        # Initialize transfer engine manager for this cache engine instance
        self.transfer_engine_manager = None
        # Only initialize transfer engine for worker role
        if connector_role == "worker":
            if TRANSFER_ENGINE_AVAILABLE:
                self.transfer_engine_manager = TransferEngineManager(self.config)
            else:
                logger.error("Transfer engine not available")
                assert self.transfer_engine_manager is not None, "Transfer engine must be initialized if enabled"
        else:
            logger.info(f"Transfer engine disabled (connector_role={connector_role})")


        # Initialize GPU VRAM segment manager for this cache engine instance
        self.segment_manager = None
        # Only initialize segment manager for worker role
        if connector_role == "worker":
            if self.config.get_extra_config_value("enable_gpu_vram_segments", True):
                self.segment_manager = GPUVRAMSegmentManager(
                    self.config, 
                    self.metadata.worker_id, 
                    self.transfer_engine_manager,  # Pass transfer engine manager for segment buffer registration
                    self.vram_metadata_client  # Pass GPU VRAM Pool Manager client for metadata registration
                )
            else:
                logger.error("GPU VRAM segments disabled in configuration")
                assert self.segment_manager is not None, "GPU VRAM segments must be initialized if enabled"
        else:
            logger.info(f"GPU VRAM segments disabled (connector_role={connector_role})")

        
        # Initialize storage backend - use Mooncake if available
        # Only initialize storage backend for worker role (scheduler doesn't need storage)
        self.storage_backend = None
        if connector_role == "worker":
            self.storage_backend = MooncakeStorageBackend(
                self.metadata.worker_id, 
                self.config, 
            )
            assert self.storage_backend is not None, "Storage backend must be initialized"
        else:
            logger.info("Storage backend disabled for scheduler role")
        
        # Initialize TestTokenDatabase for tokens processing
        self.token_database = TestTokenDatabase(chunk_size=256, save_unfull_chunk=True)
        assert self.token_database is not None, "Token database must be initialized"
        
        # Initialize BlockedKVPagedMemConnector if not provided
        # Only initialize GPU connector for worker role (scheduler doesn't need GPU operations)
        if self.gpu_connector is None:
            if connector_role == "worker":
                # extract kv cache shape params from metadata
                num_layers = metadata.kv_shape[0] if len(metadata.kv_shape) >= 5 else 32
                block_size = 16  # vLLM default block size
                num_kv_heads = metadata.kv_shape[-2] if len(metadata.kv_shape) >= 4 else 32
                head_size = metadata.kv_shape[-1] if len(metadata.kv_shape) >= 4 else 128
                
                self.gpu_connector = BlockedKVPagedMemConnector(
                    num_layers=num_layers,
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    use_gpu=True,
                    dtype=metadata.kv_dtype,
                    device=f"cuda:{metadata.worker_id}" if metadata.worker_id is not None else "cuda:0"
                )

                assert self.gpu_connector is not None, "GPU connector must be initialized for worker role"
            else:
                # Scheduler doesn't need GPU connector
                self.gpu_connector = None
                logger.info("No GPUConnector for scheduler role")

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
        logger.info(f"TestCacheEngine initialized for GPU {self.metadata.worker_id} with connector_role={connector_role}")


    def _perform_cross_gpu_transfer(self, entry, source_gpu: int, target_gpu: int, target_buffer: Optional[int] = None) -> bool:
        """
        Perform synchronous cross-GPU transfer using this Vcache engine's transfer engine.
        
        Args:
            entry: GPU VRAM entry
            source_gpu: Source GPU ID
            target_gpu: Target GPU ID
            target_buffer: Optional pre-allocated target buffer address. If None, will allocate automatically.
            
        Returns:
            True if transfer successful, False otherwise
        """
        if not self.transfer_engine_manager:
            logger.error("Transfer engine not available for cross-GPU transfer")
            return False
        
        try:
            logger.info(f"Starting cross-GPU transfer: GPU {source_gpu} -> GPU {target_gpu}, size: {entry.tensor_size} bytes")
            
            # Get the actual tensor data from source GPU using entry's buffer_pointer
            source_buffer = entry.buffer_pointer
            if source_buffer is None:
                logger.error(f"Failed to get source buffer address from entry for GPU {source_gpu}")
                return False
            
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
            
            # Perform actual cross-GPU transfer using transfer engine
            # Use entry's resident hostname as target hostname
            target_hostname = entry.resident_hostname if hasattr(entry, 'resident_hostname') else None

            if target_hostname is None:
                logger.error("Target hostname not available in entry for cross-GPU transfer")
                return False
            
            success = self.transfer_engine_manager.transfer_gpu_to_gpu(
                target_hostname=target_hostname,
                source_gpu=source_gpu,
                target_gpu=target_gpu,
                source_buffer=source_buffer,
                target_buffer=target_buffer,
                size=entry.tensor_size
            )
                  
            if success:
                logger.info(f"Cross-GPU transfer completed: GPU {source_gpu} -> GPU {target_gpu}, size: {entry.tensor_size} bytes")
                # Note: Registration of transferred entry is now handled by the caller
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
    
        logger.info(f"Registering transferred entry in GPU VRAM pool: GPU {target_gpu}, address: {hex(target_buffer)}")
        
        # Use the same token IDs, shape, and dtype as the original entry
        # Need to get the cache_key from the original entry
        if not hasattr(original_entry, 'key') or original_entry.key is None:
            logger.error("Original entry does not have a cache key, cannot register transferred copy")
            assert False, "Original entry must have a cache key for registration"
        
        success = self.vram_metadata_client.register_kvcache(
            cache_key=original_entry.key,  # Use the same cache key as original
            token_ids=original_entry.token_ids,
            gpu_id=target_gpu,
            tensor_shape=original_entry.tensor_shape,
            tensor_dtype=original_entry.tensor_dtype,
            tensor_size=original_entry.tensor_size,
            buffer_pointer=target_buffer,
            segment_id=None,  
            resident_hostname=self.config.get_extra_config_value("local_hostname_TE", "localhost"),
            kv_cache_structure=getattr(original_entry, 'kv_cache_structure', None)  # Pass KV cache structure if available
        )
        
        if success:
            logger.info(f"Successfully registered transferred entry in GPU VRAM pool: {len(original_entry.token_ids)} tokens on GPU {target_gpu} at address {hex(target_buffer)}")
        else:
            logger.error(f"Failed to register transferred entry in GPU VRAM pool")
            assert False, "Failed to register transferred entry in GPU VRAM pool"
                

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        tokens: List[int],
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        retrieve operation with GPU VRAM pool and Mooncake store integration.
        Supports mask for partial loading and slot mapping for vLLM integration.
        
        Workflow:
        1. Process tokens to generate all_chunks
        2. For all chunks, determine which need retrieval
        3. For chunks needing retrieval, perform lookup prefix to get all hit chunk info
        4. For local chunks (no transfer needed): batch get vram unit, then copy all to vllm kvcache
        5. For remote chunks (needs transfer): for each vram unit, allocate space in segment, get address,
           execute transfer, then copy each to vllm kvcache
        6. If no GPU VRAM hits or processing failed, try Mooncake storage:
            a. Generate lookup chunks
            b. Call storage_backend.lookup to get hit tokens and chunk info
            c. Process Mooncake hit by retrieving actual KV cache data and placing it in vLLM kvcaches
        
        Args:
            tokens: Input tokens
            mask: Optional mask for tokens (True = needs loading)
            **kwargs: Additional arguments including:
                - kvcaches: List of KV cache tensors from vLLM
                - slot_mapping: vLLM slot mapping tensor
                
        Returns:
            Boolean mask indicating retrieved tokens
        """

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )
        
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
        
        # Get kvcaches from kwargs
        kvcaches = kwargs.get("kvcaches")
        assert kvcaches is not None, "kvcaches is required for retrieve operation"
        
        # Get slot mapping from kwargs
        slot_mapping = kwargs.get("slot_mapping")
        assert slot_mapping is not None, "slot_mapping is required for retrieve operation"
        
        logger.info(f"Retrieve operation: {len(tokens)} tokens, mask={mask is not None}, "
                   f"kvcaches={len(kvcaches)}, slot_mapping={slot_mapping is not None}")
        
        # Step 1: Process tokens to generate all_chunks
        
        # Generate all chunks using token database with correct parameters
        all_chunks = []
        # Convert torch.dtype to string using TORCH_DTYPE_TO_STR_DTYPE mapping
        # This ensures consistency with CacheEngineKey's internal representation
        from lmcache.utils import TORCH_DTYPE_TO_STR_DTYPE
        kv_dtype_str = TORCH_DTYPE_TO_STR_DTYPE.get(self.metadata.kv_dtype, "half")
        for start, end, cache_key in self.token_database.process_tokens(
            tokens=tokens, 
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
        chunks_needing_retrieval = [(start, end, cache_key) for start, end, cache_key in all_chunks if cache_key is not None]
        
        # Step 3: For chunks needing retrieval, perform lookup prefix to get all hit chunk info
        gpu_vram_hits = []  # List of (start, end, cache_key, gpu_id, needs_transfer, entry)
        
        if self.vram_metadata_client is not None and chunks_needing_retrieval:
            # Extract just the cache keys for lookup
            cache_keys = [cache_key for _, _, cache_key in chunks_needing_retrieval]
            
            # Batch get entries for all cache keys
            entries = self.vram_metadata_client.batch_get_entry(cache_keys)
            
            logger.info(f"batch_get_entry - Got {len(entries)} entries, {len([e for e in entries if e])} non-None entries")
            
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
                    logger.debug(f"GPU VRAM hit for chunk [{start}, {end}), key: {cache_key}, chunk_hash: {cache_key.chunk_hash if hasattr(cache_key, 'chunk_hash') else 'N/A'}, gpu_id: {entry.gpu_id}, needs_transfer: {needs_transfer}")
                else:
                    logger.debug(f"No GPU VRAM hit for chunk [{start}, {end}), key: {cache_key}, chunk_hash: {cache_key.chunk_hash if hasattr(cache_key, 'chunk_hash') else 'N/A'}")
        
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
                    break
            
            if not continuous_hits:
                logger.debug("No continuous GPU VRAM hits from the beginning")
            else:
                logger.debug(f"Found {len(continuous_hits)} continuous GPU VRAM hits from the beginning, covering tokens [0, {expected_start})")
                
                # Separate local and remote hits
                local_hits = [(start, end, cache_key, entry) 
                             for start, end, cache_key, _, needs_transfer, entry in continuous_hits 
                             if not needs_transfer]
                remote_hits = [(start, end, cache_key, gpu_id, entry) 
                              for start, end, cache_key, gpu_id, needs_transfer, entry in continuous_hits 
                              if needs_transfer]
                
                logger.info(f"Processing {len(local_hits)} local GPU VRAM hits and {len(remote_hits)} remote GPU VRAM hits")
                
                # Process local hits (no transfer needed)
                local_success = False
                remote_success = False

                if local_hits:
                    local_success = self._process_local_gpu_vram_hits(
                        local_hits=local_hits,
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping,
                        ret_mask=ret_mask
                    )
                
                # Process remote hits (needs transfer)
                if remote_hits:
                    remote_success =self._process_remote_gpu_vram_hits(
                        remote_hits=remote_hits,
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping,
                        ret_mask=ret_mask
                    )

                assert local_success and remote_success, "Failed to process GPU VRAM hits"
                
        else:
            logger.info("No GPU VRAM hits found")
        
        # Step 5: If no GPU VRAM hits or processing failed, try Mooncake storage
        if ret_mask.sum().item() == 0:
            logger.info("No GPU VRAM hits processed successfully, fallback to Mooncake storage")
            
            # storage_backend.lookup returns a tuple (hit_tokens, chunk_info_list)
            lookup_result = self.storage_backend.lookup(tokens, chunks_needing_retrieval)
            
            if isinstance(lookup_result, tuple) and len(lookup_result) >= 2:
                storage_hit_tokens = lookup_result[0]
                chunk_info_list = lookup_result[1]
            else:
                storage_hit_tokens = lookup_result
                chunk_info_list = None
            
            if storage_hit_tokens == 0:
                logger.info(f"mooncake backend retrieve miss: No hits found for {num_required_tokens} tokens")
                return ret_mask
            
            logger.info(f"Mooncake storage hit: {storage_hit_tokens} tokens")
            
            # Process Mooncake storage hit with chunk_info_list
            # chunk_info_list now contains (start, end, cache_key) for each hit chunk
            # These chunks are continuous from the beginning (start=0)
            mooncake_success = self._process_mooncake_hit(
                tokens=tokens,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                ret_mask=ret_mask,
                chunks_needing_retrieval=chunk_info_list  # Directly use chunk_info_list from lookup
            )
            
            if mooncake_success:
                logger.info(f"Successfully processed Mooncake hit: {storage_hit_tokens} tokens")
            else:
                logger.error(f"Failed to process Mooncake hit")
                assert False, "Failed to process Mooncake hit"
        
        return ret_mask

    def _process_mooncake_hit(
        self,
        tokens: List[int],
        kvcaches: List[torch.Tensor],
        slot_mapping: Optional[torch.Tensor],
        ret_mask: torch.Tensor,
        all_chunks: List[Tuple[int, int, CacheEngineKey]] = None
    ) -> bool:
        """
        Process Mooncake storage hit by retrieving actual KV cache data and placing it in vLLM kvcaches.
        Only processes continuous hits from the beginning (start=0).
        Uses GPUConnector for batch upload to vLLM kvcaches.
        
        Args:
            tokens: Input tokens
            storage_hit_tokens: Number of storage hit tokens (from lookup)
            kvcaches: List of vLLM KV cache tensors
            slot_mapping: vLLM slot mapping tensor
            ret_mask: Return mask to mark retrieved tokens
            all_chunks: List of chunks needing retrieval (start, end, cache_key)
            
        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing Mooncake storage tokens")
        
        
        # Step 1: Use provided chunks
        if all_chunks is None:
            logger.error("chunks_needing_retrieval must be provided")
            return False
        
        # MooncakeStorageBackend.lookup already returns sorted and continuous chunks from the beginning (start=0)
        # We can directly use the chunks without additional sorting or continuity checks
        
        logger.info(f"Found {len(all_chunks)} Mooncake hit chunks from the beginning")
        
        # Step 2: For each hit chunk, retrieve from Mooncake backend using unified cache key
        total_retrieved_tokens = 0
        all_retrieved_data = []  # Store (start, end, retrieved_tokens, kv_cache_tensor, kv_cache_structure) for each chunk
        
        for start, end, cache_key in all_chunks:
            chunk_tokens = tokens[start:end]
            logger.debug(f"Retrieving hit chunk/prefix [{start}, {end}): {len(chunk_tokens)} tokens with unified cache key {cache_key}")
            
            retrieved_tokens, kv_cache_tensor, kv_cache_structure = self.storage_backend.retrieve(
                cache_key=cache_key, 
                tokens=chunk_tokens
            )
            
            if len(retrieved_tokens) == 0 or kv_cache_tensor is None:
                logger.error(f"Mooncake retrieve failed for chunk [{start}, {end}): no data found with unified cache key")
                return False
            
            logger.info(f"Mooncake retrieve successful for chunk [{start}, {end}): {len(retrieved_tokens)} tokens retrieved, KV cache tensor shape: {kv_cache_tensor.shape}")
            
            # Verify that retrieved tokens match the chunk tokens
            if retrieved_tokens != chunk_tokens:
                logger.error(f"Token mismatch for chunk [{start}, {end}): retrieved {len(retrieved_tokens)} tokens but expected {len(chunk_tokens)}")
                return False
            
            all_retrieved_data.append((start, end, retrieved_tokens, kv_cache_tensor, kv_cache_structure))
            total_retrieved_tokens += len(retrieved_tokens)
            logger.debug(f"Successfully retrieved chunk [{start}, {end}) using unified cache key")
        
        if total_retrieved_tokens == 0:
            logger.error("Mooncake retrieve failed: no chunks retrieved successfully")
            return False
        
        logger.info(f"Mooncake retrieve successful: {total_retrieved_tokens} tokens retrieved from {len(all_retrieved_data)} chunks")
        
        # Step 3: Check if we have GPUConnector
        if self.gpu_connector is None:
            logger.error(f"GPU connector is not available, cannot process Mooncake hits with GPU connector")
            return False
        
        # Step 4: Prepare batch upload parameters for GPUConnector
        blocked_kv_data_list = []
        slot_mapping_list = []
        starts = []
        ends = []
        
        for start, end, retrieved_tokens, kv_cache_tensor, kv_cache_structure in all_retrieved_data:
            num_tokens = end - start
            logger.debug(f"Preparing Mooncake chunk [{start}, {end}): {num_tokens} tokens for upload")
            
            # The kv_cache_tensor is already in the correct format from Mooncake storage
            # Shape: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
            # We can use it directly for batch upload
            
            # Verify the tensor shape
            if len(kv_cache_tensor.shape) != 6:
                logger.error(f"Mooncake KV cache tensor has unexpected shape: {kv_cache_tensor.shape}, expected 6D")
                return False
            
            # Create slot mapping for this chunk
            if slot_mapping is not None:
                chunk_slot_mapping = slot_mapping[start:end]
            else:
                return False
            
            blocked_kv_data_list.append(kv_cache_tensor)
            slot_mapping_list.append(chunk_slot_mapping)
            starts.append(0)  # Start from beginning of each chunk
            ends.append(num_tokens)  # Upload all tokens in chunk
        
        if not blocked_kv_data_list:
            logger.error("No Mooncake chunks prepared for upload")
            return False
        
        # Step 5: Perform batch upload using GPUConnector
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
                for i in range(start, min(end, len(tokens))):
                    ret_mask[i] = True
                total_copied_tokens += num_tokens
                logger.debug(f"Marked {num_tokens} tokens for Mooncake chunk [{start}, {end}) as retrieved")
            
            logger.info(f"Processed {len(all_retrieved_data)} Mooncake hits via batch upload, copied {total_copied_tokens} tokens")
            return total_copied_tokens > 0
            
        except Exception as e:
            logger.error(f"Batch upload failed for Mooncake chunks: {e}")
            return False

    def _process_local_gpu_vram_hits(
        self,
        local_hits: List[Tuple[int, int, CacheEngineKey, Any]],  # (start, end, cache_key, entry)
        kvcaches: List[torch.Tensor],
        slot_mapping: Optional[torch.Tensor],
        ret_mask: torch.Tensor
    ) -> bool:
        """
        Process local GPU VRAM hits (no transfer needed).
        Get VRAM unit from local segment and upload to vLLM kvcaches using GPUConnector.
        
        Args:
            local_hits: List of local GPU VRAM hits
            kvcaches: List of vLLM KV cache tensors
            slot_mapping: vLLM slot mapping tensor
            ret_mask: Return mask to mark retrieved tokens
            
        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing {len(local_hits)} local GPU VRAM hits (no transfer needed)")
        
        if not local_hits:
            logger.warning("No local hits to process")
            return False

      
        # Prepare batch upload parameters
        blocked_kv_data_list = []
        slot_mapping_list = []
        starts = []
        ends = []
        
        for start, end, cache_key, _ in local_hits:
            num_tokens = end - start
            logger.debug(f"Preparing chunk [{start}, {end}): {num_tokens} tokens in local for upload")
            
            # Get VRAM unit for this chunk
            vram_unit = self.segment_manager.get_vram_unit(cache_key)
            
            assert vram_unit is not None, f"Failed to get VRAM unit in local for chunk [{start}, {end})"

            # Get the tensor data from VRAM unit
            # The VRAM unit stores flattened tensor data, we need to restore it to original shape
            if not hasattr(vram_unit, 'original_shape') or vram_unit.original_shape is None:
                logger.error(f"VRAM unit for chunk [{start}, {end}) does not have original_shape metadata")
                assert False, f"VRAM unit for chunk [{start}, {end}) must have original_shape metadata"
            
            original_shape = vram_unit.original_shape
            vram_tensor = vram_unit.kv_cache_tensor
            
            # Restore the tensor to its original shape
            # Original shape should be: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
            
            restored_tensor = vram_tensor.view(original_shape)
            logger.debug(f"Restored tensor for chunk [{start}, {end}): shape={restored_tensor.shape}")
            
            # Create slot mapping for this chunk
            assert slot_mapping is not None, "slot_mapping must be provided for local GPU VRAM hits"

            chunk_slot_mapping = slot_mapping[start:end]
            
            blocked_kv_data_list.append(restored_tensor)
            slot_mapping_list.append(chunk_slot_mapping)
            starts.append(0)  # Start from beginning of each chunk
            ends.append(num_tokens)  # Upload all tokens in chunk
        
        assert blocked_kv_data_list is not None, "No VRAM units prepared for upload"
        
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
            
            logger.info(f"Successfully uploaded {len(blocked_kv_data_list)} chunks in local via batch upload")
            
            # Mark retrieved tokens in return mask
            total_copied_tokens = 0
            for idx, (start, end, cache_key, entry) in enumerate(local_hits):
                if idx >= len(blocked_kv_data_list):
                    continue
                
                num_tokens = end - start
                for i in range(start, min(end, len(ret_mask))):
                    ret_mask[i] = True
                total_copied_tokens += num_tokens
            
            logger.info(f"Processed {len(local_hits)} local GPU VRAM hits via batch upload, copied {total_copied_tokens} tokens")
            return total_copied_tokens > 0
            
        except Exception as e:
            logger.error(f"local hit Batch upload failed: {e}")
            return False

    def _process_remote_gpu_vram_hits(
        self,
        remote_hits: List[Tuple[int, int, CacheEngineKey, int, Any]],  # (start, end, cache_key, gpu_id, entry)
        kvcaches: List[torch.Tensor],
        slot_mapping: Optional[torch.Tensor],
        ret_mask: torch.Tensor
    ) -> bool:
        """
        Process remote GPU VRAM hits (needs transfer).
        Transfer remote data to local segment, then upload to vLLM kvcaches using GPUConnector.
        
        Args:
            remote_hits: List of remote GPU VRAM hits
            kvcaches: List of vLLM KV cache tensors
            slot_mapping: vLLM slot mapping tensor
            ret_mask: Return mask to mark retrieved tokens
            
        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing {len(remote_hits)} remote GPU VRAM hits (needs transfer)")
        
        if not remote_hits:
            logger.warning("No remote hits to process")
            return False
        
        # Prepare batch upload parameters
        blocked_kv_data_list = []
        slot_mapping_list = []
        starts = []
        ends = []
        
        for start, end, cache_key, source_gpu_id, entry in remote_hits:
            num_tokens = end - start
            logger.debug(f"Processing remote hit chunk [{start}, {end}): {num_tokens} tokens from GPU {source_gpu_id}")
            
            segment_address = None
            
            assert self.segment_manager is not None

            segment_id, segment_offset = self.segment_manager.allocate_in_segment(entry.tensor_size)

            if segment_id is None:
                logger.error(f"Failed to allocate segment space for {entry.tensor_size} bytes")
                return False

            segment_address = self.segment_manager.get_buffer_address(segment_id, segment_offset)
            logger.debug(f"Allocated segment space: {entry.tensor_size} bytes at address {hex(segment_address)} for remote GPU VRAM hit data")
        
            if segment_address is None:
                logger.error(f"Failed to allocate segment space for chunk [{start}, {end})")
                return False
            
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
                self.segment_manager.free_segment_space(segment_id, segment_offset, entry.tensor_size)
                return False
            
            logger.info(f"Successfully transferred {num_tokens} tokens from GPU {source_gpu_id} to segment space")
            
            # Register transferred entry in GPU VRAM pool
            self._register_transferred_entry(entry, self.metadata.worker_id, segment_address)
            
            # Create VRAM unit for transferred data at allocated space
            vram_unit = None
            if self.segment_manager is not None:
                vram_unit = self.segment_manager.create_vram_unit(
                    cache_key=cache_key,
                    token_ids=entry.token_ids,
                    segment_id=segment_id,
                    offset=segment_offset,
                    allocated_size=entry.tensor_size,
                    dtype=entry.tensor_dtype,
                    original_shape=entry.tensor_shape  # Use entry's tensor_shape as original_shape
                )
            
            if vram_unit is None:
                logger.error(f"Failed to create VRAM unit for transferred chunk [{start}, {end})")
                return False
            
            # Get the tensor data from VRAM unit
            # The VRAM unit stores flattened tensor data, we need to restore it to original shape
            if not hasattr(vram_unit, 'original_shape') or vram_unit.original_shape is None:
                logger.error(f"VRAM unit for chunk [{start}, {end}) does not have original_shape metadata")
                return False
            
            original_shape = vram_unit.original_shape
            vram_tensor = vram_unit.kv_cache_tensor
            
            # Restore the tensor to its original shape
            # Original shape should be: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
            if len(original_shape) != 6:
                logger.error(f"Expected 6D original shape for chunk [{start}, {end}), got: {original_shape}")
                return False
            
            restored_tensor = vram_tensor.view(original_shape)
            logger.debug(f"Restored tensor for chunk [{start}, {end}): shape={restored_tensor.shape}")
            
            # Create slot mapping for this chunk
            if slot_mapping is not None:
                chunk_slot_mapping = slot_mapping[start:end]
            else:
                logger.error("slot_mapping must be provided for remote GPU VRAM hits")
                return False
            
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
                    ret_mask[i] = True
                total_copied_tokens += num_tokens
                logger.info(f"Marked {num_tokens} tokens for chunk [{start}, {end}) as retrieved")
            
            logger.info(f"Processed {len(remote_hits)} remote GPU VRAM hits via batch upload, copied {total_copied_tokens} tokens")
            return total_copied_tokens > 0
            
        except Exception as e:
            logger.error(f"Batch upload failed for transferred chunks: {e}")
            return False


    @_lmcache_nvtx_annotate
    def store(
        self,
        tokens: List[int],
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> None:
        """
        store operation
        Saves vLLM KV cache to VRAM segment.
        
        Args:
            tokens: Input tokens
            mask: Optional mask for tokens (True = needs storing)
            **kwargs: Additional arguments including:
                - kvcaches: List of vLLM KV cache tensors
                - slot_mapping: vLLM slot mapping tensor
                - offset: Offset for partial storage
        """
        
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
        
        # Step 1: Process tokens using class member TokenDatabase
        
        assert hasattr(self, 'token_database') and self.token_database is not None
        
        # Process tokens to generate cache keys using class member token_database
        all_chunks = []
        # Convert torch.dtype to string using TORCH_DTYPE_TO_STR_DTYPE mapping
        from lmcache.utils import TORCH_DTYPE_TO_STR_DTYPE
        kv_dtype_str = TORCH_DTYPE_TO_STR_DTYPE.get(self.metadata.kv_dtype, "half")
        for start, end, cache_key in self.token_database.process_tokens(
            tokens=tokens, 
            mask=mask, 
            make_key=True,
            model_name="test_model",
            worker_id=self.metadata.worker_id,
            world_size=self.metadata.world_size,
            kv_dtype=kv_dtype_str
        ):
            all_chunks.append((start, end, cache_key))
            chunk_tokens = tokens[start:end]
        
        # for each chunk
        for start, end, cache_key in all_chunks:
            chunk_tokens = tokens[start:end]
            logger.debug(f"Processing chunk/prefix [{start}, {end}): {len(chunk_tokens)} tokens, key: {cache_key}")
            
            # check if already exists in GPU VRAM
            gpu_vram_exists = False
            if self.vram_metadata_client is not None:
                gpu_vram_exists = self.vram_metadata_client.contains(cache_key)
            
            
            # if exists in vram, skip storage
            if gpu_vram_exists :
                logger.debug(f"Chunk [{start}, {end}) already exists in GPU VRAM, skipping storage")
                continue
            
            # Step 2: Calculate KV cache size for this chunk and allocate VRAM segment
            chunk_kv_cache_size = 0
            
            # Get actual number of layers
            if isinstance(kvcaches, dict):
                num_layers = len(kvcaches)
                first_kv_cache = list(kvcaches.values())[0]
            else:
                num_layers = len(kvcaches)  # use list length as number of layers
                first_kv_cache = kvcaches[0]
            
            # vLLM KV Cache shape: (num_blocks, 2, block_size, num_kv_heads, head_size)
            # Dimension 0: num_blocks (number of blocks)
            # Dimension 1: 2 (Key and Value)
            # Dimension 2: block_size (number of tokens per block)
            # Dimension 3: num_kv_heads (number of KV heads)
            # Dimension 4: head_size (size of each head)
            
            if len(first_kv_cache.shape) != 5:
                logger.error(f"Unsupported KV cache shape: {first_kv_cache.shape}. Only vLLM FlashAttention KV Cache structure (5D) is supported.")
                raise ValueError(f"Unsupported KV cache shape: {first_kv_cache.shape}. Expected 5D vLLM FlashAttention KV Cache structure.")
            
            # Size per token: 2 * num_kv_heads * head_size * element_size
            num_kv_heads = first_kv_cache.shape[3]
            head_size = first_kv_cache.shape[4]
            element_size = first_kv_cache.element_size()
            
            # Calculate size per token
            token_size = 2 * num_kv_heads * head_size * element_size
            # Calculate total size for all layers
            chunk_kv_cache_size = len(chunk_tokens) * token_size * num_layers
            
            # Allocate VRAM segment for this chunk and create VRAM Unit
            vram_kvcache_unit = None
            segment_id = None
            segment_offset = None
            
            assert self.segment_manager is not None, "Segment Manager must be available for store operation"
            # Use new Segment Manager method to create VRAM Unit
            # Use the first KV cache for shape and dtype (first_kv_cache already defined above)
            kv_dtype = first_kv_cache.dtype
            
            # create correct KV cache shape for this chunk
            # block_size is fixed, we need to adjust num_blocks to accommodate all tokens in the chunk

            # Standard vLLM FlashAttention KV Cache structure
            num_blocks, kv_pairs, block_size, num_kv_heads, head_size = first_kv_cache.shape
            # Calculate required number of blocks
            chunk_blocks = (len(chunk_tokens) + block_size - 1) // block_size
            kv_shape = (chunk_blocks, kv_pairs, block_size, num_kv_heads, head_size)
            
            # Allocate segment space and get segment information
            segment_id, segment_offset = self.segment_manager.allocate_in_segment(chunk_kv_cache_size)

            if segment_id is None or segment_offset is None:
                logger.error(f"Failed to allocate segment space for chunk [{start}, {end}) of size {chunk_kv_cache_size} bytes")
                continue
            
            # Use new create_vram_unit method to create VRAM Unit (for 1D flattened data)
            # Need to pass original_shape parameter, here we don't know the original shape, so pass None
            vram_kvcache_unit = self.segment_manager.create_vram_unit(
                cache_key=cache_key,
                token_ids=chunk_tokens,
                segment_id=segment_id,
                offset=segment_offset,
                allocated_size=chunk_kv_cache_size,
                dtype=kv_dtype,
                original_shape=None  # Don't know original shape here, pass None
            )
            
            if vram_kvcache_unit is not None:
                logger.debug(f"Created VRAM Unit: {cache_key} at segment {segment_id}, "
                            f"offset {segment_offset}, size {chunk_kv_cache_size} bytes, "
                            f"shape {kv_shape}, tokens {len(chunk_tokens)}")
            else:
                logger.error(f"Failed to create VRAM Unit for chunk: {len(chunk_tokens)} tokens")
                self.segment_manager.free_segment_space(segment_id, segment_offset, chunk_kv_cache_size)
                continue
        
            # Step 3: Copy vLLM KV cache data to VRAM segment
            allocated_buffer_address = self.segment_manager.get_buffer_address(segment_id, segment_offset)
            if allocated_buffer_address is None:
                logger.error(f"Failed to get buffer address for segment {segment_id}, offset {segment_offset}")
                self.segment_manager.free_segment_space(segment_id, segment_offset, chunk_kv_cache_size)
                continue
            
            # Step 4: Use GPU connector to download blocked KV cache data directly
            copy_success = False
            combined_tensor = None
            
            assert self.gpu_connector is not None, "GPU connector must be available for store operation"
            
            # prepare slot mapping for this chunk
            assert slot_mapping is not None, "slot_mapping must be provided for store operation"

            chunk_slot_mapping = slot_mapping[start:end]
            
            # ensure chunk_slot_mapping is on the same device as kvcaches
            if kvcaches and len(kvcaches) > 0:
                device = kvcaches[0].device
                if not chunk_slot_mapping.is_cuda:
                    chunk_slot_mapping = chunk_slot_mapping.to(device)
            
            # initialize kvcaches pointer in connector
            if hasattr(self.gpu_connector, 'initialize_kvcaches_ptr'):
                self.gpu_connector.initialize_kvcaches_ptr(kvcaches=kvcaches)
            
            # download kvcaches
            # returned tensor shape: (num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size)
            combined_tensor = self.gpu_connector.download_blocked_kv(
                vllm_kvcaches=kvcaches,
                slot_mapping=chunk_slot_mapping,
                start=0,  # from beginning of chunk
                end=len(chunk_tokens),  # to the end of chunk
                dtype=kv_dtype,
                device=kvcaches[0].device  # to GPU device
            )
            
            if combined_tensor is not None:
                logger.debug(f"Successfully downloaded blocked KV cache data: shape={combined_tensor.shape}, "
                            f"device={combined_tensor.device}, dtype={combined_tensor.dtype}")          
            else:
                logger.error("download_blocked_kv returned None")
                self.segment_manager.free_segment_space(segment_id, segment_offset, chunk_kv_cache_size)
                continue
                      
            # if download success, create VRAM unit
            # flatten combined_tensor and copy to VRAM segment
            combined_size = combined_tensor.numel() * combined_tensor.element_size()
            
            flattened_tensor = combined_tensor.flatten()
            
            vram_flat_tensor = vram_kvcache_unit.kv_cache_tensor
            
            if vram_flat_tensor.numel() < flattened_tensor.numel():
                logger.error(f"VRAM unit tensor size insufficient: {vram_flat_tensor.numel()} < {flattened_tensor.numel()}")
                self.segment_manager.free_segment_space(segment_id, segment_offset, chunk_kv_cache_size)
                continue
            else:
                # copy flattened data
                vram_flat_tensor.copy_(flattened_tensor)
                # set original shape metadata
                vram_kvcache_unit.original_shape = combined_tensor.shape
                
                logger.debug(f"Copied flattened tensor to VRAM segment: {combined_size} bytes")
                           
            # Step 5: Register to GPU VRAM pool using unified cache key
            # combined_tensor: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
            combined_shape = combined_tensor.shape
            self._register_to_gpu_vram_pool(
                cache_key=cache_key, 
                tokens=chunk_tokens,
                gpu_vram_address=allocated_buffer_address, 
                segment_id=segment_id,
                kv_shape=combined_shape,  # original shape
                kv_dtype=kv_dtype,
                total_size=chunk_kv_cache_size  # size of this chunk
            )
            
            # Step 6: Store to Mooncake backend
            # combined_tensor: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
            actual_num_layers = combined_tensor.shape[0]
            kv_cache_structure = {
                "num_layers": actual_num_layers,
                "layer_shapes": [combined_tensor[i].shape for i in range(actual_num_layers)],
                "layer_dtypes": [str(combined_tensor[i].dtype) for i in range(actual_num_layers)],
                "layer_sizes": [combined_tensor[i].numel() * combined_tensor[i].element_size() for i in range(actual_num_layers)],  # 每层的大小
                "vllm_layout": "blocked",
                "block_size": getattr(kvcaches[0], 'block_size', 16) if hasattr(kvcaches[0], 'block_size') else 16,
                "num_heads": kv_shape[-2] if len(kv_shape) >= 4 else 32,
                "head_size": kv_shape[-1] if len(kv_shape) >= 4 else 128,
                "chunk_tokens": len(chunk_tokens), 
                "is_chunk": True,  
                "cache_key": {  
                    "chunk_hash": cache_key.chunk_hash,
                    "model_name": cache_key.model_name,
                    "worker_id": cache_key.worker_id,
                    "world_size": cache_key.world_size
                }
            }
            
            # split combined_tensor into per-layer tensors
            layer_tensors = [combined_tensor[i] for i in range(actual_num_layers)]
            
            unified_kv_cache_structure = kv_cache_structure.copy()
            unified_kv_cache_structure["all_layers_data"] = True
            unified_kv_cache_structure["total_layers"] = actual_num_layers
            
            store_success = self.storage_backend.store(
                tokens=chunk_tokens,
                kvcaches=layer_tensors, 
                kv_cache_structure=unified_kv_cache_structure,
                cache_key=cache_key 
            )
            
            if store_success:
                logger.debug(f"Successfully stored KV cache for chunk [{start}, {end}): {len(chunk_tokens)} tokens with unified cache key, {actual_num_layers} layers")
            else:
                logger.error(f"Failed to store KV cache for chunk [{start}, {end})")
            
            # free combined tensor to release memory
            del combined_tensor
            import gc
            gc.collect()
            logger.debug(f"Released CPU tensor memory for chunk {start}:{end}")

    
        logger.info(f"Store operation completed for {num_to_store_tokens} tokens")

    def _register_to_gpu_vram_pool(
        self, 
        cache_key: CacheEngineKey,  # Add cache_key parameter
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
        
        # Use the passed parameters, which have been correctly calculated in the store function as chunk-specific values
        
        # Validate required parameters
        if kv_shape is None:
            logger.error("KV cache shape is required for registration")
            return
        
        if kv_dtype is None:
            logger.error("KV cache dtype is required for registration")
            return
        
        if total_size is None:
            logger.error("Total size is required for registration")
            return
        
        # Build KV cache structure information for this chunk
        # In the store function, we pass combined_tensor.shape (6D shape)
        # combined_tensor.shape: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
        # So kv_shape is already a 6D shape, no need to add layer dimension
        combined_shape = kv_shape
        
        # Extract number of layers from combined_shape
        num_layers = combined_shape[0] if len(combined_shape) == 6 else 1
        
        # Build complete KV cache structure information
        # combined_shape is 6D shape: [num_layers, chunk_blocks, 2, block_size, num_kv_heads, head_size]
        # We need to extract correct parameters from combined_shape
        if len(combined_shape) == 6:
            num_layers = combined_shape[0]
            chunk_blocks = combined_shape[1]
            kv_pairs = combined_shape[2]
            block_size = combined_shape[3]
            num_heads = combined_shape[4]
            head_size = combined_shape[5]
        else:
            # If not 6D shape, use default values
            num_layers = 1
            chunk_blocks = 1
            kv_pairs = 2
            block_size = 16
            num_heads = 32
            head_size = 128
        
        kv_cache_structure = {
            "num_layers": num_layers,  # Get actual number of layers from combined_shape
            "layer_shapes": [combined_shape],  # Use combined shape
            "layer_dtypes": [str(kv_dtype)],
            "layer_sizes": [total_size],  # Size of the entire chunk
            "vllm_layout": "blocked",  # vLLM uses blocked memory layout
            "block_size": block_size,  # Get block_size from combined_shape
            "num_heads": num_heads,  # Get num_heads from combined_shape
            "head_size": head_size,  # Get head_size from combined_shape
            "chunk_tokens": len(token_ids),  # Add chunk token count information
            "is_chunk": True,  # Mark this as chunk data
            "combined_tensor": True  # Mark this as combined tensor
        }
        logger.debug(f"Extracted chunk KV cache structure: {kv_cache_structure}")
        
        # Use provided GPU VRAM address
        buffer_pointer = gpu_vram_address
        if buffer_pointer is None:
            logger.warning("No GPU VRAM address provided for KV cache registration")
            return
        
        # Register in GPU VRAM pool with chunk-specific vLLM KV cache metadata
        # Use TestTokenDatabase generated cache_key to ensure key consistency
        logger.info(f"DEBUG: Registering to GPU VRAM pool - Key: {cache_key}, chunk_hash: {cache_key.chunk_hash}, "
                   f"worker_id: {self.metadata.worker_id}, tokens: {len(token_ids)}, "
                   f"shape: {combined_shape}, dtype: {kv_dtype}")
        
        success = self.vram_metadata_client.register_kvcache(
            cache_key=cache_key,  # Pass TestTokenDatabase generated CacheEngineKey
            token_ids=token_ids,
            gpu_id=self.metadata.worker_id,
            tensor_shape=combined_shape,  # Use combined shape
            tensor_dtype=kv_dtype,  # Use chunk-specific dtype
            tensor_size=total_size,  # Use chunk-specific size
            buffer_pointer=buffer_pointer,
            segment_id=segment_id,
            resident_hostname=self.config.get_extra_config_value("local_hostname_TE", "localhost"),
            kv_cache_structure=kv_cache_structure  # Use chunk-specific KV cache structure
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
        
        # 1. GPU VRAM pool lookup (priority) - use token database for chunk processing
        gpu_vram_hit_tokens = 0
        if self.vram_metadata_client is not None:
            # Generate all possible chunks and prefix chunks
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
            
            # Directly pass all_chunks to vram_metadata_client.lookup_prefix, let vram manager find the most suitable chunk based on all chunks
            # No longer call each chunk individually, but process in batch
            gpu_vram_hit_tokens, chunk_info_list = self.vram_metadata_client.lookup_prefix(
                token_ids=token_list,
                max_tokens=len(token_list),
                current_gpu_id=self.metadata.worker_id,
                all_chunks=all_chunks  # Pass all chunks information
            )
            
            # Ensure only continuous hit tokens from the beginning are returned
            if gpu_vram_hit_tokens > 0 and chunk_info_list:
                # Check if hit tokens start from the beginning
                # If gpu_vram_hit_tokens doesn't equal any chunk's end value, it's not continuous hit from the beginning
                # Here we assume vram_metadata_client.lookup_prefix has already returned continuous hits from the beginning
                
                # Extract information from chunk_info_list for logging
                # chunk_info_list contains detailed information for each matched chunk: ((start, end), gpu_id, needs_transfer)
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
        
            # 2. If GPU VRAM has no hit, then check Mooncake storage backend - use token database for chunk processing
        storage_hit_tokens = 0
        if gpu_vram_hit_tokens == 0 and self.storage_backend is not None:
            # Only check Mooncake when GPU VRAM has no hit, and storage_backend is not None
            # Use token database to generate chunks to check maximum matching chunks
            # Directly pass all_chunks to Mooncake storage backend's lookup function
            
            # First sort chunks
            all_chunks.sort(key=lambda x: x[0])
            
            # Only check continuous chunks from the beginning
            continuous_hit_tokens = 0
            expected_start = 0
            
            for start, end, cache_key in all_chunks:
                if start == expected_start:
                    # Check if this chunk exists in Mooncake
                    # Note: need to pass the chunk's range in original tokens, not from 0
                    # So should pass [(start, end, cache_key)] not [(0, len(chunk_tokens), cache_key)]
                    chunk_tokens = token_list[start:end]
                    chunk_hit_tokens, chunk_info_list = self.storage_backend.lookup(token_list, [(start, end, cache_key)])
                    
                    if chunk_hit_tokens == (end - start):
                        # Entire chunk hit
                        continuous_hit_tokens += chunk_hit_tokens
                        expected_start = end
                        logger.debug(f"Mooncake continuous hit chunk [{start}, {end}): {chunk_hit_tokens} tokens")
                    else:
                        # Chunk not fully hit, stop checking
                        logger.info(f"Breaking at chunk [{start}, {end}): expected {end-start} tokens, got {chunk_hit_tokens} tokens")
                        break
                else:
                    # Encountered gap, stop checking
                    logger.info(f"Breaking at gap: expected start={expected_start}, got start={start}")
                    break
            
            storage_hit_tokens = continuous_hit_tokens
            
            if storage_hit_tokens > 0:
                logger.info(f"Mooncake storage lookup hit: {storage_hit_tokens} continuous tokens from the beginning")
            else:
                logger.info("No Mooncake storage hits found for continuous chunks from the beginning")
        elif gpu_vram_hit_tokens == 0 and self.storage_backend is None:
            logger.debug("Storage backend is None, skipping Mooncake lookup for scheduler role")
        
        # total_hit_tokens should be either gpu_vram_hit_tokens or storage_hit_tokens, not the sum of both
        # Because we only check Mooncake when GPU VRAM has no hit
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
        
        # Add storage backend statistics (if exists)
        if self.storage_backend is not None:
            stats["storage_backend_stats"] = self.storage_backend.get_stats()
        else:
            stats["storage_backend_stats"] = {"status": "disabled", "reason": "scheduler_role"}
        
        # Add GPU VRAM pool statistics
        if self.vram_metadata_client is not None:
            gpu_vram_stats = self.vram_metadata_client.get_stats()
            stats["gpu_vram_pool_stats"] = gpu_vram_stats
            
            # Calculate GPU VRAM hit rate
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

