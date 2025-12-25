# SPDX-License-Identifier: Apache-2.0
"""
Test Cache Engine Connector for vLLM

This connector integrates the TestCacheEngine system with vLLM's KV connector interface,
providing enhanced features like GPU VRAM pool, cross-GPU transfers, and Mooncake storage.
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union
import os

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.utils import CacheEngineKey, cdiv
from lmcache.test.test_config import VCacheConfig
from lmcache.test.test_cache_engine_system import TestCacheEngine, MockGPUConnector
from lmcache.test.blocked_kv_paged_connector import BlockedKVPagedMemConnector
from lmcache.test.test_mooncake_lookup_client import TestMooncakeLookupClient

# vLLM imports
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger as vllm_init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    """Extract request-specific configurations from sampling parameters.
    
    Args:
        sampling_params: vLLM sampling parameters
        
    Returns:
        Dictionary of request configurations or None if not found
    """
    request_configs = None
    if sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                # Test system uses test_cache_engine prefix instead of lmcache
                if k.startswith("test_cache_engine."):
                    if request_configs is None:
                        request_configs = {}
                    # Store with the original key
                    request_configs[k] = v
    return request_configs


logger = init_logger(__name__)
vllm_logger = vllm_init_logger(__name__)


@dataclass
class LoadSpec:
    """Load specification for partial cache hits."""
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in TestCacheEngine
    test_cache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    """Save specification for partial cache saves."""
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class RequestTracker:
    """Track request state for TestCacheEngine connector."""
    # Request id
    req_id: str
    # Total prompt token length
    prompt_len: int
    # The token ids that has been scheduled so far
    token_ids: list[int]
    # The block ids that has been allocated so far
    allocated_block_ids: list[int]
    # The number of tokens that has been saved
    num_saved_tokens: int = 0
    # Load specification for cache hits
    load_spec: Optional[LoadSpec] = None
    # Whether the request is in decode phase
    is_decode_phase: bool = False
    # Whether the request cache should be saved
    skip_save: bool = False
    # Request-specific configurations
    request_configs: Optional[dict] = None

    @staticmethod
    def from_new_request(
        request: "Request",
        num_tokens_to_compute: int,
        test_cache_cached_tokens: int,
        skip_save: bool,
    ) -> "RequestTracker":
        """Create the request tracker from a new request."""
        # Handle different block_ids formats
        if not isinstance(request.block_ids[0], list):
            unfolded_block_ids = request.block_ids.copy()
        else:
            # Only one KVCacheGroup is supported for now
            unfolded_block_ids = request.block_ids[0].copy()

        # Extract request configurations
        request_configs = extract_request_configs(request.sampling_params)

        return RequestTracker(
            req_id=request.req_id,
            prompt_len=len(request.prompt_token_ids),
            token_ids=request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=test_cache_cached_tokens,
            skip_save=skip_save,
            request_configs=request_configs,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
    ) -> None:
        """Update the request tracker when a running request is scheduled again."""
        self.token_ids.extend(new_token_ids)

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            pass
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        
        self.allocated_block_ids.extend(new_block_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    """Request metadata for TestCacheEngine connector."""
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]
    # Slot mapping
    slot_mapping: Optional[torch.Tensor] = None
    # Load specification
    load_spec: Optional[LoadSpec] = None
    # Save specification
    save_spec: Optional[SaveSpec] = None
    # Allocated block IDs
    allocated_block_ids: Optional[list[int]] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        test_cache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker."""
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len == tracker.prompt_len:
            is_last_prefill = True

        # Calculate save specification
        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, test_cache_chunk_size) * test_cache_chunk_size
        )

        # Check if request_configs has test_cache_engine.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("test_cache_engine.skip_save", False)

        skip_save = (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // test_cache_chunk_size * test_cache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        # Update saved tokens count
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Prepare token IDs
        token_ids = input_token_ids[:num_tokens_to_save]

        # Calculate slot mapping
        num_blocks = len(tracker.allocated_block_ids)
        if len(token_ids) > num_blocks * block_size:
            logger.warning(
                f"Token count ({len(token_ids)}) exceeds block capacity ({num_blocks * block_size})"
            )

        slot_mapping = None
        if tracker.allocated_block_ids:
            block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
            block_offsets = torch.arange(0, block_size, dtype=torch.long)
            slot_mapping = (
                block_offsets.reshape((1, block_size))
                + block_ids.reshape((num_blocks, 1)) * block_size
            )
            slot_mapping = slot_mapping.flatten()[: len(token_ids)]

        # Only load if in can_load state
        if load_spec is not None and not load_spec.can_load:
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            load_spec=load_spec,
            save_spec=save_spec,
            allocated_block_ids=tracker.allocated_block_ids,
        )


class TestCacheEngineConnectorMetadata(KVConnectorMetadata):
    """TestCacheEngine connector metadata for communication between scheduler and worker."""
    
    def __init__(self):
        # List of request metadata for TestCacheEngine
        self.requests: list[ReqMeta] = []
    
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.
        
        Args:
            req_meta: Request metadata object
        """
        self.requests.append(req_meta)


class TestCacheEngineConnectorV1(KVConnectorBase_V1):
    """
    vLLM Connector using TestCacheEngine system.
    
    This connector provides enhanced KV cache functionality including:
    - GPU VRAM pool management
    - Cross-GPU transfers using Mooncake transfer engine
    - GPU VRAM segment management
    - Mooncake storage backend integration
    - Hierarchical lookup (GPU VRAM + Mooncake fallback)
    """
    
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
        kv_cache_config: Any = None,
    ):
        """
        Initialize the TestCacheEngine connector.
        
        Args:
            vllm_config: vLLM configuration
            role: Connector role (scheduler or worker)
            kv_cache_config: Optional KV cache configuration
        """
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        
        self.vllm_config = vllm_config
        self._parent = parent
        # Create TestCacheEngine configuration
        self.test_config = self._create_test_config(vllm_config)
        self.test_metadata = self._create_test_metadata(vllm_config, role)
        
        # Create GPU connector based on role
        # Scheduler only needs lookup capabilities, not actual GPU data transfer
        # Worker needs BlockedKVPagedMemConnector for actual GPU data transfer
        if role == KVConnectorRole.SCHEDULER:
            # Scheduler doesn't need GPU connector for data transfer, only for lookup
            # Pass None to TestCacheEngine for scheduler role
            gpu_connector = None
            logger.info(f"Scheduler role: No GPU connector needed for lookup operations")
        else:
            # Worker needs actual GPU connector for data transfer
            # 从metadata中获取参数
            num_layers = self.test_metadata.kv_shape[0] if len(self.test_metadata.kv_shape) >= 5 else 32
            block_size = vllm_config.cache_config.block_size
            num_kv_heads = self.test_metadata.kv_shape[-2] if len(self.test_metadata.kv_shape) >= 4 else 32
            head_size = self.test_metadata.kv_shape[-1] if len(self.test_metadata.kv_shape) >= 4 else 128
            
            try:
                # 确保device参数是torch.device对象，而不是字符串
                device_str = f"cuda:{self.test_metadata.worker_id}" if self.test_metadata.worker_id is not None else "cuda:0"
                device_obj = torch.device(device_str)
                
                gpu_connector = BlockedKVPagedMemConnector(
                    num_layers=num_layers,
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    use_gpu=True,
                    dtype=self.test_metadata.kv_dtype,
                    device=device_obj
                )
                logger.info(f"Worker role: BlockedKVPagedMemConnector initialized: layers={num_layers}, block_size={block_size}, heads={num_kv_heads}, head_size={head_size}, device={device_str}")
            except Exception as e:
                logger.error(f"Failed to initialize BlockedKVPagedMemConnector: {e}, falling back to MockGPUConnector")
                gpu_connector = MockGPUConnector()
        
        # Initialize TestCacheEngine
        self.test_engine = TestCacheEngine(
            config=self.test_config,
            metadata=self.test_metadata,
            gpu_connector=gpu_connector
        )
        
        # Initialize MooncakeLookupClient for scheduler role
        self.mooncake_lookup_client = None
        if role == KVConnectorRole.SCHEDULER:
            try:
                # Get master address from config
                master_addr = self.test_config.get_extra_config_value("master_server_address", "127.0.0.1:50051")
                self.mooncake_lookup_client = TestMooncakeLookupClient(vllm_config, master_addr)
                logger.info(f"MooncakeLookupClient initialized for scheduler role with master address: {master_addr}")
            except ImportError as e:
                logger.warning(f"Failed to import MooncakeLookupClient: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize MooncakeLookupClient: {e}")
        
        # Initialize KV caches dictionary
        self.kv_caches: dict[str, torch.Tensor] = {}
        
        # Request tracking
        self._request_trackers: dict[str, RequestTracker] = {}
        self._unfinished_requests: dict[str, "Request"] = {}
        
        # Load specifications for partial hits
        self.load_specs: dict[str, LoadSpec] = {}
        
        # Configuration flags
        self._discard_partial_chunks = True
        self._save_decode_cache = False
        self._force_skip_save = False
        
        # Statistics
        self.stats = {
            "total_lookups": 0,
            "total_retrieves": 0,
            "total_stores": 0,
            "hits": 0,
            "misses": 0,
            "cross_gpu_transfers": 0,
            "partial_hits": 0,
            "full_hits": 0,
            "load_errors": 0,
            "save_errors": 0,
            "transfer_errors": 0
        }
        
        # Block size for slot mapping
        self._block_size = vllm_config.cache_config.block_size
        self._chunk_size = 256  # Default chunk size for TestCacheEngine
        
        # Events tracking
        self._events: list[Any] = []
        
        logger.info(f"TestCacheEngineConnector initialized for role {role}")
    
    def _create_test_config(self, vllm_config: "VllmConfig") -> VCacheConfig:
        """Create TestCacheEngine configuration from vLLM config."""
        # Extract configuration from vLLM extra config
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        )
        
        # Check if config file path is provided in vLLM extra config
        config_file_path = kv_connector_extra_config.get("test_engine_config_file")
        
        if config_file_path and os.path.exists(config_file_path):
            # Load configuration from YAML file
            config = VCacheConfig.from_file(config_file_path)
            logger.info(f"Loaded TestCacheEngine configuration from: {config_file_path}")
        else:
            # Use default configuration
            config = VCacheConfig.from_defaults()
            logger.info("Using default TestCacheEngine configuration")
        
        # Set connector_role based on vLLM role
        if self._role == KVConnectorRole.SCHEDULER:
            config.connector_role = "scheduler"
            # Scheduler needs vram_metadata_client for lookup operations
            config.enable_gpu_vram_pool = True
            logger.info("Set connector_role to 'scheduler' with enable_gpu_vram_pool=True for lookup")
        else:
            config.connector_role = "worker"
            logger.info("Set connector_role to 'worker'")
        
        # No need to manually optimize other configuration - TestCacheEngine will handle it based on connector_role
        logger.info(f"Configuration created with connector_role={config.connector_role}")
        
        return config
    
    def _create_test_metadata(self, vllm_config: "VllmConfig", role: KVConnectorRole) -> LMCacheEngineMetadata:
        """Create TestCacheEngine metadata from vLLM config."""
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config
        
        # Get KV cache dtype
        kv_dtype = torch.float16  # Default
        if hasattr(cache_config, 'cache_dtype'):
            cache_dtype = cache_config.cache_dtype
            logger.info(f"Raw cache_dtype from config: {cache_dtype}, type: {type(cache_dtype)}")
            # 确保dtype是torch.dtype对象，而不是字符串
            if isinstance(cache_dtype, str):
                # 将字符串转换为torch.dtype对象
                if cache_dtype == "float16" or cache_dtype == "half":
                    kv_dtype = torch.float16
                elif cache_dtype == "float32" or cache_dtype == "float":
                    kv_dtype = torch.float32
                elif cache_dtype == "bfloat16":
                    kv_dtype = torch.bfloat16
                else:
                    logger.warning(f"Unsupported dtype string: {cache_dtype}, using float16 as default")
                    kv_dtype = torch.float16
            else:
                kv_dtype = cache_dtype
        logger.info(f"Final kv_dtype: {kv_dtype}, type: {type(kv_dtype)}")
        
        # Calculate KV shape
        num_layer = model_config.get_num_layers(parallel_config)
        num_kv_head = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        chunk_size = 256  # Default chunk size
        
        kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)
        
        # Determine role string
        role_str = "scheduler" if role == KVConnectorRole.SCHEDULER else "worker"
        
        return LMCacheEngineMetadata(
            model_name=model_config.model,
            world_size=parallel_config.world_size,
            worker_id=parallel_config.rank,
            fmt="vllm",
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=False,
            role=role_str
        )
    
    
    # ==============================
    # Worker-side methods
    # ==============================
    
    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading KV cache from TestCacheEngine to vLLM's paged buffer.
        Implements batch mode logic (all layers at once) similar to LMCache v1 adapter.
        
        Args:
            forward_context: Forward context containing attention layers
            **kwargs: Additional arguments for load operation
        """
        logger.info("Starting KV cache load from TestCacheEngine in batch mode (all layers at once)")
        
        # Initialize KV caches if not already done (first load check)
        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)
            logger.info(f"First load: Initialized {len(self.kv_caches)} KV cache connections")
        else:
            logger.debug(f"Subsequent load: Using existing {len(self.kv_caches)} KV cache connections")
        
        # Get connector metadata - follow vllm v1 adapter pattern
        # Use _get_connector_metadata() which will handle the case where metadata might be None
        try:
            metadata = self._parent._get_connector_metadata()
            logger.info(f"Got connector metadata with {len(metadata.requests) if hasattr(metadata, 'requests') else 0} requests")
        except AssertionError:
            # Parent class _get_connector_metadata() asserts metadata is not None
            # If we get here, it means no metadata was bound yet
            logger.warning("No connector metadata available for load operation")
            return
        
        # Check if metadata has requests attribute (should be TestCacheEngineConnectorMetadata)
        if not hasattr(metadata, 'requests'):
            logger.warning("Connector metadata doesn't have 'requests' attribute")
            return
        
        # Get KV caches as references to vLLM's memory regions
        kvcaches = list(self.kv_caches.values())
        
        # Process each request in metadata - implement batch mode logic
        for request in metadata.requests:
            # Extract request information from ReqMeta object
            token_ids = request.token_ids
            request_id = request.req_id
            load_spec = request.load_spec
            slot_mapping = request.slot_mapping
            
            if not token_ids:
                logger.debug(f"Request {request_id}: No token IDs, skipping load")
                continue
            
            # Check if this request has cache hits (load_spec indicates cache hits)
            if not load_spec:
                logger.debug(f"Request {request_id}: No cache hits, skipping load")
                continue
            
            # Extract load specification for partial hit handling
            vllm_cached_tokens = load_spec.vllm_cached_tokens
            test_cache_cached_tokens = load_spec.test_cache_cached_tokens
            
            # Calculate the actual tokens that need to be loaded
            tokens_to_load = test_cache_cached_tokens - vllm_cached_tokens
            if tokens_to_load <= 0:
                logger.debug(f"Request {request_id}: No tokens to load (vllm_cached={vllm_cached_tokens}, test_cache_cached={test_cache_cached_tokens})")
                continue
            
            # Create token mask for partial loading - follow LMCache connector implementation
            # True means the token needs to be loaded from TestCacheEngine
            token_mask = torch.ones(len(token_ids), dtype=torch.bool)
            
            # Calculate masked token count - align to chunk boundaries like LMCache connector
            # Use cdiv for proper alignment to chunk boundaries
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._chunk_size * self._chunk_size
            )
            token_mask[:masked_token_count] = False  # vLLM already has these tokens
            
            # Use provided slot mapping - should not be None if ReqMeta was created
            if slot_mapping is None:
                logger.warning(f"Request {request_id}: No slot mapping available, skipping load")
                continue
            
            logger.info(
                f"Request {request_id}: Loading {tokens_to_load}/{len(token_ids)} tokens "
                f"(vllm_cached={vllm_cached_tokens}, test_cache_cached={test_cache_cached_tokens}, "
                f"masked_tokens={masked_token_count})"
            )
            
            try:
                # Use TestCacheEngine for retrieval with partial hit parameters
                # Follow LMCache connector batch mode pattern: pass only cached tokens with aligned mask
                ret_mask = self.test_engine.retrieve(
                    tokens=token_ids[:test_cache_cached_tokens],  # Only cached tokens
                    mask=token_mask[:test_cache_cached_tokens],   # Mask for partial loading
                    kvcaches=kvcaches,                            # vLLM KV cache references
                    slot_mapping=slot_mapping[:test_cache_cached_tokens] if slot_mapping else None,
                    skip_contains_check=True                      # Skip contains check since we already know cache hits
                )
                
                num_retrieved_tokens = ret_mask.sum().item() if ret_mask is not None else 0
                logger.info(
                    f"Request {request_id}: Retrieved {num_retrieved_tokens}/{tokens_to_load} tokens "
                    f"(partial hit: {test_cache_cached_tokens < len(token_ids)})"
                )
                
                self.stats["total_retrieves"] += 1
                if num_retrieved_tokens > 0:
                    self.stats["hits"] += 1
                else:
                    self.stats["misses"] += 1
                    
                # Record partial hit statistics
                if test_cache_cached_tokens < len(token_ids):
                    self.stats["partial_hits"] += 1
                else:
                    self.stats["full_hits"] += 1
                    
            except Exception as e:
                logger.error(f"Error loading KV cache for request {request_id}: {e}")
                self.stats["load_errors"] += 1
                # Continue with other requests even if one fails
    
    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Wait for specific layer KV cache to be loaded.
        In batch mode (all layers at once), this is a no-op operation.
        
        Args:
            layer_name: Name of the layer to wait for
        """
        # In batch mode, all layers are loaded at once in start_load_kv
        # No layer-wise waiting is needed, similar to LMCache v1 adapter
        logger.debug(f"Batch mode: layer {layer_name} load completed (no-op in batch mode)")
    
    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Save KV cache from vLLM to TestCacheEngine.
        In batch mode (all layers at once), this method should return early like in LMCache v1 adapter.
        
        Args:
            layer_name: Name of the layer being saved (ignored in batch mode)
            kv_layer: KV cache tensor for the layer (not directly used, for interface compatibility)
            attn_metadata: Attention metadata
            **kwargs: Additional arguments for save operation
        """
        # In batch mode, save_kv_layer should return early (no per-layer operations)
        # Actual saving happens in wait_for_save method
        logger.debug(f"save_kv_layer called for layer {layer_name} in batch mode - returning early")
        return
    
    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """
        Wait for all save operations to complete.
        In batch mode, this is where actual saving happens (all layers at once).
        """
        logger.info("Starting KV cache save to TestCacheEngine in batch mode (all layers at once)")
        
        # Ensure KV caches are initialized
        if len(self.kv_caches) == 0:
            logger.warning("KV caches not initialized, cannot save")
            return
        
        # Get connector metadata
        try:
            metadata = self._parent._get_connector_metadata()
        except AssertionError:
            logger.warning("No connector metadata available for save operation")
            return
        
        # Get KV caches as references to vLLM's memory regions
        kvcaches = list(self.kv_caches.values())
        
        # Process each request in metadata for saving
        for request in metadata.requests:
            # Extract request information from ReqMeta object
            token_ids = request.token_ids
            request_id = request.req_id
            save_spec = request.save_spec
            slot_mapping = request.slot_mapping
            
            if not token_ids:
                logger.debug(f"Request {request_id}: No token IDs, skipping save")
                continue
            
            # Check if this request should be saved
            if not save_spec or not save_spec.can_save:
                logger.debug(f"Request {request_id}: Not scheduled for save, skipping")
                continue
            
            # Calculate tokens to save based on skip_leading_tokens
            skip_leading_tokens = save_spec.skip_leading_tokens
            if skip_leading_tokens >= len(token_ids):
                logger.debug(f"Request {request_id}: All tokens already saved, skipping")
                continue
            
            # Align to chunk boundaries
            skip_leading_tokens = (
                skip_leading_tokens
                // self._chunk_size * self._chunk_size
            )
            
            # Create store mask
            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False
            
            logger.info(
                f"Request {request_id}: Saving {len(token_ids) - skip_leading_tokens}/{len(token_ids)} tokens "
                f"(skip_leading_tokens={skip_leading_tokens})"
            )
            
            try:
                # Use TestCacheEngine for storage
                self.test_engine.store(
                    tokens=token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens
                )
                
                self.stats["total_stores"] += 1
                logger.info(f"Request {request_id}: Saved {len(token_ids) - skip_leading_tokens} tokens")
                
            except Exception as e:
                logger.error(f"Error saving KV cache for request {request_id}: {e}")
                self.stats["save_errors"] += 1
                # Continue with other requests even if one fails
        
        logger.info("All save operations completed in batch mode")
    
    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Get requests that have finished asynchronous transfer.
        
        Args:
            finished_req_ids: Set of finished request IDs
            
        Returns:
            Tuple of (sending/saving IDs, receiving/loading IDs)
        """
        # TestCacheEngine doesn't use asynchronous transfers
        return None, None
    
    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get block IDs that failed to load.
        
        Returns:
            Set of block IDs with load errors
        """
        # TestCacheEngine doesn't track individual block errors
        return set()
    
    # ==============================
    # Scheduler-side methods
    # ==============================
    
    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Get number of new tokens that can be loaded from external KV cache.
        Implements hierarchical lookup logic similar to LMCache v1 adapter:
        1. First check GPU VRAM pool for cross-GPU cache hits
        2. Then check MooncakeStore for remote cache hits
        
        Args:
            request: Request object
            num_computed_tokens: Number of locally computed tokens
            
        Returns:
            An optional number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
            If None, it means that the connector needs more time to
            determine the number of matched tokens, and the scheduler
            should query for this request again later.
        """
        self.stats["total_lookups"] += 1
        
        token_ids = request.prompt_token_ids
        
        # Step 1: Use TestCacheEngine for GPU VRAM pool lookup
        gpu_vram_hit_tokens = self.test_engine.lookup(token_ids)
        
        # Step 2: If no GPU VRAM hits, try Mooncake lookup (only for scheduler role)
        mooncake_hit_tokens = 0
        if gpu_vram_hit_tokens == 0 and self._role == KVConnectorRole.SCHEDULER and self.mooncake_lookup_client is not None:
            try:
                mooncake_hit_tokens = self.mooncake_lookup_client.lookup(token_ids, lookup_id=request.request_id)
                logger.info(f"Request {request.request_id}: Mooncake lookup returned {mooncake_hit_tokens} hit tokens")
            except Exception as e:
                logger.warning(f"Request {request.request_id}: Mooncake lookup failed: {e}")
        
        # Calculate total hit tokens
        total_hit_tokens = gpu_vram_hit_tokens if gpu_vram_hit_tokens > 0 else mooncake_hit_tokens
        
        if total_hit_tokens == 0:
            logger.debug(
                f"Request {request.request_id}: No external cache hits found "
                f"({len(token_ids)} tokens, GPU VRAM hits: {gpu_vram_hit_tokens}, Mooncake hits: {mooncake_hit_tokens})"
            )
            # !!!!if no tokens hit, return 0. if return none, vLLM will keep querying, a cyclic lock will happen
            return 0
        
        # Calculate number of tokens to allocate
        need_to_allocate = total_hit_tokens - num_computed_tokens
        
        # Full prompt hit case - need to recompute last token
        if total_hit_tokens == request.num_tokens:
            need_to_allocate -= 1
            self.stats["full_hits"] += 1
        else:
            self.stats["partial_hits"] += 1
        
        logger.info(
            f"Request {request.request_id}: Total tokens {request.num_tokens}, "
            f"GPU VRAM hits: {gpu_vram_hit_tokens}, Mooncake hits: {mooncake_hit_tokens}, "
            f"Total hit: {total_hit_tokens}, Need to load: {need_to_allocate}, "
            f"Partial hit: {total_hit_tokens < request.num_tokens}"
        )
        
        # Store LoadSpec for partial hit handling
        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            test_cache_cached_tokens=total_hit_tokens,
            can_load=False
        )
        
        if need_to_allocate <= 0:
            return 0
        
        return need_to_allocate
    
    @_lmcache_nvtx_annotate
    def update_state_after_alloc(
        self, request: "Request", num_external_tokens: int
    ):
        """
        Update connector state after block allocation.
        
        Args:
            request: Request object
            num_external_tokens: Number of external tokens
        """
        # Track unfinished requests
        self._unfinished_requests[request.request_id] = request
        
        # Update load spec if this request has cache hits
        if request.request_id in self.load_specs:
            load_spec = self.load_specs[request.request_id]
            if num_external_tokens == 0:
                # No need to load anything
                load_spec.can_load = False
            else:
                # Only check for non-prompt-hit case
                if load_spec.test_cache_cached_tokens != request.num_tokens:
                    # Verify the number of tokens matches
                    expected_tokens = load_spec.test_cache_cached_tokens - load_spec.vllm_cached_tokens
                    if num_external_tokens != expected_tokens:
                        logger.warning(
                            f"Mismatch in number of tokens for request {request.request_id}: "
                            f"expected {expected_tokens}, got {num_external_tokens}"
                        )
                
                load_spec.can_load = True
        
        logger.debug(
            f"Updated state for request {request.request_id}: "
            f"{num_external_tokens} external tokens"
        )
    
    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build connector metadata for this scheduling step.
        
        Args:
            scheduler_output: Scheduler output
            
        Returns:
            Connector metadata
        """
        logger.info(f"Building connector metadata for scheduler step. Role: {self._role}")
        logger.info(f"Finished requests: {scheduler_output.finished_req_ids}")
        logger.info(f"New requests count: {len(scheduler_output.scheduled_new_reqs)}")
        logger.info(f"Cached requests type: {type(scheduler_output.scheduled_cached_reqs)}")
        
        metadata = TestCacheEngineConnectorMetadata()
        
        # Process finished requests
        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            logger.debug(f"Cleaned up finished request: {finished_req_id}")
        
        # Process new requests
        new_request_count = 0
        for request in scheduler_output.scheduled_new_reqs:
            # Get load spec for this request
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            test_cache_cached_tokens = 0
            if load_spec is not None:
                test_cache_cached_tokens = load_spec.test_cache_cached_tokens
            
            logger.info(f"Processing new request {request.req_id}: "
                       f"num_tokens_to_compute={num_tokens_to_compute}, "
                       f"test_cache_cached_tokens={test_cache_cached_tokens}, "
                       f"load_spec={load_spec is not None}")
            
            # Create request tracker
            request_tracker = RequestTracker.from_new_request(
                request,
                num_tokens_to_compute,
                test_cache_cached_tokens,
                self._force_skip_save,
            )
            self._request_trackers[request.req_id] = request_tracker
            
            # Create request metadata
            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
            )
            if req_meta is not None:
                metadata.add_request(req_meta)
                new_request_count += 1
                logger.info(f"Added request {request.req_id} to metadata with "
                           f"{len(req_meta.token_ids)} tokens, "
                           f"load_spec={req_meta.load_spec is not None}, "
                           f"save_spec={req_meta.save_spec is not None}")
            else:
                logger.info(f"Request {request.req_id} not added to metadata (no load/save needed)")
        
        # Process cached requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        cached_request_count = 0
        
        # Handle different vLLM versions
        if isinstance(cached_reqs, list):
            # Old vLLM version format
            logger.info(f"Processing {len(cached_reqs)} cached requests (old format)")
            for req in cached_reqs:
                request_tracker = self._request_trackers[req.req_id]
                request_tracker.update(req.new_token_ids, req.new_block_ids)
                
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._chunk_size,
                    load_spec=None,
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    metadata.add_request(req_meta)
                    cached_request_count += 1
                    logger.info(f"Added cached request {req.req_id} to metadata with "
                               f"{len(req_meta.token_ids)} tokens")
        else:
            # New vLLM version format
            logger.info(f"Processing {len(cached_reqs.req_ids)} cached requests (new format)")
            for i, req_id in enumerate(cached_reqs.req_ids):
                request_tracker = self._request_trackers[req_id]
                num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                
                if req_id in self._unfinished_requests:
                    request = self._unfinished_requests[req_id]
                    num_current_tokens = len(request_tracker.token_ids)
                    new_token_ids = request.all_token_ids[
                        num_current_tokens : num_current_tokens + num_new_tokens
                    ]
                else:
                    raise ValueError(
                        f"Request {req_id} is not in _unfinished_requests, "
                        f"but it is scheduled to be cached"
                    )
                
                new_block_ids = cached_reqs.new_block_ids[i]
                request_tracker.update(new_token_ids, new_block_ids)
                
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._chunk_size,
                    load_spec=None,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self._save_decode_cache,
                )
                if req_meta is not None:
                    metadata.add_request(req_meta)
                    cached_request_count += 1
                    logger.info(f"Added cached request {req_id} to metadata with "
                               f"{len(req_meta.token_ids)} tokens, "
                               f"new_tokens={num_new_tokens}")
        
        # Set the connector metadata for worker-side access
        self._connector_metadata = metadata
        
        logger.info(f"Successfully built connector metadata: "
                   f"total_requests={len(metadata.requests)} "
                   f"(new={new_request_count}, cached={cached_request_count})")
        
        # Debug: print details of each request in metadata
        for i, req in enumerate(metadata.requests):
            logger.debug(f"Request {i}: id={req.req_id}, tokens={len(req.token_ids)}, "
                        f"load_spec={req.load_spec is not None}, "
                        f"save_spec={req.save_spec is not None}")
        
        return metadata
    
    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Handle request completion.
        
        Args:
            request: Finished request
            block_ids: Block IDs being freed
            
        Returns:
            Tuple of (should defer block freeing, optional return parameters)
        """
        # TestCacheEngine doesn't use asynchronous saving
        return False, None
    
    # ==============================
    # Helper methods
    # ==============================

    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        """Initialize KV caches from forward context - similar to LMCache implementation."""
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("Layer %s does not have kv_cache, skipping", layer_name)
                continue
            
            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]
    
    
    # Remove the overridden _get_connector_metadata method
    # Use the parent class implementation from KVConnectorBase_V1
    
    
    def get_stats(self) -> dict:
        """Get connector statistics."""
        stats = self.stats.copy()
        
        # Add TestCacheEngine statistics with error handling
        try:
            test_engine_stats = self.test_engine.get_stats()
            stats["test_engine_stats"] = test_engine_stats
        except Exception as e:
            logger.warning(f"Failed to get TestCacheEngine statistics: {e}")
            stats["test_engine_stats"] = {"error": str(e), "status": "unavailable"}
        
        stats.update({
            "role": str(self._role),
            "worker_id": self.test_metadata.worker_id
        })
        
        return stats
    
    # ==============================
    # Additional required methods
    # ==============================
    
    # Note: We don't implement register_kv_caches because we follow the same
    # pattern as LMCache connector - kv caches are initialized in start_load_kv
    # via _init_kv_caches_from_forward_context
    
    def shutdown(self):
        """Shutdown the connector and clean up resources."""
        logger.info("Shutting down TestCacheEngineConnector")
        
        try:
            if hasattr(self, 'test_engine'):
                self.test_engine.close()
                logger.info("TestCacheEngine closed successfully")
        except Exception as e:
            logger.error(f"Error closing TestCacheEngine: {e}")
            self.stats["transfer_errors"] += 1
        
        # Clear all internal state
        self.kv_caches.clear()
        self._request_trackers.clear()
        self._unfinished_requests.clear()
        self.load_specs.clear()
        self._events.clear()
        
        logger.info("TestCacheEngineConnector shutdown completed")
    
    def get_kv_connector_stats(self) -> Optional[Any]:
        """Get KV connector statistics."""
        # Return a simple stats object compatible with vLLM's stats system
        return self.get_stats()
    
    def update_connector_output(self, connector_output: Any):
        """Update connector state from worker-side connector output."""
        # TestCacheEngine doesn't need to update from worker output
        pass
    
    def take_events(self):
        """Take KV cache events from the connector."""
        events = self._events.copy()
        self._events.clear()
        return events
    
    def get_handshake_metadata(self) -> Optional[Any]:
        """Get handshake metadata for connector communication."""
        # TestCacheEngine doesn't require handshake metadata
        return None
    
    def set_xfer_handshake_metadata(self, metadata: dict[int, Any]) -> None:
        """Set handshake metadata for connector communication."""
        # TestCacheEngine doesn't require handshake metadata
        pass
    
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """Get the required KV cache layout for this connector."""
        # TestCacheEngine works with standard vLLM layouts
        return None
    
    def get_finished_count(self) -> int | None:
        """Get the count of requests expected to complete send/receive operations."""
        # TestCacheEngine doesn't use asynchronous transfers
        return None
    
    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata."""
        self._connector_metadata = None
    
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler."""
        logger.info(f"bind_connector_metadata called for role {self._role}. "
                   f"Metadata type: {type(connector_metadata)}, "
                   f"has requests: {hasattr(connector_metadata, 'requests')}")
        
        if hasattr(connector_metadata, 'requests'):
            logger.info(f"Metadata contains {len(connector_metadata.requests)} requests")
            for i, req in enumerate(connector_metadata.requests):
                logger.debug(f"Request {i}: id={req.req_id}, tokens={len(req.token_ids)}, "
                           f"load_spec={req.load_spec is not None}, "
                           f"save_spec={req.save_spec is not None}")
        
        self._connector_metadata = connector_metadata
        logger.info(f"Connector metadata bound successfully for role {self._role}")
