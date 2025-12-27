# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class VCacheEngineConnectorV1(KVConnectorBase_V1):
    """
    vLLM Connector using VCacheEngine system.
    
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
        kv_cache_config: "KVCacheConfig",
    ):
        """
        Initialize the VCacheEngine connector.
        
        Args:
            vllm_config: vLLM configuration
            role: Connector role (scheduler or worker)
            kv_cache_config: KV cache configuration
        """
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        
        assert vllm_config.kv_transfer_config is not None
        
        # Check if we should use native implementation
        use_native = vllm_config.kv_transfer_config.get_from_extra_config(
            "use_native", True
        )
        
        if use_native:
            logger.info("Initializing native VCacheEngine connector")
            from lmcache.vcache.vcache_engine_connector import (
                VCacheEngineConnectorV1 as VCacheEngineConnectorLatestImpl,
            )
            cls = VCacheEngineConnectorLatestImpl
        else:
            logger.info("Initializing latest VCacheEngine connector")
            from lmcache.vcache.vcache_engine_connector import (
                VCacheEngineConnectorV1 as VCacheEngineConnectorLatestImpl,
            )
            cls = VCacheEngineConnectorLatestImpl

        # Initialize the actual implementation
        self.vcache_engine = cls(vllm_config, role, kv_cache_config, parent=self)

    # ==============================
    # Worker-side methods
    # ==============================
    
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self.vcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self.vcache_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self.vcache_engine.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self.vcache_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return self.vcache_engine.get_finished(finished_req_ids)


    # ==============================
    # Scheduler-side methods
    # ==============================
    
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        return self.vcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        ), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self.vcache_engine.update_state_after_alloc(request, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self.vcache_engine.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return self.vcache_engine.request_finished(request, block_ids)
'''
    def get_stats(self) -> dict:
        """
        Get connector statistics.

        Returns:
            Dictionary containing connector statistics
        """
        return self.vcache_engine.get_stats()
'''
    # ==============================
    # Additional required methods
    # ==============================