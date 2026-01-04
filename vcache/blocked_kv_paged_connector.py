"""
Blocked KV Paged GPU Memory Connector for vLLM
"""

# Standard
from typing import List, Optional, Union
import torch


from lmcache.vcache.vcache_logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryObj

if torch.cuda.is_available():
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


class BlockedKVPagedMemConnector(GPUConnectorInterface):
    """
    
    compatible with vcache engine format
    - vLLM: [2, num_blocks, block_size, num_kv_heads, head_size]
    - vcache engine: [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
    """
    
    def __init__(
        self,
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        **kwargs,
    ):
        """
        init Blocked KV Paged Memory Connector
        
        Args:
            num_layers: model layers
            block_size: vLLM block size
            num_kv_heads: KV heads number
            head_size: head size
            **kwargs:
                - dtype:
                - device:
                - chunk_size
        """
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        
        self.hidden_dim_size = num_kv_heads * head_size
        
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0
        
        # vLLM KV caches reference
        self.kvcaches: Optional[List[torch.Tensor]] = None
        
        self.use_mla = False

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()
        
        logger.info(f"BlockedKVPagedMemConnector initialized: "
                   f"layers={num_layers}, block_size={block_size}, "
                   f"heads={num_kv_heads}, head_size={head_size}")
    
    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        """
        initialize KV cache pointers
        
        Args:
            kv_caches: vLLM KV cache list,
            
        Returns:
            GPU tensor of pointers
        """
        self.device = kv_caches[0].device
        assert self.device.type == "cuda", "The device should be CUDA."
        
        idx = self.device.index
        
        # delete existing pointers
        if idx in self.kv_cache_pointers_on_gpu:
            del self.kv_cache_pointers_on_gpu[idx]
        
        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        
        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.num_layers, dtype=torch.int64, device=self.device
        )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        assert kv_caches[0].dim() == 5
        
        if kv_caches[0].shape[0] == 2:
            # [2, num_blocks, block_size, num_heads, head_size]
            self.page_buffer_size = kv_caches[0].shape[1] * kv_caches[0].shape[2]
        elif kv_caches[0].shape[1] == 2:
            # [num_blocks, 2, block_size, num_heads, head_size]
            self.page_buffer_size = kv_caches[0].shape[0] * kv_caches[0].shape[2]
        else:
            raise ValueError(f"unable to determine shape={kv_caches[0].shape}")
        
        logger.debug(f"KV cache pointers initialized: {len(kv_caches)} layers, "
                   f"page_buffer_size={self.page_buffer_size}")
              
        return self.kv_cache_pointers_on_gpu[idx]
    
    def _convert_to_flash_attention_format(
        self,
        input_tensor: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        convert blocked KV cache to Flash Attention
        
        Args:
            input_tensor:  [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
            start: start token index
            end: 
            
        Returns:
            Flash Attention tensor: [2, num_layers, num_tokens, num_heads*head_size]
        """
        num_tokens = end - start
        
        if input_tensor.dim() == 6:
            # [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
            num_blocks = input_tensor.shape[1]
            assert input_tensor.shape[0] == self.num_layers, \
                f"Expected {self.num_layers} layers, got {input_tensor.shape[0]}"
            assert input_tensor.shape[2] == 2, f"Expected 2 KV pairs, got {input_tensor.shape[2]}"
            assert input_tensor.shape[3] == self.block_size, \
                f"Expected block_size={self.block_size}, got {input_tensor.shape[3]}"
            assert input_tensor.shape[4] == self.num_kv_heads, \
                f"Expected {self.num_kv_heads} heads, got {input_tensor.shape[4]}"
            assert input_tensor.shape[5] == self.head_size, \
                f"Expected head_size={self.head_size}, got {input_tensor.shape[5]}"
            
            total_tokens = num_blocks * self.block_size
            if num_tokens != total_tokens:
                logger.warning(
                    f"Token count mismatch: expected {total_tokens} "
                    f"(num_blocks={num_blocks} * block_size={self.block_size}), "
                    f"got {num_tokens}. Using first {num_tokens} tokens."
                )
            
            # convert [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
            # -> [num_layers, 2, num_blocks * block_size, num_kv_heads, head_size]
            # -> [2, num_layers, num_blocks * block_size, num_kv_heads, head_size]
            # -> [2, num_layers, num_tokens, num_kv_heads * head_size]
            
            # 1. [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
            #    -> [num_layers, 2, num_blocks * block_size, num_kv_heads, head_size]
            reshaped = input_tensor.permute(0, 2, 1, 3, 4, 5).contiguous()
            reshaped = reshaped.view(self.num_layers, 2, num_blocks * self.block_size, self.num_kv_heads, self.head_size)
            
            # 2. select tokens
            selected_tokens = reshaped[:, :, start:end, :, :]
            
            # 3. Flash Attention: [num_layers, 2, num_tokens, num_kv_heads, head_size]
            #    -> [2, num_layers, num_tokens, num_kv_heads, head_size]
            #    -> [2, num_layers, num_tokens, num_kv_heads * head_size]
            flash_attention_tensor = selected_tokens.permute(1, 0, 2, 3, 4).contiguous()
            flash_attention_tensor = flash_attention_tensor.view(2, self.num_layers, num_tokens, self.hidden_dim_size)
        else:
            raise ValueError(
                f"Unsupported tensor dimension: {input_tensor.dim()}. "
                f"expect ([num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]) dimensions."
            )
        
        logger.debug(
            f"Format conversion: {input_tensor.shape} -> {flash_attention_tensor.shape}"
        )
        
        return flash_attention_tensor
    
    def _convert_from_flash_attention_format(
        self,
        flash_attention_tensor: torch.Tensor,
        num_blocks: Optional[int] = None
    ) -> torch.Tensor:
        """
        convert Flash Attention to blocked KV cache
        
        Args:
            flash_attention_tensor:  [2, num_layers, num_tokens, num_heads*head_size]
            num_blocks: 
            
        Returns:
            blocked KV cache tensor: [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
        """
        assert flash_attention_tensor.dim() == 4, \
            f"Expected 4D Flash Attention tensor, got {flash_attention_tensor.dim()}D"
        assert flash_attention_tensor.shape[0] == 2, \
            f"Expected first dimension to be 2 (KV pairs), got {flash_attention_tensor.shape[0]}"
        assert flash_attention_tensor.shape[1] == self.num_layers, \
            f"Expected {self.num_layers} layers, got {flash_attention_tensor.shape[1]}"
        assert flash_attention_tensor.shape[3] == self.hidden_dim_size, \
            f"Expected hidden dimension size {self.hidden_dim_size}, got {flash_attention_tensor.shape[3]}"
        
        num_tokens = flash_attention_tensor.shape[2]
        
        if num_blocks is None:
            num_blocks = num_tokens // self.block_size
            if num_tokens % self.block_size != 0:
                logger.warning(
                    f"Token count {num_tokens} is not divisible by block_size {self.block_size}. "
                    f"Using {num_blocks} full blocks and ignoring remaining tokens."
                )
        
        # [2, num_layers, num_tokens, num_heads*head_size]
        # -> [2, num_layers, num_tokens, num_kv_heads, head_size]
        # -> [num_layers, 2, num_tokens, num_kv_heads, head_size]
        # -> [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
        
        # [2, num_layers, num_tokens, num_heads*head_size]
        #    -> [2, num_layers, num_tokens, num_kv_heads, head_size]
        reshaped = flash_attention_tensor.view(2, self.num_layers, num_tokens, self.num_kv_heads, self.head_size)
        
        #  [2, num_layers, num_tokens, num_kv_heads, head_size]
        #    -> [num_layers, 2, num_tokens, num_kv_heads, head_size]
        blocked_5d = reshaped.permute(1, 0, 2, 3, 4).contiguous()
        
        # [num_layers, 2, num_tokens, num_kv_heads, head_size]
        #    -> [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
        # reshape to [num_layers, 2, num_blocks, block_size, num_kv_heads, head_size]
        blocked_6d = blocked_5d.view(self.num_layers, 2, num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        
        # [num_layers, 2, num_blocks, block_size, num_kv_heads, head_size]
        #    -> [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
        blocked_6d = blocked_6d.permute(0, 2, 1, 3, 4, 5).contiguous()
        
        logger.debug(
            f"Reverse format conversion: {flash_attention_tensor.shape} -> {blocked_6d.shape}"
        )
        
        return blocked_6d
    
    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError(
            "to_gpu method is not implemented for BlockedKVPagedMemConnector. "
            "Use upload_blocked_kv method instead for test cache engine."
        )
    
    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):

        raise NotImplementedError(
            "from_gpu method is not implemented for BlockedKVPagedMemConnector. "
            "Use download_blocked_kv method instead for test cache engine."
        )
    
    def batched_to_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        raise NotImplementedError(
            "batched_to_gpu method is not implemented for BlockedKVPagedMemConnector. "
            "Use batch_download_blocked_kv method instead for test cache engine."
        )
    
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        raise NotImplementedError(
            "batched_from_gpu method is not implemented for BlockedKVPagedMemConnector. "
            "Use batch_download_blocked_kv method instead for test cache engine."
        )
    
    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])
    
    def initialize_kvcaches_ptr(self, **kwargs):
        """
        initilize kvcache ptr
        
        Args:
            **kwargs: kvcaches
        """
        self.kvcaches = kwargs["kvcaches"]
        logger.debug(f"Initialized kvcaches pointer with {len(self.kvcaches)} layers")
        
        if self.kvcaches and len(self.kvcaches) > 0:
            try:
                self._initialize_pointers(self.kvcaches)
            except Exception as e:
                logger.error(f"Failed to initialize pointers during initialize_kvcaches_ptr: {e}")
    
    def upload_blocked_kv(
        self,
        blocked_kv_data: torch.Tensor,
        vllm_kvcaches: List[torch.Tensor],
        slot_mapping: torch.Tensor,
        start: int,
        end: int,
        **kwargs
    ) -> None:
        """
        upload blocked KV cache。
        
        Args:
            blocked_kv_data: KV cache
                            - [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
            vllm_kvcaches: vLLM KV cache list
            slot_mapping: slot mapping tensor
            start: 
            end: 
            **kwargs: 
        """
        # 1. convert to Flash Attention
        flash_attention_tensor = self._convert_to_flash_attention_format(blocked_kv_data, start, end)
        
        logger.debug(f"Converted to Flash Attention format: {blocked_kv_data.shape} -> {flash_attention_tensor.shape}")
        
        # 2. get ptr
        device = vllm_kvcaches[0].device
        idx = device.index
        
        if idx in self.kv_cache_pointers_on_gpu:
            kv_cache_pointers = self.kv_cache_pointers_on_gpu[idx]
            logger.debug(f"Using existing pointers for device {device}")
        else:
            logger.debug(f"Pointers not initialized for device {device}, initializing now")
            kv_cache_pointers = self._initialize_pointers(vllm_kvcaches)
        
        # 3. check slot_mapping valid
        if start >= len(slot_mapping) or end > len(slot_mapping):
            raise ValueError(
                f"slot_mapping out of bounds: start={start}, end={end}, "
                f"slot_mapping length={len(slot_mapping)}"
            )
        
        # 4. get slot mapping slice
        slot_mapping_slice = slot_mapping[start:end]
        logger.debug(f"Upload blocked KV: start={start}, end={end}, num_tokens={end-start}")
        
        # check slot_mapping in page_buffer_size
        if len(slot_mapping_slice) > 0:
            max_slot = slot_mapping_slice.max().item()
            if max_slot >= self.page_buffer_size:
                raise ValueError(
                    f"slot_mapping out of bound: "
                    f"max_slot={max_slot} >= page_buffer_size={self.page_buffer_size}"
                )
        
        lmc_ops.multi_layer_kv_transfer(
            flash_attention_tensor,     # [2, num_layers, num_tokens, hidden_dim]
            kv_cache_pointers,          # kv cache ptrs
            slot_mapping_slice,         # slot mapping
            self.device,
            self.page_buffer_size,
            False,                      # direction=False (to vllm)
            self.use_mla,
        )
        
        torch.cuda.synchronize(self.device)
        
        logger.info(
            f"Upload completed: {end-start} tokens (start={start}, end={end}) "
            f"for all {self.num_layers} layers"
        )
    
    def download_blocked_kv(
        self,
        vllm_kvcaches: List[torch.Tensor],
        slot_mapping: torch.Tensor,
        start: int,
        end: int,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        download blocked KV cache from vllm
        
        Args:
            vllm_kvcaches: vLLM KV cache list
            slot_mapping: slot mapping tensor
            start: 
            end: 
            dtype: tensor dtype
            device: output tensor device
            **kwargs:
            
        Returns:
            blocked KV cache [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
        """
        device_idx = vllm_kvcaches[0].device.index
        
        if device_idx in self.kv_cache_pointers_on_gpu:
            kv_cache_pointers = self.kv_cache_pointers_on_gpu[device_idx]
            logger.info(f"Using existing pointers for device {vllm_kvcaches[0].device} in download")
            
            if len(kv_cache_pointers) > 0:
                stored_ptr = kv_cache_pointers[0].item()
                current_ptr = vllm_kvcaches[0].data_ptr()

                if stored_ptr != current_ptr:
                    logger.warning(f"Pointer mismatch! Reinitializing pointers for new kvcache addresses")
                    kv_cache_pointers = self._initialize_pointers(vllm_kvcaches)
        else:
            logger.info(f"Pointers not initialized for device {vllm_kvcaches[0].device}, initializing now in download")
            kv_cache_pointers = self._initialize_pointers(vllm_kvcaches)
        
        num_tokens = end - start
        num_blocks = num_tokens // self.block_size
        if num_tokens % self.block_size != 0:
            logger.warning(
                f"Token count {num_tokens} is not divisible by block_size {self.block_size}. "
                f"Using {num_blocks} full blocks and ignoring remaining tokens."
            )
        
        if device is None:
            device = vllm_kvcaches[0].device
        
        # [2, num_layers, num_tokens, hidden_dim]
        flash_attention_shape = (2, self.num_layers, num_tokens, self.hidden_dim_size)
        
        temp_flash_buffer = torch.empty(
            flash_attention_shape,
            dtype=dtype,
            device=vllm_kvcaches[0].device
        )
        
        if start >= len(slot_mapping) or end > len(slot_mapping):
            raise ValueError(
                f"slot_mapping out of bounds: start={start}, end={end}, "
                f"slot_mapping length={len(slot_mapping)}"
            )
        
        slot_mapping_slice = slot_mapping[start:end]
        if len(slot_mapping_slice) > 0:
            min_slot = slot_mapping_slice.min().item()
            max_slot = slot_mapping_slice.max().item()
            if max_slot >= self.page_buffer_size:
                raise ValueError(
                    f"slot_mapping out of bound: max_slot={max_slot} >= page_buffer_size={self.page_buffer_size}"
                )
        
        # multi_layer_kv_transfer
        lmc_ops.multi_layer_kv_transfer(
            temp_flash_buffer,          # output Flash Attention
            kv_cache_pointers,          # kvcache ptr
            slot_mapping_slice,         # slot mapping
            vllm_kvcaches[0].device,    # GPU device
            self.page_buffer_size,      
            True,                       # direction=True (vllm ->)
            self.use_mla,              
        )
        
        torch.cuda.synchronize(vllm_kvcaches[0].device)
        
        # convert to 6d in vcache
        blocked_6d = self._convert_from_flash_attention_format(temp_flash_buffer, num_blocks)
        
        logger.debug(f"Converted from Flash Attention format: {temp_flash_buffer.shape} -> {blocked_6d.shape}")
        
        if device != vllm_kvcaches[0].device:
            blocked_6d = blocked_6d.to(device)
        
        logger.info(
            f"Download completed: {num_tokens} tokens (start={start}, end={end}) "
            f"for all {self.num_layers} layers"
        )
        
        return blocked_6d
    
    def batch_upload_blocked_kv(
        self,
        blocked_kv_data_list: List[torch.Tensor],
        vllm_kvcaches: List[torch.Tensor],
        slot_mapping_list: List[torch.Tensor],
        starts: List[int],
        ends: List[int],
        **kwargs
    ) -> None:
        """
        batch upload blocked KV cache。
        
        Args:
            blocked_kv_data_list: blocked KV cache list：
                                 - [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
            vllm_kvcaches: vLLM KV cache list
            slot_mapping_list: slot mapping tensor list
            starts: start token idx list
            ends: 
            **kwargs: 
        """
        assert len(blocked_kv_data_list) == len(slot_mapping_list) == len(starts) == len(ends), \
            "All input lists must have the same length"
        
        logger.debug(f"Batch upload: {len(blocked_kv_data_list)} chunks")
        
        with torch.cuda.stream(self.load_stream):
            for idx, (blocked_kv_data, slot_mapping, start, end) in enumerate(zip(
                blocked_kv_data_list, slot_mapping_list, starts, ends, strict=False
            )):
                logger.debug(f"Processing chunk {idx}: start={start}, end={end}, tokens={end-start}")
                
                self.upload_blocked_kv(
                    blocked_kv_data=blocked_kv_data,
                    vllm_kvcaches=vllm_kvcaches,
                    slot_mapping=slot_mapping,
                    start=start,
                    end=end,
                    **kwargs
                )
        
        self.load_stream.synchronize()
        
        total_tokens = sum(end - start for start, end in zip(starts, ends, strict=False))
        logger.debug(
            f"Batch upload completed: {len(blocked_kv_data_list)} batches, "
            f"{total_tokens} total tokens for all {self.num_layers} layers"
        )
    
    def batch_download_blocked_kv(
        self,
        vllm_kvcaches: List[torch.Tensor],
        slot_mapping_list: List[torch.Tensor],
        starts: List[int],
        ends: List[int],
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        batch download blocked KV cache。
        
        Args:
            vllm_kvcaches: vLLM KV cache list
            slot_mapping_list: slot mapping tensor list
            starts: start token idx list
            ends: 
            dtype: output tensor dytpe
            device: 
            **kwargs: 
            
        Returns:
            blocked KV cache list [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
        """
        assert len(slot_mapping_list) == len(starts) == len(ends), \
            "All input lists must have the same length"
        
        results = []
        
        for slot_mapping, start, end in zip(slot_mapping_list, starts, ends, strict=False):
            blocked_6d = self.download_blocked_kv(
                vllm_kvcaches=vllm_kvcaches,
                slot_mapping=slot_mapping,
                start=start,
                end=end,
                dtype=dtype,
                device=device,
                **kwargs
            )
            results.append(blocked_6d)
        
        total_tokens = sum(end - start for start, end in zip(starts, ends, strict=False))
        logger.info(
            f"Batch download completed: {len(results)} batches, "
            f"{total_tokens} total tokens for all {self.num_layers} layers"
        )
        
        return results
