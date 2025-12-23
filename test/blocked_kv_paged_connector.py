# SPDX-License-Identifier: Apache-2.0
"""
Blocked KV Paged Memory Connector for vLLM

结合 VLLMPagedMemGPUConnectorV2 的指针管理和 BlockedKVGPUConnector 的6D转换逻辑，
专门用于test cache engine的数据格式。
"""

# Standard
from typing import List, Optional, Tuple, Union
import abc

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

if torch.cuda.is_available():
    # First Party
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


class BlockedKVPagedMemConnector(GPUConnectorInterface):
    """
    专门处理 [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size] 格式的
    KV cache数据，一次性上传/下载全部层到vLLM。
    
    结合了 VLLMPagedMemGPUConnectorV2 的指针管理和 BlockedKVGPUConnector 的6D转换逻辑。
    
    数据格式说明：
    - num_layers: 模型层数
    - num_blocks: block数量
    - 2: Key和Value
    - block_size: vLLM block大小（通常为16）
    - num_kv_heads: KV头数量
    - head_size: 每个头的大小
    
    与vLLM KV cache结构匹配：
    - vLLM格式: [2, num_blocks, block_size, num_kv_heads, head_size]
    - 本connector格式: [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
    """
    
    def __init__(
        self,
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        初始化Blocked KV Paged Memory Connector。
        
        Args:
            num_layers: 模型层数
            block_size: vLLM block大小
            num_kv_heads: KV头数量
            head_size: 每个头的大小
            use_gpu: 是否使用GPU缓冲区
            **kwargs: 额外参数，包括：
                - dtype: 数据类型
                - device: 设备
                - chunk_size: chunk大小（可选）
        """
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        
        # 计算隐藏维度大小
        self.hidden_dim_size = num_kv_heads * head_size
        
        # KV cache指针管理（与VLLMPagedMemGPUConnectorV2一致）
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0
        
        # vLLM KV cache引用
        self.kvcaches: Optional[List[torch.Tensor]] = None
        
        # GPU缓冲区（可选）
        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = kwargs.get("use_mla", False)
        
        if use_gpu:
            assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, "device should be provided to create a GPU buffer."
            
            dtype = kwargs["dtype"]
            device = kwargs["device"]
            
            # 创建GPU缓冲区，形状为 [2, num_layers, chunk_size, hidden_dim_size]
            # 与VLLMPagedMemGPUConnectorV2一致
            chunk_size = kwargs.get("chunk_size", block_size)
            shape = self.get_shape(chunk_size)
            self.gpu_buffer = torch.empty(
                shape, dtype=dtype, device=device
            )
            logger.info(f"GPU buffer created with shape: {shape}")
        
        # CUDA流管理
        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()
        
        logger.info(f"BlockedKVPagedMemConnector initialized: "
                   f"layers={num_layers}, block_size={block_size}, "
                   f"heads={num_kv_heads}, head_size={head_size}")
    
    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        """
        初始化KV cache指针（与VLLMPagedMemGPUConnectorV2._initialize_pointers一致）。
        
        Args:
            kv_caches: vLLM KV cache列表，每层一个tensor
            
        Returns:
            GPU上的指针数组
        """
        self.device = kv_caches[0].device
        assert self.device.type == "cuda", "The device should be CUDA."
        
        idx = self.device.index
        
        # 关键修复：清空缓存确保使用新指针
        # 删除旧的指针缓存，强制重新创建
        if idx in self.kv_cache_pointers_on_gpu:
            del self.kv_cache_pointers_on_gpu[idx]
        
        logger.info(f"强制重新初始化指针 for device {self.device} (已清空缓存)")
        
        # 填充CPU端指针数组
        # 注意：multi_layer_kv_transfer期望的是int64_t*数组，每个int64_t*指向一个vLLM KV cache
        # 所以我们需要创建一个int64_t*数组，每个元素是vLLM KV cache的地址
        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        
        # 创建GPU上的指针数组
        # 注意：multi_layer_kv_transfer期望的是int64_t*数组，所以我们需要创建一个int64_t*数组
        # 每个元素是vLLM KV cache的地址
        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.num_layers, dtype=torch.int64, device=self.device
        )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)
        
        # 计算页面缓冲区大小
        if self.use_mla:
            # MLA格式: [num_blocks, block_size, head_size]
            assert kv_caches[0].dim() == 3
            self.page_buffer_size = kv_caches[0].shape[0] * kv_caches[0].shape[1]
        else:
            # 标准vLLM格式: [2, num_blocks, block_size, num_heads, head_size]
            assert kv_caches[0].dim() == 5
            
            # 确定哪个维度是2（K/V）
            if kv_caches[0].shape[0] == 2:
                # 格式: [2, num_blocks, block_size, num_heads, head_size]
                self.page_buffer_size = kv_caches[0].shape[1] * kv_caches[0].shape[2]
            elif kv_caches[0].shape[1] == 2:
                # 格式: [num_blocks, 2, block_size, num_heads, head_size]
                self.page_buffer_size = kv_caches[0].shape[0] * kv_caches[0].shape[2]
            else:
                raise ValueError(f"无法确定vLLM KV cache格式: shape={kv_caches[0].shape}")
        
        logger.info(f"KV cache pointers initialized: {len(kv_caches)} layers, "
                   f"page_buffer_size={self.page_buffer_size}")
        
        # 调试：打印指针值
        logger.debug(f"指针数组值:")
        for i, ptr in enumerate(self.kv_cache_pointers_on_gpu[idx]):
            logger.debug(f"  第{i}层: {hex(ptr.item())}")
        
        return self.kv_cache_pointers_on_gpu[idx]
    
    def _convert_to_flash_attention_format(
        self,
        input_tensor: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        将blocked KV cache数据转换为Flash Attention格式（KV_2LTD格式）。
        
        Args:
            input_tensor: 输入tensor，形状应为：
                         - [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
                         - [num_layers, 2, block_size, num_kv_heads, head_size]
            start: 起始token索引
            end: 结束token索引
            
        Returns:
            转换后的Flash Attention格式tensor: [2, num_layers, num_tokens, num_heads*head_size]
            
        Raises:
            ValueError: 如果输入tensor维度不支持
        """
        num_tokens = end - start
        
        # 检查输入tensor维度并转换为Flash Attention格式
        if input_tensor.dim() == 6:
            # 格式: [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
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
            
            # 计算总token数
            total_tokens = num_blocks * self.block_size
            if num_tokens != total_tokens:
                logger.warning(
                    f"Token count mismatch: expected {total_tokens} (num_blocks={num_blocks} * block_size={self.block_size}), "
                    f"got {num_tokens}. Using first {num_tokens} tokens."
                )
            
            # 转换格式: [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
            # -> [num_layers, 2, num_blocks * block_size, num_kv_heads, head_size]
            # -> [2, num_layers, num_blocks * block_size, num_kv_heads, head_size]
            # -> [2, num_layers, num_tokens, num_kv_heads * head_size]
            
            # 1. 合并blocks维度: [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
            #    -> [num_layers, 2, num_blocks * block_size, num_kv_heads, head_size]
            reshaped = input_tensor.permute(0, 2, 1, 3, 4, 5).contiguous()
            reshaped = reshaped.view(self.num_layers, 2, num_blocks * self.block_size, self.num_kv_heads, self.head_size)
            
            # 2. 选择需要的tokens
            selected_tokens = reshaped[:, :, start:end, :, :]
            
            # 3. 转换为Flash Attention格式: [num_layers, 2, num_tokens, num_kv_heads, head_size]
            #    -> [2, num_layers, num_tokens, num_kv_heads, head_size]
            #    -> [2, num_layers, num_tokens, num_kv_heads * head_size]
            flash_attention_tensor = selected_tokens.permute(1, 0, 2, 3, 4).contiguous()
            flash_attention_tensor = flash_attention_tensor.view(2, self.num_layers, num_tokens, self.hidden_dim_size)
            
        elif input_tensor.dim() == 5:
            # 格式: [num_layers, 2, block_size, num_kv_heads, head_size]
            assert input_tensor.shape[0] == self.num_layers, \
                f"Expected {self.num_layers} layers, got {input_tensor.shape[0]}"
            assert input_tensor.shape[1] == 2, f"Expected 2 KV pairs, got {input_tensor.shape[1]}"
            assert input_tensor.shape[2] == self.block_size, \
                f"Expected block_size={self.block_size}, got {input_tensor.shape[2]}"
            assert input_tensor.shape[3] == self.num_kv_heads, \
                f"Expected {self.num_kv_heads} heads, got {input_tensor.shape[3]}"
            assert input_tensor.shape[4] == self.head_size, \
                f"Expected head_size={self.head_size}, got {input_tensor.shape[4]}"
            
            if num_tokens != self.block_size:
                logger.warning(
                    f"Token count mismatch: expected {self.block_size}, got {num_tokens}. "
                    f"Using first {num_tokens} tokens from each block."
                )
            
            # 转换格式: [num_layers, 2, num_tokens, num_kv_heads, head_size]
            # -> [2, num_layers, num_tokens, num_kv_heads, head_size]
            # -> [2, num_layers, num_tokens, num_kv_heads * head_size]
            selected_tokens = input_tensor[:, :, start:end, :, :]
            flash_attention_tensor = selected_tokens.permute(1, 0, 2, 3, 4).contiguous()
            flash_attention_tensor = flash_attention_tensor.view(2, self.num_layers, num_tokens, self.hidden_dim_size)
            
        else:
            raise ValueError(
                f"Unsupported tensor dimension: {input_tensor.dim()}. "
                f"Expected 5 ([num_layers, 2, block_size, num_kv_heads, head_size]) or "
                f"6 ([num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]) dimensions."
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
        将Flash Attention格式数据转换为blocked KV cache格式。
        
        Args:
            flash_attention_tensor: Flash Attention格式tensor: [2, num_layers, num_tokens, num_heads*head_size]
            num_blocks: block数量（可选，如果知道的话）
            
        Returns:
            转换后的blocked KV cache格式tensor: [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
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
        
        # 计算block数量
        if num_blocks is None:
            num_blocks = num_tokens // self.block_size
            if num_tokens % self.block_size != 0:
                logger.warning(
                    f"Token count {num_tokens} is not divisible by block_size {self.block_size}. "
                    f"Using {num_blocks} full blocks and ignoring remaining tokens."
                )
        
        # 转换格式: [2, num_layers, num_tokens, num_heads*head_size]
        # -> [2, num_layers, num_tokens, num_kv_heads, head_size]
        # -> [num_layers, 2, num_tokens, num_kv_heads, head_size]
        # -> [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
        
        # 1. 恢复隐藏维度: [2, num_layers, num_tokens, num_heads*head_size]
        #    -> [2, num_layers, num_tokens, num_kv_heads, head_size]
        reshaped = flash_attention_tensor.view(2, self.num_layers, num_tokens, self.num_kv_heads, self.head_size)
        
        # 2. 转换维度顺序: [2, num_layers, num_tokens, num_kv_heads, head_size]
        #    -> [num_layers, 2, num_tokens, num_kv_heads, head_size]
        blocked_5d = reshaped.permute(1, 0, 2, 3, 4).contiguous()
        
        # 3. 分割为blocks: [num_layers, 2, num_tokens, num_kv_heads, head_size]
        #    -> [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
        # 首先reshape为 [num_layers, 2, num_blocks, block_size, num_kv_heads, head_size]
        blocked_6d = blocked_5d.view(self.num_layers, 2, num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        
        # 4. 调整维度顺序: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_size]
        #    -> [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
        blocked_6d = blocked_6d.permute(0, 2, 1, 3, 4, 5).contiguous()
        
        logger.debug(
            f"Reverse format conversion: {flash_attention_tensor.shape} -> {blocked_6d.shape}"
        )
        
        return blocked_6d
    
    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """
        to_gpu方法的占位符实现。
        
        注意：test cache engine不使用MemoryObj，所以这个方法只是占位符。
        实际使用应该调用upload_blocked_kv方法。
        """
        raise NotImplementedError(
            "to_gpu method is not implemented for BlockedKVPagedMemConnector. "
            "Use upload_blocked_kv method instead for test cache engine."
        )
    
    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """
        from_gpu方法的占位符实现。
        
        注意：test cache engine不使用MemoryObj，所以这个方法只是占位符。
        实际使用应该调用download_blocked_kv方法。
        """
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
        """
        批量上传KV cache数据到vLLM。
        
        注意：对于blocked布局，通常一次上传一个block的所有层数据，
        所以这里主要处理多个blocks的情况。
        """
        # 简化实现：顺序处理每个memory_obj
        with torch.cuda.stream(self.load_stream):
            if isinstance(memory_objs[0], list):
                # 多层格式：List[List[MemoryObj]]
                for layer_memory_objs, start, end in zip(memory_objs, starts, ends, strict=False):
                    for memory_obj in layer_memory_objs:
                        self.to_gpu(memory_obj, start, end, **kwargs)
            else:
                # 单层格式：List[MemoryObj]
                for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                    self.to_gpu(memory_obj, start, end, **kwargs)
        
        self.load_stream.synchronize()
    
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        批量从vLLM下载KV cache数据。
        """
        # 简化实现：顺序处理每个memory_obj
        if isinstance(memory_objs[0], list):
            # 多层格式
            for layer_memory_objs, start, end in zip(memory_objs, starts, ends, strict=False):
                for memory_obj in layer_memory_objs:
                    self.from_gpu(memory_obj, start, end, **kwargs)
        else:
            # 单层格式
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.from_gpu(memory_obj, start, end, **kwargs)
    
    def get_shape(self, num_tokens: int) -> torch.Size:
        """
        获取给定token数量的数据形状（与VLLMPagedMemGPUConnectorV2一致）。
        
        Args:
            num_tokens: token数量
            
        Returns:
            数据形状 [2, num_layers, num_tokens, hidden_dim_size]
        """
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])
    
    def initialize_kvcaches_ptr(self, **kwargs):
        """
        初始化KV cache指针。
        
        Args:
            **kwargs: 必须包含'kvcaches'参数
        """
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]
            logger.info(f"Initialized kvcaches pointer with {len(self.kvcaches)} layers")
            
            # 立即初始化指针，而不是等到upload时才初始化
            # 这样可以确保指针在batch_upload_blocked_kv之前就已经准备好
            if self.kvcaches and len(self.kvcaches) > 0:
                try:
                    logger.info(f"Calling _initialize_pointers during initialize_kvcaches_ptr")
                    kv_cache_pointers = self._initialize_pointers(self.kvcaches)
                    logger.info(f"Successfully initialized pointers: shape={kv_cache_pointers.shape}, device={kv_cache_pointers.device}")
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
        直接上传blocked KV cache数据，不经过MemoryObj转换。
        类似于VLLMPagedMemGPUConnectorV2.to_gpu，但处理的是6D格式数据。
        
        Args:
            blocked_kv_data: KV cache数据，形状应为：
                            - [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
                            - [num_layers, 2, block_size, num_kv_heads, head_size]
            vllm_kvcaches: vLLM KV cache列表
            slot_mapping: slot mapping tensor
            start: 起始token索引
            end: 结束token索引
            **kwargs: 额外参数
        """
        # 1. 转换为Flash Attention格式（KV_2LTD格式）
        flash_attention_tensor = self._convert_to_flash_attention_format(blocked_kv_data, start, end)
        
        logger.debug(f"Converted to Flash Attention format: {blocked_kv_data.shape} -> {flash_attention_tensor.shape}")
        
        # 2. 获取指针（如果已经初始化过，直接使用；否则初始化）
        device = vllm_kvcaches[0].device
        idx = device.index
        
        if idx in self.kv_cache_pointers_on_gpu:
            # 指针已经初始化，直接使用
            kv_cache_pointers = self.kv_cache_pointers_on_gpu[idx]
            logger.debug(f"Using existing pointers for device {device}")
        else:
            # 指针未初始化，需要初始化
            logger.info(f"Pointers not initialized for device {device}, initializing now")
            kv_cache_pointers = self._initialize_pointers(vllm_kvcaches)
        
        # 3. 检查slot_mapping索引是否有效
        if start >= len(slot_mapping) or end > len(slot_mapping):
            raise ValueError(
                f"slot_mapping索引越界: start={start}, end={end}, "
                f"slot_mapping长度={len(slot_mapping)}"
            )
        
        # 4. 获取slot_mapping切片
        slot_mapping_slice = slot_mapping[start:end]
        logger.info(f"Upload blocked KV: start={start}, end={end}, num_tokens={end-start}")
        
        # 检查slot_mapping值是否在page_buffer_size范围内
        if len(slot_mapping_slice) > 0:
            max_slot = slot_mapping_slice.max().item()
            if max_slot >= self.page_buffer_size:
                raise ValueError(
                    f"slot_mapping值越界: max_slot={max_slot} >= page_buffer_size={self.page_buffer_size}"
                )
        
        lmc_ops.multi_layer_kv_transfer(
            flash_attention_tensor,     # [2, num_layers, num_tokens, hidden_dim] - KV_2LTD格式
            kv_cache_pointers,          # 所有层KV cache指针
            slot_mapping_slice,         # slot mapping
            self.device,
            self.page_buffer_size,
            False,                      # direction=False (LMCache→GPU)
            self.use_mla,
        )
        
        # 同步以确保数据已复制
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
        直接从vLLM下载blocked KV cache数据，不经过MemoryObj转换。
        类似于VLLMPagedMemGPUConnectorV2.from_gpu，但返回的是6D格式数据。
        
        Args:
            vllm_kvcaches: vLLM KV cache列表
            slot_mapping: slot mapping tensor
            start: 起始token索引
            end: 结束token索引
            dtype: 输出tensor的数据类型
            device: 输出tensor的设备（如果为None，则使用vllm_kvcaches的设备）
            **kwargs: 额外参数
            
        Returns:
            下载的blocked KV cache数据，形状为 [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
        """
        # 1. 获取指针（如果已经初始化过，直接使用；否则初始化）
        # 注意：initialize_kvcaches_ptr应该已经调用了_initialize_pointers
        # 这里我们检查指针是否已经初始化，避免重复初始化
        device_idx = vllm_kvcaches[0].device.index
        
        # 关键调试：检查当前指针是否指向正确的kvcache地址
        logger.info(f"=== DEBUG: Download pointer check ===")
        logger.info(f"Device index: {device_idx}")
        logger.info(f"Current vllm_kvcaches[0] address: {hex(vllm_kvcaches[0].data_ptr())}")
        
        if device_idx in self.kv_cache_pointers_on_gpu:
            # 指针已经初始化，直接使用
            kv_cache_pointers = self.kv_cache_pointers_on_gpu[device_idx]
            logger.info(f"Using existing pointers for device {vllm_kvcaches[0].device} in download")
            
            # 检查指针是否指向正确的地址
            if len(kv_cache_pointers) > 0:
                stored_ptr = kv_cache_pointers[0].item()
                current_ptr = vllm_kvcaches[0].data_ptr()
                logger.info(f"Stored pointer[0]: {hex(stored_ptr)}")
                logger.info(f"Current vllm_kvcaches[0] pointer: {hex(current_ptr)}")
                logger.info(f"Pointers match: {stored_ptr == current_ptr}")
                
                # 如果指针不匹配，需要重新初始化
                if stored_ptr != current_ptr:
                    logger.warning(f"Pointer mismatch! Reinitializing pointers for new kvcache addresses")
                    kv_cache_pointers = self._initialize_pointers(vllm_kvcaches)
        else:
            # 指针未初始化，需要初始化
            logger.info(f"Pointers not initialized for device {vllm_kvcaches[0].device}, initializing now in download")
            kv_cache_pointers = self._initialize_pointers(vllm_kvcaches)
        
        # 2. 计算token数量和block数量
        num_tokens = end - start
        num_blocks = num_tokens // self.block_size
        if num_tokens % self.block_size != 0:
            logger.warning(
                f"Token count {num_tokens} is not divisible by block_size {self.block_size}. "
                f"Using {num_blocks} full blocks and ignoring remaining tokens."
            )
        
        # 3. 确定输出设备
        if device is None:
            device = vllm_kvcaches[0].device
        
        # 4. 创建临时Flash Attention格式缓冲区（KV_2LTD格式）
        # 形状: [2, num_layers, num_tokens, hidden_dim]
        flash_attention_shape = (2, self.num_layers, num_tokens, self.hidden_dim_size)
        
        # temp_flash_buffer必须在vllm_kvcaches的设备上创建，因为multi_layer_kv_transfer需要GPU缓冲区
        temp_flash_buffer = torch.empty(
            flash_attention_shape,
            dtype=dtype,
            device=vllm_kvcaches[0].device
        )
        
        # 5. 检查slot_mapping索引是否有效
        if start >= len(slot_mapping) or end > len(slot_mapping):
            raise ValueError(
                f"slot_mapping索引越界: start={start}, end={end}, "
                f"slot_mapping长度={len(slot_mapping)}"
            )
        
        # 6. 检查slot_mapping值是否在page_buffer_size范围内
        slot_mapping_slice = slot_mapping[start:end]
        if len(slot_mapping_slice) > 0:
            min_slot = slot_mapping_slice.min().item()
            max_slot = slot_mapping_slice.max().item()
            if max_slot >= self.page_buffer_size:
                raise ValueError(
                    f"slot_mapping值越界: max_slot={max_slot} >= page_buffer_size={self.page_buffer_size}"
                )
        
        # 7. 调用multi_layer_kv_transfer（与VLLMPagedMemGPUConnectorV2.from_gpu一致）
        # 注意：direction=True表示GPU→LMCache（下载）
        lmc_ops.multi_layer_kv_transfer(
            temp_flash_buffer,          # 输出: Flash Attention格式
            kv_cache_pointers,          # 输入: vLLM分页内存指针
            slot_mapping_slice,         # slot mapping
            vllm_kvcaches[0].device,    # GPU设备
            self.page_buffer_size,      # 页面缓冲区大小
            True,                       # direction=True (GPU→LMCache)
            self.use_mla,               # 格式标志
        )
        
        # 8. 同步CUDA以确保数据已下载
        torch.cuda.synchronize(vllm_kvcaches[0].device)
        
        # 9. 将Flash Attention格式转换为6维blocked格式
        blocked_6d = self._convert_from_flash_attention_format(temp_flash_buffer, num_blocks)
        
        logger.debug(f"Converted from Flash Attention format: {temp_flash_buffer.shape} -> {blocked_6d.shape}")
        
        # 10. 如果需要，将数据移动到指定设备
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
        批量上传blocked KV cache数据。
        
        Args:
            blocked_kv_data_list: blocked KV cache数据列表，每个元素形状应为：
                                 - [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
                                 - [num_layers, 2, block_size, num_kv_heads, head_size]
            vllm_kvcaches: vLLM KV cache列表
            slot_mapping_list: slot mapping tensor列表
            starts: 起始token索引列表
            ends: 结束token索引列表
            **kwargs: 额外参数
        """
        assert len(blocked_kv_data_list) == len(slot_mapping_list) == len(starts) == len(ends), \
            "All input lists must have the same length"
        
        logger.info(f"Batch upload: {len(blocked_kv_data_list)} chunks")
        
        # 模仿原来的batch操作，循环调用单个上传函数
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
        logger.info(
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
        批量下载blocked KV cache数据。
        
        Args:
            vllm_kvcaches: vLLM KV cache列表
            slot_mapping_list: slot mapping tensor列表
            starts: 起始token索引列表
            ends: 结束token索引列表
            dtype: 输出tensor的数据类型
            device: 输出tensor的设备（如果为None，则使用vllm_kvcaches的设备）
            **kwargs: 额外参数
            
        Returns:
            下载的blocked KV cache数据列表，每个元素形状为 [num_layers, num_blocks, 2, block_size, num_kv_heads, head_size]
        """
        assert len(slot_mapping_list) == len(starts) == len(ends), \
            "All input lists must have the same length"
        
        # 模仿原来的batch操作，循环调用单个下载函数
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


# 使用示例
def example_usage():
    """
    BlockedKVPagedMemConnector使用示例
    """
    import torch
    
    # 配置参数
    num_layers = 32
    block_size = 16
    num_kv_heads = 32
    head_size = 128
    
    # 创建connector
    connector = BlockedKVPagedMemConnector(
        num_layers=num_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        use_gpu=True,
        dtype=torch.float16,
        device="cuda:0",
        chunk_size=256
    )
    
    # 模拟blocked KV cache数据 [num_layers, 2, block_size, num_kv_heads, head_size]
    blocked_kv_data = torch.randn(
        num_layers, 2, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device="cuda:0"
    )
    
    print("BlockedKVPagedMemConnector created successfully")
    print(f"Upload data shape: {blocked_kv_data.shape}")
    print(f"Connector supports {num_layers} layers, block_size={block_size}")
    print(f"Hidden dimension size: {connector.hidden_dim_size}")


if __name__ == "__main__":
    example_usage()
