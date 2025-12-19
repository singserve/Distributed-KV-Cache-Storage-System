"""
Test Cache Engine GPU VRAM KV Cache数据结构
简化的数据结构，用于在GPU VRAM segment的buffer中存储KV cache
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch
import time


from typing import Union
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey

@dataclass
class TestVRAMKVCacheUnit:
    """
    Test Cache Engine中用于GPU VRAM segment存储的KV cache单元
    增强的数据结构，包含完整的segment信息
    """
    
    # 核心标识信息
    cache_key: Union[str, CacheEngineKey, LayerCacheEngineKey]
    token_ids: List[int]
    
    # GPU VRAM segment存储信息
    segment_id: str
    segment_offset: int  # segment内的偏移量
    allocated_size: int  # 分配的大小（字节）
    segment_base_address: int  # segment基地址
    
    # KV cache tensor数据 - 直接存储tensor
    kv_cache_tensor: torch.Tensor
    
    # GPU信息
    gpu_id: int
    
    # 原始tensor形状信息（用于恢复展平的数据）
    original_shape: Optional[Tuple[int, ...]] = None
    
    # 访问时间信息
    last_access_time: float = field(default_factory=time.time)
    
    @property
    def buffer_pointer(self) -> int:
        """从kv_cache_tensor获取GPU VRAM地址"""
        return self.kv_cache_tensor.data_ptr()
    
    @property
    def buffer_size(self) -> int:
        """计算buffer大小"""
        return self.kv_cache_tensor.element_size() * self.kv_cache_tensor.nelement()
    
    @property
    def tensor_shape(self) -> Tuple[int, ...]:
        """获取tensor形状"""
        return tuple(self.kv_cache_tensor.shape)
    
    @property
    def tensor_dtype(self) -> torch.dtype:
        """获取tensor数据类型"""
        return self.kv_cache_tensor.dtype
    
    @property
    def tensor_size(self) -> int:
        """计算tensor大小（字节）"""
        return self.kv_cache_tensor.element_size() * self.kv_cache_tensor.nelement()
    
    @property
    def vram_address(self) -> int:
        """获取VRAM地址（segment基地址 + 偏移量）"""
        return self.segment_base_address + self.segment_offset
    
    @property
    def cache_key_str(self) -> str:
        """获取cache key的字符串表示"""
        if isinstance(self.cache_key, (CacheEngineKey, LayerCacheEngineKey)):
            return self.cache_key.to_string()
        return str(self.cache_key)
    
    def get_tensor_view(self, device: str = "cuda") -> torch.Tensor:
        """
        获取KV cache tensor视图
        由于kv_cache_tensor已经是VRAM中的tensor，直接返回即可
        """
        return self.kv_cache_tensor
    
    def update_access_time(self) -> None:
        """更新最后访问时间"""
        self.last_access_time = time.time()
    
    def get_metadata(self) -> dict:
        """获取增强的元数据"""
        return {
            "cache_key": self.cache_key_str,
            "token_ids": self.token_ids,
            "gpu_id": self.gpu_id,
            "segment_id": self.segment_id,
            "segment_offset": self.segment_offset,
            "allocated_size": self.allocated_size,
            "segment_base_address": hex(self.segment_base_address),
            "vram_address": hex(self.vram_address),
            "buffer_pointer": hex(self.buffer_pointer),
            "tensor_shape": self.tensor_shape,
            "tensor_dtype": str(self.tensor_dtype),
            "tensor_size": self.tensor_size,
            "buffer_size": self.buffer_size,
            "last_access_time": self.last_access_time
        }

