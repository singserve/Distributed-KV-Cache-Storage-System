from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch
import time


from typing import Union
from lmcache.utils import CacheEngineKey

@dataclass
class VRAMKVCacheUnit:
    """
    Data unit to store KVCache chunk in GPU VRAM segment
    """
    
    # cache key and token ids
    cache_key: Union[str, CacheEngineKey]
    token_ids: List[int]
    
    # GPU VRAM segment info
    segment_id: str
    segment_offset: int 
    allocated_size: int 
    segment_base_address: int
    
    # KV cache tensor: data pointer
    kv_cache_tensor: torch.Tensor
    
    gpu_id: int
    
    # original shape for reshaping
    original_shape: Optional[Tuple[int, ...]] = None
    
    last_access_time: float = field(default_factory=time.time)
    
    @property
    def buffer_pointer(self) -> int:
        return self.kv_cache_tensor.data_ptr()
    
    @property
    def buffer_size(self) -> int:
        return self.kv_cache_tensor.element_size() * self.kv_cache_tensor.nelement()
    
    @property
    def tensor_shape(self) -> Tuple[int, ...]:
        return tuple(self.kv_cache_tensor.shape)
    
    @property
    def tensor_dtype(self) -> torch.dtype:
        return self.kv_cache_tensor.dtype
    
    @property
    def vram_address(self) -> int:
        return self.segment_base_address + self.segment_offset
    
    @property
    def cache_key_str(self) -> str:
        if isinstance(self.cache_key, CacheEngineKey):
            return self.cache_key.to_string()
        return str(self.cache_key)
    
    def get_tensor_view(self, device: str = "cuda") -> torch.Tensor:
        return self.kv_cache_tensor
    
    def update_access_time(self) -> None:
        self.last_access_time = time.time()
    
    def get_metadata(self) -> dict:
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
            "buffer_size": self.buffer_size,
            "last_access_time": self.last_access_time
        }
