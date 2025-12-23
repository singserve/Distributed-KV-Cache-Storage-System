"""
Test Cache Engine Token Database
基于LMCache前缀缓存机制的token处理类
实现了前缀哈希链机制，支持vLLM兼容的前缀缓存
"""

from typing import List, Optional, Tuple, Union, Iterable, Any
import torch
import os


class TestTokenDatabase:
    """
    Test Cache Engine的token处理类
    基于LMCache的ChunkedTokenDatabase实现前缀缓存机制
    """
    
    def __init__(self, chunk_size: int = 256, save_unfull_chunk: bool = True):
        """
        初始化TestTokenDatabase
        
        Args:
            chunk_size: token分块大小
            save_unfull_chunk: 是否保存不完整的chunk
        """
        self.chunk_size = chunk_size
        self.save_unfull_chunk = save_unfull_chunk
        
        # 使用前缀哈希链机制
        self.hash_func = self._get_hash_function()
        
        # 初始化NONE_HASH（类似LMCache的实现）
        self.NONE_HASH = self._init_none_hash()
        
        # 检查分布式一致性设置
        self._check_distributed_consistency()
    
    def _get_hash_function(self):
        """
        获取哈希函数，支持vLLM兼容性
        
        Returns:
            哈希函数
        """
        # 尝试从vLLM导入哈希函数
        try:
            # 优先尝试vLLM >= PR#27151的路径
            from vllm.utils.hashing import get_hash_fn_by_name
            return get_hash_fn_by_name("sha256_cbor")
        except ImportError:
            try:
                # 尝试vLLM < PR#27151的路径
                from vllm.utils import get_hash_fn_by_name
                return get_hash_fn_by_name("sha256_cbor")
            except ImportError:
                # 回退到内置哈希函数
                self._warn_hash_consistency()
                return hash
    
    def _init_none_hash(self) -> int:
        """
        初始化NONE_HASH，类似LMCache的实现
        
        Returns:
            NONE_HASH值
        """
        # 为了测试一致性，总是返回0
        # 这样可以确保store和retrieve时使用相同的NONE_HASH
        return 0
    
    def _check_distributed_consistency(self):
        """
        检查分布式一致性设置
        """
        if os.getenv("PYTHONHASHSEED") is None:
            print("WARNING: Using builtin hash without PYTHONHASHSEED set. "
                  "For production environments, set PYTHONHASHSEED=0 "
                  "to ensure consistent hashing across processes.")
    
    def _warn_hash_consistency(self):
        """
        哈希一致性警告
        """
        print("WARNING: Using builtin hash function. "
              "For vLLM compatibility, install vLLM and use sha256_cbor hash.")
    
    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        **kwargs
    ) -> Iterable[Tuple[int, int, Optional[Any]]]:
        """
        处理tokens并生成cache key，实现前缀缓存机制
        以chunk为粒度，每个chunk和之前的chunks作为一个前缀
        
        Args:
            tokens: 输入的tokens
            mask: token mask，True表示需要处理的token
            make_key: 是否生成cache key，False则返回hash值
            
        Returns:
            生成器，返回(start, end, key/hash)元组，如果chunk的mask为false则key为None
            - 当make_key=True时，key是CacheEngineKey对象
            - 当make_key=False时，key是int（哈希值）
            - 当chunk被跳过时，key是None
        """
        if tokens is None:
            raise ValueError("tokens must be provided")
        
        # 将tokens转换为列表
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.cpu().tolist()
        else:
            token_list = tokens
        
        total_len = len(token_list)
        
        # 使用前缀哈希链处理tokens
        prefix_hash = self.NONE_HASH  # 从NONE_HASH开始
        chunk_generator = self._chunk_tokens(token_list)
        
        for chunk_id, chunk_tokens in enumerate(chunk_generator):
            chunk_start_idx = chunk_id * self.chunk_size
            chunk_end_idx = chunk_start_idx + len(chunk_tokens)
            
            # 检查当前chunk的mask情况
            chunk_has_valid_tokens = True
            if mask is not None:
                # 获取当前chunk对应的mask部分
                chunk_mask = mask[chunk_start_idx:chunk_end_idx]
                # 如果chunk中所有token的mask都为False，则跳过该chunk
                if not chunk_mask.any():
                    chunk_has_valid_tokens = False
            
            # 即使chunk被mask跳过，也要更新prefix_hash以保持哈希链的一致性
            # 这是为了确保store和retrieve使用相同的哈希链
            prefix_hash = self._hash_tokens(chunk_tokens, prefix_hash)
            
            # 如果chunk没有有效token，跳过生成key
            if not chunk_has_valid_tokens:
                # 返回None作为key，表示该chunk不需要处理
                yield chunk_start_idx, chunk_end_idx, None
                continue
            
            # 当前chunk的hash就是prefix_hash（已经包含了之前所有chunks的哈希链）
            current_prefix_hash = prefix_hash
            
            # 返回chunk的范围，而不是前缀的范围
            # chunk的范围是chunk在原始tokens中的起始和结束位置
            start_idx = chunk_start_idx
            end_idx = chunk_end_idx
            
            if make_key:
                # 生成cache key
                cache_key = self._make_cache_key(current_prefix_hash, **kwargs)
                yield start_idx, end_idx, cache_key
            else:
                yield start_idx, end_idx, current_prefix_hash
            
            # 注意：这里不重置prefix_hash，保持链式结构
            # 下一个chunk会继续使用当前的prefix_hash作为前缀
    
    
    def _chunk_tokens(self, tokens: List[int]) -> Iterable[List[int]]:
        """
        将tokens分块，类似LMCache的实现
        
        Args:
            tokens: 输入的token列表
            
        Returns:
            生成器，返回每个chunk的token列表
        """
        if self.save_unfull_chunk:
            end = len(tokens)
        else:
            end = len(tokens) - (len(tokens) % self.chunk_size)
        
        for i in range(0, end, self.chunk_size):
            yield tokens[i:i + self.chunk_size]
    
    
    def _hash_tokens(self, tokens: List[int], prefix_hash: Optional[int] = None) -> int:
        """
        计算tokens的哈希值，包含前缀信息
        
        Args:
            tokens: token列表
            prefix_hash: 前缀哈希值
            
        Returns:
            哈希值
        """
        tokens_tuple = tuple(tokens)
        return self.hash_func((prefix_hash, tokens_tuple))
    
    def _make_cache_key(self, chunk_hash: int, **kwargs) -> "CacheEngineKey":
        """
        根据哈希值生成cache key，使用LMCache的CacheEngineKey格式
        
        Args:
            chunk_hash: chunk的哈希值
            **kwargs: 其他参数
            
        Returns:
            CacheEngineKey对象
        """
        model_name = kwargs.get('model_name', 'test_model')
        # 使用固定的worker_id=0，确保相同chunk在不同GPU之间可以共享
        # 数据不应该绑定到特定的worker_id
        worker_id = 0  # 固定为0，而不是从kwargs获取
        world_size = kwargs.get('world_size', 1)
        kv_dtype_str = kwargs.get('kv_dtype', 'float16')
        
        # 导入CacheEngineKey
        from lmcache.utils import CacheEngineKey, STR_DTYPE_TO_TORCH_DTYPE
        
        # 转换dtype字符串为torch.dtype
        kv_dtype = STR_DTYPE_TO_TORCH_DTYPE.get(kv_dtype_str, torch.float16)
        
        # 创建CacheEngineKey对象
        # fmt参数通常是"vllm"或"lmcache"，这里使用"vllm"表示vLLM格式
        cache_key = CacheEngineKey(
            fmt="vllm",  # 使用vLLM格式
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,  # 固定为0
            chunk_hash=chunk_hash,
            dtype=kv_dtype,
            request_configs=None
        )
        
        return cache_key
    
    def lookup_prefix(self, tokens: Union[torch.Tensor, List[int]]) -> int:
        """
        前缀缓存查找，返回最长连续命中的前缀长度
        
        Args:
            tokens: 输入的tokens
            
        Returns:
            最长连续命中的前缀长度
        """
        # 这个函数需要与存储后端配合使用
        # 这里返回模拟的前缀命中长度
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.cpu().tolist()
        else:
            token_list = tokens
        
        # 模拟前缀缓存查找逻辑
        prefix_hash = self.NONE_HASH
        max_hit_length = 0
        
        for chunk_tokens in self._chunk_tokens(token_list):
            prefix_hash = self._hash_tokens(chunk_tokens, prefix_hash)
            
            # 这里应该检查存储后端是否包含这个key
            # 为了测试目的，我们假设第一个chunk总是命中
            if max_hit_length == 0:
                max_hit_length = len(chunk_tokens)
            else:
                # 模拟后续chunk的检查
                break
        
        return max_hit_length
