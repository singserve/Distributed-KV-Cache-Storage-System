"""
VCache Utilities 
"""

from typing import Optional, Any, Dict
import torch


class VCacheKey:
    """
    Simplified cache key structure 
    
    Attributes:
        fmt: Format identifier (e.g., 'vllm', 'pt')
        model_name: Name of the model
        chunk_hash: Hash of the chunk/prefix
        dtype: Data type of the KV cache (torch.dtype)
    """
    
    def __init__(
        self,
        fmt: str = "vllm",
        model_name: str = "test_model",
        chunk_hash: Any = 0,
        dtype: torch.dtype = torch.float16,
    ):
        self.fmt = fmt
        self.model_name = model_name
        self.chunk_hash = chunk_hash
        self.dtype = dtype
    
    def to_string(self) -> str:
        """Convert key to string representation."""
        # Create a deterministic string representation
        dtype_str = dtype_to_str(self.dtype)
        return f"fmt={self.fmt}|model={self.model_name}|hash={self.chunk_hash}|dtype={dtype_str}"
    
    def __repr__(self) -> str:
        dtype_str = dtype_to_str(self.dtype)
        return (f"VCacheKey(fmt='{self.fmt}', model_name='{self.model_name}', "
                f"chunk_hash={self.chunk_hash}, dtype={dtype_str})")
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, VCacheKey):
            return False
        
        return (self.fmt == other.fmt and
                self.model_name == other.model_name and
                self.chunk_hash == other.chunk_hash and
                self.dtype == other.dtype)
    
    def __hash__(self) -> int:
        """Compute hash for use in dictionaries."""
        return hash((
            self.fmt,
            self.model_name,
            self.chunk_hash,
            self.dtype
        ))
    


def dtype_to_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to string representation."""
    if dtype == torch.float16:
        return "half"
    elif dtype == torch.float32:
        return "float"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    else:
        return str(dtype)


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string representation to torch.dtype."""
    dtype_str = dtype_str.lower()
    if dtype_str in ["half", "float16", "torch.float16"]:
        return torch.float16
    elif dtype_str in ["float", "float32", "torch.float32"]:
        return torch.float32
    elif dtype_str in ["bfloat16", "torch.bfloat16"]:
        return torch.bfloat16
    else:
        # Default to float16 if unknown
        return torch.float16


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b
