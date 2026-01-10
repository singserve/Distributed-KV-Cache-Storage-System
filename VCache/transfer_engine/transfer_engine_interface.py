"""
Transfer Engine Interface

This module defines the interface for transfer engines that handle GPU-to-GPU data transfers.
Different implementations (NVLINK, Mooncake, etc.) should implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple


class TransferEngineInterface(ABC):
    """
    Abstract base class for transfer engines.
    
    This interface defines the common API for all transfer engine implementations.
    Implementations should provide concrete methods for GPU-to-GPU data transfer,
    status monitoring, and resource management.
    """
    
    @abstractmethod
    def transfer_gpu_to_gpu(
        self,
        source_gpu: int,
        target_gpu: int,
        source_buffer: int,
        target_buffer: int,
        size: int,
        src_hostname: str = "localhost",
        target_hostname: str = "localhost",
        src_offset: int = 0,
        dst_offset: int = 0,
        **kwargs
    ) -> bool:
        """
        Transfer data between GPUs.
        
        Args:
            source_gpu: Source GPU ID
            target_gpu: Target GPU ID
            source_buffer: Source GPU memory address (int) or buffer object
            target_buffer: Target GPU memory address (int) or buffer object
            size: Size to transfer in bytes
            src_hostname: Hostname where source GPU is located (default: "localhost")
            target_hostname: Hostname where target GPU is located (default: "localhost")
            src_offset: Offset into source buffer (bytes)
            dst_offset: Offset into destination buffer (bytes)
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            True if transfer successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get transfer engine status and statistics.
        
        Returns:
            Dictionary containing engine status, statistics, and health information
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        Shutdown the transfer engine and release all resources.
        
        Returns:
            True if shutdown successful, False otherwise
        """
        pass
    
    @abstractmethod
    def register_segment(
        self, 
        segment_id: str, 
        base_address: int, 
        gpu_id: int, 
        size: int
    ) -> bool:
        """
        Register a GPU memory segment with the transfer engine.
        
        Args:
            segment_id: Unique segment identifier
            base_address: Base GPU memory address of the segment
            gpu_id: GPU device ID
            size: Segment size in bytes
            
        Returns:
            True if registration successful, False otherwise
        """
        pass
    
    @abstractmethod
    def unregister_segment(
        self, 
        segment_id: str, 
        base_address: int, 
        gpu_id: int
    ) -> bool:
        """
        Unregister a GPU memory segment from the transfer engine.
        
        Args:
            segment_id: Unique segment identifier
            base_address: Base GPU memory address of the segment
            gpu_id: GPU device ID
            
        Returns:
            True if unregistration successful, False otherwise
        """
        pass
    
    # Optional convenience methods with default implementations
    
    def is_available(self) -> bool:
        """
        Check if the transfer engine is available and ready for transfers.
        
        Returns:
            True if engine is available, False otherwise
        """
        return True
    
