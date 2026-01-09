"""
Mooncake Transfer Engine implementation for TransferEngineInterface.
This engine uses Mooncake's transfer engine for cross-GPU data transfers.
"""


from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import threading
import time
import os
import torch

from lmcache.vcache.vcache_logging import init_logger
from lmcache.vcache.transfer_engine_interface import TransferEngineInterface

logger = init_logger(__name__)

# Mooncake imports
try:
    from mooncake.engine import TransferEngine
    TRANSFER_ENGINE_AVAILABLE = True
    logger.info("Mooncake transfer engine is available")
except ImportError:
    TRANSFER_ENGINE_AVAILABLE = False
    logger.warning("Mooncake transfer engine not available, cross-GPU transfers disabled")

class MooncakeTransferEngine(TransferEngineInterface):
    """
    Mooncake Transfer Engine implementation for TransferEngineInterface.
    
    This engine uses Mooncake's transfer engine for cross-GPU data transfers
    """
    
    def __init__(self, 
        config, 
        gpu_id: int, 
        ipc_client=None
    ):
        """
        Initialize Mooncake transfer engine for a specific GPU.
        
        Args:
            config: Configuration object
            gpu_id: GPU ID for this engine (0, 1, etc.)
            ipc_client: Optional VRAM metadata IPC client used to request IPC handles
        """
        self.config = config
        self.gpu_id = gpu_id
        self.ipc_client = ipc_client
        
        self.lock = threading.RLock()
        self.engine = None
        self.initialized = False
        
        # Registered memory addresses for cleanup
        self._registered_segments: List[int] = []
        
        # Initialize the engine
        self._initialize_transfer_engine()
        
        logger.info(f"Mooncake Transfer Engine initialized for GPU {gpu_id}. "
                    f"Mooncake available: {TRANSFER_ENGINE_AVAILABLE}")

    def _initialize_transfer_engine(self):
        """Initialize Mooncake transfer engine if available."""
        if not TRANSFER_ENGINE_AVAILABLE:
            return
        
        try:
            self.engine = TransferEngine()
            
            # Get configuration from config
            local_hostname_TE = self.config.get_extra_config_value("local_hostname_TE", "localhost")
            metadata_server = self.config.get_extra_config_value("metadata_server", "http://localhost:8080/metadata")
            protocol_TE = self.config.get_extra_config_value("protocol_TE", "nvlink")
            device_name = self.config.get_extra_config_value("device_name", "")

            # Set environment variables for Mooncake
            os.environ['MC_LEGACY_RPC_PORT_BINDING'] = '1'
            os.environ['MC_FORCE_MNNVL'] = '1'
            os.environ['MC_USE_NVLINK_IPC'] = '1'
            
            ret = self.engine.initialize(
                local_hostname_TE,
                metadata_server,
                protocol_TE,
                device_name
            )
            
            # Reset environment variable
            os.environ['MC_FORCE_MNNVL'] = '0'
            
            if ret == 0:
                self.initialized = True
            else:
                logger.error(f"Failed to initialize Mooncake transfer engine: {ret}")
                
        except Exception as e:
            logger.error(f"Error initializing Mooncake transfer engine: {e}")

    def register_segment(self, 
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
        if not self.initialized or not self.engine:
            logger.warning("Transfer engine not initialized, cannot register segment")
            return False
        
        try:
            # Register the segment memory with transfer engine
            ret = self.engine.register_memory(base_address, size)
            
            if ret == 0:
                logger.info(f"Successfully registered segment {segment_id}, "
                             f"GPU {gpu_id}: "
                             f"address={hex(base_address)}, "
                             f"size={size} bytes")
                
                # Save base_address for cleanup during shutdown
                with self.lock:
                    self._registered_segments.append(base_address)
                
                return True
            else:
                logger.error(f"Failed to register segment {segment_id}"
                             f" on GPU {gpu_id}: "
                             f" error code {ret}")
                return False
                
        except Exception as e:
            logger.error(f"Exception registering segment {segment_id}: {e}")
            return False
    
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
        if not self.initialized or not self.engine:
            logger.warning("Transfer engine not initialized, cannot unregister segment")
            return False
        
        try:
            # Remove from registered segments list if present
            with self.lock:
                if base_address in self._registered_segments:
                    self._registered_segments.remove(base_address)
            
            # Unregister the memory from transfer engine
            ret = self.engine.unregister_memory(base_address)
            
            if ret == 0:
                logger.info(f"Successfully unregistered segment {segment_id}, "
                            f"GPU {gpu_id}: "
                            f"address={hex(base_address)}")
                return True
            else:
                logger.error(f"Failed to unregister segment {segment_id} "
                             f"on GPU {gpu_id}: "
                             f"error code {ret}")
                return False
                
        except Exception as e:
            logger.error(f"Exception unregistering segment {segment_id}: {e}")
            return False
    
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
        Transfer data between GPUs using Mooncake transfer engine.
        
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
        if not self.initialized or not self.engine:
            logger.warning("Mooncake transfer engine not available for cross-GPU transfer")
            return False
        
        logger.info(f"Transferring from host:{src_hostname}, "
                    f"GPU{source_gpu}:{hex(source_buffer)} "
                    f"to host:{target_hostname}, "
                    f"GPU{target_gpu}:{hex(target_buffer)}, "
                    f"size: {size} bytes")
        
        try:
            # Ensure proper device context for IPC operations
            original_device = torch.cuda.current_device()
            
            # Set device to target GPU
            torch.cuda.set_device(target_gpu)
            
            local_hostname = self.config.get_extra_config_value("local_hostname_TE", "localhost")
            
            # Perform synchronous transfer - this reads from remote GPU to local GPU
            start_time = time.time()
            
            # Determine transfer direction based on hostnames
            if src_hostname != local_hostname:
                # Remote to local transfer
                ret = self.engine.transfer_sync_read(
                    src_hostname,      # Remote hostname where source data resides
                    target_buffer,     # Local buffer to receive data
                    source_buffer,     # Remote buffer to read from
                    size               # Size of data to transfer
                )
            else:
                # Local to remote transfer
                ret = self.engine.transfer_sync_write(
                    target_hostname,   # Remote hostname where target data resides
                    source_buffer,     # Local buffer to send data from
                    target_buffer,     # Remote buffer to write to
                    size               # Size of data to transfer
                )
            
            transfer_time = time.time() - start_time
            
            # Restore original device
            torch.cuda.set_device(original_device)
            
            if ret == 0:
                logger.info(f"Cross-GPU transfer successful: "
                            f"{src_hostname}:GPU{source_gpu} -> {target_hostname}:GPU{target_gpu}, "
                            f"size: {size} bytes, time: {transfer_time:.3f}s")
                return True
            else:
                logger.error(f"Cross-GPU transfer failed: "
                             f"{src_hostname}:GPU{source_gpu} -> {target_hostname}:GPU{target_gpu}, "
                            f"error code: {ret}")
                return False
                
        except Exception as e:
            logger.error(f"Exception during cross-GPU transfer: {e}")
            # Restore original device in case of error
            try:
                torch.cuda.set_device(original_device)
            except:
                pass
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get transfer engine status.
        
        Returns:
            Dictionary containing engine status and health information
        """
        with self.lock:
            status = {}
            
            # Add engine status information
            status['engine_type'] = 'MooncakeTransferEngine'
            status['initialized'] = self.initialized
            status['mooncake_available'] = TRANSFER_ENGINE_AVAILABLE
            status['gpu_id'] = self.gpu_id
            
            # Add registered segments count
            status['registered_segments'] = len(self._registered_segments)
            
            # Add configuration info
            status['local_hostname'] = self.config.get_extra_config_value("local_hostname_TE", "localhost")
            status['protocol'] = self.config.get_extra_config_value("protocol_TE", "nvlink")
            
            return status

    def shutdown(self) -> bool:
        """
        Shutdown the transfer engine and release all resources.
        
        Returns:
            True if shutdown successful, False otherwise
        """
        with self.lock:
            logger.info("Shutting down Mooncake transfer engine")
            
            if not self.initialized or not self.engine:
                logger.info("Mooncake transfer engine not initialized, nothing to shutdown")
                return True
            
            try:
                # Unregister all registered memory addresses first
                for base_address in self._registered_segments:
                    try:
                        self.engine.unregister_memory(base_address)
                        logger.debug(f"Unregistered memory at "
                                     f"address {hex(base_address)} from transfer engine")
                    except Exception as e:
                        logger.warning(f"Failed to unregister memory at "
                                       f"address {hex(base_address)}: {e}")
                self._registered_segments.clear()
                
                # If transfer engine has a shutdown method, call it
                if hasattr(self.engine, 'shutdown'):
                    ret = self.engine.shutdown()
                    if ret == 0:
                        logger.debug("Mooncake transfer engine shutdown completed successfully")
                    else:
                        logger.error(f"Mooncake transfer engine shutdown failed with error code: {ret}")
                        return False
                
                # Reset engine state
                self.engine = None
                self.initialized = False
                
                logger.info("Mooncake transfer engine shutdown completed")
                return True
                
            except Exception as e:
                logger.error(f"Error during Mooncake transfer engine shutdown: {e}")
                return False

    # Optional convenience methods with default implementations
    
    def is_available(self) -> bool:
        """
        Check if the transfer engine is available and ready for transfers.
        
        Returns:
            True if engine is available, False otherwise
        """
        return self.initialized and TRANSFER_ENGINE_AVAILABLE
    
    def __del__(self):
        """Destructor to ensure proper shutdown."""
        if hasattr(self, 'initialized') and self.initialized:
            self.shutdown()
