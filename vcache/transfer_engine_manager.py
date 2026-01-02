
from dataclasses import dataclass
import threading
import time
import os
import torch
from lmcache.logging import init_logger

logger = init_logger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("CuPy is available")
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available")

# NVLINK Transfer Engine imports
try:
    from vcache.nvlink_transfer_engine import DistributedNVLINKTransferEngine
    TRANSFER_ENGINE_AVAILABLE = True
    logger.info("NVLINK transfer engine is available")
except ImportError:
    TRANSFER_ENGINE_AVAILABLE = False
    logger.warning("NVLINK transfer engine not available, cross-GPU transfers disabled")

class TransferEngineManager:
    """Manages NVLINK transfer engine for cross-GPU data transfers."""
    
    def __init__(self, config, ipc_client=None):
        self.config = config
        self.ipc_client = ipc_client
        self.engine = None
        self.initialized = False
        self.lock = threading.RLock()
        
        self._initialize_transfer_engine()

    def _initialize_transfer_engine(self):
        """Initialize NVLINK transfer engine if available."""
        if not TRANSFER_ENGINE_AVAILABLE:
            logger.warning("Transfer engine not available, cross-GPU transfers disabled")
            return
        
        try:
            # Get GPU ID from config
            gpu_id = self.config.get_extra_config_value("gpu_id", 0)
            
            # Initialize NVLINK transfer engine, pass IPC client if provided
            self.engine = DistributedNVLINKTransferEngine(
                config=self.config,
                gpu_id=gpu_id,
                segment_manager=None,  # NVLINK engine doesn't need segment registration
                ipc_client=self.ipc_client
            )
            
            self.initialized = True
            logger.info(f"NVLINK transfer engine initialized successfully for GPU {gpu_id}")
                
        except Exception as e:
            logger.error(f"Error initializing NVLINK transfer engine: {e}")

    
    def transfer_gpu_to_gpu(self,  
                            source_gpu: int, 
                            target_gpu: int, 
                            source_buffer, 
                            target_buffer, 
                            size: int,
                            src_offset: int = 0,
                            dst_offset: int = 0) -> bool:
        """Transfer data between GPUs using NVLINK transfer engine.

        ipc_handle: when transferring from a remote process allocation,
        pass the serialized cudaIpcMemHandle bytes so the engine can open the handle
        locally and perform the transfer.
        """
        if not self.initialized or not self.engine:
            logger.warning("Transfer engine not available for cross-GPU transfer")
            return False
        
        try:    
            # Convert buffer addresses to integers if they're not already
            source_address = source_buffer if isinstance(source_buffer, int) else int(source_buffer)
            target_address = target_buffer if isinstance(target_buffer, int) else int(target_buffer)
            
            # If an IPC client is available, try to get an IPC mem handle for the source buffer
            ipc_handle = None
            if self.ipc_client is not None:
                try:
                    # Pass GPU ID to avoid conflicts between different GPUs with the same address
                    res = self.ipc_client.get_ipc_mem_handle(source_address, source_gpu)
                    if res:
                        handle_bytes, handle_gpu_id, base_pointer, segment_size = res
                        if handle_gpu_id != source_gpu:
                            logger.warning(f"IPC handle GPU id ({handle_gpu_id}) does not match requested source_gpu ({source_gpu})")
                        ipc_handle = handle_bytes
                        logger.debug(f"Obtained IPC mem handle for source buffer {hex(source_address)} on GPU {source_gpu} via IPC client: segment base {hex(base_pointer)}, size {segment_size}")
                except Exception as e:
                    logger.error(f"Failed to obtain IPC mem handle via IPC client: {e}")

            # Perform synchronous transfer using NVLINK engine
            success = self.engine.transfer_sync(
                source_gpu=source_gpu,
                target_gpu=target_gpu,
                source_address=source_address,
                target_address=target_address,
                size=size,
                ipc_handle=ipc_handle,
                src_offset=src_offset,
                dst_offset=dst_offset
            )
            
            if success:
                logger.info(f"Cross-GPU transfer successful: "
                            f"GPU {source_gpu} -> GPU {target_gpu}, "
                            f"size: {size} bytes")
            else:
                logger.error(f"Cross-GPU transfer failed: "
                             f"GPU {source_gpu} -> GPU {target_gpu}")
            
            return success
                
        except Exception as e:
            logger.error(f"Exception during cross-GPU transfer: {e}")
            return False
    def shutdown(self) -> bool:
        """
        Shutdown the transfer engine and release all resources.
        This should be called when the program is exiting.
        
        Returns:
            True if shutdown successful, False otherwise
        """
        with self.lock:
            logger.info("Shutting down transfer engine")
            
            if not self.initialized or not self.engine:
                logger.info("Transfer engine not initialized, nothing to shutdown")
                return True
            
            try:
                # Call engine's shutdown method if it exists
                if hasattr(self.engine, 'shutdown'):
                    self.engine.shutdown()
                    logger.info("Transfer engine shutdown completed successfully")
                else:
                    # If no shutdown method, just log and continue
                    logger.info("Transfer engine does not have shutdown method, skipping")
                
                # Reset engine state
                self.engine = None
                self.initialized = False
                
                logger.info("Transfer engine shutdown completed")
                return True
                
            except Exception as e:
                logger.error(f"Error during transfer engine shutdown: {e}")
                return False
