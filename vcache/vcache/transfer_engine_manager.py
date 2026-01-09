
from dataclasses import dataclass
import threading
from lmcache.vcache.vcache_logging import init_logger

logger = init_logger(__name__)

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
        """Initialize transfer engine based on configuration."""
        # Get engine type from config
        engine_type = self.config.get_extra_config_value("transfer_engine_type", "nvlink")
        gpu_id = self.config.get_extra_config_value("gpu_id", 0)
        
        logger.info(f"Initializing transfer engine: type={engine_type}, gpu_id={gpu_id}")
        
        # Try to initialize the selected engine
        if engine_type == "nvlink":
            try:
                from lmcache.vcache.nvlink_transfer_engine import DistributedNVLINKTransferEngine              
                # Initialize NVLINK transfer engine
                self.engine = DistributedNVLINKTransferEngine(
                    config=self.config,
                    gpu_id=gpu_id,
                    ipc_client=self.ipc_client
                )
                
                self.initialized = True
                logger.info(f"NVLINK transfer engine initialized successfully for GPU {gpu_id}")
                    
            except Exception as e:
                logger.error(f"Error initializing NVLINK transfer engine: {e}")
                raise
        
        elif engine_type == "mooncake":
            try:
                from lmcache.vcache.mooncake_transfer_engine import MooncakeTransferEngine 
                # Initialize Mooncake transfer engine
                self.engine = MooncakeTransferEngine(
                    config=self.config,
                    gpu_id=gpu_id,
                    ipc_client=self.ipc_client
                )
                
                self.initialized = True
                logger.info(f"Mooncake transfer engine initialized successfully for GPU {gpu_id}")
                    
            except Exception as e:
                logger.error(f"Error initializing Mooncake transfer engine: {e}")
                raise
        
        else:
            error_msg = f"Unknown transfer engine type: {engine_type}. Supported types: 'nvlink', 'mooncake'"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def register_segment(self, segment_id: str, base_address: int, gpu_id: int, size: int) -> bool:
        """
        Register a GPU memory segment with the metadata server
        
        Args:
            segment_id: Unique segment identifier
            base_address: Base GPU memory address of the segment
            gpu_id: GPU device ID
            size: Segment size in bytes
            
        Returns:
            True if registration successful, False otherwise
        """
        if not self.initialized or not self.engine:
            logger.warning("Transfer engine not available, cannot register segment")
            return False
        
        return self.engine.register_segment(segment_id, base_address, gpu_id, size)

    
    def unregister_segment(self, segment_id: str, base_address: int, gpu_id: int) -> bool:
        """
        Unregister a GPU memory segment from the metadata server.
        
        Args:
            segment_id: Unique segment identifier
            base_address: Base GPU memory address of the segment
            gpu_id: GPU device ID
            
        Returns:
            True if unregistration successful, False otherwise
        """
        if not self.initialized or not self.engine:
            logger.warning("Transfer engine not available, cannot unregister segment")
            return False
        return self.engine.unregister_segment(segment_id, base_address, gpu_id)


    
    def transfer_gpu_to_gpu(
        self,  
        source_gpu: int, 
        target_gpu: int, 
        source_buffer, 
        target_buffer, 
        size: int,
        src_hostname: str = "localhost",
        target_hostname: str = "localhost",
        src_offset: int = 0,
        dst_offset: int = 0,
        **kwargs
    ) -> bool:
        """Transfer data between GPUs using transfer engine.

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
                        ipc_handle = handle_bytes
                        logger.debug(f"Obtained IPC mem handle for source buffer {hex(source_address)} "
                                     f"on GPU {source_gpu} via IPC client: "
                                     f"segment base {hex(base_pointer)}, "
                                     f"size {segment_size}")
                except Exception as e:
                    logger.error(f"Failed to obtain IPC mem handle via IPC client: {e}")

            # Add ipc_handle to kwargs if available
            if ipc_handle is not None:
                kwargs['ipc_handle'] = ipc_handle

            # Check if engine has the new transfer_gpu_to_gpu method (with hostname parameters)
            # Use the new interface with hostname parameters
            success = self.engine.transfer_gpu_to_gpu(
                source_gpu=source_gpu,
                target_gpu=target_gpu,
                source_buffer=source_address,
                target_buffer=target_address,
                size=size,
                src_hostname=src_hostname,
                target_hostname=target_hostname,
                src_offset=src_offset,
                dst_offset=dst_offset,
                **kwargs
            )

            if success:
                logger.info(f"Cross-GPU transfer successful: "
                            f"{src_hostname}:GPU {source_gpu} -> {target_hostname}:GPU {target_gpu}, "
                            f"size: {size} bytes")
            else:
                logger.error(f"Cross-GPU transfer failed: "
                             f"{src_hostname}:GPU {source_gpu} -> {target_hostname}:GPU {target_gpu}")
            
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
