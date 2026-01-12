"""
Distributed NVLINK Transfer Engine for direct GPU-to-GPU data transfer.
Each VCacheEngine instance has its own transfer engine that can directly
transfer data between GPU memory addresses using NVLINK.

"""

import threading
import time
import torch
import ctypes
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from lmcache.vcache.logging.vcache_logging import init_logger
from lmcache.vcache.transfer_engine.transfer_engine_interface import TransferEngineInterface

logger = init_logger(__name__)

@dataclass
class TransferRequest:

    """Transfer request for NVLINK transfer."""
    request_id: str
    source_gpu: int
    target_gpu: int
    source_address: int 
    target_address: int 
    size: int  # Size in bytes
    ipc_handle: Optional[bytes] = None  # Serialized cudaIpcMemHandle if source is remote
    src_offset: int = 0  # offset into source buffer (bytes)
    dst_offset: int = 0  # offset into destination buffer (bytes)
    sync: bool = True  # Whether to wait for transfer completion
    status: str = "pending"  # pending, in_progress, submitted, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    completion_event: Optional[torch.cuda.Event] = None  # CUDA event for async transfer completion


class DistributedNVLINKTransferEngine(TransferEngineInterface):
    """
    Distributed NVLINK Transfer Engine for direct GPU-to-GPU data transfer.
    
    Each VCacheEngine instance has its own transfer engine that can:
    1. Perform direct GPU-to-GPU transfers using NVLINK
    2. Manage local transfer queues and statistics
    """
    
    def __init__(
        self, 
        config, 
        gpu_id: int, 
        ipc_client=None
    ):
        """
        Initialize distributed NVLINK transfer engine for a specific GPU.
        
        Args:
            config: Configuration object
            gpu_id: GPU ID for this engine (0, 1, etc.)
            ipc_client: Optional VRAM metadata IPC client used to request IPC handles
        """
        self.config = config
        self.gpu_id = gpu_id
        self.ipc_client = ipc_client
        
        self.lock = threading.RLock()
        
        self.cuda_available = torch.cuda.is_available()
        self.nvlink_available = self._check_nvlink_availability()
        
        # Native CUDA library for NVLINK transfers
        self.cuda_lib = None
        self._load_cuda_library()
        
        # Transfer queue management
        self.transfer_queue: List[TransferRequest] = []
        self.active_transfers: Dict[str, TransferRequest] = {}
        self.completed_transfers: Dict[str, TransferRequest] = {}
        
        # Statistics tracking
        self.stats = {
            "successful_transfers": 0,
            "failed_transfers": 0,
            "total_transfer_bytes": 0,
            "successful_registers": 0,
            "failed_registers": 0,
            "successful_unregisters": 0,
            "failed_unregisters": 0
        }
        
        # Configuration
        self.max_concurrent_transfers = config.get_extra_config_value(
            "max_concurrent_transfers", 2
        )
        self.transfer_timeout_sec = config.get_extra_config_value(
            "transfer_timeout_sec", 30.0
        )
        
        # Initialize transfer worker thread
        self._worker_thread = None
        self._worker_running = False
        self._worker_event = threading.Event()
        
        # Initialize the engine
        self._initialize_engine()
        
        logger.info(f"Distributed NVLINK Transfer Engine initialized for GPU {gpu_id}. "
                   f"NVLINK available: {self.nvlink_available}")

    
    def _load_cuda_library(self):
        """Load the native CUDA library for NVLINK transfers."""
        if not self.cuda_available:
            logger.warning("CUDA not available, cannot load CUDA library")
            return
        
        try:
            import ctypes
            libname = "./native/build/libnvlink_transfer.so"
            self.cuda_lib = ctypes.CDLL(libname)
            logger.debug(f"Loaded native nvlink helper: {libname}")
            
            # Set up function signatures
            self.cuda_lib.perform_direct_transfer.argtypes = [
                ctypes.c_int,          # source_gpu
                ctypes.c_ulonglong,    # source_ptr
                ctypes.c_int,          # target_gpu
                ctypes.c_ulonglong,    # target_ptr
                ctypes.c_ulonglong,    # size
                ctypes.c_void_p,       # src_ipc_handle (pointer or None)
                ctypes.c_uint,         # src_ipc_handle_size
                ctypes.c_void_p,       # dst_ipc_handle
                ctypes.c_uint,         # dst_ipc_handle_size
                ctypes.c_ulonglong,    # src_offset
                ctypes.c_ulonglong,    # dst_offset
                ctypes.c_void_p        # stream
            ]
            self.cuda_lib.perform_direct_transfer.restype = ctypes.c_int
            
            # Set up error string function
            self.cuda_lib.cudaGetErrorString.argtypes = [ctypes.c_int]
            self.cuda_lib.cudaGetErrorString.restype = ctypes.c_char_p
            
        except Exception as e:
            logger.error(f"Native nvlink helper not available: {e}")
            self.cuda_lib = None
    
    def _check_nvlink_availability(self) -> bool:
        """
        Check if NVLINK is available between GPUs.
        
        Returns:
            True if NVLINK is available, False otherwise
        """
        if not self.cuda_available:
            logger.warning("CUDA not available, NVLINK transfers disabled")
            return False
        
        try:
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2:
                logger.warning(f"Only {num_gpus} GPU(s) available, "
                               f"NVLINK transfers require at least 2 GPUs")
                return False
            
            # Check peer-to-peer access between GPUs
            nvlink_available = False
            
            for i in range(num_gpus):
                for j in range(i + 1, num_gpus):
                    try:
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        if can_access:
                            logger.debug(f"GPU {i} can access GPU {j} (peer access enabled)")
                            nvlink_available = True
                    except Exception as e:
                        logger.debug(f"Error checking peer access between GPU {i} and {j}: {e}")
            
            if not nvlink_available:
                logger.warning("No peer-to-peer access detected between GPUs.")
            
            return nvlink_available
            
        except Exception as e:
            logger.error(f"Error checking NVLINK availability: {e}")
            return False
    
    def _initialize_engine(self):
        """Initialize the transfer engine and enable peer access between GPUs."""
        if not self.cuda_available:
            logger.warning("CUDA not available, cannot initialize transfer engine")
            return
        
        try:
            num_gpus = torch.cuda.device_count()
            logger.info(f"Initializing NVLINK transfer engine for {num_gpus} GPUs")
            
            # Enable peer access between all GPU pairs
            for i in range(num_gpus):
                for j in range(num_gpus):
                    if i != j:
                        try:
                            # check if peer access is already enabled
                            if torch.cuda.can_device_access_peer(i, j):
                                logger.debug(f"Peer access already enabled from GPU {i} to GPU {j}")
                            else:
                                # enable peer access
                                torch.cuda.set_device(i)
                                torch.cuda.enable_peer_access(j)
                                logger.debug(f"Enabled peer access from GPU {i} to GPU {j}")
                        except RuntimeError as e:
                            logger.warning(f"Cannot enable peer access from GPU {i} to GPU {j}: {e}")
            # Start the transfer worker thread
            self._start_worker_thread()
            
        except Exception as e:
            logger.error(f"Error initializing transfer engine: {e}")
        
    def _start_worker_thread(self):
        """Start the transfer worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("Transfer worker thread already running")
            return
        
        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._transfer_worker,
            name="NVLINKTransferWorker",
            daemon=True
        )
        self._worker_thread.start()
        logger.info("Transfer worker thread started")
    
    def _transfer_worker(self):
        """Worker thread that processes transfer requests from the queue."""
        logger.info("Transfer worker thread started")
        
        while self._worker_running:
            try:
                # Wait for work or timeout
                self._worker_event.wait(timeout=0.1)
                self._worker_event.clear()
                
                # Process transfers if we have capacity
                with self.lock:
                    active_count = len(self.active_transfers)
                    
                    # Start new transfers if we have capacity
                    while (active_count < self.max_concurrent_transfers and 
                           self.transfer_queue):
                        
                        # Get the next transfer request (FIFO)
                        request = self.transfer_queue.pop(0)
                        request.status = "in_progress"
                        request.start_time = time.time()
                        self.active_transfers[request.request_id] = request
                        
                        # Start the transfer in a separate thread
                        transfer_thread = threading.Thread(
                            target=self._execute_transfer,
                            args=(request,),
                            name=f"Transfer-{request.request_id}",
                            daemon=True
                        )
                        transfer_thread.start()
                        
                        active_count += 1
                        logger.debug(f"Started transfer {request.request_id}: "
                                   f"GPU {request.source_gpu} -> GPU {request.target_gpu}, "
                                   f"size: {request.size} bytes")
                
                # Clean up completed transfers
                self._cleanup_completed_transfers()
                
            except Exception as e:
                logger.error(f"Error in transfer worker: {e}")
                time.sleep(1.0)
        
        logger.debug("Transfer worker thread stopped")
    
    def _execute_transfer(self, request: TransferRequest):
        """
        Execute a single transfer request.
        
        Args:
            request: Transfer request to execute
        """
        try:
            # Validate request parameters
            if request.size <= 0:
                raise ValueError(f"Invalid transfer size: {request.size}")
            
            if request.source_gpu == request.target_gpu:
                raise ValueError(f"Source and target GPU are the same: {request.source_gpu}")
            
            # Perform the transfer (asynchronous)
            success, completion_event = self._perform_direct_transfer(
                source_gpu=request.source_gpu,
                target_gpu=request.target_gpu,
                source_address=request.source_address,
                target_address=request.target_address,
                size=request.size,
                ipc_handle=request.ipc_handle,
                src_offset=request.src_offset,
                dst_offset=request.dst_offset,
            )
            
            if not success:
                request.status = "failed"
                request.error_message = "Transfer submission failed"
                request.end_time = time.time()
                logger.error(f"Transfer {request.request_id} submission failed: "
                           f"GPU {request.source_gpu} -> GPU {request.target_gpu}")
                with self.lock:
                    self.stats["failed_transfers"] += 1
                return
            
            # Store completion event for later synchronization
            request.completion_event = completion_event
            
            # If sync is True, wait for completion now
            if request.sync and completion_event is not None:
                completion_event.synchronize()
                request.end_time = time.time()
                request.status = "completed"
                logger.info(f"Transfer {request.request_id} completed synchronously: "
                          f"GPU {request.source_gpu} -> GPU {request.target_gpu}, "
                          f"size: {request.size} bytes, "
                          f"time: {request.end_time - request.start_time:.3f}s")
                with self.lock:
                    self.stats["successful_transfers"] += 1
                    self.stats["total_transfer_bytes"] += request.size
            else:
                # For async transfers, mark as "submitted" not "completed"
                request.status = "submitted"
                logger.info(f"Transfer {request.request_id} submitted asynchronously: "
                          f"GPU {request.source_gpu} -> GPU {request.target_gpu}, "
                          f"size: {request.size} bytes")
                # Async transfers will update stats when they complete via wait_for_transfer
            
        except Exception as e:
            request.status = "failed"
            request.end_time = time.time()
            request.error_message = str(e)
            logger.error(f"Error executing transfer {request.request_id}: {e}")
            with self.lock:
                self.stats["failed_transfers"] += 1
        
        finally:
            # Move from active to completed/submitted
            with self.lock:
                if request.request_id in self.active_transfers:
                    del self.active_transfers[request.request_id]
                
                # Keep only recent completed transfers to avoid memory leak
                if len(self.completed_transfers) > 1000:
                    oldest_ids = sorted(self.completed_transfers.keys())[:100]
                    for old_id in oldest_ids:
                        del self.completed_transfers[old_id]
                
                self.completed_transfers[request.request_id] = request
    
    def _perform_direct_transfer(
        self, 
        source_gpu: int, 
        target_gpu: int, 
        source_address: int, 
        target_address: int, 
        size: int,
        ipc_handle: Optional[bytes] = None,
        src_offset: int = 0,
        dst_offset: int = 0
    ):
        """Perform direct GPU-to-GPU transfer using NVLINK with low-level CUDA APIs.
        This function is asynchronous - it submits the transfer and returns immediately.
        Synchronization is done through the recorded CUDA event.

        Args:
            ipc_handle: serialized cudaIpcMemHandle if is remote
        
        Returns:
            Tuple of (success: bool, completion_event: Optional[torch.cuda.Event])
            If success is False, completion_event is None.
        """
        if not self.cuda_available:
            logger.error("CUDA not available for direct transfer")
            return False, None
        
        if self.cuda_lib is None:
            logger.error("CUDA library not loaded for direct transfer")
            return False, None
        
        original_device = torch.cuda.current_device()
        
        try:
            device_count = torch.cuda.device_count()
            if source_gpu >= device_count or target_gpu >= device_count:
                logger.error(f"Invalid GPU IDs: source_gpu={source_gpu}, "
                             f"target_gpu={target_gpu}")
                return False, None

            # Allow one address to be zero if ipc_handle is provided
            if (target_address == 0 and ipc_handle is None) or \
               (source_address == 0 and ipc_handle is None):
                logger.error(f"Invalid memory address; "
                             f"note: address may be 0 if ipc_handle is provided")
                return False, None
            
            can_access = torch.cuda.can_device_access_peer(source_gpu, target_gpu)
            
            if not can_access:
                logger.error(f"Peer access not enabled "
                             f"from GPU {source_gpu} to GPU {target_gpu}")
                return False, None
            
            torch.cuda.set_device(source_gpu)
            
            stream = torch.cuda.Stream(device=source_gpu)

            # Prepare handle buffer and sizes
            _tmp_src_buf = None
            src_handle_buf = ctypes.c_void_p(0)
            src_handle_size = ctypes.c_uint(0)
            if ipc_handle is not None:
                _tmp_src_buf = ctypes.create_string_buffer(ipc_handle)
                src_handle_buf = ctypes.cast(_tmp_src_buf, ctypes.c_void_p)
                src_handle_size = ctypes.c_uint(len(ipc_handle))

            # Stream pointer
            stream_ptr = ctypes.c_void_p(stream.cuda_stream)

            # Call native helper using pre-loaded library
            res = self.cuda_lib.perform_direct_transfer(
                    ctypes.c_int(source_gpu),
                    ctypes.c_ulonglong(int(source_address)),
                    ctypes.c_int(target_gpu),
                    ctypes.c_ulonglong(int(target_address)),
                    ctypes.c_ulonglong(size),
                    src_handle_buf,
                    src_handle_size,
                    None,
                    ctypes.c_uint(0),
                    ctypes.c_ulonglong(src_offset),
                    ctypes.c_ulonglong(dst_offset),
                    stream_ptr
                )
            if res != 0:
                try:
                    error_msg = self.cuda_lib.cudaGetErrorString(res)
                    logger.error(f"Native transfer failed: {error_msg.decode() if error_msg else res}")
                except Exception:
                    logger.error(f"Native transfer failed with code: {res}")

                return False, None

            # Create and record completion event (asynchronous)
            completion_event = torch.cuda.Event(enable_timing=False)
            completion_event.record(stream)

            logger.debug(f"Direct transfer submitted asynchronously: "
                         f"GPU {source_gpu}->{target_gpu}, "
                         f"size={size} bytes ({size/1024/1024:.2f} MB)")
            return True, completion_event

       
        except Exception as e:
            logger.error(f"Error in direct transfer: {e}", exc_info=True)
            return False, None
            
        finally:
            torch.cuda.set_device(original_device)
  
    
    def _cleanup_completed_transfers(self):
        """Clean up old completed transfers."""
        with self.lock:
            # Keep only recent completed transfers to avoid memory leak
            if len(self.completed_transfers) > 1000:
                oldest_ids = sorted(self.completed_transfers.keys())[:100]
                for old_id in oldest_ids:
                    del self.completed_transfers[old_id]
    
    # ==================== Public API ====================
    
    def submit_transfer(
        self, 
        source_gpu: int, 
        target_gpu: int, 
        source_address: int, 
        target_address: int, 
        size: int,
        ipc_handle: Optional[bytes] = None,
        src_offset: int = 0,
        dst_offset: int = 0,
        sync: bool = True
    ) -> str:
        """
        Submit a transfer request to the engine.
        
        Args:
            source_gpu: Source GPU ID
            target_gpu: Target GPU ID
            source_address: Source GPU memory address
            target_address: Target GPU memory address
            size: Size to transfer in bytes
            ipc_handle : Optional serialized cudaIpcMemHandle if one address is remote
            src_offset: Offset into source buffer (bytes)
            dst_offset: Offset into destination buffer (bytes)
            sync: Whether to wait for transfer completion. 
            
        Returns:
            Transfer request ID
        """
        if not self.cuda_available:
            raise RuntimeError("CUDA not available, cannot submit transfer")
        
        if size <= 0:
            raise ValueError(f"Invalid transfer size: {size}")
        
        # Generate unique request ID
        with self.lock:
            # Use a counter for unique IDs instead of queue length
            if not hasattr(self, '_transfer_counter'):
                self._transfer_counter = 0
            self._transfer_counter += 1
            request_id = f"transfer_{int(time.time() * 1000)}_{self._transfer_counter}"
        
        # Create transfer request
        request = TransferRequest(
            request_id=request_id,
            source_gpu=source_gpu,
            target_gpu=target_gpu,
            source_address=source_address,
            target_address=target_address,
            size=size,
            ipc_handle=ipc_handle,
            src_offset=src_offset,
            dst_offset=dst_offset,
            sync=sync,
            status="pending"
        )
        
        # Add to queue
        with self.lock:
            self.transfer_queue.append(request)
            self._worker_event.set()
            
            logger.debug(f"Submitted transfer request {request_id}: "
                       f"GPU {source_gpu} -> GPU {target_gpu}, "
                       f"size: {size} bytes, sync: {sync}")
        
        return request_id
    
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
        # Extract ipc_handle from kwargs if provided
        ipc_handle = kwargs.get('ipc_handle', None)
        
        # Call transfer_sync method with the same parameters
        return self.transfer_sync(
            source_gpu=source_gpu,
            target_gpu=target_gpu,
            source_address=source_buffer,
            target_address=target_buffer,
            size=size,
            ipc_handle=ipc_handle,
            src_offset=src_offset,
            dst_offset=dst_offset
        )
    
    def transfer_sync(
        self,
        source_gpu: int,
        target_gpu: int,
        source_address: int,
        target_address: int,
        size: int,
        ipc_handle: Optional[bytes] = None,
        src_offset: int = 0,
        dst_offset: int = 0
    ) -> bool:
        """
        Perform synchronous transfer (blocking).
        
        Args:
            source_gpu: Source GPU ID
            target_gpu: Target GPU ID
            source_address: Source GPU memory address
            target_address: Target GPU memory address
            size: Size to transfer in bytes
            ipc_handle: Optional serialized cudaIpcMemHandle if one address is remote
            src_offset: Offset into source buffer (bytes)
            dst_offset: Offset into destination buffer (bytes)

        Returns:
            True if transfer successful, False otherwise
        """
        # Submit transfer with sync=True (explicitly synchronous)
        request_id = self.submit_transfer(
            source_gpu=source_gpu,
            target_gpu=target_gpu,
            source_address=source_address,
            target_address=target_address,
            size=size,
            ipc_handle=ipc_handle,
            src_offset=src_offset,
            dst_offset=dst_offset,
            sync=True  # Explicitly set sync=True for synchronous transfer
        )
        
        # Wait for transfer to complete
        return self.wait_for_transfer(request_id, timeout=self.transfer_timeout_sec)
    
    def wait_for_transfer(self, request_id: str, timeout: float = 30.0) -> bool:
        """
        Wait for a transfer to complete.
        
        Args:
            request_id: Transfer request ID
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if transfer completed successfully, False otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                # Check if transfer is in completed transfers
                if request_id in self.completed_transfers:
                    request = self.completed_transfers[request_id]
                    
                    # If status is "completed", return True
                    if request.status == "completed":
                        return True
                    # If status is "submitted", check if the CUDA event has completed
                    elif request.status == "submitted" and request.completion_event is not None:
                        # Check if the CUDA event has completed
                        if request.completion_event.query():
                            # Update status to completed
                            request.status = "completed"
                            request.end_time = time.time()
                            logger.debug(f"Transfer {request_id} completed asynchronously")
                            # Update stats for async transfer completion
                            self.stats["successful_transfers"] += 1
                            self.stats["total_transfer_bytes"] += request.size
                            return True
                        # Event not completed yet, continue waiting
                
                # Check if transfer is still active (being executed)
                if request_id in self.active_transfers:
                    # Still in progress, wait a bit
                    pass
                # Check if transfer is pending in queue
                elif any(req.request_id == request_id for req in self.transfer_queue):
                    # Still waiting in queue, wait a bit
                    pass
                else:
                    # Transfer not found in any queue - this shouldn't happen normally
                    # But we'll give it a moment in case it's being moved between states
                    logger.debug(f"Transfer request {request_id} not found in active/pending/completed")
            
            time.sleep(0.01)  # Small sleep to avoid busy waiting
        
        # Log error and provide more context
        with self.lock:
            if request_id not in self.completed_transfers and \
               request_id not in self.active_transfers and \
               not any(req.request_id == request_id for req in self.transfer_queue):
                logger.error(f"Timeout waiting for transfer {request_id} - transfer not found in any queue")
            else:
                logger.error(f"Timeout waiting for transfer {request_id}")
        
        return False
    
    def get_transfer_status(self, request_id: str) -> Optional[Dict]:
        """
        Get status of a transfer request.
        
        Args:
            request_id: Transfer request ID
            
        Returns:
            Dictionary with transfer status or None if not found
        """
        with self.lock:
            # Check completed transfers first
            if request_id in self.completed_transfers:
                request = self.completed_transfers[request_id]
                return {
                    'request_id': request.request_id,
                    'status': request.status,
                    'source_gpu': request.source_gpu,
                    'target_gpu': request.target_gpu,
                    'size': request.size,
                    'start_time': request.start_time,
                    'end_time': request.end_time,
                    'duration': request.end_time - request.start_time if request.start_time and request.end_time else None,
                    'error_message': request.error_message
                }
            
            # Check active transfers
            if request_id in self.active_transfers:
                request = self.active_transfers[request_id]
                return {
                    'request_id': request.request_id,
                    'status': request.status,
                    'source_gpu': request.source_gpu,
                    'target_gpu': request.target_gpu,
                    'size': request.size,
                    'start_time': request.start_time,
                    'end_time': None,
                    'duration': time.time() - request.start_time if request.start_time else None,
                    'error_message': None
                }
            
            # Check pending transfers in queue
            for request in self.transfer_queue:
                if request.request_id == request_id:
                    return {
                        'request_id': request.request_id,
                        'status': request.status,
                        'source_gpu': request.source_gpu,
                        'target_gpu': request.target_gpu,
                        'size': request.size,
                        'start_time': None,
                        'end_time': None,
                        'duration': None,
                        'error_message': None
                    }
        
        return None
    
    def cancel_transfer(self, request_id: str) -> bool:
        """
        Cancel a pending transfer request.
        
        Args:
            request_id: Transfer request ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        with self.lock:
            # Check if in queue
            for i, request in enumerate(self.transfer_queue):
                if request.request_id == request_id:
                    # Remove from queue
                    self.transfer_queue.pop(i)
                    request.status = "cancelled"
                    self.completed_transfers[request_id] = request
                    logger.info(f"Cancelled transfer request {request_id}")
                    return True
            
            # Check if active (cannot cancel active transfers)
            if request_id in self.active_transfers:
                logger.warning(f"Cannot cancel active transfer {request_id}")
                return False
            
            # Check if already completed
            if request_id in self.completed_transfers:
                logger.warning(f"Transfer {request_id} already completed")
                return False
        
        logger.warning(f"Transfer request {request_id} not found")
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get transfer engine status.
        
        Returns:
            Dictionary containing engine status and health information
        """
        with self.lock:
            status = {
                'engine_type': 'DistributedNVLINKTransferEngine',
                'gpu_id': self.gpu_id,
                'max_concurrent_transfers': self.max_concurrent_transfers,
                'transfer_timeout_sec': self.transfer_timeout_sec,
                'pending_transfers': len(self.transfer_queue),
                'active_transfers': len(self.active_transfers),
                'completed_transfers_count': len(self.completed_transfers),
            }
            
            # Add statistics to status
            status.update(self.stats)
            
            return status
    
    # need modification here
    def batch_transfer(
        self, 
        source_gpu: int,
        target_gpu: int,
        transfers: list,
        ipc_handles: Optional[List[bytes]] = None,
        src_offsets: Optional[List[int]] = None,
        dst_offsets: Optional[List[int]] = None
    ) -> list:
        """
        Batch transfer multiple buffers from source GPU to target GPU.
        
        This method is optimized for transferring multiple small files or buffers.
        It submits all transfers concurrently and waits for them to complete,
        amortizing the overhead cost across multiple transfers.
        
        Args:
            source_gpu: Source GPU ID
            target_gpu: Target GPU ID
            transfers: List of tuples (source_address, target_address, size_bytes)
                      Example: [(src_addr1, dst_addr1, size1), (src_addr2, dst_addr2, size2), ...]
            ipc_handles: Optional list of IPC handles for each transfer (same length as transfers)
            src_offsets: Optional list of source offsets for each transfer (same length as transfers)
            dst_offsets: Optional list of destination offsets for each transfer (same length as transfers)
        
        Returns:
            List of transfer results (one dict per transfer) with keys:
            - 'request_id': Unique transfer request ID
            - 'status': 'completed' or 'failed'
            - 'size': Size of transfer
            - 'duration_sec': Time taken for this transfer
            - 'error': Error message if failed
        """
        if not transfers:
            logger.warning("batch_transfer called with empty transfer list")
            return []
        
        # Validate optional parameter lengths
        num_transfers = len(transfers)
        if ipc_handles is not None and len(ipc_handles) != num_transfers:
            raise ValueError(f"ipc_handles length ({len(ipc_handles)}) must match transfers length ({num_transfers})")
        if src_offsets is not None and len(src_offsets) != num_transfers:
            raise ValueError(f"src_offsets length ({len(src_offsets)}) must match transfers length ({num_transfers})")
        if dst_offsets is not None and len(dst_offsets) != num_transfers:
            raise ValueError(f"dst_offsets length ({len(dst_offsets)}) must match transfers length ({num_transfers})")
        
        logger.info(f"Batch transfer: {num_transfers} transfers, "
                   f"total size: {sum(t[2] for t in transfers) / (1024**2):.2f}MB")
        
        # Step 1: Submit all transfers concurrently with sync=False for better performance
        request_ids = []
        for i, (src_addr, dst_addr, size) in enumerate(transfers):
            try:
                # Get optional parameters for this transfer
                ipc_handle = ipc_handles[i] if ipc_handles is not None else None
                src_offset = src_offsets[i] if src_offsets is not None else 0
                dst_offset = dst_offsets[i] if dst_offsets is not None else 0
                
                request_id = self.submit_transfer(
                    source_gpu=source_gpu,
                    target_gpu=target_gpu,
                    source_address=src_addr,
                    target_address=dst_addr,
                    size=size,
                    ipc_handle=ipc_handle,
                    src_offset=src_offset,
                    dst_offset=dst_offset,
                    sync=False  # Use async submission for batch transfers
                )
                request_ids.append(request_id)
            except Exception as e:
                logger.error(f"Failed to submit batch transfer {i}: {e}")
                return []
        
        logger.debug(f"Submitted {len(request_ids)} concurrent transfers (async)")
        
        # Step 2: Wait for all transfers to complete
        results = []
        for request_id in request_ids:
            try:
                status = self.wait_for_transfer(request_id)
                
                # Get transfer details
                transfer_info = self.get_transfer_status(request_id)
                
                results.append({
                    'request_id': request_id,
                    'status': transfer_info['status'] if transfer_info else 'unknown',
                    'size': transfer_info['size'] if transfer_info else 0,
                    'duration_sec': transfer_info['duration'] if transfer_info and transfer_info['duration'] else None,
                    'error': transfer_info['error_message'] if transfer_info and transfer_info['error_message'] else None
                })
            except Exception as e:
                logger.error(f"Failed to wait for transfer {request_id}: {e}")
                results.append({
                    'request_id': request_id,
                    'status': 'failed',
                    'size': 0,
                    'duration_sec': None,
                    'error': str(e)
                })
        
        # Statistics
        successful = sum(1 for r in results if r['status'] == 'completed')
        failed = len(results) - successful
        total_bytes = sum(r['size'] for r in results)
        
        logger.info(f"Batch transfer complete: {successful}/{len(results)} successful, "
                   f"{total_bytes / (1024**2):.2f}MB transferred")
        
        if failed > 0:
            logger.warning(f"Batch transfer: {failed} transfers failed")
        
        return results
    
    def batch_transfer_sync(
        self, 
        source_gpu: int,
        target_gpu: int,
        transfers: list,
        timeout: float = 30.0,
        ipc_handles: Optional[List[bytes]] = None,
        src_offsets: Optional[List[int]] = None,
        dst_offsets: Optional[List[int]] = None
    ) -> dict:
        """
        Synchronous batch transfer with optimized waiting mechanism.
        
        Args:
            source_gpu: Source GPU ID
            target_gpu: Target GPU ID
            transfers: List of (source_addr, target_addr, size) tuples
            timeout: Timeout in seconds for entire batch
            ipc_handles: Optional list of IPC handles for each transfer (same length as transfers)
            src_offsets: Optional list of source offsets for each transfer (same length as transfers)
            dst_offsets: Optional list of destination offsets for each transfer (same length as transfers)
            
        Returns:
            Dictionary with:
            - 'success': bool indicating if all transfers completed successfully
            - 'total_bytes': Total bytes transferred
            - 'total_time_sec': Time taken for batch
        """
        if not transfers:
            return {
                'success': False,
                'total_bytes': 0,
                'total_time_sec': 0
            }
        
        # Validate optional parameter lengths
        num_transfers = len(transfers)
        if ipc_handles is not None and len(ipc_handles) != num_transfers:
            raise ValueError(f"ipc_handles length ({len(ipc_handles)}) must match transfers length ({num_transfers})")
        if src_offsets is not None and len(src_offsets) != num_transfers:
            raise ValueError(f"src_offsets length ({len(src_offsets)}) must match transfers length ({num_transfers})")
        if dst_offsets is not None and len(dst_offsets) != num_transfers:
            raise ValueError(f"dst_offsets length ({len(dst_offsets)}) must match transfers length ({num_transfers})")
        
        start_time = time.time()
        
        # Phase 1: Submit all transfers with sync=False for better performance
        request_ids = []
        total_size = 0
        for i, (src_addr, dst_addr, size) in enumerate(transfers):
            try:
                # Get optional parameters for this transfer
                ipc_handle = ipc_handles[i] if ipc_handles is not None else None
                src_offset = src_offsets[i] if src_offsets is not None else 0
                dst_offset = dst_offsets[i] if dst_offsets is not None else 0
                
                request_id = self.submit_transfer(
                    source_gpu=source_gpu,
                    target_gpu=target_gpu,
                    source_address=src_addr,
                    target_address=dst_addr,
                    size=size,
                    ipc_handle=ipc_handle,
                    src_offset=src_offset,
                    dst_offset=dst_offset,
                    sync=False  # Use async submission for batch transfers
                )
                request_ids.append(request_id)
                total_size += size
            except Exception as e:
                logger.error(f"Failed to submit batch transfer {i}: {e}")
                return {
                    'success': False,
                    'total_bytes': 0,
                    'total_time_sec': time.time() - start_time
                }
        
        logger.debug(f"Submitted {len(request_ids)} concurrent transfers (async), total size: {total_size/1024/1024:.2f}MB")
        
        # Phase 2: Wait for all transfers with OPTIMIZED polling
        # Check all transfers together to reduce lock contention
        completed = set()
        failed = set()
        
        # Adaptive polling interval based on remaining transfers
        poll_interval = 0.001  # Start with 1ms
        max_poll_interval = 0.1  # Maximum 100ms
        
        while len(completed) + len(failed) < len(request_ids):
            # Check all pending transfers in one pass
            pending_count = 0
            for request_id in request_ids:
                if request_id in completed or request_id in failed:
                    continue
                
                # Non-blocking status check
                status = self.get_transfer_status(request_id)
                
                if status:
                    if status['status'] == 'completed':
                        completed.add(request_id)
                    elif status['status'] == 'failed':
                        failed.add(request_id)
                    else:
                        pending_count += 1
                else:
                    pending_count += 1
            
            # Adaptive sleep based on pending count
            if pending_count > 0:
                # Increase sleep time as more transfers complete
                remaining_ratio = pending_count / len(request_ids)
                adaptive_sleep = min(poll_interval * (1.0 / max(remaining_ratio, 0.1)), max_poll_interval)
                time.sleep(adaptive_sleep)
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Batch transfer timeout after {elapsed:.1f}s")
                break
        
        # Calculate metrics
        total_time = time.time() - start_time
        successful = len(completed)
        failed_count = len(failed)
        
        # Calculate total bytes transferred (only from successful transfers)
        total_bytes = 0
        for i, (src_addr, dst_addr, size) in enumerate(transfers):
            if i < len(request_ids):
                request_id = request_ids[i]
                if request_id in completed:
                    total_bytes += size
        
        logger.info(f"Batch transfer complete: {successful}/{len(request_ids)} successful, "
                   f"{total_bytes / (1024**2):.2f}MB transferred in {total_time:.3f}s")
        
        if failed_count > 0:
            logger.warning(f"Batch transfer: {failed_count} transfers failed")
        
        return {
            'success': failed_count == 0,
            'total_bytes': total_bytes,
            'total_time_sec': total_time
        }
    
    def clear_completed_transfers(self) -> int:
        """
        Clear completed transfers from memory.
        
        Returns:
            Number of transfers cleared
        """
        with self.lock:
            count = len(self.completed_transfers)
            self.completed_transfers.clear()
            logger.info(f"Cleared {count} completed transfers from memory")
            return count
    
    def shutdown(self):
        """
        Shutdown the transfer engine.
        
        This will:
        1. Stop the worker thread
        2. Wait for active transfers to complete
        3. Clear all queues
        """
        logger.info("Shutting down NVLINK transfer engine")
        
        # Stop worker thread
        self._worker_running = False
        self._worker_event.set()
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            logger.info("Transfer worker thread stopped")
        
        # Clear all queues
        with self.lock:
            self.transfer_queue.clear()
            self.active_transfers.clear()
            self.completed_transfers.clear()
        
        logger.info("NVLINK transfer engine shutdown completed")
    
    def register_segment(
        self,
        segment_id: str, 
        base_address: int, 
        gpu_id: int, 
        size: int
    ) -> bool:
        """
        Register a GPU memory segment with the metadata server 
        by obtaining and registering its IPC handle.
        
        Args:
            segment_id: Unique segment identifier
            base_address: Base GPU memory address of the segment
            gpu_id: GPU device ID
            size: Segment size in bytes
            
        Returns:
            True if registration successful, False otherwise
        """
        if not self.ipc_client:
            logger.warning("IPC client not available, cannot register segment")
            with self.lock:
                self.stats["failed_registers"] += 1
            return False
        
        try:
            
            # Get IPC handle for the allocated GPU memory
            libcudart = ctypes.CDLL("libcudart.so")
            cudaIpcGetMemHandle = libcudart.cudaIpcGetMemHandle
            cudaIpcGetMemHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            cudaIpcGetMemHandle.restype = ctypes.c_int
            
            IPC_HANDLE_SIZE = 64
            handle_buf = ctypes.create_string_buffer(IPC_HANDLE_SIZE)
            rc = cudaIpcGetMemHandle(ctypes.cast(handle_buf, ctypes.c_void_p), ctypes.c_void_p(base_address))
            
            if rc != 0:
                logger.error(f"Failed to get IPC handle for segment {segment_id}: "
                             f"CUDA error code {rc}")
                with self.lock:
                    self.stats["failed_registers"] += 1
                return False
            
            handle_bytes = handle_buf.raw
            
            # Register the IPC handle with the metadata server
            success = self.ipc_client.register_segment_ipc_handle(
                segment_id=segment_id,
                buffer_pointer=base_address,
                handle_bytes=handle_bytes,
                gpu_id=gpu_id,
                size=size
            )
            
            if success:
                logger.debug(f"Registered segment {segment_id} via IPC client")
                with self.lock:
                    self.stats["successful_registers"] += 1
            else:
                logger.error(f"Failed to register segment {segment_id} via IPC client")
                with self.lock:
                    self.stats["failed_registers"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error registering segment {segment_id}: {e}")
            with self.lock:
                self.stats["failed_registers"] += 1
            return False
    
    def unregister_segment(
        self, 
        segment_id: str, 
        base_address: int, 
        gpu_id: int
    ) -> bool:
        """
        Unregister a GPU memory segment from the metadata server.
        
        Args:
            segment_id: Unique segment identifier
            base_address: Base GPU memory address of the segment
            gpu_id: GPU device ID
            
        Returns:
            True if unregistration successful, False otherwise
        """
        if not self.ipc_client:
            logger.warning("IPC client not available, cannot unregister segment")
            with self.lock:
                self.stats["failed_unregisters"] += 1
            return False
        
        try:
            # Unregister the IPC handle from the metadata server
            success = self.ipc_client.unregister_segment_ipc_handle(
                segment_id=segment_id,
                buffer_pointer=base_address,
                gpu_id=gpu_id
            )
            
            if success:
                logger.debug(f"Unregistered segment {segment_id} via IPC client")
                with self.lock:
                    self.stats["successful_unregisters"] += 1
            else:
                logger.error(f"Failed to unregister segment {segment_id} via IPC client")
                with self.lock:
                    self.stats["failed_unregisters"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error unregistering segment {segment_id}: {e}")
            with self.lock:
                self.stats["failed_unregisters"] += 1
            return False
    
    def __del__(self):
        """Destructor to ensure proper shutdown."""
        if hasattr(self, '_worker_running') and self._worker_running:
            self.shutdown()
