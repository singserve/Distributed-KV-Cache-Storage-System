"""
Distributed NVLINK Transfer Engine for direct GPU-to-GPU data transfer.
Each VCacheEngine instance has its own transfer engine that can directly
transfer data between GPU memory addresses using NVLINK.

"""

import threading
import time
import torch
from typing import Optional, Dict, List
from dataclasses import dataclass
from lmcache.logging import init_logger

logger = init_logger(__name__)

@dataclass
class TransferRequest:

    """Transfer request for NVLINK transfer."""
    request_id: str
    source_gpu: int
    target_gpu: int
    source_address: int  # GPU memory address
    target_address: int  # GPU memory address
    size: int  # Size in bytes
    ipc_handle: Optional[bytes] = None  # Serialized cudaIpcMemHandle if source is remote
    src_offset: int = 0  # offset into source buffer (bytes)
    dst_offset: int = 0  # offset into destination buffer (bytes)
    status: str = "pending"  # pending, in_progress, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None


class DistributedNVLINKTransferEngine:
    """
    Distributed NVLINK Transfer Engine for direct GPU-to-GPU data transfer.
    
    Each VCacheEngine instance has its own transfer engine that can:
    1. Push data from its own GPU to other GPUs
    2. Pull data from other GPUs to its own GPU
    3. Manage local transfer queues and statistics
    4. Coordinate with segment manager for address resolution
    
    This is a pure distributed architecture where each engine controls its own transfers.
    """
    
    def __init__(self, config, gpu_id: int, segment_manager=None, ipc_client=None):
        """
        Initialize distributed NVLINK transfer engine for a specific GPU.
        
        Args:
            config: Configuration object
            gpu_id: GPU ID for this engine (0, 1, etc.)
            segment_manager: GPUVRAMSegmentManager instance for this GPU (optional)
            ipc_client: Optional VRAM metadata IPC client used to request IPC handles
        """
        self.config = config
        self.gpu_id = gpu_id
        self.segment_manager = segment_manager
        self.ipc_client = ipc_client
        
        self.lock = threading.RLock()
        
        # Check if CUDA and NVLINK are available
        self.cuda_available = torch.cuda.is_available()
        self.nvlink_available = self._check_nvlink_availability()
        
        # Transfer queue management (local to this engine)
        self.transfer_queue: List[TransferRequest] = []
        self.active_transfers: Dict[str, TransferRequest] = {}
        self.completed_transfers: Dict[str, TransferRequest] = {}
        
        # Statistics for this engine
        self.stats = {
            'total_transfers': 0,
            'successful_transfers': 0,
            'failed_transfers': 0,
            'total_bytes_transferred': 0,
            'avg_transfer_time': 0.0,
            'peak_bandwidth_gbps': 0.0,
            'queue_size': 0,
            'active_transfers_count': 0
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
                logger.warning(f"Only {num_gpus} GPU(s) available, NVLINK transfers require at least 2 GPUs")
                return False
            
            # Check peer-to-peer access between GPUs
            nvlink_available = False
            
            for i in range(num_gpus):
                for j in range(i + 1, num_gpus):
                    try:
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        if can_access:
                            logger.info(f"GPU {i} can access GPU {j} (peer access enabled)")
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
            
            logger.info("NVLINK transfer engine initialization completed")
            
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
        
        logger.info("Transfer worker thread stopped")
    
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
            
            # Perform the transfer
            success = self._perform_direct_transfer(
                source_gpu=request.source_gpu,
                target_gpu=request.target_gpu,
                source_address=request.source_address,
                target_address=request.target_address,
                size=request.size,
                ipc_handle=request.ipc_handle,
                src_offset=request.src_offset,
                dst_offset=request.dst_offset,
            )
            
            # Update request status
            request.end_time = time.time()
            if success:
                request.status = "completed"
                logger.info(f"Transfer {request.request_id} completed successfully: "
                          f"GPU {request.source_gpu} -> GPU {request.target_gpu}, "
                          f"size: {request.size} bytes, "
                          f"time: {request.end_time - request.start_time:.3f}s")
            else:
                request.status = "failed"
                request.error_message = "Transfer failed"
                logger.error(f"Transfer {request.request_id} failed: "
                           f"GPU {request.source_gpu} -> GPU {request.target_gpu}")
            
        except Exception as e:
            request.status = "failed"
            request.end_time = time.time()
            request.error_message = str(e)
            logger.error(f"Error executing transfer {request.request_id}: {e}")
        
        finally:
            # Move from active to completed
            with self.lock:
                if request.request_id in self.active_transfers:
                    del self.active_transfers[request.request_id]
                
                # Keep only recent completed transfers to avoid memory leak
                if len(self.completed_transfers) > 1000:
                    oldest_ids = sorted(self.completed_transfers.keys())[:100]
                    for old_id in oldest_ids:
                        del self.completed_transfers[old_id]
                
                self.completed_transfers[request.request_id] = request
    
    def _perform_direct_transfer(self, 
                            source_gpu: int, 
                            target_gpu: int, 
                            source_address: int, 
                            target_address: int, 
                            size: int,
                            ipc_handle: Optional[bytes] = None,
                            src_offset: int = 0,
                            dst_offset: int = 0) -> bool:
        """Perform direct GPU-to-GPU transfer using NVLINK with low-level CUDA APIs.

        Args:
            ipc_handle: Optional serialized cudaIpcMemHandle if source allocation is remote (not yet mapped here)
        """
        if not self.cuda_available:
            logger.error("CUDA not available for direct transfer")
            return False
        
        original_device = torch.cuda.current_device()
        
        try:
            # If an IPC handle is provided, log it (actual mapping not implemented here)
            if ipc_handle is not None:
                logger.debug("IPC handle provided to _perform_direct_transfer")

            device_count = torch.cuda.device_count()
            if source_gpu >= device_count or target_gpu >= device_count:
                logger.error(f"Invalid GPU IDs: source_gpu={source_gpu}, target_gpu={target_gpu}")
                return False
            
            if size <= 0 or size > 1024 * 1024 * 1024:  # 1GB max
                logger.error(f"Invalid transfer size: {size} bytes")
                return False

            # Allow source_address==0 when using an IPC handle (remote buffer); target must be a valid local pointer
            if target_address == 0 or (source_address == 0 and ipc_handle is None):
                logger.error(f"Invalid memory address: source={hex(source_address)}, target={hex(target_address)}; "
                             f"note: source may be 0 if ipc_handle is provided")
                return False
            
            can_access = torch.cuda.can_device_access_peer(source_gpu, target_gpu)
            logger.debug(f"Peer access from GPU {source_gpu} to GPU {target_gpu}: {can_access}")
            
            if not can_access:
                logger.error(f"Peer access not enabled from GPU {source_gpu} to GPU {target_gpu}")
                return False
            
            torch.cuda.set_device(source_gpu)
            
            stream = torch.cuda.Stream(device=source_gpu)
            
            import ctypes

            libname = "libnvlink_transfer.so"
            try:
                cuda_lib = ctypes.CDLL(libname)
                logger.debug(f"Loaded native nvlink helper: {libname}")
            except Exception as e:
                logger.error(f"Native nvlink helper not available: {e}")
                return False

            # native library exports perform_direct_transfer
            perform_direct_transfer_native = cuda_lib.perform_direct_transfer
            perform_direct_transfer_native.argtypes = [
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
            perform_direct_transfer_native.restype = ctypes.c_int

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

            # Call native helper (must be available)
            res = perform_direct_transfer_native(
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
                    cuda_get_error_string = cuda_lib.cudaGetErrorString
                    cuda_get_error_string.argtypes = [ctypes.c_int]
                    cuda_get_error_string.restype = ctypes.c_char_p
                    error_msg = cuda_get_error_string(res)
                    logger.error(f"Native transfer failed: {error_msg.decode() if error_msg else res}")
                except Exception:
                    logger.error(f"Native transfer failed with code: {res}")

                return False

            # Wait for copy to complete
            stream.synchronize()

            logger.info(f"Direct transfer completed: GPU {source_gpu}->{target_gpu}, "
                        f"size={size} bytes ({size/1024/1024:.2f} MB)")
            return True


            
        except Exception as e:
            logger.error(f"Error in direct transfer: {e}", exc_info=True)
            return False
            
        finally:
            torch.cuda.set_device(original_device)
  
    
    def _cleanup_completed_transfers(self):
        """Clean up old completed transfers and update statistics."""
        with self.lock:
            # Update statistics
            self.stats['queue_size'] = len(self.transfer_queue)
            self.stats['active_transfers_count'] = len(self.active_transfers)
            
            # Calculate transfer statistics from completed transfers
            successful_transfers = 0
            failed_transfers = 0
            total_bytes = 0
            total_time = 0.0
            
            for request in self.completed_transfers.values():
                if request.status == "completed":
                    successful_transfers += 1
                    total_bytes += request.size
                    if request.start_time and request.end_time:
                        total_time += (request.end_time - request.start_time)
                elif request.status == "failed":
                    failed_transfers += 1
            
            # Update stats
            self.stats['successful_transfers'] = successful_transfers
            self.stats['failed_transfers'] = failed_transfers
            self.stats['total_transfers'] = successful_transfers + failed_transfers
            self.stats['total_bytes_transferred'] = total_bytes
            
            if successful_transfers > 0:
                self.stats['avg_transfer_time'] = total_time / successful_transfers
                if total_time > 0:
                    bandwidth_gbps = (total_bytes / (1024**3)) / (total_time / successful_transfers)
                    self.stats['peak_bandwidth_gbps'] = max(
                        self.stats['peak_bandwidth_gbps'], bandwidth_gbps
                    )
    
    # ==================== Public API ====================
    
    def submit_transfer(self, 
                       source_gpu: int, 
                       target_gpu: int, 
                       source_address: int, 
                       target_address: int, 
                       size: int,
                       ipc_handle: Optional[bytes] = None,
                       src_offset: int = 0,
                       dst_offset: int = 0) -> str:
        """
        Submit a transfer request to the engine.
        
        Args:
            source_gpu: Source GPU ID
            target_gpu: Target GPU ID
            source_address: Source GPU memory address
            target_address: Target GPU memory address
            size: Size to transfer in bytes
            
        Returns:
            Transfer request ID
        """
        if not self.cuda_available:
            raise RuntimeError("CUDA not available, cannot submit transfer")
        
        if size <= 0:
            raise ValueError(f"Invalid transfer size: {size}")
        
        # Generate unique request ID (using atomic counter for uniqueness)
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
            status="pending"
        )
        
        # Add to queue
        with self.lock:
            self.transfer_queue.append(request)
            self._worker_event.set()
            
            logger.debug(f"Submitted transfer request {request_id}: "
                       f"GPU {source_gpu} -> GPU {target_gpu}, "
                       f"size: {size} bytes")
        
        return request_id
    
    def transfer_sync(self,
                     source_gpu: int,
                     target_gpu: int,
                     source_address: int,
                     target_address: int,
                     size: int,
                     ipc_handle: Optional[bytes] = None,
                     src_offset: int = 0,
                     dst_offset: int = 0) -> bool:
        """
        Perform synchronous transfer (blocking).
        
        Args:
            source_gpu: Source GPU ID
            target_gpu: Target GPU ID
            source_address: Source GPU memory address
            target_address: Target GPU memory address
            size: Size to transfer in bytes
            ipc_handle: Optional serialized cudaIpcMemHandle if source allocation is remote
            
        Returns:
            True if transfer successful, False otherwise
        """
        # Submit transfer (propagate ipc_handle to the worker request)
        request_id = self.submit_transfer(
            source_gpu=source_gpu,
            target_gpu=target_gpu,
            source_address=source_address,
            target_address=target_address,
            size=size,
            ipc_handle=ipc_handle,
            src_offset=src_offset,
            dst_offset=dst_offset
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
                # Check if transfer is completed
                if request_id in self.completed_transfers:
                    request = self.completed_transfers[request_id]
                    return request.status == "completed"
                
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
    
    def get_stats(self) -> Dict:
        """
        Get transfer engine statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            stats = self.stats.copy()
            
            # Add queue information
            stats['pending_transfers'] = len(self.transfer_queue)
            stats['active_transfers'] = len(self.active_transfers)
            stats['completed_transfers_count'] = len(self.completed_transfers)
            
            # Calculate success rate
            total = stats['successful_transfers'] + stats['failed_transfers']
            if total > 0:
                stats['success_rate'] = (stats['successful_transfers'] / total) * 100
            else:
                stats['success_rate'] = 0.0
            
            return stats
    
    def batch_transfer(self, 
                      source_gpu: int,
                      target_gpu: int,
                      transfers: list) -> list:
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
        
        Returns:
            List of transfer results (one dict per transfer) with keys:
            - 'request_id': Unique transfer request ID
            - 'status': 'completed' or 'failed'
            - 'size': Size of transfer
            - 'duration_sec': Time taken for this transfer
            - 'error': Error message if failed
            
        Example:
            >>> # Transfer 10 files of 64KB each
            >>> transfers = [(addr1, addr1', 65536), (addr2, addr2', 65536), ...]
            >>> results = engine.batch_transfer(source_gpu=1, target_gpu=0, transfers=transfers)
            >>> for result in results:
            ...     print(f"Transfer {result['request_id']}: {result['status']}")
        """
        if not transfers:
            logger.warning("batch_transfer called with empty transfer list")
            return []
        
        logger.info(f"Batch transfer: {len(transfers)} transfers, "
                   f"total size: {sum(t[2] for t in transfers) / (1024**2):.2f}MB")
        
        # Step 1: Submit all transfers concurrently
        request_ids = []
        for src_addr, dst_addr, size in transfers:
            try:
                request_id = self.submit_transfer(
                    source_gpu=source_gpu,
                    target_gpu=target_gpu,
                    source_address=src_addr,
                    target_address=dst_addr,
                    size=size
                )
                request_ids.append(request_id)
            except Exception as e:
                logger.error(f"Failed to submit batch transfer: {e}")
                return []
        
        logger.debug(f"Submitted {len(request_ids)} concurrent transfers")
        
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
    
    def batch_transfer_sync(self, 
                           source_gpu: int,
                           target_gpu: int,
                           transfers: list,
                           timeout: float = 30.0) -> dict:
        """
        Synchronous batch transfer with optimized waiting mechanism.
        
        IMPORTANT: Uses optimized polling to avoid lock contention.
        Do NOT use sequential wait_for_transfer() calls!
        
        Args:
            source_gpu: Source GPU ID
            target_gpu: Target GPU ID
            transfers: List of (source_addr, target_addr, size) tuples
            timeout: Timeout in seconds for entire batch
            
        Returns:
            Dictionary with:
            - 'success': bool indicating if all transfers completed
            - 'results': List of individual transfer results
            - 'total_bytes': Total bytes transferred
            - 'total_time_sec': Time taken for batch
            - 'throughput_gbps': Aggregated throughput
            - 'num_transfers': Number of transfers
            - 'num_failed': Number of failed transfers
        """
        if not transfers:
            return {
                'success': False,
                'results': [],
                'total_bytes': 0,
                'total_time_sec': 0,
                'throughput_gbps': 0,
                'num_transfers': 0,
                'num_failed': 0
            }
        
        start_time = time.time()
        
        # Phase 1: Submit all transfers
        request_ids = []
        for src_addr, dst_addr, size in transfers:
            try:
                request_id = self.submit_transfer(
                    source_gpu=source_gpu,
                    target_gpu=target_gpu,
                    source_address=src_addr,
                    target_address=dst_addr,
                    size=size
                )
                request_ids.append(request_id)
            except Exception as e:
                logger.error(f"Failed to submit batch transfer: {e}")
                return {
                    'success': False,
                    'results': [],
                    'total_bytes': 0,
                    'total_time_sec': time.time() - start_time,
                    'throughput_gbps': 0,
                    'num_transfers': 0,
                    'num_failed': len(request_ids)
                }
        
        logger.debug(f"Submitted {len(request_ids)} concurrent transfers")
        
        # Phase 2: Wait for all transfers with OPTIMIZED polling
        # KEY OPTIMIZATION: Check all transfers together to reduce lock contention
        completed = set()
        failed = set()
        
        while len(completed) + len(failed) < len(request_ids):
            # Check all pending transfers in one pass
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
        
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Batch transfer timeout after {elapsed:.1f}s")
                break
        
        # Phase 3: Collect results
        results = []
        for request_id in request_ids:
            status = self.get_transfer_status(request_id)
            if status:
                results.append({
                    'request_id': request_id,
                    'status': status['status'],
                    'size': status['size'],
                    'duration_sec': status['duration'],
                    'error': status['error_message']
                })
            else:
                results.append({
                    'request_id': request_id,
                    'status': 'unknown',
                    'size': 0,
                    'duration_sec': None,
                    'error': 'Could not retrieve status'
                })
        
        # Calculate metrics
        total_time = time.time() - start_time
        successful = len(completed)
        failed_count = len(failed)
        total_bytes = sum(r['size'] for r in results if r['status'] == 'completed')
        
        # Calculate throughput
        throughput_gbps = 0
        if total_time > 0:
            throughput_gbps = (total_bytes / (1024**3)) / total_time
        
        logger.info(f"Batch transfer complete: {successful}/{len(request_ids)} successful, "
                   f"{total_bytes / (1024**2):.2f}MB transferred in {total_time:.3f}s "
                   f"({throughput_gbps:.2f} Gbps)")
        
        if failed_count > 0:
            logger.warning(f"Batch transfer: {failed_count} transfers failed")
        
        return {
            'success': failed_count == 0,
            'results': results,
            'total_bytes': total_bytes,
            'total_time_sec': total_time,
            'throughput_gbps': throughput_gbps,
            'num_transfers': len(request_ids),
            'num_failed': failed_count
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
            
            # Reset statistics
            self.stats = {
                'total_transfers': 0,
                'successful_transfers': 0,
                'failed_transfers': 0,
                'total_bytes_transferred': 0,
                'avg_transfer_time': 0.0,
                'peak_bandwidth_gbps': 0.0,
                'queue_size': 0,
                'active_transfers_count': 0
            }
        
        logger.info("NVLINK transfer engine shutdown completed")
    
    def __del__(self):
        """Destructor to ensure proper shutdown."""
        if hasattr(self, '_worker_running') and self._worker_running:
            self.shutdown()
