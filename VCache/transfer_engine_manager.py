# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import asyncio
import threading
import time
import ctypes
import uuid
import os
import json

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)

# Try to import cupy, but make it optional
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("CuPy is available")
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available, intra-GPU transfers using CuPy will be disabled")

# Mooncake imports
try:
    from mooncake.engine import TransferEngine
    TRANSFER_ENGINE_AVAILABLE = True
    logger.info("Mooncake transfer engine is available")
except ImportError:
    TRANSFER_ENGINE_AVAILABLE = False
    logger.warning("Mooncake transfer engine not available, cross-GPU transfers disabled")

# Import GPU VRAM segment class
from lmcache.test.gpu_vram_segment_manager import GPUVRAMSegment

class TransferEngineManager:
    """Manages Mooncake transfer engine for cross-GPU data transfers."""
    
    def __init__(self, config):
        self.config = config
        self.engine = None
        self.initialized = False
        self.lock = threading.RLock()
        
        # Transfer statistics
        self.transfer_stats = {
            'total_transfers': 0,
            'successful_transfers': 0,
            'failed_transfers': 0,
            'total_bytes_transferred': 0,
            'avg_transfer_time': 0.0
        }
        
        self._initialize_transfer_engine()

    def _initialize_transfer_engine(self):
        """Initialize Mooncake transfer engine if available."""
        if not TRANSFER_ENGINE_AVAILABLE:
            logger.warning("Transfer engine not available, cross-GPU transfers disabled")
            return
        
        try:
            self.engine = TransferEngine()
            
            # Get configuration from LMCache config
            local_hostname_TE = self.config.get_extra_config_value("local_hostname_TE", "localhost")
            metadata_server = self.config.get_extra_config_value("metadata_server", "http://localhost:8080/metadata")
            protocol_TE = self.config.get_extra_config_value("protocol_TE", "nvlink")
            device_name = self.config.get_extra_config_value("device_name", "")

            os.environ['MC_LEGACY_RPC_PORT_BINDING'] = '1'
            os.environ['MC_FORCE_MNNVL'] = '1'
            os.environ['MC_USE_NVLINK_IPC'] = '1'
            ret = self.engine.initialize(
                local_hostname_TE,
                metadata_server,
                protocol_TE,
                device_name
            )
            os.environ['MC_FORCE_MNNVL'] = '0'
            
            if ret == 0:
                self.initialized = True
                logger.info(f"Transfer engine initialized successfully on {local_hostname_TE}")
            else:
                logger.error(f"Failed to initialize transfer engine: {ret}")
                
        except Exception as e:
            logger.error(f"Error initializing transfer engine: {e}")

    def _register_segment_with_engine(self, segment: GPUVRAMSegment):
        """
        Register a single GPU VRAM segment with the transfer engine.
        
        Args:
            segment: GPU VRAM segment to register
        """
        try:
            # Register the segment memory with transfer engine
            ret = self.engine.register_memory(segment.base_address, segment.size)
            
            if ret == 0:
                host = self.config.get_extra_config_value("local_hostname_TE", "localhost")
                logger.info(f"Successfully registered segment on {host},id:{segment.segment_id} on GPU {segment.gpu_id}: "
                           f"address={hex(segment.base_address)}, size={segment.size} bytes")
                
                # Store registration info for cleanup
                if not hasattr(self, '_registered_segments'):
                    self._registered_segments = []
                self._registered_segments.append(segment)
                
            else:
                logger.error(f"Failed to register segment {segment.segment_id} on GPU {segment.gpu_id}: error code {ret}")
                
        except Exception as e:
            logger.error(f"Exception registering segment {segment.segment_id}: {e}")

    def unregister_memory(self, buffer_address: int) -> bool:
        """
        Unregister memory from transfer engine.
        
        Args:
            buffer_address: Memory address to unregister
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized or not self.engine:
            return False
        
        try:
            ret = self.engine.unregister_memory(buffer_address)
            
            if ret == 0:
                logger.info(f"Successfully unregistered memory at address {hex(buffer_address)}")
                return True
            else:
                logger.error(f"Failed to unregister memory at address {hex(buffer_address)}: error code {ret}")
                return False
                
        except Exception as e:
            logger.error(f"Exception unregistering memory at {hex(buffer_address)}: {e}")
            return False
    
    def transfer_gpu_to_gpu(self, target_hostname: str, source_gpu: int, target_gpu: int, 
                           source_buffer, target_buffer, size: int) -> bool:
        """Transfer data between GPUs using transfer engine."""
        if not self.initialized or not self.engine:
            logger.warning("Transfer engine not available for cross-GPU transfer")
            return False
        
        logger.info(f"going to transfer from host:{target_hostname},GPU{source_gpu}:{source_buffer} to GPU{target_gpu}:{target_buffer}, size is {size} ")
        
        try:
            # Ensure proper device context for IPC operations
            original_device = torch.cuda.current_device()
            
            # Set device to target GPU (where we want to receive data)
            torch.cuda.set_device(target_gpu)
            local_hostname = self.config.get_extra_config_value("local_hostname_TE", "localhost")
            logger.info(f'from target_host:{target_hostname},target_gpu:{source_gpu},target_buffer:{source_buffer}, registered_segments:{self._registered_segments}')
            logger.info(f'to source_host:{local_hostname},source_gpu:{target_gpu},source_buffer:{target_buffer}')
            
            # Perform synchronous transfer - this reads from remote GPU to local GPU
            start_time = time.time()
            ret = self.engine.transfer_sync_read(
                target_hostname,      # Remote hostname where source data resides
                target_buffer,        # Local buffer to receive data
                source_buffer,        # Remote buffer to read from
                size                  # Size of data to transfer
            )
            transfer_time = time.time() - start_time
            
            # Restore original device
            torch.cuda.set_device(original_device)
            
            # Update statistics
            with self.lock:
                self.transfer_stats['total_transfers'] += 1
                self.transfer_stats['total_bytes_transferred'] += size
                
                if ret == 0:
                    self.transfer_stats['successful_transfers'] += 1
                    logger.info(f"Cross-GPU transfer successful: GPU {source_gpu} -> GPU {target_gpu}, "
                               f"size: {size} bytes, time: {transfer_time:.3f}s")
                    return True
                else:
                    self.transfer_stats['failed_transfers'] += 1
                    logger.error(f"Cross-GPU transfer failed: GPU {source_gpu} -> GPU {target_gpu}, "
                                f"error code: {ret}")
                    return False
                
        except Exception as e:
            logger.error(f"Exception during cross-GPU transfer: {e}")
            # Restore original device in case of error
            try:
                torch.cuda.set_device(original_device)
            except:
                pass
            with self.lock:
                self.transfer_stats['failed_transfers'] += 1
            return False
    
    def prefetch_to_gpu(self, source_gpu: int, target_gpu: int, 
                       source_buffer, target_buffer, size: int) -> int:
        """Initiate asynchronous prefetch transfer."""
        if not self.initialized or not self.engine:
            return -1
        
        try:
            target_hostname = f"gpu{target_gpu}.localhost"
            
            # Submit asynchronous transfer
            batch_id = self.engine.transfer_submit_write(
                target_hostname=target_hostname,
                buffer=source_buffer,
                peer_buffer_address=target_buffer,
                length=size
            )
            
            if batch_id != -1:
                logger.debug(f"Prefetch initiated: GPU {source_gpu} -> GPU {target_gpu}, "
                           f"batch_id: {batch_id}")
            
            return batch_id
            
        except Exception as e:
            logger.error(f"Exception during prefetch: {e}")
            return -1
    
    def check_transfer_status(self, batch_id: int) -> int:
        """Check status of asynchronous transfer."""
        if not self.initialized or not self.engine:
            return -1
        
        try:
            return self.engine.transfer_check_status(batch_id)
        except Exception as e:
            logger.error(f"Exception checking transfer status: {e}")
            return -1

    def transfer_within_gpu(self, source_address: int, target_address: int, size: int, gpu_id: int) -> bool:
        """
        More practical implementation using CuPy as intermediary.
        """
        try:
            # Save current device
            original_device = torch.cuda.current_device()
            torch.cuda.set_device(gpu_id)
            
            logger.info(f"Starting intra-GPU memory copy on GPU {gpu_id}: {hex(source_address)} -> {hex(target_address)}, size: {size} bytes")
            
            if not CUPY_AVAILABLE:
                logger.warning("CuPy not available, using torch.cuda.memcpy for intra-GPU transfer")
                # 使用torch.cuda.memcpy进行GPU内拷贝
                import ctypes
                from ctypes import c_void_p
                
                # 加载CUDA驱动
                try:
                    cuda = ctypes.CDLL("nvcuda.dll" if os.name == 'nt' else "libcuda.so")
                except Exception as e:
                    logger.error(f"Failed to load CUDA driver: {e}")
                    return False
                
                # 定义cudaMemcpy函数
                cuda.cudaMemcpy.restype = ctypes.c_int
                cuda.cudaMemcpy.argtypes = [c_void_p, c_void_p, ctypes.c_size_t, ctypes.c_int]
                
                # CUDA内存拷贝类型 - 设备到设备
                cudaMemcpyDeviceToDevice = 2
                
                # 执行设备到设备的内存拷贝
                result = cuda.cudaMemcpy(
                    c_void_p(target_address),  # 目标地址
                    c_void_p(source_address),  # 源地址
                    size,                      # 大小
                    cudaMemcpyDeviceToDevice   # 拷贝方向
                )
                
                if result == 0:  # cudaSuccess
                    logger.info(f"Intra-GPU memory copy successful using torch.cuda.memcpy: {size} bytes from {hex(source_address)} to {hex(target_address)}")
                else:
                    logger.error(f"Intra-GPU memory copy failed with CUDA error: {result}")
                    return False
            else:
                # 使用CuPy进行GPU内拷贝
                with cp.cuda.Device(gpu_id):
                    # 将源地址包装为CuPy数组
                    src_mem = cp.cuda.UnownedMemory(source_address, size, None)
                    src_ptr = cp.cuda.MemoryPointer(src_mem, 0)
                    
                    # 将目标地址包装为CuPy数组
                    dst_mem = cp.cuda.UnownedMemory(target_address, size, None)
                    dst_ptr = cp.cuda.MemoryPointer(dst_mem, 0)
                    
                    # 创建CuPy数组视图
                    # 假设数据为float32类型
                    element_size = cp.dtype(cp.int64).itemsize
                    num_elements = size // element_size
                    
                    src_arr = cp.ndarray((num_elements,), dtype=cp.int64, memptr=src_ptr)
                    dst_arr = cp.ndarray((num_elements,), dtype=cp.int64, memptr=dst_ptr)
                    
                    # 方法1: 使用CuPy数组拷贝（推荐）
                    cp.copyto(dst_arr, src_arr)
                    
                    # 同步
                    cp.cuda.Stream.null.synchronize()
                
                logger.info(f"Intra-GPU memory copy successful using CuPy: {size} bytes from {hex(source_address)} to {hex(target_address)}")
            
            # Update statistics
            with self.lock:
                self.transfer_stats['total_transfers'] += 1
                self.transfer_stats['successful_transfers'] += 1
                self.transfer_stats['total_bytes_transferred'] += size
            
            # Restore original device
            torch.cuda.set_device(original_device)
            return True
                
        except Exception as e:
            logger.error(f"Intra-GPU memory copy failed: {e}")
            # Update statistics
            with self.lock:
                self.transfer_stats['total_transfers'] += 1
                self.transfer_stats['failed_transfers'] += 1
            
            # Restore original device in case of error
            try:
                torch.cuda.set_device(original_device)
            except:
                pass
            return False

    
    
    def get_stats(self) -> dict:
        """Get transfer statistics."""
        with self.lock:
            stats = self.transfer_stats.copy()
            if stats['total_transfers'] > 0:
                stats['success_rate'] = (stats['successful_transfers'] / stats['total_transfers']) * 100
            else:
                stats['success_rate'] = 0.0
            return stats

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
                # Unregister all registered segments first
                if hasattr(self, '_registered_segments'):
                    for segment in self._registered_segments:
                        try:
                            self.unregister_memory(segment.base_address)
                            logger.debug(f"Unregistered segment {segment.segment_id} from transfer engine")
                        except Exception as e:
                            logger.warning(f"Failed to unregister segment {segment.segment_id}: {e}")
                    self._registered_segments.clear()
                
                # If transfer engine has a shutdown method, call it
                if hasattr(self.engine, 'shutdown'):
                    ret = self.engine.shutdown()
                    if ret == 0:
                        logger.info("Transfer engine shutdown completed successfully")
                    else:
                        logger.error(f"Transfer engine shutdown failed with error code: {ret}")
                        return False
                else:
                    # If no shutdown method, just log and continue
                    logger.info("Transfer engine does not have shutdown method, skipping")
                
                # Reset engine state
                self.engine = None
                self.initialized = False
                
                # Clear statistics
                self.transfer_stats = {
                    'total_transfers': 0,
                    'successful_transfers': 0,
                    'failed_transfers': 0,
                    'total_bytes_transferred': 0,
                    'avg_transfer_time': 0.0
                }
                
                logger.info("Transfer engine shutdown completed")
                return True
                
            except Exception as e:
                logger.error(f"Error during transfer engine shutdown: {e}")

