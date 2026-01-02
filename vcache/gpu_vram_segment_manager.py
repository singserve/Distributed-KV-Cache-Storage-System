'''
VRAM management
Memory block and Segment manage metadata such as allocation status, size, and pointers to next blocks.
Segment manager handles real GPU memory allcation/deallocation and it allocates all memory when initialized.
'''


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import threading
import time
import torch
import ctypes

from lmcache.logging import init_logger
from lmcache.vcache.vram_kvcache_unit import VRAMKVCacheUnit
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


@dataclass
class MemoryBlock:
    """Memory block within a segment."""
    start: int  # Start offset in bytes
    size: int   # Size in bytes
    is_allocated: bool = False  # True if allocated, False if free
    next: Optional["MemoryBlock"] = None  # Next block in linked list
    
    @property
    def end(self) -> int:
        """End offset of the block."""
        return self.start + self.size
    
    def split(self, size: int) -> Optional["MemoryBlock"]:
        """
        Split the block into two blocks.
        
        Args:
            size: Size for the first block (must be <= self.size)
            
        Returns:
            New free block if split successful, None otherwise
        """
        if size <= 0 or size >= self.size:
            return None
        
        # Create new free block for the remaining space
        new_block = MemoryBlock(
            start=self.start + size,
            size=self.size - size,
            is_allocated=False,
            next=self.next  # Keep the original next pointer
        )
        
        # Update current block size
        self.size = size
        # Note: We don't set self.next here because the caller
        # will handle linking the blocks appropriately based on
        # whether they're in free or allocated lists
        
        return new_block
    

@dataclass
class GPUVRAMSegment:
    """GPU VRAM segment for data exchange"""
    segment_id: str  # Unique segment identifier
    gpu_id: int  # GPU device ID
    base_address: int  # Base GPU memory address
    size: int  # Segment size in bytes
    is_active: bool = True  # Whether segment is active for allocation
    created_time: float = 0.0
    last_access_time: float = 0.0
    
    # Memory management using two linked lists
    free_blocks_head: Optional[MemoryBlock] = None  # Linked list of free blocks
    allocated_blocks_head: Optional[MemoryBlock] = None  # Linked list of allocated blocks
    
    # VRAM Unit management 
    _vram_units: Dict[Union[str, CacheEngineKey], VRAMKVCacheUnit] = None  # cache_key -> VRAM unit
    
    def __post_init__(self):
        if self.created_time == 0.0:
            self.created_time = time.time()
        if self.last_access_time == 0.0:
            self.last_access_time = time.time()
        
        # Initialize with one free block covering the entire segment
        self.free_blocks_head = MemoryBlock(start=0, size=self.size, is_allocated=False)
        self.allocated_blocks_head = None
        
        # Initialize VRAM unit management
        self._vram_units = {}
    
    @property
    def used_size(self) -> int:
        """Calculate total used size from allocated blocks."""
        total_used = 0
        current = self.allocated_blocks_head
        while current:
            total_used += current.size
            current = current.next
        return total_used
    
    @property
    def free_size(self) -> int:
        """Calculate total free size from free blocks."""
        total_free = 0
        current = self.free_blocks_head
        while current:
            total_free += current.size
            current = current.next
        return total_free
    
    def allocate(self, size: int) -> Optional[Tuple[int, MemoryBlock]]:
        """
        Allocate a block of memory from this segment.
        
        Args:
            size: Required size in bytes
            
        Returns:
            Tuple of (offset, allocated_block) or None if allocation failed
        """
        if not self.is_active:
            return None
        
        # Find best fit free block
        prev = None
        current = self.free_blocks_head
        best_fit_prev = None
        best_fit_block = None
        best_fit_size = float('inf')
        
        while current:
            if not current.is_allocated and current.size >= size:
                # Found a free block that can fit the request
                if current.size < best_fit_size:
                    best_fit_prev = prev
                    best_fit_block = current
                    best_fit_size = current.size
            prev = current
            current = current.next
        
        if best_fit_block is None:
            return None
        
        # Remove from free list
        if best_fit_prev:
            best_fit_prev.next = best_fit_block.next
        else:
            self.free_blocks_head = best_fit_block.next
        
        # Split if the block is larger than needed
        remaining_block = best_fit_block.split(size)
        if remaining_block:
            # Add remaining block back to free list
            remaining_block.next = self.free_blocks_head
            self.free_blocks_head = remaining_block
            
            # After adding the remaining block, we need to merge adjacent free blocks
            # to avoid fragmentation. Use the more efficient method first.
            if not self._merge_with_adjacent_free_blocks(remaining_block):
                # If the efficient method didn't merge, fall back to full merge
                self._merge_free_blocks()
        
        # Mark as allocated
        best_fit_block.is_allocated = True
        
        # Add to allocated list (sorted by start address)
        self._add_to_allocated_list(best_fit_block)
        
        # Update access time
        self.last_access_time = time.time()
        
        return best_fit_block.start, best_fit_block
    
    def free(self, block: MemoryBlock) -> bool:
        """
        Free an allocated block.
        
        Args:
            block: Block to free
            
        Returns:
            True if successful, False otherwise
        """
        if not block.is_allocated:
            return False
        
        # Remove from allocated list
        if not self._remove_from_allocated_list(block):
            return False
        
        # Mark as free
        block.is_allocated = False
        
        # Add to free list (sorted by start address)
        self._add_to_free_list(block)
        
        # Merge with adjacent free blocks
        # First try the efficient method that only checks adjacent blocks
        if not self._merge_with_adjacent_free_blocks(block):
            # If the efficient method didn't merge
            # fall back to full merge to handle complex cases
            self._merge_free_blocks()
        
        # Update access time
        self.last_access_time = time.time()
        
        return True
    
    def _add_to_allocated_list(self, block: MemoryBlock):
        """Add block to allocated list (sorted by start address)."""
        if self.allocated_blocks_head is None:
            self.allocated_blocks_head = block
            block.next = None
            return
        
        # Insert in sorted order
        prev = None
        current = self.allocated_blocks_head
        
        while current and current.start < block.start:
            prev = current
            current = current.next
        
        if prev is None:
            # Insert at head
            block.next = self.allocated_blocks_head
            self.allocated_blocks_head = block
        else:
            # Insert after prev
            block.next = current
            prev.next = block
    
    def _remove_from_allocated_list(self, block: MemoryBlock) -> bool:
        """Remove block from allocated list."""
        prev = None
        current = self.allocated_blocks_head
        
        while current:
            if current is block:
                if prev:
                    prev.next = current.next
                else:
                    self.allocated_blocks_head = current.next
                return True
            prev = current
            current = current.next
        
        return False
    
    def _add_to_free_list(self, block: MemoryBlock):
        """Add block to free list (sorted by start address)."""
        if self.free_blocks_head is None:
            self.free_blocks_head = block
            block.next = None
            return
        
        # Insert in sorted order
        prev = None
        current = self.free_blocks_head
        
        while current and current.start < block.start:
            prev = current
            current = current.next
        
        if prev is None:
            # Insert at head
            block.next = self.free_blocks_head
            self.free_blocks_head = block
        else:
            # Insert after prev
            block.next = current
            prev.next = block
    
    def _merge_free_blocks(self):
        """Merge all free blocks in the free list."""
        if self.free_blocks_head is None:
            return
        
        # Sort free blocks by start address first
        blocks = []
        current = self.free_blocks_head
        while current:
            blocks.append(current)
            current = current.next
        
        # Sort by start address
        blocks.sort(key=lambda b: b.start)
        
        # Rebuild linked list and merge adjacent blocks
        self.free_blocks_head = None
        prev_block = None
        
        for block in blocks:
            if prev_block is None:
                self.free_blocks_head = block
                block.next = None
                prev_block = block
            else:
                # Check if blocks are adjacent
                if prev_block.end == block.start:
                    # Merge with previous block
                    prev_block.size += block.size
                    # Don't add current block to list
                else:
                    # Add as separate block
                    prev_block.next = block
                    block.next = None
                    prev_block = block
    
    def _merge_with_adjacent_free_blocks(self, new_block: MemoryBlock) -> bool:
        """
        Merge a newly added free block with adjacent free blocks.
        This is more efficient than rebuilding the entire free list.
        
        The free list is maintained in sorted order by start address,
        so we can efficiently find adjacent blocks.
        
        Args:
            new_block: The newly added free block
            
        Returns:
            True if any merge occurred, False otherwise
        """
        if self.free_blocks_head is None:
            return False
        
        merged = False
        
        # Find the position of new_block in the free list
        prev = None
        current = self.free_blocks_head
        
        # First, try to find if new_block is adjacent to any existing block
        while current:
            if current.end == new_block.start:
                # Case 1: current block is immediately before new_block
                # Merge new_block into current block
                current.size += new_block.size
                merged = True
                
                # After merging with new_block, check if current is now adjacent to next block
                # This handles: free_block_A + new_block + free_block_B
                while current.next and current.end == current.next.start:
                    # Keep merging with consecutive adjacent blocks
                    current.size += current.next.size
                    current.next = current.next.next
                    merged = True
                
                break
            elif new_block.end == current.start:
                # Case 2: new_block is immediately before current block
                # Merge new_block into current block
                current.start = new_block.start
                current.size += new_block.size
                merged = True
                
                # After merging new_block into current, check if prev is adjacent to current
                # This handles: free_block_A + new_block + free_block_B
                if prev and prev.end == current.start:
                    # Merge prev with current
                    prev.size += current.size
                    prev.next = current.next
                    # Now check if the merged block is adjacent to the next block
                    while prev.next and prev.end == prev.next.start:
                        prev.size += prev.next.size
                        prev.next = prev.next.next
                
                break
            
            prev = current
            current = current.next
        
        return merged
    
    def get_block_by_offset(self, offset: int) -> Optional[MemoryBlock]:
        """Find allocated block by offset."""
        current = self.allocated_blocks_head
        while current:
            if current.start == offset:
                return current
            current = current.next
        return None
    
    def get_stats(self) -> dict:
        """Get segment statistics."""
        allocated_count = 0
        free_count = 0
        largest_free_block = 0
        
        current = self.allocated_blocks_head
        while current:
            allocated_count += 1
            current = current.next
        
        current = self.free_blocks_head
        while current:
            free_count += 1
            if current.size > largest_free_block:
                largest_free_block = current.size
            current = current.next
        
        return {
            "segment_id": self.segment_id,
            "total_size": self.size,
            "used_size": self.used_size,
            "free_size": self.free_size,
            "utilization": (self.used_size / self.size) * 100 if self.size > 0 else 0,
            "allocated_blocks_count": allocated_count,
            "free_blocks_count": free_count,
            "largest_free_block": largest_free_block
        }
    
    # VRAM Unit Management Methods
    def register_vram_unit(self, vram_unit: VRAMKVCacheUnit) -> bool:
        """
        Register a VRAM unit in this segment.
        
        Args:
            vram_unit: VRAM unit to register
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = vram_unit.cache_key
        if cache_key in self._vram_units:
            logger.debug(f"VRAM unit {cache_key} already registered in segment {self.segment_id}")
            return True
        
        self._vram_units[cache_key] = vram_unit
        logger.debug(f"Registered VRAM unit {cache_key} in segment {self.segment_id}")
        return True
    
    def unregister_vram_unit(self, cache_key: Union[str, CacheEngineKey]) -> bool:
        """
        Unregister a VRAM unit from this segment.
        
        Args:
            cache_key: Cache key of the VRAM unit to unregister
            
        Returns:
            True if successful, False otherwise
        """
        if cache_key not in self._vram_units:
            logger.warning(f"VRAM unit {cache_key} not found in segment {self.segment_id}")
            return False
        
        del self._vram_units[cache_key]
        logger.debug(f"Unregistered VRAM unit {cache_key} from segment {self.segment_id}")
        return True
    
    def get_vram_unit(self, cache_key: Union[str, CacheEngineKey]) -> Optional[VRAMKVCacheUnit]:
        """
        Get a VRAM unit from this segment and update its access time.
        
        Args:
            cache_key: Cache key of the VRAM unit
            
        Returns:
            VRAM unit or None if not found
        """
        vram_unit = self._vram_units.get(cache_key)
        if vram_unit:
            vram_unit.update_access_time()
            logger.debug(f"Updated access time for VRAM unit {cache_key} in segment {self.segment_id}")
        return vram_unit
    
    def get_all_vram_units(self) -> List[VRAMKVCacheUnit]:
        """
        Get all VRAM units in this segment.
        
        Returns:
            List of all VRAM units
        """
        return list(self._vram_units.values())
    
    def get_vram_unit_count(self) -> int:
        """
        Get the number of VRAM units in this segment.
        
        Returns:
            Number of VRAM units
        """
        return len(self._vram_units)
    
    def get_oldest_vram_unit(self) -> Optional[Tuple[Union[str, CacheEngineKey], VRAMKVCacheUnit]]:
        """
        Get the oldest VRAM unit in this segment (LRU).
        
        Returns:
            Tuple of (cache_key, vram_unit) or None if no VRAM units
        """
        if not self._vram_units:
            return None
        
        oldest_key = None
        oldest_time = float('inf')
        
        for cache_key, vram_unit in self._vram_units.items():
            if vram_unit.last_access_time < oldest_time:
                oldest_time = vram_unit.last_access_time
                oldest_key = cache_key
        
        if oldest_key:
            return oldest_key, self._vram_units[oldest_key]
        return None
    
    def clear_vram_units(self) -> bool:
        """
        Clear all VRAM units from this segment.
        
        Returns:
            True if successful
        """
        self._vram_units.clear()
        logger.debug(f"Cleared all VRAM units from segment {self.segment_id}")
        return True
    
    def get_vram_unit_stats(self) -> dict:
        """
        Get VRAM unit statistics for this segment.
        
        Returns:
            Dictionary with VRAM unit statistics
        """
        total_allocated_size = 0
        for vram_unit in self._vram_units.values():
            total_allocated_size += vram_unit.allocated_size
        
        return {
            "segment_id": self.segment_id,
            "vram_unit_count": len(self._vram_units),
            "total_allocated_size": total_allocated_size,
            "vram_unit_keys": list(self._vram_units.keys())
        }


class GPUVRAMSegmentManager:
    """
    GPU VRAM Segment Manager for managing GPU memory segments.
    Each vcache engine instance has its own segment manager for local GPU VRAM management.
    Handles segment allocation, deallocation, and space management for a specific GPU.
    handles real GPU memory allcation when _allocate_gpu_segment is called.
    after that, memory allocation just slice from pre-allocated segments.
    """
    
    def __init__(self, config, gpu_id: int, transfer_engine_manager=None, vram_metadata_client=None):
        self.config = config
        self.gpu_id = gpu_id
        self.transfer_engine_manager = transfer_engine_manager
        self.vram_metadata_client = vram_metadata_client 
        
        # GPU VRAM segment management for this specific GPU
        self.segments: List[GPUVRAMSegment] = []  # List of segments on this GPU
        self.segment_size_mb = config.get_extra_config_value("gpu_vram_segment_size_mb", 256)  # Default 256MB per segment
        
        # Store tensor references to prevent garbage collection
        self._segment_tensors: Dict[str, torch.Tensor] = {}
        
        # Initialize GPU segments for this specific GPU
        self._initialize_gpu_segments()
        
        logger.info(f"GPU VRAM Segment Manager initialized for GPU {gpu_id}")

    def _initialize_gpu_segments(self):
        """Initialize GPU VRAM segments for this specific GPU."""
        try:
            # Allocate initial segment for this GPU
            self._allocate_gpu_segment()
            logger.debug(f"Initialized GPU VRAM segments for GPU {self.gpu_id}")
        except Exception as e:
            logger.error(f"Failed to initialize GPU VRAM segments for GPU {self.gpu_id}: {e}")


    def _allocate_gpu_segment(self) -> Optional[GPUVRAMSegment]:
        """
        Allocate a GPU VRAM segment on this GPU using CUDA methods.
        
        Returns:
            GPUVRAMSegment object if successful, None otherwise
        """
        try:
            # Save current device
            original_device = torch.cuda.current_device()
            torch.cuda.set_device(self.gpu_id)
            
            # Calculate segment size in bytes (MB to bytes)
            segment_size_bytes = self.segment_size_mb * 1024 * 1024
            
            # Allocate GPU memory using PyTorch
            # We'll create a tensor of the required size to get GPU memory
            num_elements = segment_size_bytes
            
            # use uint8 tensor for 1-byte alignment
            tensor = torch.zeros(num_elements, dtype=torch.uint8, device='cuda')
            
            # Get the actual GPU memory address
            base_address = tensor.data_ptr()
            
            # Create segment ID
            segment_id = f"gpu_{self.gpu_id}_segment_{len(self.segments)}"
            
            # Create segment object
            segment = GPUVRAMSegment(
                segment_id=segment_id,
                gpu_id=self.gpu_id,
                base_address=base_address,
                size=segment_size_bytes
            )
            
            # Store tensor reference to prevent garbage collection
            self._segment_tensors[segment_id] = tensor
            
            # Add to segment tracking
            self.segments.append(segment)
            
            logger.info(f"Allocated GPU VRAM segment on GPU {self.gpu_id}: "
                        f"{self.segment_size_mb}MB, "
                        f"address: {hex(base_address)}, "
                        f"segment_id: {segment_id}")
            
            # Register IPC handle for the segment if IPC client is available
            assert self.vram_metadata_client is not None

            # Get IPC handle for the allocated GPU memory
            libcudart = ctypes.CDLL("libcudart.so")
            cudaIpcGetMemHandle = libcudart.cudaIpcGetMemHandle
            cudaIpcGetMemHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            cudaIpcGetMemHandle.restype = ctypes.c_int
            
            IPC_HANDLE_SIZE = 64
            handle_buf = ctypes.create_string_buffer(IPC_HANDLE_SIZE)
            rc = cudaIpcGetMemHandle(ctypes.cast(handle_buf, ctypes.c_void_p), ctypes.c_void_p(base_address))
            
            if rc == 0:
                handle_bytes = handle_buf.raw
                # Register the IPC handle with the metadata server
                success = self.vram_metadata_client.register_segment_ipc_handle(
                    segment_id=segment_id,
                    buffer_pointer=base_address,
                    handle_bytes=handle_bytes,
                    gpu_id=self.gpu_id,
                    size=segment_size_bytes
                )
                
                if success:
                    logger.debug(f"Registered IPC handle for segment {segment_id} via IPC client")
                else:
                    logger.error(f"Failed to register IPC handle for segment {segment_id} via IPC client")
            else:
                logger.error(f"Failed to get IPC handle for segment {segment_id}: CUDA error code {rc}")

            
            # NVLINK transfer engine doesn't need segment registration
            # The engine can directly transfer between GPU memory addresses
            if self.transfer_engine_manager and self.transfer_engine_manager.initialized:
                logger.debug(f"NVLINK transfer engine available for segment {segment_id}, no registration needed")
            
            # Restore original device
            torch.cuda.set_device(original_device)
            
            return segment
            
        except Exception as e:
            logger.error(f"Failed to allocate GPU VRAM segment on GPU {self.gpu_id}: {e}")
            # Restore original device in case of error
            torch.cuda.set_device(original_device)
            return None

    def get_available_segment(self, required_size: int) -> Optional[GPUVRAMSegment]:
        """
        Find an available GPU VRAM segment with sufficient space on this GPU.
        If no segment has a large enough contiguous free block, use LRU to evict VRAM units
        and merge freed space with adjacent free blocks.
        
        Args:
            required_size: Required size in bytes
            
        Returns:
            Available GPUVRAMSegment or None if no suitable segment found
        """
        # First pass: check if any segment has a large enough contiguous free block
        for segment in self.segments:
            if segment.is_active:
                current = segment.free_blocks_head
                while current:
                    if not current.is_allocated and current.size >= required_size:
                        return segment
                    current = current.next
        
        # No segment has a large enough contiguous free block
        # Try to evict LRU VRAM units to create larger contiguous free blocks
        logger.info(f"No segment has contiguous free block of size {required_size} bytes, "
                    f"attempting LRU eviction on GPU {self.gpu_id}")
        
        # Keep track of eviction attempts to avoid infinite loops
        max_eviction_attempts = 100  # Safety limit
        eviction_attempts = 0
        
        while eviction_attempts < max_eviction_attempts:
            # Find the segment with the oldest VRAM unit
            oldest_segment = None
            oldest_key = None
            oldest_time = float('inf')
            
            for segment in self.segments:
                if not segment.is_active:
                    continue
                    
                # Get oldest VRAM unit in this segment
                oldest_result = segment.get_oldest_vram_unit()
                if oldest_result:
                    cache_key, vram_unit = oldest_result
                    if vram_unit.last_access_time < oldest_time:
                        oldest_time = vram_unit.last_access_time
                        oldest_key = cache_key
                        oldest_segment = segment
            
            if oldest_key is None or oldest_segment is None:
                logger.error(f"No VRAM units to evict on GPU {self.gpu_id}")
                break
            
            logger.info(f"Evicting LRU VRAM unit {oldest_key} from segment {oldest_segment.segment_id}, "
                       f"last accessed at {oldest_time}, size: {vram_unit.allocated_size} bytes")
            
            # Evict the VRAM unit using the manager's remove method
            # This will free the memory block and unregister the VRAM unit
            success = self.remove_vram_unit(oldest_key)
            if not success:
                logger.error(f"Failed to evict VRAM unit {oldest_key}")
                break
            
            eviction_attempts += 1
            
            # After eviction, check if this segment now has a large enough contiguous free block
            current = oldest_segment.free_blocks_head
            while current:
                if not current.is_allocated and current.size >= required_size:
                    logger.info(f"After evicting {oldest_key}, "
                                f"segment {oldest_segment.segment_id} now has "
                                f"contiguous free block of size {current.size} >= {required_size}")
                    return oldest_segment
                current = current.next
            
            # Continue evicting more VRAM units if still not enough space
            logger.debug(f"Still not enough contiguous space after evicting {oldest_key}, "
                        f"continuing LRU eviction...")
        
        if eviction_attempts >= max_eviction_attempts:
            logger.warning(f"Reached maximum eviction attempts ({max_eviction_attempts}) "
                          f"but still no contiguous free block of size {required_size}")
        
        # If LRU eviction didn't work, try to allocate a new segment
        logger.info(f"LRU eviction didn't create enough contiguous space, "
                   f"allocating new segment on GPU {self.gpu_id}")
        new_segment = self._allocate_gpu_segment()
        return new_segment

    def allocate_in_segment(self, size: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Allocate space in a GPU VRAM segment for data storage on this GPU.
        only update the segment metadata, actual GPU memory is pre-allocated.
        
        Args:
            size: Required size in bytes
            
        Returns:
            Tuple of (segment_id, offset) or (None, None) if allocation failed
        """
        segment = self.get_available_segment(size)
        if not segment:
            logger.error(f"Failed to find available segment on GPU {self.gpu_id} for {size} bytes")
            return None, None
        
        # Use segment's new allocate method
        allocation_result = segment.allocate(size)
        if not allocation_result:
            logger.error(f"Failed to allocate {size} bytes in segment {segment.segment_id}")
            return None, None
        
        offset, allocated_block = allocation_result
        
        logger.debug(f"Allocated {size} bytes "
                     f"in segment {segment.segment_id} "
                     f"on GPU {self.gpu_id}, "
                     f"offset: {offset}, "
                     f"block_size: {allocated_block.size}")
        
        return segment.segment_id, offset

    def free_segment_space(self, segment_id: str, offset: int, size: int) -> bool:
        """
        Free previously allocated space in a GPU VRAM segment.
        only update the segment metadata, actual GPU memory is pre-allocated.
        
        Args:
            segment_id: Segment ID
            offset: Offset within segment
            size: Size to free in bytes
            
        Returns:
            True if successful, False otherwise
        """
        segment = self.get_segment_by_id(segment_id)
        if not segment:
            logger.error(f"Segment {segment_id} not found on GPU {self.gpu_id}")
            return False
        
        # Find the allocated block at this offset
        block = segment.get_block_by_offset(offset)
        if not block:
            logger.error(f"No allocated block found at offset {offset} in segment {segment_id}")
            return False
        
        # Verify block size matches
        if block.size != size:
            logger.warning(f"Block size mismatch: expected {size}, got {block.size}")
        
        # Free the block
        success = segment.free(block)
        if success:
            logger.debug(f"Freed {size} bytes in segment {segment_id} at offset {offset}")
        else:
            logger.error(f"Failed to free block at offset {offset} in segment {segment_id}")
        
        return success

    def get_segment_by_id(self, segment_id: str) -> Optional[GPUVRAMSegment]:
        """
        Get segment by segment ID on this GPU.
        
        Args:
            segment_id: Segment ID to find
            
        Returns:
            GPUVRAMSegment or None if not found
        """
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        return None

    def get_buffer_address(self, segment_id: str, offset: int) -> Optional[int]:
        """
        Calculate buffer address from segment ID and offset on this GPU.
        
        Args:
            segment_id: Segment ID
            offset: Offset within segment
            
        Returns:
            GPU buffer address or None if segment not found
        """
        segment = self.get_segment_by_id(segment_id)
        if not segment:
            logger.error(f"Segment {segment_id} not found on GPU {self.gpu_id}")
            return None
        
        if offset < 0 or offset > segment.size:
            logger.error(f"Offset {offset} out of bounds for segment {segment_id} (size: {segment.size})")
            return None
        
        return segment.base_address + offset


    def get_segment_stats(self) -> dict:
        """Get GPU VRAM segment statistics for this GPU."""
        stats = {
            "gpu_id": self.gpu_id,
            "total_segments": len(self.segments),
            "total_segment_size_bytes": 0,
            "total_used_segment_bytes": 0,
            "segment_utilization": {}
        }
        
        for segment in self.segments:
            stats["total_segment_size_bytes"] += segment.size
            stats["total_used_segment_bytes"] += segment.used_size
            
            # Calculate segment utilization
            utilization = (segment.used_size / segment.size) * 100 if segment.size > 0 else 0
            stats["segment_utilization"][segment.segment_id] = {
                "size_bytes": segment.size,
                "used_bytes": segment.used_size,
                "utilization_percent": utilization,
                "entry_count": segment.get_vram_unit_count()
            }
        
        return stats

    def cleanup_segment(self, segment_id: str) -> bool:
        """
        Clean up a GPU VRAM segment by removing all associated entries and VRAM units on this GPU.
        this is to clean up segment metadata, actual GPU memory is pre-allocated.
        
        Args:
            segment_id: Segment ID to clean up
            
        Returns:
            True if successful, False otherwise
        """
        segment = self.get_segment_by_id(segment_id)
        if segment is None:
            logger.warning(f"Segment {segment_id} not found for cleanup on GPU {self.gpu_id}")
            return False
        
        # clean segment VRAM units
        vram_units = segment.get_all_vram_units()
        for vram_unit in vram_units:
            cache_key = vram_unit.cache_key
            # free segment memory block
            block = segment.get_block_by_offset(vram_unit.segment_offset)
            if block:
                segment.free(block)
            # unregister VRAM unit
            segment.unregister_vram_unit(cache_key)
        
        # Find and reset the segment
        for seg in self.segments:
            if seg.segment_id == segment_id:
                # Free all allocated blocks
                current = seg.allocated_blocks_head
                while current:
                    next_block = current.next
                    seg.free(current)
                    current = next_block
                
                # Reset segment to initial state
                seg.free_blocks_head = MemoryBlock(start=0, size=seg.size, is_allocated=False)
                seg.allocated_blocks_head = None
                seg.last_access_time = time.time()
                
                seg.clear_vram_units()
                
                logger.info(f"Cleaned up segment {segment_id} "
                            f"on GPU {self.gpu_id}, "
                            f"freed all allocated blocks, reset free list, cleared VRAM units")
                return True
        
        logger.error(f"Segment {segment_id} not found in GPU segments on GPU {self.gpu_id}")
        return False


    def create_vram_unit(
        self,
        cache_key: Union[str, CacheEngineKey],
        token_ids: List[int],
        segment_id: str,
        offset: int,
        allocated_size: int,
        dtype: torch.dtype,
        original_shape: Optional[Tuple[int, ...]] = None
    ) -> Optional[VRAMKVCacheUnit]:
        """
        create a VRAM unit that references a flattened tensor slice from a GPU VRAM segment.
        use preallocated metadata (memory block in segment) to create the VRAM unit
        use metadata to create a flattened tensor slice from segment memory.
        
        Args:
            cache_key: cache key
            token_ids: token
            segment_id: segment ID
            offset: segment offset in bytes
            allocated_size: in bytes
            dtype: tensor data type
            original_shape: original shape of the tensor before flattening
            
        Returns:
            TestVRAMKVCacheUnit or None if creation failed
        """
        segment = self.get_segment_by_id(segment_id)
        if segment is None:
            logger.error(f"Segment {segment_id} not found")
            return None
        
        if offset < 0 or offset + allocated_size > segment.size:
            logger.error(f"Invalid allocation: offset={offset}, "
                         f"size={allocated_size}, "
                         f"segment_size={segment.size}")
            return None
        
        if segment_id not in self._segment_tensors:
            logger.error(f"Segment tensor not found for segment {segment_id}")
            return None
        
        segment_tensor = self._segment_tensors[segment_id]
        
        # get element size in bytes for the given dtype
        element_size_bytes = torch.tensor([], dtype=dtype).element_size()
        num_elements = allocated_size // element_size_bytes
        
        # check allocated_size is multiple of element size
        if allocated_size % element_size_bytes != 0:
            logger.error(f"Allocated size {allocated_size} bytes "
                         f"is not multiple of element size {element_size_bytes} bytes")
            return None
        
        # check bounds
        if offset + allocated_size > segment.size:
            logger.error(f"Not enough space in segment: "
                         f"offset={offset}, "
                         f"allocated_size={allocated_size}, "
                         f"segment_size={segment.size}")
            return None
        
        try:
            # slice the segment tensor to get the byte range
            # segment_tensor is uint8 tensor
            byte_slice = segment_tensor[offset:offset + allocated_size]
            
            # reshape to 1D tensor of the correct dtype
            flat_tensor = byte_slice.view(dtype)
            
            # check number of elements
            if len(flat_tensor) != num_elements:
                logger.error(f"Slice size mismatch: expected {num_elements} elements, got {len(flat_tensor)}")
                return None
            
            # check tenor size
            actual_tensor_size = flat_tensor.element_size() * flat_tensor.nelement()
            if actual_tensor_size != allocated_size:
                logger.error(f"Tensor size mismatch: expected {allocated_size}, got {actual_tensor_size}")
                return None
            
            vram_unit = VRAMKVCacheUnit(
                cache_key=cache_key,
                token_ids=token_ids,
                segment_id=segment_id,
                segment_offset=offset,
                allocated_size=allocated_size,
                segment_base_address=segment.base_address,
                kv_cache_tensor=flat_tensor, 
                gpu_id=self.gpu_id,
                original_shape=original_shape 
            )
            
            success = segment.register_vram_unit(vram_unit)
            if not success:
                logger.error(f"Failed to register VRAM unit {cache_key} in segment {segment_id}")
                return None
            
            logger.info(f"Created VRAM unit for flattened data: {cache_key} at segment {segment_id}, "
                       f"offset {offset}, size {allocated_size} bytes, dtype {dtype}, "
                       f"elements: {num_elements}, original_shape: {original_shape}")
            
            return vram_unit
            
        except Exception as e:
            logger.error(f"Failed to create flattened tensor slice from segment {segment_id}: {e}")
            return None

    def shutdown(self) -> bool:
        """
        Shutdown the GPU VRAM segment manager and release all resources.
        This should be called when the program is exiting.
        
        Returns:
            True if shutdown successful, False otherwise
        """
        logger.info(f"Shutting down GPU VRAM segment manager for GPU {self.gpu_id}")
        
        # Clean up all segments
        cleanup_success = self.cleanup_all_segments()
        return cleanup_success
