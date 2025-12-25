from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import threading
import time
from collections import defaultdict
import torch

from lmcache.logging import init_logger
from lmcache.test.test_vram_kvcache_unit import TestVRAMKVCacheUnit
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey

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
    
    def merge_with_next(self) -> bool:
        """
        Merge this block with the next block if both are free.
        
        Returns:
            True if merge successful, False otherwise
        """
        if self.next is None or self.is_allocated or self.next.is_allocated:
            return False
        
        # Merge with next block
        self.size += self.next.size
        self.next = self.next.next
        return True


@dataclass
class GPUVRAMSegment:
    """GPU VRAM segment for data exchange, inspired by MooncakeStore segment management."""
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
    
    def __post_init__(self):
        if self.created_time == 0.0:
            self.created_time = time.time()
        if self.last_access_time == 0.0:
            self.last_access_time = time.time()
        
        # Initialize with one free block covering the entire segment
        self.free_blocks_head = MemoryBlock(start=0, size=self.size, is_allocated=False)
        self.allocated_blocks_head = None
    
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
                # This handles cases where the new block might be adjacent to multiple blocks
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
            # If the efficient method didn't merge (e.g., block is between two free blocks),
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
        """Merge adjacent free blocks in the free list."""
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
        
        This method handles three-way merges when new_block is between
        two free blocks (e.g., free_block_A -> new_block -> free_block_B).
        
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
        
        # If new_block wasn't merged yet, it might be between two free blocks
        # but we didn't find it in the first pass. This can happen if the free list
        # has multiple blocks and new_block is inserted in the middle.
        # We'll let the fallback _merge_free_blocks() handle this case.
        
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


class GPUVRAMSegmentManager:
    """
    GPU VRAM Segment Manager for managing GPU memory segments.
    Each vcache engine instance has its own segment manager for local GPU VRAM management.
    Handles segment allocation, deallocation, and space management for a specific GPU.
    """
    
    def __init__(self, config, gpu_id: int, transfer_engine_manager=None, vram_metadata_client=None):
        self.config = config
        self.gpu_id = gpu_id
        self.transfer_engine_manager = transfer_engine_manager
        self.vram_metadata_client = vram_metadata_client 
        self.lock = threading.RLock()
        
        # GPU VRAM segment management for this specific GPU
        self.segments: List[GPUVRAMSegment] = []  # List of segments on this GPU
        self.segment_size_mb = config.get_extra_config_value("gpu_vram_segment_size_mb", 256)  # Default 256MB per segment
        
        # Store tensor references to prevent garbage collection
        self._segment_tensors: Dict[str, torch.Tensor] = {}
        
        # VRAM Unit management
        self._vram_units: Dict[Union[str, CacheEngineKey], TestVRAMKVCacheUnit] = {}  # cache_key -> VRAM unit
        self._segment_to_units: Dict[str, List[Union[str, CacheEngineKey]]] = defaultdict(list)  # segment_id -> [cache_key]
        
        # Initialize GPU segments for this specific GPU
        self._initialize_gpu_segments()
        
        logger.info(f"GPU VRAM Segment Manager initialized for GPU {gpu_id} with VRAM Unit management")

    def _initialize_gpu_segments(self):
        """Initialize GPU VRAM segments for this specific GPU."""
        try:
            # Allocate initial segment for this GPU
            self._allocate_gpu_segment()
            logger.info(f"Initialized GPU VRAM segments for GPU {self.gpu_id}")
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
            
            logger.info(f"Allocated GPU VRAM segment on GPU {self.gpu_id}: {self.segment_size_mb}MB, "
                       f"address: {hex(base_address)}, segment_id: {segment_id}")
            
            # Register segment with transfer engine if available
            if self.transfer_engine_manager and self.transfer_engine_manager.initialized:
                try:
                    # Use transfer engine manager's method to register segment
                    if hasattr(self.transfer_engine_manager, '_register_segment_with_engine'):
                        self.transfer_engine_manager._register_segment_with_engine(segment)
                        logger.info(f"Successfully registered segment {segment_id} with transfer engine")
                    else:
                        logger.warning("Transfer engine manager does not have _register_segment_with_engine method")
                except Exception as e:
                    logger.error(f"Failed to register segment {segment_id} with transfer engine: {e}")
            
            # Restore original device
            torch.cuda.set_device(original_device)
            
            return segment
            
        except Exception as e:
            logger.error(f"Failed to allocate GPU VRAM segment on GPU {self.gpu_id}: {e}")
            # Restore original device in case of error
            try:
                torch.cuda.set_device(original_device)
            except:
                pass
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
        max_eviction_attempts = len(self._vram_units) * 2  # Safety limit
        eviction_attempts = 0
        
        while eviction_attempts < max_eviction_attempts:
            # Find the oldest VRAM unit
            oldest_key = None
            oldest_time = float('inf')
            
            for cache_key, vram_unit in self._vram_units.items():
                if vram_unit.last_access_time < oldest_time:
                    oldest_time = vram_unit.last_access_time
                    oldest_key = cache_key
            
            if oldest_key is None:
                logger.error(f"No VRAM units to evict on GPU {self.gpu_id}")
                break
            
            # Get the segment for this VRAM unit
            vram_unit = self._vram_units[oldest_key]
            segment_id = vram_unit.segment_id
            segment = self.get_segment_by_id(segment_id)
            
            if segment is None:
                logger.error(f"Segment {segment_id} not found for VRAM unit {oldest_key}")
                break
            
            logger.info(f"Evicting LRU VRAM unit {oldest_key} from segment {segment_id}, "
                       f"last accessed at {oldest_time}, size: {vram_unit.allocated_size} bytes")
            
            # Evict the VRAM unit
            success = self.remove_vram_unit(oldest_key)
            if not success:
                logger.error(f"Failed to evict VRAM unit {oldest_key}")
                break
            
            eviction_attempts += 1
            
            # After eviction, the segment will merge the freed block with adjacent free blocks
            # Check if this segment now has a large enough contiguous free block
            current = segment.free_blocks_head
            while current:
                if not current.is_allocated and current.size >= required_size:
                    logger.info(f"After evicting {oldest_key}, segment {segment_id} now has "
                               f"contiguous free block of size {current.size} >= {required_size}")
                    return segment
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
        
        logger.debug(f"Allocated {size} bytes in segment {segment.segment_id} on GPU {self.gpu_id}, "
                    f"offset: {offset}, block_size: {allocated_block.size}")
        
        return segment.segment_id, offset

    def free_segment_space(self, segment_id: str, offset: int, size: int) -> bool:
        """
        Free previously allocated space in a GPU VRAM segment.
        
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

    def register_entry_in_segment(self, segment_id: str, entry_key) -> bool:
        """
        Register an entry in a segment on this GPU.
        
        Args:
            segment_id: Segment ID
            entry_key: Entry key to register
            
        Returns:
            True if successful, False otherwise
        """
        # 使用_segment_to_units来跟踪entry
        # 检查entry是否已经存在，避免重复添加
        if entry_key in self._segment_to_units.get(segment_id, []):
            logger.debug(f"Entry {entry_key} already registered in segment {segment_id}")
            return True
            
        self._segment_to_units[segment_id].append(entry_key)
        logger.debug(f"Registered entry {entry_key} in segment {segment_id} on GPU {self.gpu_id}")
        return True

    def unregister_entry_from_segment(self, segment_id: str, entry_key) -> bool:
        """
        Unregister an entry from a segment on this GPU.
        
        Args:
            segment_id: Segment ID
            entry_key: Entry key to unregister
            
        Returns:
            True if successful, False otherwise
        """
        # 使用_segment_to_units来跟踪entry
        if segment_id not in self._segment_to_units:
            logger.warning(f"Segment {segment_id} not found for entry unregistration on GPU {self.gpu_id}")
            return False
        
        if entry_key in self._segment_to_units[segment_id]:
            self._segment_to_units[segment_id].remove(entry_key)
            logger.debug(f"Unregistered entry {entry_key} from segment {segment_id} on GPU {self.gpu_id}")
            return True
        
        logger.warning(f"Entry {entry_key} not found in segment {segment_id} on GPU {self.gpu_id}")
        return False

    def get_segment_stats(self) -> dict:
        """Get GPU VRAM segment statistics for this GPU."""
        with self.lock:
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
                    "entry_count": len(self._segment_to_units.get(segment.segment_id, []))
                }
            
            return stats

    def cleanup_segment(self, segment_id: str) -> bool:
        """
        Clean up a GPU VRAM segment by removing all associated entries and VRAM units on this GPU.
        
        Args:
            segment_id: Segment ID to clean up
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # 获取该segment中的所有VRAM units
            units_to_remove = []
            for cache_key, vram_unit in self._vram_units.items():
                if vram_unit.segment_id == segment_id:
                    units_to_remove.append(cache_key)
            
            # 移除所有VRAM units
            for cache_key in units_to_remove:
                self.remove_vram_unit(cache_key)
            
            # 使用_segment_to_units来清理entries
            if segment_id not in self._segment_to_units:
                logger.warning(f"Segment {segment_id} not found for cleanup on GPU {self.gpu_id}")
                return False
            
            # Remove all entries associated with this segment
            entries_to_remove = list(self._segment_to_units[segment_id])
            for entry_key in entries_to_remove:
                self.unregister_entry_from_segment(segment_id, entry_key)
            
            # Clear segment entries
            self._segment_to_units[segment_id].clear()
            
            # Find and reset the segment
            for segment in self.segments:
                if segment.segment_id == segment_id:
                    # Free all allocated blocks
                    current = segment.allocated_blocks_head
                    while current:
                        next_block = current.next
                        segment.free(current)
                        current = next_block
                    
                    # Reset segment to initial state
                    segment.free_blocks_head = MemoryBlock(start=0, size=segment.size, is_allocated=False)
                    segment.allocated_blocks_head = None
                    segment.last_access_time = time.time()
                    
                    logger.info(f"Cleaned up segment {segment_id} on GPU {self.gpu_id}, "
                               f"freed all allocated blocks, reset free list")
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
    ) -> Optional[TestVRAMKVCacheUnit]:
        """
        在指定segment位置创建VRAM unit，用于存储展平的一维数据
        
        Args:
            cache_key: 缓存键（可以是字符串、CacheEngineKey或LayerCacheEngineKey）
            token_ids: token序列
            segment_id: segment ID
            offset: segment内的偏移量（字节）
            allocated_size: 分配的大小（字节）
            dtype: tensor数据类型
            original_shape: 原始tensor形状（用于恢复展平的数据）
            
        Returns:
            TestVRAMKVCacheUnit实例或None
        """
        with self.lock:
            # 获取segment信息
            segment = self.get_segment_by_id(segment_id)
            if segment is None:
                logger.error(f"Segment {segment_id} not found")
                return None
            
            # 验证偏移量和大小
            if offset < 0 or offset + allocated_size > segment.size:
                logger.error(f"Invalid allocation: offset={offset}, size={allocated_size}, segment_size={segment.size}")
                return None
            
            # 获取segment tensor
            if segment_id not in self._segment_tensors:
                logger.error(f"Segment tensor not found for segment {segment_id}")
                return None
            
            segment_tensor = self._segment_tensors[segment_id]
            
            # 计算在目标dtype中的元素数量
            element_size_bytes = torch.tensor([], dtype=dtype).element_size()
            num_elements = allocated_size // element_size_bytes
            
            # 验证分配的大小是否足够容纳整数个元素
            if allocated_size % element_size_bytes != 0:
                logger.error(f"Allocated size {allocated_size} bytes is not multiple of element size {element_size_bytes} bytes")
                return None
            
            # 检查是否有足够的空间（以字节为单位）
            if offset + allocated_size > segment.size:
                logger.error(f"Not enough space in segment: offset={offset}, "
                           f"allocated_size={allocated_size}, segment_size={segment.size}")
                return None
            
            try:
                # 直接切片：获取segment中对应位置的slice
                # segment_tensor是uint8类型的一维tensor
                byte_slice = segment_tensor[offset:offset + allocated_size]
                
                # 将uint8切片重新解释为目标dtype的一维tensor
                # 使用view()方法改变dtype
                flat_tensor = byte_slice.view(dtype)
                
                # 验证切片大小
                if len(flat_tensor) != num_elements:
                    logger.error(f"Slice size mismatch: expected {num_elements} elements, got {len(flat_tensor)}")
                    return None
                
                # 验证tensor大小
                actual_tensor_size = flat_tensor.element_size() * flat_tensor.nelement()
                if actual_tensor_size != allocated_size:
                    logger.error(f"Tensor size mismatch: expected {allocated_size}, got {actual_tensor_size}")
                    return None
                
                # 创建VRAM unit（存储一维展平数据）
                vram_unit = TestVRAMKVCacheUnit(
                    cache_key=cache_key,
                    token_ids=token_ids,
                    segment_id=segment_id,
                    segment_offset=offset,
                    allocated_size=allocated_size,
                    segment_base_address=segment.base_address,
                    kv_cache_tensor=flat_tensor,  # 一维展平数据
                    gpu_id=self.gpu_id,
                    original_shape=original_shape  # 设置原始形状
                )
                
                # 注册到内部管理
                self._register_vram_unit(vram_unit)
                
                # 注册到segment entries（保持向后兼容）
                self.register_entry_in_segment(segment_id, cache_key)
                
                logger.info(f"Created VRAM unit for flattened data: {cache_key} at segment {segment_id}, "
                           f"offset {offset}, size {allocated_size} bytes, dtype {dtype}, "
                           f"elements: {num_elements}, original_shape: {original_shape}")
                
                return vram_unit
                
            except Exception as e:
                logger.error(f"Failed to create flattened tensor slice from segment {segment_id}: {e}")
                return None

    def _register_vram_unit(self, vram_unit: TestVRAMKVCacheUnit):
        """内部注册VRAM unit"""
        cache_key = vram_unit.cache_key
        self._vram_units[cache_key] = vram_unit
        self._segment_to_units[vram_unit.segment_id].append(cache_key)
        
        logger.debug(f"Registered VRAM unit: {cache_key} in segment {vram_unit.segment_id}")

    def get_vram_unit(self, cache_key: Union[str, CacheEngineKey]) -> Optional[TestVRAMKVCacheUnit]:
        """根据cache key获取VRAM unit，并更新访问时间"""
        with self.lock:
            vram_unit = self._vram_units.get(cache_key)
            if vram_unit:
                vram_unit.update_access_time()
                logger.debug(f"Updated access time for VRAM unit: {cache_key}")
            return vram_unit

    def batch_get_vram_units(self, cache_keys: List[Union[str, CacheEngineKey]]) -> List[Optional[TestVRAMKVCacheUnit]]:
        """
        批量获取VRAM units，并更新访问时间
        
        Args:
            cache_keys: cache key列表
            
        Returns:
            对应的VRAM unit列表，如果某个key不存在则返回None
        """
        with self.lock:
            result = []
            for cache_key in cache_keys:
                vram_unit = self._vram_units.get(cache_key)
                if vram_unit:
                    vram_unit.update_access_time()
                    logger.debug(f"Updated access time for VRAM unit: {cache_key}")
                result.append(vram_unit)
            return result

    def remove_vram_unit(self, cache_key: Union[str, CacheEngineKey]) -> bool:
        """
        移除VRAM unit并释放segment中的内存块
        
        Args:
            cache_key: 要移除的cache key
            
        Returns:
            是否成功移除
        """
        with self.lock:
            if cache_key not in self._vram_units:
                return False
            
            vram_unit = self._vram_units[cache_key]
            segment_id = vram_unit.segment_id
            offset = vram_unit.segment_offset
            size = vram_unit.allocated_size
            
            # 首先释放segment中的内存块
            segment = self.get_segment_by_id(segment_id)
            if segment:
                # 找到对应的内存块
                block = segment.get_block_by_offset(offset)
                if block:
                    # 释放内存块
                    success = segment.free(block)
                    if not success:
                        logger.warning(f"Failed to free memory block at offset {offset} in segment {segment_id}")
                else:
                    logger.warning(f"Memory block not found at offset {offset} in segment {segment_id}")
            
            # 清理segment索引
            if cache_key in self._segment_to_units[segment_id]:
                self._segment_to_units[segment_id].remove(cache_key)
                if not self._segment_to_units[segment_id]:
                    del self._segment_to_units[segment_id]
            
            # 清理segment entries
            self.unregister_entry_from_segment(segment_id, cache_key)
            
            # 移除VRAM unit
            del self._vram_units[cache_key]
            
            # 清理GPU VRAM Pool Manager中的metadata（如果可用）
            if self.vram_metadata_client is not None:
                try:
                    # 检查cache_key是否是CacheEngineKey类型
                    if isinstance(cache_key, CacheEngineKey):
                        # 调用GPU VRAM Pool Manager的remove方法清理metadata
                        metadata_removed = self.vram_metadata_client.remove(cache_key)
                        if metadata_removed:
                            logger.info(f"Successfully removed metadata from GPU VRAM Pool Manager for key: {cache_key}")
                        else:
                            logger.warning(f"Failed to remove metadata from GPU VRAM Pool Manager for key: {cache_key}")
                    else:
                        logger.warning(f"Cache key is not CacheEngineKey type, cannot remove from GPU VRAM Pool Manager: {type(cache_key)}")
                except Exception as e:
                    logger.error(f"Error removing metadata from GPU VRAM Pool Manager: {e}")
            
            logger.info(f"Removed VRAM unit: {cache_key} from segment {segment_id}, "
                       f"freed {size} bytes at offset {offset}")
            return True

    def get_vram_units_by_segment(self, segment_id: str) -> List[TestVRAMKVCacheUnit]:
        """获取指定segment中的所有VRAM units"""
        with self.lock:
            cache_keys = self._segment_to_units.get(segment_id, [])
            return [self._vram_units[key] for key in cache_keys if key in self._vram_units]

    def get_vram_unit_stats(self) -> dict:
        """获取VRAM unit统计信息"""
        with self.lock:
            stats = {
                "total_vram_units": len(self._vram_units),
                "vram_units_by_segment": {},
                "total_allocated_size": 0
            }
            
            # 按segment统计VRAM units
            for segment_id, cache_keys in self._segment_to_units.items():
                stats["vram_units_by_segment"][segment_id] = len(cache_keys)
            
            # 计算总分配大小
            for vram_unit in self._vram_units.values():
                stats["total_allocated_size"] += vram_unit.allocated_size
            
            return stats


    def cleanup_all_segments(self) -> bool:
        """
        清理所有segments和VRAM units
        This should be called when the program is shutting down.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        with self.lock:
            logger.info(f"Starting cleanup of all GPU VRAM segments and VRAM units on GPU {self.gpu_id}")
            
            # 清理所有VRAM units
            self._vram_units.clear()
            self._segment_to_units.clear()
            
            # Get all segment IDs from this GPU
            all_segment_ids = [segment.segment_id for segment in self.segments]
            
            # Clean up all segment entries
            for segment_id in all_segment_ids:
                self.cleanup_segment(segment_id)
            
            # Release all GPU memory allocations
            for segment_id, tensor in self._segment_tensors.items():
                try:
                    # Delete the tensor to release GPU memory
                    del tensor
                    logger.debug(f"Released GPU memory for segment {segment_id} on GPU {self.gpu_id}")
                except Exception as e:
                    logger.error(f"Failed to release GPU memory for segment {segment_id} on GPU {self.gpu_id}: {e}")
            
            # Clear the segment tensors dictionary
            self._segment_tensors.clear()
            
            # Clear all segment tracking
            self.segments.clear()
            
            logger.info(f"Successfully cleaned up all GPU VRAM segments and VRAM units on GPU {self.gpu_id}")
            return True

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
