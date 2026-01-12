"""
Stats Collector Module

This module provides a unified interface for collecting statistics from all
VCache engine components. It aggregates stats from:
1. VCache Engine itself
2. GPU VRAM Segment Manager
3. Transfer Engine Manager
4. VRAM Metadata IPC Client
5. Mooncake Storage Backend
6. Token Database

"""

from typing import Dict, Any, Optional, List
import time
from dataclasses import dataclass, asdict
import threading

from lmcache.vcache.logging.vcache_logging import init_logger

logger = init_logger(__name__)


@dataclass
class SegmentStats:
    """Statistics for a GPU VRAM segment."""
    segment_id: str
    total_size_bytes: int
    used_bytes: int
    free_bytes: int
    utilization_percent: float
    allocated_blocks_count: int
    free_blocks_count: int
    largest_free_block: int
    vram_unit_count: int
    total_allocated_size: int


@dataclass
class GPUVRAMStats:
    """Statistics for GPU VRAM management."""
    gpu_id: int
    total_segments: int
    total_allocations: int
    total_deallocations: int
    vram_unit_creations: int
    vram_unit_deletions: int
    segment_evictions: int
    segments: List[SegmentStats]  


@dataclass
class TransferEngineStats:
    """Statistics for transfer engine."""
    engine_type: str
    gpu_id: int
    max_concurrent_transfers: int = 0
    transfer_timeout_sec: float = 0.0
    pending_transfers: int = 0
    active_transfers: int = 0
    completed_transfers_count: int = 0
    successful_transfers: int = 0
    failed_transfers: int = 0
    total_transfer_bytes: int = 0
    successful_registers: int = 0
    failed_registers: int = 0
    successful_unregisters: int = 0
    failed_unregisters: int = 0
    engine_status: Dict[str, Any] = None


@dataclass
class VRAMMetadataStats:
    """Statistics for VRAM metadata IPC client."""
    total_requests: int
    failed_requests: int
    is_connected: bool
    server_address: str
    server_stats: Dict[str, Any]


@dataclass
class StorageBackendStats:
    """Statistics for storage backend."""
    retrieves: int
    stores: int
    lookups: int
    contains: int
    total_hit_tokens: int
    total_entries: int
    total_size_bytes: int


@dataclass
class TokenDatabaseStats:
    """Statistics for token database."""
    total_tokens_processed: int
    total_chunks_generated: int
    chunk_size: int
    save_unfull_chunk: bool
    hash_function: str = ""


@dataclass
class VCacheEngineStats:
    """Main VCache engine statistics."""
    # Basic info
    worker_id: int
    connector_role: str
    start_time: float
    uptime_seconds: float
    
    # Operation counts
    hits: int
    misses: int
    total_lookups: int
    total_stores: int
    total_retrieves: int
    gpu_vram_hits: int
    gpu_vram_misses: int
    cross_gpu_transfers: int
    
    # Subsystem stats
    gpu_vram_stats: Optional[GPUVRAMStats] = None
    transfer_engine_stats: Optional[TransferEngineStats] = None
    vram_metadata_stats: Optional[VRAMMetadataStats] = None
    storage_backend_stats: Optional[StorageBackendStats] = None
    token_database_stats: Optional[TokenDatabaseStats] = None


class StatsCollector:
    """
    Main statistics collector for VCache engine.
    
    This class provides methods to collect and aggregate statistics from
    all VCache engine components.
    """
    
    def __init__(self):
        self.lock = threading.RLock()
        self.start_time = time.time()
        
    def collect_vcache_engine_stats(
        self, 
        vcache_engine
    ) -> VCacheEngineStats:
        """
        Collect statistics from VCacheEngine instance.
        
        Args:
            vcache_engine: VCacheEngine instance
            
        Returns:
            VCacheEngineStats object with all collected statistics
        """
        with self.lock:
            # Calculate uptime
            uptime = time.time() - self.start_time
            
            # Collect subsystem stats
            gpu_vram_stats = None
            transfer_engine_stats = None
            vram_metadata_stats = None
            storage_backend_stats = None
            token_database_stats = None
            
            # Collect GPU VRAM segment manager stats
            if hasattr(vcache_engine, 'segment_manager') and vcache_engine.segment_manager:
                gpu_vram_stats = self._collect_gpu_vram_stats(vcache_engine.segment_manager)
            
            # Collect transfer engine manager stats
            if hasattr(vcache_engine, 'transfer_engine_manager') and vcache_engine.transfer_engine_manager:
                transfer_engine_stats = self._collect_transfer_engine_stats(vcache_engine.transfer_engine_manager)
            
            # Collect VRAM metadata IPC client stats
            if hasattr(vcache_engine, 'vram_metadata_client') and vcache_engine.vram_metadata_client:
                vram_metadata_stats = self._collect_vram_metadata_stats(vcache_engine.vram_metadata_client)
            
            # Collect storage backend stats
            if hasattr(vcache_engine, 'storage_backend') and vcache_engine.storage_backend:
                storage_backend_stats = self._collect_storage_backend_stats(vcache_engine.storage_backend)
            
            # Collect token database stats
            if hasattr(vcache_engine, 'token_database') and vcache_engine.token_database:
                token_database_stats = self._collect_token_database_stats(vcache_engine.token_database)
            
            return VCacheEngineStats(
                worker_id=vcache_engine.metadata.worker_id,
                connector_role=vcache_engine.config.connector_role,
                start_time=self.start_time,
                uptime_seconds=uptime,
                hits=vcache_engine.stats.get("hits", 0),
                misses=vcache_engine.stats.get("misses", 0),
                total_lookups=vcache_engine.stats.get("total_lookups", 0),
                total_stores=vcache_engine.stats.get("total_stores", 0),
                total_retrieves=vcache_engine.stats.get("total_retrieves", 0),
                gpu_vram_hits=vcache_engine.stats.get("gpu_vram_hits", 0),
                gpu_vram_misses=vcache_engine.stats.get("gpu_vram_misses", 0),
                cross_gpu_transfers=vcache_engine.stats.get("cross_gpu_transfers", 0),
                gpu_vram_stats=gpu_vram_stats,
                transfer_engine_stats=transfer_engine_stats,
                vram_metadata_stats=vram_metadata_stats,
                storage_backend_stats=storage_backend_stats,
                token_database_stats=token_database_stats
            )
    
    def _collect_gpu_vram_stats(self, segment_manager) -> Optional[GPUVRAMStats]:
        """Collect GPU VRAM segment manager statistics."""
        try:
            if hasattr(segment_manager, 'get_stats'):
                stats = segment_manager.get_stats()
                
                gpu_id = stats.get("gpu_id", 0)
                total_segments = stats.get("total_segments", 0)
                total_allocations = stats.get("total_allocations", 0)
                total_deallocations = stats.get("total_deallocations", 0)
                vram_unit_creations = stats.get("vram_unit_creations", 0)
                vram_unit_deletions = stats.get("vram_unit_deletions", 0)
                segment_evictions = stats.get("segment_evictions", 0)
                segment_details = stats.get("segments", [])
                
                # Calculate total segment size and used size from segments
                total_segment_size_bytes = 0
                total_used_segment_bytes = 0
                segments = []
                for segment_data in segment_details:
                    total_size = segment_data.get("total_size", 0)
                    used_size = segment_data.get("used_size", 0)
                    free_size = segment_data.get("free_size", 0)
                    
                    total_segment_size_bytes += total_size
                    total_used_segment_bytes += used_size
                    
                    # Calculate utilization percentage
                    utilization = (used_size / total_size * 100) if total_size > 0 else 0
                    
                    segments.append(SegmentStats(
                        segment_id=segment_data.get("segment_id", ""),
                        total_size_bytes=total_size,
                        used_bytes=used_size,
                        free_bytes=free_size,
                        utilization_percent=utilization,
                        allocated_blocks_count=segment_data.get("allocated_blocks_count", 0),
                        free_blocks_count=segment_data.get("free_blocks_count", 0),
                        largest_free_block=segment_data.get("largest_free_block", 0),
                        vram_unit_count=segment_data.get("vram_unit_count", 0),
                        total_allocated_size=segment_data.get("total_allocated_size", 0)
                    ))
                
                return GPUVRAMStats(
                    gpu_id=gpu_id,
                    total_segments=total_segments,
                    total_allocations=total_allocations,
                    total_deallocations=total_deallocations,
                    vram_unit_creations=vram_unit_creations,
                    vram_unit_deletions=vram_unit_deletions,
                    segment_evictions=segment_evictions,
                    segments=segments
                )
        except Exception as e:
            logger.error(f"Failed to collect GPU VRAM stats: {e}")
        
        return None
    
    def _collect_transfer_engine_stats(self, transfer_engine_manager) -> Optional[TransferEngineStats]:
        """Collect transfer engine manager statistics."""
        try:
            if hasattr(transfer_engine_manager, 'get_stats'):
                stats = transfer_engine_manager.get_stats()
                
                return TransferEngineStats(
                    engine_type=stats.get("engine_type", "unknown"),
                    gpu_id=stats.get("gpu_id", 0),
                    max_concurrent_transfers=stats.get("max_concurrent_transfers", 0),
                    transfer_timeout_sec=stats.get("transfer_timeout_sec", 0.0),
                    pending_transfers=stats.get("pending_transfers", 0),
                    active_transfers=stats.get("active_transfers", 0),
                    completed_transfers_count=stats.get("completed_transfers_count", 0),
                    successful_transfers=stats.get("successful_transfers", 0),
                    failed_transfers=stats.get("failed_transfers", 0),
                    total_transfer_bytes=stats.get("total_transfer_bytes", 0),
                    successful_registers=stats.get("successful_registers", 0),
                    failed_registers=stats.get("failed_registers", 0),
                    successful_unregisters=stats.get("successful_unregisters", 0),
                    failed_unregisters=stats.get("failed_unregisters", 0),
                    engine_status=stats
                )
        except Exception as e:
            logger.error(f"Failed to collect transfer engine stats: {e}")
        
        return None
    
    def _collect_vram_metadata_stats(self, vram_metadata_client) -> Optional[VRAMMetadataStats]:
        """Collect VRAM metadata IPC client statistics."""
        try:
            if hasattr(vram_metadata_client, 'get_stats'):
                stats = vram_metadata_client.get_stats()
                client_stats = stats.get("client_stats", {})
                server_stats = stats.get("server_stats", {})
                
                return VRAMMetadataStats(
                    total_requests=client_stats.get("total_requests", 0),
                    failed_requests=client_stats.get("failed_requests", 0),
                    is_connected=client_stats.get("is_connected", False),
                    server_address=client_stats.get("server_address", ""),
                    server_stats=server_stats
                )
        except Exception as e:
            logger.error(f"Failed to collect VRAM metadata stats: {e}")
        
        return None
    
    def _collect_storage_backend_stats(self, storage_backend) -> Optional[StorageBackendStats]:
        """Collect storage backend statistics."""
        try:
            if hasattr(storage_backend, 'get_stats'):
                stats = storage_backend.get_stats()
                
                return StorageBackendStats(
                    retrieves=stats.get("retrieves", 0),
                    stores=stats.get("stores", 0),
                    lookups=stats.get("lookups", 0),
                    contains=stats.get("contains", 0),
                    total_hit_tokens=stats.get("total_hit_tokens", 0),
                    total_entries=stats.get("total_entries", 0),
                    total_size_bytes=stats.get("total_size_bytes", 0)
                )
        except Exception as e:
            logger.error(f"Failed to collect storage backend stats: {e}")
        
        return None
    
    def _collect_token_database_stats(self, token_database) -> Optional[TokenDatabaseStats]:
        """Collect token database statistics."""
        try:
            if hasattr(token_database, 'get_stats'):
                stats = token_database.get_stats()
                
                return TokenDatabaseStats(
                    total_tokens_processed=stats.get("total_tokens_processed", 0),
                    total_chunks_generated=stats.get("total_chunks_generated", 0),
                    chunk_size=stats.get("chunk_size", 0),
                    save_unfull_chunk=stats.get("save_unfull_chunk", True),
                    hash_function=stats.get("hash_function", "")
                )
        except Exception as e:
            logger.error(f"Failed to collect token database stats: {e}")
        
        return None
    
    def format_stats(self, stats: VCacheEngineStats) -> Dict[str, Any]:
        """
        Format statistics for display or serialization.
        
        Args:
            stats: VCacheEngineStats object
            
        Returns:
            Dictionary with formatted statistics
        """
        result = {
            "vcache_engine": {
                "worker_id": stats.worker_id,
                "connector_role": stats.connector_role,
                "start_time": stats.start_time,
                "uptime_seconds": stats.uptime_seconds,
                "operation_counts": {
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "total_lookups": stats.total_lookups,
                    "total_stores": stats.total_stores,
                    "total_retrieves": stats.total_retrieves,
                    "gpu_vram_hits": stats.gpu_vram_hits,
                    "gpu_vram_misses": stats.gpu_vram_misses,
                    "cross_gpu_transfers": stats.cross_gpu_transfers
                }
            }
        }
        
        # Add subsystem stats if available
        if stats.gpu_vram_stats:
            result["gpu_vram"] = asdict(stats.gpu_vram_stats)
        
        if stats.transfer_engine_stats:
            result["transfer_engine"] = asdict(stats.transfer_engine_stats)
        
        if stats.vram_metadata_stats:
            result["vram_metadata"] = asdict(stats.vram_metadata_stats)
        
        if stats.storage_backend_stats:
            result["storage_backend"] = asdict(stats.storage_backend_stats)
        
        if stats.token_database_stats:
            result["token_database"] = asdict(stats.token_database_stats)
        
        return result
    
    def get_summary(self, stats: VCacheEngineStats) -> str:
        """
        Get a human-readable summary of statistics.
        
        Args:
            stats: VCacheEngineStats object
            
        Returns:
            String summary
        """
        summary = [
            f"VCache Engine Stats (GPU {stats.worker_id}, {stats.connector_role}):",
            f"  Uptime: {stats.uptime_seconds:.2f}s",
            f"  Operations: {stats.total_lookups} lookups, {stats.total_stores} stores, {stats.total_retrieves} retrieves",
            f"  Hits: {stats.hits}, Misses: {stats.misses}",
            f"  GPU VRAM Hits: {stats.gpu_vram_hits}, GPU VRAM Misses: {stats.gpu_vram_misses}",
            f"  Cross-GPU Transfers: {stats.cross_gpu_transfers}"
        ]
        
        if stats.gpu_vram_stats:
            # Calculate total segment size and used size from segments
            total_segment_size_bytes = 0
            total_used_segment_bytes = 0
            for segment in stats.gpu_vram_stats.segments:
                total_segment_size_bytes += segment.total_size_bytes
                total_used_segment_bytes += segment.used_bytes
            
            summary.extend([
                f"  GPU VRAM: {stats.gpu_vram_stats.total_segments} segments, "
                f"{total_used_segment_bytes:,} bytes used / "
                f"{total_segment_size_bytes:,} bytes total, "
                f"Allocations: {stats.gpu_vram_stats.total_allocations}, "
                f"Deallocations: {stats.gpu_vram_stats.total_deallocations}"
            ])
        
        if stats.vram_metadata_stats:
            summary.extend([
                f"  VRAM Metadata: {stats.vram_metadata_stats.total_requests} requests, "
                f"Failed: {stats.vram_metadata_stats.failed_requests}"
            ])
        
        if stats.storage_backend_stats:
            summary.extend([
                f"  Storage Backend: {stats.storage_backend_stats.stores} stores, "
                f"{stats.storage_backend_stats.retrieves} retrieves, "
                f"{stats.storage_backend_stats.total_hit_tokens} hit tokens"
            ])
        
        return "\n".join(summary)


# Global stats collector instance
_stats_collector = None

def get_stats_collector() -> StatsCollector:
    """Get or create the global stats collector instance."""
    global _stats_collector
    if _stats_collector is None:
        _stats_collector = StatsCollector()
    return _stats_collector
