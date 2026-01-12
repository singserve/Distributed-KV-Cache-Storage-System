"""
Stats Calculator Module

This module provides statistical calculations for VCache engine statistics.
It computes derived metrics such as averages, rates, totals, and other
statistical measures from raw statistics data.
"""

from typing import Dict, Any, Optional, List, Tuple
import time
import math
from dataclasses import dataclass, asdict

from lmcache.vcache.logging.vcache_logging import init_logger

logger = init_logger(__name__)


@dataclass
class CalculatedMetrics:
    """Container for calculated statistical metrics."""
    
    # VCache Engine metrics
    hit_rate: float = 0.0
    gpu_vram_hit_rate: float = 0.0
    cross_gpu_transfer_rate: float = 0.0
    
    # GPU VRAM metrics
    avg_segment_utilization: float = 0.0
    total_segment_efficiency: float = 0.0
    
    # Transfer Engine metrics
    transfer_success_rate: float = 0.0
    
    # VRAM Metadata metrics
    metadata_request_success_rate: float = 0.0
    
    # Storage Backend metrics
    storage_hit_rate: float = 0.0


class StatsCalculator:
    """
    Statistical calculator for VCache engine statistics.
    
    This class provides methods to calculate various derived metrics
    from raw statistics data collected by the StatsCollector.
    """
    
    def __init__(self):
        """Initialize the stats calculator."""
        pass
    
    def calculate_all_metrics(self, stats_dict: Dict[str, Any]) -> CalculatedMetrics:
        """
        Calculate all statistical metrics from raw statistics.
        
        Args:
            stats_dict: Dictionary containing raw statistics from StatsCollector
            
        Returns:
            CalculatedMetrics object with all calculated metrics
        """
        metrics = CalculatedMetrics()
        
        # Extract main stats
        vcache_stats = stats_dict.get("vcache_engine", {})
        gpu_vram_stats = stats_dict.get("gpu_vram", {})
        transfer_stats = stats_dict.get("transfer_engine", {})
        metadata_stats = stats_dict.get("vram_metadata", {})
        storage_stats = stats_dict.get("storage_backend", {})
        
        # Calculate VCache Engine metrics
        metrics = self._calculate_vcache_metrics(metrics, vcache_stats)
        
        # Calculate GPU VRAM metrics
        metrics = self._calculate_gpu_vram_metrics(metrics, gpu_vram_stats)
        
        # Calculate Transfer Engine metrics
        metrics = self._calculate_transfer_metrics(metrics, transfer_stats)
        
        # Calculate VRAM Metadata metrics
        metrics = self._calculate_metadata_metrics(metrics, metadata_stats)
        
        # Calculate Storage Backend metrics
        metrics = self._calculate_storage_metrics(metrics, storage_stats)
        
        return metrics
    
    def _calculate_vcache_metrics(self, metrics: CalculatedMetrics, 
                                 stats: Dict[str, Any]) -> CalculatedMetrics:
        """Calculate VCache engine specific metrics."""
        if not stats:
            return metrics
        
        op_counts = stats.get("operation_counts", {})
        
        # Hit rate
        hits = op_counts.get("hits", 0)
        misses = op_counts.get("misses", 0)
        total_lookups = hits + misses
        metrics.hit_rate = hits / max(total_lookups, 1)
        
        # GPU VRAM hit rate
        gpu_hits = op_counts.get("gpu_vram_hits", 0)
        gpu_misses = op_counts.get("gpu_vram_misses", 0)
        total_gpu_ops = gpu_hits + gpu_misses
        metrics.gpu_vram_hit_rate = gpu_hits / max(total_gpu_ops, 1)
        
        # Cross-GPU transfer rate
        cross_gpu_transfers = op_counts.get("cross_gpu_transfers", 0)
        total_retrieves = op_counts.get("total_retrieves", 0)
        metrics.cross_gpu_transfer_rate = cross_gpu_transfers / max(total_retrieves, 1)
        
        return metrics
    
    def _calculate_gpu_vram_metrics(self, metrics: CalculatedMetrics,
                                   stats: Dict[str, Any]) -> CalculatedMetrics:
        """Calculate GPU VRAM specific metrics."""
        if not stats:
            return metrics
        
        # Average segment utilization and calculate total size/used size
        segments = stats.get("segments", [])
        total_size = 0
        used_size = 0
        if segments:
            total_utilization = 0
            for segment in segments:
                segment_size = segment.get("total_size_bytes", 0)
                segment_used = segment.get("used_bytes", 0)
                segment_utilization = segment.get("utilization_percent", 0)
                
                total_size += segment_size
                used_size += segment_used
                total_utilization += segment_utilization
            
            metrics.avg_segment_utilization = total_utilization / len(segments)
        
        # Total segment efficiency
        metrics.total_segment_efficiency = used_size / max(total_size, 1)
        
        return metrics
    
    def _calculate_transfer_metrics(self, metrics: CalculatedMetrics,
                                   stats: Dict[str, Any]) -> CalculatedMetrics:
        """Calculate transfer engine specific metrics."""
        if not stats:
            return metrics
        
        # Transfer success rate
        successful = stats.get("successful_transfers", 0)
        failed = stats.get("failed_transfers", 0)
        total = successful + failed
        metrics.transfer_success_rate = successful / max(total, 1)
        
        return metrics
    
    def _calculate_metadata_metrics(self, metrics: CalculatedMetrics,
                                   stats: Dict[str, Any]) -> CalculatedMetrics:
        """Calculate VRAM metadata specific metrics."""
        if not stats:
            return metrics
        
        # Metadata request success rate
        total_requests = stats.get("total_requests", 0)
        failed_requests = stats.get("failed_requests", 0)
        successful_requests = total_requests - failed_requests
        metrics.metadata_request_success_rate = successful_requests / max(total_requests, 1)
        
        return metrics
    
    def _calculate_storage_metrics(self, metrics: CalculatedMetrics,
                                  stats: Dict[str, Any]) -> CalculatedMetrics:
        """Calculate storage backend specific metrics."""
        if not stats:
            return metrics
        
        # Storage hit rate - calculate based on lookups and total_hit_tokens
        total_hit_tokens = stats.get("total_hit_tokens", 0)
        lookups = stats.get("lookups", 0)
        if lookups > 0:
            # Estimate average tokens per lookup (assuming chunk_size of 256)
            # This is an approximation
            metrics.storage_hit_rate = total_hit_tokens / (lookups * 256)
        
        return metrics
    
    def _calculate_token_metrics(self, metrics: CalculatedMetrics,
                                stats: Dict[str, Any]) -> CalculatedMetrics:
        """Calculate token database specific metrics."""
        pass
    
    def get_calculated_metrics_dict(self, metrics: CalculatedMetrics) -> Dict[str, Any]:
        """
        Convert CalculatedMetrics to dictionary format.
        
        Args:
            metrics: CalculatedMetrics object
            
        Returns:
            Dictionary representation of calculated metrics
        """
        return {
            "vcache_engine_metrics": {
                "hit_rate": metrics.hit_rate,
                "gpu_vram_hit_rate": metrics.gpu_vram_hit_rate,
                "cross_gpu_transfer_rate": metrics.cross_gpu_transfer_rate
            },
            "gpu_vram_metrics": {
                "avg_segment_utilization": metrics.avg_segment_utilization,
                "total_segment_efficiency": metrics.total_segment_efficiency
            },
            "transfer_engine_metrics": {
                "transfer_success_rate": metrics.transfer_success_rate
            },
            "vram_metadata_metrics": {
                "metadata_request_success_rate": metrics.metadata_request_success_rate
            },
            "storage_backend_metrics": {
                "storage_hit_rate": metrics.storage_hit_rate
            }
        }
    
    def get_metrics_summary(self, metrics: CalculatedMetrics) -> str:
        """
        Get a human-readable summary of calculated metrics.
        
        Args:
            metrics: CalculatedMetrics object
            
        Returns:
            String summary of calculated metrics
        """
        summary = [
            "=== Calculated Statistics Metrics ===",
            f"VCache Engine:",
            f"  Hit Rate: {metrics.hit_rate:.2%}",
            f"  GPU VRAM Hit Rate: {metrics.gpu_vram_hit_rate:.2%}",
            f"  Cross-GPU Transfer Rate: {metrics.cross_gpu_transfer_rate:.2%}",
            "",
            f"GPU VRAM:",
            f"  Avg Segment Utilization: {metrics.avg_segment_utilization:.2f}%",
            f"  Total Segment Efficiency: {metrics.total_segment_efficiency:.2%}",
            "",
            f"Transfer Engine:",
            f"  Success Rate: {metrics.transfer_success_rate:.2%}",
            "",
            f"VRAM Metadata:",
            f"  Request Success Rate: {metrics.metadata_request_success_rate:.2%}",
            "",
            f"Storage Backend:",
            f"  Storage Hit Rate: {metrics.storage_hit_rate:.2%}",
            ""
        ]
        
        return "\n".join(summary)
