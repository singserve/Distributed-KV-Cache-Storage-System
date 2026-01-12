"""
Stats Manager Module

This module provides a simple manager for collecting statistics and
calculating derived metrics in one integrated interface.
"""

from typing import Dict, Any
import threading

from lmcache.vcache.logging.vcache_logging import init_logger
from lmcache.vcache.stats.stats_collector import StatsCollector
from lmcache.vcache.stats.stats_calculator import StatsCalculator

logger = init_logger(__name__)


class StatsManager:
    """
    Simple statistics manager for VCache engine.
    
    This class provides a unified interface for:
    1. Collecting raw statistics from all VCache components
    2. Calculating derived metrics (averages, rates, totals, etc.)
    """
    
    def __init__(self):
        """Initialize the stats manager."""
        self.lock = threading.RLock()
        
        # Initialize components
        self.stats_collector = StatsCollector()
        self.stats_calculator = StatsCalculator()
        
        logger.debug("StatsManager initialized")
    
    def collect_and_calculate(self, vcache_engine) -> Dict[str, Any]:
        """
        Collect statistics and calculate derived metrics in one operation.
        
        Args:
            vcache_engine: VCacheEngine instance
            
        Returns:
            Dictionary with both raw and calculated statistics
        """
        with self.lock:
            # Collect raw statistics
            raw_stats = self.stats_collector.collect_vcache_engine_stats(vcache_engine)
            
            # Format raw stats to dictionary
            raw_stats_dict = self.stats_collector.format_stats(raw_stats)
            
            # Calculate derived metrics
            calculated_metrics = self.stats_calculator.calculate_all_metrics(raw_stats_dict)
            
            # Get calculated metrics as dictionary
            calculated_dict = self.stats_calculator.get_calculated_metrics_dict(calculated_metrics)
            
            # Return comprehensive results
            return {
                "raw_statistics": raw_stats_dict,
                "calculated_metrics": calculated_dict,
                "summary": {
                    "raw": self.stats_collector.get_summary(raw_stats),
                    "calculated": self.stats_calculator.get_metrics_summary(calculated_metrics)
                }
            }
    
    def get_status(self, vcache_engine) -> Dict[str, Any]:
        """
        Get current status including both raw stats and calculated metrics.
        Alias for collect_and_calculate.
        
        Args:
            vcache_engine: VCacheEngine instance
            
        Returns:
            Dictionary with current status information
        """
        return self.collect_and_calculate(vcache_engine)
