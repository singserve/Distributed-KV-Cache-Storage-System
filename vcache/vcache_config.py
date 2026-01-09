"""
This module provides a VCacheConfig class with direct attributes for all configuration.
"""

from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class VCacheConfig:
    # Connector role: "scheduler" or "worker"
    connector_role: str = "worker"  # "scheduler" or "worker"
    
    # Used by TestCacheEngine for GPU VRAM pool initialization
    enable_gpu_vram_pool: bool = True
    use_vram_metadata_server: bool = True
    
    # Used by GPUVRAMPoolManager
    max_gpu_vram_metadata_size: int = 10000
    
    # GPU VRAM Segment Management
    gpu_vram_segment_size_mb: int = 631242752
    enable_gpu_vram_segments: bool = True
    
    # Used by MooncakeStorageBackend
    local_hostname: str = "localhost"
    global_segment_size: int = 3200  # MB
    local_buffer_size: int = 512    # MB
    master_server_address: str = "127.0.0.1:50051"
    metadata_server: str = "http://127.0.0.1:8080/metadata"

    # VRAM Metadata IPC Server
    vram_metadata_ipc_address: str = "192.168.1.86"
    vram_metadata_ipc_port:int = 9091

    # TransferEngineManager
    transfer_engine_type: str = "nvlink"
    gpu_id: int = 0

    # nvlink_transfer_engine
    max_concurrent_transfers: int = 8
    transfer_timeout_sec: int = 30

    # mooncake_transfer_engine
    local_hostname_TE: str = "localhost"  
    protocol: str = "rdma"
    protocol_TE: str = "nvlink" 
    device_name: str = "mlx5_0"

    @staticmethod
    def from_defaults() -> "VCacheConfig":
        """Create VCacheConfig with default values."""
        return VCacheConfig()
    
    @staticmethod
    def from_file(file_path: str) -> "VCacheConfig":
        """Load VCacheConfig from YAML file."""
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config = VCacheConfig()
        
        # Update attributes from YAML data
        for key, value in config_data.items():
            if key == "extra_config":
                # Handle extra_config section
                if isinstance(value, dict):
                    for extra_key, extra_value in value.items():
                        if hasattr(config, extra_key):
                            setattr(config, extra_key, extra_value)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def get_extra_config_value(self, key: str, default_value: Optional[str] = None) -> Optional[str]:
        """Compatibility method to support existing code that uses get_extra_config_value."""
        if hasattr(self, key):
            return getattr(self, key)
        return default_value
