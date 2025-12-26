import sys
import time
import signal
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from the same directory
from vcache_config import VCacheConfig

"""
Simple script to start the VRAM metadata IPC server only.
"""
def create_config() -> VCacheConfig:
    """Create configuration for a specific GPU"""
    config_file = f"test_system_config_gpu0.yaml"
    print(f"Loading configuration from {config_file}")
    return VCacheConfig.from_file(config_file)

def start_vram_metadata_ipc_server():
    """Start VRAM metadata IPC server"""
    try:
        print("Starting VRAM metadata IPC server...")
        
        # Import and start the VRAM metadata IPC server
        # Import from the same directory
        from vram_metadata_ipc_server import start_vram_metadata_ipc_server
        
        config = create_config()
        server = start_vram_metadata_ipc_server(config=config)
        
        print("VRAM metadata IPC server started on port 9091")
        print("Server is running...")
        print("Press Ctrl+C to stop the server")
        
        return server
        
    except ImportError as e:
        print(f"Failed to import VRAM metadata IPC server: {e}")
        print("Make sure LMCache is properly installed")
        return None
    except Exception as e:
        print(f"Failed to start VRAM metadata IPC server: {e}")
        return None

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    print("\nShutting down IPC server...")
    sys.exit(0)

def main():
    """Main function to start the IPC server"""
    print("=== VRAM Metadata IPC Server ===")
    os.environ["LMCache_LOG_LEVEL"] = "DEBUG"
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the IPC server
    server = start_vram_metadata_ipc_server()
    
    if server is None:
        print("Failed to start IPC server. Exiting.")
        return
    
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("IPC server shutdown complete")

if __name__ == "__main__":
    main()
