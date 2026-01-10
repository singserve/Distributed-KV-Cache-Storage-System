"""
Test script for VCache logging system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vcache.vcache_logging import init_logger

def test_logging_basic():
    """Test basic logging functionality."""
    print("Testing basic logging functionality...")
    
    # Get a logger
    logger = init_logger(__name__)
    
    # Test all log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test with formatting
    logger.info("Formatted message: %s %d", "test", 123)
    
    print("Basic logging test completed")

def test_logging_exception():
    """Test exception logging."""
    print("\nTesting exception logging...")
    
    logger = init_logger(__name__)
    
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        logger.exception("Caught an exception: %s", str(e))
    
    print("Exception logging test completed")

def test_logging_different_modules():
    """Test logging from different modules."""
    print("\nTesting logging from different modules...")
    
    # Simulate different modules
    module1_logger = init_logger("module1")
    module2_logger = init_logger("module2")
    
    module1_logger.info("Message from module1")
    module2_logger.info("Message from module2")
    
    print("Multiple modules test completed")

def test_log_level_control():
    """Test log level control."""
    print("\nTesting log level control...")
    
    from vcache.vcache_logging import set_log_level
    
    # Set to DEBUG level
    set_log_level("DEBUG")
    logger = init_logger(__name__)
    logger.debug("This debug message should appear")
    
    # Set to WARNING level
    set_log_level("WARNING")
    logger.debug("This debug message should NOT appear")
    logger.info("This info message should NOT appear")
    logger.warning("This warning message should appear")
    
    # Reset to INFO
    set_log_level("INFO")
    print("Log level control test completed")

def test_file_logging():
    """Test file logging."""
    print("\nTesting file logging...")
    
    from vcache.vcache_logging import add_file_logging
    import tempfile
    import time
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_file = f.name
    
    try:
        # Add file logging
        add_file_logging(log_file, level="DEBUG")
        
        logger = init_logger(__name__)
        logger.info("This message should go to both console and file")
        logger.debug("This debug message should go to file")
        
        # Give time for logs to be written
        time.sleep(0.1)
        
        # Read the log file
        with open(log_file, 'r') as f:
            content = f.read()
            print(f"Log file content ({len(content)} bytes):")
            print(content[:500] + "..." if len(content) > 500 else content)
        
        print("File logging test completed")
    finally:
        # Clean up
        try:
            os.unlink(log_file)
        except:
            pass

def main():
    """Run all tests."""
    print("=" * 60)
    print("VCache Logging System Test Suite")
    print("=" * 60)
    
    test_logging_basic()
    test_logging_exception()
    test_logging_different_modules()
    test_log_level_control()
    test_file_logging()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
