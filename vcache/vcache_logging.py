"""
A simple unified logging system for the VCache distributed KV cache storage system.
Provides consistent logging across all components with configurable log levels.
"""

import logging
import sys
import os
from typing import Optional, Dict, Any
import threading

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Log level mapping
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

class VCacheLogFormatter(logging.Formatter):
    """Custom formatter for VCache logs."""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        if fmt is None:
            fmt = DEFAULT_LOG_FORMAT
        if datefmt is None:
            datefmt = DEFAULT_DATE_FORMAT
        super().__init__(fmt, datefmt)

# Global state
_loggers = {}
_default_level = logging.INFO
_handlers = []
_lock = threading.RLock()  
_initialized = False

def _initialize():
    """Initialize the logging system."""
    global _initialized
    if _initialized:
        return
    
    with _lock:
        if _initialized:
            return
        
        # Clear any existing handlers from root logger
        logging.root.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(VCacheLogFormatter())
        console_handler.setLevel(_default_level)
        
        # Add handler to root logger
        logging.root.addHandler(console_handler)
        logging.root.setLevel(_default_level)
        
        _handlers.append(console_handler)
        
        _initialized = True
    
    _initialize_from_environment()

def _initialize_from_environment():
    """Initialize logging from environment variables."""
    import os
    
    # Check for log file path in environment variable
    log_file = os.environ.get('VCACHE_LOG_FILE')
    if log_file:
        # Get log level from environment or use default
        log_level = os.environ.get('VCACHE_LOG_LEVEL', 'INFO')
        
        # Add file logging
        try:
            add_file_logging(log_file, level=log_level)
            print(f"VCache logging: Log file configured from environment: {log_file} (level: {log_level})")
        except Exception as e:
            print(f"VCache logging: Failed to configure log file from environment: {e}")
    
    # Check for log level override
    env_log_level = os.environ.get('VCACHE_LOG_LEVEL')
    if env_log_level:
        try:
            set_log_level(env_log_level)
            print(f"VCache logging: Log level set from environment: {env_log_level}")
        except Exception as e:
            print(f"VCache logging: Failed to set log level from environment: {e}")

def init_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Initialize and return a logger for the given module.
    
    This is the main entry point for getting a logger in VCache components.
    
    Args:
        name: Logger name (usually __name__)
        level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    # Ensure logging system is initialized
    if not _initialized:
        _initialize()
    
    with _lock:
        if name in _loggers:
            return _loggers[name]
        
        # Create new logger
        logger = logging.getLogger(name)
        
        # Set log level
        if level is not None:
            log_level = LOG_LEVELS.get(level.upper(), _default_level)
            logger.setLevel(log_level)
        else:
            logger.setLevel(_default_level)
        
        # Don't propagate to root logger
        logger.propagate = False
        
        # Add handlers if logger doesn't have any
        if not logger.handlers:
            for handler in _handlers:
                logger.addHandler(handler)
        
        _loggers[name] = logger
        return logger

def set_log_level(level: str):
    """
    Set the global log level.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _default_level
    
    with _lock:
        log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
        _default_level = log_level
        
        # Update root logger
        logging.root.setLevel(log_level)
        
        # Update all existing loggers
        for logger in _loggers.values():
            logger.setLevel(log_level)
        
        # Update all handlers
        for handler in _handlers:
            handler.setLevel(log_level)

def add_file_logging(filepath: str, level: Optional[str] = None):
    """
    Add file logging to the specified path.
    
    Args:
        filepath: Path to log file
        level: Optional log level for file handler
    """
    from logging.handlers import RotatingFileHandler
    
    with _lock:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            filepath,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        
        # Set formatter
        file_handler.setFormatter(VCacheLogFormatter())
        
        # Set level
        if level is not None:
            log_level = LOG_LEVELS.get(level.upper(), _default_level)
            file_handler.setLevel(log_level)
        else:
            file_handler.setLevel(_default_level)
        
        # Add handler to root logger
        logging.root.addHandler(file_handler)
        
        # Add handler to all existing loggers
        for logger in _loggers.values():
            logger.addHandler(file_handler)
        
        _handlers.append(file_handler)

def setup_default_logging():
    """Set up default logging configuration."""
    # This is a no-op now since initialization happens on first logger creation
    # But we keep it for API compatibility
    if not _initialized:
        _initialize()

def shutdown_logging():
    """Shutdown the logging system."""
    with _lock:
        for handler in _handlers:
            handler.close()
        _handlers.clear()
        _loggers.clear()
        global _initialized
        _initialized = False

# Export public API
__all__ = [
    'init_logger',
    'set_log_level',
    'add_file_logging',
    'setup_default_logging',
    'shutdown_logging',
]
