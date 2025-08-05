"""
Structured Logging Configuration
Provides centralized logging setup with proper formatting and levels.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_type: str = "standard",
    enable_console: bool = True
) -> None:
    """
    Setup centralized logging for the image enhancement pipeline
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_type: Format type (standard, detailed, json)
        enable_console: Whether to enable console logging
    """
    
    # Define log formats
    formats = {
        "standard": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        "json": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]"
    }
    
    log_format = formats.get(format_type, formats["standard"])
    
    # Create logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {},
        "loggers": {
            "": {  # Root logger
                "handlers": [],
                "level": level,
                "propagate": False
            }
        }
    }
    
    # Add console handler if enabled
    if enable_console:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        }
        config["loggers"][""]["handlers"].append("console")
    
    # Add file handler if log file specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": level,
            "formatter": "standard",
            "filename": log_file,
            "mode": "a"
        }
        config["loggers"][""]["handlers"].append("file")
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {level}, Console: {enable_console}, File: {log_file}")


def setup_pipeline_logging(
    base_level: str = "INFO",
    pipeline_level: str = None,
    services_level: str = None,
    log_dir: str = "logs"
) -> None:
    """
    Setup specialized logging for pipeline components
    
    Args:
        base_level: Base logging level for all loggers
        pipeline_level: Specific level for pipeline loggers
        services_level: Specific level for service loggers
        log_dir: Directory for log files
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Setup main logging
    main_log_file = log_path / "pipeline.log"
    setup_logging(
        level=base_level,
        log_file=str(main_log_file),
        format_type="detailed",
        enable_console=True
    )
    
    # Configure specific loggers
    loggers_config = {
        "pipelines": pipeline_level or base_level,
        "services": services_level or base_level,
        "core": base_level
    }
    
    for logger_name, level in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Add separate file handler for each component
        component_log_file = log_path / f"{logger_name}.log"
        file_handler = logging.FileHandler(component_log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


class PipelineLogger:
    """
    Specialized logger for pipeline operations with context management
    """
    
    def __init__(self, name: str, context: Dict[str, Any] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def _format_message(self, message: str) -> str:
        """Format message with context information"""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{message} [{context_str}]"
        return message
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self.logger.error(self._format_message(message), **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self.logger.critical(self._format_message(message), **kwargs)
    
    def update_context(self, **kwargs):
        """Update logger context"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear logger context"""
        self.context.clear()


class PerformanceLogger:
    """
    Logger for performance metrics and timing information
    """
    
    def __init__(self, name: str = "performance"):
        self.logger = logging.getLogger(f"performance.{name}")
        self.metrics = {}
    
    def log_timing(self, operation: str, duration: float, **metadata):
        """Log timing information for an operation"""
        self.logger.info(
            f"TIMING: {operation} completed in {duration:.3f}s",
            extra={"operation": operation, "duration": duration, **metadata}
        )
        self.metrics[operation] = {
            "duration": duration,
            "timestamp": __import__('time').time(),
            **metadata
        }
    
    def log_memory_usage(self, operation: str, memory_info: Dict[str, Any]):
        """Log memory usage information"""
        self.logger.info(
            f"MEMORY: {operation} - {memory_info}",
            extra={"operation": operation, "memory_info": memory_info}
        )
    
    def log_batch_metrics(self, batch_size: int, processing_time: float, success_rate: float):
        """Log batch processing metrics"""
        throughput = batch_size / processing_time if processing_time > 0 else 0
        self.logger.info(
            f"BATCH: {batch_size} images in {processing_time:.2f}s "
            f"(throughput: {throughput:.2f} img/s, success: {success_rate:.1%})",
            extra={
                "batch_size": batch_size,
                "processing_time": processing_time,
                "throughput": throughput,
                "success_rate": success_rate
            }
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset collected metrics"""
        self.metrics.clear()


def get_pipeline_logger(name: str, **context) -> PipelineLogger:
    """Get a pipeline logger with context"""
    return PipelineLogger(f"pipelines.{name}", context)


def get_service_logger(name: str, **context) -> PipelineLogger:
    """Get a service logger with context"""
    return PipelineLogger(f"services.{name}", context)


def get_performance_logger(name: str) -> PerformanceLogger:
    """Get a performance logger"""
    return PerformanceLogger(name)


# Context manager for logging operations
class LoggingContext:
    """Context manager for logging operations with automatic cleanup"""
    
    def __init__(self, logger: PipelineLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.start_context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = __import__('time').time()
        self.logger.update_context(operation=self.operation, **self.start_context)
        self.logger.info(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = __import__('time').time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {duration:.3f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")
        
        self.logger.clear_context()


# Default logging setup function
def configure_default_logging():
    """Configure default logging for the pipeline"""
    setup_pipeline_logging(
        base_level="INFO",
        pipeline_level="INFO",
        services_level="DEBUG",
        log_dir="logs"
    )