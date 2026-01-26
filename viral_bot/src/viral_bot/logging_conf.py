"""
Structured logging configuration using structlog.

Provides consistent, parseable logs for production monitoring.
"""

import logging
import sys
from typing import Optional

import structlog
from structlog.types import Processor


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    run_id: Optional[str] = None
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output logs as JSON (good for production)
        run_id: Optional run ID to include in all log entries
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    
    # Build processor chain
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_output:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Pretty console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Bind run_id if provided
    if run_id:
        structlog.contextvars.bind_contextvars(run_id=run_id)


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Optional logger name (usually __name__)
    
    Returns:
        Configured structlog logger
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger=name)
    return logger


# Convenience function for binding context
def bind_context(**kwargs) -> None:
    """Bind additional context to all subsequent log entries."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context."""
    structlog.contextvars.clear_contextvars()
