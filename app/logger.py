"""Application-wide logging setup.

Import get_logger() anywhere in the app to obtain a configured logger. The
root log level and format are driven by the LOG_LEVEL and LOG_FORMAT env vars.
"""

import logging
import sys
from functools import lru_cache

from app.config import get_settings


def configure_logging() -> None:
    """Configure the root logger once based on application settings."""
    settings = get_settings()

    level = getattr(logging, settings.log_level.upper(), logging.DEBUG)

    # Avoid duplicate handlers when the module is reloaded (e.g., uvicorn --reload)
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(settings.log_format))
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party libraries unless DEBUG is requested
    if level > logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


@lru_cache()
def get_logger(name: str) -> logging.Logger:
    """Return a logger instance with the requested dotted name."""
    return logging.getLogger(name)
