import logging
import sys
from typing import Optional


def setup_logger(name: Optional[str] = None, log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup a standardized logger for the application.
    Uses Rich for console output if available, otherwise falls back to standard StreamHandler.
    """
    logger = logging.getLogger(name)

    # If the logger already has handlers, assume it's already configured to avoid duplicate logs
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, markup=True, show_time=True, show_level=True, show_path=True)
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


# Create a default root logger configuration to catch everything if not explicitly named
def configure_root_logger(log_level: int = logging.INFO):
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(log_level)
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, markup=True)
        except ImportError:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
        root_logger.addHandler(handler)


configure_root_logger()
logger = setup_logger("sd_merger")
