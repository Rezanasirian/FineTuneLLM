"""
Centralized Logging System
Provides structured logging with file rotation and different log levels
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class LoggerSetup:
    """Setup and configure loggers for the application."""

    _loggers = {}

    @classmethod
    def setup_logger(
        cls,
        name: str,
        log_file: Optional[str] = None,
        level: str = "INFO",
        log_dir: str = "/../../app_log",
        console_output: bool = True,
        file_output: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ) -> logging.Logger:
        """
        Setup a logger with file and console handlers.

        Args:
            name: Logger name
            log_file: Log file name (default: {name}.log)
            level: Logging level
            log_dir: Directory for log files
            console_output: Enable console output
            file_output: Enable file output
            max_bytes: Max size of log file before rotation
            backup_count: Number of backup files to keep

        Returns:
            Configured logger
        """
        # Return existing logger if already configured
        if name in cls._loggers:
            return cls._loggers[name]

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = False

        # Clear existing handlers
        logger.handlers.clear()

        # Format strings
        detailed_format = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
        )
        simple_format = '%(asctime)s - %(levelname)s - %(message)s'

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = ColoredFormatter(simple_format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler
        if file_output:
            # Create log directory
            log_path = Path(log_dir)
            print(f"log path {log_path}")
            log_path.mkdir(parents=True, exist_ok=True)

            # Log file path
            if log_file is None:
                log_file = f"{name}.log"
            file_path = log_path / log_file

            # Rotating file handler
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_formatter = logging.Formatter(detailed_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # Store logger
        cls._loggers[name] = logger

        return logger

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get existing logger or create new one."""
        if name not in cls._loggers:
            return cls.setup_logger(name)
        return cls._loggers[name]

    @classmethod
    def setup_application_loggers(cls, base_level: str = "INFO"):
        """Setup all application loggers."""
        # Main application logger
        cls.setup_logger("app", level=base_level)

        # Component loggers
        cls.setup_logger("training", log_file="./../app_log/training.log", level="INFO")
        cls.setup_logger("evaluation", log_file="./../app_log/evaluation.log", level="INFO")
        cls.setup_logger("api", log_file="./../app_log/api.log", level="INFO")
        cls.setup_logger("feedback", log_file="./../app_log/feedback.log", level="INFO")
        cls.setup_logger("pipeline", log_file="./../app_log/pipeline.log", level="INFO")
        cls.setup_logger("monitoring", log_file="./../app_log/monitoring.log", level="WARNING")

        return cls.get_logger("app")


# Convenience function
def get_logger(name: str = "app") -> logging.Logger:
    """Get a logger instance."""
    return LoggerSetup.get_logger(name)


# Setup default loggers when module is imported
try:
    LoggerSetup.setup_application_loggers()
except Exception as e:
    print(f"Warning: Could not setup loggers: {e}")


# if __name__ == "__main__":
#     # Test logging
#     logger = get_logger("reza")
#     #
#     logger.debug("This is a debug message")
#     logger.info("This is an info message")
#     logger.warning("This is a warning message")
#     logger.error("This is an error message")
    # logger.critical("This is a critical message")
    #
    # # Test with different logger
    # api_logger = get_logger("api")
    # api_logger.info("API logger test")