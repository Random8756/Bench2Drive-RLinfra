"""
Simple logger module for the environment.
Provides a unified logging interface across all modules.
"""

import logging
import sys


class Logger:
    """
    A simple logger wrapper that provides info, warning, error, and debug methods.
    """
    
    def __init__(self, name="EnvLogger", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        root_logger = logging.getLogger()

        # Avoid adding multiple handlers if logger already has handlers.
        # If the process already configured root logging, let this named logger
        # propagate to root instead of attaching its own stdout handler.
        if not self.logger.handlers and not root_logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)

            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.propagate = False
        elif self.logger.handlers:
            # Keep dedicated handlers single-sourced even if root logging is
            # configured later in the same process.
            self.logger.propagate = False
    
    def info(self, msg):
        """Log info message"""
        self.logger.info(msg)
    
    def warning(self, msg):
        """Log warning message"""
        self.logger.warning(msg)
    
    def error(self, msg):
        """Log error message"""
        self.logger.error(msg)
    
    def debug(self, msg):
        """Log debug message"""
        self.logger.debug(msg)
    
    def critical(self, msg):
        """Log critical message"""
        self.logger.critical(msg)
    
    def set_level(self, level):
        """Set logging level"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)


# Create a global logger instance
log = Logger(name="Carla")
