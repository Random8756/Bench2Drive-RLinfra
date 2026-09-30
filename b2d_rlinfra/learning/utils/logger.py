"""
Logging Utilities.

TensorBoard logging and console output for RL training.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import time
from collections import defaultdict

import numpy as np

logger = logging.getLogger("Policy")


class Logger:
    """
    Logger for RL training with TensorBoard support.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        output_formats: Optional[List[str]] = None,
        verbose: int = 1,
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files.
            output_formats: List of output formats ['stdout', 'tensorboard'].
            verbose: Verbosity level (0: silent, 1: info, 2: debug).
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        supported_formats = {'stdout', 'tensorboard'}
        self.output_formats = [
            fmt for fmt in (output_formats or ['stdout', 'tensorboard'])
            if fmt in supported_formats
        ]
        self.verbose = verbose
        
        # Name-value pairs to log
        self.name_to_value: Dict[str, float] = {}
        self.name_to_count: Dict[str, int] = defaultdict(int)
        self.name_to_excluded: Dict[str, str] = {}
        
        # Timestep tracking
        self.num_timesteps = 0
        
        # TensorBoard writer
        self._tb_writer = None
        if 'tensorboard' in self.output_formats:
            self._setup_tensorboard()
        
        # Timing
        self._start_time = time.time()
    
    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.output_formats.remove('tensorboard')
    
    def record(self, key: str, value: Any, exclude: Optional[str] = None) -> None:
        """
        Record a value for logging.
        
        Args:
            key: Key name for the value.
            value: Value to log.
            exclude: Optional format to exclude from logging.
        """
        self.name_to_value[key] = value
        if exclude is not None:
            self.name_to_excluded[key] = exclude
    
    def record_mean(self, key: str, value: float) -> None:
        """
        Record a value that will be averaged over dump calls.
        
        Args:
            key: Key name for the value.
            value: Value to add to the running mean.
        """
        if key in self.name_to_value:
            count = self.name_to_count[key]
            self.name_to_value[key] = (self.name_to_value[key] * count + value) / (count + 1)
        else:
            self.name_to_value[key] = value
        self.name_to_count[key] += 1
    
    def dump(self, step: Optional[int] = None) -> None:
        """
        Write all recorded values to output formats.
        
        Args:
            step: Global step for logging (default: num_timesteps).
        """
        if step is None:
            step = self.num_timesteps
        
        # Log to stdout
        if 'stdout' in self.output_formats and self.verbose >= 2:
            self._dump_stdout()
        
        # Log to TensorBoard
        if 'tensorboard' in self.output_formats and self._tb_writer is not None:
            self._dump_tensorboard(step)
        
        # Clear recorded values
        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()
    
    def _dump_stdout(self) -> None:
        """Write to stdout."""
        import sys
        if not self.name_to_value:
            return  # Nothing to log
        
        lines = ["-" * 55]
        for key in sorted(self.name_to_value.keys()):
            if self.name_to_excluded.get(key) == 'stdout':
                continue
            value = self.name_to_value[key]
            if isinstance(value, float):
                lines.append(f"| {key:35s} | {value:12.5g} |")
            else:
                lines.append(f"| {key:35s} | {str(value):>12s} |")
        lines.append("-" * 55)
        
        # Print all at once to avoid interleaving
        output = "\n".join(lines)
        print(output, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
    
    def _dump_tensorboard(self, step: int) -> None:
        """Write to TensorBoard."""
        for key, value in self.name_to_value.items():
            if self.name_to_excluded.get(key) == 'tensorboard':
                continue
            if isinstance(value, (int, float, np.number)):
                self._tb_writer.add_scalar(key, value, step)
    
    def set_timesteps(self, num_timesteps: int) -> None:
        """Set the current timestep for logging."""
        self.num_timesteps = num_timesteps
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since logger creation."""
        return time.time() - self._start_time
    
    def close(self) -> None:
        """Close all output formats."""
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def configure_logger(
    log_dir: Union[str, Path],
    output_formats: Optional[List[str]] = None,
    verbose: int = 1,
) -> Logger:
    """
    Configure and return a logger instance.
    
    Args:
        log_dir: Directory for log files.
        output_formats: List of output formats.
        verbose: Verbosity level.
        
    Returns:
        Configured Logger instance.
    """
    return Logger(log_dir, output_formats, verbose)
