"""
Training Callbacks.

Callback hooks for injecting custom logic during training.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from pathlib import Path
import logging

import numpy as np

if TYPE_CHECKING:
    from ..algorithms.base_algorithm import BaseAlgorithm

logger = logging.getLogger("Policy")


class BaseCallback(ABC):
    """
    Base class for callback.
    
    Callbacks are called at specific points during training to inject custom logic.
    """
    
    def __init__(self, verbose: int = 0):
        """
        Initialize callback.
        
        Args:
            verbose: Verbosity level (0: no output, 1: info, 2: debug).
        """
        self.verbose = verbose
        self.model: Optional['BaseAlgorithm'] = None
        self.n_calls = 0
        self.num_timesteps = 0
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}
        self.logger = None
    
    def init_callback(self, model: 'BaseAlgorithm') -> None:
        """
        Initialize callback with model reference.
        
        Args:
            model: The RL algorithm instance.
        """
        self.model = model
        self.logger = model.logger
        self._init_callback()
    
    def _init_callback(self) -> None:
        """Override to perform initialization."""
        pass
    
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Called before training starts.
        
        Args:
            locals_: Local variables from the training loop.
            globals_: Global variables.
        """
        self.locals = locals_
        self.globals = globals_
        self._on_training_start()
    
    def _on_training_start(self) -> None:
        """Override to add custom logic at training start."""
        pass
    
    def on_rollout_start(self) -> None:
        """Called at the start of a rollout collection."""
        self._on_rollout_start()
    
    def _on_rollout_start(self) -> None:
        """Override to add custom logic at rollout start."""
        pass
    
    def on_step(self) -> bool:
        """
        Called after each step in the environment.
        
        Returns:
            True to continue training, False to stop.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps
        return self._on_step()
    
    @abstractmethod
    def _on_step(self) -> bool:
        """
        Override to add custom logic after each step.
        
        Returns:
            True to continue training, False to stop.
        """
        return True
    
    def on_rollout_end(self) -> None:
        """Called at the end of a rollout collection."""
        self._on_rollout_end()
    
    def _on_rollout_end(self) -> None:
        """Override to add custom logic at rollout end."""
        pass

    def on_update_end(self) -> None:
        """Called after a learner update finishes."""
        if self.model is not None:
            self.num_timesteps = self.model.num_timesteps
        self._on_update_end()

    def _on_update_end(self) -> None:
        """Override to add custom logic after a learner update."""
        pass
    
    def on_training_end(self) -> None:
        """Called when training ends."""
        self._on_training_end()
    
    def _on_training_end(self) -> None:
        """Override to add custom logic at training end."""
        pass
    
    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update local variables for callback.
        
        Args:
            locals_: Local variables from the training loop.
        """
        self.locals = locals_
        if self.model is not None:
            self.num_timesteps = self.model.num_timesteps


class CallbackList(BaseCallback):
    """
    Callback that wraps a list of callbacks.
    """
    
    def __init__(self, callbacks: List[BaseCallback]):
        super().__init__()
        self.callbacks = callbacks
    
    def init_callback(self, model: 'BaseAlgorithm') -> None:
        super().init_callback(model)
        for callback in self.callbacks:
            callback.init_callback(model)
    
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_training_start(locals_, globals_)
    
    def on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()
    
    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            continue_training = callback.on_step() and continue_training
        return continue_training
    
    def on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def on_update_end(self) -> None:
        for callback in self.callbacks:
            callback.on_update_end()
    
    def on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()
    
    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """Update locals for all callbacks."""
        self.locals = locals_
        for callback in self.callbacks:
            callback.update_locals(locals_)


class CheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints during training.

    NOTE: save_freq is in TIMESTEPS, not callback calls.
    The callback tracks num_timesteps and saves when threshold is crossed.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: Union[str, Path],
        name_prefix: str = 'rl_model',
        verbose: int = 0,
        save_on: str = 'step',
    ):
        """
        Initialize checkpoint callback.

        Args:
            save_freq: Save frequency (in timesteps, not callback calls!).
            save_path: Directory to save checkpoints.
            name_prefix: Prefix for checkpoint files.
            verbose: Verbosity level.
            save_on: Hook used for periodic saves: 'step' or 'update_end'.
        """
        super().__init__(verbose)
        if save_on not in ('step', 'update_end'):
            raise ValueError("save_on must be either 'step' or 'update_end'")
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_on = save_on
        self._last_save_timesteps = 0

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self._last_save_timesteps = self.model.num_timesteps

    def _maybe_save_checkpoint(self) -> None:
        if self.model is None:
            return

        self.num_timesteps = self.model.num_timesteps
        if (self.num_timesteps - self._last_save_timesteps) >= self.save_freq:
            self._last_save_timesteps = self.num_timesteps

            path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(str(path))
            if self.verbose >= 2:
                logger.debug("Checkpoint callback saved path=%s steps=%d", path, self.num_timesteps)

    def _on_step(self) -> bool:
        if self.save_on == 'step':
            self._maybe_save_checkpoint()

        return True

    def _on_update_end(self) -> None:
        if self.save_on == 'update_end':
            self._maybe_save_checkpoint()
