"""
Learning Rate and Parameter Schedules.
"""

from typing import Callable, Union


Schedule = Callable[[float], float]


def constant_schedule(value: float) -> Schedule:
    """
    Constant schedule returning the same value regardless of progress.
    
    Args:
        value: Constant value to return.
        
    Returns:
        Schedule function.
    """
    def func(progress_remaining: float) -> float:
        return value
    return func


def linear_schedule(initial_value: float, final_value: float = 0.0) -> Schedule:
    """
    Linear schedule from initial_value to final_value.
    
    Args:
        initial_value: Starting value.
        final_value: Ending value (default: 0).
        
    Returns:
        Schedule function that takes progress_remaining (1.0 -> 0.0) and returns value.
    """
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


def get_schedule_fn(value: Union[float, Schedule]) -> Schedule:
    """
    Get a schedule function from a value or existing schedule.
    
    Args:
        value: Either a float (converted to constant schedule) or a Schedule function.
        
    Returns:
        Schedule function.
    """
    if callable(value):
        return value
    return constant_schedule(value)


def polynomial_schedule(
    initial_value: float,
    final_value: float = 0.0,
    power: float = 1.0,
) -> Schedule:
    """
    Polynomial schedule from initial_value to final_value.
    
    Args:
        initial_value: Starting value.
        final_value: Ending value.
        power: Polynomial power (1.0 = linear).
        
    Returns:
        Schedule function.
    """
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * (progress_remaining ** power)
    return func


def exponential_schedule(
    initial_value: float,
    final_value: float = 0.0,
    decay_rate: float = 0.99,
) -> Schedule:
    """
    Exponential decay schedule.
    
    Args:
        initial_value: Starting value.
        final_value: Minimum value.
        decay_rate: Decay rate per step.
        
    Returns:
        Schedule function.
    """
    def func(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        value = initial_value * (decay_rate ** progress)
        return max(final_value, value)
    return func
