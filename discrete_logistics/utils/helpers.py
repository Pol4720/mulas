"""
Helper Utilities Module
=======================

General helper functions and utilities.
"""

import time
import logging
from typing import Optional, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime


class Timer:
    """
    Context manager and decorator for timing code execution.
    
    Can be used as:
    1. Context manager: with Timer() as t: ...
    2. Decorator: @Timer.measure
    3. Manual: t = Timer(); t.start(); ... ; t.stop()
    
    Examples
    --------
    >>> with Timer() as t:
    ...     # code to time
    >>> print(f"Elapsed: {t.elapsed:.3f}s")
    
    >>> @Timer.measure
    ... def my_function():
    ...     pass
    """
    
    def __init__(self, name: str = "Timer", logger: Optional[logging.Logger] = None):
        """
        Initialize timer.
        
        Parameters
        ----------
        name : str
            Name for logging purposes
        logger : logging.Logger, optional
            Logger instance for output
        """
        self.name = name
        self.logger = logger
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._elapsed: float = 0.0
    
    def start(self):
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._end_time = None
    
    def stop(self) -> float:
        """
        Stop the timer.
        
        Returns
        -------
        float
            Elapsed time in seconds
        """
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
        
        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time
        return self._elapsed
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        
        if self._end_time is None:
            # Timer still running
            return time.perf_counter() - self._start_time
        
        return self._elapsed
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000
    
    def __enter__(self) -> 'Timer':
        """Enter context manager."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Exit context manager."""
        self.stop()
        if self.logger:
            self.logger.info(f"{self.name}: {self.elapsed:.4f}s")
    
    @staticmethod
    def measure(func: Callable) -> Callable:
        """
        Decorator to measure function execution time.
        
        Parameters
        ----------
        func : Callable
            Function to decorate
            
        Returns
        -------
        Callable
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            timer = Timer(name=func.__name__)
            timer.start()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                timer.stop()
                print(f"{func.__name__}: {timer.elapsed:.4f}s")
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


@dataclass
class ProgressTracker:
    """
    Tracks progress of long-running operations.
    
    Provides:
    - Progress percentage
    - Estimated time remaining
    - Progress callbacks
    
    Examples
    --------
    >>> tracker = ProgressTracker(total=100)
    >>> for i in range(100):
    ...     # do work
    ...     tracker.update(i + 1)
    ...     print(tracker.get_progress_string())
    """
    
    total: int
    current: int = 0
    description: str = "Progress"
    start_time: float = field(default_factory=time.perf_counter)
    _callback: Optional[Callable[[int, int, str], None]] = field(default=None, repr=False)
    
    def update(self, current: Optional[int] = None, description: Optional[str] = None):
        """
        Update progress.
        
        Parameters
        ----------
        current : int, optional
            Current progress (increments by 1 if not provided)
        description : str, optional
            Update description
        """
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        if description:
            self.description = description
        
        if self._callback:
            self._callback(self.current, self.total, self.description)
    
    def set_callback(self, callback: Callable[[int, int, str], None]):
        """Set progress callback function."""
        self._callback = callback
    
    @property
    def percentage(self) -> float:
        """Get progress percentage."""
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.perf_counter() - self.start_time
    
    @property
    def estimated_remaining(self) -> float:
        """Get estimated remaining time in seconds."""
        if self.current == 0:
            return float('inf')
        
        rate = self.elapsed_time / self.current
        remaining = self.total - self.current
        return rate * remaining
    
    @property
    def estimated_total(self) -> float:
        """Get estimated total time in seconds."""
        if self.current == 0:
            return float('inf')
        
        rate = self.elapsed_time / self.current
        return rate * self.total
    
    def get_progress_string(self) -> str:
        """Get formatted progress string."""
        elapsed = self.elapsed_time
        remaining = self.estimated_remaining
        
        if remaining == float('inf'):
            eta = "?"
        else:
            eta = f"{remaining:.1f}s"
        
        return (f"{self.description}: {self.current}/{self.total} "
                f"({self.percentage:.1f}%) - Elapsed: {elapsed:.1f}s - ETA: {eta}")
    
    def reset(self, total: Optional[int] = None):
        """Reset tracker."""
        if total is not None:
            self.total = total
        self.current = 0
        self.start_time = time.perf_counter()
    
    def is_complete(self) -> bool:
        """Check if progress is complete."""
        return self.current >= self.total


def setup_logging(
    name: str = "discrete_logistics",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    name : str
        Logger name
    level : int
        Logging level (e.g., logging.INFO)
    log_file : str, optional
        File to write logs to
    format_string : str, optional
        Custom format string
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


@contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr.
    
    Useful for running algorithms that print progress.
    
    Examples
    --------
    >>> with suppress_output():
    ...     noisy_function()
    """
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    
    Parameters
    ----------
    seconds : float
        Time in seconds
        
    Returns
    -------
    str
        Formatted time string
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_number(n: float, precision: int = 2) -> str:
    """
    Format number with appropriate suffix.
    
    Parameters
    ----------
    n : float
        Number to format
    precision : int
        Decimal precision
        
    Returns
    -------
    str
        Formatted number string
    """
    if abs(n) < 1000:
        return f"{n:.{precision}f}"
    elif abs(n) < 1000000:
        return f"{n/1000:.{precision}f}K"
    elif abs(n) < 1000000000:
        return f"{n/1000000:.{precision}f}M"
    else:
        return f"{n/1000000000:.{precision}f}B"


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns
    -------
    float
        Memory usage in megabytes
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


class Memoize:
    """
    Memoization decorator with optional cache size limit.
    
    Examples
    --------
    >>> @Memoize(maxsize=100)
    ... def expensive_function(x):
    ...     return x ** 2
    """
    
    def __init__(self, maxsize: Optional[int] = None):
        """
        Initialize memoization decorator.
        
        Parameters
        ----------
        maxsize : int, optional
            Maximum cache size (None for unlimited)
        """
        self.maxsize = maxsize
        self.cache = {}
        self.order = []
    
    def __call__(self, func: Callable) -> Callable:
        """Apply decorator to function."""
        def wrapper(*args, **kwargs):
            # Create hashable key
            key = (args, tuple(sorted(kwargs.items())))
            
            if key in self.cache:
                return self.cache[key]
            
            result = func(*args, **kwargs)
            
            # Check cache size
            if self.maxsize and len(self.cache) >= self.maxsize:
                # Remove oldest entry
                oldest = self.order.pop(0)
                del self.cache[oldest]
            
            self.cache[key] = result
            self.order.append(key)
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.cache = self.cache
        wrapper.clear_cache = lambda: self.cache.clear()
        
        return wrapper
