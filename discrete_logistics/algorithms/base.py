"""
Base classes for algorithm implementations.

Provides abstract base class and registry for all algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Callable, Any
from dataclasses import dataclass, field
import time

try:
    from ..core.problem import Problem, Solution, Bin, Item
except ImportError:
    from discrete_logistics.core.problem import Problem, Solution, Bin, Item


@dataclass
class AlgorithmStep:
    """
    Represents a single step in algorithm execution for visualization.
    
    Attributes:
        step_number: Sequential step number
        action: Description of the action taken
        item: Item being processed (if any)
        bin_id: Target bin (if any)
        bins_state: Copy of all bins' state at this step
        metrics: Current solution metrics
        extra_data: Algorithm-specific data for visualization
    """
    step_number: int
    action: str
    item: Optional[Item] = None
    bin_id: Optional[int] = None
    bins_state: Optional[List[Dict]] = None
    metrics: Optional[Dict] = None
    extra_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "step_number": self.step_number,
            "action": self.action,
            "item": self.item.to_dict() if self.item else None,
            "bin_id": self.bin_id,
            "bins_state": self.bins_state,
            "metrics": self.metrics,
            "extra_data": self.extra_data
        }


class Algorithm(ABC):
    """
    Abstract base class for all algorithms.
    
    Provides common interface and functionality for solving the
    Balanced Multi-Bin Packing Problem.
    
    Complexity Analysis (to be overridden by subclasses):
    - Time Complexity: O(?)
    - Space Complexity: O(?)
    - Approximation Ratio: ?
    """
    
    # Class attributes for complexity documentation
    time_complexity: str = "Unknown"
    space_complexity: str = "Unknown"
    approximation_ratio: str = "Unknown"
    description: str = "Base algorithm class"
    
    def __init__(self, track_steps: bool = False, verbose: bool = False):
        """
        Initialize algorithm.
        
        Args:
            track_steps: Whether to record execution steps for visualization
            verbose: Whether to print progress information
        """
        self.track_steps = track_steps
        self.verbose = verbose
        self.steps: List[AlgorithmStep] = []
        self.step_counter = 0
        self._start_time = 0
        self._iterations = 0
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return algorithm name."""
        pass
    
    @abstractmethod
    def solve(self, problem: Problem) -> Solution:
        """
        Solve the problem and return a solution.
        
        Args:
            problem: Problem instance to solve
            
        Returns:
            Solution object with bin assignments
        """
        pass
    
    def _start_timer(self):
        """Start execution timer."""
        self._start_time = time.perf_counter()
        self.steps = []
        self.step_counter = 0
        self._iterations = 0
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.perf_counter() - self._start_time
    
    def _record_step(
        self,
        action: str,
        bins: List[Bin],
        item: Optional[Item] = None,
        bin_id: Optional[int] = None,
        extra_data: Optional[Dict] = None
    ):
        """Record an algorithm step for visualization."""
        if not self.track_steps:
            return
        
        self.step_counter += 1
        
        # Capture bins state
        bins_state = []
        for b in bins:
            bins_state.append({
                "id": b.id,
                "items": [i.id for i in b.items],
                "current_weight": b.current_weight,
                "current_value": b.current_value,
                "remaining_capacity": b.remaining_capacity
            })
        
        # Calculate current metrics
        values = [b.current_value for b in bins]
        metrics = {
            "value_difference": max(values) - min(values) if values else 0,
            "max_value": max(values) if values else 0,
            "min_value": min(values) if values else 0,
            "total_items": sum(len(b.items) for b in bins)
        }
        
        step = AlgorithmStep(
            step_number=self.step_counter,
            action=action,
            item=item,
            bin_id=bin_id,
            bins_state=bins_state,
            metrics=metrics,
            extra_data=extra_data or {}
        )
        
        self.steps.append(step)
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.name}] {message}")
    
    def get_complexity_info(self) -> Dict:
        """Return complexity analysis information."""
        return {
            "name": self.name,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "approximation_ratio": self.approximation_ratio,
            "description": self.description
        }
    
    def get_steps(self) -> List[AlgorithmStep]:
        """Get recorded execution steps."""
        return self.steps
    
    def get_steps_as_dicts(self) -> List[Dict]:
        """Get recorded execution steps as dictionaries."""
        return [step.to_dict() for step in self.steps]


class AlgorithmRegistry:
    """
    Registry for algorithm implementations.
    
    Allows dynamic registration and retrieval of algorithms.
    """
    
    _algorithms: Dict[str, Type[Algorithm]] = {}
    _instances: Dict[str, Algorithm] = {}
    
    @classmethod
    def register(cls, name: str, algorithm_class: Type[Algorithm]):
        """Register an algorithm class."""
        cls._algorithms[name] = algorithm_class
    
    @classmethod
    def get(cls, name: str, **kwargs) -> Algorithm:
        """Get an algorithm instance by name."""
        if name not in cls._algorithms:
            raise ValueError(f"Unknown algorithm: {name}. Available: {list(cls._algorithms.keys())}")
        return cls._algorithms[name](**kwargs)
    
    @classmethod
    def get_all(cls, **kwargs) -> Dict[str, Algorithm]:
        """Get instances of all registered algorithms."""
        return {name: algo(**kwargs) for name, algo in cls._algorithms.items()}
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """List all registered algorithm names."""
        return list(cls._algorithms.keys())
    
    @classmethod
    def get_info(cls, name: str) -> Dict:
        """Get complexity information for an algorithm."""
        if name not in cls._algorithms:
            raise ValueError(f"Unknown algorithm: {name}")
        algo = cls._algorithms[name]()
        return algo.get_complexity_info()
    
    @classmethod
    def get_all_info(cls) -> List[Dict]:
        """Get complexity information for all algorithms."""
        return [cls.get_info(name) for name in cls._algorithms.keys()]


def register_algorithm(name: str):
    """
    Decorator to register an algorithm class.
    
    Usage:
        @register_algorithm("my_algorithm")
        class MyAlgorithm(Algorithm):
            ...
    """
    def decorator(cls: Type[Algorithm]) -> Type[Algorithm]:
        AlgorithmRegistry.register(name, cls)
        return cls
    return decorator
