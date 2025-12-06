"""
Core problem definitions and data structures for the Balanced Multi-Bin Packing Problem.

Mathematical Formulation:
-------------------------
Given:
- n items with weights w_i and values v_i for i ∈ {1, ..., n}
- k bins (mules) with individual weight capacities C_j for j ∈ {1, ..., k}
- 
Objective:
    Minimize max{V_j - V_l : j, l ∈ {1, ..., k}}
    where V_j = Σ v_i for all items i assigned to bin j

Subject to:
    - Σ w_i ≤ C_j for all items i assigned to bin j (capacity constraint per bin)
    - Each item assigned to exactly one bin (assignment constraint)
    - All k bins must be used (coverage constraint)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from copy import deepcopy
import json


@dataclass
class Item:
    """
    Represents an item to be packed.
    
    Attributes:
        id: Unique identifier for the item
        weight: Weight of the item (must satisfy capacity constraints)
        value: Value of the item (affects balance objective)
        name: Optional descriptive name
    """
    id: int
    weight: float
    value: float
    name: Optional[str] = None
    
    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"Item weight cannot be negative: {self.weight}")
        if self.value < 0:
            raise ValueError(f"Item value cannot be negative: {self.value}")
        if self.name is None:
            self.name = f"Item_{self.id}"
    
    def __repr__(self) -> str:
        return f"Item(id={self.id}, w={self.weight:.2f}, v={self.value:.2f})"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Item):
            return self.id == other.id
        return False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "weight": self.weight,
            "value": self.value,
            "name": self.name
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Item':
        return cls(**data)


@dataclass
class Bin:
    """
    Represents a bin (mule) with capacity constraint.
    
    Attributes:
        id: Unique identifier for the bin
        capacity: Maximum weight capacity
        items: List of items assigned to this bin
        name: Optional descriptive name
    """
    id: int
    capacity: float
    items: List[Item] = field(default_factory=list)
    name: Optional[str] = None
    
    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError(f"Bin capacity must be positive: {self.capacity}")
        if self.name is None:
            self.name = f"Mule_{self.id}"
    
    @property
    def current_weight(self) -> float:
        """Total weight of items in this bin."""
        return sum(item.weight for item in self.items)
    
    @property
    def current_value(self) -> float:
        """Total value of items in this bin."""
        return sum(item.value for item in self.items)
    
    @property
    def remaining_capacity(self) -> float:
        """Remaining weight capacity."""
        return self.capacity - self.current_weight
    
    @property
    def utilization(self) -> float:
        """Weight utilization percentage (0-1)."""
        return self.current_weight / self.capacity if self.capacity > 0 else 0
    
    @property
    def item_count(self) -> int:
        """Number of items in this bin."""
        return len(self.items)
    
    def can_fit(self, item: Item) -> bool:
        """Check if item can fit in this bin."""
        return item.weight <= self.remaining_capacity
    
    def add_item(self, item: Item) -> bool:
        """
        Attempt to add an item to this bin.
        
        Returns:
            True if item was added, False if it doesn't fit.
        """
        if self.can_fit(item):
            self.items.append(item)
            return True
        return False
    
    def remove_item(self, item: Item) -> bool:
        """
        Remove an item from this bin.
        
        Returns:
            True if item was removed, False if not found.
        """
        if item in self.items:
            self.items.remove(item)
            return True
        return False
    
    def clear(self) -> List[Item]:
        """Remove and return all items from this bin."""
        items = self.items.copy()
        self.items = []
        return items
    
    def __repr__(self) -> str:
        return (f"Bin(id={self.id}, items={len(self.items)}, "
                f"weight={self.current_weight:.2f}/{self.capacity:.2f}, "
                f"value={self.current_value:.2f})")
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def copy(self) -> 'Bin':
        """Create a deep copy of this bin."""
        new_bin = Bin(self.id, self.capacity, name=self.name)
        new_bin.items = [item for item in self.items]
        return new_bin
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "capacity": self.capacity,
            "items": [item.to_dict() for item in self.items],
            "name": self.name
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Bin':
        items = [Item.from_dict(item_data) for item_data in data.get("items", [])]
        return cls(
            id=data["id"],
            capacity=data["capacity"],
            items=items,
            name=data.get("name")
        )


@dataclass
class Solution:
    """
    Represents a complete solution to the bin packing problem.
    
    Attributes:
        bins: List of bins with assigned items
        algorithm_name: Name of the algorithm that produced this solution
        execution_time: Time taken to find this solution (seconds)
        iterations: Number of iterations/steps taken
        metadata: Additional algorithm-specific information
    """
    bins: List[Bin]
    algorithm_name: str = "Unknown"
    execution_time: float = 0.0
    iterations: int = 0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if solution respects all capacity constraints."""
        return all(bin.current_weight <= bin.capacity for bin in self.bins)
    
    @property
    def bin_values(self) -> List[float]:
        """List of total values for each bin."""
        return [bin.current_value for bin in self.bins]
    
    @property
    def bin_weights(self) -> List[float]:
        """List of total weights for each bin."""
        return [bin.current_weight for bin in self.bins]
    
    @property
    def max_value(self) -> float:
        """Maximum value among all bins."""
        return max(self.bin_values) if self.bins else 0
    
    @property
    def min_value(self) -> float:
        """Minimum value among all bins."""
        return min(self.bin_values) if self.bins else 0
    
    @property
    def value_difference(self) -> float:
        """
        Main objective: difference between max and min bin values.
        This is the makespan of the value distribution.
        """
        if not self.bins:
            return 0
        values = self.bin_values
        return max(values) - min(values)
    
    @property
    def balance_factor(self) -> float:
        """
        Balance factor: ratio of min to max value (1.0 = perfectly balanced).
        Returns 1.0 if max_value is 0.
        """
        if self.max_value == 0:
            return 1.0
        return self.min_value / self.max_value
    
    @property
    def total_value(self) -> float:
        """Sum of all item values."""
        return sum(self.bin_values)
    
    @property
    def total_weight(self) -> float:
        """Sum of all item weights."""
        return sum(self.bin_weights)
    
    @property
    def average_utilization(self) -> float:
        """Average weight utilization across all bins."""
        if not self.bins:
            return 0
        return np.mean([bin.utilization for bin in self.bins])
    
    @property
    def value_variance(self) -> float:
        """Variance of bin values."""
        if not self.bins:
            return 0
        return np.var(self.bin_values)
    
    @property
    def value_std(self) -> float:
        """Standard deviation of bin values."""
        return np.sqrt(self.value_variance)
    
    @property
    def coefficient_of_variation(self) -> float:
        """Coefficient of variation of bin values."""
        mean_val = np.mean(self.bin_values)
        if mean_val == 0:
            return 0
        return self.value_std / mean_val
    
    def get_item_assignment(self) -> Dict[int, int]:
        """Get mapping of item_id -> bin_id."""
        assignment = {}
        for bin in self.bins:
            for item in bin.items:
                assignment[item.id] = bin.id
        return assignment
    
    def get_metrics(self) -> Dict:
        """Get comprehensive solution metrics."""
        return {
            "value_difference": self.value_difference,
            "balance_factor": self.balance_factor,
            "total_value": self.total_value,
            "total_weight": self.total_weight,
            "max_value": self.max_value,
            "min_value": self.min_value,
            "value_variance": self.value_variance,
            "value_std": self.value_std,
            "cv": self.coefficient_of_variation,
            "average_utilization": self.average_utilization,
            "is_valid": self.is_valid,
            "num_bins": len(self.bins),
            "num_items": sum(bin.item_count for bin in self.bins),
            "execution_time": self.execution_time,
            "iterations": self.iterations,
            "algorithm": self.algorithm_name
        }
    
    def copy(self) -> 'Solution':
        """Create a deep copy of this solution."""
        return Solution(
            bins=[bin.copy() for bin in self.bins],
            algorithm_name=self.algorithm_name,
            execution_time=self.execution_time,
            iterations=self.iterations,
            metadata=deepcopy(self.metadata)
        )
    
    def __repr__(self) -> str:
        return (f"Solution(algorithm={self.algorithm_name}, "
                f"value_diff={self.value_difference:.2f}, "
                f"balance={self.balance_factor:.3f}, "
                f"valid={self.is_valid})")
    
    def to_dict(self) -> dict:
        return {
            "bins": [bin.to_dict() for bin in self.bins],
            "algorithm_name": self.algorithm_name,
            "execution_time": self.execution_time,
            "iterations": self.iterations,
            "metadata": self.metadata,
            "metrics": self.get_metrics()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Solution':
        bins = [Bin.from_dict(bin_data) for bin_data in data["bins"]]
        return cls(
            bins=bins,
            algorithm_name=data.get("algorithm_name", "Unknown"),
            execution_time=data.get("execution_time", 0.0),
            iterations=data.get("iterations", 0),
            metadata=data.get("metadata", {})
        )
    
    def to_json(self, filepath: str):
        """Save solution to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Solution':
        """Load solution from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class Problem:
    """
    Represents an instance of the Balanced Multi-Bin Packing Problem.
    
    Attributes:
        items: List of items to be packed
        num_bins: Number of bins (mules) available
        bin_capacities: Weight capacity of each bin (list of k capacities)
        name: Optional instance name
        optimal_value: Known optimal value (if available)
    """
    items: List[Item]
    num_bins: int
    bin_capacities: List[float]
    name: str = "Instance"
    optimal_value: Optional[float] = None
    
    def __post_init__(self):
        if self.num_bins <= 0:
            raise ValueError(f"Number of bins must be positive: {self.num_bins}")
        if not self.bin_capacities:
            raise ValueError("Bin capacities list cannot be empty")
        if len(self.bin_capacities) != self.num_bins:
            raise ValueError(
                f"Number of capacities ({len(self.bin_capacities)}) must match "
                f"number of bins ({self.num_bins})"
            )
        for i, cap in enumerate(self.bin_capacities):
            if cap <= 0:
                raise ValueError(f"Bin {i} capacity must be positive: {cap}")
        if not self.items:
            raise ValueError("Items list cannot be empty")
        
        # Validate items fit in at least one bin
        max_capacity = max(self.bin_capacities)
        for item in self.items:
            if item.weight > max_capacity:
                raise ValueError(
                    f"Item {item.id} weight ({item.weight}) exceeds "
                    f"maximum bin capacity ({max_capacity})"
                )
    
    @property
    def n_items(self) -> int:
        """Number of items."""
        return len(self.items)
    
    @property
    def total_weight(self) -> float:
        """Total weight of all items."""
        return sum(item.weight for item in self.items)
    
    @property
    def total_value(self) -> float:
        """Total value of all items."""
        return sum(item.value for item in self.items)
    
    @property
    def total_capacity(self) -> float:
        """Total capacity across all bins."""
        return sum(self.bin_capacities)
    
    @property
    def min_bin_capacity(self) -> float:
        """Minimum bin capacity."""
        return min(self.bin_capacities)
    
    @property
    def max_bin_capacity(self) -> float:
        """Maximum bin capacity."""
        return max(self.bin_capacities)
    
    @property
    def is_feasible(self) -> bool:
        """Check if problem is potentially feasible (total weight fits)."""
        return self.total_weight <= self.total_capacity
    
    @property
    def ideal_value_per_bin(self) -> float:
        """Ideal (perfectly balanced) value per bin."""
        return self.total_value / self.num_bins
    
    @property
    def weight_bounds(self) -> Tuple[float, float]:
        """Min and max item weights."""
        weights = [item.weight for item in self.items]
        return min(weights), max(weights)
    
    @property
    def value_bounds(self) -> Tuple[float, float]:
        """Min and max item values."""
        values = [item.value for item in self.items]
        return min(values), max(values)
    
    def create_empty_bins(self) -> List[Bin]:
        """Create k empty bins with individual capacities."""
        return [Bin(i, self.bin_capacities[i]) for i in range(self.num_bins)]
    
    def create_empty_solution(self, algorithm_name: str = "Unknown") -> Solution:
        """Create an empty solution with bins ready for assignment."""
        return Solution(
            bins=self.create_empty_bins(),
            algorithm_name=algorithm_name
        )
    
    def validate_solution(self, solution: Solution) -> Tuple[bool, List[str]]:
        """
        Validate a solution against problem constraints.
        
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check number of bins
        if len(solution.bins) != self.num_bins:
            violations.append(
                f"Wrong number of bins: {len(solution.bins)} vs {self.num_bins}"
            )
        
        # Check capacity constraints
        for bin in solution.bins:
            if bin.current_weight > bin.capacity + 1e-9:  # Small tolerance
                violations.append(
                    f"Bin {bin.id} exceeds capacity: "
                    f"{bin.current_weight:.2f} > {bin.capacity:.2f}"
                )
        
        # Check all items assigned exactly once
        assigned_items = set()
        for bin in solution.bins:
            for item in bin.items:
                if item.id in assigned_items:
                    violations.append(f"Item {item.id} assigned multiple times")
                assigned_items.add(item.id)
        
        expected_items = set(item.id for item in self.items)
        missing = expected_items - assigned_items
        extra = assigned_items - expected_items
        
        if missing:
            violations.append(f"Missing items: {missing}")
        if extra:
            violations.append(f"Extra items: {extra}")
        
        return len(violations) == 0, violations
    
    def get_statistics(self) -> Dict:
        """Get problem statistics."""
        weights = [item.weight for item in self.items]
        values = [item.value for item in self.items]
        
        return {
            "name": self.name,
            "n_items": self.n_items,
            "num_bins": self.num_bins,
            "bin_capacities": self.bin_capacities,
            "total_weight": self.total_weight,
            "total_value": self.total_value,
            "total_capacity": self.total_capacity,
            "is_feasible": self.is_feasible,
            "ideal_value_per_bin": self.ideal_value_per_bin,
            "weight_mean": np.mean(weights),
            "weight_std": np.std(weights),
            "weight_min": min(weights),
            "weight_max": max(weights),
            "value_mean": np.mean(values),
            "value_std": np.std(values),
            "value_min": min(values),
            "value_max": max(values),
            "capacity_utilization": self.total_weight / self.total_capacity
        }
    
    def __repr__(self) -> str:
        return (f"Problem(name={self.name}, items={self.n_items}, "
                f"bins={self.num_bins}, capacities={self.bin_capacities})")
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "items": [item.to_dict() for item in self.items],
            "num_bins": self.num_bins,
            "bin_capacities": self.bin_capacities,
            "optimal_value": self.optimal_value,
            "statistics": self.get_statistics()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Problem':
        items = [Item.from_dict(item_data) for item_data in data["items"]]
        # Support legacy format with single bin_capacity
        if "bin_capacities" in data:
            bin_capacities = data["bin_capacities"]
        else:
            # Legacy support: single capacity for all bins
            bin_capacities = [data["bin_capacity"]] * data["num_bins"]
        return cls(
            items=items,
            num_bins=data["num_bins"],
            bin_capacities=bin_capacities,
            name=data.get("name", "Instance"),
            optimal_value=data.get("optimal_value")
        )
    
    def to_json(self, filepath: str):
        """Save problem instance to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Problem':
        """Load problem instance from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_lower_bound(self) -> float:
        """
        Calculate a lower bound on the optimal value difference.
        
        For perfectly balanced distribution:
        LB = 0 if total_value is divisible by num_bins
        Otherwise, LB depends on the value distribution.
        """
        # Simple lower bound: if perfect balance is possible
        ideal = self.total_value / self.num_bins
        
        # Lower bound based on value distribution
        # At minimum, the difference is 0 (perfect balance)
        # But item granularity may prevent this
        max_item_value = max(item.value for item in self.items)
        
        # A tighter bound considers that moving one item changes difference
        # by at least the minimum item value
        min_item_value = min(item.value for item in self.items)
        
        return 0  # Trivial lower bound, can be tightened with problem-specific analysis
    
    def get_upper_bound(self) -> float:
        """
        Calculate an upper bound on the optimal value difference.
        
        UB: Assign all items to one bin (if feasible), difference = total_value
        """
        return self.total_value
