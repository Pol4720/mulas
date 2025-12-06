"""
Dynamic Programming Algorithm for Balanced Multi-Bin Packing.

For small instances, DP can find optimal solutions by exploring
the state space systematically.

Note: Due to the problem's complexity, DP is only practical for
very small instances (n ≤ 20, k ≤ 5).
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
try:
    from ..core.problem import Problem, Solution, Bin, Item
    from .base import Algorithm, register_algorithm
except ImportError:
    from discrete_logistics.core.problem import Problem, Solution, Bin, Item
    from discrete_logistics.algorithms.base import Algorithm, register_algorithm


@register_algorithm("dynamic_programming")
class DynamicProgramming(Algorithm):
    """
    Dynamic Programming Algorithm for Balanced Multi-Bin Packing.
    
    Uses subset DP to enumerate possible bin assignments and find
    the optimal balance. Due to exponential state space, only
    suitable for small instances.
    
    Complexity Analysis:
    - Time: O(k * 3^n) in worst case (subset enumeration)
    - Space: O(k * 2^n) for memoization
    - Approximation: Optimal (exact algorithm)
    
    The algorithm works by:
    1. Generate all possible subsets of items for each bin
    2. Filter subsets that satisfy weight constraint
    3. Use DP to find k-partition minimizing value difference
    
    State: dp[mask] = list of (value_tuple) where mask represents
    assigned items and value_tuple is values for bins 0..k-1
    """
    
    time_complexity = "O(k · 3^n)"
    space_complexity = "O(k · 2^n)"
    approximation_ratio = "Optimal (exact)"
    description = "Exact algorithm using dynamic programming, feasible only for small instances"
    
    MAX_ITEMS = 20  # Safety limit
    MAX_BINS = 8
    
    def __init__(self, track_steps: bool = False, verbose: bool = False,
                 max_items: int = 20, max_bins: int = 8):
        super().__init__(track_steps, verbose)
        self.max_items = min(max_items, self.MAX_ITEMS)
        self.max_bins = min(max_bins, self.MAX_BINS)
    
    @property
    def name(self) -> str:
        return "Dynamic Programming"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting DP on {problem.n_items} items, {problem.num_bins} bins")
        
        # Check size limits
        if problem.n_items > self.max_items:
            self._log(f"Instance too large for DP (n={problem.n_items} > {self.max_items})")
            return self._fallback_greedy(problem)
        
        if problem.num_bins > self.max_bins:
            self._log(f"Too many bins for DP (k={problem.num_bins} > {self.max_bins})")
            return self._fallback_greedy(problem)
        
        n = problem.n_items
        k = problem.num_bins
        items = problem.items
        capacity = problem.bin_capacity
        
        # Precompute subset properties
        self._log("Precomputing feasible subsets...")
        feasible_subsets = self._compute_feasible_subsets(items, capacity)
        self._log(f"Found {len(feasible_subsets)} feasible subsets")
        
        self._record_step(
            f"Computed {len(feasible_subsets)} feasible subsets",
            problem.create_empty_bins(),
            extra_data={"n_feasible": len(feasible_subsets)}
        )
        
        # DP: Find best k-partition
        best_assignment = self._find_best_partition(
            items, k, feasible_subsets
        )
        
        if best_assignment is None:
            self._log("No feasible partition found!")
            return self._fallback_greedy(problem)
        
        # Build solution from assignment
        solution = problem.create_empty_solution(self.name)
        bins = solution.bins
        
        for bin_id, item_ids in enumerate(best_assignment):
            for item_id in item_ids:
                item = next(i for i in items if i.id == item_id)
                bins[bin_id].add_item(item)
        
        self._record_step(
            f"Found optimal partition with diff={solution.value_difference:.2f}",
            bins,
            extra_data={"assignment": best_assignment}
        )
        
        solution.execution_time = self._get_elapsed_time()
        solution.iterations = self._iterations
        solution.metadata["exact"] = True
        
        self._log(f"Completed in {solution.execution_time:.4f}s")
        
        return solution
    
    def _compute_feasible_subsets(
        self,
        items: List[Item],
        capacity: float
    ) -> Dict[int, Tuple[float, float]]:
        """
        Compute all subsets of items that fit within capacity.
        
        Returns:
            Dict mapping bitmask -> (total_weight, total_value)
        """
        n = len(items)
        feasible = {}
        
        for mask in range(1 << n):
            self._iterations += 1
            total_weight = 0
            total_value = 0
            
            for i in range(n):
                if mask & (1 << i):
                    total_weight += items[i].weight
                    total_value += items[i].value
            
            if total_weight <= capacity:
                feasible[mask] = (total_weight, total_value)
        
        # Include empty set
        feasible[0] = (0, 0)
        
        return feasible
    
    def _find_best_partition(
        self,
        items: List[Item],
        k: int,
        feasible: Dict[int, Tuple[float, float]]
    ) -> Optional[List[List[int]]]:
        """
        Find the best k-partition using DP.
        
        Returns:
            List of k lists, each containing item IDs for that bin
            Returns None if no feasible partition exists
        """
        n = len(items)
        full_mask = (1 << n) - 1
        
        # dp[j][mask] = (best_max, best_min, assignment)
        # j = number of bins used
        # mask = items assigned so far
        # assignment = list of sets for each bin
        
        INF = float('inf')
        
        # Initialize: 1 bin
        dp = [{} for _ in range(k + 1)]
        
        for mask, (weight, value) in feasible.items():
            dp[1][mask] = (value, value, [[self._mask_to_items(mask, items)]])
        
        # DP transition: add bins one by one
        for j in range(2, k + 1):
            self._log(f"DP: Processing {j} bins...")
            
            for prev_mask in dp[j - 1]:
                prev_max, prev_min, prev_assign = dp[j - 1][prev_mask]
                remaining = full_mask ^ prev_mask  # Items not yet assigned
                
                # Try all feasible subsets of remaining items
                subset = remaining
                while subset > 0:
                    self._iterations += 1
                    
                    if subset in feasible:
                        _, new_value = feasible[subset]
                        new_mask = prev_mask | subset
                        
                        new_max = max(prev_max, new_value)
                        new_min = min(prev_min, new_value)
                        new_diff = new_max - new_min
                        
                        # Check if this is better
                        if new_mask not in dp[j]:
                            new_assign = prev_assign + [self._mask_to_items(subset, items)]
                            dp[j][new_mask] = (new_max, new_min, new_assign)
                        else:
                            old_max, old_min, _ = dp[j][new_mask]
                            old_diff = old_max - old_min
                            
                            if new_diff < old_diff:
                                new_assign = prev_assign + [self._mask_to_items(subset, items)]
                                dp[j][new_mask] = (new_max, new_min, new_assign)
                    
                    # Next subset of remaining
                    subset = (subset - 1) & remaining
        
        # Find best complete partition
        if full_mask in dp[k]:
            _, _, assignment = dp[k][full_mask]
            return assignment
        
        # If exact k bins not possible, try with some empty bins
        best_diff = INF
        best_assign = None
        
        for mask in dp[k]:
            if mask == full_mask:
                max_v, min_v, assign = dp[k][mask]
                diff = max_v - min_v
                if diff < best_diff:
                    best_diff = diff
                    best_assign = assign
        
        return best_assign
    
    def _mask_to_items(self, mask: int, items: List[Item]) -> List[int]:
        """Convert bitmask to list of item IDs."""
        result = []
        for i, item in enumerate(items):
            if mask & (1 << i):
                result.append(item.id)
        return result
    
    def _fallback_greedy(self, problem: Problem) -> Solution:
        """Fall back to greedy when DP is infeasible."""
        from .greedy import BestFitDecreasing
        
        self._log("Falling back to Best Fit Decreasing")
        greedy = BestFitDecreasing(track_steps=self.track_steps, verbose=self.verbose)
        solution = greedy.solve(problem)
        solution.algorithm_name = f"{self.name} (fallback to {greedy.name})"
        solution.metadata["fallback"] = True
        
        return solution


class DPState:
    """
    Helper class for DP state management.
    """
    
    def __init__(self, k: int):
        self.k = k
        self.values = [0.0] * k
        self.assignment = [[] for _ in range(k)]
    
    @property
    def max_value(self) -> float:
        return max(self.values)
    
    @property
    def min_value(self) -> float:
        return min(self.values)
    
    @property
    def difference(self) -> float:
        return self.max_value - self.min_value
    
    def copy(self) -> 'DPState':
        new_state = DPState(self.k)
        new_state.values = self.values.copy()
        new_state.assignment = [lst.copy() for lst in self.assignment]
        return new_state
    
    def assign(self, item: Item, bin_id: int):
        self.values[bin_id] += item.value
        self.assignment[bin_id].append(item.id)
    
    def __hash__(self) -> int:
        return hash(tuple(sorted(self.values)))
    
    def dominates(self, other: 'DPState') -> bool:
        """Check if this state dominates another."""
        return self.difference <= other.difference
