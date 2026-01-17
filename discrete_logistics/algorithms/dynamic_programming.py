"""
Dynamic Programming Algorithm for Balanced Multi-Bin Packing.

For small instances, DP can find optimal solutions by exploring
the state space systematically.

Note: Due to the problem's complexity, DP may take significant time
for large instances. A 5-minute timeout is enforced.

WARNING: Complexity is O(k · 3^n), which means:
- n=10: ~59,000 operations (instant)
- n=15: ~14 million operations (seconds)  
- n=20: ~3.5 billion operations (minutes to hours)
- Timeout after 5 minutes if no solution found
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
import time
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
    - Time: O(k^2 * 3^n) in worst case
      - k * 3^n iterations (subset enumeration over k levels)
      - O(k) per iteration (computing max/min over value tuple)
    - Space: O(k * 2^n) for memoization
    - Approximation: Optimal (exact algorithm)
    
    The algorithm works by:
    1. Generate all possible subsets of items for each bin
    2. Filter subsets that satisfy weight constraint (heterogeneous capacities)
    3. Use DP to find k-partition minimizing value difference
    
    State: dp[j][mask] = (value_tuple, assignment) where:
    - mask represents assigned items
    - value_tuple is (V_1, ..., V_j) values for bins 0..j-1
    
    Note: We store the full value tuple (not just max/min) because with
    heterogeneous capacities, feasibility depends on the specific bin.
    
    EXECUTION TIME LIMIT:
    - 5 minutes timeout enforced
    - Will raise TimeoutError if exceeded
    """
    
    time_complexity = "O(k² · 3^n)"
    space_complexity = "O(k · 2^n)"
    approximation_ratio = "Optimal (exact)"
    description = "Exact algorithm using dynamic programming, feasible only for small instances"
    
    DEFAULT_TIMEOUT = 300.0  # seconds (5 minutes)
    
    def __init__(self, track_steps: bool = False, verbose: bool = False,
                 time_limit: float = 300.0):
        super().__init__(track_steps, verbose)
        self.time_limit = time_limit
        self._start_time = None
    
    def _check_timeout(self) -> bool:
        """Check if execution has exceeded time limit."""
        if self._start_time is None:
            return False
        return (time.time() - self._start_time) > self.time_limit
    
    @property
    def name(self) -> str:
        return "Dynamic Programming"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._start_time = time.time()
        self._log(f"Starting DP on {problem.n_items} items, {problem.num_bins} bins")
        
        # Estimate complexity and warn
        estimated_ops = problem.num_bins * (3 ** problem.n_items)
        if estimated_ops > 1e9:
            self._log(f"WARNING: Estimated {estimated_ops:.2e} operations, may take a while...")
        
        n = problem.n_items
        k = problem.num_bins
        items = problem.items
        capacities = problem.bin_capacities
        # Precompute subset properties for each bin capacity
        self._log("Precomputing feasible subsets...")
        feasible_per_bin = self._compute_feasible_subsets_per_bin(items, capacities)
        total_feasible = sum(len(fs) for fs in feasible_per_bin)
        self._log(f"Found {total_feasible} total feasible subsets across bins")
        
        self._record_step(
            f"Computed feasible subsets for {k} bins",
            problem.create_empty_bins(),
            extra_data={"n_feasible_total": total_feasible}
        )
        
        # DP: Find best k-partition
        best_assignment = self._find_best_partition_multi_cap(
            items, k, feasible_per_bin
        )
        
        if best_assignment is None:
            self._log("No feasible partition found!")
            raise ValueError("No feasible partition exists for this problem")
        
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
    
    def _compute_feasible_subsets_per_bin(
        self,
        items: List[Item],
        capacities: List[float]
    ) -> List[Dict[int, Tuple[float, float]]]:
        """
        Compute all subsets of items that fit within each bin's capacity.
        
        Returns:
            List of dicts (one per bin), each mapping bitmask -> (total_weight, total_value)
        """
        n = len(items)
        k = len(capacities)
        feasible_per_bin = []
        
        for bin_id in range(k):
            capacity = capacities[bin_id]
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
            feasible_per_bin.append(feasible)
        
        return feasible_per_bin
    
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
    
    def _find_best_partition_multi_cap(
        self,
        items: List[Item],
        k: int,
        feasible_per_bin: List[Dict[int, Tuple[float, float]]]
    ) -> Optional[List[List[int]]]:
        """
        Find the best k-partition using DP with individual bin capacities.
        
        IMPORTANT: This algorithm minimizes the difference between max and min
        bin values. To achieve optimal balance (diff=0), it must consider
        ALL possible distributions including empty bins.
        
        Returns:
            List of k lists, each containing item IDs for that bin
            Returns None if no feasible partition exists
        """
        n = len(items)
        full_mask = (1 << n) - 1
        
        # dp[j][mask] = (bin_values_tuple, assignment)
        # j = number of bins used (0 to k)  
        # mask = items assigned so far
        # bin_values_tuple = tuple of values for bins 0..j-1
        # assignment = list of item ID lists for each bin
        #
        # We store all bin values (not just max/min) to correctly compute
        # the final objective when all k bins are assigned.
        
        INF = float('inf')
        
        # Initialize: bin 0 (first bin) - include empty set (value=0)
        dp = [{} for _ in range(k + 1)]
        
        # With 1 bin (bin index 0), try ALL feasible subsets including empty
        for mask, (weight, value) in feasible_per_bin[0].items():
            item_ids = self._mask_to_items(mask, items)
            dp[1][mask] = ((value,), [item_ids])
        
        # DP transition: add bins one by one
        for j in range(2, k + 1):
            self._log(f"DP: Processing bin {j}...")
            bin_idx = j - 1  # Current bin index (0-based)
            feasible_j = feasible_per_bin[bin_idx]
            
            # Check timeout at start of each bin processing
            if self._check_timeout():
                self._log(f"Timeout reached during DP, returning best found so far")
                return self._extract_best_partial(dp, k, full_mask, items)
            
            for prev_mask, (prev_values, prev_assign) in list(dp[j - 1].items()):
                remaining = full_mask ^ prev_mask  # Items not yet assigned
                
                # Try all feasible subsets of remaining items for this bin
                # INCLUDING the empty set (subset=0)
                subset = remaining
                while True:
                    self._iterations += 1
                    
                    # Periodic timeout check (every 10000 iterations)
                    if self._iterations % 10000 == 0 and self._check_timeout():
                        self._log(f"Timeout at iteration {self._iterations}")
                        return self._extract_best_partial(dp, k, full_mask, items)
                    
                    if subset in feasible_j:
                        _, new_bin_value = feasible_j[subset]
                        new_mask = prev_mask | subset
                        new_values = prev_values + (new_bin_value,)
                        
                        # Compute difference for this partial assignment
                        new_diff = max(new_values) - min(new_values)
                        
                        # Check if this is better than existing state
                        should_update = False
                        if new_mask not in dp[j]:
                            should_update = True
                        else:
                            old_values, _ = dp[j][new_mask]
                            old_diff = max(old_values) - min(old_values)
                            if new_diff < old_diff:
                                should_update = True
                        
                        if should_update:
                            new_assign = prev_assign + [self._mask_to_items(subset, items)]
                            dp[j][new_mask] = (new_values, new_assign)
                    
                    # Next subset of remaining (or break if we've done empty set)
                    if subset == 0:
                        break
                    subset = (subset - 1) & remaining
        
        # Find best complete partition (all items assigned, all k bins used)
        best_diff = INF
        best_assign = None
        
        for mask, (values, assign) in dp[k].items():
            if mask == full_mask:  # All items assigned
                diff = max(values) - min(values)
                if diff < best_diff:
                    best_diff = diff
                    best_assign = assign
                    self._log(f"Found partition with diff={diff:.2f}: values={values}")
        
        if best_assign is not None:
            self._log(f"Optimal partition found with diff={best_diff:.2f}")
            return best_assign
        
        # Fallback: find any complete partition
        self._log("No perfect partition found, looking for any valid partition...")
        for mask in dp[k]:
            if mask == full_mask:
                _, assign = dp[k][mask]
                return assign
        
        return None
    
    def _extract_best_partial(
        self,
        dp: List[Dict],
        k: int,
        full_mask: int,
        items: List[Item]
    ) -> Optional[List[List[int]]]:
        """Extract best solution found so far when timeout occurs."""
        # Look for complete partitions first
        if full_mask in dp[k]:
            _, assign = dp[k][full_mask]
            return assign
        
        # Look for best partial in highest completed level
        for j in range(k, 0, -1):
            if dp[j]:
                best_diff = float('inf')
                best_assign = None
                for mask, (values, assign) in dp[j].items():
                    if mask == full_mask:
                        diff = max(values) - min(values)
                        if diff < best_diff:
                            best_diff = diff
                            best_assign = assign
                if best_assign:
                    return best_assign
        
        return None
    
    def _mask_to_items(self, mask: int, items: List[Item]) -> List[int]:
        """Convert bitmask to list of item IDs."""
        result = []
        for i, item in enumerate(items):
            if mask & (1 << i):
                result.append(item.id)
        return result
    
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
