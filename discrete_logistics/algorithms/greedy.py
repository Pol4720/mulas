"""
Greedy Algorithms for Balanced Multi-Bin Packing.

Implements various greedy heuristics:
- First Fit Decreasing (FFD)
- Best Fit Decreasing (BFD)
- Worst Fit Decreasing (WFD)
- Round Robin Greedy
- Largest Difference First (LDF)

Greedy algorithms provide fast, reasonable solutions but no optimality guarantee.
"""

from typing import List, Optional, Callable
try:
    from ..core.problem import Problem, Solution, Bin, Item
    from .base import Algorithm, register_algorithm
except ImportError:
    from discrete_logistics.core.problem import Problem, Solution, Bin, Item
    from discrete_logistics.algorithms.base import Algorithm, register_algorithm


@register_algorithm("first_fit_decreasing")
class FirstFitDecreasing(Algorithm):
    """
    First Fit Decreasing (FFD) Algorithm.
    
    Strategy: Sort items by value (descending), assign each item to the
    first bin where it fits and that minimizes the current value difference.
    
    Complexity Analysis:
    - Time: O(n log n + n * k) where n = items, k = bins
    - Space: O(n + k)
    - Approximation: No guaranteed bound for balance objective
    
    Pseudocode:
    1. Sort items by value in descending order
    2. For each item:
       a. Find first bin that can fit the item
       b. If tie, prefer bin with lower current value (balancing)
       c. Assign item to selected bin
    3. Return solution
    """
    
    time_complexity = "O(n log n + n·k)"
    space_complexity = "O(n + k)"
    approximation_ratio = "No guaranteed bound"
    description = "Assigns items to first fitting bin, sorted by decreasing value"
    
    @property
    def name(self) -> str:
        return "First Fit Decreasing"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting FFD on {problem.n_items} items, {problem.num_bins} bins")
        
        # Create solution with empty bins
        solution = problem.create_empty_solution(self.name)
        bins = solution.bins
        
        # Sort items by value (descending)
        sorted_items = sorted(problem.items, key=lambda x: x.value, reverse=True)
        
        self._record_step(
            "Initialize: Sorted items by value (descending)",
            bins,
            extra_data={"sorted_order": [i.id for i in sorted_items]}
        )
        
        # Assign each item
        for item in sorted_items:
            self._iterations += 1
            
            # Find first bin that can fit the item
            # Prefer bins with lower value for balancing
            best_bin = None
            best_value = float('inf')
            
            for bin in bins:
                if bin.can_fit(item):
                    if best_bin is None or bin.current_value < best_value:
                        best_bin = bin
                        best_value = bin.current_value
            
            if best_bin is not None:
                best_bin.add_item(item)
                self._record_step(
                    f"Assign item {item.id} (v={item.value:.1f}) to bin {best_bin.id}",
                    bins,
                    item=item,
                    bin_id=best_bin.id
                )
            else:
                self._log(f"Warning: Item {item.id} could not be assigned!")
        
        solution.execution_time = self._get_elapsed_time()
        solution.iterations = self._iterations
        
        self._log(f"Completed in {solution.execution_time:.4f}s, "
                  f"value_diff={solution.value_difference:.2f}")
        
        return solution


@register_algorithm("best_fit_decreasing")
class BestFitDecreasing(Algorithm):
    """
    Best Fit Decreasing (BFD) Algorithm.
    
    Strategy: Sort items by value (descending), assign each item to the
    bin that results in the best balance after assignment.
    
    Complexity Analysis:
    - Time: O(n log n + n * k)
    - Space: O(n + k)
    - Approximation: No guaranteed bound
    
    Pseudocode:
    1. Sort items by value in descending order
    2. For each item:
       a. For each bin that can fit the item:
          - Calculate resulting value difference if item assigned
       b. Choose bin that minimizes the max value difference
       c. Assign item to selected bin
    3. Return solution
    """
    
    time_complexity = "O(n log n + n·k)"
    space_complexity = "O(n + k)"
    approximation_ratio = "No guaranteed bound"
    description = "Assigns items to bin that best maintains balance"
    
    @property
    def name(self) -> str:
        return "Best Fit Decreasing"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting BFD on {problem.n_items} items, {problem.num_bins} bins")
        
        solution = problem.create_empty_solution(self.name)
        bins = solution.bins
        
        sorted_items = sorted(problem.items, key=lambda x: x.value, reverse=True)
        
        self._record_step(
            "Initialize: Sorted items by value (descending)",
            bins,
            extra_data={"sorted_order": [i.id for i in sorted_items]}
        )
        
        for item in sorted_items:
            self._iterations += 1
            
            best_bin = None
            best_diff = float('inf')
            
            # Find bin that results in minimum value difference
            for bin in bins:
                if bin.can_fit(item):
                    # Simulate adding item
                    temp_value = bin.current_value + item.value
                    
                    # Calculate resulting difference
                    all_values = [b.current_value for b in bins]
                    all_values[bin.id] = temp_value
                    
                    diff = max(all_values) - min(all_values)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_bin = bin
            
            if best_bin is not None:
                best_bin.add_item(item)
                self._record_step(
                    f"Assign item {item.id} (v={item.value:.1f}) to bin {best_bin.id} "
                    f"(best diff: {best_diff:.1f})",
                    bins,
                    item=item,
                    bin_id=best_bin.id,
                    extra_data={"resulting_diff": best_diff}
                )
            else:
                self._log(f"Warning: Item {item.id} could not be assigned!")
        
        solution.execution_time = self._get_elapsed_time()
        solution.iterations = self._iterations
        
        return solution


@register_algorithm("worst_fit_decreasing")
class WorstFitDecreasing(Algorithm):
    """
    Worst Fit Decreasing (WFD) Algorithm.
    
    Strategy: Assign items to the bin with the most remaining capacity.
    Tends to spread items more evenly by weight.
    
    Complexity Analysis:
    - Time: O(n log n + n * k)
    - Space: O(n + k)
    - Approximation: No guaranteed bound
    
    Pseudocode:
    1. Sort items by value (descending)
    2. For each item:
       a. Find bin with most remaining capacity that can fit item
       b. Assign item to that bin
    3. Return solution
    """
    
    time_complexity = "O(n log n + n·k)"
    space_complexity = "O(n + k)"
    approximation_ratio = "No guaranteed bound"
    description = "Assigns items to bin with most remaining capacity"
    
    @property
    def name(self) -> str:
        return "Worst Fit Decreasing"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting WFD on {problem.n_items} items, {problem.num_bins} bins")
        
        solution = problem.create_empty_solution(self.name)
        bins = solution.bins
        
        sorted_items = sorted(problem.items, key=lambda x: x.value, reverse=True)
        
        self._record_step(
            "Initialize: Sorted items by value (descending)",
            bins,
            extra_data={"sorted_order": [i.id for i in sorted_items]}
        )
        
        for item in sorted_items:
            self._iterations += 1
            
            # Find bin with most remaining capacity
            best_bin = None
            max_remaining = -1
            
            for bin in bins:
                if bin.can_fit(item) and bin.remaining_capacity > max_remaining:
                    max_remaining = bin.remaining_capacity
                    best_bin = bin
            
            if best_bin is not None:
                best_bin.add_item(item)
                self._record_step(
                    f"Assign item {item.id} to bin {best_bin.id} "
                    f"(max capacity: {max_remaining:.1f})",
                    bins,
                    item=item,
                    bin_id=best_bin.id
                )
            else:
                self._log(f"Warning: Item {item.id} could not be assigned!")
        
        solution.execution_time = self._get_elapsed_time()
        solution.iterations = self._iterations
        
        return solution


@register_algorithm("round_robin_greedy")
class RoundRobinGreedy(Algorithm):
    """
    Round Robin Greedy Algorithm.
    
    Strategy: Distribute items in round-robin fashion after sorting,
    always assigning to the bin with minimum current value.
    
    Complexity Analysis:
    - Time: O(n log n + n * k)
    - Space: O(n + k)
    - Approximation: Provides 4/3 approximation for makespan scheduling
    
    Pseudocode:
    1. Sort items by value in descending order
    2. For each item:
       a. Find bin with minimum current value that can fit item
       b. Assign item to that bin
    3. Return solution
    """
    
    time_complexity = "O(n log n + n·k)"
    space_complexity = "O(n + k)"
    approximation_ratio = "4/3 for makespan (LPT bound)"
    description = "Round-robin assignment to bin with minimum value"
    
    @property
    def name(self) -> str:
        return "Round Robin Greedy"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting Round Robin on {problem.n_items} items, {problem.num_bins} bins")
        
        solution = problem.create_empty_solution(self.name)
        bins = solution.bins
        
        # Sort items by value descending (LPT-style)
        sorted_items = sorted(problem.items, key=lambda x: x.value, reverse=True)
        
        self._record_step(
            "Initialize: Sorted items by value (descending) - LPT style",
            bins,
            extra_data={"sorted_order": [i.id for i in sorted_items]}
        )
        
        for item in sorted_items:
            self._iterations += 1
            
            # Find bin with minimum value that can fit item
            feasible_bins = [b for b in bins if b.can_fit(item)]
            
            if feasible_bins:
                min_bin = min(feasible_bins, key=lambda b: b.current_value)
                min_bin.add_item(item)
                
                self._record_step(
                    f"Assign item {item.id} (v={item.value:.1f}) to bin {min_bin.id} "
                    f"(min value: {min_bin.current_value - item.value:.1f})",
                    bins,
                    item=item,
                    bin_id=min_bin.id
                )
            else:
                self._log(f"Warning: Item {item.id} could not be assigned!")
        
        solution.execution_time = self._get_elapsed_time()
        solution.iterations = self._iterations
        
        return solution


@register_algorithm("largest_difference_first")
class LargestDifferenceFirst(Algorithm):
    """
    Largest Difference First (LDF) Algorithm.
    
    Strategy: Iteratively assign the item that most reduces the
    maximum difference between bin values.
    
    Complexity Analysis:
    - Time: O(n² * k) due to recalculation at each step
    - Space: O(n + k)
    - Approximation: No guaranteed bound, but often produces good balance
    
    Pseudocode:
    1. While unassigned items remain:
       a. For each unassigned item:
          - For each bin that can fit it:
            * Calculate resulting value difference
          - Track best (item, bin) pair
       b. Assign best item to best bin
    2. Return solution
    """
    
    time_complexity = "O(n²·k)"
    space_complexity = "O(n + k)"
    approximation_ratio = "No guaranteed bound"
    description = "Greedily minimizes value difference at each step"
    
    @property
    def name(self) -> str:
        return "Largest Difference First"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting LDF on {problem.n_items} items, {problem.num_bins} bins")
        
        solution = problem.create_empty_solution(self.name)
        bins = solution.bins
        
        unassigned = list(problem.items)
        
        self._record_step(
            "Initialize: All items unassigned",
            bins,
            extra_data={"unassigned": [i.id for i in unassigned]}
        )
        
        while unassigned:
            self._iterations += 1
            
            best_item = None
            best_bin = None
            best_diff = float('inf')
            
            for item in unassigned:
                for bin in bins:
                    if bin.can_fit(item):
                        # Simulate assignment
                        temp_values = [b.current_value for b in bins]
                        temp_values[bin.id] += item.value
                        
                        diff = max(temp_values) - min(temp_values)
                        
                        if diff < best_diff:
                            best_diff = diff
                            best_item = item
                            best_bin = bin
            
            if best_item is not None and best_bin is not None:
                best_bin.add_item(best_item)
                unassigned.remove(best_item)
                
                self._record_step(
                    f"Assign item {best_item.id} to bin {best_bin.id} "
                    f"(best diff: {best_diff:.1f})",
                    bins,
                    item=best_item,
                    bin_id=best_bin.id,
                    extra_data={
                        "resulting_diff": best_diff,
                        "remaining": len(unassigned)
                    }
                )
            else:
                # Fallback: try to assign any remaining item to any available bin
                assigned_any = False
                for item in unassigned:
                    for bin in bins:
                        if bin.can_fit(item):
                            bin.add_item(item)
                            unassigned.remove(item)
                            self._log(f"Fallback: assigned item {item.id} to bin {bin.id}")
                            assigned_any = True
                            break
                    if assigned_any:
                        break
                
                if not assigned_any:
                    # No item could be assigned - exit to prevent infinite loop
                    self._log("Warning: Could not assign remaining items!")
                    break
        
        solution.execution_time = self._get_elapsed_time()
        solution.iterations = self._iterations
        
        return solution


# Additional utility functions for greedy algorithms

def greedy_by_criterion(
    problem: Problem,
    sort_key: Callable[[Item], float],
    bin_selector: Callable[[List[Bin], Item], Optional[Bin]],
    algorithm_name: str = "Custom Greedy",
    reverse_sort: bool = True
) -> Solution:
    """
    Generic greedy algorithm framework.
    
    Args:
        problem: Problem instance
        sort_key: Function to extract sort key from item
        bin_selector: Function to select target bin for an item
        algorithm_name: Name for the solution
        reverse_sort: Whether to sort in descending order
        
    Returns:
        Solution with greedy assignments
    """
    solution = problem.create_empty_solution(algorithm_name)
    bins = solution.bins
    
    sorted_items = sorted(problem.items, key=sort_key, reverse=reverse_sort)
    
    for item in sorted_items:
        target_bin = bin_selector(bins, item)
        if target_bin is not None:
            target_bin.add_item(item)
    
    return solution
