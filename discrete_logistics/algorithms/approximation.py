"""
Approximation Algorithms for Balanced Multi-Bin Packing.

Implements algorithms with theoretical performance guarantees:
- LPT (Longest Processing Time) Approximation
- Multi-Way Number Partitioning

These algorithms provide provable bounds on solution quality.
"""

from typing import List, Optional, Tuple
import numpy as np
try:
    from ..core.problem import Problem, Solution, Bin, Item
    from .base import Algorithm, register_algorithm
except ImportError:
    from discrete_logistics.core.problem import Problem, Solution, Bin, Item
    from discrete_logistics.algorithms.base import Algorithm, register_algorithm


@register_algorithm("lpt_approximation")
class LPTApproximation(Algorithm):
    """
    Longest Processing Time (LPT) Approximation Algorithm.
    
    Classic approximation algorithm for makespan minimization,
    adapted for the balanced bin packing objective.
    
    Complexity Analysis:
    - Time: O(n log n + n log k) with heap-based implementation
    - Space: O(n + k)
    - Approximation Ratio: 4/3 - 1/(3k) for makespan
    
    The algorithm sorts items by value (descending) and assigns each
    item to the bin with minimum current value that can fit it.
    
    Theorem: For makespan scheduling on identical machines,
    LPT achieves approximation ratio 4/3 - 1/(3m) where m = number of machines.
    
    For our balanced objective (minimize max-min difference):
    - No tight theoretical bound, but empirically effective
    - Works well when values are close to uniformly distributed
    
    Pseudocode:
    1. Sort items by value in decreasing order
    2. For each item (in sorted order):
       a. Find bin with minimum value that can fit item
       b. Assign item to that bin
    3. Return solution
    """
    
    time_complexity = "O(n log n + n log k)"
    space_complexity = "O(n + k)"
    approximation_ratio = "4/3 - 1/(3k) for makespan"
    description = "Classic LPT scheduling adapted for balanced packing"
    
    @property
    def name(self) -> str:
        return "LPT Approximation"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting LPT on {problem.n_items} items, {problem.num_bins} bins")
        
        solution = problem.create_empty_solution(self.name)
        bins = solution.bins
        k = problem.num_bins
        
        # Sort items by value (descending) - LPT order
        sorted_items = sorted(problem.items, key=lambda x: x.value, reverse=True)
        
        self._record_step(
            "LPT: Items sorted by value (descending)",
            bins,
            extra_data={
                "sorted_order": [i.id for i in sorted_items],
                "values": [i.value for i in sorted_items]
            }
        )
        
        # Use heap for efficient minimum finding
        import heapq
        
        # Heap entries: (current_value, bin_id)
        heap = [(0.0, i) for i in range(k)]
        heapq.heapify(heap)
        
        # Track bin weights separately
        bin_weights = [0.0] * k
        
        for item in sorted_items:
            self._iterations += 1
            
            # Find bin with minimum value that can fit item
            assigned = False
            temp_popped = []
            
            while heap and not assigned:
                current_val, bin_id = heapq.heappop(heap)
                
                if bin_weights[bin_id] + item.weight <= bins[bin_id].capacity:
                    # Assign item to this bin
                    bins[bin_id].add_item(item)
                    bin_weights[bin_id] += item.weight
                    new_val = current_val + item.value
                    heapq.heappush(heap, (new_val, bin_id))
                    assigned = True
                    
                    self._record_step(
                        f"Assign item {item.id} (v={item.value:.1f}) to bin {bin_id}",
                        bins,
                        item=item,
                        bin_id=bin_id
                    )
                else:
                    temp_popped.append((current_val, bin_id))
            
            # Push back bins that couldn't fit the item
            for entry in temp_popped:
                heapq.heappush(heap, entry)
            
            if not assigned:
                self._log(f"Warning: Item {item.id} could not be assigned!")
        
        solution.execution_time = self._get_elapsed_time()
        solution.iterations = self._iterations
        
        # Calculate theoretical bound
        total_value = sum(item.value for item in problem.items)
        max_item_value = max(item.value for item in problem.items)
        theoretical_opt = max(total_value / k, max_item_value)
        
        solution.metadata.update({
            "theoretical_opt_lb": theoretical_opt,
            "approximation_ratio_bound": 4/3 - 1/(3*k)
        })
        
        self._log(f"LPT completed: diff={solution.value_difference:.2f}")
        
        return solution


@register_algorithm("multiway_partition")
class MultiWayPartition(Algorithm):
    """
    Multi-Way Number Partitioning using Karmarkar-Karp Differencing.
    
    Adaptation of the Karmarkar-Karp algorithm for multi-way partitioning.
    Uses differencing method to reduce problem size.
    
    Complexity Analysis:
    - Time: O(n log n) for basic differencing
    - Space: O(n)
    - Approximation: O(1/n^θ(log n)) for 2-way partition
    
    For k-way partitioning, uses recursive differencing approach.
    
    The KK algorithm works by:
    1. Repeatedly replacing two largest numbers with their difference
    2. This simulates placing them in different bins
    3. For k>2, uses more sophisticated grouping
    
    Pseudocode (for k=2):
    1. Create max-heap with all item values
    2. While heap has more than one element:
       a. Pop two largest values a, b
       b. Push |a - b| back to heap
    3. Final value is minimum achievable difference
    
    For k>2, we use a recursive approach with grouping.
    """
    
    time_complexity = "O(n log n)"
    space_complexity = "O(n)"
    approximation_ratio = "O(1/n^θ(log n)) for k=2"
    description = "Karmarkar-Karp differencing for multi-way partition"
    
    @property
    def name(self) -> str:
        return "Multi-Way Partition"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting Multi-Way Partition: n={problem.n_items}, k={problem.num_bins}")
        
        k = problem.num_bins
        
        if k == 2:
            return self._solve_two_way(problem)
        else:
            return self._solve_k_way(problem)
    
    def _solve_two_way(self, problem: Problem) -> Solution:
        """
        Solve 2-way partition using Karmarkar-Karp differencing.
        """
        import heapq
        
        n = problem.n_items
        items = problem.items
        capacities = problem.bin_capacities
        
        # Create max-heap with (value, [item_ids])
        # Using negative values for max-heap behavior
        heap = [(-item.value, [item.id]) for item in items]
        heapq.heapify(heap)
        
        self._record_step(
            "KK: Initial heap with item values",
            problem.create_empty_bins(),
            extra_data={"heap_size": len(heap)}
        )
        
        # Differencing process
        while len(heap) > 1:
            self._iterations += 1
            
            # Pop two largest
            neg_val1, items1 = heapq.heappop(heap)
            neg_val2, items2 = heapq.heappop(heap)
            
            val1, val2 = -neg_val1, -neg_val2
            
            # Push difference with combined item list
            diff = abs(val1 - val2)
            if val1 >= val2:
                combined = items1 + items2
            else:
                combined = items2 + items1
            
            if diff > 0:
                heapq.heappush(heap, (-diff, combined))
        
        # Build solution from final grouping
        # Items in the final group go to bin 0
        # Others go to bin 1
        
        solution = problem.create_empty_solution(self.name)
        bins = solution.bins
        
        if heap:
            _, final_group = heap[0]
            final_set = set(final_group)
        else:
            final_set = set()
        
        for item in items:
            if item.id in final_set:
                # Alternate between bins to respect capacity
                if bins[0].can_fit(item):
                    bins[0].add_item(item)
                elif bins[1].can_fit(item):
                    bins[1].add_item(item)
            else:
                if bins[1].can_fit(item):
                    bins[1].add_item(item)
                elif bins[0].can_fit(item):
                    bins[0].add_item(item)
        
        # Post-process to ensure feasibility and better balance
        solution = self._rebalance(solution, problem)
        
        solution.execution_time = self._get_elapsed_time()
        solution.iterations = self._iterations
        
        return solution
    
    def _solve_k_way(self, problem: Problem) -> Solution:
        """
        Solve k-way partition using greedy differencing approach.
        """
        k = problem.num_bins
        items = sorted(problem.items, key=lambda x: x.value, reverse=True)
        capacities = problem.bin_capacities
        
        # Use greedy assignment with look-ahead
        solution = problem.create_empty_solution(self.name)
        bins = solution.bins
        
        for item in items:
            self._iterations += 1
            
            # Find bin that minimizes resulting max-min difference
            best_bin = None
            best_diff = float('inf')
            
            for bin in bins:
                if not bin.can_fit(item):
                    continue
                
                # Simulate assignment
                current_values = [b.current_value for b in bins]
                current_values[bin.id] += item.value
                
                diff = max(current_values) - min(current_values)
                if diff < best_diff:
                    best_diff = diff
                    best_bin = bin
            
            if best_bin:
                best_bin.add_item(item)
        
        # Apply differencing-based local improvement
        solution = self._local_improvement(solution, problem)
        
        solution.execution_time = self._get_elapsed_time()
        solution.iterations = self._iterations
        
        return solution
    
    def _rebalance(self, solution: Solution, problem: Problem) -> Solution:
        """Rebalance solution using local moves."""
        bins = solution.bins
        # Each bin has its own capacity via bins[i].capacity
        
        improved = True
        max_iter = 100
        iter_count = 0
        
        while improved and iter_count < max_iter:
            iter_count += 1
            improved = False
            
            # Find bins with max and min values
            values = [(b.current_value, b.id) for b in bins]
            max_val, max_bin_id = max(values)
            min_val, min_bin_id = min(values)
            
            if max_bin_id == min_bin_id:
                break
            
            current_diff = max_val - min_val
            
            # Try moving items from max bin to min bin
            max_bin = bins[max_bin_id]
            min_bin = bins[min_bin_id]
            
            for item in max_bin.items:
                if min_bin.can_fit(item):
                    # Calculate new difference
                    new_max_val = max_val - item.value
                    new_min_val = min_val + item.value
                    
                    # Recalculate actual max/min
                    test_values = [b.current_value for b in bins]
                    test_values[max_bin_id] -= item.value
                    test_values[min_bin_id] += item.value
                    
                    new_diff = max(test_values) - min(test_values)
                    
                    if new_diff < current_diff:
                        max_bin.remove_item(item)
                        min_bin.add_item(item)
                        improved = True
                        break
        
        return solution
    
    def _local_improvement(self, solution: Solution, problem: Problem) -> Solution:
        """Apply local search improvement."""
        bins = solution.bins
        # Each bin has its own capacity via bins[i].capacity
        
        improved = True
        max_iter = 50
        iter_count = 0
        
        while improved and iter_count < max_iter:
            iter_count += 1
            improved = False
            
            current_diff = solution.value_difference
            
            # Try all single-item moves
            for src_bin in bins:
                for item in src_bin.items[:]:  # Copy list to allow modification
                    for dst_bin in bins:
                        if dst_bin.id == src_bin.id:
                            continue
                        if not dst_bin.can_fit(item):
                            continue
                        
                        # Test move
                        src_bin.remove_item(item)
                        dst_bin.add_item(item)
                        
                        new_diff = solution.value_difference
                        
                        if new_diff < current_diff:
                            current_diff = new_diff
                            improved = True
                        else:
                            # Undo move
                            dst_bin.remove_item(item)
                            src_bin.add_item(item)
                    
                    if improved:
                        break
                if improved:
                    break
        
        return solution


@register_algorithm("dual_approximation")
class DualApproximation(Algorithm):
    """
    Dual Approximation Algorithm.
    
    Uses binary search on the objective value combined with
    a feasibility test.
    
    Complexity Analysis:
    - Time: O(n log n log V) where V is value range
    - Space: O(n + k)
    - Approximation: Depends on feasibility oracle
    
    The algorithm:
    1. Binary search on target difference D
    2. For each D, check if items can be packed with max-min ≤ D
    3. Use First Fit Decreasing as feasibility oracle
    """
    
    time_complexity = "O(n log n log V)"
    space_complexity = "O(n + k)"
    approximation_ratio = "Problem-dependent"
    description = "Binary search with feasibility oracle"
    
    @property
    def name(self) -> str:
        return "Dual Approximation"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        
        # Get bounds on optimal difference
        total_value = problem.total_value
        k = problem.num_bins
        
        # Lower bound: 0 (perfect balance)
        # Upper bound: total_value (all in one bin)
        lo = 0.0
        hi = total_value
        
        # Binary search precision
        epsilon = min(item.value for item in problem.items) / 2
        
        best_solution = None
        
        while hi - lo > epsilon:
            self._iterations += 1
            mid = (lo + hi) / 2
            
            # Try to find solution with difference ≤ mid
            solution = self._feasibility_check(problem, mid)
            
            if solution and solution.is_valid:
                if solution.value_difference <= mid + epsilon:
                    best_solution = solution
                    hi = solution.value_difference
                else:
                    lo = mid
            else:
                lo = mid
        
        if best_solution is None:
            # Fallback to greedy
            from .greedy import BestFitDecreasing
            greedy = BestFitDecreasing()
            best_solution = greedy.solve(problem)
        
        best_solution.algorithm_name = self.name
        best_solution.execution_time = self._get_elapsed_time()
        best_solution.iterations = self._iterations
        
        return best_solution
    
    def _feasibility_check(
        self,
        problem: Problem,
        target_diff: float
    ) -> Optional[Solution]:
        """
        Check if a solution with difference ≤ target_diff is achievable.
        Uses constrained greedy approach.
        """
        k = problem.num_bins
        capacities = problem.bin_capacities
        total_value = problem.total_value
        
        # Target per-bin value for balance
        target_per_bin = total_value / k
        
        # Value bounds based on target difference
        min_target = target_per_bin - target_diff / 2
        max_target = target_per_bin + target_diff / 2
        
        solution = problem.create_empty_solution("Feasibility Check")
        bins = solution.bins
        
        sorted_items = sorted(problem.items, key=lambda x: x.value, reverse=True)
        
        for item in sorted_items:
            # Find bin closest to target that can fit item
            best_bin = None
            best_score = float('inf')
            
            for bin in bins:
                if not bin.can_fit(item):
                    continue
                
                new_value = bin.current_value + item.value
                
                # Score: distance from target, penalize exceeding max_target
                if new_value <= max_target:
                    score = abs(new_value - target_per_bin)
                else:
                    score = (new_value - max_target) * 10 + target_diff
                
                if score < best_score:
                    best_score = score
                    best_bin = bin
            
            if best_bin:
                best_bin.add_item(item)
        
        return solution
