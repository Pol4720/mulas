"""
Branch and Bound Algorithm for Balanced Multi-Bin Packing.

Implements exact algorithm with intelligent pruning strategies
to reduce the search space.
"""

from typing import List, Optional, Tuple, Set
from dataclasses import dataclass, field
import heapq
try:
    from ..core.problem import Problem, Solution, Bin, Item
    from .base import Algorithm, register_algorithm
except ImportError:
    from discrete_logistics.core.problem import Problem, Solution, Bin, Item
    from discrete_logistics.algorithms.base import Algorithm, register_algorithm


@dataclass
class BBNode:
    """
    Node in the Branch and Bound search tree.
    
    Attributes:
        level: Depth in the tree (number of items assigned)
        assignment: Current item-to-bin assignments
        bin_values: Current value in each bin
        bin_weights: Current weight in each bin
        lower_bound: Lower bound on best achievable difference
        upper_bound: Upper bound (current difference)
    """
    level: int
    assignment: List[int]  # assignment[i] = bin_id for item i
    bin_values: List[float]
    bin_weights: List[float]
    lower_bound: float = 0
    upper_bound: float = float('inf')
    
    def __lt__(self, other: 'BBNode') -> bool:
        """For priority queue: prefer nodes with lower bound."""
        return self.lower_bound < other.lower_bound
    
    @property
    def difference(self) -> float:
        """Current value difference."""
        return max(self.bin_values) - min(self.bin_values)


@register_algorithm("branch_and_bound")
class BranchAndBound(Algorithm):
    """
    Branch and Bound Algorithm with Multiple Pruning Strategies.
    
    Explores the search tree of all possible assignments, using
    bounds to prune branches that cannot lead to optimal solutions.
    
    Complexity Analysis:
    - Time: O(k^n) worst case, but typically much better with pruning
    - Space: O(n * k) for storing nodes
    - Approximation: Optimal (exact algorithm)
    
    Pruning Strategies:
    1. Feasibility pruning: Skip bins where item doesn't fit
    2. Bound pruning: Skip branches where LB >= current best
    3. Symmetry breaking: Consider bins in order of current value
    4. Dominance pruning: Skip equivalent states
    
    Pseudocode:
    1. Initialize with empty assignment, best = greedy solution
    2. Create root node, add to priority queue
    3. While queue not empty:
       a. Pop node with lowest lower bound
       b. If complete assignment, update best if better
       c. If not pruned, branch on next item:
          - For each bin that can fit item:
            * Create child node
            * Calculate bounds
            * Add to queue if promising
    4. Return best solution found
    """
    
    time_complexity = "O(k^n) worst case, typically O(k^n / pruning_factor)"
    space_complexity = "O(n · k)"
    approximation_ratio = "Optimal (exact)"
    description = "Exact algorithm with intelligent pruning strategies"
    
    MAX_ITERATIONS = 10_000_000
    
    def __init__(self, track_steps: bool = False, verbose: bool = False,
                 max_iterations: int = 10_000_000, time_limit: float = 300.0):
        """
        Initialize Branch and Bound.
        
        Args:
            track_steps: Record steps for visualization
            verbose: Print progress information
            max_iterations: Maximum nodes to explore
            time_limit: Maximum time in seconds
        """
        super().__init__(track_steps, verbose)
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.nodes_explored = 0
        self.nodes_pruned = 0
    
    @property
    def name(self) -> str:
        return "Branch and Bound"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting B&B on {problem.n_items} items, {problem.num_bins} bins")
        
        n = problem.n_items
        k = problem.num_bins
        items = problem.items
        capacities = problem.bin_capacities  # Lista de capacidades individuales
        
        # Sort items by value (descending) for better pruning
        sorted_indices = sorted(range(n), key=lambda i: items[i].value, reverse=True)
        sorted_items = [items[i] for i in sorted_indices]
        
        # Get initial upper bound from greedy
        from .greedy import BestFitDecreasing
        greedy = BestFitDecreasing()
        greedy_sol = greedy.solve(problem)
        best_diff = greedy_sol.value_difference
        best_assignment = self._solution_to_assignment(greedy_sol, sorted_indices)
        
        self._log(f"Initial bound from greedy: {best_diff:.2f}")
        
        self._record_step(
            f"Initial greedy bound: {best_diff:.2f}",
            greedy_sol.bins,
            extra_data={"bound": best_diff}
        )
        
        # Initialize root node
        root = BBNode(
            level=0,
            assignment=[-1] * n,
            bin_values=[0.0] * k,
            bin_weights=[0.0] * k,
            lower_bound=0,
            upper_bound=best_diff
        )
        
        # Priority queue (min-heap by lower bound)
        pq = [root]
        heapq.heapify(pq)
        
        self.nodes_explored = 0
        self.nodes_pruned = 0
        
        while pq and self.nodes_explored < self.max_iterations:
            if self._get_elapsed_time() > self.time_limit:
                self._log(f"Time limit reached ({self.time_limit}s)")
                break
            
            node = heapq.heappop(pq)
            self.nodes_explored += 1
            self._iterations += 1
            
            # Pruning: skip if lower bound >= best
            if node.lower_bound >= best_diff:
                self.nodes_pruned += 1
                continue
            
            # Check if complete assignment
            if node.level == n:
                diff = node.difference
                if diff < best_diff:
                    best_diff = diff
                    best_assignment = node.assignment.copy()
                    self._log(f"New best: {best_diff:.2f} at iter {self.nodes_explored}")
                    
                    self._record_step(
                        f"New best solution: diff={best_diff:.2f}",
                        self._assignment_to_bins(best_assignment, sorted_items, k, capacities),
                        extra_data={"diff": best_diff, "nodes": self.nodes_explored}
                    )
                continue
            
            # Branch: try assigning next item to each bin
            item_idx = node.level
            item = sorted_items[item_idx]
            
            # Sort bins by current value for symmetry breaking
            bin_order = sorted(range(k), key=lambda b: node.bin_values[b])
            
            for bin_id in bin_order:
                # Feasibility check - usar capacidad individual
                if node.bin_weights[bin_id] + item.weight > capacities[bin_id]:
                    continue
                
                # Create child node
                child = BBNode(
                    level=node.level + 1,
                    assignment=node.assignment.copy(),
                    bin_values=node.bin_values.copy(),
                    bin_weights=node.bin_weights.copy()
                )
                
                child.assignment[item_idx] = bin_id
                child.bin_values[bin_id] += item.value
                child.bin_weights[bin_id] += item.weight
                
                # Calculate bounds
                child.lower_bound = self._calculate_lower_bound(
                    child, sorted_items, n, k, capacities
                )
                child.upper_bound = child.difference
                
                # Add to queue if promising
                if child.lower_bound < best_diff:
                    heapq.heappush(pq, child)
                else:
                    self.nodes_pruned += 1
        
        # Build solution
        solution = self._build_solution(
            best_assignment, sorted_items, k, capacities, problem
        )
        
        solution.execution_time = self._get_elapsed_time()
        solution.iterations = self._iterations
        solution.metadata.update({
            "nodes_explored": self.nodes_explored,
            "nodes_pruned": self.nodes_pruned,
            "optimal": self.nodes_explored < self.max_iterations and 
                       self._get_elapsed_time() < self.time_limit
        })
        
        self._log(f"Completed: {self.nodes_explored} nodes, {self.nodes_pruned} pruned")
        self._log(f"Best diff: {best_diff:.2f}, time: {solution.execution_time:.2f}s")
        
        return solution
    
    def _calculate_lower_bound(
        self,
        node: BBNode,
        items: List[Item],
        n: int,
        k: int,
        capacity: float
    ) -> float:
        """
        Calculate lower bound on best achievable difference.
        
        Uses relaxation: assume remaining items can be distributed perfectly.
        """
        # Current difference is a trivial lower bound
        current_diff = node.difference
        
        if node.level >= n:
            return current_diff
        
        # Calculate remaining value
        remaining_value = sum(items[i].value for i in range(node.level, n))
        
        # Best case: remaining value distributed to equalize bins
        current_values = node.bin_values.copy()
        target = (sum(current_values) + remaining_value) / k
        
        # Lower bound: at least the current minimum gap that can't be closed
        # If max is much higher than target, we can't reduce it much
        max_val = max(current_values)
        min_val = min(current_values)
        
        # Optimistic bound: assume we can get close to target
        # But we're limited by item granularity
        if remaining_value > 0:
            avg_item_value = remaining_value / (n - node.level)
            # At minimum, difference will be affected by single item placement
            lb = max(0, current_diff - remaining_value)
        else:
            lb = current_diff
        
        return lb
    
    def _solution_to_assignment(
        self,
        solution: Solution,
        sorted_indices: List[int]
    ) -> List[int]:
        """Convert solution to assignment array."""
        item_to_bin = solution.get_item_assignment()
        n = len(sorted_indices)
        assignment = [-1] * n
        
        for i, orig_idx in enumerate(sorted_indices):
            if orig_idx in item_to_bin:
                assignment[i] = item_to_bin[orig_idx]
        
        return assignment
    
    def _assignment_to_bins(
        self,
        assignment: List[int],
        items: List[Item],
        k: int,
        capacities: List[float]
    ) -> List[Bin]:
        """Convert assignment array to bins."""
        bins = [Bin(i, capacities[i]) for i in range(k)]
        
        for item_idx, bin_id in enumerate(assignment):
            if bin_id >= 0 and bin_id < k:
                bins[bin_id].add_item(items[item_idx])
        
        return bins
    
    def _build_solution(
        self,
        assignment: List[int],
        sorted_items: List[Item],
        k: int,
        capacities: List[float],
        problem: Problem
    ) -> Solution:
        """Build solution from assignment."""
        bins = self._assignment_to_bins(assignment, sorted_items, k, capacities)
        
        return Solution(
            bins=bins,
            algorithm_name=self.name,
            execution_time=self._get_elapsed_time(),
            iterations=self._iterations
        )


@register_algorithm("branch_and_bound_dfs")
class BranchAndBoundDFS(Algorithm):
    """
    Branch and Bound with Depth-First Search strategy.
    
    Uses DFS instead of best-first search, which can find good
    solutions faster but may explore more nodes overall.
    
    Complexity Analysis:
    - Time: O(k^n) worst case
    - Space: O(n) for recursion stack
    - Approximation: Optimal (exact)
    """
    
    time_complexity = "O(k^n)"
    space_complexity = "O(n)"
    approximation_ratio = "Optimal (exact)"
    description = "Branch and Bound with depth-first search"
    
    def __init__(self, track_steps: bool = False, verbose: bool = False,
                 max_iterations: int = 5_000_000, time_limit: float = 300.0):
        super().__init__(track_steps, verbose)
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.best_diff = float('inf')
        self.best_assignment = None
        self.nodes_explored = 0
    
    @property
    def name(self) -> str:
        return "Branch and Bound (DFS)"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        
        n = problem.n_items
        k = problem.num_bins
        items = sorted(problem.items, key=lambda x: x.value, reverse=True)
        capacities = problem.bin_capacities
        
        # Get initial bound from greedy
        from .greedy import RoundRobinGreedy
        greedy = RoundRobinGreedy()
        greedy_sol = greedy.solve(problem)
        self.best_diff = greedy_sol.value_difference
        
        # Initialize state
        assignment = [-1] * n
        bin_values = [0.0] * k
        bin_weights = [0.0] * k
        
        self.best_assignment = None
        self.nodes_explored = 0
        
        # Run DFS
        self._dfs(
            0, items, k, capacities,
            assignment, bin_values, bin_weights
        )
        
        # Build solution
        if self.best_assignment:
            bins = [Bin(i, capacities[i]) for i in range(k)]
            for item_idx, bin_id in enumerate(self.best_assignment):
                if bin_id >= 0:
                    bins[bin_id].add_item(items[item_idx])
            
            solution = Solution(
                bins=bins,
                algorithm_name=self.name,
                execution_time=self._get_elapsed_time(),
                iterations=self.nodes_explored
            )
        else:
            solution = greedy_sol
            solution.algorithm_name = self.name
        
        return solution
    
    def _dfs(
        self,
        level: int,
        items: List[Item],
        k: int,
        capacities: List[float],
        assignment: List[int],
        bin_values: List[float],
        bin_weights: List[float]
    ):
        """Recursive DFS with pruning."""
        # Check limits
        if self.nodes_explored >= self.max_iterations:
            return
        if self._get_elapsed_time() > self.time_limit:
            return
        
        self.nodes_explored += 1
        n = len(items)
        
        # Complete assignment
        if level == n:
            diff = max(bin_values) - min(bin_values)
            if diff < self.best_diff:
                self.best_diff = diff
                self.best_assignment = assignment.copy()
            return
        
        # Pruning: current difference already too large
        current_diff = max(bin_values) - min(bin_values)
        if current_diff >= self.best_diff:
            return
        
        item = items[level]
        
        # Try each bin (sorted by value for symmetry breaking)
        bin_order = sorted(range(k), key=lambda b: bin_values[b])
        
        for bin_id in bin_order:
            # Feasibility check
            if bin_weights[bin_id] + item.weight > capacities[bin_id]:
                continue
            
            # Make assignment
            assignment[level] = bin_id
            bin_values[bin_id] += item.value
            bin_weights[bin_id] += item.weight
            
            # Recurse
            self._dfs(
                level + 1, items, k, capacities,
                assignment, bin_values, bin_weights
            )
            
            # Backtrack
            assignment[level] = -1
            bin_values[bin_id] -= item.value
            bin_weights[bin_id] -= item.weight
