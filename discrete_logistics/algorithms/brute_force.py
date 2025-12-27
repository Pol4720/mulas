"""
Brute Force Algorithm for Balanced Multi-Bin Packing.

Enumerates ALL possible k^n assignments to find the globally optimal solution.
This serves as a baseline to compare heuristic solutions on small instances.

WARNING: Complexity is O(k^n), making it feasible only for very small instances.
- n=8, k=3:  6,561 combinations (instant)
- n=10, k=3: 59,049 combinations (< 1 second)
- n=12, k=3: 531,441 combinations (seconds)
- n=15, k=3: 14,348,907 combinations (minutes)
- n=20, k=3: 3.5 billion combinations (impractical)

Reference:
    Garey & Johnson (1979) "Computers and Intractability"
    - Multi-processor scheduling is NP-hard
    - Brute force is the only way to guarantee optimality without pruning
"""

from typing import List, Optional, Tuple, Generator
from itertools import product
import time

try:
    from ..core.problem import Problem, Solution, Bin, Item
    from .base import Algorithm, register_algorithm
except ImportError:
    from discrete_logistics.core.problem import Problem, Solution, Bin, Item
    from discrete_logistics.algorithms.base import Algorithm, register_algorithm


@register_algorithm("brute_force")
class BruteForce(Algorithm):
    """
    Brute Force Exhaustive Search Algorithm.
    
    Enumerates every possible assignment of n items to k bins,
    evaluates each assignment's feasibility and objective value,
    and returns the globally optimal solution.
    
    This algorithm guarantees finding the true optimal solution,
    making it the gold standard for comparing heuristic solutions.
    
    Complexity Analysis (SRTBOT Framework):
    ----------------------------------------
    S - Subproblems: All k^n possible assignments
    R - Relation: Enumerate all, pick best feasible
    T - Topological order: Sequential enumeration
    B - Base case: Empty assignment
    O - Original problem: Full assignment
    T - Time complexity: O(k^n · n) 
        - k^n assignments to enumerate
        - O(n) to evaluate each (compute bin totals)
    
    Space Complexity: O(n + k)
        - Store current best assignment: O(n)
        - Store bin totals for evaluation: O(k)
    
    Approximation Ratio: Optimal (exact algorithm)
    
    Practical Limits:
        - n ≤ 10: < 1 second (k=3)
        - n ≤ 12: < 10 seconds (k=3)
        - n ≤ 14: < 60 seconds (k=3)
        - n > 14: Not recommended
    """
    
    time_complexity = "O(k^n · n)"
    space_complexity = "O(n + k)"
    approximation_ratio = "Optimal (exact - exhaustive enumeration)"
    description = "Exhaustive search that guarantees global optimum"
    
    MAX_ITEMS = 14  # Safety limit
    DEFAULT_TIMEOUT = 120.0
    
    def __init__(self, track_steps: bool = False, verbose: bool = False,
                 max_items: int = 14, time_limit: float = 120.0):
        """
        Initialize Brute Force algorithm.
        
        Args:
            track_steps: Record steps for visualization
            verbose: Print progress information
            max_items: Maximum number of items to accept (safety limit)
            time_limit: Maximum execution time in seconds
        """
        super().__init__(track_steps, verbose)
        self.max_items = min(max_items, self.MAX_ITEMS)
        self.time_limit = time_limit
        self._start_time = None
        self._total_evaluated = 0
        self._feasible_count = 0
        
    @property
    def name(self) -> str:
        return "Brute Force"
    
    def _check_timeout(self) -> bool:
        """Check if execution has exceeded time limit."""
        if self._start_time is None:
            return False
        return (time.time() - self._start_time) > self.time_limit
    
    def _evaluate_assignment(
        self, 
        assignment: Tuple[int, ...], 
        items: List[Item],
        capacities: List[float],
        k: int
    ) -> Tuple[bool, float, List[float], List[float]]:
        """
        Evaluate a complete assignment.
        
        Args:
            assignment: Tuple where assignment[i] = bin for item i
            items: List of items
            capacities: List of bin capacities
            k: Number of bins
            
        Returns:
            (is_feasible, objective, bin_values, bin_weights)
        """
        bin_weights = [0.0] * k
        bin_values = [0.0] * k
        
        for item_idx, bin_idx in enumerate(assignment):
            item = items[item_idx]
            bin_weights[bin_idx] += item.weight
            bin_values[bin_idx] += item.value
        
        # Check feasibility (capacity constraints)
        is_feasible = all(
            bin_weights[j] <= capacities[j] 
            for j in range(k)
        )
        
        if is_feasible:
            # Objective: minimize max - min value
            objective = max(bin_values) - min(bin_values)
        else:
            objective = float('inf')
        
        return is_feasible, objective, bin_values, bin_weights
    
    def _generate_all_assignments(self, n: int, k: int) -> Generator[Tuple[int, ...], None, None]:
        """
        Generate all k^n possible assignments.
        
        Uses itertools.product for memory-efficient generation.
        
        Args:
            n: Number of items
            k: Number of bins
            
        Yields:
            Tuple representing assignment (length n, values 0 to k-1)
        """
        # Each item can go to any of k bins
        return product(range(k), repeat=n)
    
    def solve(self, problem: Problem) -> Solution:
        """
        Find the globally optimal solution by exhaustive enumeration.
        
        Args:
            problem: Problem instance
            
        Returns:
            Optimal Solution (guaranteed)
        """
        self._start_timer()
        self._start_time = time.time()
        self._total_evaluated = 0
        self._feasible_count = 0
        
        n = problem.n_items
        k = problem.num_bins
        items = problem.items
        capacities = problem.bin_capacities
        
        self._log(f"Starting Brute Force on {n} items, {k} bins")
        total_combinations = k ** n
        self._log(f"Total combinations to evaluate: {total_combinations:,}")
        
        # Safety check
        if n > self.max_items:
            self._log(f"ERROR: Instance too large (n={n} > {self.max_items})")
            raise ValueError(
                f"Brute Force requires n ≤ {self.max_items}. "
                f"Instance has n={n}. Use Branch and Bound or heuristics instead."
            )
        
        # Estimate time and warn
        if total_combinations > 1_000_000:
            estimated_time = total_combinations / 500_000  # rough estimate
            self._log(f"WARNING: Estimated time ~{estimated_time:.1f}s")
        
        # Track best solution
        best_assignment: Optional[Tuple[int, ...]] = None
        best_objective = float('inf')
        best_values: Optional[List[float]] = None
        
        # Progress reporting interval
        report_interval = max(1, total_combinations // 100)
        
        # Enumerate ALL assignments
        for idx, assignment in enumerate(self._generate_all_assignments(n, k)):
            self._iterations += 1
            self._total_evaluated += 1
            
            # Progress reporting
            if idx % report_interval == 0:
                progress = (idx + 1) / total_combinations * 100
                self._log(f"Progress: {progress:.1f}% ({idx+1:,}/{total_combinations:,})")
            
            # Timeout check
            if idx % 10000 == 0 and self._check_timeout():
                self._log(f"TIMEOUT after evaluating {idx:,} assignments")
                break
            
            # Evaluate this assignment
            is_feasible, objective, bin_values, bin_weights = self._evaluate_assignment(
                assignment, items, capacities, k
            )
            
            if is_feasible:
                self._feasible_count += 1
                
                if objective < best_objective:
                    best_objective = objective
                    best_assignment = assignment
                    best_values = bin_values.copy()
                    
                    self._log(f"New best: objective={objective:.4f} at assignment #{idx}")
                    
                    self._record_step(
                        f"New best: diff={objective:.4f}",
                        None,  # We'll build bins later
                        extra_data={
                            "assignment": list(assignment),
                            "bin_values": bin_values,
                            "evaluated": idx + 1
                        }
                    )
        
        # Build final solution
        solution = problem.create_empty_solution(self.name)
        
        if best_assignment is not None:
            for item_idx, bin_idx in enumerate(best_assignment):
                item = items[item_idx]
                solution.bins[bin_idx].add_item(item)
            
            self._log(f"Optimal solution found: diff={best_objective:.4f}")
        else:
            self._log("WARNING: No feasible solution found!")
        
        solution.execution_time = self._get_elapsed_time()
        solution.iterations = self._iterations
        
        # Store metadata for analysis
        solution.metadata.update({
            "exact": True,
            "optimal": True,
            "total_combinations": total_combinations,
            "evaluated_combinations": self._total_evaluated,
            "feasible_combinations": self._feasible_count,
            "feasibility_rate": self._feasible_count / max(1, self._total_evaluated),
            "optimal_assignment": list(best_assignment) if best_assignment else None,
            "timed_out": self._check_timeout()
        })
        
        self._log(f"Completed in {solution.execution_time:.4f}s")
        self._log(f"Evaluated: {self._total_evaluated:,} | Feasible: {self._feasible_count:,}")
        
        return solution
    
    @staticmethod
    def estimate_time(n: int, k: int, ops_per_second: int = 500_000) -> float:
        """
        Estimate execution time for given instance size.
        
        Args:
            n: Number of items
            k: Number of bins
            ops_per_second: Estimated operations per second
            
        Returns:
            Estimated time in seconds
        """
        total = k ** n
        return total / ops_per_second
    
    @staticmethod
    def max_feasible_n(k: int, time_limit: float = 60.0, ops_per_second: int = 500_000) -> int:
        """
        Calculate maximum n that can be solved within time limit.
        
        Args:
            k: Number of bins
            time_limit: Maximum time in seconds
            ops_per_second: Estimated operations per second
            
        Returns:
            Maximum n value
        """
        import math
        max_ops = time_limit * ops_per_second
        # k^n = max_ops => n = log_k(max_ops)
        return int(math.log(max_ops) / math.log(k))


# Alias for clarity
@register_algorithm("exhaustive_search")
class ExhaustiveSearch(BruteForce):
    """Alias for BruteForce - exhaustive enumeration algorithm."""
    
    @property
    def name(self) -> str:
        return "Exhaustive Search"
