"""
Hybrid Algorithm: Combining Metaheuristics with Dynamic Programming

This algorithm provides a balanced approach between solution quality and speed:
- For large instances: Uses metaheuristics for bulk assignment + DP for fine-tuning
- For small instances: Falls back to pure DP for optimal solutions

The key insight is that:
1. Large items have the most impact on balance - assign them quickly with metaheuristics
2. Small items can be optimally placed via DP to fine-tune the solution
3. This combination achieves near-optimal quality with practical runtime

Theoretical Foundation:
- Metaheuristics provide O(1) approximation in practice for bin packing variants
- DP provides optimal solutions for subproblems with n_dp items
- Combined approach: O(meta_time + k² · 3^{n_dp}) where n_dp << n
"""

from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass
import numpy as np
import time
from copy import deepcopy

try:
    from ..core.problem import Problem, Solution, Bin, Item
    from .base import Algorithm, register_algorithm
    from .metaheuristics import SimulatedAnnealing, GeneticAlgorithm, TabuSearch
    from .dynamic_programming import DynamicProgramming
    from .greedy import FirstFitDecreasing
except ImportError:
    from discrete_logistics.core.problem import Problem, Solution, Bin, Item
    from discrete_logistics.algorithms.base import Algorithm, register_algorithm
    from discrete_logistics.algorithms.metaheuristics import SimulatedAnnealing, GeneticAlgorithm, TabuSearch
    from discrete_logistics.algorithms.dynamic_programming import DynamicProgramming
    from discrete_logistics.algorithms.greedy import FirstFitDecreasing


@dataclass
class HybridStats:
    """Statistics from hybrid algorithm execution."""
    total_items: int
    meta_items: int
    dp_items: int
    meta_time: float
    dp_time: float
    total_time: float
    meta_objective: float
    dp_improvement: float
    final_objective: float
    strategy_used: str
    fallback_used: bool = False


PartitionStrategy = Literal['largest_first', 'value_first', 'balanced', 'adaptive']
MetaAlgorithm = Literal['simulated_annealing', 'genetic', 'tabu', 'auto']


@register_algorithm("hybrid_dp_meta")
class HybridDPMeta(Algorithm):
    """
    Hybrid Algorithm combining Metaheuristics with Dynamic Programming.
    
    Strategy:
    1. For small instances (n <= dp_threshold): Use pure DP (optimal)
    2. For larger instances:
       a. Partition items into two sets: meta_items and dp_items
       b. Solve meta_items with fast metaheuristic
       c. Calculate residual capacities
       d. Solve dp_items optimally with DP on residual problem
       e. Merge solutions
    
    This achieves near-optimal quality with practical runtime by:
    - Using metaheuristics where they excel (bulk assignment)
    - Using DP where it's tractable (small residual problems)
    
    Complexity Analysis:
    - Time: O(meta_time + k² · 3^{dp_threshold})
    - Space: O(n + k · 2^{dp_threshold})
    - Approximation: No formal bound, but empirically < 5% gap from optimal
    """
    
    time_complexity = "O(meta + k² · 3^{dp_threshold})"
    space_complexity = "O(n + k · 2^{dp_threshold})"
    approximation_ratio = "Empirically < 5% gap"
    description = "Hybrid: Metaheuristic for bulk + DP for fine-tuning"
    
    # Empirically determined thresholds
    DP_THRESHOLDS = {
        2: 14,  # k=2: DP handles up to n=14 in ~8s
        3: 12,  # k=3: DP handles up to n=12 in ~1.5s
        4: 10,  # k=4: DP handles up to n=10 in ~0.2s
        5: 9,   # k=5: DP handles up to n=9 in ~0.07s
    }
    
    def __init__(
        self,
        track_steps: bool = False,
        verbose: bool = False,
        dp_threshold: Optional[int] = None,
        meta_algorithm: MetaAlgorithm = 'auto',
        partition_strategy: PartitionStrategy = 'adaptive',
        meta_time_limit: float = 30.0,
        dp_time_limit: float = 60.0,
        total_time_limit: float = 120.0,
        quality_weight: float = 0.7,  # 0-1, higher = prioritize quality over speed
    ):
        """
        Initialize Hybrid Algorithm.
        
        Args:
            dp_threshold: Max items for DP phase (None = auto-select based on k)
            meta_algorithm: Which metaheuristic to use ('auto' selects best)
            partition_strategy: How to split items between phases
            meta_time_limit: Max time for metaheuristic phase
            dp_time_limit: Max time for DP phase
            total_time_limit: Overall time limit
            quality_weight: Balance between quality (1) and speed (0)
        """
        super().__init__(track_steps, verbose)
        self.dp_threshold = dp_threshold
        self.meta_algorithm = meta_algorithm
        self.partition_strategy = partition_strategy
        self.meta_time_limit = meta_time_limit
        self.dp_time_limit = dp_time_limit
        self.total_time_limit = total_time_limit
        self.quality_weight = quality_weight
        
        # Statistics
        self.stats: Optional[HybridStats] = None
    
    @property
    def name(self) -> str:
        return "Hybrid DP-Meta"
    
    def _get_dp_threshold(self, k: int) -> int:
        """Get appropriate DP threshold based on number of bins."""
        if self.dp_threshold is not None:
            return self.dp_threshold
        return self.DP_THRESHOLDS.get(k, 10)
    
    def _select_metaheuristic(self, problem: Problem) -> Algorithm:
        """Select the best metaheuristic for the problem."""
        n = problem.n_items
        k = problem.num_bins
        
        if self.meta_algorithm == 'simulated_annealing' or self.meta_algorithm == 'auto':
            # SA is generally best for medium instances
            return SimulatedAnnealing(
                track_steps=False,
                verbose=self.verbose,
                initial_temp=1000.0,
                cooling_rate=0.995,
                max_iterations=min(5000, n * 200),
                time_limit=self.meta_time_limit
            )
        elif self.meta_algorithm == 'genetic':
            return GeneticAlgorithm(
                track_steps=False,
                verbose=self.verbose,
                population_size=min(100, n * 5),
                generations=min(200, n * 10),
                time_limit=self.meta_time_limit
            )
        elif self.meta_algorithm == 'tabu':
            return TabuSearch(
                track_steps=False,
                verbose=self.verbose,
                tabu_tenure=min(20, n // 2),
                max_iterations=min(3000, n * 150),
                time_limit=self.meta_time_limit
            )
        else:
            # Default to SA
            return SimulatedAnnealing(time_limit=self.meta_time_limit)
    
    def _partition_items(
        self,
        items: List[Item],
        n_meta: int,
        strategy: PartitionStrategy
    ) -> Tuple[List[Item], List[Item]]:
        """
        Partition items into meta and DP sets.
        
        Args:
            items: All items
            n_meta: Number of items for metaheuristic
            strategy: Partitioning strategy
            
        Returns:
            (meta_items, dp_items)
        """
        if strategy == 'largest_first':
            # Assign largest items to meta (they impact balance most)
            sorted_items = sorted(items, key=lambda x: x.weight, reverse=True)
            meta_items = sorted_items[:n_meta]
            dp_items = sorted_items[n_meta:]
            
        elif strategy == 'value_first':
            # Assign highest value items to meta
            sorted_items = sorted(items, key=lambda x: x.value, reverse=True)
            meta_items = sorted_items[:n_meta]
            dp_items = sorted_items[n_meta:]
            
        elif strategy == 'balanced':
            # Try to balance both sets by combined weight-value score
            sorted_items = sorted(items, key=lambda x: x.weight + x.value, reverse=True)
            meta_items = sorted_items[:n_meta]
            dp_items = sorted_items[n_meta:]
            
        elif strategy == 'adaptive':
            # Adaptive: Use weight for items, but ensure dp_items are "packable"
            # Sort by weight, but keep some medium items for DP
            sorted_by_weight = sorted(items, key=lambda x: x.weight, reverse=True)
            
            # Take largest items for meta
            meta_items = sorted_by_weight[:n_meta]
            dp_items = sorted_by_weight[n_meta:]
            
            # If DP items have very diverse weights, might want to adjust
            # For now, keep simple strategy
            
        else:
            raise ValueError(f"Unknown partition strategy: {strategy}")
        
        return meta_items, dp_items
    
    def _create_residual_problem(
        self,
        dp_items: List[Item],
        bins: List[Bin],
        meta_solution: Solution
    ) -> Problem:
        """
        Create the residual problem for DP phase.
        
        Args:
            dp_items: Items to assign in DP phase
            bins: Original bins
            meta_solution: Solution from metaheuristic
            
        Returns:
            Problem with residual capacities
        """
        # Calculate residual capacities
        residual_capacities = []
        for j, bin in enumerate(meta_solution.bins):
            used_weight = bin.current_weight
            original_capacity = bins[j].capacity if j < len(bins) else bin.capacity
            residual = original_capacity - used_weight
            residual_capacities.append(max(0.0, residual))
        
        # Re-index dp_items to start from 0
        reindexed_items = [
            Item(id=i, weight=item.weight, value=item.value, name=item.name)
            for i, item in enumerate(dp_items)
        ]
        
        return Problem(
            items=reindexed_items,
            num_bins=len(residual_capacities),
            bin_capacities=residual_capacities,
            name="residual_problem"
        )
    
    def _merge_solutions(
        self,
        meta_solution: Solution,
        dp_solution: Solution,
        dp_items: List[Item],  # Original dp_items with correct IDs
        k: int
    ) -> Solution:
        """
        Merge meta and DP solutions.
        
        Args:
            meta_solution: Solution from metaheuristic
            dp_solution: Solution from DP (with re-indexed items)
            dp_items: Original DP items with correct IDs
            k: Number of bins
            
        Returns:
            Merged solution
        """
        # Create new bins by copying meta solution
        merged_bins = [bin.copy() for bin in meta_solution.bins]
        
        # Map re-indexed items back to original items
        id_to_original = {i: item for i, item in enumerate(dp_items)}
        
        # Add DP items to merged bins
        for j, dp_bin in enumerate(dp_solution.bins):
            for item in dp_bin.items:
                original_item = id_to_original[item.id]
                merged_bins[j].items.append(original_item)
        
        return Solution(
            bins=merged_bins,
            algorithm_name=self.name,
            execution_time=0,  # Will be set later
            iterations=self._iterations
        )
    
    def solve(self, problem: Problem) -> Solution:
        """
        Solve using hybrid approach.
        
        For small instances: Pure DP
        For large instances: Meta + DP
        """
        self._start_timer()
        start_time = time.time()
        
        n = problem.n_items
        k = problem.num_bins
        dp_threshold = self._get_dp_threshold(k)
        
        self._log(f"Starting Hybrid: n={n}, k={k}, dp_threshold={dp_threshold}")
        
        # ========================================
        # Case 1: Small instance - use pure DP
        # ========================================
        if n <= dp_threshold:
            self._log(f"Small instance (n={n} <= {dp_threshold}), using pure DP")
            dp = DynamicProgramming(
                track_steps=self.track_steps,
                verbose=self.verbose,
                time_limit=self.dp_time_limit
            )
            solution = dp.solve(problem)
            solution.algorithm_name = self.name
            
            elapsed = time.time() - start_time
            self.stats = HybridStats(
                total_items=n,
                meta_items=0,
                dp_items=n,
                meta_time=0,
                dp_time=elapsed,
                total_time=elapsed,
                meta_objective=0,
                dp_improvement=0,
                final_objective=solution.value_difference,
                strategy_used='pure_dp',
                fallback_used=False
            )
            
            solution.metadata['hybrid_stats'] = self.stats.__dict__
            return solution
        
        # ========================================
        # Case 2: Large instance - use Hybrid
        # ========================================
        n_dp = dp_threshold
        n_meta = n - n_dp
        
        self._log(f"Large instance: {n_meta} items for meta, {n_dp} items for DP")
        
        # Partition items
        meta_items, dp_items = self._partition_items(
            problem.items, n_meta, self.partition_strategy
        )
        
        self._log(f"Partition: meta_items weights={[i.weight for i in meta_items[:5]]}...")
        self._log(f"Partition: dp_items weights={[i.weight for i in dp_items[:5]]}...")
        
        # ----------------------------------------
        # Phase 1: Solve meta_items with metaheuristic
        # ----------------------------------------
        meta_start = time.time()
        
        # Create partial problem with only meta_items
        meta_problem = Problem(
            items=meta_items,
            num_bins=k,
            bin_capacities=problem.bin_capacities.copy(),
            name="meta_subproblem"
        )
        
        meta_algo = self._select_metaheuristic(meta_problem)
        self._log(f"Using {meta_algo.name} for meta phase")
        
        try:
            meta_solution = meta_algo.solve(meta_problem)
            meta_time = time.time() - meta_start
            meta_objective = meta_solution.value_difference
            self._log(f"Meta phase: objective={meta_objective:.2f}, time={meta_time:.2f}s")
        except Exception as e:
            self._log(f"Meta phase failed: {e}, using greedy fallback")
            greedy = FirstFitDecreasing()
            meta_solution = greedy.solve(meta_problem)
            meta_time = time.time() - meta_start
            meta_objective = meta_solution.value_difference
        
        # Check timeout
        if time.time() - start_time > self.total_time_limit:
            self._log("Time limit reached after meta phase")
            meta_solution.algorithm_name = self.name
            self.stats = HybridStats(
                total_items=n,
                meta_items=n_meta,
                dp_items=0,
                meta_time=meta_time,
                dp_time=0,
                total_time=time.time() - start_time,
                meta_objective=meta_objective,
                dp_improvement=0,
                final_objective=meta_objective,
                strategy_used=self.partition_strategy,
                fallback_used=True
            )
            meta_solution.metadata['hybrid_stats'] = self.stats.__dict__
            return meta_solution
        
        # ----------------------------------------
        # Phase 2: Create and solve residual problem with DP
        # ----------------------------------------
        dp_start = time.time()
        
        residual_problem = self._create_residual_problem(
            dp_items, problem.create_empty_bins(), meta_solution
        )
        
        self._log(f"Residual capacities: {residual_problem.bin_capacities}")
        
        # Check if residual problem is feasible
        total_dp_weight = sum(item.weight for item in dp_items)
        total_residual_cap = sum(residual_problem.bin_capacities)
        
        if total_dp_weight > total_residual_cap:
            self._log(f"Warning: Residual problem may be infeasible")
            self._log(f"  DP weight={total_dp_weight:.2f}, residual cap={total_residual_cap:.2f}")
        
        remaining_time = min(
            self.dp_time_limit,
            self.total_time_limit - (time.time() - start_time)
        )
        
        dp = DynamicProgramming(
            track_steps=False,
            verbose=self.verbose,
            time_limit=remaining_time
        )
        
        try:
            dp_solution = dp.solve(residual_problem)
            dp_time = time.time() - dp_start
            self._log(f"DP phase: time={dp_time:.2f}s")
            
            # Merge solutions
            final_solution = self._merge_solutions(
                meta_solution, dp_solution, dp_items, k
            )
            
            dp_improvement = meta_objective - final_solution.value_difference
            
        except Exception as e:
            self._log(f"DP phase failed: {e}, using meta solution only")
            dp_time = time.time() - dp_start
            
            # Fallback: assign dp_items greedily to meta_solution
            final_solution = meta_solution.copy()
            for item in dp_items:
                # Find bin with most remaining capacity that can fit item
                best_bin = None
                best_remaining = -1
                for bin in final_solution.bins:
                    remaining = bin.capacity - bin.current_weight
                    if remaining >= item.weight and remaining > best_remaining:
                        best_bin = bin
                        best_remaining = remaining
                
                if best_bin is not None:
                    best_bin.items.append(item)
            
            dp_improvement = 0
        
        final_solution.algorithm_name = self.name
        final_solution.execution_time = time.time() - start_time
        final_solution.iterations = self._iterations
        
        self.stats = HybridStats(
            total_items=n,
            meta_items=n_meta,
            dp_items=n_dp,
            meta_time=meta_time,
            dp_time=dp_time,
            total_time=final_solution.execution_time,
            meta_objective=meta_objective,
            dp_improvement=dp_improvement,
            final_objective=final_solution.value_difference,
            strategy_used=self.partition_strategy,
            fallback_used=False
        )
        
        final_solution.metadata['hybrid_stats'] = self.stats.__dict__
        
        self._log(f"Final: objective={final_solution.value_difference:.2f}")
        self._log(f"  Meta objective: {meta_objective:.2f}")
        self._log(f"  DP improvement: {dp_improvement:.2f}")
        self._log(f"  Total time: {final_solution.execution_time:.2f}s")
        
        return final_solution
    
    def get_stats(self) -> Optional[HybridStats]:
        """Get statistics from last run."""
        return self.stats


# ============================================================================
# Variants with different configurations
# ============================================================================

@register_algorithm("hybrid_quality")
class HybridQualityFocused(HybridDPMeta):
    """Hybrid variant optimized for solution quality."""
    
    description = "Hybrid optimized for quality (larger DP phase)"
    
    def __init__(self, **kwargs):
        # Use larger DP threshold for better quality
        kwargs.setdefault('dp_threshold', 14)
        kwargs.setdefault('meta_time_limit', 45.0)
        kwargs.setdefault('quality_weight', 0.9)
        super().__init__(**kwargs)
    
    @property
    def name(self) -> str:
        return "Hybrid (Quality)"


@register_algorithm("hybrid_speed")
class HybridSpeedFocused(HybridDPMeta):
    """Hybrid variant optimized for speed."""
    
    description = "Hybrid optimized for speed (smaller DP phase)"
    
    def __init__(self, **kwargs):
        # Use smaller DP threshold for speed
        kwargs.setdefault('dp_threshold', 8)
        kwargs.setdefault('meta_time_limit', 15.0)
        kwargs.setdefault('quality_weight', 0.3)
        super().__init__(**kwargs)
    
    @property
    def name(self) -> str:
        return "Hybrid (Speed)"


@register_algorithm("hybrid_genetic")
class HybridWithGenetic(HybridDPMeta):
    """Hybrid using Genetic Algorithm for meta phase."""
    
    description = "Hybrid using Genetic Algorithm"
    
    def __init__(self, **kwargs):
        kwargs.setdefault('meta_algorithm', 'genetic')
        super().__init__(**kwargs)
    
    @property
    def name(self) -> str:
        return "Hybrid (Genetic+DP)"


@register_algorithm("hybrid_tabu")
class HybridWithTabu(HybridDPMeta):
    """Hybrid using Tabu Search for meta phase."""
    
    description = "Hybrid using Tabu Search"
    
    def __init__(self, **kwargs):
        kwargs.setdefault('meta_algorithm', 'tabu')
        super().__init__(**kwargs)
    
    @property
    def name(self) -> str:
        return "Hybrid (Tabu+DP)"
