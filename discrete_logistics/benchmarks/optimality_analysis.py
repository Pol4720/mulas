"""
Optimality Analysis Module
==========================

Compares heuristic solutions against the optimal solution from brute force.
Calculates optimality gaps, success rates, and performance characteristics.

Reference:
    - Measuring solution quality: (heuristic - optimal) / optimal * 100%
    - Graham (1969) for LPT approximation analysis
    - Coffman et al. (1996) "Approximation algorithms for bin packing"
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.problem import Problem, Solution, Item
from core.instance_generator import InstanceGenerator
from algorithms.brute_force import BruteForce
from algorithms.greedy import (
    FirstFitDecreasing, BestFitDecreasing, WorstFitDecreasing,
    RoundRobinGreedy, LargestDifferenceFirst
)
from algorithms.metaheuristics import SimulatedAnnealing, GeneticAlgorithm, TabuSearch
from algorithms.approximation import LPTApproximation, MultiWayPartition
from algorithms.branch_and_bound import BranchAndBound
from algorithms.dynamic_programming import DynamicProgramming


@dataclass
class OptimalityResult:
    """
    Result of comparing a heuristic solution to the optimal.
    
    Attributes:
        algorithm_name: Name of the heuristic algorithm
        problem_name: Name of the problem instance
        heuristic_objective: Objective value from heuristic
        optimal_objective: Objective value from brute force
        gap: Optimality gap as percentage
        is_optimal: Whether heuristic found the optimal
        heuristic_time: Time for heuristic (seconds)
        brute_force_time: Time for brute force (seconds)
        speedup: How much faster is heuristic vs brute force
    """
    algorithm_name: str
    problem_name: str
    heuristic_objective: float
    optimal_objective: float
    gap: float  # percentage: (h - opt) / opt * 100
    is_optimal: bool
    heuristic_time: float
    brute_force_time: float
    speedup: float  # brute_force_time / heuristic_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm_name,
            'problem': self.problem_name,
            'heuristic_obj': self.heuristic_objective,
            'optimal_obj': self.optimal_objective,
            'gap_pct': self.gap,
            'is_optimal': self.is_optimal,
            'heuristic_time': self.heuristic_time,
            'bf_time': self.brute_force_time,
            'speedup': self.speedup
        }


@dataclass
class AlgorithmProfile:
    """
    Performance profile for an algorithm across multiple instances.
    
    Attributes:
        algorithm_name: Name of the algorithm
        results: List of OptimalityResult for each instance
        summary: Aggregated statistics
    """
    algorithm_name: str
    results: List[OptimalityResult] = field(default_factory=list)
    
    @property
    def summary(self) -> Dict[str, float]:
        """Calculate summary statistics."""
        if not self.results:
            return {}
        
        gaps = [r.gap for r in self.results if r.optimal_objective > 0]
        times = [r.heuristic_time for r in self.results]
        optimal_count = sum(1 for r in self.results if r.is_optimal)
        
        return {
            'n_instances': len(self.results),
            'optimal_found_count': optimal_count,
            'optimal_found_rate': optimal_count / len(self.results) * 100,
            'gap_mean': float(np.mean(gaps)) if gaps else 0,
            'gap_std': float(np.std(gaps)) if len(gaps) > 1 else 0,
            'gap_max': float(np.max(gaps)) if gaps else 0,
            'gap_min': float(np.min(gaps)) if gaps else 0,
            'gap_median': float(np.median(gaps)) if gaps else 0,
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times)) if len(times) > 1 else 0,
            'avg_speedup': float(np.mean([r.speedup for r in self.results]))
        }


class OptimalityAnalyzer:
    """
    Analyzes algorithm performance against optimal solutions.
    
    Uses brute force to establish optimal baselines, then compares
    heuristic solutions to measure quality gaps and success rates.
    """
    
    # Default algorithms to analyze
    DEFAULT_ALGORITHMS = {
        'FFD': FirstFitDecreasing,
        'BFD': BestFitDecreasing,
        'WFD': WorstFitDecreasing,
        'RoundRobin': RoundRobinGreedy,
        'LDF': LargestDifferenceFirst,
        'LPT': LPTApproximation,
        'SA': SimulatedAnnealing,
        'GA': GeneticAlgorithm,
        'TabuSearch': TabuSearch,
    }
    
    def __init__(
        self,
        algorithms: Optional[Dict[str, type]] = None,
        brute_force_timeout: float = 60.0,
        verbose: bool = False
    ):
        """
        Initialize the analyzer.
        
        Args:
            algorithms: Dict of algorithm name -> class (uses defaults if None)
            brute_force_timeout: Max time for brute force per instance
            verbose: Print progress information
        """
        self.algorithms = algorithms or self.DEFAULT_ALGORITHMS
        self.bf_timeout = brute_force_timeout
        self.verbose = verbose
        self.profiles: Dict[str, AlgorithmProfile] = {}
        self._optimal_cache: Dict[str, Tuple[float, float]] = {}  # problem_name -> (obj, time)
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[OptimalityAnalyzer] {msg}")
    
    def compute_optimal(self, problem: Problem) -> Tuple[float, float, bool]:
        """
        Compute optimal solution using brute force.
        
        Args:
            problem: Problem instance
            
        Returns:
            (optimal_objective, time, success)
        """
        # Check cache
        if problem.name in self._optimal_cache:
            obj, t = self._optimal_cache[problem.name]
            return obj, t, True
        
        self._log(f"Computing optimal for {problem.name} (n={problem.n_items}, k={problem.num_bins})")
        
        try:
            bf = BruteForce(time_limit=self.bf_timeout, verbose=False)
            start = time.time()
            solution = bf.solve(problem)
            elapsed = time.time() - start
            
            obj = solution.value_difference
            self._optimal_cache[problem.name] = (obj, elapsed)
            
            self._log(f"  Optimal: {obj:.4f} in {elapsed:.3f}s")
            return obj, elapsed, True
            
        except ValueError as e:
            # Instance too large
            self._log(f"  Brute force failed: {e}")
            return float('inf'), 0, False
        except Exception as e:
            self._log(f"  Error: {e}")
            return float('inf'), 0, False
    
    def analyze_algorithm(
        self,
        algorithm_name: str,
        algorithm_class: type,
        problem: Problem,
        optimal_obj: float,
        bf_time: float,
        **algo_params
    ) -> OptimalityResult:
        """
        Analyze a single algorithm on a single instance.
        
        Args:
            algorithm_name: Name of algorithm
            algorithm_class: Algorithm class
            problem: Problem instance
            optimal_obj: Optimal objective from brute force
            bf_time: Time taken by brute force
            **algo_params: Parameters for algorithm constructor
            
        Returns:
            OptimalityResult with comparison data
        """
        # Create algorithm instance
        algo = algorithm_class(**algo_params)
        
        # Solve
        start = time.time()
        solution = algo.solve(problem)
        elapsed = time.time() - start
        
        heuristic_obj = solution.value_difference
        
        # Calculate gap
        if optimal_obj > 0:
            gap = (heuristic_obj - optimal_obj) / optimal_obj * 100
        elif optimal_obj == 0:
            gap = heuristic_obj * 100  # Any non-zero is bad
        else:
            gap = 0  # Both are zero
        
        is_optimal = abs(heuristic_obj - optimal_obj) < 1e-6
        speedup = bf_time / elapsed if elapsed > 0 else float('inf')
        
        return OptimalityResult(
            algorithm_name=algorithm_name,
            problem_name=problem.name,
            heuristic_objective=heuristic_obj,
            optimal_objective=optimal_obj,
            gap=gap,
            is_optimal=is_optimal,
            heuristic_time=elapsed,
            brute_force_time=bf_time,
            speedup=speedup
        )
    
    def analyze_all(
        self,
        problems: List[Problem],
        algo_params: Optional[Dict[str, dict]] = None
    ) -> Dict[str, AlgorithmProfile]:
        """
        Analyze all algorithms on all problems.
        
        Args:
            problems: List of problem instances
            algo_params: Optional params per algorithm {algo_name: {param: value}}
            
        Returns:
            Dict of algorithm name -> AlgorithmProfile
        """
        algo_params = algo_params or {}
        
        # Initialize profiles
        self.profiles = {name: AlgorithmProfile(name) for name in self.algorithms}
        
        for i, problem in enumerate(problems):
            self._log(f"\nProblem {i+1}/{len(problems)}: {problem.name}")
            
            # Get optimal solution
            optimal_obj, bf_time, success = self.compute_optimal(problem)
            
            if not success:
                self._log(f"  Skipping (brute force failed)")
                continue
            
            # Test each algorithm
            for algo_name, algo_class in self.algorithms.items():
                params = algo_params.get(algo_name, {})
                
                # Set reasonable defaults for metaheuristics
                if algo_name == 'SA' and 'max_iterations' not in params:
                    params['max_iterations'] = 500
                elif algo_name == 'GA' and 'generations' not in params:
                    params['generations'] = 100
                elif algo_name == 'TabuSearch' and 'max_iterations' not in params:
                    params['max_iterations'] = 500
                
                try:
                    result = self.analyze_algorithm(
                        algo_name, algo_class, problem,
                        optimal_obj, bf_time, **params
                    )
                    self.profiles[algo_name].results.append(result)
                    
                    status = "✓ OPTIMAL" if result.is_optimal else f"gap={result.gap:.2f}%"
                    self._log(f"  {algo_name}: {result.heuristic_objective:.4f} ({status}) [{result.heuristic_time:.4f}s]")
                    
                except Exception as e:
                    self._log(f"  {algo_name}: ERROR - {e}")
        
        return self.profiles
    
    def get_summary_table(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all algorithms."""
        return {name: profile.summary for name, profile in self.profiles.items()}
    
    def get_best_algorithm_per_instance(self) -> Dict[str, str]:
        """Determine which algorithm was best for each instance."""
        instance_best = {}
        
        # Collect all results by problem
        by_problem: Dict[str, List[OptimalityResult]] = {}
        for profile in self.profiles.values():
            for result in profile.results:
                if result.problem_name not in by_problem:
                    by_problem[result.problem_name] = []
                by_problem[result.problem_name].append(result)
        
        # Find best for each problem
        for problem_name, results in by_problem.items():
            best = min(results, key=lambda r: r.heuristic_objective)
            instance_best[problem_name] = best.algorithm_name
        
        return instance_best
    
    def export_results(self, filepath: str):
        """Export results to JSON file."""
        data = {
            'summaries': self.get_summary_table(),
            'details': {
                name: [r.to_dict() for r in profile.results]
                for name, profile in self.profiles.items()
            },
            'optimal_cache': {k: {'obj': v[0], 'time': v[1]} 
                            for k, v in self._optimal_cache.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self):
        """Print a formatted summary of results."""
        print("\n" + "=" * 80)
        print("OPTIMALITY ANALYSIS SUMMARY")
        print("=" * 80)
        
        summaries = self.get_summary_table()
        
        # Header
        print(f"\n{'Algorithm':<15} {'Opt%':<8} {'AvgGap%':<10} {'MaxGap%':<10} {'AvgTime':<10} {'Speedup':<10}")
        print("-" * 70)
        
        # Sort by optimal rate descending
        sorted_algos = sorted(summaries.items(), 
                             key=lambda x: x[1].get('optimal_found_rate', 0), 
                             reverse=True)
        
        for algo_name, stats in sorted_algos:
            if not stats:
                continue
            print(f"{algo_name:<15} "
                  f"{stats.get('optimal_found_rate', 0):<8.1f} "
                  f"{stats.get('gap_mean', 0):<10.2f} "
                  f"{stats.get('gap_max', 0):<10.2f} "
                  f"{stats.get('time_mean', 0):<10.4f} "
                  f"{stats.get('avg_speedup', 0):<10.1f}")
        
        print("-" * 70)
        print(f"Total instances analyzed: {len(self._optimal_cache)}")
        print("=" * 80)


def run_standard_analysis(
    max_items: int = 10,
    num_bins_options: List[int] = [2, 3],
    verbose: bool = True
) -> OptimalityAnalyzer:
    """
    Run a standard optimality analysis.
    
    Args:
        max_items: Maximum items per instance
        num_bins_options: List of bin counts to test
        verbose: Print progress
        
    Returns:
        Configured OptimalityAnalyzer with results
    """
    # Generate test instances
    gen = InstanceGenerator(seed=42)
    problems = gen.generate_test_suite_for_brute_force(
        max_items=max_items,
        num_bins_options=num_bins_options
    )
    
    print(f"Generated {len(problems)} test instances")
    
    # Run analysis
    analyzer = OptimalityAnalyzer(verbose=verbose)
    analyzer.analyze_all(problems)
    
    # Print results
    analyzer.print_summary()
    
    return analyzer


if __name__ == "__main__":
    # Run standard analysis
    analyzer = run_standard_analysis(max_items=10, verbose=True)
    
    # Export results
    analyzer.export_results("optimality_analysis_results.json")
