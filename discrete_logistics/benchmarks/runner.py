"""
Benchmark Runner Module
======================

Provides tools for running systematic benchmarks
across multiple algorithms and problem instances.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
from datetime import datetime
import traceback

import sys
sys.path.insert(0, str(__file__).rsplit('benchmarks', 1)[0])

from core.problem import Problem, Solution
from algorithms.base import Algorithm


@dataclass
class BenchmarkResult:
    """
    Container for benchmark results of a single algorithm run.
    
    Attributes
    ----------
    algorithm_name : str
        Name of the algorithm
    problem_name : str
        Name of the problem instance
    objective : float
        Objective value achieved
    execution_time : float
        Time taken in seconds
    feasible : bool
        Whether solution is feasible
    solution : Optional[Solution]
        The solution object (if stored)
    metrics : Dict[str, float]
        Additional metrics
    error : Optional[str]
        Error message if failed
    """
    algorithm_name: str
    problem_name: str
    objective: float
    execution_time: float
    feasible: bool
    solution: Optional[Solution] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm': self.algorithm_name,
            'problem': self.problem_name,
            'objective': self.objective,
            'time': self.execution_time,
            'feasible': self.feasible,
            'metrics': self.metrics,
            'error': self.error,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary."""
        return cls(
            algorithm_name=data['algorithm'],
            problem_name=data['problem'],
            objective=data['objective'],
            execution_time=data['time'],
            feasible=data['feasible'],
            metrics=data.get('metrics', {}),
            error=data.get('error'),
            timestamp=data.get('timestamp', '')
        )


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark execution.
    
    Attributes
    ----------
    time_limit : float
        Maximum time per algorithm run
    num_runs : int
        Number of runs per algorithm/problem pair
    store_solutions : bool
        Whether to store full solutions
    parallel : bool
        Whether to run in parallel
    max_workers : int
        Maximum parallel workers
    seed : Optional[int]
        Random seed for reproducibility
    """
    time_limit: float = 60.0
    num_runs: int = 5
    store_solutions: bool = False
    parallel: bool = False
    max_workers: int = 4
    seed: Optional[int] = 42
    verbose: bool = True


class BenchmarkRunner:
    """
    Executes systematic benchmarks on multiple algorithms and instances.
    
    This class provides:
    - Automated execution across algorithm/instance combinations
    - Time-limited execution with timeout handling
    - Statistical aggregation of multiple runs
    - Progress tracking and logging
    - Result serialization
    
    Example
    -------
    >>> runner = BenchmarkRunner()
    >>> runner.add_algorithm('FFD', FirstFitDecreasing())
    >>> runner.add_algorithm('SA', SimulatedAnnealing())
    >>> runner.add_problems(test_instances)
    >>> results = runner.run()
    >>> runner.export_results('benchmark_results.json')
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark runner.
        
        Parameters
        ----------
        config : BenchmarkConfig, optional
            Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.algorithms: Dict[str, Algorithm] = {}
        self.problems: Dict[str, Problem] = {}
        self.results: List[BenchmarkResult] = []
        self._progress_callback: Optional[Callable] = None
    
    def add_algorithm(self, name: str, algorithm: Algorithm):
        """
        Add an algorithm to the benchmark.
        
        Parameters
        ----------
        name : str
            Unique identifier for the algorithm
        algorithm : Algorithm
            Algorithm instance to benchmark
        """
        self.algorithms[name] = algorithm
    
    def add_problem(self, name: str, problem: Problem):
        """
        Add a problem instance to the benchmark.
        
        Parameters
        ----------
        name : str
            Unique identifier for the problem
        problem : Problem
            Problem instance
        """
        self.problems[name] = problem
    
    def add_problems(self, problems: Dict[str, Problem]):
        """Add multiple problems at once."""
        self.problems.update(problems)
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def run(self) -> List[BenchmarkResult]:
        """
        Execute the benchmark suite.
        
        Returns
        -------
        List[BenchmarkResult]
            All benchmark results
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        self.results = []
        total_runs = len(self.algorithms) * len(self.problems) * self.config.num_runs
        current_run = 0
        
        for algo_name, algorithm in self.algorithms.items():
            for problem_name, problem in self.problems.items():
                for run_idx in range(self.config.num_runs):
                    current_run += 1
                    
                    if self._progress_callback:
                        self._progress_callback(
                            current_run, total_runs,
                            f"Running {algo_name} on {problem_name} (run {run_idx + 1})"
                        )
                    
                    if self.config.verbose:
                        print(f"[{current_run}/{total_runs}] {algo_name} on {problem_name} "
                              f"(run {run_idx + 1}/{self.config.num_runs})")
                    
                    result = self._run_single(algorithm, problem, algo_name, problem_name)
                    self.results.append(result)
        
        return self.results
    
    def _run_single(self, algorithm: Algorithm, problem: Problem,
                    algo_name: str, problem_name: str) -> BenchmarkResult:
        """Execute a single benchmark run."""
        try:
            # Time the execution
            start_time = time.perf_counter()
            
            if self.config.parallel:
                solution = self._run_with_timeout(algorithm, problem)
            else:
                solution = algorithm.solve(problem)
            
            execution_time = time.perf_counter() - start_time
            
            # Calculate metrics
            objective = self._calculate_objective(solution)
            feasible = self._check_feasibility(solution, problem)
            metrics = self._calculate_metrics(solution, problem)
            
            return BenchmarkResult(
                algorithm_name=algo_name,
                problem_name=problem_name,
                objective=objective,
                execution_time=execution_time,
                feasible=feasible,
                solution=solution if self.config.store_solutions else None,
                metrics=metrics
            )
            
        except TimeoutError:
            return BenchmarkResult(
                algorithm_name=algo_name,
                problem_name=problem_name,
                objective=float('inf'),
                execution_time=self.config.time_limit,
                feasible=False,
                error="Timeout exceeded"
            )
        except Exception as e:
            return BenchmarkResult(
                algorithm_name=algo_name,
                problem_name=problem_name,
                objective=float('inf'),
                execution_time=0,
                feasible=False,
                error=f"{type(e).__name__}: {str(e)}"
            )
    
    def _run_with_timeout(self, algorithm: Algorithm, problem: Problem) -> Solution:
        """Run algorithm with timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(algorithm.solve, problem)
            return future.result(timeout=self.config.time_limit)
    
    def _calculate_objective(self, solution: Solution) -> float:
        """Calculate objective value from solution."""
        if not solution or not solution.bins:
            return float('inf')
        
        bin_values = [sum(item.value for item in bin_obj.items) 
                     for bin_obj in solution.bins]
        
        return max(bin_values) - min(bin_values) if bin_values else float('inf')
    
    def _check_feasibility(self, solution: Solution, problem: Problem) -> bool:
        """Check if solution is feasible."""
        if not solution or not solution.bins:
            return False
        
        # Check all items assigned
        assigned_items = set()
        for bin_obj in solution.bins:
            for item in bin_obj.items:
                assigned_items.add(item.id)
        
        if len(assigned_items) != len(problem.items):
            return False
        
        # Check capacity constraints
        for bin_obj in solution.bins:
            total_weight = sum(item.weight for item in bin_obj.items)
            # Use the capacity from the bin object itself
            if total_weight > bin_obj.capacity:
                return False
        
        return True
    
    def _calculate_metrics(self, solution: Solution, problem: Problem) -> Dict[str, float]:
        """Calculate additional metrics."""
        if not solution or not solution.bins:
            return {}
        
        bin_weights = [sum(item.weight for item in bin_obj.items) 
                      for bin_obj in solution.bins]
        bin_values = [sum(item.value for item in bin_obj.items) 
                     for bin_obj in solution.bins]
        
        total_capacity = sum(problem.bin_capacities)
        total_weight = sum(bin_weights)
        
        return {
            'weight_utilization': total_weight / total_capacity if total_capacity > 0 else 0,
            'value_std': float(np.std(bin_values)),
            'value_range': max(bin_values) - min(bin_values) if bin_values else 0,
            'weight_balance': 1 - (np.std(bin_weights) / np.mean(bin_weights)) if np.mean(bin_weights) > 0 else 0,
            'avg_items_per_bin': len(problem.items) / len(solution.bins),
            'max_bin_weight': max(bin_weights) if bin_weights else 0,
            'min_bin_weight': min(bin_weights) if bin_weights else 0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of benchmark results.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics by algorithm
        """
        summary = {}
        
        for algo_name in self.algorithms:
            algo_results = [r for r in self.results if r.algorithm_name == algo_name]
            
            if not algo_results:
                continue
            
            objectives = [r.objective for r in algo_results if r.objective < float('inf')]
            times = [r.execution_time for r in algo_results]
            feasible_count = sum(1 for r in algo_results if r.feasible)
            
            summary[algo_name] = {
                'num_runs': len(algo_results),
                'feasible_rate': feasible_count / len(algo_results),
                'avg_objective': np.mean(objectives) if objectives else float('inf'),
                'std_objective': np.std(objectives) if objectives else 0,
                'best_objective': min(objectives) if objectives else float('inf'),
                'worst_objective': max(objectives) if objectives else float('inf'),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'total_time': sum(times)
            }
        
        return summary
    
    def get_comparison_table(self) -> str:
        """
        Generate a comparison table in markdown format.
        
        Returns
        -------
        str
            Markdown table
        """
        summary = self.get_summary()
        
        lines = [
            "| Algorithm | Avg Obj | Std Obj | Best | Feasible% | Avg Time |",
            "|-----------|---------|---------|------|-----------|----------|"
        ]
        
        for algo_name, stats in sorted(summary.items(), key=lambda x: x[1]['avg_objective']):
            lines.append(
                f"| {algo_name} | {stats['avg_objective']:.4f} | "
                f"{stats['std_objective']:.4f} | {stats['best_objective']:.4f} | "
                f"{stats['feasible_rate']*100:.1f}% | {stats['avg_time']:.3f}s |"
            )
        
        return '\n'.join(lines)
    
    def export_results(self, filepath: str):
        """
        Export results to JSON file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        """
        data = {
            'config': {
                'time_limit': self.config.time_limit,
                'num_runs': self.config.num_runs,
                'seed': self.config.seed
            },
            'algorithms': list(self.algorithms.keys()),
            'problems': list(self.problems.keys()),
            'results': [r.to_dict() for r in self.results],
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_results(cls, filepath: str) -> 'BenchmarkRunner':
        """
        Load results from JSON file.
        
        Parameters
        ----------
        filepath : str
            Input file path
            
        Returns
        -------
        BenchmarkRunner
            Runner with loaded results
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = BenchmarkConfig(
            time_limit=data['config']['time_limit'],
            num_runs=data['config']['num_runs'],
            seed=data['config'].get('seed')
        )
        
        runner = cls(config)
        runner.results = [BenchmarkResult.from_dict(r) for r in data['results']]
        
        return runner


class BenchmarkSuite:
    """
    Pre-configured benchmark suites for different scenarios.
    """
    
    @staticmethod
    def quick_test(algorithms: Dict[str, Algorithm], 
                   problems: Dict[str, Problem]) -> BenchmarkRunner:
        """Quick benchmark with minimal runs."""
        config = BenchmarkConfig(
            time_limit=10.0,
            num_runs=1,
            verbose=True
        )
        runner = BenchmarkRunner(config)
        
        for name, algo in algorithms.items():
            runner.add_algorithm(name, algo)
        runner.add_problems(problems)
        
        return runner
    
    @staticmethod
    def standard_benchmark(algorithms: Dict[str, Algorithm],
                          problems: Dict[str, Problem]) -> BenchmarkRunner:
        """Standard benchmark with statistical significance."""
        config = BenchmarkConfig(
            time_limit=60.0,
            num_runs=10,
            seed=42,
            verbose=True
        )
        runner = BenchmarkRunner(config)
        
        for name, algo in algorithms.items():
            runner.add_algorithm(name, algo)
        runner.add_problems(problems)
        
        return runner
    
    @staticmethod
    def comprehensive_benchmark(algorithms: Dict[str, Algorithm],
                               problems: Dict[str, Problem]) -> BenchmarkRunner:
        """Comprehensive benchmark with detailed analysis."""
        config = BenchmarkConfig(
            time_limit=120.0,
            num_runs=30,
            store_solutions=True,
            seed=42,
            verbose=True
        )
        runner = BenchmarkRunner(config)
        
        for name, algo in algorithms.items():
            runner.add_algorithm(name, algo)
        runner.add_problems(problems)
        
        return runner
