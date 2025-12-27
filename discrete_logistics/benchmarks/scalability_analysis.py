"""
Brute Force Scalability Analysis
================================

Determines the practical limits of brute force algorithm by measuring
execution time as problem size increases.

Key Questions:
    1. What is the maximum n solvable in 1s, 10s, 60s?
    2. How does k (number of bins) affect feasibility?
    3. Time complexity verification: O(k^n * n)
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.problem import Problem, Item
from core.instance_generator import InstanceGenerator
from algorithms.brute_force import BruteForce


@dataclass
class ScalabilityResult:
    """Result from testing a single (n, k) configuration."""
    n_items: int
    num_bins: int
    time_seconds: float
    success: bool
    optimal_diff: float = 0.0
    assignments_enumerated: int = 0
    
    @property
    def search_space_size(self) -> int:
        """Theoretical search space: k^n"""
        return self.num_bins ** self.n_items


@dataclass
class ScalabilityProfile:
    """Complete scalability analysis results."""
    results: List[ScalabilityResult] = field(default_factory=list)
    
    def max_n_for_time(self, k: int, max_time: float) -> Optional[int]:
        """Find maximum n solvable within time limit for given k."""
        relevant = [r for r in self.results 
                   if r.num_bins == k and r.success and r.time_seconds <= max_time]
        if not relevant:
            return None
        return max(r.n_items for r in relevant)
    
    def time_for_config(self, n: int, k: int) -> Optional[float]:
        """Get time for specific (n, k) configuration."""
        for r in self.results:
            if r.n_items == n and r.num_bins == k:
                return r.time_seconds
        return None
    
    def to_dict(self) -> Dict:
        return {
            'results': [
                {
                    'n': r.n_items,
                    'k': r.num_bins,
                    'time': r.time_seconds,
                    'success': r.success,
                    'opt_diff': r.optimal_diff,
                    'search_space': r.search_space_size
                }
                for r in self.results
            ],
            'limits': {
                f'k={k}': {
                    '1s': self.max_n_for_time(k, 1.0),
                    '10s': self.max_n_for_time(k, 10.0),
                    '60s': self.max_n_for_time(k, 60.0)
                }
                for k in sorted(set(r.num_bins for r in self.results))
            }
        }


class ScalabilityAnalyzer:
    """
    Analyzes brute force scalability.
    
    Tests progressively larger instances to find practical limits.
    """
    
    def __init__(
        self,
        time_limit_per_test: float = 120.0,
        seed: int = 42,
        verbose: bool = True
    ):
        self.time_limit = time_limit_per_test
        self.seed = seed
        self.verbose = verbose
        self.generator = InstanceGenerator(seed=seed)
        self.profile = ScalabilityProfile()
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def test_config(self, n: int, k: int, capacity: int = 1000) -> ScalabilityResult:
        """
        Test brute force on a single (n, k) configuration.
        
        Args:
            n: Number of items
            k: Number of bins
            capacity: Bin capacity
            
        Returns:
            ScalabilityResult with timing and success info
        """
        # Generate instance
        rng = np.random.default_rng(self.seed + n * 100 + k)
        values = rng.integers(1, capacity // 2, size=n).tolist()
        weights = rng.integers(1, capacity // 4, size=n).tolist()
        
        items = [
            Item(id=i, value=float(values[i]), weight=float(weights[i]))
            for i in range(n)
        ]
        
        problem = Problem(
            items=items,
            num_bins=k,
            bin_capacities=[float(capacity)] * k,
            name=f"scale_n{n}_k{k}"
        )
        
        # Run brute force with time limit
        bf = BruteForce(time_limit=self.time_limit, verbose=False)
        
        try:
            start = time.time()
            solution = bf.solve(problem)
            elapsed = time.time() - start
            
            return ScalabilityResult(
                n_items=n,
                num_bins=k,
                time_seconds=elapsed,
                success=True,
                optimal_diff=solution.value_difference,
                assignments_enumerated=bf._assignments_enumerated if hasattr(bf, '_assignments_enumerated') else 0
            )
            
        except ValueError as e:
            # Instance too large (rejected before running)
            return ScalabilityResult(
                n_items=n,
                num_bins=k,
                time_seconds=self.time_limit,
                success=False
            )
        except Exception as e:
            self._log(f"  Error: {e}")
            return ScalabilityResult(
                n_items=n,
                num_bins=k,
                time_seconds=self.time_limit,
                success=False
            )
    
    def run_scalability_analysis(
        self,
        n_range: Tuple[int, int] = (4, 20),
        k_values: List[int] = [2, 3, 4, 5],
        early_stop_time: float = 60.0
    ) -> ScalabilityProfile:
        """
        Run full scalability analysis.
        
        Args:
            n_range: (min_n, max_n) range to test
            k_values: List of bin counts to test
            early_stop_time: Stop increasing n if time exceeds this
            
        Returns:
            ScalabilityProfile with all results
        """
        self._log("=" * 60)
        self._log("BRUTE FORCE SCALABILITY ANALYSIS")
        self._log("=" * 60)
        self._log(f"Time limit per test: {self.time_limit}s")
        self._log(f"Early stop threshold: {early_stop_time}s")
        self._log("")
        
        self.profile = ScalabilityProfile()
        
        for k in k_values:
            self._log(f"\n{'='*40}")
            self._log(f"Testing k={k} bins")
            self._log(f"{'='*40}")
            
            stop_n = False
            
            for n in range(n_range[0], n_range[1] + 1):
                if stop_n:
                    break
                    
                search_space = k ** n
                self._log(f"\n  n={n}: Search space = {k}^{n} = {search_space:,}")
                
                result = self.test_config(n, k)
                self.profile.results.append(result)
                
                if result.success:
                    self._log(f"    Time: {result.time_seconds:.4f}s")
                    self._log(f"    Optimal diff: {result.optimal_diff:.4f}")
                    
                    if result.time_seconds > early_stop_time:
                        self._log(f"    EARLY STOP: Time > {early_stop_time}s")
                        stop_n = True
                else:
                    self._log(f"    FAILED (timeout or too large)")
                    stop_n = True
        
        return self.profile
    
    def print_summary(self):
        """Print summary of scalability limits."""
        self._log("\n" + "=" * 60)
        self._log("SCALABILITY LIMITS SUMMARY")
        self._log("=" * 60)
        
        k_values = sorted(set(r.num_bins for r in self.profile.results))
        
        self._log(f"\n{'k bins':<10} {'Max n (1s)':<12} {'Max n (10s)':<12} {'Max n (60s)':<12}")
        self._log("-" * 50)
        
        for k in k_values:
            n_1s = self.profile.max_n_for_time(k, 1.0)
            n_10s = self.profile.max_n_for_time(k, 10.0)
            n_60s = self.profile.max_n_for_time(k, 60.0)
            
            self._log(f"k={k:<7} {str(n_1s):<12} {str(n_10s):<12} {str(n_60s):<12}")
        
        self._log("-" * 50)
        self._log("\nTime Complexity: O(k^n * n)")
        self._log("Memory Complexity: O(n + k)")
        
        # Time progression analysis
        self._log("\n" + "-" * 50)
        self._log("TIME PROGRESSION (k=2)")
        self._log("-" * 50)
        
        k2_results = [r for r in self.profile.results if r.num_bins == 2 and r.success]
        k2_results.sort(key=lambda r: r.n_items)
        
        for r in k2_results:
            ratio = ""
            if r.n_items > 4:
                prev = next((x for x in k2_results if x.n_items == r.n_items - 1), None)
                if prev and prev.time_seconds > 0:
                    ratio = f"(×{r.time_seconds/prev.time_seconds:.1f})"
            self._log(f"  n={r.n_items:2}: {r.time_seconds:8.4f}s  {ratio}")
        
        self._log("\nExpected ratio for k=2: ~2x per n increase")
    
    def export_results(self, filepath: str):
        """Export results to JSON."""
        data = self.profile.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        self._log(f"\nResults exported to {filepath}")


def run_quick_analysis(verbose: bool = True) -> ScalabilityProfile:
    """Run a quick scalability analysis."""
    analyzer = ScalabilityAnalyzer(
        time_limit_per_test=30.0,
        verbose=verbose
    )
    
    profile = analyzer.run_scalability_analysis(
        n_range=(4, 16),
        k_values=[2, 3, 4],
        early_stop_time=15.0
    )
    
    analyzer.print_summary()
    return profile


def run_full_analysis(verbose: bool = True) -> ScalabilityProfile:
    """Run comprehensive scalability analysis."""
    analyzer = ScalabilityAnalyzer(
        time_limit_per_test=120.0,
        verbose=verbose
    )
    
    profile = analyzer.run_scalability_analysis(
        n_range=(4, 22),
        k_values=[2, 3, 4, 5],
        early_stop_time=60.0
    )
    
    analyzer.print_summary()
    analyzer.export_results("brute_force_scalability.json")
    
    return profile


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_analysis()
    else:
        run_quick_analysis()
