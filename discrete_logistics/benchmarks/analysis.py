"""
Statistical Analysis Module
===========================

Provides statistical analysis tools for benchmark results.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import json

import sys
sys.path.insert(0, str(__file__).rsplit('benchmarks', 1)[0])

from benchmarks.runner import BenchmarkResult, BenchmarkRunner


@dataclass
class StatisticalTest:
    """
    Result of a statistical test.
    
    Attributes
    ----------
    test_name : str
        Name of the statistical test
    statistic : float
        Test statistic value
    p_value : float
        P-value of the test
    significant : bool
        Whether result is significant at alpha level
    alpha : float
        Significance level
    conclusion : str
        Human-readable conclusion
    """
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    conclusion: str = ""


class StatisticalAnalyzer:
    """
    Performs statistical analysis on benchmark results.
    
    Provides:
    - Descriptive statistics
    - Hypothesis testing (t-test, Wilcoxon, ANOVA)
    - Effect size calculations
    - Confidence intervals
    """
    
    def __init__(self, results: List[BenchmarkResult]):
        """
        Initialize analyzer with benchmark results.
        
        Parameters
        ----------
        results : List[BenchmarkResult]
            Benchmark results to analyze
        """
        self.results = results
        self._group_by_algorithm()
    
    def _group_by_algorithm(self):
        """Group results by algorithm name."""
        self.by_algorithm: Dict[str, List[BenchmarkResult]] = {}
        
        for result in self.results:
            if result.algorithm_name not in self.by_algorithm:
                self.by_algorithm[result.algorithm_name] = []
            self.by_algorithm[result.algorithm_name].append(result)
    
    def descriptive_stats(self, algorithm: str) -> Dict[str, float]:
        """
        Calculate descriptive statistics for an algorithm.
        
        Parameters
        ----------
        algorithm : str
            Algorithm name
            
        Returns
        -------
        Dict[str, float]
            Descriptive statistics
        """
        results = self.by_algorithm.get(algorithm, [])
        
        if not results:
            return {}
        
        objectives = [r.objective for r in results if r.objective < float('inf')]
        times = [r.execution_time for r in results]
        
        if not objectives:
            return {'n': len(results), 'feasible_count': 0}
        
        return {
            'n': len(results),
            'feasible_count': sum(1 for r in results if r.feasible),
            'feasible_rate': sum(1 for r in results if r.feasible) / len(results),
            'obj_mean': float(np.mean(objectives)),
            'obj_std': float(np.std(objectives, ddof=1)),
            'obj_min': float(np.min(objectives)),
            'obj_max': float(np.max(objectives)),
            'obj_median': float(np.median(objectives)),
            'obj_q1': float(np.percentile(objectives, 25)),
            'obj_q3': float(np.percentile(objectives, 75)),
            'obj_iqr': float(np.percentile(objectives, 75) - np.percentile(objectives, 25)),
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times, ddof=1)),
            'time_min': float(np.min(times)),
            'time_max': float(np.max(times)),
        }
    
    def all_descriptive_stats(self) -> Dict[str, Dict[str, float]]:
        """Get descriptive statistics for all algorithms."""
        return {algo: self.descriptive_stats(algo) for algo in self.by_algorithm}
    
    def paired_t_test(self, algo1: str, algo2: str, alpha: float = 0.05) -> StatisticalTest:
        """
        Perform paired t-test between two algorithms.
        
        Parameters
        ----------
        algo1, algo2 : str
            Algorithm names to compare
        alpha : float
            Significance level
            
        Returns
        -------
        StatisticalTest
            Test results
        """
        results1 = self.by_algorithm.get(algo1, [])
        results2 = self.by_algorithm.get(algo2, [])
        
        # Match by problem instance
        paired_data = self._get_paired_data(results1, results2)
        
        if len(paired_data) < 2:
            return StatisticalTest(
                test_name="Paired t-test",
                statistic=0,
                p_value=1,
                significant=False,
                alpha=alpha,
                conclusion="Insufficient paired data"
            )
        
        x1 = [p[0] for p in paired_data]
        x2 = [p[1] for p in paired_data]
        
        statistic, p_value = stats.ttest_rel(x1, x2)
        significant = p_value < alpha
        
        if significant:
            winner = algo1 if np.mean(x1) < np.mean(x2) else algo2
            conclusion = f"{winner} performs significantly better (p={p_value:.4f})"
        else:
            conclusion = f"No significant difference between algorithms (p={p_value:.4f})"
        
        return StatisticalTest(
            test_name="Paired t-test",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=significant,
            alpha=alpha,
            conclusion=conclusion
        )
    
    def wilcoxon_test(self, algo1: str, algo2: str, alpha: float = 0.05) -> StatisticalTest:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
        
        Parameters
        ----------
        algo1, algo2 : str
            Algorithm names to compare
        alpha : float
            Significance level
            
        Returns
        -------
        StatisticalTest
            Test results
        """
        results1 = self.by_algorithm.get(algo1, [])
        results2 = self.by_algorithm.get(algo2, [])
        
        paired_data = self._get_paired_data(results1, results2)
        
        if len(paired_data) < 6:  # Wilcoxon needs minimum samples
            return StatisticalTest(
                test_name="Wilcoxon signed-rank",
                statistic=0,
                p_value=1,
                significant=False,
                alpha=alpha,
                conclusion="Insufficient data (n >= 6 required)"
            )
        
        x1 = [p[0] for p in paired_data]
        x2 = [p[1] for p in paired_data]
        
        try:
            statistic, p_value = stats.wilcoxon(x1, x2)
            significant = p_value < alpha
            
            if significant:
                winner = algo1 if np.median(x1) < np.median(x2) else algo2
                conclusion = f"{winner} performs significantly better (p={p_value:.4f})"
            else:
                conclusion = f"No significant difference (p={p_value:.4f})"
                
        except ValueError as e:
            return StatisticalTest(
                test_name="Wilcoxon signed-rank",
                statistic=0,
                p_value=1,
                significant=False,
                alpha=alpha,
                conclusion=f"Test failed: {str(e)}"
            )
        
        return StatisticalTest(
            test_name="Wilcoxon signed-rank",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=significant,
            alpha=alpha,
            conclusion=conclusion
        )
    
    def anova_test(self, algorithms: Optional[List[str]] = None, 
                   alpha: float = 0.05) -> StatisticalTest:
        """
        Perform one-way ANOVA across multiple algorithms.
        
        Parameters
        ----------
        algorithms : List[str], optional
            Algorithms to compare (all if None)
        alpha : float
            Significance level
            
        Returns
        -------
        StatisticalTest
            Test results
        """
        if algorithms is None:
            algorithms = list(self.by_algorithm.keys())
        
        groups = []
        for algo in algorithms:
            results = self.by_algorithm.get(algo, [])
            objectives = [r.objective for r in results if r.objective < float('inf')]
            if objectives:
                groups.append(objectives)
        
        if len(groups) < 2:
            return StatisticalTest(
                test_name="One-way ANOVA",
                statistic=0,
                p_value=1,
                significant=False,
                alpha=alpha,
                conclusion="Need at least 2 groups with data"
            )
        
        statistic, p_value = stats.f_oneway(*groups)
        significant = p_value < alpha
        
        if significant:
            conclusion = f"Significant difference among algorithms (p={p_value:.4f})"
        else:
            conclusion = f"No significant difference (p={p_value:.4f})"
        
        return StatisticalTest(
            test_name="One-way ANOVA",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=significant,
            alpha=alpha,
            conclusion=conclusion
        )
    
    def kruskal_wallis_test(self, algorithms: Optional[List[str]] = None,
                           alpha: float = 0.05) -> StatisticalTest:
        """
        Perform Kruskal-Wallis H-test (non-parametric alternative to ANOVA).
        
        Parameters
        ----------
        algorithms : List[str], optional
            Algorithms to compare
        alpha : float
            Significance level
            
        Returns
        -------
        StatisticalTest
            Test results
        """
        if algorithms is None:
            algorithms = list(self.by_algorithm.keys())
        
        groups = []
        for algo in algorithms:
            results = self.by_algorithm.get(algo, [])
            objectives = [r.objective for r in results if r.objective < float('inf')]
            if objectives:
                groups.append(objectives)
        
        if len(groups) < 2:
            return StatisticalTest(
                test_name="Kruskal-Wallis H",
                statistic=0,
                p_value=1,
                significant=False,
                alpha=alpha,
                conclusion="Need at least 2 groups with data"
            )
        
        statistic, p_value = stats.kruskal(*groups)
        significant = p_value < alpha
        
        if significant:
            conclusion = f"Significant difference among algorithms (p={p_value:.4f})"
        else:
            conclusion = f"No significant difference (p={p_value:.4f})"
        
        return StatisticalTest(
            test_name="Kruskal-Wallis H",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=significant,
            alpha=alpha,
            conclusion=conclusion
        )
    
    def effect_size_cohens_d(self, algo1: str, algo2: str) -> float:
        """
        Calculate Cohen's d effect size between two algorithms.
        
        Parameters
        ----------
        algo1, algo2 : str
            Algorithm names
            
        Returns
        -------
        float
            Cohen's d effect size
            - Small: d ≈ 0.2
            - Medium: d ≈ 0.5
            - Large: d ≈ 0.8
        """
        results1 = self.by_algorithm.get(algo1, [])
        results2 = self.by_algorithm.get(algo2, [])
        
        x1 = [r.objective for r in results1 if r.objective < float('inf')]
        x2 = [r.objective for r in results2 if r.objective < float('inf')]
        
        if not x1 or not x2:
            return 0.0
        
        mean_diff = np.mean(x1) - np.mean(x2)
        pooled_std = np.sqrt(
            ((len(x1) - 1) * np.var(x1, ddof=1) + (len(x2) - 1) * np.var(x2, ddof=1)) /
            (len(x1) + len(x2) - 2)
        )
        
        if pooled_std == 0:
            return 0.0
        
        return float(mean_diff / pooled_std)
    
    def confidence_interval(self, algorithm: str, 
                           confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean objective value.
        
        Parameters
        ----------
        algorithm : str
            Algorithm name
        confidence : float
            Confidence level (default 95%)
            
        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds of confidence interval
        """
        results = self.by_algorithm.get(algorithm, [])
        objectives = [r.objective for r in results if r.objective < float('inf')]
        
        if len(objectives) < 2:
            return (float('nan'), float('nan'))
        
        mean = np.mean(objectives)
        sem = stats.sem(objectives)
        
        # Calculate t-critical value
        df = len(objectives) - 1
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        
        margin = t_crit * sem
        return (float(mean - margin), float(mean + margin))
    
    def _get_paired_data(self, results1: List[BenchmarkResult], 
                         results2: List[BenchmarkResult]) -> List[Tuple[float, float]]:
        """Get paired objective values by matching problem instances."""
        # Create lookup by problem name
        lookup1 = {}
        for r in results1:
            if r.problem_name not in lookup1:
                lookup1[r.problem_name] = []
            lookup1[r.problem_name].append(r.objective)
        
        lookup2 = {}
        for r in results2:
            if r.problem_name not in lookup2:
                lookup2[r.problem_name] = []
            lookup2[r.problem_name].append(r.objective)
        
        # Match pairs
        paired = []
        common_problems = set(lookup1.keys()) & set(lookup2.keys())
        
        for problem in common_problems:
            objs1 = [o for o in lookup1[problem] if o < float('inf')]
            objs2 = [o for o in lookup2[problem] if o < float('inf')]
            
            if objs1 and objs2:
                # Use mean if multiple runs
                paired.append((np.mean(objs1), np.mean(objs2)))
        
        return paired
    
    def ranking(self) -> List[Tuple[str, float, int]]:
        """
        Rank algorithms by mean objective value.
        
        Returns
        -------
        List[Tuple[str, float, int]]
            List of (algorithm_name, mean_objective, rank)
        """
        stats_all = self.all_descriptive_stats()
        
        ranked = []
        for algo, stats_dict in stats_all.items():
            mean_obj = stats_dict.get('obj_mean', float('inf'))
            ranked.append((algo, mean_obj))
        
        # Sort by mean objective (lower is better)
        ranked.sort(key=lambda x: x[1])
        
        # Add ranks
        return [(algo, mean_obj, rank + 1) for rank, (algo, mean_obj) in enumerate(ranked)]


@dataclass
class PerformanceReport:
    """
    Comprehensive performance report for benchmark results.
    """
    
    descriptive_stats: Dict[str, Dict[str, float]]
    ranking: List[Tuple[str, float, int]]
    pairwise_tests: Dict[str, StatisticalTest]
    anova_test: StatisticalTest
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Benchmark Performance Report",
            "",
            "## Algorithm Ranking",
            "",
            "| Rank | Algorithm | Mean Objective |",
            "|------|-----------|----------------|"
        ]
        
        for algo, mean_obj, rank in self.ranking:
            lines.append(f"| {rank} | {algo} | {mean_obj:.4f} |")
        
        lines.extend([
            "",
            "## Descriptive Statistics",
            "",
            "| Algorithm | N | Mean | Std | Min | Max | Feasible% |",
            "|-----------|---|------|-----|-----|-----|-----------|"
        ])
        
        for algo, stats in self.descriptive_stats.items():
            lines.append(
                f"| {algo} | {stats.get('n', 0)} | "
                f"{stats.get('obj_mean', 0):.4f} | {stats.get('obj_std', 0):.4f} | "
                f"{stats.get('obj_min', 0):.4f} | {stats.get('obj_max', 0):.4f} | "
                f"{stats.get('feasible_rate', 0)*100:.1f}% |"
            )
        
        lines.extend([
            "",
            "## Statistical Tests",
            "",
            f"### ANOVA: {self.anova_test.conclusion}",
            f"- F-statistic: {self.anova_test.statistic:.4f}",
            f"- p-value: {self.anova_test.p_value:.4f}",
            "",
            "### Pairwise Comparisons",
            ""
        ])
        
        for pair, test in self.pairwise_tests.items():
            lines.append(f"- **{pair}**: {test.conclusion}")
        
        lines.extend([
            "",
            "## Effect Sizes (Cohen's d)",
            ""
        ])
        
        for pair, d in self.effect_sizes.items():
            magnitude = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
            lines.append(f"- {pair}: d = {d:.3f} ({magnitude})")
        
        lines.extend([
            "",
            "## Confidence Intervals (95%)",
            ""
        ])
        
        for algo, (lower, upper) in self.confidence_intervals.items():
            lines.append(f"- {algo}: [{lower:.4f}, {upper:.4f}]")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'descriptive_stats': self.descriptive_stats,
            'ranking': self.ranking,
            'pairwise_tests': {k: v.__dict__ for k, v in self.pairwise_tests.items()},
            'anova_test': self.anova_test.__dict__,
            'effect_sizes': self.effect_sizes,
            'confidence_intervals': self.confidence_intervals
        }
    
    @classmethod
    def generate(cls, runner: BenchmarkRunner) -> 'PerformanceReport':
        """
        Generate a complete performance report from benchmark results.
        
        Parameters
        ----------
        runner : BenchmarkRunner
            Completed benchmark runner
            
        Returns
        -------
        PerformanceReport
            Complete performance analysis
        """
        analyzer = StatisticalAnalyzer(runner.results)
        
        # Get all statistics
        desc_stats = analyzer.all_descriptive_stats()
        ranking = analyzer.ranking()
        
        # Pairwise tests
        algorithms = list(runner.algorithms.keys())
        pairwise_tests = {}
        effect_sizes = {}
        
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                pair_name = f"{algo1} vs {algo2}"
                pairwise_tests[pair_name] = analyzer.wilcoxon_test(algo1, algo2)
                effect_sizes[pair_name] = analyzer.effect_size_cohens_d(algo1, algo2)
        
        # ANOVA
        anova = analyzer.anova_test()
        
        # Confidence intervals
        conf_intervals = {
            algo: analyzer.confidence_interval(algo)
            for algo in algorithms
        }
        
        return cls(
            descriptive_stats=desc_stats,
            ranking=ranking,
            pairwise_tests=pairwise_tests,
            anova_test=anova,
            effect_sizes=effect_sizes,
            confidence_intervals=conf_intervals
        )
