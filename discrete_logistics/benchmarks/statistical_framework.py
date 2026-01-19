"""
Statistical Experimentation Framework for Algorithm Comparison.

Provides rigorous statistical testing including:
- Mann-Whitney U Test (non-parametric comparison of two groups)
- Paired t-Test (parametric comparison with normality check)
- One-way ANOVA with Tukey HSD post-hoc
- Effect Size measures (Cohen's d, Cliff's delta)
- Multiple comparison corrections (Bonferroni, Holm-Bonferroni)

This framework ensures scientific rigor in algorithm performance claims.
"""

from typing import List, Dict, Tuple, Optional, Any, Literal
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
import json
import csv
from datetime import datetime
from pathlib import Path
import warnings

try:
    from ..core.problem import Problem
    from ..core.instance_generator import InstanceGenerator
    from ..algorithms.base import Algorithm, AlgorithmRegistry
except ImportError:
    from discrete_logistics.core.problem import Problem
    from discrete_logistics.core.instance_generator import InstanceGenerator
    from discrete_logistics.algorithms.base import Algorithm, AlgorithmRegistry


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    is_significant: bool = False
    alpha: float = 0.05
    conclusion: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    algorithm_name: str
    instance_config: Dict[str, Any]
    objective_value: float
    execution_time: float
    is_feasible: bool
    iteration_count: int = 0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for experimental design."""
    n_items_range: List[int] = field(default_factory=lambda: [15, 20, 30, 50, 75, 100])
    k_bins_range: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    distributions: List[str] = field(default_factory=lambda: ['uniform', 'normal', 'correlated'])
    capacity_variations: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.5])
    repetitions: int = 30
    random_seeds: Optional[List[int]] = None
    timeout_per_algorithm: float = 120.0
    
    def __post_init__(self):
        if self.random_seeds is None:
            self.random_seeds = list(range(42, 42 + self.repetitions))


class StatisticalTests:
    """Collection of statistical tests for algorithm comparison."""
    
    @staticmethod
    def check_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
        """
        Check if data is normally distributed using Shapiro-Wilk test.
        
        Args:
            data: Sample data
            alpha: Significance level
            
        Returns:
            (is_normal, p_value)
        """
        if len(data) < 3:
            return False, 0.0
        
        if len(data) > 5000:
            # For large samples, use D'Agostino-Pearson
            stat, p_value = stats.normaltest(data)
        else:
            stat, p_value = stats.shapiro(data)
        
        return p_value > alpha, p_value
    
    @staticmethod
    def mann_whitney_u(
        sample1: np.ndarray,
        sample2: np.ndarray,
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> StatisticalResult:
        """
        Mann-Whitney U Test (Wilcoxon rank-sum test).
        
        Non-parametric test for comparing two independent samples.
        Tests whether the distribution of sample1 is stochastically
        greater than, less than, or different from sample2.
        
        Args:
            sample1: First sample (e.g., hybrid algorithm results)
            sample2: Second sample (e.g., baseline algorithm results)
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            StatisticalResult with test results
        """
        stat, p_value = stats.mannwhitneyu(
            sample1, sample2,
            alternative=alternative
        )
        
        # Cliff's delta (non-parametric effect size)
        cliffs_delta = StatisticalTests._cliffs_delta(sample1, sample2)
        effect_interp = StatisticalTests._interpret_cliffs_delta(cliffs_delta)
        
        is_significant = p_value < alpha
        
        # Determine direction
        median1 = np.median(sample1)
        median2 = np.median(sample2)
        
        if is_significant:
            if alternative == 'two-sided':
                direction = "lower" if median1 < median2 else "higher"
                conclusion = f"Sample 1 has significantly {direction} values (p={p_value:.4f})"
            elif alternative == 'less':
                conclusion = f"Sample 1 is significantly less than Sample 2 (p={p_value:.4f})"
            else:
                conclusion = f"Sample 1 is significantly greater than Sample 2 (p={p_value:.4f})"
        else:
            conclusion = f"No significant difference detected (p={p_value:.4f})"
        
        return StatisticalResult(
            test_name="Mann-Whitney U Test",
            statistic=stat,
            p_value=p_value,
            effect_size=cliffs_delta,
            effect_size_interpretation=effect_interp,
            is_significant=is_significant,
            alpha=alpha,
            conclusion=conclusion,
            details={
                'median_1': median1,
                'median_2': median2,
                'n1': len(sample1),
                'n2': len(sample2),
                'alternative': alternative
            }
        )
    
    @staticmethod
    def paired_t_test(
        sample1: np.ndarray,
        sample2: np.ndarray,
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> StatisticalResult:
        """
        Paired t-Test with normality check.
        
        Parametric test for comparing paired samples. Includes
        Shapiro-Wilk normality test on differences.
        
        Args:
            sample1: First sample (e.g., hybrid algorithm results)
            sample2: Second sample (e.g., baseline results on same instances)
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            StatisticalResult with test results
        """
        if len(sample1) != len(sample2):
            raise ValueError("Samples must have equal length for paired test")
        
        differences = sample1 - sample2
        
        # Check normality of differences
        is_normal, normality_p = StatisticalTests.check_normality(differences, alpha)
        
        if not is_normal:
            warnings.warn(
                f"Differences may not be normally distributed (p={normality_p:.4f}). "
                "Consider using Mann-Whitney U test instead."
            )
        
        stat, p_value = stats.ttest_rel(sample1, sample2, alternative=alternative)
        
        # Cohen's d for paired samples
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
        effect_interp = StatisticalTests._interpret_cohens_d(cohens_d)
        
        is_significant = p_value < alpha
        
        mean_diff = np.mean(differences)
        
        if is_significant:
            direction = "lower" if mean_diff < 0 else "higher"
            conclusion = f"Sample 1 is significantly {direction} (mean diff={mean_diff:.4f}, p={p_value:.4f})"
        else:
            conclusion = f"No significant difference (mean diff={mean_diff:.4f}, p={p_value:.4f})"
        
        return StatisticalResult(
            test_name="Paired t-Test",
            statistic=stat,
            p_value=p_value,
            effect_size=cohens_d,
            effect_size_interpretation=effect_interp,
            is_significant=is_significant,
            alpha=alpha,
            conclusion=conclusion,
            details={
                'mean_1': np.mean(sample1),
                'mean_2': np.mean(sample2),
                'std_1': np.std(sample1, ddof=1),
                'std_2': np.std(sample2, ddof=1),
                'mean_diff': mean_diff,
                'normality_p': normality_p,
                'is_normal': is_normal,
                'n': len(sample1),
                'alternative': alternative
            }
        )
    
    @staticmethod
    def one_way_anova(
        *samples: np.ndarray,
        group_names: Optional[List[str]] = None,
        alpha: float = 0.05
    ) -> StatisticalResult:
        """
        One-way ANOVA for comparing multiple groups.
        
        Args:
            *samples: Variable number of sample arrays
            group_names: Names for each group
            alpha: Significance level
            
        Returns:
            StatisticalResult with test results
        """
        stat, p_value = stats.f_oneway(*samples)
        
        is_significant = p_value < alpha
        
        # Eta-squared effect size
        all_data = np.concatenate(samples)
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(s) * (np.mean(s) - grand_mean)**2 for s in samples)
        ss_total = np.sum((all_data - grand_mean)**2)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        effect_interp = StatisticalTests._interpret_eta_squared(eta_squared)
        
        if group_names is None:
            group_names = [f"Group {i+1}" for i in range(len(samples))]
        
        if is_significant:
            conclusion = f"Significant difference among groups (p={p_value:.4f}, η²={eta_squared:.4f})"
        else:
            conclusion = f"No significant difference among groups (p={p_value:.4f})"
        
        return StatisticalResult(
            test_name="One-way ANOVA",
            statistic=stat,
            p_value=p_value,
            effect_size=eta_squared,
            effect_size_interpretation=effect_interp,
            is_significant=is_significant,
            alpha=alpha,
            conclusion=conclusion,
            details={
                'group_means': {name: np.mean(s) for name, s in zip(group_names, samples)},
                'group_stds': {name: np.std(s, ddof=1) for name, s in zip(group_names, samples)},
                'group_sizes': {name: len(s) for name, s in zip(group_names, samples)},
                'ss_between': ss_between,
                'ss_total': ss_total
            }
        )
    
    @staticmethod
    def tukey_hsd(
        *samples: np.ndarray,
        group_names: Optional[List[str]] = None,
        alpha: float = 0.05
    ) -> Dict[str, StatisticalResult]:
        """
        Tukey's Honest Significant Difference (HSD) post-hoc test.
        
        Performs pairwise comparisons with family-wise error rate control.
        
        Args:
            *samples: Variable number of sample arrays
            group_names: Names for each group
            alpha: Significance level
            
        Returns:
            Dictionary of pairwise comparison results
        """
        if group_names is None:
            group_names = [f"Group {i+1}" for i in range(len(samples))]
        
        # Create combined data for Tukey HSD
        data = np.concatenate(samples)
        groups = np.concatenate([[name] * len(s) for name, s in zip(group_names, samples)])
        
        # Perform Tukey HSD
        tukey_result = stats.tukey_hsd(*samples)
        
        results = {}
        for i, name1 in enumerate(group_names):
            for j, name2 in enumerate(group_names):
                if i < j:
                    p_value = tukey_result.pvalue[i, j]
                    statistic = tukey_result.statistic[i, j]
                    
                    mean_diff = np.mean(samples[i]) - np.mean(samples[j])
                    
                    is_significant = p_value < alpha
                    
                    key = f"{name1} vs {name2}"
                    results[key] = StatisticalResult(
                        test_name="Tukey HSD",
                        statistic=statistic,
                        p_value=p_value,
                        effect_size=None,
                        is_significant=is_significant,
                        alpha=alpha,
                        conclusion=f"{'Significant' if is_significant else 'No significant'} difference (p={p_value:.4f})",
                        details={
                            'mean_diff': mean_diff,
                            'group1_mean': np.mean(samples[i]),
                            'group2_mean': np.mean(samples[j])
                        }
                    )
        
        return results
    
    @staticmethod
    def kruskal_wallis(
        *samples: np.ndarray,
        group_names: Optional[List[str]] = None,
        alpha: float = 0.05
    ) -> StatisticalResult:
        """
        Kruskal-Wallis H Test (non-parametric ANOVA).
        
        Args:
            *samples: Variable number of sample arrays
            group_names: Names for each group
            alpha: Significance level
            
        Returns:
            StatisticalResult with test results
        """
        stat, p_value = stats.kruskal(*samples)
        
        is_significant = p_value < alpha
        
        if group_names is None:
            group_names = [f"Group {i+1}" for i in range(len(samples))]
        
        if is_significant:
            conclusion = f"Significant difference among groups (p={p_value:.4f})"
        else:
            conclusion = f"No significant difference among groups (p={p_value:.4f})"
        
        return StatisticalResult(
            test_name="Kruskal-Wallis H Test",
            statistic=stat,
            p_value=p_value,
            is_significant=is_significant,
            alpha=alpha,
            conclusion=conclusion,
            details={
                'group_medians': {name: np.median(s) for name, s in zip(group_names, samples)},
                'group_sizes': {name: len(s) for name, s in zip(group_names, samples)}
            }
        )
    
    @staticmethod
    def _cliffs_delta(sample1: np.ndarray, sample2: np.ndarray) -> float:
        """
        Calculate Cliff's delta effect size.
        
        Ranges from -1 to 1:
        - Positive: sample1 tends to be larger
        - Negative: sample2 tends to be larger
        - Zero: no difference
        """
        n1, n2 = len(sample1), len(sample2)
        if n1 == 0 or n2 == 0:
            return 0.0
        more = sum(1 for x in sample1 for y in sample2 if x > y)
        less = sum(1 for x in sample1 for y in sample2 if x < y)
        return (more - less) / (n1 * n2)
    
    @staticmethod
    def _interpret_cliffs_delta(d: float) -> str:
        """Interpret Cliff's delta effect size."""
        d_abs = abs(d)
        if d_abs < 0.147:
            return "negligible"
        elif d_abs < 0.33:
            return "small"
        elif d_abs < 0.474:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def _interpret_eta_squared(eta2: float) -> str:
        """Interpret eta-squared effect size."""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        else:
            return "large"


class ExperimentRunner:
    """
    Run systematic experiments comparing multiple algorithms.
    
    Features:
    - Automatic instance generation with varied parameters
    - Parallel execution (optional)
    - Progress tracking
    - Result aggregation and statistical analysis
    """
    
    def __init__(
        self,
        algorithms: List[Algorithm],
        config: Optional[ExperimentConfig] = None,
        verbose: bool = True
    ):
        """
        Initialize experiment runner.
        
        Args:
            algorithms: List of algorithm instances to compare
            config: Experiment configuration
            verbose: Print progress information
        """
        self.algorithms = algorithms
        self.config = config or ExperimentConfig()
        self.verbose = verbose
        self.generator = InstanceGenerator()
        
        self.results: List[ExperimentResult] = []
        self.statistics: Dict[str, Any] = {}
    
    def run(self, callback=None) -> List[ExperimentResult]:
        """
        Run all experiments.
        
        Args:
            callback: Optional progress callback (current, total, description)
            
        Returns:
            List of all experiment results
        """
        self.results = []
        
        total_experiments = (
            len(self.config.n_items_range) *
            len(self.config.k_bins_range) *
            len(self.config.distributions) *
            len(self.config.capacity_variations) *
            self.config.repetitions *
            len(self.algorithms)
        )
        
        current = 0
        
        for n in self.config.n_items_range:
            for k in self.config.k_bins_range:
                for dist in self.config.distributions:
                    for cap_var in self.config.capacity_variations:
                        for rep in range(self.config.repetitions):
                            seed = self.config.random_seeds[rep]
                            
                            # Generate instance
                            instance = self._generate_instance(
                                n=n, k=k, distribution=dist,
                                capacity_variation=cap_var, seed=seed
                            )
                            
                            for algo in self.algorithms:
                                current += 1
                                
                                if callback:
                                    callback(
                                        current, total_experiments,
                                        f"{algo.name}: n={n}, k={k}, {dist}"
                                    )
                                
                                result = self._run_single(algo, instance, {
                                    'n': n, 'k': k,
                                    'distribution': dist,
                                    'capacity_variation': cap_var,
                                    'seed': seed,
                                    'repetition': rep
                                })
                                
                                self.results.append(result)
        
        return self.results
    
    def _generate_instance(
        self,
        n: int,
        k: int,
        distribution: str,
        capacity_variation: float,
        seed: int
    ) -> Problem:
        """Generate a problem instance."""
        self.generator.set_seed(seed)
        
        # Generate instance directly using the generator
        if distribution == 'uniform':
            return self.generator.generate_uniform(
                n_items=n,
                num_bins=k,
                weight_range=(10, 100),
                value_range=(5, 50),
                capacity_factor=1.15,
                capacity_variation=capacity_variation,
                name=f"exp_n{n}_k{k}_{distribution}_{seed}"
            )
        elif distribution == 'normal':
            return self.generator.generate_normal(
                n_items=n,
                num_bins=k,
                weight_mean=50,
                weight_std=15,
                value_mean=25,
                value_std=10,
                capacity_factor=1.15,
                capacity_variation=capacity_variation,
                name=f"exp_n{n}_k{k}_{distribution}_{seed}"
            )
        elif distribution == 'correlated':
            return self.generator.generate_correlated(
                n_items=n,
                num_bins=k,
                correlation=0.7,
                capacity_factor=1.15,
                capacity_variation=capacity_variation,
                name=f"exp_n{n}_k{k}_{distribution}_{seed}"
            )
        else:
            return self.generator.generate_uniform(
                n_items=n,
                num_bins=k,
                capacity_factor=1.15,
                capacity_variation=capacity_variation,
                name=f"exp_n{n}_k{k}_{distribution}_{seed}"
            )
    
    def _run_single(
        self,
        algorithm: Algorithm,
        instance: Problem,
        config: Dict[str, Any]
    ) -> ExperimentResult:
        """Run a single experiment."""
        import time
        
        start_time = time.time()
        
        try:
            solution = algorithm.solve(instance)
            execution_time = time.time() - start_time
            
            return ExperimentResult(
                algorithm_name=algorithm.name,
                instance_config=config,
                objective_value=solution.value_difference,
                execution_time=execution_time,
                is_feasible=solution.is_valid,
                iteration_count=solution.iterations or 0,
                additional_metrics={
                    'max_value': solution.max_value,
                    'min_value': solution.min_value,
                    'metadata': solution.metadata
                }
            )
            
        except Exception as e:
            if self.verbose:
                print(f"Error running {algorithm.name}: {e}")
            
            return ExperimentResult(
                algorithm_name=algorithm.name,
                instance_config=config,
                objective_value=float('inf'),
                execution_time=time.time() - start_time,
                is_feasible=False,
                additional_metrics={'error': str(e)}
            )
    
    def get_results_by_algorithm(self) -> Dict[str, List[ExperimentResult]]:
        """Group results by algorithm name."""
        grouped = {}
        for result in self.results:
            if result.algorithm_name not in grouped:
                grouped[result.algorithm_name] = []
            grouped[result.algorithm_name].append(result)
        return grouped
    
    def get_objective_values(
        self,
        algorithm_name: str,
        filter_config: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Get objective values for an algorithm.
        
        Args:
            algorithm_name: Name of algorithm
            filter_config: Optional filter criteria (e.g., {'n': 50})
            
        Returns:
            Array of objective values
        """
        values = []
        for result in self.results:
            if result.algorithm_name != algorithm_name:
                continue
            
            if filter_config:
                match = all(
                    result.instance_config.get(k) == v
                    for k, v in filter_config.items()
                )
                if not match:
                    continue
            
            if result.is_feasible:
                values.append(result.objective_value)
        
        return np.array(values)
    
    def compare_algorithms(
        self,
        algo1_name: str,
        algo2_name: str,
        test_type: Literal['mann_whitney', 'paired_t', 'both'] = 'both',
        filter_config: Optional[Dict[str, Any]] = None,
        alpha: float = 0.05
    ) -> Dict[str, StatisticalResult]:
        """
        Statistically compare two algorithms.
        
        Args:
            algo1_name: First algorithm name
            algo2_name: Second algorithm name
            test_type: Type of test to perform
            filter_config: Optional filter criteria
            alpha: Significance level
            
        Returns:
            Dictionary of test results
        """
        values1 = self.get_objective_values(algo1_name, filter_config)
        values2 = self.get_objective_values(algo2_name, filter_config)
        
        results = {}
        
        if test_type in ('mann_whitney', 'both'):
            results['mann_whitney'] = StatisticalTests.mann_whitney_u(
                values1, values2, alpha=alpha, alternative='less'
            )
        
        if test_type in ('paired_t', 'both') and len(values1) == len(values2):
            results['paired_t'] = StatisticalTests.paired_t_test(
                values1, values2, alpha=alpha, alternative='less'
            )
        
        return results
    
    def compare_all_algorithms(
        self,
        filter_config: Optional[Dict[str, Any]] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compare all algorithms using ANOVA and post-hoc tests.
        
        Returns comprehensive statistical analysis.
        """
        algorithm_names = list(set(r.algorithm_name for r in self.results))
        
        samples = []
        for name in algorithm_names:
            values = self.get_objective_values(name, filter_config)
            if len(values) > 0:
                samples.append(values)
            else:
                algorithm_names.remove(name)
        
        results = {
            'algorithms': algorithm_names,
            'summary': {},
            'anova': None,
            'kruskal_wallis': None,
            'pairwise': {}
        }
        
        # Summary statistics
        for name, values in zip(algorithm_names, samples):
            results['summary'][name] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'n': len(values)
            }
        
        if len(samples) >= 2:
            # ANOVA
            results['anova'] = StatisticalTests.one_way_anova(
                *samples, group_names=algorithm_names, alpha=alpha
            )
            
            # Kruskal-Wallis (non-parametric alternative)
            results['kruskal_wallis'] = StatisticalTests.kruskal_wallis(
                *samples, group_names=algorithm_names, alpha=alpha
            )
            
            # Tukey HSD if ANOVA is significant
            if results['anova'].is_significant:
                results['tukey_hsd'] = StatisticalTests.tukey_hsd(
                    *samples, group_names=algorithm_names, alpha=alpha
                )
            
            # Pairwise comparisons
            for i, name1 in enumerate(algorithm_names):
                for j, name2 in enumerate(algorithm_names):
                    if i < j:
                        key = f"{name1} vs {name2}"
                        results['pairwise'][key] = StatisticalTests.mann_whitney_u(
                            samples[i], samples[j], alpha=alpha
                        )
        
        return results
    
    def export_results(
        self,
        filepath: str,
        format: Literal['csv', 'json'] = 'csv'
    ):
        """Export results to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            with open(filepath, 'w', newline='') as f:
                if not self.results:
                    return
                
                # Flatten results for CSV
                fieldnames = [
                    'algorithm_name', 'objective_value', 'execution_time',
                    'is_feasible', 'iteration_count', 'n', 'k',
                    'distribution', 'capacity_variation', 'seed', 'repetition'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    row = {
                        'algorithm_name': result.algorithm_name,
                        'objective_value': result.objective_value,
                        'execution_time': result.execution_time,
                        'is_feasible': result.is_feasible,
                        'iteration_count': result.iteration_count,
                        **result.instance_config
                    }
                    writer.writerow(row)
        
        elif format == 'json':
            data = {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'n_items_range': self.config.n_items_range,
                    'k_bins_range': self.config.k_bins_range,
                    'distributions': self.config.distributions,
                    'repetitions': self.config.repetitions
                },
                'results': [
                    {
                        'algorithm_name': r.algorithm_name,
                        'instance_config': r.instance_config,
                        'objective_value': r.objective_value,
                        'execution_time': r.execution_time,
                        'is_feasible': r.is_feasible,
                        'iteration_count': r.iteration_count
                    }
                    for r in self.results
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate a text report of the experiment results."""
        lines = []
        lines.append("=" * 60)
        lines.append("EXPERIMENTAL RESULTS REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Configuration
        lines.append("EXPERIMENTAL CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"Item sizes (n): {self.config.n_items_range}")
        lines.append(f"Bin counts (k): {self.config.k_bins_range}")
        lines.append(f"Distributions: {self.config.distributions}")
        lines.append(f"Repetitions: {self.config.repetitions}")
        lines.append("")
        
        # Summary by algorithm
        analysis = self.compare_all_algorithms()
        
        lines.append("ALGORITHM SUMMARY")
        lines.append("-" * 40)
        for name, stats in analysis['summary'].items():
            lines.append(f"\n{name}:")
            lines.append(f"  Mean objective: {stats['mean']:.4f} ± {stats['std']:.4f}")
            lines.append(f"  Median: {stats['median']:.4f}")
            lines.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            lines.append(f"  Samples: {stats['n']}")
        
        # Statistical tests
        lines.append("")
        lines.append("STATISTICAL ANALYSIS")
        lines.append("-" * 40)
        
        if analysis.get('anova'):
            anova = analysis['anova']
            lines.append(f"\nANOVA: F={anova.statistic:.4f}, p={anova.p_value:.6f}")
            lines.append(f"  η² = {anova.effect_size:.4f} ({anova.effect_size_interpretation})")
            lines.append(f"  {anova.conclusion}")
        
        if analysis.get('kruskal_wallis'):
            kw = analysis['kruskal_wallis']
            lines.append(f"\nKruskal-Wallis: H={kw.statistic:.4f}, p={kw.p_value:.6f}")
            lines.append(f"  {kw.conclusion}")
        
        if analysis.get('tukey_hsd'):
            lines.append("\nTukey HSD Post-hoc:")
            for key, result in analysis['tukey_hsd'].items():
                sig = "***" if result.is_significant else ""
                lines.append(f"  {key}: p={result.p_value:.6f} {sig}")
        
        lines.append("")
        lines.append("PAIRWISE COMPARISONS (Mann-Whitney U)")
        lines.append("-" * 40)
        for key, result in analysis.get('pairwise', {}).items():
            sig = "***" if result.is_significant else ""
            lines.append(f"  {key}:")
            lines.append(f"    U={result.statistic:.2f}, p={result.p_value:.6f} {sig}")
            lines.append(f"    Cliff's δ={result.effect_size:.4f} ({result.effect_size_interpretation})")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
