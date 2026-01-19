"""
Tests for Hybrid Algorithm and Statistical Framework.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from discrete_logistics.core.problem import Problem, Item
from discrete_logistics.algorithms.hybrid import (
    HybridDPMeta,
    HybridQualityFocused,
    HybridSpeedFocused,
    HybridWithGenetic,
    HybridWithTabu,
    HybridStats
)
from discrete_logistics.benchmarks.statistical_framework import (
    StatisticalTests,
    ExperimentRunner,
    ExperimentConfig,
    StatisticalResult
)


class TestHybridAlgorithm:
    """Test suite for HybridDPMeta algorithm."""
    
    @pytest.fixture
    def small_problem(self):
        """Create a small problem instance (n <= dp_threshold)."""
        items = [
            Item(id=i, weight=np.random.uniform(10, 50), value=np.random.uniform(5, 25))
            for i in range(10)
        ]
        return Problem(
            items=items,
            num_bins=2,
            bin_capacities=[200.0, 200.0],
            name="small_test"
        )
    
    @pytest.fixture
    def medium_problem(self):
        """Create a medium problem instance (n > dp_threshold)."""
        items = [
            Item(id=i, weight=np.random.uniform(10, 50), value=np.random.uniform(5, 25))
            for i in range(30)
        ]
        return Problem(
            items=items,
            num_bins=3,
            bin_capacities=[300.0, 300.0, 300.0],
            name="medium_test"
        )
    
    @pytest.fixture
    def large_problem(self):
        """Create a larger problem instance."""
        items = [
            Item(id=i, weight=np.random.uniform(10, 50), value=np.random.uniform(5, 25))
            for i in range(50)
        ]
        total_weight = sum(item.weight for item in items)
        capacity = total_weight / 4 * 1.2
        return Problem(
            items=items,
            num_bins=4,
            bin_capacities=[capacity] * 4,
            name="large_test"
        )
    
    def test_hybrid_solves_small_problem(self, small_problem):
        """Test hybrid algorithm on small problems (should use pure DP)."""
        hybrid = HybridDPMeta(dp_threshold=12, verbose=False)
        solution = hybrid.solve(small_problem)
        
        assert solution is not None
        assert solution.is_valid
        assert solution.value_difference >= 0
        
        # Check stats
        stats = hybrid.get_stats()
        assert stats is not None
        assert stats.strategy_used == 'pure_dp'
        assert stats.meta_items == 0
        assert stats.dp_items == small_problem.n_items
    
    def test_hybrid_solves_medium_problem(self, medium_problem):
        """Test hybrid algorithm on medium problems (should use hybrid approach)."""
        hybrid = HybridDPMeta(dp_threshold=10, meta_time_limit=10, verbose=False)
        solution = hybrid.solve(medium_problem)
        
        assert solution is not None
        assert solution.value_difference >= 0
        
        # Check stats
        stats = hybrid.get_stats()
        assert stats is not None
        assert stats.meta_items > 0
        assert stats.dp_items > 0
        assert stats.meta_items + stats.dp_items == medium_problem.n_items
    
    def test_hybrid_partition_strategies(self, medium_problem):
        """Test different partition strategies."""
        strategies = ['largest_first', 'value_first', 'balanced', 'adaptive']
        
        for strategy in strategies:
            hybrid = HybridDPMeta(
                dp_threshold=10,
                partition_strategy=strategy,
                meta_time_limit=5,
                verbose=False
            )
            solution = hybrid.solve(medium_problem)
            
            assert solution is not None
            stats = hybrid.get_stats()
            assert stats.strategy_used == strategy
    
    def test_hybrid_meta_algorithms(self, medium_problem):
        """Test different metaheuristic choices."""
        algorithms = ['simulated_annealing', 'genetic', 'tabu', 'auto']
        
        for algo in algorithms:
            hybrid = HybridDPMeta(
                dp_threshold=10,
                meta_algorithm=algo,
                meta_time_limit=5,
                verbose=False
            )
            solution = hybrid.solve(medium_problem)
            
            assert solution is not None
            assert solution.value_difference >= 0
    
    def test_hybrid_variants(self, medium_problem):
        """Test hybrid algorithm variants."""
        variants = [
            HybridQualityFocused(verbose=False),
            HybridSpeedFocused(verbose=False),
            HybridWithGenetic(verbose=False),
            HybridWithTabu(verbose=False)
        ]
        
        for variant in variants:
            solution = variant.solve(medium_problem)
            
            assert solution is not None
            assert variant.get_stats() is not None
    
    def test_hybrid_stats_structure(self, medium_problem):
        """Test that HybridStats contains all expected fields."""
        hybrid = HybridDPMeta(dp_threshold=10, meta_time_limit=5, verbose=False)
        solution = hybrid.solve(medium_problem)
        
        stats = hybrid.get_stats()
        
        assert hasattr(stats, 'total_items')
        assert hasattr(stats, 'meta_items')
        assert hasattr(stats, 'dp_items')
        assert hasattr(stats, 'meta_time')
        assert hasattr(stats, 'dp_time')
        assert hasattr(stats, 'total_time')
        assert hasattr(stats, 'meta_objective')
        assert hasattr(stats, 'dp_improvement')
        assert hasattr(stats, 'final_objective')
        assert hasattr(stats, 'strategy_used')
    
    def test_hybrid_time_limits(self, medium_problem):
        """Test that hybrid respects time limits."""
        hybrid = HybridDPMeta(
            dp_threshold=10,
            meta_time_limit=2,
            dp_time_limit=2,
            total_time_limit=5,
            verbose=False
        )
        
        import time
        start = time.time()
        solution = hybrid.solve(medium_problem)
        elapsed = time.time() - start
        
        # Should complete within reasonable time
        assert elapsed < 15  # generous bound considering overhead


class TestStatisticalTests:
    """Test suite for statistical testing framework."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return {
            'normal1': np.random.normal(50, 10, 30),
            'normal2': np.random.normal(55, 10, 30),
            'uniform1': np.random.uniform(30, 70, 30),
            'different': np.random.normal(100, 5, 30)
        }
    
    def test_normality_check(self, sample_data):
        """Test Shapiro-Wilk normality check."""
        is_normal, p_value = StatisticalTests.check_normality(sample_data['normal1'])
        
        assert isinstance(is_normal, (bool, np.bool_))
        assert 0 <= p_value <= 1
    
    def test_mann_whitney_u(self, sample_data):
        """Test Mann-Whitney U test."""
        result = StatisticalTests.mann_whitney_u(
            sample_data['normal1'],
            sample_data['normal2']
        )
        
        assert isinstance(result, StatisticalResult)
        assert result.test_name == "Mann-Whitney U Test"
        assert 0 <= result.p_value <= 1
        assert result.effect_size is not None
        assert result.effect_size_interpretation in ['negligible', 'small', 'medium', 'large']
    
    def test_mann_whitney_u_significant(self, sample_data):
        """Test Mann-Whitney detects significant differences."""
        result = StatisticalTests.mann_whitney_u(
            sample_data['normal1'],
            sample_data['different'],
            alpha=0.05
        )
        
        assert result.is_significant  # These distributions are very different
    
    def test_paired_t_test(self, sample_data):
        """Test paired t-test."""
        result = StatisticalTests.paired_t_test(
            sample_data['normal1'],
            sample_data['normal2']
        )
        
        assert isinstance(result, StatisticalResult)
        assert result.test_name == "Paired t-Test"
        assert 0 <= result.p_value <= 1
        assert result.effect_size is not None
        assert 'normality_p' in result.details
    
    def test_one_way_anova(self, sample_data):
        """Test one-way ANOVA."""
        result = StatisticalTests.one_way_anova(
            sample_data['normal1'],
            sample_data['normal2'],
            sample_data['different'],
            group_names=['Group A', 'Group B', 'Group C']
        )
        
        assert isinstance(result, StatisticalResult)
        assert result.test_name == "One-way ANOVA"
        assert 0 <= result.p_value <= 1
        assert result.effect_size is not None  # eta-squared
        assert 'group_means' in result.details
    
    def test_kruskal_wallis(self, sample_data):
        """Test Kruskal-Wallis H test."""
        result = StatisticalTests.kruskal_wallis(
            sample_data['normal1'],
            sample_data['normal2'],
            sample_data['different']
        )
        
        assert isinstance(result, StatisticalResult)
        assert result.test_name == "Kruskal-Wallis H Test"
        assert 0 <= result.p_value <= 1
    
    def test_tukey_hsd(self, sample_data):
        """Test Tukey HSD post-hoc test."""
        results = StatisticalTests.tukey_hsd(
            sample_data['normal1'],
            sample_data['normal2'],
            sample_data['different'],
            group_names=['A', 'B', 'C']
        )
        
        assert isinstance(results, dict)
        assert len(results) == 3  # 3 pairwise comparisons
        
        for key, result in results.items():
            assert isinstance(result, StatisticalResult)
            assert result.test_name == "Tukey HSD"
    
    def test_cliffs_delta(self):
        """Test Cliff's delta effect size calculation."""
        sample1 = np.array([1, 2, 3, 4, 5])
        sample2 = np.array([6, 7, 8, 9, 10])
        
        delta = StatisticalTests._cliffs_delta(sample1, sample2)
        
        assert -1 <= delta <= 1
        assert delta < 0  # sample1 is less than sample2
    
    def test_effect_size_interpretations(self):
        """Test effect size interpretation functions."""
        # Cohen's d
        assert StatisticalTests._interpret_cohens_d(0.1) == "negligible"
        assert StatisticalTests._interpret_cohens_d(0.3) == "small"
        assert StatisticalTests._interpret_cohens_d(0.6) == "medium"
        assert StatisticalTests._interpret_cohens_d(1.0) == "large"
        
        # Cliff's delta
        assert StatisticalTests._interpret_cliffs_delta(0.1) == "negligible"
        assert StatisticalTests._interpret_cliffs_delta(0.2) == "small"
        assert StatisticalTests._interpret_cliffs_delta(0.4) == "medium"
        assert StatisticalTests._interpret_cliffs_delta(0.6) == "large"


class TestExperimentRunner:
    """Test suite for experiment runner."""
    
    @pytest.fixture
    def simple_config(self):
        """Create simple experiment config for fast tests."""
        return ExperimentConfig(
            n_items_range=[15],
            k_bins_range=[2],
            distributions=['uniform'],
            capacity_variations=[0.0],
            repetitions=3
        )
    
    def test_experiment_runner_initialization(self, simple_config):
        """Test experiment runner can be initialized."""
        from discrete_logistics.algorithms.greedy import FirstFitDecreasing
        
        algorithms = [FirstFitDecreasing()]
        runner = ExperimentRunner(algorithms, simple_config, verbose=False)
        
        assert runner is not None
        assert len(runner.algorithms) == 1
    
    def test_experiment_runner_execution(self, simple_config):
        """Test experiment runner can execute experiments."""
        from discrete_logistics.algorithms.greedy import FirstFitDecreasing
        
        algorithms = [FirstFitDecreasing()]
        runner = ExperimentRunner(algorithms, simple_config, verbose=False)
        
        results = runner.run()
        
        assert len(results) == 3  # 3 repetitions
        for result in results:
            assert result.algorithm_name == "First Fit Decreasing"
            assert result.objective_value >= 0
            assert result.execution_time >= 0
    
    def test_experiment_runner_comparison(self, simple_config):
        """Test experiment runner can compare algorithms."""
        from discrete_logistics.algorithms.greedy import FirstFitDecreasing, BestFitDecreasing
        
        # Use a larger config for meaningful comparison
        larger_config = ExperimentConfig(
            n_items_range=[15],
            k_bins_range=[2],
            distributions=['uniform'],
            capacity_variations=[0.0],
            repetitions=10  # Need more samples for statistical tests
        )
        
        algorithms = [FirstFitDecreasing(), BestFitDecreasing()]
        runner = ExperimentRunner(algorithms, larger_config, verbose=False)
        
        results = runner.run()
        
        # Compare algorithms
        comparison = runner.compare_algorithms(
            "First Fit Decreasing",
            "Best Fit Decreasing"
        )
        
        assert 'mann_whitney' in comparison
    
    def test_experiment_runner_all_comparison(self, simple_config):
        """Test comprehensive comparison of all algorithms."""
        from discrete_logistics.algorithms.greedy import FirstFitDecreasing, BestFitDecreasing
        
        algorithms = [FirstFitDecreasing(), BestFitDecreasing()]
        runner = ExperimentRunner(algorithms, simple_config, verbose=False)
        
        results = runner.run()
        analysis = runner.compare_all_algorithms()
        
        assert 'algorithms' in analysis
        assert 'summary' in analysis
        assert 'pairwise' in analysis
    
    def test_experiment_runner_report(self, simple_config):
        """Test report generation."""
        from discrete_logistics.algorithms.greedy import FirstFitDecreasing
        
        algorithms = [FirstFitDecreasing()]
        runner = ExperimentRunner(algorithms, simple_config, verbose=False)
        
        results = runner.run()
        report = runner.generate_report()
        
        assert isinstance(report, str)
        assert "EXPERIMENTAL RESULTS REPORT" in report


class TestHybridVsBaseline:
    """Integration tests comparing hybrid to baseline algorithms."""
    
    def test_hybrid_improves_over_greedy(self):
        """Test that hybrid generally performs at least as well as greedy."""
        np.random.seed(42)
        
        # Create moderate problem
        items = [
            Item(id=i, weight=np.random.uniform(10, 50), value=np.random.uniform(5, 25))
            for i in range(25)
        ]
        
        total_weight = sum(item.weight for item in items)
        capacity = total_weight / 3 * 1.15
        
        problem = Problem(
            items=items,
            num_bins=3,
            bin_capacities=[capacity] * 3,
            name="hybrid_vs_greedy_test"
        )
        
        # Run hybrid
        hybrid = HybridDPMeta(dp_threshold=10, meta_time_limit=10, verbose=False)
        hybrid_solution = hybrid.solve(problem)
        
        # Run greedy
        from discrete_logistics.algorithms.greedy import FirstFitDecreasing
        greedy = FirstFitDecreasing()
        greedy_solution = greedy.solve(problem)
        
        # Hybrid should be competitive (within 50% worse at absolute worst)
        assert hybrid_solution.value_difference <= greedy_solution.value_difference * 1.5 + 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
