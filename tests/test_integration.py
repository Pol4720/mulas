"""
Integration tests for the dashboard and end-to-end workflows.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from discrete_logistics.core.problem import Problem, Item
from discrete_logistics.core.instance_generator import InstanceGenerator
from discrete_logistics.algorithms import AlgorithmRegistry


class TestInstanceGenerator:
    """Tests for instance generation."""
    
    def test_generate_random_instance(self):
        """Test random instance generation."""
        generator = InstanceGenerator(seed=42)
        problem = generator.generate_uniform(
            n_items=10,
            num_bins=3
        )
        
        assert problem.n_items == 10
        assert problem.num_bins == 3
        assert all(item.weight > 0 for item in problem.items)
        assert all(item.value > 0 for item in problem.items)
    
    def test_generate_with_different_capacities(self):
        """Test generation with heterogeneous capacities."""
        generator = InstanceGenerator(seed=42)
        problem = generator.generate_uniform(
            n_items=8,
            num_bins=3,
            capacity_variation=0.3  # 30% variation in capacities
        )
        
        # With variation > 0, capacities should differ
        assert len(set(problem.bin_capacities)) > 1 or problem.num_bins == 1
    
    def test_reproducibility(self):
        """Test that same seed produces same instance."""
        generator1 = InstanceGenerator(seed=123)
        generator2 = InstanceGenerator(seed=123)
        
        p1 = generator1.generate_uniform(n_items=5, num_bins=2)
        p2 = generator2.generate_uniform(n_items=5, num_bins=2)
        
        for i1, i2 in zip(p1.items, p2.items):
            assert i1.weight == i2.weight
            assert i1.value == i2.value


class TestAlgorithmRegistry:
    """Tests for algorithm registry."""
    
    def test_registry_has_all_algorithms(self):
        """Test that all algorithms are registered."""
        algorithms = AlgorithmRegistry.list_algorithms()
        
        expected = [
            'first_fit_decreasing',
            'best_fit_decreasing',
            'worst_fit_decreasing',
            'round_robin_greedy',
            'largest_difference_first',
            'simulated_annealing',
            'genetic_algorithm',
            'tabu_search',
            'branch_and_bound',
            'dynamic_programming',
            'lpt_approximation',
            'multiway_partition'
        ]
        
        for name in expected:
            assert name in algorithms, f"Algorithm {name} not registered"
    
    def test_get_algorithm_by_name(self):
        """Test retrieving algorithm by name."""
        alg = AlgorithmRegistry.get('first_fit_decreasing')
        assert alg is not None
        assert alg.name == "First Fit Decreasing"


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_complete_workflow(self):
        """Test complete problem solving workflow."""
        # 1. Generate instance
        generator = InstanceGenerator(seed=42)
        problem = generator.generate_uniform(n_items=12, num_bins=3)
        
        # 2. Solve with multiple algorithms
        results = {}
        for alg_name in ['first_fit_decreasing', 'simulated_annealing', 'lpt_approximation']:
            alg = AlgorithmRegistry.get(alg_name)
            if hasattr(alg, 'max_iterations'):
                alg.max_iterations = 500  # Limit for testing
            solution = alg.solve(problem)
            results[alg_name] = solution
        
        # 3. Verify all solutions are valid
        for name, solution in results.items():
            # All bins should be within capacity (solution is valid)
            for bin in solution.bins:
                assert bin.current_weight <= bin.capacity
            assert solution.value_difference >= 0
            
            # Check all items assigned
            all_items = set()
            for bin in solution.bins:
                for item in bin.items:
                    all_items.add(item.id)
            
            assert len(all_items) == problem.n_items
        
        # 4. Verify execution time recorded
        for name, solution in results.items():
            assert solution.execution_time >= 0
    
    def test_benchmark_workflow(self):
        """Test running multiple instances for benchmarking."""
        # Generate multiple instances with different seeds
        instances = []
        for i in range(3):
            generator = InstanceGenerator(seed=i)
            instances.append(generator.generate_uniform(n_items=8, num_bins=2))
        
        # Solve each with greedy
        alg = AlgorithmRegistry.get('best_fit_decreasing')
        
        for problem in instances:
            solution = alg.solve(problem)
            assert solution.value_difference >= 0
