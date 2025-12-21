"""
Unit tests for all algorithms.

Tests verify:
1. Algorithm produces valid solutions (all items assigned, capacity respected)
2. Algorithm respects individual bin capacities
3. Algorithm improves or maintains solution quality
4. Algorithm handles edge cases
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from discrete_logistics.core.problem import Problem, Item
from discrete_logistics.algorithms.greedy import (
    FirstFitDecreasing,
    BestFitDecreasing,
    WorstFitDecreasing,
    RoundRobinGreedy,
    LargestDifferenceFirst
)
from discrete_logistics.algorithms.metaheuristics import (
    SimulatedAnnealing,
    GeneticAlgorithm,
    TabuSearch
)
from discrete_logistics.algorithms.branch_and_bound import BranchAndBound
from discrete_logistics.algorithms.dynamic_programming import DynamicProgramming
from discrete_logistics.algorithms.approximation import (
    LPTApproximation,
    MultiWayPartition
)


def validate_solution(solution, problem):
    """Helper to validate a solution."""
    # Check all items are assigned exactly once
    assigned_items = set()
    for bin in solution.bins:
        for item in bin.items:
            assert item.id not in assigned_items, f"Item {item.id} assigned multiple times"
            assigned_items.add(item.id)
    
    all_item_ids = {item.id for item in problem.items}
    assert assigned_items == all_item_ids, f"Not all items assigned: {all_item_ids - assigned_items}"
    
    # Check capacity constraints
    for i, bin in enumerate(solution.bins):
        capacity = problem.bin_capacities[i]
        assert bin.current_weight <= capacity + 1e-6, \
            f"Bin {i} exceeds capacity: {bin.current_weight} > {capacity}"
    
    # Check value difference is non-negative
    assert solution.value_difference >= 0
    
    return True


# ============================================================================
# Greedy Algorithm Tests
# ============================================================================

class TestFirstFitDecreasing:
    """Tests for First Fit Decreasing algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that FFD produces a valid solution."""
        alg = FirstFitDecreasing()
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)
    
    def test_heterogeneous_capacities(self, heterogeneous_capacity_problem):
        """Test FFD with different bin capacities."""
        alg = FirstFitDecreasing()
        solution = alg.solve(heterogeneous_capacity_problem)
        assert validate_solution(solution, heterogeneous_capacity_problem)
    
    def test_tight_capacity(self, tight_capacity_problem):
        """Test FFD with tight capacity constraints."""
        alg = FirstFitDecreasing()
        solution = alg.solve(tight_capacity_problem)
        assert validate_solution(solution, tight_capacity_problem)
    
    def test_single_bin(self, single_bin_problem):
        """Test FFD with single bin."""
        alg = FirstFitDecreasing()
        solution = alg.solve(single_bin_problem)
        assert validate_solution(solution, single_bin_problem)


class TestBestFitDecreasing:
    """Tests for Best Fit Decreasing algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that BFD produces a valid solution."""
        alg = BestFitDecreasing()
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)
    
    def test_heterogeneous_capacities(self, heterogeneous_capacity_problem):
        """Test BFD with different bin capacities."""
        alg = BestFitDecreasing()
        solution = alg.solve(heterogeneous_capacity_problem)
        assert validate_solution(solution, heterogeneous_capacity_problem)


class TestWorstFitDecreasing:
    """Tests for Worst Fit Decreasing algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that WFD produces a valid solution."""
        alg = WorstFitDecreasing()
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)
    
    def test_heterogeneous_capacities(self, heterogeneous_capacity_problem):
        """Test WFD with different bin capacities."""
        alg = WorstFitDecreasing()
        solution = alg.solve(heterogeneous_capacity_problem)
        assert validate_solution(solution, heterogeneous_capacity_problem)


class TestRoundRobinGreedy:
    """Tests for Round Robin Greedy algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that RR produces a valid solution."""
        alg = RoundRobinGreedy()
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)
    
    def test_heterogeneous_capacities(self, heterogeneous_capacity_problem):
        """Test RR with different bin capacities."""
        alg = RoundRobinGreedy()
        solution = alg.solve(heterogeneous_capacity_problem)
        assert validate_solution(solution, heterogeneous_capacity_problem)


class TestLargestDifferenceFirst:
    """Tests for Largest Difference First algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that LDF produces a valid solution."""
        alg = LargestDifferenceFirst()
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)


# ============================================================================
# Metaheuristic Tests
# ============================================================================

class TestSimulatedAnnealing:
    """Tests for Simulated Annealing algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that SA produces a valid solution."""
        alg = SimulatedAnnealing(max_iterations=500)
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)
    
    def test_heterogeneous_capacities(self, heterogeneous_capacity_problem):
        """Test SA with different bin capacities."""
        alg = SimulatedAnnealing(max_iterations=500)
        solution = alg.solve(heterogeneous_capacity_problem)
        assert validate_solution(solution, heterogeneous_capacity_problem)
    
    def test_improves_over_greedy(self, medium_problem):
        """Test that SA can improve over greedy solutions."""
        greedy = FirstFitDecreasing()
        greedy_sol = greedy.solve(medium_problem)
        
        sa = SimulatedAnnealing(max_iterations=2000)
        sa_sol = sa.solve(medium_problem)
        
        assert validate_solution(sa_sol, medium_problem)
        # SA should be no worse than greedy (usually better)
        assert sa_sol.value_difference <= greedy_sol.value_difference * 1.1


class TestGeneticAlgorithm:
    """Tests for Genetic Algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that GA produces a valid solution."""
        alg = GeneticAlgorithm(population_size=20, generations=50)
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)
    
    def test_heterogeneous_capacities(self, heterogeneous_capacity_problem):
        """Test GA with different bin capacities."""
        alg = GeneticAlgorithm(population_size=20, generations=50)
        solution = alg.solve(heterogeneous_capacity_problem)
        assert validate_solution(solution, heterogeneous_capacity_problem)


class TestTabuSearch:
    """Tests for Tabu Search algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that TS produces a valid solution."""
        alg = TabuSearch(max_iterations=200)
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)
    
    def test_heterogeneous_capacities(self, heterogeneous_capacity_problem):
        """Test TS with different bin capacities."""
        alg = TabuSearch(max_iterations=200)
        solution = alg.solve(heterogeneous_capacity_problem)
        assert validate_solution(solution, heterogeneous_capacity_problem)


# ============================================================================
# Exact Algorithm Tests
# ============================================================================

class TestBranchAndBound:
    """Tests for Branch and Bound algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that B&B produces a valid solution."""
        alg = BranchAndBound(max_iterations=10000, time_limit=10)
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)
    
    def test_heterogeneous_capacities(self, heterogeneous_capacity_problem):
        """Test B&B with different bin capacities."""
        alg = BranchAndBound(max_iterations=10000, time_limit=10)
        solution = alg.solve(heterogeneous_capacity_problem)
        assert validate_solution(solution, heterogeneous_capacity_problem)
    
    def test_optimal_simple(self, tight_capacity_problem):
        """Test that B&B finds optimal for simple problem."""
        alg = BranchAndBound(max_iterations=50000, time_limit=30)
        solution = alg.solve(tight_capacity_problem)
        assert validate_solution(solution, tight_capacity_problem)
        
        # For this specific problem, optimal is perfect balance
        assert solution.value_difference == 0


class TestDynamicProgramming:
    """Tests for Dynamic Programming algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that DP produces a valid solution."""
        alg = DynamicProgramming()
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)
    
    def test_heterogeneous_capacities(self, heterogeneous_capacity_problem):
        """Test DP with different bin capacities."""
        alg = DynamicProgramming()
        solution = alg.solve(heterogeneous_capacity_problem)
        assert validate_solution(solution, heterogeneous_capacity_problem)
    
    def test_small_instance(self, small_problem):
        """Test DP on a small instance."""
        alg = DynamicProgramming()
        solution = alg.solve(small_problem)
        assert validate_solution(solution, small_problem)
    
    def test_fallback_for_large_instance(self, medium_problem):
        """Test that DP falls back to greedy for large instances."""
        alg = DynamicProgramming(max_items=10)  # Force fallback
        solution = alg.solve(medium_problem)
        assert validate_solution(solution, medium_problem)


# ============================================================================
# Approximation Algorithm Tests
# ============================================================================

class TestLPTApproximation:
    """Tests for LPT Approximation algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that LPT produces a valid solution."""
        alg = LPTApproximation()
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)
    
    def test_heterogeneous_capacities(self, heterogeneous_capacity_problem):
        """Test LPT with different bin capacities."""
        alg = LPTApproximation()
        solution = alg.solve(heterogeneous_capacity_problem)
        assert validate_solution(solution, heterogeneous_capacity_problem)


class TestMultiWayPartition:
    """Tests for Multi-Way Partition algorithm."""
    
    def test_basic_solution(self, simple_problem):
        """Test that MWP produces a valid solution."""
        alg = MultiWayPartition()
        solution = alg.solve(simple_problem)
        assert validate_solution(solution, simple_problem)
    
    def test_heterogeneous_capacities(self, heterogeneous_capacity_problem):
        """Test MWP with different bin capacities."""
        alg = MultiWayPartition()
        solution = alg.solve(heterogeneous_capacity_problem)
        assert validate_solution(solution, heterogeneous_capacity_problem)


# ============================================================================
# Algorithm Comparison Tests
# ============================================================================

class TestAlgorithmComparison:
    """Tests comparing algorithm results."""
    
    def test_all_algorithms_produce_valid_solutions(self, simple_problem):
        """Test that all algorithms produce valid solutions."""
        algorithms = [
            FirstFitDecreasing(),
            BestFitDecreasing(),
            WorstFitDecreasing(),
            RoundRobinGreedy(),
            LargestDifferenceFirst(),
            SimulatedAnnealing(max_iterations=100),
            GeneticAlgorithm(population_size=10, generations=20),
            TabuSearch(max_iterations=100),
            BranchAndBound(max_iterations=1000, time_limit=5),
            DynamicProgramming(),
            LPTApproximation(),
            MultiWayPartition()
        ]
        
        for alg in algorithms:
            solution = alg.solve(simple_problem)
            assert validate_solution(solution, simple_problem), \
                f"Algorithm {alg.name} failed validation"
    
    def test_exact_algorithms_no_worse_than_greedy(self, small_problem):
        """Test that exact algorithms produce solutions no worse than greedy."""
        greedy = FirstFitDecreasing()
        greedy_sol = greedy.solve(small_problem)
        
        bb = BranchAndBound(max_iterations=50000, time_limit=30)
        bb_sol = bb.solve(small_problem)
        
        dp = DynamicProgramming()
        dp_sol = dp.solve(small_problem)
        
        assert bb_sol.value_difference <= greedy_sol.value_difference + 1e-6
        assert dp_sol.value_difference <= greedy_sol.value_difference + 1e-6


# ============================================================================
# Robustness and Edge Case Tests
# ============================================================================

class TestRobustness:
    """Robustness tests to catch edge cases and common bugs."""
    
    def test_string_item_ids(self):
        """Test all algorithms work with string item IDs (common bug source)."""
        items = [
            Item(id=f"item_{i}", weight=10 + i, value=50 + i * 10)
            for i in range(15)
        ]
        problem = Problem(
            items=items,
            num_bins=3,
            bin_capacities=[100.0, 100.0, 100.0]
        )
        
        # Test all algorithms with string IDs
        algorithms = [
            FirstFitDecreasing(),
            BestFitDecreasing(),
            WorstFitDecreasing(),
            RoundRobinGreedy(),
            LargestDifferenceFirst(),
            SimulatedAnnealing(max_iterations=50),
            GeneticAlgorithm(population_size=10, generations=10),
            TabuSearch(max_iterations=50),
        ]
        
        for alg in algorithms:
            try:
                solution = alg.solve(problem)
                assert validate_solution(solution, problem), \
                    f"Algorithm {alg.name} failed validation with string IDs"
            except Exception as e:
                pytest.fail(f"Algorithm {alg.name} crashed with string IDs: {e}")
    
    def test_large_instance(self):
        """Test algorithms on larger instances (20+ items)."""
        items = [
            Item(id=f"i{i:03d}", weight=5 + (i % 10), value=20 + (i * 3) % 50)
            for i in range(25)
        ]
        problem = Problem(
            items=items,
            num_bins=5,
            bin_capacities=[80.0] * 5
        )
        
        # Test heuristics on larger instance
        algorithms = [
            FirstFitDecreasing(),
            BestFitDecreasing(),
            LargestDifferenceFirst(),
            SimulatedAnnealing(max_iterations=100),
            TabuSearch(max_iterations=100),
        ]
        
        for alg in algorithms:
            solution = alg.solve(problem)
            assert validate_solution(solution, problem), \
                f"Algorithm {alg.name} failed on large instance"
    
    def test_all_same_value_items(self):
        """Test with items that all have the same value (edge case for balancing)."""
        items = [
            Item(id=f"equal_{i}", weight=10, value=100)
            for i in range(6)
        ]
        problem = Problem(
            items=items,
            num_bins=3,
            bin_capacities=[50.0] * 3
        )
        
        dp = DynamicProgramming()
        solution = dp.solve(problem)
        
        # Should achieve perfect balance (diff=0) with same-value items
        assert solution.value_difference == 0, \
            f"DP should find perfect balance with same-value items, got diff={solution.value_difference}"
    
    def test_unbalanceable_instance(self):
        """Test instance that cannot be perfectly balanced."""
        items = [
            Item(id="big", weight=10, value=1000),
            Item(id="small1", weight=5, value=1),
            Item(id="small2", weight=5, value=1),
        ]
        problem = Problem(
            items=items,
            num_bins=2,
            bin_capacities=[50.0, 50.0]
        )
        
        # All algorithms should still produce valid solutions
        algorithms = [
            FirstFitDecreasing(),
            TabuSearch(max_iterations=50),
            DynamicProgramming(),
        ]
        
        for alg in algorithms:
            solution = alg.solve(problem)
            assert validate_solution(solution, problem)
    
    def test_empty_bins_allowed(self):
        """Test that algorithms handle cases where some bins should be empty."""
        items = [
            Item(id="single", weight=10, value=100),
        ]
        problem = Problem(
            items=items,
            num_bins=3,
            bin_capacities=[50.0] * 3
        )
        
        algorithms = [
            FirstFitDecreasing(),
            DynamicProgramming(),
        ]
        
        for alg in algorithms:
            solution = alg.solve(problem)
            assert validate_solution(solution, problem)
            # Exactly one bin should have the item
            non_empty = sum(1 for b in solution.bins if b.items)
            assert non_empty == 1, f"Expected 1 non-empty bin, got {non_empty}"


class TestDashboardIntegration:
    """Tests for dashboard integration correctness."""
    
    def test_all_algorithms_available_in_dashboard(self):
        """Verify all algorithms in selector can be instantiated."""
        from discrete_logistics.dashboard.shared import create_algorithm_instance
        
        algorithm_names = [
            'FirstFitDecreasing',
            'BestFitDecreasing', 
            'WorstFitDecreasing',
            'RoundRobinGreedy',
            'LargestDifferenceFirst',
            'SimulatedAnnealing',
            'GeneticAlgorithm',
            'TabuSearch',
            'BranchAndBound',
            'DynamicProgramming',
        ]
        
        for name in algorithm_names:
            params = {}
            alg = create_algorithm_instance(name, params)
            assert alg is not None, f"Algorithm {name} not available in dashboard"
            assert hasattr(alg, 'solve'), f"Algorithm {name} missing solve method"
    
    def test_dashboard_algorithm_execution(self):
        """Test that all dashboard algorithms can execute on a problem."""
        from discrete_logistics.dashboard.shared import create_algorithm_instance
        
        items = [
            Item(id=f"test_{i}", weight=10, value=50 + i * 5)
            for i in range(8)
        ]
        problem = Problem(items=items, num_bins=3, bin_capacities=[50.0] * 3)
        
        algorithm_names = [
            'FirstFitDecreasing',
            'BestFitDecreasing',
            'LargestDifferenceFirst',
            'SimulatedAnnealing',
            'TabuSearch',
        ]
        
        for name in algorithm_names:
            params = {'max_iterations': 50} if name in ['SimulatedAnnealing', 'TabuSearch'] else {}
            alg = create_algorithm_instance(name, params)
            
            try:
                solution = alg.solve(problem)
                assert validate_solution(solution, problem), \
                    f"Dashboard algorithm {name} produced invalid solution"
            except Exception as e:
                pytest.fail(f"Dashboard algorithm {name} crashed: {e}")
