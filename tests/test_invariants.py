"""
Invariant Tests Module
======================

Rigorous tests for verifying algorithm and solution invariants.
Based on the Correctness Review Plan.
"""

import pytest
import numpy as np
from typing import List, Set
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from discrete_logistics.core.problem import Problem, Item, Bin, Solution
from discrete_logistics.core.instance_generator import InstanceGenerator
from discrete_logistics.algorithms.greedy import (
    FirstFitDecreasing,
    BestFitDecreasing,
    WorstFitDecreasing,
    RoundRobinGreedy
)
from discrete_logistics.algorithms.metaheuristics import (
    SimulatedAnnealing,
    GeneticAlgorithm,
    TabuSearch
)


# ============================================================================
# Invariant Helper Functions
# ============================================================================

def check_item_invariants(item: Item) -> bool:
    """
    Check all item invariants.
    
    INV-ITEM-1: Weight non-negative
    INV-ITEM-2: Value non-negative
    """
    assert item.weight >= 0, f"INV-ITEM-1 violated: item.weight = {item.weight} < 0"
    assert item.value >= 0, f"INV-ITEM-2 violated: item.value = {item.value} < 0"
    return True


def check_bin_invariants(bin_obj: Bin, tolerance: float = 1e-6) -> bool:
    """
    Check all bin invariants.
    
    INV-BIN-1: Capacity positive
    INV-BIN-2: Current weight does not exceed capacity
    INV-BIN-3: Weight consistency
    INV-BIN-4: Value consistency
    """
    # INV-BIN-1
    assert bin_obj.capacity > 0, f"INV-BIN-1 violated: capacity = {bin_obj.capacity} <= 0"
    
    # INV-BIN-2
    assert bin_obj.current_weight <= bin_obj.capacity + tolerance, \
        f"INV-BIN-2 violated: current_weight = {bin_obj.current_weight} > capacity = {bin_obj.capacity}"
    
    # INV-BIN-3
    expected_weight = sum(item.weight for item in bin_obj.items)
    assert abs(bin_obj.current_weight - expected_weight) < tolerance, \
        f"INV-BIN-3 violated: current_weight = {bin_obj.current_weight} != sum(weights) = {expected_weight}"
    
    # INV-BIN-4
    expected_value = sum(item.value for item in bin_obj.items)
    assert abs(bin_obj.current_value - expected_value) < tolerance, \
        f"INV-BIN-4 violated: current_value = {bin_obj.current_value} != sum(values) = {expected_value}"
    
    return True


def check_solution_invariants(solution: Solution, problem: Problem, tolerance: float = 1e-6) -> bool:
    """
    Check all solution invariants.
    
    INV-SOL-1: Complete assignment (every item assigned exactly once)
    INV-SOL-2: No duplication
    INV-SOL-3: Feasibility (capacity constraints)
    INV-SOL-4: Value difference non-negative
    INV-SOL-5: Objective consistency
    """
    # Collect all assigned items
    assigned_items: Set[int] = set()
    all_item_ids = {item.id for item in problem.items}
    
    for bin_obj in solution.bins:
        for item in bin_obj.items:
            # INV-SOL-2: No duplication
            assert item.id not in assigned_items, \
                f"INV-SOL-2 violated: Item {item.id} assigned to multiple bins"
            assigned_items.add(item.id)
    
    # INV-SOL-1: Complete assignment
    assert assigned_items == all_item_ids, \
        f"INV-SOL-1 violated: Missing items {all_item_ids - assigned_items}"
    
    # INV-SOL-3: Feasibility
    for i, bin_obj in enumerate(solution.bins):
        capacity = problem.bin_capacities[i] if i < len(problem.bin_capacities) else problem.bin_capacities[0]
        assert bin_obj.current_weight <= capacity + tolerance, \
            f"INV-SOL-3 violated: Bin {i} exceeds capacity ({bin_obj.current_weight} > {capacity})"
    
    # INV-SOL-4: Value difference non-negative
    assert solution.value_difference >= -tolerance, \
        f"INV-SOL-4 violated: value_difference = {solution.value_difference} < 0"
    
    # INV-SOL-5: Objective consistency
    bin_values = [bin_obj.current_value for bin_obj in solution.bins]
    expected_diff = max(bin_values) - min(bin_values) if bin_values else 0
    assert abs(solution.value_difference - expected_diff) < tolerance, \
        f"INV-SOL-5 violated: value_difference = {solution.value_difference} != max-min = {expected_diff}"
    
    return True


# ============================================================================
# Item Invariant Tests
# ============================================================================

class TestItemInvariants:
    """Tests for Item invariants."""
    
    def test_inv_item_1_weight_nonnegative(self):
        """INV-ITEM-1: Item weight cannot be negative."""
        # Valid item
        item = Item(id=1, weight=10.0, value=20.0)
        assert check_item_invariants(item)
        
        # Zero weight is valid
        item_zero = Item(id=2, weight=0.0, value=5.0)
        assert check_item_invariants(item_zero)
        
        # Negative weight should raise error
        with pytest.raises(ValueError):
            Item(id=3, weight=-5.0, value=10.0)
    
    def test_inv_item_2_value_nonnegative(self):
        """INV-ITEM-2: Item value cannot be negative."""
        # Valid item
        item = Item(id=1, weight=10.0, value=20.0)
        assert check_item_invariants(item)
        
        # Zero value is valid
        item_zero = Item(id=2, weight=5.0, value=0.0)
        assert check_item_invariants(item_zero)
        
        # Negative value should raise error
        with pytest.raises(ValueError):
            Item(id=3, weight=10.0, value=-5.0)
    
    def test_inv_item_3_unique_ids(self):
        """INV-ITEM-3: Items should have unique identifiers."""
        item1 = Item(id=1, weight=10.0, value=20.0)
        item2 = Item(id=2, weight=15.0, value=25.0)
        
        assert item1.id != item2.id
        assert hash(item1) != hash(item2)
        assert item1 != item2


# ============================================================================
# Bin Invariant Tests
# ============================================================================

class TestBinInvariants:
    """Tests for Bin invariants."""
    
    def test_inv_bin_1_capacity_positive(self):
        """INV-BIN-1: Bin capacity must be positive."""
        # Valid bin
        bin_obj = Bin(id=0, capacity=100.0)
        assert bin_obj.capacity > 0
        
        # Zero or negative capacity should raise error
        with pytest.raises(ValueError):
            Bin(id=1, capacity=0.0)
        
        with pytest.raises(ValueError):
            Bin(id=2, capacity=-50.0)
    
    def test_inv_bin_2_weight_within_capacity(self):
        """INV-BIN-2: Current weight should not exceed capacity."""
        bin_obj = Bin(id=0, capacity=100.0)
        
        # Add items that fit
        item1 = Item(id=1, weight=30.0, value=10.0)
        item2 = Item(id=2, weight=40.0, value=20.0)
        
        assert bin_obj.add_item(item1)
        assert bin_obj.add_item(item2)
        
        assert check_bin_invariants(bin_obj)
        
        # Item that doesn't fit
        item3 = Item(id=3, weight=50.0, value=30.0)
        assert not bin_obj.can_fit(item3)
    
    def test_inv_bin_3_weight_consistency(self):
        """INV-BIN-3: current_weight should equal sum of item weights."""
        bin_obj = Bin(id=0, capacity=100.0)
        
        items = [
            Item(id=1, weight=10.0, value=5.0),
            Item(id=2, weight=20.0, value=10.0),
            Item(id=3, weight=15.0, value=7.5)
        ]
        
        for item in items:
            bin_obj.add_item(item)
        
        expected_weight = sum(item.weight for item in items)
        assert abs(bin_obj.current_weight - expected_weight) < 1e-6
    
    def test_inv_bin_4_value_consistency(self):
        """INV-BIN-4: current_value should equal sum of item values."""
        bin_obj = Bin(id=0, capacity=100.0)
        
        items = [
            Item(id=1, weight=10.0, value=5.0),
            Item(id=2, weight=20.0, value=10.0),
            Item(id=3, weight=15.0, value=7.5)
        ]
        
        for item in items:
            bin_obj.add_item(item)
        
        expected_value = sum(item.value for item in items)
        assert abs(bin_obj.current_value - expected_value) < 1e-6


# ============================================================================
# Solution Invariant Tests
# ============================================================================

class TestSolutionInvariants:
    """Tests for Solution invariants using all algorithms."""
    
    @pytest.fixture
    def simple_problem(self):
        """Create a simple test problem."""
        items = [
            Item(id=i, weight=np.random.uniform(5, 20), value=np.random.uniform(10, 50))
            for i in range(15)
        ]
        return Problem(
            items=items,
            num_bins=3,
            bin_capacities=[100.0, 100.0, 100.0],
            name="test_simple"
        )
    
    @pytest.fixture
    def heterogeneous_problem(self):
        """Create a problem with heterogeneous bin capacities."""
        items = [
            Item(id=i, weight=np.random.uniform(5, 25), value=np.random.uniform(10, 40))
            for i in range(20)
        ]
        return Problem(
            items=items,
            num_bins=4,
            bin_capacities=[80.0, 100.0, 120.0, 90.0],
            name="test_heterogeneous"
        )
    
    def test_ffd_solution_invariants(self, simple_problem):
        """Test FFD algorithm satisfies all solution invariants."""
        algo = FirstFitDecreasing()
        solution = algo.solve(simple_problem)
        
        assert check_solution_invariants(solution, simple_problem)
    
    def test_bfd_solution_invariants(self, simple_problem):
        """Test BFD algorithm satisfies all solution invariants."""
        algo = BestFitDecreasing()
        solution = algo.solve(simple_problem)
        
        assert check_solution_invariants(solution, simple_problem)
    
    def test_wfd_solution_invariants(self, simple_problem):
        """Test WFD algorithm satisfies all solution invariants."""
        algo = WorstFitDecreasing()
        solution = algo.solve(simple_problem)
        
        assert check_solution_invariants(solution, simple_problem)
    
    def test_round_robin_solution_invariants(self, simple_problem):
        """Test Round Robin algorithm satisfies all solution invariants."""
        algo = RoundRobinGreedy()
        solution = algo.solve(simple_problem)
        
        assert check_solution_invariants(solution, simple_problem)
    
    def test_sa_solution_invariants(self, simple_problem):
        """Test Simulated Annealing algorithm satisfies all solution invariants."""
        algo = SimulatedAnnealing(max_iterations=500)
        solution = algo.solve(simple_problem)
        
        assert check_solution_invariants(solution, simple_problem)
    
    def test_ga_solution_invariants(self, simple_problem):
        """Test Genetic Algorithm satisfies all solution invariants."""
        algo = GeneticAlgorithm(population_size=20, generations=50)
        solution = algo.solve(simple_problem)
        
        assert check_solution_invariants(solution, simple_problem)
    
    def test_tabu_solution_invariants(self, simple_problem):
        """Test Tabu Search algorithm satisfies all solution invariants."""
        algo = TabuSearch(max_iterations=500)
        solution = algo.solve(simple_problem)
        
        assert check_solution_invariants(solution, simple_problem)
    
    def test_heterogeneous_capacity_invariants(self, heterogeneous_problem):
        """Test invariants with heterogeneous bin capacities."""
        algorithms = [
            FirstFitDecreasing(),
            BestFitDecreasing(),
            WorstFitDecreasing(),
            RoundRobinGreedy(),
            SimulatedAnnealing(max_iterations=500),
        ]
        
        for algo in algorithms:
            solution = algo.solve(heterogeneous_problem)
            assert check_solution_invariants(solution, heterogeneous_problem), \
                f"Invariants violated for {algo.name}"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_bin(self):
        """Test with a single bin."""
        items = [Item(id=i, weight=10.0, value=5.0) for i in range(5)]
        problem = Problem(
            items=items,
            num_bins=1,
            bin_capacities=[100.0],
            name="single_bin"
        )
        
        algo = FirstFitDecreasing()
        solution = algo.solve(problem)
        
        assert check_solution_invariants(solution, problem)
        assert solution.value_difference == 0  # Only one bin, difference is 0
    
    def test_single_item(self):
        """Test with a single item."""
        items = [Item(id=0, weight=10.0, value=50.0)]
        problem = Problem(
            items=items,
            num_bins=3,
            bin_capacities=[50.0, 50.0, 50.0],
            name="single_item"
        )
        
        algo = FirstFitDecreasing()
        solution = algo.solve(problem)
        
        assert check_solution_invariants(solution, problem)
    
    def test_identical_items(self):
        """Test with identical items."""
        items = [Item(id=i, weight=10.0, value=20.0) for i in range(9)]
        problem = Problem(
            items=items,
            num_bins=3,
            bin_capacities=[50.0, 50.0, 50.0],
            name="identical_items"
        )
        
        # Round Robin should distribute evenly
        algo = RoundRobinGreedy()
        solution = algo.solve(problem)
        
        assert check_solution_invariants(solution, problem)
        
        # With identical items and even distribution, difference should be 0
        bin_counts = [len(b.items) for b in solution.bins]
        if all(c == 3 for c in bin_counts):  # Perfect distribution
            assert solution.value_difference == 0
    
    def test_zero_value_items(self):
        """Test with items that have zero value."""
        items = [Item(id=i, weight=10.0, value=0.0) for i in range(6)]
        problem = Problem(
            items=items,
            num_bins=2,
            bin_capacities=[50.0, 50.0],
            name="zero_values"
        )
        
        algo = FirstFitDecreasing()
        solution = algo.solve(problem)
        
        assert check_solution_invariants(solution, problem)
        assert solution.value_difference == 0  # All values are 0
    
    def test_zero_weight_items(self):
        """Test with items that have zero weight (ghost items)."""
        items = [Item(id=i, weight=0.0, value=10.0) for i in range(6)]
        problem = Problem(
            items=items,
            num_bins=2,
            bin_capacities=[50.0, 50.0],
            name="zero_weights"
        )
        
        algo = FirstFitDecreasing()
        solution = algo.solve(problem)
        
        assert check_solution_invariants(solution, problem)
    
    def test_tight_capacity(self):
        """Test with tight capacity constraints."""
        items = [
            Item(id=0, weight=45.0, value=100.0),
            Item(id=1, weight=45.0, value=100.0),
            Item(id=2, weight=10.0, value=50.0),
        ]
        problem = Problem(
            items=items,
            num_bins=2,
            bin_capacities=[55.0, 55.0],
            name="tight_capacity"
        )
        
        algo = FirstFitDecreasing()
        solution = algo.solve(problem)
        
        assert check_solution_invariants(solution, problem)


# ============================================================================
# Algorithm-Specific Invariant Tests
# ============================================================================

class TestSimulatedAnnealingInvariants:
    """Tests specific to Simulated Annealing invariants."""
    
    def test_inv_sa_1_temperature_decreasing(self):
        """INV-SA-1: Temperature should decrease over time."""
        items = [Item(id=i, weight=10.0, value=20.0) for i in range(10)]
        problem = Problem(
            items=items,
            num_bins=3,
            bin_capacities=[50.0, 50.0, 50.0],
            name="sa_test"
        )
        
        algo = SimulatedAnnealing(
            initial_temp=1000.0,
            min_temp=0.01,
            cooling_rate=0.95,
            max_iterations=100
        )
        
        solution = algo.solve(problem)
        
        # Check temperature history if available
        if hasattr(algo, 'temperature_history') and algo.temperature_history:
            temps = algo.temperature_history
            # Temperature should be non-increasing
            for i in range(1, len(temps)):
                assert temps[i] <= temps[i-1] + 1e-6, \
                    f"INV-SA-1 violated: T[{i}] = {temps[i]} > T[{i-1}] = {temps[i-1]}"
    
    def test_inv_sa_4_best_monotonic(self):
        """INV-SA-4: Best solution should never get worse."""
        items = [Item(id=i, weight=np.random.uniform(5, 15), value=np.random.uniform(10, 30)) for i in range(15)]
        problem = Problem(
            items=items,
            num_bins=3,
            bin_capacities=[60.0, 60.0, 60.0],
            name="sa_best_test"
        )
        
        algo = SimulatedAnnealing(max_iterations=500)
        solution = algo.solve(problem)
        
        # If we track best objective over time, it should be monotonically non-increasing
        if hasattr(algo, 'best_history'):
            bests = algo.best_history
            for i in range(1, len(bests)):
                assert bests[i] <= bests[i-1] + 1e-6, \
                    f"INV-SA-4 violated: best[{i}] = {bests[i]} > best[{i-1}] = {bests[i-1]}"


class TestGeneticAlgorithmInvariants:
    """Tests specific to Genetic Algorithm invariants."""
    
    def test_inv_ga_1_population_constant(self):
        """INV-GA-1: Population size should remain constant."""
        items = [Item(id=i, weight=10.0, value=20.0) for i in range(10)]
        problem = Problem(
            items=items,
            num_bins=3,
            bin_capacities=[50.0, 50.0, 50.0],
            name="ga_test"
        )
        
        pop_size = 30
        algo = GeneticAlgorithm(population_size=pop_size, generations=50)
        solution = algo.solve(problem)
        
        # Check that final population matches expected size
        if hasattr(algo, 'population'):
            assert len(algo.population) == pop_size, \
                f"INV-GA-1 violated: |population| = {len(algo.population)} != {pop_size}"
    
    def test_inv_ga_3_fitness_nonnegative(self):
        """INV-GA-3: Fitness values should be non-negative."""
        items = [Item(id=i, weight=10.0, value=20.0) for i in range(10)]
        problem = Problem(
            items=items,
            num_bins=3,
            bin_capacities=[50.0, 50.0, 50.0],
            name="ga_fitness_test"
        )
        
        algo = GeneticAlgorithm(population_size=20, generations=30)
        solution = algo.solve(problem)
        
        # Objective (which is related to fitness) should be non-negative
        assert solution.value_difference >= 0, \
            f"INV-GA-3 violated: fitness/objective = {solution.value_difference} < 0"


# ============================================================================
# Regression Tests
# ============================================================================

class TestRegressionCases:
    """Regression tests with known optimal solutions."""
    
    def test_trivial_perfect_balance(self):
        """Trivial case where perfect balance is achievable."""
        items = [
            Item(id=0, weight=10.0, value=50.0),
            Item(id=1, weight=10.0, value=50.0),
        ]
        problem = Problem(
            items=items,
            num_bins=2,
            bin_capacities=[50.0, 50.0],
            name="trivial_balance"
        )
        
        algo = BestFitDecreasing()
        solution = algo.solve(problem)
        
        assert check_solution_invariants(solution, problem)
        # Perfect balance should be achievable
        assert solution.value_difference == 0 or solution.value_difference < 1e-6
    
    def test_known_optimal(self):
        """Test case with known optimal solution."""
        items = [
            Item(id=0, weight=30.0, value=60.0),
            Item(id=1, weight=20.0, value=40.0),
            Item(id=2, weight=10.0, value=20.0),
        ]
        problem = Problem(
            items=items,
            num_bins=2,
            bin_capacities=[50.0, 50.0],
            name="known_optimal"
        )
        
        # Optimal: Bin0 = {item0}, Bin1 = {item1, item2}
        # Values: 60 and 60 => difference = 0
        
        algo = SimulatedAnnealing(max_iterations=1000)
        solution = algo.solve(problem)
        
        assert check_solution_invariants(solution, problem)
        # SA should be able to find optimal or near-optimal
        assert solution.value_difference <= 20  # Allow some slack


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
