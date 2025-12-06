"""
Unit tests for core problem definitions.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from discrete_logistics.core.problem import Problem, Solution, Item, Bin


class TestItem:
    """Tests for Item class."""
    
    def test_item_creation(self):
        """Test basic item creation."""
        item = Item(id=0, weight=10.0, value=100.0)
        assert item.id == 0
        assert item.weight == 10.0
        assert item.value == 100.0
        assert item.name == "Item_0"
    
    def test_item_with_name(self):
        """Test item creation with custom name."""
        item = Item(id=1, weight=20.0, value=200.0, name="Custom")
        assert item.name == "Custom"
    
    def test_item_negative_weight_raises(self):
        """Test that negative weight raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            Item(id=0, weight=-10.0, value=100.0)
    
    def test_item_negative_value_raises(self):
        """Test that negative value raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            Item(id=0, weight=10.0, value=-100.0)
    
    def test_item_equality(self):
        """Test item equality is based on id."""
        item1 = Item(id=0, weight=10.0, value=100.0)
        item2 = Item(id=0, weight=20.0, value=200.0)  # Same id
        item3 = Item(id=1, weight=10.0, value=100.0)  # Different id
        
        assert item1 == item2
        assert item1 != item3
    
    def test_item_hash(self):
        """Test item hashing for use in sets/dicts."""
        item1 = Item(id=0, weight=10.0, value=100.0)
        item2 = Item(id=0, weight=20.0, value=200.0)
        
        items_set = {item1, item2}
        assert len(items_set) == 1  # Same id means same hash
    
    def test_item_to_dict(self):
        """Test item serialization."""
        item = Item(id=0, weight=10.0, value=100.0, name="Test")
        data = item.to_dict()
        
        assert data["id"] == 0
        assert data["weight"] == 10.0
        assert data["value"] == 100.0
        assert data["name"] == "Test"
    
    def test_item_from_dict(self):
        """Test item deserialization."""
        data = {"id": 0, "weight": 10.0, "value": 100.0, "name": "Test"}
        item = Item.from_dict(data)
        
        assert item.id == 0
        assert item.weight == 10.0


class TestBin:
    """Tests for Bin class."""
    
    def test_bin_creation(self):
        """Test basic bin creation."""
        bin = Bin(id=0, capacity=100.0)
        assert bin.id == 0
        assert bin.capacity == 100.0
        assert bin.items == []
    
    def test_bin_invalid_capacity(self):
        """Test that non-positive capacity raises error."""
        with pytest.raises(ValueError):
            Bin(id=0, capacity=0)
        with pytest.raises(ValueError):
            Bin(id=0, capacity=-10)
    
    def test_bin_add_item(self):
        """Test adding items to bin."""
        bin = Bin(id=0, capacity=100.0)
        item = Item(id=0, weight=30.0, value=50.0)
        
        bin.add_item(item)
        assert len(bin.items) == 1
        assert bin.current_weight == 30.0
        assert bin.current_value == 50.0
    
    def test_bin_can_fit(self):
        """Test capacity checking."""
        bin = Bin(id=0, capacity=50.0)
        item1 = Item(id=0, weight=30.0, value=50.0)
        item2 = Item(id=1, weight=30.0, value=50.0)
        
        assert bin.can_fit(item1)
        bin.add_item(item1)
        assert not bin.can_fit(item2)  # Would exceed capacity
    
    def test_bin_remaining_capacity(self):
        """Test remaining capacity calculation."""
        bin = Bin(id=0, capacity=100.0)
        item = Item(id=0, weight=30.0, value=50.0)
        
        assert bin.remaining_capacity == 100.0
        bin.add_item(item)
        assert bin.remaining_capacity == 70.0


class TestProblem:
    """Tests for Problem class."""
    
    def test_problem_creation(self, simple_items):
        """Test basic problem creation."""
        problem = Problem(
            items=simple_items,
            num_bins=2,
            bin_capacities=[50.0, 60.0]
        )
        
        assert problem.n_items == 5
        assert problem.num_bins == 2
        assert len(problem.bin_capacities) == 2
    
    def test_problem_single_capacity(self, simple_items):
        """Test problem with single capacity for all bins."""
        # Note: This implementation requires explicit capacities for each bin
        problem = Problem(
            items=simple_items,
            num_bins=3,
            bin_capacities=[50.0, 50.0, 50.0]  # Must provide for each bin
        )
        
        assert len(problem.bin_capacities) == 3
        assert all(c == 50.0 for c in problem.bin_capacities)
    
    def test_problem_properties(self, simple_problem):
        """Test problem computed properties."""
        assert simple_problem.total_weight == 82.0  # Sum of all weights
        assert simple_problem.total_value == 820.0  # Sum of all values
    
    def test_problem_create_empty_solution(self, simple_problem):
        """Test creating empty solution from problem."""
        solution = simple_problem.create_empty_solution("Test Algorithm")
        
        assert len(solution.bins) == 2
        assert all(len(b.items) == 0 for b in solution.bins)
        assert solution.algorithm_name == "Test Algorithm"


class TestSolution:
    """Tests for Solution class."""
    
    def test_solution_value_difference(self, simple_problem):
        """Test value difference calculation."""
        solution = simple_problem.create_empty_solution("Test")
        bins = solution.bins
        
        # Assign items: bin 0 gets item0 (v=100), bin 1 gets item1 (v=200)
        bins[0].add_item(simple_problem.items[0])
        bins[1].add_item(simple_problem.items[1])
        
        assert solution.value_difference == 100.0  # 200 - 100
    
    def test_solution_all_items_assigned(self, simple_problem):
        """Test checking if all items are assigned."""
        solution = simple_problem.create_empty_solution("Test")
        
        # Empty solution has no items assigned
        assigned = set()
        for bin in solution.bins:
            for item in bin.items:
                assigned.add(item.id)
        assert len(assigned) == 0
    
    def test_solution_copy(self, simple_problem):
        """Test solution deep copy."""
        solution = simple_problem.create_empty_solution("Test")
        solution.bins[0].add_item(simple_problem.items[0])
        
        copy = solution.copy()
        
        assert copy is not solution
        assert len(copy.bins[0].items) == 1
        
        # Modifying copy doesn't affect original
        copy.bins[0].add_item(simple_problem.items[1])
        assert len(solution.bins[0].items) == 1
