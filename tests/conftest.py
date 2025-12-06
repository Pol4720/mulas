"""
Test configuration and fixtures for the test suite.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from discrete_logistics.core.problem import Problem, Solution, Item, Bin


@pytest.fixture
def simple_items():
    """Create a simple list of items for testing."""
    return [
        Item(id=0, weight=10, value=100, name="Item_0"),
        Item(id=1, weight=20, value=200, name="Item_1"),
        Item(id=2, weight=15, value=150, name="Item_2"),
        Item(id=3, weight=25, value=250, name="Item_3"),
        Item(id=4, weight=12, value=120, name="Item_4"),
    ]


@pytest.fixture
def simple_problem(simple_items):
    """Create a simple problem instance for testing."""
    return Problem(
        items=simple_items,
        num_bins=2,
        bin_capacities=[50.0, 60.0],
        name="Test Problem"
    )


@pytest.fixture
def small_problem():
    """Create a small problem instance suitable for exact algorithms."""
    items = [
        Item(id=i, weight=w, value=v)
        for i, (w, v) in enumerate([
            (5, 50), (10, 100), (8, 80), (12, 120),
            (6, 60), (15, 150), (7, 70), (9, 90)
        ])
    ]
    return Problem(
        items=items,
        num_bins=3,
        bin_capacities=[30.0, 35.0, 40.0],
        name="Small Test Problem"
    )


@pytest.fixture
def medium_problem():
    """Create a medium-sized problem for metaheuristic testing."""
    import random
    random.seed(42)
    
    items = [
        Item(
            id=i,
            weight=random.uniform(5, 25),
            value=random.uniform(50, 250)
        )
        for i in range(20)
    ]
    
    return Problem(
        items=items,
        num_bins=4,
        bin_capacities=[80.0, 90.0, 85.0, 95.0],
        name="Medium Test Problem"
    )


@pytest.fixture
def heterogeneous_capacity_problem():
    """Create a problem with very different bin capacities."""
    items = [
        Item(id=0, weight=30, value=300),
        Item(id=1, weight=20, value=200),
        Item(id=2, weight=15, value=150),
        Item(id=3, weight=10, value=100),
        Item(id=4, weight=5, value=50),
    ]
    return Problem(
        items=items,
        num_bins=3,
        bin_capacities=[35.0, 25.0, 45.0],  # Very different capacities
        name="Heterogeneous Capacity Problem"
    )


@pytest.fixture
def tight_capacity_problem():
    """Create a problem where capacity is tight."""
    items = [
        Item(id=0, weight=10, value=100),
        Item(id=1, weight=10, value=100),
        Item(id=2, weight=10, value=100),
        Item(id=3, weight=10, value=100),
    ]
    return Problem(
        items=items,
        num_bins=2,
        bin_capacities=[20.0, 20.0],  # Exactly fits 2 items per bin
        name="Tight Capacity Problem"
    )


@pytest.fixture
def single_bin_problem():
    """Create a problem with only one bin."""
    items = [
        Item(id=0, weight=10, value=100),
        Item(id=1, weight=20, value=200),
        Item(id=2, weight=15, value=150),
    ]
    return Problem(
        items=items,
        num_bins=1,
        bin_capacities=[50.0],
        name="Single Bin Problem"
    )
