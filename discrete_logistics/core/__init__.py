"""
Core module containing problem definitions and data structures.
"""

try:
    from .problem import Problem, Item, Bin, Solution
    from .instance_generator import InstanceGenerator
except ImportError:
    from discrete_logistics.core.problem import Problem, Item, Bin, Solution
    from discrete_logistics.core.instance_generator import InstanceGenerator

__all__ = [
    "Problem",
    "Item",
    "Bin", 
    "Solution",
    "InstanceGenerator",
]
