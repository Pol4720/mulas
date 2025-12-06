"""
Core module containing problem definitions and data structures.
"""

from .problem import Problem, Item, Bin, Solution
from .instance_generator import InstanceGenerator

__all__ = [
    "Problem",
    "Item",
    "Bin", 
    "Solution",
    "InstanceGenerator",
]
