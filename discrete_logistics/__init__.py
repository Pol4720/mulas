"""
Discrete Logistics Transport Problem - Balanced Multi-Bin Packing with Capacity Constraints

A comprehensive implementation of various algorithms for solving the NP-hard balanced
bin packing problem with applications to logistics and resource distribution.

Author: DAA Project Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DAA Project Team"

from .core.problem import Problem, Item, Bin, Solution
from .core.instance_generator import InstanceGenerator

__all__ = [
    "Problem",
    "Item", 
    "Bin",
    "Solution",
    "InstanceGenerator",
]
