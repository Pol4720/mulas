"""
Benchmarks Module
=================

Provides comprehensive benchmarking tools for comparing
algorithm performance on various test instances.

Components:
-----------
- runner: BenchmarkRunner for executing tests
- instances: Standard test instance sets
- analysis: Statistical analysis tools
- optimality_analysis: Compare heuristics vs brute force optimal
- scalability_analysis: Measure brute force scalability limits
"""

from .runner import BenchmarkRunner, BenchmarkResult
from .instances import TestInstanceSet, StandardInstances
from .analysis import StatisticalAnalyzer, PerformanceReport
from .optimality_analysis import OptimalityAnalyzer, OptimalityResult, AlgorithmProfile
from .scalability_analysis import ScalabilityAnalyzer, ScalabilityResult, ScalabilityProfile

__all__ = [
    'BenchmarkRunner',
    'BenchmarkResult',
    'TestInstanceSet',
    'StandardInstances',
    'StatisticalAnalyzer',
    'PerformanceReport',
    'OptimalityAnalyzer',
    'OptimalityResult',
    'AlgorithmProfile',
    'ScalabilityAnalyzer',
    'ScalabilityResult',
    'ScalabilityProfile',
]
