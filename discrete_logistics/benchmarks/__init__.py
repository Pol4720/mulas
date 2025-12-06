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
"""

from .runner import BenchmarkRunner, BenchmarkResult
from .instances import TestInstanceSet, StandardInstances
from .analysis import StatisticalAnalyzer, PerformanceReport

__all__ = [
    'BenchmarkRunner',
    'BenchmarkResult',
    'TestInstanceSet',
    'StandardInstances',
    'StatisticalAnalyzer',
    'PerformanceReport'
]
