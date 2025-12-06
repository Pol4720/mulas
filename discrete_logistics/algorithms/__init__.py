"""
Algorithm implementations for the Balanced Multi-Bin Packing Problem.
"""

try:
    from .base import Algorithm, AlgorithmRegistry
    from .greedy import (
        FirstFitDecreasing,
        BestFitDecreasing,
        WorstFitDecreasing,
        RoundRobinGreedy,
        LargestDifferenceFirst
    )
    from .dynamic_programming import DynamicProgramming
    from .branch_and_bound import BranchAndBound
    from .metaheuristics import (
        SimulatedAnnealing,
        GeneticAlgorithm,
        TabuSearch
    )
    from .approximation import (
        LPTApproximation,
        MultiWayPartition
    )
except ImportError:
    from discrete_logistics.algorithms.base import Algorithm, AlgorithmRegistry
    from discrete_logistics.algorithms.greedy import (
        FirstFitDecreasing,
        BestFitDecreasing,
        WorstFitDecreasing,
        RoundRobinGreedy,
        LargestDifferenceFirst
    )
    from discrete_logistics.algorithms.dynamic_programming import DynamicProgramming
    from discrete_logistics.algorithms.branch_and_bound import BranchAndBound
    from discrete_logistics.algorithms.metaheuristics import (
        SimulatedAnnealing,
        GeneticAlgorithm,
        TabuSearch
    )
    from discrete_logistics.algorithms.approximation import (
        LPTApproximation,
        MultiWayPartition
    )

__all__ = [
    "Algorithm",
    "AlgorithmRegistry",
    "FirstFitDecreasing",
    "BestFitDecreasing", 
    "WorstFitDecreasing",
    "RoundRobinGreedy",
    "LargestDifferenceFirst",
    "DynamicProgramming",
    "BranchAndBound",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "TabuSearch",
    "LPTApproximation",
    "MultiWayPartition",
]
