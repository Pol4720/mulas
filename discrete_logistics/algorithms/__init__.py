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
    from .brute_force import BruteForce, ExhaustiveSearch
    from .metaheuristics import (
        SimulatedAnnealing,
        GeneticAlgorithm,
        TabuSearch
    )
    from .approximation import (
        LPTApproximation,
        MultiWayPartition
    )
    from .hybrid import (
        HybridDPMeta,
        HybridQualityFocused,
        HybridSpeedFocused,
        HybridWithGenetic,
        HybridWithTabu
    )
    from .external_adapters import (
        GeminiHGADP,
        QwenSADP
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
    from discrete_logistics.algorithms.brute_force import BruteForce, ExhaustiveSearch
    from discrete_logistics.algorithms.metaheuristics import (
        SimulatedAnnealing,
        GeneticAlgorithm,
        TabuSearch
    )
    from discrete_logistics.algorithms.approximation import (
        LPTApproximation,
        MultiWayPartition
    )
    from discrete_logistics.algorithms.hybrid import (
        HybridDPMeta,
        HybridQualityFocused,
        HybridSpeedFocused,
        HybridWithGenetic,
        HybridWithTabu
    )
    from discrete_logistics.algorithms.external_adapters import (
        GeminiHGADP,
        QwenSADP
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
    "BruteForce",
    "ExhaustiveSearch",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "TabuSearch",
    "LPTApproximation",
    "MultiWayPartition",
    "HybridDPMeta",
    "HybridQualityFocused",
    "HybridSpeedFocused",
    "HybridWithGenetic",
    "HybridWithTabu",
    "GeminiHGADP",
    "QwenSADP",
]
