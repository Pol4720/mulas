"""
Mathematical Formalization of the Balanced Multi-Bin Packing Problem.

This module provides formal mathematical definitions, ILP formulations,
and theoretical foundations for the problem.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# For optional ILP solving
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from ..core.problem import Problem, Solution, Item, Bin


@dataclass
class MathematicalModel:
    """
    Formal mathematical model for the Balanced Multi-Bin Packing Problem.
    
    PROBLEM DEFINITION
    ==================
    
    Given:
    ------
    - n items: I = {1, 2, ..., n}
    - k bins (mules): B = {1, 2, ..., k}
    - Weight function: w: I → ℝ⁺
    - Value function: v: I → ℝ⁺
    - Bin capacity: W ∈ ℝ⁺
    
    Decision Variables:
    -------------------
    - x_{ij} ∈ {0, 1}: 1 if item i is assigned to bin j, 0 otherwise
    
    Auxiliary Variables:
    --------------------
    - V_j = Σᵢ v_i · x_{ij}: Total value in bin j
    - V_max = max_j V_j: Maximum bin value
    - V_min = min_j V_j: Minimum bin value
    
    Objective Function:
    -------------------
    minimize Z = V_max - V_min
    
    This is the "makespan" or "balance" objective, seeking to minimize
    the difference between the most and least loaded bins.
    
    Constraints:
    ------------
    1. Assignment: Σⱼ x_{ij} = 1 ∀i ∈ I
       (Each item assigned to exactly one bin)
    
    2. Capacity: Σᵢ w_i · x_{ij} ≤ W ∀j ∈ B
       (No bin exceeds weight capacity)
    
    3. Coverage: Σᵢ x_{ij} ≥ 1 ∀j ∈ B
       (All bins must be used - optional)
    
    4. Binary: x_{ij} ∈ {0, 1}
    
    ALTERNATIVE FORMULATIONS
    ========================
    
    Min-Max Formulation:
    --------------------
    minimize V_max
    subject to:
        Σⱼ x_{ij} = 1 ∀i
        Σᵢ w_i · x_{ij} ≤ W ∀j
        Σᵢ v_i · x_{ij} ≤ V_max ∀j
        x_{ij} ∈ {0, 1}
    
    This minimizes the maximum load, related to makespan scheduling.
    
    Variance Minimization:
    ----------------------
    minimize Σⱼ (V_j - V̄)²
    where V̄ = (Σᵢ v_i) / k
    
    This minimizes the variance of bin values, promoting balance.
    (Non-linear objective, requires linearization or different solver)
    
    PROBLEM VARIANTS
    ================
    
    1. Identical Bins: All bins have the same capacity (standard case)
    2. Variable Capacity: Each bin j has capacity W_j
    3. Multi-Dimensional: Items have multiple attributes (weight, volume, etc.)
    4. Online Version: Items arrive sequentially, irrevocable decisions
    5. Stochastic: Item properties are random variables
    
    RELATIONSHIP TO OTHER PROBLEMS
    ==============================
    
    - Bin Packing: Minimize number of bins (different objective)
    - Multiprocessor Scheduling: Minimize makespan (P||C_max)
    - Number Partitioning: k=2 case is equivalent to PARTITION
    - Balanced Graph Partitioning: Similar balance objective
    - Fair Division: Envy-free allocation concerns
    """
    
    n_items: int
    n_bins: int
    weights: List[float]
    values: List[float]
    capacity: float
    
    def __init__(self, problem: Problem):
        """Initialize model from problem instance."""
        self.n_items = problem.n_items
        self.n_bins = problem.num_bins
        self.weights = [item.weight for item in problem.items]
        self.values = [item.value for item in problem.items]
        self.capacity = problem.bin_capacity
        self.problem = problem
    
    def get_latex_formulation(self) -> str:
        """
        Return LaTeX formulation of the problem.
        
        Returns:
            LaTeX string for mathematical formulation
        """
        return r"""
\section{Mathematical Formulation}

\subsection{Problem Definition}

Given:
\begin{itemize}
    \item Set of items $I = \{1, 2, \ldots, n\}$
    \item Set of bins $B = \{1, 2, \ldots, k\}$
    \item Weight function $w: I \rightarrow \mathbb{R}^+$
    \item Value function $v: I \rightarrow \mathbb{R}^+$
    \item Bin capacity $W \in \mathbb{R}^+$
\end{itemize}

\subsection{Decision Variables}

\begin{equation}
x_{ij} = \begin{cases}
1 & \text{if item } i \text{ is assigned to bin } j \\
0 & \text{otherwise}
\end{cases}
\end{equation}

\subsection{Auxiliary Variables}

Total value in bin $j$:
\begin{equation}
V_j = \sum_{i \in I} v_i \cdot x_{ij}
\end{equation}

Maximum and minimum bin values:
\begin{equation}
V_{\max} = \max_{j \in B} V_j, \quad V_{\min} = \min_{j \in B} V_j
\end{equation}

\subsection{Objective Function}

\begin{equation}
\text{minimize } Z = V_{\max} - V_{\min}
\end{equation}

\subsection{Constraints}

Assignment constraint (each item to exactly one bin):
\begin{equation}
\sum_{j \in B} x_{ij} = 1, \quad \forall i \in I
\end{equation}

Capacity constraint (no bin exceeds capacity):
\begin{equation}
\sum_{i \in I} w_i \cdot x_{ij} \leq W, \quad \forall j \in B
\end{equation}

Binary constraint:
\begin{equation}
x_{ij} \in \{0, 1\}, \quad \forall i \in I, \forall j \in B
\end{equation}
"""
    
    def get_instance_data(self) -> Dict:
        """Get formatted instance data for display."""
        return {
            "n": self.n_items,
            "k": self.n_bins,
            "W": self.capacity,
            "weights": self.weights,
            "values": self.values,
            "total_weight": sum(self.weights),
            "total_value": sum(self.values),
            "avg_weight": np.mean(self.weights),
            "avg_value": np.mean(self.values),
            "ideal_per_bin": sum(self.values) / self.n_bins
        }
    
    def verify_solution(self, assignment: Dict[int, int]) -> Tuple[bool, List[str]]:
        """
        Verify if a solution satisfies all constraints.
        
        Args:
            assignment: Dict mapping item index to bin index
            
        Returns:
            Tuple of (is_feasible, list_of_violations)
        """
        violations = []
        
        # Check assignment constraint
        assigned_items = set(assignment.keys())
        expected_items = set(range(self.n_items))
        
        if assigned_items != expected_items:
            missing = expected_items - assigned_items
            extra = assigned_items - expected_items
            if missing:
                violations.append(f"Missing items: {missing}")
            if extra:
                violations.append(f"Invalid items: {extra}")
        
        # Check capacity constraints
        bin_weights = {j: 0 for j in range(self.n_bins)}
        for i, j in assignment.items():
            if 0 <= j < self.n_bins:
                bin_weights[j] += self.weights[i]
        
        for j, weight in bin_weights.items():
            if weight > self.capacity:
                violations.append(
                    f"Bin {j} exceeds capacity: {weight:.2f} > {self.capacity:.2f}"
                )
        
        return len(violations) == 0, violations


class ILPFormulation:
    """
    Integer Linear Programming formulation using PuLP.
    
    Provides exact solution capability for small instances.
    """
    
    def __init__(self, problem: Problem):
        """Initialize ILP formulation."""
        if not PULP_AVAILABLE:
            raise ImportError(
                "PuLP is required for ILP formulation. "
                "Install with: pip install pulp"
            )
        
        self.problem = problem
        self.model = None
        self.x = None  # Decision variables
        self.v_max = None
        self.v_min = None
    
    def build_model(self, time_limit: int = 60) -> pulp.LpProblem:
        """
        Build the ILP model.
        
        Args:
            time_limit: Solver time limit in seconds
            
        Returns:
            PuLP LpProblem object
        """
        n = self.problem.n_items
        k = self.problem.num_bins
        items = self.problem.items
        capacity = self.problem.bin_capacity
        
        # Create problem
        self.model = pulp.LpProblem("Balanced_Bin_Packing", pulp.LpMinimize)
        
        # Decision variables: x[i][j] = 1 if item i in bin j
        self.x = pulp.LpVariable.dicts(
            "x",
            ((i, j) for i in range(n) for j in range(k)),
            cat=pulp.LpBinary
        )
        
        # Auxiliary variables for min-max
        self.v_max = pulp.LpVariable("V_max", lowBound=0)
        self.v_min = pulp.LpVariable("V_min", lowBound=0)
        
        # Objective: minimize V_max - V_min
        self.model += self.v_max - self.v_min, "Minimize_Difference"
        
        # Constraints
        
        # 1. Assignment: each item to exactly one bin
        for i in range(n):
            self.model += (
                pulp.lpSum(self.x[i, j] for j in range(k)) == 1,
                f"Assignment_{i}"
            )
        
        # 2. Capacity: no bin exceeds capacity
        for j in range(k):
            self.model += (
                pulp.lpSum(items[i].weight * self.x[i, j] for i in range(n)) <= capacity,
                f"Capacity_{j}"
            )
        
        # 3. V_max >= V_j for all j
        for j in range(k):
            self.model += (
                self.v_max >= pulp.lpSum(items[i].value * self.x[i, j] for i in range(n)),
                f"VMax_{j}"
            )
        
        # 4. V_min <= V_j for all j
        for j in range(k):
            self.model += (
                self.v_min <= pulp.lpSum(items[i].value * self.x[i, j] for i in range(n)),
                f"VMin_{j}"
            )
        
        return self.model
    
    def solve(
        self,
        solver: Optional[str] = None,
        time_limit: int = 60,
        gap: float = 0.01
    ) -> Optional[Solution]:
        """
        Solve the ILP model.
        
        Args:
            solver: Solver to use (None for default)
            time_limit: Time limit in seconds
            gap: Optimality gap tolerance
            
        Returns:
            Solution object if successful, None otherwise
        """
        if self.model is None:
            self.build_model(time_limit)
        
        # Configure solver
        if solver == "CBC" or solver is None:
            slv = pulp.PULP_CBC_CMD(
                timeLimit=time_limit,
                gapRel=gap,
                msg=False
            )
        elif solver == "GLPK":
            slv = pulp.GLPK_CMD(
                timeLimit=time_limit,
                msg=False
            )
        else:
            slv = pulp.getSolver(solver)
        
        # Solve
        status = self.model.solve(slv)
        
        if status != pulp.LpStatusOptimal:
            return None
        
        # Extract solution
        n = self.problem.n_items
        k = self.problem.num_bins
        items = self.problem.items
        
        assignment = {}
        for i in range(n):
            for j in range(k):
                if pulp.value(self.x[i, j]) > 0.5:
                    assignment[i] = j
                    break
        
        # Build solution object
        bins = [Bin(j, self.problem.bin_capacity) for j in range(k)]
        for i, j in assignment.items():
            bins[j].add_item(items[i])
        
        solution = Solution(
            bins=bins,
            algorithm_name="ILP (Exact)",
            metadata={
                "status": pulp.LpStatus[status],
                "objective": pulp.value(self.model.objective),
                "v_max": pulp.value(self.v_max),
                "v_min": pulp.value(self.v_min),
                "solver": solver or "CBC"
            }
        )
        
        return solution
    
    def get_model_statistics(self) -> Dict:
        """Get statistics about the ILP model."""
        if self.model is None:
            return {}
        
        return {
            "num_variables": len(self.model.variables()),
            "num_constraints": len(self.model.constraints),
            "num_binary": sum(
                1 for v in self.model.variables()
                if v.cat == pulp.LpBinary
            ),
            "num_continuous": sum(
                1 for v in self.model.variables()
                if v.cat == pulp.LpContinuous
            )
        }
    
    def export_lp(self, filepath: str):
        """Export model to LP format file."""
        if self.model:
            self.model.writeLP(filepath)
    
    def export_mps(self, filepath: str):
        """Export model to MPS format file."""
        if self.model:
            self.model.writeMPS(filepath)
