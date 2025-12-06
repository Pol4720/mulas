"""
Complexity Analysis and NP-Hardness Proofs.

Provides formal complexity analysis for the Balanced Multi-Bin Packing Problem.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ComplexityClass:
    """Represents a computational complexity class."""
    name: str
    description: str
    contains: List[str]
    
    def __repr__(self) -> str:
        return f"ComplexityClass({self.name})"


# Standard complexity classes
P = ComplexityClass(
    "P",
    "Problems solvable in polynomial time by a deterministic Turing machine",
    ["SORTING", "SHORTEST_PATH", "MST", "LINEAR_PROGRAMMING"]
)

NP = ComplexityClass(
    "NP",
    "Problems verifiable in polynomial time by a deterministic Turing machine",
    ["SAT", "CLIQUE", "VERTEX_COVER", "PARTITION", "BIN_PACKING", "BALANCED_BIN_PACKING"]
)

NP_HARD = ComplexityClass(
    "NP-hard",
    "Problems at least as hard as the hardest problems in NP",
    ["SAT", "TSP", "PARTITION", "BIN_PACKING", "BALANCED_BIN_PACKING"]
)

NP_COMPLETE = ComplexityClass(
    "NP-complete",
    "Problems that are both in NP and NP-hard",
    ["SAT", "3SAT", "PARTITION", "SUBSET_SUM", "BIN_PACKING"]
)


class ComplexityAnalysis:
    """
    Comprehensive complexity analysis for algorithms.
    """
    
    ALGORITHMS = {
        "first_fit_decreasing": {
            "name": "First Fit Decreasing",
            "time": "O(n log n + n·k)",
            "space": "O(n + k)",
            "best_case": "O(n log n)",
            "worst_case": "O(n log n + n·k)",
            "average_case": "O(n log n + n·k)",
            "approximation": "No guaranteed bound for balance objective",
            "description": """
Time Complexity Analysis:
- Sorting: O(n log n) using comparison-based sort
- Assignment loop: O(n) iterations
- Each iteration: O(k) to find best bin
- Total: O(n log n + n·k)

Space Complexity Analysis:
- Input storage: O(n) for items
- Bin storage: O(k) for bins
- Sorted list: O(n) - can be done in-place
- Total: O(n + k)
            """
        },
        "best_fit_decreasing": {
            "name": "Best Fit Decreasing",
            "time": "O(n log n + n·k)",
            "space": "O(n + k)",
            "best_case": "O(n log n)",
            "worst_case": "O(n log n + n·k)",
            "average_case": "O(n log n + n·k)",
            "approximation": "No guaranteed bound",
            "description": """
Time Complexity Analysis:
- Similar to FFD but evaluates resulting balance for each bin
- Sorting: O(n log n)
- For each item: evaluate all k bins, O(k) per item
- Total: O(n log n + n·k)
            """
        },
        "worst_fit_decreasing": {
            "name": "Worst Fit Decreasing",
            "time": "O(n log n + n log k)",
            "space": "O(n + k)",
            "best_case": "O(n log n)",
            "worst_case": "O(n log n + n log k)",
            "average_case": "O(n log n + n log k)",
            "approximation": "No guaranteed bound",
            "description": """
Time Complexity Analysis:
- Sorting: O(n log n)
- Using max-heap for bins: O(log k) per insertion/update
- n items × O(log k) operations
- Total: O(n log n + n log k)
            """
        },
        "round_robin_greedy": {
            "name": "Round Robin Greedy (LPT)",
            "time": "O(n log n + n log k)",
            "space": "O(n + k)",
            "best_case": "O(n log n)",
            "worst_case": "O(n log n + n log k)",
            "average_case": "O(n log n + n log k)",
            "approximation": "4/3 - 1/(3k) for makespan",
            "description": """
Time Complexity Analysis:
- Sorting: O(n log n)
- Using min-heap for bin values: O(log k) per operation
- n items × O(log k) operations
- Total: O(n log n + n log k)

Approximation Ratio (LPT):
- For makespan scheduling: 4/3 - 1/(3m)
- Graham (1969) proved this bound
- For balance objective: no tight bound known
            """
        },
        "largest_difference_first": {
            "name": "Largest Difference First",
            "time": "O(n² · k)",
            "space": "O(n + k)",
            "best_case": "O(n · k)",
            "worst_case": "O(n² · k)",
            "average_case": "O(n² · k)",
            "approximation": "No guaranteed bound",
            "description": """
Time Complexity Analysis:
- For each of n items (outer loop): O(n)
- Evaluate all unassigned items: O(n - i) at step i
- For each item, check all k bins: O(k)
- Total: O(n · n · k) = O(n² · k)
            """
        },
        "dynamic_programming": {
            "name": "Dynamic Programming",
            "time": "O(k · 3^n)",
            "space": "O(k · 2^n)",
            "best_case": "O(2^n)",
            "worst_case": "O(k · 3^n)",
            "average_case": "O(k · 3^n)",
            "approximation": "Optimal (exact)",
            "description": """
Time Complexity Analysis:
- Enumerate all subsets: O(2^n)
- For k-partition: combine subsets
- Each subset can go to k bins
- Worst case: O(k · 3^n) for exhaustive DP

Space Complexity Analysis:
- Memoization table: O(k · 2^n) entries
- Each entry stores bin values and assignments

Note: Only practical for small instances (n ≤ 20)
            """
        },
        "branch_and_bound": {
            "name": "Branch and Bound",
            "time": "O(k^n) worst case",
            "space": "O(n · k)",
            "best_case": "Polynomial with good pruning",
            "worst_case": "O(k^n)",
            "average_case": "Depends on instance and bounds",
            "approximation": "Optimal (exact)",
            "description": """
Time Complexity Analysis:
- Search tree has k^n nodes in worst case
- Each node: O(k) work for branching
- Pruning reduces effective search space
- Best-first search prioritizes promising nodes

Pruning Strategies:
1. Feasibility: O(1) check
2. Bound comparison: O(1)
3. Symmetry breaking: reduces by factor of k!
4. Dominance: O(k) check

Effective complexity depends heavily on:
- Quality of initial bound (greedy solution)
- Tightness of lower bounds
- Problem structure
            """
        },
        "simulated_annealing": {
            "name": "Simulated Annealing",
            "time": "O(max_iter · n · k)",
            "space": "O(n + k)",
            "best_case": "O(max_iter)",
            "worst_case": "O(max_iter · n · k)",
            "average_case": "O(max_iter · n)",
            "approximation": "No theoretical bound",
            "description": """
Time Complexity Analysis:
- max_iter iterations of cooling loop
- Each iteration: generate neighbor O(n)
- Evaluate neighbor: O(k)
- Total: O(max_iter · n)

Convergence Analysis:
- Temperature T decreases: T_{i+1} = α · T_i
- After m iterations: T_m = T_0 · α^m
- Reaches T_min when m = log(T_min/T_0) / log(α)

Acceptance probability:
- P(accept) = exp(-ΔE / T)
- As T → 0, only improvements accepted
            """
        },
        "genetic_algorithm": {
            "name": "Genetic Algorithm",
            "time": "O(generations · pop_size · n)",
            "space": "O(pop_size · n)",
            "best_case": "O(generations · pop_size)",
            "worst_case": "O(generations · pop_size · n · k)",
            "average_case": "O(generations · pop_size · n)",
            "approximation": "No theoretical bound",
            "description": """
Time Complexity Analysis:
- generations iterations
- Population size: pop_size
- Each generation:
  * Selection: O(pop_size · tournament_size)
  * Crossover: O(n) per pair
  * Mutation: O(n) per individual
  * Fitness evaluation: O(n) per individual
  * Repair: O(n · k) worst case
- Total: O(generations · pop_size · n)

Population Diversity:
- Helps escape local optima
- Diversity calculation: O(pop_size² · n)
            """
        },
        "tabu_search": {
            "name": "Tabu Search",
            "time": "O(max_iter · n · k)",
            "space": "O(tabu_tenure + n · k)",
            "best_case": "O(max_iter)",
            "worst_case": "O(max_iter · n · k)",
            "average_case": "O(max_iter · n · k)",
            "approximation": "No theoretical bound",
            "description": """
Time Complexity Analysis:
- max_iter iterations
- Neighborhood generation: O(n · k) moves possible
- Tabu check: O(1) with hash table
- Best non-tabu selection: O(n · k)
- Total: O(max_iter · n · k)

Memory Structure:
- Tabu list: O(tabu_tenure) entries
- Frequency memory: O(n · k) entries
            """
        },
        "lpt_approximation": {
            "name": "LPT Approximation",
            "time": "O(n log n + n log k)",
            "space": "O(n + k)",
            "best_case": "O(n log n)",
            "worst_case": "O(n log n + n log k)",
            "average_case": "O(n log n + n log k)",
            "approximation": "4/3 - 1/(3k) for makespan",
            "description": """
LPT (Longest Processing Time) Analysis:

Time Complexity:
- Sort by value: O(n log n)
- Heap operations: O(n log k)
- Total: O(n log n + n log k)

Approximation Ratio (Graham 1969):
- For makespan (max value): ratio ≤ 4/3 - 1/(3k)
- This is tight: examples exist achieving this ratio
- For balance (max - min): no tight bound known

Proof Sketch (Makespan):
Let OPT be optimal makespan
Let ALG be LPT makespan
Let j* be the bin with maximum value after LPT
Let i be the last item assigned to j*

Case 1: v_i ≤ OPT/3
  ALG ≤ (sum of values)/k + v_i ≤ OPT + OPT/3 = 4/3 OPT

Case 2: v_i > OPT/3
  At most 2 items per bin in OPT
  LPT places items optimally in this case
            """
        },
        "multiway_partition": {
            "name": "Multi-Way Partition (KK)",
            "time": "O(n log n)",
            "space": "O(n)",
            "best_case": "O(n log n)",
            "worst_case": "O(n log n)",
            "average_case": "O(n log n)",
            "approximation": "O(1/n^θ(log n)) for k=2",
            "description": """
Karmarkar-Karp Differencing Analysis:

For k=2 (Two-Way Partition):
- Heap operations: O(n log n)
- Approximation: remarkably good

Approximation Quality (k=2):
- Expected difference: O(n^(-θ(log n)))
- This is pseudo-polynomial in the best case
- Much better than greedy methods empirically

For k>2:
- Extended differencing or greedy
- No tight bounds available
- Heuristic quality depends on instance

Historical Note:
- Karmarkar & Karp (1982)
- Significant improvement over previous methods
- Related to the complete anytime algorithm
            """
        }
    }
    
    @classmethod
    def get_algorithm_complexity(cls, algorithm_name: str) -> Dict:
        """Get complexity information for an algorithm."""
        return cls.ALGORITHMS.get(algorithm_name, {})
    
    @classmethod
    def get_all_complexities(cls) -> Dict:
        """Get complexity information for all algorithms."""
        return cls.ALGORITHMS
    
    @classmethod
    def compare_complexities(cls, algorithms: List[str]) -> str:
        """Generate complexity comparison table in LaTeX."""
        latex = r"""
\begin{table}[h]
\centering
\caption{Algorithm Complexity Comparison}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Algorithm} & \textbf{Time} & \textbf{Space} & \textbf{Approx. Ratio} \\
\hline
"""
        for algo in algorithms:
            if algo in cls.ALGORITHMS:
                info = cls.ALGORITHMS[algo]
                latex += f"{info['name']} & {info['time']} & {info['space']} & {info['approximation']} \\\\\n"
        
        latex += r"""
\hline
\end{tabular}
\end{table}
"""
        return latex


class NPHardnessProof:
    """
    NP-Hardness proof for the Balanced Multi-Bin Packing Problem.
    """
    
    @staticmethod
    def get_proof() -> str:
        """
        Return the NP-hardness proof.
        
        The proof is by reduction from PARTITION.
        """
        return r"""
\section{NP-Hardness Proof}

\subsection{Theorem}
The Balanced Multi-Bin Packing Problem is NP-hard.

\subsection{Proof}
We prove NP-hardness by polynomial-time reduction from the PARTITION problem,
which is known to be NP-complete (Karp, 1972).

\subsubsection{PARTITION Problem}
\textbf{Instance:} A set $S = \{a_1, a_2, \ldots, a_n\}$ of positive integers.

\textbf{Question:} Can $S$ be partitioned into two subsets $S_1, S_2$ such that
$\sum_{a_i \in S_1} a_i = \sum_{a_j \in S_2} a_j$?

\subsubsection{Reduction}
Given an instance of PARTITION with set $S = \{a_1, \ldots, a_n\}$,
we construct an instance of Balanced Multi-Bin Packing as follows:

\begin{itemize}
    \item Number of items: $n$
    \item Number of bins: $k = 2$
    \item For each $a_i \in S$:
        \begin{itemize}
            \item Weight $w_i = a_i$
            \item Value $v_i = a_i$
        \end{itemize}
    \item Bin capacity: $W = \sum_{i=1}^{n} a_i$
\end{itemize}

\subsubsection{Correctness}
\textbf{($\Rightarrow$)} If PARTITION has a solution $(S_1, S_2)$:
\begin{itemize}
    \item Assign items corresponding to $S_1$ to bin 1
    \item Assign items corresponding to $S_2$ to bin 2
    \item Both bins have value $\frac{1}{2}\sum a_i$
    \item Value difference = 0 (optimal)
\end{itemize}

\textbf{($\Leftarrow$)} If Balanced Bin Packing has solution with difference 0:
\begin{itemize}
    \item Both bins have equal total value
    \item Since $v_i = a_i$, the items form equal-sum subsets
    \item This is a valid PARTITION solution
\end{itemize}

\subsubsection{Polynomial Time}
The reduction takes $O(n)$ time:
\begin{itemize}
    \item Copy $n$ values
    \item Set $k = 2$
    \item Compute capacity sum in $O(n)$
\end{itemize}

\subsection{Corollary}
Since PARTITION is strongly NP-complete, and our reduction preserves
the number encoding, Balanced Multi-Bin Packing is also NP-hard in the
strong sense when the number of bins is part of the input.

\subsection{Implications}
\begin{enumerate}
    \item No polynomial-time exact algorithm exists unless P = NP
    \item Approximation algorithms and heuristics are necessary for large instances
    \item Pseudo-polynomial algorithms exist (e.g., DP) for bounded input values
    \item For fixed $k$, the problem may have better approximation ratios
\end{enumerate}

\subsection{Related Hardness Results}
\begin{itemize}
    \item \textbf{3-PARTITION}: Strongly NP-complete (Garey \& Johnson, 1979)
    \item \textbf{Bin Packing}: NP-hard to approximate within $3/2 - \epsilon$
    \item \textbf{Multiprocessor Scheduling}: NP-hard, 4/3-approximable
\end{itemize}
"""
    
    @staticmethod
    def get_membership_proof() -> str:
        """
        Return proof that the decision version is in NP.
        """
        return r"""
\section{NP Membership}

\subsection{Theorem}
The decision version of Balanced Multi-Bin Packing is in NP.

\subsection{Decision Version}
\textbf{Instance:} Items $I$, bins $B$, capacity $W$, target difference $D$.

\textbf{Question:} Does there exist an assignment of items to bins such that:
\begin{enumerate}
    \item Each item is assigned to exactly one bin
    \item No bin exceeds capacity $W$
    \item The difference between max and min bin values is at most $D$
\end{enumerate}

\subsection{Proof}
\textbf{Certificate:} An assignment function $f: I \rightarrow B$

\textbf{Verification Algorithm:}
\begin{enumerate}
    \item Check each item is assigned: $O(n)$
    \item Compute bin weights and check capacity: $O(n + k)$
    \item Compute bin values: $O(n)$
    \item Find max and min values: $O(k)$
    \item Check difference $\leq D$: $O(1)$
\end{enumerate}

Total verification time: $O(n + k)$ = polynomial in input size.

\subsection{Conclusion}
Therefore, Balanced Multi-Bin Packing (decision) is in NP,
and combined with NP-hardness, the decision version is NP-complete.
"""
    
    @staticmethod
    def get_full_latex_document() -> str:
        """Return complete LaTeX document with all proofs."""
        return r"""
\documentclass{article}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm, algorithmic}

\newtheorem{theorem}{Theorem}
\newtheorem{corollary}{Corollary}
\newtheorem{lemma}{Lemma}
\newtheorem{definition}{Definition}

\title{Complexity Analysis of the Balanced Multi-Bin Packing Problem}
\author{DAA Project}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
We analyze the computational complexity of the Balanced Multi-Bin Packing
problem, proving its NP-hardness via reduction from PARTITION and
establishing its membership in NP.
\end{abstract}

""" + NPHardnessProof.get_proof() + NPHardnessProof.get_membership_proof() + r"""

\section{Bibliography}
\begin{thebibliography}{9}
\bibitem{garey1979}
Garey, M. R., \& Johnson, D. S. (1979).
\textit{Computers and Intractability: A Guide to the Theory of NP-Completeness}.
W. H. Freeman.

\bibitem{graham1969}
Graham, R. L. (1969).
Bounds on multiprocessing timing anomalies.
\textit{SIAM Journal on Applied Mathematics}, 17(2), 416-429.

\bibitem{karmarkar1982}
Karmarkar, N., \& Karp, R. M. (1982).
The differencing method of set partitioning.
\textit{Technical Report UCB/CSD-82-113}, UC Berkeley.

\bibitem{karp1972}
Karp, R. M. (1972).
Reducibility among combinatorial problems.
In \textit{Complexity of Computer Computations} (pp. 85-103). Springer.
\end{thebibliography}

\end{document}
"""
