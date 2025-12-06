"""
Pseudocode generation for algorithm documentation.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class AlgorithmDescription:
    """Complete description of an algorithm."""
    name: str
    category: str
    description: str
    pseudocode: str
    complexity: Dict[str, str]
    advantages: List[str]
    disadvantages: List[str]
    best_for: List[str]
    references: List[str]


class PseudocodeGenerator:
    """
    Generates pseudocode and algorithm descriptions.
    """
    
    ALGORITHMS = {
        "first_fit_decreasing": AlgorithmDescription(
            name="First Fit Decreasing (FFD)",
            category="Greedy",
            description="""
First Fit Decreasing is a classic bin packing heuristic that sorts items
by value in descending order and assigns each item to the first bin that
can accommodate it while minimizing the current imbalance.
            """,
            pseudocode="""
Algorithm: First Fit Decreasing (FFD)
Input: Items I with weights w[] and values v[], bins B, capacity W
Output: Assignment of items to bins

1.  SORT items by value in DESCENDING order
2.  INITIALIZE all bins as empty
3.  FOR each item i in sorted order DO
4.      best_bin ← NULL
5.      best_value ← ∞
6.      FOR each bin b in B DO
7.          IF weight(b) + w[i] ≤ W THEN
8.              IF value(b) < best_value THEN
9.                  best_bin ← b
10.                 best_value ← value(b)
11.             END IF
12.         END IF
13.     END FOR
14.     IF best_bin ≠ NULL THEN
15.         ASSIGN item i to best_bin
16.     ELSE
17.         REPORT infeasible
18.     END IF
19. END FOR
20. RETURN assignment
            """,
            complexity={
                "time": "O(n log n + n·k)",
                "space": "O(n + k)"
            },
            advantages=[
                "Simple to implement",
                "Fast execution",
                "Deterministic results",
                "Good for quick baseline"
            ],
            disadvantages=[
                "No optimality guarantee",
                "Can perform poorly on adversarial instances",
                "Sensitive to item ordering"
            ],
            best_for=[
                "Quick approximate solutions",
                "Large instances where exact methods fail",
                "Initial solutions for metaheuristics"
            ],
            references=[
                "Johnson, D. S. (1973). Near-optimal bin packing algorithms."
            ]
        ),
        
        "branch_and_bound": AlgorithmDescription(
            name="Branch and Bound",
            category="Exact",
            description="""
Branch and Bound is an exact algorithm that systematically explores the
solution space using a tree structure. It uses bounds to prune branches
that cannot lead to optimal solutions.
            """,
            pseudocode="""
Algorithm: Branch and Bound for Balanced Bin Packing
Input: Items I, bins B, capacity W
Output: Optimal assignment minimizing value difference

1.  SORT items by value in DESCENDING order
2.  best_solution ← GREEDY_SOLUTION()
3.  best_diff ← difference(best_solution)
4.  root ← CREATE_NODE(level=0, assignment=[], bounds)
5.  priority_queue ← {root}
6.  
7.  WHILE priority_queue NOT EMPTY DO
8.      node ← EXTRACT_MIN(priority_queue)  // by lower bound
9.      
10.     // Pruning
11.     IF node.lower_bound ≥ best_diff THEN
12.         CONTINUE  // Prune this branch
13.     END IF
14.     
15.     // Check if complete solution
16.     IF node.level = n THEN
17.         IF node.difference < best_diff THEN
18.             best_solution ← node.assignment
19.             best_diff ← node.difference
20.         END IF
21.         CONTINUE
22.     END IF
23.     
24.     // Branch: try assigning next item to each bin
25.     item ← items[node.level]
26.     FOR each bin b in B (sorted by current value) DO
27.         IF can_fit(b, item) THEN
28.             child ← CREATE_CHILD(node, item, b)
29.             child.lower_bound ← COMPUTE_LOWER_BOUND(child)
30.             IF child.lower_bound < best_diff THEN
31.                 INSERT(priority_queue, child)
32.             END IF
33.         END IF
34.     END FOR
35. END WHILE
36. 
37. RETURN best_solution

Function: COMPUTE_LOWER_BOUND(node)
    // Optimistic bound: assume remaining items can be distributed perfectly
    current_diff ← max(node.values) - min(node.values)
    remaining_value ← sum of unassigned item values
    
    // Lower bound is at least the current difference
    // minus the best possible reduction from remaining items
    RETURN max(0, current_diff - remaining_value)
            """,
            complexity={
                "time": "O(k^n) worst case",
                "space": "O(n·k)"
            },
            advantages=[
                "Guarantees optimal solution",
                "Can prove optimality",
                "Effective pruning for many instances"
            ],
            disadvantages=[
                "Exponential worst-case complexity",
                "Memory intensive for large trees",
                "Performance depends on bound quality"
            ],
            best_for=[
                "Small to medium instances (n ≤ 30)",
                "When optimality proof is required",
                "Instances with good structure"
            ],
            references=[
                "Land, A. H., & Doig, A. G. (1960). An automatic method for solving discrete programming problems."
            ]
        ),
        
        "simulated_annealing": AlgorithmDescription(
            name="Simulated Annealing",
            category="Metaheuristic",
            description="""
Simulated Annealing is inspired by the physical annealing process in
metallurgy. It probabilistically accepts worse solutions to escape
local optima, with the probability decreasing over time.
            """,
            pseudocode="""
Algorithm: Simulated Annealing for Balanced Bin Packing
Input: Items I, bins B, capacity W
Parameters: T_initial, T_min, cooling_rate α, max_iterations
Output: Best solution found

1.  current ← GREEDY_SOLUTION()
2.  best ← current
3.  T ← T_initial
4.  
5.  WHILE T > T_min AND iterations < max_iterations DO
6.      // Generate neighbor by move or swap
7.      neighbor ← GENERATE_NEIGHBOR(current)
8.      
9.      IF neighbor is feasible THEN
10.         ΔE ← f(neighbor) - f(current)  // f = value difference
11.         
12.         IF ΔE < 0 THEN
13.             // Better solution: always accept
14.             current ← neighbor
15.             IF f(current) < f(best) THEN
16.                 best ← current
17.             END IF
18.         ELSE
19.             // Worse solution: accept with probability
20.             probability ← exp(-ΔE / T)
21.             IF RANDOM() < probability THEN
22.                 current ← neighbor
23.             END IF
24.         END IF
25.     END IF
26.     
27.     // Cool down
28.     T ← T × α
29. END WHILE
30. 
31. RETURN best

Function: GENERATE_NEIGHBOR(solution)
    IF RANDOM() < 0.5 THEN
        // Move: relocate one item
        item ← RANDOM_ITEM(solution)
        source_bin ← bin containing item
        target_bin ← RANDOM_BIN(≠ source_bin)
        IF target_bin can fit item THEN
            MOVE item from source_bin to target_bin
        END IF
    ELSE
        // Swap: exchange items between bins
        bin1, bin2 ← TWO_RANDOM_BINS()
        item1 ← RANDOM_ITEM(bin1)
        item2 ← RANDOM_ITEM(bin2)
        IF swap is feasible THEN
            SWAP items
        END IF
    END IF
    RETURN modified solution
            """,
            complexity={
                "time": "O(max_iter · n)",
                "space": "O(n + k)"
            },
            advantages=[
                "Escapes local optima",
                "Simple to implement",
                "Few parameters to tune",
                "Anytime algorithm"
            ],
            disadvantages=[
                "No optimality guarantee",
                "Requires parameter tuning",
                "Convergence can be slow"
            ],
            best_for=[
                "Large instances",
                "When good solutions needed quickly",
                "Complex solution landscapes"
            ],
            references=[
                "Kirkpatrick, S., et al. (1983). Optimization by simulated annealing.",
                "Černý, V. (1985). Thermodynamical approach to the traveling salesman problem."
            ]
        ),
        
        "genetic_algorithm": AlgorithmDescription(
            name="Genetic Algorithm",
            category="Metaheuristic",
            description="""
Genetic Algorithms evolve a population of solutions using selection,
crossover, and mutation operators inspired by biological evolution.
            """,
            pseudocode="""
Algorithm: Genetic Algorithm for Balanced Bin Packing
Input: Items I, bins B, capacity W
Parameters: pop_size, generations, crossover_rate, mutation_rate
Output: Best solution found

1.  // Initialize population
2.  population ← []
3.  FOR i ← 1 to pop_size DO
4.      individual ← RANDOM_OR_GREEDY_SOLUTION()
5.      individual.fitness ← EVALUATE(individual)
6.      ADD individual to population
7.  END FOR
8.  
9.  best ← BEST_INDIVIDUAL(population)
10. 
11. FOR gen ← 1 to generations DO
12.     new_population ← []
13.     
14.     // Elitism: keep best individuals
15.     ADD TOP_K(population, elitism_count) to new_population
16.     
17.     WHILE |new_population| < pop_size DO
18.         // Selection (tournament)
19.         parent1 ← TOURNAMENT_SELECT(population)
20.         parent2 ← TOURNAMENT_SELECT(population)
21.         
22.         // Crossover
23.         IF RANDOM() < crossover_rate THEN
24.             child1, child2 ← CROSSOVER(parent1, parent2)
25.         ELSE
26.             child1, child2 ← COPY(parent1), COPY(parent2)
27.         END IF
28.         
29.         // Mutation
30.         MUTATE(child1, mutation_rate)
31.         MUTATE(child2, mutation_rate)
32.         
33.         // Repair infeasible solutions
34.         REPAIR(child1)
35.         REPAIR(child2)
36.         
37.         // Evaluate fitness
38.         child1.fitness ← EVALUATE(child1)
39.         child2.fitness ← EVALUATE(child2)
40.         
41.         ADD child1, child2 to new_population
42.     END WHILE
43.     
44.     population ← new_population
45.     
46.     IF BEST_INDIVIDUAL(population).fitness < best.fitness THEN
47.         best ← BEST_INDIVIDUAL(population)
48.     END IF
49. END FOR
50. 
51. RETURN best

Function: CROSSOVER(parent1, parent2)
    // Uniform crossover
    child1, child2 ← empty chromosomes
    FOR i ← 1 to n DO
        IF RANDOM() < 0.5 THEN
            child1[i] ← parent1[i]
            child2[i] ← parent2[i]
        ELSE
            child1[i] ← parent2[i]
            child2[i] ← parent1[i]
        END IF
    END FOR
    RETURN child1, child2

Function: MUTATE(individual, rate)
    FOR i ← 1 to n DO
        IF RANDOM() < rate THEN
            individual[i] ← RANDOM_BIN()
        END IF
    END FOR
            """,
            complexity={
                "time": "O(generations · pop_size · n)",
                "space": "O(pop_size · n)"
            },
            advantages=[
                "Explores diverse solutions",
                "Robust to local optima",
                "Parallelizable",
                "Handles complex constraints"
            ],
            disadvantages=[
                "Many parameters to tune",
                "Computationally expensive",
                "May converge prematurely"
            ],
            best_for=[
                "Very large instances",
                "Complex solution spaces",
                "When diversity is important"
            ],
            references=[
                "Holland, J. H. (1975). Adaptation in Natural and Artificial Systems.",
                "Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning."
            ]
        ),
        
        "tabu_search": AlgorithmDescription(
            name="Tabu Search",
            category="Metaheuristic",
            description="""
Tabu Search uses memory structures to avoid cycling and escape local
optima by forbidding recently visited solutions.
            """,
            pseudocode="""
Algorithm: Tabu Search for Balanced Bin Packing
Input: Items I, bins B, capacity W
Parameters: tabu_tenure, max_iterations
Output: Best solution found

1.  current ← GREEDY_SOLUTION()
2.  best ← current
3.  tabu_list ← {}  // Maps moves to iteration they become non-tabu
4.  
5.  FOR iter ← 1 to max_iterations DO
6.      // Generate all neighbors
7.      neighbors ← GENERATE_ALL_NEIGHBORS(current)
8.      
9.      best_neighbor ← NULL
10.     best_move ← NULL
11.     
12.     FOR each (neighbor, move) in neighbors DO
13.         is_tabu ← (move in tabu_list AND tabu_list[move] > iter)
14.         
15.         // Aspiration criterion: accept if better than best ever
16.         IF is_tabu AND f(neighbor) < f(best) THEN
17.             is_tabu ← FALSE
18.         END IF
19.         
20.         IF NOT is_tabu THEN
21.             IF best_neighbor = NULL OR f(neighbor) < f(best_neighbor) THEN
22.                 best_neighbor ← neighbor
23.                 best_move ← move
24.             END IF
25.         END IF
26.     END FOR
27.     
28.     // Move to best neighbor
29.     current ← best_neighbor
30.     
31.     // Add reverse move to tabu list
32.     reverse_move ← REVERSE(best_move)
33.     tabu_list[reverse_move] ← iter + tabu_tenure
34.     
35.     // Update best
36.     IF f(current) < f(best) THEN
37.         best ← current
38.     END IF
39.     
40.     // Clean expired tabu entries
41.     REMOVE entries from tabu_list where value ≤ iter
42. END FOR
43. 
44. RETURN best

Function: GENERATE_ALL_NEIGHBORS(solution)
    neighbors ← []
    FOR each item i in solution DO
        source_bin ← bin containing i
        FOR each bin b ≠ source_bin DO
            IF b can fit item i THEN
                neighbor ← COPY(solution)
                MOVE i from source_bin to b in neighbor
                move ← (i, source_bin, b)
                ADD (neighbor, move) to neighbors
            END IF
        END FOR
    END FOR
    RETURN neighbors
            """,
            complexity={
                "time": "O(max_iter · n · k)",
                "space": "O(tabu_tenure + n·k)"
            },
            advantages=[
                "Effective local search",
                "Avoids cycling",
                "Can escape local optima",
                "Deterministic with same seed"
            ],
            disadvantages=[
                "Tabu tenure needs tuning",
                "Can be slow for large neighborhoods",
                "Memory intensive for long tenure"
            ],
            best_for=[
                "Medium to large instances",
                "When local search gets stuck",
                "Fine-tuning good solutions"
            ],
            references=[
                "Glover, F. (1986). Future paths for integer programming and links to artificial intelligence.",
                "Glover, F., & Laguna, M. (1997). Tabu Search."
            ]
        ),
        
        "dynamic_programming": AlgorithmDescription(
            name="Dynamic Programming",
            category="Exact",
            description="""
Dynamic Programming solves the problem by breaking it into subproblems
and storing intermediate results. Only practical for small instances
due to exponential state space.
            """,
            pseudocode="""
Algorithm: Dynamic Programming for Balanced Bin Packing
Input: Items I, bins B, capacity W
Output: Optimal assignment (for small n)

1.  n ← |I|
2.  k ← |B|
3.  
4.  // Precompute feasible subsets
5.  feasible ← {}
6.  FOR mask ← 0 to 2^n - 1 DO
7.      weight ← 0
8.      value ← 0
9.      FOR i ← 0 to n-1 DO
10.         IF bit i is set in mask THEN
11.             weight ← weight + w[i]
12.             value ← value + v[i]
13.         END IF
14.     END FOR
15.     IF weight ≤ W THEN
16.         feasible[mask] ← (weight, value)
17.     END IF
18. END FOR
19. 
20. // DP: dp[j][mask] = best (max_val, min_val) using j bins for items in mask
21. dp ← array of size [k+1][2^n]
22. 
23. // Base case: 1 bin
24. FOR mask in feasible DO
25.     dp[1][mask] ← (feasible[mask].value, feasible[mask].value, [mask])
26. END FOR
27. 
28. // Fill DP table
29. FOR j ← 2 to k DO
30.     FOR prev_mask in dp[j-1] DO
31.         remaining ← FULL_MASK XOR prev_mask
32.         
33.         // Try all subsets of remaining items
34.         subset ← remaining
35.         WHILE subset > 0 DO
36.             IF subset in feasible THEN
37.                 new_mask ← prev_mask OR subset
38.                 new_val ← feasible[subset].value
39.                 
40.                 prev_max, prev_min, prev_assign ← dp[j-1][prev_mask]
41.                 new_max ← max(prev_max, new_val)
42.                 new_min ← min(prev_min, new_val)
43.                 new_diff ← new_max - new_min
44.                 
45.                 IF new_mask not in dp[j] OR new_diff < current_diff THEN
46.                     dp[j][new_mask] ← (new_max, new_min, prev_assign + [subset])
47.                 END IF
48.             END IF
49.             subset ← (subset - 1) AND remaining
50.         END WHILE
51.     END FOR
52. END FOR
53. 
54. // Extract solution
55. FULL_MASK ← 2^n - 1
56. IF FULL_MASK in dp[k] THEN
57.     RETURN dp[k][FULL_MASK].assignment
58. ELSE
59.     RETURN infeasible
60. END IF
            """,
            complexity={
                "time": "O(k · 3^n)",
                "space": "O(k · 2^n)"
            },
            advantages=[
                "Guarantees optimal solution",
                "Systematic exploration",
                "Can be adapted for variants"
            ],
            disadvantages=[
                "Exponential complexity",
                "Only for small instances (n ≤ 20)",
                "High memory requirements"
            ],
            best_for=[
                "Small instances where optimality needed",
                "Benchmarking other algorithms",
                "Understanding problem structure"
            ],
            references=[
                "Bellman, R. (1957). Dynamic Programming.",
                "Korf, R. E. (1998). A complete anytime algorithm for number partitioning."
            ]
        )
    }
    
    @classmethod
    def get_pseudocode(cls, algorithm_name: str) -> str:
        """Get pseudocode for an algorithm."""
        if algorithm_name in cls.ALGORITHMS:
            return cls.ALGORITHMS[algorithm_name].pseudocode
        return "Pseudocode not available"
    
    @classmethod
    def get_description(cls, algorithm_name: str) -> AlgorithmDescription:
        """Get full description of an algorithm."""
        return cls.ALGORITHMS.get(algorithm_name)
    
    @classmethod
    def get_all_pseudocodes(cls) -> Dict[str, str]:
        """Get pseudocodes for all algorithms."""
        return {name: algo.pseudocode for name, algo in cls.ALGORITHMS.items()}
    
    @classmethod
    def generate_latex_document(cls) -> str:
        """Generate LaTeX document with all algorithms."""
        latex = r"""
\documentclass{article}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{amsmath}

\title{Algorithm Pseudocode Documentation}
\author{DAA Project}
\date{\today}

\begin{document}
\maketitle

\tableofcontents
\newpage

"""
        for name, algo in cls.ALGORITHMS.items():
            latex += f"""
\\section{{{algo.name}}}
\\subsection{{Category}}
{algo.category}

\\subsection{{Description}}
{algo.description}

\\subsection{{Complexity}}
\\begin{{itemize}}
    \\item Time: {algo.complexity['time']}
    \\item Space: {algo.complexity['space']}
\\end{{itemize}}

\\subsection{{Pseudocode}}
\\begin{{verbatim}}
{algo.pseudocode}
\\end{{verbatim}}

\\subsection{{Advantages}}
\\begin{{itemize}}
"""
            for adv in algo.advantages:
                latex += f"    \\item {adv}\n"
            
            latex += """\\end{itemize}

\\subsection{Disadvantages}
\\begin{itemize}
"""
            for dis in algo.disadvantages:
                latex += f"    \\item {dis}\n"
            
            latex += """\\end{itemize}

\\subsection{Best For}
\\begin{itemize}
"""
            for best in algo.best_for:
                latex += f"    \\item {best}\n"
            
            latex += """\\end{itemize}

\\subsection{References}
\\begin{itemize}
"""
            for ref in algo.references:
                latex += f"    \\item {ref}\n"
            
            latex += "\\end{itemize}\n\n"
        
        latex += "\\end{document}"
        return latex
