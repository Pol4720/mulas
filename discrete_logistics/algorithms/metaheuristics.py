"""
Metaheuristic Algorithms for Balanced Multi-Bin Packing.

Implements:
- Simulated Annealing (SA)
- Genetic Algorithm (GA)
- Tabu Search (TS)

These algorithms provide high-quality solutions for larger instances
where exact methods are impractical.
"""

from typing import List, Dict, Tuple, Optional, Set, Callable
from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy
import random
import math

try:
    from ..core.problem import Problem, Solution, Bin, Item
    from .base import Algorithm, register_algorithm
except ImportError:
    from discrete_logistics.core.problem import Problem, Solution, Bin, Item
    from discrete_logistics.algorithms.base import Algorithm, register_algorithm


# ============================================================================
# Simulated Annealing
# ============================================================================

@register_algorithm("simulated_annealing")
class SimulatedAnnealing(Algorithm):
    """
    Simulated Annealing for Balanced Multi-Bin Packing.
    
    Probabilistically accepts worse solutions to escape local optima,
    with probability decreasing over time (cooling schedule).
    
    Complexity Analysis:
    - Time: O(max_iterations * neighborhood_size)
    - Space: O(n + k)
    - Approximation: No guaranteed bound, but often near-optimal
    
    Pseudocode:
    1. Generate initial solution (greedy)
    2. Set initial temperature T = T_initial
    3. While T > T_min and iterations < max_iterations:
       a. Generate neighbor solution by moving/swapping items
       b. Calculate ΔE = f(neighbor) - f(current)
       c. If ΔE < 0, accept neighbor
       d. Else accept with probability exp(-ΔE/T)
       e. Cool down: T = T * cooling_rate
    4. Return best solution found
    """
    
    time_complexity = "O(max_iter · neighborhood_size)"
    space_complexity = "O(n + k)"
    approximation_ratio = "No theoretical bound"
    description = "Probabilistic local search with temperature-based acceptance"
    
    def __init__(
        self,
        track_steps: bool = False,
        verbose: bool = False,
        initial_temp: float = 1000.0,
        min_temp: float = 0.01,
        cooling_rate: float = 0.995,
        max_iterations: int = 10000,
        neighborhood_size: int = 5
    ):
        """
        Initialize Simulated Annealing.
        
        Args:
            initial_temp: Starting temperature
            min_temp: Minimum temperature (stopping criterion)
            cooling_rate: Temperature decay factor (0 < α < 1)
            max_iterations: Maximum iterations
            neighborhood_size: Number of neighbors to consider per iteration
        """
        super().__init__(track_steps, verbose)
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.neighborhood_size = neighborhood_size
        
        # Statistics
        self.accepted_moves = 0
        self.rejected_moves = 0
        self.temperature_history = []
        self.objective_history = []
    
    @property
    def name(self) -> str:
        return "Simulated Annealing"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting SA: T0={self.initial_temp}, α={self.cooling_rate}")
        
        # Get initial solution from greedy
        from .greedy import RoundRobinGreedy
        greedy = RoundRobinGreedy()
        current = greedy.solve(problem)
        
        best = current.copy()
        best.algorithm_name = self.name
        
        self._log(f"Initial solution: diff={current.value_difference:.2f}")
        
        T = self.initial_temp
        self.accepted_moves = 0
        self.rejected_moves = 0
        self.temperature_history = [T]
        self.objective_history = [current.value_difference]
        
        self._record_step(
            f"Initial: diff={current.value_difference:.2f}",
            current.bins,
            extra_data={"temperature": T}
        )
        
        iteration = 0
        while T > self.min_temp and iteration < self.max_iterations:
            iteration += 1
            self._iterations += 1
            
            # Generate neighbor
            neighbor = self._get_neighbor(current, problem)
            
            if neighbor is None:
                continue
            
            # Calculate energy difference
            current_energy = current.value_difference
            neighbor_energy = neighbor.value_difference
            delta_e = neighbor_energy - current_energy
            
            # Accept or reject
            if delta_e < 0:
                # Better solution: always accept
                current = neighbor
                self.accepted_moves += 1
                
                if current.value_difference < best.value_difference:
                    best = current.copy()
                    best.algorithm_name = self.name
                    self._log(f"New best: {best.value_difference:.2f} at T={T:.2f}")
                    
                    self._record_step(
                        f"New best: diff={best.value_difference:.2f}",
                        best.bins,
                        extra_data={"temperature": T, "iteration": iteration}
                    )
            else:
                # Worse solution: accept with probability
                prob = math.exp(-delta_e / T)
                if random.random() < prob:
                    current = neighbor
                    self.accepted_moves += 1
                else:
                    self.rejected_moves += 1
            
            # Cool down
            T *= self.cooling_rate
            
            # Record history
            self.temperature_history.append(T)
            self.objective_history.append(current.value_difference)
            
            # Periodic logging
            if iteration % 1000 == 0:
                self._log(f"Iter {iteration}: T={T:.4f}, "
                         f"current={current.value_difference:.2f}, "
                         f"best={best.value_difference:.2f}")
        
        best.execution_time = self._get_elapsed_time()
        best.iterations = iteration
        best.metadata.update({
            "final_temperature": T,
            "accepted_moves": self.accepted_moves,
            "rejected_moves": self.rejected_moves,
            "acceptance_rate": self.accepted_moves / max(1, self.accepted_moves + self.rejected_moves)
        })
        
        self._log(f"Completed: {iteration} iterations, best={best.value_difference:.2f}")
        
        return best
    
    def _get_neighbor(self, solution: Solution, problem: Problem) -> Optional[Solution]:
        """Generate a neighbor solution by moving or swapping items."""
        neighbor = solution.copy()
        bins = neighbor.bins
        
        # Choose move type
        move_type = random.choice(["move", "swap"])
        
        if move_type == "move":
            return self._move_item(neighbor, problem.bin_capacities)
        else:
            return self._swap_items(neighbor, problem.bin_capacities)
    
    def _move_item(self, solution: Solution, capacities: List[float]) -> Optional[Solution]:
        """Move a random item from one bin to another."""
        bins = solution.bins
        
        # Find bins with items
        source_bins = [b for b in bins if b.items]
        if not source_bins:
            return None
        
        source = random.choice(source_bins)
        if not source.items:
            return None
        
        item = random.choice(source.items)
        
        # Find feasible target bins (considering individual capacities)
        target_bins = [b for b in bins if b.id != source.id and 
                      b.current_weight + item.weight <= capacities[b.id]]
        if not target_bins:
            return None
        
        target = random.choice(target_bins)
        
        # Execute move
        source.remove_item(item)
        target.add_item(item)
        
        return solution
    
    def _swap_items(self, solution: Solution, capacities: List[float]) -> Optional[Solution]:
        """Swap items between two bins."""
        bins = solution.bins
        
        # Find two bins with items
        bins_with_items = [b for b in bins if b.items]
        if len(bins_with_items) < 2:
            return None
        
        bin1, bin2 = random.sample(bins_with_items, 2)
        
        item1 = random.choice(bin1.items)
        item2 = random.choice(bin2.items)
        
        # Check feasibility (using individual capacities)
        new_weight1 = bin1.current_weight - item1.weight + item2.weight
        new_weight2 = bin2.current_weight - item2.weight + item1.weight
        
        if new_weight1 > capacities[bin1.id] or new_weight2 > capacities[bin2.id]:
            return None
        
        # Execute swap
        bin1.remove_item(item1)
        bin2.remove_item(item2)
        bin1.add_item(item2)
        bin2.add_item(item1)
        
        return solution
    
    def get_convergence_data(self) -> Dict:
        """Get convergence data for plotting."""
        return {
            "temperatures": self.temperature_history,
            "objectives": self.objective_history,
            "accepted_moves": self.accepted_moves,
            "rejected_moves": self.rejected_moves
        }


# ============================================================================
# Genetic Algorithm
# ============================================================================

@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population."""
    chromosome: List[int]  # chromosome[i] = bin assignment for item i
    fitness: float = float('inf')
    
    def __lt__(self, other: 'Individual') -> bool:
        return self.fitness < other.fitness


@register_algorithm("genetic_algorithm")
class GeneticAlgorithm(Algorithm):
    """
    Genetic Algorithm for Balanced Multi-Bin Packing.
    
    Evolves a population of solutions using selection, crossover,
    and mutation operators.
    
    Complexity Analysis:
    - Time: O(generations * population_size * n)
    - Space: O(population_size * n)
    - Approximation: No guaranteed bound
    
    Pseudocode:
    1. Initialize population with random/greedy solutions
    2. Evaluate fitness of all individuals
    3. For each generation:
       a. Select parents using tournament selection
       b. Apply crossover to create offspring
       c. Apply mutation to offspring
       d. Repair infeasible solutions
       e. Evaluate fitness
       f. Select survivors for next generation
    4. Return best individual
    """
    
    time_complexity = "O(generations · pop_size · n)"
    space_complexity = "O(pop_size · n)"
    approximation_ratio = "No theoretical bound"
    description = "Evolutionary algorithm with crossover and mutation"
    
    def __init__(
        self,
        track_steps: bool = False,
        verbose: bool = False,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        elitism_count: int = 2
    ):
        """
        Initialize Genetic Algorithm.
        
        Args:
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation per gene
            tournament_size: Size of tournament for selection
            elitism_count: Number of best individuals to preserve
        """
        super().__init__(track_steps, verbose)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        
        self.fitness_history = []
        self.diversity_history = []
    
    @property
    def name(self) -> str:
        return "Genetic Algorithm"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting GA: pop={self.population_size}, gen={self.generations}")
        
        n = problem.n_items
        k = problem.num_bins
        items = problem.items
        capacities = problem.bin_capacities
        
        # Initialize population
        population = self._initialize_population(problem)
        
        # Evaluate initial fitness
        for ind in population:
            ind.fitness = self._evaluate_fitness(ind, items, k, capacities)
        
        population.sort()
        best = population[0]
        
        self._log(f"Initial best fitness: {best.fitness:.2f}")
        self.fitness_history = [best.fitness]
        
        self._record_step(
            f"Initial population: best={best.fitness:.2f}",
            self._chromosome_to_bins(best.chromosome, items, k, capacities),
            extra_data={"generation": 0}
        )
        
        # Evolution loop
        for gen in range(self.generations):
            self._iterations += 1
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            new_population.extend(population[:self.elitism_count])
            
            # Fill rest with offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, k)
                else:
                    child1 = Individual(parent1.chromosome.copy())
                    child2 = Individual(parent2.chromosome.copy())
                
                # Mutation
                self._mutate(child1, k)
                self._mutate(child2, k)
                
                # Repair infeasible solutions
                self._repair(child1, items, k, capacities)
                self._repair(child2, items, k, capacities)
                
                # Evaluate fitness
                child1.fitness = self._evaluate_fitness(child1, items, k, capacities)
                child2.fitness = self._evaluate_fitness(child2, items, k, capacities)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Update population
            population = sorted(new_population)[:self.population_size]
            
            # Update best
            if population[0].fitness < best.fitness:
                best = population[0]
                self._log(f"Gen {gen}: New best = {best.fitness:.2f}")
                
                self._record_step(
                    f"Gen {gen}: New best = {best.fitness:.2f}",
                    self._chromosome_to_bins(best.chromosome, items, k, capacities),
                    extra_data={"generation": gen}
                )
            
            self.fitness_history.append(best.fitness)
            
            # Calculate diversity
            diversity = self._calculate_diversity(population)
            self.diversity_history.append(diversity)
            
            if gen % 20 == 0:
                self._log(f"Gen {gen}: best={best.fitness:.2f}, "
                         f"avg={np.mean([i.fitness for i in population]):.2f}, "
                         f"div={diversity:.2f}")
        
        # Build solution
        solution = Solution(
            bins=self._chromosome_to_bins(best.chromosome, items, k, capacities),
            algorithm_name=self.name,
            execution_time=self._get_elapsed_time(),
            iterations=self._iterations,
            metadata={
                "final_fitness": best.fitness,
                "generations": self.generations,
                "population_size": self.population_size
            }
        )
        
        return solution
    
    def _initialize_population(self, problem: Problem) -> List[Individual]:
        """Initialize population with mix of greedy and random solutions."""
        population = []
        n = problem.n_items
        k = problem.num_bins
        
        # Add greedy solutions
        from .greedy import FirstFitDecreasing, BestFitDecreasing, RoundRobinGreedy
        
        for greedy_class in [FirstFitDecreasing, BestFitDecreasing, RoundRobinGreedy]:
            greedy = greedy_class()
            sol = greedy.solve(problem)
            chromosome = self._solution_to_chromosome(sol, n)
            population.append(Individual(chromosome))
        
        # Fill rest with random solutions
        while len(population) < self.population_size:
            chromosome = [random.randint(0, k - 1) for _ in range(n)]
            population.append(Individual(chromosome))
        
        return population
    
    def _evaluate_fitness(
        self,
        individual: Individual,
        items: List[Item],
        k: int,
        capacities: List[float]
    ) -> float:
        """
        Evaluate fitness of an individual.
        
        Fitness = value_difference + penalty for constraint violations
        """
        bins = self._chromosome_to_bins(individual.chromosome, items, k, capacities)
        
        # Calculate value difference
        values = [b.current_value for b in bins]
        diff = max(values) - min(values)
        
        # Penalty for weight violations
        penalty = 0
        for b in bins:
            if b.current_weight > b.capacity:
                penalty += (b.current_weight - b.capacity) * 100
        
        return diff + penalty
    
    def _tournament_select(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return min(tournament, key=lambda x: x.fitness)
    
    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        k: int
    ) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        n = len(parent1.chromosome)
        child1_chrom = []
        child2_chrom = []
        
        for i in range(n):
            if random.random() < 0.5:
                child1_chrom.append(parent1.chromosome[i])
                child2_chrom.append(parent2.chromosome[i])
            else:
                child1_chrom.append(parent2.chromosome[i])
                child2_chrom.append(parent1.chromosome[i])
        
        return Individual(child1_chrom), Individual(child2_chrom)
    
    def _mutate(self, individual: Individual, k: int):
        """Mutate individual by randomly changing bin assignments."""
        for i in range(len(individual.chromosome)):
            if random.random() < self.mutation_rate:
                individual.chromosome[i] = random.randint(0, k - 1)
    
    def _repair(
        self,
        individual: Individual,
        items: List[Item],
        k: int,
        capacities: List[float]
    ):
        """Repair infeasible solution by moving items from overloaded bins."""
        bins = self._chromosome_to_bins(individual.chromosome, items, k, capacities)
        
        for _ in range(100):  # Max repair iterations
            # Find overloaded bins
            overloaded = [b for b in bins if b.current_weight > b.capacity]
            if not overloaded:
                break
            
            # Move item from most overloaded bin
            worst_bin = max(overloaded, key=lambda b: b.current_weight)
            if not worst_bin.items:
                break
            
            # Find smallest item that can be moved
            moveable = sorted(worst_bin.items, key=lambda x: x.weight)
            
            for item in moveable:
                # Find bin with most remaining capacity
                target = min(
                    [b for b in bins if b.id != worst_bin.id],
                    key=lambda b: b.current_weight
                )
                
                if target.can_fit(item):
                    worst_bin.remove_item(item)
                    target.add_item(item)
                    
                    # Update chromosome
                    item_idx = items.index(item)
                    individual.chromosome[item_idx] = target.id
                    break
    
    def _calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0
        
        # Calculate average Hamming distance
        total_distance = 0
        pairs = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = sum(
                    1 for a, b in zip(population[i].chromosome, population[j].chromosome)
                    if a != b
                )
                total_distance += dist
                pairs += 1
        
        return total_distance / pairs if pairs > 0 else 0
    
    def _solution_to_chromosome(self, solution: Solution, n: int) -> List[int]:
        """Convert solution to chromosome representation."""
        assignment = solution.get_item_assignment()
        return [assignment.get(i, 0) for i in range(n)]
    
    def _chromosome_to_bins(
        self,
        chromosome: List[int],
        items: List[Item],
        k: int,
        capacities: List[float]
    ) -> List[Bin]:
        """Convert chromosome to bins."""
        bins = [Bin(i, capacities[i]) for i in range(k)]
        
        for item_idx, bin_id in enumerate(chromosome):
            if 0 <= bin_id < k:
                bins[bin_id].items.append(items[item_idx])
        
        return bins
    
    def get_convergence_data(self) -> Dict:
        """Get convergence data for plotting."""
        return {
            "fitness_history": self.fitness_history,
            "diversity_history": self.diversity_history
        }


# ============================================================================
# Tabu Search
# ============================================================================

@dataclass
class TabuMove:
    """Represents a move in tabu search."""
    item_id: int
    from_bin: int
    to_bin: int
    
    def __hash__(self) -> int:
        return hash((self.item_id, self.from_bin, self.to_bin))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TabuMove):
            return False
        return (self.item_id == other.item_id and
                self.from_bin == other.from_bin and
                self.to_bin == other.to_bin)


@register_algorithm("tabu_search")
class TabuSearch(Algorithm):
    """
    Tabu Search for Balanced Multi-Bin Packing.
    
    Uses a tabu list to avoid cycling and escape local optima.
    
    Complexity Analysis:
    - Time: O(max_iterations * n * k) for neighborhood evaluation
    - Space: O(tabu_tenure + n * k)
    - Approximation: No guaranteed bound
    
    Pseudocode:
    1. Generate initial solution (greedy)
    2. Initialize empty tabu list
    3. For each iteration:
       a. Generate all neighbors (item moves)
       b. Find best non-tabu neighbor (or aspiration criterion)
       c. Move to best neighbor
       d. Add move to tabu list
       e. Update best if improved
       f. Remove old moves from tabu list
    4. Return best solution found
    """
    
    time_complexity = "O(max_iter · n · k)"
    space_complexity = "O(tabu_tenure + n·k)"
    approximation_ratio = "No theoretical bound"
    description = "Local search with memory to avoid cycling"
    
    def __init__(
        self,
        track_steps: bool = False,
        verbose: bool = False,
        max_iterations: int = 1000,
        tabu_tenure: int = 20,
        aspiration: bool = True,
        intensification_freq: int = 50,
        diversification_freq: int = 100
    ):
        """
        Initialize Tabu Search.
        
        Args:
            max_iterations: Maximum iterations
            tabu_tenure: Number of iterations a move stays tabu
            aspiration: Allow tabu moves if they improve best
            intensification_freq: Frequency of intensification
            diversification_freq: Frequency of diversification
        """
        super().__init__(track_steps, verbose)
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.aspiration = aspiration
        self.intensification_freq = intensification_freq
        self.diversification_freq = diversification_freq
        
        self.objective_history = []
    
    @property
    def name(self) -> str:
        return "Tabu Search"
    
    def solve(self, problem: Problem) -> Solution:
        self._start_timer()
        self._log(f"Starting Tabu Search: tenure={self.tabu_tenure}")
        
        n = problem.n_items
        k = problem.num_bins
        items = problem.items
        capacities = problem.bin_capacities
        
        # Get initial solution from greedy
        from .greedy import BestFitDecreasing
        greedy = BestFitDecreasing()
        current = greedy.solve(problem)
        
        best = current.copy()
        best.algorithm_name = self.name
        
        self._log(f"Initial solution: diff={current.value_difference:.2f}")
        
        # Tabu list: maps move -> iteration when it becomes non-tabu
        tabu_list: Dict[TabuMove, int] = {}
        
        # Frequency memory for diversification
        move_frequency: Dict[int, Dict[int, int]] = {
            i: {j: 0 for j in range(k)} for i in range(n)
        }
        
        self.objective_history = [current.value_difference]
        
        self._record_step(
            f"Initial: diff={current.value_difference:.2f}",
            current.bins,
            extra_data={"tabu_size": 0}
        )
        
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            self._iterations += 1
            
            # Generate and evaluate all neighbors
            neighbors = self._generate_neighbors(current, items, k, capacities)
            
            if not neighbors:
                self._log("No feasible neighbors found")
                break
            
            # Find best admissible neighbor
            best_neighbor = None
            best_neighbor_diff = float('inf')
            best_move = None
            
            for neighbor, move in neighbors:
                diff = neighbor.value_difference
                
                # Check if move is tabu
                is_tabu = move in tabu_list and tabu_list[move] > iteration
                
                # Aspiration criterion: accept if better than best ever
                if is_tabu and self.aspiration and diff < best.value_difference:
                    is_tabu = False
                
                if not is_tabu and diff < best_neighbor_diff:
                    best_neighbor = neighbor
                    best_neighbor_diff = diff
                    best_move = move
            
            if best_neighbor is None:
                # All neighbors are tabu, pick least-tabu
                min_tenure = float('inf')
                for neighbor, move in neighbors:
                    if move in tabu_list and tabu_list[move] < min_tenure:
                        min_tenure = tabu_list[move]
                        best_neighbor = neighbor
                        best_move = move
            
            if best_neighbor is None:
                break
            
            # Move to best neighbor
            current = best_neighbor
            
            # Update tabu list
            if best_move:
                # Add reverse move to tabu list
                reverse_move = TabuMove(
                    best_move.item_id,
                    best_move.to_bin,
                    best_move.from_bin
                )
                tabu_list[reverse_move] = iteration + self.tabu_tenure
                
                # Update frequency
                move_frequency[best_move.item_id][best_move.to_bin] += 1
            
            # Clean old entries from tabu list
            tabu_list = {m: t for m, t in tabu_list.items() if t > iteration}
            
            # Update best
            if current.value_difference < best.value_difference:
                best = current.copy()
                best.algorithm_name = self.name
                no_improvement_count = 0
                
                self._log(f"Iter {iteration}: New best = {best.value_difference:.2f}")
                
                self._record_step(
                    f"Iter {iteration}: New best = {best.value_difference:.2f}",
                    best.bins,
                    extra_data={"iteration": iteration}
                )
            else:
                no_improvement_count += 1
            
            self.objective_history.append(current.value_difference)
            
            # Intensification
            if iteration % self.intensification_freq == 0 and no_improvement_count > 0:
                current = best.copy()
                self._log(f"Intensification: returning to best solution")
            
            # Diversification
            if iteration % self.diversification_freq == 0 and no_improvement_count > 20:
                current = self._diversify(current, move_frequency, items, k, capacities)
                self._log(f"Diversification: perturbing solution")
            
            if iteration % 100 == 0:
                self._log(f"Iter {iteration}: current={current.value_difference:.2f}, "
                         f"best={best.value_difference:.2f}, tabu_size={len(tabu_list)}")
        
        best.execution_time = self._get_elapsed_time()
        best.iterations = self._iterations
        best.metadata["tabu_tenure"] = self.tabu_tenure
        
        return best
    
    def _generate_neighbors(
        self,
        solution: Solution,
        items: List[Item],
        k: int,
        capacities: List[float]
    ) -> List[Tuple[Solution, TabuMove]]:
        """Generate all neighbor solutions by moving items."""
        neighbors = []
        bins = solution.bins
        
        # Item ID to index mapping
        item_idx = {item.id: i for i, item in enumerate(items)}
        
        for bin in bins:
            for item in bin.items:
                # Try moving to each other bin
                for target_bin in bins:
                    if target_bin.id == bin.id:
                        continue
                    
                    # Check feasibility
                    if not target_bin.can_fit(item):
                        continue
                    
                    # Create neighbor
                    neighbor = solution.copy()
                    neighbor.bins[bin.id].remove_item(item)
                    neighbor.bins[target_bin.id].add_item(item)
                    
                    move = TabuMove(item.id, bin.id, target_bin.id)
                    neighbors.append((neighbor, move))
        
        return neighbors
    
    def _diversify(
        self,
        solution: Solution,
        move_frequency: Dict[int, Dict[int, int]],
        items: List[Item],
        k: int,
        capacities: List[float]
    ) -> Solution:
        """Diversify by making less-frequent moves."""
        result = solution.copy()
        
        # Make several random moves to less-visited configurations
        for _ in range(k):
            bins = result.bins
            
            # Find item to move (prefer items with high frequency in current bin)
            best_item = None
            best_bin = None
            max_freq = -1
            
            for bin in bins:
                if not bin.items:
                    continue
                for item in bin.items:
                    freq = move_frequency[item.id][bin.id]
                    if freq > max_freq:
                        max_freq = freq
                        best_item = item
                        best_bin = bin
            
            if best_item and best_bin:
                # Move to least-frequented feasible bin
                min_freq = float('inf')
                target = None
                
                for bin in bins:
                    if bin.id != best_bin.id and bin.can_fit(best_item):
                        freq = move_frequency[best_item.id][bin.id]
                        if freq < min_freq:
                            min_freq = freq
                            target = bin
                
                if target:
                    best_bin.remove_item(best_item)
                    target.add_item(best_item)
        
        return result
    
    def get_convergence_data(self) -> Dict:
        """Get convergence data for plotting."""
        return {
            "objective_history": self.objective_history
        }
