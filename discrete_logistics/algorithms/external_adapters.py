"""
External Algorithm Adapters
===========================

Adaptadores para integrar algoritmos externos (Gemini H-GADP, Qwen SA-DP)
con la infraestructura del proyecto.

Estos adaptadores convierten las estructuras de datos del proyecto
a las estructuras esperadas por los algoritmos externos y viceversa.
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from ..algorithms.base import Algorithm
from ..core.problem import Problem, Solution, Bin, Item

# ============================================================================
# Gemini H-GADP Adapter
# ============================================================================

class GeminiHGADP(Algorithm):
    """
    Adaptador para el algoritmo H-GADP de Gemini.
    
    Hybrid Genetic Algorithm with Dynamic Programming:
    - Fase 1: GA para items "core" (más valiosos)
    - Fase 2: DP (Knapsack) para items "tail"
    - Fase 3: Greedy para items residuales
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        pop_size: int = 40,
        generations: int = 50,
        mutation_rate: float = 0.1,
        time_limit: float = 120.0,
        verbose: bool = False
    ):
        self.alpha = alpha
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.time_limit = time_limit
        self.verbose = verbose
    
    @property
    def name(self) -> str:
        return "H-GADP (Gemini)"
    
    def solve(self, problem: Problem) -> Solution:
        """Ejecuta el algoritmo H-GADP adaptado."""
        start_time = time.time()
        
        # Convertir estructuras de datos del proyecto a las de Gemini
        gemini_items = []
        for item in problem.items:
            gemini_items.append(_GeminiItem(
                id=item.id,
                weight=int(item.weight),
                value=int(item.value)
            ))
        
        gemini_bins = []
        for i, capacity in enumerate(problem.bin_capacities):
            gemini_bins.append(_GeminiBin(
                id=i,
                capacity=int(capacity)
            ))
        
        # Ejecutar algoritmo H-GADP
        result_bins, imbalance = self._h_gadp(
            items=gemini_items,
            bins_input=gemini_bins,
            alpha=self.alpha
        )
        
        # Convertir resultado a Solution del proyecto
        solution_bins = []
        for i, g_bin in enumerate(result_bins):
            # Crear bin con los items asignados
            new_bin = Bin(
                id=g_bin.id,
                capacity=problem.bin_capacities[i]
            )
            for g_item in g_bin.items:
                # Encontrar item original y agregarlo usando add_item
                original_item = next(it for it in problem.items if it.id == g_item.id)
                new_bin.items.append(original_item)  # Direct append since we control the assignment
            solution_bins.append(new_bin)
        
        execution_time = time.time() - start_time
        
        return Solution(
            bins=solution_bins,
            algorithm_name=self.name,
            execution_time=execution_time,
            iterations=self.generations,
            metadata={
                'alpha': self.alpha,
                'imbalance': imbalance,
                'pop_size': self.pop_size
            }
        )
    
    def _h_gadp(self, items: List['_GeminiItem'], bins_input: List['_GeminiBin'], alpha: float = 0.3):
        """Implementación del algoritmo H-GADP."""
        import random
        
        start_time = time.time()
        
        # Copiar bins
        bins = [b.copy() for b in bins_input]
        
        # Ordenar items por valor descendente
        sorted_items = sorted(items, key=lambda x: x.value, reverse=True)
        
        # Corte Alpha
        cut_idx = max(1, int(len(items) * alpha))
        core_items = sorted_items[:cut_idx]
        tail_items = sorted_items[cut_idx:]
        
        # Limitar tail_items a máximo 12
        max_tail_items = 12
        if len(tail_items) > max_tail_items:
            excess = tail_items[max_tail_items:]
            core_items = core_items + excess
            tail_items = tail_items[:max_tail_items]
        
        if self.verbose:
            print(f"H-GADP: Total={len(items)} | Core={len(core_items)} | Tail={len(tail_items)}")
        
        # Fase 1: GA para Core Items
        if core_items:
            best_chrom = self._genetic_algorithm(core_items, bins)
            for i, bin_idx in enumerate(best_chrom):
                if bin_idx < len(bins):
                    bins[bin_idx].add_item(core_items[i])
        
        # Fase 2: DP para Tail Items
        if tail_items:
            total_val = sum(i.value for i in items)
            target_val = total_val / len(bins) if bins else 0
            
            bins.sort(key=lambda b: b.current_value)
            remaining_tail = tail_items[:]
            
            for b in bins:
                if not remaining_tail:
                    break
                residual_cap = int(b.capacity - b.current_weight)
                if residual_cap > 0:
                    chosen = self._knapsack_dp(remaining_tail, residual_cap)
                    for item in chosen:
                        b.add_item(item)
                        remaining_tail.remove(item)
            
            # Fase 3: Greedy para residuales
            for item in remaining_tail:
                best_bin = None
                min_imbalance = float('inf')
                for b in bins:
                    if b.can_fit(item):
                        dist = abs((b.current_value + item.value) - target_val)
                        if dist < min_imbalance:
                            min_imbalance = dist
                            best_bin = b
                if best_bin:
                    best_bin.add_item(item)
        
        final_vals = [b.current_value for b in bins]
        imbalance = max(final_vals) - min(final_vals) if final_vals else 0
        
        return bins, imbalance
    
    def _genetic_algorithm(self, items: List['_GeminiItem'], bins: List['_GeminiBin']) -> List[int]:
        """Algoritmo genético simplificado."""
        import random
        
        k = len(bins)
        n = len(items)
        
        if n == 0 or k == 0:
            return []
        
        # Población inicial
        population = [[random.randint(0, k - 1) for _ in range(n)] for _ in range(self.pop_size)]
        
        start_time = time.time()
        best_fitness = float('inf')
        no_improvement = 0
        
        for gen in range(self.generations):
            if time.time() - start_time >= self.time_limit:
                break
            
            # Evaluar fitness
            fitness_scores = [(ind, self._calculate_fitness(ind, items, bins)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1])
            
            current_best = fitness_scores[0][1]
            if current_best < best_fitness:
                best_fitness = current_best
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= 100:
                break
            
            # Nueva generación
            next_gen = [fitness_scores[0][0]]
            
            while len(next_gen) < self.pop_size:
                # Torneo
                candidates = random.sample(fitness_scores, min(3, len(fitness_scores)))
                parent1 = min(candidates, key=lambda x: x[1])[0]
                candidates = random.sample(fitness_scores, min(3, len(fitness_scores)))
                parent2 = min(candidates, key=lambda x: x[1])[0]
                
                # Crossover
                child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(n)]
                
                # Mutación
                if random.random() < self.mutation_rate:
                    idx = random.randint(0, n - 1)
                    child[idx] = random.randint(0, k - 1)
                
                next_gen.append(child)
            
            population = next_gen
        
        return min(population, key=lambda x: self._calculate_fitness(x, items, bins))
    
    def _calculate_fitness(self, chromosome: List[int], items: List['_GeminiItem'], bins: List['_GeminiBin']) -> float:
        """Calcula fitness (desbalance + penalización)."""
        k = len(bins)
        bin_weights = [0] * k
        bin_values = [0] * k
        penalty = 0
        
        for i, bin_idx in enumerate(chromosome):
            if bin_idx < k:
                item = items[i]
                bin_weights[bin_idx] += item.weight
                bin_values[bin_idx] += item.value
                
                if bin_weights[bin_idx] > bins[bin_idx].capacity:
                    penalty += (bin_weights[bin_idx] - bins[bin_idx].capacity) * 10000
        
        spread = max(bin_values) - min(bin_values) if bin_values else 0
        return spread + penalty
    
    def _knapsack_dp(self, items: List['_GeminiItem'], capacity: int) -> List['_GeminiItem']:
        """Resuelve 0/1 Knapsack con DP."""
        if not items or capacity <= 0:
            return []
        
        n = min(len(items), 12)  # Limitar a 12 items
        items = items[:n]
        capacity = min(capacity, 10000)  # Limitar capacidad
        
        # DP table
        K = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            item = items[i-1]
            item_weight = int(item.weight)
            for w in range(1, capacity + 1):
                if item_weight <= w:
                    K[i][w] = max(item.value + K[i-1][w - item_weight], K[i-1][w])
                else:
                    K[i][w] = K[i-1][w]
        
        # Backtracking
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            if K[i][w] != K[i-1][w]:
                selected.append(items[i-1])
                w -= int(items[i-1].weight)
        
        return selected


# ============================================================================
# Qwen SA-DP Adapter
# ============================================================================

class QwenSADP(Algorithm):
    """
    Adaptador para el algoritmo Hybrid SA-DP de Qwen.
    
    Simulated Annealing with Dynamic Programming:
    - Fase 1: SA para items pesados
    - Fase 2: DP para balance de items restantes
    - Fase 3: Búsqueda local para refinamiento
    """
    
    def __init__(
        self,
        alpha: float = 0.65,
        epsilon: float = 0.1,
        time_limit: float = 120.0,
        verbose: bool = False
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.time_limit = time_limit
        self.verbose = verbose
    
    @property
    def name(self) -> str:
        return "SA-DP (Qwen)"
    
    def solve(self, problem: Problem) -> Solution:
        """Ejecuta el algoritmo SA-DP adaptado."""
        start_time = time.time()
        
        # Convertir estructuras
        items = [_QwenItem(id=it.id, weight=float(it.weight), value=float(it.value)) 
                 for it in problem.items]
        bins = [_QwenBin(id=i, capacity=float(cap)) 
                for i, cap in enumerate(problem.bin_capacities)]
        
        # Ejecutar algoritmo
        assignment = self._hybrid_sa_dp(items, bins)
        
        # Construir solución
        solution_bins = []
        for i, capacity in enumerate(problem.bin_capacities):
            new_bin = Bin(id=i, capacity=capacity)
            for item in problem.items:
                if assignment.get(item.id) == i:
                    new_bin.items.append(item)  # Direct append
            solution_bins.append(new_bin)
        
        execution_time = time.time() - start_time
        
        return Solution(
            bins=solution_bins,
            algorithm_name=self.name,
            execution_time=execution_time,
            iterations=0,
            metadata={
                'alpha': self.alpha,
                'epsilon': self.epsilon
            }
        )
    
    def _hybrid_sa_dp(self, items: List['_QwenItem'], bins: List['_QwenBin']) -> Dict[int, int]:
        """Implementación del algoritmo híbrido SA-DP."""
        import random
        import math
        
        if not items or not bins:
            return {}
        
        # Ordenar items por peso
        items_sorted = sorted(items, key=lambda x: x.weight, reverse=True)
        
        # Calcular umbral
        total_capacity = sum(b.capacity for b in bins)
        tau = total_capacity / (len(bins) * max(1, len(items)))
        
        # Separar items
        heavy_items = [it for it in items_sorted if it.weight >= tau]
        max_heavy = max(1, int(self.alpha * len(items)))
        I_h = heavy_items[:max_heavy]
        I_dp = [it for it in items if it not in I_h]
        
        # Limitar DP items
        if len(I_dp) > 12:
            excess = I_dp[12:]
            I_h = I_h + excess
            I_dp = I_dp[:12]
        
        if self.verbose:
            print(f"SA-DP: Heavy={len(I_h)} | DP={len(I_dp)}")
        
        # Fase 1: SA para items pesados
        solution = self._ffd_init(I_h, bins)
        if I_h:
            solution = self._simulated_annealing(solution, I_h, bins)
        
        # Fase 2: DP para balance
        if I_dp:
            # Actualizar capacidades residuales
            residual_bins = [_QwenBin(b.id, b.capacity) for b in bins]
            for item in I_h:
                bin_id = solution.get(item.id)
                if bin_id is not None:
                    for b in residual_bins:
                        if b.id == bin_id:
                            b.add_item(item)
                            break
            
            base_values = [b.current_value for b in residual_bins]
            dp_solution = self._dp_balance(I_dp, residual_bins, base_values)
            solution.update(dp_solution)
        
        return solution
    
    def _ffd_init(self, items: List['_QwenItem'], bins: List['_QwenBin']) -> Dict[int, int]:
        """First-Fit Decreasing initialization."""
        sorted_items = sorted(items, key=lambda x: x.weight, reverse=True)
        for b in bins:
            b.reset()
        
        solution = {}
        for item in sorted_items:
            assigned = False
            for b in bins:
                if b.can_add(item):
                    b.add_item(item)
                    solution[item.id] = b.id
                    assigned = True
                    break
            if not assigned:
                best_bin = max(bins, key=lambda b: b.capacity - b.current_weight)
                best_bin.add_item(item)
                solution[item.id] = best_bin.id
        
        return solution
    
    def _simulated_annealing(self, initial_solution: Dict[int, int], 
                             items: List['_QwenItem'], bins: List['_QwenBin']) -> Dict[int, int]:
        """Simulated Annealing simplificado."""
        import random
        import math
        
        current_solution = initial_solution.copy()
        current_cost = self._calculate_cost(current_solution, items, bins)
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        T = 100.0
        cooling_rate = 0.95
        min_temp = 0.001
        
        start_time = time.time()
        item_ids = [it.id for it in items]
        bin_ids = [b.id for b in bins]
        
        iteration = 0
        no_improve = 0
        
        while T > min_temp and no_improve < 1000:
            if time.time() - start_time >= self.time_limit * 0.4:
                break
            
            # Generar vecino
            new_solution = current_solution.copy()
            item_id = random.choice(item_ids)
            current_bin = new_solution[item_id]
            possible_bins = [bid for bid in bin_ids if bid != current_bin]
            if possible_bins:
                new_solution[item_id] = random.choice(possible_bins)
            
            new_cost = self._calculate_cost(new_solution, items, bins)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_solution = new_solution
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            T *= cooling_rate
            iteration += 1
        
        return best_solution
    
    def _calculate_cost(self, solution: Dict[int, int], items: List['_QwenItem'], 
                        bins: List['_QwenBin']) -> float:
        """Calcula costo de una solución."""
        for b in bins:
            b.reset()
        
        for item in items:
            bin_id = solution.get(item.id)
            if bin_id is not None:
                for b in bins:
                    if b.id == bin_id:
                        b.add_item(item)
                        break
        
        bin_values = [b.current_value for b in bins]
        balance_cost = max(bin_values) - min(bin_values) if bin_values else 0
        
        # Penalización por sobrecarga
        penalty = 0
        for b in bins:
            if b.current_weight > b.capacity:
                penalty += 10000 * (b.current_weight - b.capacity)
        
        return balance_cost + penalty
    
    def _dp_balance(self, items: List['_QwenItem'], bins: List['_QwenBin'], 
                    base_values: List[float]) -> Dict[int, int]:
        """DP para balance de items restantes."""
        if not items:
            return {}
        
        solution = {}
        remaining = items[:]
        
        # Ordenar bins por valor actual (menor primero)
        sorted_bins = sorted(enumerate(bins), key=lambda x: base_values[x[0]])
        
        for idx, b in sorted_bins:
            if not remaining:
                break
            
            residual_cap = int(b.capacity - b.current_weight)
            if residual_cap > 0:
                # Knapsack DP simple
                chosen = self._simple_knapsack(remaining, residual_cap)
                for item in chosen:
                    solution[item.id] = b.id
                    remaining.remove(item)
        
        # Asignar residuales con greedy
        for item in remaining:
            best_bin = min(bins, key=lambda b: b.current_value)
            solution[item.id] = best_bin.id
        
        return solution
    
    def _simple_knapsack(self, items: List['_QwenItem'], capacity: int) -> List['_QwenItem']:
        """Knapsack DP simple."""
        if not items or capacity <= 0:
            return []
        
        n = min(len(items), 12)
        items = items[:n]
        capacity = min(capacity, 10000)
        
        K = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            item = items[i-1]
            w = int(item.weight)
            for c in range(1, capacity + 1):
                if w <= c:
                    K[i][c] = max(item.value + K[i-1][c - w], K[i-1][c])
                else:
                    K[i][c] = K[i-1][c]
        
        selected = []
        c = capacity
        for i in range(n, 0, -1):
            if K[i][c] != K[i-1][c]:
                selected.append(items[i-1])
                c -= int(items[i-1].weight)
        
        return selected


# ============================================================================
# Helper Classes (Internal Data Structures)
# ============================================================================

class _GeminiItem:
    """Item interno para Gemini."""
    def __init__(self, id: int, weight: int, value: int):
        self.id = id
        self.weight = weight
        self.value = value
        self.ratio = value / weight if weight > 0 else 0

class _GeminiBin:
    """Bin interno para Gemini."""
    def __init__(self, id: int, capacity: int):
        self.id = id
        self.capacity = capacity
        self.current_weight = 0
        self.current_value = 0
        self.items = []
    
    def can_fit(self, item: _GeminiItem) -> bool:
        return self.current_weight + item.weight <= self.capacity
    
    def add_item(self, item: _GeminiItem):
        self.items.append(item)
        self.current_weight += item.weight
        self.current_value += item.value
    
    def copy(self):
        new_bin = _GeminiBin(self.id, self.capacity)
        new_bin.current_weight = self.current_weight
        new_bin.current_value = self.current_value
        new_bin.items = self.items[:]
        return new_bin


@dataclass
class _QwenItem:
    """Item interno para Qwen."""
    id: int
    weight: float
    value: float

@dataclass
class _QwenBin:
    """Bin interno para Qwen."""
    id: int
    capacity: float
    current_weight: float = 0.0
    current_value: float = 0.0
    
    def reset(self):
        self.current_weight = 0.0
        self.current_value = 0.0
    
    def can_add(self, item: _QwenItem) -> bool:
        return self.current_weight + item.weight <= self.capacity
    
    def add_item(self, item: _QwenItem):
        self.current_weight += item.weight
        self.current_value += item.value
    
    def remove_item(self, item: _QwenItem):
        self.current_weight -= item.weight
        self.current_value -= item.value
