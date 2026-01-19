# Algoritmo Híbrido SA-DP para Empaquetamiento Balanceado
# Implementación optimizada en Python

import random
import math
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

@dataclass
class Item:
    """Representa un ítem a empaquetar."""
    id: int
    weight: float
    value: float
    
    def __repr__(self) -> str:
        return f"Item(id={self.id}, weight={self.weight:.2f}, value={self.value:.2f})"

@dataclass
class Bin:
    """Representa un contenedor con capacidad limitada."""
    id: int
    capacity: float
    current_weight: float = 0.0
    current_value: float = 0.0
    
    def reset(self) -> None:
        """Reinicia el contenedor a su estado inicial."""
        self.current_weight = 0.0
        self.current_value = 0.0
        
    def can_add(self, item: Item) -> bool:
        """Verifica si el ítem cabe en el contenedor."""
        return self.current_weight + item.weight <= self.capacity
    
    def add_item(self, item: Item) -> None:
        """Agrega un ítem al contenedor."""
        self.current_weight += item.weight
        self.current_value += item.value
        
    def remove_item(self, item: Item) -> None:
        """Elimina un ítem del contenedor."""
        self.current_weight -= item.weight
        self.current_value -= item.value
    
    def __repr__(self) -> str:
        return (f"Bin(id={self.id}, capacity={self.capacity:.2f}, "
                f"current_weight={self.current_weight:.2f}, "
                f"current_value={self.current_value:.2f})")

def calculate_initial_temp(items: List[Item], bins: List[Bin], 
                          num_samples: int = 100) -> float:
    """
    Calcula la temperatura inicial para Simulated Annealing basada en la 
    desviación estándar de cambios aleatorios en la función objetivo.
    """
    if not items:
        return 1.0
        
    # Generar solución inicial con FFD
    solution = ffd_initialization(items, bins)
    
    # Evaluar solución inicial
    current_cost = calculate_cost(solution, items, bins)
    delta_costs = []
    
    # Generar cambios aleatorios y medir sus impactos
    for _ in range(num_samples):
        # Seleccionar ítem y nuevo contenedor aleatorios
        item_idx = random.randint(0, len(items)-1)
        item_id = items[item_idx].id
        current_bin = solution[item_id]
        new_bin_idx = random.randint(0, len(bins)-1)
        new_bin_id = bins[new_bin_idx].id
        
        if current_bin == new_bin_id:
            continue
            
        # Calcular el costo del movimiento
        temp_solution = solution.copy()
        temp_solution[item_id] = new_bin_id
        new_cost = calculate_cost(temp_solution, items, bins)
        delta = new_cost - current_cost
        if delta > 0:  # Solo considerar empeoramientos
            delta_costs.append(delta)
    
    if not delta_costs:
        return 1.0
        
    std_dev = np.std(delta_costs) if len(delta_costs) > 1 else max(delta_costs)
    # Temperatura para una probabilidad inicial de aceptación ~80%
    return max(1.0, std_dev / max(0.001, -math.log(0.8))) if std_dev > 0 else 1.0

def calculate_cost(solution: Dict[int, int], items: List[Item], bins: List[Bin], 
                  lambda_penalty: float = None) -> float:
    """
    Calcula el costo de una solución, considerando el desbalance y penalizaciones
    por violaciones de capacidad.
    """
    # Si no se proporciona lambda_penalty, calcular automáticamente
    if lambda_penalty is None:
        max_value = max(item.value for item in items) if items else 1.0
        lambda_penalty = len(bins) * max_value * 10
    
    # Reiniciar contenedores
    for bin in bins:
        bin.reset()
        
    # Asignar ítems según la solución
    for item in items:
        bin_id = solution[item.id]
        for bin in bins:
            if bin.id == bin_id:
                bin.add_item(item)
                break
    
    # Calcular valores de contenedores y verificar violaciones
    bin_values = [bin.current_value for bin in bins]
    bin_weights = [bin.current_weight for bin in bins]
    
    # Calcular desbalance
    max_value = max(bin_values)
    min_value = min(bin_values)
    balance_cost = max_value - min_value
    
    # Calcular penalización por sobrecarga
    overload_penalty = 0.0
    for i, bin in enumerate(bins):
        if bin_weights[i] > bin.capacity:
            overload_penalty += lambda_penalty * (bin_weights[i] - bin.capacity) / bin.capacity
    
    return balance_cost + overload_penalty

def ffd_initialization(items: List[Item], bins: List[Bin]) -> Dict[int, int]:
    """
    Genera una solución inicial usando First-Fit Decreasing.
    """
    # Ordenar ítems por peso descendente
    sorted_items = sorted(items, key=lambda x: x.weight, reverse=True)
    
    # Reiniciar contenedores
    for bin in bins:
        bin.reset()
        
    solution = {}
    
    # Asignar cada ítem al primer contenedor disponible
    for item in sorted_items:
        assigned = False
        for bin in bins:
            if bin.can_add(item):
                bin.add_item(item)
                solution[item.id] = bin.id
                assigned = True
                break
        
        # Si no se pudo asignar, asignar al contenedor con mayor capacidad restante
        if not assigned:
            # Encontrar contenedor con mayor capacidad restante
            best_bin = max(bins, key=lambda b: b.capacity - b.current_weight)
            best_bin.add_item(item)
            solution[item.id] = best_bin.id
    
    return solution

def simulated_annealing(initial_solution: Dict[int, int], items: List[Item], 
                       bins: List[Bin], T0: float = 100.0, cooling_rate: float = 0.95,
                       min_temp: float = 0.001, max_iterations: int = 10000,
                       no_improve_limit: int = 1000, max_time: float = 120.0,
                       convergence_limit: int = 10000) -> Dict[int, int]:
    """
    Implementación de Simulated Annealing con operadores especializados para el problema.
    MODIFICACIÓN: Añadido límite de tiempo (2 minutos) y convergencia de 10000 iteraciones.
    """
    # Parámetros automáticos si no se proporcionan
    if T0 <= 0:
        T0 = calculate_initial_temp(items, bins)
    
    # Calcular lambda_penalty automáticamente
    max_value = max(item.value for item in items) if items else 1.0
    lambda_penalty = len(bins) * max_value * 10
    
    current_solution = initial_solution.copy()
    current_cost = calculate_cost(current_solution, items, bins, lambda_penalty)
    best_solution = current_solution.copy()
    best_cost = current_cost
    
    T = T0
    iteration = 0
    no_improve_count = 0
    
    # Pre-calcular índices de ítems y bins para acceso rápido
    item_ids = [item.id for item in items]
    bin_ids = [bin.id for bin in bins]
    
    start_time = time.time()
    
    # MODIFICACIÓN: Añadir verificación de tiempo y usar convergence_limit
    while T > min_temp and iteration < max_iterations and no_improve_count < convergence_limit:
        # Verificar límite de tiempo
        if time.time() - start_time >= max_time:
            break
        
        # Seleccionar operador de vecindad según probabilidades
        rand = random.random()
        if rand < 0.7:  # Single Move (70%)
            neighbor = single_move(current_solution, item_ids, bin_ids)
        elif rand < 0.9:  # Swap (20%)
            neighbor = swap_move(current_solution, item_ids, bin_ids)
        else:  # Bin Reset (10%)
            neighbor = bin_reset_move(current_solution, item_ids, bin_ids, bins)
        
        # Evaluar vecino
        neighbor_cost = calculate_cost(neighbor, items, bins, lambda_penalty)
        
        # Criterio de aceptación de Metropolis
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_solution = neighbor
            current_cost = neighbor_cost
            
            # Actualizar mejor solución
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Enfriamiento
        T *= cooling_rate
        iteration += 1
    
    return best_solution

def single_move(solution: Dict[int, int], item_ids: List[int], bin_ids: List[int]) -> Dict[int, int]:
    """Operador de movimiento simple: reasigna un ítem a un contenedor aleatorio."""
    new_solution = solution.copy()
    item_id = random.choice(item_ids)
    current_bin = new_solution[item_id]
    
    # Seleccionar un nuevo contenedor diferente
    possible_bins = [bid for bid in bin_ids if bid != current_bin]
    if possible_bins:
        new_solution[item_id] = random.choice(possible_bins)
    
    return new_solution

def swap_move(solution: Dict[int, int], item_ids: List[int], bin_ids: List[int]) -> Dict[int, int]:
    """Operador de intercambio: intercambia las asignaciones de dos ítems."""
    if len(item_ids) < 2:
        return solution.copy()
        
    new_solution = solution.copy()
    item1_id, item2_id = random.sample(item_ids, 2)
    
    # Intercambiar asignaciones si están en contenedores diferentes
    if solution[item1_id] != solution[item2_id]:
        new_solution[item1_id] = solution[item2_id]
        new_solution[item2_id] = solution[item1_id]
    
    return new_solution

def bin_reset_move(solution: Dict[int, int], item_ids: List[int], bin_ids: List[int], 
                  bins: List[Bin]) -> Dict[int, int]:
    """Operador de reinicio de contenedor: vacía un contenedor y reasigna sus ítems."""
    if not bin_ids:
        return solution.copy()
        
    new_solution = solution.copy()
    bin_id = random.choice(bin_ids)
    
    # Obtener ítems asignados a este contenedor
    bin_items = [item_id for item_id, b_id in solution.items() if b_id == bin_id]
    
    if not bin_items:
        return new_solution
        
    # Reasignar aleatoriamente los ítems del contenedor
    for item_id in bin_items:
        # Seleccionar un contenedor diferente al actual
        possible_bins = [bid for bid in bin_ids if bid != bin_id]
        if possible_bins:
            new_solution[item_id] = random.choice(possible_bins)
    
    return new_solution

def update_bin_residuals(original_bins: List[Bin], solution: Dict[int, int], 
                        items: List[Item]) -> List[Bin]:
    """
    Crea una copia de los contenedores con capacidades residuales después de asignar
    los ítems en la solución.
    """
    # Crear copias de los contenedores originales
    residual_bins = [Bin(bin.id, bin.capacity) for bin in original_bins]
    
    # Aplicar la solución para actualizar pesos y valores
    for item in items:
        bin_id = solution[item.id]
        for bin in residual_bins:
            if bin.id == bin_id:
                bin.add_item(item)
                break
    
    return residual_bins

def dp_balance(items: List[Item], residual_bins: List[Bin], base_values: List[float],
              epsilon: float = 0.1, max_time: float = 30.0) -> Dict[int, int]:
    """
    Programación Dinámica para balancear los ítems restantes dadas las capacidades
    residuales y valores base de los contenedores.
    
    ESTRATEGIA ADAPTATIVA:
    - Máximo 12 ítems para DP
    - Si la DP tarda demasiado, reduce el número de ítems progresivamente
    - Usa discretización agresiva para reducir el espacio de estados
    - Los ítems no procesados por DP se asignan con heurística balanceada
    """
    if not items or not residual_bins:
        return {}
    
    k = len(residual_bins)
    n_original = len(items)
    
    # Ordenar ítems por valor descendente (los más valiosos primero para DP)
    items_sorted = sorted(items, key=lambda x: x.value, reverse=True)
    
    # Determinar tamaño inicial para DP (máximo 12)
    max_dp_items = min(12, n_original)
    
    # Si k=1, todos los ítems van al único contenedor
    if k == 1:
        solution = {}
        for item in items:
            solution[item.id] = residual_bins[0].id
        return solution
    
    # Función interna para ejecutar DP con un subconjunto de ítems
    def run_dp_with_timeout(dp_items: List[Item], time_limit: float) -> Optional[Dict[int, int]]:
        """Ejecuta DP con límite de tiempo. Retorna None si se agota el tiempo."""
        if not dp_items:
            return {}
        
        dp_start = time.time()
        n = len(dp_items)
        
        # Discretización más agresiva basada en el número de ítems y bins
        # Más ítems/bins = discretización más gruesa
        aggression_factor = max(1.0, (n * k) / 20.0)
        effective_epsilon = min(0.5, epsilon * aggression_factor)
        
        # Calcular valores totales y máximos para escalar
        total_value = sum(item.value for item in dp_items)
        max_value = max(item.value for item in dp_items) if dp_items else 1.0
        
        # Factor de escalado agresivo
        delta = effective_epsilon * max_value / max(1, n)
        delta = max(delta, max_value / 50.0)  # Máximo 50 niveles por valor
        
        # Escalar valores
        scaled_items = []
        for item in dp_items:
            scaled_value = int(item.value / delta)
            scaled_items.append((item.id, item.weight, scaled_value))
        
        # Capacidades residuales
        residual_capacities = [bin.capacity - bin.current_weight for bin in residual_bins]
        scaled_base_values = [int(val / delta) for val in base_values]
        
        # DP iterativa con tabla (más controlable que recursiva)
        # Estado: para cada ítem, guardamos las mejores asignaciones parciales
        # Formato: {(valores_por_bin): asignación}
        
        # Limitar el número máximo de estados para evitar explosión
        max_states = 10000
        
        # Estado inicial: valores base de cada bin
        initial_state = tuple(scaled_base_values)
        
        # dp_table[i] = dict de {estado: (desbalance, asignaciones)}
        current_states = {initial_state: ([], [0.0] * k)}  # (asignaciones, pesos_usados)
        
        for item_idx, (item_id, item_weight, item_scaled_value) in enumerate(scaled_items):
            # Verificar tiempo
            if time.time() - dp_start > time_limit:
                return None  # Timeout
            
            next_states = {}
            
            for state, (assignments, weights_used) in current_states.items():
                for bin_idx in range(k):
                    # Verificar capacidad
                    new_weight = weights_used[bin_idx] + item_weight
                    if new_weight > residual_capacities[bin_idx]:
                        continue
                    
                    # Calcular nuevo estado
                    new_state = list(state)
                    new_state[bin_idx] += item_scaled_value
                    new_state = tuple(new_state)
                    
                    # Calcular desbalance del nuevo estado
                    new_desbalance = max(new_state) - min(new_state)
                    
                    # Nueva asignación y pesos
                    new_assignments = assignments + [bin_idx]
                    new_weights = weights_used.copy()
                    new_weights[bin_idx] = new_weight
                    
                    # Guardar si es mejor o nuevo
                    if new_state not in next_states:
                        next_states[new_state] = (new_assignments, new_weights)
                    else:
                        existing_desbalance = max(next_states[new_state][0]) - min(next_states[new_state][0]) if next_states[new_state][0] else float('inf')
                        if new_desbalance < existing_desbalance:
                            next_states[new_state] = (new_assignments, new_weights)
            
            # Podar estados si hay demasiados (mantener los mejores)
            if len(next_states) > max_states:
                # Ordenar por desbalance y mantener los mejores
                sorted_states = sorted(
                    next_states.items(),
                    key=lambda x: max(x[0]) - min(x[0])
                )
                next_states = dict(sorted_states[:max_states])
            
            current_states = next_states
            
            # Si no quedan estados válidos, fallar
            if not current_states:
                return None
        
        # Encontrar el mejor estado final
        if not current_states:
            return None
        
        best_state = min(current_states.keys(), key=lambda s: max(s) - min(s))
        best_assignments, _ = current_states[best_state]
        
        # Construir solución
        solution = {}
        for idx, bin_idx in enumerate(best_assignments):
            item_id = dp_items[idx].id
            solution[item_id] = residual_bins[bin_idx].id
        
        return solution
    
    # Estrategia adaptativa: intentar con tamaños decrecientes si hay timeout
    dp_start_time = time.time()
    dp_solution = None
    items_for_dp = []
    items_remaining = []
    
    for attempt_size in range(max_dp_items, 0, -2):  # Reducir de 2 en 2
        # Verificar tiempo total disponible
        elapsed = time.time() - dp_start_time
        if elapsed > max_time * 0.8:  # Usar máximo 80% del tiempo en intentos
            break
        
        # Seleccionar ítems para este intento
        items_for_dp = items_sorted[:attempt_size]
        items_remaining = items_sorted[attempt_size:]
        
        # Calcular tiempo para este intento (más tiempo para tamaños menores)
        time_for_attempt = min(max_time - elapsed, max_time / (max_dp_items / attempt_size))
        
        dp_solution = run_dp_with_timeout(items_for_dp, time_for_attempt)
        
        if dp_solution is not None:
            break  # Éxito
    
    # Si la DP falló completamente, usar heurística para todos los ítems
    if dp_solution is None:
        items_for_dp = []
        items_remaining = items_sorted
        dp_solution = {}
    
    # Asignar ítems restantes (no procesados por DP) con heurística balanceada
    if items_remaining:
        # Calcular cargas actuales de los bins (incluyendo asignaciones de DP)
        bin_loads = list(base_values)
        bin_weights = [bin.current_weight for bin in residual_bins]
        
        for item_id, bin_idx in dp_solution.items():
            # Encontrar el ítem y actualizar cargas
            for item in items_for_dp:
                if item.id == item_id:
                    # bin_idx es el id del bin, encontrar su índice
                    for j, bin in enumerate(residual_bins):
                        if bin.id == bin_idx:
                            bin_loads[j] += item.value
                            bin_weights[j] += item.weight
                            break
                    break
        
        # Asignar ítems restantes balanceando cargas
        for item in items_remaining:
            # Encontrar bin con menor carga que tenga capacidad
            best_bin_idx = None
            best_load = float('inf')
            
            for j in range(k):
                if bin_weights[j] + item.weight <= residual_bins[j].capacity:
                    if bin_loads[j] < best_load:
                        best_load = bin_loads[j]
                        best_bin_idx = j
            
            # Si no cabe en ninguno, usar el de mayor capacidad restante
            if best_bin_idx is None:
                best_bin_idx = max(range(k), 
                                   key=lambda j: residual_bins[j].capacity - bin_weights[j])
            
            dp_solution[item.id] = residual_bins[best_bin_idx].id
            bin_loads[best_bin_idx] += item.value
            bin_weights[best_bin_idx] += item.weight
    
    return dp_solution

def merge_assignments(sa_solution: Dict[int, int], dp_solution: Dict[int, int]) -> Dict[int, int]:
    """Combina las soluciones de SA y DP en una solución completa."""
    full_solution = sa_solution.copy()
    full_solution.update(dp_solution)
    return full_solution

def local_search(solution: Dict[int, int], items: List[Item], bins: List[Bin],
                max_iterations: int = 1000, max_time: float = 30.0) -> Dict[int, int]:
    """
    Búsqueda local para mejorar la solución final mediante movimientos de 2-3 ítems.
    MODIFICACIÓN: Añadido límite de tiempo.
    """
    start_time = time.time()
    
    current_solution = solution.copy()
    current_cost = calculate_cost(current_solution, items, bins)
    
    # Pre-calcular estructuras para acceso rápido
    item_dict = {item.id: item for item in items}
    bin_ids = [bin.id for bin in bins]
    
    for _ in range(max_iterations):
        # Verificar límite de tiempo
        if time.time() - start_time >= max_time:
            break
        
        improved = False
        
        # Intentar movimientos de 2 ítems
        item_ids = list(current_solution.keys())
        if len(item_ids) >= 2:
            for _ in range(10):  # Probar 10 movimientos aleatorios
                item1_id, item2_id = random.sample(item_ids, 2)
                bin1_id, bin2_id = current_solution[item1_id], current_solution[item2_id]
                
                if bin1_id == bin2_id:
                    continue
                    
                # Probar intercambio
                new_solution = current_solution.copy()
                new_solution[item1_id] = bin2_id
                new_solution[item2_id] = bin1_id
                
                new_cost = calculate_cost(new_solution, items, bins)
                if new_cost < current_cost:
                    current_solution = new_solution
                    current_cost = new_cost
                    improved = True
                    break
        
        # Si no se mejoró con 2-ítems, intentar movimientos simples
        if not improved and item_ids:
            for _ in range(10):
                item_id = random.choice(item_ids)
                current_bin = current_solution[item_id]
                new_bin_id = random.choice([bid for bid in bin_ids if bid != current_bin])
                
                new_solution = current_solution.copy()
                new_solution[item_id] = new_bin_id
                
                new_cost = calculate_cost(new_solution, items, bins)
                if new_cost < current_cost:
                    current_solution = new_solution
                    current_cost = new_cost
                    improved = True
                    break
        
        if not improved:
            break
    
    return current_solution

def hybrid_sa_dp(items: List[Item], bins: List[Bin], alpha: float = 0.65, 
                epsilon: float = 0.1, verbose: bool = False, 
                max_time: float = 120.0) -> Dict[int, int]:
    """
    Algoritmo híbrido SA-DP para empaquetamiento balanceado en contenedores heterogéneos.
    
    Args:
        items: Lista de ítems a empaquetar, cada uno con peso y valor
        bins: Lista de contenedores con capacidades heterogéneas
        alpha: Proporción de ítems "grandes" para SA (0-1)
        epsilon: Parámetro de precisión para FPTAS en DP
        verbose: Si es True, imprime información de depuración
        max_time: Tiempo máximo total de ejecución en segundos (default: 120s = 2 min)
        
    Returns:
        Diccionario con asignación de ítems a contenedores {item_id: bin_id}
    """
    start_time = time.time()
    
    if verbose:
        print(f"Iniciando algoritmo híbrido SA-DP con {len(items)} ítems y {len(bins)} contenedores")
    
    # 1. Preprocesamiento - ordenar ítems por peso descendente
    items_sorted = sorted(items, key=lambda x: x.weight, reverse=True)
    
    # Calcular umbral tau
    total_capacity = sum(bin.capacity for bin in bins)
    tau = total_capacity / (len(bins) * max(1, len(items)))
    
    if verbose:
        print(f"Umbral tau calculado: {tau:.2f}")
    
    # 2. Fragmentación adaptativa
    # Seleccionar ítems grandes o top alpha*n por peso
    heavy_items = [item for item in items_sorted if item.weight >= tau]
    max_heavy = max(1, int(alpha * len(items)))
    I_h = heavy_items[:max_heavy]
    I_dp = [item for item in items if item not in I_h]
    
    # MODIFICACIÓN: Limitar I_dp a máximo 12 ítems
    max_dp_items = 12
    if len(I_dp) > max_dp_items:
        # Ajustar I_h para que I_dp tenga exactamente 12 ítems
        num_to_move = len(I_dp) - max_dp_items
        # Los últimos items de I_dp (más ligeros) se mueven a I_h
        items_to_move = I_dp[-num_to_move:]
        I_h = I_h + items_to_move
        I_dp = I_dp[:max_dp_items]
    
    if verbose:
        print(f"Fragmentación: |I_h| = {len(I_h)}, |I_dp| = {len(I_dp)}")
    
    # 3. Simulated Annealing para I_h
    if I_h:
        initial_sol = ffd_initialization(I_h, bins)
        T0 = calculate_initial_temp(I_h, bins)
        
        if verbose:
            print(f"Temperatura inicial para SA: {T0:.2f}")
        
        # MODIFICACIÓN: Calcular tiempo restante para SA (reservar 10s para DP y local_search)
        elapsed = time.time() - start_time
        remaining_time = max(10.0, max_time - elapsed - 10.0)  # Reservar 10s para fases posteriores
        
        best_sol = simulated_annealing(
            initial_sol, I_h, bins,
            T0=T0,
            cooling_rate=0.95,
            min_temp=0.001,
            max_iterations=5000,
            no_improve_limit=500,
            max_time=remaining_time,  # Usar tiempo restante
            convergence_limit=10000
        )
    else:
        best_sol = {}
    
    sa_time = time.time()
    if verbose:
        print(f"SA completado en {sa_time - start_time:.2f} segundos")
    
    # 4. Actualizar capacidades residuales
    residual_bins = update_bin_residuals(bins, best_sol, I_h)
    base_values = [bin.current_value for bin in residual_bins]
    
    # Verificar factibilidad
    infeasible = any(bin.current_weight > bin.capacity for bin in residual_bins)
    if infeasible and verbose:
        print("¡ADVERTENCIA! Solución de SA no factible. Se aplicará reparación.")
    
    # 5. Programación Dinámica para I_dp
    dp_solution = {}
    if I_dp:
        # Verificar si queda tiempo
        elapsed = time.time() - start_time
        time_for_dp = max(5.0, max_time - elapsed - 5.0)  # Reservar 5s para local_search
        
        if elapsed < max_time - 5.0:
            dp_start = time.time()
            dp_solution = dp_balance(I_dp, residual_bins, base_values, epsilon, max_time=time_for_dp)
            dp_time = time.time() - dp_start
            
            if verbose:
                print(f"DP completado en {dp_time:.2f} segundos")
        elif verbose:
            print("DP omitida por límite de tiempo")
    
    # 6. Combinar soluciones
    full_assignment = merge_assignments(best_sol, dp_solution)
    
    # 7. Post-optimización: Búsqueda local en frontera
    # Calcular tiempo restante para local_search
    elapsed = time.time() - start_time
    remaining_time = max(1.0, max_time - elapsed)
    final_solution = local_search(full_assignment, items, bins, max_iterations=200, max_time=remaining_time)
    
    total_time = time.time() - start_time
    if verbose:
        final_cost = calculate_cost(final_solution, items, bins)
        print(f"Algoritmo completado en {total_time:.2f} segundos")
        print(f"Costo final: {final_cost:.2f}")
    
    return final_solution

def generate_random_instance(num_items: int, num_bins: int, 
                           weight_range: Tuple[float, float] = (1.0, 100.0),
                           value_range: Tuple[float, float] = (1.0, 50.0),
                           capacity_factor: float = 1.2) -> Tuple[List[Item], List[Bin]]:
    """
    Genera una instancia aleatoria del problema.
    
    Args:
        num_items: Número de ítems a generar
        num_bins: Número de contenedores a generar
        weight_range: Rango para pesos de ítems (min, max)
        value_range: Rango para valores de ítems (min, max)
        capacity_factor: Factor para calcular capacidades de contenedores (relativo al promedio)
    
    Returns:
        Tupla con (lista de ítems, lista de contenedores)
    """
    # Generar ítems
    items = []
    for i in range(num_items):
        weight = random.uniform(weight_range[0], weight_range[1])
        value = random.uniform(value_range[0], value_range[1])
        items.append(Item(i, weight, value))
    
    # Calcular capacidad total necesaria
    total_weight = sum(item.weight for item in items)
    avg_capacity = total_weight * capacity_factor / num_bins
    
    # Generar contenedores con capacidades heterogéneas
    bins = []
    for j in range(num_bins):
        # Variación del 20% alrededor de la capacidad promedio
        capacity = avg_capacity * random.uniform(0.8, 1.2)
        bins.append(Bin(j, capacity))
    
    return items, bins

def evaluate_solution(solution: Dict[int, int], items: List[Item], bins: List[Bin]) -> Dict[str, float]:
    """
    Evalúa una solución y retorna métricas de desempeño.
    """
    # Reiniciar contenedores
    for bin in bins:
        bin.reset()
    
    # Aplicar solución
    for item in items:
        bin_id = solution[item.id]
        for bin in bins:
            if bin.id == bin_id:
                bin.add_item(item)
                break
    
    # Calcular métricas
    bin_values = [bin.current_value for bin in bins]
    bin_weights = [bin.current_weight for bin in bins]
    capacities = [bin.capacity for bin in bins]
    
    # Desbalance de valores
    max_value = max(bin_values)
    min_value = min(bin_values)
    balance = max_value - min_value
    rel_balance = balance / max(1e-6, sum(bin_values) / len(bins)) * 100
    
    # Utilización de capacidades
    utilizations = [w / c for w, c in zip(bin_weights, capacities)]
    avg_utilization = sum(utilizations) / len(utilizations) * 100
    min_utilization = min(utilizations) * 100
    max_utilization = max(utilizations) * 100
    
    # Factibilidad
    overloads = [max(0, w - c) for w, c in zip(bin_weights, capacities)]
    is_feasible = all(w <= c for w, c in zip(bin_weights, capacities))
    
    return {
        "desbalance_absoluto": balance,
        "desbalance_relativo": rel_balance,
        "utilizacion_promedio": avg_utilization,
        "utilizacion_min": min_utilization,
        "utilizacion_max": max_utilization,
        "es_factible": is_feasible,
        "sobrecarga_total": sum(overloads)
    }

def main_example():
    """
    Ejemplo de uso del algoritmo híbrido.
    """
    # Generar instancia de ejemplo
    random.seed(42)  # Para reproducibilidad
    items, bins = generate_random_instance(
        num_items=100,
        num_bins=5,
        weight_range=(1, 30),
        value_range=(1, 20),
        capacity_factor=1.3
    )
    
    print("=== Instancia Generada ===")
    print(f"Número de ítems: {len(items)}")
    print(f"Número de contenedores: {len(bins)}")
    print("\nCapacidades de contenedores:")
    for bin in bins:
        print(f"  {bin}")
    print(f"\nPeso total de ítems: {sum(item.weight for item in items):.2f}")
    print(f"Capacidad total de contenedores: {sum(bin.capacity for bin in bins):.2f}")
    
    # Ejecutar algoritmo híbrido
    print("\n=== Ejecutando Algoritmo Híbrido SA-DP ===")
    solution = hybrid_sa_dp(
        items, 
        bins, 
        alpha=0.65, 
        epsilon=0.1,
        verbose=True
    )
    
    # Evaluar solución
    metrics = evaluate_solution(solution, items, bins)
    
    print("\n=== Resultados de la Solución ===")
    print(f"Factibilidad: {'SÍ' if metrics['es_factible'] else 'NO'}")
    print(f"Desbalance relativo: {metrics['desbalance_relativo']:.2f}%")
    print(f"Utilización promedio: {metrics['utilizacion_promedio']:.2f}%")
    print(f"Utilización mínima: {metrics['utilizacion_min']:.2f}%")
    print(f"Utilización máxima: {metrics['utilizacion_max']:.2f}%")
    
    # Mostrar asignación por contenedor
    bin_assignments = defaultdict(list)
    for item_id, bin_id in solution.items():
        bin_assignments[bin_id].append(item_id)
    
    print("\nAsignación por contenedor:")
    for bin_id, item_ids in sorted(bin_assignments.items()):
        bin_items = [item for item in items if item.id in item_ids]
        total_weight = sum(item.weight for item in bin_items)
        total_value = sum(item.value for item in bin_items)
        bin = next(b for b in bins if b.id == bin_id)
        print(f"Contenedor {bin_id}: {len(item_ids)} ítems, "
              f"peso={total_weight:.2f}/{bin.capacity:.2f}, "
              f"valor={total_value:.2f}")
    
    return solution, metrics

if __name__ == "__main__":
    # Ejecutar ejemplo
    solution, metrics = main_example()

# === CÓMO INVOCAR EL ALGORITMO ===
# 
# Para usar el algoritmo en tu propio código:
# 
# 1. Importa las clases y funciones necesarias:
#    from este_archivo import Item, Bin, hybrid_sa_dp, evaluate_solution
# 
# 2. Crea tus ítems y contenedores:
#    items = [
#        Item(0, weight=10.5, value=20.0),
#        Item(1, weight=5.2, value=15.0),
#        # ... más ítems
#    ]
#    bins = [
#        Bin(0, capacity=50.0),
#        Bin(1, capacity=30.0),
#        # ... más contenedores
#    ]
# 
# 3. Ejecuta el algoritmo:
#    solution = hybrid_sa_dp(
#        items, 
#        bins,
#        alpha=0.65,    # Proporción de ítems para SA (0-1)
#        epsilon=0.1,   # Precisión para DP (menor = más preciso pero más lento)
#        verbose=True   # Muestra información detallada durante la ejecución
#    )
# 
# 4. Evalúa los resultados:
#    metrics = evaluate_solution(solution, items, bins)
#    print(f"Desbalance relativo: {metrics['desbalance_relativo']:.2f}%")
# 
# 5. Usa la solución (diccionario {item_id: bin_id}):
#    for item_id, bin_id in solution.items():
#        print(f"Ítem {item_id} asignado al contenedor {bin_id}")
# 
# Parámetros recomendados:
# - alpha: 0.65 para instancias medianas (50-200 ítems)
# - epsilon: 0.1 para balance calidad/rendimiento
# - Para instancias más grandes, aumenta alpha (0.75-0.85)
# - Para mayor precisión, reduce epsilon (0.05)
# 
# Requisitos:
# - Python 3.7+
# - numpy
# 
# Instalación de dependencias:
# pip install numpy