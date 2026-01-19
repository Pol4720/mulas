import random
import copy
import math
import time
from typing import List, Tuple

# --- ESTRUCTURAS DE DATOS ---

class Item:
    def __init__(self, id: int, weight: int, value: int):
        self.id = id
        self.weight = weight
        self.value = value
        self.ratio = value / weight if weight > 0 else 0

    def __repr__(self):
        return f"Item(id={self.id}, w={self.weight}, v={self.value})"

class Bin:
    def __init__(self, id: int, capacity: int):
        self.id = id
        self.capacity = capacity
        self.current_weight = 0
        self.current_value = 0
        self.items = []

    def can_fit(self, item: Item) -> bool:
        return self.current_weight + item.weight <= self.capacity

    def add_item(self, item: Item):
        self.items.append(item)
        self.current_weight += item.weight
        self.current_value += item.value

    def clear(self):
        self.items = []
        self.current_weight = 0
        self.current_value = 0

    def copy(self):
        new_bin = Bin(self.id, self.capacity)
        new_bin.current_weight = self.current_weight
        new_bin.current_value = self.current_value
        new_bin.items = self.items[:]
        return new_bin

    def __repr__(self):
        return f"Bin {self.id} [W:{self.current_weight}/{self.capacity}, V:{self.current_value}]"

# --- FASE 2: METAHEURÍSTICA (CORE - GA) ---

class GeneticCoreSolver:
    def __init__(self, items: List[Item], bins: List[Bin], pop_size=50, generations=100, 
                 mutation_rate=0.1, max_time=120.0, convergence_limit=10000):
        self.items = items
        self.bins_template = bins
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.k = len(bins)
        # MODIFICACIÓN: Añadir límite de tiempo (2 minutos) y convergencia
        self.max_time = max_time
        self.convergence_limit = convergence_limit
        
        # Penalización severa para infactibilidad
        self.penalty_weight = 10000 

    def _calculate_fitness(self, chromosome: List[int]) -> float:
        """
        Retorna el desbalance (Spread). Si es infactible, suma penalización.
        Objetivo: Minimizar este valor.
        """
        # Reiniciar contadores temporales (más rápido que crear objetos Bin completos)
        bin_weights = [0] * self.k
        bin_values = [0] * self.k
        penalty = 0

        for i, bin_idx in enumerate(chromosome):
            item = self.items[i]
            bin_weights[bin_idx] += item.weight
            bin_values[bin_idx] += item.value
            
            # Chequeo de capacidad
            if bin_weights[bin_idx] > self.bins_template[bin_idx].capacity:
                penalty += (bin_weights[bin_idx] - self.bins_template[bin_idx].capacity) * self.penalty_weight

        spread = max(bin_values) - min(bin_values)
        return spread + penalty

    def solve(self) -> Tuple[List[int], float]:
        # MODIFICACIÓN: Iniciar temporizador
        start_time = time.time()
        
        # Población Inicial: Lista de cromosomas (cada uno es una lista de índices de bins)
        population = [[random.randint(0, self.k - 1) for _ in self.items] for _ in range(self.pop_size)]
        
        best_fitness = float('inf')
        no_improvement = 0
        
        for gen in range(self.generations):
            # MODIFICACIÓN: Verificar límite de tiempo
            if time.time() - start_time >= self.max_time:
                break
            
            # Evaluar
            fitness_scores = [(ind, self._calculate_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1]) # Ordenar por menor fitness (minimización)
            
            # MODIFICACIÓN: Verificar convergencia
            current_best = fitness_scores[0][1]
            if current_best < best_fitness:
                best_fitness = current_best
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= self.convergence_limit:
                break
            
            # Elitismo: Guardar el mejor
            next_gen = [fitness_scores[0][0]]
            
            # Selección y Crossover (Torneo)
            while len(next_gen) < self.pop_size:
                parent1 = self._tournament(fitness_scores)
                parent2 = self._tournament(fitness_scores)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                next_gen.append(child)
            
            population = next_gen

        # Retornar mejor solución encontrada
        best_chrom = min(population, key=self._calculate_fitness)
        return best_chrom, self._calculate_fitness(best_chrom)

    def _tournament(self, ranked_pop):
        # Selecciona 3 al azar y se queda con el mejor
        candidates = random.sample(ranked_pop, 3)
        return min(candidates, key=lambda x: x[1])[0]

    def _crossover(self, p1, p2):
        # Uniform Crossover
        return [p1[i] if random.random() < 0.5 else p2[i] for i in range(len(p1))]

    def _mutate(self, chrom):
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(chrom) - 1)
            chrom[idx] = random.randint(0, self.k - 1)
        return chrom

# --- FASE 3: PROGRAMACIÓN DINÁMICA (TAIL - KNAPSACK) ---

def solve_knapsack(items: List[Item], capacity: int) -> List[Item]:
    """
    Resuelve 0/1 Knapsack maximizando Valor sujeto a Capacidad.
    Retorna la lista de ítems seleccionados.
    MODIFICACIÓN: Limitada a máximo 12 ítems.
    """
    if not items or capacity <= 0:
        return []

    n = len(items)
    
    # MODIFICACIÓN: Limitar DP a máximo 12 ítems
    if n > 12:
        items = items[:12]
        n = 12
    
    # Tabla DP: K[i][w] = max valor con items 0..i y peso w
    # Nota: Si la capacidad es gigante (>10,000), usar un diccionario o escalado.
    # Aquí asumimos capacidades razonables para la demo.
    K = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        item = items[i-1]
        for w in range(1, capacity + 1):
            if item.weight <= w:
                K[i][w] = max(item.value + K[i-1][w - item.weight], K[i-1][w])
            else:
                K[i][w] = K[i-1][w]

    # Reconstrucción de la solución (Backtracking)
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if K[i][w] != K[i-1][w]:
            selected.append(items[i-1])
            w -= items[i-1].weight

    return selected

# --- ALGORITMO PRINCIPAL: H-GADP ---

def h_gadp(items: List[Item], bins_input: List[Bin], alpha=0.3):
    """
    Hybrid Genetic Algorithm with Dynamic Programming
    """
    start_time = time.time()
    
    # 0. Preprocesamiento y Clonado
    # Importante: Trabajar con copias para no mutar entradas externas
    bins = [b.copy() for b in bins_input]
    
    # Ordenar items por valor descendente
    sorted_items = sorted(items, key=lambda x: x.value, reverse=True)
    
    # Corte Alpha
    cut_idx = int(len(items) * alpha)
    core_items = sorted_items[:cut_idx]
    tail_items = sorted_items[cut_idx:]
    
    # MODIFICACIÓN: Limitar tail_items a máximo 12 ítems
    max_tail_items = 12
    if len(tail_items) > max_tail_items:
        # Mover el exceso al core
        excess = tail_items[max_tail_items:]
        core_items = core_items + excess
        tail_items = tail_items[:max_tail_items]
    
    print(f"--- Iniciando H-GADP ---")
    print(f"Total Items: {len(items)} | Core: {len(core_items)} | Tail: {len(tail_items)}")

    # 1. Fase Core (Genetic Algorithm)
    print("Ejecutando Fase 1: GA en Core Items...")
    # MODIFICACIÓN: Añadir parámetros de tiempo y convergencia
    ga_solver = GeneticCoreSolver(core_items, bins, pop_size=40, generations=50,
                                   max_time=120.0, convergence_limit=10000)
    best_chrom, _ = ga_solver.solve()

    # Asignar items core a los bins
    for i, bin_idx in enumerate(best_chrom):
        bins[bin_idx].add_item(core_items[i])

    # 2. Fase Tail (Dynamic Programming)
    print("Ejecutando Fase 2: DP Secuencial en Tail Items...")
    
    # Calcular target ideal (promedio)
    total_val = sum(i.value for i in items)
    target_val = total_val / len(bins)
    
    # Ordenar bins: Los que están más lejos del target (más pobres) van primero
    # para darles oportunidad de elegir los mejores items del tail.
    bins.sort(key=lambda b: b.current_value) # Menor valor primero

    remaining_tail = tail_items[:]
    
    for b in bins:
        if not remaining_tail:
            break
            
        residual_cap = b.capacity - b.current_weight
        
        # Si el bin ya está muy lleno o supera el target, pasamos (o le damos poco)
        # Queremos maximizar valor en este espacio libre
        if residual_cap > 0:
            chosen = solve_knapsack(remaining_tail, residual_cap)
            
            # Asignar y remover del pool
            for item in chosen:
                b.add_item(item)
                remaining_tail.remove(item)

    # 3. Fase Limpieza (Greedy Best Fit)
    # Si sobraron items porque la DP no encontró hueco perfecto
    print(f"Ejecutando Fase 3: Greedy para {len(remaining_tail)} items residuales...")
    
    for item in remaining_tail:
        # Encontrar el bin donde quepa y minimize el nuevo desbalance
        best_bin = None
        min_added_imbalance = float('inf')
        
        current_vals = [b.current_value for b in bins]
        
        for b in bins:
            if b.can_fit(item):
                # Simular asignación
                # El nuevo desbalance se estima simplificadamente
                # como la distancia al target promedio
                dist = abs((b.current_value + item.value) - target_val)
                
                if dist < min_added_imbalance:
                    min_added_imbalance = dist
                    best_bin = b
        
        if best_bin:
            best_bin.add_item(item)
        else:
            print(f"WARNING: Item {item.id} (w={item.weight}) no pudo ser asignado (Contenedores llenos).")

    # --- RESULTADOS ---
    end_time = time.time()
    
    final_vals = [b.current_value for b in bins]
    final_weights = [b.current_weight for b in bins]
    imbalance = max(final_vals) - min(final_vals)
    
    print("\n--- Resultados Finales ---")
    for b in bins:
        print(b)
    print(f"Desbalance Final (Max V - Min V): {imbalance}")
    print(f"Tiempo de Ejecución: {end_time - start_time:.4f}s")
    
    return bins, imbalance

# --- BLOQUE DE EJECUCIÓN ---

if __name__ == "__main__":
    # --- CONFIGURACIÓN DEL ESCENARIO DE PRUEBA ---
    random.seed(42) # Reproducibilidad

    # 1. Crear Contenedores Heterogéneos
    # 3 Contenedores con capacidades variables
    capacities = [100, 120, 90] 
    bins_test = [Bin(id=i, capacity=c) for i, c in enumerate(capacities)]

    # 2. Generar Items (Simulación)
    # Generamos 50 items.
    # Unos pocos pesados y valiosos (Rocas), muchos ligeros (Arena)
    items_test = []
    
    # Generar items correlacionados (Valor ~ Peso) para hacerlo difícil
    for i in range(50):
        w = random.randint(1, 25)
        # El valor es proporcional al peso con algo de ruido
        v = int(w * 10 + random.randint(-5, 5))
        if v < 1: v = 1
        items_test.append(Item(i, w, v))

    # --- INVOCACIÓN DEL ALGORITMO ---
    print(">>> Iniciando Test de Algoritmo Híbrido H-GADP <<<")
    solution_bins, final_diff = h_gadp(
        items=items_test, 
        bins_input=bins_test, 
        alpha=0.30 # 30% items tratados por GA, 70% por DP
    )



# from h_gadp_optimizer import h_gadp, Item, Bin

# # 1. Definir tus datos
# mis_items = [Item(1, 10, 20), Item(2, 5, 15), Item(3, 8, 10)] # etc...
# mis_contenedores = [Bin(0, 50), Bin(1, 60)]

# # 2. Ejecutar
# resultado_bins, desbalance = h_gadp(mis_items, mis_contenedores, alpha=0.3)

# # 3. Ver resultados
# print(f"Desbalance logrado: {desbalance}")