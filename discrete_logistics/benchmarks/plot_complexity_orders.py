"""
Gráfico de Comparación de Órdenes de Complejidad Teóricos.

Genera un gráfico comparativo mostrando las curvas teóricas de complejidad
de los diferentes algoritmos implementados, con n en el eje X y los demás
parámetros fijos.

Este gráfico ilustra visualmente el crecimiento asintótico de cada algoritmo.
"""
from pathlib import Path
import sys
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Allow running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_PNG = RESULTS_DIR / "complexity_orders_comparison.png"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Parámetros fijos
K_FIXED = 3  # número de contenedores
ITERATIONS_METAHEURISTICS = 1000  # iteraciones para metaheurísticas
POPULATION_GA = 100  # población para algoritmos genéticos
GENERATIONS_GA = 100  # generaciones para algoritmos genéticos

# Rango de n para el gráfico
N_MIN = 5
N_MAX = 30  # Limitado para que las curvas exponenciales sean visibles
N_POINTS = 100


def complexity_brute_force(n, k):
    """O(k^n * n) - Exponencial fuerte"""
    return k**n * n


def complexity_branch_bound(n, k):
    """O(k^n) peor caso - Exponencial"""
    return k**n


def complexity_dynamic_programming(n, k):
    """O(k^2 * 3^n) - Exponencial muy fuerte"""
    return k**2 * 3**n


def complexity_greedy_nk(n, k):
    """O(n log n + n*k) - Cuasi-lineal con FFD, BFD"""
    return n * np.log2(n + 1) + n * k


def complexity_greedy_nlogk(n, k):
    """O(n log n + n log k) - Cuasi-lineal con WFD, LPT, RoundRobin"""
    return n * np.log2(n + 1) + n * np.log2(k + 1)


def complexity_ldf(n, k):
    """O(n^2 * k) - Cuadrático"""
    return n**2 * k


def complexity_kk(n):
    """O(n log n) - Cuasi-lineal (KK)"""
    return n * np.log2(n + 1)


def complexity_simulated_annealing(n, iterations):
    """O(I * n) - Lineal con iteraciones"""
    return iterations * n


def complexity_genetic_algorithm(n, population, generations):
    """O(G * P * n) - Lineal con población y generaciones"""
    return generations * population * n


def complexity_tabu_search(n, iterations):
    """O(I * N) donde N es el tamaño del vecindario ~ n"""
    return iterations * n


def plot_complexity_orders():
    """Genera el gráfico de comparación de órdenes de complejidad."""
    logging.info("="*70)
    logging.info("GENERANDO GRÁFICO DE ÓRDENES DE COMPLEJIDAD")
    logging.info("="*70)
    logging.info(f"Parámetros fijos:")
    logging.info(f"  - k (contenedores): {K_FIXED}")
    logging.info(f"  - Iteraciones metaheurísticas: {ITERATIONS_METAHEURISTICS}")
    logging.info(f"  - Población GA: {POPULATION_GA}")
    logging.info(f"  - Generaciones GA: {GENERATIONS_GA}")
    logging.info(f"Rango de n: {N_MIN} - {N_MAX}")
    logging.info("")
    
    # Crear rango de n
    n_values = np.linspace(N_MIN, N_MAX, N_POINTS)
    
    # Calcular valores de complejidad para cada algoritmo
    logging.info("Calculando valores de complejidad...")
    
    # Algoritmos exactos (exponenciales)
    brute_force_vals = np.array([complexity_brute_force(n, K_FIXED) for n in n_values])
    branch_bound_vals = np.array([complexity_branch_bound(n, K_FIXED) for n in n_values])
    dp_vals = np.array([complexity_dynamic_programming(n, K_FIXED) for n in n_values])
    
    # Algoritmos greedy (polinomiales/cuasi-lineales)
    greedy_nk_vals = np.array([complexity_greedy_nk(n, K_FIXED) for n in n_values])
    greedy_nlogk_vals = np.array([complexity_greedy_nlogk(n, K_FIXED) for n in n_values])
    ldf_vals = np.array([complexity_ldf(n, K_FIXED) for n in n_values])
    kk_vals = np.array([complexity_kk(n) for n in n_values])
    
    # Metaheurísticas (lineales con parámetros)
    sa_vals = np.array([complexity_simulated_annealing(n, ITERATIONS_METAHEURISTICS) for n in n_values])
    ga_vals = np.array([complexity_genetic_algorithm(n, POPULATION_GA, GENERATIONS_GA) for n in n_values])
    tabu_vals = np.array([complexity_tabu_search(n, ITERATIONS_METAHEURISTICS) for n in n_values])
    
    logging.info("✓ Valores calculados")
    logging.info("")
    
    # Crear figura con 2 subplots: uno para todos, otro sin exponenciales
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Subplot 1: Todos los algoritmos (escala logarítmica) ---
    logging.info("Generando subplot 1: Todos los algoritmos (escala log)...")
    
    # Algoritmos exactos
    ax1.plot(n_values, brute_force_vals, 'r-', linewidth=2.5, label='Brute Force: $O(k^n \\cdot n)$', alpha=0.8)
    ax1.plot(n_values, branch_bound_vals, 'darkred', linewidth=2.5, label='Branch & Bound: $O(k^n)$', linestyle='--', alpha=0.8)
    ax1.plot(n_values, dp_vals, 'purple', linewidth=2.5, label='Dynamic Programming: $O(k^2 \\cdot 3^n)$', linestyle='-.', alpha=0.8)
    
    # Algoritmos greedy
    ax1.plot(n_values, greedy_nk_vals, 'blue', linewidth=2, label='FFD/BFD: $O(n \\log n + n \\cdot k)$', alpha=0.7)
    ax1.plot(n_values, greedy_nlogk_vals, 'cyan', linewidth=2, label='WFD/LPT/RR: $O(n \\log n + n \\log k)$', alpha=0.7)
    ax1.plot(n_values, ldf_vals, 'green', linewidth=2, label='LDF: $O(n^2 \\cdot k)$', alpha=0.7)
    ax1.plot(n_values, kk_vals, 'lightgreen', linewidth=2, label='KK: $O(n \\log n)$', alpha=0.7)
    
    # Metaheurísticas
    ax1.plot(n_values, sa_vals, 'orange', linewidth=2, label=f'Sim. Annealing: $O(I \\cdot n)$, $I={ITERATIONS_METAHEURISTICS}$', linestyle=':', alpha=0.7)
    ax1.plot(n_values, ga_vals, 'brown', linewidth=2, label=f'Genetic: $O(G \\cdot P \\cdot n)$, $G={GENERATIONS_GA}, P={POPULATION_GA}$', linestyle=':', alpha=0.7)
    ax1.plot(n_values, tabu_vals, 'pink', linewidth=2, label=f'Tabu Search: $O(I \\cdot n)$, $I={ITERATIONS_METAHEURISTICS}$', linestyle=':', alpha=0.7)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Número de ítems (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Operaciones (escala logarítmica)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Comparación de Órdenes de Complejidad (k={K_FIXED})\nEscala Logarítmica', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # --- Subplot 2: Solo algoritmos polinomiales/cuasi-lineales (escala lineal) ---
    logging.info("Generando subplot 2: Algoritmos polinomiales (escala lineal)...")
    
    # Solo greedy y metaheurísticas
    ax2.plot(n_values, greedy_nk_vals, 'blue', linewidth=2.5, label='FFD/BFD: $O(n \\log n + n \\cdot k)$', alpha=0.8)
    ax2.plot(n_values, greedy_nlogk_vals, 'cyan', linewidth=2.5, label='WFD/LPT/RR: $O(n \\log n + n \\log k)$', alpha=0.8)
    ax2.plot(n_values, ldf_vals, 'green', linewidth=2.5, label='LDF: $O(n^2 \\cdot k)$', alpha=0.8)
    ax2.plot(n_values, kk_vals, 'lightgreen', linewidth=2.5, label='KK: $O(n \\log n)$', alpha=0.8)
    ax2.plot(n_values, sa_vals, 'orange', linewidth=2.5, label=f'Sim. Annealing: $O(I \\cdot n)$', linestyle='--', alpha=0.8)
    ax2.plot(n_values, ga_vals, 'brown', linewidth=2.5, label=f'Genetic: $O(G \\cdot P \\cdot n)$', linestyle='--', alpha=0.8)
    ax2.plot(n_values, tabu_vals, 'pink', linewidth=2.5, label=f'Tabu Search: $O(I \\cdot n)$', linestyle='--', alpha=0.8)
    
    ax2.set_xlabel('Número de ítems (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Operaciones', fontsize=12, fontweight='bold')
    ax2.set_title(f'Algoritmos Polinomiales y Metaheurísticas (k={K_FIXED})\nEscala Lineal', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Formatear eje Y con separadores de miles
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    
    # Guardar
    logging.info("")
    logging.info("Guardando gráfico...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    logging.info(f"✓ Gráfico guardado: {OUTPUT_PNG}")
    
    # Mostrar
    logging.info("")
    logging.info("="*70)
    logging.info("GRÁFICO GENERADO EXITOSAMENTE")
    logging.info("="*70)
    logging.info(f"📊 Archivo: {OUTPUT_PNG}")
    logging.info("")
    logging.info("Interpretación:")
    logging.info("  • Subplot 1 (izquierda): Muestra todos los algoritmos en escala logarítmica")
    logging.info("    - Algoritmos exponenciales crecen dramáticamente")
    logging.info("    - Algoritmos polinomiales son líneas casi planas en comparación")
    logging.info("")
    logging.info("  • Subplot 2 (derecha): Enfoque en algoritmos prácticos (escala lineal)")
    logging.info("    - KK y greedy cuasi-lineales: crecimiento muy lento")
    logging.info("    - LDF cuadrático: crece más rápido pero aún manejable")
    logging.info("    - Metaheurísticas: lineales con parámetros fijos")
    logging.info("")
    logging.info("💡 Conclusión: Los algoritmos exponenciales son inviables para n > 20,")
    logging.info("   mientras que greedy y metaheurísticas escalan bien a problemas grandes.")
    logging.info("="*70)


def main():
    """Función principal."""
    start_time = datetime.now()
    logging.info("")
    logging.info("#"*70)
    logging.info("# GENERACIÓN DE GRÁFICO DE ÓRDENES DE COMPLEJIDAD")
    logging.info("#"*70)
    logging.info(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("")
    
    plot_complexity_orders()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info("")
    logging.info(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Duración: {duration:.2f} segundos")
    logging.info("#"*70)
    logging.info("")
    logging.info("✅ Generación completada exitosamente")


if __name__ == "__main__":
    main()
