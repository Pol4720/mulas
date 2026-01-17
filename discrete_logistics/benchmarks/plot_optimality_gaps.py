"""
Análisis de Gap de Optimalidad - Comparación de Heurísticas vs Soluciones Óptimas.

Calcula y visualiza el gap de optimalidad para los algoritmos aproximados/heurísticos
comparándolos con las soluciones óptimas obtenidas por Branch & Bound.

Gap(%) = ((Heurístico - Óptimo) / Óptimo) × 100

Genera gráficos mostrando:
1. Gap promedio por algoritmo
2. Gap vs tamaño de instancia (n)
3. Gap vs número de contenedores (k)
"""
from pathlib import Path
import sys
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Allow running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RESULTS_DIR = Path(__file__).resolve().parent / "results"
INPUT_CSV = RESULTS_DIR / "runtime_vs_size.csv"
OUTPUT_DIR = RESULTS_DIR
OUTPUT_PNG_1 = OUTPUT_DIR / "gap_by_algorithm.png"
OUTPUT_PNG_2 = OUTPUT_DIR / "gap_vs_size.png"
OUTPUT_PNG_3 = OUTPUT_DIR / "gap_vs_k.png"
OUTPUT_CSV = OUTPUT_DIR / "optimality_gaps.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Algoritmos a analizar (excluye exactos que son el baseline)
HEURISTIC_ALGORITHMS = [
    "FirstFitDecreasing",
    "BestFitDecreasing",
    "WorstFitDecreasing",
    "RoundRobinGreedy",
    "LargestDifferenceFirst",
    "LPT",
    "KK",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "TabuSearch"
]

# Mapeo de nombres para visualización
ALGORITHM_DISPLAY_NAMES = {
    "FirstFitDecreasing": "FFD",
    "BestFitDecreasing": "BFD",
    "WorstFitDecreasing": "WFD",
    "RoundRobinGreedy": "Round Robin",
    "LargestDifferenceFirst": "LDF",
    "LPT": "LPT",
    "KK": "KK",
    "SimulatedAnnealing": "Simulated Annealing",
    "GeneticAlgorithm": "Genetic Algorithm",
    "TabuSearch": "Tabu Search",
    "BranchAndBound": "Branch & Bound (Óptimo)"
}


def load_data() -> pd.DataFrame:
    """Carga los datos del benchmark."""
    logging.info("="*70)
    logging.info("CARGANDO DATOS")
    logging.info("="*70)
    
    if not INPUT_CSV.exists():
        logging.error(f"❌ Archivo no encontrado: {INPUT_CSV}")
        logging.error("Primero ejecuta: python discrete_logistics/benchmarks/run_algorithms.py")
        sys.exit(1)
    
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"✓ Datos cargados: {len(df)} filas")
    logging.info(f"  Algoritmos: {df['algorithm'].nunique()}")
    logging.info(f"  Problemas: {df['problem'].nunique()}")
    logging.info("")
    
    return df


def compute_optimality_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el gap de optimalidad para cada heurística.
    
    Gap(%) = ((Heurístico - Óptimo) / Óptimo) × 100
    
    Usa Branch & Bound como referencia de solución óptima.
    """
    logging.info("="*70)
    logging.info("CALCULANDO GAPS DE OPTIMALIDAD")
    logging.info("="*70)
    
    # Filtrar solo ejecuciones exitosas
    df_success = df[df['status'] == 'ok'].copy()
    
    # Obtener soluciones óptimas de Branch & Bound
    optimal_solutions = df_success[df_success['algorithm'] == 'BranchAndBound'][
        ['problem', 'objective']
    ].rename(columns={'objective': 'optimal_objective'})
    
    logging.info(f"✓ Soluciones óptimas disponibles: {len(optimal_solutions)} problemas")
    
    # Filtrar heurísticas
    df_heuristics = df_success[df_success['algorithm'].isin(HEURISTIC_ALGORITHMS)].copy()
    
    # Merge con soluciones óptimas
    df_gaps = df_heuristics.merge(optimal_solutions, on='problem', how='inner')
    
    logging.info(f"✓ Instancias con ambos resultados: {len(df_gaps)}")
    
    # Calcular gap absoluto primero
    df_gaps['gap_absolute'] = df_gaps['objective'] - df_gaps['optimal_objective']
    
    # Filtrar casos donde el óptimo es muy cercano a 0 (< 0.01)
    # Estos casos pueden causar gaps porcentuales extremadamente altos
    threshold = 0.01
    valid_cases = df_gaps['optimal_objective'] >= threshold
    
    logging.info(f"  Casos con óptimo >= {threshold}: {valid_cases.sum()} / {len(df_gaps)}")
    logging.info(f"  Casos filtrados (óptimo muy cercano a 0): {(~valid_cases).sum()}")
    
    # Calcular gap solo para casos válidos
    # Gap(%) = ((Heurístico - Óptimo) / Óptimo) × 100
    df_gaps['gap_percent'] = np.nan
    df_gaps.loc[valid_cases, 'gap_percent'] = (
        (df_gaps.loc[valid_cases, 'objective'] - df_gaps.loc[valid_cases, 'optimal_objective']) / 
        df_gaps.loc[valid_cases, 'optimal_objective']
    ) * 100
    
    # Para casos con óptimo cercano a 0, usar gap absoluto como indicador
    df_gaps.loc[~valid_cases, 'gap_percent'] = df_gaps.loc[~valid_cases, 'gap_absolute']
    
    # Asegurar gaps no negativos (no puede ser mejor que el óptimo)
    df_gaps['gap_percent'] = df_gaps['gap_percent'].clip(lower=0)
    
    # Eliminar outliers extremos (gaps > 1000%) para mejor visualización
    # Guardarlos en una columna separada
    df_gaps['is_outlier'] = df_gaps['gap_percent'] > 1000
    outliers_count = df_gaps['is_outlier'].sum()
    
    if outliers_count > 0:
        logging.info(f"  Outliers extremos detectados (gap > 1000%): {outliers_count}")
        logging.info(f"  Estos se limitarán a 1000% para visualización")
        df_gaps['gap_percent_display'] = df_gaps['gap_percent'].clip(upper=1000)
    else:
        df_gaps['gap_percent_display'] = df_gaps['gap_percent']
    
    logging.info("")
    logging.info("Resumen de gaps calculados:")
    logging.info(f"  Total de comparaciones: {len(df_gaps)}")
    
    # Estadísticas solo de casos válidos (no outliers extremos)
    valid_gaps = df_gaps[~df_gaps['is_outlier']]['gap_percent']
    logging.info(f"  Casos sin outliers: {len(valid_gaps)}")
    logging.info(f"  Gap promedio (sin outliers): {valid_gaps.mean():.2f}%")
    logging.info(f"  Gap mediano (sin outliers): {valid_gaps.median():.2f}%")
    logging.info(f"  Gap mínimo: {df_gaps['gap_percent'].min():.2f}%")
    logging.info(f"  Gap máximo: {df_gaps['gap_percent'].max():.2f}%")
    logging.info("")
    
    # Guardar resultados
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_gaps.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"✓ Gaps guardados en: {OUTPUT_CSV}")
    logging.info("")
    
    return df_gaps


def plot_gap_by_algorithm(df_gaps: pd.DataFrame) -> None:
    """Gráfico 1: Gap promedio por algoritmo."""
    logging.info("="*70)
    logging.info("GRÁFICO 1: GAP PROMEDIO POR ALGORITMO")
    logging.info("="*70)
    
    # Calcular estadísticas por algoritmo (usando gap_percent_display para limitar outliers)
    stats = df_gaps.groupby('algorithm')['gap_percent_display'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    # Ordenar por gap promedio
    stats = stats.sort_values('mean')
    
    # Mapear nombres
    stats['display_name'] = stats['algorithm'].map(ALGORITHM_DISPLAY_NAMES)
    
    logging.info("Gaps por algoritmo:")
    for _, row in stats.iterrows():
        logging.info(f"  {row['display_name']}: {row['mean']:.2f}% ± {row['std']:.2f}% "
                    f"(min={row['min']:.2f}%, max={row['max']:.2f}%, n={row['count']})")
    logging.info("")
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Barplot con error bars
    x_pos = np.arange(len(stats))
    bars = ax.bar(x_pos, stats['mean'], yerr=stats['std'], 
                   capsize=5, alpha=0.8, color='steelblue', edgecolor='black')
    
    # Colorear barras según gap
    colors = ['green' if gap < 5 else 'orange' if gap < 10 else 'red' 
              for gap in stats['mean']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.7)
    
    ax.set_xlabel('Algoritmo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap de Optimalidad (%)', fontsize=12, fontweight='bold')
    ax.set_title('Gap Promedio de Optimalidad por Algoritmo\n(con respecto a Branch & Bound)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats['display_name'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Añadir líneas de referencia
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Gap 5%')
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Gap 10%')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_1, dpi=300, bbox_inches='tight')
    logging.info(f"✓ Gráfico guardado: {OUTPUT_PNG_1}")
    logging.info("")


def plot_gap_vs_size(df_gaps: pd.DataFrame) -> None:
    """Gráfico 2: Gap vs tamaño de instancia (n)."""
    logging.info("="*70)
    logging.info("GRÁFICO 2: GAP VS TAMAÑO DE INSTANCIA")
    logging.info("="*70)
    
    # Mapear nombres para visualización
    df_gaps['display_name'] = df_gaps['algorithm'].map(ALGORITHM_DISPLAY_NAMES)
    
    # Usar gap_percent_display para visualización (con outliers limitados)
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot para cada algoritmo
    for algo in sorted(df_gaps['algorithm'].unique()):
        data = df_gaps[df_gaps['algorithm'] == algo]
        display_name = ALGORITHM_DISPLAY_NAMES[algo]
        
        # Calcular gap promedio por n
        gap_by_n = data.groupby('n_items')['gap_percent_display'].agg(['mean', 'std']).reset_index()
        
        # Plot línea con área de error
        ax.plot(gap_by_n['n_items'], gap_by_n['mean'], 
               marker='o', linewidth=2, label=display_name, alpha=0.8)
        ax.fill_between(gap_by_n['n_items'], 
                        gap_by_n['mean'] - gap_by_n['std'],
                        gap_by_n['mean'] + gap_by_n['std'],
                        alpha=0.2)
    
    ax.set_xlabel('Número de Ítems (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap de Optimalidad (%)', fontsize=12, fontweight='bold')
    ax.set_title('Evolución del Gap de Optimalidad con el Tamaño de Instancia', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Líneas de referencia
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_2, dpi=300, bbox_inches='tight')
    logging.info(f"✓ Gráfico guardado: {OUTPUT_PNG_2}")
    logging.info("")


def plot_gap_vs_k(df_gaps: pd.DataFrame) -> None:
    """Gráfico 3: Gap vs número de contenedores (k)."""
    logging.info("="*70)
    logging.info("GRÁFICO 3: GAP VS NÚMERO DE CONTENEDORES")
    logging.info("="*70)
    
    # Mapear nombres para visualización
    df_gaps['display_name'] = df_gaps['algorithm'].map(ALGORITHM_DISPLAY_NAMES)
    
    # Crear gráfico de caja (boxplot)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Preparar datos para boxplot
    k_values = sorted(df_gaps['k'].unique())
    data_by_k = []
    labels = []
    
    for algo in sorted(df_gaps['algorithm'].unique()):
        for k in k_values:
            data = df_gaps[(df_gaps['algorithm'] == algo) & (df_gaps['k'] == k)]['gap_percent_display']
            if len(data) > 0:
                data_by_k.append(data)
                labels.append(f"{ALGORITHM_DISPLAY_NAMES[algo]}\nk={k}")
    
    # Crear boxplot
    positions = np.arange(len(data_by_k))
    bp = ax.boxplot(data_by_k, positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='green', markersize=5))
    
    ax.set_xlabel('Algoritmo y Número de Contenedores', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap de Optimalidad (%)', fontsize=12, fontweight='bold')
    ax.set_title('Distribución del Gap de Optimalidad por Algoritmo y Número de Contenedores', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(positions[::len(k_values)])  # Un tick por algoritmo
    ax.set_xticklabels([ALGORITHM_DISPLAY_NAMES[algo] for algo in sorted(df_gaps['algorithm'].unique())],
                       rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Añadir líneas de referencia
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.3, label='Gap 5%')
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.3, label='Gap 10%')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_3, dpi=300, bbox_inches='tight')
    logging.info(f"✓ Gráfico guardado: {OUTPUT_PNG_3}")
    logging.info("")


def plot_gap_heatmap(df_gaps: pd.DataFrame) -> None:
    """Gráfico adicional: Heatmap de gaps por (n, k) para cada algoritmo."""
    logging.info("="*70)
    logging.info("GRÁFICO ADICIONAL: HEATMAP DE GAPS")
    logging.info("="*70)
    
    # Crear subplots para cada algoritmo
    algorithms = sorted(df_gaps['algorithm'].unique())
    n_algos = len(algorithms)
    n_cols = 3
    n_rows = (n_algos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_algos > 1 else [axes]
    
    for idx, algo in enumerate(algorithms):
        data = df_gaps[df_gaps['algorithm'] == algo]
        
        # Crear matriz pivot
        pivot = data.pivot_table(values='gap_percent', index='k', columns='n_items', aggfunc='mean')
        
        # Heatmap
        sns.heatmap(pivot, ax=axes[idx], cmap='RdYlGn_r', annot=True, fmt='.1f',
                   cbar_kws={'label': 'Gap (%)'}, vmin=0, vmax=20)
        axes[idx].set_title(ALGORITHM_DISPLAY_NAMES[algo], fontweight='bold')
        axes[idx].set_xlabel('Número de ítems (n)')
        axes[idx].set_ylabel('Número de contenedores (k)')
    
    # Ocultar subplots vacíos
    for idx in range(n_algos, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Heatmap de Gaps de Optimalidad por Tamaño de Instancia',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_heatmap = OUTPUT_DIR / "gap_heatmap.png"
    plt.savefig(output_heatmap, dpi=300, bbox_inches='tight')
    logging.info(f"✓ Heatmap guardado: {output_heatmap}")
    logging.info("")


def generate_summary_report(df_gaps: pd.DataFrame) -> None:
    """Genera un resumen textual de los resultados."""
    logging.info("="*70)
    logging.info("RESUMEN DE ANÁLISIS DE GAPS")
    logging.info("="*70)
    
    # Resumen general
    valid_gaps = df_gaps[~df_gaps['is_outlier']]['gap_percent']
    logging.info(f"Total de comparaciones: {len(df_gaps)}")
    logging.info(f"Casos sin outliers extremos: {len(valid_gaps)}")
    logging.info(f"Gap promedio (sin outliers): {valid_gaps.mean():.2f}%")
    logging.info(f"Gap mediano (sin outliers): {valid_gaps.median():.2f}%")
    logging.info("")
    
    # Top 3 mejores algoritmos (usando gap_percent_display)
    best = df_gaps.groupby('algorithm')['gap_percent_display'].mean().sort_values().head(3)
    logging.info("🏆 Top 3 Mejores Algoritmos (menor gap):")
    for i, (algo, gap) in enumerate(best.items(), 1):
        logging.info(f"  {i}. {ALGORITHM_DISPLAY_NAMES[algo]}: {gap:.2f}%")
    logging.info("")
    
    # Top 3 peores algoritmos
    worst = df_gaps.groupby('algorithm')['gap_percent_display'].mean().sort_values().tail(3)
    logging.info("⚠️  Top 3 Algoritmos con Mayor Gap:")
    for i, (algo, gap) in enumerate(worst.items(), 1):
        logging.info(f"  {i}. {ALGORITHM_DISPLAY_NAMES[algo]}: {gap:.2f}%")
    logging.info("")
    
    # Análisis por rango de gap (usando gap_percent_display)
    excellent = (df_gaps['gap_percent_display'] < 5).sum()
    good = ((df_gaps['gap_percent_display'] >= 5) & (df_gaps['gap_percent_display'] < 10)).sum()
    acceptable = ((df_gaps['gap_percent_display'] >= 10) & (df_gaps['gap_percent_display'] < 20)).sum()
    poor = (df_gaps['gap_percent_display'] >= 20).sum()
    
    total = len(df_gaps)
    logging.info("Distribución de Calidad de Soluciones:")
    logging.info(f"  Excelente (< 5%):      {excellent:4d} ({excellent/total*100:.1f}%)")
    logging.info(f"  Bueno (5-10%):         {good:4d} ({good/total*100:.1f}%)")
    logging.info(f"  Aceptable (10-20%):    {acceptable:4d} ({acceptable/total*100:.1f}%)")
    logging.info(f"  Pobre (≥ 20%):         {poor:4d} ({poor/total*100:.1f}%)")
    logging.info("")
    
    logging.info("="*70)


def main():
    """Función principal."""
    start_time = datetime.now()
    logging.info("")
    logging.info("#"*70)
    logging.info("# ANÁLISIS DE GAPS DE OPTIMALIDAD")
    logging.info("#"*70)
    logging.info(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("")
    
    # Cargar datos
    df = load_data()
    
    # Calcular gaps
    df_gaps = compute_optimality_gaps(df)
    
    # Generar gráficos
    plot_gap_by_algorithm(df_gaps)
    plot_gap_vs_size(df_gaps)
    plot_gap_vs_k(df_gaps)
    plot_gap_heatmap(df_gaps)
    
    # Generar resumen
    generate_summary_report(df_gaps)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Duración: {duration:.2f} segundos")
    logging.info("#"*70)
    logging.info("")
    logging.info("✅ Análisis completado exitosamente")
    logging.info("")
    logging.info("📊 Archivos generados:")
    logging.info(f"   - {OUTPUT_CSV}")
    logging.info(f"   - {OUTPUT_PNG_1}")
    logging.info(f"   - {OUTPUT_PNG_2}")
    logging.info(f"   - {OUTPUT_PNG_3}")
    logging.info(f"   - {OUTPUT_DIR / 'gap_heatmap.png'}")
    logging.info("")


if __name__ == "__main__":
    main()
