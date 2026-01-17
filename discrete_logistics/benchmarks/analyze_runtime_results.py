"""
Runtime vs instance size benchmark - Analysis Phase.

Loads collected benchmark data from CSV and generates visualizations
and statistical analysis of algorithm performance.

Requires that runtime_vs_size.py has been run first to generate the data.
"""
from pathlib import Path
import sys
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

# Allow running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discrete_logistics.visualizations.plots import BenchmarkPlotter

# Paths
RESULTS_DIR = Path(__file__).resolve().parent / "results"
INPUT_CSV = RESULTS_DIR / "runtime_vs_size.csv"
OUTPUT_HTML_BASE = RESULTS_DIR / "runtime_vs_size"
STATS_CSV = RESULTS_DIR / "runtime_stats.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_results() -> Optional[pd.DataFrame]:
    """Load benchmark results from CSV."""
    logging.info("="*70)
    logging.info("CARGANDO RESULTADOS DE BENCHMARKS")
    logging.info("="*70)
    
    if not INPUT_CSV.exists():
        logging.error(f"❌ Archivo de resultados no encontrado: {INPUT_CSV}")
        logging.error("")
        logging.error("Primero ejecuta la recolección de datos:")
        logging.error("  python discrete_logistics/benchmarks/runtime_vs_size.py")
        return None
    
    try:
        df = pd.read_csv(INPUT_CSV)
        logging.info(f"✓ Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
        logging.info(f"  Archivo: {INPUT_CSV}")
        logging.info("")
        
        # Mostrar información básica
        logging.info("Resumen de datos:")
        logging.info(f"  - Algoritmos únicos: {df['algorithm'].nunique()}")
        logging.info(f"  - Problemas únicos: {df['problem'].nunique()}")
        logging.info(f"  - Rango de n: {df['n_items'].min()} - {df['n_items'].max()}")
        logging.info(f"  - Valores de k: {sorted(df['k'].unique())}")
        
        # Contar éxitos vs errores
        success_count = len(df[df['status'] == 'ok'])
        error_count = len(df[df['status'] == 'error'])
        logging.info(f"  - Ejecuciones exitosas: {success_count}")
        logging.info(f"  - Ejecuciones con error: {error_count}")
        logging.info("")
        
        return df
        
    except Exception as e:
        logging.error(f"❌ Error al cargar el archivo CSV: {e}")
        return None


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistical summary of runtime performance."""
    logging.info("="*70)
    logging.info("CALCULANDO ESTADÍSTICAS")
    logging.info("="*70)
    
    # Filtrar solo ejecuciones exitosas
    df_success = df[df['status'] == 'ok'].copy()
    
    if len(df_success) == 0:
        logging.warning("⚠ No hay ejecuciones exitosas para analizar")
        return pd.DataFrame()
    
    # Agrupar por algoritmo y calcular estadísticas
    stats = df_success.groupby('algorithm')['execution_time'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]).reset_index()
    
    # Ordenar por tiempo medio
    stats = stats.sort_values('mean')
    
    logging.info(f"✓ Estadísticas calculadas para {len(stats)} algoritmos")
    logging.info("")
    logging.info("Top 5 algoritmos más rápidos (tiempo medio):")
    for idx, row in stats.head(5).iterrows():
        logging.info(f"  {row['algorithm']:25s} - {row['mean']:.4f}s (±{row['std']:.4f}s)")
    
    logging.info("")
    logging.info("Top 5 algoritmos más lentos (tiempo medio):")
    for idx, row in stats.tail(5).iterrows():
        logging.info(f"  {row['algorithm']:25s} - {row['mean']:.4f}s (±{row['std']:.4f}s)")
    
    logging.info("")
    
    return stats


def generate_visualizations(df: pd.DataFrame) -> None:
    """Generate interactive visualizations of the results - one graph per k value."""
    logging.info("="*70)
    logging.info("GENERANDO VISUALIZACIONES")
    logging.info("="*70)
    
    # Filtrar solo ejecuciones exitosas para la visualización principal
    df_plot = df[df['status'] == 'ok'].copy()
    
    if len(df_plot) == 0:
        logging.warning("⚠ No hay datos exitosos para visualizar")
        return
    
    # Obtener valores únicos de k y ordenarlos
    k_values = sorted(df_plot['k'].unique())
    
    if len(k_values) == 0:
        logging.warning("⚠ No hay valores de k en los datos")
        return
    
    logging.info(f"Generando {len(k_values)} gráficos (uno por cada valor de k)...")
    logging.info("")
    
    plotter = BenchmarkPlotter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generar un gráfico para cada valor de k
    for k_val in k_values:
        # Filtrar datos para este valor de k
        df_k = df_plot[df_plot['k'] == k_val].copy()
        
        if len(df_k) == 0:
            logging.warning(f"  ⚠ No hay datos para k={k_val}")
            continue
        
        logging.info(f"  Generando gráfico para k={k_val}...")
        
        fig = plotter.plot_scaling_analysis(
            df_k,
            x_var="n_items",
            y_var="execution_time",
            title=f"Runtime vs Instance Size (Escala Logarítmica) - k={k_val}",
        )
        
        # Personalizar el gráfico
        fig.update_yaxes(type="log", title="Tiempo de Ejecución (s, escala log)")
        fig.update_xaxes(title="Número de Items (n)")
        fig.update_layout(
            height=700,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Guardar gráfico con nombre específico para cada k
        output_file = OUTPUT_HTML_BASE.parent / f"runtime_vs_size_k{k_val}.html"
        fig.write_html(output_file)
        logging.info(f"    ✓ Gráfico guardado: {output_file}")
    
    logging.info("")
    logging.info(f"✓ Todos los gráficos generados en: {RESULTS_DIR}")
    logging.info(f"  Abrir los archivos HTML en navegador para visualización interactiva")
    logging.info("")


def save_statistics(stats: pd.DataFrame) -> None:
    """Save statistical summary to CSV."""
    if len(stats) == 0:
        logging.warning("⚠ No hay estadísticas para guardar")
        return
    
    logging.info("Guardando estadísticas...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(STATS_CSV, index=False)
    logging.info(f"  ✓ Estadísticas guardadas: {STATS_CSV}")
    logging.info("")


def analyze_scaling_behavior(df: pd.DataFrame) -> None:
    """Analyze and report scaling behavior of algorithms."""
    logging.info("="*70)
    logging.info("ANÁLISIS DE ESCALABILIDAD")
    logging.info("="*70)
    
    df_success = df[df['status'] == 'ok'].copy()
    
    if len(df_success) == 0:
        logging.warning("⚠ No hay datos para analizar escalabilidad")
        return
    
    # Analizar crecimiento del runtime con n para cada algoritmo
    logging.info("Análisis del crecimiento del runtime con el tamaño de la instancia:")
    logging.info("")
    
    for algo in sorted(df_success['algorithm'].unique()):
        algo_data = df_success[df_success['algorithm'] == algo].copy()
        algo_data = algo_data.sort_values('n_items')
        
        # Comparar tiempo en n_min vs n_max
        n_min = algo_data['n_items'].min()
        n_max = algo_data['n_items'].max()
        
        time_at_min = algo_data[algo_data['n_items'] == n_min]['execution_time'].mean()
        time_at_max = algo_data[algo_data['n_items'] == n_max]['execution_time'].mean()
        
        if time_at_min > 0:
            growth_factor = time_at_max / time_at_min
            logging.info(f"  {algo:25s}")
            logging.info(f"    n={n_min:2d}: {time_at_min:.4f}s → n={n_max:2d}: {time_at_max:.4f}s")
            logging.info(f"    Factor de crecimiento: {growth_factor:.1f}x")
            logging.info("")


def print_summary(df: pd.DataFrame) -> None:
    """Print final summary of the analysis."""
    logging.info("="*70)
    logging.info("RESUMEN DEL ANÁLISIS")
    logging.info("="*70)
    
    total_runs = len(df)
    success_runs = len(df[df['status'] == 'ok'])
    error_runs = len(df[df['status'] == 'error'])
    
    logging.info(f"📊 Total de ejecuciones: {total_runs}")
    logging.info(f"✓  Exitosas: {success_runs} ({100*success_runs/total_runs:.1f}%)")
    logging.info(f"✗  Con errores: {error_runs} ({100*error_runs/total_runs:.1f}%)")
    logging.info("")
    logging.info("📁 Archivos generados:")
    logging.info(f"   - Gráficos interactivos: {RESULTS_DIR}/runtime_vs_size_k*.html")
    logging.info(f"   - Estadísticas: {STATS_CSV}")
    logging.info("")
    logging.info("💡 Para ver el gráfico interactivo, abre el archivo HTML en un navegador")
    logging.info("="*70)


def main() -> None:
    start_time = datetime.now()
    logging.info("")
    logging.info("#"*70)
    logging.info("# ANÁLISIS DE RESULTADOS: Runtime vs Instance Size")
    logging.info("#"*70)
    logging.info(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("")
    
    # Cargar datos
    df = load_results()
    if df is None:
        logging.error("❌ No se pudieron cargar los datos. Análisis cancelado.")
        return
    
    # Vista previa
    logging.info("Vista previa de los datos:")
    logging.info("\n" + df.head(10).to_string())
    logging.info("")
    
    # Calcular estadísticas
    stats = compute_statistics(df)
    if len(stats) > 0:
        save_statistics(stats)
    
    # Analizar escalabilidad
    analyze_scaling_behavior(df)
    
    # Generar visualizaciones
    generate_visualizations(df)
    
    # Resumen final
    print_summary(df)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info("")
    logging.info(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Duración del análisis: {duration:.2f} segundos")
    logging.info("#"*70)


if __name__ == "__main__":
    main()
