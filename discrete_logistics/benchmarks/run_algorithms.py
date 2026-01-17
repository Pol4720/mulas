"""
Runtime vs instance size benchmark - Data Collection Phase.

Generates random instances with n in [5, 50] and k in {3, 4, 5}, runs ALL
available algorithms with a timeout, and saves raw results to CSV.

Use analyze_runtime_results.py to analyze and visualize the collected data.

Includes detailed logging to track progress through the benchmark execution.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import sys
import logging
from datetime import datetime
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time
from copy import deepcopy

import pandas as pd

# Allow running this file directly (`python discrete_logistics/benchmarks/runtime_vs_size.py`)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discrete_logistics.core.instance_generator import InstanceGenerator
from discrete_logistics.core.problem import Problem
from discrete_logistics.benchmarks.runner import BenchmarkRunner, BenchmarkConfig, BenchmarkResult
from discrete_logistics.algorithms.brute_force import BruteForce
from discrete_logistics.algorithms.approximation import (
    LPTApproximation,
    MultiWayPartition,
)
from discrete_logistics.algorithms.greedy import (
    FirstFitDecreasing,
    BestFitDecreasing,
    WorstFitDecreasing,
    RoundRobinGreedy,
    LargestDifferenceFirst
)
from discrete_logistics.algorithms.dynamic_programming import DynamicProgramming
from discrete_logistics.algorithms.branch_and_bound import BranchAndBound
from discrete_logistics.algorithms.metaheuristics import (
    SimulatedAnnealing,
    GeneticAlgorithm,
    TabuSearch
)
from discrete_logistics.benchmarks.runner import BenchmarkResult


N_VALUES: List[int] = list(range(5, 51, 5))
K_VALUES: List[int] = [3, 4, 5]
TIME_LIMIT = 60.0  # seconds (1 minute per instance)
SEED = 123
OUT_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
INSTANCES_CHECKPOINT = CHECKPOINT_DIR / "instances.pkl"
RESULTS_CHECKPOINT = CHECKPOINT_DIR / "results_partial.csv"
PROGRESS_FILE = CHECKPOINT_DIR / "progress.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def save_checkpoint_instances(problems: Dict[str, Problem]) -> None:
    """Guarda las instancias generadas en un checkpoint."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(INSTANCES_CHECKPOINT, 'wb') as f:
        pickle.dump(problems, f)
    logging.info(f"  ✓ Checkpoint de instancias guardado: {INSTANCES_CHECKPOINT}")


def load_checkpoint_instances() -> Optional[Dict[str, Problem]]:
    """Carga las instancias desde un checkpoint si existe."""
    if INSTANCES_CHECKPOINT.exists():
        try:
            with open(INSTANCES_CHECKPOINT, 'rb') as f:
                problems = pickle.load(f)
            logging.info(f"  ✓ Checkpoint de instancias cargado: {len(problems)} instancias")
            return problems
        except Exception as e:
            logging.warning(f"  ⚠ Error al cargar checkpoint de instancias: {e}")
            return None
    return None


def save_progress(completed_runs: List[str]) -> None:
    """Guarda el progreso de ejecuciones completadas."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    progress_data = {
        'completed_runs': completed_runs,
        'timestamp': datetime.now().isoformat(),
        'total_expected': len(N_VALUES) * len(K_VALUES)
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f, indent=2)


def load_progress() -> List[str]:
    """Carga el progreso de ejecuciones completadas."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress_data = json.load(f)
            completed = progress_data.get('completed_runs', [])
            logging.info(f"  ✓ Progreso cargado: {len(completed)} ejecuciones completadas")
            return completed
        except Exception as e:
            logging.warning(f"  ⚠ Error al cargar progreso: {e}")
            return []
    return []


def save_partial_results(df: pd.DataFrame) -> None:
    """Guarda resultados parciales."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CHECKPOINT, index=False)
    logging.debug(f"  → Checkpoint de resultados guardado: {len(df)} filas")


def load_partial_results() -> Optional[pd.DataFrame]:
    """Carga resultados parciales si existen."""
    if RESULTS_CHECKPOINT.exists():
        try:
            df = pd.read_csv(RESULTS_CHECKPOINT)
            logging.info(f"  ✓ Resultados parciales cargados: {len(df)} filas")
            return df
        except Exception as e:
            logging.warning(f"  ⚠ Error al cargar resultados parciales: {e}")
            return None
    return None


def build_instances(n_values: Iterable[int], k_values: Iterable[int], force_regenerate: bool = False) -> Dict[str, Problem]:
    """Generate uniform random instances for all (n, k) pairs."""
    logging.info("="*70)
    logging.info("GENERANDO INSTANCIAS DE PRUEBA")
    logging.info("="*70)
    
    # Intentar cargar desde checkpoint
    if not force_regenerate:
        problems = load_checkpoint_instances()
        if problems is not None:
            logging.info(f"✓ Usando instancias del checkpoint ({len(problems)} instancias)")
            logging.info("")
            return problems
    
    logging.info("Generando nuevas instancias...")
    gen = InstanceGenerator(seed=SEED)
    problems: Dict[str, Problem] = {}
    
    total_instances = len(list(k_values)) * len(list(n_values))
    current = 0

    for k in k_values:
        for n in n_values:
            current += 1
            name = f"n{n}_k{k}"
            logging.info(f"[{current}/{total_instances}] Generando instancia: {name} (n={n} items, k={k} bins)")
            
            problems[name] = gen.generate_uniform(
                n_items=n,
                num_bins=k,
                capacity_factor=1.5,
                name=name,
            )
    
    logging.info(f"✓ Se generaron {len(problems)} instancias exitosamente")
    
    # Guardar checkpoint
    save_checkpoint_instances(problems)
    logging.info("")

    return problems


def run_benchmark(problems: Dict[str, Problem], resume: bool = True) -> pd.DataFrame:
    """Execute benchmarks and return a tidy DataFrame."""
    logging.info("="*70)
    logging.info("CONFIGURANDO ALGORITMOS PARA BENCHMARK")
    logging.info("="*70)
    
    # Cargar resultados parciales si existen
    existing_results = []
    completed_runs = set()
    
    if resume:
        partial_df = load_partial_results()
        if partial_df is not None and len(partial_df) > 0:
            existing_results = partial_df.to_dict('records')
            # Identificar runs completados (algoritmo + problema)
            for _, row in partial_df.iterrows():
                completed_runs.add(f"{row['algorithm']}:{row['problem']}")
            logging.info(f"  ✓ Reanudando desde checkpoint: {len(completed_runs)} ejecuciones ya completadas")
    
    # Función auxiliar para crear una instancia fresca de cada algoritmo
    def create_algorithm(algo_name: str):
        """Crea una instancia fresca del algoritmo especificado."""
        if algo_name == "BruteForce":
            return BruteForce(time_limit=TIME_LIMIT, max_items=14, track_steps=False, verbose=False)
        elif algo_name == "DynamicProgramming":
            return DynamicProgramming(time_limit=TIME_LIMIT, track_steps=False, verbose=False)
        elif algo_name == "BranchAndBound":
            return BranchAndBound(time_limit=TIME_LIMIT, track_steps=False, verbose=False)
        elif algo_name == "FirstFitDecreasing":
            return FirstFitDecreasing(track_steps=False, verbose=False)
        elif algo_name == "BestFitDecreasing":
            return BestFitDecreasing(track_steps=False, verbose=False)
        elif algo_name == "WorstFitDecreasing":
            return WorstFitDecreasing(track_steps=False, verbose=False)
        elif algo_name == "RoundRobinGreedy":
            return RoundRobinGreedy(track_steps=False, verbose=False)
        elif algo_name == "LargestDifferenceFirst":
            return LargestDifferenceFirst(track_steps=False, verbose=False)
        elif algo_name == "LPT":
            return LPTApproximation(track_steps=False, verbose=False)
        elif algo_name == "KK":
            return MultiWayPartition(track_steps=False, verbose=False)
        elif algo_name == "SimulatedAnnealing":
            return SimulatedAnnealing(track_steps=False, verbose=False, max_iterations=10000, initial_temp=100.0, cooling_rate=0.995)
        elif algo_name == "GeneticAlgorithm":
            return GeneticAlgorithm(track_steps=False, verbose=False, population_size=50, generations=100)
        elif algo_name == "TabuSearch":
            return TabuSearch(track_steps=False, verbose=False, max_iterations=1000, tabu_tenure=10)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Lista de nombres de algoritmos a ejecutar
    algorithm_names = [
        "LargestDifferenceFirst",
        "BruteForce",
        "DynamicProgramming", 
        "BranchAndBound",
        "FirstFitDecreasing",
        "BestFitDecreasing",
        "WorstFitDecreasing",
        "RoundRobinGreedy",
        "LPT",
        "KK",
        "SimulatedAnnealing",
        "GeneticAlgorithm",
        "TabuSearch"
    ]
    
    logging.info(f"✓ Total de algoritmos: {len(algorithm_names)}")
    logging.info(f"✓ Total de instancias: {len(problems)}")
    logging.info(f"✓ Total de ejecuciones: {len(algorithm_names) * len(problems)}")
    logging.info("")
    
    logging.info("="*70)
    logging.info("INICIANDO EJECUCIÓN DE BENCHMARKS")
    logging.info("="*70)
    
    total_runs = len(algorithm_names) * len(problems)
    if completed_runs:
        logging.info(f"Total ejecuciones: {total_runs}")
        logging.info(f"Ya completadas: {len(completed_runs)}")
        logging.info(f"Pendientes: {total_runs - len(completed_runs)}")
        logging.info("")
    
    # Ejecutar solo las pendientes
    records = existing_results.copy()  # Mantener resultados previos
    success_count = len([r for r in existing_results if r.get('status') == 'ok'])
    error_count = len([r for r in existing_results if r.get('status') == 'error'])
    timeout_count = len([r for r in existing_results 
                        if r.get('error') and isinstance(r.get('error'), str) and 
                        ('requires n ≤' in r.get('error') or 'Timeout exceeded' in r.get('error'))])
    
    current_run = len(completed_runs)
    checkpoint_interval = 10  # Guardar cada 10 ejecuciones
    
    for algo_name in algorithm_names:
        for problem_name, problem in problems.items():
            run_id = f"{algo_name}:{problem_name}"
            
            # Skip si ya está completado
            if run_id in completed_runs:
                continue
            
            current_run += 1
            logging.info(f"[{current_run}/{total_runs}] Ejecutando {algo_name} en {problem_name}...")
            
            # Crear instancia fresca del algoritmo
            algorithm = create_algorithm(algo_name)
            
            # Ejecutar con aislamiento por copia y timeout real
            start_time = time.time()
            try:
                problem_copy = deepcopy(problem)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(algorithm.solve, problem_copy)
                    try:
                        solution = future.result(timeout=TIME_LIMIT)
                    except FutureTimeoutError:
                        try:
                            future.cancel()
                        finally:
                            raise
                elapsed = time.time() - start_time

                # Crear resultado exitoso
                result = BenchmarkResult(
                    algorithm_name=algo_name,
                    problem_name=problem_name,
                    objective=solution.value_difference,
                    execution_time=elapsed,
                    feasible=solution.is_valid,
                    error=None if solution.is_valid else "Infeasible solution"
                )
                logging.info(f"  ✓ Completado: {elapsed:.4f}s, objetivo={solution.value_difference:.2f}")

            except FutureTimeoutError:
                elapsed = TIME_LIMIT
                logging.warning("  ⏱ Timeout excedido; cancelando ejecución")
                result = BenchmarkResult(
                    algorithm_name=algo_name,
                    problem_name=problem_name,
                    objective=float('inf'),
                    execution_time=elapsed,
                    feasible=False,
                    error="Timeout exceeded"
                )

            except KeyboardInterrupt:
                # Si se interrumpe, guardar y terminar
                logging.warning("  ⏹ Interrupción recibida")
                break

            except Exception as e:
                elapsed = time.time() - start_time
                logging.error(f"  → Error inesperado: {str(e)[:80]}...")
                result = BenchmarkResult(
                    algorithm_name=algo_name,
                    problem_name=problem_name,
                    objective=float('inf'),
                    execution_time=elapsed,
                    feasible=False,
                    error=f"Exception: {str(e)[:100]}"
                )
            
            # Procesar resultado
            exec_time = result.execution_time
            status = "ok"
            
            if result.error:
                status = "error"
                error_count += 1
                if "requires n ≤" in result.error or "Timeout exceeded" in result.error or "timed out" in result.error.lower():
                    exec_time = TIME_LIMIT
                    timeout_count += 1
                logging.warning(f"  → Error: {result.error[:80]}...")
            else:
                success_count += 1
                logging.info(f"  → OK (tiempo={exec_time:.4f}s, obj={result.objective:.2f})")
            
            record = {
                "algorithm": result.algorithm_name,
                "problem": result.problem_name,
                "n_items": problem.n_items,
                "k": problem.num_bins,
                "execution_time": exec_time,
                "feasible": result.feasible,
                "objective": result.objective,
                "error": result.error,
                "status": status,
            }
            
            records.append(record)
            completed_runs.add(run_id)
            
            # Guardar checkpoint periódicamente
            if current_run % checkpoint_interval == 0:
                df_temp = pd.DataFrame.from_records(records)
                save_partial_results(df_temp)
                save_progress(sorted(list(completed_runs)))
                logging.info(f"  💾 Checkpoint guardado ({current_run}/{total_runs} completadas)")
    
    logging.info("")
    logging.info("="*70)
    logging.info("PROCESANDO RESULTADOS FINALES")
    logging.info("="*70)
    logging.info(f"✓ Ejecuciones exitosas: {success_count}")
    logging.info(f"✗ Errores: {error_count} (timeouts: {timeout_count})")
    logging.info(f"Total: {len(records)}")
    
    # Guardar checkpoint final
    df_final = pd.DataFrame.from_records(records)
    save_partial_results(df_final)
    save_progress(sorted(list(completed_runs)))

    return df_final


def clean_checkpoints() -> None:
    """Limpia los archivos de checkpoint después de completar exitosamente."""
    try:
        if INSTANCES_CHECKPOINT.exists():
            INSTANCES_CHECKPOINT.unlink()
            logging.info("  ✓ Checkpoint de instancias eliminado")
        if RESULTS_CHECKPOINT.exists():
            RESULTS_CHECKPOINT.unlink()
            logging.info("  ✓ Checkpoint de resultados eliminado")
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
            logging.info("  ✓ Archivo de progreso eliminado")
    except Exception as e:
        logging.warning(f"  ⚠ Error al limpiar checkpoints: {e}")


def save_outputs(df: pd.DataFrame) -> None:
    """Persist CSV with benchmark results."""
    logging.info("")
    logging.info("="*70)
    logging.info("GUARDANDO RESULTADOS FINALES")
    logging.info("="*70)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "runtime_vs_size.csv"
    logging.info(f"Guardando CSV: {csv_path}")
    df.to_csv(csv_path, index=False)
    logging.info(f"  ✓ CSV guardado ({len(df)} filas, {len(df.columns)} columnas)")
    
    logging.info("")
    logging.info("="*70)
    logging.info("RECOLECCIÓN DE DATOS COMPLETADA")
    logging.info("="*70)
    logging.info(f"📊 Resultados guardados en: {csv_path}")
    logging.info(f"")
    logging.info(f"💡 Para analizar los resultados:")
    logging.info(f"   python discrete_logistics/benchmarks/analyze_runtime_results.py")
    logging.info("="*70)


def main() -> None:
    start_time = datetime.now()
    logging.info("")
    logging.info("#"*70)
    logging.info("# RECOLECCIÓN DE DATOS: Runtime vs Instance Size")
    logging.info("#"*70)
    logging.info(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Configuración:")
    logging.info(f"  - Rango de n: {N_VALUES[0]} a {N_VALUES[-1]} (paso {N_VALUES[1]-N_VALUES[0]})")
    logging.info(f"  - Valores de k: {K_VALUES}")
    logging.info(f"  - Timeout por instancia: {TIME_LIMIT} segundos (1 minuto)")
    logging.info(f"  - Semilla: {SEED}")
    logging.info(f"  - Track steps: Desactivado (para mejor rendimiento)")
    logging.info("")
    
    problems = build_instances(N_VALUES, K_VALUES)
    df = run_benchmark(problems)
    
    logging.info("")
    logging.info("Vista previa de resultados:")
    logging.info("\n" + df.head(10).to_string())
    
    save_outputs(df)
    
    # Limpiar checkpoints después de completar exitosamente
    logging.info("")
    logging.info("Limpiando archivos temporales...")
    clean_checkpoints()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info("")
    logging.info(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Duración total: {duration:.2f} segundos ({duration/60:.2f} minutos)")
    logging.info("#"*70)
    logging.info("")
    logging.info("✅ Recolección de datos completada exitosamente")
    logging.info("")
    logging.info("🔍 Siguiente paso - Analizar los resultados:")
    logging.info("   python discrete_logistics/benchmarks/analyze_runtime_results.py")


if __name__ == "__main__":
    main()
