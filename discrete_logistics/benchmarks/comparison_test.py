"""
Script de Comparación de Solvers de Testing.

Genera instancias aleatorias de diferentes tamaños (pequeñas, medianas, grandes)
y ejecuta los cuatro solvers de testing (deepseek, chapgpt, qwen, gemini) para comparar:
- Tiempos de ejecución
- Calidad de soluciones (valor de desbalance)
- Rendimiento relativo

Uso:
    python comparison_test.py

Estructura de tamaños:
    - PEQUEÑAS: n ∈ [5, 10], k ∈ [3, 4]
    - MEDIANAS: n ∈ [15, 25], k ∈ [3, 4, 5]
    - GRANDES: n ∈ [30, 50], k ∈ [4, 5]
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import logging
import json
import time
from datetime import datetime
from copy import deepcopy
import traceback

import pandas as pd
import numpy as np

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar componentes principales
from discrete_logistics.core.instance_generator import InstanceGenerator
from discrete_logistics.core.problem import Problem

# Importar solvers de testing
sys.path.insert(0, str(PROJECT_ROOT / "discrete_logistics/testing"))
from deepseek import SA_DP_Hybrid as deepseek_solver, Item as DeepseekItem, Solution as DeepseekSolution
from chapgpt import hybrid_packing as chapgpt_solver
from qwen import hybrid_sa_dp as qwen_solver, Item as QwenItem, Bin as QwenBin
from gemini import GeneticCoreSolver as gemini_solver, Item as GeminiItem, Bin as GeminiBin

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directorio de resultados
RESULTS_DIR = Path(__file__).resolve().parent / "comparison_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constantes de configuración
INSTANCE_SIZES = {
    "PEQUEÑAS": {
        "n_ranges": [(5, 30)],
        "k_values": [3, 4,5],
        "num_instances": 30,
    },
    "MEDIANAS": {
        "n_ranges": [(30, 70)],
        "k_values": [ 5, 6,7],
        "num_instances": 20,
    },
    "GRANDES": {
        "n_ranges": [(70, 150)],
        "k_values": [7, 8,9],
        "num_instances": 10,
    },
}

# Timeouts para solvers
TIMEOUTS = {
    "deepseek": 60.0,
    "chapgpt": 60.0,
    "qwen":120.0,
    "gemini": 120.0,
}

# Semilla para reproducibilidad
SEED = 42


class ComparisonRunner:
    """
    Ejecutor de pruebas de comparación entre solvers de testing.
    """

    def __init__(self):
        self.results: List[Dict] = []
        self.generator = InstanceGenerator(seed=SEED)
        self.instance_count = 0

    def generate_instance(self, n: int, k: int) -> Optional[Tuple[List[Dict], List[int]]]:
        """Genera una instancia aleatoria compatible con los solvers de testing."""
        try:
            problem = self.generator.generate_uniform(
                n_items=n,
                num_bins=k,
                weight_range=(1, 100),
                value_range=(1, 100),
                capacity_factor=1.5,
                capacity_variation=0.2,
            )
            
            # Convertir a formato de testing (listas simples)
            items = []
            for item in problem.items:
                items.append({
                    'id': item.id,
                    'weight': int(item.weight),
                    'value': int(item.value)
                })
            
            capacities = [int(c) for c in problem.bin_capacities]
            
            return items, capacities
        except Exception as e:
            logger.error(f"Error generando instancia (n={n}, k={k}): {e}")
            traceback.print_exc()
            return None

    def run_deepseek(self, items: List[Dict], capacities: List[int], timeout: float) -> Tuple[Optional[float], Optional[float], str]:
        """Ejecuta el solver deepseek."""
        try:
            start_time = time.time()
            
            # Convertir a formato de deepseek
            deepseek_items = [DeepseekItem(id=i['id'], weight=i['weight'], value=i['value']) for i in items]
            
            # Ejecutar
            result = deepseek_solver(
                items=deepseek_items,
                k=len(capacities),
                capacities=capacities,
                alpha=0.7,
                verbose=False
            )

            if isinstance(result, DeepseekSolution):
                imbalance = result.imbalance
            elif isinstance(result, tuple) and len(result) >= 2:
                _, imbalance = result[0], result[1]
            else:
                raise ValueError("Formato de retorno inesperado de deepseek_solver")
            
            exec_time = time.time() - start_time
            return imbalance, exec_time, "success"
        except Exception as e:
            logger.warning(f"Error ejecutando deepseek: {e}")
            return None, None, f"error: {type(e).__name__}"

    def run_chapgpt(self, items: List[Dict], capacities: List[int], timeout: float) -> Tuple[Optional[float], Optional[float], str]:
        """Ejecuta el solver chapgpt."""
        try:
            start_time = time.time()
            
            # Convertir a formato de chapgpt (listas simples)
            weights = [i['weight'] for i in items]
            values = [i['value'] for i in items]
            
            # Ejecutar
            result = chapgpt_solver(
                w=weights,
                v=values,
                C=capacities,
                alpha=0.6,
                max_n_dp=12
            )
            
            exec_time = time.time() - start_time
            imbalance = result.get('range', None)
            
            return imbalance, exec_time, "success"
        except Exception as e:
            logger.warning(f"Error ejecutando chapgpt: {e}")
            return None, None, f"error: {type(e).__name__}"

    def run_qwen(self, items: List[Dict], capacities: List[int], timeout: float) -> Tuple[Optional[float], Optional[float], str]:
        """Ejecuta el solver qwen."""
        try:
            start_time = time.time()
            
            # Convertir a formato de qwen
            qwen_items = [QwenItem(id=i['id'], weight=float(i['weight']), value=float(i['value'])) for i in items]
            qwen_bins = [QwenBin(id=j, capacity=float(c)) for j, c in enumerate(capacities)]
            
            # Ejecutar con límite de tiempo
            assignment = qwen_solver(
                items=qwen_items,
                bins=qwen_bins,
                alpha=0.65,
                verbose=False,
                max_time=timeout  # Pasar el timeout al solver
            )
            
            exec_time = time.time() - start_time
            
            # Calcular imbalance
            bin_values = [0.0] * len(capacities)
            for item_id, bin_id in assignment.items():
                for item in qwen_items:
                    if item.id == item_id:
                        bin_values[bin_id] += item.value
                        break
            
            imbalance = max(bin_values) - min(bin_values) if bin_values else None
            
            return imbalance, exec_time, "success"
        except Exception as e:
            logger.warning(f"Error ejecutando qwen: {e}")
            return None, None, f"error: {type(e).__name__}"

    def run_gemini(self, items: List[Dict], capacities: List[int], timeout: float) -> Tuple[Optional[float], Optional[float], str]:
        """Ejecuta el solver gemini."""
        try:
            start_time = time.time()
            
            # Convertir a formato de gemini
            gemini_items = [GeminiItem(id=i['id'], weight=i['weight'], value=i['value']) for i in items]
            gemini_bins = [GeminiBin(id=j, capacity=c) for j, c in enumerate(capacities)]
            
            # Ejecutar solver
            solver = gemini_solver(items=gemini_items, bins=gemini_bins, pop_size=50, generations=100, 
                                   max_time=timeout, convergence_limit=10000)
            best_chromosome, fitness = solver.solve()
            
            exec_time = time.time() - start_time
            imbalance = fitness  # El fitness es el spread/imbalance
            
            return imbalance, exec_time, "success"
        except Exception as e:
            logger.warning(f"Error ejecutando gemini: {e}")
            return None, None, f"error: {type(e).__name__}"

    def run_comparison_suite(self):
        """Ejecuta la suite completa de comparaciones."""
        logger.info("=" * 80)
        logger.info("INICIANDO COMPARACIÓN DE SOLVERS")
        logger.info("=" * 80)

        # Diccionario de solvers
        solvers = {
            # "deepseek": self.run_deepseek,
            # "chapgpt": self.run_chapgpt,
            "qwen": self.run_qwen,
            "gemini": self.run_gemini,
        }

        # Iterar sobre categorías de tamaño
        for size_category, config in INSTANCE_SIZES.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"CATEGORÍA: {size_category}")
            logger.info(f"{'='*80}")

            # Iterar sobre rangos de n
            for n_min, n_max in config["n_ranges"]:
                for k in config["k_values"]:
                    for instance_num in range(config["num_instances"]):
                        # Generar n aleatorio en el rango
                        n = np.random.randint(n_min, n_max + 1)

                        logger.info(
                            f"\n[{size_category}] Instancia {instance_num + 1}/{config['num_instances']}: "
                            f"n={n}, k={k}"
                        )

                        # Generar instancia
                        instance = self.generate_instance(n, k)
                        if instance is None:
                            continue

                        items, capacities = instance
                        self.instance_count += 1

                        # Ejecutar cada solver
                        for solver_name, solver_func in solvers.items():
                            timeout = TIMEOUTS.get(solver_name, 60.0)

                            logger.info(f"  Ejecutando {solver_name}...")

                            imbalance, exec_time, status = solver_func(
                                items, capacities, timeout
                            )

                            # Registrar resultado
                            result = {
                                "timestamp": datetime.now().isoformat(),
                                "size_category": size_category,
                                "n_items": n,
                                "k_bins": k,
                                "instance_num": instance_num + 1,
                                "solver": solver_name,
                                "imbalance": imbalance,
                                "execution_time": exec_time,
                                "status": status,
                            }
                            self.results.append(result)

                            if status == "success":
                                logger.info(
                                    f"    ✓ imbalance={imbalance:.2f}, time={exec_time:.3f}s"
                                )
                            else:
                                logger.info(f"    ✗ {status}")

        logger.info(f"\n{'='*80}")
        logger.info(f"COMPARACIÓN COMPLETADA: {self.instance_count} instancias procesadas")
        logger.info(f"{'='*80}")

    def save_results(self) -> Path:
        """Guarda los resultados en archivos CSV."""
        df = pd.DataFrame(self.results)

        # Guardar CSV completo
        csv_path = RESULTS_DIR / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"\n✓ Resultados guardados: {csv_path}")

        return csv_path

    def print_summary(self):
        """Imprime un resumen comparativo de los resultados."""
        if len(self.results) == 0:
            logger.warning("No hay resultados para mostrar")
            return

        df = pd.DataFrame(self.results)

        logger.info("\n" + "=" * 80)
        logger.info("RESUMEN COMPARATIVO")
        logger.info("=" * 80)

        # 1. Resumen por solver
        logger.info("\n1. RENDIMIENTO POR SOLVER")
        logger.info("-" * 80)

        for solver in df["solver"].unique():
            solver_data = df[df["solver"] == solver]
            successful = solver_data[solver_data["status"] == "success"]

            logger.info(f"\n{solver}:")
            logger.info(f"  Total ejecuciones: {len(solver_data)}")
            logger.info(f"  Exitosas: {len(successful)}")
            logger.info(f"  Errores: {len(solver_data[solver_data['status'] != 'success'])}")

            if len(successful) > 0:
                logger.info(
                    f"  Imbalance promedio: {successful['imbalance'].mean():.2f} "
                    f"(min: {successful['imbalance'].min():.2f}, "
                    f"max: {successful['imbalance'].max():.2f})"
                )
                logger.info(
                    f"  Tiempo promedio: {successful['execution_time'].mean():.3f}s "
                    f"(min: {successful['execution_time'].min():.3f}s, "
                    f"max: {successful['execution_time'].max():.3f}s)"
                )

        # 2. Resumen por categoría de tamaño
        logger.info(f"\n{'=' * 80}")
        logger.info("2. RENDIMIENTO POR CATEGORÍA DE TAMAÑO")
        logger.info("-" * 80)

        for size_cat in ["PEQUEÑAS", "MEDIANAS", "GRANDES"]:
            size_data = df[df["size_category"] == size_cat]
            if len(size_data) == 0:
                continue

            logger.info(f"\n{size_cat}:")
            logger.info(f"  Instancias: {size_data[['n_items', 'k_bins']].drop_duplicates().shape[0]}")

            for solver in df["solver"].unique():
                solver_size_data = size_data[
                    (size_data["solver"] == solver) & (size_data["status"] == "success")
                ]

                if len(solver_size_data) > 0:
                    avg_imbalance = solver_size_data["imbalance"].mean()
                    avg_time = solver_size_data["execution_time"].mean()
                    logger.info(
                        f"    {solver}: imbalance={avg_imbalance:.2f}, "
                        f"tiempo={avg_time:.3f}s ({len(solver_size_data)} exitosas)"
                    )

        # 3. Comparación de calidad en instancias donde todos los solvers tuvieron éxito
        logger.info(f"\n{'=' * 80}")
        logger.info("3. COMPARACIÓN DE CALIDAD (instancias con todos los solvers exitosos)")
        logger.info("-" * 80)

        # Agrupar por instancia
        for size_cat in ["PEQUEÑAS", "MEDIANAS", "GRANDES"]:
            size_data = df[df["size_category"] == size_cat]
            if len(size_data) == 0:
                continue

            instances = size_data.groupby(["n_items", "k_bins", "instance_num"])

            complete_instances = 0
            for (n, k, inst_num), group in instances:
                successful_solvers = group[group["status"] == "success"]
                if len(successful_solvers) == 4:  # 4 solvers
                    complete_instances += 1
                    best_imbalance = successful_solvers["imbalance"].min()
                    worst_imbalance = successful_solvers["imbalance"].max()
                    avg_imbalance = successful_solvers["imbalance"].mean()

                    logger.info(
                        f"  n={n}, k={k}: mejor={best_imbalance:.2f}, "
                        f"peor={worst_imbalance:.2f}, promedio={avg_imbalance:.2f}"
                    )

                    for _, row in successful_solvers.iterrows():
                        gap = ((row["imbalance"] - best_imbalance) / best_imbalance * 100) if best_imbalance > 0 else 0
                        logger.info(
                            f"      {row['solver']}: imbalance={row['imbalance']:.2f} "
                            f"(+{gap:.1f}%), tiempo={row['execution_time']:.3f}s"
                        )

            if complete_instances > 0:
                logger.info(f"\n  Total instancias completas ({size_cat}): {complete_instances}")

        logger.info(f"\n{'=' * 80}")


def main():
    """Función principal."""
    try:
        runner = ComparisonRunner()
        runner.run_comparison_suite()
        runner.save_results()
        runner.print_summary()

        logger.info("\n✓ Comparación completada exitosamente")

    except Exception as e:
        logger.error(f"Error fatal: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
