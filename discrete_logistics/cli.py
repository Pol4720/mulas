"""
Command Line Interface for discrete_logistics package.
"""

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="discrete-logistics")
def main():
    """
    Discrete Logistics - Balanced Multi-Bin Packing Solver
    
    Una herramienta para resolver problemas de empaquetado multi-contenedor
    balanceado con restricciones de capacidad.
    """
    pass


@main.command()
def dashboard():
    """Iniciar el dashboard interactivo de Streamlit."""
    try:
        from discrete_logistics.dashboard.app import main as run_dashboard
        import subprocess
        import sys
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "discrete_logistics/dashboard/app.py"
        ])
    except ImportError as e:
        click.echo(f"Error: Dependencias del dashboard no instaladas. {e}")
        click.echo("Instala con: pip install discrete-logistics[dashboard]")


@main.command()
@click.option('--items', '-n', default=10, help='Número de ítems')
@click.option('--bins', '-k', default=3, help='Número de contenedores')
@click.option('--capacity', '-c', default=100.0, help='Capacidad de los contenedores')
@click.option('--algorithm', '-a', default='lpt', 
              type=click.Choice(['ffd', 'bfd', 'wfd', 'lpt', 'sa', 'ga', 'tabu', 'bb', 'dp']),
              help='Algoritmo a usar')
def solve(items, bins, capacity, algorithm):
    """Resolver una instancia aleatoria del problema."""
    click.echo(f"Generando instancia con {items} ítems y {bins} contenedores...")
    
    from discrete_logistics.core.instance_generator import InstanceGenerator
    from discrete_logistics.algorithms.greedy import (
        FirstFitDecreasing, BestFitDecreasing, WorstFitDecreasing, RoundRobinGreedy
    )
    from discrete_logistics.algorithms.metaheuristics import (
        SimulatedAnnealing, GeneticAlgorithm, TabuSearch
    )
    from discrete_logistics.algorithms.branch_and_bound import BranchAndBound
    from discrete_logistics.algorithms.dynamic_programming import DynamicProgramming
    from discrete_logistics.algorithms.approximation import LPTApproximation
    
    # Generar instancia
    generator = InstanceGenerator()
    problem = generator.generate_random(
        n_items=items,
        num_bins=bins,
        capacity=capacity
    )
    
    # Seleccionar algoritmo
    algorithms = {
        'ffd': FirstFitDecreasing(),
        'bfd': BestFitDecreasing(),
        'wfd': WorstFitDecreasing(),
        'lpt': LPTApproximation(),
        'sa': SimulatedAnnealing(),
        'ga': GeneticAlgorithm(),
        'tabu': TabuSearch(),
        'bb': BranchAndBound(),
        'dp': DynamicProgramming(),
    }
    
    algo = algorithms.get(algorithm)
    if algo is None:
        click.echo(f"Algoritmo '{algorithm}' no encontrado.")
        return
    
    click.echo(f"Ejecutando {algo.name}...")
    
    # Resolver
    import time
    start = time.time()
    solution = algo.solve(problem)
    elapsed = time.time() - start
    
    # Mostrar resultados
    click.echo(f"\n{'='*50}")
    click.echo(f"Resultados - {algo.name}")
    click.echo(f"{'='*50}")
    click.echo(f"Diferencia de valores: {solution.value_difference:.2f}")
    click.echo(f"Tiempo de ejecución: {elapsed:.4f}s")
    click.echo(f"Solución factible: {'Sí' if solution.is_feasible else 'No'}")
    
    for i, bin_obj in enumerate(solution.bins):
        total_weight = sum(item.weight for item in bin_obj.items)
        total_value = sum(item.value for item in bin_obj.items)
        click.echo(f"  Contenedor {i+1}: {len(bin_obj.items)} ítems, "
                   f"peso={total_weight:.1f}, valor={total_value:.1f}")


@main.command()
def info():
    """Mostrar información del paquete."""
    click.echo("""
╔══════════════════════════════════════════════════════════════╗
║           Discrete Logistics - Bin Packing Solver            ║
╠══════════════════════════════════════════════════════════════╣
║  Versión: 0.1.0                                              ║
║  Problema: Balanced Multi-Bin Packing                        ║
║  Complejidad: NP-Completo                                    ║
╠══════════════════════════════════════════════════════════════╣
║  Algoritmos disponibles:                                     ║
║  • Greedy: FFD, BFD, WFD, LPT                                ║
║  • Metaheurísticas: SA, GA, Tabu Search                      ║
║  • Exactos: Branch & Bound, Programación Dinámica            ║
╠══════════════════════════════════════════════════════════════╣
║  Uso: discrete-logistics --help                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
