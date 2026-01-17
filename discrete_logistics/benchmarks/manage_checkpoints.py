"""
Utilidad para gestionar checkpoints de la recolección de datos runtime_vs_size.

Permite ver el estado y limpiar checkpoints de ejecuciones parciales.
Útil cuando se interrumpe la recolección de datos y se quiere reiniciar o continuar.
"""
import sys
from pathlib import Path
import json
import argparse

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = Path(__file__).resolve().parent / "results" / "checkpoints"
INSTANCES_CHECKPOINT = CHECKPOINT_DIR / "instances.pkl"
RESULTS_CHECKPOINT = CHECKPOINT_DIR / "results_partial.csv"
PROGRESS_FILE = CHECKPOINT_DIR / "progress.json"


def show_status():
    """Muestra el estado actual de los checkpoints."""
    print("=" * 70)
    print("ESTADO DE CHECKPOINTS")
    print("=" * 70)
    
    if not CHECKPOINT_DIR.exists():
        print("❌ No hay checkpoints guardados.")
        return
    
    has_checkpoints = False
    
    # Instancias
    if INSTANCES_CHECKPOINT.exists():
        size_mb = INSTANCES_CHECKPOINT.stat().st_size / (1024 * 1024)
        print(f"✓ Instancias: {INSTANCES_CHECKPOINT}")
        print(f"  Tamaño: {size_mb:.2f} MB")
        has_checkpoints = True
    else:
        print("✗ No hay checkpoint de instancias")
    
    # Resultados parciales
    if RESULTS_CHECKPOINT.exists():
        size_mb = RESULTS_CHECKPOINT.stat().st_size / (1024 * 1024)
        
        # Contar líneas (filas)
        with open(RESULTS_CHECKPOINT, 'r') as f:
            num_lines = sum(1 for _ in f) - 1  # -1 por el header
        
        print(f"✓ Resultados parciales: {RESULTS_CHECKPOINT}")
        print(f"  Tamaño: {size_mb:.2f} MB")
        print(f"  Filas: {num_lines}")
        has_checkpoints = True
    else:
        print("✗ No hay checkpoint de resultados")
    
    # Progreso
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            progress_data = json.load(f)
        
        completed = len(progress_data.get('completed_runs', []))
        timestamp = progress_data.get('timestamp', 'desconocido')
        
        print(f"✓ Progreso: {PROGRESS_FILE}")
        print(f"  Ejecuciones completadas: {completed}")
        print(f"  Última actualización: {timestamp}")
        has_checkpoints = True
    else:
        print("✗ No hay archivo de progreso")
    
    if not has_checkpoints:
        print("\n❌ No hay checkpoints guardados.")
    
    print("=" * 70)


def clean_checkpoints():
    """Limpia todos los checkpoints."""
    print("=" * 70)
    print("LIMPIANDO CHECKPOINTS")
    print("=" * 70)
    
    deleted = False
    
    if INSTANCES_CHECKPOINT.exists():
        INSTANCES_CHECKPOINT.unlink()
        print(f"✓ Eliminado: {INSTANCES_CHECKPOINT.name}")
        deleted = True
    
    if RESULTS_CHECKPOINT.exists():
        RESULTS_CHECKPOINT.unlink()
        print(f"✓ Eliminado: {RESULTS_CHECKPOINT.name}")
        deleted = True
    
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print(f"✓ Eliminado: {PROGRESS_FILE.name}")
        deleted = True
    
    if deleted:
        print("\n✓ Checkpoints limpiados exitosamente.")
    else:
        print("\n⚠ No había checkpoints para limpiar.")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Gestiona checkpoints del análisis runtime_vs_size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python manage_checkpoints.py status    # Ver estado
  python manage_checkpoints.py clean     # Limpiar checkpoints
        """
    )
    
    parser.add_argument(
        'action',
        choices=['status', 'clean'],
        help='Acción a realizar'
    )
    
    args = parser.parse_args()
    
    if args.action == 'status':
        show_status()
    elif args.action == 'clean':
        clean_checkpoints()


if __name__ == "__main__":
    main()
