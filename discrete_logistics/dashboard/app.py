"""
Main Streamlit Application
=========================

This is the main entry point for the Balanced Multi-Bin Packing
interactive dashboard.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import time
from typing import Dict, Any, Optional

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Empaquetado Multi-Contenedor Balanceado",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import components
import sys
from pathlib import Path

# Add package root to path for absolute imports
_dashboard_dir = Path(__file__).parent  # dashboard folder
_pkg_root = _dashboard_dir.parent       # discrete_logistics folder
_workspace_root = _pkg_root.parent      # mulas folder

# Insert in reverse order of priority
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))
if str(_pkg_root.parent) not in sys.path:
    sys.path.insert(0, str(_pkg_root.parent))

from discrete_logistics.dashboard.components import (
    ThemeManager,
    ProblemConfigurator,
    AlgorithmSelector,
    ResultsDisplay,
    VisualizationPanel,
    ExportManager
)

from discrete_logistics.core.problem import Problem, Solution, Item, Bin
from discrete_logistics.core.instance_generator import InstanceGenerator
from discrete_logistics.algorithms import AlgorithmRegistry
from discrete_logistics.algorithms.greedy import FirstFitDecreasing, BestFitDecreasing, WorstFitDecreasing, RoundRobinGreedy
from discrete_logistics.algorithms.metaheuristics import SimulatedAnnealing, GeneticAlgorithm, TabuSearch
from discrete_logistics.algorithms.branch_and_bound import BranchAndBound
from discrete_logistics.algorithms.dynamic_programming import DynamicProgramming


def init_session_state():
    """Initialize session state variables."""
    if 'current_problem' not in st.session_state:
        st.session_state['current_problem'] = None
    if 'results' not in st.session_state:
        st.session_state['results'] = {}
    if 'convergence_history' not in st.session_state:
        st.session_state['convergence_history'] = {}
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'dark'


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=BinPacking", use_container_width=True)
        st.markdown("# 📦 Empaquetado Multi-Contenedor")
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navegación")
        page = st.radio(
            "Seleccionar Página",
            options=['🏠 Inicio', '🔬 Solucionador', '📊 Análisis', '📚 Teoría', '⚙️ Configuración'],
            label_visibility='collapsed'
        )
        
        st.markdown("---")
        
        # Quick settings
        st.markdown("### Ajustes Rápidos")
        theme = st.selectbox(
            "Tema",
            options=['Oscuro', 'Claro'],
            index=0 if st.session_state['theme'] == 'dark' else 1
        )
        st.session_state['theme'] = 'dark' if theme == 'Oscuro' else 'light'
        
        # Apply theme
        ThemeManager.apply_theme(st.session_state['theme'])
        
        st.markdown("---")
        
        # Info section
        with st.expander("ℹ️ Acerca de"):
            st.markdown("""
            **Empaquetado Multi-Contenedor Balanceado**
            
            Una herramienta interactiva para resolver el
            problema NP-difícil de empaquetado en contenedores
            con restricciones de balance.
            
            Características:
            - Múltiples algoritmos
            - Visualización en tiempo real
            - Análisis de benchmarks
            
            *Proyecto DAA - 2024*
            """)
        
        return page


def render_home_page():
    """Render the home page."""
    st.markdown("""
    # 🏠 Bienvenido al Solucionador de Empaquetado Multi-Contenedor Balanceado
    
    Este dashboard interactivo te permite explorar y resolver el problema de
    **Empaquetado Multi-Contenedor Balanceado con Restricciones de Capacidad**.
    
    ## 📋 Descripción del Problema
    
    Dado:
    - Un conjunto de **n ítems**, cada uno con peso y valor
    - **k contenedores** con capacidades individuales C_j
    
    Objetivo:
    - Minimizar la **diferencia máxima** de valores totales entre contenedores
    - Respetando las **restricciones de capacidad**
    
    ## 🎯 Características
    
    | Característica | Descripción |
    |----------------|-------------|
    | 🔬 Múltiples Algoritmos | Voraz, Metaheurísticas, Métodos Exactos |
    | 📊 Visualizaciones | Gráficos interactivos y animaciones |
    | 📈 Benchmarking | Comparar rendimiento de algoritmos |
    | 📚 Teoría | Formalización matemática y demostraciones |
    
    ## 🚀 Comenzar
    
    1. Navega a la página **Solucionador**
    2. Configura tu instancia del problema
    3. Selecciona algoritmos a ejecutar
    4. ¡Analiza los resultados!
    
    ---
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Algoritmos", "9+", help="Algoritmos disponibles")
    with col2:
        st.metric("Complejidad", "NP-Difícil", help="Clase de complejidad del problema")
    with col3:
        st.metric("Máx Ítems", "100", help="Cantidad de ítems soportada")
    with col4:
        st.metric("Visualizaciones", "5+", help="Tipos de gráficos disponibles")


def render_solver_page(problem_config: ProblemConfigurator, 
                       algo_selector: AlgorithmSelector,
                       results_display: ResultsDisplay,
                       viz_panel: VisualizationPanel):
    """Render the main solver page."""
    st.markdown("# 🔬 Solucionador del Problema")
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Problem configuration
        problem = problem_config.render()
        
        if problem:
            problem_config.render_problem_summary(problem)
    
    with col2:
        # Algorithm selection
        algorithm_configs = algo_selector.render()
        
        # Run button
        if st.button("▶️ Ejecutar Algoritmos", type="primary", use_container_width=True):
            if problem is None:
                st.error("¡Por favor genera una instancia del problema primero!")
            elif not algorithm_configs:
                st.error("¡Por favor selecciona al menos un algoritmo!")
            else:
                run_algorithms(problem, algorithm_configs)
    
    # Results section
    st.markdown("---")
    
    if st.session_state.get('results'):
        results_display.render_results(st.session_state['results'])
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.get('convergence_history'):
                viz_panel.render_convergence_plot(st.session_state['convergence_history'])
        
        with col2:
            viz_panel.render_performance_radar(st.session_state['results'])
        
        # Export options
        ExportManager.render_export_buttons(st.session_state['results'])


def run_algorithms(problem: Problem, algorithm_configs: list):
    """Execute selected algorithms on the problem."""
    results = {}
    convergence_history = {}
    
    progress_bar = st.progress(0, text="Ejecutando algoritmos...")
    
    for idx, (algo_name, params) in enumerate(algorithm_configs):
        progress_bar.progress(
            (idx + 1) / len(algorithm_configs),
            text=f"Ejecutando {algo_name}..."
        )
        
        try:
            # Create algorithm instance
            algorithm = create_algorithm_instance(algo_name, params)
            
            if algorithm is None:
                st.warning(f"Algoritmo {algo_name} no disponible")
                continue
            
            # Run algorithm
            start_time = time.time()
            solution = algorithm.solve(problem)
            end_time = time.time()
            
            # Calculate metrics
            objective = calculate_objective(solution)
            balance_score = calculate_balance_score(solution)
            feasible = check_feasibility(solution, problem)
            
            results[algo_name] = {
                'solution': solution,
                'objective': objective,
                'balance_score': balance_score,
                'time': end_time - start_time,
                'feasible': feasible,
                'stability': 0.9  # Placeholder
            }
            
            # Store convergence history if available
            if hasattr(algorithm, 'history'):
                convergence_history[algo_name] = algorithm.history
                
        except Exception as e:
            st.error(f"Error ejecutando {algo_name}: {str(e)}")
            results[algo_name] = {
                'objective': float('inf'),
                'time': 0,
                'feasible': False,
                'error': str(e)
            }
    
    progress_bar.progress(1.0, text="¡Completado!")
    
    st.session_state['results'] = results
    st.session_state['convergence_history'] = convergence_history
    
    st.success(f"✅ Se completaron {len(results)} algoritmo(s)")


def create_algorithm_instance(algo_name: str, params: Dict[str, Any]):
    """Create an algorithm instance by name."""
    algorithm_map = {
        'FirstFitDecreasing': FirstFitDecreasing,
        'BestFitDecreasing': BestFitDecreasing,
        'WorstFitDecreasing': WorstFitDecreasing,
        'RoundRobinGreedy': RoundRobinGreedy,
        'SimulatedAnnealing': lambda: SimulatedAnnealing(**params),
        'GeneticAlgorithm': lambda: GeneticAlgorithm(**params),
        'TabuSearch': lambda: TabuSearch(**params),
        'BranchAndBound': lambda: BranchAndBound(**params) if params else BranchAndBound(),
        'DynamicProgramming': DynamicProgramming,
    }
    
    if algo_name in algorithm_map:
        creator = algorithm_map[algo_name]
        if callable(creator) and not isinstance(creator, type):
            return creator()
        return creator()
    
    return None


def calculate_objective(solution: Solution) -> float:
    """Calculate the objective value (max value difference)."""
    if not solution or not solution.bins:
        return float('inf')
    
    bin_values = [sum(item.value for item in bin_obj.items) for bin_obj in solution.bins]
    
    if not bin_values:
        return 0.0
    
    return max(bin_values) - min(bin_values)


def calculate_balance_score(solution: Solution) -> float:
    """Calculate balance score (0 = worst, 1 = perfect)."""
    if not solution or not solution.bins:
        return 0.0
    
    bin_values = [sum(item.value for item in bin_obj.items) for bin_obj in solution.bins]
    
    if not bin_values or max(bin_values) == 0:
        return 1.0
    
    mean_value = np.mean(bin_values)
    std_value = np.std(bin_values)
    
    # Coefficient of variation inverted
    if mean_value == 0:
        return 1.0
    
    cv = std_value / mean_value
    return max(0, 1 - cv)


def check_feasibility(solution: Solution, problem: Problem) -> bool:
    """Check if solution respects capacity constraints."""
    if not solution or not solution.bins:
        return False
    
    for i, bin_obj in enumerate(solution.bins):
        total_weight = sum(item.weight for item in bin_obj.items)
        capacity = problem.bin_capacities[i] if i < len(problem.bin_capacities) else problem.bin_capacities[0]
        if total_weight > capacity:
            return False
    
    return True


def render_analysis_page(viz_panel: VisualizationPanel):
    """Render the analysis page."""
    st.markdown("# 📊 Análisis de Resultados")
    
    if not st.session_state.get('results'):
        st.info("No hay resultados para analizar. ¡Ejecuta algunos algoritmos primero!")
        return
    
    results = st.session_state['results']
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📈 Rendimiento", "🔄 Convergencia", "📊 Comparación"])
    
    with tab1:
        st.markdown("### Métricas de Rendimiento de Algoritmos")
        
        # Create performance dataframe
        import pandas as pd
        
        perf_data = []
        for algo, result in results.items():
            perf_data.append({
                'Algoritmo': algo,
                'Objetivo': result.get('objective', '-'),
                'Tiempo (s)': f"{result.get('time', 0):.4f}",
                'Balance': f"{result.get('balance_score', 0):.2%}",
                'Factible': '✅' if result.get('feasible') else '❌'
            })
        
        df = pd.DataFrame(perf_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Performance chart
        viz_panel.render_performance_radar(results)
    
    with tab2:
        st.markdown("### Análisis de Convergencia")
        
        if st.session_state.get('convergence_history'):
            viz_panel.render_convergence_plot(st.session_state['convergence_history'])
        else:
            st.info("No hay datos de convergencia disponibles. Ejecuta algoritmos metaheurísticos para ver la convergencia.")
    
    with tab3:
        st.markdown("### Comparación de Algoritmos")
        
        # Create comparison visualizations
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        algos = list(results.keys())
        objectives = [results[a].get('objective', 0) for a in algos]
        times = [results[a].get('time', 0) for a in algos]
        
        fig.add_trace(go.Bar(
            x=algos,
            y=objectives,
            name='Valor Objetivo',
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title='Valores Objetivos por Algoritmo',
            xaxis_title='Algoritmo',
            yaxis_title='Valor Objetivo',
            template='plotly_dark' if st.session_state['theme'] == 'dark' else 'plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_theory_page():
    """Render the theory page."""
    st.markdown("# 📚 Fundamentos Teóricos")
    
    tabs = st.tabs([
        "📐 Formalización",
        "🔢 Complejidad NP", 
        "🔗 Reducciones",
        "📝 Algoritmos",
        "📖 Referencias"
    ])
    
    with tabs[0]:
        st.markdown("""
        ## Formalización Matemática
        
        ### Definición del Problema
        
        El **Problema de Empaquetado Multi-Contenedor Balanceado** se define formalmente como:
        
        **Dado:**
        - Un conjunto de ítems $I = \\{1, 2, ..., n\\}$
        - Cada ítem $i$ tiene peso $w_i$ y valor $v_i$
        - $k$ contenedores con capacidades individuales $C_j$
        
        **Variables de Decisión:**
        - $x_{ij} \\in \\{0, 1\\}$: 1 si el ítem $i$ es asignado al contenedor $j$
        - $z$: makespan (valor total máximo en cualquier contenedor)
        
        ### Formulación ILP (Programación Lineal Entera)
        
        $$\\min z$$
        
        Sujeto a:
        
        $$\\sum_{j=1}^{k} x_{ij} = 1 \\quad \\forall i \\in I$$
        
        $$\\sum_{i=1}^{n} w_i \\cdot x_{ij} \\leq C_j \\quad \\forall j = 1,...,k$$
        
        $$\\sum_{i=1}^{n} v_i \\cdot x_{ij} \\leq z \\quad \\forall j = 1,...,k$$
        
        $$x_{ij} \\in \\{0, 1\\}, z \\geq 0$$
        
        ### Problema de Optimización vs. Decisión
        
        - **Optimización (BALANCED-BIN-PACKING-OPT):** Minimizar la diferencia máxima de valores
        - **Decisión (BALANCED-BIN-PACKING-DEC):** ¿Existe asignación con diferencia ≤ B?
        
        El problema de decisión es **NP-completo**, lo que implica que el problema de optimización es **NP-hard**.
        """)
    
    with tabs[1]:
        st.markdown("""
        ## Análisis de Complejidad Computacional
        
        ### NP-Completitud del Problema de Decisión
        
        **Teorema:** BALANCED-BIN-PACKING-DEC ∈ NP-completo
        
        **Demostración (esquema):**
        
        #### Parte 1: BALANCED-BIN-PACKING-DEC ∈ NP
        
        **Certificado:** Una asignación σ: I → {1,...,k}
        
        **Verificación en tiempo polinomial:**
        1. Verificar asignación completa: O(n)
        2. Calcular pesos por bin: O(n)
        3. Verificar capacidades: O(k)
        4. Calcular valores por bin: O(n)
        5. Verificar diferencia ≤ B: O(k)
        
        **Total:** O(n + k) → Polinomial ✓
        
        #### Parte 2: NP-Hardness
        
        Se demuestra mediante reducción desde **3-PARTITION**, un problema fuertemente NP-completo.
        
        ### Clases de Complejidad Relevantes
        
        | Clase | Descripción | Nuestro Problema |
        |-------|-------------|------------------|
        | P | Resoluble en tiempo polinomial | ❌ (asumiendo P ≠ NP) |
        | NP | Verificable en tiempo polinomial | ✅ (versión decisión) |
        | NP-completo | Más difícil en NP | ✅ (versión decisión) |
        | NP-hard | Al menos tan difícil como NP-completo | ✅ (versión optimización) |
        
        ### Implicaciones Prácticas
        
        1. **No existe algoritmo exacto eficiente** (asumiendo P ≠ NP)
        2. **Necesidad de aproximaciones** y heurísticas
        3. **Inaproximabilidad:** No existe PTAS general
        4. **Instancias grandes:** Requieren métodos aproximados
        """)
    
    with tabs[2]:
        st.markdown("""
        ## Cadena de Reducciones
        
        ### De PARTITION a Nuestro Problema
        
        ```
        PARTITION (Karp 1972, NP-completo)
             ↓ reducción polinomial
        3-PARTITION (Fuertemente NP-completo)
             ↓ reducción polinomial
        BIN PACKING CLÁSICO
             ↓ generalización
        BALANCED-BIN-PACKING ← Nuestro problema
        ```
        
        ### Problema 3-PARTITION
        
        **Entrada:** 
        - Conjunto A = {a₁, a₂, ..., a₃ₘ} de 3m enteros
        - Valor objetivo B tal que Σaᵢ = mB
        - Restricción: B/4 < aᵢ < B/2 para todo i
        
        **Pregunta:** ¿Se puede particionar A en m subconjuntos de 3 elementos cada uno, donde cada subconjunto suma exactamente B?
        
        **Importancia:** 3-PARTITION es **fuertemente NP-completo**:
        - Permanece NP-completo incluso con representación unaria
        - No tiene pseudo-polinomial (a diferencia de KNAPSACK)
        
        ### Reducción: 3-PARTITION ≤ₚ BALANCED-BIN-PACKING
        
        **Construcción:**
        
        Dada instancia de 3-PARTITION con {a₁,...,a₃ₘ} y objetivo B:
        
        1. **Crear ítems:** Para cada aᵢ → Item(peso=aᵢ, valor=aᵢ)
        2. **Número de bins:** k = m
        3. **Capacidades:** Cⱼ = B para todo j (uniforme)
        4. **Umbral:** β = 0 (balance perfecto)
        
        **Correctitud (⇒):**
        - Si existe 3-partición válida → todos los bins tienen valor B
        - Diferencia máxima = B - B = 0 ≤ β ✓
        
        **Correctitud (⇐):**
        - Si diferencia = 0 → todos bins tienen igual valor
        - Como Σvᵢ = mB y k = m → cada bin tiene valor B
        - Restricciones B/4 < aᵢ < B/2 → exactamente 3 elementos por bin
        - Esto constituye una 3-partición válida ✓
        
        ### Consecuencias
        
        **Corolario 1:** BALANCED-BIN-PACKING-OPT es NP-hard
        
        *Prueba:* Si existiera algoritmo polinomial para optimización, resolvería decisión en tiempo polinomial → P = NP.
        
        **Corolario 2:** Capacidades heterogéneas son ≥ difíciles que uniformes
        
        *Prueba:* Caso uniforme es instancia particular del heterogéneo.
        """)
    
    with tabs[3]:
        st.markdown("""
        ## Descripción de Algoritmos
        
        ### Algoritmos Voraces (Greedy)
        
        **First Fit Decreasing (FFD):**
        1. Ordenar ítems por peso (descendente)
        2. Para cada ítem, colocar en primer contenedor con espacio
        3. Complejidad: O(n log n)
        4. Aproximación: Sin garantía para objetivo de balance
        
        **Best Fit Decreasing (BFD):**
        1. Ordenar ítems por peso (descendente)
        2. Para cada ítem, elegir bin con mínimo espacio restante que quepa
        3. Proporciona empaquetado más compacto
        4. Complejidad: O(n²)
        
        **Worst Fit Decreasing (WFD):**
        1. Ordenar ítems por peso (descendente)
        2. Para cada ítem, elegir bin con máximo espacio restante
        3. Favorece el balance (distribuye carga)
        4. Complejidad: O(n log n)
        
        ### Programación Dinámica
        
        **Enfoque:** Construcción óptima de k-particiones
        
        **Estado:** DP[j][mask] = mejor solución con j bins asignando ítems en mask
        
        **Transición:**
        ```
        Para cada bin j:
            Para cada subconjunto S factible en bin j:
                DP[j][mask ∪ S] = mejor de:
                    - DP[j][mask ∪ S] actual
                    - DP[j-1][mask] + S en bin j
        ```
        
        **Complejidad:**
        - Tiempo: O(k · 3ⁿ) [iterar particiones]
        - Espacio: O(k · 2ⁿ)
        - Práctico: n ≤ 20
        
        **Optimización:** Pre-computar subconjuntos factibles por bin (capacidades heterogéneas)
        
        ### Branch and Bound
        
        **Estrategia:** Exploración sistemática con poda
        
        **Componentes:**
        1. **Branching:** Asignar ítem i a cada bin j posible
        2. **Bounding:** Calcular cota inferior del objetivo
        3. **Pruning:** Descartar ramas con cota ≥ mejor solución
        
        **Cotas Utilizadas:**
        - Cota trivial: diferencia actual
        - Cota optimista: distribuir valor restante uniformemente
        - Cota por relajación lineal
        
        **Complejidad:**
        - Peor caso: O(kⁿ)
        - Mejor caso: Poda extensiva reduce búsqueda
        - Práctico: n ≤ 25 con buenas cotas
        
        ### Metaheurísticas
        
        **Recocido Simulado (Simulated Annealing):**
        - Búsqueda local probabilística
        - Acepta soluciones peores con probabilidad e^(-Δ/T)
        - Temperatura T decrece (cooling schedule)
        - Escapa de óptimos locales
        
        **Algoritmo Genético:**
        - Población de soluciones evoluciona
        - Operadores: selección, cruce, mutación
        - Explora espacio de soluciones diverso
        - Balance exploración/explotación
        
        **Búsqueda Tabú:**
        - Búsqueda local con memoria
        - Lista tabú evita ciclos
        - Intensificación y diversificación
        - Memoria a corto y largo plazo
        
        ### Complejidades Comparadas
        
        | Algoritmo | Tiempo | Espacio | Optimalidad |
        |-----------|--------|---------|-------------|
        | FFD | O(n log n) | O(n) | No garantizada |
        | BFD | O(n²) | O(n) | No garantizada |
        | DP | O(k·3ⁿ) | O(k·2ⁿ) | **Óptima** |
        | B&B | O(kⁿ) peor | O(n) | **Óptima** |
        | SA | O(I·n) | O(n) | Aproximación |
        | GA | O(G·P·n) | O(P·n) | Aproximación |
        
        *Donde: I=iteraciones, G=generaciones, P=población*
        """)
    
    with tabs[4]:
        st.markdown("""
        ## Referencias Fundamentales
        
        ### Complejidad Computacional
        
        1. **Garey, M.R., & Johnson, D.S. (1979).** *Computers and Intractability: 
           A Guide to the Theory of NP-Completeness*. W.H. Freeman.
           - Teoría fundamental de NP-completitud
           - Demostración de 3-PARTITION como NP-completo
        
        2. **Karp, R.M. (1972).** "Reducibility among combinatorial problems." 
           *Complexity of Computer Computations*, 85-103.
           - 21 problemas NP-completos originales
           - Incluye PARTITION
        
        ### Bin Packing
        
        3. **Martello, S., & Toth, P. (1990).** *Knapsack Problems: Algorithms 
           and Computer Implementations*. Wiley.
           - Algoritmos exactos y aproximados
           - Programación dinámica avanzada
        
        4. **Coffman, E.G., Garey, M.R., & Johnson, D.S. (1996).** 
           "Approximation algorithms for bin packing: A survey." 
           *Approximation Algorithms for NP-hard Problems*, 46-93.
           - Estado del arte en aproximación
           - Análisis de FFD, BFD, etc.
        
        5. **Graham, R.L. (1969).** "Bounds on multiprocessing timing anomalies." 
           *SIAM Journal on Applied Mathematics*, 17(2), 416-429.
           - Algoritmo LPT para scheduling
           - Análisis de aproximación
        
        ### Metaheurísticas
        
        6. **Kirkpatrick, S., Gelatt, C.D., & Vecchi, M.P. (1983).** 
           "Optimization by simulated annealing." *Science*, 220(4598), 671-680.
           - Introducción del Simulated Annealing
           - Fundamento termodinámico
        
        7. **Goldberg, D.E. (1989).** *Genetic Algorithms in Search, Optimization 
           and Machine Learning*. Addison-Wesley.
           - Algoritmos genéticos fundamentales
           - Teoría de schemas
        
        8. **Glover, F. (1986).** "Future paths for integer programming and 
           links to artificial intelligence." *Computers & Operations Research*, 13(5), 533-549.
           - Introducción de Búsqueda Tabú
           - Estrategias de memoria
        
        ### Artículos Recientes
        
        9. **Delorme, M., Iori, M., & Martello, S. (2016).** 
           "Bin packing and cutting stock problems: Mathematical models and exact algorithms." 
           *European Journal of Operational Research*, 255(1), 1-20.
           - Survey moderno de bin packing
           - Modelos ILP avanzados
        
        10. **Baldi, M.M., Crainic, T.G., Perboli, G., & Tadei, R. (2012).**
            "The generalized bin packing problem."
            *Transportation Research Part E*, 48(6), 1205-1220.
            - Generalizaciones del problema
            - Aplicaciones logísticas
        """)


def render_settings_page():
    """Render the settings page."""
    st.markdown("# ⚙️ Configuración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Apariencia")
        
        theme = st.selectbox(
            "Tema de Color",
            options=['Oscuro', 'Claro'],
            index=0 if st.session_state['theme'] == 'dark' else 1
        )
        st.session_state['theme'] = 'dark' if theme == 'Oscuro' else 'light'
        
        st.markdown("### Rendimiento")
        
        max_iterations = st.number_input(
            "Máx. Iteraciones por Defecto",
            min_value=100,
            max_value=100000,
            value=10000
        )
        
        time_limit = st.number_input(
            "Límite de Tiempo por Defecto (segundos)",
            min_value=1,
            max_value=300,
            value=60
        )
    
    with col2:
        st.markdown("### Datos")
        
        if st.button("🗑️ Limpiar Resultados"):
            st.session_state['results'] = {}
            st.session_state['convergence_history'] = {}
            st.success("¡Resultados limpiados!")
        
        if st.button("🔄 Reiniciar Problema"):
            st.session_state['current_problem'] = None
            st.success("¡Problema reiniciado!")
        
        st.markdown("### Exportar")
        
        if st.button("📤 Exportar Todos los Datos"):
            st.info("Funcionalidad de exportación - ¡próximamente!")
    
    st.markdown("---")
    st.markdown("""
    ### Información del Sistema
    
    - **Versión:** 0.1.0
    - **Python:** 3.11+
    - **Streamlit:** Última versión
    """)


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Create component instances
    theme = st.session_state['theme']
    problem_config = ProblemConfigurator(theme=theme)
    algo_selector = AlgorithmSelector(theme=theme)
    results_display = ResultsDisplay(theme=theme)
    viz_panel = VisualizationPanel(theme=theme)
    
    # Render sidebar and get current page
    page = render_sidebar()
    
    # Render appropriate page
    if page == '🏠 Inicio':
        render_home_page()
    elif page == '🔬 Solucionador':
        render_solver_page(problem_config, algo_selector, results_display, viz_panel)
    elif page == '📊 Análisis':
        render_analysis_page(viz_panel)
    elif page == '📚 Teoría':
        render_theory_page()
    elif page == '⚙️ Configuración':
        render_settings_page()


if __name__ == "__main__":
    main()
