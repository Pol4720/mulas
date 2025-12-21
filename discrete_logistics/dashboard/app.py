"""
Main Streamlit Application
=========================

This is the main entry point for the Balanced Multi-Bin Packing
interactive dashboard.

Run with: streamlit run app.py

Features:
- Modern glassmorphism UI design
- Interactive visualizations with Plotly
- Advanced animations and micro-interactions
- Comprehensive algorithm comparison
"""

import streamlit as st
import numpy as np
import time
from typing import Dict, Any, Optional

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="🎯 Multi-Bin Packing Solver | DAA",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Pol4720/mulas',
        'Report a bug': 'https://github.com/Pol4720/mulas/issues',
        'About': '''
        ## Multi-Bin Packing Solver
        
        Advanced optimization tool for the Balanced Multi-Bin Packing Problem.
        
        **Features:**
        - 12 algorithms (Greedy, Metaheuristics, Exact)
        - Interactive visualizations
        - Comprehensive benchmarking
        - Theoretical foundations
        
        *Universidad de La Habana - DAA Project*
        '''
    }
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

# Import modern components
from discrete_logistics.dashboard.modern_components import (
    ModernTheme,
    ModernMetricCard,
    ModernCharts,
    ModernProgress
)
from discrete_logistics.dashboard.modern_animations import (
    ModernAnimations,
    LottieAnimations,
    GlassmorphicCards,
    InteractiveChartEnhancements
)

# Import interactive components
from discrete_logistics.dashboard.interactive_components import (
    InteractiveTooltips,
    SolutionComparator,
    ExecutionTimeline,
    ParameterTuner,
    InteractiveBinVisualizer,
    QuickStatsDashboard
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
        st.session_state['theme'] = 'light'
    if 'show_confetti' not in st.session_state:
        st.session_state['show_confetti'] = False
    if 'animation_enabled' not in st.session_state:
        st.session_state['animation_enabled'] = True


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        # Modern Logo Header with Animation
        st.markdown("""
        <div style="text-align: center; padding: 24px 0;">
            <div style="
                font-size: 4rem; 
                margin-bottom: 8px;
                animation: float 3s ease-in-out infinite;
            ">📦</div>
            <div style="
                font-weight: 800; 
                font-size: 1.25rem; 
                background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.02em;
            ">
                Multi-Bin Packing
            </div>
            <div style="
                font-size: 0.75rem; 
                color: #64748B;
                margin-top: 4px;
                font-weight: 500;
            ">
                🔬 Solucionador Interactivo v2.0
            </div>
        </div>
        <style>
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-8px); }
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation with icons
        st.markdown("### 🧭 Navegación")
        page = st.radio(
            "Seleccionar Página",
            options=[
                '🏠 Inicio', 
                '🔬 Solucionador', 
                '📊 Análisis', 
                '📚 Teoría', 
                '⚙️ Configuración'
            ],
            label_visibility='collapsed'
        )
        
        st.markdown("---")
        
        # Apply modern theme
        ModernTheme.apply_modern_css()
        ThemeManager.apply_theme('light')
        
        # Quick Stats with modern styling
        st.markdown("### 📈 Estadísticas Rápidas")
        
        if st.session_state.get('current_problem'):
            prob = st.session_state['current_problem']
            
            # Animated stats
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="
                    background: rgba(79, 70, 229, 0.1);
                    border-radius: 12px;
                    padding: 12px;
                    text-align: center;
                ">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #4F46E5;">{prob.n_items}</div>
                    <div style="font-size: 0.7rem; color: #64748B;">Ítems</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="
                    background: rgba(16, 185, 129, 0.1);
                    border-radius: 12px;
                    padding: 12px;
                    text-align: center;
                ">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #10B981;">{prob.num_bins}</div>
                    <div style="font-size: 0.7rem; color: #64748B;">Contenedores</div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.get('results'):
                best = min(st.session_state['results'].values(), 
                          key=lambda x: x.get('objective', float('inf')))
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                    margin-top: 12px;
                ">
                    <div style="font-size: 0.7rem; color: #64748B; margin-bottom: 4px;">🏆 Mejor Diferencia</div>
                    <div style="font-size: 1.75rem; font-weight: 800; color: #EC4899;">{best.get('objective', 0):.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("_Genera una instancia para ver estadísticas_")
        
        st.markdown("---")
        
        # Info section with modern expander
        with st.expander("ℹ️ Acerca de", expanded=False):
            st.markdown("""
            **🎯 Multi-Bin Packing Solver**
            
            Herramienta avanzada para resolver el
            problema NP-completo de empaquetado balanceado.
            
            **✨ Características:**
            - 🔬 12 algoritmos implementados
            - 📊 Visualizaciones interactivas
            - 🎨 UI moderna con glassmorphism
            - 📈 Benchmarking avanzado
            - 📚 Fundamentos teóricos
            
            ---
            *🎓 Proyecto DAA*
            *Universidad de La Habana*
            """)
        
        # Version info with animation
        st.markdown("""
        <div style="
            position: fixed; 
            bottom: 20px; 
            left: 20px; 
            font-size: 0.7rem; 
            color: #94A3B8;
            display: flex;
            align-items: center;
            gap: 6px;
        ">
            <span style="
                display: inline-block;
                width: 8px;
                height: 8px;
                background: #10B981;
                border-radius: 50%;
                animation: pulse 2s infinite;
            "></span>
            v2.0.0 | Modern UI
        </div>
        <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
        </style>
        """, unsafe_allow_html=True)
        
        return page


def render_home_page():
    """Render the home page with modern animations."""
    # Hero Section with Animated Gradient
    st.markdown("""
    <div style="text-align: center; padding: 60px 0 40px;">
        <div style="
            font-size: 5rem; 
            margin-bottom: 20px;
            animation: float 3s ease-in-out infinite;
        ">🎯</div>
        <h1 style="
            font-size: 3rem; 
            margin-bottom: 16px;
            background: linear-gradient(270deg, #4F46E5, #7C3AED, #EC4899, #4F46E5);
            background-size: 400% 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 8s ease infinite;
        ">
            Solucionador de Empaquetado Multi-Contenedor
        </h1>
        <p style="
            font-size: 1.25rem; 
            color: #64748B; 
            max-width: 700px; 
            margin: 0 auto;
            line-height: 1.7;
        ">
            Explora y resuelve problemas de empaquetado balanceado con 
            <strong>visualizaciones interactivas</strong> y 
            <strong>múltiples algoritmos de optimización</strong>.
        </p>
    </div>
    <style>
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-15px); }
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Problem Description Card with Glassmorphism
    st.markdown("""
    <div class="glass-card" style="animation: fadeInUp 0.6s ease-out;">
        <h3 style="
            margin-top: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        ">
            <span style="font-size: 1.5rem;">📋</span>
            Descripción del Problema
        </h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 20px;">
            <div>
                <p style="color: #4F46E5; font-weight: 600; margin-bottom: 8px;">📥 Entrada:</p>
                <ul style="color: #475569; line-height: 1.8; margin: 0; padding-left: 20px;">
                    <li>Un conjunto de <strong>n ítems</strong>, cada uno con peso <em>w<sub>i</sub></em> y valor <em>v<sub>i</sub></em></li>
                    <li><strong>k contenedores</strong> con capacidades individuales <em>C<sub>j</sub></em></li>
                </ul>
            </div>
            <div>
                <p style="color: #10B981; font-weight: 600; margin-bottom: 8px;">🎯 Objetivo:</p>
                <ul style="color: #475569; line-height: 1.8; margin: 0; padding-left: 20px;">
                    <li>Minimizar la <strong>diferencia máxima</strong> de valores totales entre contenedores</li>
                    <li>Respetando las <strong>restricciones de capacidad</strong> de cada contenedor</li>
                </ul>
            </div>
        </div>
    </div>
    <style>
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("")
    
    # Features Grid with Modern Cards
    st.markdown("### ✨ Características Principales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        ("🔬", "12 Algoritmos", "Voraz, Metaheurísticas y Métodos Exactos", "#4F46E5", col1, 0),
        ("📊", "Visualizaciones", "Gráficos interactivos y animaciones", "#7C3AED", col2, 100),
        ("📈", "Benchmarking", "Comparación de rendimiento avanzada", "#EC4899", col3, 200),
        ("📚", "Teoría", "NP-Completitud y reducciones formales", "#10B981", col4, 300),
    ]
    
    for icon, title, desc, color, col, delay in features:
        with col:
            st.markdown(f"""
            <div class="glass-card" style="
                text-align: center; 
                min-height: 200px;
                animation: fadeInUp 0.6s ease-out {delay}ms both;
            ">
                <div style="
                    font-size: 3rem; 
                    margin-bottom: 16px;
                    background: linear-gradient(135deg, {color} 0%, {color}99 100%);
                    border-radius: 16px;
                    padding: 16px;
                    display: inline-block;
                ">{icon}</div>
                <h4 style="margin: 12px 0 8px; color: #1E293B; font-weight: 700;">{title}</h4>
                <p style="font-size: 0.85rem; color: #64748B; margin: 0; line-height: 1.5;">
                    {desc}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("")
    
    # Quick stats metrics with animation
    st.markdown("### 📊 Resumen del Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔢 Algoritmos", "12", delta="4 exactos", delta_color="normal")
    with col2:
        st.metric("⚡ Complejidad", "NP-Completo", help="Clase de complejidad del problema de decisión")
    with col3:
        st.metric("📦 Máx Ítems", "100", help="Cantidad de ítems soportada")
    with col4:
        st.metric("📈 Visualizaciones", "8+", help="Tipos de gráficos disponibles")
    
    # Getting Started Section with animated steps
    st.markdown("---")
    st.markdown("### 🚀 ¿Cómo comenzar?")
    
    steps = [
        ("1️⃣", "Navegar", "Ve a la página de <strong>Solucionador</strong>"),
        ("2️⃣", "Configurar", "Define tu <strong>instancia</strong> del problema"),
        ("3️⃣", "Seleccionar", "Elige los <strong>algoritmos</strong> a ejecutar"),
        ("4️⃣", "Analizar", "¡Explora los <strong>resultados</strong>!"),
    ]
    
    cols = st.columns(4)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="glass-card" style="
                text-align: center;
                padding: 20px;
                animation: fadeInUp 0.5s ease-out {i * 100}ms both;
            ">
                <div style="
                    font-size: 2rem;
                    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
                    border-radius: 12px;
                    padding: 10px;
                    display: inline-block;
                    margin-bottom: 12px;
                ">{num}</div>
                <div style="font-weight: 700; color: #1E293B; margin-bottom: 6px;">{title}</div>
                <div style="font-size: 0.8rem; color: #64748B;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def render_solver_page(problem_config: ProblemConfigurator, 
                       algo_selector: AlgorithmSelector,
                       results_display: ResultsDisplay,
                       viz_panel: VisualizationPanel):
    """Render the main solver page with modern UI."""
    
    # Page header with gradient
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="
            font-size: 2.5rem;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        ">🔬 Solucionador del Problema</h1>
        <p style="color: #64748B;">Configura, ejecuta y analiza algoritmos de optimización</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two-column layout with modern cards
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; display: flex; align-items: center; gap: 8px;">
                <span>📦</span> Configuración del Problema
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Problem configuration
        problem = problem_config.render()
        
        if problem:
            problem_config.render_problem_summary(problem)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; display: flex; align-items: center; gap: 8px;">
                <span>⚙️</span> Selección de Algoritmos
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Algorithm selection
        algorithm_configs = algo_selector.render()
        
        # Modern run button
        st.markdown("")
        if st.button("🚀 Ejecutar Algoritmos", type="primary", use_container_width=True):
            if problem is None:
                st.error("⚠️ ¡Por favor genera una instancia del problema primero!")
            elif not algorithm_configs:
                st.error("⚠️ ¡Por favor selecciona al menos un algoritmo!")
            else:
                run_algorithms(problem, algorithm_configs)
    
    # Results section
    st.markdown("---")
    
    if st.session_state.get('results'):
        # Show confetti for new results
        if st.session_state.get('show_confetti') and st.session_state.get('animation_enabled', True):
            try:
                ModernAnimations.confetti_effect()
            except:
                pass  # Gracefully handle if animations fail
            st.session_state['show_confetti'] = False
        
        results_display.render_results(st.session_state['results'])
        
        # Additional visualizations with modern layout
        st.markdown("### 📈 Análisis Visual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.get('convergence_history'):
                viz_panel.render_convergence_plot(st.session_state['convergence_history'])
            else:
                st.info("💡 Ejecuta algoritmos metaheurísticos para ver gráficos de convergencia")
        
        with col2:
            viz_panel.render_performance_radar(st.session_state['results'])
        
        # Export options with modern styling
        st.markdown("---")
        ExportManager.render_export_buttons(st.session_state['results'])


def run_algorithms(problem: Problem, algorithm_configs: list):
    """Execute selected algorithms on the problem with modern progress UI."""
    results = {}
    convergence_history = {}
    
    # Modern progress indicator
    progress_container = st.container()
    with progress_container:
        st.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <div style="font-size: 2rem; margin-bottom: 10px;">⚡</div>
            <div style="font-weight: 600; color: #4F46E5;">Ejecutando algoritmos...</div>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for idx, (algo_name, params) in enumerate(algorithm_configs):
        progress = (idx + 1) / len(algorithm_configs)
        progress_bar.progress(progress)
        status_text.markdown(f"""
        <div style="text-align: center; color: #64748B;">
            🔄 Procesando: <strong>{algo_name}</strong> ({idx + 1}/{len(algorithm_configs)})
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Create algorithm instance
            algorithm = create_algorithm_instance(algo_name, params)
            
            if algorithm is None:
                st.warning(f"⚠️ Algoritmo {algo_name} no disponible")
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
            elif hasattr(algorithm, 'objective_history'):
                convergence_history[algo_name] = algorithm.objective_history
                
        except Exception as e:
            st.error(f"❌ Error ejecutando {algo_name}: {str(e)}")
            results[algo_name] = {
                'objective': float('inf'),
                'time': 0,
                'feasible': False,
                'error': str(e)
            }
    
    # Clear progress UI
    progress_bar.progress(1.0)
    
    st.session_state['results'] = results
    st.session_state['convergence_history'] = convergence_history
    st.session_state['show_confetti'] = True
    
    # Success message with animation
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.1) 100%);
        border-left: 4px solid #10B981;
        border-radius: 0 16px 16px 0;
        padding: 20px;
        margin: 20px 0;
        animation: slideInLeft 0.5s ease-out;
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 2rem;">✅</span>
            <div>
                <div style="font-weight: 700; color: #065F46; font-size: 1.1rem;">
                    ¡Completado exitosamente!
                </div>
                <div style="color: #047857; font-size: 0.9rem;">
                    Se ejecutaron {len(results)} algoritmo(s) correctamente
                </div>
            </div>
        </div>
    </div>
    <style>
        @keyframes slideInLeft {{
            from {{ opacity: 0; transform: translateX(-30px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
    </style>
    """, unsafe_allow_html=True)


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
    """Render the analysis page with modern interactive components."""
    
    # Page header with gradient
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="
            font-size: 2.5rem;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        ">📊 Análisis de Resultados</h1>
        <p style="color: #64748B;">Explora y compara el rendimiento de los algoritmos</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get('results'):
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            border: 1px dashed #6366F140;
        ">
            <div style="font-size: 4rem; margin-bottom: 16px;">📭</div>
            <h3 style="color: #6366F1; margin-bottom: 8px;">No hay resultados para analizar</h3>
            <p style="color: #64748B;">
                ¡Ejecuta algunos algoritmos en la página de Solucionador primero!
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    results = st.session_state['results']
    
    # Quick Stats Dashboard
    st.markdown("### 📈 Resumen Rápido")
    
    # Calculate quick stats
    objectives = [r.get('objective', float('inf')) for r in results.values() if r.get('objective') != float('inf')]
    times = [r.get('time', 0) for r in results.values()]
    best_algo = min(results.items(), key=lambda x: x[1].get('objective', float('inf')))[0]
    fastest_algo = min(results.items(), key=lambda x: x[1].get('time', float('inf')))[0]
    
    stats = {
        'Mejor Objetivo': {
            'value': min(objectives) if objectives else 0,
            'icon': '🏆',
            'color': '#10B981'
        },
        'Tiempo Promedio': {
            'value': f"{np.mean(times):.3f}s" if times else "0s",
            'icon': '⏱️',
            'color': '#6366F1'
        },
        'Algoritmos': {
            'value': str(len(results)),
            'icon': '🔬',
            'color': '#8B5CF6'
        },
        'Mejor Algoritmo': {
            'value': best_algo[:15] + '...' if len(best_algo) > 15 else best_algo,
            'icon': '🥇',
            'color': '#F59E0B'
        }
    }
    
    QuickStatsDashboard.render_stats_row(stats)
    
    st.markdown("---")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Rendimiento", 
        "🔄 Convergencia", 
        "⚔️ Comparación",
        "📦 Visualización de Bins",
        "ℹ️ Info Algoritmos"
    ])

    with tab1:
        st.markdown("### 📊 Métricas de Rendimiento")
        
        # Create performance dataframe with modern styling
        import pandas as pd
        
        perf_data = []
        sorted_objs = sorted(objectives) if objectives else []
        for algo, result in results.items():
            obj = result.get('objective', float('inf'))
            # Determine ranking medal
            ranking = ''
            if algo == best_algo:
                ranking = '🥇'
            elif len(sorted_objs) > 1 and obj == sorted_objs[1]:
                ranking = '🥈'
            
            perf_data.append({
                'Algoritmo': algo,
                'Objetivo': f"{obj:.2f}" if obj != float('inf') else '∞',
                'Tiempo (s)': f"{result.get('time', 0):.4f}",
                'Balance': f"{result.get('balance_score', 0):.1%}",
                'Factible': '✅' if result.get('feasible') else '❌',
                'Ranking': ranking
            })
        
        df = pd.DataFrame(perf_data)
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Algoritmo': st.column_config.TextColumn('🔬 Algoritmo', width='medium'),
                'Objetivo': st.column_config.TextColumn('🎯 Objetivo', width='small'),
                'Tiempo (s)': st.column_config.TextColumn('⏱️ Tiempo', width='small'),
                'Balance': st.column_config.TextColumn('⚖️ Balance', width='small'),
                'Factible': st.column_config.TextColumn('✓ Válido', width='small'),
                'Ranking': st.column_config.TextColumn('🏅', width='small'),
            }
        )
        
        # Performance visualization
        st.markdown("")
        viz_panel.render_performance_radar(results)

    with tab2:
        st.markdown("### 🔄 Análisis de Convergencia")
        
        if st.session_state.get('convergence_history'):
            # Timeline visualization
            history_data = st.session_state['convergence_history']
            
            algo_selected = st.selectbox(
                "Seleccionar algoritmo:",
                options=list(history_data.keys()),
                key="convergence_algo_select"
            )
            
            if algo_selected and algo_selected in history_data:
                history = history_data[algo_selected]
                
                # Create animated convergence timeline
                fig = ExecutionTimeline.create_convergence_timeline(history)
                st.plotly_chart(fig, use_container_width=True)
            
            # Multi-algorithm comparison
            st.markdown("---")
            st.markdown("#### Comparación Multi-Algoritmo")
            viz_panel.render_convergence_plot(history_data)
        else:
            st.markdown("""
            <div style="
                background: rgba(245, 158, 11, 0.1);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                border-left: 4px solid #F59E0B;
            ">
                <div style="font-size: 2rem; margin-bottom: 8px;">💡</div>
                <p style="color: #92400E; margin: 0;">
                    <strong>No hay datos de convergencia disponibles.</strong><br>
                    Ejecuta algoritmos metaheurísticos (SA, GA, TabuSearch) para ver gráficos de convergencia.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### ⚔️ Comparación Directa")
        
        if len(results) >= 2:
            col1, col2 = st.columns(2)
            
            algo_names = list(results.keys())
            
            with col1:
                algo1 = st.selectbox("🔵 Algoritmo 1:", algo_names, index=0, key="compare_algo1")
            with col2:
                algo2 = st.selectbox("🔴 Algoritmo 2:", algo_names, index=min(1, len(algo_names)-1), key="compare_algo2")
            
            if algo1 != algo2:
                # Render comparison header
                SolutionComparator.render_comparison_header(algo1, algo2)
                
                # Metric comparisons
                SolutionComparator.render_metric_comparison(
                    "Objetivo (Diferencia)",
                    results[algo1].get('objective', 0),
                    results[algo2].get('objective', 0),
                    lower_is_better=True
                )
                
                SolutionComparator.render_metric_comparison(
                    "Tiempo de Ejecución",
                    results[algo1].get('time', 0),
                    results[algo2].get('time', 0),
                    unit="s",
                    lower_is_better=True
                )
                
                SolutionComparator.render_metric_comparison(
                    "Score de Balance",
                    results[algo1].get('balance_score', 0) * 100,
                    results[algo2].get('balance_score', 0) * 100,
                    unit="%",
                    lower_is_better=False
                )
                
                # Radar comparison chart
                st.markdown("---")
                fig = SolutionComparator.create_comparison_chart(
                    results[algo1], results[algo2], algo1, algo2
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Selecciona dos algoritmos diferentes para comparar")
        else:
            st.info("ℹ️ Necesitas ejecutar al menos 2 algoritmos para usar la comparación")

    with tab4:
        st.markdown("### 📦 Visualización de Soluciones")
        
        algo_for_viz = st.selectbox(
            "Seleccionar algoritmo para visualizar:",
            options=list(results.keys()),
            key="viz_algo_select"
        )
        
        if algo_for_viz:
            result = results[algo_for_viz]
            solution = result.get('solution')
            
            if solution and hasattr(solution, 'bins'):
                # Prepare data for visualization
                bins_data = []
                for bin_obj in solution.bins:
                    items_data = []
                    for item in bin_obj.items:
                        items_data.append({
                            'id': item.id,
                            'weight': item.weight,
                            'value': item.value
                        })
                    bins_data.append({'items': items_data})
                
                # Get capacities from problem
                problem = st.session_state.get('current_problem')
                capacities = problem.bin_capacities if problem else [100.0] * len(bins_data)
                
                # Render bin summary cards
                InteractiveBinVisualizer.render_bin_summary_cards(bins_data, capacities)
                
                st.markdown("")
                
                # Render interactive bin visualization
                fig = InteractiveBinVisualizer.create_interactive_bins(bins_data, capacities)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ No hay datos de solución disponibles para este algoritmo")

    with tab5:
        st.markdown("### ℹ️ Información de Algoritmos")
        
        st.markdown("""
        <p style="color: #64748B; margin-bottom: 20px;">
            Selecciona un algoritmo para ver información detallada sobre su funcionamiento,
            complejidad y características.
        </p>
        """, unsafe_allow_html=True)
        
        # Map internal names to display names
        algo_display_names = {
            'FirstFitDecreasing': 'First Fit Decreasing',
            'BestFitDecreasing': 'Best Fit Decreasing',
            'WorstFitDecreasing': 'Worst Fit Decreasing',
            'RoundRobinGreedy': 'Round Robin',
            'SimulatedAnnealing': 'Simulated Annealing',
            'GeneticAlgorithm': 'Genetic Algorithm',
            'TabuSearch': 'Tabu Search',
            'BranchAndBound': 'Branch and Bound',
            'DynamicProgramming': 'Dynamic Programming'
        }
        
        # Select algorithm
        selected_algo = st.selectbox(
            "Seleccionar algoritmo:",
            options=list(InteractiveTooltips.ALGORITHM_INFO.keys()),
            key="info_algo_select"
        )
        
        if selected_algo:
            InteractiveTooltips.render_algorithm_card(selected_algo)


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
        
        **Enfoque:** Construcción óptima de k-particiones mediante el esquema SRTBOT
        
        #### Esquema SRTBOT
        
        **S - Subproblemas:**
        - $DP[j][mask]$ = mejor solución con $j$ bins asignando ítems en $mask$
        - $mask$ es una máscara de bits representando el subconjunto de ítems asignados
        - Número de subproblemas: $O(k \\cdot 2^n)$
        
        **R - Relación de Recurrencia:**
        $$DP[j][mask \\cup S] = \\min_{S \\in Factible(j)} \\left\\{ \\max(V_{max}, V(S)) - \\min(V_{min}, V(S)) \\right\\}$$
        
        Donde:
        - $S$ es un subconjunto factible para el bin $j$
        - $V(S)$ es el valor total del subconjunto
        - $V_{max}, V_{min}$ son los valores extremos actuales
        
        **T - Topología:**
        1. Pre-computar subconjuntos factibles por bin
        2. Iterar $j = 1, 2, ..., k$
        3. Para cada $j$, iterar máscaras por cardinalidad creciente
        
        **B - Casos Base:**
        - $DP[1][S] = (V(S), V(S), [S])$ para $S \\in Factible(1)$
        - Con un solo bin, la diferencia es 0
        
        **O - Problema Original:**
        - $DP[k][full\\_mask]$ donde $full\\_mask = 2^n - 1$
        
        **T - Tiempo de Ejecución:**
        - Pre-computación: $O(k \\cdot 2^n \\cdot n)$
        - DP principal: $O(k \\cdot 3^n)$ (iterar particiones)
        - Espacio: $O(k \\cdot 2^n)$
        
        **Complejidad $O(3^n)$:** Surge de $\\sum_{m=0}^{n} \\binom{n}{m} 2^m = 3^n$ (teorema del binomio)
        
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
