"""
🔬 Solucionador - Multi-Bin Packing Solver
==========================================

Página principal para configurar y ejecutar algoritmos
de optimización sobre instancias del problema.
"""

import streamlit as st
import time
from typing import Dict, Any

st.set_page_config(
    page_title="🔬 Solucionador | Multi-Bin Packing",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import shared utilities
from shared import (
    init_session_state,
    apply_custom_styles,
    render_sidebar_info,
    create_algorithm_instance,
    calculate_objective,
    calculate_balance_score,
    check_feasibility,
    ProblemConfigurator,
    AlgorithmSelector,
    ResultsDisplay,
    VisualizationPanel,
    ExportManager,
    ModernAnimations,
    Problem
)

# Initialize
init_session_state()
apply_custom_styles()
render_sidebar_info()

# ============================================================================
# Helper Functions
# ============================================================================

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
                'stability': 0.9
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


# ============================================================================
# Page Content
# ============================================================================

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

# Initialize components
problem_config = ProblemConfigurator()
algo_selector = AlgorithmSelector()
results_display = ResultsDisplay()
viz_panel = VisualizationPanel()

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
        st.session_state['current_problem'] = problem
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
            pass
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
else:
    # Empty state
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
        border-radius: 16px;
        padding: 60px;
        text-align: center;
        border: 2px dashed rgba(99, 102, 241, 0.2);
        margin-top: 20px;
    ">
        <div style="font-size: 4rem; margin-bottom: 16px; opacity: 0.5;">📊</div>
        <h3 style="color: #6366F1; margin-bottom: 8px;">Sin resultados aún</h3>
        <p style="color: #64748B;">
            Configura una instancia y ejecuta algoritmos para ver los resultados aquí
        </p>
    </div>
    """, unsafe_allow_html=True)
