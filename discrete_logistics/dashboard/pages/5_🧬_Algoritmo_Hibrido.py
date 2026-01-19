"""
🧬 Comparación de Algoritmos - Statistical Algorithm Comparison
===============================================================

Página para comparación estadística rigurosa de algoritmos:
- Selección dinámica de algoritmos a comparar
- Clasificación de instancias por tamaño (pequeñas, medianas, grandes)
- Tests estadísticos: Mann-Whitney, t-Student, ANOVA con Tukey HSD
- Determinación automática del mejor algoritmo
- Visualizaciones interactivas completas
- Exportación de resultados
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
import io

st.set_page_config(
    page_title="🧬 Comparación de Algoritmos | Multi-Bin Packing",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import shared utilities
from shared import (
    init_session_state,
    apply_custom_styles,
    Problem,
)

# Import algorithms and statistical framework
import sys
from pathlib import Path

_dashboard_dir = Path(__file__).parent.parent
_pkg_root = _dashboard_dir.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from discrete_logistics.algorithms.hybrid import (
    HybridDPMeta,
    HybridQualityFocused,
    HybridSpeedFocused,
    HybridWithGenetic,
    HybridWithTabu,
)
from discrete_logistics.algorithms.metaheuristics import SimulatedAnnealing, GeneticAlgorithm, TabuSearch
from discrete_logistics.algorithms.dynamic_programming import DynamicProgramming
from discrete_logistics.algorithms.greedy import (
    FirstFitDecreasing, 
    BestFitDecreasing, 
    WorstFitDecreasing,
    RoundRobinGreedy,
    LargestDifferenceFirst
)
from discrete_logistics.algorithms.approximation import LPTApproximation, MultiWayPartition
from discrete_logistics.algorithms.external_adapters import GeminiHGADP, QwenSADP
from discrete_logistics.benchmarks.statistical_framework import (
    StatisticalTests,
    ExperimentRunner,
    ExperimentConfig,
    StatisticalResult
)
from discrete_logistics.core.instance_generator import InstanceGenerator

# Initialize
init_session_state()
apply_custom_styles()

# ============================================================================
# Available Algorithms Registry
# ============================================================================

AVAILABLE_ALGORITHMS = {
    # Hybrid Algorithms
    "🧬 Hybrid DP-Meta": lambda params: HybridDPMeta(
        dp_threshold=params.get('dp_threshold', 10),
        meta_time_limit=params.get('meta_time_limit', 30),
        verbose=False
    ),
    "🧬 Hybrid (Calidad)": lambda params: HybridQualityFocused(verbose=False),
    "🧬 Hybrid (Velocidad)": lambda params: HybridSpeedFocused(verbose=False),
    "🧬 Hybrid + Genético": lambda params: HybridWithGenetic(verbose=False),
    "🧬 Hybrid + Tabú": lambda params: HybridWithTabu(verbose=False),
    
    # External Algorithms (Gemini & Qwen)
    "🤖 H-GADP (Gemini)": lambda params: GeminiHGADP(
        alpha=0.3,
        time_limit=params.get('time_limit', 30),
        verbose=False
    ),
    "🤖 SA-DP (Qwen)": lambda params: QwenSADP(
        alpha=0.65,
        time_limit=params.get('time_limit', 30),
        verbose=False
    ),
    
    # Metaheuristics
    "🌡️ Simulated Annealing": lambda params: SimulatedAnnealing(
        time_limit=params.get('time_limit', 30),
        verbose=False
    ),
    "🧬 Algoritmo Genético": lambda params: GeneticAlgorithm(
        time_limit=params.get('time_limit', 30),
        verbose=False
    ),
    "📋 Búsqueda Tabú": lambda params: TabuSearch(
        time_limit=params.get('time_limit', 30),
        verbose=False
    ),
    
    # Greedy Algorithms
    "📦 First Fit Decreasing": lambda params: FirstFitDecreasing(verbose=False),
    "📦 Best Fit Decreasing": lambda params: BestFitDecreasing(verbose=False),
    "📦 Worst Fit Decreasing": lambda params: WorstFitDecreasing(verbose=False),
    "🔄 Round Robin Greedy": lambda params: RoundRobinGreedy(verbose=False),
    "📊 Largest Diff First": lambda params: LargestDifferenceFirst(verbose=False),
    
    # Approximation Algorithms
    "⚖️ LPT Approximation": lambda params: LPTApproximation(verbose=False),
    "🔀 MultiWay Partition": lambda params: MultiWayPartition(verbose=False),
}

# Instance size classification
INSTANCE_SIZES = {
    "pequeña": {"n_range": [10, 15], "description": "10-15 ítems"},
    "mediana": {"n_range": [20, 30, 40], "description": "20-40 ítems"},
    "grande": {"n_range": [50, 75, 100], "description": "50-100 ítems"},
}

# ============================================================================
# Custom Styles
# ============================================================================

st.markdown("""
<style>
.comparison-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
}

.comparison-header h1 {
    color: white;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.comparison-header p {
    opacity: 0.9;
    font-size: 1.1rem;
}

.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #667eea;
    margin-bottom: 1rem;
}

.stat-card.winner {
    border-left-color: #10B981;
    background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
}

.stat-card.significant {
    border-left-color: #10B981;
}

.stat-card.not-significant {
    border-left-color: #F59E0B;
    background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
}

.stat-title {
    font-weight: 600;
    color: #374151;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1F2937;
}

.stat-subtitle {
    color: #6B7280;
    font-size: 0.85rem;
}

.winner-badge {
    display: inline-block;
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 1rem;
    margin: 0.5rem 0;
}

.size-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 15px;
    font-weight: 500;
    font-size: 0.8rem;
    margin-right: 0.5rem;
}

.size-small {
    background: #DBEAFE;
    color: #1D4ED8;
}

.size-medium {
    background: #FEF3C7;
    color: #D97706;
}

.size-large {
    background: #FEE2E2;
    color: #DC2626;
}

.algorithm-chip {
    display: inline-block;
    background: #F3F4F6;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    margin: 0.25rem;
    font-size: 0.85rem;
}

.p-value-significant {
    color: #059669;
    font-weight: bold;
}

.p-value-not-significant {
    color: #D97706;
}

.results-table {
    width: 100%;
    border-collapse: collapse;
}

.results-table th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    text-align: left;
}

.results-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #E5E7EB;
}

.results-table tr:hover {
    background: #F9FAFB;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================

st.markdown("""
<div class="comparison-header">
    <h1>🧬 Comparación Estadística de Algoritmos</h1>
    <p>Análisis riguroso con clasificación por tamaño de instancia y determinación del mejor algoritmo</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Session State
# ============================================================================

if 'experiment_results' not in st.session_state:
    st.session_state['experiment_results'] = None
if 'selected_algorithms' not in st.session_state:
    st.session_state['selected_algorithms'] = []
if 'experiment_config' not in st.session_state:
    st.session_state['experiment_config'] = None

# ============================================================================
# Sidebar - Dynamic Algorithm Selection
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuración del Experimento")
    
    st.markdown("#### 🤖 Selección de Algoritmos")
    st.caption("Selecciona los algoritmos a comparar:")
    
    # Group algorithms by category
    algo_categories = {
        "🧬 Híbridos": [k for k in AVAILABLE_ALGORITHMS.keys() if "Hybrid" in k or "🧬 Hybrid" in k],
        "🤖 Externos (Gemini/Qwen)": ["🤖 H-GADP (Gemini)", "🤖 SA-DP (Qwen)"],
        "🔥 Metaheurísticas": ["🌡️ Simulated Annealing", "🧬 Algoritmo Genético", "📋 Búsqueda Tabú"],
        "📦 Greedy": [k for k in AVAILABLE_ALGORITHMS.keys() if "Fit" in k or "Round" in k or "Largest" in k],
        "⚖️ Aproximación": ["⚖️ LPT Approximation", "🔀 MultiWay Partition"],
    }
    
    selected_algos = []
    
    for category, algos in algo_categories.items():
        with st.expander(category, expanded=True):
            for algo in algos:
                if algo in AVAILABLE_ALGORITHMS:
                    if st.checkbox(algo, value=(algo in ["🧬 Hybrid DP-Meta", "🌡️ Simulated Annealing", "📦 First Fit Decreasing"]), key=f"algo_{algo}"):
                        selected_algos.append(algo)
    
    st.session_state['selected_algorithms'] = selected_algos
    
    st.markdown("---")
    st.markdown("#### 📏 Clasificación de Instancias")
    
    use_small = st.checkbox("📘 Pequeñas (10-15 ítems)", value=True)
    use_medium = st.checkbox("📙 Medianas (20-40 ítems)", value=True)
    use_large = st.checkbox("📕 Grandes (50-100 ítems)", value=True)
    
    st.markdown("---")
    st.markdown("#### 🎲 Parámetros de Instancias")
    
    k_bins = st.multiselect(
        "Contenedores (k)",
        [2, 3, 4, 5],
        default=[2, 3]
    )
    
    distributions = st.multiselect(
        "Distribuciones",
        ["uniform", "normal", "correlated"],
        default=["uniform", "normal"],
        format_func=lambda x: {"uniform": "Uniforme", "normal": "Normal", "correlated": "Correlacionada"}.get(x, x)
    )
    
    repetitions = st.slider(
        "Repeticiones",
        5, 50, 15,
        help="Más repeticiones = resultados más robustos"
    )
    
    time_limit = st.slider(
        "Tiempo límite por algoritmo (s)",
        10, 120, 30
    )
    
    alpha_level = st.selectbox(
        "Nivel de significancia (α)",
        [0.01, 0.05, 0.10],
        index=1,
        format_func=lambda x: f"{x} ({int((1-x)*100)}% confianza)"
    )

# ============================================================================
# Main Content
# ============================================================================

# Build instance sizes based on selection
selected_n_values = []
size_labels = {}
if use_small:
    selected_n_values.extend(INSTANCE_SIZES["pequeña"]["n_range"])
    for n in INSTANCE_SIZES["pequeña"]["n_range"]:
        size_labels[n] = "pequeña"
if use_medium:
    selected_n_values.extend(INSTANCE_SIZES["mediana"]["n_range"])
    for n in INSTANCE_SIZES["mediana"]["n_range"]:
        size_labels[n] = "mediana"
if use_large:
    selected_n_values.extend(INSTANCE_SIZES["grande"]["n_range"])
    for n in INSTANCE_SIZES["grande"]["n_range"]:
        size_labels[n] = "grande"

# Display selected configuration
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🤖 Algoritmos Seleccionados")
    if selected_algos:
        for algo in selected_algos:
            st.markdown(f'<span class="algorithm-chip">{algo}</span>', unsafe_allow_html=True)
    else:
        st.warning("Selecciona al menos 2 algoritmos en la barra lateral")

with col2:
    st.markdown("#### 📏 Tamaños de Instancia")
    if use_small:
        st.markdown('<span class="size-badge size-small">Pequeñas: 10-15</span>', unsafe_allow_html=True)
    if use_medium:
        st.markdown('<span class="size-badge size-medium">Medianas: 20-40</span>', unsafe_allow_html=True)
    if use_large:
        st.markdown('<span class="size-badge size-large">Grandes: 50-100</span>', unsafe_allow_html=True)

with col3:
    st.markdown("#### 📊 Configuración")
    st.markdown(f"**Contenedores:** {k_bins}")
    st.markdown(f"**Distribuciones:** {len(distributions)}")
    st.markdown(f"**Repeticiones:** {repetitions}")

st.markdown("---")

# ============================================================================
# Run Experiment Button
# ============================================================================

if len(selected_algos) >= 2 and len(selected_n_values) > 0:
    total_experiments = len(selected_n_values) * len(k_bins) * len(distributions) * repetitions * len(selected_algos)
    
    st.info(f"📊 **Total de experimentos:** {total_experiments} ({len(selected_algos)} algoritmos × {len(selected_n_values)} tamaños × {len(k_bins)} k × {len(distributions)} distribuciones × {repetitions} repeticiones)")
    
    if st.button("🚀 Ejecutar Experimento Completo", type="primary", use_container_width=True):
        
        # Create algorithm instances
        algo_params = {'time_limit': time_limit, 'dp_threshold': 10, 'meta_time_limit': time_limit}
        algorithms = []
        for algo_name in selected_algos:
            try:
                algo_instance = AVAILABLE_ALGORITHMS[algo_name](algo_params)
                algorithms.append((algo_name, algo_instance))
            except Exception as e:
                st.warning(f"No se pudo crear {algo_name}: {e}")
        
        # Configure experiment
        config = ExperimentConfig(
            n_items_range=sorted(selected_n_values),
            k_bins_range=k_bins,
            distributions=distributions,
            capacity_variations=[0.0],
            repetitions=repetitions
        )
        
        # Create runner with algorithm instances
        runner = ExperimentRunner(
            [algo for _, algo in algorithms],
            config,
            verbose=False
        )
        
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_estimate = st.empty()
        
        start_time = time.time()
        
        def update_progress(current, total, desc):
            progress = current / total
            progress_bar.progress(progress)
            elapsed = time.time() - start_time
            if current > 0:
                eta = (elapsed / current) * (total - current)
                time_estimate.markdown(f"⏱️ Tiempo restante estimado: **{eta:.0f}s**")
            status_text.markdown(f"🔄 **{desc}** ({current}/{total})")
        
        # Run experiments
        results = runner.run(callback=update_progress)
        
        progress_bar.progress(100)
        status_text.markdown("✅ **Experimento completado**")
        time_estimate.markdown(f"⏱️ Tiempo total: **{time.time() - start_time:.1f}s**")
        
        # Create mapping from internal names to display names
        name_to_display = {}
        for display_name, algo_instance in algorithms:
            internal_name = algo_instance.name
            name_to_display[internal_name] = display_name
        
        # Store results with metadata
        st.session_state['experiment_results'] = {
            'runner': runner,
            'results': results,
            'config': config,
            'algorithms': algorithms,
            'algo_names': selected_algos,
            'size_labels': size_labels,
            'alpha': alpha_level,
            'name_to_display': name_to_display
        }
        
        time.sleep(1)
        st.rerun()

else:
    if len(selected_algos) < 2:
        st.warning("⚠️ Selecciona al menos **2 algoritmos** en la barra lateral para comparar.")
    if len(selected_n_values) == 0:
        st.warning("⚠️ Selecciona al menos **un tamaño de instancia**.")

# ============================================================================
# Display Results
# ============================================================================

if st.session_state['experiment_results'] is not None:
    exp_data = st.session_state['experiment_results']
    runner = exp_data['runner']
    algo_names = exp_data['algo_names']
    size_labels = exp_data['size_labels']
    alpha = exp_data['alpha']
    name_to_display = exp_data.get('name_to_display', {})
    
    st.markdown("---")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Resumen y Ganadores",
        "📊 Análisis por Tamaño",
        "📈 Tests Estadísticos",
        "📉 Visualizaciones",
        "💾 Exportar"
    ])
    
    # ========================================================================
    # Tab 1: Summary and Winners
    # ========================================================================
    with tab1:
        st.markdown("### 🏆 Resumen de Resultados y Determinación del Mejor Algoritmo")
        
        # Debug info
        total_results = len(runner.results)
        feasible_results = sum(1 for r in runner.results if r.is_feasible)
        
        if total_results == 0:
            st.error("❌ No se ejecutaron experimentos. Por favor, vuelve a ejecutar.")
        else:
            # Show debug info in expander
            with st.expander("📊 Información de depuración", expanded=False):
                st.write(f"**Total de resultados:** {total_results}")
                st.write(f"**Resultados factibles:** {feasible_results}")
                st.write(f"**Resultados no factibles:** {total_results - feasible_results}")
                
                if feasible_results < total_results:
                    # Sample some non-feasible results
                    non_feasible = [r for r in runner.results if not r.is_feasible][:3]
                    for r in non_feasible:
                        st.write(f"- {r.algorithm_name}: obj={r.objective_value}, config={r.instance_config}")
        
        # Get all objective values by algorithm (include all results with finite objective)
        all_results_df = []
        for result in runner.results:
            # Include result if feasible OR has finite objective value
            if result.is_feasible or (result.objective_value != float('inf') and result.objective_value is not None):
                n = result.instance_config['n']
                # Use display name if available, otherwise use internal name
                display_name = name_to_display.get(result.algorithm_name, result.algorithm_name)
                all_results_df.append({
                    'Algoritmo': display_name,
                    'Objetivo': result.objective_value if result.objective_value != float('inf') else None,
                    'Tiempo': result.execution_time,
                    'n': n,
                    'k': result.instance_config['k'],
                    'Distribución': result.instance_config['distribution'],
                    'Tamaño': size_labels.get(n, 'desconocido'),
                    'Factible': result.is_feasible
                })
        
        df_all = pd.DataFrame(all_results_df)
        
        # Remove rows with None objective
        if len(df_all) > 0 and 'Objetivo' in df_all.columns:
            df_all = df_all.dropna(subset=['Objetivo'])
        
        if len(df_all) > 0:
            # Overall summary
            summary_stats = df_all.groupby('Algoritmo').agg({
                'Objetivo': ['mean', 'std', 'median', 'min', 'max', 'count'],
                'Tiempo': ['mean', 'std']
            }).round(4)
            summary_stats.columns = ['Media Obj', 'Std Obj', 'Mediana Obj', 'Min Obj', 'Max Obj', 'N', 'Media Tiempo', 'Std Tiempo']
            summary_stats = summary_stats.sort_values('Media Obj')
            
            # Determine overall winner
            winner_name = summary_stats.index[0]
            winner_stats = summary_stats.iloc[0]
            
            # Winner announcement
            st.markdown(f"""
            <div class="stat-card winner" style="text-align: center;">
                <div class="stat-title">🏆 MEJOR ALGORITMO GLOBAL</div>
                <div class="winner-badge">{winner_name}</div>
                <div style="margin-top: 1rem;">
                    <span style="font-size: 1.5rem; font-weight: bold; color: #059669;">
                        {winner_stats['Media Obj']:.4f}
                    </span>
                    <span style="color: #6B7280;"> ± {winner_stats['Std Obj']:.4f}</span>
                </div>
                <div class="stat-subtitle">Objetivo promedio (menor es mejor)</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Summary table
            st.markdown("#### 📋 Tabla Resumen de Todos los Algoritmos")
            
            # Create a formatted DataFrame for display
            display_df = summary_stats.reset_index()
            display_df.columns = ['Algoritmo', 'Media', 'Desv.Est.', 'Mediana', 'Mínimo', 'Máximo', 'N', 'Tiempo (s)', 'Tiempo Std']
            
            # Add ranking
            display_df['Ranking'] = range(1, len(display_df) + 1)
            display_df = display_df[['Ranking', 'Algoritmo', 'Media', 'Desv.Est.', 'Mediana', 'Mínimo', 'Máximo', 'N', 'Tiempo (s)']]
            
            st.dataframe(
                display_df.style.format({
                    'Media': '{:.4f}',
                    'Desv.Est.': '{:.4f}',
                    'Mediana': '{:.4f}',
                    'Mínimo': '{:.4f}',
                    'Máximo': '{:.4f}',
                    'Tiempo (s)': '{:.3f}'
                }).background_gradient(subset=['Media'], cmap='RdYlGn_r'),
                use_container_width=True,
                hide_index=True
            )
            
            # Statistical significance of winner
            st.markdown("#### 🔬 Significancia Estadística del Ganador")
            
            winner_values = df_all[df_all['Algoritmo'] == winner_name]['Objetivo'].values
            
            sig_results = []
            for algo in summary_stats.index[1:]:  # Compare winner vs all others
                other_values = df_all[df_all['Algoritmo'] == algo]['Objetivo'].values
                if len(winner_values) > 0 and len(other_values) > 0:
                    test_result = StatisticalTests.mann_whitney_u(
                        winner_values, other_values,
                        alpha=alpha,
                        alternative='less'
                    )
                    sig_results.append({
                        'Comparación': f"{winner_name} vs {algo}",
                        'p-valor': test_result.p_value,
                        'Significativo': "✅ Sí" if test_result.is_significant else "❌ No",
                        "Cliff's δ": test_result.effect_size,
                        'Efecto': test_result.effect_size_interpretation
                    })
            
            if sig_results:
                df_sig = pd.DataFrame(sig_results)
                st.dataframe(
                    df_sig.style.format({
                        'p-valor': '{:.6f}',
                        "Cliff's δ": '{:.4f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Count significant wins
                n_significant = sum(1 for r in sig_results if "✅" in r['Significativo'])
                st.success(f"✅ **{winner_name}** es significativamente mejor que **{n_significant}/{len(sig_results)}** algoritmos (α={alpha})")
        else:
            st.warning("No hay resultados factibles para mostrar.")
    
    # ========================================================================
    # Tab 2: Analysis by Size
    # ========================================================================
    with tab2:
        st.markdown("### 📊 Análisis Diferenciado por Tamaño de Instancia")
        
        if len(df_all) > 0:
            # Winners by size
            sizes_present = df_all['Tamaño'].unique()
            
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            
            size_winners = {}
            
            for i, size in enumerate(['pequeña', 'mediana', 'grande']):
                if size in sizes_present:
                    with cols[i]:
                        df_size = df_all[df_all['Tamaño'] == size]
                        
                        size_summary = df_size.groupby('Algoritmo')['Objetivo'].agg(['mean', 'std', 'count']).round(4)
                        size_summary = size_summary.sort_values('mean')
                        
                        if len(size_summary) > 0:
                            size_winner = size_summary.index[0]
                            size_winners[size] = size_winner
                            
                            size_class = {'pequeña': 'size-small', 'mediana': 'size-medium', 'grande': 'size-large'}[size]
                            size_emoji = {'pequeña': '📘', 'mediana': '📙', 'grande': '📕'}[size]
                            
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-title">{size_emoji} Instancias {size.upper()}s</div>
                                <div class="winner-badge" style="font-size: 0.85rem;">{size_winner}</div>
                                <div style="margin-top: 0.5rem;">
                                    <span style="font-size: 1.2rem; font-weight: bold;">
                                        {size_summary.loc[size_winner, 'mean']:.4f}
                                    </span>
                                    <span style="color: #6B7280; font-size: 0.9rem;"> ± {size_summary.loc[size_winner, 'std']:.4f}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Detailed table by size
            st.markdown("#### 📋 Rendimiento Detallado por Tamaño")
            
            for size in ['pequeña', 'mediana', 'grande']:
                if size in sizes_present:
                    with st.expander(f"{'📘' if size == 'pequeña' else '📙' if size == 'mediana' else '📕'} Instancias {size.title()}s", expanded=True):
                        df_size = df_all[df_all['Tamaño'] == size]
                        
                        size_stats = df_size.groupby('Algoritmo').agg({
                            'Objetivo': ['mean', 'std', 'median'],
                            'Tiempo': 'mean'
                        }).round(4)
                        size_stats.columns = ['Media', 'Desv.Est.', 'Mediana', 'Tiempo (s)']
                        size_stats = size_stats.sort_values('Media')
                        size_stats['Ranking'] = range(1, len(size_stats) + 1)
                        
                        st.dataframe(
                            size_stats[['Ranking', 'Media', 'Desv.Est.', 'Mediana', 'Tiempo (s)']].style.format({
                                'Media': '{:.4f}',
                                'Desv.Est.': '{:.4f}',
                                'Mediana': '{:.4f}',
                                'Tiempo (s)': '{:.3f}'
                            }).background_gradient(subset=['Media'], cmap='RdYlGn_r'),
                            use_container_width=True
                        )
            
            # Visualization: Performance by size
            st.markdown("#### 📈 Visualización del Rendimiento por Tamaño")
            
            # Heatmap of rankings by size
            ranking_data = []
            for size in ['pequeña', 'mediana', 'grande']:
                if size in sizes_present:
                    df_size = df_all[df_all['Tamaño'] == size]
                    size_means = df_size.groupby('Algoritmo')['Objetivo'].mean().sort_values()
                    for rank, (algo, mean) in enumerate(size_means.items(), 1):
                        ranking_data.append({
                            'Algoritmo': algo,
                            'Tamaño': size.title(),
                            'Ranking': rank,
                            'Media': mean
                        })
            
            if ranking_data:
                df_ranking = pd.DataFrame(ranking_data)
                
                # Create heatmap
                fig_heatmap = px.density_heatmap(
                    df_ranking,
                    x='Tamaño',
                    y='Algoritmo',
                    z='Ranking',
                    color_continuous_scale='RdYlGn_r',
                    title='Ranking de Algoritmos por Tamaño de Instancia (menor es mejor)'
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Line chart showing performance across sizes
                fig_line = px.line(
                    df_ranking,
                    x='Tamaño',
                    y='Media',
                    color='Algoritmo',
                    markers=True,
                    title='Objetivo Promedio por Tamaño de Instancia'
                )
                fig_line.update_layout(height=400)
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("No hay datos para análisis por tamaño.")
    
    # ========================================================================
    # Tab 3: Statistical Tests
    # ========================================================================
    with tab3:
        st.markdown("### 📈 Tests Estadísticos Completos")
        
        if len(df_all) > 0:
            unique_algos = df_all['Algoritmo'].unique()
            
            # ANOVA
            st.markdown("#### 🧪 ANOVA (Análisis de Varianza)")
            
            samples = [df_all[df_all['Algoritmo'] == algo]['Objetivo'].values for algo in unique_algos]
            samples = [s for s in samples if len(s) > 0]
            
            if len(samples) >= 2:
                anova_result = StatisticalTests.one_way_anova(
                    *samples,
                    group_names=list(unique_algos),
                    alpha=alpha
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sig_class = "significant" if anova_result.is_significant else "not-significant"
                    st.markdown(f"""
                    <div class="stat-card {sig_class}">
                        <div class="stat-title">Estadístico F</div>
                        <div class="stat-value">{anova_result.statistic:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stat-card {sig_class}">
                        <div class="stat-title">p-valor</div>
                        <div class="stat-value {'p-value-significant' if anova_result.is_significant else 'p-value-not-significant'}">{anova_result.p_value:.6f}</div>
                        <div class="stat-subtitle">α = {alpha}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="stat-card {sig_class}">
                        <div class="stat-title">η² (Tamaño del Efecto)</div>
                        <div class="stat-value">{anova_result.effect_size:.4f}</div>
                        <div class="stat-subtitle">{anova_result.effect_size_interpretation}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if anova_result.is_significant:
                    st.success(f"✅ **ANOVA significativo** (p={anova_result.p_value:.6f}): Existen diferencias significativas entre los algoritmos.")
                else:
                    st.warning(f"⚠️ **ANOVA no significativo** (p={anova_result.p_value:.6f}): No hay evidencia de diferencias significativas.")
                
                # Tukey HSD if ANOVA significant
                if anova_result.is_significant:
                    st.markdown("#### 📊 Post-hoc: Tukey HSD (Comparaciones Múltiples)")
                    
                    try:
                        tukey_results = StatisticalTests.tukey_hsd(
                            *samples,
                            group_names=list(unique_algos),
                            alpha=alpha
                        )
                        
                        tukey_data = []
                        for key, result in tukey_results.items():
                            tukey_data.append({
                                'Comparación': key,
                                'Estadístico': result.statistic,
                                'p-valor': result.p_value,
                                'Significativo': "✅" if result.is_significant else "❌",
                                'Diferencia Media': result.details.get('mean_diff', 0)
                            })
                        
                        df_tukey = pd.DataFrame(tukey_data)
                        st.dataframe(
                            df_tukey.style.format({
                                'Estadístico': '{:.4f}',
                                'p-valor': '{:.6f}',
                                'Diferencia Media': '{:.4f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    except Exception as e:
                        st.info(f"Tukey HSD no disponible: {e}")
            
            # Kruskal-Wallis (non-parametric alternative)
            st.markdown("#### 🧪 Kruskal-Wallis (Alternativa No Paramétrica)")
            
            if len(samples) >= 2:
                kw_result = StatisticalTests.kruskal_wallis(
                    *samples,
                    group_names=list(unique_algos),
                    alpha=alpha
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    sig_class = "significant" if kw_result.is_significant else "not-significant"
                    st.markdown(f"""
                    <div class="stat-card {sig_class}">
                        <div class="stat-title">Estadístico H</div>
                        <div class="stat-value">{kw_result.statistic:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stat-card {sig_class}">
                        <div class="stat-title">p-valor</div>
                        <div class="stat-value">{kw_result.p_value:.6f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Pairwise Mann-Whitney U tests
            st.markdown("#### 🔄 Comparaciones Pareadas (Mann-Whitney U)")
            
            pairwise_data = []
            algos_list = list(unique_algos)
            
            for i, algo1 in enumerate(algos_list):
                for j, algo2 in enumerate(algos_list):
                    if i < j:
                        values1 = df_all[df_all['Algoritmo'] == algo1]['Objetivo'].values
                        values2 = df_all[df_all['Algoritmo'] == algo2]['Objetivo'].values
                        
                        if len(values1) > 0 and len(values2) > 0:
                            mw_result = StatisticalTests.mann_whitney_u(
                                values1, values2,
                                alpha=alpha
                            )
                            
                            pairwise_data.append({
                                'Algoritmo 1': algo1,
                                'Algoritmo 2': algo2,
                                'U': mw_result.statistic,
                                'p-valor': mw_result.p_value,
                                "Cliff's δ": mw_result.effect_size,
                                'Efecto': mw_result.effect_size_interpretation,
                                'Significativo': "✅" if mw_result.is_significant else "❌",
                                'Mejor': algo1 if mw_result.details['median_1'] < mw_result.details['median_2'] else algo2
                            })
            
            if pairwise_data:
                df_pairwise = pd.DataFrame(pairwise_data)
                st.dataframe(
                    df_pairwise.style.format({
                        'U': '{:.2f}',
                        'p-valor': '{:.6f}',
                        "Cliff's δ": '{:.4f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Summary of significant differences
                n_sig = sum(1 for p in pairwise_data if p['Significativo'] == "✅")
                st.info(f"📊 **{n_sig}/{len(pairwise_data)}** comparaciones muestran diferencias significativas (α={alpha})")
        else:
            st.warning("No hay datos suficientes para tests estadísticos.")
    
    # ========================================================================
    # Tab 4: Visualizations
    # ========================================================================
    with tab4:
        st.markdown("### 📉 Visualizaciones Interactivas")
        
        if len(df_all) > 0:
            # Box plot
            st.markdown("#### 📦 Distribución de Objetivos por Algoritmo")
            
            fig_box = px.box(
                df_all,
                x='Algoritmo',
                y='Objetivo',
                color='Algoritmo',
                title='Distribución de Objetivos por Algoritmo',
                points='outliers'
            )
            fig_box.update_layout(
                showlegend=False,
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Violin plot
            st.markdown("#### 🎻 Violín Plot con Densidad")
            
            fig_violin = px.violin(
                df_all,
                x='Algoritmo',
                y='Objetivo',
                color='Algoritmo',
                box=True,
                points='outliers',
                title='Distribución de Objetivos con Densidad'
            )
            fig_violin.update_layout(
                showlegend=False,
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            
            # Box plot by size
            st.markdown("#### 📊 Distribución por Tamaño de Instancia")
            
            fig_box_size = px.box(
                df_all,
                x='Tamaño',
                y='Objetivo',
                color='Algoritmo',
                title='Distribución de Objetivos por Tamaño y Algoritmo'
            )
            fig_box_size.update_layout(height=500)
            st.plotly_chart(fig_box_size, use_container_width=True)
            
            # Scatter: Objective vs Time
            st.markdown("#### ⏱️ Objetivo vs Tiempo de Ejecución")
            
            fig_scatter = px.scatter(
                df_all,
                x='Tiempo',
                y='Objetivo',
                color='Algoritmo',
                size='n',
                hover_data=['k', 'Distribución', 'Tamaño'],
                title='Relación Objetivo-Tiempo por Algoritmo',
                labels={'Tiempo': 'Tiempo (s)', 'Objetivo': 'Objetivo (menor es mejor)'}
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Mean comparison with confidence intervals
            st.markdown("#### 📈 Medias con Intervalos de Confianza (95%)")
            
            ci_data = []
            for algo in df_all['Algoritmo'].unique():
                values = df_all[df_all['Algoritmo'] == algo]['Objetivo'].values
                mean = np.mean(values)
                se = np.std(values, ddof=1) / np.sqrt(len(values))
                ci = 1.96 * se
                ci_data.append({
                    'Algoritmo': algo,
                    'Media': mean,
                    'CI_lower': mean - ci,
                    'CI_upper': mean + ci
                })
            
            df_ci = pd.DataFrame(ci_data).sort_values('Media')
            
            fig_ci = go.Figure()
            fig_ci.add_trace(go.Scatter(
                x=df_ci['Algoritmo'],
                y=df_ci['Media'],
                mode='markers',
                marker=dict(size=12, color='#667eea'),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=df_ci['CI_upper'] - df_ci['Media'],
                    arrayminus=df_ci['Media'] - df_ci['CI_lower']
                ),
                name='Media ± IC 95%'
            ))
            fig_ci.update_layout(
                title='Comparación de Medias con Intervalos de Confianza',
                yaxis_title='Objetivo (menor es mejor)',
                xaxis_title='Algoritmo',
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_ci, use_container_width=True)
            
            # Heatmap: Performance by n and k
            st.markdown("#### 🗺️ Mapa de Calor: Rendimiento por n y k")
            
            # Select algorithm for heatmap
            selected_algo_heatmap = st.selectbox(
                "Selecciona algoritmo para el mapa de calor:",
                df_all['Algoritmo'].unique()
            )
            
            df_algo = df_all[df_all['Algoritmo'] == selected_algo_heatmap]
            pivot_table = df_algo.pivot_table(
                values='Objetivo',
                index='n',
                columns='k',
                aggfunc='mean'
            )
            
            fig_heatmap = px.imshow(
                pivot_table,
                labels=dict(x='k (contenedores)', y='n (ítems)', color='Objetivo'),
                title=f'Rendimiento de {selected_algo_heatmap} por n y k',
                color_continuous_scale='RdYlGn_r'
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("No hay datos para visualizar.")
    
    # ========================================================================
    # Tab 5: Export
    # ========================================================================
    with tab5:
        st.markdown("### 💾 Exportar Resultados")
        
        if len(df_all) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📄 Datos del Experimento")
                
                # CSV export
                csv_content = df_all.to_csv(index=False)
                st.download_button(
                    "📥 Descargar CSV Completo",
                    csv_content,
                    file_name=f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Summary CSV
                summary_csv = df_all.groupby('Algoritmo').agg({
                    'Objetivo': ['mean', 'std', 'median', 'min', 'max', 'count'],
                    'Tiempo': ['mean', 'std']
                }).round(4).to_csv()
                
                st.download_button(
                    "📥 Descargar Resumen CSV",
                    summary_csv,
                    file_name=f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # JSON export
                json_data = {
                    'timestamp': datetime.now().isoformat(),
                    'config': {
                        'n_values': exp_data['config'].n_items_range,
                        'k_values': exp_data['config'].k_bins_range,
                        'distributions': exp_data['config'].distributions,
                        'repetitions': exp_data['config'].repetitions,
                        'alpha': alpha
                    },
                    'algorithms': algo_names,
                    'results': df_all.to_dict('records'),
                    'summary': df_all.groupby('Algoritmo')['Objetivo'].agg(['mean', 'std', 'median']).to_dict()
                }
                
                st.download_button(
                    "📥 Descargar JSON",
                    json.dumps(json_data, indent=2, default=str),
                    file_name=f"experiment_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                st.markdown("#### 📊 Informe Estadístico")
                
                # Generate text report
                report_lines = []
                report_lines.append("=" * 60)
                report_lines.append("INFORME DE COMPARACIÓN ESTADÍSTICA DE ALGORITMOS")
                report_lines.append("=" * 60)
                report_lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append("")
                report_lines.append("CONFIGURACIÓN DEL EXPERIMENTO")
                report_lines.append("-" * 40)
                report_lines.append(f"Algoritmos: {', '.join(algo_names)}")
                report_lines.append(f"Tamaños n: {exp_data['config'].n_items_range}")
                report_lines.append(f"Contenedores k: {exp_data['config'].k_bins_range}")
                report_lines.append(f"Distribuciones: {exp_data['config'].distributions}")
                report_lines.append(f"Repeticiones: {exp_data['config'].repetitions}")
                report_lines.append(f"Nivel de significancia: α = {alpha}")
                report_lines.append("")
                report_lines.append("RESULTADOS GLOBALES")
                report_lines.append("-" * 40)
                
                summary_stats = df_all.groupby('Algoritmo')['Objetivo'].agg(['mean', 'std', 'median']).round(4)
                summary_stats = summary_stats.sort_values('mean')
                
                for i, (algo, row) in enumerate(summary_stats.iterrows(), 1):
                    report_lines.append(f"{i}. {algo}: {row['mean']:.4f} ± {row['std']:.4f} (mediana: {row['median']:.4f})")
                
                report_lines.append("")
                report_lines.append(f"🏆 MEJOR ALGORITMO GLOBAL: {summary_stats.index[0]}")
                report_lines.append("")
                report_lines.append("=" * 60)
                
                report_text = "\n".join(report_lines)
                
                st.download_button(
                    "📥 Descargar Informe (.txt)",
                    report_text,
                    file_name=f"statistical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                with st.expander("👁️ Vista previa del informe"):
                    st.code(report_text, language=None)
            
            st.markdown("---")
            st.markdown("#### 🖼️ Exportar Gráficos")
            st.info("""
            Para exportar los gráficos como imágenes:
            1. Pasa el cursor sobre el gráfico que deseas exportar
            2. Haz clic en el ícono de cámara 📷 en la esquina superior derecha
            3. La imagen se descargará automáticamente en formato PNG
            """)
        else:
            st.warning("No hay resultados para exportar.")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    <p>🧬 <strong>Comparación Estadística de Algoritmos</strong> - Diseño y Análisis de Algoritmos</p>
    <p style="font-size: 0.85rem;">
        Tests estadísticos: Mann-Whitney U, t-Student pareado, ANOVA con Tukey HSD | Tamaños del efecto: Cliff's δ, η²
    </p>
</div>
""", unsafe_allow_html=True)
