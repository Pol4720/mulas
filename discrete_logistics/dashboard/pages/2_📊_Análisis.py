"""
📊 Análisis - Multi-Bin Packing Solver
======================================

Página de análisis detallado de resultados con visualizaciones
interactivas y comparación de algoritmos.
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="📊 Análisis | Multi-Bin Packing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import shared utilities
from shared import (
    init_session_state,
    apply_custom_styles,
    render_sidebar_info,
    VisualizationPanel,
    QuickStatsDashboard,
    SolutionComparator,
    ExecutionTimeline,
    InteractiveBinVisualizer,
    InteractiveTooltips
)

# Initialize
init_session_state()
apply_custom_styles()
render_sidebar_info()

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
    ">📊 Análisis de Resultados</h1>
    <p style="color: #64748B;">Explora y compara el rendimiento de los algoritmos</p>
</div>
""", unsafe_allow_html=True)

# Check if we have results
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
    st.stop()

results = st.session_state['results']
viz_panel = VisualizationPanel()

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
    
    perf_data = []
    sorted_objs = sorted(objectives) if objectives else []
    
    for algo, result in results.items():
        obj = result.get('objective', float('inf'))
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
    
    st.markdown("")
    viz_panel.render_performance_radar(results)

with tab2:
    st.markdown("### 🔄 Análisis de Convergencia")
    
    if st.session_state.get('convergence_history'):
        history_data = st.session_state['convergence_history']
        
        algo_selected = st.selectbox(
            "Seleccionar algoritmo:",
            options=list(history_data.keys()),
            key="convergence_algo_select"
        )
        
        if algo_selected and algo_selected in history_data:
            history = history_data[algo_selected]
            
            fig = ExecutionTimeline.create_convergence_timeline(history)
            st.plotly_chart(fig, use_container_width=True)
        
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
            SolutionComparator.render_comparison_header(algo1, algo2)
            
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
            
            problem = st.session_state.get('current_problem')
            capacities = problem.bin_capacities if problem else [100.0] * len(bins_data)
            
            InteractiveBinVisualizer.render_bin_summary_cards(bins_data, capacities)
            
            st.markdown("")
            
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
    
    selected_algo = st.selectbox(
        "Seleccionar algoritmo:",
        options=list(InteractiveTooltips.ALGORITHM_INFO.keys()),
        key="info_algo_select"
    )
    
    if selected_algo:
        InteractiveTooltips.render_algorithm_card(selected_algo)
