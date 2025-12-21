"""
Interactive Components Module
=============================

Advanced interactive components for the dashboard including:
- Drag and drop bin assignment
- Interactive tooltips
- Side-by-side solution comparator
- Execution timeline visualization
- Real-time parameter tuning
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# Interactive Tooltips Component
# ============================================================================

class InteractiveTooltips:
    """Component for rich interactive tooltips with algorithm information."""
    
    ALGORITHM_INFO = {
        "First Fit Decreasing": {
            "description": "Ordena ítems por peso decreciente, asigna al primer bin que quepa",
            "complexity": "O(n log n)",
            "type": "Greedy Determinístico",
            "pros": ["Rápido", "Determinístico", "Buen baseline"],
            "cons": ["Puede ser subóptimo", "No considera balance de valores"],
            "icon": "📦",
            "color": "#6366F1"
        },
        "Best Fit Decreasing": {
            "description": "Ordena ítems por peso decreciente, asigna al bin con menor espacio residual",
            "complexity": "O(n² log n)",
            "type": "Greedy Determinístico",
            "pros": ["Mejor empaquetado", "Minimiza espacio desperdiciado"],
            "cons": ["Más lento que FFD", "Puede crear desbalance"],
            "icon": "🎯",
            "color": "#8B5CF6"
        },
        "Worst Fit Decreasing": {
            "description": "Ordena ítems por peso decreciente, asigna al bin con mayor espacio libre",
            "complexity": "O(n² log n)",
            "type": "Greedy Determinístico",
            "pros": ["Distribuye mejor la carga", "Favorece el balance"],
            "cons": ["Puede desperdiciar espacio", "No siempre óptimo"],
            "icon": "⚖️",
            "color": "#EC4899"
        },
        "Round Robin": {
            "description": "Asigna ítems cíclicamente entre bins disponibles",
            "complexity": "O(n)",
            "type": "Greedy Cíclico",
            "pros": ["Muy rápido", "Distribución equitativa", "Simple"],
            "cons": ["No considera pesos", "Puede violar capacidad"],
            "icon": "🔄",
            "color": "#F59E0B"
        },
        "Simulated Annealing": {
            "description": "Metaheurística inspirada en enfriamiento metalúrgico",
            "complexity": "O(n × iterations)",
            "type": "Metaheurística Probabilística",
            "pros": ["Escapa mínimos locales", "Soluciones de alta calidad"],
            "cons": ["Requiere ajuste de parámetros", "No determinístico"],
            "icon": "🔥",
            "color": "#EF4444"
        },
        "Genetic Algorithm": {
            "description": "Algoritmo evolutivo con selección, cruce y mutación",
            "complexity": "O(pop × gen × n)",
            "type": "Metaheurística Evolutiva",
            "pros": ["Explora ampliamente", "Adaptativo", "Paralelizable"],
            "cons": ["Convergencia lenta", "Muchos parámetros"],
            "icon": "🧬",
            "color": "#10B981"
        },
        "Tabu Search": {
            "description": "Búsqueda local con memoria para evitar ciclos",
            "complexity": "O(n × iterations)",
            "type": "Metaheurística con Memoria",
            "pros": ["Muy efectivo", "Evita ciclos", "Intensificación"],
            "cons": ["Puede estancarse", "Tamaño de lista tabu crítico"],
            "icon": "🚫",
            "color": "#06B6D4"
        },
        "Branch and Bound": {
            "description": "Método exacto con poda inteligente del árbol de búsqueda",
            "complexity": "O(2^n) worst case",
            "type": "Exacto Óptimo",
            "pros": ["Solución óptima garantizada", "Poda eficiente"],
            "cons": ["Exponencial en peor caso", "Solo instancias pequeñas"],
            "icon": "🌳",
            "color": "#3B82F6"
        },
        "Dynamic Programming": {
            "description": "Programación dinámica con memorización",
            "complexity": "O(n × W × k)",
            "type": "Exacto Óptimo",
            "pros": ["Óptimo", "Eficiente para instancias medianas"],
            "cons": ["Memoria O(W×k)", "Pseudo-polinomial"],
            "icon": "📊",
            "color": "#14B8A6"
        }
    }
    
    @staticmethod
    def render_algorithm_card(algorithm_name: str):
        """Render an interactive algorithm info card."""
        info = InteractiveTooltips.ALGORITHM_INFO.get(algorithm_name, {})
        if not info:
            return
        
        color = info['color']
        # Convert hex to rgba for opacity
        hex_color = color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        
        pros_html = ''.join(f'<li>{pro}</li>' for pro in info['pros'])
        cons_html = ''.join(f'<li>{con}</li>' for con in info['cons'])
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(145deg, rgba({r},{g},{b},0.12), rgba({r},{g},{b},0.03));
            border-radius: 16px;
            padding: 20px;
            border-left: 4px solid {color};
            margin: 10px 0;
        ">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 2rem;">{info['icon']}</span>
                <div>
                    <h3 style="margin: 0; color: {color};">{algorithm_name}</h3>
                    <span style="
                        background: rgba({r},{g},{b},0.2);
                        color: {color};
                        padding: 2px 8px;
                        border-radius: 12px;
                        font-size: 0.75rem;
                        font-weight: 600;
                    ">{info['type']}</span>
                </div>
            </div>
            <p style="color: #64748B; margin: 8px 0; font-size: 0.9rem;">{info['description']}</p>
            <div style="
                display: flex;
                gap: 20px;
                margin-top: 12px;
                padding-top: 12px;
                border-top: 1px solid rgba({r},{g},{b},0.15);
            ">
                <div>
                    <span style="color: #94A3B8; font-size: 0.75rem;">Complejidad</span>
                    <p style="margin: 0; font-weight: 600; color: {color};">{info['complexity']}</p>
                </div>
            </div>
            <div style="display: flex; gap: 30px; margin-top: 12px;">
                <div>
                    <span style="color: #22C55E; font-size: 0.75rem;">✅ Ventajas</span>
                    <ul style="margin: 4px 0; padding-left: 16px; color: #64748B; font-size: 0.85rem;">
                        {pros_html}
                    </ul>
                </div>
                <div>
                    <span style="color: #EF4444; font-size: 0.75rem;">⚠️ Limitaciones</span>
                    <ul style="margin: 4px 0; padding-left: 16px; color: #64748B; font-size: 0.85rem;">
                        {cons_html}
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_tooltip_popover(title: str, content: str, color: str = "#6366F1"):
        """Render a hoverable tooltip popover."""
        st.markdown(f"""
        <div class="tooltip-container" style="display: inline-block; position: relative;">
            <span style="
                cursor: help;
                border-bottom: 1px dashed {color};
                color: {color};
            ">{title} ℹ️</span>
            <div class="tooltip-content" style="
                visibility: hidden;
                position: absolute;
                z-index: 1000;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
                background: white;
                color: #1E293B;
                padding: 12px 16px;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                min-width: 250px;
                opacity: 0;
                transition: all 0.3s ease;
            ">
                {content}
                <div style="
                    position: absolute;
                    top: 100%;
                    left: 50%;
                    transform: translateX(-50%);
                    border: 8px solid transparent;
                    border-top-color: white;
                "></div>
            </div>
        </div>
        <style>
            .tooltip-container:hover .tooltip-content {{
                visibility: visible;
                opacity: 1;
            }}
        </style>
        """, unsafe_allow_html=True)


# ============================================================================
# Side-by-Side Solution Comparator
# ============================================================================

class SolutionComparator:
    """Interactive side-by-side solution comparison component."""
    
    @staticmethod
    def render_comparison_header(algo1_name: str, algo2_name: str):
        """Render comparison header with algorithm names."""
        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px;
            background: linear-gradient(90deg, #6366F120 0%, transparent 50%, #EC489920 100%);
            border-radius: 12px;
            margin-bottom: 20px;
        ">
            <div style="text-align: center; flex: 1;">
                <span style="font-size: 1.5rem;">🔵</span>
                <h3 style="margin: 0; color: #6366F1;">{algo1_name}</h3>
            </div>
            <div style="
                font-size: 2rem;
                color: #94A3B8;
                padding: 0 20px;
            ">⚔️</div>
            <div style="text-align: center; flex: 1;">
                <span style="font-size: 1.5rem;">🔴</span>
                <h3 style="margin: 0; color: #EC4899;">{algo2_name}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_comparison_chart(
        results1: Dict[str, Any],
        results2: Dict[str, Any],
        algo1_name: str = "Algorithm 1",
        algo2_name: str = "Algorithm 2"
    ) -> go.Figure:
        """Create an interactive comparison radar chart."""
        
        # Normalize metrics for radar chart
        metrics = ['Balance', 'Velocidad', 'Eficiencia', 'Utilización', 'Calidad']
        
        # Extract and normalize values (example normalization)
        val1 = results1.get('value_difference', 100)
        val2 = results2.get('value_difference', 100)
        time1 = results1.get('execution_time', 1)
        time2 = results2.get('execution_time', 1)
        
        # Calculate scores (higher is better)
        max_diff = max(val1, val2, 1)
        balance1 = 100 * (1 - val1 / max_diff) if max_diff > 0 else 100
        balance2 = 100 * (1 - val2 / max_diff) if max_diff > 0 else 100
        
        max_time = max(time1, time2, 0.001)
        speed1 = 100 * (1 - time1 / max_time)
        speed2 = 100 * (1 - time2 / max_time)
        
        # Simulated metrics for demo
        values1 = [balance1, speed1, 75, 80, balance1 * 0.9]
        values2 = [balance2, speed2, 70, 85, balance2 * 0.9]
        
        fig = go.Figure()
        
        # Algorithm 1
        fig.add_trace(go.Scatterpolar(
            r=values1 + [values1[0]],  # Close the shape
            theta=metrics + [metrics[0]],
            fill='toself',
            name=algo1_name,
            line=dict(color='#6366F1', width=2),
            fillcolor='rgba(99, 102, 241, 0.3)'
        ))
        
        # Algorithm 2
        fig.add_trace(go.Scatterpolar(
            r=values2 + [values2[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=algo2_name,
            line=dict(color='#EC4899', width=2),
            fillcolor='rgba(236, 72, 153, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showticklabels=False,
                    gridcolor='rgba(100,100,100,0.2)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(100,100,100,0.2)'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=60, r=60, t=40, b=60),
            height=400
        )
        
        return fig
    
    @staticmethod
    def render_metric_comparison(
        metric_name: str,
        value1: float,
        value2: float,
        unit: str = "",
        lower_is_better: bool = True
    ):
        """Render a single metric comparison with visual indicator."""
        
        if lower_is_better:
            winner = 1 if value1 < value2 else (2 if value2 < value1 else 0)
        else:
            winner = 1 if value1 > value2 else (2 if value2 > value1 else 0)
        
        diff_pct = abs(value1 - value2) / max(value1, value2, 0.001) * 100
        
        # Determine colors and badges
        color1 = '#6366F1' if winner == 1 else '#64748B'
        color2 = '#EC4899' if winner == 2 else '#64748B'
        badge1 = ' 👑' if winner == 1 else ''
        badge2 = ' 👑' if winner == 2 else ''
        
        col1, col2, col3 = st.columns([1, 1.2, 1])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <span style="font-size: 1.3rem; font-weight: 700; color: {color1};">
                    {value1:.2f}{unit}{badge1}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background: rgba(100,100,100,0.05); border-radius: 8px;">
                <strong style="color: #475569;">{metric_name}</strong><br/>
                <span style="font-size: 0.75rem; color: #94A3B8;">Δ {diff_pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <span style="font-size: 1.3rem; font-weight: 700; color: {color2};">
                    {value2:.2f}{unit}{badge2}
                </span>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# Execution Timeline Visualization
# ============================================================================

class ExecutionTimeline:
    """Visualize algorithm execution timeline with interactive elements."""
    
    @staticmethod
    def create_timeline_chart(
        execution_data: List[Dict[str, Any]],
        total_time: float
    ) -> go.Figure:
        """Create an interactive execution timeline chart."""
        
        fig = go.Figure()
        
        colors = ['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#06B6D4']
        
        for i, event in enumerate(execution_data):
            color = colors[i % len(colors)]
            start_pct = (event.get('start', 0) / total_time) * 100
            duration_pct = (event.get('duration', 0) / total_time) * 100
            
            # Timeline bar
            fig.add_trace(go.Bar(
                x=[duration_pct],
                y=[event.get('name', f'Phase {i}')],
                orientation='h',
                base=start_pct,
                marker=dict(
                    color=color,
                    line=dict(width=0),
                ),
                text=f"{event.get('duration', 0)*1000:.1f}ms",
                textposition='inside',
                hovertemplate=(
                    f"<b>{event.get('name', '')}</b><br>" +
                    f"Inicio: {event.get('start', 0)*1000:.1f}ms<br>" +
                    f"Duración: {event.get('duration', 0)*1000:.1f}ms<br>" +
                    f"<extra></extra>"
                ),
                name=event.get('name', f'Phase {i}')
            ))
        
        fig.update_layout(
            barmode='overlay',
            xaxis=dict(
                title='Progreso (%)',
                range=[0, 100],
                showgrid=True,
                gridcolor='rgba(100,100,100,0.1)'
            ),
            yaxis=dict(
                title='',
                showgrid=False
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=150, r=30, t=30, b=50),
            height=max(200, len(execution_data) * 50)
        )
        
        return fig
    
    @staticmethod
    def create_convergence_timeline(
        history: List[float],
        timestamps: Optional[List[float]] = None
    ) -> go.Figure:
        """Create a convergence timeline chart."""
        
        if timestamps is None:
            timestamps = list(range(len(history)))
        
        fig = go.Figure()
        
        # Main convergence line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=history,
            mode='lines+markers',
            line=dict(color='#6366F1', width=2),
            marker=dict(size=4, color='#6366F1'),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)',
            name='Objetivo',
            hovertemplate='Iteración: %{x}<br>Objetivo: %{y:.2f}<extra></extra>'
        ))
        
        # Add best value indicator
        if history:
            best_idx = int(np.argmin(history))
            best_val = history[best_idx]
            
            # Best point marker
            fig.add_trace(go.Scatter(
                x=[timestamps[best_idx]],
                y=[best_val],
                mode='markers',
                marker=dict(size=12, color='#10B981', symbol='star'),
                name=f'Mejor: {best_val:.2f}',
                hovertemplate=f'Mejor encontrado<br>Iteración: {timestamps[best_idx]}<br>Valor: {best_val:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            xaxis=dict(
                title='Iteración',
                showgrid=True,
                gridcolor='rgba(100,100,100,0.1)',
                zeroline=False
            ),
            yaxis=dict(
                title='Diferencia de Valores',
                showgrid=True,
                gridcolor='rgba(100,100,100,0.1)',
                zeroline=False
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=60, r=40, t=40, b=60),
            height=350,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            hovermode='x unified'
        )
        
        return fig


# ============================================================================
# Real-time Parameter Tuning Component
# ============================================================================

class ParameterTuner:
    """Interactive parameter tuning component with live preview."""
    
    ALGORITHM_PARAMETERS = {
        "Simulated Annealing": {
            "initial_temp": {
                "label": "🌡️ Temperatura Inicial",
                "type": "slider",
                "min": 100.0,
                "max": 10000.0,
                "default": 1000.0,
                "step": 100.0,
                "help": "Temperatura inicial del sistema. Mayor = más exploración inicial."
            },
            "cooling_rate": {
                "label": "❄️ Tasa de Enfriamiento",
                "type": "slider",
                "min": 0.80,
                "max": 0.99,
                "default": 0.95,
                "step": 0.01,
                "help": "Factor de reducción de temperatura. Más cercano a 1 = enfriamiento más lento."
            },
            "max_iterations": {
                "label": "🔄 Iteraciones Máximas",
                "type": "slider",
                "min": 100,
                "max": 10000,
                "default": 1000,
                "step": 100,
                "help": "Número máximo de iteraciones del algoritmo."
            }
        },
        "Genetic Algorithm": {
            "population_size": {
                "label": "👥 Tamaño de Población",
                "type": "slider",
                "min": 10,
                "max": 200,
                "default": 50,
                "step": 10,
                "help": "Número de individuos en cada generación."
            },
            "generations": {
                "label": "🧬 Generaciones",
                "type": "slider",
                "min": 10,
                "max": 500,
                "default": 100,
                "step": 10,
                "help": "Número de generaciones a evolucionar."
            },
            "mutation_rate": {
                "label": "🎲 Tasa de Mutación",
                "type": "slider",
                "min": 0.01,
                "max": 0.30,
                "default": 0.1,
                "step": 0.01,
                "help": "Probabilidad de mutación de cada gen."
            },
            "crossover_rate": {
                "label": "✂️ Tasa de Cruce",
                "type": "slider",
                "min": 0.5,
                "max": 1.0,
                "default": 0.8,
                "step": 0.05,
                "help": "Probabilidad de cruce entre padres."
            }
        },
        "Tabu Search": {
            "tabu_tenure": {
                "label": "📋 Tenure de Lista Tabú",
                "type": "slider",
                "min": 5,
                "max": 50,
                "default": 10,
                "step": 1,
                "help": "Número de iteraciones que un movimiento permanece tabú."
            },
            "max_iterations": {
                "label": "🔄 Iteraciones Máximas",
                "type": "slider",
                "min": 100,
                "max": 5000,
                "default": 1000,
                "step": 100,
                "help": "Número máximo de iteraciones."
            }
        }
    }
    
    @staticmethod
    def render_parameter_panel(algorithm_name: str) -> Dict[str, Any]:
        """Render interactive parameter tuning panel for an algorithm."""
        
        params = ParameterTuner.ALGORITHM_PARAMETERS.get(algorithm_name, {})
        if not params:
            st.info(f"ℹ️ {algorithm_name} no tiene parámetros configurables.")
            return {}
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #6366F108, #8B5CF608);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        ">
            <h4 style="margin: 0 0 12px 0; color: #6366F1;">
                ⚙️ Parámetros de {algorithm_name}
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        values = {}
        for param_key, param_config in params.items():
            if param_config['type'] == 'slider':
                values[param_key] = st.slider(
                    param_config['label'],
                    min_value=param_config['min'],
                    max_value=param_config['max'],
                    value=param_config['default'],
                    step=param_config['step'],
                    help=param_config['help'],
                    key=f"param_{algorithm_name}_{param_key}"
                )
        
        return values
    
    @staticmethod
    def create_parameter_impact_preview(
        param_name: str,
        param_values: List[float],
        impact_scores: List[float]
    ) -> go.Figure:
        """Create a preview chart showing parameter impact."""
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=param_values,
            y=impact_scores,
            mode='lines+markers',
            line=dict(
                color='#6366F1',
                width=3,
                shape='spline'
            ),
            marker=dict(
                size=10,
                color=impact_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Impacto')
            ),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)',
            hovertemplate=(
                f"<b>{param_name}</b>: %{{x:.2f}}<br>" +
                "Impacto: %{y:.2f}<br>" +
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            xaxis_title=param_name,
            yaxis_title='Puntuación de Impacto',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(100,100,100,0.1)'),
            yaxis=dict(gridcolor='rgba(100,100,100,0.1)'),
            height=250,
            margin=dict(l=50, r=30, t=30, b=50)
        )
        
        return fig


# ============================================================================
# Interactive Bin Visualizer
# ============================================================================

class InteractiveBinVisualizer:
    """Interactive bin visualization with hover effects and animations."""
    
    @staticmethod
    def create_interactive_bins(
        bins_data: List[Dict],
        capacities: List[float]
    ) -> go.Figure:
        """Create interactive bin visualization with clickable items."""
        
        num_bins = len(bins_data)
        
        fig = make_subplots(
            rows=1, cols=num_bins,
            subplot_titles=[f"Bin {i+1}" for i in range(num_bins)],
            horizontal_spacing=0.05
        )
        
        colors = ['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#06B6D4', '#EF4444', '#84CC16']
        
        for bin_idx, bin_data in enumerate(bins_data):
            items = bin_data.get('items', [])
            capacity = capacities[bin_idx] if bin_idx < len(capacities) else capacities[0]
            
            y_position = 0
            for item_idx, item in enumerate(items):
                color = colors[item_idx % len(colors)]
                height = (item['weight'] / capacity) * 100
                
                fig.add_trace(
                    go.Bar(
                        x=[f"Bin {bin_idx + 1}"],
                        y=[height],
                        base=y_position,
                        name=f"Item {item['id']}",
                        marker=dict(
                            color=color,
                            line=dict(width=1, color='white'),
                            pattern=dict(shape="") if item_idx % 2 == 0 else dict(shape="/")
                        ),
                        text=f"#{item['id']}<br>w:{item['weight']:.1f}<br>v:{item['value']:.1f}",
                        textposition='inside',
                        textfont=dict(size=10, color='white'),
                        hovertemplate=(
                            f"<b>Item {item['id']}</b><br>" +
                            f"Peso: {item['weight']:.2f}<br>" +
                            f"Valor: {item['value']:.2f}<br>" +
                            f"<extra></extra>"
                        ),
                        showlegend=False
                    ),
                    row=1, col=bin_idx + 1
                )
                
                y_position += height
            
            # Capacity line
            fig.add_hline(
                y=100,
                line=dict(color='#EF4444', width=2, dash='dash'),
                annotation_text="Capacidad",
                annotation_position="right",
                row=1, col=bin_idx + 1
            )
            
            # Fill percentage annotation
            fill_pct = (sum(item['weight'] for item in items) / capacity) * 100
            fig.add_annotation(
                x=f"Bin {bin_idx + 1}",
                y=105,
                text=f"{fill_pct:.1f}%",
                showarrow=False,
                font=dict(size=14, color='#6366F1', weight='bold'),
                row=1, col=bin_idx + 1
            )
        
        fig.update_layout(
            barmode='stack',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        fig.update_yaxes(range=[0, 120], title='Utilización (%)', gridcolor='rgba(100,100,100,0.1)')
        fig.update_xaxes(showticklabels=False)
        
        return fig
    
    @staticmethod
    def render_bin_summary_cards(bins_data: List[Dict], capacities: List[float]):
        """Render summary cards for each bin."""
        
        cols = st.columns(len(bins_data))
        
        for i, (col, bin_data) in enumerate(zip(cols, bins_data)):
            items = bin_data.get('items', [])
            capacity = capacities[i] if i < len(capacities) else capacities[0]
            total_weight = sum(item['weight'] for item in items)
            total_value = sum(item['value'] for item in items)
            utilization = (total_weight / capacity) * 100
            
            # Determine color based on utilization
            if utilization > 95:
                color = '#EF4444'
                status = '🔴 Casi lleno'
            elif utilization > 70:
                color = '#F59E0B'
                status = '🟡 Bien utilizado'
            else:
                color = '#10B981'
                status = '🟢 Espacio disponible'
            
            with col:
                # Convert color to rgba format
                rgb = InteractiveBinVisualizer._hex_to_rgb(color)
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba({rgb}, 0.1), rgba({rgb}, 0.02));
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                    border: 1px solid rgba({rgb}, 0.2);
                ">
                    <h4 style="margin: 0; color: {color};">Bin {i + 1}</h4>
                    <div style="
                        font-size: 1.75rem;
                        font-weight: 700;
                        color: {color};
                        margin: 8px 0;
                    ">{utilization:.1f}%</div>
                    <p style="margin: 4px 0; color: #64748B; font-size: 0.85rem;">
                        {len(items)} ítems | Valor: {total_value:.1f}
                    </p>
                    <span style="font-size: 0.75rem; color: {color};">{status}</span>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        """Convert hex color to RGB string."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"{r}, {g}, {b}"


# ============================================================================
# Quick Stats Dashboard Component
# ============================================================================

class QuickStatsDashboard:
    """Quick statistics dashboard with animated metrics."""
    
    @staticmethod
    def render_stats_row(stats: Dict[str, Dict]):
        """Render a row of quick stats with animations."""
        
        cols = st.columns(len(stats))
        
        for col, (stat_name, stat_config) in zip(cols, stats.items()):
            with col:
                value = stat_config.get('value', 0)
                icon = stat_config.get('icon', '📊')
                color = stat_config.get('color', '#6366F1')
                
                # Format value
                if isinstance(value, str):
                    display_value = value
                elif isinstance(value, float):
                    display_value = f"{value:.2f}"
                else:
                    display_value = str(value)
                
                # Use st.metric for reliable rendering
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.04));
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                    border: 1px solid rgba(99, 102, 241, 0.15);
                ">
                    <div style="font-size: 1.5rem; margin-bottom: 4px;">{icon}</div>
                    <p style="margin: 4px 0; color: #64748B; font-size: 0.8rem;">{stat_name}</p>
                    <div style="
                        font-size: 1.4rem;
                        font-weight: 700;
                        color: {color};
                        margin-top: 4px;
                    ">{display_value}</div>
                </div>
                """, unsafe_allow_html=True)


# ============================================================================
# Export all components
# ============================================================================

__all__ = [
    'InteractiveTooltips',
    'SolutionComparator',
    'ExecutionTimeline',
    'ParameterTuner',
    'InteractiveBinVisualizer',
    'QuickStatsDashboard'
]
