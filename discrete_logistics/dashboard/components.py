"""
Dashboard Components Module
==========================

Reusable UI components for the Streamlit dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import json

import sys
from pathlib import Path

# Add the discrete_logistics package to path
_pkg_root = Path(__file__).parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))
if str(_pkg_root.parent) not in sys.path:
    sys.path.insert(0, str(_pkg_root.parent))

from discrete_logistics.core.problem import Problem, Solution, Item
from discrete_logistics.core.instance_generator import InstanceGenerator
from discrete_logistics.algorithms import AlgorithmRegistry


@dataclass
class DashboardConfig:
    """Configuration for dashboard appearance and behavior."""
    theme: str = "dark"
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    success_color: str = "#2ca02c"
    warning_color: str = "#d62728"
    chart_height: int = 400
    animation_speed: float = 0.5
    

class ThemeManager:
    """Manages dashboard theme (dark/light mode)."""
    
    DARK_THEME = {
        'bg_color': '#0e1117',
        'card_bg': '#1e1e1e',
        'text_color': '#ffffff',
        'secondary_text': '#b0b0b0',
        'accent_color': '#1f77b4',
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'error': '#d62728',
        'plotly_template': 'plotly_dark'
    }
    
    LIGHT_THEME = {
        'bg_color': '#ffffff',
        'card_bg': '#f5f5f5',
        'text_color': '#000000',
        'secondary_text': '#666666',
        'accent_color': '#1f77b4',
        'success': '#28a745',
        'warning': '#ffc107',
        'error': '#dc3545',
        'plotly_template': 'plotly_white'
    }
    
    @classmethod
    def get_theme(cls, theme_name: str) -> Dict[str, str]:
        """Get theme configuration by name."""
        if theme_name.lower() == 'dark':
            return cls.DARK_THEME
        return cls.LIGHT_THEME
    
    @classmethod
    def apply_theme(cls, theme_name: str):
        """Apply CSS styling based on theme."""
        theme = cls.get_theme(theme_name)
        st.markdown(f"""
        <style>
            .stApp {{
                background-color: {theme['bg_color']};
            }}
            .metric-card {{
                background-color: {theme['card_bg']};
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            }}
            .success-text {{
                color: {theme['success']};
            }}
            .warning-text {{
                color: {theme['warning']};
            }}
            .error-text {{
                color: {theme['error']};
            }}
            .section-header {{
                color: {theme['accent_color']};
                font-size: 1.5em;
                font-weight: bold;
                margin-bottom: 20px;
            }}
        </style>
        """, unsafe_allow_html=True)


class ProblemConfigurator:
    """
    UI component for configuring bin packing problem instances.
    
    Provides controls for:
    - Number of items and bins
    - Capacity constraints
    - Weight and value distributions
    - Instance generation methods
    """
    
    def __init__(self, theme: str = "dark"):
        self.theme = ThemeManager.get_theme(theme)
        self.generator = InstanceGenerator()
    
    def render(self) -> Optional[Problem]:
        """Render the problem configuration panel."""
        st.markdown("### 📦 Configuración del Problema")
        
        with st.expander("Parámetros Básicos", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                n_items = st.slider(
                    "Número de Ítems (n)",
                    min_value=3,
                    max_value=100,
                    value=20,
                    help="Cantidad total de ítems a empacar"
                )
            
            with col2:
                n_bins = st.slider(
                    "Número de Contenedores (k)",
                    min_value=2,
                    max_value=20,
                    value=4,
                    help="Cantidad de contenedores disponibles"
                )
            
            # Capacity configuration mode
            capacity_mode = st.radio(
                "Configuración de Capacidad",
                options=['Uniforme', 'Individual', 'Variable'],
                horizontal=True,
                help="Uniforme: igual para todos, Individual: configurar cada uno, Variable: variación aleatoria"
            )
            
            if capacity_mode == 'Uniforme':
                base_capacity = st.number_input(
                    "Capacidad del Contenedor (C)",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    help="Capacidad máxima de peso para todos los contenedores"
                )
                bin_capacities = [float(base_capacity)] * n_bins
            elif capacity_mode == 'Individual':
                st.markdown("**Establecer capacidad para cada contenedor:**")
                bin_capacities = []
                cols = st.columns(min(n_bins, 5))
                for i in range(n_bins):
                    with cols[i % 5]:
                        cap = st.number_input(
                            f"Cont. {i+1}",
                            min_value=10,
                            max_value=1000,
                            value=100,
                            key=f"cap_{i}"
                        )
                        bin_capacities.append(float(cap))
            else:  # Variable
                col1, col2 = st.columns(2)
                with col1:
                    base_capacity = st.number_input(
                        "Capacidad Base",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        help="Capacidad promedio"
                    )
                with col2:
                    capacity_variation = st.slider(
                        "Variación %",
                        min_value=0,
                        max_value=50,
                        value=20,
                        help="Porcentaje de variación respecto a la base"
                    )
                # Generate variable capacities
                var = capacity_variation / 100.0
                bin_capacities = [
                    float(base_capacity * (1 + np.random.uniform(-var, var)))
                    for _ in range(n_bins)
                ]
                st.caption(f"Capacidades generadas: {[f'{c:.1f}' for c in bin_capacities]}")
        
        with st.expander("Configuración de Distribución", expanded=False):
            distribution = st.selectbox(
                "Distribución de Peso/Valor",
                options=['uniform', 'normal', 'bimodal', 'clustered'],
                format_func=lambda x: {'uniform': 'Uniforme', 'normal': 'Normal', 'bimodal': 'Bimodal', 'clustered': 'Agrupada'}[x],
                help="Distribución estadística para generar pesos y valores de ítems"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                weight_range = st.slider(
                    "Rango de Peso",
                    min_value=1,
                    max_value=100,
                    value=(5, 30),
                    help="Peso mínimo y máximo de los ítems"
                )
            
            with col2:
                value_range = st.slider(
                    "Rango de Valor", 
                    min_value=1,
                    max_value=100,
                    value=(10, 50),
                    help="Valor mínimo y máximo de los ítems"
                )
            
            correlation = st.slider(
                "Correlación Peso-Valor",
                min_value=-1.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Correlación entre pesos y valores de los ítems"
            )
        
        with st.expander("Opciones Avanzadas", expanded=False):
            seed = st.number_input(
                "Semilla Aleatoria (0 para aleatorio)",
                min_value=0,
                max_value=99999,
                value=42,
                help="Semilla para generación reproducible de instancias"
            )
            
            preset = st.selectbox(
                "Cargar Instancia Predefinida",
                options=['Personalizada', 'Pequeña (Fácil)', 'Mediana', 'Grande (Difícil)', 'Benchmark A', 'Benchmark B'],
                help="Cargar una instancia de prueba predefinida"
            )
            
            if preset != 'Personalizada':
                n_items, n_bins, bin_capacities = self._get_preset_params(preset)
        
        # Generate problem button
        if st.button("🎲 Generar Instancia", type="primary", use_container_width=True):
            with st.spinner("Generando instancia del problema..."):
                problem = self._generate_problem(
                    n_items=n_items,
                    n_bins=n_bins,
                    bin_capacities=bin_capacities,
                    distribution=distribution,
                    weight_range=weight_range,
                    value_range=value_range,
                    correlation=correlation,
                    seed=seed if seed != 0 else None
                )
                
                # Store in session state
                st.session_state['current_problem'] = problem
                st.success(f"✅ Instancia generada con {n_items} ítems y {n_bins} contenedores")
                
                return problem
        
        # Return existing problem if available
        return st.session_state.get('current_problem')
    
    def _get_preset_params(self, preset: str) -> Tuple[int, int, List[float]]:
        """Get parameters for preset instances."""
        presets = {
            'Pequeña (Fácil)': (10, 3, [50.0, 50.0, 50.0]),
            'Mediana': (30, 5, [100.0, 100.0, 100.0, 100.0, 100.0]),
            'Grande (Difícil)': (50, 8, [150.0] * 8),
            'Benchmark A': (25, 4, [80.0, 85.0, 75.0, 90.0]),
            'Benchmark B': (40, 6, [120.0, 110.0, 130.0, 115.0, 125.0, 120.0]),
        }
        return presets.get(preset, (20, 4, [100.0] * 4))
    
    def _generate_problem(self, n_items: int, n_bins: int, bin_capacities: List[float],
                         distribution: str, weight_range: Tuple[int, int],
                         value_range: Tuple[int, int], correlation: float,
                         seed: Optional[int]) -> Problem:
        """Generate a problem instance with given parameters."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate items based on distribution
        items = []
        for i in range(n_items):
            if distribution == 'uniform':
                weight = np.random.uniform(weight_range[0], weight_range[1])
                base_value = np.random.uniform(value_range[0], value_range[1])
            elif distribution == 'normal':
                weight = np.clip(
                    np.random.normal((weight_range[0] + weight_range[1]) / 2, 
                                    (weight_range[1] - weight_range[0]) / 4),
                    weight_range[0], weight_range[1]
                )
                base_value = np.random.normal((value_range[0] + value_range[1]) / 2,
                                             (value_range[1] - value_range[0]) / 4)
            elif distribution == 'bimodal':
                if np.random.random() < 0.5:
                    weight = np.random.normal(weight_range[0] + 5, 3)
                else:
                    weight = np.random.normal(weight_range[1] - 5, 3)
                weight = np.clip(weight, weight_range[0], weight_range[1])
                base_value = np.random.uniform(value_range[0], value_range[1])
            else:  # clustered
                cluster = np.random.choice(3)
                centers = [weight_range[0], (weight_range[0] + weight_range[1]) / 2, weight_range[1]]
                weight = np.clip(np.random.normal(centers[cluster], 5), 
                               weight_range[0], weight_range[1])
                base_value = np.random.uniform(value_range[0], value_range[1])
            
            # Apply correlation
            if correlation != 0:
                value = base_value + correlation * (weight - np.mean(weight_range)) * 0.5
                value = np.clip(value, value_range[0], value_range[1])
            else:
                value = base_value
            
            items.append(Item(
                id=f"item_{i+1}",
                weight=round(weight, 2),
                value=round(value, 2)
            ))
        
        return Problem(
            items=items,
            num_bins=n_bins,
            bin_capacities=bin_capacities,
            name=f"instance_{n_items}_{n_bins}"
        )
    
    def render_problem_summary(self, problem: Problem):
        """Display a summary of the current problem instance."""
        st.markdown("### 📊 Resumen de la Instancia")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Ítems", len(problem.items))
        with col2:
            st.metric("Contenedores", problem.num_bins)
        with col3:
            st.metric("Capacidad Total", f"{sum(problem.bin_capacities):.1f}")
        with col4:
            total_weight = sum(item.weight for item in problem.items)
            st.metric("Peso Total", f"{total_weight:.1f}")
        
        # Show individual bin capacities
        st.markdown("**Capacidades de los Contenedores:**")
        cap_cols = st.columns(min(problem.num_bins, 6))
        for i, cap in enumerate(problem.bin_capacities):
            with cap_cols[i % 6]:
                st.caption(f"Cont. {i+1}: {cap:.1f}")
        
        # Items distribution chart
        fig = go.Figure()
        
        weights = [item.weight for item in problem.items]
        values = [item.value for item in problem.items]
        
        fig.add_trace(go.Scatter(
            x=weights,
            y=values,
            mode='markers',
            marker=dict(
                size=10,
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Valor")
            ),
            text=[item.id for item in problem.items],
            hovertemplate='<b>%{text}</b><br>Peso: %{x:.2f}<br>Valor: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Distribución de los Ítems",
            xaxis_title="Peso",
            yaxis_title="Valor",
            template=self.theme['plotly_template'],
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)


class AlgorithmSelector:
    """
    UI component for selecting and configuring algorithms.
    """
    
    ALGORITHM_INFO = {
        'FirstFitDecreasing': {
            'name': 'First Fit Decreasing (FFD)',
            'category': 'Voraz',
            'complexity': 'O(n log n)',
            'description': 'Ordena ítems por peso y asigna cada uno al primer contenedor con espacio.',
            'params': {}
        },
        'BestFitDecreasing': {
            'name': 'Best Fit Decreasing (BFD)',
            'category': 'Voraz',
            'complexity': 'O(n² log n)',
            'description': 'Asigna cada ítem al contenedor con menor capacidad restante.',
            'params': {}
        },
        'WorstFitDecreasing': {
            'name': 'Worst Fit Decreasing (WFD)',
            'category': 'Voraz', 
            'complexity': 'O(n log n)',
            'description': 'Asigna ítems a contenedores con mayor capacidad restante para balance.',
            'params': {}
        },
        'RoundRobinGreedy': {
            'name': 'Round Robin Voraz',
            'category': 'Voraz',
            'complexity': 'O(n log n)',
            'description': 'Distribuye ítems uniformemente entre contenedores en forma circular.',
            'params': {}
        },
        'LargestDifferenceFirst': {
            'name': 'Mayor Diferencia Primero',
            'category': 'Voraz',
            'complexity': 'O(n log n)',
            'description': 'Prioriza reducir la mayor diferencia de valor entre contenedores.',
            'params': {}
        },
        'SimulatedAnnealing': {
            'name': 'Recocido Simulado',
            'category': 'Metaheurística',
            'complexity': 'O(iteraciones × n)',
            'description': 'Optimización probabilística inspirada en el recocido metalúrgico.',
            'params': {
                'initial_temp': (100.0, 1.0, 1000.0, 'Temperatura inicial'),
                'cooling_rate': (0.995, 0.9, 0.999, 'Tasa de enfriamiento'),
                'max_iterations': (10000, 100, 100000, 'Iteraciones máximas')
            }
        },
        'GeneticAlgorithm': {
            'name': 'Algoritmo Genético',
            'category': 'Metaheurística',
            'complexity': 'O(generaciones × pob × n)',
            'description': 'Algoritmo evolutivo usando selección, cruce y mutación.',
            'params': {
                'population_size': (50, 10, 200, 'Tamaño de población'),
                'generations': (100, 10, 500, 'Número de generaciones'),
                'mutation_rate': (0.1, 0.01, 0.5, 'Probabilidad de mutación')
            }
        },
        'TabuSearch': {
            'name': 'Búsqueda Tabú',
            'category': 'Metaheurística',
            'complexity': 'O(iteraciones × vecinos)',
            'description': 'Búsqueda local con memoria para evitar ciclos.',
            'params': {
                'tabu_tenure': (10, 1, 50, 'Tamaño lista tabú'),
                'max_iterations': (1000, 100, 10000, 'Iteraciones máximas')
            }
        },
        'BranchAndBound': {
            'name': 'Branch and Bound',
            'category': 'Exacto',
            'complexity': 'O(k^n) peor caso',
            'description': 'Enumeración sistemática con poda inteligente.',
            'params': {
                'time_limit': (60.0, 1.0, 300.0, 'Límite de tiempo (segundos)')
            }
        },
        'DynamicProgramming': {
            'name': 'Programación Dinámica',
            'category': 'Exacto',
            'complexity': 'O(n × C^k)',
            'description': 'Solución óptima para instancias pequeñas usando memoización.',
            'params': {}
        }
    }
    
    def __init__(self, theme: str = "dark"):
        self.theme = ThemeManager.get_theme(theme)
        self.registry = AlgorithmRegistry()
    
    def render(self) -> List[Tuple[str, Dict]]:
        """Render algorithm selection panel."""
        st.markdown("### ⚙️ Selección de Algoritmos")
        
        # Category filter
        categories = ['Todos', 'Voraz', 'Metaheurística', 'Exacto']
        selected_category = st.radio(
            "Filtrar por Categoría",
            options=categories,
            horizontal=True
        )
        
        # Algorithm selection
        available_algorithms = self._filter_algorithms(selected_category)
        
        # Determine default selection based on available algorithms
        default_selection = []
        if 'FirstFitDecreasing' in available_algorithms:
            default_selection.append('FirstFitDecreasing')
        elif available_algorithms:
            default_selection.append(list(available_algorithms.keys())[0])
        
        selected_algorithms = st.multiselect(
            "Seleccionar Algoritmos a Ejecutar",
            options=list(available_algorithms.keys()),
            default=default_selection if default_selection else None,
            format_func=lambda x: f"{available_algorithms[x]['name']} ({available_algorithms[x]['category']})"
        )
        
        # Algorithm details and parameters
        algorithm_configs = []
        
        for algo_name in selected_algorithms:
            info = self.ALGORITHM_INFO.get(algo_name, {})
            
            with st.expander(f"🔧 {info.get('name', algo_name)}", expanded=False):
                st.markdown(f"**Categoría:** {info.get('category', 'Desconocida')}")
                st.markdown(f"**Complejidad:** {info.get('complexity', 'Desconocida')}")
                st.markdown(f"**Descripción:** {info.get('description', 'Sin descripción')}")
                
                # Parameter configuration
                params = {}
                if info.get('params'):
                    st.markdown("**Parámetros:**")
                    for param_name, (default, min_val, max_val, desc) in info['params'].items():
                        if isinstance(default, float):
                            params[param_name] = st.slider(
                                desc, min_value=min_val, max_value=max_val,
                                value=default, key=f"{algo_name}_{param_name}"
                            )
                        else:
                            params[param_name] = st.slider(
                                desc, min_value=int(min_val), max_value=int(max_val),
                                value=int(default), key=f"{algo_name}_{param_name}"
                            )
                
                algorithm_configs.append((algo_name, params))
        
        return algorithm_configs
    
    def _filter_algorithms(self, category: str) -> Dict[str, Dict]:
        """Filter algorithms by category."""
        if category == 'Todos':
            return self.ALGORITHM_INFO
        return {
            name: info for name, info in self.ALGORITHM_INFO.items()
            if info['category'] == category
        }


class ResultsDisplay:
    """
    UI component for displaying algorithm results.
    """
    
    def __init__(self, theme: str = "dark"):
        self.theme = ThemeManager.get_theme(theme)
    
    def render_results(self, results: Dict[str, Any]):
        """Display results from algorithm execution."""
        st.markdown("### 📈 Results")
        
        if not results:
            st.info("No results to display. Run algorithms first.")
            return
        
        # Summary metrics
        self._render_summary_metrics(results)
        
        # Detailed comparison
        self._render_comparison_table(results)
        
        # Best solution visualization
        self._render_best_solution(results)
    
    def _render_summary_metrics(self, results: Dict[str, Any]):
        """Render summary metrics cards."""
        col1, col2, col3 = st.columns(3)
        
        best_algo = min(results.keys(), key=lambda x: results[x].get('objective', float('inf')))
        best_result = results[best_algo]
        
        with col1:
            st.metric(
                "Best Algorithm",
                best_algo,
                help="Algorithm with lowest objective value"
            )
        
        with col2:
            st.metric(
                "Best Objective",
                f"{best_result.get('objective', 0):.4f}",
                help="Minimum value difference achieved"
            )
        
        with col3:
            st.metric(
                "Execution Time",
                f"{best_result.get('time', 0):.3f}s",
                help="Time to find best solution"
            )
    
    def _render_comparison_table(self, results: Dict[str, Any]):
        """Render comparison table of all algorithms."""
        st.markdown("#### Algorithm Comparison")
        
        data = []
        for algo_name, result in results.items():
            data.append({
                'Algorithm': algo_name,
                'Objective': result.get('objective', '-'),
                'Balance Score': result.get('balance_score', '-'),
                'Time (s)': result.get('time', '-'),
                'Feasible': '✅' if result.get('feasible', False) else '❌'
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Objective')
        
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _render_best_solution(self, results: Dict[str, Any]):
        """Visualize the best solution found."""
        st.markdown("#### Best Solution Visualization")
        
        best_algo = min(results.keys(), key=lambda x: results[x].get('objective', float('inf')))
        solution = results[best_algo].get('solution')
        
        if solution is None:
            st.warning("No solution available for visualization.")
            return
        
        # Create bin visualization
        fig = self._create_bin_visualization(solution)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_bin_visualization(self, solution: Solution) -> go.Figure:
        """Create a visual representation of bin contents."""
        fig = make_subplots(
            rows=1, cols=len(solution.bins),
            subplot_titles=[f"Bin {i+1}" for i in range(len(solution.bins))]
        )
        
        colors = px.colors.qualitative.Set3
        
        for bin_idx, bin_obj in enumerate(solution.bins):
            y_offset = 0
            for item_idx, item in enumerate(bin_obj.items):
                fig.add_trace(
                    go.Bar(
                        x=[1],
                        y=[item.weight],
                        base=[y_offset],
                        name=item.id,
                        marker_color=colors[item_idx % len(colors)],
                        text=f"{item.id}<br>w={item.weight:.1f}<br>v={item.value:.1f}",
                        textposition='inside',
                        hoverinfo='text',
                        showlegend=False
                    ),
                    row=1, col=bin_idx + 1
                )
                y_offset += item.weight
            
            # Add capacity line
            fig.add_hline(
                y=solution.bins[0].capacity if hasattr(solution.bins[0], 'capacity') else 100,
                line_dash="dash",
                line_color="red",
                row=1, col=bin_idx + 1
            )
        
        fig.update_layout(
            title="Bin Contents (Weight Distribution)",
            template=self.theme['plotly_template'],
            height=400,
            showlegend=False,
            barmode='stack'
        )
        
        return fig


class VisualizationPanel:
    """
    UI component for various visualizations.
    """
    
    def __init__(self, theme: str = "dark"):
        self.theme = ThemeManager.get_theme(theme)
    
    def render_convergence_plot(self, history: Dict[str, List[float]]):
        """Render convergence plot for iterative algorithms."""
        st.markdown("#### Convergence Analysis")
        
        fig = go.Figure()
        
        for algo_name, values in history.items():
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode='lines',
                name=algo_name
            ))
        
        fig.update_layout(
            title="Algorithm Convergence",
            xaxis_title="Iteration",
            yaxis_title="Objective Value",
            template=self.theme['plotly_template'],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_radar(self, results: Dict[str, Any]):
        """Render radar chart comparing algorithm performance."""
        st.markdown("#### Performance Comparison")
        
        metrics = ['Speed', 'Quality', 'Balance', 'Feasibility', 'Stability']
        
        fig = go.Figure()
        
        for algo_name, result in results.items():
            # Normalize metrics to 0-1 scale
            values = [
                1 - min(result.get('time', 1) / 10, 1),  # Speed (inverse of time)
                1 - min(result.get('objective', 1) / 100, 1),  # Quality
                result.get('balance_score', 0.5),  # Balance
                1.0 if result.get('feasible', False) else 0.0,  # Feasibility
                result.get('stability', 0.8)  # Stability
            ]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=algo_name
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            template=self.theme['plotly_template'],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_bin_heatmap(self, solution: Solution):
        """Render heatmap of bin utilization."""
        st.markdown("#### Bin Utilization Heatmap")
        
        # Create utilization matrix
        bin_data = []
        for bin_obj in solution.bins:
            weight_util = sum(item.weight for item in bin_obj.items) / bin_obj.capacity
            value_sum = sum(item.value for item in bin_obj.items)
            item_count = len(bin_obj.items)
            bin_data.append([weight_util, value_sum / 100, item_count / 10])
        
        fig = go.Figure(data=go.Heatmap(
            z=bin_data,
            x=['Weight Util.', 'Value Sum', 'Item Count'],
            y=[f"Bin {i+1}" for i in range(len(solution.bins))],
            colorscale='RdYlGn',
            showscale=True
        ))
        
        fig.update_layout(
            title="Bin Utilization Metrics",
            template=self.theme['plotly_template'],
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)


class ExportManager:
    """
    Handles export functionality for results and solutions.
    """
    
    @staticmethod
    def export_to_json(data: Any, filename: str) -> str:
        """Export data to JSON format."""
        return json.dumps(data, indent=2, default=str)
    
    @staticmethod
    def export_to_csv(results: Dict[str, Any]) -> str:
        """Export results to CSV format."""
        rows = []
        for algo_name, result in results.items():
            row = {
                'algorithm': algo_name,
                'objective': result.get('objective', ''),
                'time': result.get('time', ''),
                'feasible': result.get('feasible', ''),
                'balance_score': result.get('balance_score', '')
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
    
    @staticmethod
    def render_export_buttons(results: Dict[str, Any]):
        """Render export buttons in the UI."""
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            json_data = ExportManager.export_to_json(results, "results")
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name="results.json",
                mime="application/json"
            )
        
        with col2:
            csv_data = ExportManager.export_to_csv(results)
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv"
            )
        
        with col3:
            st.button("📋 Copy to Clipboard", disabled=True, help="Coming soon")
