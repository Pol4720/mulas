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
        st.markdown("### 📦 Problem Configuration")
        
        with st.expander("Basic Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_items = st.slider(
                    "Number of Items (n)",
                    min_value=3,
                    max_value=100,
                    value=20,
                    help="Total number of items to pack"
                )
            
            with col2:
                n_bins = st.slider(
                    "Number of Bins (k)",
                    min_value=2,
                    max_value=20,
                    value=4,
                    help="Number of available bins"
                )
            
            with col3:
                capacity = st.number_input(
                    "Bin Capacity (C)",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    help="Maximum weight capacity per bin"
                )
        
        with st.expander("Distribution Settings", expanded=False):
            distribution = st.selectbox(
                "Weight/Value Distribution",
                options=['uniform', 'normal', 'bimodal', 'clustered'],
                help="Statistical distribution for generating item weights and values"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                weight_range = st.slider(
                    "Weight Range",
                    min_value=1,
                    max_value=100,
                    value=(5, 30),
                    help="Min and max item weights"
                )
            
            with col2:
                value_range = st.slider(
                    "Value Range", 
                    min_value=1,
                    max_value=100,
                    value=(10, 50),
                    help="Min and max item values"
                )
            
            correlation = st.slider(
                "Weight-Value Correlation",
                min_value=-1.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Correlation between item weights and values"
            )
        
        with st.expander("Advanced Options", expanded=False):
            seed = st.number_input(
                "Random Seed (0 for random)",
                min_value=0,
                max_value=99999,
                value=42,
                help="Seed for reproducible instance generation"
            )
            
            preset = st.selectbox(
                "Load Preset Instance",
                options=['Custom', 'Small (Easy)', 'Medium', 'Large (Hard)', 'Benchmark A', 'Benchmark B'],
                help="Load a predefined test instance"
            )
            
            if preset != 'Custom':
                n_items, n_bins, capacity = self._get_preset_params(preset)
        
        # Generate problem button
        if st.button("🎲 Generate Instance", type="primary", use_container_width=True):
            with st.spinner("Generating problem instance..."):
                problem = self._generate_problem(
                    n_items=n_items,
                    n_bins=n_bins,
                    capacity=capacity,
                    distribution=distribution,
                    weight_range=weight_range,
                    value_range=value_range,
                    correlation=correlation,
                    seed=seed if seed != 0 else None
                )
                
                # Store in session state
                st.session_state['current_problem'] = problem
                st.success(f"✅ Generated instance with {n_items} items and {n_bins} bins")
                
                return problem
        
        # Return existing problem if available
        return st.session_state.get('current_problem')
    
    def _get_preset_params(self, preset: str) -> Tuple[int, int, int]:
        """Get parameters for preset instances."""
        presets = {
            'Small (Easy)': (10, 3, 50),
            'Medium': (30, 5, 100),
            'Large (Hard)': (50, 8, 150),
            'Benchmark A': (25, 4, 80),
            'Benchmark B': (40, 6, 120),
        }
        return presets.get(preset, (20, 4, 100))
    
    def _generate_problem(self, n_items: int, n_bins: int, capacity: int,
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
            bin_capacity=capacity,
            name=f"instance_{n_items}_{n_bins}_{capacity}"
        )
    
    def render_problem_summary(self, problem: Problem):
        """Display a summary of the current problem instance."""
        st.markdown("### 📊 Instance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Items", len(problem.items))
        with col2:
            st.metric("Bins", problem.num_bins)
        with col3:
            st.metric("Capacity", problem.bin_capacity)
        with col4:
            total_weight = sum(item.weight for item in problem.items)
            st.metric("Total Weight", f"{total_weight:.1f}")
        
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
                colorbar=dict(title="Value")
            ),
            text=[item.id for item in problem.items],
            hovertemplate='<b>%{text}</b><br>Weight: %{x:.2f}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Items Distribution",
            xaxis_title="Weight",
            yaxis_title="Value",
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
            'category': 'Greedy',
            'complexity': 'O(n log n)',
            'description': 'Sorts items by weight and assigns each to the first bin that fits.',
            'params': {}
        },
        'BestFitDecreasing': {
            'name': 'Best Fit Decreasing (BFD)',
            'category': 'Greedy',
            'complexity': 'O(n² log n)',
            'description': 'Assigns each item to the bin with minimum remaining capacity.',
            'params': {}
        },
        'WorstFitDecreasing': {
            'name': 'Worst Fit Decreasing (WFD)',
            'category': 'Greedy', 
            'complexity': 'O(n log n)',
            'description': 'Assigns items to bins with maximum remaining capacity for balance.',
            'params': {}
        },
        'RoundRobinGreedy': {
            'name': 'Round Robin Greedy',
            'category': 'Greedy',
            'complexity': 'O(n log n)',
            'description': 'Distributes items evenly across bins in round-robin fashion.',
            'params': {}
        },
        'LargestDifferenceFirst': {
            'name': 'Largest Difference First',
            'category': 'Greedy',
            'complexity': 'O(n log n)',
            'description': 'Prioritizes reducing the largest value difference between bins.',
            'params': {}
        },
        'SimulatedAnnealing': {
            'name': 'Simulated Annealing',
            'category': 'Metaheuristic',
            'complexity': 'O(iterations × n)',
            'description': 'Probabilistic optimization inspired by metallurgical annealing.',
            'params': {
                'initial_temp': (100.0, 1.0, 1000.0, 'Initial temperature'),
                'cooling_rate': (0.995, 0.9, 0.999, 'Temperature cooling rate'),
                'max_iterations': (10000, 100, 100000, 'Maximum iterations')
            }
        },
        'GeneticAlgorithm': {
            'name': 'Genetic Algorithm',
            'category': 'Metaheuristic',
            'complexity': 'O(generations × pop × n)',
            'description': 'Evolutionary algorithm using selection, crossover, and mutation.',
            'params': {
                'population_size': (50, 10, 200, 'Population size'),
                'generations': (100, 10, 500, 'Number of generations'),
                'mutation_rate': (0.1, 0.01, 0.5, 'Mutation probability')
            }
        },
        'TabuSearch': {
            'name': 'Tabu Search',
            'category': 'Metaheuristic',
            'complexity': 'O(iterations × neighbors)',
            'description': 'Local search with memory to avoid cycling.',
            'params': {
                'tabu_tenure': (10, 1, 50, 'Tabu list size'),
                'max_iterations': (1000, 100, 10000, 'Maximum iterations')
            }
        },
        'BranchAndBound': {
            'name': 'Branch and Bound',
            'category': 'Exact',
            'complexity': 'O(k^n) worst case',
            'description': 'Systematic enumeration with intelligent pruning.',
            'params': {
                'time_limit': (60.0, 1.0, 300.0, 'Time limit in seconds')
            }
        },
        'DynamicProgramming': {
            'name': 'Dynamic Programming',
            'category': 'Exact',
            'complexity': 'O(n × C^k)',
            'description': 'Optimal solution for small instances using memoization.',
            'params': {}
        }
    }
    
    def __init__(self, theme: str = "dark"):
        self.theme = ThemeManager.get_theme(theme)
        self.registry = AlgorithmRegistry()
    
    def render(self) -> List[Tuple[str, Dict]]:
        """Render algorithm selection panel."""
        st.markdown("### ⚙️ Algorithm Selection")
        
        # Category filter
        categories = ['All', 'Greedy', 'Metaheuristic', 'Exact']
        selected_category = st.radio(
            "Filter by Category",
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
            "Select Algorithms to Run",
            options=list(available_algorithms.keys()),
            default=default_selection if default_selection else None,
            format_func=lambda x: f"{available_algorithms[x]['name']} ({available_algorithms[x]['category']})"
        )
        
        # Algorithm details and parameters
        algorithm_configs = []
        
        for algo_name in selected_algorithms:
            info = self.ALGORITHM_INFO.get(algo_name, {})
            
            with st.expander(f"🔧 {info.get('name', algo_name)}", expanded=False):
                st.markdown(f"**Category:** {info.get('category', 'Unknown')}")
                st.markdown(f"**Complexity:** {info.get('complexity', 'Unknown')}")
                st.markdown(f"**Description:** {info.get('description', 'No description')}")
                
                # Parameter configuration
                params = {}
                if info.get('params'):
                    st.markdown("**Parameters:**")
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
        if category == 'All':
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
