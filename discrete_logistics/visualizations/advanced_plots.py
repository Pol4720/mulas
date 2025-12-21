"""
Advanced Plotly Visualizations Module
=====================================

Enhanced Plotly visualizations with modern styling,
animations, 3D options, and advanced interactivity.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple


class PlotlyModernTheme:
    """
    Modern Plotly theme configuration.
    """
    
    # Color palettes
    PALETTE_PRIMARY = ['#4F46E5', '#7C3AED', '#EC4899', '#10B981', '#F59E0B', '#06B6D4']
    PALETTE_GRADIENT = ['#4F46E5', '#6366F1', '#818CF8', '#A5B4FC', '#C7D2FE', '#E0E7FF']
    PALETTE_RAINBOW = ['#EF4444', '#F59E0B', '#10B981', '#06B6D4', '#3B82F6', '#8B5CF6', '#EC4899']
    
    # Custom colorscales
    COLORSCALE_INDIGO = [
        [0, '#E0E7FF'],
        [0.25, '#A5B4FC'],
        [0.5, '#818CF8'],
        [0.75, '#6366F1'],
        [1, '#4F46E5']
    ]
    
    COLORSCALE_AURORA = [
        [0, '#10B981'],
        [0.33, '#4F46E5'],
        [0.66, '#7C3AED'],
        [1, '#EC4899']
    ]
    
    @classmethod
    def get_layout(cls, title: str = "", height: int = 400) -> Dict:
        """Get modern layout configuration."""
        return {
            'title': {
                'text': title,
                'font': {
                    'family': 'Inter, sans-serif',
                    'size': 20,
                    'color': '#1E293B'
                },
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.95
            },
            'font': {
                'family': 'Inter, sans-serif',
                'size': 12,
                'color': '#475569'
            },
            'paper_bgcolor': 'rgba(255, 255, 255, 0.8)',
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'height': height,
            'margin': {'l': 60, 'r': 40, 't': 80, 'b': 60},
            'hoverlabel': {
                'bgcolor': 'white',
                'font_size': 13,
                'font_family': 'Inter, sans-serif',
                'bordercolor': '#E2E8F0'
            },
            'legend': {
                'bgcolor': 'rgba(255, 255, 255, 0.9)',
                'bordercolor': '#E2E8F0',
                'borderwidth': 1,
                'font': {'size': 11},
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5
            },
            'xaxis': {
                'gridcolor': 'rgba(226, 232, 240, 0.5)',
                'linecolor': '#E2E8F0',
                'tickfont': {'size': 11},
                'zeroline': False,
                'showgrid': True,
                'gridwidth': 1
            },
            'yaxis': {
                'gridcolor': 'rgba(226, 232, 240, 0.5)',
                'linecolor': '#E2E8F0',
                'tickfont': {'size': 11},
                'zeroline': False,
                'showgrid': True,
                'gridwidth': 1
            }
        }


class AdvancedSolutionPlots:
    """
    Advanced visualization plots for solutions.
    """
    
    @staticmethod
    def animated_bin_filling(
        solution,
        title: str = "Proceso de Llenado de Contenedores"
    ) -> go.Figure:
        """
        Create an animated visualization of bin filling process.
        
        Args:
            solution: Solution object with bins
            title: Chart title
            
        Returns:
            Animated Plotly Figure
        """
        bins = solution.bins
        n_bins = len(bins)
        
        # Collect all items with their bin assignments
        frames_data = []
        cumulative_values = [0] * n_bins
        cumulative_weights = [0] * n_bins
        
        # Initial frame
        frames_data.append({
            'values': [0] * n_bins,
            'weights': [0] * n_bins,
            'label': 'Inicio'
        })
        
        # Create frames for each item assignment
        item_num = 0
        for bin_idx, bin_obj in enumerate(bins):
            for item in bin_obj.items:
                cumulative_values[bin_idx] += item.value
                cumulative_weights[bin_idx] += item.weight
                item_num += 1
                frames_data.append({
                    'values': cumulative_values.copy(),
                    'weights': cumulative_weights.copy(),
                    'label': f'Item {item_num}: v={item.value:.1f}'
                })
        
        # Create figure
        fig = go.Figure()
        
        # Initial bars
        fig.add_trace(go.Bar(
            x=[f'Bin {i}' for i in range(n_bins)],
            y=frames_data[0]['values'],
            marker_color=PlotlyModernTheme.PALETTE_PRIMARY[:n_bins],
            text=[f'{v:.1f}' for v in frames_data[0]['values']],
            textposition='outside',
            name='Valor'
        ))
        
        # Create animation frames
        frames = []
        for i, data in enumerate(frames_data):
            frames.append(go.Frame(
                data=[go.Bar(
                    x=[f'Bin {j}' for j in range(n_bins)],
                    y=data['values'],
                    marker_color=PlotlyModernTheme.PALETTE_PRIMARY[:n_bins],
                    text=[f'{v:.1f}' for v in data['values']],
                    textposition='outside'
                )],
                name=str(i),
                layout=go.Layout(title=f"{title} - {data['label']}")
            ))
        
        fig.frames = frames
        
        # Animation controls
        layout = PlotlyModernTheme.get_layout(title, 500)
        layout.update({
            'updatemenus': [{
                'type': 'buttons',
                'showactive': False,
                'y': 1.15,
                'x': 0.5,
                'xanchor': 'center',
                'buttons': [
                    {
                        'label': '▶ Reproducir',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                        }]
                    },
                    {
                        'label': '⏸ Pausar',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            'sliders': [{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'prefix': 'Paso: ',
                    'visible': True,
                    'xanchor': 'right',
                    'font': {'size': 12}
                },
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.05,
                'y': 0,
                'steps': [
                    {
                        'args': [[str(i)], {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 200}
                        }],
                        'label': str(i),
                        'method': 'animate'
                    }
                    for i in range(len(frames))
                ]
            }]
        })
        
        fig.update_layout(**layout)
        
        return fig
    
    @staticmethod
    def value_balance_sankey(
        solution,
        title: str = "Flujo de Valores: Items → Bins"
    ) -> go.Figure:
        """
        Create a Sankey diagram showing item-to-bin value flow.
        
        Args:
            solution: Solution object
            title: Chart title
            
        Returns:
            Plotly Figure with Sankey diagram
        """
        bins = solution.bins
        
        # Prepare Sankey data
        sources = []
        targets = []
        values = []
        labels = []
        colors = []
        
        item_idx = 0
        n_bins = len(bins)
        
        # Items are sources (indices 0 to total_items-1)
        # Bins are targets (indices total_items to total_items + n_bins - 1)
        
        total_items = sum(len(b.items) for b in bins)
        
        # Add item labels
        for bin_obj in bins:
            for item in bin_obj.items:
                labels.append(f'Item {item.id}')
        
        # Add bin labels
        for i, bin_obj in enumerate(bins):
            labels.append(f'Bin {i}')
        
        # Create flows
        for bin_idx, bin_obj in enumerate(bins):
            for item in bin_obj.items:
                sources.append(item_idx)
                targets.append(total_items + bin_idx)
                values.append(item.value)
                colors.append(PlotlyModernTheme.PALETTE_PRIMARY[bin_idx % len(PlotlyModernTheme.PALETTE_PRIMARY)])
                item_idx += 1
        
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color='white', width=2),
                label=labels,
                color=PlotlyModernTheme.PALETTE_GRADIENT
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=[c.replace(')', ', 0.3)').replace('rgb', 'rgba') 
                       if c.startswith('rgb') else c + '50' for c in colors]
            )
        ))
        
        layout = PlotlyModernTheme.get_layout(title, 500)
        fig.update_layout(**layout)
        
        return fig
    
    @staticmethod
    def bin_visualization_3d(
        solution,
        title: str = "Visualización 3D de Contenedores"
    ) -> go.Figure:
        """
        Create a 3D visualization of bins and their contents.
        
        Args:
            solution: Solution object
            title: Chart title
            
        Returns:
            3D Plotly Figure
        """
        bins = solution.bins
        n_bins = len(bins)
        
        fig = go.Figure()
        
        # Create 3D bars for each bin
        for i, bin_obj in enumerate(bins):
            # Bin position
            x_pos = i * 2
            
            # Add bin container (wireframe)
            capacity = bin_obj.capacity
            fig.add_trace(go.Mesh3d(
                x=[x_pos-0.4, x_pos+0.4, x_pos+0.4, x_pos-0.4, x_pos-0.4, x_pos+0.4, x_pos+0.4, x_pos-0.4],
                y=[0, 0, 0, 0, 1, 1, 1, 1],
                z=[0, 0, capacity, capacity, 0, 0, capacity, capacity],
                opacity=0.1,
                color='gray',
                name=f'Capacidad Bin {i}',
                showlegend=False
            ))
            
            # Add filled portion (value representation)
            fill_height = bin_obj.current_value / max(b.current_value for b in bins) * capacity if max(b.current_value for b in bins) > 0 else 0
            
            color = PlotlyModernTheme.PALETTE_PRIMARY[i % len(PlotlyModernTheme.PALETTE_PRIMARY)]
            
            fig.add_trace(go.Mesh3d(
                x=[x_pos-0.35, x_pos+0.35, x_pos+0.35, x_pos-0.35, x_pos-0.35, x_pos+0.35, x_pos+0.35, x_pos-0.35],
                y=[0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9],
                z=[0, 0, fill_height, fill_height, 0, 0, fill_height, fill_height],
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                opacity=0.8,
                color=color,
                name=f'Bin {i}: V={bin_obj.current_value:.1f}',
                hoverinfo='name'
            ))
        
        layout = PlotlyModernTheme.get_layout(title, 600)
        layout.update({
            'scene': {
                'xaxis': {'title': 'Bins', 'tickvals': [i*2 for i in range(n_bins)], 'ticktext': [f'Bin {i}' for i in range(n_bins)]},
                'yaxis': {'title': '', 'showticklabels': False},
                'zaxis': {'title': 'Valor/Capacidad'},
                'camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1}}
            }
        })
        
        fig.update_layout(**layout)
        
        return fig
    
    @staticmethod
    def treemap_distribution(
        solution,
        title: str = "Distribución de Items por Contenedor"
    ) -> go.Figure:
        """
        Create a treemap showing item distribution across bins.
        
        Args:
            solution: Solution object
            title: Chart title
            
        Returns:
            Plotly Treemap Figure
        """
        bins = solution.bins
        
        # Prepare hierarchical data
        labels = ['Total']
        parents = ['']
        values = [sum(b.current_value for b in bins)]
        colors = ['#F8FAFC']
        
        for i, bin_obj in enumerate(bins):
            bin_label = f'Bin {i}'
            labels.append(bin_label)
            parents.append('Total')
            values.append(bin_obj.current_value)
            colors.append(PlotlyModernTheme.PALETTE_PRIMARY[i % len(PlotlyModernTheme.PALETTE_PRIMARY)])
            
            for item in bin_obj.items:
                labels.append(f'{item.id}')
                parents.append(bin_label)
                values.append(item.value)
                # Lighter shade of bin color
                colors.append(PlotlyModernTheme.PALETTE_GRADIENT[i % len(PlotlyModernTheme.PALETTE_GRADIENT)])
        
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker_colors=colors,
            branchvalues='total',
            textinfo='label+value+percent parent',
            textfont={'size': 12, 'family': 'Inter, sans-serif'},
            hovertemplate='<b>%{label}</b><br>Valor: %{value:.2f}<br>%{percentParent:.1%} del padre<extra></extra>'
        ))
        
        layout = PlotlyModernTheme.get_layout(title, 500)
        fig.update_layout(**layout)
        
        return fig


class AdvancedBenchmarkPlots:
    """
    Advanced benchmark comparison visualizations.
    """
    
    @staticmethod
    def parallel_coordinates(
        results: Dict[str, Dict],
        title: str = "Comparación Multidimensional de Algoritmos"
    ) -> go.Figure:
        """
        Create a parallel coordinates plot for algorithm comparison.
        
        Args:
            results: Dictionary of algorithm results
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        # Prepare data
        data = []
        for i, (algo_name, result) in enumerate(results.items()):
            data.append({
                'Algorithm': algo_name,
                'Objective': result.get('objective', 0),
                'Time': result.get('time', 0),
                'Balance': result.get('balance_score', 0),
                'Feasibility': 1 if result.get('feasible', False) else 0,
                'Index': i
            })
        
        df = pd.DataFrame(data)
        
        fig = go.Figure(go.Parcoords(
            line=dict(
                color=df['Index'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Algoritmo')
            ),
            dimensions=[
                dict(
                    range=[df['Objective'].min() * 0.9, df['Objective'].max() * 1.1],
                    label='Objetivo',
                    values=df['Objective']
                ),
                dict(
                    range=[0, df['Time'].max() * 1.1],
                    label='Tiempo (s)',
                    values=df['Time']
                ),
                dict(
                    range=[0, 1],
                    label='Balance',
                    values=df['Balance']
                ),
                dict(
                    range=[0, 1],
                    label='Factible',
                    values=df['Feasibility'],
                    tickvals=[0, 1],
                    ticktext=['No', 'Sí']
                )
            ]
        ))
        
        layout = PlotlyModernTheme.get_layout(title, 500)
        fig.update_layout(**layout)
        
        return fig
    
    @staticmethod
    def bubble_chart(
        results: Dict[str, Dict],
        title: str = "Análisis de Trade-offs: Tiempo vs Calidad"
    ) -> go.Figure:
        """
        Create a bubble chart showing time-quality trade-offs.
        
        Args:
            results: Dictionary of algorithm results
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        for i, (algo_name, result) in enumerate(results.items()):
            objective = result.get('objective', 0)
            time_val = result.get('time', 0)
            balance = result.get('balance_score', 0.5)
            feasible = result.get('feasible', False)
            
            color = PlotlyModernTheme.PALETTE_PRIMARY[i % len(PlotlyModernTheme.PALETTE_PRIMARY)]
            
            fig.add_trace(go.Scatter(
                x=[time_val],
                y=[objective],
                mode='markers+text',
                marker=dict(
                    size=balance * 60 + 20,  # Size based on balance
                    color=color,
                    opacity=0.7,
                    line=dict(color='white', width=2)
                ),
                text=[algo_name],
                textposition='top center',
                textfont=dict(size=11),
                name=algo_name,
                hovertemplate=(
                    f'<b>{algo_name}</b><br>'
                    f'Tiempo: %{{x:.4f}}s<br>'
                    f'Objetivo: %{{y:.4f}}<br>'
                    f'Balance: {balance:.2%}<br>'
                    f'Factible: {"Sí" if feasible else "No"}'
                    '<extra></extra>'
                )
            ))
        
        layout = PlotlyModernTheme.get_layout(title, 500)
        layout['xaxis']['title'] = 'Tiempo de Ejecución (s)'
        layout['yaxis']['title'] = 'Valor Objetivo (menor es mejor)'
        
        # Add annotation
        fig.add_annotation(
            text="Tamaño = Puntuación de Balance",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color='gray')
        )
        
        fig.update_layout(**layout)
        
        return fig
    
    @staticmethod
    def algorithm_ranking_bars(
        results: Dict[str, Dict],
        title: str = "Ranking de Algoritmos"
    ) -> go.Figure:
        """
        Create a horizontal bar chart ranking algorithms.
        
        Args:
            results: Dictionary of algorithm results
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        # Sort by objective
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get('objective', float('inf'))
        )
        
        algos = [item[0] for item in sorted_results]
        objectives = [item[1].get('objective', 0) for item in sorted_results]
        times = [item[1].get('time', 0) for item in sorted_results]
        feasible = [item[1].get('feasible', False) for item in sorted_results]
        
        # Create colors based on ranking
        colors = []
        for i, f in enumerate(feasible):
            if i == 0:  # Best
                colors.append('#10B981')  # Green
            elif not f:  # Infeasible
                colors.append('#EF4444')  # Red
            else:
                colors.append(PlotlyModernTheme.PALETTE_PRIMARY[i % len(PlotlyModernTheme.PALETTE_PRIMARY)])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=algos,
            x=objectives,
            orientation='h',
            marker_color=colors,
            text=[f'{obj:.4f}' for obj in objectives],
            textposition='auto',
            textfont=dict(color='white', size=12),
            hovertemplate='<b>%{y}</b><br>Objetivo: %{x:.4f}<extra></extra>'
        ))
        
        # Add medal emojis for top 3
        for i in range(min(3, len(algos))):
            medals = ['🥇', '🥈', '🥉']
            fig.add_annotation(
                x=objectives[i] + max(objectives) * 0.05,
                y=algos[i],
                text=medals[i],
                showarrow=False,
                font=dict(size=20)
            )
        
        layout = PlotlyModernTheme.get_layout(title, 400 + len(algos) * 30)
        layout['xaxis']['title'] = 'Valor Objetivo (menor es mejor)'
        layout['yaxis']['title'] = ''
        layout['yaxis']['categoryorder'] = 'array'
        layout['yaxis']['categoryarray'] = list(reversed(algos))
        
        fig.update_layout(**layout)
        
        return fig
    
    @staticmethod
    def convergence_comparison(
        history: Dict[str, List[float]],
        title: str = "Comparación de Convergencia"
    ) -> go.Figure:
        """
        Create an animated convergence comparison chart.
        
        Args:
            history: Dictionary mapping algorithm names to convergence histories
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        for i, (algo_name, values) in enumerate(history.items()):
            color = PlotlyModernTheme.PALETTE_PRIMARY[i % len(PlotlyModernTheme.PALETTE_PRIMARY)]
            
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode='lines',
                name=algo_name,
                line=dict(
                    color=color,
                    width=2,
                    shape='spline'
                ),
                fill='tonexty' if i > 0 else 'tozeroy',
                fillcolor=f'rgba{tuple(list(int(color[j:j+2], 16) for j in (1, 3, 5)) + [0.1])}',
                hovertemplate=f'<b>{algo_name}</b><br>Iteración: %{{x}}<br>Objetivo: %{{y:.4f}}<extra></extra>'
            ))
        
        layout = PlotlyModernTheme.get_layout(title, 450)
        layout['xaxis']['title'] = 'Iteración'
        layout['yaxis']['title'] = 'Valor Objetivo'
        
        fig.update_layout(**layout)
        
        return fig


class AdvancedStatisticalPlots:
    """
    Statistical analysis visualizations.
    """
    
    @staticmethod
    def box_violin_comparison(
        data: Dict[str, List[float]],
        title: str = "Distribución de Resultados por Algoritmo"
    ) -> go.Figure:
        """
        Create a combined box and violin plot.
        
        Args:
            data: Dictionary mapping algorithm names to lists of values
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        for i, (algo_name, values) in enumerate(data.items()):
            color = PlotlyModernTheme.PALETTE_PRIMARY[i % len(PlotlyModernTheme.PALETTE_PRIMARY)]
            
            fig.add_trace(go.Violin(
                y=values,
                name=algo_name,
                box_visible=True,
                meanline_visible=True,
                fillcolor=f'{color}40',
                line_color=color,
                opacity=0.7
            ))
        
        layout = PlotlyModernTheme.get_layout(title, 450)
        layout['yaxis']['title'] = 'Valor Objetivo'
        layout['violinmode'] = 'group'
        
        fig.update_layout(**layout)
        
        return fig
    
    @staticmethod
    def correlation_heatmap(
        metrics: pd.DataFrame,
        title: str = "Correlación entre Métricas"
    ) -> go.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            metrics: DataFrame with metrics
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        corr = metrics.corr()
        
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=PlotlyModernTheme.COLORSCALE_INDIGO,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={'size': 12},
            hoverongaps=False,
            hovertemplate='%{x} vs %{y}<br>Correlación: %{z:.3f}<extra></extra>'
        ))
        
        layout = PlotlyModernTheme.get_layout(title, 500)
        fig.update_layout(**layout)
        
        return fig
