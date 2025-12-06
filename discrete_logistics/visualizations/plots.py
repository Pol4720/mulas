"""
Static plotting utilities for solutions and benchmarks.

Uses Plotly for interactive scientific visualizations.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from ..core.problem import Problem, Solution, Bin


# Scientific color palette
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#28A745",
    "warning": "#F18F01",
    "danger": "#C73E1D",
    "info": "#17A2B8",
    "light": "#F8F9FA",
    "dark": "#343A40",
    "bins": px.colors.qualitative.Set2,
    "sequential": px.colors.sequential.Viridis,
    "diverging": px.colors.diverging.RdBu
}


class SolutionPlotter:
    """
    Creates visualizations for individual solutions.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize plotter.
        
        Args:
            theme: Plotly template (plotly_white, plotly_dark, etc.)
        """
        self.theme = theme
    
    def plot_bin_distribution(
        self,
        solution: Solution,
        show_items: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create bar chart showing value and weight distribution across bins.
        
        Args:
            solution: Solution to visualize
            show_items: Whether to show individual items in bins
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        bins = solution.bins
        n_bins = len(bins)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Value Distribution", "Weight Distribution"),
            vertical_spacing=0.15
        )
        
        # Value distribution
        values = [b.current_value for b in bins]
        fig.add_trace(
            go.Bar(
                x=[f"Bin {i}" for i in range(n_bins)],
                y=values,
                marker_color=COLORS["bins"][:n_bins],
                text=[f"{v:.1f}" for v in values],
                textposition="outside",
                name="Value"
            ),
            row=1, col=1
        )
        
        # Add ideal line
        avg_value = sum(values) / n_bins
        fig.add_hline(
            y=avg_value, line_dash="dash", line_color="red",
            annotation_text=f"Ideal: {avg_value:.1f}",
            row=1, col=1
        )
        
        # Weight distribution
        weights = [b.current_weight for b in bins]
        capacities = [b.capacity for b in bins]
        
        fig.add_trace(
            go.Bar(
                x=[f"Bin {i}" for i in range(n_bins)],
                y=weights,
                marker_color=COLORS["bins"][:n_bins],
                text=[f"{w:.1f}/{c:.1f}" for w, c in zip(weights, capacities)],
                textposition="outside",
                name="Weight"
            ),
            row=2, col=1
        )
        
        # Add capacity line
        fig.add_hline(
            y=capacities[0], line_dash="dash", line_color="red",
            annotation_text=f"Capacity: {capacities[0]:.1f}",
            row=2, col=1
        )
        
        fig.update_layout(
            title=title or f"Solution: {solution.algorithm_name}",
            template=self.theme,
            showlegend=False,
            height=600
        )
        
        return fig
    
    def plot_bin_contents(
        self,
        solution: Solution,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create stacked bar chart showing items in each bin.
        
        Args:
            solution: Solution to visualize
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        bins = solution.bins
        
        fig = go.Figure()
        
        # Get all items and their bin assignments
        all_items = []
        for bin in bins:
            for item in bin.items:
                all_items.append({
                    "bin": f"Bin {bin.id}",
                    "item": f"Item {item.id}",
                    "value": item.value,
                    "weight": item.weight
                })
        
        df = pd.DataFrame(all_items)
        
        if not df.empty:
            # Create stacked bar chart
            for bin in bins:
                bin_items = [i for i in bin.items]
                
                fig.add_trace(go.Bar(
                    name=f"Bin {bin.id}",
                    x=[f"Bin {bin.id}"],
                    y=[bin.current_value],
                    text=[f"{len(bin_items)} items<br>V={bin.current_value:.1f}<br>W={bin.current_weight:.1f}"],
                    textposition="inside",
                    marker_color=COLORS["bins"][bin.id % len(COLORS["bins"])]
                ))
        
        fig.update_layout(
            title=title or f"Bin Contents: {solution.algorithm_name}",
            template=self.theme,
            barmode="group",
            xaxis_title="Bins",
            yaxis_title="Total Value",
            height=400
        )
        
        return fig
    
    def plot_balance_gauge(
        self,
        solution: Solution,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create gauge chart showing balance factor.
        
        Args:
            solution: Solution to visualize
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        balance = solution.balance_factor
        diff = solution.value_difference
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=balance * 100,
            title={"text": "Balance Factor (%)"},
            delta={"reference": 100, "decreasing": {"color": "red"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLORS["primary"]},
                "steps": [
                    {"range": [0, 60], "color": "#ffcccc"},
                    {"range": [60, 80], "color": "#ffffcc"},
                    {"range": [80, 100], "color": "#ccffcc"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90
                }
            }
        ))
        
        fig.update_layout(
            title=title or f"Balance: {solution.algorithm_name}",
            template=self.theme,
            height=300,
            annotations=[
                dict(
                    text=f"Value Difference: {diff:.2f}",
                    x=0.5, y=-0.15,
                    showarrow=False,
                    font=dict(size=14)
                )
            ]
        )
        
        return fig
    
    def plot_solution_comparison(
        self,
        solutions: List[Solution],
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create comparison chart for multiple solutions.
        
        Args:
            solutions: List of solutions to compare
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Value Difference (lower is better)",
                "Balance Factor (higher is better)",
                "Execution Time (seconds)",
                "Iterations"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        names = [s.algorithm_name for s in solutions]
        diffs = [s.value_difference for s in solutions]
        balances = [s.balance_factor * 100 for s in solutions]
        times = [s.execution_time for s in solutions]
        iters = [s.iterations for s in solutions]
        
        colors = [COLORS["bins"][i % len(COLORS["bins"])] for i in range(len(solutions))]
        
        # Value Difference
        fig.add_trace(
            go.Bar(x=names, y=diffs, marker_color=colors, text=[f"{d:.2f}" for d in diffs],
                   textposition="outside"),
            row=1, col=1
        )
        
        # Balance Factor
        fig.add_trace(
            go.Bar(x=names, y=balances, marker_color=colors, text=[f"{b:.1f}%" for b in balances],
                   textposition="outside"),
            row=1, col=2
        )
        
        # Execution Time
        fig.add_trace(
            go.Bar(x=names, y=times, marker_color=colors, text=[f"{t:.4f}s" for t in times],
                   textposition="outside"),
            row=2, col=1
        )
        
        # Iterations
        fig.add_trace(
            go.Bar(x=names, y=iters, marker_color=colors, text=[str(i) for i in iters],
                   textposition="outside"),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title or "Algorithm Comparison",
            template=self.theme,
            showlegend=False,
            height=700
        )
        
        return fig
    
    def plot_items_scatter(
        self,
        problem: Problem,
        solution: Optional[Solution] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create scatter plot of items colored by bin assignment.
        
        Args:
            problem: Problem instance
            solution: Optional solution for coloring
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        items = problem.items
        
        weights = [i.weight for i in items]
        values = [i.value for i in items]
        ids = [i.id for i in items]
        
        if solution:
            assignment = solution.get_item_assignment()
            colors = [assignment.get(i.id, 0) for i in items]
            color_scale = "Viridis"
        else:
            colors = values
            color_scale = "Viridis"
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=weights,
            y=values,
            mode="markers",
            marker=dict(
                size=12,
                color=colors,
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(title="Bin" if solution else "Value")
            ),
            text=[f"Item {id}<br>Weight: {w:.1f}<br>Value: {v:.1f}"
                  for id, w, v in zip(ids, weights, values)],
            hovertemplate="%{text}<extra></extra>"
        ))
        
        fig.update_layout(
            title=title or "Items Distribution",
            template=self.theme,
            xaxis_title="Weight",
            yaxis_title="Value",
            height=500
        )
        
        return fig


class BenchmarkPlotter:
    """
    Creates visualizations for benchmark results.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        self.theme = theme
    
    def plot_performance_heatmap(
        self,
        results: pd.DataFrame,
        metric: str = "value_difference",
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create heatmap showing algorithm performance across instances.
        
        Args:
            results: DataFrame with columns [algorithm, instance, metric]
            metric: Metric to visualize
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        # Pivot table
        pivot = results.pivot_table(
            values=metric,
            index="algorithm",
            columns="instance",
            aggfunc="mean"
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn_r",  # Red=bad, Green=good
            text=np.round(pivot.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=title or f"Performance Heatmap: {metric}",
            template=self.theme,
            xaxis_title="Instance",
            yaxis_title="Algorithm",
            height=400 + len(pivot.index) * 30
        )
        
        return fig
    
    def plot_scaling_analysis(
        self,
        results: pd.DataFrame,
        x_var: str = "n_items",
        y_var: str = "execution_time",
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create line plot showing how metrics scale with problem size.
        
        Args:
            results: DataFrame with columns [algorithm, n_items, metric]
            x_var: Variable for x-axis
            y_var: Variable for y-axis
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = px.line(
            results,
            x=x_var,
            y=y_var,
            color="algorithm",
            markers=True,
            template=self.theme,
            title=title or f"Scaling: {y_var} vs {x_var}"
        )
        
        fig.update_layout(
            xaxis_title=x_var.replace("_", " ").title(),
            yaxis_title=y_var.replace("_", " ").title(),
            legend_title="Algorithm",
            height=500
        )
        
        return fig
    
    def plot_boxplot_comparison(
        self,
        results: pd.DataFrame,
        metric: str = "value_difference",
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create box plot comparing algorithm performance distributions.
        
        Args:
            results: DataFrame with columns [algorithm, metric]
            metric: Metric to visualize
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = px.box(
            results,
            x="algorithm",
            y=metric,
            color="algorithm",
            template=self.theme,
            title=title or f"Distribution: {metric}"
        )
        
        fig.update_layout(
            xaxis_title="Algorithm",
            yaxis_title=metric.replace("_", " ").title(),
            showlegend=False,
            height=500
        )
        
        return fig
    
    def plot_pareto_front(
        self,
        results: pd.DataFrame,
        x_metric: str = "execution_time",
        y_metric: str = "value_difference",
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create scatter plot with Pareto front highlighting.
        
        Args:
            results: DataFrame with columns [algorithm, x_metric, y_metric]
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = px.scatter(
            results,
            x=x_metric,
            y=y_metric,
            color="algorithm",
            hover_data=["instance"] if "instance" in results.columns else None,
            template=self.theme,
            title=title or f"Pareto Analysis: {y_metric} vs {x_metric}"
        )
        
        # Find Pareto front
        x_vals = results[x_metric].values
        y_vals = results[y_metric].values
        
        pareto_mask = self._get_pareto_mask(x_vals, y_vals)
        pareto_points = results[pareto_mask].sort_values(x_metric)
        
        # Add Pareto front line
        fig.add_trace(go.Scatter(
            x=pareto_points[x_metric],
            y=pareto_points[y_metric],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Pareto Front"
        ))
        
        fig.update_layout(
            xaxis_title=x_metric.replace("_", " ").title(),
            yaxis_title=y_metric.replace("_", " ").title(),
            height=500
        )
        
        return fig
    
    def _get_pareto_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Find Pareto-optimal points (minimizing both x and y)."""
        n = len(x)
        mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if x[j] <= x[i] and y[j] <= y[i]:
                        if x[j] < x[i] or y[j] < y[i]:
                            mask[i] = False
                            break
        
        return mask


class ConvergencePlotter:
    """
    Creates visualizations for algorithm convergence.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        self.theme = theme
    
    def plot_convergence(
        self,
        history: List[float],
        algorithm_name: str = "Algorithm",
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create line plot showing objective value over iterations.
        
        Args:
            history: List of objective values
            algorithm_name: Name of the algorithm
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(history))),
            y=history,
            mode="lines",
            name=algorithm_name,
            line=dict(color=COLORS["primary"], width=2)
        ))
        
        fig.update_layout(
            title=title or f"Convergence: {algorithm_name}",
            template=self.theme,
            xaxis_title="Iteration",
            yaxis_title="Objective Value",
            height=400
        )
        
        return fig
    
    def plot_multi_convergence(
        self,
        histories: Dict[str, List[float]],
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create line plot comparing convergence of multiple algorithms.
        
        Args:
            histories: Dict mapping algorithm name to history list
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        colors = COLORS["bins"]
        
        for i, (name, history) in enumerate(histories.items()):
            fig.add_trace(go.Scatter(
                x=list(range(len(history))),
                y=history,
                mode="lines",
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=title or "Convergence Comparison",
            template=self.theme,
            xaxis_title="Iteration",
            yaxis_title="Objective Value",
            legend_title="Algorithm",
            height=500
        )
        
        return fig
    
    def plot_sa_temperature(
        self,
        temperatures: List[float],
        objectives: List[float],
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create dual-axis plot for Simulated Annealing temperature and objective.
        
        Args:
            temperatures: List of temperature values
            objectives: List of objective values
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(objectives))),
                y=objectives,
                name="Objective",
                line=dict(color=COLORS["primary"])
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(temperatures))),
                y=temperatures,
                name="Temperature",
                line=dict(color=COLORS["secondary"], dash="dash")
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title=title or "Simulated Annealing Progress",
            template=self.theme,
            height=500
        )
        
        fig.update_xaxes(title_text="Iteration")
        fig.update_yaxes(title_text="Objective Value", secondary_y=False)
        fig.update_yaxes(title_text="Temperature", secondary_y=True)
        
        return fig
    
    def plot_ga_evolution(
        self,
        fitness_history: List[float],
        diversity_history: List[float],
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create plot for Genetic Algorithm evolution.
        
        Args:
            fitness_history: Best fitness per generation
            diversity_history: Population diversity per generation
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(fitness_history))),
                y=fitness_history,
                name="Best Fitness",
                line=dict(color=COLORS["success"], width=2)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(diversity_history))),
                y=diversity_history,
                name="Diversity",
                line=dict(color=COLORS["info"], dash="dot")
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title=title or "Genetic Algorithm Evolution",
            template=self.theme,
            height=500
        )
        
        fig.update_xaxes(title_text="Generation")
        fig.update_yaxes(title_text="Best Fitness", secondary_y=False)
        fig.update_yaxes(title_text="Population Diversity", secondary_y=True)
        
        return fig
