"""
Animation module for algorithm visualization.

Supports both Manim (for high-quality exports) and Plotly (for interactive web).
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np

# Try importing manim, fall back gracefully if not available
try:
    from manim import *
    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.problem import Problem, Solution, Bin, Item
from ..algorithms.base import AlgorithmStep


class AlgorithmAnimator:
    """
    Creates animations of algorithm execution.
    
    Supports:
    - Plotly animated figures (web-compatible)
    - Manim scenes (high-quality video export)
    """
    
    def __init__(self, fps: int = 30, theme: str = "plotly_white"):
        """
        Initialize animator.
        
        Args:
            fps: Frames per second for animations
            theme: Plotly theme for web animations
        """
        self.fps = fps
        self.theme = theme
        self.colors = [
            "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
            "#28A745", "#17A2B8", "#6C757D", "#343A40"
        ]
    
    def create_plotly_animation(
        self,
        steps: List[AlgorithmStep],
        problem: Problem,
        title: str = "Algorithm Animation"
    ) -> go.Figure:
        """
        Create Plotly animation from algorithm steps.
        
        Args:
            steps: List of algorithm execution steps
            problem: Problem instance
            title: Animation title
            
        Returns:
            Plotly Figure with animation frames
        """
        if not steps:
            return self._create_empty_figure(title)
        
        k = problem.num_bins
        
        # Create figure with animation frames
        fig = go.Figure()
        
        # Initial frame (empty bins)
        initial_trace = go.Bar(
            x=[f"Bin {i}" for i in range(k)],
            y=[0] * k,
            marker_color=self.colors[:k],
            text=["0.0"] * k,
            textposition="outside",
            name="Values"
        )
        fig.add_trace(initial_trace)
        
        # Create frames for each step
        frames = []
        
        for step in steps:
            if step.bins_state is None:
                continue
            
            values = [bs["current_value"] for bs in step.bins_state]
            weights = [bs["current_weight"] for bs in step.bins_state]
            
            frame_data = [go.Bar(
                x=[f"Bin {i}" for i in range(k)],
                y=values,
                marker_color=self.colors[:k],
                text=[f"V:{v:.1f}<br>W:{w:.1f}" for v, w in zip(values, weights)],
                textposition="outside"
            )]
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(step.step_number),
                layout=go.Layout(
                    title=f"Step {step.step_number}: {step.action}"
                )
            ))
        
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="Bins",
            yaxis_title="Value",
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.15,
                    x=0.5,
                    xanchor="center",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 200}
                            }]
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        )
                    ]
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue={
                        "prefix": "Step: ",
                        "visible": True,
                        "xanchor": "right"
                    },
                    pad={"b": 10, "t": 50},
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[[f.name], {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 200}
                            }],
                            label=str(i),
                            method="animate"
                        )
                        for i, f in enumerate(frames)
                    ]
                )
            ],
            height=600
        )
        
        return fig
    
    def create_step_by_step_figure(
        self,
        steps: List[AlgorithmStep],
        problem: Problem,
        step_index: int = 0
    ) -> go.Figure:
        """
        Create figure for a specific step (for manual stepping).
        
        Args:
            steps: List of algorithm execution steps
            problem: Problem instance
            step_index: Index of step to visualize
            
        Returns:
            Plotly Figure for the specified step
        """
        if not steps or step_index >= len(steps):
            return self._create_empty_figure("No Step Data")
        
        step = steps[step_index]
        k = problem.num_bins
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Bin Values", "Bin Weights",
                "Items per Bin", "Current Metrics"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        if step.bins_state:
            values = [bs["current_value"] for bs in step.bins_state]
            weights = [bs["current_weight"] for bs in step.bins_state]
            item_counts = [len(bs["items"]) for bs in step.bins_state]
            
            # Values
            fig.add_trace(
                go.Bar(
                    x=[f"Bin {i}" for i in range(k)],
                    y=values,
                    marker_color=self.colors[:k],
                    text=[f"{v:.1f}" for v in values],
                    textposition="outside"
                ),
                row=1, col=1
            )
            
            # Weights
            fig.add_trace(
                go.Bar(
                    x=[f"Bin {i}" for i in range(k)],
                    y=weights,
                    marker_color=self.colors[:k],
                    text=[f"{w:.1f}" for w in weights],
                    textposition="outside"
                ),
                row=1, col=2
            )
            
            # Item counts
            fig.add_trace(
                go.Bar(
                    x=[f"Bin {i}" for i in range(k)],
                    y=item_counts,
                    marker_color=self.colors[:k],
                    text=[str(c) for c in item_counts],
                    textposition="outside"
                ),
                row=2, col=1
            )
            
            # Metrics indicator
            if step.metrics:
                fig.add_trace(
                    go.Indicator(
                        mode="number+delta",
                        value=step.metrics.get("value_difference", 0),
                        title={"text": "Value Difference"},
                        number={"font": {"size": 40}},
                        delta={"reference": 0}
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=f"Step {step.step_number}: {step.action}",
            template=self.theme,
            showlegend=False,
            height=600
        )
        
        return fig
    
    def create_branch_bound_tree(
        self,
        steps: List[AlgorithmStep],
        max_depth: int = 5
    ) -> go.Figure:
        """
        Create visualization of Branch and Bound decision tree.
        
        Args:
            steps: Algorithm steps with tree structure
            max_depth: Maximum depth to visualize
            
        Returns:
            Plotly Figure with tree visualization
        """
        # Build tree structure from steps
        nodes = []
        edges = []
        
        for i, step in enumerate(steps[:100]):  # Limit for visualization
            level = step.extra_data.get("level", 0)
            if level > max_depth:
                continue
            
            nodes.append({
                "id": i,
                "level": level,
                "label": f"Step {step.step_number}",
                "value": step.metrics.get("value_difference", 0) if step.metrics else 0
            })
        
        if not nodes:
            return self._create_empty_figure("No Tree Data")
        
        # Position nodes by level
        levels = {}
        for node in nodes:
            level = node["level"]
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        # Calculate positions
        max_width = max(len(v) for v in levels.values()) if levels else 1
        
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for level, level_nodes in levels.items():
            width = len(level_nodes)
            for i, node in enumerate(level_nodes):
                x = (i - width/2 + 0.5) * (max_width / max(width, 1))
                y = -level
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{node['label']}<br>Diff: {node['value']:.2f}")
                node_color.append(node['value'])
        
        fig = go.Figure()
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                size=20,
                color=node_color,
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="Diff")
            ),
            text=[n["label"] for n in nodes],
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text"
        ))
        
        fig.update_layout(
            title="Branch and Bound Decision Tree",
            template=self.theme,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def _create_empty_figure(self, title: str) -> go.Figure:
        """Create empty figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            template=self.theme,
            height=400
        )
        return fig


def create_algorithm_animation(
    problem: Problem,
    algorithm_name: str,
    output_format: str = "html"
) -> Any:
    """
    Convenience function to run algorithm and create animation.
    
    Args:
        problem: Problem instance
        algorithm_name: Name of registered algorithm
        output_format: 'html', 'json', or 'figure'
        
    Returns:
        Animation in specified format
    """
    from ..algorithms import AlgorithmRegistry
    
    # Get algorithm with step tracking
    algorithm = AlgorithmRegistry.get(algorithm_name, track_steps=True)
    solution = algorithm.solve(problem)
    steps = algorithm.get_steps()
    
    # Create animation
    animator = AlgorithmAnimator()
    fig = animator.create_plotly_animation(steps, problem, f"{algorithm_name} Animation")
    
    if output_format == "html":
        return fig.to_html(include_plotlyjs=True, full_html=True)
    elif output_format == "json":
        return fig.to_json()
    else:
        return fig


# Manim-based animations (only if manim is available)
if MANIM_AVAILABLE:
    
    class BinPackingScene(Scene):
        """
        Manim scene for high-quality algorithm animation.
        """
        
        def __init__(self, steps: List[AlgorithmStep], problem: Problem, **kwargs):
            super().__init__(**kwargs)
            self.steps = steps
            self.problem = problem
            self.k = problem.num_bins
            self.colors = [BLUE, GREEN, RED, ORANGE, PURPLE, YELLOW, PINK, TEAL]
        
        def construct(self):
            # Title
            title = Text(f"Bin Packing Algorithm", font_size=36)
            title.to_edge(UP)
            self.play(Write(title))
            
            # Create bins
            bin_width = 1.5
            bin_height = 4
            bin_spacing = 0.3
            total_width = self.k * (bin_width + bin_spacing) - bin_spacing
            start_x = -total_width / 2
            
            bins = VGroup()
            bin_labels = VGroup()
            value_labels = VGroup()
            
            for i in range(self.k):
                x = start_x + i * (bin_width + bin_spacing) + bin_width / 2
                
                # Bin rectangle
                bin_rect = Rectangle(
                    width=bin_width,
                    height=bin_height,
                    stroke_color=self.colors[i % len(self.colors)],
                    fill_opacity=0.1
                )
                bin_rect.move_to([x, 0, 0])
                bins.add(bin_rect)
                
                # Label
                label = Text(f"Bin {i}", font_size=18)
                label.next_to(bin_rect, DOWN)
                bin_labels.add(label)
                
                # Value label
                val_label = Text("V: 0.0", font_size=14)
                val_label.next_to(bin_rect, UP)
                value_labels.add(val_label)
            
            self.play(
                *[Create(b) for b in bins],
                *[Write(l) for l in bin_labels],
                *[Write(v) for v in value_labels]
            )
            
            # Animate steps
            for step in self.steps[:50]:  # Limit steps for animation
                if step.bins_state is None:
                    continue
                
                # Update value labels
                new_labels = []
                for i, bs in enumerate(step.bins_state):
                    new_text = f"V: {bs['current_value']:.1f}"
                    new_label = Text(new_text, font_size=14)
                    new_label.move_to(value_labels[i].get_center())
                    new_labels.append(new_label)
                
                # Step description
                step_text = Text(step.action[:50], font_size=16)
                step_text.to_edge(DOWN)
                
                self.play(
                    *[Transform(value_labels[i], new_labels[i]) for i in range(self.k)],
                    Write(step_text),
                    run_time=0.5
                )
                
                self.wait(0.3)
                self.play(FadeOut(step_text), run_time=0.2)
            
            self.wait(1)
    
    
    class ConvergenceScene(Scene):
        """
        Manim scene for convergence animation.
        """
        
        def __init__(self, history: List[float], algorithm_name: str, **kwargs):
            super().__init__(**kwargs)
            self.history = history
            self.algorithm_name = algorithm_name
        
        def construct(self):
            # Title
            title = Text(f"{self.algorithm_name} Convergence", font_size=32)
            title.to_edge(UP)
            self.play(Write(title))
            
            # Create axes
            axes = Axes(
                x_range=[0, len(self.history), len(self.history)//10],
                y_range=[0, max(self.history) * 1.1, max(self.history)//5],
                x_length=10,
                y_length=5,
                axis_config={"include_tip": True}
            )
            axes.center()
            
            x_label = Text("Iteration", font_size=18)
            x_label.next_to(axes.x_axis, DOWN)
            
            y_label = Text("Objective", font_size=18)
            y_label.next_to(axes.y_axis, LEFT)
            y_label.rotate(PI/2)
            
            self.play(Create(axes), Write(x_label), Write(y_label))
            
            # Animate convergence line
            points = [
                axes.c2p(i, v) for i, v in enumerate(self.history)
            ]
            
            line = VMobject()
            line.set_points_smoothly(points[:1])
            line.set_color(BLUE)
            
            self.add(line)
            
            for i in range(1, len(points), max(1, len(points)//100)):
                new_line = VMobject()
                new_line.set_points_smoothly(points[:i+1])
                new_line.set_color(BLUE)
                self.play(
                    Transform(line, new_line),
                    run_time=0.05
                )
            
            # Final value
            final_val = Text(f"Final: {self.history[-1]:.2f}", font_size=24, color=GREEN)
            final_val.to_edge(RIGHT)
            self.play(Write(final_val))
            
            self.wait(2)
