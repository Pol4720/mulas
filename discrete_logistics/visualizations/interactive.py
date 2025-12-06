"""
Interactive dashboard components for algorithm visualization.
"""

from typing import List, Dict, Optional, Callable, Any
import numpy as np

from ..core.problem import Problem, Solution
from ..algorithms.base import AlgorithmStep


class InteractiveDashboard:
    """
    Base class for interactive dashboard components.
    
    Provides building blocks for Streamlit/Dash interfaces.
    """
    
    def __init__(self):
        self.current_step = 0
        self.steps: List[AlgorithmStep] = []
        self.problem: Optional[Problem] = None
        self.solution: Optional[Solution] = None
    
    def set_data(
        self,
        problem: Problem,
        solution: Solution,
        steps: List[AlgorithmStep]
    ):
        """Set the data for visualization."""
        self.problem = problem
        self.solution = solution
        self.steps = steps
        self.current_step = 0
    
    def get_step_data(self, step_index: int) -> Dict:
        """Get data for a specific step."""
        if not self.steps or step_index >= len(self.steps):
            return {}
        
        step = self.steps[step_index]
        return {
            "step_number": step.step_number,
            "action": step.action,
            "bins_state": step.bins_state,
            "metrics": step.metrics,
            "item": step.item.to_dict() if step.item else None,
            "bin_id": step.bin_id,
            "extra_data": step.extra_data
        }
    
    def get_metrics_summary(self) -> Dict:
        """Get current metrics summary."""
        if not self.solution:
            return {}
        
        return self.solution.get_metrics()
    
    def get_comparison_data(self, solutions: List[Solution]) -> List[Dict]:
        """Get comparison data for multiple solutions."""
        return [s.get_metrics() for s in solutions]
    
    def next_step(self) -> bool:
        """Move to next step."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            return True
        return False
    
    def prev_step(self) -> bool:
        """Move to previous step."""
        if self.current_step > 0:
            self.current_step -= 1
            return True
        return False
    
    def reset(self):
        """Reset to first step."""
        self.current_step = 0
    
    def goto_step(self, step: int):
        """Go to specific step."""
        if 0 <= step < len(self.steps):
            self.current_step = step


class DashboardConfig:
    """Configuration for dashboard appearance and behavior."""
    
    # Theme settings
    LIGHT_THEME = {
        "bg_color": "#FFFFFF",
        "text_color": "#333333",
        "primary_color": "#2E86AB",
        "secondary_color": "#A23B72",
        "success_color": "#28A745",
        "warning_color": "#F18F01",
        "danger_color": "#C73E1D",
        "chart_template": "plotly_white"
    }
    
    DARK_THEME = {
        "bg_color": "#1E1E1E",
        "text_color": "#E0E0E0",
        "primary_color": "#4FC3F7",
        "secondary_color": "#F06292",
        "success_color": "#81C784",
        "warning_color": "#FFB74D",
        "danger_color": "#E57373",
        "chart_template": "plotly_dark"
    }
    
    # Animation settings
    ANIMATION_SPEED = {
        "slow": 1000,
        "normal": 500,
        "fast": 200
    }
    
    @classmethod
    def get_theme(cls, dark_mode: bool = False) -> Dict:
        """Get theme configuration."""
        return cls.DARK_THEME if dark_mode else cls.LIGHT_THEME
    
    @classmethod
    def get_css(cls, dark_mode: bool = False) -> str:
        """Get CSS styles for the dashboard."""
        theme = cls.get_theme(dark_mode)
        
        return f"""
        <style>
            .main {{
                background-color: {theme['bg_color']};
                color: {theme['text_color']};
            }}
            .metric-card {{
                background-color: {theme['bg_color']};
                border: 1px solid {theme['primary_color']};
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: {theme['primary_color']};
            }}
            .metric-label {{
                font-size: 0.9em;
                color: {theme['text_color']};
                opacity: 0.8;
            }}
            .step-indicator {{
                background-color: {theme['secondary_color']};
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                display: inline-block;
            }}
            .algorithm-card {{
                background: linear-gradient(135deg, {theme['primary_color']}20, {theme['secondary_color']}20);
                border-radius: 15px;
                padding: 20px;
                margin: 10px 0;
            }}
            .bin-visualization {{
                display: flex;
                justify-content: center;
                gap: 20px;
                margin: 20px 0;
            }}
            .bin-container {{
                border: 2px solid {theme['primary_color']};
                border-radius: 10px;
                padding: 10px;
                min-width: 100px;
                text-align: center;
            }}
            .progress-bar {{
                height: 10px;
                background-color: {theme['primary_color']}30;
                border-radius: 5px;
                overflow: hidden;
            }}
            .progress-fill {{
                height: 100%;
                background-color: {theme['primary_color']};
                transition: width 0.3s ease;
            }}
        </style>
        """


def create_metric_card(label: str, value: Any, delta: Optional[float] = None) -> str:
    """Create HTML for a metric card."""
    delta_html = ""
    if delta is not None:
        color = "green" if delta >= 0 else "red"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<span style="color: {color}; font-size: 0.8em;">{arrow} {abs(delta):.2f}</span>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def create_bin_visualization(bins_state: List[Dict], capacity: float) -> str:
    """Create HTML visualization of bins."""
    html_parts = ['<div class="bin-visualization">']
    
    for bs in bins_state:
        fill_pct = (bs["current_weight"] / capacity) * 100
        html_parts.append(f"""
        <div class="bin-container">
            <div style="font-weight: bold;">Bin {bs['id']}</div>
            <div>Items: {len(bs['items'])}</div>
            <div>Value: {bs['current_value']:.1f}</div>
            <div>Weight: {bs['current_weight']:.1f}/{capacity:.1f}</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {min(fill_pct, 100):.1f}%;"></div>
            </div>
        </div>
        """)
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)


def create_step_timeline(steps: List[AlgorithmStep], current: int) -> str:
    """Create HTML timeline of steps."""
    html_parts = ['<div style="display: flex; overflow-x: auto; gap: 10px; padding: 10px;">']
    
    for i, step in enumerate(steps[:50]):  # Limit display
        is_current = i == current
        bg_color = "#2E86AB" if is_current else "#E0E0E0"
        text_color = "white" if is_current else "black"
        
        html_parts.append(f"""
        <div style="
            min-width: 60px;
            padding: 5px 10px;
            background-color: {bg_color};
            color: {text_color};
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
        " title="{step.action}">
            {step.step_number}
        </div>
        """)
    
    if len(steps) > 50:
        html_parts.append(f'<div style="padding: 5px;">...+{len(steps)-50} more</div>')
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)
