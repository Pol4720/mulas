"""
Shared Utilities for Dashboard Pages
====================================

Common imports, session state initialization, and helper functions
shared across all dashboard pages.
"""

import streamlit as st
import numpy as np
import time
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add package root to path for absolute imports
_dashboard_dir = Path(__file__).parent
_pkg_root = _dashboard_dir.parent
_workspace_root = _pkg_root.parent

if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))
if str(_pkg_root.parent) not in sys.path:
    sys.path.insert(0, str(_pkg_root.parent))

# Core imports
from discrete_logistics.dashboard.components import (
    ThemeManager,
    ProblemConfigurator,
    AlgorithmSelector,
    ResultsDisplay,
    VisualizationPanel,
    ExportManager
)

from discrete_logistics.dashboard.modern_components import (
    ModernTheme,
    ModernMetricCard,
    ModernCharts,
    ModernProgress
)

from discrete_logistics.dashboard.modern_animations import (
    ModernAnimations,
    LottieAnimations,
    GlassmorphicCards,
    InteractiveChartEnhancements
)

from discrete_logistics.dashboard.interactive_components import (
    InteractiveTooltips,
    SolutionComparator,
    ExecutionTimeline,
    ParameterTuner,
    InteractiveBinVisualizer,
    QuickStatsDashboard
)

from discrete_logistics.core.problem import Problem, Solution, Item, Bin
from discrete_logistics.core.instance_generator import InstanceGenerator
from discrete_logistics.algorithms.greedy import (
    FirstFitDecreasing, 
    BestFitDecreasing, 
    WorstFitDecreasing, 
    RoundRobinGreedy
)
from discrete_logistics.algorithms.metaheuristics import (
    SimulatedAnnealing, 
    GeneticAlgorithm, 
    TabuSearch
)
from discrete_logistics.algorithms.branch_and_bound import BranchAndBound
from discrete_logistics.algorithms.dynamic_programming import DynamicProgramming


def init_session_state():
    """Initialize session state variables."""
    if 'current_problem' not in st.session_state:
        st.session_state['current_problem'] = None
    if 'results' not in st.session_state:
        st.session_state['results'] = {}
    if 'convergence_history' not in st.session_state:
        st.session_state['convergence_history'] = {}
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'light'
    if 'show_confetti' not in st.session_state:
        st.session_state['show_confetti'] = False
    if 'animation_enabled' not in st.session_state:
        st.session_state['animation_enabled'] = True


def apply_custom_styles():
    """Apply modern CSS styles to the page."""
    ModernTheme.apply_modern_css()
    ThemeManager.apply_theme('light')
    
    # Additional navigation styling for multipage
    st.markdown("""
    <style>
        /* Sidebar navigation styling */
        [data-testid="stSidebarNav"] {
            background: linear-gradient(180deg, rgba(79, 70, 229, 0.05) 0%, transparent 100%);
            padding-top: 1rem;
        }
        
        [data-testid="stSidebarNav"] li {
            margin: 0.25rem 0.5rem;
        }
        
        [data-testid="stSidebarNav"] a {
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stSidebarNav"] a:hover {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%) !important;
            transform: translateX(4px);
        }
        
        [data-testid="stSidebarNav"] a[aria-selected="true"] {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
            color: white !important;
        }
        
        [data-testid="stSidebarNav"] span {
            font-weight: 600 !important;
        }
        
        /* Hide default sidebar title */
        [data-testid="stSidebarNav"]::before {
            content: "";
            display: block;
            height: 0;
        }
        
        /* Page container animation */
        .main .block-container {
            animation: fadeIn 0.5s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar_info():
    """Render sidebar information panel."""
    with st.sidebar:
        # Logo header
        st.markdown("""
        <div style="text-align: center; padding: 16px 0 24px;">
            <div style="
                font-size: 3rem; 
                margin-bottom: 8px;
                animation: float 3s ease-in-out infinite;
            ">📦</div>
            <div style="
                font-weight: 800; 
                font-size: 1.1rem; 
                background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">
                Multi-Bin Packing
            </div>
            <div style="font-size: 0.7rem; color: #64748B; margin-top: 4px;">
                Solucionador v2.0
            </div>
        </div>
        <style>
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-6px); }
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📈 Estado Actual")
        
        if st.session_state.get('current_problem'):
            prob = st.session_state['current_problem']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="
                    background: rgba(79, 70, 229, 0.1);
                    border-radius: 10px;
                    padding: 10px;
                    text-align: center;
                ">
                    <div style="font-size: 1.25rem; font-weight: 700; color: #4F46E5;">{prob.n_items}</div>
                    <div style="font-size: 0.65rem; color: #64748B;">Ítems</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="
                    background: rgba(16, 185, 129, 0.1);
                    border-radius: 10px;
                    padding: 10px;
                    text-align: center;
                ">
                    <div style="font-size: 1.25rem; font-weight: 700; color: #10B981;">{prob.num_bins}</div>
                    <div style="font-size: 0.65rem; color: #64748B;">Bins</div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.get('results'):
                best = min(st.session_state['results'].values(), 
                          key=lambda x: x.get('objective', float('inf')))
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
                    border-radius: 10px;
                    padding: 12px;
                    text-align: center;
                    margin-top: 10px;
                ">
                    <div style="font-size: 0.65rem; color: #64748B;">🏆 Mejor</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #EC4899;">{best.get('objective', 0):.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("_Sin instancia activa_")
        
        st.markdown("---")
        
        # About
        with st.expander("ℹ️ Acerca de"):
            st.markdown("""
            **Multi-Bin Packing Solver**
            
            Herramienta para el problema NP-completo de empaquetado balanceado.
            
            - 🔬 12 algoritmos
            - 📊 Visualizaciones
            - 📚 Teoría incluida
            
            *🎓 Proyecto DAA - UH*
            """)


def create_algorithm_instance(algo_name: str, params: Dict[str, Any]):
    """Create an algorithm instance by name."""
    # Import LargestDifferenceFirst
    from discrete_logistics.algorithms.greedy import LargestDifferenceFirst
    
    algorithm_map = {
        'FirstFitDecreasing': FirstFitDecreasing,
        'BestFitDecreasing': BestFitDecreasing,
        'WorstFitDecreasing': WorstFitDecreasing,
        'RoundRobinGreedy': RoundRobinGreedy,
        'LargestDifferenceFirst': LargestDifferenceFirst,
        'SimulatedAnnealing': lambda: SimulatedAnnealing(**params),
        'GeneticAlgorithm': lambda: GeneticAlgorithm(**params),
        'TabuSearch': lambda: TabuSearch(**params),
        'BranchAndBound': lambda: BranchAndBound(**params) if params else BranchAndBound(),
        'DynamicProgramming': DynamicProgramming,
    }
    
    if algo_name in algorithm_map:
        creator = algorithm_map[algo_name]
        if callable(creator) and not isinstance(creator, type):
            return creator()
        return creator()
    
    return None


def calculate_objective(solution: Solution) -> float:
    """Calculate the objective value (max value difference)."""
    if not solution or not solution.bins:
        return float('inf')
    
    bin_values = [sum(item.value for item in bin_obj.items) for bin_obj in solution.bins]
    
    if not bin_values:
        return 0.0
    
    return max(bin_values) - min(bin_values)


def calculate_balance_score(solution: Solution) -> float:
    """Calculate balance score (0 = worst, 1 = perfect)."""
    if not solution or not solution.bins:
        return 0.0
    
    bin_values = [sum(item.value for item in bin_obj.items) for bin_obj in solution.bins]
    
    if not bin_values or max(bin_values) == 0:
        return 1.0
    
    mean_value = np.mean(bin_values)
    std_value = np.std(bin_values)
    
    if mean_value == 0:
        return 1.0
    
    cv = std_value / mean_value
    return max(0, 1 - cv)


def check_feasibility(solution: Solution, problem: Problem) -> bool:
    """Check if solution respects capacity constraints."""
    if not solution or not solution.bins:
        return False
    
    for i, bin_obj in enumerate(solution.bins):
        total_weight = sum(item.weight for item in bin_obj.items)
        capacity = problem.bin_capacities[i] if i < len(problem.bin_capacities) else problem.bin_capacities[0]
        if total_weight > capacity:
            return False
    
    return True


# Export all utilities
__all__ = [
    'init_session_state',
    'apply_custom_styles',
    'render_sidebar_info',
    'create_algorithm_instance',
    'calculate_objective',
    'calculate_balance_score',
    'check_feasibility',
    'ModernTheme',
    'ModernMetricCard',
    'ModernCharts',
    'ModernProgress',
    'ModernAnimations',
    'LottieAnimations',
    'GlassmorphicCards',
    'InteractiveChartEnhancements',
    'InteractiveTooltips',
    'SolutionComparator',
    'ExecutionTimeline',
    'ParameterTuner',
    'InteractiveBinVisualizer',
    'QuickStatsDashboard',
    'ProblemConfigurator',
    'AlgorithmSelector',
    'ResultsDisplay',
    'VisualizationPanel',
    'ExportManager',
    'Problem',
    'Solution',
    'Item',
    'Bin',
    'InstanceGenerator',
    'FirstFitDecreasing',
    'BestFitDecreasing',
    'WorstFitDecreasing',
    'RoundRobinGreedy',
    'SimulatedAnnealing',
    'GeneticAlgorithm',
    'TabuSearch',
    'BranchAndBound',
    'DynamicProgramming',
]
