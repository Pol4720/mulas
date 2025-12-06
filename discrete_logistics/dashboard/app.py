"""
Main Streamlit Application
=========================

This is the main entry point for the Balanced Multi-Bin Packing
interactive dashboard.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import time
from typing import Dict, Any, Optional

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Balanced Multi-Bin Packing",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import components
import sys
from pathlib import Path

# Add package root to path for absolute imports
_pkg_root = Path(__file__).parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))
if str(_pkg_root.parent) not in sys.path:
    sys.path.insert(0, str(_pkg_root.parent))

from discrete_logistics.dashboard.components import (
    ThemeManager,
    ProblemConfigurator,
    AlgorithmSelector,
    ResultsDisplay,
    VisualizationPanel,
    ExportManager
)

from discrete_logistics.core.problem import Problem, Solution, Item, Bin
from discrete_logistics.core.instance_generator import InstanceGenerator
from discrete_logistics.algorithms import AlgorithmRegistry
from discrete_logistics.algorithms.greedy import FirstFitDecreasing, BestFitDecreasing, WorstFitDecreasing, RoundRobinGreedy
from discrete_logistics.algorithms.metaheuristics import SimulatedAnnealing, GeneticAlgorithm, TabuSearch
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
        st.session_state['theme'] = 'dark'


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=BinPacking", use_container_width=True)
        st.markdown("# 📦 Multi-Bin Packing")
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            options=['🏠 Home', '🔬 Solver', '📊 Analysis', '📚 Theory', '⚙️ Settings'],
            label_visibility='collapsed'
        )
        
        st.markdown("---")
        
        # Quick settings
        st.markdown("### Quick Settings")
        theme = st.selectbox(
            "Theme",
            options=['Dark', 'Light'],
            index=0 if st.session_state['theme'] == 'dark' else 1
        )
        st.session_state['theme'] = theme.lower()
        
        # Apply theme
        ThemeManager.apply_theme(st.session_state['theme'])
        
        st.markdown("---")
        
        # Info section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Balanced Multi-Bin Packing**
            
            An interactive tool for solving the 
            NP-hard bin packing problem with 
            balance constraints.
            
            Features:
            - Multiple algorithms
            - Real-time visualization
            - Benchmark analysis
            
            *DAA Project - 2024*
            """)
        
        return page


def render_home_page():
    """Render the home page."""
    st.markdown("""
    # 🏠 Welcome to Balanced Multi-Bin Packing Solver
    
    This interactive dashboard allows you to explore and solve the 
    **Balanced Multi-Bin Packing with Capacity Constraints** problem.
    
    ## 📋 Problem Description
    
    Given:
    - A set of **n items**, each with a weight and value
    - **k bins** with capacity C
    
    Objective:
    - Minimize the **maximum difference** in total values between bins
    - While respecting **capacity constraints**
    
    ## 🎯 Features
    
    | Feature | Description |
    |---------|-------------|
    | 🔬 Multiple Algorithms | Greedy, Metaheuristics, Exact methods |
    | 📊 Visualizations | Interactive charts and animations |
    | 📈 Benchmarking | Compare algorithm performance |
    | 📚 Theory | Mathematical formalization and proofs |
    
    ## 🚀 Getting Started
    
    1. Navigate to the **Solver** page
    2. Configure your problem instance
    3. Select algorithms to run
    4. Analyze the results!
    
    ---
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Algorithms", "9+", help="Available algorithms")
    with col2:
        st.metric("Complexity", "NP-Hard", help="Problem complexity class")
    with col3:
        st.metric("Max Items", "100", help="Supported item count")
    with col4:
        st.metric("Visualizations", "5+", help="Chart types available")


def render_solver_page(problem_config: ProblemConfigurator, 
                       algo_selector: AlgorithmSelector,
                       results_display: ResultsDisplay,
                       viz_panel: VisualizationPanel):
    """Render the main solver page."""
    st.markdown("# 🔬 Problem Solver")
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Problem configuration
        problem = problem_config.render()
        
        if problem:
            problem_config.render_problem_summary(problem)
    
    with col2:
        # Algorithm selection
        algorithm_configs = algo_selector.render()
        
        # Run button
        if st.button("▶️ Run Algorithms", type="primary", use_container_width=True):
            if problem is None:
                st.error("Please generate a problem instance first!")
            elif not algorithm_configs:
                st.error("Please select at least one algorithm!")
            else:
                run_algorithms(problem, algorithm_configs)
    
    # Results section
    st.markdown("---")
    
    if st.session_state.get('results'):
        results_display.render_results(st.session_state['results'])
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.get('convergence_history'):
                viz_panel.render_convergence_plot(st.session_state['convergence_history'])
        
        with col2:
            viz_panel.render_performance_radar(st.session_state['results'])
        
        # Export options
        ExportManager.render_export_buttons(st.session_state['results'])


def run_algorithms(problem: Problem, algorithm_configs: list):
    """Execute selected algorithms on the problem."""
    results = {}
    convergence_history = {}
    
    progress_bar = st.progress(0, text="Running algorithms...")
    
    for idx, (algo_name, params) in enumerate(algorithm_configs):
        progress_bar.progress(
            (idx + 1) / len(algorithm_configs),
            text=f"Running {algo_name}..."
        )
        
        try:
            # Create algorithm instance
            algorithm = create_algorithm_instance(algo_name, params)
            
            if algorithm is None:
                st.warning(f"Algorithm {algo_name} not available")
                continue
            
            # Run algorithm
            start_time = time.time()
            solution = algorithm.solve(problem)
            end_time = time.time()
            
            # Calculate metrics
            objective = calculate_objective(solution)
            balance_score = calculate_balance_score(solution)
            feasible = check_feasibility(solution, problem)
            
            results[algo_name] = {
                'solution': solution,
                'objective': objective,
                'balance_score': balance_score,
                'time': end_time - start_time,
                'feasible': feasible,
                'stability': 0.9  # Placeholder
            }
            
            # Store convergence history if available
            if hasattr(algorithm, 'history'):
                convergence_history[algo_name] = algorithm.history
                
        except Exception as e:
            st.error(f"Error running {algo_name}: {str(e)}")
            results[algo_name] = {
                'objective': float('inf'),
                'time': 0,
                'feasible': False,
                'error': str(e)
            }
    
    progress_bar.progress(1.0, text="Complete!")
    
    st.session_state['results'] = results
    st.session_state['convergence_history'] = convergence_history
    
    st.success(f"✅ Completed {len(results)} algorithm(s)")


def create_algorithm_instance(algo_name: str, params: Dict[str, Any]):
    """Create an algorithm instance by name."""
    algorithm_map = {
        'FirstFitDecreasing': FirstFitDecreasing,
        'BestFitDecreasing': BestFitDecreasing,
        'WorstFitDecreasing': WorstFitDecreasing,
        'RoundRobinGreedy': RoundRobinGreedy,
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
    
    # Coefficient of variation inverted
    if mean_value == 0:
        return 1.0
    
    cv = std_value / mean_value
    return max(0, 1 - cv)


def check_feasibility(solution: Solution, problem: Problem) -> bool:
    """Check if solution respects capacity constraints."""
    if not solution or not solution.bins:
        return False
    
    for bin_obj in solution.bins:
        total_weight = sum(item.weight for item in bin_obj.items)
        if total_weight > problem.bin_capacity:
            return False
    
    return True


def render_analysis_page(viz_panel: VisualizationPanel):
    """Render the analysis page."""
    st.markdown("# 📊 Results Analysis")
    
    if not st.session_state.get('results'):
        st.info("No results to analyze. Run some algorithms first!")
        return
    
    results = st.session_state['results']
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "🔄 Convergence", "📊 Comparison"])
    
    with tab1:
        st.markdown("### Algorithm Performance Metrics")
        
        # Create performance dataframe
        import pandas as pd
        
        perf_data = []
        for algo, result in results.items():
            perf_data.append({
                'Algorithm': algo,
                'Objective': result.get('objective', '-'),
                'Time (s)': f"{result.get('time', 0):.4f}",
                'Balance': f"{result.get('balance_score', 0):.2%}",
                'Feasible': '✅' if result.get('feasible') else '❌'
            })
        
        df = pd.DataFrame(perf_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Performance chart
        viz_panel.render_performance_radar(results)
    
    with tab2:
        st.markdown("### Convergence Analysis")
        
        if st.session_state.get('convergence_history'):
            viz_panel.render_convergence_plot(st.session_state['convergence_history'])
        else:
            st.info("No convergence data available. Run metaheuristic algorithms to see convergence.")
    
    with tab3:
        st.markdown("### Algorithm Comparison")
        
        # Create comparison visualizations
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        algos = list(results.keys())
        objectives = [results[a].get('objective', 0) for a in algos]
        times = [results[a].get('time', 0) for a in algos]
        
        fig.add_trace(go.Bar(
            x=algos,
            y=objectives,
            name='Objective Value',
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title='Objective Values by Algorithm',
            xaxis_title='Algorithm',
            yaxis_title='Objective Value',
            template='plotly_dark' if st.session_state['theme'] == 'dark' else 'plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_theory_page():
    """Render the theory page."""
    st.markdown("# 📚 Theoretical Background")
    
    tabs = st.tabs([
        "📐 Formalization",
        "🔢 Complexity", 
        "📝 Algorithms",
        "📖 References"
    ])
    
    with tabs[0]:
        st.markdown("""
        ## Mathematical Formalization
        
        ### Problem Definition
        
        The **Balanced Multi-Bin Packing Problem** can be formally defined as:
        
        **Given:**
        - A set of items $I = \\{1, 2, ..., n\\}$
        - Each item $i$ has weight $w_i$ and value $v_i$
        - $k$ identical bins with capacity $C$
        
        **Decision Variables:**
        - $x_{ij} \\in \\{0, 1\\}$: 1 if item $i$ is assigned to bin $j$
        - $z$: makespan (maximum total value in any bin)
        
        ### ILP Formulation
        
        $$\\min z$$
        
        Subject to:
        
        $$\\sum_{j=1}^{k} x_{ij} = 1 \\quad \\forall i \\in I$$
        
        $$\\sum_{i=1}^{n} w_i \\cdot x_{ij} \\leq C \\quad \\forall j = 1,...,k$$
        
        $$\\sum_{i=1}^{n} v_i \\cdot x_{ij} \\leq z \\quad \\forall j = 1,...,k$$
        
        $$x_{ij} \\in \\{0, 1\\}, z \\geq 0$$
        """)
    
    with tabs[1]:
        st.markdown("""
        ## Complexity Analysis
        
        ### NP-Hardness Proof
        
        The Balanced Multi-Bin Packing Problem is **NP-hard**.
        
        **Proof:** By reduction from 3-PARTITION.
        
        Given an instance of 3-PARTITION with integers $a_1, ..., a_{3m}$ 
        and target $B = \\frac{1}{m}\\sum a_i$, we construct a bin packing instance:
        
        1. Create item $i$ with weight $w_i = v_i = a_i$
        2. Set $k = m$ bins with capacity $C = B$
        
        A valid 3-PARTITION exists if and only if we can achieve 
        objective value 0 (perfect balance).
        
        ### Algorithm Complexities
        
        | Algorithm | Time | Space |
        |-----------|------|-------|
        | FFD | $O(n \\log n)$ | $O(n)$ |
        | BFD | $O(n^2)$ | $O(n)$ |
        | Simulated Annealing | $O(I \\cdot n)$ | $O(n)$ |
        | Genetic Algorithm | $O(G \\cdot P \\cdot n)$ | $O(P \\cdot n)$ |
        | Branch & Bound | $O(k^n)$ worst | $O(n)$ |
        """)
    
    with tabs[2]:
        st.markdown("""
        ## Algorithm Descriptions
        
        ### Greedy Algorithms
        
        **First Fit Decreasing (FFD):**
        1. Sort items by weight (descending)
        2. For each item, place in first bin that fits
        3. Complexity: $O(n \\log n)$
        
        **Best Fit Decreasing (BFD):**
        1. Sort items by weight (descending)
        2. For each item, place in bin with minimum remaining space
        3. Provides tighter packing
        
        ### Metaheuristics
        
        **Simulated Annealing:**
        - Probabilistic local search
        - Accepts worse solutions with probability $e^{-\\Delta/T}$
        - Temperature decreases over time (cooling schedule)
        
        **Genetic Algorithm:**
        - Population-based evolutionary approach
        - Uses selection, crossover, and mutation operators
        - Explores diverse solution space
        """)
    
    with tabs[3]:
        st.markdown("""
        ## References
        
        1. Garey, M.R., & Johnson, D.S. (1979). *Computers and Intractability: 
           A Guide to the Theory of NP-Completeness*. W.H. Freeman.
        
        2. Martello, S., & Toth, P. (1990). *Knapsack Problems: Algorithms 
           and Computer Implementations*. Wiley.
        
        3. Coffman, E.G., Garey, M.R., & Johnson, D.S. (1996). 
           "Approximation algorithms for bin packing: A survey." 
           *Approximation Algorithms for NP-hard Problems*, 46-93.
        
        4. Kirkpatrick, S., Gelatt, C.D., & Vecchi, M.P. (1983). 
           "Optimization by simulated annealing." *Science*, 220(4598), 671-680.
        
        5. Glover, F. (1986). "Future paths for integer programming and 
           links to artificial intelligence." *Computers & Operations Research*.
        """)


def render_settings_page():
    """Render the settings page."""
    st.markdown("# ⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Appearance")
        
        theme = st.selectbox(
            "Color Theme",
            options=['Dark', 'Light'],
            index=0 if st.session_state['theme'] == 'dark' else 1
        )
        st.session_state['theme'] = theme.lower()
        
        st.markdown("### Performance")
        
        max_iterations = st.number_input(
            "Default Max Iterations",
            min_value=100,
            max_value=100000,
            value=10000
        )
        
        time_limit = st.number_input(
            "Default Time Limit (seconds)",
            min_value=1,
            max_value=300,
            value=60
        )
    
    with col2:
        st.markdown("### Data")
        
        if st.button("🗑️ Clear Results"):
            st.session_state['results'] = {}
            st.session_state['convergence_history'] = {}
            st.success("Results cleared!")
        
        if st.button("🔄 Reset Problem"):
            st.session_state['current_problem'] = None
            st.success("Problem reset!")
        
        st.markdown("### Export")
        
        if st.button("📤 Export All Data"):
            st.info("Export functionality - coming soon!")
    
    st.markdown("---")
    st.markdown("""
    ### System Information
    
    - **Version:** 0.1.0
    - **Python:** 3.11+
    - **Streamlit:** Latest
    """)


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Create component instances
    theme = st.session_state['theme']
    problem_config = ProblemConfigurator(theme=theme)
    algo_selector = AlgorithmSelector(theme=theme)
    results_display = ResultsDisplay(theme=theme)
    viz_panel = VisualizationPanel(theme=theme)
    
    # Render sidebar and get current page
    page = render_sidebar()
    
    # Render appropriate page
    if page == '🏠 Home':
        render_home_page()
    elif page == '🔬 Solver':
        render_solver_page(problem_config, algo_selector, results_display, viz_panel)
    elif page == '📊 Analysis':
        render_analysis_page(viz_panel)
    elif page == '📚 Theory':
        render_theory_page()
    elif page == '⚙️ Settings':
        render_settings_page()


if __name__ == "__main__":
    main()
