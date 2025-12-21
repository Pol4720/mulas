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
    theme: str = "light"
    primary_color: str = "#4F46E5"
    secondary_color: str = "#10B981"
    success_color: str = "#22C55E"
    warning_color: str = "#F59E0B"
    chart_height: int = 400
    animation_speed: float = 0.5
    

class ThemeManager:
    """Manages dashboard theme (light mode only)."""
    
    LIGHT_THEME = {
        'bg_color': '#F8FAFC',
        'card_bg': '#FFFFFF',
        'text_color': '#1E293B',
        'secondary_text': '#64748B',
        'accent_color': '#4F46E5',
        'accent_light': '#818CF8',
        'success': '#22C55E',
        'warning': '#F59E0B',
        'error': '#EF4444',
        'border': '#E2E8F0',
        'plotly_template': 'plotly_white',
        'gradient_start': '#4F46E5',
        'gradient_end': '#7C3AED'
    }
    
    @classmethod
    def get_theme(cls, theme_name: str = "light") -> Dict[str, str]:
        """Get theme configuration (always returns light theme)."""
        return cls.LIGHT_THEME
    
    @classmethod
    def apply_theme(cls, theme_name: str = "light"):
        """Apply CSS styling for light mode with modern animations."""
        theme = cls.get_theme(theme_name)
        st.markdown(f"""
        <style>
            /* ============================================ */
            /* Global Styles */
            /* ============================================ */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            .stApp {{
                background: linear-gradient(135deg, {theme['bg_color']} 0%, #EEF2FF 100%);
                font-family: 'Inter', sans-serif;
            }}
            
            /* ============================================ */
            /* Header Animations */
            /* ============================================ */
            h1, h2, h3 {{
                background: linear-gradient(135deg, {theme['gradient_start']} 0%, {theme['gradient_end']} 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                animation: fadeInDown 0.6s ease-out;
            }}
            
            @keyframes fadeInDown {{
                from {{
                    opacity: 0;
                    transform: translateY(-20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            
            /* ============================================ */
            /* Card Styles */
            /* ============================================ */
            .metric-card {{
                background: {theme['card_bg']};
                padding: 24px;
                border-radius: 16px;
                margin: 12px 0;
                border: 1px solid {theme['border']};
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                animation: slideInUp 0.5s ease-out;
            }}
            
            .metric-card:hover {{
                transform: translateY(-4px);
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                border-color: {theme['accent_light']};
            }}
            
            @keyframes slideInUp {{
                from {{
                    opacity: 0;
                    transform: translateY(30px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            
            /* ============================================ */
            /* Staggered Animation Classes */
            /* ============================================ */
            .stagger-1 {{ animation-delay: 0.1s; }}
            .stagger-2 {{ animation-delay: 0.2s; }}
            .stagger-3 {{ animation-delay: 0.3s; }}
            .stagger-4 {{ animation-delay: 0.4s; }}
            .stagger-5 {{ animation-delay: 0.5s; }}
            
            /* ============================================ */
            /* Button Styles */
            /* ============================================ */
            .stButton > button {{
                background: linear-gradient(135deg, {theme['accent_color']} 0%, {theme['gradient_end']} 100%);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
            }}
            
            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.5);
            }}
            
            .stButton > button:active {{
                transform: translateY(0);
            }}
            
            /* ============================================ */
            /* Input Styles */
            /* ============================================ */
            .stSlider > div > div {{
                background-color: {theme['accent_color']};
            }}
            
            .stSelectbox > div > div {{
                border-radius: 12px;
                border: 2px solid {theme['border']};
                transition: border-color 0.3s ease;
            }}
            
            .stSelectbox > div > div:focus-within {{
                border-color: {theme['accent_color']};
            }}
            
            .stNumberInput > div > div > input {{
                border-radius: 12px;
                border: 2px solid {theme['border']};
            }}
            
            .stNumberInput > div > div > input:focus {{
                border-color: {theme['accent_color']};
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
            }}
            
            /* ============================================ */
            /* Sidebar Styles */
            /* ============================================ */
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {theme['card_bg']} 0%, {theme['bg_color']} 100%);
                border-right: 1px solid {theme['border']};
            }}
            
            [data-testid="stSidebar"] .block-container {{
                padding-top: 2rem;
            }}
            
            /* ============================================ */
            /* Metric Styles */
            /* ============================================ */
            [data-testid="stMetricValue"] {{
                font-size: 2rem;
                font-weight: 700;
                color: {theme['text_color']};
            }}
            
            [data-testid="stMetricLabel"] {{
                font-size: 0.875rem;
                font-weight: 500;
                color: {theme['secondary_text']};
            }}
            
            [data-testid="stMetricDelta"] {{
                font-size: 0.875rem;
            }}
            
            /* ============================================ */
            /* Expander Styles */
            /* ============================================ */
            .streamlit-expanderHeader {{
                background-color: {theme['card_bg']};
                border-radius: 12px;
                border: 1px solid {theme['border']};
                font-weight: 600;
            }}
            
            .streamlit-expanderContent {{
                background-color: {theme['card_bg']};
                border-radius: 0 0 12px 12px;
                border: 1px solid {theme['border']};
                border-top: none;
            }}
            
            /* ============================================ */
            /* Table Styles */
            /* ============================================ */
            .dataframe {{
                border-radius: 12px !important;
                overflow: hidden;
                border: 1px solid {theme['border']} !important;
            }}
            
            .dataframe th {{
                background: linear-gradient(135deg, {theme['accent_color']} 0%, {theme['gradient_end']} 100%) !important;
                color: white !important;
                font-weight: 600 !important;
            }}
            
            .dataframe td {{
                border-color: {theme['border']} !important;
            }}
            
            .dataframe tr:hover td {{
                background-color: {theme['bg_color']} !important;
            }}
            
            /* ============================================ */
            /* Status Colors */
            /* ============================================ */
            .success-text {{
                color: {theme['success']};
                font-weight: 600;
            }}
            
            .warning-text {{
                color: {theme['warning']};
                font-weight: 600;
            }}
            
            .error-text {{
                color: {theme['error']};
                font-weight: 600;
            }}
            
            /* ============================================ */
            /* Section Headers */
            /* ============================================ */
            .section-header {{
                color: {theme['accent_color']};
                font-size: 1.5em;
                font-weight: 700;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 3px solid {theme['accent_color']};
            }}
            
            /* ============================================ */
            /* Progress Bar Animation */
            /* ============================================ */
            .stProgress > div > div > div {{
                background: linear-gradient(90deg, {theme['accent_color']} 0%, {theme['gradient_end']} 50%, {theme['accent_color']} 100%);
                background-size: 200% 100%;
                animation: shimmer 2s infinite;
            }}
            
            @keyframes shimmer {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}
            
            /* ============================================ */
            /* Spinner Animation */
            /* ============================================ */
            .stSpinner > div {{
                border-top-color: {theme['accent_color']} !important;
            }}
            
            /* ============================================ */
            /* Success/Info/Warning/Error Boxes */
            /* ============================================ */
            .stSuccess {{
                background-color: rgba(34, 197, 94, 0.1);
                border-left: 4px solid {theme['success']};
                border-radius: 0 12px 12px 0;
            }}
            
            .stInfo {{
                background-color: rgba(79, 70, 229, 0.1);
                border-left: 4px solid {theme['accent_color']};
                border-radius: 0 12px 12px 0;
            }}
            
            .stWarning {{
                background-color: rgba(245, 158, 11, 0.1);
                border-left: 4px solid {theme['warning']};
                border-radius: 0 12px 12px 0;
            }}
            
            .stError {{
                background-color: rgba(239, 68, 68, 0.1);
                border-left: 4px solid {theme['error']};
                border-radius: 0 12px 12px 0;
            }}
            
            /* ============================================ */
            /* Tabs Styles */
            /* ============================================ */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
                background-color: {theme['bg_color']};
                border-radius: 12px;
                padding: 4px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 500;
                transition: all 0.3s ease;
            }}
            
            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, {theme['accent_color']} 0%, {theme['gradient_end']} 100%);
                color: white;
            }}
            
            /* ============================================ */
            /* Pulse Animation for Important Elements */
            /* ============================================ */
            @keyframes pulse {{
                0%, 100% {{
                    opacity: 1;
                }}
                50% {{
                    opacity: 0.6;
                }}
            }}
            
            .pulse {{
                animation: pulse 2s infinite;
            }}
            
            /* ============================================ */
            /* Float Animation */
            /* ============================================ */
            @keyframes float {{
                0%, 100% {{
                    transform: translateY(0);
                }}
                50% {{
                    transform: translateY(-10px);
                }}
            }}
            
            .float {{
                animation: float 3s ease-in-out infinite;
            }}
            
            /* ============================================ */
            /* Radio Button Styles */
            /* ============================================ */
            .stRadio > div {{
                gap: 12px;
            }}
            
            .stRadio > div > label {{
                background-color: {theme['card_bg']};
                padding: 12px 20px;
                border-radius: 10px;
                border: 2px solid {theme['border']};
                transition: all 0.3s ease;
            }}
            
            .stRadio > div > label:hover {{
                border-color: {theme['accent_light']};
                background-color: rgba(79, 70, 229, 0.05);
            }}
            
            /* ============================================ */
            /* Code Block Styles */
            /* ============================================ */
            code {{
                background-color: {theme['bg_color']};
                padding: 2px 6px;
                border-radius: 6px;
                font-size: 0.875em;
            }}
            
            pre {{
                background-color: #1E293B !important;
                border-radius: 12px !important;
                padding: 16px !important;
            }}
            
            /* ============================================ */
            /* Enhanced Animations */
            /* ============================================ */
            
            /* Bounce animation for notifications */
            @keyframes bounce {{
                0%, 20%, 50%, 80%, 100% {{
                    transform: translateY(0);
                }}
                40% {{
                    transform: translateY(-12px);
                }}
                60% {{
                    transform: translateY(-6px);
                }}
            }}
            
            .bounce {{
                animation: bounce 1s ease;
            }}
            
            /* Scale in animation for charts */
            @keyframes scaleIn {{
                from {{
                    opacity: 0;
                    transform: scale(0.9);
                }}
                to {{
                    opacity: 1;
                    transform: scale(1);
                }}
            }}
            
            .scale-in {{
                animation: scaleIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            /* Fade slide for sequential content */
            @keyframes fadeSlideRight {{
                from {{
                    opacity: 0;
                    transform: translateX(-20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateX(0);
                }}
            }}
            
            .fade-slide-right {{
                animation: fadeSlideRight 0.4s ease-out forwards;
            }}
            
            /* Glow effect for active elements */
            @keyframes glow {{
                0%, 100% {{
                    box-shadow: 0 0 5px rgba(79, 70, 229, 0.3);
                }}
                50% {{
                    box-shadow: 0 0 20px rgba(79, 70, 229, 0.6);
                }}
            }}
            
            .glow {{
                animation: glow 2s ease-in-out infinite;
            }}
            
            /* Gradient text animation */
            @keyframes gradientShift {{
                0% {{
                    background-position: 0% 50%;
                }}
                50% {{
                    background-position: 100% 50%;
                }}
                100% {{
                    background-position: 0% 50%;
                }}
            }}
            
            .gradient-text-animated {{
                background: linear-gradient(270deg, {theme['gradient_start']}, {theme['gradient_end']}, #EC4899, {theme['gradient_start']});
                background-size: 400% 400%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: gradientShift 8s ease infinite;
            }}
            
            /* Result card reveal */
            @keyframes revealUp {{
                0% {{
                    opacity: 0;
                    transform: translateY(40px);
                    filter: blur(10px);
                }}
                100% {{
                    opacity: 1;
                    transform: translateY(0);
                    filter: blur(0);
                }}
            }}
            
            .reveal-up {{
                animation: revealUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
            }}
            
            /* Counter animation helper */
            @keyframes countUp {{
                from {{
                    opacity: 0;
                }}
                to {{
                    opacity: 1;
                }}
            }}
            
            /* Ripple effect on buttons */
            .stButton > button {{
                position: relative;
                overflow: hidden;
            }}
            
            .stButton > button::after {{
                content: '';
                position: absolute;
                width: 100%;
                height: 100%;
                top: 0;
                left: 0;
                background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
                transform: scale(0);
                opacity: 0;
            }}
            
            .stButton > button:active::after {{
                transform: scale(2);
                opacity: 1;
                transition: transform 0.3s, opacity 0.3s;
            }}
            
            /* Smooth transitions for all interactive elements */
            * {{
                transition-property: background-color, border-color, color, box-shadow, transform;
                transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
                transition-duration: 0.15s;
            }}
            
            /* Chart container animation */
            [data-testid="stPlotlyChart"] {{
                animation: scaleIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            /* DataFrame animation */
            [data-testid="stDataFrame"] {{
                animation: fadeSlideRight 0.4s ease-out;
            }}
            
            /* Loading shimmer effect */
            @keyframes loadingShimmer {{
                0% {{
                    background-position: -200% 0;
                }}
                100% {{
                    background-position: 200% 0;
                }}
            }}
            
            .loading-shimmer {{
                background: linear-gradient(
                    90deg,
                    {theme['bg_color']} 25%,
                    {theme['card_bg']} 50%,
                    {theme['bg_color']} 75%
                );
                background-size: 200% 100%;
                animation: loadingShimmer 1.5s infinite;
            }}
            
            /* Tooltip styles */
            [data-tooltip] {{
                position: relative;
            }}
            
            [data-tooltip]::before {{
                content: attr(data-tooltip);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%) translateY(-8px);
                background: {theme['text_color']};
                color: {theme['card_bg']};
                padding: 6px 12px;
                border-radius: 8px;
                font-size: 0.75rem;
                white-space: nowrap;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.2s, transform 0.2s;
            }}
            
            [data-tooltip]:hover::before {{
                opacity: 1;
                transform: translateX(-50%) translateY(-4px);
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
            'complexity': 'O(k · 3^n)',
            'description': '⚠️ Solo para instancias pequeñas (n≤15). Solución óptima garantizada.',
            'params': {
                'time_limit': (60.0, 5.0, 300.0, 'Límite de tiempo (segundos)')
            },
            'warning': 'Muy lento para n>15 items. Usa Branch & Bound para instancias mayores.'
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
        
        # Quick selection buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            select_all = st.button("✅ Seleccionar Todos", use_container_width=True, type="primary")
        with col2:
            clear_all = st.button("🗑️ Limpiar", use_container_width=True)
        
        # Handle selection state
        if 'selected_algos' not in st.session_state:
            st.session_state.selected_algos = ['FirstFitDecreasing'] if 'FirstFitDecreasing' in available_algorithms else []
        
        if select_all:
            st.session_state.selected_algos = list(available_algorithms.keys())
        if clear_all:
            st.session_state.selected_algos = []
        
        # Filter session state to only include available algorithms
        valid_selection = [a for a in st.session_state.selected_algos if a in available_algorithms]
        
        selected_algorithms = st.multiselect(
            "Seleccionar Algoritmos a Ejecutar",
            options=list(available_algorithms.keys()),
            default=valid_selection if valid_selection else None,
            format_func=lambda x: f"{available_algorithms[x]['name']} ({available_algorithms[x]['category']})",
            key="algo_multiselect"
        )
        
        # Update session state
        st.session_state.selected_algos = selected_algorithms
        
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
        st.markdown("### 📈 Resultados")
        
        if not results:
            st.info("No hay resultados para mostrar. Ejecute los algoritmos primero.")
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
                "Mejor Algoritmo",
                best_algo,
                help="Algoritmo con menor valor objetivo"
            )
        
        with col2:
            st.metric(
                "Mejor Objetivo",
                f"{best_result.get('objective', 0):.4f}",
                help="Mínima diferencia de valor alcanzada"
            )
        
        with col3:
            st.metric(
                "Tiempo de Ejecución",
                f"{best_result.get('time', 0):.3f}s",
                help="Tiempo para encontrar la mejor solución"
            )
    
    def _render_comparison_table(self, results: Dict[str, Any]):
        """Render comparison table of all algorithms."""
        st.markdown("#### Comparación de Algoritmos")
        
        # Add explanation for balance score
        st.info(
            "ℹ️ **Puntuación Balance** = 1 - (desviación estándar / media) de los valores por contenedor. "
            "**1.0 = balance perfecto** (todos los bins tienen el mismo valor), "
            "**0.0 = máximo desbalance**."
        )
        
        data = []
        for algo_name, result in results.items():
            balance = result.get('balance_score', 0)
            data.append({
                'Algoritmo': algo_name,
                'Objetivo (diff)': f"{result.get('objective', 0):.2f}",
                'Balance': f"{balance:.1%}" if isinstance(balance, (int, float)) else '-',
                'Tiempo (s)': f"{result.get('time', 0):.4f}" if isinstance(result.get('time'), (int, float)) else '-',
                'Factible': '✅' if result.get('feasible', False) else '❌'
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Objetivo (diff)')
        
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _render_best_solution(self, results: Dict[str, Any]):
        """Visualize the best solution found."""
        st.markdown("#### Visualización de la Mejor Solución")
        
        best_algo = min(results.keys(), key=lambda x: results[x].get('objective', float('inf')))
        solution = results[best_algo].get('solution')
        
        if solution is None:
            st.warning("No hay solución disponible para visualizar.")
            return
        
        # Create bin visualization
        fig = self._create_bin_visualization(solution)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_bin_visualization(self, solution: Solution) -> go.Figure:
        """Create a visual representation of bin contents."""
        num_bins = len(solution.bins)
        
        # Create figure with single x-axis for all bins side by side
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        # Collect bin info for proper capacity lines
        max_capacity = 0
        for bin_obj in solution.bins:
            cap = getattr(bin_obj, 'capacity', 100)
            if cap > max_capacity:
                max_capacity = cap
        
        for bin_idx, bin_obj in enumerate(solution.bins):
            if not bin_obj.items:
                # Add placeholder for empty bin
                fig.add_trace(
                    go.Bar(
                        x=[f"Cont. {bin_idx + 1}"],
                        y=[0],
                        name=f"Cont. {bin_idx + 1} (vacío)",
                        marker_color='rgba(200, 200, 200, 0.3)',
                        text="Vacío",
                        textposition='inside',
                        hoverinfo='text',
                        showlegend=False
                    )
                )
            else:
                y_offset = 0
                for item_idx, item in enumerate(bin_obj.items):
                    fig.add_trace(
                        go.Bar(
                            x=[f"Cont. {bin_idx + 1}"],
                            y=[item.weight],
                            base=[y_offset],
                            name=item.id,
                            marker_color=colors[item_idx % len(colors)],
                            text=f"{item.id}<br>w={item.weight:.1f}<br>v={item.value:.1f}",
                            textposition='inside',
                            hovertemplate=f"<b>{item.id}</b><br>Peso: {item.weight:.2f}<br>Valor: {item.value:.2f}<extra></extra>",
                            showlegend=False
                        )
                    )
                    y_offset += item.weight
        
        # Add capacity line if available
        bin_capacities = []
        for i, bin_obj in enumerate(solution.bins):
            cap = getattr(bin_obj, 'capacity', max_capacity)
            bin_capacities.append(cap)
        
        # Add capacity indicators
        if max_capacity > 0:
            x_labels = [f"Cont. {i + 1}" for i in range(num_bins)]
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=bin_capacities,
                    mode='lines+markers',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(color='red', size=8),
                    name='Capacidad máx.',
                    hovertemplate="Capacidad: %{y:.1f}<extra></extra>"
                )
            )
        
        # Add summary metrics below
        bin_weights = []
        bin_values = []
        for bin_obj in solution.bins:
            bin_weights.append(sum(item.weight for item in bin_obj.items))
            bin_values.append(sum(item.value for item in bin_obj.items))
        
        fig.update_layout(
            title=dict(
                text="Contenido de Contenedores (Distribución de Peso)",
                font=dict(size=16)
            ),
            template=self.theme['plotly_template'],
            height=450,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            barmode='relative',
            xaxis=dict(title="Contenedor", tickfont=dict(size=12)),
            yaxis=dict(title="Peso Acumulado", tickfont=dict(size=12)),
            annotations=[
                dict(
                    x=f"Cont. {i + 1}",
                    y=bin_weights[i] + max_capacity * 0.05,
                    text=f"v={bin_values[i]:.0f}",
                    showarrow=False,
                    font=dict(size=10, color='#4F46E5')
                ) for i in range(num_bins) if bin_values[i] > 0
            ]
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
        st.markdown("#### Análisis de Convergencia")
        
        fig = go.Figure()
        
        for algo_name, values in history.items():
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode='lines',
                name=algo_name
            ))
        
        fig.update_layout(
            title="Convergencia de Algoritmos",
            xaxis_title="Iteración",
            yaxis_title="Valor Objetivo",
            template=self.theme['plotly_template'],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_radar(self, results: Dict[str, Any]):
        """Render radar chart comparing algorithm performance."""
        st.markdown("#### Comparación de Rendimiento")
        
        metrics = ['Velocidad', 'Calidad', 'Balance', 'Factibilidad', 'Estabilidad']
        
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
        st.markdown("#### Mapa de Calor de Utilización")
        
        # Create utilization matrix
        bin_data = []
        for bin_obj in solution.bins:
            weight_util = sum(item.weight for item in bin_obj.items) / bin_obj.capacity
            value_sum = sum(item.value for item in bin_obj.items)
            item_count = len(bin_obj.items)
            bin_data.append([weight_util, value_sum / 100, item_count / 10])
        
        fig = go.Figure(data=go.Heatmap(
            z=bin_data,
            x=['Util. Peso', 'Suma Valor', 'Cant. Ítems'],
            y=[f"Cont. {i+1}" for i in range(len(solution.bins))],
            colorscale='RdYlGn',
            showscale=True
        ))
        
        fig.update_layout(
            title="Métricas de Utilización de Contenedores",
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
        st.markdown("### 💾 Exportar Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            json_data = ExportManager.export_to_json(results, "results")
            st.download_button(
                label="📄 Descargar JSON",
                data=json_data,
                file_name="resultados.json",
                mime="application/json"
            )
        
        with col2:
            csv_data = ExportManager.export_to_csv(results)
            st.download_button(
                label="📊 Descargar CSV",
                data=csv_data,
                file_name="resultados.csv",
                mime="text/csv"
            )
        
        with col3:
            st.button("📋 Copiar al Portapapeles", disabled=True, help="Próximamente")
