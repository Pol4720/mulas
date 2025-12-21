"""
Modern Components Module
========================

Enhanced Streamlit components with modern UI/UX patterns,
glassmorphism design, and smooth animations.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import json


class ModernTheme:
    """
    Modern theme configuration with glassmorphism and gradients.
    """
    
    # Color palette
    COLORS = {
        'primary': '#4F46E5',
        'primary_light': '#818CF8',
        'primary_dark': '#3730A3',
        'secondary': '#7C3AED',
        'accent': '#EC4899',
        'success': '#10B981',
        'warning': '#F59E0B',
        'error': '#EF4444',
        'info': '#06B6D4',
        'background': '#F8FAFC',
        'surface': '#FFFFFF',
        'text': '#1E293B',
        'text_secondary': '#64748B',
        'border': '#E2E8F0',
    }
    
    # Gradients
    GRADIENTS = {
        'primary': 'linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)',
        'secondary': 'linear-gradient(135deg, #7C3AED 0%, #EC4899 100%)',
        'success': 'linear-gradient(135deg, #10B981 0%, #34D399 100%)',
        'sunset': 'linear-gradient(135deg, #F59E0B 0%, #EF4444 100%)',
        'ocean': 'linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)',
        'aurora': 'linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%)',
    }
    
    @classmethod
    def apply_modern_css(cls):
        """Apply comprehensive modern CSS styling."""
        st.markdown(f"""
        <style>
            /* ============================================ */
            /* CSS Variables for Theming */
            /* ============================================ */
            :root {{
                --primary: {cls.COLORS['primary']};
                --primary-light: {cls.COLORS['primary_light']};
                --secondary: {cls.COLORS['secondary']};
                --accent: {cls.COLORS['accent']};
                --success: {cls.COLORS['success']};
                --warning: {cls.COLORS['warning']};
                --error: {cls.COLORS['error']};
                --background: {cls.COLORS['background']};
                --surface: {cls.COLORS['surface']};
                --text: {cls.COLORS['text']};
                --text-secondary: {cls.COLORS['text_secondary']};
                --border: {cls.COLORS['border']};
                
                --gradient-primary: {cls.GRADIENTS['primary']};
                --gradient-aurora: {cls.GRADIENTS['aurora']};
                
                --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                --shadow-glow: 0 0 20px rgba(79, 70, 229, 0.3);
                
                --radius-sm: 8px;
                --radius-md: 12px;
                --radius-lg: 16px;
                --radius-xl: 24px;
                --radius-full: 9999px;
                
                --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
                --transition-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
                --transition-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            /* ============================================ */
            /* Base Styles */
            /* ============================================ */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            .stApp {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, var(--background) 0%, #EEF2FF 50%, #FDF4FF 100%);
            }}
            
            /* ============================================ */
            /* Glassmorphism Cards */
            /* ============================================ */
            .glass-card {{
                background: rgba(255, 255, 255, 0.7);
                backdrop-filter: blur(16px) saturate(180%);
                -webkit-backdrop-filter: blur(16px) saturate(180%);
                border-radius: var(--radius-xl);
                border: 1px solid rgba(255, 255, 255, 0.5);
                box-shadow: var(--shadow-lg);
                padding: 24px;
                transition: all var(--transition-normal);
            }}
            
            .glass-card:hover {{
                transform: translateY(-4px);
                box-shadow: var(--shadow-xl), var(--shadow-glow);
                border-color: var(--primary-light);
            }}
            
            .glass-card-dark {{
                background: rgba(30, 41, 59, 0.8);
                backdrop-filter: blur(16px) saturate(180%);
                -webkit-backdrop-filter: blur(16px) saturate(180%);
                border-radius: var(--radius-xl);
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: var(--shadow-lg);
                color: white;
            }}
            
            /* ============================================ */
            /* Neumorphism Effect */
            /* ============================================ */
            .neu-card {{
                background: var(--background);
                border-radius: var(--radius-lg);
                box-shadow: 
                    8px 8px 16px rgba(174, 174, 192, 0.4),
                    -8px -8px 16px rgba(255, 255, 255, 0.8);
                padding: 24px;
                transition: all var(--transition-normal);
            }}
            
            .neu-card:hover {{
                box-shadow: 
                    4px 4px 8px rgba(174, 174, 192, 0.4),
                    -4px -4px 8px rgba(255, 255, 255, 0.8);
            }}
            
            .neu-inset {{
                box-shadow: 
                    inset 4px 4px 8px rgba(174, 174, 192, 0.4),
                    inset -4px -4px 8px rgba(255, 255, 255, 0.8);
            }}
            
            /* ============================================ */
            /* Gradient Headings */
            /* ============================================ */
            h1, h2, h3 {{
                background: var(--gradient-aurora);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: 700;
            }}
            
            /* ============================================ */
            /* Modern Buttons */
            /* ============================================ */
            .stButton > button {{
                background: var(--gradient-primary) !important;
                color: white !important;
                border: none !important;
                border-radius: var(--radius-lg) !important;
                padding: 14px 28px !important;
                font-weight: 600 !important;
                font-size: 15px !important;
                letter-spacing: 0.025em !important;
                box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4) !important;
                transition: all var(--transition-normal) !important;
                position: relative !important;
                overflow: hidden !important;
            }}
            
            .stButton > button:hover {{
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px 0 rgba(79, 70, 229, 0.5) !important;
            }}
            
            .stButton > button:active {{
                transform: translateY(0) !important;
            }}
            
            .stButton > button::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(
                    90deg,
                    transparent,
                    rgba(255, 255, 255, 0.3),
                    transparent
                );
                transition: 0.5s;
            }}
            
            .stButton > button:hover::before {{
                left: 100%;
            }}
            
            /* ============================================ */
            /* Animated Metrics */
            /* ============================================ */
            [data-testid="stMetric"] {{
                background: var(--surface);
                border-radius: var(--radius-lg);
                padding: 20px;
                border: 1px solid var(--border);
                box-shadow: var(--shadow-md);
                transition: all var(--transition-normal);
            }}
            
            [data-testid="stMetric"]:hover {{
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg);
                border-color: var(--primary-light);
            }}
            
            [data-testid="stMetricValue"] {{
                font-size: 2.25rem !important;
                font-weight: 800 !important;
                background: var(--gradient-primary);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            
            /* ============================================ */
            /* Modern Inputs */
            /* ============================================ */
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input {{
                border-radius: var(--radius-md) !important;
                border: 2px solid var(--border) !important;
                padding: 12px 16px !important;
                font-size: 15px !important;
                transition: all var(--transition-fast) !important;
            }}
            
            .stTextInput > div > div > input:focus,
            .stNumberInput > div > div > input:focus {{
                border-color: var(--primary) !important;
                box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1) !important;
            }}
            
            /* ============================================ */
            /* Modern Sliders */
            /* ============================================ */
            .stSlider > div > div > div {{
                background: var(--gradient-primary) !important;
            }}
            
            .stSlider > div > div > div > div {{
                background: var(--surface) !important;
                border: 3px solid var(--primary) !important;
                box-shadow: var(--shadow-md) !important;
            }}
            
            /* ============================================ */
            /* Modern Selectbox */
            /* ============================================ */
            .stSelectbox > div > div {{
                border-radius: var(--radius-md) !important;
                border: 2px solid var(--border) !important;
                transition: all var(--transition-fast) !important;
            }}
            
            .stSelectbox > div > div:hover {{
                border-color: var(--primary-light) !important;
            }}
            
            /* ============================================ */
            /* Modern Tabs */
            /* ============================================ */
            .stTabs [data-baseweb="tab-list"] {{
                background: rgba(255, 255, 255, 0.5);
                backdrop-filter: blur(10px);
                border-radius: var(--radius-lg);
                padding: 6px;
                gap: 6px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                border-radius: var(--radius-md) !important;
                padding: 12px 24px !important;
                font-weight: 600 !important;
                transition: all var(--transition-fast) !important;
            }}
            
            .stTabs [aria-selected="true"] {{
                background: var(--gradient-primary) !important;
                color: white !important;
                box-shadow: var(--shadow-md) !important;
            }}
            
            /* ============================================ */
            /* Modern Expanders */
            /* ============================================ */
            .streamlit-expanderHeader {{
                background: var(--surface) !important;
                border-radius: var(--radius-lg) !important;
                border: 1px solid var(--border) !important;
                font-weight: 600 !important;
                transition: all var(--transition-fast) !important;
            }}
            
            .streamlit-expanderHeader:hover {{
                background: var(--background) !important;
                border-color: var(--primary-light) !important;
            }}
            
            .streamlit-expanderContent {{
                background: var(--surface) !important;
                border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
                border: 1px solid var(--border) !important;
                border-top: none !important;
            }}
            
            /* ============================================ */
            /* Modern Sidebar */
            /* ============================================ */
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, var(--surface) 0%, var(--background) 100%);
                border-right: 1px solid var(--border);
            }}
            
            [data-testid="stSidebar"] > div:first-child {{
                padding-top: 2rem;
            }}
            
            /* ============================================ */
            /* Success/Error/Warning/Info Messages */
            /* ============================================ */
            .stSuccess {{
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.1) 100%);
                border-left: 4px solid var(--success);
                border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
            }}
            
            .stError {{
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(248, 113, 113, 0.1) 100%);
                border-left: 4px solid var(--error);
                border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
            }}
            
            .stWarning {{
                background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(251, 191, 36, 0.1) 100%);
                border-left: 4px solid var(--warning);
                border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
            }}
            
            .stInfo {{
                background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
                border-left: 4px solid var(--primary);
                border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
            }}
            
            /* ============================================ */
            /* Modern Progress Bar */
            /* ============================================ */
            .stProgress > div > div > div {{
                background: var(--gradient-aurora) !important;
                background-size: 200% 200%;
                animation: gradientMove 2s ease infinite;
                border-radius: var(--radius-full);
            }}
            
            @keyframes gradientMove {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}
            
            /* ============================================ */
            /* Animations */
            /* ============================================ */
            @keyframes fadeInUp {{
                from {{
                    opacity: 0;
                    transform: translateY(30px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            
            @keyframes slideInLeft {{
                from {{
                    opacity: 0;
                    transform: translateX(-30px);
                }}
                to {{
                    opacity: 1;
                    transform: translateX(0);
                }}
            }}
            
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
            
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.7; }}
            }}
            
            @keyframes float {{
                0%, 100% {{ transform: translateY(0px); }}
                50% {{ transform: translateY(-10px); }}
            }}
            
            @keyframes shimmer {{
                0% {{ background-position: -200% 0; }}
                100% {{ background-position: 200% 0; }}
            }}
            
            .animate-fade-in-up {{
                animation: fadeInUp 0.6s ease-out forwards;
            }}
            
            .animate-slide-in {{
                animation: slideInLeft 0.5s ease-out forwards;
            }}
            
            .animate-scale-in {{
                animation: scaleIn 0.4s ease-out forwards;
            }}
            
            .animate-pulse {{
                animation: pulse 2s ease-in-out infinite;
            }}
            
            .animate-float {{
                animation: float 3s ease-in-out infinite;
            }}
            
            /* Animation delays for staggered effects */
            .delay-100 {{ animation-delay: 100ms; }}
            .delay-200 {{ animation-delay: 200ms; }}
            .delay-300 {{ animation-delay: 300ms; }}
            .delay-400 {{ animation-delay: 400ms; }}
            .delay-500 {{ animation-delay: 500ms; }}
            
            /* ============================================ */
            /* Chart Containers */
            /* ============================================ */
            [data-testid="stPlotlyChart"] {{
                animation: scaleIn 0.5s ease-out;
                border-radius: var(--radius-lg);
                overflow: hidden;
            }}
            
            /* ============================================ */
            /* DataFrame Styling */
            /* ============================================ */
            [data-testid="stDataFrame"] {{
                animation: slideInLeft 0.4s ease-out;
            }}
            
            [data-testid="stDataFrame"] > div {{
                border-radius: var(--radius-lg) !important;
                overflow: hidden;
            }}
            
            /* ============================================ */
            /* Scrollbar Styling */
            /* ============================================ */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: var(--background);
                border-radius: var(--radius-full);
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: var(--border);
                border-radius: var(--radius-full);
                transition: background var(--transition-fast);
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: var(--primary-light);
            }}
            
            /* ============================================ */
            /* Tooltip Enhancement */
            /* ============================================ */
            [data-baseweb="tooltip"] {{
                background: var(--surface) !important;
                border-radius: var(--radius-md) !important;
                box-shadow: var(--shadow-lg) !important;
                border: 1px solid var(--border) !important;
            }}
        </style>
        """, unsafe_allow_html=True)


class ModernMetricCard:
    """
    Modern metric card component with animations.
    """
    
    @staticmethod
    def render(
        title: str,
        value: str,
        delta: Optional[str] = None,
        delta_type: str = "positive",  # positive, negative, neutral
        icon: str = "📊",
        animation_delay: int = 0
    ):
        """
        Render a modern metric card.
        
        Args:
            title: Card title
            value: Main value to display
            delta: Optional change indicator
            delta_type: Type of change for styling
            icon: Emoji icon for the card
            animation_delay: Animation delay in milliseconds
        """
        delta_html = ""
        if delta:
            colors = {
                "positive": "#10B981",
                "negative": "#EF4444",
                "neutral": "#64748B"
            }
            icons = {
                "positive": "↑",
                "negative": "↓",
                "neutral": "→"
            }
            color = colors.get(delta_type, colors["neutral"])
            icon_delta = icons.get(delta_type, icons["neutral"])
            delta_html = f'''
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    color: {color};
                    font-size: 0.875rem;
                    font-weight: 600;
                    margin-top: 8px;
                ">
                    <span>{icon_delta}</span>
                    <span>{delta}</span>
                </div>
            '''
        
        st.markdown(f'''
            <div class="glass-card animate-fade-in-up" style="animation-delay: {animation_delay}ms;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <div style="
                        font-size: 2rem;
                        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
                        border-radius: 12px;
                        padding: 10px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">{icon}</div>
                    <span style="
                        color: #64748B;
                        font-size: 0.875rem;
                        font-weight: 500;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                    ">{title}</span>
                </div>
                <div style="
                    font-size: 2.5rem;
                    font-weight: 800;
                    background: linear-gradient(135deg, #1E293B 0%, #475569 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    line-height: 1.2;
                ">{value}</div>
                {delta_html}
            </div>
        ''', unsafe_allow_html=True)


class ModernCharts:
    """
    Enhanced Plotly charts with modern styling.
    """
    
    COLORSCALE = [
        '#4F46E5', '#7C3AED', '#EC4899', '#10B981',
        '#F59E0B', '#06B6D4', '#8B5CF6', '#EF4444'
    ]
    
    @classmethod
    def get_layout_defaults(cls) -> Dict:
        """Get default layout configuration for modern charts."""
        return {
            'template': 'plotly_white',
            'font': {
                'family': 'Inter, sans-serif',
                'size': 12,
                'color': '#334155'
            },
            'paper_bgcolor': 'rgba(255,255,255,0.8)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'colorway': cls.COLORSCALE,
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
                'font': {'size': 12}
            },
            'xaxis': {
                'gridcolor': '#F1F5F9',
                'gridwidth': 1,
                'linecolor': '#E2E8F0',
                'tickfont': {'size': 11}
            },
            'yaxis': {
                'gridcolor': '#F1F5F9',
                'gridwidth': 1,
                'linecolor': '#E2E8F0',
                'tickfont': {'size': 11}
            },
            'margin': {'l': 60, 'r': 30, 't': 60, 'b': 50}
        }
    
    @classmethod
    def animated_bar_chart(
        cls,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: str = "",
        height: int = 400
    ) -> go.Figure:
        """
        Create an animated bar chart with modern styling.
        
        Args:
            data: DataFrame with the data
            x: Column name for x-axis
            y: Column name for y-axis
            color: Optional column for color
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly Figure
        """
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            color_discrete_sequence=cls.COLORSCALE
        )
        
        # Apply modern styling
        layout = cls.get_layout_defaults()
        layout['height'] = height
        layout['title'] = {
            'text': title,
            'font': {'size': 18, 'weight': 700},
            'x': 0.5,
            'xanchor': 'center'
        }
        
        fig.update_layout(**layout)
        
        # Add animation on load
        fig.update_traces(
            marker_line_width=0,
            marker_cornerradius=8,
            hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
        )
        
        return fig
    
    @classmethod
    def gradient_line_chart(
        cls,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str = "",
        height: int = 400,
        fill: bool = True
    ) -> go.Figure:
        """
        Create a line chart with gradient fill.
        
        Args:
            data: DataFrame with the data
            x: Column name for x-axis
            y: Column name for y-axis
            title: Chart title
            height: Chart height
            fill: Whether to fill area under line
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[x],
            y=data[y],
            mode='lines',
            line=dict(
                color='#4F46E5',
                width=3,
                shape='spline'
            ),
            fill='tozeroy' if fill else None,
            fillcolor='rgba(79, 70, 229, 0.1)',
            hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
        ))
        
        layout = cls.get_layout_defaults()
        layout['height'] = height
        layout['title'] = {
            'text': title,
            'font': {'size': 18, 'weight': 700},
            'x': 0.5,
            'xanchor': 'center'
        }
        
        fig.update_layout(**layout)
        
        return fig
    
    @classmethod
    def radar_chart(
        cls,
        categories: List[str],
        values: Dict[str, List[float]],
        title: str = "",
        height: int = 400
    ) -> go.Figure:
        """
        Create a radar/spider chart with modern styling.
        
        Args:
            categories: List of category names
            values: Dict mapping series names to lists of values
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        for i, (name, vals) in enumerate(values.items()):
            color = cls.COLORSCALE[i % len(cls.COLORSCALE)]
            # Close the polygon
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=cats_closed,
                fill='toself',
                name=name,
                line=dict(color=color, width=2),
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor='#E2E8F0',
                    linecolor='#E2E8F0'
                ),
                angularaxis=dict(
                    gridcolor='#E2E8F0',
                    linecolor='#E2E8F0'
                ),
                bgcolor='rgba(255,255,255,0.8)'
            ),
            showlegend=True,
            height=height,
            title={
                'text': title,
                'font': {'size': 18, 'weight': 700, 'family': 'Inter, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            font={'family': 'Inter, sans-serif'},
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#E2E8F0',
                borderwidth=1
            )
        )
        
        return fig
    
    @classmethod
    def donut_chart(
        cls,
        labels: List[str],
        values: List[float],
        title: str = "",
        height: int = 400,
        hole_size: float = 0.6
    ) -> go.Figure:
        """
        Create a donut chart with modern styling.
        
        Args:
            labels: List of segment labels
            values: List of segment values
            title: Chart title
            height: Chart height
            hole_size: Size of the center hole (0-1)
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=hole_size,
            marker=dict(colors=cls.COLORSCALE),
            textinfo='percent+label',
            textfont=dict(size=12, family='Inter, sans-serif'),
            hovertemplate='<b>%{label}</b><br>%{value:.2f}<br>%{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            height=height,
            title={
                'text': title,
                'font': {'size': 18, 'weight': 700, 'family': 'Inter, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            font={'family': 'Inter, sans-serif'},
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#E2E8F0',
                borderwidth=1
            ),
            annotations=[dict(
                text=f'<b>Total</b><br>{sum(values):.1f}',
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )]
        )
        
        return fig
    
    @classmethod
    def heatmap_chart(
        cls,
        z: List[List[float]],
        x: List[str],
        y: List[str],
        title: str = "",
        height: int = 400
    ) -> go.Figure:
        """
        Create a heatmap with modern styling.
        
        Args:
            z: 2D array of values
            x: X-axis labels
            y: Y-axis labels
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[
                [0, '#F8FAFC'],
                [0.25, '#C7D2FE'],
                [0.5, '#818CF8'],
                [0.75, '#4F46E5'],
                [1, '#3730A3']
            ],
            hovertemplate='%{x}<br>%{y}<br>Value: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            height=height,
            title={
                'text': title,
                'font': {'size': 18, 'weight': 700, 'family': 'Inter, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            font={'family': 'Inter, sans-serif'},
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis={'tickfont': {'size': 11}},
            yaxis={'tickfont': {'size': 11}}
        )
        
        return fig


class ModernProgress:
    """
    Modern progress indicators.
    """
    
    @staticmethod
    def circular_progress(
        progress: float,
        label: str = "",
        color: str = "#4F46E5"
    ):
        """
        Display a circular progress indicator.
        
        Args:
            progress: Progress value (0-100)
            label: Optional label
            color: Progress color
        """
        svg = f'''
        <div style="display: flex; flex-direction: column; align-items: center; padding: 20px;">
            <svg width="120" height="120" viewBox="0 0 120 120">
                <circle
                    cx="60" cy="60" r="50"
                    fill="none"
                    stroke="#E2E8F0"
                    stroke-width="10"
                />
                <circle
                    cx="60" cy="60" r="50"
                    fill="none"
                    stroke="{color}"
                    stroke-width="10"
                    stroke-linecap="round"
                    stroke-dasharray="314.159"
                    stroke-dashoffset="{314.159 * (1 - progress / 100)}"
                    transform="rotate(-90 60 60)"
                    style="transition: stroke-dashoffset 1s ease-out;"
                />
                <text
                    x="60" y="65"
                    text-anchor="middle"
                    font-size="24"
                    font-weight="700"
                    fill="{color}"
                    font-family="Inter, sans-serif"
                >{progress:.0f}%</text>
            </svg>
            {f'<div style="color: #64748B; margin-top: 8px; font-size: 0.875rem;">{label}</div>' if label else ''}
        </div>
        '''
        st.markdown(svg, unsafe_allow_html=True)
    
    @staticmethod
    def step_progress(
        steps: List[str],
        current_step: int
    ):
        """
        Display a step-by-step progress indicator.
        
        Args:
            steps: List of step labels
            current_step: Current step index (0-based)
        """
        steps_html = ""
        for i, step in enumerate(steps):
            is_complete = i < current_step
            is_current = i == current_step
            
            if is_complete:
                status_class = "complete"
                icon = "✓"
                bg_color = "#10B981"
            elif is_current:
                status_class = "current"
                icon = str(i + 1)
                bg_color = "#4F46E5"
            else:
                status_class = "pending"
                icon = str(i + 1)
                bg_color = "#E2E8F0"
            
            text_color = "white" if is_complete or is_current else "#64748B"
            
            steps_html += f'''
                <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
                    <div style="
                        width: 40px;
                        height: 40px;
                        border-radius: 50%;
                        background: {bg_color};
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: {text_color};
                        font-weight: 600;
                        font-size: 14px;
                        transition: all 0.3s ease;
                    ">{icon}</div>
                    <div style="
                        margin-top: 8px;
                        font-size: 0.75rem;
                        color: {'#1E293B' if is_current else '#64748B'};
                        font-weight: {'600' if is_current else '400'};
                        text-align: center;
                    ">{step}</div>
                </div>
            '''
            
            # Add connector line (except for last item)
            if i < len(steps) - 1:
                line_color = "#10B981" if is_complete else "#E2E8F0"
                steps_html += f'''
                    <div style="
                        flex: 1;
                        height: 2px;
                        background: {line_color};
                        margin-top: 20px;
                        transition: background 0.3s ease;
                    "></div>
                '''
        
        st.markdown(f'''
            <div style="display: flex; align-items: flex-start; padding: 20px 0;">
                {steps_html}
            </div>
        ''', unsafe_allow_html=True)
