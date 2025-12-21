"""
Multi-Bin Packing Solver - Main Entry Point
============================================

This is the main entry point for the Streamlit multipage application.
Run with: streamlit run app.py

The application uses Streamlit's native multipage feature.
Pages are located in the 'pages/' directory.
"""

import streamlit as st

st.set_page_config(
    page_title="🎯 Multi-Bin Packing Solver | DAA",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Pol4720/mulas',
        'Report a bug': 'https://github.com/Pol4720/mulas/issues',
        'About': '''
        ## Multi-Bin Packing Solver
        
        Advanced optimization tool for the Balanced Multi-Bin Packing Problem.
        
        **Features:**
        - 12 algorithms (Greedy, Metaheuristics, Exact)
        - Interactive visualizations
        - Comprehensive benchmarking
        - Theoretical foundations
        
        *Universidad de La Habana - DAA Project*
        '''
    }
)

# Import shared utilities
from shared import (
    init_session_state,
    apply_custom_styles,
    render_sidebar_info
)

# Initialize session state
init_session_state()

# Apply custom styles
apply_custom_styles()

# Render sidebar info
render_sidebar_info()

# ============================================================================
# Main Page Content (Home)
# ============================================================================

# Hero Section with Animated Gradient
st.markdown("""
<div style="text-align: center; padding: 50px 0 30px;">
    <div style="
        font-size: 5rem; 
        margin-bottom: 20px;
        animation: float 3s ease-in-out infinite;
    ">🎯</div>
    <h1 style="
        font-size: 3rem; 
        margin-bottom: 16px;
        background: linear-gradient(270deg, #4F46E5, #7C3AED, #EC4899, #4F46E5);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 8s ease infinite;
    ">
        Solucionador de Empaquetado Multi-Contenedor
    </h1>
    <p style="
        font-size: 1.25rem; 
        color: #64748B; 
        max-width: 700px; 
        margin: 0 auto;
        line-height: 1.7;
    ">
        Explora y resuelve problemas de empaquetado balanceado con 
        <strong>visualizaciones interactivas</strong> y 
        <strong>múltiples algoritmos de optimización</strong>.
    </p>
</div>
<style>
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("---")

# Problem Description Card
st.markdown("""
<div class="glass-card" style="animation: fadeInUp 0.6s ease-out;">
    <h3 style="
        margin-top: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <span style="font-size: 1.5rem;">📋</span>
        Descripción del Problema
    </h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 20px;">
        <div>
            <p style="color: #4F46E5; font-weight: 600; margin-bottom: 8px;">📥 Entrada:</p>
            <ul style="color: #475569; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li>Un conjunto de <strong>n ítems</strong>, cada uno con peso <em>w<sub>i</sub></em> y valor <em>v<sub>i</sub></em></li>
                <li><strong>k contenedores</strong> con capacidades individuales <em>C<sub>j</sub></em></li>
            </ul>
        </div>
        <div>
            <p style="color: #10B981; font-weight: 600; margin-bottom: 8px;">🎯 Objetivo:</p>
            <ul style="color: #475569; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li>Minimizar la <strong>diferencia máxima</strong> de valores totales entre contenedores</li>
                <li>Respetando las <strong>restricciones de capacidad</strong> de cada contenedor</li>
            </ul>
        </div>
    </div>
</div>
<style>
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("")
st.markdown("")

# Features Grid
st.markdown("### ✨ Características Principales")

col1, col2, col3, col4 = st.columns(4)

features = [
    ("🔬", "12 Algoritmos", "Voraz, Metaheurísticas y Métodos Exactos", "#4F46E5", col1, 0),
    ("📊", "Visualizaciones", "Gráficos interactivos y animaciones", "#7C3AED", col2, 100),
    ("📈", "Benchmarking", "Comparación de rendimiento avanzada", "#EC4899", col3, 200),
    ("📚", "Teoría", "NP-Completitud y reducciones formales", "#10B981", col4, 300),
]

for icon, title, desc, color, col, delay in features:
    with col:
        st.markdown(f"""
        <div class="glass-card" style="
            text-align: center; 
            min-height: 200px;
            animation: fadeInUp 0.6s ease-out {delay}ms both;
        ">
            <div style="
                font-size: 3rem; 
                margin-bottom: 16px;
                background: linear-gradient(135deg, {color} 0%, {color}99 100%);
                border-radius: 16px;
                padding: 16px;
                display: inline-block;
            ">{icon}</div>
            <h4 style="margin: 12px 0 8px; color: #1E293B; font-weight: 700;">{title}</h4>
            <p style="font-size: 0.85rem; color: #64748B; margin: 0; line-height: 1.5;">
                {desc}
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")
st.markdown("")

# Quick stats
st.markdown("### 📊 Resumen del Sistema")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔢 Algoritmos", "12", delta="4 exactos", delta_color="normal")
with col2:
    st.metric("⚡ Complejidad", "NP-Completo", help="Clase de complejidad del problema")
with col3:
    st.metric("📦 Máx Ítems", "100", help="Cantidad de ítems soportada")
with col4:
    st.metric("📈 Visualizaciones", "8+", help="Tipos de gráficos disponibles")

# Getting Started Section
st.markdown("---")
st.markdown("### 🚀 ¿Cómo comenzar?")

steps = [
    ("1️⃣", "Solucionador", "Ve a la página de <strong>Solucionador</strong>"),
    ("2️⃣", "Configurar", "Define tu <strong>instancia</strong> del problema"),
    ("3️⃣", "Ejecutar", "Elige los <strong>algoritmos</strong> a ejecutar"),
    ("4️⃣", "Analizar", "¡Explora los <strong>resultados</strong>!"),
]

cols = st.columns(4)
for i, (num, title, desc) in enumerate(steps):
    with cols[i]:
        st.markdown(f"""
        <div class="glass-card" style="
            text-align: center;
            padding: 20px;
            animation: fadeInUp 0.5s ease-out {i * 100}ms both;
        ">
            <div style="
                font-size: 2rem;
                background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
                border-radius: 12px;
                padding: 10px;
                display: inline-block;
                margin-bottom: 12px;
            ">{num}</div>
            <div style="font-weight: 700; color: #1E293B; margin-bottom: 6px;">{title}</div>
            <div style="font-size: 0.8rem; color: #64748B;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Navigation hint
st.markdown("")
st.markdown("""
<div style="
    text-align: center;
    padding: 30px;
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%);
    border-radius: 16px;
    margin-top: 20px;
">
    <p style="color: #64748B; margin: 0;">
        👈 Usa el <strong>menú lateral</strong> para navegar entre las páginas
    </p>
</div>
""", unsafe_allow_html=True)
