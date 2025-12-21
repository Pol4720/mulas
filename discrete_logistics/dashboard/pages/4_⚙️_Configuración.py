"""
⚙️ Configuración - Multi-Bin Packing Solver
============================================

Página de configuración y ajustes del dashboard.
"""

import streamlit as st

st.set_page_config(
    page_title="⚙️ Configuración | Multi-Bin Packing",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import shared utilities
from shared import (
    init_session_state,
    apply_custom_styles,
    render_sidebar_info,
    ParameterTuner,
    InteractiveTooltips
)

# Initialize
init_session_state()
apply_custom_styles()
render_sidebar_info()

# ============================================================================
# Page Content
# ============================================================================

# Page header
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="
        font-size: 2.5rem;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    ">⚙️ Configuración</h1>
    <p style="color: #64748B;">Personaliza la experiencia del dashboard</p>
</div>
""", unsafe_allow_html=True)

# Configuration sections
tab1, tab2, tab3 = st.tabs([
    "🎨 Apariencia",
    "🔧 Parámetros de Algoritmos",
    "📊 Datos"
])

with tab1:
    st.markdown("### 🎨 Configuración de Apariencia")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top: 0;">🌓 Tema</h4>
        </div>
        """, unsafe_allow_html=True)
        
        theme = st.radio(
            "Seleccionar tema:",
            options=["🌞 Claro", "🌙 Oscuro", "🔄 Sistema"],
            horizontal=True,
            key="theme_selector"
        )
        
        if theme == "🌙 Oscuro":
            st.info("💡 El tema oscuro se implementará en una futura versión.")
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top: 0;">✨ Animaciones</h4>
        </div>
        """, unsafe_allow_html=True)
        
        animations_enabled = st.toggle(
            "Habilitar animaciones",
            value=st.session_state.get('animation_enabled', True),
            key="animation_toggle"
        )
        st.session_state['animation_enabled'] = animations_enabled
        
        if animations_enabled:
            st.success("✅ Animaciones habilitadas")
        else:
            st.info("💤 Animaciones deshabilitadas")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Vista previa de colores")
    
    cols = st.columns(6)
    colors = [
        ("#4F46E5", "Índigo"),
        ("#7C3AED", "Violeta"),
        ("#EC4899", "Rosa"),
        ("#10B981", "Esmeralda"),
        ("#F59E0B", "Ámbar"),
        ("#06B6D4", "Cian")
    ]
    
    for col, (color, name) in zip(cols, colors):
        with col:
            st.markdown(f"""
            <div style="
                background: {color};
                border-radius: 12px;
                padding: 30px 10px;
                text-align: center;
            ">
                <span style="color: white; font-weight: 600; font-size: 0.8rem;">{name}</span>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 🔧 Parámetros Predeterminados de Algoritmos")
    
    st.markdown("""
    <p style="color: #64748B; margin-bottom: 20px;">
        Configura los parámetros predeterminados para cada algoritmo metaheurístico.
        Estos valores se usarán como valores iniciales al seleccionar el algoritmo.
    </p>
    """, unsafe_allow_html=True)
    
    # Algorithm parameter configurations
    algo_tabs = st.tabs(["🔥 Simulated Annealing", "🧬 Genetic Algorithm", "🚫 Tabu Search"])
    
    with algo_tabs[0]:
        st.markdown("#### Simulated Annealing")
        
        col1, col2 = st.columns(2)
        with col1:
            sa_temp = st.slider(
                "🌡️ Temperatura Inicial",
                min_value=100.0,
                max_value=10000.0,
                value=1000.0,
                step=100.0,
                help="Temperatura inicial del sistema"
            )
            
            sa_cooling = st.slider(
                "❄️ Tasa de Enfriamiento",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Factor de reducción de temperatura"
            )
        
        with col2:
            sa_iterations = st.slider(
                "🔄 Iteraciones Máximas",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Número máximo de iteraciones"
            )
            
            sa_min_temp = st.slider(
                "🧊 Temperatura Mínima",
                min_value=0.001,
                max_value=10.0,
                value=0.01,
                step=0.001,
                format="%.3f",
                help="Temperatura de parada"
            )
    
    with algo_tabs[1]:
        st.markdown("#### Genetic Algorithm")
        
        col1, col2 = st.columns(2)
        with col1:
            ga_pop = st.slider(
                "👥 Tamaño de Población",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Número de individuos"
            )
            
            ga_gens = st.slider(
                "🧬 Generaciones",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Número de generaciones"
            )
        
        with col2:
            ga_mut = st.slider(
                "🎲 Tasa de Mutación",
                min_value=0.01,
                max_value=0.30,
                value=0.1,
                step=0.01,
                help="Probabilidad de mutación"
            )
            
            ga_cross = st.slider(
                "✂️ Tasa de Cruce",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Probabilidad de cruce"
            )
    
    with algo_tabs[2]:
        st.markdown("#### Tabu Search")
        
        col1, col2 = st.columns(2)
        with col1:
            ts_tenure = st.slider(
                "📋 Tenure de Lista Tabú",
                min_value=5,
                max_value=50,
                value=10,
                step=1,
                help="Iteraciones que un movimiento permanece tabú"
            )
        
        with col2:
            ts_iterations = st.slider(
                "🔄 Iteraciones Máximas",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Número máximo de iteraciones"
            )
    
    # Save button
    st.markdown("")
    if st.button("💾 Guardar Configuración", type="primary"):
        st.session_state['default_params'] = {
            'SimulatedAnnealing': {
                'initial_temp': sa_temp,
                'cooling_rate': sa_cooling,
                'max_iterations': sa_iterations,
                'min_temp': sa_min_temp
            },
            'GeneticAlgorithm': {
                'population_size': ga_pop,
                'generations': ga_gens,
                'mutation_rate': ga_mut,
                'crossover_rate': ga_cross
            },
            'TabuSearch': {
                'tabu_tenure': ts_tenure,
                'max_iterations': ts_iterations
            }
        }
        st.success("✅ Configuración guardada correctamente")

with tab3:
    st.markdown("### 📊 Gestión de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top: 0;">🗑️ Limpiar Datos</h4>
            <p style="color: #64748B; font-size: 0.9rem;">
                Elimina los resultados y la instancia actual de la sesión.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Limpiar Resultados", type="secondary"):
            st.session_state['results'] = {}
            st.session_state['convergence_history'] = {}
            st.success("✅ Resultados eliminados")
        
        if st.button("🗑️ Limpiar Instancia", type="secondary"):
            st.session_state['current_problem'] = None
            st.success("✅ Instancia eliminada")
        
        if st.button("🗑️ Limpiar Todo", type="primary"):
            st.session_state['results'] = {}
            st.session_state['convergence_history'] = {}
            st.session_state['current_problem'] = None
            st.success("✅ Todos los datos eliminados")
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top: 0;">📈 Estado de la Sesión</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Session stats
        has_problem = st.session_state.get('current_problem') is not None
        num_results = len(st.session_state.get('results', {}))
        has_history = bool(st.session_state.get('convergence_history'))
        
        st.markdown(f"""
        <div style="
            background: rgba(79, 70, 229, 0.05);
            border-radius: 12px;
            padding: 16px;
        ">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #64748B;">Instancia activa:</span>
                <span style="font-weight: 600;">{'✅ Sí' if has_problem else '❌ No'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #64748B;">Resultados:</span>
                <span style="font-weight: 600;">{num_results}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #64748B;">Historial de convergencia:</span>
                <span style="font-weight: 600;">{'✅ Sí' if has_history else '❌ No'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
    <p>Multi-Bin Packing Solver v2.0 | 🎓 Proyecto DAA - Universidad de La Habana</p>
</div>
""", unsafe_allow_html=True)
