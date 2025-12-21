"""
📚 Teoría - Multi-Bin Packing Solver
====================================

Página con los fundamentos teóricos del problema:
formalización matemática, complejidad computacional,
reducciones y descripción de algoritmos.
"""

import streamlit as st

st.set_page_config(
    page_title="📚 Teoría | Multi-Bin Packing",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import shared utilities
from shared import (
    init_session_state,
    apply_custom_styles,
    render_sidebar_info
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
    ">📚 Fundamentos Teóricos</h1>
    <p style="color: #64748B;">Complejidad computacional, formalizaciones y algoritmos</p>
</div>
""", unsafe_allow_html=True)

# Tabs for different theory sections
tabs = st.tabs([
    "📐 Formalización",
    "🔢 Complejidad NP", 
    "🔗 Reducciones",
    "📝 Algoritmos",
    "📖 Referencias"
])

with tabs[0]:
    st.markdown("""
    ## Formalización Matemática
    
    ### Definición del Problema
    
    El **Problema de Empaquetado Multi-Contenedor Balanceado** se define formalmente como:
    
    **Dado:**
    - Un conjunto de ítems $I = \\{1, 2, ..., n\\}$
    - Cada ítem $i$ tiene peso $w_i$ y valor $v_i$
    - $k$ contenedores con capacidades individuales $C_j$
    
    **Variables de Decisión:**
    - $x_{ij} \\in \\{0, 1\\}$: 1 si el ítem $i$ es asignado al contenedor $j$
    - $z$: makespan (valor total máximo en cualquier contenedor)
    
    ### Formulación ILP (Programación Lineal Entera)
    
    $$\\min z$$
    
    Sujeto a:
    
    $$\\sum_{j=1}^{k} x_{ij} = 1 \\quad \\forall i \\in I$$
    
    $$\\sum_{i=1}^{n} w_i \\cdot x_{ij} \\leq C_j \\quad \\forall j = 1,...,k$$
    
    $$\\sum_{i=1}^{n} v_i \\cdot x_{ij} \\leq z \\quad \\forall j = 1,...,k$$
    
    $$x_{ij} \\in \\{0, 1\\}, z \\geq 0$$
    
    ### Problema de Optimización vs. Decisión
    
    - **Optimización (BALANCED-BIN-PACKING-OPT):** Minimizar la diferencia máxima de valores
    - **Decisión (BALANCED-BIN-PACKING-DEC):** ¿Existe asignación con diferencia ≤ B?
    
    El problema de decisión es **NP-completo**, lo que implica que el problema de optimización es **NP-hard**.
    """)

with tabs[1]:
    st.markdown("""
    ## Análisis de Complejidad Computacional
    
    ### NP-Completitud del Problema de Decisión
    
    **Teorema:** BALANCED-BIN-PACKING-DEC ∈ NP-completo
    
    **Demostración (esquema):**
    
    #### Parte 1: BALANCED-BIN-PACKING-DEC ∈ NP
    
    **Certificado:** Una asignación σ: I → {1,...,k}
    
    **Verificación en tiempo polinomial:**
    1. Verificar asignación completa: O(n)
    2. Calcular pesos por bin: O(n)
    3. Verificar capacidades: O(k)
    4. Calcular valores por bin: O(n)
    5. Verificar diferencia ≤ B: O(k)
    
    **Total:** O(n + k) → Polinomial ✓
    
    #### Parte 2: NP-Hardness
    
    Se demuestra mediante reducción desde **3-PARTITION**, un problema fuertemente NP-completo.
    
    ### Clases de Complejidad Relevantes
    
    | Clase | Descripción | Nuestro Problema |
    |-------|-------------|------------------|
    | P | Resoluble en tiempo polinomial | ❌ (asumiendo P ≠ NP) |
    | NP | Verificable en tiempo polinomial | ✅ (versión decisión) |
    | NP-completo | Más difícil en NP | ✅ (versión decisión) |
    | NP-hard | Al menos tan difícil como NP-completo | ✅ (versión optimización) |
    
    ### Implicaciones Prácticas
    
    1. **No existe algoritmo exacto eficiente** (asumiendo P ≠ NP)
    2. **Necesidad de aproximaciones** y heurísticas
    3. **Inaproximabilidad:** No existe PTAS general
    4. **Instancias grandes:** Requieren métodos aproximados
    """)

with tabs[2]:
    st.markdown("""
    ## Cadena de Reducciones
    
    ### De PARTITION a Nuestro Problema
    """)
    
    st.code("""
    PARTITION (Karp 1972, NP-completo)
         ↓ reducción polinomial
    3-PARTITION (Fuertemente NP-completo)
         ↓ reducción polinomial
    BIN PACKING CLÁSICO
         ↓ generalización
    BALANCED-BIN-PACKING ← Nuestro problema
    """, language="text")
    
    st.markdown("""
    ### Problema 3-PARTITION
    
    **Entrada:** 
    - Conjunto A = {a₁, a₂, ..., a₃ₘ} de 3m enteros
    - Valor objetivo B tal que Σaᵢ = mB
    - Restricción: B/4 < aᵢ < B/2 para todo i
    
    **Pregunta:** ¿Se puede particionar A en m subconjuntos de 3 elementos cada uno, donde cada subconjunto suma exactamente B?
    
    **Importancia:** 3-PARTITION es **fuertemente NP-completo**:
    - Permanece NP-completo incluso con representación unaria
    - No tiene pseudo-polinomial (a diferencia de KNAPSACK)
    
    ### Reducción: 3-PARTITION ≤ₚ BALANCED-BIN-PACKING
    
    **Construcción:**
    
    Dada instancia de 3-PARTITION con {a₁,...,a₃ₘ} y objetivo B:
    
    1. **Crear ítems:** Para cada aᵢ → Item(peso=aᵢ, valor=aᵢ)
    2. **Número de bins:** k = m
    3. **Capacidades:** Cⱼ = B para todo j (uniforme)
    4. **Umbral:** β = 0 (balance perfecto)
    
    **Correctitud (⇒):**
    - Si existe 3-partición válida → todos los bins tienen valor B
    - Diferencia máxima = B - B = 0 ≤ β ✓
    
    **Correctitud (⇐):**
    - Si diferencia = 0 → todos bins tienen igual valor
    - Como Σvᵢ = mB y k = m → cada bin tiene valor B
    - Restricciones B/4 < aᵢ < B/2 → exactamente 3 elementos por bin
    - Esto constituye una 3-partición válida ✓
    
    ### Consecuencias
    
    **Corolario 1:** BALANCED-BIN-PACKING-OPT es NP-hard
    
    *Prueba:* Si existiera algoritmo polinomial para optimización, resolvería decisión en tiempo polinomial → P = NP.
    
    **Corolario 2:** Capacidades heterogéneas son ≥ difíciles que uniformes
    
    *Prueba:* Caso uniforme es instancia particular del heterogéneo.
    """)

with tabs[3]:
    st.markdown("""
    ## Descripción de Algoritmos
    
    ### Algoritmos Voraces (Greedy)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **First Fit Decreasing (FFD):**
        1. Ordenar ítems por peso (descendente)
        2. Para cada ítem, colocar en primer contenedor con espacio
        3. Complejidad: O(n log n)
        4. Aproximación: Sin garantía para objetivo de balance
        
        **Best Fit Decreasing (BFD):**
        1. Ordenar ítems por peso (descendente)
        2. Para cada ítem, elegir bin con mínimo espacio restante que quepa
        3. Proporciona empaquetado más compacto
        4. Complejidad: O(n²)
        """)
    
    with col2:
        st.markdown("""
        **Worst Fit Decreasing (WFD):**
        1. Ordenar ítems por peso (descendente)
        2. Para cada ítem, elegir bin con máximo espacio restante
        3. Favorece el balance (distribuye carga)
        4. Complejidad: O(n log n)
        
        **Round Robin:**
        1. Asignar ítems cíclicamente entre bins
        2. Muy rápido: O(n)
        3. Simple pero puede violar capacidad
        """)
    
    st.markdown("---")
    st.markdown("### Programación Dinámica")
    
    st.markdown("""
    **Enfoque:** Construcción óptima de k-particiones mediante el esquema SRTBOT
    
    #### Esquema SRTBOT
    
    **S - Subproblemas:**
    - $DP[j][mask]$ = mejor solución con $j$ bins asignando ítems en $mask$
    - Número de subproblemas: $O(k \\cdot 2^n)$
    
    **R - Relación de Recurrencia:**
    $$DP[j][mask \\cup S] = \\min_{S \\in Factible(j)} \\left\\{ \\max(V_{max}, V(S)) - \\min(V_{min}, V(S)) \\right\\}$$
    
    **T - Tiempo de Ejecución:**
    - Pre-computación: $O(k \\cdot 2^n \\cdot n)$
    - DP principal: $O(k \\cdot 3^n)$
    - Espacio: $O(k \\cdot 2^n)$
    """)
    
    st.markdown("---")
    st.markdown("### Metaheurísticas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔥 Simulated Annealing:**
        - Búsqueda local probabilística
        - Acepta soluciones peores con P = e^(-Δ/T)
        - Temperatura T decrece
        - Escapa de óptimos locales
        """)
    
    with col2:
        st.markdown("""
        **🧬 Algoritmo Genético:**
        - Población de soluciones evoluciona
        - Operadores: selección, cruce, mutación
        - Explora espacio diverso
        - Balance exploración/explotación
        """)
    
    with col3:
        st.markdown("""
        **🚫 Búsqueda Tabú:**
        - Búsqueda local con memoria
        - Lista tabú evita ciclos
        - Intensificación y diversificación
        - Muy efectivo en la práctica
        """)
    
    st.markdown("---")
    st.markdown("### Complejidades Comparadas")
    
    complexity_data = {
        'Algoritmo': ['FFD', 'BFD', 'WFD', 'DP', 'B&B', 'SA', 'GA'],
        'Tiempo': ['O(n log n)', 'O(n²)', 'O(n log n)', 'O(k·3ⁿ)', 'O(kⁿ) peor', 'O(I·n)', 'O(G·P·n)'],
        'Espacio': ['O(n)', 'O(n)', 'O(n)', 'O(k·2ⁿ)', 'O(n)', 'O(n)', 'O(P·n)'],
        'Optimalidad': ['No', 'No', 'No', '✅ Óptima', '✅ Óptima', 'Aprox', 'Aprox']
    }
    
    import pandas as pd
    df = pd.DataFrame(complexity_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

with tabs[4]:
    st.markdown("""
    ## Referencias Fundamentales
    
    ### Complejidad Computacional
    
    1. **Garey, M.R., & Johnson, D.S. (1979).** *Computers and Intractability: 
       A Guide to the Theory of NP-Completeness*. W.H. Freeman.
       - Teoría fundamental de NP-completitud
       - Demostración de 3-PARTITION como NP-completo
    
    2. **Karp, R.M. (1972).** "Reducibility among combinatorial problems." 
       *Complexity of Computer Computations*, 85-103.
       - 21 problemas NP-completos originales
       - Incluye PARTITION
    
    ### Bin Packing
    
    3. **Martello, S., & Toth, P. (1990).** *Knapsack Problems: Algorithms 
       and Computer Implementations*. Wiley.
       - Algoritmos exactos y aproximados
       - Programación dinámica avanzada
    
    4. **Coffman, E.G., Garey, M.R., & Johnson, D.S. (1996).** 
       "Approximation algorithms for bin packing: A survey."
       *Approximation Algorithms for NP-Hard Problems*, 46-93.
       - Análisis de FFD, BFD, WFD
       - Cotas de aproximación
    
    ### Metaheurísticas
    
    5. **Kirkpatrick, S., Gelatt, C.D., & Vecchi, M.P. (1983).** 
       "Optimization by simulated annealing." *Science*, 220(4598), 671-680.
       - Artículo fundacional de SA
    
    6. **Glover, F. (1989).** "Tabu search—Part I." 
       *ORSA Journal on Computing*, 1(3), 190-206.
       - Fundamentos de búsqueda tabú
    
    7. **Holland, J.H. (1992).** *Adaptation in Natural and Artificial Systems*.
       MIT Press.
       - Algoritmos genéticos
    
    ### Balanceo de Carga
    
    8. **Graham, R.L. (1969).** "Bounds on multiprocessing timing anomalies."
       *SIAM Journal on Applied Mathematics*, 17(2), 416-429.
       - Makespan scheduling
       - Análisis de LPT
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Enlaces Útiles
    
    - 📘 [Complexity Zoo](https://complexityzoo.net/) - Clases de complejidad
    - 📗 [Algorithm Visualizer](https://algorithm-visualizer.org/) - Visualizaciones
    - 📙 [OEIS](https://oeis.org/) - Secuencias relacionadas
    """)
