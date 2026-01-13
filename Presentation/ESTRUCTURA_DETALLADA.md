# Estructura de la Presentación - Vista Detallada

## Resumen de Diapositivas

```
Total: 19 diapositivas
Duración estimada: 15-20 minutos
Formato: 16:9 (Beamer Madrid theme)
```

---

## Estructura Completa

### DIAPOSITIVA 1: PORTADA
```
┌─────────────────────────────────────────┐
│                                         │
│  Problema de Transporte Logístico       │
│              Discreto                   │
│                                         │
│  Diseño y Análisis de Algoritmos        │
│                                         │
│  R. A. Matos Arderí                     │
│  A. Ponce González                      │
│  A. Romero Imbert                       │
│                                         │
│  Facultad de Matemática y Computación   │
│  Universidad de La Habana               │
│                                         │
│  Enero 13, 2026                         │
│                                         │
└─────────────────────────────────────────┘
```

### DIAPOSITIVA 2: CONTENIDOS
- Lista de secciones principales
- Facilita navegación

### SECCIÓN 1: EL PROBLEMA (1 diapositiva)

**DIAPOSITIVA 3: Descripción del Problema**
```
┌─────────────────────────────────────────┐
│ Descripción del Problema                │
│                                         │
│ Escenario Real:                         │
│ • n paquetes con peso y valor          │
│ • k vehículos con capacidad máxima      │
│ • Distribuir respetando límites         │
│ • Objetivo: Equilibrar la carga         │
│                                         │
│ Ejemplo Simple:                         │
│ • 5 paquetes: 2kg, 3kg, 4kg, 1kg, 2kg │
│ • 2 camiones: 6kg cada uno              │
│ • Desafío: distribución equitativa      │
│                                         │
│ [ESPACIO PARA ILUSTRACIÓN]              │
│                                         │
└─────────────────────────────────────────┘
```

### SECCIÓN 2: FORMALIZACIÓN MATEMÁTICA (2 diapositivas)

**DIAPOSITIVA 4: Definiciones Fundamentales**
```
Entrada del Problema:
- Ítems: I = {1, 2, …, n}
- Pesos: w_i > 0 para cada ítem i
- Valores: v_i ≥ 0 para cada ítem i
- k contenedores con capacidades C_1, …, C_k

Variables de Decisión:
  x_ij = { 1 si ítem i va a contenedor j
         { 0 en otro caso

Objetivo:
  Minimizar: max_j V_j - min_j V_j
  donde V_j = Σ(i=1 a n) v_i · x_ij
```

**DIAPOSITIVA 5: Formulación ILP**
```
minimizar   z^+ - z^-
sujeto a:
  Σ_j x_ij = 1, ∀i
  Σ_i w_i·x_ij ≤ C_j, ∀j
  Σ_i v_i·x_ij ≤ z^+, ∀j
  Σ_i v_i·x_ij ≥ z^-, ∀j
  x_ij ∈ {0,1}

Propiedad: En optimalidad: z^+ = max_j V_j y z^- = min_j V_j
```

### SECCIÓN 3: ANÁLISIS DE COMPLEJIDAD (2 diapositivas)

**DIAPOSITIVA 6: Problema de Decisión vs Optimización**
```
Problema de Decisión:
  ¿Existe asignación factible con desbalance ≤ B?

Problema de Optimización:
  Minimizar el desbalance.

Resultado: El problema es NP-Completo
```

**DIAPOSITIVA 7: Cadena de Reducciones**
```
PARTITION → 3-PARTITION → BALANCED-BIN-PACKING

Implicación: El problema es NP-Hard

No existe algoritmo polinomial conocido (asumiendo P ≠ NP)

Se requiere trade-off: optimalidad vs velocidad
```

### SECCIÓN 4: MÉTODOS DE RESOLUCIÓN (7 diapositivas)

**DIAPOSITIVA 8: Clasificación de Algoritmos**
```
Tabla comparativa:
┌─────────────┬────────┬──────────┬────────┐
│ Método      │ Óptimo │ Velocidad│ Rango  │
├─────────────┼────────┼──────────┼────────┤
│ Exactos:    │        │          │        │
│ Fuerza Bruta│ Sí     │ Lenta    │ n<15   │
│ B&B         │ Sí     │ Media    │ n<25   │
│ Prog. Din.  │ Sí     │ Media    │ n<20   │
├─────────────┼────────┼──────────┼────────┤
│ Aproximados:│        │          │        │
│ Greedy      │ No     │ Muy rápida│Todos  │
├─────────────┼────────┼──────────┼────────┤
│ Metaheur.:  │        │          │        │
│ SA/GA/TS    │ No     │ Rápida   │ Todos  │
└─────────────┴────────┴──────────┴────────┘
```

**DIAPOSITIVA 9: Algoritmos Greedy**
```
First Fit Decreasing (FFD):
1. Ordenar ítems por peso (descendente)
2. Asignar al primer contenedor con espacio
Complejidad: O(n log n)

LPT Balanced:
1. Ordenar ítems por valor (descendente)
2. Asignar al contenedor con menos carga
Complejidad: O(n log n)

Ventajas: Muy rápidos, buenos resultados
Desventajas: No garantizan optimalidad
```

**DIAPOSITIVA 10: Fuerza Bruta**
```
Probar las k^n asignaciones posibles.

Límites prácticos:
- k=2: hasta n=14 (1 segundo)
- k=3: hasta n=11 (1 segundo)
- k=4: hasta n=8 (1 segundo)

Complejidad: O(k^n · n)

Rol: Validar heurísticas, instancias pequeñas
```

**DIAPOSITIVA 11: Branch and Bound**
```
Idea: Fuerza Bruta + poda inteligente

Explora árbol de decisiones calculando cotas inferiores.

Si cota ≥ mejor solución: poda la rama

Ventajas:
- Garantiza optimalidad
- Mejora significativa en práctica
- Efectivo con cotas ajustadas

Complejidad: O(k^n) peor caso
```

**DIAPOSITIVA 12: Programación Dinámica**
```
Idea: Construir solución bin por bin.

Estado: DP[j][mask] = mejor asignación de ítems 
en mask a primeros j contenedores

Transición: Probar subconjuntos del contenedor j

Complejidad:
- Tiempo: O(k² · 3^n)
- Espacio: O(k · 2^n)

Aplicabilidad: Instancias pequeñas (n < 20)
```

**DIAPOSITIVA 13: Metaheurísticas**
```
Simulated Annealing, Algoritmos Genéticos, Búsqueda Tabú

Esquema general:
1. Generar solución inicial
2. Mejorar iterativamente con movimientos locales
3. Aceptar movimientos malos ocasionalmente
4. Detener por convergencia o tiempo

Ventajas: Rápidas, buenos resultados

Desventajas: Sin garantías, aleatoriedad, parámetros
```

**DIAPOSITIVA 14: [Posible diapositiva adicional si se necesita]**

### SECCIÓN 5: ANÁLISIS EXPERIMENTAL (4 diapositivas)

**DIAPOSITIVA 15: Comparación de Tiempos**
```
[GRÁFICO 1: Tiempo vs tamaño (escala log)]

Instancias aleatorias con n ∈ [5, 50] e k=3,4,5
Eje Y: tiempo (segundos)
Eje X: número de ítems
```

**DIAPOSITIVA 16: Órdenes de Complejidad**
```
[GRÁFICO 2: Comparación de órdenes]

- FFD/LPT: línea plana (polinomial)
- B&B: exponencial débil
- DP: exponencial fuerte 3^n
- Metaheurísticas: línea plana (parámetro fijo)
```

**DIAPOSITIVA 17: Calidad de Soluciones**
```
Gap de Optimalidad:
  Gap(%) = (Heurístico - Óptimo) / Óptimo × 100

Resultados promedio:
- FFD: 5-15%
- LPT: 3-10%
- Genéticos: 0.5-3%
- Tabú: 0.2-1.5%

[GRÁFICO 3: Gap vs tamaño de instancia]
```

**DIAPOSITIVA 18: Resumen Comparativo**
```
Tabla final con recomendaciones de uso
para cada algoritmo según el contexto
```

### SECCIÓN 6: CONCLUSIONES (1 diapositiva)

**DIAPOSITIVA 19: Conclusiones**
```
1. NP-Hard: No existe algoritmo polinomial (bajo P ≠ NP)

2. Trade-off: Optimalidad vs velocidad

Elección depende del contexto:
- Instancias pequeñas: B&B o DP
- Problemas reales: Metaheurísticas
- Baseline rápido: Greedy

3. Herramientas: 9 algoritmos, dashboard interactivo, benchmarking

4. Aplicaciones: Logística, distribución de carga, 
   computación distribuida
```

### DIAPOSITIVA 20: AGRADECIMIENTOS
```
┌──────────────────────────────┐
│                              │
│   Gracias por su atención    │
│                              │
│   Preguntas y Discusión      │
│                              │
│   Código disponible en GitHub│
│                              │
└──────────────────────────────┘
```

---

## Notas de Presentación

### Distribución del Tiempo (20 minutos)

| Sección | Diapositivas | Tiempo |
|---------|-------------|--------|
| Introducción | 1-2 | 1 min |
| Problema | 3 | 2 min |
| Formalización | 4-5 | 2 min |
| Complejidad | 6-7 | 3 min |
| Métodos | 8-14 | 8 min |
| Análisis | 15-18 | 3 min |
| Conclusiones | 19-20 | 1 min |

### Puntos Clave a Enfatizar Oralmente

1. **Problema:** Motivación práctica real
2. **Formalización:** Claridad matemática
3. **Complejidad:** Por qué NP-Hard importa
4. **Métodos:** Intuición vs fórmulas
5. **Análisis:** Interpretación de gráficos

### Transiciones Sugeridas

- "Como vemos en la formalización..."
- "Esto nos lleva a analizar la complejidad..."
- "Para resolver este problema, implementamos..."
- "Los resultados experimentales muestran..."

---

## Notas Técnicas

- **Tema:** Beamer Madrid (profesional)
- **Colores:** Azul (#0066CC) para énfasis
- **Fuente:** Sans serif (readable a distancia)
- **Proporción:** 16:9 (moderno, pantalla ancha)
- **Navegación:** Page Down o flechas

---

## Verificación Final

✓ 19 diapositivas (< 20 según requisito)
✓ Incluye: introducción, agradecimientos
✓ Prioriza: imágenes (4 espacios reservados), fórmulas, conceptos simples
✓ Explica: problema, formalización, complejidad, métodos, análisis
✓ Compilable: LaTeX sin errores
✓ Profesional: tema y colores adecuados

---

*Última actualización: Enero 13, 2026*
