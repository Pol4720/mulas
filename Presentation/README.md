# Presentación: Problema de Transporte Logístico Discreto

## Descripción

Esta presentación resume el informe académico sobre el "Problema de Transporte Logístico Discreto" (Balanced Multi-Bin Packing with Capacity Constraints). La presentación tiene un total de **19 diapositivas** y está diseñada para ser expuesta oralmente.

## Estructura de la Presentación

### 1. Introducción y Portada (1 diapositiva)
- Título, autores e institución

### 2. Índice (1 diapositiva)
- Tabla de contenidos

### Sección 1: El Problema (1 diapositiva)
- **Descripción del Problema:** Presentación clara del escenario logístico real
- Incluye espacio para una ilustración con camiones y paquetes

### Sección 2: Formalización Matemática (2 diapositivas)
- **Definiciones Fundamentales:** Ítems, pesos, valores, contenedores y variables de decisión
- **Formulación ILP:** Planteo como programa lineal entero con función objetivo y restricciones
- Énfasis en la propiedad del óptimo

### Sección 3: Análisis de Complejidad (2 diapositivas)
- **Problema de Decisión vs Optimización:** Distinción entre ambos conceptos
- **Cadena de Reducciones:** PARTITION → 3-PARTITION → BALANCED-BIN-PACKING
- Conclusión: Problema NP-Hard, trade-off optimalidad vs velocidad

### Sección 4: Métodos de Resolución (7 diapositivas)
- **Clasificación de Algoritmos:** Tabla comparativa de exactos, aproximados y metaheurísticas
- **Algoritmos Greedy:** FFD y LPT con complejidades y características
- **Fuerza Bruta:** Límites prácticos y papel en validación
- **Branch and Bound:** Idea de poda inteligente y complejidad
- **Programación Dinámica:** Estado, transiciones y complejidad exponencial
- **Metaheurísticas:** Esquema general y características

### Sección 5: Análisis Experimental (4 diapositivas)
- **Comparación de Tiempos:** Gráfico de tiempo vs tamaño (escala logarítmica)
- **Órdenes de Complejidad:** Visualización de diferentes comportamientos exponenciales
- **Calidad de Soluciones:** Gap de optimalidad promedio para cada algoritmo
- **Resumen Comparativo:** Tabla con óptimo, velocidad y caso de uso

### Sección 6: Conclusiones (1 diapositiva)
- Resumen de hallazgos principales
- Recomendaciones de uso según contexto

### Agradecimientos (1 diapositiva)
- Cierre y apertura a preguntas

## Características de Diseño

### Prioridad: Imágenes y Fórmulas
- Se proporciona espacio explícito para ilustraciones
- Se utilizan fórmulas matemáticas claras y formales
- Textos concisos para complementar exposición oral
- Evita explicaciones extensas (se proporcionan oralmente)

### Colores y Tema
- Tema **Madrid** de Beamer
- Colores personalizados: Azul (#0066CC) para énfasis
- Estructura profesional y clara

### Contenido Visual Reservado
Se dejan espacios explícitos para los siguientes gráficos (a insertar posteriormente):
1. **Gráfico 1:** Tiempo de ejecución vs tamaño de instancia (escala log)
2. **Gráfico 2:** Comparación de órdenes de complejidad
3. **Gráfico 3:** Gap de optimalidad vs tamaño de instancia
4. **Ilustración 1:** Ejemplo visual del problema (camiones y paquetes)

## Requisitos para Compilación

```bash
# Compilar a PDF
pdflatex presentation.tex

# O con compilación múltiple (recomendado para tabla de contenidos)
pdflatex presentation.tex
pdflatex presentation.tex
```

## Requisitos de Sistema

- **TeX Live** o **MiKTeX** (distribuición LaTeX)
- Paquetes necesarios (incluidos en cualquier instalación estándar):
  - `beamer` (presentaciones)
  - `amsmath`, `amssymb` (símbolos matemáticos)
  - `booktabs` (tablas profesionales)
  - `xcolor`, `tikz` (gráficos)
  - `babel` (soporte para español)

## Archivos

- `presentation.tex` - Código fuente de la presentación
- `presentation.pdf` - PDF compilado (195 KB)

## Cómo Añadir Gráficos

Para añadir los gráficos, modifica las líneas correspondientes en el archivo `.tex`:

```tex
% Ejemplo para el Gráfico 1
\includegraphics[width=0.8\textwidth]{graficos/tiempo_vs_tamaño.png}
```

## Sugerencias para la Exposición

1. **Duración:** 15-20 minutos
2. **Énfasis oral en:**
   - Motivación del problema real
   - Intuición detrás de cada algoritmo
   - Interpretación de gráficos
   - Recomendaciones prácticas

3. **Puntos clave a expandir oralmente:**
   - ¿Por qué NP-Hard es importante?
   - Diferencia entre garantía de optimalidad y velocidad
   - Cuándo usar cada algoritmo

## Notas Finales

- Las diapositivas siguen el principio "menos es más"
- Los gráficos son esenciales para la comprensión
- Se deja ample espacio para notas personales del presentador
- La estructura permite seguimiento fácil para la audiencia

---

**Autores:** Richard Alejandro Matos Arderí, Abel Ponce González, Abraham Romero Imbert

**Institución:** Facultad de Matemática y Computación, Universidad de La Habana

**Fecha:** Enero 2026
