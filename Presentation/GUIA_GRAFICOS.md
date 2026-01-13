# Guía para Insertar Gráficos en la Presentación

## Resumen Ejecutivo

La presentación contiene **3 espacios reservados para gráficos** que deben ser generados a partir de los datos de los benchmarks del proyecto. Estos gráficos son esenciales para la calidad visual y comprensión de la presentación.

---

## Gráfico 1: Tiempo de Ejecución vs Tamaño de Instancia

### Ubicación en la presentación
**Diapositiva 16:** "Comparación de Tiempos"

### Descripción
Gráfico que compara los tiempos de ejecución de todos los algoritmos implementados en función del número de ítems. Utiliza escala logarítmica en ambos ejes.

### Especificaciones técnicas

| Aspecto | Especificación |
|---------|--------|
| **Escala** | Log-Log (ambos ejes logarítmicos) |
| **Eje X** | Número de ítems ($n$): rango [5, 50] |
| **Eje Y** | Tiempo de ejecución (segundos) |
| **Variables** | $k = 3, 4, 5$ (número de contenedores) |
| **Líneas** | Una línea por algoritmo (9 en total) |
| **Tipo** | Líneas con marcadores (para cada tamaño de $n$) |

### Algoritmos a incluir (en orden sugerido)
1. **FFD (First Fit Decreasing)** - Línea plana/casi plana (O(n log n))
2. **LPT Balanced** - Línea plana/casi plana (O(n log n))
3. **Simulated Annealing** - Línea plana (tiempo fijo)
4. **Algoritmo Genético** - Línea plana (tiempo fijo)
5. **Búsqueda Tabú** - Línea plana (tiempo fijo)
6. **Fuerza Bruta** - Exponencial (O(k^n))
7. **Branch and Bound** - Exponencial débil
8. **Programación Dinámica** - Exponencial fuerte (O(3^n))

### Datos esperados
Los datos se encuentran probablemente en:
- `/home/abraham/Escritorio/mulas/discrete_logistics/benchmarks/`
- O en archivos CSV/JSON generados por `runner.py` o `scalability_analysis.py`

### Cómo insertar en LaTeX

Reemplaza la línea en la diapositiva 16:
```tex
\textbf{[GRÁFICO 1: Tiempo vs tamaño (escala log)]}
```

Por:
```tex
\includegraphics[width=0.85\textwidth]{../benchmarks/timing_comparison.png}
```

Asume que el gráfico está en `benchmarks/timing_comparison.png`

---

## Gráfico 2: Órdenes de Complejidad

### Ubicación en la presentación
**Diapositiva 17:** "Órdenes de Complejidad"

### Descripción
Visualización que muestra el comportamiento teórico vs empírico de los órdenes de complejidad. Comparar diferentes tasas de crecimiento.

### Especificaciones técnicas

| Aspecto | Especificación |
|---------|--------|
| **Tipo** | Líneas de referencia teórica + datos empíricos |
| **Eje X** | Número de ítems |
| **Eje Y** | Tiempo (segundos, escala logarítmica) |
| **Curvas teóricas** | $O(n)$, $O(n \log n)$, $O(2^n)$, $O(3^n)$, $O(k^n)$ |
| **Datos empíricos** | Puntos con barras de error (si se tienen múltiples ejecuciones) |

### Estructura sugerida

```
Línea 1: O(1) o O(n) - Greedy/Metaheurísticas (plana o casi)
Línea 2: O(n log n) - Greedy (crecimiento leve)
Línea 3: O(2^n) - Branch & Bound (exponencial débil)
Línea 4: O(3^n) - Programación Dinámica (exponencial fuerte)
Línea 5: O(k^n) - Fuerza Bruta (exponencial más fuerte)
```

### Cómo insertar

```tex
\includegraphics[width=0.85\textwidth]{../benchmarks/complexity_orders.png}
```

---

## Gráfico 3: Gap de Optimalidad vs Tamaño de Instancia

### Ubicación en la presentación
**Diapositiva 18:** "Calidad de Soluciones"

### Descripción
Muestra cuán lejos está la solución de cada heurística del óptimo (garantizado por algoritmos exactos).

### Especificaciones técnicas

| Aspecto | Especificación |
|---------|--------|
| **Métrica** | Gap de Optimalidad = (Heurístico - Óptimo) / Óptimo × 100 |
| **Eje X** | Tamaño de instancia (n) |
| **Eje Y** | Gap (%) |
| **Rango Y** | [0%, 20%] (típicamente) |
| **Líneas** | FFD, LPT, SA, GA, TS |

### Valores esperados (del informe)

- **FFD:** 5-15% de gap
- **LPT:** 3-10% de gap
- **Simulated Annealing:** ~5-8% de gap
- **Algoritmos Genéticos:** 0.5-3% de gap
- **Búsqueda Tabú:** 0.2-1.5% de gap

### Cómo insertar

```tex
\includegraphics[width=0.85\textwidth]{../benchmarks/optimality_gap.png}
```

---

## Ilustración 1: Descripción del Problema

### Ubicación en la presentación
**Diapositiva 2:** "Descripción del Problema"

### Descripción
Ilustración visual que muestre:
- Múltiples paquetes con diferentes pesos/tamaños
- Múltiples vehículos/contenedores
- Distribución desbalanceada (ejemplo malo)
- Distribución balanceada (ejemplo bueno)

### Sugerencias de contenido

**Lado izquierdo (Desbalanceado):**
- Camión 1: 4 paquetes pequeños (carga baja)
- Camión 2: 1 paquete grande (carga alta)
- Diferencia grande entre cargas

**Lado derecho (Balanceado):**
- Camión 1: Mezcla equilibrada
- Camión 2: Mezcla equilibrada
- Cargas similares

### Herramientas sugeridas
- **TikZ** (si se integra en LaTeX directamente)
- **Inkscape** (exportar a PNG/PDF)
- **Python + Matplotlib** (para gráficos basados en datos)

### Cómo insertar

```tex
\includegraphics[width=0.6\textwidth]{ilustraciones/problema_ilustracion.png}
```

---

## Guía de Generación de Gráficos desde Datos

### Opción 1: Desde Python (Recomendado)

```python
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos de benchmarks
# ... (detalles específicos según la estructura de datos)

# Gráfico 1
plt.figure(figsize=(10, 6))
for algorithm in algorithms:
    plt.loglog(sizes, times[algorithm], marker='o', label=algorithm)
plt.xlabel('Número de ítems (n)')
plt.ylabel('Tiempo (segundos)')
plt.title('Comparación de Tiempos de Ejecución')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('timing_comparison.png', dpi=300, bbox_inches='tight')
```

### Opción 2: Desde LaTeX puro (pgfplots)

Si tienes experiencia con `pgfplots`, puedes generar los gráficos directamente en LaTeX cargando archivos de datos CSV.

---

## Checklist de Implementación

- [ ] Acceder a los datos de benchmarks en `discrete_logistics/benchmarks/`
- [ ] Generar Gráfico 1 (tiempo vs tamaño)
- [ ] Generar Gráfico 2 (órdenes de complejidad)
- [ ] Generar Gráfico 3 (gap de optimalidad)
- [ ] Generar Ilustración 1 (descripción del problema)
- [ ] Crear directorio `benchmarks/` o `ilustraciones/` en `Presentation/`
- [ ] Actualizar rutas en `presentation.tex`
- [ ] Compilar y verificar en PDF
- [ ] Ajustar tamaños y posiciones si es necesario

---

## Archivos Relevantes del Proyecto

Para generar los datos de los gráficos, consulta:

| Archivo | Propósito |
|---------|----------|
| `discrete_logistics/benchmarks/scalability_analysis.py` | Análisis de escalabilidad |
| `discrete_logistics/benchmarks/runner.py` | Ejecución de benchmarks |
| `discrete_logistics/benchmarks/analysis.py` | Análisis de resultados |
| `discrete_logistics/visualizations/plots.py` | Funciones de visualización |

---

## Notas Finales

1. Los gráficos deben ser **profesionales y claros**
2. Incluir **leyendas legibles** y **etiquetas de ejes**
3. Usar **colores distintos** para cada algoritmo
4. Escala logarítmica es esencial para ver todos los algoritmos
5. Resolución mínima: **300 DPI** para impresión

---

**Última actualización:** Enero 13, 2026
