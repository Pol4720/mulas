# Diseño del Algoritmo Híbrido HybridDPMeta

## 1. Motivación y Contexto

### Problema
- **DP es exacto pero lento**: O(k² · 3^n) - solo práctico para n ≤ 15
- **Metaheurísticas son rápidas pero no óptimas**: No tienen garantías teóricas
- **Necesidad**: Un algoritmo que combine lo mejor de ambos mundos

### Idea Central
Dividir el problema en dos fases:
1. **Fase Rápida (Metaheurística)**: Asignar un porcentaje de ítems rápidamente
2. **Fase Exacta (DP)**: Resolver el subproblema restante de forma óptima

## 2. Diseño del Algoritmo HybridDPMeta

### Parámetros Clave
- `dp_threshold`: Número máximo de ítems que DP puede manejar eficientemente (default: 12)
- `meta_algorithm`: Metaheurística a usar ('simulated_annealing', 'genetic', 'tabu')
- `partition_strategy`: Cómo dividir los ítems ('largest_first', 'random', 'clustering')
- `quality_weight`: Peso para balance calidad/tiempo en selección de estrategia

### Estrategias de Partición

#### 2.1 Largest First (Recomendada)
Asignar primero los ítems más grandes con metaheurística:
- Ítems grandes tienen mayor impacto en el balance
- Dejar ítems pequeños para DP permite ajuste fino
- Justificación: Los ítems pequeños son más fáciles de "encajar" óptimamente

#### 2.2 Value-Based
Asignar ítems por valor:
- Ítems de alto valor primero (metaheurística)
- Ítems de bajo valor después (DP para ajuste fino del balance)

#### 2.3 Clustering
Agrupar ítems similares:
- Crear clusters por peso/valor
- Asignar clusters grandes con meta, individuales con DP

### Pseudocódigo

```
ALGORITHM HybridDPMeta(items, bins, dp_threshold=12):
    n = |items|
    k = |bins|
    
    IF n <= dp_threshold:
        RETURN DP.solve(items, bins)  # Exacto para instancias pequeñas
    
    # Fase 1: Partición de ítems
    n_meta = n - dp_threshold  # Ítems para metaheurística
    n_dp = dp_threshold        # Ítems para DP
    
    meta_items, dp_items = partition_items(items, n_meta, strategy)
    
    # Fase 2: Resolver con metaheurística
    # Crear problema parcial con todos los bins pero solo meta_items
    partial_problem = Problem(meta_items, bins)
    meta_solution = Metaheuristic.solve(partial_problem)
    
    # Fase 3: Calcular capacidades residuales
    residual_capacities = []
    FOR j = 1 TO k:
        used_weight = sum(item.weight for item in meta_solution.bins[j])
        residual = bins[j].capacity - used_weight
        residual_capacities.append(residual)
    
    # Fase 4: Resolver subproblema exactamente
    # dp_items debe asignarse a bins con capacidades residuales
    residual_problem = Problem(dp_items, bins_with_residual_capacities)
    dp_solution = DP.solve(residual_problem)
    
    # Fase 5: Combinar soluciones
    final_solution = merge_solutions(meta_solution, dp_solution)
    
    RETURN final_solution
```

### Optimizaciones

1. **Adaptive Threshold**: Ajustar dp_threshold basado en k y tiempo disponible
2. **Early Termination**: Si metaheurística encuentra solución muy buena, no usar DP
3. **Iterative Refinement**: Permitir múltiples iteraciones meta→DP
4. **Caching**: Cache de subproblemas DP para problemas similares

## 3. Framework de Experimentación

### Diseño Experimental

#### Variables Independientes
- Tamaño del problema (n): [15, 20, 25, 30, 40, 50, 75, 100]
- Número de bins (k): [2, 3, 4, 5]
- Tipo de distribución: [uniform, normal, correlated, clustered]
- Variación de capacidad: [0, 0.2, 0.4]

#### Variables Dependientes
- Calidad de solución (objetivo: max-min)
- Tiempo de ejecución
- Gap de optimalidad (para instancias pequeñas con solución conocida)

#### Métricas Compuestas
- **Score Ponderado**: α × (1 - normalized_objective) + (1-α) × (1 - normalized_time)
- **Ratio Calidad/Tiempo**: objective_improvement / time_increase

### Pruebas Estadísticas

#### 1. Comparación de Medianas (Mann-Whitney U)
- H0: No hay diferencia entre algoritmos
- H1: Híbrido es mejor
- α = 0.05

#### 2. Comparación de Medias (t-Student pareado)
- Para distribuciones normales
- Test de normalidad previo (Shapiro-Wilk)

#### 3. Análisis de Varianza (ANOVA)
- Comparar múltiples algoritmos simultáneamente
- Post-hoc: Tukey HSD

#### 4. Efecto del Tamaño (Cohen's d)
- Medir magnitud del efecto, no solo significancia

### Número de Repeticiones
- Mínimo 30 repeticiones por configuración (CLT)
- 50 repeticiones para alta confiabilidad
- Seeds fijos para reproducibilidad

## 4. Estructura de Archivos

```
discrete_logistics/
├── algorithms/
│   └── hybrid.py                 # Algoritmo híbrido
├── benchmarks/
│   ├── hybrid_experiment.py      # Framework experimental
│   ├── statistical_tests.py      # Tests estadísticos
│   └── results/
│       └── hybrid/               # Resultados del híbrido
│           ├── raw_results.csv
│           ├── statistical_analysis.json
│           └── figures/
└── dashboard/
    └── pages/
        └── 5_🧬_Algoritmo_Hibrido.py  # Página especial
```

## 5. Métricas de Éxito

1. **Calidad**: Híbrido debe tener gap < 5% respecto a DP en instancias pequeñas
2. **Velocidad**: Híbrido debe ser >10x más rápido que DP para n > 20
3. **Escalabilidad**: Híbrido debe manejar n = 100 en < 60 segundos
4. **Robustez**: Resultados consistentes (bajo CV) en múltiples repeticiones
5. **Estadística**: p-value < 0.05 en comparaciones con otros algoritmos

## 6. Plan de Implementación

### Fase 1: Algoritmo Base (Día 1)
- [ ] Implementar clase HybridDPMeta
- [ ] Estrategias de partición
- [ ] Integración con DP y SA existentes

### Fase 2: Experimentación (Día 2)
- [ ] Framework de benchmarking
- [ ] Generación de instancias variadas
- [ ] Ejecución paralela de experimentos

### Fase 3: Análisis Estadístico (Día 2-3)
- [ ] Implementar tests estadísticos
- [ ] Análisis de resultados
- [ ] Visualizaciones

### Fase 4: Dashboard (Día 3)
- [ ] Página interactiva
- [ ] Gráficos exportables
- [ ] Documentación integrada
