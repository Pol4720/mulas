# Índice de Archivos - Presentación

## Archivos Principales

### 1. **presentation.pdf** (196 KB) ⭐
   - **Propósito:** PDF compilado, listo para usar
   - **Uso:** Abrir con cualquier lector PDF, presentar en pantalla
   - **Características:** 19 diapositivas, 16:9, tema Madrid
   - **Acción:** Usar directamente para presentación

### 2. **presentation.tex** (9.3 KB)
   - **Propósito:** Código fuente de la presentación
   - **Uso:** Editar y recompilar si se necesitan cambios
   - **Lenguaje:** LaTeX/Beamer
   - **Acción:** Modificar contenido, insertar gráficos, recompilar

---

## Documentación

### 3. **README.md** (4.9 KB)
   - **Propósito:** Documentación general de la presentación
   - **Contiene:**
     - Descripción completa
     - Estructura de secciones
     - Requisitos de compilación
     - Instrucciones de uso
   - **Público:** Cualquiera que quiera entender la presentación
   - **Acción:** Leer para contexto general

### 4. **GUIA_GRAFICOS.md** (7 KB) 📊
   - **Propósito:** Especificaciones para insertar gráficos
   - **Contiene:**
     - 3 gráficos: tiempo, complejidad, optimality gap
     - 1 ilustración: descripción visual del problema
     - Especificaciones técnicas detalladas
     - Cómo insertar en LaTeX
     - Valores esperados
   - **Público:** Desarrolladores/diseñadores que generan gráficos
   - **Acción:** Usar para generar/insertar visualizaciones

### 5. **ESTRUCTURA_DETALLADA.md** (8+ KB)
   - **Propósito:** Vista detallada de contenido de cada diapositiva
   - **Contiene:**
     - Resumen de estructura (19 diapositivas)
     - Mock-up de cada diapositiva
     - Distribución de tiempo
     - Puntos clave a enfatizar
     - Notas técnicas
   - **Público:** Presentador, para practicar
   - **Acción:** Usar como guía durante ensayo

### 6. **RESUMEN.txt** (8.8 KB)
   - **Propósito:** Resumen ejecutivo de lo completado
   - **Contiene:**
     - Archivos generados
     - Estadísticas
     - Características principales
     - Elementos visuales pendientes
     - Instrucciones de uso
     - Próximos pasos
   - **Público:** Referencia rápida
   - **Acción:** Consultar para estado general

### 7. **INDICE.md** (este archivo)
   - **Propósito:** Guía de navegación de toda la documentación
   - **Contiene:** Descripción de cada archivo
   - **Acción:** Saber qué leer según necesidad

---

## Flujo de Uso Recomendado

### Para Presentación Inmediata:
```
1. Abrir: presentation.pdf
2. Consultar: ESTRUCTURA_DETALLADA.md (para puntos clave)
3. Presentar
```

### Para Mejorar la Presentación:
```
1. Leer: GUIA_GRAFICOS.md
2. Generar: gráficos usando datos de benchmarks
3. Editar: presentation.tex
4. Compilar: pdflatex presentation.tex
5. Verificar: presentation.pdf
```

### Para Entender Completamente:
```
1. Leer: README.md (contexto general)
2. Revisar: ESTRUCTURA_DETALLADA.md (contenido)
3. Consultar: GUIA_GRAFICOS.md (visualizaciones)
4. Leer: RESUMEN.txt (logros y próximos pasos)
```

---

## Mapa de Decisiones

```
¿Necesito presentar ahora?
├─ Sí → Abrir presentation.pdf
└─ No, primero quiero mejorarla
   ├─ ¿Insertar gráficos?
   │  └─ Leer: GUIA_GRAFICOS.md
   ├─ ¿Cambiar contenido?
   │  └─ Editar: presentation.tex
   └─ ¿Practicar presentación?
      └─ Usar: ESTRUCTURA_DETALLADA.md

¿No sé por dónde empezar?
└─ Leer en orden: README.md → ESTRUCTURA_DETALLADA.md → RESUMEN.txt
```

---

## Checklist Pre-Presentación

- [ ] Compilar: `pdflatex presentation.tex`
- [ ] Verificar: `pdfinfo presentation.pdf | grep Pages` (debe ser 19)
- [ ] Revisar: Primeras 3 diapositivas
- [ ] Revisar: Secciones críticas (6-7, 15-18)
- [ ] Practicar: Con ESTRUCTURA_DETALLADA.md como guía
- [ ] Preparar: Transiciones entre secciones
- [ ] Comprobar: Proyector/pantalla funciona correctamente
- [ ] Backup: Copiar presentation.pdf a dispositivo USB

---

## Información de Contacto / Repositorio

**Proyecto:** Problema de Transporte Logístico Discreto

**Autores:**
- Richard Alejandro Matos Arderí
- Abel Ponce González
- Abraham Romero Imbert

**Institución:** Facultad de Matemática y Computación, Universidad de La Habana

**Repositorio:** https://github.com/Pol4720/mulas

**Fecha de Presentación:** Enero 2026

---

## Requisitos Técnicos

| Componente | Requisito |
|-----------|-----------|
| Sistema Operativo | Cualquiera (Linux, Windows, macOS) |
| Lector PDF | Adobe Reader, Evince, Preview, etc. |
| Editor LaTeX | TeX Live / MiKTeX (si se modifica .tex) |
| Compilador | pdflatex (incluido en TeX Live) |
| Espacio Disco | ~500 MB (instalación TeX Live) |

---

## Estadísticas Generales

| Métrica | Valor |
|---------|-------|
| Total de archivos | 10 |
| Total de documentación | ~35 KB |
| PDF compilado | 196 KB |
| Diapositivas | 19 |
| Secciones principales | 6 |
| Gráficos/Ilustraciones pendientes | 4 |
| Tiempo de presentación | 15-20 min |

---

## Historial de Cambios

| Fecha | Cambio |
|-------|--------|
| 2026-01-13 | Presentación completada con 19 diapositivas |
| 2026-01-13 | Documentación completa generada |
| 2026-01-13 | Espacios visuales identificados |

---

## Preguntas Frecuentes

**P: ¿Puedo editar las diapositivas?**
A: Sí, modifica presentation.tex y recompila con `pdflatex presentation.tex`

**P: ¿Dónde inserto gráficos?**
A: Lee GUIA_GRAFICOS.md para instrucciones detalladas

**P: ¿Cuánto debo practicar?**
A: Al menos 2-3 veces. Usa ESTRUCTURA_DETALLADA.md como guía.

**P: ¿Qué paquetes LaTeX necesito?**
A: Instala TeX Live completo (incluye todo lo necesario)

**P: ¿Puedo cambiar el tema?**
A: Sí, en la línea `\usetheme{Madrid}` de presentation.tex

---

## Recursos Útiles

- **LaTeX/Beamer:** https://www.overleaf.com/learn/latex/Beamer
- **Tema Madrid:** Documentación en TeX Live
- **Matplotlib (para gráficos):** https://matplotlib.org/
- **GitHub (repositorio):** https://github.com/Pol4720/mulas

---

**Última actualización:** 13 de enero de 2026

*Todos los archivos están organizados en:*
*/home/abraham/Escritorio/mulas/Presentation/*
