# ⚡ INICIO RÁPIDO - Presentación

## 3 Pasos para Presentar Ahora

### Paso 1: Abre el PDF
```bash
# Linux/macOS
open /home/abraham/Escritorio/mulas/Presentation/presentation.pdf

# O simplemente abre con tu lector PDF favorito:
# - Adobe Reader
# - Evince (GNOME)
# - Preview (macOS)
# - Sumatra PDF (Windows)
```

### Paso 2: Activa Modo Presentación
- **Adobe Reader:** Ctrl+L (o Cmd+L en Mac)
- **Evince:** F5 o Ctrl+Shift+P
- **Preview:** Flechas para navegar
- **Genérico:** Page Down para siguiente diapositiva

### Paso 3: ¡Presenta!
- **Siguiente:** Flecha derecha, Page Down, o Click
- **Anterior:** Flecha izquierda, Page Up, o Click
- **Salir:** ESC

---

## 📋 Cheatsheet de la Presentación

### Estructura Rápida

```
Intro (2 min)
├─ Portada
└─ Contenidos

Problema (2 min)
└─ Descripción + Ejemplo

Formalización (2 min)
├─ Definiciones
└─ Modelo ILP

Complejidad (3 min)
├─ Decisión vs Optimización
└─ Reducciones → NP-Hard

Métodos (8 min)
├─ Clasificación
├─ Greedy (FFD, LPT)
├─ Fuerza Bruta
├─ Branch & Bound
├─ Programación Dinámica
└─ Metaheurísticas

Análisis (3 min)
├─ Tiempos
├─ Complejidad
├─ Calidad
└─ Resumen

Cierre (1 min)
└─ Conclusiones + Agradecimientos

TOTAL: 19 diapositivas, 20 minutos
```

---

## 💡 Puntos Clave a Explicar Oralmente

### Sobre el Problema
- "Imagina una empresa de transporte..."
- "Tenemos n paquetes y k vehículos..."
- "El desafío: equilibrar la carga..."

### Sobre la Complejidad
- "Este problema es NP-Hard..."
- "No existe algoritmo polinomial conocido..."
- "Debemos elegir entre optimalidad y velocidad..."

### Sobre los Métodos
- "Fuerza bruta lo intenta TODO (exhaustivo)..."
- "Greedy elige lo mejor LOCALMENTE..."
- "Programación Dinámica lo divide en SUBPROBLEMAS..."
- "Metaheurísticas buscan de forma INTELIGENTE..."

### Sobre Resultados
- "Como ven en el gráfico: el tiempo crece exponencialmente..."
- "Las metaheurísticas dan buenos resultados en SEGUNDOS..."
- "Los métodos exactos son LENTOS pero GARANTIZAN optimalidad..."

---

## 🎯 Timing Recomendado

| Fase | Duración | Diapositivas |
|------|----------|-------------|
| Introducción | 1 min | 1-2 |
| Problema | 2 min | 3 |
| Formalización | 2 min | 4-5 |
| Complejidad | 3 min | 6-7 |
| Métodos | 8 min | 8-14 |
| Análisis | 3 min | 15-18 |
| Conclusiones | 1 min | 19-20 |
| **TOTAL** | **20 min** | **19 diaps** |

---

## ❓ Preguntas Probables & Respuestas

**P: ¿Por qué es NP-Hard?**
A: Porque hemos reducido desde 3-PARTITION, que es un problema NP-completo conocido. Cualquier solución para nuestro problema resolvería 3-PARTITION.

**P: ¿Qué significa exactamente NP-Hard?**
A: Significa que no existe un algoritmo conocido que lo resuelva en tiempo polinomial. Se cree que no existe (asumiendo P ≠ NP), pero nunca se ha demostrado.

**P: ¿Cuál es el mejor algoritmo?**
A: Depende del contexto. Para instancias pequeñas: Branch & Bound. Para problemas reales: Búsqueda Tabú o Genéticos. Para baseline rápido: Greedy.

**P: ¿Por qué no usar siempre metaheurísticas?**
A: Porque no garantizan optimalidad. A veces necesitas la mejor solución garantizada, no solo una buena aproximación.

**P: ¿Cómo compilaste esto?**
A: Con LaTeX/Beamer. El comando es: `pdflatex presentation.tex`

---

## 🛠️ Si Algo Sale Mal

### No abre el PDF
- Asegúrate de tener un lector PDF instalado
- Intenta: `file presentation.pdf`
- Si está corrupto, recompila: `pdflatex presentation.tex`

### Las diapositivas se ven pequeñas
- Usa zoom del lector PDF (Ctrl++ o Cmd++)
- O abre en pantalla completa

### Quiero cambiar algo
- Edita `presentation.tex`
- Recompila: `pdflatex presentation.tex`
- Verifica: `pdfinfo presentation.pdf | grep Pages`

### Necesito agregar gráficos
- Lee: `GUIA_GRAFICOS.md`
- Genera los gráficos
- Inserta en `presentation.tex`
- Recompila

---

## 📚 Documentación Completa

Si necesitas más información, consulta:

| Archivo | Para qué |
|---------|----------|
| `README.md` | Descripción general |
| `ESTRUCTURA_DETALLADA.md` | Guía para practicar |
| `GUIA_GRAFICOS.md` | Insertar gráficos |
| `INDICE.md` | Navegar documentación |
| `RESUMEN.txt` | Estado general |

---

## ✅ Checklist Pre-Presentación

```
□ PDF abierto y funciona
□ Modo presentación testado
□ Primeras 2 diapositivas revisadas
□ Proyector/pantalla conectada
□ Puntero disponible
□ Agua/bebida nearby
□ Practicado 2-3 veces
□ Notas personales listas
□ Backup en USB (recomendado)
```

---

## 🚀 Listo para Presentar

Estás todo listo. La presentación:

✅ Tiene 19 diapositivas (< 20 requeridas)
✅ Resume el informe completo
✅ Incluye: problema, formalización, complejidad, métodos, análisis
✅ Prioriza: imágenes (4 espacios), fórmulas, conceptos
✅ Está compilada y funcional
✅ Bien documentada

**¡Que te vaya bien!**

---

## 📞 Información Rápida

- **Ubicación:** `/home/abraham/Escritorio/mulas/Presentation/`
- **PDF Principal:** `presentation.pdf` (196 KB)
- **Fuente:** `presentation.tex` (9.3 KB)
- **Documentación:** `README.md`, `INDICE.md`
- **Duración:** 15-20 minutos
- **Autores:** Richard Matos, Abel Ponce, Abraham Romero
- **Institución:** Facultad de Matemática y Computación, Universidad de La Habana

---

*Última actualización: 13 de enero de 2026*

**¡Mucho éxito en tu presentación! 🎉**
