# 🎬 Guión para Video de Presentación del Proyecto
## Predicción de Cancelaciones Hoteleras - 10 minutos

---

## 👥 **Personajes**
- **PRESENTADOR A (Juan)**: Desarrollador principal del proyecto
- **PRESENTADOR B (Laura)**: Experta en machine learning / Co-presentadora

---

## ⏱️ **Estructura Temporal**
- **Introducción**: 0:00 - 1:30 (1.5 min)
- **Problema y Datos**: 1:30 - 3:00 (1.5 min)
- **Metodología**: 3:00 - 5:00 (2 min)
- **Resultados Principales**: 5:00 - 7:30 (2.5 min)
- **Técnicas Avanzadas**: 7:30 - 9:00 (1.5 min)
- **Conclusiones y Demo**: 9:00 - 10:00 (1 min)

---

## 🎥 **GUIÓN COMPLETO**

### **[0:00 - 1:30] INTRODUCCIÓN**

**[FADE IN - Logo del proyecto en pantalla]**

**JUAN**: ¡Hola! Soy Juan Arismendy, y junto con Laura, vamos a presentarles nuestro proyecto de **predicción de cancelaciones hoteleras** usando machine learning.

**LAURA**: ¡Hola! Soy Laura. En los próximos 10 minutos, veremos cómo desarrollamos una solución que supera el estado del arte y está lista para implementación real.

**JUAN**: Este proyecto es especialmente relevante porque las cancelaciones representan un problema de **millones de dólares** para la industria hotelera. 

**[MOSTRAR: Gráfica de impacto económico]**

**LAURA**: Nuestro objetivo era claro: crear un sistema que prediga cancelaciones con alta precisión, usando técnicas avanzadas de machine learning y optimización de características.

**JUAN**: Y los resultados... ¡son impresionantes! Logramos un **F1-Score de 0.814** y un **AUC-ROC de 0.933**, posicionándonos en el **percentil 95+** comparado con la literatura existente.

---

### **[1:30 - 3:00] PROBLEMA Y DATOS**

**[TRANSICIÓN - Mostrar dataset]**

**LAURA**: Empezamos con un dataset real de **119,390 reservas hoteleras** de Kaggle, con información detallada de hoteles urbanos y resort entre 2015-2017.

**JUAN**: El dataset incluye **36 características** como tiempo de anticipación, tipo de cliente, país de origen, tarifa diaria, y por supuesto, si la reserva fue cancelada o no.

**[MOSTRAR: Visualización del dataset]**

**LAURA**: Un dato crucial: el **37.8% de las reservas se cancelan**. Esto significa que tenemos un problema de **clases desbalanceadas** que requiere técnicas especializadas.

**JUAN**: Para hacer el análisis más eficiente, trabajamos con una **muestra estratificada del 10%** - casi 12,000 registros - manteniendo la proporción original de cancelaciones.

**[MOSTRAR: Gráfica de distribución de clases]**

**LAURA**: Realizamos una limpieza rigurosa: eliminamos registros inconsistentes, manejamos valores faltantes, y removimos variables que causarían "data leakage" como el estado final de la reserva.

---

### **[3:00 - 5:00] METODOLOGÍA**

**[TRANSICIÓN - Mostrar pipeline de ML]**

**JUAN**: Nuestra metodología se basa en tres pilares fundamentales: **evaluación comparativa**, **optimización de hiperparámetros**, y **validación rigurosa**.

**LAURA**: Implementamos **5 algoritmos** diferentes: Regresión Logística, K-Nearest Neighbors, Random Forest, Redes Neuronales, y Support Vector Machines.

**[MOSTRAR: Diagrama de los 5 modelos]**

**JUAN**: Cada modelo pasó por un proceso de **Grid Search** con validación cruzada de 5 folds para encontrar los mejores hiperparámetros.

**LAURA**: Para el desbalance de clases, aplicamos **SMOTE** (Synthetic Minority Oversampling Technique) que genera muestras sintéticas de la clase minoritaria.

**[MOSTRAR: Visualización de SMOTE]**

**JUAN**: Nuestro pipeline incluye preprocesamiento automático: **StandardScaler** para variables numéricas y **OneHotEncoder** para categóricas.

**LAURA**: Y algo crucial: usamos **F1-Score** como métrica principal porque balancea precisión y recall, ideal para problemas con clases desbalanceadas.

---

### **[5:00 - 7:30] RESULTADOS PRINCIPALES**

**[TRANSICIÓN - Mostrar tabla de resultados]**

**LAURA**: ¡Y aquí están los resultados! **Random Forest** lidera con un F1-Score de **0.814** y AUC-ROC de **0.933**.

**[MOSTRAR: Tabla comparativa de modelos]**

**JUAN**: **SVM** queda en segundo lugar con **0.797** de F1-Score, seguido por Regresión Logística con **0.768**. KNN y MLP tuvieron rendimientos menores.

**LAURA**: Pero lo más impresionante es la **comparación con el estado del arte**. Mientras la literatura reporta F1-Scores entre 0.65-0.85, nosotros logramos **0.814**.

**[MOSTRAR: Gráfica de comparación con literatura]**

**JUAN**: En AUC-ROC, la literatura va de 0.80-0.92, y nosotros alcanzamos **0.933**, ubicándonos en el **percentil 95+**.

**LAURA**: Esto significa que nuestro modelo no solo es competitivo, sino que **supera significativamente** los resultados publicados anteriormente.

**[MOSTRAR: Gráfica de curvas ROC]**

**JUAN**: Las matrices de confusión muestran que Random Forest tiene excelente balance entre detectar cancelaciones reales y evitar falsas alarmas.

---

### **[7:30 - 9:00] TÉCNICAS AVANZADAS**

**[TRANSICIÓN - Mostrar análisis de características]**

**LAURA**: Pero no nos detuvimos ahí. Implementamos **dos técnicas avanzadas** para optimizar el rendimiento: selección secuencial y extracción PCA.

**JUAN**: La **selección secuencial forward** redujo las características de 33 a 20 - una **reducción del 39.4%** - y sorprendentemente ¡**mejoró** el F1-Score de Random Forest en 9.7%!

**[MOSTRAR: Gráfica de selección de características]**

**LAURA**: Esto demuestra que "menos es más" - eliminamos ruido y nos enfocamos en las características más predictivas.

**JUAN**: Por otro lado, **PCA** con 95% de varianza explicada logró una **reducción del 42.9%** con solo **3.5% de pérdida** en Random Forest.

**[MOSTRAR: Gráfica de análisis PCA]**

**LAURA**: Esto es crucial para implementaciones con recursos limitados - podemos mantener casi el mismo rendimiento con menos de la mitad de las características.

**JUAN**: Las características más importantes incluyen: tiempo de anticipación, tipo de depósito, tipo de cliente, y tarifa diaria promedio.

---

### **[9:00 - 10:00] CONCLUSIONES Y DEMO**

**[TRANSICIÓN - Mostrar dashboard final]**

**LAURA**: En resumen, desarrollamos una **solución integral** que no solo predice cancelaciones, sino que está optimizada para implementación real.

**JUAN**: Nuestras **dos recomendaciones** principales son: Random Forest con selección secuencial para **máximo rendimiento**, o SVM con PCA para **recursos limitados**.

**[MOSTRAR: Roadmap de implementación]**

**LAURA**: El proyecto está **completamente documentado** con código reproducible, visualizaciones profesionales, y guías de implementación.

**JUAN**: Todo el código está disponible en nuestro repositorio de GitHub, organizado profesionalmente con scripts de producción claramente identificados.

**[MOSTRAR: Estructura del repositorio]**

**LAURA**: Este proyecto demuestra que con **metodología rigurosa** y **técnicas avanzadas**, podemos crear soluciones que superan el estado del arte.

**JUAN**: ¡Gracias por su atención! Las preguntas son bienvenidas, y esperamos que este proyecto inspire futuras investigaciones en predicción hotelera.

**[FADE OUT - Información de contacto]**

---

## 📋 **ELEMENTOS VISUALES REQUERIDOS**

### **Gráficas Principales** (usar archivos del proyecto):
1. **model_comparison_chart.png** - Comparación de modelos
2. **state_of_art_comparison.png** - Comparación con literatura
3. **roc_curves.png** - Curvas ROC
4. **feature_selection_results.png** - Selección de características
5. **pca_analysis.png** - Análisis PCA
6. **comprehensive_results_summary.png** - Resumen integral
7. **conclusions_dashboard.png** - Dashboard final

### **Elementos Adicionales**:
- Logo del proyecto
- Gráfica de impacto económico (crear)
- Visualización del dataset
- Diagrama del pipeline ML
- Visualización de SMOTE
- Roadmap de implementación
- Estructura del repositorio

---

## 🎯 **CONSEJOS PARA LA GRABACIÓN**

### **Técnicos**:
- **Ritmo**: Mantener 150-160 palabras por minuto
- **Transiciones**: Usar elementos visuales para cambios de sección
- **Énfasis**: Destacar números clave y resultados
- **Interacción**: Alternar entre presentadores naturalmente

### **Visuales**:
- **Pantalla dividida**: Presentadores + contenido
- **Zoom**: Destacar números importantes en gráficas
- **Animaciones**: Transiciones suaves entre secciones
- **Consistencia**: Mantener estilo visual uniforme

### **Contenido**:
- **Storytelling**: Problema → Solución → Resultados → Impacto
- **Datos concretos**: Siempre respaldar afirmaciones con números
- **Comparaciones**: Mostrar superioridad vs. estado del arte
- **Practicidad**: Enfatizar aplicabilidad real

---

## ⏰ **CRONOMETRAJE DETALLADO**

| Sección | Inicio | Duración | Palabras Aprox. | Enfoque |
|---------|--------|----------|-----------------|---------|
| Introducción | 0:00 | 1:30 | 225 | Hook + Objetivos |
| Problema/Datos | 1:30 | 1:30 | 225 | Contexto + Dataset |
| Metodología | 3:00 | 2:00 | 300 | Técnicas + Pipeline |
| Resultados | 5:00 | 2:30 | 375 | Números + Comparaciones |
| Técnicas Avanzadas | 7:30 | 1:30 | 225 | Optimización |
| Conclusiones | 9:00 | 1:00 | 150 | Resumen + CTA |

**Total: 10:00 minutos | ~1,500 palabras**

---

## 🎬 **CHECKLIST PRE-GRABACIÓN**

### **Preparación**:
- [ ] Revisar guión completo
- [ ] Preparar todas las visualizaciones
- [ ] Configurar software de grabación
- [ ] Probar audio y video
- [ ] Ensayar transiciones entre presentadores

### **Materiales**:
- [ ] Todas las gráficas exportadas en alta resolución
- [ ] Logo del proyecto
- [ ] Información de contacto
- [ ] Link al repositorio
- [ ] Notas de respaldo para cada sección

### **Técnico**:
- [ ] Iluminación adecuada
- [ ] Audio claro sin eco
- [ ] Fondo profesional
- [ ] Conexión estable a internet
- [ ] Software de edición preparado

---

**🎯 ¡Este guión está diseñado para crear un video profesional, dinámico y educativo que destaque los logros del proyecto en exactamente 10 minutos!** 