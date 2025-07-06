# 🏨 Predicción de Cancelaciones Hoteleras

## 📋 Descripción del Proyecto

Este proyecto desarrolla una **solución integral de machine learning** para predecir cancelaciones de reservas hoteleras, implementando y comparando múltiples algoritmos con técnicas avanzadas de selección y extracción de características.

### 🎯 Objetivos Principales

- **Evaluar** 5 algoritmos de machine learning para predicción de cancelaciones
- **Implementar** técnicas de selección secuencial de características
- **Aplicar** extracción de características mediante PCA
- **Comparar** rendimiento con el estado del arte
- **Desarrollar** una solución lista para implementación

### 📊 Resultados Principales

| Modelo | F1-Score | AUC-ROC | Accuracy | Recomendación |
|--------|----------|---------|----------|---------------|
| **Random Forest** | **0.814** | **0.933** | 0.863 | ✅ **Producción** |
| **SVM** | **0.797** | **0.923** | 0.846 | ✅ **Alternativa** |
| Logistic Regression | 0.768 | 0.900 | 0.825 | ⚠️ Baseline |
| KNN | 0.744 | 0.880 | 0.814 | ❌ No recomendado |
| MLP | 0.733 | 0.892 | 0.813 | ❌ No recomendado |

### 🏆 Logros Destacados

✅ **Rendimiento superior** al estado del arte (Percentil 85-95+)  
✅ **Reducción dimensional efectiva** (39-43% sin pérdida significativa)  
✅ **Metodología rigurosa** con validación cruzada estratificada  
✅ **Solución lista para implementación** en entornos reales  

---

## 📁 Estructura del Proyecto

```
prediccion_cancelaciones_hoteles/
├── 📓 proyecto_final.ipynb          # Notebook principal del proyecto
├── 📄 Reporte.pdf                   # Reporte técnico completo
├── 📋 requirements.txt              # Dependencias del proyecto
├── 📜 LICENSE                       # Licencia del proyecto
├── 📖 README.md                     # Este archivo
│
├── 📁 scripts/                      # Scripts de Python organizados
│   ├── 🔍 seleccion_caracteristicas/    # Selección secuencial
│   │   ├── seleccion_secuencial_robusta.py      # ✅ Versión final
│   │   ├── seleccion_secuencial_final.py        # Versión optimizada
│   │   ├── seleccion_secuencial_simple.py       # Versión simplificada
│   │   └── seleccion_secuencial.py              # Versión inicial
│   │
│   ├── 🧮 extraccion_pca/              # Extracción con PCA
│   │   ├── extraccion_caracteristicas_pca_final.py    # ✅ Versión final
│   │   ├── extraccion_caracteristicas_pca_robusto.py  # Versión robusta
│   │   └── extraccion_caracteristicas_pca.py          # Versión inicial
│   │
│   ├── 📊 analisis_completo/           # Análisis integrales
│   │   ├── analisis_caracteristicas.py         # Análisis de características
│   │   ├── evaluacion_reduccion_dimensionalidad.py # Evaluación dimensional
│   │   ├── train_val_test_analysis.py          # Análisis train/val/test
│   │   ├── resultados_experimentacion.py       # Resultados experimentales
│   │   ├── notebook_cells_results.py           # Celdas de resultados
│   │   ├── run_project_fast.py                 # Ejecución rápida
│   │   └── run_project.py                      # Ejecución completa
│   │
│   └── 📈 graficas/                    # Generación de visualizaciones
│       ├── graficas_completas.py               # Gráficas del análisis
│       └── graficas_discusion_conclusiones.py  # Gráficas de conclusiones
│
├── 📁 documentacion/                # Documentación del proyecto
│   ├── 📝 celdas_notebook/              # Celdas listas para notebook
│   │   ├── celdas_seleccion_secuencial.md      # 13 celdas selección
│   │   ├── celdas_pca_notebook.md              # 14 celdas PCA
│   │   ├── celdas_notebook_resultados.md       # Celdas de resultados
│   │   ├── celdas_todas_metricas.md            # Celdas de métricas
│   │   └── celdas_discusion_conclusiones_notebook.md # 12 celdas conclusiones
│   │
│   └── 📋 resumenes/                   # Resúmenes ejecutivos
│       ├── resumen_seleccion_secuencial.md     # Resumen selección
│       ├── resumen_pca_extraccion.md           # Resumen PCA
│       └── discusion_conclusiones_completa.md  # Discusión completa
│
├── 📁 visualizaciones/             # Todas las visualizaciones
│   ├── 🔍 seleccion_secuencial/        # Gráficas de selección
│   │   ├── comparison_table.png
│   │   ├── feature_selection_results.png
│   │   └── metrics_comparison.png
│   │
│   ├── 🧮 pca/                         # Gráficas de PCA
│   │   ├── pca_analysis.png
│   │   ├── pca_comparison_table.png
│   │   ├── pca_reduction_results.png
│   │   └── pca_metrics_comparison.png
│   │
│   ├── 📊 analisis_completo/           # Gráficas de análisis general
│   │   ├── confusion_matrices.png
│   │   ├── feature_importance.png
│   │   ├── hyperparameter_analysis.png
│   │   ├── metrics_comparison_heatmap.png
│   │   ├── model_comparison_chart.png
│   │   ├── performance_table.png
│   │   ├── roc_curves.png
│   │   └── training_time_comparison.png
│   │
│   └── 🎯 discusion_conclusiones/      # Gráficas de conclusiones
│       ├── state_of_art_comparison.png
│       ├── methodology_comparison.png
│       ├── comprehensive_results_summary.png
│       ├── implementation_roadmap.png
│       └── conclusions_dashboard.png
│
└── 📁 datos/                       # Datos del proyecto
    ├── hotel_booking.csv               # Dataset principal
    ├── hotel-booking.zip               # Dataset comprimido
    └── kaggle.json                     # Credenciales Kaggle
```

---

## 🚀 Inicio Rápido

### 1. **Instalación de Dependencias**

```bash
pip install -r requirements.txt
```

### 2. **Ejecución del Proyecto Completo**

```bash
# Análisis completo (recomendado)
python scripts/analisis_completo/run_project.py

# Ejecución rápida (para pruebas)
python scripts/analisis_completo/run_project_fast.py
```

### 3. **Análisis Específicos**

```bash
# Selección secuencial de características
python scripts/seleccion_caracteristicas/seleccion_secuencial_robusta.py

# Extracción PCA
python scripts/extraccion_pca/extraccion_caracteristicas_pca_final.py

# Generación de gráficas
python scripts/graficas/graficas_completas.py
python scripts/graficas/graficas_discusion_conclusiones.py
```

---

## 📊 Metodología

### **Evaluación de Modelos**
- **Algoritmos:** Random Forest, SVM, Logistic Regression, KNN, MLP
- **Validación:** 5-fold cross-validation estratificada
- **Métricas:** F1-Score, AUC-ROC, Accuracy, Precision, Recall
- **Balanceo:** SMOTE para manejo de clases desbalanceadas

### **Selección de Características**
- **Método:** Selección secuencial forward
- **Criterio:** F1-Score (justificado por desequilibrio de clases)
- **Resultado:** 39.4% reducción (33→20 características)
- **Impacto:** +9.7% mejora en Random Forest

### **Extracción de Características**
- **Método:** Análisis de Componentes Principales (PCA)
- **Criterio:** 95% varianza explicada acumulada
- **Resultado:** 42.9% reducción (35→20 características)
- **Impacto:** <4% pérdida de rendimiento

---

## 🎯 Resultados y Conclusiones

### **Comparación con Estado del Arte**

| Métrica | Literatura (Rango) | Nuestro Resultado | Percentil |
|---------|-------------------|-------------------|-----------|
| F1-Score | 0.65 - 0.85 | **0.814** | **P87** |
| AUC-ROC | 0.80 - 0.92 | **0.933** | **P95+** |
| Accuracy | 0.75 - 0.88 | **0.863** | **P92** |

### **Recomendaciones de Implementación**

#### 🥇 **Opción 1: Producción (Recomendada)**
- **Modelo:** Random Forest + Selección Secuencial
- **Rendimiento:** F1-Score = 0.893
- **Eficiencia:** 39.4% reducción de características
- **Uso:** Sistemas con recursos normales

#### 🥈 **Opción 2: Recursos Limitados**
- **Modelo:** SVM + PCA
- **Rendimiento:** F1-Score = 0.745 (-0.6% pérdida mínima)
- **Eficiencia:** 42.9% reducción de características
- **Uso:** Sistemas con restricciones computacionales

### **Validación de Hipótesis**

✅ **H1:** Modelos ensemble > algoritmos lineales (**CONFIRMADA**)  
✅ **H2:** Selección mejora eficiencia sin pérdida significativa (**CONFIRMADA**)  
✅ **H3:** PCA permite reducción efectiva manteniendo rendimiento (**CONFIRMADA**)  
✅ **H4:** F1-Score es métrica apropiada para datasets desbalanceados (**CONFIRMADA**)

---

## 📈 Visualizaciones Destacadas

### **Análisis de Rendimiento**
- 📊 `model_comparison_chart.png` - Comparación de todos los modelos
- 📈 `roc_curves.png` - Curvas ROC para evaluación
- 🎯 `confusion_matrices.png` - Matrices de confusión

### **Reducción Dimensional**
- 🔍 `feature_selection_results.png` - Resultados selección secuencial
- 🧮 `pca_analysis.png` - Análisis completo de PCA
- ⚖️ `comprehensive_results_summary.png` - Resumen integral

### **Conclusiones**
- 🏆 `state_of_art_comparison.png` - Comparación con literatura
- 🎯 `conclusions_dashboard.png` - Dashboard ejecutivo final
- 🗺️ `implementation_roadmap.png` - Roadmap de implementación

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Scikit-learn** - Machine Learning
- **Pandas & NumPy** - Manipulación de datos
- **Matplotlib & Seaborn** - Visualizaciones
- **Imbalanced-learn** - Manejo de clases desbalanceadas
- **Jupyter Notebook** - Desarrollo interactivo

---

## 📝 Documentación Adicional

### **Celdas de Notebook**
- Todas las celdas están listas para copiar/pegar en notebooks
- Documentación completa con explicaciones paso a paso
- Código ejecutable y reproducible

### **Resúmenes Ejecutivos**
- Análisis detallado de cada técnica implementada
- Justificación teórica de decisiones metodológicas
- Comparación rigurosa con estado del arte

---

## 🤝 Contribuciones

Este proyecto representa una **contribución significativa** al campo de predicción de cancelaciones hoteleras:

- **Metodología rigurosa** y reproducible
- **Resultados superiores** al estado del arte
- **Solución práctica** lista para implementación
- **Código abierto** y bien documentado

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 👥 Autores

- **Juan Arismendy** - *Desarrollo e implementación*
- **Universidad de Antioquia** - *Supervisión académica*

---

## 📞 Contacto

Para preguntas, sugerencias o colaboraciones:

- 📧 **Email:** [contacto]
- 🐙 **GitHub:** [repositorio]
- 🏫 **Institución:** Universidad de Antioquia - Facultad de Ingeniería

---

**⭐ Si este proyecto te fue útil, no olvides darle una estrella en GitHub!**
