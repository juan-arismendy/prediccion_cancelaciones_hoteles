# 📁 Estructura del Proyecto: Predicción de Cancelaciones Hoteleras

## 🎯 Descripción General
Este proyecto implementa modelos de machine learning para predecir cancelaciones de reservas hoteleras, incluyendo análisis de características, reducción de dimensionalidad y evaluación comparativa.

## 📂 Estructura de Carpetas

### 🚀 **Scripts/** - Código fuente principal
```
scripts/
├── analisis/                    # Scripts de análisis de características
│   ├── analisis_seleccion_secuencial.py
│   ├── analisis_extraccion_pca.py
│   └── analisis_recomendacion_final.py
├── entrenamiento/               # Scripts de entrenamiento de modelos
│   ├── proyecto_final.py
│   ├── proyecto_final_comentado.py
│   └── proyecto_final_with_persistence.py
├── utilidades/                  # Herramientas y utilidades
│   ├── model_persistence.py
│   ├── model_persistence_fast.py
│   ├── quick_access_models.py
│   └── config_joblib.py
└── visualizacion/              # Scripts de generación de gráficas
    ├── generar_graficas_pca.py
    └── crear_imagenes_video.py
```

### 📊 **Resultados/** - Outputs del análisis
```
resultados/
├── analisis/                   # Resultados tabulares
│   ├── sequential_selection_results.csv
│   └── pca_extraction_results.csv
└── graficas/                   # Imágenes generadas
    ├── sequential_selection_*.png
    ├── pca_*.png
    └── otros gráficos...
```

### 📚 **Documentación/** - Documentación técnica y presentación
```
documentacion/
├── tecnica/                    # Documentación técnica
│   ├── extract_docx_content.py
│   ├── README_persistence.md
│   ├── resumen_mejoras_implementadas.md
│   ├── mejoras_reporte_sugeridas.md
│   └── seccion_*.md
└── presentacion/               # Material de presentación
    ├── guion_video_presentacion.md
    ├── notas_presentacion_video.md
    ├── lista_imagenes_cronologica_video.md
    └── imagenes_video_completas.md
```

### 💾 **Datos y Modelos**
```
datos/                          # Dataset original
saved_models/                   # Modelos entrenados y resultados
├── models/                     # Modelos serializados
├── results/                    # Resultados de evaluación
└── metadata/                   # Metadatos de sesiones
```

## 🎯 **Archivos Principales en Raíz**

### 📋 **Archivos de Configuración**
- `README.md` - Documentación principal del proyecto
- `requirements.txt` - Dependencias del proyecto
- `.gitignore` - Archivos a ignorar en Git
- `LICENSE` - Licencia del proyecto

### 📊 **Archivos de Datos**
- `proyecto_final.ipynb` - Notebook principal con análisis completo
- `Reporte.pdf` - Reporte final del proyecto

### 🎬 **Archivos de Presentación**
- `ORGANIZACION_PROYECTO.md` - Guía de organización
- `ESTRUCTURA_PROYECTO.md` - Este archivo

## 🚀 **Cómo Usar la Estructura**

### 1. **Entrenamiento de Modelos**
```bash
cd scripts/entrenamiento/
python proyecto_final_with_persistence.py
```

### 2. **Análisis de Características**
```bash
cd scripts/analisis/
python analisis_seleccion_secuencial.py
python analisis_extraccion_pca.py
python analisis_recomendacion_final.py
```

### 3. **Acceso Rápido a Modelos**
```bash
cd scripts/utilidades/
python quick_access_models.py
```

### 4. **Generación de Gráficas**
```bash
cd scripts/visualizacion/
python generar_graficas_pca.py
```

## 📈 **Flujo de Trabajo Recomendado**

1. **Preparación**: Revisar `README.md` y `requirements.txt`
2. **Entrenamiento**: Ejecutar scripts en `scripts/entrenamiento/`
3. **Análisis**: Ejecutar scripts en `scripts/analisis/`
4. **Visualización**: Generar gráficas con scripts en `scripts/visualizacion/`
5. **Documentación**: Revisar resultados en `resultados/` y documentación en `documentacion/`

## 🔧 **Características de la Organización**

### ✅ **Ventajas**
- **Separación clara**: Código, datos, resultados y documentación separados
- **Reutilización**: Scripts modulares y reutilizables
- **Trazabilidad**: Resultados organizados por tipo de análisis
- **Escalabilidad**: Fácil agregar nuevos análisis o modelos

### 📋 **Convenciones**
- Scripts de análisis: `analisis_*.py`
- Scripts de entrenamiento: `proyecto_final*.py`
- Resultados: Archivos `.csv` y `.png`
- Documentación: Archivos `.md`

## 🎯 **Recomendación Final del Proyecto**

**Modelo Original**: RandomForest (F1-Score: 0.814)
**Técnica de Reducción**: PCA
**Modelo Final**: SVM con PCA (F1-Score: 0.781)
**Reducción**: 99.9% de dimensionalidad

---

*Esta estructura facilita la navegación, mantenimiento y extensión del proyecto de predicción de cancelaciones hoteleras.* 