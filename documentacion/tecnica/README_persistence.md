# 🚀 Sistema de Persistencia de Modelos - Predicción de Cancelaciones Hoteleras

## 📋 Descripción General

Este sistema permite **guardar y cargar modelos entrenados** para evitar reentrenamiento cada vez que necesites usar los modelos para análisis adicionales como selección de características, PCA, o experimentos.

## 🎯 Beneficios

- ⚡ **Ahorro de tiempo**: No necesitas reentrenar modelos cada vez
- 💾 **Persistencia completa**: Guarda modelos, resultados y metadatos
- 🔄 **Reutilización**: Acceso rápido a modelos para experimentos
- 📊 **Trazabilidad**: Mantiene historial de entrenamientos
- 🛠️ **Flexibilidad**: Carga modelos específicos o sesiones completas

## 📁 Estructura de Archivos

```
prediccion_cancelaciones_hoteles/
├── proyecto_final.py                    # Script original (sin persistencia)
├── proyecto_final_with_persistence.py   # Script con persistencia integrada
├── model_persistence.py                 # Sistema de persistencia
├── quick_access_models.py              # Acceso rápido a modelos
├── saved_models/                       # Directorio de modelos guardados
│   ├── models/                         # Modelos entrenados (.pkl)
│   ├── results/                        # Resultados de evaluación (.json)
│   └── metadata/                       # Metadatos de entrenamiento (.json)
└── README_persistence.md               # Este archivo
```

## 🚀 Uso Rápido

### 1. Entrenamiento Inicial (Primera vez)

```bash
# Ejecutar entrenamiento completo con persistencia
python proyecto_final_with_persistence.py
```

**Lo que hace:**
- Entrena los 5 modelos (LogisticRegression, KNN, RandomForest, MLP, SVM)
- Guarda automáticamente todos los modelos y resultados
- Crea una sesión con timestamp único

### 2. Acceso Rápido (Veces posteriores)

```bash
# Cargar modelos sin reentrenar
python quick_access_models.py
```

**Lo que hace:**
- Carga automáticamente la sesión más reciente
- Muestra resumen de resultados
- Hace disponibles los modelos para uso

## 📊 Funciones Principales

### `ModelPersistenceManager`

Clase principal para gestionar la persistencia:

```python
from model_persistence import ModelPersistenceManager

# Inicializar
persistence_manager = ModelPersistenceManager()

# Guardar modelo
persistence_manager.save_model(model, "RandomForest", "v1.0")

# Cargar modelo
model = persistence_manager.load_model("RandomForest", "v1.0")

# Guardar resultados
persistence_manager.save_results(results, "RandomForest", "v1.0")

# Cargar resultados
results = persistence_manager.load_results("RandomForest", "v1.0")
```

### `train_and_save_models(X, y, session_name=None)`

Entrena y guarda todos los modelos:

```python
from model_persistence import train_and_save_models

# Entrenar y guardar
session_name, models_dict, results_dict = train_and_save_models(X, y)

print(f"Sesión guardada: {session_name}")
print(f"Modelos: {list(models_dict.keys())}")
```

### `load_trained_models(session_name=None)`

Carga modelos guardados:

```python
from model_persistence import load_trained_models

# Cargar sesión más reciente
models_dict, results_dict, session_info = load_trained_models()

# Cargar sesión específica
models_dict, results_dict, session_info = load_trained_models("session_20231201_143022")
```

## 🎯 Ejemplos de Uso

### Ejemplo 1: Cargar Mejor Modelo

```python
from quick_access_models import get_best_model

# Obtener el mejor modelo según F1-Score
best_model, best_results = get_best_model()

print(f"Mejor modelo: F1-Score = {best_results['F1-Score']:.3f}")

# Usar el modelo para predicciones
predictions = best_model.predict(X_new)
```

### Ejemplo 2: Cargar Modelo Específico

```python
from quick_access_models import get_model_by_name

# Cargar Random Forest específicamente
rf_model, rf_results, rf_metadata = get_model_by_name("RandomForest")

print(f"Random Forest F1-Score: {rf_results['F1-Score']:.3f}")
```

### Ejemplo 3: Comparar Todos los Modelos

```python
from quick_access_models import compare_models

# Comparar rendimiento de todos los modelos
comparison = compare_models()

# Resultado: Tabla ordenada por F1-Score
```

### Ejemplo 4: Predicciones con Modelo Guardado

```python
from quick_access_models import predict_with_saved_model

# Realizar predicciones con modelo guardado
predictions, probabilities = predict_with_saved_model("RandomForest", X_new)

print(f"Predicciones: {predictions.shape}")
print(f"Probabilidades: {probabilities.shape}")
```

## 🔧 Integración con Análisis Adicionales

### Para Selección de Características

```python
# Cargar modelos para análisis de características
from quick_access_models import main
models_dict, results_dict, session_info = main()

# Usar modelos para selección secuencial
rf_model = models_dict['RandomForest']
svm_model = models_dict['SVM']

# Continuar con análisis de características...
```

### Para Análisis PCA

```python
# Cargar modelos para análisis PCA
from quick_access_models import get_best_model
best_model, best_results = get_best_model()

# Usar el mejor modelo para análisis PCA
# El modelo ya incluye preprocesamiento y SMOTE
```

## 📈 Gestión de Sesiones

### Listar Sesiones Disponibles

```python
from model_persistence import ModelPersistenceManager

persistence_manager = ModelPersistenceManager()
sessions = persistence_manager.list_available_sessions()

print("Sesiones disponibles:")
for session in sessions:
    print(f"  - {session}")
```

### Información de Modelo Específico

```python
# Obtener información detallada de un modelo
model_info = persistence_manager.get_model_info("RandomForest")

print("Archivos disponibles:")
print(f"  Modelos: {model_info['model_files']}")
print(f"  Resultados: {model_info['result_files']}")
print(f"  Metadatos: {model_info['metadata_files']}")
```

## 🛠️ Configuración Avanzada

### Personalizar Directorio de Guardado

```python
# Cambiar directorio de guardado
persistence_manager = ModelPersistenceManager(base_path="mi_directorio_modelos")
```

### Guardar Sesión con Nombre Personalizado

```python
# Guardar con nombre específico
session_name = "experimento_caracteristicas_v1"
session_name, models, results = train_and_save_models(X, y, session_name)
```

## 🔍 Estructura de Datos Guardados

### Modelos (.pkl)
```python
# Contiene pipeline completo:
# - Preprocesamiento (StandardScaler + OneHotEncoder)
# - SMOTE (balanceo de clases)
# - Clasificador entrenado
```

### Resultados (.json)
```python
{
  "F1-Score": 0.823,
  "AUC-ROC": 0.891,
  "Accuracy": 0.856,
  "Precision": 0.789,
  "Recall": 0.861,
  "F1-CI": 0.045,
  "AUC-CI": 0.032
}
```

### Metadatos (.json)
```python
{
  "best_params": {"n_estimators": 200, "max_depth": 30},
  "best_score": 0.823,
  "cv_results": {...},
  "dataset_shape": [10000, 35],
  "training_timestamp": "2023-12-01T14:30:22"
}
```

## ⚠️ Consideraciones Importantes

### Compatibilidad de Versiones
- Los modelos se guardan con pickle, asegúrate de usar la misma versión de scikit-learn
- Si actualizas librerías, reentrena los modelos

### Espacio en Disco
- Cada modelo puede ocupar varios MB
- Los archivos se guardan en `saved_models/`
- Puedes eliminar sesiones antiguas manualmente

### Reproducibilidad
- Todos los modelos usan `random_state=42`
- Los resultados son reproducibles
- Los metadatos incluyen timestamp de entrenamiento

## 🚀 Flujo de Trabajo Recomendado

1. **Primera ejecución**: `python proyecto_final_with_persistence.py`
2. **Análisis adicionales**: `python quick_access_models.py`
3. **Experimentación**: Usar funciones de carga rápida
4. **Nuevo entrenamiento**: Solo cuando cambies datos o hiperparámetros

## 💡 Tips y Trucos

- **Acceso rápido**: Usa `get_best_model()` para obtener el mejor modelo automáticamente
- **Comparación**: Usa `compare_models()` para ver todos los resultados
- **Debugging**: Los metadatos incluyen información detallada del entrenamiento
- **Backup**: Copia la carpeta `saved_models/` para respaldo

## 🔧 Solución de Problemas

### Error: "No se encontraron modelos guardados"
```bash
# Ejecutar entrenamiento inicial
python proyecto_final_with_persistence.py
```

### Error: "Error cargando modelo"
```bash
# Verificar archivos
ls saved_models/models/
ls saved_models/results/

# Reentrenar si es necesario
python proyecto_final_with_persistence.py
```

### Error: "Versión incompatible"
```bash
# Reentrenar con versión actual
python proyecto_final_with_persistence.py
```

---

**¡Con este sistema, nunca más tendrás que esperar horas para reentrenar modelos!** 🎉 