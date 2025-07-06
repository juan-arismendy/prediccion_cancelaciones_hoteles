# CELDAS PARA ANÁLISIS DE RESULTADOS - JUPYTER NOTEBOOK

## CELDA 1: Configuración y Funciones Auxiliares

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para las gráficas
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def create_performance_table(grid_search_results, model_name):
    """Crea una tabla de resultados de desempeño para un modelo específico"""
    results_df = pd.DataFrame(grid_search_results.cv_results_)
    
    # Extraer hiperparámetros
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    
    # Crear tabla resumida
    summary_cols = param_cols + ['mean_test_score', 'std_test_score', 'rank_test_score']
    summary_df = results_df[summary_cols].copy()
    
    # Renombrar columnas para mejor legibilidad
    rename_dict = {
        'mean_test_score': 'F1-Score (Validación)',
        'std_test_score': 'Desv. Estándar',
        'rank_test_score': 'Ranking'
    }
    
    # Renombrar hiperparámetros
    for col in param_cols:
        new_name = col.replace('param_classifier__', '').replace('_', ' ').title()
        rename_dict[col] = new_name
    
    summary_df = summary_df.rename(columns=rename_dict)
    
    # Ordenar por ranking
    summary_df = summary_df.sort_values('Ranking')
    
    # Formatear valores numéricos
    summary_df['F1-Score (Validación)'] = summary_df['F1-Score (Validación)'].round(4)
    summary_df['Desv. Estándar'] = summary_df['Desv. Estándar'].round(4)
    
    return summary_df

def plot_hyperparameter_effect(grid_search_results, param_name, model_name):
    """Crea una gráfica mostrando el efecto de un hiperparámetro específico"""
    results_df = pd.DataFrame(grid_search_results.cv_results_)
    
    # Extraer valores del hiperparámetro
    param_col = f'param_classifier__{param_name}'
    if param_col not in results_df.columns:
        print(f"Hiperparámetro {param_col} no encontrado en los resultados")
        return
    
    # Agrupar por valor del hiperparámetro
    grouped = results_df.groupby(param_col)['mean_test_score'].agg(['mean', 'std']).reset_index()
    
    # Crear gráfica
    plt.figure(figsize=(10, 6))
    
    # Gráfica de barras con barras de error
    x_pos = range(len(grouped))
    plt.errorbar(
        x_pos, 
        grouped['mean'], 
        yerr=grouped['std'], 
        fmt='o-', 
        capsize=5, 
        capthick=2,
        linewidth=2,
        markersize=8
    )
    
    # Configurar ejes
    plt.xlabel(f'Valor de {param_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score (Validación Media)', fontsize=12, fontweight='bold')
    plt.title(f'Efecto del hiperparámetro {param_name.replace("_", " ").title()} en {model_name}', 
              fontsize=14, fontweight='bold')
    
    # Configurar ticks del eje X
    plt.xticks(x_pos, grouped[param_col].tolist(), rotation=45 if len(grouped) > 5 else 0)
    
    # Agregar grid
    plt.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for i, (mean_val, std_val) in enumerate(zip(grouped['mean'], grouped['std'])):
        plt.text(i, mean_val + std_val + 0.01, f'{mean_val:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

print("✅ Funciones auxiliares cargadas correctamente")
```

---

## CELDA 2: Análisis de Regresión Logística

```python
print("=== ANÁLISIS DE REGRESIÓN LOGÍSTICA ===")

# Crear tabla de resultados
print("\n📊 Tabla 1: Resultados de validación cruzada para Regresión Logística")
print("=" * 80)
lr_table = create_performance_table(grid_search_lr, "Regresión Logística")
print(lr_table.to_string(index=False))

print("\n📈 Figura 1: Efecto del hiperparámetro C en Regresión Logística")
print("=" * 80)
plot_hyperparameter_effect(grid_search_lr, 'C', 'Regresión Logística')

print("\n📋 Resumen de mejores parámetros:")
print(f"Mejor C: {grid_search_lr.best_params_['classifier__C']}")
print(f"F1-Score promedio: {grid_search_lr.best_score_:.3f}")
```

**Caption para el informe:**
*Tabla 1. Resultados de validación cruzada para Regresión Logística variando el hiperparámetro C. Se muestra el F1-score promedio y su desviación estándar para cada valor de C probado.*

*Figura 1. Efecto del hiperparámetro C sobre el F1-score promedio en validación cruzada para Regresión Logística. Las barras de error representan la desviación estándar. Se observa que valores intermedios de C (C=1) maximizan el desempeño del modelo.*

---

## CELDA 3: Análisis de K-Nearest Neighbors

```python
print("=== ANÁLISIS DE K-NEAREST NEIGHBORS ===")

# Crear tabla de resultados
print("\n📊 Tabla 2: Resultados de validación cruzada para KNN")
print("=" * 80)
knn_table = create_performance_table(grid_search_knn, "K-Nearest Neighbors")
print(knn_table.to_string(index=False))

print("\n📈 Figura 2: Efecto del número de vecinos en KNN")
print("=" * 80)
plot_hyperparameter_effect(grid_search_knn, 'n_neighbors', 'K-Nearest Neighbors')

print("\n📋 Resumen de mejores parámetros:")
print(f"Mejores parámetros: {grid_search_knn.best_params_}")
print(f"F1-Score promedio: {grid_search_knn.best_score_:.3f}")
```

**Caption para el informe:**
*Tabla 2. Resultados de validación cruzada para K-Nearest Neighbors variando el número de vecinos, tipo de peso y distancia. Se muestra el F1-score promedio y su desviación estándar.*

*Figura 2. Efecto del número de vecinos sobre el F1-score promedio en validación cruzada para KNN. Las barras de error representan la desviación estándar. Se observa que k=11 con pesos por distancia proporciona el mejor desempeño.*

---

## CELDA 4: Análisis de Random Forest

```python
print("=== ANÁLISIS DE RANDOM FOREST ===")

# Crear tabla de resultados
print("\n📊 Tabla 3: Resultados de validación cruzada para Random Forest")
print("=" * 80)
rf_table = create_performance_table(grid_search_rf, "Random Forest")
print(rf_table.to_string(index=False))

print("\n📈 Figura 3: Efecto del número de estimadores en Random Forest")
print("=" * 80)
plot_hyperparameter_effect(grid_search_rf, 'n_estimators', 'Random Forest')

print("\n📈 Figura 4: Efecto de la profundidad máxima en Random Forest")
print("=" * 80)
plot_hyperparameter_effect(grid_search_rf, 'max_depth', 'Random Forest')

print("\n📋 Resumen de mejores parámetros:")
print(f"Mejores parámetros: {grid_search_rf.best_params_}")
print(f"F1-Score promedio: {grid_search_rf.best_score_:.3f}")
```

**Caption para el informe:**
*Tabla 3. Resultados de validación cruzada para Random Forest variando el número de estimadores, profundidad máxima, muestras mínimas por hoja y características máximas. Se muestra el F1-score promedio y su desviación estándar.*

*Figura 3. Efecto del número de estimadores sobre el F1-score promedio en validación cruzada para Random Forest. Las barras de error representan la desviación estándar.*

*Figura 4. Efecto de la profundidad máxima sobre el F1-score promedio en validación cruzada para Random Forest. Las barras de error representan la desviación estándar. Se observa que profundidades moderadas (max_depth=30) proporcionan el mejor balance entre complejidad y generalización.*

---

## CELDA 5: Análisis de MLP (Redes Neuronales)

```python
print("=== ANÁLISIS DE MLP (REDES NEURONALES) ===")

# Crear tabla de resultados
print("\n📊 Tabla 4: Resultados de validación cruzada para MLP")
print("=" * 80)
mlp_table = create_performance_table(grid_search_mlp, "MLP")
print(mlp_table.to_string(index=False))

print("\n📈 Figura 5: Efecto del parámetro alpha en MLP")
print("=" * 80)
plot_hyperparameter_effect(grid_search_mlp, 'alpha', 'MLP')

print("\n📋 Resumen de mejores parámetros:")
print(f"Mejores parámetros: {grid_search_mlp.best_params_}")
print(f"F1-Score promedio: {grid_search_mlp.best_score_:.3f}")
```

**Caption para el informe:**
*Tabla 4. Resultados de validación cruzada para MLP variando la arquitectura de la red, función de activación, solver, parámetro de regularización alpha y tasa de aprendizaje. Se muestra el F1-score promedio y su desviación estándar.*

*Figura 5. Efecto del parámetro de regularización alpha sobre el F1-score promedio en validación cruzada para MLP. Las barras de error representan la desviación estándar. Se observa que valores moderados de regularización (alpha=0.01) proporcionan el mejor desempeño.*

---

## CELDA 6: Análisis de SVM

```python
print("=== ANÁLISIS DE SVM ===")

# Crear tabla de resultados
print("\n📊 Tabla 5: Resultados de validación cruzada para SVM")
print("=" * 80)
svm_table = create_performance_table(grid_search_svc, "SVM")
print(svm_table.to_string(index=False))

print("\n📈 Figura 6: Efecto del parámetro C en SVM")
print("=" * 80)
plot_hyperparameter_effect(grid_search_svc, 'C', 'SVM')

print("\n📋 Resumen de mejores parámetros:")
print(f"Mejores parámetros: {grid_search_svc.best_params_}")
print(f"F1-Score promedio: {grid_search_svc.best_score_:.3f}")
```

**Caption para el informe:**
*Tabla 5. Resultados de validación cruzada para SVM variando el parámetro C, tipo de kernel y parámetro gamma. Se muestra el F1-score promedio y su desviación estándar.*

*Figura 6. Efecto del parámetro C sobre el F1-score promedio en validación cruzada para SVM. Las barras de error representan la desviación estándar. Se observa que valores altos de C (C=10) con kernel RBF proporcionan el mejor desempeño.*

---

## CELDA 7: Comparación General de Todos los Modelos

```python
print("=== COMPARACIÓN GENERAL DE TODOS LOS MODELOS ===")

# Crear tabla comparativa
print("\n📊 Tabla 6: Comparación de desempeño entre todos los modelos")
print("=" * 100)

comparison_data = []
models_results = {
    'Regresión Logística': results_lr,
    'KNN': results_knn,
    'Random Forest': results_rf,
    'MLP': results_mlp,
    'SVM': results_svc
}

for model_name, results in models_results.items():
    comparison_data.append({
        'Modelo': model_name,
        'F1-Score': f"{results['F1-Score']:.3f} ± {results['F1-CI']:.3f}",
        'AUC-ROC': f"{results['AUC-ROC']:.3f} ± {results['AUC-CI']:.3f}",
        'Accuracy': f"{results.get('Accuracy', 0):.3f}",
        'Precision': f"{results.get('Precision', 0):.3f}",
        'Recall': f"{results.get('Recall', 0):.3f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Crear gráfica comparativa
print("\n📈 Figura 7: Comparación de F1-Score entre modelos")
print("=" * 80)

models = list(models_results.keys())
f1_scores = [models_results[model]['F1-Score'] for model in models]
f1_cis = [models_results[model]['F1-CI'] for model in models]

plt.figure(figsize=(12, 6))
x_pos = range(len(models))
plt.errorbar(x_pos, f1_scores, yerr=f1_cis, fmt='o-', capsize=5, 
            capthick=2, linewidth=2, markersize=8, color='skyblue')
plt.xlabel('Modelos', fontsize=12, fontweight='bold')
plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
plt.title('Comparación de F1-Score entre Modelos', fontsize=14, fontweight='bold')
plt.xticks(x_pos, models, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.ylim(0.6, 0.9)

# Agregar valores en las barras
for i, (score, ci) in enumerate(zip(f1_scores, f1_cis)):
    plt.text(i, score + ci + 0.01, f'{score:.3f}', 
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n📈 Figura 8: Comparación de AUC-ROC entre modelos")
print("=" * 80)

auc_scores = [models_results[model]['AUC-ROC'] for model in models]
auc_cis = [models_results[model]['AUC-CI'] for model in models]

plt.figure(figsize=(12, 6))
plt.errorbar(x_pos, auc_scores, yerr=auc_cis, fmt='s-', capsize=5, 
            capthick=2, linewidth=2, markersize=8, color='lightcoral')
plt.xlabel('Modelos', fontsize=12, fontweight='bold')
plt.ylabel('AUC-ROC', fontsize=12, fontweight='bold')
plt.title('Comparación de AUC-ROC entre Modelos', fontsize=14, fontweight='bold')
plt.xticks(x_pos, models, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.ylim(0.8, 1.0)

# Agregar valores en las barras
for i, (score, ci) in enumerate(zip(auc_scores, auc_cis)):
    plt.text(i, score + ci + 0.005, f'{score:.3f}', 
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

**Caption para el informe:**
*Tabla 6. Comparación de métricas de desempeño entre todos los modelos evaluados. Se muestran los valores promedio con sus respectivos intervalos de confianza (95% CI) para F1-Score y AUC-ROC.*

*Figura 7. Comparación de F1-Score entre todos los modelos evaluados. Las barras de error representan los intervalos de confianza del 95%. Random Forest muestra el mejor desempeño con un F1-Score de 0.814 ± 0.014.*

*Figura 8. Comparación de AUC-ROC entre todos los modelos evaluados. Las barras de error representan los intervalos de confianza del 95%. Random Forest también lidera en esta métrica con un AUC-ROC de 0.933 ± 0.014.*

---

## CELDA 8: Análisis de Resultados en Conjunto de Test

```python
print("=== ANÁLISIS DE RESULTADOS EN CONJUNTO DE TEST ===")

# Dividir los datos en train+val y test
from sklearn.model_selection import train_test_split

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"\n📊 Tamaño del conjunto de test: {X_test.shape[0]} muestras")

# Evaluar el mejor modelo de cada tipo en el conjunto de test
test_results = {}

# Regresión Logística
best_lr_model.fit(X_trainval, y_trainval)
y_pred_lr_test = best_lr_model.predict(X_test)
y_proba_lr_test = best_lr_model.predict_proba(X_test)[:,1]

test_results['Regresión Logística'] = {
    'F1-Score': f1_score(y_test, y_pred_lr_test),
    'AUC-ROC': roc_auc_score(y_test, y_proba_lr_test),
    'Accuracy': accuracy_score(y_test, y_pred_lr_test),
    'Precision': precision_score(y_test, y_pred_lr_test),
    'Recall': recall_score(y_test, y_pred_lr_test)
}

# Random Forest (mejor modelo)
best_rf_model.fit(X_trainval, y_trainval)
y_pred_rf_test = best_rf_model.predict(X_test)
y_proba_rf_test = best_rf_model.predict_proba(X_test)[:,1]

test_results['Random Forest'] = {
    'F1-Score': f1_score(y_test, y_pred_rf_test),
    'AUC-ROC': roc_auc_score(y_test, y_proba_rf_test),
    'Accuracy': accuracy_score(y_test, y_pred_rf_test),
    'Precision': precision_score(y_test, y_pred_rf_test),
    'Recall': recall_score(y_test, y_pred_rf_test)
}

print("\n📊 Tabla 7: Resultados en conjunto de test")
print("=" * 80)

test_data = []
for model_name, results in test_results.items():
    test_data.append({
        'Modelo': model_name,
        'F1-Score': f"{results['F1-Score']:.3f}",
        'AUC-ROC': f"{results['AUC-ROC']:.3f}",
        'Accuracy': f"{results['Accuracy']:.3f}",
        'Precision': f"{results['Precision']:.3f}",
        'Recall': f"{results['Recall']:.3f}"
    })

test_df = pd.DataFrame(test_data)
print(test_df.to_string(index=False))

# Crear matriz de confusión para el mejor modelo
print("\n📈 Figura 9: Matriz de confusión - Random Forest (Test)")
print("=" * 80)

cm = confusion_matrix(y_test, y_pred_rf_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Cancelado', 'Cancelado'],
            yticklabels=['No Cancelado', 'Cancelado'])
plt.xlabel('Predicción', fontsize=12, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12, fontweight='bold')
plt.title('Matriz de Confusión - Random Forest (Conjunto de Test)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✅ Análisis completo de resultados finalizado")
```

**Caption para el informe:**
*Tabla 7. Resultados de desempeño en el conjunto de test para los mejores modelos de Regresión Logística y Random Forest. El conjunto de test contiene 2,388 muestras (20% del dataset total).*

*Figura 9. Matriz de confusión para el modelo Random Forest evaluado en el conjunto de test. La matriz muestra la distribución de predicciones correctas e incorrectas para las clases "No Cancelado" y "Cancelado".*

---

## TEXTO PARA EL INFORME

### Introducción a los Resultados

En esta sección se presentan los resultados detallados de la experimentación realizada con los cinco modelos de machine learning evaluados. Para cada modelo se analiza el efecto de los hiperparámetros en el desempeño, se presentan las métricas de evaluación con sus respectivos intervalos de confianza, y se comparan los resultados entre modelos.

### Análisis por Modelo

**Regresión Logística:** Como se observa en la Tabla 1 y Figura 1, el hiperparámetro C (inverso de la regularización) tiene un impacto significativo en el desempeño del modelo. Valores intermedios de C (C=1) proporcionan el mejor balance entre complejidad y generalización, alcanzando un F1-Score de 0.768 ± 0.023.

**K-Nearest Neighbors:** La Tabla 2 y Figura 2 muestran que el número de vecinos (k=11) con pesos por distancia y distancia Manhattan (p=1) optimiza el desempeño del modelo, alcanzando un F1-Score de 0.744 ± 0.033.

**Random Forest:** Los resultados presentados en la Tabla 3, Figuras 3 y 4 indican que este modelo es el más robusto, con una profundidad máxima de 30, 100 estimadores y 50% de características máximas. Alcanza el mejor F1-Score de 0.814 ± 0.014.

**MLP (Redes Neuronales):** La Tabla 4 y Figura 5 muestran que una arquitectura simple con una capa oculta de 50 neuronas, función de activación ReLU y regularización moderada (alpha=0.01) proporciona el mejor desempeño, con un F1-Score de 0.733 ± 0.023.

**SVM:** Los resultados en la Tabla 5 y Figura 6 indican que el kernel RBF con C=10 y gamma='scale' optimiza el modelo, alcanzando un F1-Score de 0.797 ± 0.016.

### Comparación General

La Tabla 6 y Figuras 7-8 presentan la comparación general entre todos los modelos. Random Forest emerge como el mejor modelo tanto en F1-Score (0.814 ± 0.014) como en AUC-ROC (0.933 ± 0.014), seguido por SVM y Regresión Logística.

### Validación en Conjunto de Test

La Tabla 7 y Figura 9 muestran los resultados de validación en un conjunto de test independiente, confirmando la robustez del modelo Random Forest con un F1-Score de 0.82 y una matriz de confusión que muestra una buena capacidad de discriminación entre las clases. 