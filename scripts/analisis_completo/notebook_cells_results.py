# ============================================================================
# CELDA 1: CONFIGURACIÓN Y FUNCIONES AUXILIARES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
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

# ============================================================================
# CELDA 2: ANÁLISIS DE REGRESIÓN LOGÍSTICA
# ============================================================================

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

# ============================================================================
# CELDA 3: ANÁLISIS DE K-NEAREST NEIGHBORS
# ============================================================================

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

# ============================================================================
# CELDA 4: ANÁLISIS DE RANDOM FOREST
# ============================================================================

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

# ============================================================================
# CELDA 5: ANÁLISIS DE MLP (REDES NEURONALES)
# ============================================================================

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

# ============================================================================
# CELDA 6: ANÁLISIS DE SVM
# ============================================================================

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

# ============================================================================
# CELDA 7: COMPARACIÓN GENERAL DE TODOS LOS MODELOS
# ============================================================================

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

# ============================================================================
# CELDA 8: ANÁLISIS DE RESULTADOS EN TEST (si tienes un conjunto de test)
# ============================================================================

print("=== ANÁLISIS DE RESULTADOS EN CONJUNTO DE TEST ===")

# Si quieres evaluar en un conjunto de test separado, puedes usar este código:
# Primero, divide los datos en train+val y test
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

# Random Forest (ejemplo con el mejor modelo)
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