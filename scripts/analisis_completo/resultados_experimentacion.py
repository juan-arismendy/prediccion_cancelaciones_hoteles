#!/usr/bin/env python3
"""
Script para generar tablas y gráficas de resultados de experimentación
para el proyecto de predicción de cancelaciones de hoteles.
Incluye análisis de hiperparámetros, intervalos de confianza y métricas de desempeño.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
    """
    Crea una tabla de resultados de desempeño para un modelo específico
    """
    results_df = pd.DataFrame(grid_search_results.cv_results_)
    
    # Extraer hiperparámetros
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    
    # Crear tabla resumida
    summary_cols = param_cols + ['mean_test_score', 'std_test_score', 'rank_test_score']
    summary_df = results_df[summary_cols].copy()
    
    # Renombrar columnas para mejor legibilidad
    rename_dict = {}
    rename_dict['mean_test_score'] = 'F1-Score (Validación)'
    rename_dict['std_test_score'] = 'Desv. Estándar'
    rename_dict['rank_test_score'] = 'Ranking'
    
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

def plot_hyperparameter_effect(grid_search_results, param_name, model_name, metric='mean_test_score'):
    """
    Crea una gráfica mostrando el efecto de un hiperparámetro específico
    """
    results_df = pd.DataFrame(grid_search_results.cv_results_)
    
    # Extraer valores del hiperparámetro
    param_col = f'param_classifier__{param_name}'
    if param_col not in results_df.columns:
        print(f"Hiperparámetro {param_col} no encontrado en los resultados")
        return
    
    # Agrupar por valor del hiperparámetro
    grouped = results_df.groupby(param_col)[metric].agg(['mean', 'std']).reset_index()
    
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
    plt.xticks(x_pos, list(grouped[param_col]), rotation=45 if len(grouped) > 5 else 0)
    
    # Agregar grid
    plt.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for i, (mean_val, std_val) in enumerate(zip(grouped['mean'], grouped['std'])):
        plt.text(i, mean_val + std_val + 0.01, f'{mean_val:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def create_model_comparison_table(all_results):
    """
    Crea una tabla comparativa de todos los modelos
    """
    comparison_data = []
    
    for model_name, results in all_results.items():
        comparison_data.append({
            'Modelo': model_name,
            'F1-Score': f"{results['F1-Score']:.3f} ± {results['F1-CI']:.3f}",
            'AUC-ROC': f"{results['AUC-ROC']:.3f} ± {results['AUC-CI']:.3f}",
            'Mejor Hiperparámetro': get_best_hyperparameter(model_name)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def get_best_hyperparameter(model_name):
    """
    Retorna el mejor hiperparámetro para cada modelo
    """
    best_params = {
        'Logistic Regression': 'C = 1',
        'KNN': 'n_neighbors = 11, weights = distance, p = 1',
        'Random Forest': 'n_estimators = 100, max_depth = 30, max_features = 0.5',
        'MLP': 'hidden_layer_sizes = (50,), activation = relu, alpha = 0.01',
        'SVM': 'C = 10, kernel = rbf, gamma = scale'
    }
    return best_params.get(model_name, 'N/A')

def plot_model_comparison(all_results):
    """
    Crea una gráfica comparativa de todos los modelos
    """
    models = list(all_results.keys())
    f1_scores = [all_results[model]['F1-Score'] for model in models]
    f1_cis = [all_results[model]['F1-CI'] for model in models]
    auc_scores = [all_results[model]['AUC-ROC'] for model in models]
    auc_cis = [all_results[model]['AUC-CI'] for model in models]
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfica F1-Score
    x_pos = range(len(models))
    bars1 = ax1.errorbar(x_pos, f1_scores, yerr=f1_cis, fmt='o-', capsize=5, 
                        capthick=2, linewidth=2, markersize=8, color='skyblue')
    ax1.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de F1-Score entre Modelos', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.6, 0.9)
    
    # Agregar valores en las barras
    for i, (score, ci) in enumerate(zip(f1_scores, f1_cis)):
        ax1.text(i, score + ci + 0.01, f'{score:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Gráfica AUC-ROC
    bars2 = ax2.errorbar(x_pos, auc_scores, yerr=auc_cis, fmt='s-', capsize=5, 
                        capthick=2, linewidth=2, markersize=8, color='lightcoral')
    ax2.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax2.set_title('Comparación de AUC-ROC entre Modelos', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.8, 1.0)
    
    # Agregar valores en las barras
    for i, (score, ci) in enumerate(zip(auc_scores, auc_cis)):
        ax2.text(i, score + ci + 0.005, f'{score:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """
    Crea una matriz de confusión para un modelo
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Cancelado', 'Cancelado'],
                yticklabels=['No Cancelado', 'Cancelado'])
    plt.xlabel('Predicción', fontsize=12, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12, fontweight='bold')
    plt.title(f'Matriz de Confusión - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def evaluate_model_train_val_test(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    """
    Evalúa un modelo en conjuntos de entrenamiento, validación y test
    """
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Predecir en cada conjunto
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    
    y_pred_val = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val)[:, 1]
    
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas para cada conjunto
    results = {}
    
    for dataset_name, y_true, y_pred, y_proba in [
        ('Train', y_train, y_pred_train, y_proba_train),
        ('Validation', y_val, y_pred_val, y_proba_val),
        ('Test', y_test, y_pred_test, y_proba_test)
    ]:
        results[dataset_name] = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'AUC-ROC': roc_auc_score(y_true, y_proba)
        }
    
    return results

def create_train_val_test_comparison_table(all_models_results):
    """
    Crea una tabla comparativa de resultados Train/Validation/Test para todos los modelos
    """
    comparison_data = []
    
    for model_name, results in all_models_results.items():
        for dataset in ['Train', 'Validation', 'Test']:
            comparison_data.append({
                'Modelo': model_name,
                'Conjunto': dataset,
                'Accuracy': f"{results[dataset]['Accuracy']:.3f}",
                'Precision': f"{results[dataset]['Precision']:.3f}",
                'Recall': f"{results[dataset]['Recall']:.3f}",
                'F1-Score': f"{results[dataset]['F1-Score']:.3f}",
                'AUC-ROC': f"{results[dataset]['AUC-ROC']:.3f}"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def plot_train_val_test_comparison(all_models_results, metric='F1-Score'):
    """
    Crea una gráfica comparativa de métricas Train/Validation/Test
    """
    models = list(all_models_results.keys())
    datasets = ['Train', 'Validation', 'Test']
    
    # Preparar datos
    data = []
    for model in models:
        for dataset in datasets:
            data.append({
                'Modelo': model,
                'Conjunto': dataset,
                metric: all_models_results[model][dataset][metric]
            })
    
    df = pd.DataFrame(data)
    
    # Crear gráfica
    plt.figure(figsize=(14, 8))
    
    # Gráfica de barras agrupadas
    x = np.arange(len(models))
    width = 0.25
    
    for i, dataset in enumerate(datasets):
        values = [df[(df['Modelo'] == model) & (df['Conjunto'] == dataset)][metric].iloc[0] 
                 for model in models]
        plt.bar(x + i*width, values, width, label=dataset, alpha=0.8)
    
    plt.xlabel('Modelos', fontsize=12, fontweight='bold')
    plt.ylabel(metric, fontsize=12, fontweight='bold')
    plt.title(f'Comparación de {metric} - Train vs Validation vs Test', 
              fontsize=14, fontweight='bold')
    plt.xticks(x + width, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def generate_train_val_test_analysis(X, y, preprocessor):
    """
    Genera análisis completo Train/Validation/Test para todos los modelos
    """
    print("=== ANÁLISIS TRAIN/VALIDATION/TEST ===")
    
    # Dividir datos: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print(f"Tamaños de conjuntos:")
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Validation: {X_val.shape[0]} muestras")
    print(f"  Test: {X_test.shape[0]} muestras")
    
    # Definir modelos con mejores hiperparámetros
    models = {
        'Logistic Regression': ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(random_state=42, solver='liblinear', penalty='l1', C=1))
        ]),
        'KNN': ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', KNeighborsClassifier(n_neighbors=11, weights='distance', p=1))
        ]),
        'Random Forest': ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=30, max_features=0.5))
        ]),
        'SVM': ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', SVC(random_state=42, probability=True, C=10, kernel='rbf', gamma='scale'))
        ])
    }
    
    # Evaluar cada modelo
    all_results = {}
    for model_name, model in models.items():
        print(f"\n--- Evaluando {model_name} ---")
        results = evaluate_model_train_val_test(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name)
        all_results[model_name] = results
        
        # Mostrar resultados
        for dataset in ['Train', 'Validation', 'Test']:
            print(f"  {dataset}: F1={results[dataset]['F1-Score']:.3f}, AUC={results[dataset]['AUC-ROC']:.3f}")
    
    # Crear tabla comparativa
    print("\n📊 Tabla: Comparación Train/Validation/Test")
    print("=" * 80)
    comparison_table = create_train_val_test_comparison_table(all_results)
    print(comparison_table.to_string(index=False))
    
    # Crear gráficas
    print("\n📈 Gráficas: Comparación Train/Validation/Test")
    print("=" * 80)
    plot_train_val_test_comparison(all_results, 'F1-Score')
    plot_train_val_test_comparison(all_results, 'AUC-ROC')
    
    return all_results

def generate_all_results_analysis():
    """
    Función principal que genera todo el análisis de resultados
    """
    print("=== ANÁLISIS COMPLETO DE RESULTADOS DE EXPERIMENTACIÓN ===\n")
    
    # Simular resultados basados en tu notebook (ajusta estos valores según tus resultados reales)
    all_results = {
        'Logistic Regression': {
            'F1-Score': 0.768,
            'AUC-ROC': 0.900,
            'F1-CI': 0.023,
            'AUC-CI': 0.017,
            'grid_search': None  # Aquí irían los resultados reales de GridSearchCV
        },
        'KNN': {
            'F1-Score': 0.744,
            'AUC-ROC': 0.880,
            'F1-CI': 0.033,
            'AUC-CI': 0.015,
            'grid_search': None
        },
        'Random Forest': {
            'F1-Score': 0.814,
            'AUC-ROC': 0.933,
            'F1-CI': 0.014,
            'AUC-CI': 0.014,
            'grid_search': None
        },
        'MLP': {
            'F1-Score': 0.733,
            'AUC-ROC': 0.892,
            'F1-CI': 0.023,
            'AUC-CI': 0.021,
            'grid_search': None
        },
        'SVM': {
            'F1-Score': 0.797,
            'AUC-ROC': 0.923,
            'F1-CI': 0.016,
            'AUC-CI': 0.013,
            'grid_search': None
        }
    }
    
    # 1. Tabla comparativa de todos los modelos
    print("1. TABLA COMPARATIVA DE TODOS LOS MODELOS")
    print("=" * 50)
    comparison_table = create_model_comparison_table(all_results)
    print(comparison_table.to_string(index=False))
    print("\n")
    
    # 2. Gráfica comparativa
    print("2. GRÁFICA COMPARATIVA DE MODELOS")
    print("=" * 50)
    plot_model_comparison(all_results)
    
    # 3. Análisis individual por modelo
    print("3. ANÁLISIS INDIVIDUAL POR MODELO")
    print("=" * 50)
    
    # Para cada modelo, mostrar análisis detallado
    for model_name in all_results.keys():
        print(f"\n--- {model_name.upper()} ---")
        print(f"F1-Score: {all_results[model_name]['F1-Score']:.3f} ± {all_results[model_name]['F1-CI']:.3f}")
        print(f"AUC-ROC: {all_results[model_name]['AUC-ROC']:.3f} ± {all_results[model_name]['AUC-CI']:.3f}")
        print(f"Mejor hiperparámetro: {get_best_hyperparameter(model_name)}")
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("\nPara usar con tus resultados reales:")
    print("1. Reemplaza los valores en 'all_results' con tus resultados reales")
    print("2. Asigna los objetos GridSearchCV reales a 'grid_search'")
    print("3. Ejecuta las funciones de gráficas individuales para cada modelo")

if __name__ == "__main__":
    generate_all_results_analysis() 