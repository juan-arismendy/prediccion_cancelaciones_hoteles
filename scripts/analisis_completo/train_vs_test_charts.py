#!/usr/bin/env python3
"""
Script para generar gráficas específicas de Train vs Test
para detectar overfitting y evaluar generalización de modelos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_and_prepare_data():
    """Carga y prepara los datos"""
    print("=== CARGANDO DATOS ===")
    
    # Cargar datos desde la carpeta datos
    df = pd.read_csv('datos/hotel_booking.csv')
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Reducir dataset para velocidad (10% como en el notebook)
    df = df.sample(frac=0.1, random_state=42)
    print(f"Dataset reducido: {df.shape[0]} filas")
    
    # Limpieza básica
    df['children'] = df['children'].fillna(0)
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    df = df[df['adr'] >= 0]
    df = df[df['adults'] + df['children'] + df['babies'] > 0]
    df = df.drop(columns=['reservation_status', 'reservation_status_date'])
    
    # Separar features y target
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    
    # Crear preprocessor
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    print(f"Distribución del target: {y.value_counts(normalize=True).to_dict()}")
    return X, y, preprocessor

def evaluate_train_test_performance(model, X_train, y_train, X_test, y_test):
    """Evalúa el rendimiento en train y test"""
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Predecir en train y test
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    results = {
        'Train': {
            'Accuracy': accuracy_score(y_train, y_pred_train),
            'Precision': precision_score(y_train, y_pred_train),
            'Recall': recall_score(y_train, y_pred_train),
            'F1-Score': f1_score(y_train, y_pred_train),
            'AUC-ROC': roc_auc_score(y_train, y_proba_train)
        },
        'Test': {
            'Accuracy': accuracy_score(y_test, y_pred_test),
            'Precision': precision_score(y_test, y_pred_test),
            'Recall': recall_score(y_test, y_pred_test),
            'F1-Score': f1_score(y_test, y_pred_test),
            'AUC-ROC': roc_auc_score(y_test, y_proba_test)
        }
    }
    
    return results

def create_train_test_comparison_chart(all_results):
    """Crea gráfica comparativa de Train vs Test"""
    models = list(all_results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        train_scores = [all_results[model]['Train'][metric] for model in models]
        test_scores = [all_results[model]['Test'][metric] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, test_scores, width, label='Test', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Modelos', fontweight='bold')
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'Train vs Test - {metric}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Remover el subplot extra
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('visualizaciones/analisis_completo/train_vs_test_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_overfitting_analysis_chart(all_results):
    """Crea gráfica específica de análisis de overfitting"""
    models = list(all_results.keys())
    
    # Calcular diferencias Train - Test
    f1_differences = []
    auc_differences = []
    
    for model in models:
        f1_diff = all_results[model]['Train']['F1-Score'] - all_results[model]['Test']['F1-Score']
        auc_diff = all_results[model]['Train']['AUC-ROC'] - all_results[model]['Test']['AUC-ROC']
        f1_differences.append(f1_diff)
        auc_differences.append(auc_diff)
    
    # Crear gráfica
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfica de diferencias F1-Score
    colors = ['green' if diff < 0.05 else 'orange' if diff < 0.1 else 'red' for diff in f1_differences]
    bars1 = ax1.bar(models, f1_differences, color=colors, alpha=0.7)
    ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Umbral de Precaución (0.05)')
    ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Umbral de Overfitting (0.10)')
    ax1.set_xlabel('Modelos', fontweight='bold')
    ax1.set_ylabel('Diferencia F1-Score (Train - Test)', fontweight='bold')
    ax1.set_title('Análisis de Overfitting - F1-Score', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores
    for bar, diff in zip(bars1, f1_differences):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{diff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfica de diferencias AUC-ROC
    colors = ['green' if diff < 0.03 else 'orange' if diff < 0.06 else 'red' for diff in auc_differences]
    bars2 = ax2.bar(models, auc_differences, color=colors, alpha=0.7)
    ax2.axhline(y=0.03, color='orange', linestyle='--', alpha=0.7, label='Umbral de Precaución (0.03)')
    ax2.axhline(y=0.06, color='red', linestyle='--', alpha=0.7, label='Umbral de Overfitting (0.06)')
    ax2.set_xlabel('Modelos', fontweight='bold')
    ax2.set_ylabel('Diferencia AUC-ROC (Train - Test)', fontweight='bold')
    ax2.set_title('Análisis de Overfitting - AUC-ROC', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Agregar valores
    for bar, diff in zip(bars2, auc_differences):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{diff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizaciones/analisis_completo/overfitting_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_learning_curves(X, y, preprocessor):
    """Crea curvas de aprendizaje para detectar overfitting"""
    # Modelos a evaluar
    models = {
        'Random Forest': ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=30))
        ]),
        'SVM': ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', SVC(random_state=42, probability=True, C=10, kernel='rbf'))
        ])
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (model_name, model) in enumerate(models.items()):
        ax = axes[idx]
        
        # Generar curva de aprendizaje
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=3, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='f1', random_state=42
        )
        
        # Calcular medias y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Graficar
        ax.plot(train_sizes, train_mean, 'o-', label='Train Score', color='blue', linewidth=2)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', label='Validation Score', color='red', linewidth=2)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        ax.set_xlabel('Tamaño del Conjunto de Entrenamiento', fontweight='bold')
        ax.set_ylabel('F1-Score', fontweight='bold')
        ax.set_title(f'Curva de Aprendizaje - {model_name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizaciones/analisis_completo/learning_curves.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_train_test_analysis():
    """Función principal para generar análisis Train vs Test"""
    print("🔍 ANÁLISIS TRAIN vs TEST - DETECCIÓN DE OVERFITTING")
    print("=" * 60)
    
    # Cargar datos
    X, y, preprocessor = load_and_prepare_data()
    
    # Dividir en train y test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTamaños de conjuntos:")
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Test: {X_test.shape[0]} muestras")
    
    # Definir modelos con hiperparámetros optimizados
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
    
    # Evaluar modelos
    all_results = {}
    print("\n📊 Evaluando modelos...")
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        results = evaluate_train_test_performance(model, X_train, y_train, X_test, y_test)
        all_results[model_name] = results
        
        # Mostrar resultados
        train_f1 = results['Train']['F1-Score']
        test_f1 = results['Test']['F1-Score']
        diff_f1 = train_f1 - test_f1
        
        print(f"  Train F1: {train_f1:.3f}")
        print(f"  Test F1:  {test_f1:.3f}")
        print(f"  Diferencia: {diff_f1:.3f}")
        
        if diff_f1 > 0.1:
            print(f"  🔴 OVERFITTING DETECTADO")
        elif diff_f1 > 0.05:
            print(f"  🟡 Posible overfitting")
        else:
            print(f"  🟢 Modelo bien balanceado")
    
    # Crear tabla resumen
    print("\n📋 TABLA RESUMEN TRAIN vs TEST")
    print("=" * 80)
    
    summary_data = []
    for model_name, results in all_results.items():
        summary_data.append({
            'Modelo': model_name,
            'Train F1': f"{results['Train']['F1-Score']:.3f}",
            'Test F1': f"{results['Test']['F1-Score']:.3f}",
            'Diferencia F1': f"{results['Train']['F1-Score'] - results['Test']['F1-Score']:.3f}",
            'Train AUC': f"{results['Train']['AUC-ROC']:.3f}",
            'Test AUC': f"{results['Test']['AUC-ROC']:.3f}",
            'Diferencia AUC': f"{results['Train']['AUC-ROC'] - results['Test']['AUC-ROC']:.3f}",
            'Estado': 'Overfitting' if (results['Train']['F1-Score'] - results['Test']['F1-Score']) > 0.1 
                     else 'Precaución' if (results['Train']['F1-Score'] - results['Test']['F1-Score']) > 0.05 
                     else 'Balanceado'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Generar gráficas
    print("\n📈 Generando gráficas...")
    create_train_test_comparison_chart(all_results)
    create_overfitting_analysis_chart(all_results)
    create_learning_curves(X, y, preprocessor)
    
    print("\n✅ ANÁLISIS COMPLETADO")
    print("📁 Gráficas guardadas en visualizaciones/analisis_completo/")
    print("   - train_vs_test_comparison.png")
    print("   - overfitting_analysis.png")
    print("   - learning_curves.png")
    
    return all_results

if __name__ == "__main__":
    generate_train_test_analysis() 