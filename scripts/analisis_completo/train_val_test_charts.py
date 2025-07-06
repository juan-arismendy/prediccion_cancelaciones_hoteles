#!/usr/bin/env python3
"""
Script mejorado para generar gráficas de Train vs Validation vs Test
Incluye análisis de overfitting y generalización
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

# Configuración para evitar problemas de GUI
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI

# Configuración de estilo
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_and_prepare_data():
    """Carga y prepara los datos"""
    print("=== CARGANDO DATOS ===")
    
    # Cargar datos
    df = pd.read_csv('datos/hotel_booking.csv')
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Reducir dataset para velocidad
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
    
    return X, y, preprocessor

def evaluate_train_val_test(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evalúa un modelo en train, validation y test"""
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Predecir en cada conjunto
    sets = {
        'Train': (X_train, y_train),
        'Validation': (X_val, y_val),
        'Test': (X_test, y_test)
    }
    
    results = {}
    for set_name, (X_set, y_set) in sets.items():
        y_pred = model.predict(X_set)
        y_proba = model.predict_proba(X_set)[:, 1]
        
        results[set_name] = {
            'Accuracy': accuracy_score(y_set, y_pred),
            'Precision': precision_score(y_set, y_pred),
            'Recall': recall_score(y_set, y_pred),
            'F1-Score': f1_score(y_set, y_pred),
            'AUC-ROC': roc_auc_score(y_set, y_proba)
        }
    
    return results

def create_train_val_test_comparison(all_results):
    """Crea gráfica comparativa de Train vs Validation vs Test"""
    models = list(all_results.keys())
    metrics = ['F1-Score', 'AUC-ROC']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Preparar datos
        train_scores = [all_results[model]['Train'][metric] for model in models]
        val_scores = [all_results[model]['Validation'][metric] for model in models]
        test_scores = [all_results[model]['Test'][metric] for model in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        # Crear barras
        bars1 = ax.bar(x - width, train_scores, width, label='Train', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, val_scores, width, label='Validation', alpha=0.8, color='lightcoral')
        bars3 = ax.bar(x + width, test_scores, width, label='Test', alpha=0.8, color='lightgreen')
        
        # Configurar gráfica
        ax.set_xlabel('Modelos', fontweight='bold')
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'Comparación {metric} - Train vs Validation vs Test', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizaciones/analisis_completo/train_val_test_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("✅ Gráfica guardada: train_val_test_comparison.png")
    plt.close()

def create_overfitting_analysis(all_results):
    """Crea análisis específico de overfitting"""
    models = list(all_results.keys())
    
    # Calcular diferencias
    train_val_f1 = []
    val_test_f1 = []
    train_val_auc = []
    val_test_auc = []
    
    for model in models:
        # Diferencias F1-Score
        tv_f1 = all_results[model]['Train']['F1-Score'] - all_results[model]['Validation']['F1-Score']
        vt_f1 = all_results[model]['Validation']['F1-Score'] - all_results[model]['Test']['F1-Score']
        train_val_f1.append(tv_f1)
        val_test_f1.append(vt_f1)
        
        # Diferencias AUC-ROC
        tv_auc = all_results[model]['Train']['AUC-ROC'] - all_results[model]['Validation']['AUC-ROC']
        vt_auc = all_results[model]['Validation']['AUC-ROC'] - all_results[model]['Test']['AUC-ROC']
        train_val_auc.append(tv_auc)
        val_test_auc.append(vt_auc)
    
    # Crear gráfica
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # F1-Score: Train vs Validation (Overfitting)
    colors1 = ['green' if diff < 0.05 else 'orange' if diff < 0.1 else 'red' for diff in train_val_f1]
    bars1 = ax1.bar(models, train_val_f1, color=colors1, alpha=0.7)
    ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Precaución (0.05)')
    ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Overfitting (0.10)')
    ax1.set_title('Overfitting Analysis - F1 Score\n(Train - Validation)', fontweight='bold')
    ax1.set_ylabel('Diferencia F1-Score', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for bar, diff in zip(bars1, train_val_f1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{diff:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # F1-Score: Validation vs Test (Generalización)
    colors2 = ['green' if abs(diff) < 0.03 else 'orange' if abs(diff) < 0.06 else 'red' for diff in val_test_f1]
    bars2 = ax2.bar(models, val_test_f1, color=colors2, alpha=0.7)
    ax2.axhline(y=0.03, color='orange', linestyle='--', alpha=0.7, label='Precaución (±0.03)')
    ax2.axhline(y=-0.03, color='orange', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.06, color='red', linestyle='--', alpha=0.7, label='Problema (±0.06)')
    ax2.axhline(y=-0.06, color='red', linestyle='--', alpha=0.7)
    ax2.set_title('Generalization Analysis - F1 Score\n(Validation - Test)', fontweight='bold')
    ax2.set_ylabel('Diferencia F1-Score', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    for bar, diff in zip(bars2, val_test_f1):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{diff:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # AUC-ROC: Train vs Validation (Overfitting)
    colors3 = ['green' if diff < 0.03 else 'orange' if diff < 0.06 else 'red' for diff in train_val_auc]
    bars3 = ax3.bar(models, train_val_auc, color=colors3, alpha=0.7)
    ax3.axhline(y=0.03, color='orange', linestyle='--', alpha=0.7, label='Precaución (0.03)')
    ax3.axhline(y=0.06, color='red', linestyle='--', alpha=0.7, label='Overfitting (0.06)')
    ax3.set_title('Overfitting Analysis - AUC-ROC\n(Train - Validation)', fontweight='bold')
    ax3.set_ylabel('Diferencia AUC-ROC', fontweight='bold')
    ax3.set_xlabel('Modelos', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    for bar, diff in zip(bars3, train_val_auc):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{diff:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # AUC-ROC: Validation vs Test (Generalización)
    colors4 = ['green' if abs(diff) < 0.02 else 'orange' if abs(diff) < 0.04 else 'red' for diff in val_test_auc]
    bars4 = ax4.bar(models, val_test_auc, color=colors4, alpha=0.7)
    ax4.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Precaución (±0.02)')
    ax4.axhline(y=-0.02, color='orange', linestyle='--', alpha=0.7)
    ax4.axhline(y=0.04, color='red', linestyle='--', alpha=0.7, label='Problema (±0.04)')
    ax4.axhline(y=-0.04, color='red', linestyle='--', alpha=0.7)
    ax4.set_title('Generalization Analysis - AUC-ROC\n(Validation - Test)', fontweight='bold')
    ax4.set_ylabel('Diferencia AUC-ROC', fontweight='bold')
    ax4.set_xlabel('Modelos', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    for bar, diff in zip(bars4, val_test_auc):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{diff:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visualizaciones/analisis_completo/overfitting_detailed_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✅ Gráfica guardada: overfitting_detailed_analysis.png")
    plt.close()

def create_performance_summary_table(all_results):
    """Crea tabla resumen de rendimiento"""
    print("\n📋 TABLA RESUMEN DETALLADA")
    print("=" * 100)
    
    summary_data = []
    for model_name, results in all_results.items():
        # Calcular diferencias
        train_val_f1 = results['Train']['F1-Score'] - results['Validation']['F1-Score']
        val_test_f1 = results['Validation']['F1-Score'] - results['Test']['F1-Score']
        train_val_auc = results['Train']['AUC-ROC'] - results['Validation']['AUC-ROC']
        val_test_auc = results['Validation']['AUC-ROC'] - results['Test']['AUC-ROC']
        
        # Determinar estado
        overfitting_status = 'Severo' if train_val_f1 > 0.1 else 'Moderado' if train_val_f1 > 0.05 else 'Bueno'
        generalization_status = 'Problema' if abs(val_test_f1) > 0.06 else 'Precaución' if abs(val_test_f1) > 0.03 else 'Bueno'
        
        summary_data.append({
            'Modelo': model_name,
            'Train F1': f"{results['Train']['F1-Score']:.3f}",
            'Val F1': f"{results['Validation']['F1-Score']:.3f}",
            'Test F1': f"{results['Test']['F1-Score']:.3f}",
            'Train-Val Δ': f"{train_val_f1:.3f}",
            'Val-Test Δ': f"{val_test_f1:.3f}",
            'Train AUC': f"{results['Train']['AUC-ROC']:.3f}",
            'Val AUC': f"{results['Validation']['AUC-ROC']:.3f}",
            'Test AUC': f"{results['Test']['AUC-ROC']:.3f}",
            'Overfitting': overfitting_status,
            'Generalización': generalization_status
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    return summary_df

def generate_train_val_test_analysis():
    """Función principal"""
    print("🔍 ANÁLISIS COMPLETO: TRAIN vs VALIDATION vs TEST")
    print("=" * 60)
    
    # Cargar datos
    X, y, preprocessor = load_and_prepare_data()
    
    # Dividir datos: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nTamaños de conjuntos:")
    print(f"  Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Validation: {X_val.shape[0]} muestras ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Definir modelos
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
        print(f"\n--- Entrenando {model_name} ---")
        try:
            results = evaluate_train_val_test(model, X_train, y_train, X_val, y_val, X_test, y_test)
            all_results[model_name] = results
            
            # Mostrar resultados básicos
            print(f"  Train F1: {results['Train']['F1-Score']:.3f}")
            print(f"  Val F1:   {results['Validation']['F1-Score']:.3f}")
            print(f"  Test F1:  {results['Test']['F1-Score']:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error en {model_name}: {e}")
            continue
    
    if not all_results:
        print("❌ No se pudieron evaluar los modelos")
        return
    
    # Crear tabla resumen
    summary_df = create_performance_summary_table(all_results)
    
    # Generar gráficas
    print("\n📈 Generando gráficas...")
    try:
        create_train_val_test_comparison(all_results)
        create_overfitting_analysis(all_results)
        
        print("\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("📁 Gráficas guardadas en visualizaciones/analisis_completo/:")
        print("   - train_val_test_comparison.png")
        print("   - overfitting_detailed_analysis.png")
        
    except Exception as e:
        print(f"❌ Error generando gráficas: {e}")
    
    return all_results, summary_df

if __name__ == "__main__":
    results, summary = generate_train_val_test_analysis() 