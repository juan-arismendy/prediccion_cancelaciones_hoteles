#!/usr/bin/env python3
"""
Gráfica simple y clara de Train vs Test
Muestra directamente la comparación de rendimiento
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Configuración para evitar problemas de GUI
import matplotlib
matplotlib.use('Agg')

# Configuración de estilo
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

def load_data():
    """Carga y prepara los datos"""
    print("📊 Cargando datos...")
    
    # Cargar datos
    df = pd.read_csv('datos/hotel_booking.csv')
    
    # Reducir dataset para velocidad
    df = df.sample(frac=0.15, random_state=42)
    print(f"Dataset: {df.shape[0]} filas")
    
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

def evaluate_model_simple(model, X_train, y_train, X_test, y_test):
    """Evalúa modelo en train y test"""
    # Entrenar
    model.fit(X_train, y_train)
    
    # Evaluar en train
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    # Evaluar en test
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'Train': {
            'Accuracy': accuracy_score(y_train, y_train_pred),
            'F1-Score': f1_score(y_train, y_train_pred),
            'AUC-ROC': roc_auc_score(y_train, y_train_proba)
        },
        'Test': {
            'Accuracy': accuracy_score(y_test, y_test_pred),
            'F1-Score': f1_score(y_test, y_test_pred),
            'AUC-ROC': roc_auc_score(y_test, y_test_proba)
        }
    }

def create_train_vs_test_chart(results):
    """Crea gráfica Train vs Test"""
    
    # Preparar datos
    models = list(results.keys())
    metrics = ['Accuracy', 'F1-Score', 'AUC-ROC']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎯 ANÁLISIS TRAIN vs TEST - Detección de Overfitting', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Gráfica 1: Comparación de métricas
    ax1 = axes[0, 0]
    
    train_f1 = [results[m]['Train']['F1-Score'] for m in models]
    test_f1 = [results[m]['Test']['F1-Score'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_f1, width, label='Train', 
                    color='lightblue', alpha=0.8, edgecolor='navy')
    bars2 = ax1.bar(x + width/2, test_f1, width, label='Test', 
                    color='lightcoral', alpha=0.8, edgecolor='darkred')
    
    ax1.set_xlabel('Modelos', fontweight='bold')
    ax1.set_ylabel('F1-Score', fontweight='bold')
    ax1.set_title('F1-Score: Train vs Test', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfica 2: Diferencias (Overfitting)
    ax2 = axes[0, 1]
    
    differences = [results[m]['Train']['F1-Score'] - results[m]['Test']['F1-Score'] for m in models]
    colors = ['green' if d < 0.05 else 'orange' if d < 0.1 else 'red' for d in differences]
    
    bars = ax2.bar(models, differences, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.8, label='Precaución (0.05)')
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.8, label='Overfitting (0.10)')
    ax2.set_xlabel('Modelos', fontweight='bold')
    ax2.set_ylabel('Diferencia F1 (Train - Test)', fontweight='bold')
    ax2.set_title('🚨 Detección de Overfitting', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Agregar valores
    for bar, diff in zip(bars, differences):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{diff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfica 3: AUC-ROC Comparison
    ax3 = axes[1, 0]
    
    train_auc = [results[m]['Train']['AUC-ROC'] for m in models]
    test_auc = [results[m]['Test']['AUC-ROC'] for m in models]
    
    bars1 = ax3.bar(x - width/2, train_auc, width, label='Train', 
                    color='lightgreen', alpha=0.8, edgecolor='darkgreen')
    bars2 = ax3.bar(x + width/2, test_auc, width, label='Test', 
                    color='lightyellow', alpha=0.8, edgecolor='orange')
    
    ax3.set_xlabel('Modelos', fontweight='bold')
    ax3.set_ylabel('AUC-ROC', fontweight='bold')
    ax3.set_title('AUC-ROC: Train vs Test', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Agregar valores
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfica 4: Tabla resumen
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Crear tabla
    table_data = []
    for model in models:
        train_f1 = results[model]['Train']['F1-Score']
        test_f1 = results[model]['Test']['F1-Score']
        diff = train_f1 - test_f1
        status = '✅ Bueno' if diff < 0.05 else '⚠️ Moderado' if diff < 0.1 else '❌ Severo'
        
        table_data.append([
            model,
            f'{train_f1:.3f}',
            f'{test_f1:.3f}',
            f'{diff:.3f}',
            status
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Modelo', 'Train F1', 'Test F1', 'Diferencia', 'Estado'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Colorear celdas según estado
    for i in range(len(models)):
        diff = float(table_data[i][3])
        if diff < 0.05:
            table[(i+1, 4)].set_facecolor('#90EE90')  # Verde claro
        elif diff < 0.1:
            table[(i+1, 4)].set_facecolor('#FFE4B5')  # Naranja claro
        else:
            table[(i+1, 4)].set_facecolor('#FFB6C1')  # Rojo claro
    
    ax4.set_title('📋 Resumen de Overfitting', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('visualizaciones/analisis_completo/train_vs_test_final.png', 
                dpi=300, bbox_inches='tight')
    print("✅ Gráfica guardada: train_vs_test_final.png")
    plt.close()

def main():
    """Función principal"""
    print("🎯 GENERANDO GRÁFICA TRAIN vs TEST")
    print("=" * 50)
    
    # Cargar datos
    X, y, preprocessor = load_data()
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape[0]} muestras")
    print(f"Test: {X_test.shape[0]} muestras")
    
    # Definir modelos optimizados
    models = {
        'Logistic Regression': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(random_state=42, C=1, solver='liblinear'))
        ]),
        'Random Forest': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=20))
        ]),
        'SVM': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', SVC(random_state=42, probability=True, C=10, kernel='rbf'))
        ]),
        'KNN': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', KNeighborsClassifier(n_neighbors=11, weights='distance'))
        ])
    }
    
    # Evaluar modelos
    results = {}
    print("\n📊 Evaluando modelos...")
    
    for name, model in models.items():
        print(f"  Entrenando {name}...")
        try:
            results[name] = evaluate_model_simple(model, X_train, y_train, X_test, y_test)
            train_f1 = results[name]['Train']['F1-Score']
            test_f1 = results[name]['Test']['F1-Score']
            print(f"    Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Crear gráfica
    if results:
        print("\n📈 Generando gráfica...")
        create_train_vs_test_chart(results)
        print("\n✅ ¡Gráfica generada exitosamente!")
        print("📁 Ubicación: visualizaciones/analisis_completo/train_vs_test_final.png")
    else:
        print("❌ No se pudieron evaluar los modelos")

if __name__ == "__main__":
    main() 