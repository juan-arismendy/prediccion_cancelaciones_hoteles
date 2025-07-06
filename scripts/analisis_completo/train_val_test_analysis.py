#!/usr/bin/env python3
"""
Script para generar análisis Train/Validation/Test de los modelos
de predicción de cancelaciones de hoteles.
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

# Configuración de estilo para las gráficas
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_and_prepare_data():
    """
    Carga y prepara los datos para el análisis
    """
    print("=== CARGANDO Y PREPARANDO DATOS ===")
    
    # Cargar datos
    df = pd.read_csv('hotel_booking.csv')
    
    # Limpiar datos (misma lógica que en tu notebook)
    df['children'] = df['children'].fillna(0)
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    
    # Filtrar datos válidos
    df = df[df['adr'] >= 0]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['adr'], inplace=True)
    
    # Remover filas sin huéspedes
    initial_rows = df.shape[0]
    df = df[df['adults'] + df['children'] + df['babies'] > 0]
    print(f"Removidas {initial_rows - df.shape[0]} filas sin huéspedes")
    
    # Remover columnas de leakage
    df = df.drop(columns=['reservation_status', 'reservation_status_date'])
    
    # Reducir tamaño del dataset para velocidad (opcional)
    df = df.sample(frac=0.1, random_state=42)
    print(f"Dataset final: {df.shape[0]} filas")
    
    # Separar features y target
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    
    # Identificar features numéricas y categóricas
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    print(f"Features numéricas: {len(numerical_features)}")
    print(f"Features categóricas: {len(categorical_features)}")
    
    # Crear preprocessor
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return X, y, preprocessor

def evaluate_model_train_val_test(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    """
    Evalúa un modelo en conjuntos de entrenamiento, validación y test
    """
    print(f"Entrenando {model_name}...")
    
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
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, dataset in enumerate(datasets):
        values = []
        for model in models:
            value = df[(df['Modelo'] == model) & (df['Conjunto'] == dataset)][metric].iloc[0]
            values.append(value)
        
        plt.bar(x + i*width, values, width, label=dataset, alpha=0.8, color=colors[i])
    
    plt.xlabel('Modelos', fontsize=12, fontweight='bold')
    plt.ylabel(metric, fontsize=12, fontweight='bold')
    plt.title(f'Comparación de {metric} - Train vs Validation vs Test', 
              fontsize=14, fontweight='bold')
    plt.xticks(x + width, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def generate_train_val_test_analysis():
    """
    Genera análisis completo Train/Validation/Test para todos los modelos
    """
    print("=== ANÁLISIS TRAIN/VALIDATION/TEST ===")
    
    # Cargar y preparar datos
    X, y, preprocessor = load_and_prepare_data()
    
    # Dividir datos: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print(f"\nTamaños de conjuntos:")
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Validation: {X_val.shape[0]} muestras")
    print(f"  Test: {X_test.shape[0]} muestras")
    
    # Definir modelos con mejores hiperparámetros (basados en tu notebook)
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
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=30, max_features='sqrt'))
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
    
    # Análisis de overfitting
    print("\n🔍 Análisis de Overfitting")
    print("=" * 80)
    for model_name, results in all_results.items():
        train_f1 = results['Train']['F1-Score']
        val_f1 = results['Validation']['F1-Score']
        test_f1 = results['Test']['F1-Score']
        
        overfitting_score = train_f1 - val_f1
        generalization_score = val_f1 - test_f1
        
        print(f"\n{model_name}:")
        print(f"  Train F1: {train_f1:.3f}")
        print(f"  Validation F1: {val_f1:.3f}")
        print(f"  Test F1: {test_f1:.3f}")
        print(f"  Overfitting (Train-Val): {overfitting_score:.3f}")
        print(f"  Generalización (Val-Test): {generalization_score:.3f}")
        
        if overfitting_score > 0.05:
            print(f"  ⚠️  Posible overfitting detectado")
        elif generalization_score > 0.05:
            print(f"  ⚠️  Posible problema de generalización")
        else:
            print(f"  ✅ Modelo bien balanceado")
    
    return all_results

if __name__ == "__main__":
    generate_train_val_test_analysis() 