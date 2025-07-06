#!/usr/bin/env python3
"""
Script para ejecutar el proyecto de predicción de cancelaciones de hoteles
Basado en el notebook proyecto_final.ipynb
"""

import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

def download_dataset():
    """Descarga el dataset desde Kaggle"""
    print("Descargando dataset desde Kaggle...")
    try:
        # Add kaggle to PATH if not found
        kaggle_path = "/Library/Frameworks/Python.framework/Versions/3.12/bin/kaggle"
        if os.path.exists(kaggle_path):
            os.environ['PATH'] = "/Library/Frameworks/Python.framework/Versions/3.12/bin:" + os.environ.get('PATH', '')
        
        result = os.system("kaggle datasets download -d mojtaba142/hotel-booking")
        if result == 0:
            os.system("unzip -o hotel-booking.zip")
            print("Dataset descargado exitosamente!")
            return True
        else:
            print("Error: No se pudo descargar el dataset. Verifica tu configuración de Kaggle.")
            return False
    except Exception as e:
        print(f"Error descargando dataset: {e}")
        return False

def load_and_clean_data():
    """Carga y limpia los datos"""
    print("\n=== CARGANDO Y LIMPIANDO DATOS ===")
    
    # Cargar datos
    try:
        df = pd.read_csv('hotel_booking.csv')
        print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    except FileNotFoundError:
        print("Error: No se encontró hotel_booking.csv")
        return None, None
    
    # Reducir tamaño del dataset (10% como en el notebook)
    percentage_to_keep = 0.1
    df = df.sample(frac=percentage_to_keep, random_state=42)
    print(f"Dataset reducido a {df.shape[0]} filas ({percentage_to_keep*100}%)")
    
    # Limpiar datos
    df['children'] = df['children'].fillna(0)
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    
    # Eliminar filas con ADR negativo o cero
    df = df[df['adr'] >= 0]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['adr'], inplace=True)
    
    # Eliminar filas sin huéspedes
    initial_rows = df.shape[0]
    df = df[df['adults'] + df['children'] + df['babies'] > 0]
    print(f"Eliminadas {initial_rows - df.shape[0]} filas sin huéspedes")
    
    # Eliminar variables de fuga
    df = df.drop(columns=['reservation_status', 'reservation_status_date'])
    
    # Separar features y target
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    
    print(f"Distribución del target: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y

def setup_preprocessing():
    """Configura el preprocesamiento"""
    print("\n=== CONFIGURANDO PREPROCESAMIENTO ===")
    
    # Identificar tipos de features
    numerical_features = ['lead_time', 'arrival_date_year', 'arrival_date_week_number', 
                         'arrival_date_day_of_month', 'stays_in_weekend_nights', 
                         'stays_in_week_nights', 'adults', 'children', 'babies', 
                         'is_repeated_guest', 'previous_cancellations', 
                         'previous_bookings_not_canceled', 'booking_changes', 'agent', 
                         'company', 'days_in_waiting_list', 'adr', 
                         'required_car_parking_spaces', 'total_of_special_requests']
    
    categorical_features = ['hotel', 'arrival_date_month', 'meal', 'country', 
                           'market_segment', 'distribution_channel', 'reserved_room_type', 
                           'assigned_room_type', 'deposit_type', 'customer_type', 
                           'name', 'email', 'phone-number', 'credit_card']
    
    # Crear preprocesador
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    print("Preprocesador configurado exitosamente")
    return preprocessor

def evaluate_model(model, X_data, y_data, model_name="Model"):
    """Evalúa un modelo usando validación cruzada"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    f1_scores, auc_roc_scores = [], []
    accuracy_scores, precision_scores, recall_scores = [], [], []
    
    print(f"\n--- Evaluando {model_name} ---")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_data, y_data)):
        X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Predecir
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Calcular métricas
        accuracy_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        auc_roc_scores.append(roc_auc_score(y_val, y_proba))
        
        print(f"  Fold {fold+1}: F1 = {f1_scores[-1]:.3f}, AUC = {auc_roc_scores[-1]:.3f}")
    
    print(f"\n{model_name} - Resultados Promedio:")
    print(f"  Accuracy: {np.mean(accuracy_scores):.3f} +/- {np.std(accuracy_scores)*2:.3f}")
    print(f"  Precision: {np.mean(precision_scores):.3f} +/- {np.std(precision_scores)*2:.3f}")
    print(f"  Recall: {np.mean(recall_scores):.3f} +/- {np.std(recall_scores)*2:.3f}")
    print(f"  F1-Score: {np.mean(f1_scores):.3f} +/- {np.std(f1_scores)*2:.3f}")
    print(f"  AUC-ROC: {np.mean(auc_roc_scores):.3f} +/- {np.std(auc_roc_scores)*2:.3f}")
    
    return {
        'F1-Score': np.mean(f1_scores),
        'AUC-ROC': np.mean(auc_roc_scores),
        'F1-CI': np.std(f1_scores) * 2,
        'AUC-CI': np.std(auc_roc_scores) * 2
    }

def train_models(X, y, preprocessor):
    """Entrena y evalúa todos los modelos"""
    print("\n=== ENTRENANDO MODELOS ===")
    
    results = {}
    
    # Modelo 1: Regresión Logística
    print("\n--- Modelo 1: Regresión Logística ---")
    pipeline_lr = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42, solver='saga', penalty='l1'))
    ])
    
    param_grid_lr = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    grid_search_lr.fit(X, y)
    
    print(f"Mejores parámetros: {grid_search_lr.best_params_}")
    results['Regresión Logística'] = evaluate_model(grid_search_lr.best_estimator_, X, y, "Regresión Logística")
    
    # Modelo 2: KNN
    print("\n--- Modelo 2: K-Nearest Neighbors ---")
    pipeline_knn = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', KNeighborsClassifier())
    ])
    
    param_grid_knn = {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1, 2]
    }
    grid_search_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    grid_search_knn.fit(X, y)
    
    print(f"Mejores parámetros: {grid_search_knn.best_params_}")
    results['KNN'] = evaluate_model(grid_search_knn.best_estimator_, X, y, "KNN")
    
    # Modelo 3: Random Forest
    print("\n--- Modelo 3: Random Forest ---")
    pipeline_rf = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid_rf = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', 0.5]
    }
    grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    grid_search_rf.fit(X, y)
    
    print(f"Mejores parámetros: {grid_search_rf.best_params_}")
    results['Random Forest'] = evaluate_model(grid_search_rf.best_estimator_, X, y, "Random Forest")
    
    # Modelo 4: Redes Neuronales (MLP)
    print("\n--- Modelo 4: Redes Neuronales (MLP) ---")
    pipeline_mlp = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', MLPClassifier(random_state=42, max_iter=500))
    ])
    
    param_grid_mlp = {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__solver': ['adam', 'sgd'],
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__learning_rate': ['constant', 'adaptive']
    }
    grid_search_mlp = GridSearchCV(pipeline_mlp, param_grid_mlp, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    grid_search_mlp.fit(X, y)
    
    print(f"Mejores parámetros: {grid_search_mlp.best_params_}")
    results['MLP'] = evaluate_model(grid_search_mlp.best_estimator_, X, y, "MLP")
    
    # Modelo 5: Support Vector Machine (SVM)
    print("\n--- Modelo 5: Support Vector Machine (SVM) ---")
    pipeline_svc = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', SVC(random_state=42, probability=True))
    ])
    
    param_grid_svc = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf', 'poly'],
        'classifier__gamma': ['scale', 'auto', 0.1, 1]
    }
    grid_search_svc = GridSearchCV(pipeline_svc, param_grid_svc, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    grid_search_svc.fit(X, y)
    
    print(f"Mejores parámetros: {grid_search_svc.best_params_}")
    results['SVM'] = evaluate_model(grid_search_svc.best_estimator_, X, y, "SVM")
    
    return results

def print_summary(results):
    """Imprime un resumen de todos los resultados"""
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    
    print(f"{'Modelo':<20} {'F1-Score':<12} {'AUC-ROC':<12}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        f1 = metrics['F1-Score']
        auc = metrics['AUC-ROC']
        print(f"{model_name:<20} {f1:.3f}±{metrics['F1-CI']:.3f} {auc:.3f}±{metrics['AUC-CI']:.3f}")
    
    # Encontrar el mejor modelo
    best_model = max(results.items(), key=lambda x: x[1]['AUC-ROC'])
    print(f"\n🏆 MEJOR MODELO: {best_model[0]}")
    print(f"   AUC-ROC: {best_model[1]['AUC-ROC']:.3f}")
    print(f"   F1-Score: {best_model[1]['F1-Score']:.3f}")

def main():
    """Función principal"""
    print("🚀 INICIANDO PROYECTO DE PREDICCIÓN DE CANCELACIONES DE HOTELES")
    print("="*70)
    
    # Descargar dataset si no existe
    if not os.path.exists('hotel_booking.csv'):
        if not download_dataset():
            print("No se pudo descargar el dataset. Asegúrate de tener configurado Kaggle.")
            return
    
    # Cargar y limpiar datos
    X, y = load_and_clean_data()
    if X is None:
        return
    
    # Configurar preprocesamiento
    preprocessor = setup_preprocessing()
    
    # Entrenar modelos
    results = train_models(X, y, preprocessor)
    
    # Mostrar resumen
    print_summary(results)
    
    print("\n✅ ¡Proyecto completado exitosamente!")
    print("📊 Revisa los resultados arriba para ver el rendimiento de cada modelo.")

if __name__ == "__main__":
    main() 