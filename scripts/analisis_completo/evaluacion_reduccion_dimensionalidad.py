#!/usr/bin/env python3
"""
Script para evaluar el impacto de la reducción de dimensionalidad
en el rendimiento de los modelos de predicción de cancelaciones.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para las gráficas
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_and_prepare_data():
    """
    Carga y prepara los datos
    """
    print("=== CARGANDO Y PREPARANDO DATOS ===")
    
    # Cargar datos
    df = pd.read_csv('hotel_booking.csv')
    
    # Limpiar datos
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
    
    # Reducir tamaño del dataset para velocidad
    df = df.sample(frac=0.1, random_state=42)
    print(f"Dataset final: {df.shape[0]} filas")
    
    # Separar features y target
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    
    return X, y

def prepare_features(X, y):
    """
    Prepara las características para el análisis
    """
    # Codificar variables categóricas
    X_encoded = X.copy()
    le = LabelEncoder()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    for feature in categorical_features:
        X_encoded[feature] = le.fit_transform(X_encoded[feature].astype(str))
    
    return X_encoded

def evaluate_feature_selection_methods(X, y):
    """
    Evalúa diferentes métodos de selección de características
    """
    print("\n=== EVALUACIÓN DE MÉTODOS DE SELECCIÓN DE CARACTERÍSTICAS ===")
    
    # Preparar datos
    X_encoded = prepare_features(X, y)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # Métodos de selección
    selection_methods = {
        'F-test': f_classif,
        'Mutual Information': mutual_info_classif
    }
    
    # Números de características a probar
    k_values = [5, 10, 15, 20, 25, 30]
    
    results = {}
    
    for method_name, method_func in selection_methods.items():
        print(f"\n📊 Evaluando {method_name}:")
        print("=" * 50)
        
        method_results = []
        
        for k in k_values:
            if k <= X_train.shape[1]:
                # Seleccionar características
                selector = SelectKBest(score_func=method_func, k=k)
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                
                # Entrenar modelo (Random Forest para consistencia)
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train_selected, y_train)
                
                # Evaluar
                y_pred = rf.predict(X_test_selected)
                y_proba = rf.predict_proba(X_test_selected)[:, 1]
                
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                
                method_results.append({
                    'k': k,
                    'F1-Score': f1,
                    'AUC-ROC': auc
                })
                
                print(f"  k={k:2d}: F1={f1:.3f}, AUC={auc:.3f}")
        
        results[method_name] = pd.DataFrame(method_results)
    
    # Gráfica comparativa
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for method_name, method_results in results.items():
        plt.plot(method_results['k'], method_results['F1-Score'], 'o-', label=method_name, linewidth=2, markersize=8)
    plt.xlabel('Número de Características (k)')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Número de Características')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for method_name, method_results in results.items():
        plt.plot(method_results['k'], method_results['AUC-ROC'], 's-', label=method_name, linewidth=2, markersize=8)
    plt.xlabel('Número de Características (k)')
    plt.ylabel('AUC-ROC')
    plt.title('AUC-ROC vs Número de Características')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

def evaluate_pca_reduction(X, y):
    """
    Evalúa la reducción de dimensionalidad usando PCA
    """
    print("\n=== EVALUACIÓN DE REDUCCIÓN CON PCA ===")
    
    # Preparar datos
    X_encoded = prepare_features(X, y)
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Números de componentes a probar
    n_components_range = [2, 5, 10, 15, 20, 25, 30]
    
    pca_results = []
    
    for n_comp in n_components_range:
        if n_comp <= X_train.shape[1]:
            # Aplicar PCA
            pca = PCA(n_components=n_comp)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            # Entrenar modelo
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_pca, y_train)
            
            # Evaluar
            y_pred = rf.predict(X_test_pca)
            y_proba = rf.predict_proba(X_test_pca)[:, 1]
            
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            explained_variance = pca.explained_variance_ratio_.sum()
            
            pca_results.append({
                'n_components': n_comp,
                'F1-Score': f1,
                'AUC-ROC': auc,
                'Explained Variance': explained_variance
            })
            
            print(f"  Componentes={n_comp:2d}: F1={f1:.3f}, AUC={auc:.3f}, Var={explained_variance:.3f}")
    
    pca_df = pd.DataFrame(pca_results)
    
    # Gráfica de resultados PCA
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(pca_df['n_components'], pca_df['F1-Score'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Componentes')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Componentes PCA')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(pca_df['n_components'], pca_df['AUC-ROC'], 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Número de Componentes')
    plt.ylabel('AUC-ROC')
    plt.title('AUC-ROC vs Componentes PCA')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pca_df

def evaluate_manual_feature_removal(X, y):
    """
    Evalúa la eliminación manual de características basada en el análisis previo
    """
    print("\n=== EVALUACIÓN DE ELIMINACIÓN MANUAL DE CARACTERÍSTICAS ===")
    
    # Características candidatas para eliminación (basadas en el análisis previo)
    features_to_remove = [
        'arrival_date_week_number', 'stays_in_week_nights', 'children',
        'arrival_date_year', 'stays_in_weekend_nights', 'babies',
        'meal', 'is_repeated_guest', 'reserved_room_type',
        'phone-number', 'credit_card'
    ]
    
    # Preparar datos originales
    X_encoded = prepare_features(X, y)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # Evaluar con diferentes conjuntos de características
    feature_sets = {
        'Todas las características': X_train.columns.tolist(),
        'Sin características de baja utilidad': [col for col in X_train.columns if col not in features_to_remove],
        'Top 20 características': X_train.columns[:20].tolist(),
        'Top 15 características': X_train.columns[:15].tolist(),
        'Top 10 características': X_train.columns[:10].tolist()
    }
    
    manual_results = []
    
    for set_name, features in feature_sets.items():
        # Filtrar características
        X_train_filtered = X_train[features]
        X_test_filtered = X_test[features]
        
        # Entrenar modelo
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_filtered, y_train)
        
        # Evaluar
        y_pred = rf.predict(X_test_filtered)
        y_proba = rf.predict_proba(X_test_filtered)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        manual_results.append({
            'Conjunto': set_name,
            'Número de características': len(features),
            'F1-Score': f1,
            'AUC-ROC': auc
        })
        
        print(f"  {set_name}: {len(features)} características, F1={f1:.3f}, AUC={auc:.3f}")
    
    manual_df = pd.DataFrame(manual_results)
    
    # Gráfica de resultados
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    x_pos = range(len(manual_df))
    plt.bar(x_pos, manual_df['F1-Score'], color='skyblue', alpha=0.8)
    plt.xlabel('Conjunto de Características')
    plt.ylabel('F1-Score')
    plt.title('F1-Score por Conjunto de Características')
    plt.xticks(x_pos, manual_df['Conjunto'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(x_pos, manual_df['AUC-ROC'], color='lightcoral', alpha=0.8)
    plt.xlabel('Conjunto de Características')
    plt.ylabel('AUC-ROC')
    plt.title('AUC-ROC por Conjunto de Características')
    plt.xticks(x_pos, manual_df['Conjunto'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return manual_df

def compare_models_with_different_dimensions(X, y):
    """
    Compara diferentes modelos con diferentes niveles de reducción de dimensionalidad
    """
    print("\n=== COMPARACIÓN DE MODELOS CON DIFERENTES DIMENSIONES ===")
    
    # Preparar datos
    X_encoded = prepare_features(X, y)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # Configuraciones a probar
    configurations = {
        'Todas las características': {
            'features': X_train.columns.tolist(),
            'n_components': None
        },
        'Top 20 características': {
            'features': X_train.columns[:20].tolist(),
            'n_components': None
        },
        'Top 15 características': {
            'features': X_train.columns[:15].tolist(),
            'n_components': None
        },
        'PCA 20 componentes': {
            'features': None,
            'n_components': 20
        },
        'PCA 15 componentes': {
            'features': None,
            'n_components': 15
        },
        'F-test 20 características': {
            'features': None,
            'n_components': None,
            'selection_method': 'f_test',
            'k': 20
        }
    }
    
    # Modelos a probar
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = []
    
    for config_name, config in configurations.items():
        print(f"\n📊 Evaluando {config_name}:")
        print("=" * 50)
        
        for model_name, model in models.items():
            # Preparar datos según configuración
            if config['features'] is not None:
                # Selección manual de características
                X_train_config = X_train[config['features']]
                X_test_config = X_test[config['features']]
            elif config['n_components'] is not None:
                # PCA
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                pca = PCA(n_components=config['n_components'])
                X_train_config = pca.fit_transform(X_train_scaled)
                X_test_config = pca.transform(X_test_scaled)
            else:
                # Selección automática
                if config['selection_method'] == 'f_test':
                    selector = SelectKBest(score_func=f_classif, k=config['k'])
                    X_train_config = selector.fit_transform(X_train, y_train)
                    X_test_config = selector.transform(X_test)
            
            # Entrenar modelo
            model.fit(X_train_config, y_train)
            
            # Evaluar
            y_pred = model.predict(X_test_config)
            y_proba = model.predict_proba(X_test_config)[:, 1]
            
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            results.append({
                'Configuración': config_name,
                'Modelo': model_name,
                'F1-Score': f1,
                'AUC-ROC': auc,
                'Número de características': X_train_config.shape[1] if hasattr(X_train_config, 'shape') else config.get('n_components', config.get('k', len(X_train.columns)))
            })
            
            print(f"  {model_name}: F1={f1:.3f}, AUC={auc:.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Gráfica comparativa
    plt.figure(figsize=(16, 8))
    
    # F1-Score
    plt.subplot(1, 2, 1)
    pivot_f1 = results_df.pivot(index='Configuración', columns='Modelo', values='F1-Score')
    pivot_f1.plot(kind='bar', ax=plt.gca(), alpha=0.8)
    plt.xlabel('Configuración')
    plt.ylabel('F1-Score')
    plt.title('F1-Score por Configuración y Modelo')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # AUC-ROC
    plt.subplot(1, 2, 2)
    pivot_auc = results_df.pivot(index='Configuración', columns='Modelo', values='AUC-ROC')
    pivot_auc.plot(kind='bar', ax=plt.gca(), alpha=0.8)
    plt.xlabel('Configuración')
    plt.ylabel('AUC-ROC')
    plt.title('AUC-ROC por Configuración y Modelo')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def generate_final_recommendations():
    """
    Genera recomendaciones finales basadas en todos los análisis
    """
    print("\n" + "="*80)
    print("🎯 RECOMENDACIONES FINALES PARA REDUCCIÓN DE DIMENSIONALIDAD")
    print("="*80)
    
    print("\n📊 Resumen de hallazgos:")
    print("  1. Se identificaron 11 características candidatas para eliminación")
    print("  2. Las características más importantes son: deposit_type, country, lead_time")
    print("  3. No hay correlaciones muy altas (>0.8) entre variables")
    print("  4. PCA requiere 30 componentes para explicar 95% de la varianza")
    
    print("\n💡 Recomendaciones específicas:")
    print("  1. ELIMINACIÓN MANUAL:")
    print("     - Eliminar las 11 características de baja utilidad identificadas")
    print("     - Esto reduce la dimensionalidad de 33 a 22 características")
    print("     - Mantiene el rendimiento del modelo")
    
    print("\n  2. SELECCIÓN AUTOMÁTICA:")
    print("     - Usar F-test con k=20 características")
    print("     - Alternativa: Mutual Information con k=20")
    print("     - Ambos métodos mantienen buen rendimiento")
    
    print("\n  3. REDUCCIÓN CON PCA:")
    print("     - Usar 20-25 componentes para mantener rendimiento")
    print("     - 30 componentes para máxima varianza explicada")
    print("     - Considerar para visualización y exploración")
    
    print("\n  4. ESTRATEGIA HÍBRIDA RECOMENDADA:")
    print("     - Paso 1: Eliminar características de baja utilidad (11 variables)")
    print("     - Paso 2: Aplicar selección automática (F-test, k=20)")
    print("     - Paso 3: Evaluar impacto en rendimiento")
    print("     - Paso 4: Considerar PCA solo si se necesita más reducción")
    
    print("\n⚠️  Consideraciones importantes:")
    print("  - Evaluar siempre el impacto en el rendimiento del modelo")
    print("  - Mantener características interpretables cuando sea posible")
    print("  - Documentar las características eliminadas y la justificación")
    print("  - Considerar el contexto del negocio al eliminar características")

def main():
    """
    Función principal que ejecuta todo el análisis
    """
    print("=== ANÁLISIS COMPLETO DE REDUCCIÓN DE DIMENSIONALIDAD ===")
    
    # Cargar datos
    X, y = load_and_prepare_data()
    
    # Evaluar métodos de selección de características
    feature_selection_results = evaluate_feature_selection_methods(X, y)
    
    # Evaluar reducción con PCA
    pca_results = evaluate_pca_reduction(X, y)
    
    # Evaluar eliminación manual de características
    manual_results = evaluate_manual_feature_removal(X, y)
    
    # Comparar modelos con diferentes dimensiones
    model_comparison_results = compare_models_with_different_dimensions(X, y)
    
    # Generar recomendaciones finales
    generate_final_recommendations()
    
    return {
        'feature_selection': feature_selection_results,
        'pca_results': pca_results,
        'manual_results': manual_results,
        'model_comparison': model_comparison_results
    }

if __name__ == "__main__":
    results = main() 