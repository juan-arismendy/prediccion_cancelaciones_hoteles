#!/usr/bin/env python3
"""
Script para análisis de características y reducción de dimensionalidad
en el proyecto de predicción de cancelaciones de hoteles.
Incluye:
- Análisis de correlación
- Capacidad discriminativa de variables
- Selección de características
- Evaluación de técnicas de reducción de dimensionalidad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr, chi2_contingency
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
    Carga y prepara los datos para el análisis de características
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
    
    print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y, df

def analyze_correlations(X, y):
    """
    Analiza correlaciones entre variables numéricas
    """
    print("\n=== ANÁLISIS DE CORRELACIONES ===")
    
    # Obtener variables numéricas
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    X_num = X[numerical_features].copy()
    
    # Agregar target para análisis
    X_num_with_target = X_num.copy()
    X_num_with_target['is_canceled'] = y
    
    # Matriz de correlación
    correlation_matrix = X_num_with_target.corr()
    
    # Crear gráfica de matriz de correlación
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlación - Variables Numéricas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Análisis de correlación con target
    print("\n📊 Correlación con variable objetivo (is_canceled):")
    print("=" * 60)
    
    target_correlations = []
    for feature in numerical_features:
        corr_pearson, p_value_pearson = pearsonr(X_num[feature], y)
        corr_spearman, p_value_spearman = spearmanr(X_num[feature], y)
        
        target_correlations.append({
            'Variable': feature,
            'Correlación Pearson': corr_pearson,
            'P-value Pearson': p_value_pearson,
            'Correlación Spearman': corr_spearman,
            'P-value Spearman': p_value_spearman,
            'Significativa': p_value_pearson < 0.05
        })
    
    corr_df = pd.DataFrame(target_correlations)
    corr_df = corr_df.sort_values('Correlación Pearson', key=abs, ascending=False)
    
    print(corr_df.to_string(index=False, float_format='%.4f'))
    
    # Identificar variables altamente correlacionadas
    print("\n🔍 Variables altamente correlacionadas (|r| > 0.8):")
    print("=" * 60)
    
    high_corr_pairs = []
    for i in range(len(numerical_features)):
        for j in range(i+1, len(numerical_features)):
            feat1, feat2 = numerical_features[i], numerical_features[j]
            corr = correlation_matrix.loc[feat1, feat2]
            if abs(corr) > 0.8:
                high_corr_pairs.append({
                    'Variable 1': feat1,
                    'Variable 2': feat2,
                    'Correlación': corr
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        print(high_corr_df.to_string(index=False, float_format='%.4f'))
    else:
        print("No se encontraron variables con correlación > 0.8")
    
    return corr_df, high_corr_pairs

def analyze_categorical_features(X, y):
    """
    Analiza variables categóricas usando chi-cuadrado y análisis de frecuencias
    """
    print("\n=== ANÁLISIS DE VARIABLES CATEGÓRICAS ===")
    
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    if not categorical_features:
        print("No hay variables categóricas para analizar")
        return pd.DataFrame()
    
    # Análisis de chi-cuadrado
    print("\n📊 Análisis Chi-Cuadrado:")
    print("=" * 60)
    
    chi2_results = []
    for feature in categorical_features:
        # Crear tabla de contingencia
        contingency_table = pd.crosstab(X[feature], y)
        
        # Calcular chi-cuadrado
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calcular Cramer's V (medida de asociación)
        n = len(X)
        min_dim = min(contingency_table.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        chi2_results.append({
            'Variable': feature,
            'Chi2': chi2,
            'P-value': p_value,
            "Cramer's V": cramer_v,
            'Significativa': p_value < 0.05,
            'Valores únicos': X[feature].nunique()
        })
    
    chi2_df = pd.DataFrame(chi2_results)
    chi2_df = chi2_df.sort_values('Chi2', ascending=False)
    
    print(chi2_df.to_string(index=False, float_format='%.4f'))
    
    # Análisis de frecuencias por clase
    print("\n📊 Análisis de frecuencias por clase:")
    print("=" * 60)
    
    for feature in categorical_features[:5]:  # Mostrar solo las primeras 5
        print(f"\n{feature}:")
        freq_table = pd.crosstab(X[feature], y, normalize='index') * 100
        print(freq_table.round(2))
    
    return chi2_df

def analyze_discriminative_power(X, y):
    """
    Analiza la capacidad discriminativa de cada variable
    """
    print("\n=== ANÁLISIS DE CAPACIDAD DISCRIMINATIVA ===")
    
    # Preparar datos
    X_encoded = X.copy()
    
    # Codificar variables categóricas
    le = LabelEncoder()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    for feature in categorical_features:
        X_encoded[feature] = le.fit_transform(X_encoded[feature].astype(str))
    
    # Análisis ANOVA F-test
    print("\n📊 ANOVA F-test (para variables numéricas):")
    print("=" * 60)
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    f_scores, p_values = f_classif(X_encoded[numerical_features], y)
    
    f_test_results = []
    for i, feature in enumerate(numerical_features):
        f_test_results.append({
            'Variable': feature,
            'F-score': f_scores[i],
            'P-value': p_values[i],
            'Significativa': p_values[i] < 0.05
        })
    
    f_test_df = pd.DataFrame(f_test_results)
    f_test_df = f_test_df.sort_values('F-score', ascending=False)
    
    print(f_test_df.to_string(index=False, float_format='%.4f'))
    
    # Análisis de información mutua
    print("\n📊 Información Mutua:")
    print("=" * 60)
    
    mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
    
    mi_results = []
    for i, feature in enumerate(X_encoded.columns):
        mi_results.append({
            'Variable': feature,
            'Información Mutua': mi_scores[i]
        })
    
    mi_df = pd.DataFrame(mi_results)
    mi_df = mi_df.sort_values('Información Mutua', ascending=False)
    
    print(mi_df.to_string(index=False, float_format='%.4f'))
    
    # Gráfica de capacidad discriminativa
    plt.figure(figsize=(14, 8))
    
    # Top 15 variables por información mutua
    top_features = mi_df.head(15)
    
    plt.subplot(1, 2, 1)
    plt.barh(range(len(top_features)), top_features['Información Mutua'])
    plt.yticks(range(len(top_features)), top_features['Variable'])
    plt.xlabel('Información Mutua')
    plt.title('Top 15 Variables por Información Mutua')
    plt.gca().invert_yaxis()
    
    # Top 15 variables por F-score
    top_f_features = f_test_df.head(15)
    
    plt.subplot(1, 2, 2)
    plt.barh(range(len(top_f_features)), top_f_features['F-score'])
    plt.yticks(range(len(top_f_features)), top_f_features['Variable'])
    plt.xlabel('F-score')
    plt.title('Top 15 Variables por F-score')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return f_test_df, mi_df

def feature_selection_analysis(X, y):
    """
    Realiza análisis de selección de características
    """
    print("\n=== ANÁLISIS DE SELECCIÓN DE CARACTERÍSTICAS ===")
    
    # Preparar datos
    X_encoded = X.copy()
    le = LabelEncoder()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    for feature in categorical_features:
        X_encoded[feature] = le.fit_transform(X_encoded[feature].astype(str))
    
    # 1. SelectKBest con diferentes métodos
    print("\n📊 SelectKBest Analysis:")
    print("=" * 60)
    
    k_values = [10, 20, 30, 40, 50]
    methods = {
        'f_classif': f_classif,
        'mutual_info_classif': mutual_info_classif
    }
    
    for method_name, method_func in methods.items():
        print(f"\n{method_name.upper()}:")
        for k in k_values:
            if k <= X_encoded.shape[1]:
                selector = SelectKBest(score_func=method_func, k=k)
                X_selected = selector.fit_transform(X_encoded, y)
                selected_features = X_encoded.columns[selector.get_support()].tolist()
                print(f"  k={k}: {len(selected_features)} características seleccionadas")
    
    # 2. Random Forest Feature Importance
    print("\n📊 Random Forest Feature Importance:")
    print("=" * 60)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_encoded, y)
    
    feature_importance = pd.DataFrame({
        'Variable': X_encoded.columns,
        'Importancia': rf.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    print(feature_importance.head(20).to_string(index=False, float_format='%.4f'))
    
    # Gráfica de importancia de características
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['Importancia'])
    plt.yticks(range(len(top_features)), top_features['Variable'])
    plt.xlabel('Importancia')
    plt.title('Top 20 Variables por Importancia (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return feature_importance

def dimensionality_reduction_analysis(X, y):
    """
    Analiza técnicas de reducción de dimensionalidad
    """
    print("\n=== ANÁLISIS DE REDUCCIÓN DE DIMENSIONALIDAD ===")
    
    # Preparar datos
    X_encoded = X.copy()
    le = LabelEncoder()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    for feature in categorical_features:
        X_encoded[feature] = le.fit_transform(X_encoded[feature].astype(str))
    
    # Estandarizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # 1. Análisis de Componentes Principales (PCA)
    print("\n📊 Análisis de Componentes Principales (PCA):")
    print("=" * 60)
    
    # PCA para diferentes números de componentes
    n_components_range = [2, 5, 10, 15, 20, 25, 30]
    explained_variances = []
    
    for n_comp in n_components_range:
        if n_comp <= X_scaled.shape[1]:
            pca = PCA(n_components=n_comp)
            pca.fit(X_scaled)
            explained_variances.append({
                'n_components': n_comp,
                'explained_variance_ratio': pca.explained_variance_ratio_.sum(),
                'cumulative_variance': pca.explained_variance_ratio_.sum()
            })
    
    pca_results = pd.DataFrame(explained_variances)
    print(pca_results.to_string(index=False, float_format='%.4f'))
    
    # Gráfica de varianza explicada
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(pca_results['n_components'], pca_results['explained_variance_ratio'], 'bo-')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada vs Número de Componentes')
    plt.grid(True, alpha=0.3)
    
    # PCA con 2 componentes para visualización
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f})')
    plt.title('PCA - Visualización 2D')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    # 2. t-SNE para visualización
    print("\n📊 t-SNE Analysis:")
    print("=" * 60)
    
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE - Visualización 2D')
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.show()
        
        print("t-SNE completado exitosamente")
    except Exception as e:
        print(f"Error en t-SNE: {e}")
    
    return pca_results

def identify_candidate_features_for_removal(X, y):
    """
    Identifica características candidatas para eliminación
    """
    print("\n=== IDENTIFICACIÓN DE CARACTERÍSTICAS CANDIDATAS PARA ELIMINACIÓN ===")
    
    # Preparar datos
    X_encoded = X.copy()
    le = LabelEncoder()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    for feature in categorical_features:
        X_encoded[feature] = le.fit_transform(X_encoded[feature].astype(str))
    
    # Criterios para eliminación
    candidates_for_removal = []
    
    # 1. Variables con baja información mutua
    mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
    mi_threshold = np.percentile(mi_scores, 25)  # Bottom 25%
    
    for i, feature in enumerate(X_encoded.columns):
        if mi_scores[i] < mi_threshold:
            candidates_for_removal.append({
                'Variable': feature,
                'Criterio': 'Baja información mutua',
                'Valor': mi_scores[i],
                'Umbral': mi_threshold
            })
    
    # 2. Variables con baja correlación con target
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    for feature in numerical_features:
        corr, p_value = pearsonr(X_encoded[feature], y)
        if abs(corr) < 0.05 and p_value > 0.05:  # Correlación baja y no significativa
            candidates_for_removal.append({
                'Variable': feature,
                'Criterio': 'Baja correlación con target',
                'Valor': corr,
                'Umbral': 0.05
            })
    
    # 3. Variables con alta correlación entre sí (redundantes)
    correlation_matrix = X_encoded.corr()
    for i in range(len(X_encoded.columns)):
        for j in range(i+1, len(X_encoded.columns)):
            feat1, feat2 = X_encoded.columns[i], X_encoded.columns[j]
            corr = correlation_matrix.loc[feat1, feat2]
            if abs(corr) > 0.9:  # Correlación muy alta
                # Mantener la variable con mayor información mutua
                mi1 = mi_scores[i]
                mi2 = mi_scores[j]
                var_to_remove = feat2 if mi1 > mi2 else feat1
                candidates_for_removal.append({
                    'Variable': var_to_remove,
                    'Criterio': f'Alta correlación con {feat1 if var_to_remove == feat2 else feat2}',
                    'Valor': corr,
                    'Umbral': 0.9
                })
    
    # Crear DataFrame de candidatos
    if candidates_for_removal:
        candidates_df = pd.DataFrame(candidates_for_removal)
        candidates_df = candidates_df.drop_duplicates(subset=['Variable'])
        candidates_df = candidates_df.sort_values('Criterio')
        
        print("\n📊 Características candidatas para eliminación:")
        print("=" * 80)
        print(candidates_df.to_string(index=False, float_format='%.4f'))
        
        # Resumen por criterio
        print("\n📊 Resumen por criterio de eliminación:")
        print("=" * 50)
        summary = candidates_df['Criterio'].value_counts()
        for criterio, count in summary.items():
            print(f"{criterio}: {count} variables")
        
        # Lista final de variables a eliminar
        variables_to_remove = candidates_df['Variable'].unique().tolist()
        print(f"\n🎯 Total de variables candidatas para eliminación: {len(variables_to_remove)}")
        print("Variables:", variables_to_remove)
        
    else:
        print("No se identificaron características candidatas para eliminación")
        variables_to_remove = []
    
    return candidates_df if candidates_for_removal else pd.DataFrame(), variables_to_remove

def generate_comprehensive_report(X, y):
    """
    Genera un reporte completo del análisis de características
    """
    print("=== REPORTE COMPLETO DE ANÁLISIS DE CARACTERÍSTICAS ===")
    print("=" * 80)
    
    # Información general del dataset
    print(f"\n📊 Información del Dataset:")
    print(f"  Total de características: {X.shape[1]}")
    print(f"  Características numéricas: {len(X.select_dtypes(include=np.number).columns)}")
    print(f"  Características categóricas: {len(X.select_dtypes(include='object').columns)}")
    print(f"  Número de muestras: {X.shape[0]}")
    
    # Análisis de correlaciones
    corr_df, high_corr_pairs = analyze_correlations(X, y)
    
    # Análisis de variables categóricas
    chi2_df = analyze_categorical_features(X, y)
    
    # Análisis de capacidad discriminativa
    f_test_df, mi_df = analyze_discriminative_power(X, y)
    
    # Selección de características
    feature_importance = feature_selection_analysis(X, y)
    
    # Reducción de dimensionalidad
    pca_results = dimensionality_reduction_analysis(X, y)
    
    # Identificación de candidatos para eliminación
    candidates_df, variables_to_remove = identify_candidate_features_for_removal(X, y)
    
    # Resumen final
    print("\n" + "="*80)
    print("🎯 RESUMEN FINAL Y RECOMENDACIONES")
    print("="*80)
    
    print(f"\n📈 Características más importantes:")
    top_features = feature_importance.head(10)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"  {i}. {row['Variable']}: {row['Importancia']:.4f}")
    
    print(f"\n🔍 Características candidatas para eliminación: {len(variables_to_remove)}")
    if variables_to_remove:
        for var in variables_to_remove:
            print(f"  - {var}")
    
    print(f"\n💡 Recomendaciones:")
    print(f"  1. Mantener las top {min(20, len(feature_importance))} características por importancia")
    print(f"  2. Considerar eliminar {len(variables_to_remove)} características de baja utilidad")
    print(f"  3. Para PCA, usar {pca_results[pca_results['explained_variance_ratio'] >= 0.95]['n_components'].iloc[0]} componentes para 95% de varianza")
    print(f"  4. Evaluar el impacto de la reducción en el rendimiento del modelo")
    
    return {
        'correlation_analysis': corr_df,
        'categorical_analysis': chi2_df,
        'discriminative_analysis': (f_test_df, mi_df),
        'feature_importance': feature_importance,
        'pca_results': pca_results,
        'candidates_for_removal': candidates_df,
        'variables_to_remove': variables_to_remove
    }

if __name__ == "__main__":
    # Cargar datos
    X, y, df = load_and_prepare_data()
    
    # Generar reporte completo
    results = generate_comprehensive_report(X, y) 