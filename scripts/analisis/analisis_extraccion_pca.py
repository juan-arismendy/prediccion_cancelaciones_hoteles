#!/usr/bin/env python3
"""
Análisis de Extracción de Características con PCA
Proyecto: Predicción de Cancelaciones Hoteleras

- Análisis PCA en los 2 mejores modelos guardados
- Justificación del criterio de selección de componentes
- Tabla de resultados con porcentaje de varianza explicada y métricas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model_persistence import load_trained_models
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficas
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# 1. Cargar modelos y resultados guardados
print("\n=== ANÁLISIS DE EXTRACCIÓN DE CARACTERÍSTICAS CON PCA ===")
models_dict, results_dict, session_info = load_trained_models()

# Verificar que los datos se cargaron correctamente
if models_dict is None or results_dict is None:
    print("❌ Error: No se pudieron cargar los modelos guardados.")
    print("Asegúrate de haber ejecutado el entrenamiento con persistencia primero.")
    exit(1)

# Seleccionar los 2 mejores modelos por F1-Score
sorted_models = sorted(results_dict.items(), key=lambda x: x[1]['F1-Score'], reverse=True)
best_models = [models_dict[sorted_models[0][0]], models_dict[sorted_models[1][0]]]
best_model_names = [sorted_models[0][0], sorted_models[1][0]]

print(f"\n🏆 Modelos seleccionados para análisis: {best_model_names}")
print(f"Resultados originales guardados:")
for name in best_model_names:
    print(f"  {name}: F1-Score = {results_dict[name]['F1-Score']:.3f}")

# Cargar los datos originales usados en el entrenamiento
file_name = 'datos/hotel_booking.csv'
df = pd.read_csv(file_name)
df = df.sample(frac=0.1, random_state=42)
df['children'] = df['children'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)
df = df[df['adr'] >= 0]
df = df[df['adults'] + df['children'] + df['babies'] > 0]
df = df.drop(columns=['reservation_status', 'reservation_status_date'])
df = df.reset_index(drop=True)
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

# 2. Justificación del criterio de selección de componentes
print("\n=== JUSTIFICACIÓN DEL CRITERIO DE SELECCIÓN DE COMPONENTES ===")
print("Criterio elegido: 95% de varianza explicada")
print("Justificación:")
print("- 95% es un estándar ampliamente aceptado que balancea reducción de dimensionalidad y preservación de información.")
print("- Permite reducir significativamente el número de características manteniendo la mayor parte de la información relevante.")
print("- Es más conservador que 90% pero menos costoso que 99% en términos computacionales.")

# 3. Análisis PCA y evaluación
results_table = []
pca_info = []

for model, model_name in zip(best_models, best_model_names):
    print(f"\n--- {model_name}: Análisis PCA ---")
    
    # Obtener resultados originales del modelo guardado
    original_results = results_dict[model_name]
    print(f"Resultados originales: F1-Score = {original_results['F1-Score']:.3f}")
    
    # Extraer el preprocesador del pipeline
    preprocessor = model.named_steps['preprocessor']
    # Preprocesar los datos
    X_proc = preprocessor.fit_transform(X)
    # Si es sparse, convertir a dense
    if hasattr(X_proc, 'toarray'):
        X_proc = X_proc.toarray()
    
    n_features = X_proc.shape[1]
    print(f"Características originales: {n_features}")
    
    # OPTIMIZACIÓN: Si hay demasiadas características, hacer preselección
    if n_features > 100:
        print("⚠️  Muchas características detectadas. Aplicando preselección antes de PCA...")
        # Usar varianza para preseleccionar características
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)  # Mantener características con varianza > 0.01
        X_preselected = selector.fit_transform(X_proc)
        print(f"Preselección: {X_preselected.shape[1]} características seleccionadas")
        
        # Si aún hay demasiadas, usar SelectKBest
        if X_preselected.shape[1] > 50:
            from sklearn.feature_selection import SelectKBest, f_classif
            k_best = SelectKBest(score_func=f_classif, k=min(50, X_preselected.shape[1]))
            X_preselected = k_best.fit_transform(X_preselected, y)
            print(f"SelectKBest: {X_preselected.shape[1]} características seleccionadas")
    else:
        X_preselected = X_proc
    
    # Aplicar PCA con manejo de errores
    print("🔄 Aplicando PCA...")
    try:
        # Intentar con 95% de varianza
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_preselected)
        
        n_components = X_pca.shape[1]
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"Componentes seleccionados: {n_components}")
        print(f"Varianza explicada: {cumulative_variance[-1]*100:.1f}%")
        print(f"Reducción de dimensionalidad: {((n_features - n_components) / n_features) * 100:.1f}%")
        
    except Exception as e:
        print(f"⚠️  Error con 95% varianza: {e}")
        print("🔄 Intentando con 90% de varianza...")
        try:
            pca = PCA(n_components=0.90, random_state=42)
            X_pca = pca.fit_transform(X_preselected)
            
            n_components = X_pca.shape[1]
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            print(f"Componentes seleccionados: {n_components}")
            print(f"Varianza explicada: {cumulative_variance[-1]*100:.1f}%")
            print(f"Reducción de dimensionalidad: {((n_features - n_components) / n_features) * 100:.1f}%")
            
        except Exception as e2:
            print(f"⚠️  Error con 90% varianza: {e2}")
            print("🔄 Usando número fijo de componentes...")
            # Usar un número fijo de componentes
            n_components_fixed = min(20, X_preselected.shape[1])
            pca = PCA(n_components=n_components_fixed, random_state=42)
            X_pca = pca.fit_transform(X_preselected)
            
            n_components = X_pca.shape[1]
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            print(f"Componentes seleccionados: {n_components}")
            print(f"Varianza explicada: {cumulative_variance[-1]*100:.1f}%")
            print(f"Reducción de dimensionalidad: {((n_features - n_components) / n_features) * 100:.1f}%")
    
    # Evaluar el modelo con las características PCA
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
    clf = model.named_steps['classifier']
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Modelo': model_name,
        'N_Original': n_features,
        'N_Componentes': n_components,
        'Reducción (%)': ((n_features - n_components) / n_features) * 100,
        'Varianza_Explicada (%)': cumulative_variance[-1] * 100,
        'F1_Original': original_results['F1-Score'],
        'F1_PCA': f1_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    results_table.append(metrics)
    
    # Guardar información PCA para gráficas
    pca_info.append({
        'model_name': model_name,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'n_components': n_components
    })
    
    print(f"F1-Score original: {metrics['F1_Original']:.3f}")
    print(f"F1-Score con PCA: {metrics['F1-Score']:.3f}")

# Verificar que tenemos resultados para procesar
if not results_table:
    print("❌ Error: No se pudieron generar resultados de PCA.")
    exit(1)

# 4. Tabla de resultados
results_df = pd.DataFrame(results_table)
print("\n=== TABLA DE RESULTADOS DE EXTRACCIÓN PCA ===")
print(results_df[['Modelo', 'N_Original', 'N_Componentes', 'Reducción (%)', 'Varianza_Explicada (%)', 'F1_Original', 'F1_PCA', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']])

# 5. Gráfica de reducción de dimensionalidad
plt.figure(figsize=(12, 8))

# Crear subplots para mostrar diferentes aspectos
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Gráfica 1: Reducción de dimensionalidad vs F1-Score
x = np.arange(len(results_df))
width = 0.35

bars1 = ax1.bar(x - width/2, results_df['Reducción (%)'], width, 
                label='Reducción (%)', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, results_df['F1-Score'] * 100, width, 
                label='F1-Score (%)', color='#A23B72', alpha=0.8)

ax1.set_xlabel('Modelos')
ax1.set_ylabel('Porcentaje')
ax1.set_title('Reducción de Dimensionalidad vs F1-Score con PCA')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Modelo'].tolist())
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

# Gráfica 2: Comparación F1-Score original vs PCA
bars3 = ax2.bar(x - width/2, results_df['F1_Original'], width, 
                label='F1-Score Original', color='#2E86AB', alpha=0.8)
bars4 = ax2.bar(x + width/2, results_df['F1_PCA'], width, 
                label='F1-Score con PCA', color='#A23B72', alpha=0.8)

ax2.set_xlabel('Modelos')
ax2.set_ylabel('F1-Score')
ax2.set_title('Comparación F1-Score: Original vs PCA')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Modelo'].tolist())
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

for bar in bars4:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# Gráfica 3: Número de características vs componentes
bars5 = ax3.bar(x - width/2, results_df['N_Original'], width, 
                label='Características Originales', color='#2E86AB', alpha=0.8)
bars6 = ax3.bar(x + width/2, results_df['N_Componentes'], width, 
                label='Componentes PCA', color='#A23B72', alpha=0.8)

ax3.set_xlabel('Modelos')
ax3.set_ylabel('Número de Características')
ax3.set_title('Reducción de Dimensionalidad: Original vs PCA')
ax3.set_xticks(x)
ax3.set_xticklabels(results_df['Modelo'].tolist())
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
for bar in bars5:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 100,
             f'{int(height):,}', ha='center', va='bottom', fontsize=9)

for bar in bars6:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

# Gráfica 4: Varianza explicada y degradación
degradacion = ((results_df['F1_Original'] - results_df['F1_PCA']) / results_df['F1_Original']) * 100
bars7 = ax4.bar(x - width/2, results_df['Varianza_Explicada (%)'], width, 
                label='Varianza Explicada (%)', color='#4ECDC4', alpha=0.8)
bars8 = ax4.bar(x + width/2, degradacion, width, 
                label='Degradación (%)', color='#FF6B6B', alpha=0.8)

ax4.set_xlabel('Modelos')
ax4.set_ylabel('Porcentaje')
ax4.set_title('Varianza Explicada vs Degradación de Rendimiento')
ax4.set_xticks(x)
ax4.set_xticklabels(results_df['Modelo'].tolist())
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
for bar in bars7:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

for bar in bars8:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('../../resultados/graficas/pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Gráfica adicional: Resumen comparativo PCA
plt.figure(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.2

# Crear barras agrupadas
bars1 = plt.bar(x - width*1.5, results_df['F1_Original'], width, 
                label='F1-Score Original', color='#2E86AB', alpha=0.8)
bars2 = plt.bar(x - width*0.5, results_df['F1_PCA'], width, 
                label='F1-Score PCA', color='#A23B72', alpha=0.8)
bars3 = plt.bar(x + width*0.5, results_df['Reducción (%)'], width, 
                label='Reducción (%)', color='#F7931E', alpha=0.8)
bars4 = plt.bar(x + width*1.5, results_df['Varianza_Explicada (%)'], width, 
                label='Varianza (%)', color='#4ECDC4', alpha=0.8)

plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Resumen Comparativo: Análisis PCA')
plt.xticks(x, results_df['Modelo'].tolist())
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

for bar in bars3:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

for bar in bars4:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../../resultados/graficas/pca_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Gráfica de varianza explicada acumulada
plt.figure(figsize=(12, 8))
for i, info in enumerate(pca_info):
    plt.subplot(2, 1, i+1)
    plt.plot(range(1, len(info['explained_variance_ratio']) + 1), 
             info['cumulative_variance'] * 100, 
             marker='o', linewidth=2, markersize=4)
    plt.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% Varianza')
    plt.axvline(x=info['n_components'], color='green', linestyle='--', alpha=0.7, 
                label=f'{info["n_components"]} componentes')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada (%)')
    plt.title(f'Varianza Explicada Acumulada - {info["model_name"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

plt.tight_layout()
plt.savefig('pca_explained_variance.png', dpi=300)
plt.show()

# 8. Guardar tabla de resultados
results_df.to_csv('../../resultados/analisis/pca_extraction_results.csv', index=False)
print("\n✅ Análisis de extracción PCA completado.")
print("📊 Resultados guardados en: ../../resultados/analisis/pca_extraction_results.csv")
print("📈 Gráficas guardadas en: ../../resultados/graficas/") 