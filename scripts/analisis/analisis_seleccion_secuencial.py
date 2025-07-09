#!/usr/bin/env python3
"""
Análisis de Selección Secuencial de Características
Proyecto: Predicción de Cancelaciones Hoteleras

- Selección secuencial (forward y backward) en los 2 mejores modelos guardados
- Justificación del criterio de selección
- Tabla de resultados con porcentaje de reducción y métricas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
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
print("\n=== ANÁLISIS DE SELECCIÓN SECUENCIAL DE CARACTERÍSTICAS ===")
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

# Identificar características numéricas y categóricas
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# 2. Justificación del criterio de selección
print("\n=== JUSTIFICACIÓN DEL CRITERIO DE SELECCIÓN ===")
print("Criterio elegido: F1-Score (media armónica de precisión y recall)")
print("Justificación:")
print("- El dataset es desbalanceado, por lo que accuracy puede ser engañoso.")
print("- F1-Score balancea precisión y recall, penalizando tanto falsos positivos como falsos negativos.")
print("- Es el criterio más robusto para problemas de clasificación desbalanceada como cancelaciones hoteleras.")

# 3. Selección secuencial y evaluación (OPTIMIZADO)
results_table = []
reduction_table = []

for model, model_name in zip(best_models, best_model_names):
    print(f"\n--- {model_name}: Selección Secuencial (Forward) ---")
    
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
    
    # OPTIMIZACIÓN: Primero reducir a las mejores características usando SelectKBest
    n_features = X_proc.shape[1]
    print(f"Características originales: {n_features}")
    
    # Si hay demasiadas características, hacer una preselección rápida
    if n_features > 50:
        print("⚠️  Muchas características detectadas. Aplicando preselección rápida...")
        # Seleccionar las mejores 30 características primero
        k_best = SelectKBest(score_func=f_classif, k=min(30, n_features))
        X_preselected = k_best.fit_transform(X_proc, y)
        selected_features = k_best.get_support()
        print(f"Preselección: {X_preselected.shape[1]} características seleccionadas")
    else:
        X_preselected = X_proc
        selected_features = np.ones(n_features, dtype=bool)
    
    # Selección secuencial en el subconjunto preseleccionado
    n_select = max(1, int(X_preselected.shape[1] * 0.6))  # Seleccionar 60% de las características
    print(f"Características a seleccionar: {n_select}")
    
    # Usar menos CV folds y menos jobs para acelerar
    sfs = SequentialFeatureSelector(
        model.named_steps['classifier'],
        n_features_to_select=n_select,
        direction='forward',
        scoring='f1',
        cv=3,  # Reducido de 5 a 3
        n_jobs=1  # Reducido de 2 a 1 para evitar problemas de memoria
    )
    
    print("🔄 Ejecutando selección secuencial...")
    sfs.fit(X_preselected, y)
    selected_idx = sfs.get_support(indices=True)
    
    # Verificar que selected_idx no sea None
    if selected_idx is None:
        print(f"❌ Error: No se pudieron seleccionar características para {model_name}")
        continue
        
    X_selected = X_preselected[:, selected_idx]
    
    # Evaluar el modelo con las características seleccionadas
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    clf = model.named_steps['classifier']
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Modelo': model_name,
        'N_Original': n_features,
        'N_Seleccionadas': len(selected_idx),
        'Reducción (%)': 100 * (n_features - len(selected_idx)) / n_features,
        'F1_Original': original_results['F1-Score'],
        'F1_Seleccion': f1_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    results_table.append(metrics)
    print(f"Características seleccionadas: {len(selected_idx)}")
    print(f"Reducción: {metrics['Reducción (%)']:.1f}%")
    print(f"F1-Score original: {metrics['F1_Original']:.3f}")
    print(f"F1-Score con selección: {metrics['F1-Score']:.3f}")

# Verificar que tenemos resultados para procesar
if not results_table:
    print("❌ Error: No se pudieron generar resultados de selección secuencial.")
    exit(1)

# 4. Tabla de resultados
results_df = pd.DataFrame(results_table)
print("\n=== TABLA DE RESULTADOS DE SELECCIÓN SECUENCIAL ===")
print(results_df[['Modelo', 'N_Original', 'N_Seleccionadas', 'Reducción (%)', 'F1_Original', 'F1_Seleccion', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']])

# 5. Gráfica de reducción
plt.figure(figsize=(12, 8))

# Crear subplots para mostrar diferentes aspectos
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Gráfica 1: Reducción de características
x = np.arange(len(results_df))
width = 0.35

bars1 = ax1.bar(x - width/2, results_df['Reducción (%)'], width, 
                label='Reducción (%)', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, results_df['F1-Score'] * 100, width, 
                label='F1-Score (%)', color='#A23B72', alpha=0.8)

ax1.set_xlabel('Modelos')
ax1.set_ylabel('Porcentaje')
ax1.set_title('Reducción de Características vs F1-Score')
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

# Gráfica 2: Comparación F1-Score original vs seleccionado
bars3 = ax2.bar(x - width/2, results_df['F1_Original'], width, 
                label='F1-Score Original', color='#2E86AB', alpha=0.8)
bars4 = ax2.bar(x + width/2, results_df['F1_Seleccion'], width, 
                label='F1-Score con Selección', color='#A23B72', alpha=0.8)

ax2.set_xlabel('Modelos')
ax2.set_ylabel('F1-Score')
ax2.set_title('Comparación F1-Score: Original vs Selección Secuencial')
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

# Gráfica 3: Número de características
bars5 = ax3.bar(x - width/2, results_df['N_Original'], width, 
                label='Características Originales', color='#2E86AB', alpha=0.8)
bars6 = ax3.bar(x + width/2, results_df['N_Seleccionadas'], width, 
                label='Características Seleccionadas', color='#A23B72', alpha=0.8)

ax3.set_xlabel('Modelos')
ax3.set_ylabel('Número de Características')
ax3.set_title('Reducción de Dimensionalidad')
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

# Gráfica 4: Degradación de rendimiento
degradacion = ((results_df['F1_Original'] - results_df['F1_Seleccion']) / results_df['F1_Original']) * 100
bars7 = ax4.bar(x, degradacion, color=['#FF6B6B' if d > 5 else '#4ECDC4' for d in degradacion], alpha=0.8)

ax4.set_xlabel('Modelos')
ax4.set_ylabel('Degradación (%)')
ax4.set_title('Degradación de Rendimiento')
ax4.set_xticks(x)
ax4.set_xticklabels(results_df['Modelo'].tolist())
ax4.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
for i, bar in enumerate(bars7):
    height = bar.get_height()
    color = 'red' if degradacion.iloc[i] > 5 else 'green'
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10, color=color)

plt.tight_layout()
plt.savefig('../../resultados/graficas/sequential_selection_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Gráfica adicional: Resumen comparativo
plt.figure(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.25

# Crear barras agrupadas
bars1 = plt.bar(x - width*1.5, results_df['F1_Original'], width, 
                label='F1-Score Original', color='#2E86AB', alpha=0.8)
bars2 = plt.bar(x - width*0.5, results_df['F1_Seleccion'], width, 
                label='F1-Score Selección', color='#A23B72', alpha=0.8)
bars3 = plt.bar(x + width*0.5, results_df['Reducción (%)'], width, 
                label='Reducción (%)', color='#F7931E', alpha=0.8)
bars4 = plt.bar(x + width*1.5, degradacion, width, 
                label='Degradación (%)', color='#FF6B6B', alpha=0.8)

plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Resumen Comparativo: Selección Secuencial')
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
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../../resultados/graficas/sequential_selection_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Guardar tabla de resultados
results_df.to_csv('../../resultados/analisis/sequential_selection_results.csv', index=False)
print("\n✅ Análisis de selección secuencial completado.")
print("📊 Resultados guardados en: ../../resultados/analisis/sequential_selection_results.csv")
print("📈 Gráficas guardadas en: ../../resultados/graficas/") 