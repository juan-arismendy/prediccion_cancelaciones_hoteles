#!/usr/bin/env python3
"""
Análisis de Recomendación Final de Modelos
Proyecto: Predicción de Cancelaciones Hoteleras

- Comparación de modelos antes y después de reducción de dimensionalidad
- Recomendación final basada en múltiples criterios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilidades'))
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

print("\n" + "="*80)
print("🎯 ANÁLISIS DE RECOMENDACIÓN FINAL DE MODELOS")
print("="*80)

# 1. Cargar resultados originales
print("\n📊 CARGANDO RESULTADOS ORIGINALES...")
models_dict, results_dict, session_info = load_trained_models()

if models_dict is None or results_dict is None:
    print("❌ Error: No se pudieron cargar los modelos guardados.")
    exit(1)

# Crear DataFrame con resultados originales
original_results = []
for model_name, results in results_dict.items():
    original_results.append({
        'Modelo': model_name,
        'F1-Score': results['F1-Score'],
        'Accuracy': results['Accuracy'],
        'Precision': results['Precision'],
        'Recall': results['Recall'],
        'AUC-ROC': results['AUC-ROC']
    })

df_original = pd.DataFrame(original_results)
df_original = df_original.sort_values('F1-Score', ascending=False)

print("\n🏆 RANKING DE MODELOS ORIGINALES (ANTES DE REDUCCIÓN):")
print(df_original[['Modelo', 'F1-Score', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']].round(3))

# 2. Cargar resultados de selección secuencial
print("\n📈 CARGANDO RESULTADOS DE SELECCIÓN SECUENCIAL...")
try:
    df_sequential = pd.read_csv('../../resultados/analisis/sequential_selection_results.csv')
    print("✅ Resultados de selección secuencial cargados")
except:
    print("❌ No se encontraron resultados de selección secuencial")
    df_sequential = pd.DataFrame()

# 3. Cargar resultados de PCA
print("\n📉 CARGANDO RESULTADOS DE PCA...")
try:
    df_pca = pd.read_csv('../../resultados/analisis/pca_extraction_results.csv')
    print("✅ Resultados de PCA cargados")
except:
    print("❌ No se encontraron resultados de PCA")
    df_pca = pd.DataFrame()

# 4. Análisis comparativo
print("\n" + "="*80)
print("📊 ANÁLISIS COMPARATIVO")
print("="*80)

# Comparar los 2 mejores modelos originales
best_models = df_original.head(2)['Modelo'].tolist()
print(f"\n🏆 Los 2 mejores modelos originales: {best_models}")

# Crear tabla comparativa
comparison_data = []

for model in best_models:
    # Resultados originales
    orig_row = df_original[df_original['Modelo'] == model].iloc[0]
    
    # Resultados selección secuencial
    seq_row = df_sequential[df_sequential['Modelo'] == model] if not df_sequential.empty else None
    
    # Resultados PCA
    pca_row = df_pca[df_pca['Modelo'] == model] if not df_pca.empty else None
    
    comparison_data.append({
        'Modelo': model,
        'F1_Original': orig_row['F1-Score'],
        'F1_Sequential': seq_row['F1-Score'].iloc[0] if seq_row is not None and not seq_row.empty else None,
        'F1_PCA': pca_row['F1-Score'].iloc[0] if pca_row is not None and not pca_row.empty else None,
        'Reduccion_Sequential': seq_row['Reducción (%)'].iloc[0] if seq_row is not None and not seq_row.empty else None,
        'Reduccion_PCA': pca_row['Reducción (%)'].iloc[0] if pca_row is not None and not pca_row.empty else None,
        'Varianza_PCA': pca_row['Varianza_Explicada (%)'].iloc[0] if pca_row is not None and not pca_row.empty else None
    })

df_comparison = pd.DataFrame(comparison_data)

print("\n📋 TABLA COMPARATIVA DE LOS 2 MEJORES MODELOS:")
print(df_comparison.round(3))

# 5. Análisis de degradación de rendimiento
print("\n📉 ANÁLISIS DE DEGRADACIÓN DE RENDIMIENTO:")
for _, row in df_comparison.iterrows():
    model = row['Modelo']
    f1_orig = row['F1_Original']
    
    print(f"\n--- {model} ---")
    print(f"F1-Score original: {f1_orig:.3f}")
    
    if row['F1_Sequential'] is not None:
        degradation_seq = ((f1_orig - row['F1_Sequential']) / f1_orig) * 100
        print(f"Selección Secuencial: {row['F1_Sequential']:.3f} (degradación: {degradation_seq:.1f}%)")
    
    if row['F1_PCA'] is not None:
        degradation_pca = ((f1_orig - row['F1_PCA']) / f1_orig) * 100
        print(f"PCA: {row['F1_PCA']:.3f} (degradación: {degradation_pca:.1f}%)")

# 6. Recomendación final
print("\n" + "="*80)
print("🎯 RECOMENDACIÓN FINAL")
print("="*80)

# Criterios de evaluación
print("\n📋 CRITERIOS DE EVALUACIÓN:")
print("1. Rendimiento (F1-Score)")
print("2. Reducción de dimensionalidad")
print("3. Interpretabilidad")
print("4. Robustez")
print("5. Eficiencia computacional")

# Análisis por técnica
print("\n📊 ANÁLISIS POR TÉCNICA:")

print("\n🔍 SELECCIÓN SECUENCIAL:")
if not df_sequential.empty:
    best_seq = df_sequential.loc[df_sequential['F1-Score'].idxmax()]
    print(f"✅ Mejor modelo: {best_seq['Modelo']} (F1: {best_seq['F1-Score']:.3f})")
    print(f"✅ Reducción: {best_seq['Reducción (%)']:.1f}%")
    print("✅ Ventajas: Mantiene características originales, más interpretable")
    print("⚠️  Desventajas: Mayor degradación de rendimiento")

print("\n📉 PCA:")
if not df_pca.empty:
    best_pca = df_pca.loc[df_pca['F1-Score'].idxmax()]
    print(f"✅ Mejor modelo: {best_pca['Modelo']} (F1: {best_pca['F1-Score']:.3f})")
    print(f"✅ Reducción: {best_pca['Reducción (%)']:.1f}%")
    print(f"✅ Varianza explicada: {best_pca['Varianza_Explicada (%)']:.1f}%")
    print("✅ Ventajas: Menor degradación, componentes ortogonales")
    print("⚠️  Desventajas: Pérdida de interpretabilidad")

# Recomendación final
print("\n🏆 RECOMENDACIÓN FINAL:")
print("="*50)

if not df_pca.empty and not df_sequential.empty:
    # Comparar las mejores técnicas
    best_seq_f1 = df_sequential['F1-Score'].max()
    best_pca_f1 = df_pca['F1-Score'].max()
    
    if best_pca_f1 > best_seq_f1:
        best_technique = "PCA"
        best_model = df_pca.loc[df_pca['F1-Score'].idxmax()]
        print(f"🎯 TÉCNICA RECOMENDADA: PCA")
        print(f"🎯 MODELO: {best_model['Modelo']}")
        print(f"🎯 F1-Score: {best_model['F1-Score']:.3f}")
        print(f"🎯 Reducción: {best_model['Reducción (%)']:.1f}%")
        print(f"🎯 Varianza explicada: {best_model['Varianza_Explicada (%)']:.1f}%")
        
        print("\n📝 JUSTIFICACIÓN:")
        print("- PCA muestra menor degradación de rendimiento")
        print("- Mantiene 95% de varianza explicada")
        print("- Componentes ortogonales mejoran estabilidad")
        print("- Mejor balance entre rendimiento y reducción")
        
    else:
        best_technique = "Selección Secuencial"
        best_model = df_sequential.loc[df_sequential['F1-Score'].idxmax()]
        print(f"🎯 TÉCNICA RECOMENDADA: Selección Secuencial")
        print(f"🎯 MODELO: {best_model['Modelo']}")
        print(f"🎯 F1-Score: {best_model['F1-Score']:.3f}")
        print(f"🎯 Reducción: {best_model['Reducción (%)']:.1f}%")
        
        print("\n📝 JUSTIFICACIÓN:")
        print("- Mantiene características originales")
        print("- Mayor interpretabilidad")
        print("- Mejor rendimiento en este caso específico")

print("\n" + "="*80)
print("📋 RESUMEN EJECUTIVO")
print("="*80)

print(f"\n📊 MODELO ORIGINAL RECOMENDADO: {df_original.iloc[0]['Modelo']}")
print(f"   F1-Score: {df_original.iloc[0]['F1-Score']:.3f}")

if not df_pca.empty and not df_sequential.empty:
    print(f"\n🔧 TÉCNICA DE REDUCCIÓN RECOMENDADA: {best_technique}")
    print(f"   Modelo final: {best_model['Modelo']}")
    print(f"   F1-Score final: {best_model['F1-Score']:.3f}")
    print(f"   Reducción de dimensionalidad: {best_model['Reducción (%)']:.1f}%")

print("\n✅ Análisis de recomendación completado.") 