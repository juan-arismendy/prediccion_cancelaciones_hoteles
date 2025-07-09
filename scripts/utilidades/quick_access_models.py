#!/usr/bin/env python3
"""
Acceso Rápido a Modelos Guardados
Permite cargar y usar modelos entrenados sin necesidad de reentrenamiento.
"""

import os
import sys
from model_persistence import ModelPersistenceManager, load_trained_models, quick_model_access

def main():
    """
    Función principal para acceso rápido a modelos
    """
    print("🚀 Acceso Rápido a Modelos de Predicción de Cancelaciones Hoteleras")
    print("=" * 70)
    
    # Inicializar gestor de persistencia
    persistence_manager = ModelPersistenceManager()
    
    # Listar sesiones disponibles
    available_sessions = persistence_manager.list_available_sessions()
    
    if not available_sessions:
        print("❌ No se encontraron modelos guardados.")
        print("💡 Primero debes ejecutar el entrenamiento completo.")
        print("   Usa: python proyecto_final.py")
        return
    
    print(f"📁 Sesiones disponibles: {len(available_sessions)}")
    for i, session in enumerate(available_sessions, 1):
        print(f"   {i}. {session}")
    
    # Cargar la sesión más reciente por defecto
    latest_session = available_sessions[0]
    print(f"\n📂 Cargando sesión más reciente: {latest_session}")
    
    try:
        # Cargar todos los modelos y resultados
        models_dict, results_dict, session_info = load_trained_models(latest_session)
        
        if models_dict is None:
            print("❌ Error cargando modelos")
            return
        
        print(f"✅ Modelos cargados exitosamente: {list(models_dict.keys())}")
        
        # Mostrar resumen de resultados
        print("\n📊 RESUMEN DE RESULTADOS:")
        print("-" * 50)
        
        for model_name, results in results_dict.items():
            print(f"\n{model_name}:")
            print(f"  F1-Score: {results['F1-Score']:.3f} ± {results['F1-CI']:.3f}")
            print(f"  AUC-ROC: {results['AUC-ROC']:.3f} ± {results['AUC-CI']:.3f}")
            print(f"  Accuracy: {results['Accuracy']:.3f}")
            print(f"  Precision: {results['Precision']:.3f}")
            print(f"  Recall: {results['Recall']:.3f}")
        
        # Encontrar el mejor modelo
        best_model_name = max(results_dict.keys(), 
                            key=lambda x: results_dict[x]['F1-Score'])
        best_f1 = results_dict[best_model_name]['F1-Score']
        
        print(f"\n🏆 MEJOR MODELO: {best_model_name}")
        print(f"   F1-Score: {best_f1:.3f}")
        
        # Información de la sesión
        print(f"\n📋 INFORMACIÓN DE LA SESIÓN:")
        print(f"   Nombre: {session_info['session_name']}")
        print(f"   Timestamp: {session_info['timestamp']}")
        print(f"   Dataset shape: {session_info['metadata']['dataset_shape']}")
        print(f"   Modelos entrenados: {session_info['models_trained']}")
        
        # Guardar referencias globales para uso posterior
        global_models = models_dict
        global_results = results_dict
        global_session_info = session_info
        
        print(f"\n✅ Modelos listos para usar!")
        print("💡 Puedes acceder a los modelos usando:")
        print("   - global_models['RandomForest']")
        print("   - global_results['RandomForest']")
        print("   - global_session_info")
        
        return models_dict, results_dict, session_info
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

def get_best_model():
    """
    Obtiene el mejor modelo según F1-Score
    """
    models_dict, results_dict, _ = load_trained_models()
    
    if models_dict is None:
        return None, None
    
    best_model_name = max(results_dict.keys(), 
                         key=lambda x: results_dict[x]['F1-Score'])
    best_model = models_dict[best_model_name]
    best_results = results_dict[best_model_name]
    
    print(f"🏆 Mejor modelo cargado: {best_model_name}")
    print(f"   F1-Score: {best_results['F1-Score']:.3f}")
    print(f"   AUC-ROC: {best_results['AUC-ROC']:.3f}")
    
    return best_model, best_results

def get_model_by_name(model_name):
    """
    Obtiene un modelo específico por nombre
    """
    model, results, metadata = quick_model_access(model_name)
    
    if model is not None:
        print(f"✅ Modelo {model_name} cargado exitosamente")
        print(f"   F1-Score: {results['F1-Score']:.3f}")
        print(f"   AUC-ROC: {results['AUC-ROC']:.3f}")
    
    return model, results, metadata

def predict_with_saved_model(model_name, X_new):
    """
    Realiza predicciones usando un modelo guardado
    
    Args:
        model_name: Nombre del modelo
        X_new: Nuevos datos para predicción
    """
    model, results, _ = quick_model_access(model_name)
    
    if model is None:
        print(f"❌ No se pudo cargar el modelo {model_name}")
        return None
    
    try:
        # Realizar predicciones
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)[:, 1]
        
        print(f"✅ Predicciones realizadas con {model_name}")
        print(f"   Forma de predicciones: {predictions.shape}")
        print(f"   Forma de probabilidades: {probabilities.shape}")
        
        return predictions, probabilities
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return None

def compare_models():
    """
    Compara todos los modelos disponibles
    """
    models_dict, results_dict, _ = load_trained_models()
    
    if models_dict is None:
        return
    
    print("📊 COMPARACIÓN DE MODELOS:")
    print("=" * 60)
    
    # Crear tabla comparativa
    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Modelo': model_name,
            'F1-Score': f"{results['F1-Score']:.3f}",
            'AUC-ROC': f"{results['AUC-ROC']:.3f}",
            'Accuracy': f"{results['Accuracy']:.3f}",
            'Precision': f"{results['Precision']:.3f}",
            'Recall': f"{results['Recall']:.3f}"
        })
    
    # Ordenar por F1-Score
    comparison_data.sort(key=lambda x: float(x['F1-Score']), reverse=True)
    
    # Mostrar tabla
    print(f"{'Modelo':<15} {'F1-Score':<10} {'AUC-ROC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)
    
    for row in comparison_data:
        print(f"{row['Modelo']:<15} {row['F1-Score']:<10} {row['AUC-ROC']:<10} {row['Accuracy']:<10} {row['Precision']:<10} {row['Recall']:<10}")
    
    return comparison_data

if __name__ == "__main__":
    # Ejecutar carga de modelos
    models, results, session_info = main()
    
    if models is not None:
        print(f"\n🎯 Modelos disponibles para uso:")
        for model_name in models.keys():
            print(f"   - {model_name}")
        
        print(f"\n💡 Ejemplos de uso:")
        print(f"   - get_best_model(): Obtiene el mejor modelo")
        print(f"   - get_model_by_name('RandomForest'): Obtiene modelo específico")
        print(f"   - compare_models(): Compara todos los modelos")
        print(f"   - predict_with_saved_model('RandomForest', X_new): Predice con modelo guardado") 