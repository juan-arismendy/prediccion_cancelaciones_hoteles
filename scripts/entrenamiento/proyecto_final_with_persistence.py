# ---------------------------------------------
# PROYECTO: Predicción de Cancelaciones Hoteleras CON PERSISTENCIA
# ---------------------------------------------

# Importación de librerías necesarias
import os  # Para operaciones del sistema de archivos
import zipfile  # Para manejar archivos comprimidos ZIP
import pandas as pd  # Para manipulación de datos en DataFrames
import numpy as np  # Para operaciones numéricas y manejo de NaN
from sklearn.model_selection import StratifiedKFold, GridSearchCV  # Para validación cruzada y búsqueda de hiperparámetros
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Para normalización y codificación de variables categóricas
from sklearn.compose import ColumnTransformer  # Para aplicar transformaciones a columnas específicas
from sklearn.pipeline import Pipeline  # Para crear flujos de procesamiento
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # Para evaluar modelos
from imblearn.over_sampling import SMOTE  # Para balancear clases usando sobremuestreo sintético
from imblearn.pipeline import Pipeline as ImbPipeline  # Pipeline compatible con imbalanced-learn

# Importar sistema de persistencia
from model_persistence import ModelPersistenceManager, train_and_save_models, load_trained_models

# -----------------------------
# 1. Carga de los datos
# -----------------------------
file_name = 'datos/hotel_booking.csv'  # Nombre del archivo de datos
try:
    # Intentamos leer el archivo CSV en un DataFrame de pandas
    fetched_df = pd.read_csv(file_name)
    print(f"\n¡Dataset '{os.path.basename(file_name)}' cargado exitosamente en DataFrame!\n")
except FileNotFoundError:
    # Si no se encuentra el archivo, mostramos un error y detenemos la ejecución
    print(f"Error: No se encontró el archivo CSV en {file_name} después de descomprimir. Verifique el contenido descomprimido.")
    exit()

# Mostramos información inicial del DataFrame para entender los tipos de datos y valores nulos
print("Información inicial del DataFrame:")
fetched_df.info()
print("\n")

# -----------------------------
# 2. Reducción del tamaño del dataset (opcional)
# -----------------------------
# Esto es útil para evitar problemas de memoria en entornos limitados
percentage_to_keep = 0.1  # Porcentaje de filas a conservar (10%)

if percentage_to_keep < 1.0:
    print(f"Reduciendo el tamaño del dataset al {percentage_to_keep*100:.2f}% de las filas originales.")
    df = fetched_df.sample(frac=percentage_to_keep, random_state=42)  # Muestreo aleatorio reproducible
    print(f"Nuevo tamaño del dataset: {df.shape[0]} filas.")
else:
    df = fetched_df.copy()

# Separamos las variables predictoras (X) y la variable objetivo (y)
X = df.drop('is_canceled', axis=1)  # X contiene todas las columnas excepto 'is_canceled'
y = df['is_canceled']  # y contiene la columna objetivo

print("\nDataFrame después de la posible reducción de tamaño:")
print(X.head())
print(f"Distribución de la variable objetivo tras la reducción:\n{y.value_counts(normalize=True)}\n")

# Identificamos las variables numéricas y categóricas para el preprocesamiento posterior
numerical_features = X.select_dtypes(include=np.number).columns.tolist()  # Lista de columnas numéricas
categorical_features = X.select_dtypes(include='object').columns.tolist()  # Lista de columnas categóricas

print(f"Características numéricas identificadas: {numerical_features}")
print(f"Características categóricas identificadas: {categorical_features}\n")

# -----------------------------
# 3. Limpieza y preprocesamiento inicial
# -----------------------------
# Rellenamos valores nulos en columnas relevantes
# Se asume que valores nulos en 'children', 'agent' y 'company' pueden ser tratados como 0
# Esto es razonable porque la ausencia de estos datos suele indicar que no aplica (ej: sin niños, sin agente, sin compañía)
df['children'] = df['children'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)

# Eliminamos filas con valores de ADR (tarifa diaria promedio) negativos o cero, ya que suelen ser errores de registro
df = df[df['adr'] >= 0]
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Reemplazamos infinitos por NaN si existen
df.dropna(subset=['adr'], inplace=True)  # Eliminamos filas donde 'adr' sea NaN

# Eliminamos filas donde no hay huéspedes (adultos, niños y bebés todos en cero)
initial_rows = df.shape[0]
df = df[df['adults'] + df['children'] + df['babies'] > 0]
print(f"Se eliminaron {initial_rows - df.shape[0]} filas con 0 huéspedes en total.\n")

# Eliminamos columnas que pueden causar fuga de información (leakage) hacia la predicción
df = df.drop(columns=['reservation_status', 'reservation_status_date'])  # Estas columnas indican directamente la cancelación

# Volvemos a separar X e y tras la limpieza
df = df.reset_index(drop=True)
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

print("DataFrame tras limpieza y preprocesamiento inicial:")
print(X.head())
print(f"Distribución de la variable objetivo:\n{y.value_counts(normalize=True)}\n")

# Reidentificamos las variables numéricas y categóricas tras la limpieza
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"Características numéricas identificadas: {numerical_features}")
print(f"Características categóricas identificadas: {categorical_features}\n")

# -----------------------------
# 4. Sistema de Persistencia de Modelos
# -----------------------------

print("🔧 Inicializando sistema de persistencia de modelos...")
persistence_manager = ModelPersistenceManager()

# Verificar si ya existen modelos guardados
available_sessions = persistence_manager.list_available_sessions()

if available_sessions:
    print(f"📁 Se encontraron {len(available_sessions)} sesiones guardadas:")
    for session in available_sessions[:3]:  # Mostrar solo las 3 más recientes
        print(f"   - {session}")
    
    # Preguntar al usuario si quiere usar modelos existentes o entrenar nuevos
    print("\n🤔 ¿Qué prefieres hacer?")
    print("1. Usar modelos guardados (más rápido)")
    print("2. Entrenar nuevos modelos")
    
    choice = input("Elige una opción (1 o 2): ").strip()
    
    if choice == "1":
        print("\n📂 Cargando modelos guardados...")
        try:
            models_dict, results_dict, session_info = load_trained_models()
            
            if models_dict is not None:
                print("✅ Modelos cargados exitosamente!")
                print(f"Modelos disponibles: {list(models_dict.keys())}")
                
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
                
                # Guardar referencias globales
                global_models = models_dict
                global_results = results_dict
                global_session_info = session_info
                
                print(f"\n✅ ¡Listo! Puedes usar los modelos guardados.")
                print("💡 Ejemplos de uso:")
                print("   - global_models['RandomForest'] para acceder al modelo")
                print("   - global_results['RandomForest'] para ver resultados")
                print("   - get_best_model() para obtener el mejor modelo")
                
                # Salir del script aquí si se cargaron modelos exitosamente
                import sys
                sys.exit(0)
                
            else:
                print("❌ Error cargando modelos. Procediendo con entrenamiento...")
                choice = "2"
        
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Procediendo con entrenamiento...")
            choice = "2"
    
    if choice == "2":
        print("\n🚀 Procediendo con entrenamiento de nuevos modelos...")

# -----------------------------
# 5. Entrenamiento y Guardado de Modelos
# -----------------------------

print("\n--- Iniciando entrenamiento y guardado de modelos ---")

# Entrenar y guardar todos los modelos
session_name, trained_models, all_results = train_and_save_models(X, y)

if session_name:
    print(f"\n🎉 Entrenamiento completado exitosamente!")
    print(f"📁 Sesión guardada: {session_name}")
    print(f"📊 Modelos entrenados: {list(trained_models.keys())}")
    
    # Mostrar resumen de resultados
    print("\n📊 RESUMEN DE RESULTADOS:")
    print("-" * 50)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"  F1-Score: {results['F1-Score']:.3f} ± {results['F1-CI']:.3f}")
        print(f"  AUC-ROC: {results['AUC-ROC']:.3f} ± {results['AUC-CI']:.3f}")
        print(f"  Accuracy: {results['Accuracy']:.3f}")
        print(f"  Precision: {results['Precision']:.3f}")
        print(f"  Recall: {results['Recall']:.3f}")
    
    # Encontrar el mejor modelo
    best_model_name = max(all_results.keys(), 
                         key=lambda x: all_results[x]['F1-Score'])
    best_f1 = all_results[best_model_name]['F1-Score']
    
    print(f"\n🏆 MEJOR MODELO: {best_model_name}")
    print(f"   F1-Score: {best_f1:.3f}")
    
    # Guardar referencias globales
    global_models = trained_models
    global_results = all_results
    
    print(f"\n✅ ¡Entrenamiento completado!")
    print("💡 Los modelos están guardados y listos para uso futuro.")
    print("💡 Para cargar modelos sin reentrenar, usa: python quick_access_models.py")
    
else:
    print("❌ Error en el entrenamiento de modelos")

print("\n--- Fin del análisis de modelos ---") 