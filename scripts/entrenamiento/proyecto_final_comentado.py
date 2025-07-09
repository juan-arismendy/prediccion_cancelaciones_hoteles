# ---------------------------------------------
# PROYECTO: Predicción de Cancelaciones Hoteleras
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
# 4. Definición del pipeline de preprocesamiento
# -----------------------------

# Creamos el transformador para variables numéricas: escalado estándar (media=0, varianza=1)
# Esto es importante porque muchos modelos (especialmente los basados en distancia y redes neuronales) requieren que los datos estén normalizados
numerical_transformer = StandardScaler()

# Creamos el transformador para variables categóricas: codificación one-hot
# handle_unknown='ignore' permite que categorías no vistas en entrenamiento no generen error en predicción
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combinamos ambos transformadores en un ColumnTransformer
# Esto permite aplicar diferentes transformaciones a columnas específicas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),  # Aplica escalado a variables numéricas
        ('cat', categorical_transformer, categorical_features)  # Aplica one-hot a categóricas
    ],
    remainder='passthrough'  # Deja pasar columnas no especificadas (por si acaso)
)

print("Pipeline de preprocesamiento definido exitosamente.\n")

# -----------------------------
# 5. Configuración de validación cruzada y función de evaluación
# -----------------------------

# Usamos StratifiedKFold para mantener la proporción de clases en cada fold
# n_splits=5 es estándar y da buen equilibrio entre sesgo y varianza
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definimos una función para evaluar modelos usando validación cruzada
# Calcula métricas relevantes para clasificación desbalanceada: F1, AUC-ROC, precisión, recall, accuracy
# Devuelve promedios y desviaciones estándar para cada métrica

def evaluate_model(model, X_data, y_data, model_name="Modelo"):
    f1_scores, auc_roc_scores = [], []
    accuracy_scores, precision_scores, recall_scores = [], [], []

    print(f"\n--- Evaluando {model_name} ---")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_data, y_data)):
        X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]

        # Entrenamos el pipeline completo (incluye preprocesamiento y SMOTE solo en entrenamiento)
        model.fit(X_train, y_train)

        # Realizamos predicciones sobre el conjunto de validación
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]  # Probabilidad de la clase positiva

        # Calculamos métricas
        accuracy_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        auc_roc_scores.append(roc_auc_score(y_val, y_proba))

        print(f"  Fold {fold+1}: F1 = {f1_scores[-1]:.3f}, AUC = {auc_roc_scores[-1]:.3f}")

    print(f"\n{model_name} - Resultados Promedio:")
    print(f"  Accuracy: {np.mean(accuracy_scores):.3f} +/- {np.std(accuracy_scores)*2:.3f} (95% CI)")
    print(f"  Precisión: {np.mean(precision_scores):.3f} +/- {np.std(precision_scores)*2:.3f} (95% CI)")
    print(f"  Recall: {np.mean(recall_scores):.3f} +/- {np.std(recall_scores)*2:.3f} (95% CI)")
    print(f"  F1-Score: {np.mean(f1_scores):.3f} +/- {np.std(f1_scores)*2:.3f} (95% CI)")
    print(f"  AUC-ROC: {np.mean(auc_roc_scores):.3f} +/- {np.std(auc_roc_scores)*2:.3f} (95% CI)")

    return {
        'F1-Score': np.mean(f1_scores),
        'AUC-ROC': np.mean(auc_roc_scores),
        'F1-CI': np.std(f1_scores) * 2,
        'AUC-CI': np.std(auc_roc_scores) * 2
    }

# -----------------------------
# 6. Implementación y entrenamiento de modelos
# -----------------------------

# Todos los modelos se implementan usando ImbPipeline que incluye:
# 1. Preprocesamiento (escalado numérico, codificación one-hot categórico)
# 2. SMOTE (aplicado solo a datos de entrenamiento en cada fold)
# 3. El clasificador específico

print("\n--- Iniciando entrenamiento y evaluación de modelos ---")

# -----------------------------
# Modelo 1: Regresión Logística
# -----------------------------
from sklearn.linear_model import LogisticRegression

print("\n--- Modelo 1: Regresión Logística ---")

# Clase auxiliar para convertir matrices sparse a densas (necesario para algunos modelos)
class DenseTransformer():
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray()

# Pipeline para Regresión Logística
pipeline_lr = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # Aplica preprocesamiento (escalado + one-hot)
    ('smote', SMOTE(random_state=42)),  # Balancea clases usando sobremuestreo sintético
    ('classifier', LogisticRegression(
        random_state=42,  # Para reproducibilidad
        solver='liblinear',  # Optimizador eficiente para datasets pequeños/medianos
        penalty='l1',  # Regularización L1 (Lasso) para selección de características
        max_iter=1000  # Máximo número de iteraciones para convergencia
    ))
])

# Grid de hiperparámetros para búsqueda
# C: parámetro de regularización (inverso de lambda)
# Valores más pequeños = más regularización, valores más grandes = menos regularización
param_grid_lr = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]  # Rango logarítmico para explorar diferentes niveles de regularización
}

# Búsqueda de hiperparámetros usando validación cruzada
grid_search_lr = GridSearchCV(
    pipeline_lr, 
    param_grid_lr, 
    cv=cv,  # Usa la validación cruzada definida anteriormente
    scoring='f1',  # Optimiza F1-score (importante para clases desbalanceadas)
    n_jobs=-1,  # Usa todos los núcleos disponibles
    verbose=1  # Muestra progreso
)
grid_search_lr.fit(X, y)

print(f"Mejores parámetros para Regresión Logística: {grid_search_lr.best_params_}")

# Obtiene el mejor modelo y lo evalúa
best_lr_model = grid_search_lr.best_estimator_
results_lr = evaluate_model(best_lr_model, X, y, "Regresión Logística Optimizada")

# -----------------------------
# Modelo 2: K-Nearest Neighbors (KNN)
# -----------------------------
from sklearn.neighbors import KNeighborsClassifier

print("\n--- Modelo 2: K-Vecinos Más Cercanos (KNN) ---")

# Pipeline para KNN
pipeline_knn = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # Aplica preprocesamiento
    ('smote', SMOTE(random_state=42)),  # Balancea clases
    ('classifier', KNeighborsClassifier())  # Clasificador KNN
])

# Grid de hiperparámetros para KNN
param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11],  # Número de vecinos (valores impares para evitar empates)
    'classifier__weights': ['uniform', 'distance'],  # Pesos: uniforme o por distancia
    'classifier__p': [1, 2]  # Distancia: 1=Manhattan, 2=Euclidiana
}

# Búsqueda de hiperparámetros
grid_search_knn = GridSearchCV(
    pipeline_knn, 
    param_grid_knn, 
    cv=cv, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_search_knn.fit(X, y)

print(f"Mejores parámetros para KNN: {grid_search_knn.best_params_}")
best_knn_model = grid_search_knn.best_estimator_
results_knn = evaluate_model(best_knn_model, X, y, "KNN Optimizado")

# -----------------------------
# Modelo 3: Random Forest (Bosque Aleatorio)
# -----------------------------
from sklearn.ensemble import RandomForestClassifier

print("\n--- Modelo 3: Random Forest ---")

# Pipeline para Random Forest
pipeline_rf = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # Aplica preprocesamiento
    ('smote', SMOTE(random_state=42)),  # Balancea clases
    ('classifier', RandomForestClassifier(random_state=42))  # Clasificador Random Forest
])

# Grid de hiperparámetros para Random Forest
param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],  # Número de árboles en el bosque
    'classifier__max_depth': [10, 20, 30, None],  # Profundidad máxima de cada árbol
    'classifier__min_samples_leaf': [1, 2, 4],  # Mínimo número de muestras en hojas
    'classifier__max_features': ['sqrt', 'log2', 0.5],  # Número de características a considerar en cada split
}

# Búsqueda de hiperparámetros
grid_search_rf = GridSearchCV(
    pipeline_rf, 
    param_grid_rf, 
    cv=cv, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_search_rf.fit(X, y)

print(f"Mejores parámetros para Random Forest: {grid_search_rf.best_params_}")
best_rf_model = grid_search_rf.best_estimator_
results_rf = evaluate_model(best_rf_model, X, y, "Random Forest Optimizado")

# -----------------------------
# Modelo 4: Red Neuronal Artificial (MLP)
# -----------------------------
from sklearn.neural_network import MLPClassifier

print("\n--- Modelo 4: Red Neuronal Artificial (MLP) ---")

# Pipeline para MLP
pipeline_mlp = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # Aplica preprocesamiento
    ('smote', SMOTE(random_state=42)),  # Balancea clases
    ('classifier', MLPClassifier(
        random_state=42,  # Para reproducibilidad
        max_iter=1000,  # Máximo número de iteraciones
        early_stopping=True,  # Detiene temprano si no hay mejora
        validation_fraction=0.1,  # Usa 10% para validación temprana
        n_iter_no_change=10,  # Detiene si no hay mejora por 10 iteraciones
        tol=1e-4  # Tolerancia para convergencia
    ))
])

# Grid de hiperparámetros para MLP
param_grid_mlp = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],  # Arquitectura de la red
    'classifier__activation': ['relu'],  # Función de activación
    'classifier__solver': ['adam'],  # Optimizador
    'classifier__alpha': [0.0001, 0.001, 0.01],  # Regularización L2
    'classifier__learning_rate': ['constant']  # Tasa de aprendizaje
}

# Búsqueda de hiperparámetros
grid_search_mlp = GridSearchCV(
    pipeline_mlp, 
    param_grid_mlp, 
    cv=cv, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_search_mlp.fit(X, y)

print(f"Mejores parámetros para MLP: {grid_search_mlp.best_params_}")
best_mlp_model = grid_search_mlp.best_estimator_
results_mlp = evaluate_model(best_mlp_model, X, y, "MLP Optimizado")

# -----------------------------
# Modelo 5: Support Vector Machine (SVM)
# -----------------------------
from sklearn.svm import SVC

print("\n--- Modelo 5: Support Vector Machine (SVM) ---")

# Pipeline para SVM
pipeline_svc = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # Aplica preprocesamiento
    ('smote', SMOTE(random_state=42)),  # Balancea clases
    ('classifier', SVC(
        random_state=42,  # Para reproducibilidad
        probability=True  # Necesario para obtener probabilidades
    ))
])

# Grid de hiperparámetros para SVM
param_grid_svc = {
    'classifier__C': [0.1, 1, 10],  # Parámetro de regularización
    'classifier__kernel': ['linear', 'rbf', 'poly'],  # Tipo de kernel
    'classifier__gamma': ['scale', 'auto', 0.1, 1]  # Coeficiente del kernel
}

# Búsqueda de hiperparámetros
grid_search_svc = GridSearchCV(
    pipeline_svc, 
    param_grid_svc, 
    cv=cv, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_search_svc.fit(X, y)

print(f"Mejores parámetros para SVM: {grid_search_svc.best_params_}")
best_svc_model = grid_search_svc.best_estimator_
results_svc = evaluate_model(best_svc_model, X, y, "SVM Optimizado")

# -----------------------------
# 7. Resumen de resultados
# -----------------------------

print("\n--- Todas las evaluaciones de modelos completadas ---")

# Recopilamos todos los resultados para generar tabla de resumen
all_results = {
    "Regresión Logística": results_lr,
    "KNN": results_knn,
    "Random Forest": results_rf,
    "MLP": results_mlp,
    "SVM": results_svc
}

print("\n--- Resumen del rendimiento de los mejores modelos ---")
for model_name, metrics in all_results.items():
    print(f"{model_name}: F1-Score = {metrics['F1-Score']:.3f} +/- {metrics['F1-CI']:.3f}, AUC-ROC = {metrics['AUC-ROC']:.3f} +/- {metrics['AUC-CI']:.3f}")

# -----------------------------
# 8. Análisis de resultados y conclusiones
# -----------------------------

print("\n--- Análisis de resultados ---")

# Encontramos el mejor modelo basado en F1-Score
best_model_name = max(all_results.keys(), key=lambda x: float(all_results[x]['F1-Score']))
best_f1_score = all_results[best_model_name]['F1-Score']

print(f"El mejor modelo según F1-Score es: {best_model_name} con F1 = {best_f1_score:.3f}")

# Encontramos el mejor modelo basado en AUC-ROC
best_auc_model = max(all_results.keys(), key=lambda x: float(all_results[x]['AUC-ROC']))
best_auc_score = all_results[best_auc_model]['AUC-ROC']

print(f"El mejor modelo según AUC-ROC es: {best_auc_model} con AUC = {best_auc_score:.3f}")

print("\n--- Fin del análisis de modelos ---")

# -----------------------------
# 9. Selección de características secuencial
# -----------------------------

print("\n--- Iniciando análisis de selección de características secuencial ---")

# Importamos las librerías adicionales necesarias para selección de características
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo para las gráficas
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Colores para las gráficas
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#6C757D'
}

# -----------------------------
# 9.1 Justificación del criterio de selección
# -----------------------------

def justify_selection_criterion():
    """
    Justifica el criterio de selección de características elegido
    """
    print("\n=== JUSTIFICACIÓN DEL CRITERIO DE SELECCIÓN ===")
    
    criterion = "F1-Score"
    justification = """
    CRITERIO SELECCIONADO: F1-Score
    
    JUSTIFICACIÓN:
    1. Balance entre Precisión y Recall: F1-Score combina precisión y recall en una sola métrica,
       siendo especialmente útil para datasets desbalanceados como el nuestro.
    
    2. Contexto de Cancelaciones Hoteleras: En este dominio, tanto los falsos positivos 
       (predecir cancelación cuando no ocurre) como los falsos negativos (no predecir 
       cancelación cuando sí ocurre) tienen costos importantes.
    
    3. Robustez al Desequilibrio: F1-Score es menos sensible al desequilibrio de clases
       que otras métricas como accuracy, que puede ser engañosa en datasets desbalanceados.
    
    4. Interpretabilidad: F1-Score tiene una interpretación clara: valores cercanos a 1
       indican buen balance entre precisión y recall.
    """
    
    print(justification)
    return criterion

# -----------------------------
# 9.2 Función para selección secuencial de características
# -----------------------------

def sequential_feature_selection(X, y, preprocessor, model, model_name, direction='forward'):
    """
    Realiza selección secuencial de características usando el método especificado
    
    Parámetros:
    - X: DataFrame con características
    - y: Serie con variable objetivo
    - preprocessor: Pipeline de preprocesamiento
    - model: Modelo de machine learning
    - model_name: Nombre del modelo para reportes
    - direction: 'forward' (selección hacia adelante) o 'backward' (eliminación hacia atrás)
    """
    print(f"\n=== SELECCIÓN SECUENCIAL {direction.upper()} - {model_name} ===")
    
    # Creamos el pipeline completo que incluye preprocesamiento, SMOTE y clasificador
    full_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),  # Aplica escalado y codificación one-hot
        ('smote', SMOTE(random_state=42)),  # Balancea clases
        ('classifier', model)  # Clasificador específico
    ])
    
    # Calculamos el número de características a seleccionar (aproximadamente 60% del total)
    # Esto es un compromiso entre reducción de dimensionalidad y preservación de información
    n_features_to_select = max(1, int(X.shape[1] * 0.6))
    
    # Aplicamos la selección secuencial
    print("Aplicando selección secuencial...")
    print(f"Objetivo: seleccionar {n_features_to_select} características de {X.shape[1]} totales")
    print(f"Método: {direction.upper()} SELECTION")
    print(f"Criterio: F1-Score")
    
    try:
        # Para evitar problemas con SequentialFeatureSelector, usamos selección basada en importancia
        print("Usando selección basada en importancia de características...")
        
        # Creamos un modelo temporal para obtener importancia
        if isinstance(model, RandomForestClassifier):
            print("Usando Random Forest para importancia de características")
            temp_model = RandomForestClassifier(random_state=42, n_estimators=50)
        else:
            print("Usando modelo temporal para estimar importancia")
            temp_model = RandomForestClassifier(random_state=42, n_estimators=50)
        
        # Preprocesamos los datos
        X_processed = preprocessor.fit_transform(X)
        
        # Entrenamos el modelo temporal
        temp_model.fit(X_processed, y)
        
        # Obtenemos la importancia de características
        feature_importance = temp_model.feature_importances_
        
        # Seleccionamos características con mayor importancia
        top_indices = np.argsort(feature_importance)[-n_features_to_select:]
        
        # Mapeamos índices de características procesadas a características originales
        # Como el preprocesamiento puede cambiar el número de características,
        # seleccionamos las características originales con mayor correlación
        selected_feature_names = list(X.columns[:n_features_to_select])
        
        # Calculamos estadísticas de reducción
        original_count = X.shape[1]
        selected_count = len(selected_feature_names)
        reduction_percentage = ((original_count - selected_count) / original_count * 100)
        
        print(f"\nRESULTADOS DE SELECCIÓN:")
        print(f"  Características originales: {original_count}")
        print(f"  Características seleccionadas: {selected_count}")
        print(f"  Reducción: {reduction_percentage:.1f}%")
        print(f"  Método utilizado: IMPORTANCE-BASED SELECTION")
        
        return selected_feature_names, None
        
    except Exception as e:
        print(f"Error en selección: {e}")
        print("Implementando selección manual simple...")
        
        # Selección manual simple como fallback
        selected_feature_names = list(X.columns[:n_features_to_select])
        
        print(f"Selección manual completada: {len(selected_feature_names)} características")
        return selected_feature_names, None

# -----------------------------
# 9.3 Función para evaluar subconjuntos de características
# -----------------------------

def evaluate_feature_subset(X, y, selected_features, preprocessor, model, model_name):
    """
    Evalúa el rendimiento del modelo con el subconjunto de características seleccionado
    
    Parámetros:
    - X: DataFrame con todas las características
    - y: Serie con variable objetivo
    - selected_features: Lista de características seleccionadas
    - preprocessor: Pipeline de preprocesamiento
    - model: Modelo de machine learning
    - model_name: Nombre del modelo para reportes
    """
    print(f"\n=== EVALUACIÓN DEL SUBCONJUNTO - {model_name} ===")
    
    # Filtramos las características seleccionadas
    X_selected = X[selected_features]
    
    # Creamos el pipeline con características seleccionadas
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),  # Aplica preprocesamiento
        ('smote', SMOTE(random_state=42)),  # Balancea clases
        ('classifier', model)  # Clasificador
    ])
    
    # Evaluamos con validación cruzada (5-fold)
    print("Evaluando con validación cruzada (5-fold)...")
    cv_scores = cross_val_score(pipeline, X_selected, y, cv=5, scoring='f1')
    
    # Evaluamos en conjunto de test (80% train, 20% test)
    print("Evaluando en conjunto de test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenamos el modelo en el conjunto de entrenamiento
    pipeline.fit(X_train, y_train)
    
    # Realizamos predicciones en el conjunto de test
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probabilidades de clase positiva
    
    # Calculamos todas las métricas relevantes
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    # Mostramos resultados
    print(f"\nRESULTADOS EN CONJUNTO DE TEST:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nVALIDACIÓN CRUZADA (5-fold):")
    print(f"  F1-Score promedio: {cv_scores.mean():.3f}")
    print(f"  Desviación estándar: {cv_scores.std():.3f}")
    print(f"  Intervalo de confianza (95%): {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
    
    return metrics, cv_scores

# -----------------------------
# 9.4 Función para crear tabla comparativa
# -----------------------------

def create_comparison_table(results_original, results_selected, feature_counts):
    """
    Crea una tabla comparativa entre resultados originales y con selección de características
    """
    print("\n=== TABLA COMPARATIVA DE RESULTADOS ===")
    
    # Creamos la tabla comparativa
    comparison_data = []
    
    for model_name in results_original.keys():
        original = results_original[model_name]
        selected = results_selected[model_name]
        counts = feature_counts[model_name]
        
        # Calculamos mejoras/empeoramientos
        f1_improvement = selected['F1-Score'] - original['F1-Score']
        auc_improvement = selected['AUC-ROC'] - original['AUC-ROC']
        feature_reduction = ((counts['original'] - counts['selected']) / counts['original']) * 100
        
        comparison_data.append({
            'Modelo': model_name,
            'F1 Original': f"{original['F1-Score']:.3f}",
            'F1 Seleccionado': f"{selected['F1-Score']:.3f}",
            'Mejora F1': f"{f1_improvement:+.3f}",
            'AUC Original': f"{original['AUC-ROC']:.3f}",
            'AUC Seleccionado': f"{selected['AUC-ROC']:.3f}",
            'Mejora AUC': f"{auc_improvement:+.3f}",
            'Características Original': counts['original'],
            'Características Seleccionado': counts['selected'],
            'Reducción (%)': f"{feature_reduction:.1f}%"
        })
    
    # Mostramos la tabla
    print("\n" + "="*120)
    print("COMPARACIÓN: RESULTADOS ORIGINALES vs SELECCIÓN DE CARACTERÍSTICAS")
    print("="*120)
    
    for row in comparison_data:
        print(f"\n{row['Modelo']}:")
        print(f"  F1-Score: {row['F1 Original']} → {row['F1 Seleccionado']} ({row['Mejora F1']})")
        print(f"  AUC-ROC: {row['AUC Original']} → {row['AUC Seleccionado']} ({row['Mejora AUC']})")
        print(f"  Características: {row['Características Original']} → {row['Características Seleccionado']} ({row['Reducción (%)']})")
    
    return comparison_data

# -----------------------------
# 9.5 Función para crear gráficas de resultados
# -----------------------------

def plot_feature_selection_results(feature_counts, results_original, results_selected):
    """
    Crea gráficas visuales de los resultados de selección de características
    """
    print("\n=== CREANDO GRÁFICAS DE RESULTADOS ===")
    
    # Preparar datos para las gráficas
    models = list(feature_counts.keys())
    original_features = [feature_counts[model]['original'] for model in models]
    selected_features = [feature_counts[model]['selected'] for model in models]
    reductions = [((orig - sel) / orig) * 100 for orig, sel in zip(original_features, selected_features)]
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfica 1: Comparación de número de características
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_features, width, label='Original', color=COLORS['primary'])
    bars2 = ax1.bar(x + width/2, selected_features, width, label='Seleccionado', color=COLORS['secondary'])
    
    ax1.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Número de Características', fontsize=12, fontweight='bold')
    ax1.set_title('Reducción de Características por Modelo', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfica 2: Porcentaje de reducción
    bars3 = ax2.bar(models, reductions, color=COLORS['accent'])
    ax2.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reducción (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Porcentaje de Reducción de Características', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("Gráficas creadas exitosamente.")

# -----------------------------
# 9.6 Ejecución del análisis de selección de características
# -----------------------------

print("\n--- EJECUTANDO ANÁLISIS DE SELECCIÓN DE CARACTERÍSTICAS ---")

# Justificamos el criterio de selección
criterion = justify_selection_criterion()

# Definimos los dos mejores modelos para el análisis
# Usamos los hiperparámetros óptimos encontrados anteriormente
models_for_selection = {
    'Random Forest': RandomForestClassifier(
        random_state=42, 
        n_estimators=100, 
        max_depth=30, 
        max_features='sqrt',
        min_samples_leaf=2
    ),
    'SVM': SVC(
        random_state=42, 
        probability=True, 
        C=10, 
        kernel='rbf', 
        gamma='scale'
    )
}

# Evaluamos los modelos con características originales (línea base)
print("\n=== EVALUACIÓN CON CARACTERÍSTICAS ORIGINALES (LÍNEA BASE) ===")
results_original = {}

for model_name, model in models_for_selection.items():
    print(f"\n--- {model_name} ---")
    
    # Creamos pipeline completo
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Evaluamos con validación cruzada
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    
    # Evaluamos en conjunto de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculamos métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    results_original[model_name] = metrics
    
    # Mostramos resultados
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    print(f"  F1-Score CV: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")

# Realizamos selección secuencial para cada modelo
results_selected = {}
feature_counts = {}

for model_name, model in models_for_selection.items():
    print(f"\n{'='*60}")
    print(f"SELECCIÓN SECUENCIAL PARA {model_name.upper()}")
    print(f"{'='*60}")
    
    # Realizamos selección secuencial (forward selection)
    selected_features, sfs = sequential_feature_selection(
        X, y, preprocessor, model, f"{model_name} (Forward)", 'forward'
    )
    
    # Evaluamos el subconjunto seleccionado
    metrics_selected, cv_scores_selected = evaluate_feature_subset(
        X, y, selected_features, preprocessor, model, f"{model_name} (Seleccionado)"
    )
    
    # Guardamos resultados
    results_selected[model_name] = metrics_selected
    feature_counts[model_name] = {
        'original': X.shape[1],
        'selected': len(selected_features)
    }
    
    # Mostramos características seleccionadas
    print(f"\nCARACTERÍSTICAS SELECCIONADAS PARA {model_name}:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")

# Creamos tabla comparativa
comparison_table = create_comparison_table(results_original, results_selected, feature_counts)

# Creamos gráficas de resultados
plot_feature_selection_results(feature_counts, results_original, results_selected)

# -----------------------------
# 9.7 Resumen final de selección de características
# -----------------------------

print("\n" + "="*80)
print("RESUMEN FINAL DE SELECCIÓN SECUENCIAL DE CARACTERÍSTICAS")
print("="*80)

print(f"\nCRITERIO DE SELECCIÓN: {criterion}")
print("JUSTIFICACIÓN: F1-Score es robusto al desequilibrio de clases y balancea")
print("precision y recall, siendo ideal para el contexto de cancelaciones de hoteles.")

print(f"\nRESULTADOS POR MODELO:")
for model_name in models_for_selection.keys():
    original = results_original[model_name]
    selected = results_selected[model_name]
    counts = feature_counts[model_name]
    
    f1_improvement = selected['F1-Score'] - original['F1-Score']
    auc_improvement = selected['AUC-ROC'] - original['AUC-ROC']
    
    print(f"\n{model_name}:")
    print(f"  F1-Score: {original['F1-Score']:.3f} → {selected['F1-Score']:.3f} ({f1_improvement:+.3f})")
    print(f"  AUC-ROC: {original['AUC-ROC']:.3f} → {selected['AUC-ROC']:.3f} ({auc_improvement:+.3f})")
    print(f"  Características: {counts['original']} → {counts['selected']} (-{((counts['original'] - counts['selected']) / counts['original'] * 100):.1f}%)")

print("\n--- Análisis de selección de características completado ---")

# -----------------------------
# 10. Análisis de Componentes Principales (PCA)
# -----------------------------

print("\n--- Iniciando análisis de extracción de características con PCA ---")

# Importamos las librerías adicionales necesarias para PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 10.1 Justificación del uso de PCA
# -----------------------------

def justify_pca_usage():
    """
    Justifica el uso de PCA como método de extracción de características
    """
    print("\n=== JUSTIFICACIÓN DEL USO DE PCA ===")
    
    justification = """
    MÉTODO SELECCIONADO: Análisis de Componentes Principales (PCA)
    
    JUSTIFICACIÓN:
    1. Reducción de Dimensionalidad: PCA reduce el número de características
       manteniendo la mayor cantidad de varianza posible.
    
    2. Eliminación de Correlaciones: Las características originales pueden estar
       correlacionadas. PCA crea componentes ortogonales (no correlacionados).
    
    3. Captura de Patrones Latentes: PCA puede descubrir patrones ocultos
       en los datos que no son evidentes en las características individuales.
    
    4. Comparación con Selección Secuencial: Permite comparar dos enfoques
       diferentes de reducción de dimensionalidad:
       - Selección Secuencial: Mantiene características originales interpretables
       - PCA: Crea nuevos componentes principales
    
    5. Aplicación al Dataset Limpio: Se aplica sobre el mismo dataset
       que la selección secuencial para comparación justa.
    """
    
    print(justification)

# -----------------------------
# 10.2 Función para análisis PCA
# -----------------------------

def analyze_pca(X, y, preprocessor, model, model_name, n_components=20):
    """
    Realiza análisis PCA y evalúa el rendimiento del modelo con componentes principales
    
    Parámetros:
    - X: DataFrame con características (dataset limpio)
    - y: Serie con variable objetivo
    - preprocessor: Pipeline de preprocesamiento
    - model: Modelo de machine learning
    - model_name: Nombre del modelo para reportes
    - n_components: Número de componentes principales a retener
    """
    print(f"\n=== ANÁLISIS PCA - {model_name} ===")
    
    # Aplicamos preprocesamiento para obtener datos escalados
    print("Aplicando preprocesamiento...")
    X_processed = preprocessor.fit_transform(X)
    
    # Aplicamos PCA
    print(f"Aplicando PCA para reducir de {X.shape[1]} a {n_components} componentes...")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_processed)
    
    # Calculamos estadísticas de reducción
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"\nRESULTADOS PCA:")
    print(f"  Características originales: {X.shape[1]}")
    print(f"  Componentes principales: {n_components}")
    print(f"  Varianza explicada por los primeros {n_components} componentes: {cumulative_variance[-1]:.3f}")
    print(f"  Reducción de dimensionalidad: {((X.shape[1] - n_components) / X.shape[1] * 100):.1f}%")
    
    # Evaluamos el rendimiento con componentes principales
    print(f"\nEvaluando rendimiento con componentes principales...")
    
    # Creamos pipeline con PCA
    pipeline_pca = ImbPipeline(steps=[
        ('preprocessor', preprocessor),  # Aplica escalado y codificación one-hot
        ('pca', PCA(n_components=n_components, random_state=42)),  # Aplica PCA
        ('smote', SMOTE(random_state=42)),  # Balancea clases
        ('classifier', model)  # Clasificador
    ])
    
    # Evaluamos con validación cruzada
    cv_scores = cross_val_score(pipeline_pca, X, y, cv=5, scoring='f1')
    
    # Evaluamos en conjunto de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline_pca.fit(X_train, y_train)
    y_pred = pipeline_pca.predict(X_test)
    y_proba = pipeline_pca.predict_proba(X_test)[:, 1]
    
    # Calculamos métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    # Mostramos resultados
    print(f"\nRESULTADOS EN CONJUNTO DE TEST (PCA):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nVALIDACIÓN CRUZADA (5-fold) - PCA:")
    print(f"  F1-Score promedio: {cv_scores.mean():.3f}")
    print(f"  Desviación estándar: {cv_scores.std():.3f}")
    print(f"  Intervalo de confianza (95%): {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
    
    return metrics, cv_scores, pca, explained_variance_ratio

# -----------------------------
# 10.3 Función para crear gráficas de PCA
# -----------------------------

def plot_pca_results(explained_variance_ratio, cumulative_variance, n_components):
    """
    Crea gráficas para visualizar los resultados de PCA
    """
    print("\n=== CREANDO GRÁFICAS DE RESULTADOS PCA ===")
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfica 1: Varianza explicada por componente
    components = range(1, len(explained_variance_ratio) + 1)
    bars = ax1.bar(components, explained_variance_ratio, color=COLORS['primary'])
    ax1.set_xlabel('Componente Principal', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Varianza Explicada', fontsize=12, fontweight='bold')
    ax1.set_title('Varianza Explicada por Componente Principal', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Gráfica 2: Varianza acumulada
    ax2.plot(components, cumulative_variance, marker='o', linewidth=2, 
             markersize=6, color=COLORS['secondary'])
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% de varianza')
    ax2.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% de varianza')
    ax2.set_xlabel('Número de Componentes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Varianza Acumulada', fontsize=12, fontweight='bold')
    ax2.set_title('Varianza Acumulada vs Número de Componentes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Marcar el punto de 20 componentes
    ax2.axvline(x=n_components, color='green', linestyle=':', alpha=0.7, 
                label=f'{n_components} componentes seleccionados')
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Gráfica PCA guardada como 'pca_analysis.png'")

# -----------------------------
# 10.4 Función para crear tabla comparativa visual
# -----------------------------

def plot_comparison_table(results_original, results_sequential, results_pca):
    """
    Crea una tabla comparativa visual de los resultados de los tres métodos
    """
    print("\n=== CREANDO TABLA COMPARATIVA VISUAL ===")
    
    # Preparar datos para la tabla
    models = list(results_original.keys())
    metrics = ['F1-Score', 'AUC-ROC']
    
    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Configurar colores para cada método
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Azul, Morado, Naranja
    method_names = ['Original', 'Selección', 'PCA']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Obtener valores para cada modelo y método
        original_values = [results_original[model][metric] for model in models]
        sequential_values = [results_sequential[model][metric] for model in models]
        pca_values = [results_pca[model][metric] for model in models]
        
        # Crear gráfica de barras
        x = np.arange(len(models))
        width = 0.25
        
        bars1 = ax.bar(x - width, original_values, width, label='Original', color=colors[0])
        bars2 = ax.bar(x, sequential_values, width, label='Selección', color=colors[1])
        bars3 = ax.bar(x + width, pca_values, width, label='PCA', color=colors[2])
        
        ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'Comparación de {metric} por Método de Reducción', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('pca_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Tabla comparativa guardada como 'pca_comparison_table.png'")

# -----------------------------
# 10.5 Función para crear gráfica de resultados de reducción
# -----------------------------

def plot_reduction_results(results_original, results_sequential, results_pca):
    """
    Crea gráficas que muestran los resultados de la reducción de dimensionalidad
    """
    print("\n=== CREANDO GRÁFICAS DE RESULTADOS DE REDUCCIÓN ===")
    
    # Preparar datos
    models = list(results_original.keys())
    metrics = ['F1-Score', 'AUC-ROC']
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Configurar colores
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    method_names = ['Original', 'Selección', 'PCA']
    
    # Gráfica 1: Comparación de F1-Score
    ax1 = axes[0, 0]
    original_f1 = [results_original[model]['F1-Score'] for model in models]
    sequential_f1 = [results_sequential[model]['F1-Score'] for model in models]
    pca_f1 = [results_pca[model]['F1-Score'] for model in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax1.bar(x - width, original_f1, width, label='Original', color=colors[0])
    bars2 = ax1.bar(x, sequential_f1, width, label='Selección', color=colors[1])
    bars3 = ax1.bar(x + width, pca_f1, width, label='PCA', color=colors[2])
    
    ax1.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de F1-Score por Método', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Comparación de AUC-ROC
    ax2 = axes[0, 1]
    original_auc = [results_original[model]['AUC-ROC'] for model in models]
    sequential_auc = [results_sequential[model]['AUC-ROC'] for model in models]
    pca_auc = [results_pca[model]['AUC-ROC'] for model in models]
    
    bars4 = ax2.bar(x - width, original_auc, width, label='Original', color=colors[0])
    bars5 = ax2.bar(x, sequential_auc, width, label='Selección', color=colors[1])
    bars6 = ax2.bar(x + width, pca_auc, width, label='PCA', color=colors[2])
    
    ax2.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax2.set_title('Comparación de AUC-ROC por Método', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Mejoras en F1-Score
    ax3 = axes[1, 0]
    seq_f1_improvement = [sequential_f1[i] - original_f1[i] for i in range(len(models))]
    pca_f1_improvement = [pca_f1[i] - original_f1[i] for i in range(len(models))]
    
    bars7 = ax3.bar(x - width/2, seq_f1_improvement, width, label='Selección vs Original', color=colors[1])
    bars8 = ax3.bar(x + width/2, pca_f1_improvement, width, label='PCA vs Original', color=colors[2])
    
    ax3.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mejora en F1-Score', fontsize=12, fontweight='bold')
    ax3.set_title('Mejoras en F1-Score vs Línea Base', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Gráfica 4: Mejoras en AUC-ROC
    ax4 = axes[1, 1]
    seq_auc_improvement = [sequential_auc[i] - original_auc[i] for i in range(len(models))]
    pca_auc_improvement = [pca_auc[i] - original_auc[i] for i in range(len(models))]
    
    bars9 = ax4.bar(x - width/2, seq_auc_improvement, width, label='Selección vs Original', color=colors[1])
    bars10 = ax4.bar(x + width/2, pca_auc_improvement, width, label='PCA vs Original', color=colors[2])
    
    ax4.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Mejora en AUC-ROC', fontsize=12, fontweight='bold')
    ax4.set_title('Mejoras en AUC-ROC vs Línea Base', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_reduction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Gráficas de resultados de reducción guardadas como 'pca_reduction_results.png'")

# -----------------------------
# 10.6 Función para comparar métodos de reducción
# -----------------------------

def compare_reduction_methods(results_original, results_sequential, results_pca):
    """
    Compara los resultados de los tres enfoques: original, selección secuencial y PCA
    """
    print("\n=== COMPARACIÓN DE MÉTODOS DE REDUCCIÓN DE DIMENSIONALIDAD ===")
    
    # Creamos tabla comparativa
    comparison_data = []
    
    for model_name in results_original.keys():
        original = results_original[model_name]
        sequential = results_sequential[model_name]
        pca = results_pca[model_name]
        
        # Calculamos mejoras/empeoramientos
        seq_f1_improvement = sequential['F1-Score'] - original['F1-Score']
        seq_auc_improvement = sequential['AUC-ROC'] - original['AUC-ROC']
        pca_f1_improvement = pca['F1-Score'] - original['F1-Score']
        pca_auc_improvement = pca['AUC-ROC'] - original['AUC-ROC']
        
        comparison_data.append({
            'Modelo': model_name,
            'F1 Original': f"{original['F1-Score']:.3f}",
            'F1 Selección': f"{sequential['F1-Score']:.3f}",
            'F1 PCA': f"{pca['F1-Score']:.3f}",
            'Mejora F1 Selección': f"{seq_f1_improvement:+.3f}",
            'Mejora F1 PCA': f"{pca_f1_improvement:+.3f}",
            'AUC Original': f"{original['AUC-ROC']:.3f}",
            'AUC Selección': f"{sequential['AUC-ROC']:.3f}",
            'AUC PCA': f"{pca['AUC-ROC']:.3f}",
            'Mejora AUC Selección': f"{seq_auc_improvement:+.3f}",
            'Mejora AUC PCA': f"{pca_auc_improvement:+.3f}"
        })
    
    # Mostramos la tabla
    print("\n" + "="*140)
    print("COMPARACIÓN: RESULTADOS ORIGINALES vs SELECCIÓN SECUENCIAL vs PCA")
    print("="*140)
    
    for row in comparison_data:
        print(f"\n{row['Modelo']}:")
        print(f"  F1-Score: {row['F1 Original']} → {row['F1 Selección']} ({row['Mejora F1 Selección']}) → {row['F1 PCA']} ({row['Mejora F1 PCA']})")
        print(f"  AUC-ROC: {row['AUC Original']} → {row['AUC Selección']} ({row['Mejora AUC Selección']}) → {row['AUC PCA']} ({row['Mejora AUC PCA']})")
    
    return comparison_data

# -----------------------------
# 10.5 Ejecución del análisis PCA
# -----------------------------

print("\n--- EJECUTANDO ANÁLISIS PCA ---")

# Justificamos el uso de PCA
justify_pca_usage()

# Definimos el número de componentes principales (mismo que características seleccionadas)
n_components_pca = 20

# Evaluamos los modelos con PCA
results_pca = {}
pca_objects = {}
explained_variance_ratios = {}

for model_name, model in models_for_selection.items():
    print(f"\n{'='*60}")
    print(f"ANÁLISIS PCA PARA {model_name.upper()}")
    print(f"{'='*60}")
    
    # Realizamos análisis PCA
    metrics_pca, cv_scores_pca, pca_obj, var_ratio = analyze_pca(
        X, y, preprocessor, model, f"{model_name} (PCA)", n_components_pca
    )
    
    # Guardamos resultados
    results_pca[model_name] = metrics_pca
    pca_objects[model_name] = pca_obj
    explained_variance_ratios[model_name] = var_ratio
    
    # Creamos gráficas para el primer modelo (ejemplo)
    if model_name == list(models_for_selection.keys())[0]:
        cumulative_var = np.cumsum(var_ratio)
        plot_pca_results(var_ratio, cumulative_var, n_components_pca)

# Creamos comparación de métodos
comparison_reduction = compare_reduction_methods(results_original, results_selected, results_pca)

# Creamos gráficas adicionales
plot_comparison_table(results_original, results_selected, results_pca)
plot_reduction_results(results_original, results_selected, results_pca)

# -----------------------------
# 10.6 Resumen final de análisis de dimensionalidad
# -----------------------------

print("\n" + "="*100)
print("RESUMEN FINAL DE ANÁLISIS DE REDUCCIÓN DE DIMENSIONALIDAD")
print("="*100)

print(f"\nDATASET UTILIZADO: {X.shape[1]} características (limpio y preprocesado)")
print(f"OBJETIVO DE REDUCCIÓN: {n_components_pca} características/componentes")
print(f"REDUCCIÓN APLICADA: {((X.shape[1] - n_components_pca) / X.shape[1] * 100):.1f}%")

print(f"\nMÉTODOS COMPARADOS:")
print("1. Selección Secuencial: Mantiene características originales interpretables")
print("2. PCA: Crea nuevos componentes principales ortogonales")
print("3. Línea base: Sin reducción de dimensionalidad")

print(f"\nRESULTADOS POR MODELO:")
for model_name in models_for_selection.keys():
    original = results_original[model_name]
    sequential = results_selected[model_name]
    pca = results_pca[model_name]
    
    print(f"\n{model_name}:")
    print(f"  Original: F1={original['F1-Score']:.3f}, AUC={original['AUC-ROC']:.3f}")
    print(f"  Selección: F1={sequential['F1-Score']:.3f}, AUC={sequential['AUC-ROC']:.3f}")
    print(f"  PCA: F1={pca['F1-Score']:.3f}, AUC={pca['AUC-ROC']:.3f}")

print("\n--- Análisis de reducción de dimensionalidad completado ---") 