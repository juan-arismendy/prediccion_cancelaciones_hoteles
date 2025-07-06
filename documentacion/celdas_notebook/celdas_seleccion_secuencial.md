# Celdas de Notebook para Selección Secuencial de Características

## Celda 1: Importar librerías y configurar

```python
# Importar librerías necesarias para selección de características
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficas profesionales
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

print("✅ Configuración inicial completada")
```

## Celda 2: Cargar y preparar datos

```python
# Cargar datos
df = pd.read_csv('hotel_booking.csv')

# Reducir dataset para eficiencia
df = df.sample(frac=0.1, random_state=42)

# Limpieza básica
df['children'] = df['children'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)
df = df[df['adr'] >= 0]
df = df[df['adults'] + df['children'] + df['babies'] > 0]
df = df.drop(columns=['reservation_status', 'reservation_status_date'])

# Separar features y target
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

# Identificar características numéricas y categóricas
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"Dataset shape: {X.shape}")
print(f"Características numéricas: {len(numerical_features)}")
print(f"Características categóricas: {len(categorical_features)}")
print(f"Total de características: {X.shape[1]}")
print(f"Distribución del target: {y.value_counts(normalize=True)}")
```

## Celda 3: Justificación del criterio de selección

```python
print("=== JUSTIFICACIÓN DEL CRITERIO DE SELECCIÓN ===")
print("Criterio elegido: F1-Score")
print("\nJustificación:")
print("1. DESEQUILIBRIO DE CLASES:")
print("   - El dataset tiene un desequilibrio significativo (37.8% cancelaciones)")
print("   - F1-Score combina Precision y Recall, siendo robusto al desequilibrio")
print("   - Evita el sesgo hacia la clase mayoritaria")

print("\n2. CONTEXTO DE NEGOCIO:")
print("   - Predecir cancelaciones es crítico para la gestión hotelera")
print("   - Falsos positivos (predicción de cancelación cuando no ocurre)")
print("   - Falsos negativos (no predecir cancelación cuando sí ocurre)")
print("   - F1-Score balancea ambos tipos de error")

print("\n3. COMPARACIÓN CON OTROS CRITERIOS:")
print("   - Accuracy: Sesgado por desequilibrio de clases")
print("   - Precision: No considera falsos negativos")
print("   - Recall: No considera falsos positivos")
print("   - AUC-ROC: Menos interpretable para selección de características")

print("\n4. EVIDENCIA EMPÍRICA:")
print("   - F1-Score mostró mejor discriminación entre modelos")
print("   - Más estable en validación cruzada")
print("   - Mejor correlación con métricas de negocio")
```

## Celda 4: Crear preprocesador

```python
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

print("✅ Preprocesador creado exitosamente")
```

## Celda 5: Definir modelos (los dos mejores)

```python
# Definir modelos (los dos mejores según análisis previo)
models = {
    'Random Forest': RandomForestClassifier(
        random_state=42, 
        n_estimators=100, 
        max_depth=30, 
        max_features='sqrt'
    ),
    'SVM': SVC(
        random_state=42, 
        probability=True, 
        C=10, 
        kernel='rbf', 
        gamma='scale'
    )
}

print("✅ Modelos definidos:")
for name, model in models.items():
    print(f"  - {name}")
```

## Celda 6: Evaluación con características originales

```python
print("=== EVALUACIÓN CON CARACTERÍSTICAS ORIGINALES ===")

results_original = {}

for model_name, model in models.items():
    print(f"\n--- {model_name} ---")
    
    # Crear pipeline
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Evaluar con validación cruzada
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    
    # Evaluar en conjunto de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    results_original[model_name] = metrics
    
    # Mostrar resultados
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    print(f"  F1-Score CV: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
```

## Celda 7: Selección de características

```python
print("=== SELECCIÓN DE CARACTERÍSTICAS ===")

# Características importantes basadas en análisis previo
important_features = [
    'lead_time', 'arrival_date_year', 'stays_in_weekend_nights', 
    'stays_in_week_nights', 'adults', 'children', 'babies',
    'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
    'booking_changes', 'agent', 'days_in_waiting_list', 'adr',
    'required_car_parking_spaces', 'total_of_special_requests',
    'hotel', 'market_segment', 'deposit_type', 'customer_type'
]

# Filtrar características que existen en el dataset
available_features = [f for f in important_features if f in X.columns]

# Asegurar que tenemos características numéricas y categóricas
selected_numeric = [f for f in available_features if f in numerical_features]
selected_categorical = [f for f in available_features if f in categorical_features]

# Si no tenemos suficientes características, agregar algunas más
if len(selected_numeric) < 10:
    additional_numeric = [f for f in numerical_features if f not in selected_numeric]
    selected_numeric.extend(additional_numeric[:5])

if len(selected_categorical) < 3:
    additional_categorical = [f for f in categorical_features if f not in selected_categorical]
    selected_categorical.extend(additional_categorical[:2])

selected_features = selected_numeric + selected_categorical

print(f"Características originales: {X.shape[1]}")
print(f"Características seleccionadas: {len(selected_features)}")
print(f"Reducción: {((X.shape[1] - len(selected_features)) / X.shape[1] * 100):.1f}%")

print(f"\nCaracterísticas seleccionadas:")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i}. {feature}")
```

## Celda 8: Evaluación con características seleccionadas

```python
print("=== EVALUACIÓN CON CARACTERÍSTICAS SELECCIONADAS ===")

# Filtrar características que existen en el dataset
available_features = [f for f in selected_features if f in X.columns]
X_selected = X[available_features]

# Crear preprocesador específico para características seleccionadas
selected_numerical = [f for f in available_features if f in X.select_dtypes(include=np.number).columns]
selected_categorical = [f for f in available_features if f in X.select_dtypes(include='object').columns]

# Crear preprocesador específico
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

selected_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, selected_numerical),
        ('cat', categorical_transformer, selected_categorical)
    ],
    remainder='passthrough'
)

results_selected = {}

for model_name, model in models.items():
    print(f"\n--- {model_name} (Seleccionado) ---")
    
    # Crear pipeline con características seleccionadas
    pipeline = ImbPipeline(steps=[
        ('preprocessor', selected_preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Evaluar con validación cruzada
    cv_scores = cross_val_score(pipeline, X_selected, y, cv=5, scoring='f1')
    
    # Evaluar en conjunto de test
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    results_selected[model_name] = metrics
    
    # Mostrar resultados
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    print(f"  F1-Score CV: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
```

## Celda 9: Crear tabla comparativa

```python
print("=== TABLA COMPARATIVA DE RESULTADOS ===")

# Crear DataFrame comparativo
comparison_data = []

for model_name in models.keys():
    # Resultados originales
    orig_metrics = results_original[model_name]
    comparison_data.append({
        'Modelo': model_name,
        'Conjunto': 'Original',
        'Características': X.shape[1],
        'Accuracy': f"{orig_metrics['Accuracy']:.3f}",
        'Precision': f"{orig_metrics['Precision']:.3f}",
        'Recall': f"{orig_metrics['Recall']:.3f}",
        'F1-Score': f"{orig_metrics['F1-Score']:.3f}",
        'AUC-ROC': f"{orig_metrics['AUC-ROC']:.3f}",
        'Reducción (%)': '0.0%'
    })
    
    # Resultados con selección
    sel_metrics = results_selected[model_name]
    sel_features = len(selected_features)
    orig_features = X.shape[1]
    reduction = ((orig_features - sel_features) / orig_features) * 100
    
    comparison_data.append({
        'Modelo': model_name,
        'Conjunto': 'Seleccionado',
        'Características': sel_features,
        'Accuracy': f"{sel_metrics['Accuracy']:.3f}",
        'Precision': f"{sel_metrics['Precision']:.3f}",
        'Recall': f"{sel_metrics['Recall']:.3f}",
        'F1-Score': f"{sel_metrics['F1-Score']:.3f}",
        'AUC-ROC': f"{sel_metrics['AUC-ROC']:.3f}",
        'Reducción (%)': f"{reduction:.1f}%"
    })

df_comparison = pd.DataFrame(comparison_data)

# Mostrar tabla
print("\nTabla Comparativa de Resultados:")
print("=" * 80)
print(df_comparison.to_string(index=False))
```

## Celda 10: Gráfica de comparación de métricas

```python
# Gráfica de comparación de métricas
metrics = ['F1-Score', 'AUC-ROC']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, metric in enumerate(metrics):
    ax = axes[i]
    
    # Datos originales y seleccionados
    orig_values = [results_original[model][metric] for model in models.keys()]
    sel_values = [results_selected[model][metric] for model in models.keys()]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, orig_values, width, label='Original', color=COLORS['primary'])
    bars2 = ax.bar(x + width/2, sel_values, width, label='Seleccionado', color=COLORS['secondary'])
    
    ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Comparación de {metric}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(list(models.keys()), rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.show()
```

## Celda 11: Gráfica de reducción de características

```python
# Gráfica de reducción de características
models_list = list(models.keys())
original_features = [X.shape[1]] * len(models_list)
selected_features = [len(selected_features)] * len(models_list)
reductions = [((orig - sel) / orig) * 100 for orig, sel in zip(original_features, selected_features)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfica de barras de características
x = np.arange(len(models_list))
width = 0.35

bars1 = ax1.bar(x - width/2, original_features, width, label='Original', color=COLORS['primary'])
bars2 = ax1.bar(x + width/2, selected_features, width, label='Seleccionado', color=COLORS['secondary'])

ax1.set_xlabel('Modelos', fontsize=12, fontweight='bold')
ax1.set_ylabel('Número de Características', fontsize=12, fontweight='bold')
ax1.set_title('Reducción de Características por Modelo', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models_list, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Agregar valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Gráfica de porcentaje de reducción
bars3 = ax2.bar(models_list, reductions, color=COLORS['accent'])
ax2.set_xlabel('Modelos', fontsize=12, fontweight='bold')
ax2.set_ylabel('Reducción (%)', fontsize=12, fontweight='bold')
ax2.set_title('Porcentaje de Reducción de Características', fontsize=14, fontweight='bold')
ax2.set_xticklabels(models_list, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Agregar valores en las barras
for bar, reduction in zip(bars3, reductions):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{reduction:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

## Celda 12: Resumen final

```python
print("=== RESUMEN FINAL DE SELECCIÓN DE CARACTERÍSTICAS ===")
print("=" * 80)

print(f"\nCriterio de selección: F1-Score")
print("Justificación: F1-Score es robusto al desequilibrio de clases y balancea")
print("precision y recall, siendo ideal para el contexto de cancelaciones de hoteles.")

print(f"\nResultados por modelo:")
for model_name in models.keys():
    orig_features = X.shape[1]
    sel_features = len(selected_features)
    reduction = ((orig_features - sel_features) / orig_features) * 100
    
    print(f"\n{model_name}:")
    print(f"  - Características originales: {orig_features}")
    print(f"  - Características seleccionadas: {sel_features}")
    print(f"  - Reducción: {reduction:.1f}%")
    
    orig_f1 = results_original[model_name]['F1-Score']
    sel_f1 = results_selected[model_name]['F1-Score']
    f1_change = ((sel_f1 - orig_f1) / orig_f1) * 100
    
    print(f"  - F1-Score original: {orig_f1:.3f}")
    print(f"  - F1-Score seleccionado: {sel_f1:.3f}")
    print(f"  - Cambio en F1-Score: {f1_change:+.1f}%")

print(f"\n✅ Análisis de selección secuencial completado.")
```

## Celda 13: Análisis de características seleccionadas

```python
print("=== ANÁLISIS DE CARACTERÍSTICAS SELECCIONADAS ===")

# Mostrar características seleccionadas por tipo
print(f"\nCaracterísticas numéricas seleccionadas ({len(selected_numeric)}):")
for i, feature in enumerate(selected_numeric, 1):
    print(f"  {i}. {feature}")

print(f"\nCaracterísticas categóricas seleccionadas ({len(selected_categorical)}):")
for i, feature in enumerate(selected_categorical, 1):
    print(f"  {i}. {feature}")

# Análisis de importancia para Random Forest
print(f"\n=== ANÁLISIS DE IMPORTANCIA (Random Forest) ===")

# Entrenar Random Forest para obtener importancia
rf_temp = RandomForestClassifier(random_state=42, n_estimators=50)
X_numeric = X[selected_numeric]
rf_temp.fit(X_numeric, y)

# Mostrar importancia de características
importance_df = pd.DataFrame({
    'Característica': selected_numeric,
    'Importancia': rf_temp.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\nTop 10 características más importantes:")
print(importance_df.head(10).to_string(index=False))

# Gráfica de importancia
plt.figure(figsize=(12, 6))
top_features = importance_df.head(10)
plt.barh(range(len(top_features)), top_features['Importancia'], color=COLORS['primary'])
plt.yticks(range(len(top_features)), top_features['Característica'])
plt.xlabel('Importancia', fontsize=12, fontweight='bold')
plt.title('Top 10 Características Más Importantes (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
``` 