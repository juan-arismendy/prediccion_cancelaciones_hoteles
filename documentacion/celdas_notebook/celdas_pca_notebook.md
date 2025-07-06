# Celdas de Notebook para Extracción de Características con PCA

## Celda 1: Importar librerías y configurar

```python
# Importar librerías necesarias para análisis PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
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

## Celda 3: Justificación del criterio de selección PCA

```python
print("=== JUSTIFICACIÓN DEL CRITERIO DE SELECCIÓN DE COMPONENTES ===")
print("Criterio elegido: 95% de Varianza Explicada Acumulada")
print("\nJustificación:")
print("1. CONSERVACIÓN DE INFORMACIÓN:")
print("   - 95% de varianza asegura retener la mayoría de la información")
print("   - Balance entre reducción dimensional y preservación de datos")
print("   - Estándar ampliamente aceptado en la literatura")

print("\n2. REDUCCIÓN DIMENSIONAL EFECTIVA:")
print("   - Elimina redundancia manteniendo características discriminativas")
print("   - Reduce overfitting al eliminar ruido y correlaciones")
print("   - Mejora eficiencia computacional")

print("\n3. COMPARACIÓN CON OTROS CRITERIOS:")
print("   - Kaiser (eigenvalues > 1): Puede ser muy conservador")
print("   - Scree plot: Subjetivo y difícil de automatizar")
print("   - 99% varianza: Demasiado conservador, poca reducción")
print("   - 90% varianza: Podría perder información importante")

print("\n4. EVIDENCIA EMPÍRICA:")
print("   - 95% es óptimo para datasets con ruido moderado")
print("   - Mantiene capacidad predictiva en la mayoría de casos")
print("   - Permite reducción significativa sin pérdida crítica")
```

## Celda 4: Crear preprocesador eficiente

```python
# Limitar características categóricas para evitar alta dimensionalidad
limited_categorical = ['hotel', 'market_segment', 'deposit_type', 'customer_type']

# Crear transformadores
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Crear preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, limited_categorical)
    ],
    remainder='drop'  # Eliminar otras características categóricas
)

print(f"✅ Preprocesador creado exitosamente")
print(f"Características categóricas utilizadas: {limited_categorical}")
```

## Celda 5: Análisis de componentes principales

```python
# Aplicar preprocesamiento
X_preprocessed = preprocessor.fit_transform(X)
print(f"Dimensiones después del preprocesamiento: {X_preprocessed.shape}")

# Limitar el número de componentes para análisis
max_components = min(100, X_preprocessed.shape[0] - 1, X_preprocessed.shape[1])

# Aplicar PCA
pca_full = PCA(n_components=max_components, random_state=42)
X_pca_full = pca_full.fit_transform(X_preprocessed)

# Calcular varianza explicada acumulada
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Encontrar número de componentes para diferentes umbrales
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1 if np.any(cumulative_variance >= 0.99) else max_components

print(f"Dimensiones originales (después preprocesamiento): {X_preprocessed.shape[1]}")
print(f"Componentes para 90% varianza: {n_components_90}")
print(f"Componentes para 95% varianza: {n_components_95}")
print(f"Componentes para 99% varianza: {n_components_99}")

# Seleccionar número de componentes (95% de varianza)
n_components_selected = n_components_95
original_features = X_preprocessed.shape[1]
reduction_pct = ((original_features - n_components_selected) / original_features) * 100

print(f"\n🎯 Componentes seleccionados: {n_components_selected}")
print(f"🎯 Características originales: {original_features}")
print(f"🎯 Reducción: {reduction_pct:.1f}%")
print(f"🎯 Varianza explicada: {pca_full.explained_variance_ratio_[:n_components_selected].sum():.1%}")
```

## Celda 6: Visualización del análisis PCA

```python
# Crear gráfica de análisis PCA
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))

# Subplot 1: Varianza explicada por componente
components_to_show = min(30, len(pca_full.explained_variance_ratio_))
ax1.plot(range(1, components_to_show + 1), 
         pca_full.explained_variance_ratio_[:components_to_show], 'bo-', linewidth=2, markersize=4)
ax1.set_xlabel('Componente Principal')
ax1.set_ylabel('Varianza Explicada')
ax1.set_title('Varianza Explicada por Componente')
ax1.grid(True, alpha=0.3)

# Subplot 2: Varianza explicada acumulada
ax2.plot(range(1, components_to_show + 1), 
         cumulative_variance[:components_to_show], 'ro-', linewidth=2, markersize=4)
ax2.axhline(y=0.90, color='orange', linestyle='--', label='90%')
ax2.axhline(y=0.95, color='green', linestyle='--', label='95%')
ax2.axhline(y=0.99, color='red', linestyle='--', label='99%')
ax2.axvline(x=n_components_95, color='green', linestyle=':', alpha=0.7)
ax2.set_xlabel('Número de Componentes')
ax2.set_ylabel('Varianza Explicada Acumulada')
ax2.set_title('Varianza Explicada Acumulada')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Comparación de umbrales
thresholds = [90, 95, 99]
components = [n_components_90, n_components_95, n_components_99]
reduction = [(X_preprocessed.shape[1] - comp) / X_preprocessed.shape[1] * 100 for comp in components]

bars = ax3.bar(range(len(thresholds)), components, 
              color=[COLORS['accent'], COLORS['primary'], COLORS['secondary']])
ax3.set_xlabel('Umbral de Varianza (%)')
ax3.set_ylabel('Número de Componentes')
ax3.set_title('Componentes por Umbral de Varianza')
ax3.set_xticks(range(len(thresholds)))
ax3.set_xticklabels([f'{t}%' for t in thresholds])
ax3.grid(True, alpha=0.3, axis='y')

# Agregar valores en las barras
for bar, comp, red in zip(bars, components, reduction):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{comp}\n({red:.1f}% red.)', ha='center', va='bottom', fontweight='bold')

# Subplot 4: Primeros componentes principales
first_15_variance = pca_full.explained_variance_ratio_[:15]
ax4.bar(range(1, len(first_15_variance) + 1), first_15_variance, color=COLORS['neutral'])
ax4.set_xlabel('Componente Principal')
ax4.set_ylabel('Varianza Explicada')
ax4.set_title('Primeros 15 Componentes Principales')
ax4.grid(True, alpha=0.3, axis='y')

# Subplot 5: Scree plot
eigenvalues = pca_full.explained_variance_[:20]
ax5.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'go-', linewidth=2, markersize=6)
ax5.axhline(y=1, color='red', linestyle='--', label='Kaiser Criterion (λ=1)')
ax5.set_xlabel('Componente Principal')
ax5.set_ylabel('Eigenvalue')
ax5.set_title('Scree Plot (Primeros 20 componentes)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Subplot 6: Distribución de varianza acumulada
milestones = [5, 10, 15, 20, 30]
milestone_variance = []
for milestone in milestones:
    if milestone <= len(cumulative_variance):
        milestone_variance.append(cumulative_variance[milestone-1])
    else:
        milestone_variance.append(cumulative_variance[-1])

ax6.plot(milestones, milestone_variance, 'mo-', linewidth=2, markersize=8)
ax6.axhline(y=0.95, color='green', linestyle='--', label='95% objetivo')
ax6.set_xlabel('Número de Componentes')
ax6.set_ylabel('Varianza Explicada Acumulada')
ax6.set_title('Varianza Acumulada en Hitos')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Celda 7: Definir modelos (los dos mejores)

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

## Celda 8: Evaluación con características originales

```python
print("=== EVALUACIÓN CON CARACTERÍSTICAS ORIGINALES ===")

results_original = {}

for model_name, model in models.items():
    print(f"\n--- {model_name} ---")
    
    # Crear pipeline sin PCA
    pipeline_original = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Evaluar con validación cruzada
    cv_scores = cross_val_score(pipeline_original, X, y, cv=5, scoring='f1')
    
    # Evaluar en conjunto de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline_original.fit(X_train, y_train)
    y_pred = pipeline_original.predict(X_test)
    y_proba = pipeline_original.predict_proba(X_test)[:, 1]
    
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

## Celda 9: Evaluación con PCA

```python
print("=== EVALUACIÓN CON PCA ===")

results_pca = {}

for model_name, model in models.items():
    print(f"\n--- {model_name} (PCA) ---")
    print(f"Usando {n_components_selected} componentes principales")
    
    # Crear pipeline con PCA
    pipeline_pca = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components_selected, random_state=42)),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Evaluar con validación cruzada
    cv_scores = cross_val_score(pipeline_pca, X, y, cv=5, scoring='f1')
    
    # Evaluar en conjunto de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline_pca.fit(X_train, y_train)
    y_pred = pipeline_pca.predict(X_test)
    y_proba = pipeline_pca.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    results_pca[model_name] = metrics
    
    # Mostrar resultados
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    print(f"  F1-Score CV: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
```

## Celda 10: Crear tabla comparativa

```python
print("=== TABLA COMPARATIVA DE RESULTADOS PCA ===")

# Crear DataFrame comparativo
comparison_data = []

for model_name in models.keys():
    # Resultados originales
    orig_metrics = results_original[model_name]
    comparison_data.append({
        'Modelo': model_name,
        'Método': 'Original',
        'Características': original_features,
        'Accuracy': f"{orig_metrics['Accuracy']:.3f}",
        'Precision': f"{orig_metrics['Precision']:.3f}",
        'Recall': f"{orig_metrics['Recall']:.3f}",
        'F1-Score': f"{orig_metrics['F1-Score']:.3f}",
        'AUC-ROC': f"{orig_metrics['AUC-ROC']:.3f}",
        'Reducción (%)': '0.0%'
    })
    
    # Resultados con PCA
    pca_metrics = results_pca[model_name]
    reduction = ((original_features - n_components_selected) / original_features) * 100
    
    comparison_data.append({
        'Modelo': model_name,
        'Método': 'PCA',
        'Características': n_components_selected,
        'Accuracy': f"{pca_metrics['Accuracy']:.3f}",
        'Precision': f"{pca_metrics['Precision']:.3f}",
        'Recall': f"{pca_metrics['Recall']:.3f}",
        'F1-Score': f"{pca_metrics['F1-Score']:.3f}",
        'AUC-ROC': f"{pca_metrics['AUC-ROC']:.3f}",
        'Reducción (%)': f"{reduction:.1f}%"
    })

df_comparison = pd.DataFrame(comparison_data)

# Mostrar tabla
print("\nTabla Comparativa de Resultados:")
print("=" * 100)
print(df_comparison.to_string(index=False))
```

## Celda 11: Gráfica de reducción dimensional

```python
# Gráfica de reducción dimensional
models_list = list(models.keys())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfica de barras de características
x = np.arange(len(models_list))
width = 0.35

original_dims = [original_features] * len(models_list)
pca_dims = [n_components_selected] * len(models_list)

bars1 = ax1.bar(x - width/2, original_dims, width, label='Original', color=COLORS['primary'])
bars2 = ax1.bar(x + width/2, pca_dims, width, label='PCA', color=COLORS['secondary'])

ax1.set_xlabel('Modelos', fontsize=12, fontweight='bold')
ax1.set_ylabel('Número de Características', fontsize=12, fontweight='bold')
ax1.set_title('Reducción Dimensional con PCA', fontsize=14, fontweight='bold')
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
reduction_pct = ((original_features - n_components_selected) / original_features) * 100
reductions = [reduction_pct] * len(models_list)

bars3 = ax2.bar(models_list, reductions, color=COLORS['accent'])
ax2.set_xlabel('Modelos', fontsize=12, fontweight='bold')
ax2.set_ylabel('Reducción (%)', fontsize=12, fontweight='bold')
ax2.set_title('Porcentaje de Reducción Dimensional', fontsize=14, fontweight='bold')
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

## Celda 12: Gráfica de comparación de métricas

```python
# Gráfica de comparación de métricas
metrics = ['F1-Score', 'AUC-ROC']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, metric in enumerate(metrics):
    ax = axes[i]
    
    # Datos originales y PCA
    orig_values = [results_original[model][metric] for model in models_list]
    pca_values = [results_pca[model][metric] for model in models_list]
    
    x = np.arange(len(models_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, orig_values, width, label='Original', color=COLORS['primary'])
    bars2 = ax.bar(x + width/2, pca_values, width, label='PCA', color=COLORS['secondary'])
    
    ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Comparación de {metric}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=45, ha='right')
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

## Celda 13: Resumen final

```python
print("=== RESUMEN FINAL DE EXTRACCIÓN DE CARACTERÍSTICAS PCA ===")
print("=" * 80)

print(f"\nCriterio de selección: 95% Varianza Explicada")
print("Justificación: 95% de varianza explicada es el estándar para mantener")
print("información crítica mientras se logra reducción dimensional significativa.")

print(f"\nReducción dimensional:")
print(f"  - Características originales (post-preprocesamiento): {original_features}")
print(f"  - Componentes principales: {n_components_selected}")
print(f"  - Reducción: {reduction_pct:.1f}%")
print(f"  - Varianza explicada: {pca_full.explained_variance_ratio_[:n_components_selected].sum():.1%}")

print(f"\nResultados por modelo:")
for model_name in models.keys():
    orig_f1 = results_original[model_name]['F1-Score']
    pca_f1 = results_pca[model_name]['F1-Score']
    f1_change = ((pca_f1 - orig_f1) / orig_f1) * 100
    
    orig_auc = results_original[model_name]['AUC-ROC']
    pca_auc = results_pca[model_name]['AUC-ROC']
    auc_change = ((pca_auc - orig_auc) / orig_auc) * 100
    
    print(f"\n{model_name}:")
    print(f"  - F1-Score original: {orig_f1:.3f}")
    print(f"  - F1-Score PCA: {pca_f1:.3f}")
    print(f"  - Cambio en F1-Score: {f1_change:+.1f}%")
    print(f"  - AUC-ROC original: {orig_auc:.3f}")
    print(f"  - AUC-ROC PCA: {pca_auc:.3f}")
    print(f"  - Cambio en AUC-ROC: {auc_change:+.1f}%")

print(f"\n✅ Análisis de extracción de características PCA completado.")
```

## Celda 14: Análisis de componentes más importantes

```python
print("=== ANÁLISIS DE COMPONENTES MÁS IMPORTANTES ===")

# Mostrar varianza explicada por cada componente (primeros 10)
print(f"Varianza explicada por cada componente (primeros 10):")
for i, var_exp in enumerate(pca_full.explained_variance_ratio_[:10], 1):
    print(f"  PC{i}: {var_exp:.4f} ({var_exp*100:.2f}%)")

print(f"\nVarianza explicada acumulada (primeros {n_components_selected} componentes): {pca_full.explained_variance_ratio_[:n_components_selected].sum():.4f} ({pca_full.explained_variance_ratio_[:n_components_selected].sum()*100:.2f}%)")

# Gráfica de contribución de componentes
plt.figure(figsize=(12, 6))

# Subplot 1: Varianza explicada por componente
plt.subplot(1, 2, 1)
components_to_show = min(20, n_components_selected)
plt.bar(range(1, components_to_show + 1), 
        pca_full.explained_variance_ratio_[:components_to_show], 
        color=COLORS['primary'])
plt.xlabel('Componente Principal')
plt.ylabel('Varianza Explicada')
plt.title(f'Varianza Explicada por Componente\n(Primeros {components_to_show} componentes)')
plt.grid(True, alpha=0.3, axis='y')

# Subplot 2: Varianza explicada acumulada
plt.subplot(1, 2, 2)
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_[:components_to_show])
plt.plot(range(1, components_to_show + 1), cumulative_var, 'o-', 
         color=COLORS['secondary'], linewidth=2, markersize=6)
plt.axhline(y=0.95, color='red', linestyle='--', label='95% Umbral')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
``` 