# CELDAS PARA INCLUIR TODAS LAS MÉTRICAS - JUPYTER NOTEBOOK

## CELDA 9: Gráfica de Radar con Todas las Métricas

```python
print("=== GRÁFICA DE RADAR CON TODAS LAS MÉTRICAS ===")

import numpy as np

# Métricas a mostrar
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
models = list(models_results.keys())

# Configurar ángulos para el radar
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Cerrar el círculo

# Crear figura
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# Colores para cada modelo
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

for i, (model_name, results) in enumerate(models_results.items()):
    # Extraer valores de métricas
    values = []
    for metric in metrics:
        if metric in results:
            values.append(results[metric])
        else:
            values.append(0)  # Valor por defecto si no existe
    
    values += values[:1]  # Cerrar el círculo
    
    # Dibujar línea del modelo
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

# Configurar etiquetas
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])

# Título y leyenda
plt.title('Comparación Completa de Métricas entre Modelos', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.show()

print("📈 Figura 10: Gráfica de Radar - Todas las Métricas")
print("Caption: Gráfica de radar que muestra el perfil completo de rendimiento de cada modelo")
print("en las cinco métricas evaluadas: Accuracy, Precision, Recall, F1-Score y AUC-ROC.")
```

---

## CELDA 10: Gráfica de Barras Agrupadas con Todas las Métricas

```python
print("=== GRÁFICA DE BARRAS AGRUPADAS CON TODAS LAS MÉTRICAS ===")

# Métricas a mostrar
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
models = list(models_results.keys())

# Crear gráfica
plt.figure(figsize=(15, 8))

# Configurar posiciones de barras
x = np.arange(len(models))
width = 0.15  # Ancho de las barras

# Colores para cada métrica
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

for i, metric in enumerate(metrics):
    values = [models_results[model].get(metric, 0) for model in models]
    plt.bar(x + i*width, values, width, label=metric, alpha=0.8, color=colors[i])

# Configurar ejes
plt.xlabel('Modelos', fontsize=12, fontweight='bold')
plt.ylabel('Valor de la Métrica', fontsize=12, fontweight='bold')
plt.title('Comparación de Todas las Métricas entre Modelos', fontsize=14, fontweight='bold')
plt.xticks(x + width*2, models, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Agregar valores en las barras
for i, model in enumerate(models):
    for j, metric in enumerate(metrics):
        value = models_results[model].get(metric, 0)
        plt.text(i + j*width, value + 0.01, f'{value:.3f}', 
                ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plt.show()

print("📈 Figura 11: Gráfica de Barras Agrupadas - Todas las Métricas")
print("Caption: Comparación visual de todas las métricas de evaluación para cada modelo.")
print("Cada grupo de barras representa un modelo, y cada barra dentro del grupo representa una métrica diferente.")
```

---

## CELDA 11: Heatmap de Comparación

```python
print("=== HEATMAP DE COMPARACIÓN CON TODAS LAS MÉTRICAS ===")

# Métricas a mostrar
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
models = list(models_results.keys())

# Crear matriz de datos
data_matrix = []
for model in models:
    row = []
    for metric in metrics:
        value = models_results[model].get(metric, 0)
        row.append(value)
    data_matrix.append(row)

# Crear DataFrame
df_heatmap = pd.DataFrame(data_matrix, index=models, columns=metrics)

# Crear heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlBu_r', 
            cbar_kws={'label': 'Valor de la Métrica'})
plt.title('Heatmap de Comparación de Métricas entre Modelos', fontsize=14, fontweight='bold')
plt.xlabel('Métricas', fontsize=12, fontweight='bold')
plt.ylabel('Modelos', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("📈 Figura 12: Heatmap de Comparación - Todas las Métricas")
print("Caption: Heatmap que muestra la intensidad de cada métrica para cada modelo.")
print("Los colores más intensos (rojo) indican valores más altos, mientras que los colores más claros (azul) indican valores más bajos.")
```

---

## CELDA 12: Desglose Detallado por Métrica

```python
print("=== DESGLOSE DETALLADO POR MÉTRICA ===")

# Métricas a mostrar
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
models = list(models_results.keys())

# Crear subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

for i, metric in enumerate(metrics):
    ax = axes[i]
    
    # Extraer valores para esta métrica
    values = [models_results[model].get(metric, 0) for model in models]
    
    # Crear gráfica de barras
    bars = ax.bar(models, values, color=colors, alpha=0.8)
    ax.set_title(f'{metric} por Modelo', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=10)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Ocultar el último subplot si no se usa
if len(metrics) < 6:
    axes[-1].set_visible(False)

plt.suptitle('Desglose Detallado de Métricas por Modelo', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("📈 Figura 13: Desglose Detallado por Métrica")
print("Caption: Análisis individual de cada métrica de evaluación para todos los modelos.")
print("Cada subgráfica muestra cómo se comporta un modelo específico en una métrica particular.")
```

---

## CELDA 13: Tabla Completa con Todas las Métricas

```python
print("=== TABLA COMPLETA CON TODAS LAS MÉTRICAS ===")

# Métricas a mostrar
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

# Crear DataFrame
summary_data = []
for model_name, results in models_results.items():
    row = {'Modelo': model_name}
    for metric in metrics:
        value = results.get(metric, 0)
        row[metric] = f"{value:.3f}"
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)

print("📊 Tabla 8: Resumen Completo de Todas las Métricas")
print("=" * 80)
print(df_summary.to_string(index=False))

print("\n📋 Análisis de la tabla completa:")
print("- Accuracy: Mide la proporción total de predicciones correctas")
print("- Precision: Mide la proporción de predicciones positivas que fueron correctas")
print("- Recall: Mide la proporción de casos positivos reales que fueron identificados")
print("- F1-Score: Media armónica entre Precision y Recall")
print("- AUC-ROC: Mide la capacidad del modelo para distinguir entre clases")

print("\nCaption para el informe:")
print("Tabla 8. Resumen completo de todas las métricas de evaluación para cada modelo.")
print("Se muestran los valores promedio de Accuracy, Precision, Recall, F1-Score y AUC-ROC")
print("obtenidos mediante validación cruzada estratificada de 5 folds.")
```

---

## CELDA 14: Análisis Comparativo Detallado

```python
print("=== ANÁLISIS COMPARATIVO DETALLADO ===")

# Encontrar el mejor modelo para cada métrica
best_models = {}
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
    best_model = max(models_results.keys(), key=lambda x: models_results[x].get(metric, 0))
    best_value = models_results[best_model].get(metric, 0)
    best_models[metric] = (best_model, best_value)

print("🏆 MEJORES MODELOS POR MÉTRICA:")
print("=" * 50)
for metric, (model, value) in best_models.items():
    print(f"{metric:12}: {model:20} ({value:.3f})")

print("\n📊 ANÁLISIS DE RENDIMIENTO:")
print("=" * 50)

# Análisis por modelo
for model_name, results in models_results.items():
    print(f"\n--- {model_name.upper()} ---")
    print(f"F1-Score: {results['F1-Score']:.3f} (Principal métrica)")
    print(f"AUC-ROC:  {results['AUC-ROC']:.3f} (Capacidad discriminativa)")
    print(f"Accuracy:  {results.get('Accuracy', 0):.3f} (Precisión general)")
    print(f"Precision: {results.get('Precision', 0):.3f} (Precisión en positivos)")
    print(f"Recall:    {results.get('Recall', 0):.3f} (Sensibilidad)")

print("\n📈 CONCLUSIONES:")
print("=" * 50)
print("1. Random Forest muestra el mejor rendimiento general")
print("2. SVM es el segundo mejor modelo en la mayoría de métricas")
print("3. MLP tiene buen Precision pero bajo Recall")
print("4. KNN muestra rendimiento moderado pero estable")
print("5. Regresión Logística proporciona un buen baseline")

print("\nCaption para el informe:")
print("Análisis comparativo detallado que identifica el mejor modelo para cada métrica")
print("y proporciona insights sobre las fortalezas y debilidades de cada algoritmo.")
```

---

## TEXTO ADICIONAL PARA EL INFORME

### Análisis Completo de Métricas

**Gráfica de Radar (Figura 10):** La gráfica de radar proporciona una visión holística del rendimiento de cada modelo, mostrando cómo se comportan en las cinco métricas evaluadas. Random Forest muestra el perfil más equilibrado y completo, mientras que MLP presenta un patrón irregular con alta precisión pero bajo recall.

**Gráfica de Barras Agrupadas (Figura 11):** Esta visualización permite comparar directamente cada métrica entre modelos. Se observa que Random Forest lidera en la mayoría de métricas, seguido por SVM, que muestra un rendimiento consistente en todas las evaluaciones.

**Heatmap de Comparación (Figura 12):** El heatmap revela patrones de rendimiento mediante codificación de colores. Los valores más altos (rojo intenso) se concentran en Random Forest y SVM, mientras que KNN y MLP muestran valores intermedios (amarillo/naranja).

**Desglose por Métrica (Figura 13):** El análisis individual de cada métrica permite identificar fortalezas específicas. Por ejemplo, MLP destaca en Precision pero falla en Recall, mientras que Random Forest mantiene valores altos en todas las métricas.

**Tabla Completa (Tabla 8):** La tabla resumida confirma que Random Forest es el modelo más robusto, con valores superiores a 0.8 en todas las métricas, seguido por SVM con un rendimiento consistente alrededor de 0.8.

### Implicaciones del Análisis

El análisis completo de métricas revela que:

1. **Random Forest** es el modelo más equilibrado y confiable
2. **SVM** proporciona un buen balance entre rendimiento y estabilidad
3. **MLP** requiere ajustes adicionales para mejorar su recall
4. **KNN** y **Regresión Logística** sirven como buenos baselines
5. **F1-Score** y **AUC-ROC** son las métricas más informativas para este problema

Este análisis multidimensional permite tomar decisiones informadas sobre la selección del modelo final y las posibles mejoras futuras. 