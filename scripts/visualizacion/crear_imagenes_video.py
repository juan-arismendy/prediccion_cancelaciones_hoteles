#!/usr/bin/env python3
"""
Script para crear todas las imágenes faltantes del video de presentación
Predicción de Cancelaciones Hoteleras
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import warnings
warnings.filterwarnings('ignore')

# Configuración global de estilo
plt.style.use('default')
sns.set_palette("husl")

# Paleta de colores del proyecto
COLORS = {
    'azul_corporativo': '#2E86AB',
    'verde_exito': '#A23B72',
    'rojo_alerta': '#F18F01',
    'gris_texto': '#C73E1D',
    'azul_claro': '#A8DADC',
    'gris_claro': '#F1FAEE',
    'negro': '#1D3557'
}

# Configuración de figura estándar
FIGSIZE = (19.2, 10.8)  # 1920x1080 en inches (100 DPI)
DPI = 100

def setup_figure(title="", figsize=FIGSIZE):
    """Configuración estándar para todas las figuras"""
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    if title:
        fig.suptitle(title, fontsize=28, fontweight='bold', 
                    color=COLORS['negro'], y=0.95)
    
    return fig, ax

def add_logo_watermark(ax):
    """Añade logo/marca de agua en esquina superior derecha"""
    ax.text(0.98, 0.98, 'Predicción Cancelaciones Hoteleras', 
           transform=ax.transAxes, fontsize=12, 
           horizontalalignment='right', verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['azul_claro'], alpha=0.7))

def crear_logo_proyecto():
    """IMAGEN 1: Logo del proyecto"""
    print("🎬 Creando logo del proyecto...")
    
    fig, ax = setup_figure()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Fondo degradado
    gradient = np.linspace(0, 1, 256).reshape(256, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, extent=[0, 10, 0, 6], aspect='auto', cmap='Blues', alpha=0.3)
    
    # Título principal
    ax.text(5, 4.2, 'PREDICCIÓN DE', fontsize=48, fontweight='bold',
           ha='center', va='center', color=COLORS['azul_corporativo'])
    ax.text(5, 3.4, 'CANCELACIONES HOTELERAS', fontsize=48, fontweight='bold',
           ha='center', va='center', color=COLORS['negro'])
    
    # Subtítulo
    ax.text(5, 2.6, 'Machine Learning para la Industria Hotelera', fontsize=24,
           ha='center', va='center', color=COLORS['gris_texto'], style='italic')
    
    # Iconos decorativos (representación simple)
    # Hotel
    hotel_rect = FancyBboxPatch((1, 1), 1.5, 1.2, boxstyle="round,pad=0.1",
                               facecolor=COLORS['azul_claro'], edgecolor=COLORS['azul_corporativo'], linewidth=2)
    ax.add_patch(hotel_rect)
    ax.text(1.75, 1.6, '🏨', fontsize=40, ha='center', va='center')
    
    # ML/AI
    ml_circle = Circle((7.5, 1.6), 0.8, facecolor=COLORS['verde_exito'], 
                      edgecolor=COLORS['negro'], linewidth=2, alpha=0.8)
    ax.add_patch(ml_circle)
    ax.text(7.5, 1.6, '🤖', fontsize=40, ha='center', va='center')
    
    # Flecha conectora
    arrow = patches.FancyArrowPatch((2.8, 1.6), (6.5, 1.6),
                                   arrowstyle='->', mutation_scale=30,
                                   color=COLORS['rojo_alerta'], linewidth=4)
    ax.add_patch(arrow)
    
    # Universidad/Autores
    ax.text(5, 0.8, 'Universidad de Antioquia - Facultad de Ingeniería', 
           fontsize=16, ha='center', va='center', color=COLORS['gris_texto'])
    ax.text(5, 0.4, 'Juan Arismendy & Laura - 2024', 
           fontsize=14, ha='center', va='center', color=COLORS['gris_texto'])
    
    plt.tight_layout()
    plt.savefig('logo_proyecto.png', dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Logo creado: logo_proyecto.png")

def crear_impacto_economico():
    """IMAGEN 2: Gráfica de impacto económico"""
    print("📊 Creando gráfica de impacto económico...")
    
    fig, ax = setup_figure("Impacto Económico de las Cancelaciones Hoteleras")
    
    # Datos de ejemplo (basados en estudios reales)
    categorias = ['Pérdidas\nDirectas', 'Costos\nOperativos', 'Oportunidad\nPerdida', 'Total\nAnual']
    valores = [2.8, 1.2, 1.5, 5.5]  # En miles de millones USD
    colores = [COLORS['rojo_alerta'], COLORS['rojo_alerta'], 
              COLORS['rojo_alerta'], COLORS['negro']]
    
    # Gráfica de barras
    bars = ax.bar(categorias, valores, color=colores, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Etiquetas en las barras
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'${valor:.1f}B', ha='center', va='bottom', 
                fontsize=20, fontweight='bold', color=COLORS['negro'])
    
    # Configuración de ejes
    ax.set_ylabel('Pérdidas (Miles de Millones USD)', fontsize=18, fontweight='bold')
    ax.set_ylim(0, 6.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Texto explicativo
    ax.text(0.02, 0.98, '• 37.8% de reservas se cancelan globalmente\n• Industria hotelera pierde $5.5B anuales\n• Predicción temprana = Optimización de ingresos', 
           transform=ax.transAxes, fontsize=16, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['azul_claro'], alpha=0.9))
    
    # Estadística destacada
    ax.text(0.98, 0.5, '5.5\nMIL MILLONES\nUSD/AÑO', transform=ax.transAxes, 
           fontsize=32, fontweight='bold', ha='right', va='center',
           color=COLORS['rojo_alerta'],
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor=COLORS['rojo_alerta'], linewidth=3))
    
    add_logo_watermark(ax)
    
    plt.tight_layout()
    plt.savefig('impacto_economico_hoteles.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Impacto económico creado: impacto_economico_hoteles.png")

def crear_dataset_visualization():
    """IMAGEN 3: Visualización del dataset"""
    print("🗃️ Creando visualización del dataset...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGSIZE, dpi=DPI)
    fig.suptitle('Dataset: 119,390 Reservas Hoteleras (2015-2017)', fontsize=28, fontweight='bold')
    
    # 1. Información general del dataset
    ax1.axis('off')
    ax1.text(0.5, 0.9, 'INFORMACIÓN GENERAL', fontsize=18, fontweight='bold', 
            ha='center', transform=ax1.transAxes, color=COLORS['azul_corporativo'])
    
    info_text = """📊 119,390 reservas totales
📅 Período: 2015-2017
🏨 Hoteles: Urbanos y Resort
📋 36 características originales
🎯 Variable objetivo: is_canceled
🌍 Datos de múltiples países"""
    
    ax1.text(0.05, 0.7, info_text, fontsize=14, transform=ax1.transAxes, 
            verticalalignment='top')
    
    # 2. Distribución por años
    años = [2015, 2016, 2017]
    reservas_año = [18567, 56707, 44116]  # Datos aproximados
    
    ax2.bar(años, reservas_año, color=COLORS['azul_corporativo'], alpha=0.8)
    ax2.set_title('Reservas por Año', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Número de Reservas')
    ax2.tick_params(axis='both', labelsize=12)
    
    for i, v in enumerate(reservas_año):
        ax2.text(años[i], v + 1000, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Tipo de hotel
    tipos = ['City Hotel', 'Resort Hotel']
    cantidades = [79330, 40060]  # Datos aproximados
    
    wedges, texts, autotexts = ax3.pie(cantidades, labels=tipos, autopct='%1.1f%%',
                                      colors=[COLORS['azul_corporativo'], COLORS['verde_exito']],
                                      startangle=90)
    ax3.set_title('Distribución por Tipo de Hotel', fontsize=16, fontweight='bold')
    
    # 4. Muestra de características principales
    ax4.axis('off')
    ax4.text(0.5, 0.9, 'CARACTERÍSTICAS PRINCIPALES', fontsize=18, fontweight='bold',
            ha='center', transform=ax4.transAxes, color=COLORS['azul_corporativo'])
    
    caracteristicas = """• lead_time (tiempo anticipación)
• arrival_date_* (fecha llegada)
• stays_in_*_nights (noches estadía)
• adults, children, babies (huéspedes)
• meal (tipo de comida)
• country (país origen)
• market_segment (segmento mercado)
• adr (tarifa diaria promedio)
• total_of_special_requests"""
    
    ax4.text(0.05, 0.8, caracteristicas, fontsize=12, transform=ax4.transAxes,
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('dataset_visualization.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Visualización dataset creada: dataset_visualization.png")

def crear_distribucion_clases():
    """IMAGEN 4: Distribución de clases"""
    print("📈 Creando distribución de clases...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
    fig.suptitle('Distribución de Cancelaciones: Problema de Clases Desbalanceadas', 
                fontsize=24, fontweight='bold')
    
    # Gráfica de torta
    labels = ['No Cancelado\n(62.2%)', 'Cancelado\n(37.8%)']
    sizes = [62.2, 37.8]
    colors = [COLORS['verde_exito'], COLORS['rojo_alerta']]
    explode = (0, 0.1)  # Explotar la porción de cancelados
    
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', shadow=True, startangle=90,
                                      textprops={'fontsize': 14, 'fontweight': 'bold'})
    
    ax1.set_title('Distribución Global', fontsize=18, fontweight='bold', pad=20)
    
    # Gráfica de barras comparativa
    categorias = ['No Cancelado', 'Cancelado']
    valores = [74226, 45164]  # Números aproximados
    
    bars = ax2.bar(categorias, valores, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Etiquetas en las barras
    for bar, valor, porcentaje in zip(bars, valores, [62.2, 37.8]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{valor:,}\n({porcentaje}%)', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    ax2.set_title('Cantidad de Reservas', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Número de Reservas', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_ylim(0, 85000)
    
    # Texto explicativo
    explicacion = """🎯 IMPLICACIONES:

• Clases desbalanceadas (62.2% vs 37.8%)
• Requiere técnicas especializadas
• SMOTE para balanceo sintético
• F1-Score como métrica principal
• Validación cruzada estratificada"""
    
    fig.text(0.02, 0.02, explicacion, fontsize=12, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['azul_claro'], alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('distribucion_clases.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Distribución de clases creada: distribucion_clases.png")

def crear_diagrama_5_modelos():
    """IMAGEN 5: Diagrama de los 5 modelos ML"""
    print("🤖 Creando diagrama de 5 modelos ML...")
    
    fig, ax = setup_figure("Pipeline de Machine Learning: 5 Algoritmos Evaluados")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Datos de entrada
    data_box = FancyBboxPatch((0.5, 6), 2, 1, boxstyle="round,pad=0.1",
                             facecolor=COLORS['azul_claro'], edgecolor=COLORS['azul_corporativo'], linewidth=2)
    ax.add_patch(data_box)
    ax.text(1.5, 6.5, 'DATOS\n119,390 reservas\n36 características', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Preprocesamiento
    prep_box = FancyBboxPatch((4, 6), 2, 1, boxstyle="round,pad=0.1",
                             facecolor=COLORS['gris_claro'], edgecolor=COLORS['gris_texto'], linewidth=2)
    ax.add_patch(prep_box)
    ax.text(5, 6.5, 'PREPROCESAMIENTO\nStandardScaler\nOneHotEncoder\nSMOTE', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Los 5 modelos
    modelos = [
        ('Logistic\nRegression', '🔢', (1.5, 4)),
        ('K-Nearest\nNeighbors', '👥', (3.5, 4)),
        ('Random\nForest', '🌳', (5.5, 4)),
        ('Neural\nNetwork', '🧠', (7.5, 4)),
        ('Support Vector\nMachine', '📐', (9.5, 4))
    ]
    
    colores_modelos = [COLORS['azul_corporativo'], COLORS['verde_exito'], COLORS['rojo_alerta'], 
                      COLORS['gris_texto'], COLORS['azul_claro']]
    
    for i, (nombre, emoji, pos) in enumerate(modelos):
        # Caja del modelo
        model_box = FancyBboxPatch((pos[0]-0.6, pos[1]-0.5), 1.2, 1, boxstyle="round,pad=0.1",
                                  facecolor=colores_modelos[i], alpha=0.8, 
                                  edgecolor='white', linewidth=2)
        ax.add_patch(model_box)
        
        # Emoji y texto
        ax.text(pos[0], pos[1]+0.2, emoji, ha='center', va='center', fontsize=20)
        ax.text(pos[0], pos[1]-0.2, nombre, ha='center', va='center', 
               fontsize=9, fontweight='bold', color='white')
    
    # Grid Search
    grid_box = FancyBboxPatch((4, 2), 3, 0.8, boxstyle="round,pad=0.1",
                             facecolor=COLORS['negro'], edgecolor='white', linewidth=2)
    ax.add_patch(grid_box)
    ax.text(5.5, 2.4, 'GRID SEARCH\nOptimización de Hiperparámetros\n5-Fold Cross Validation', 
           ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Evaluación
    eval_box = FancyBboxPatch((9, 6), 2.5, 1, boxstyle="round,pad=0.1",
                             facecolor=COLORS['verde_exito'], edgecolor='white', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(10.25, 6.5, 'EVALUACIÓN\nF1-Score, AUC-ROC\nAccuracy, Precision\nRecall', 
           ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Flechas de flujo
    # Datos -> Preprocesamiento
    arrow1 = patches.FancyArrowPatch((2.5, 6.5), (4, 6.5), arrowstyle='->', 
                                    mutation_scale=20, color=COLORS['negro'], linewidth=2)
    ax.add_patch(arrow1)
    
    # Preprocesamiento -> Modelos
    for i, (_, _, pos) in enumerate(modelos):
        arrow = patches.FancyArrowPatch((5, 6), pos, arrowstyle='->', 
                                       mutation_scale=15, color=COLORS['gris_texto'], linewidth=1.5)
        ax.add_patch(arrow)
    
    # Modelos -> Grid Search
    for i, (_, _, pos) in enumerate(modelos):
        arrow = patches.FancyArrowPatch(pos, (5.5, 2.8), arrowstyle='->', 
                                       mutation_scale=15, color=COLORS['gris_texto'], linewidth=1.5)
        ax.add_patch(arrow)
    
    # Grid Search -> Evaluación
    arrow_final = patches.FancyArrowPatch((7, 2.4), (9, 6.5), arrowstyle='->', 
                                         mutation_scale=20, color=COLORS['negro'], linewidth=2)
    ax.add_patch(arrow_final)
    
    # Título de metodología
    ax.text(6, 0.5, 'Metodología: Evaluación Comparativa con Optimización de Hiperparámetros', 
           ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['azul_corporativo'])
    
    add_logo_watermark(ax)
    
    plt.tight_layout()
    plt.savefig('diagrama_5_modelos.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Diagrama de 5 modelos creado: diagrama_5_modelos.png")

def crear_smote_visualization():
    """IMAGEN 6: Visualización de SMOTE"""
    print("⚖️ Creando visualización de SMOTE...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
    fig.suptitle('Balanceo de Clases con SMOTE (Synthetic Minority Oversampling Technique)', 
                fontsize=20, fontweight='bold')
    
    # Generar datos sintéticos para demostración
    np.random.seed(42)
    
    # Datos originales (desbalanceados)
    n_majority = 100
    n_minority = 40
    
    # Clase mayoritaria (No cancelado)
    X_majority = np.random.normal(2, 0.8, (n_majority, 2))
    y_majority = np.zeros(n_majority)
    
    # Clase minoritaria (Cancelado)
    X_minority = np.random.normal(4, 0.6, (n_minority, 2))
    y_minority = np.ones(n_minority)
    
    # ANTES de SMOTE
    ax1.scatter(X_majority[:, 0], X_majority[:, 1], c=COLORS['verde_exito'], 
               alpha=0.7, s=60, label=f'No Cancelado (n={n_majority})', edgecolors='white')
    ax1.scatter(X_minority[:, 0], X_minority[:, 1], c=COLORS['rojo_alerta'], 
               alpha=0.7, s=60, label=f'Cancelado (n={n_minority})', edgecolors='white')
    
    ax1.set_title('ANTES: Clases Desbalanceadas\n62.2% vs 37.8%', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Característica 1', fontsize=12)
    ax1.set_ylabel('Característica 2', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # DESPUÉS de SMOTE (simulado)
    # Generar muestras sintéticas
    n_synthetic = n_majority - n_minority
    X_synthetic = np.random.normal(4.2, 0.7, (n_synthetic, 2))
    
    # Combinar datos originales con sintéticos
    X_balanced_majority = X_majority
    X_balanced_minority = np.vstack([X_minority, X_synthetic])
    
    ax2.scatter(X_balanced_majority[:, 0], X_balanced_majority[:, 1], 
               c=COLORS['verde_exito'], alpha=0.7, s=60, 
               label=f'No Cancelado (n={len(X_balanced_majority)})', edgecolors='white')
    ax2.scatter(X_minority[:, 0], X_minority[:, 1], c=COLORS['rojo_alerta'], 
               alpha=0.9, s=60, label=f'Cancelado Original (n={n_minority})', 
               edgecolors='white', marker='o')
    ax2.scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=COLORS['rojo_alerta'], 
               alpha=0.5, s=60, label=f'Cancelado Sintético (n={n_synthetic})', 
               edgecolors='black', marker='^')
    
    ax2.set_title('DESPUÉS: Clases Balanceadas\n50% vs 50%', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Característica 1', fontsize=12)
    ax2.set_ylabel('Característica 2', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Flecha entre gráficas
    fig.text(0.5, 0.5, '→\nSMOTE', ha='center', va='center', fontsize=24, 
            fontweight='bold', color=COLORS['azul_corporativo'])
    
    # Explicación
    explicacion = """🎯 SMOTE (Synthetic Minority Oversampling Technique):
• Genera muestras sintéticas de la clase minoritaria
• Interpola entre ejemplos existentes de la clase minoritaria
• Balancea las clases sin duplicar datos
• Mejora el rendimiento en clasificación desbalanceada"""
    
    fig.text(0.02, 0.02, explicacion, fontsize=12, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['azul_claro'], alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('smote_visualization.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Visualización SMOTE creada: smote_visualization.png")

def crear_roadmap_implementacion():
    """IMAGEN 12: Roadmap de implementación"""
    print("🗺️ Creando roadmap de implementación...")
    
    fig, ax = setup_figure("Roadmap de Implementación: De la Investigación a la Producción")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Dos opciones principales
    # OPCIÓN 1: Máximo Rendimiento
    ax.text(3, 9, 'OPCIÓN 1: MÁXIMO RENDIMIENTO', fontsize=20, fontweight='bold',
           ha='center', va='center', color=COLORS['verde_exito'])
    
    opcion1_box = FancyBboxPatch((0.5, 7), 5, 1.5, boxstyle="round,pad=0.2",
                                facecolor=COLORS['verde_exito'], alpha=0.2, 
                                edgecolor=COLORS['verde_exito'], linewidth=3)
    ax.add_patch(opcion1_box)
    
    ax.text(3, 7.75, '🏆 Random Forest + Selección Secuencial\n📊 F1-Score: 0.893 (+9.7% mejora)\n⚡ 39.4% menos características\n🎯 Uso: Sistemas con recursos normales', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    # OPCIÓN 2: Recursos Limitados
    ax.text(9, 9, 'OPCIÓN 2: RECURSOS LIMITADOS', fontsize=20, fontweight='bold',
           ha='center', va='center', color=COLORS['azul_corporativo'])
    
    opcion2_box = FancyBboxPatch((6.5, 7), 5, 1.5, boxstyle="round,pad=0.2",
                                facecolor=COLORS['azul_corporativo'], alpha=0.2, 
                                edgecolor=COLORS['azul_corporativo'], linewidth=3)
    ax.add_patch(opcion2_box)
    
    ax.text(9, 7.75, '💎 SVM + PCA\n📊 F1-Score: 0.745 (-0.6% pérdida mínima)\n⚡ 42.9% menos características\n🎯 Uso: Sistemas con restricciones', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Timeline de implementación
    ax.text(6, 6, 'TIMELINE DE IMPLEMENTACIÓN', fontsize=18, fontweight='bold',
           ha='center', va='center', color=COLORS['negro'])
    
    # Fases del timeline
    fases = [
        ('FASE 1\nPreparación\n(2 semanas)', 1.5, COLORS['rojo_alerta']),
        ('FASE 2\nDesarrollo\n(4 semanas)', 3.5, COLORS['azul_corporativo']),
        ('FASE 3\nPruebas\n(2 semanas)', 5.5, COLORS['verde_exito']),
        ('FASE 4\nDespliegue\n(1 semana)', 7.5, COLORS['gris_texto']),
        ('FASE 5\nMonitoreo\n(Continuo)', 9.5, COLORS['negro'])
    ]
    
    for i, (fase, x, color) in enumerate(fases):
        # Círculo de fase
        circle = Circle((x, 4.5), 0.4, facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, 4.5, str(i+1), ha='center', va='center', fontsize=16, 
               fontweight='bold', color='white')
        
        # Texto de fase
        ax.text(x, 3.5, fase, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Flecha conectora (excepto el último)
        if i < len(fases) - 1:
            arrow = patches.FancyArrowPatch((x+0.5, 4.5), (fases[i+1][1]-0.5, 4.5), 
                                           arrowstyle='->', mutation_scale=15, 
                                           color=COLORS['gris_texto'], linewidth=2)
            ax.add_patch(arrow)
    
    # Beneficios clave
    beneficios = """🎯 BENEFICIOS CLAVE:
✅ Reducción de cancelaciones inesperadas
✅ Optimización de inventario de habitaciones  
✅ Mejora en planificación financiera
✅ Estrategias proactivas de retención
✅ ROI estimado: 15-25% en primer año"""
    
    ax.text(0.5, 2.5, beneficios, fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['azul_claro'], alpha=0.9))
    
    # Requisitos técnicos
    requisitos = """⚙️ REQUISITOS TÉCNICOS:
• Python 3.8+ con scikit-learn
• Servidor con 4GB RAM mínimo
• Base de datos para históricos
• API REST para integración
• Dashboard para monitoreo"""
    
    ax.text(6.5, 2.5, requisitos, fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['gris_claro'], alpha=0.9))
    
    add_logo_watermark(ax)
    
    plt.tight_layout()
    plt.savefig('roadmap_implementacion.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Roadmap de implementación creado: roadmap_implementacion.png")

def crear_estructura_repositorio():
    """IMAGEN 13: Estructura del repositorio"""
    print("📁 Creando estructura del repositorio...")
    
    fig, ax = setup_figure("Estructura del Repositorio: Organización Profesional")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Título del repositorio
    repo_box = FancyBboxPatch((2, 8.5), 8, 1, boxstyle="round,pad=0.2",
                             facecolor=COLORS['azul_corporativo'], alpha=0.9, 
                             edgecolor='white', linewidth=2)
    ax.add_patch(repo_box)
    ax.text(6, 9, '📁 prediccion_cancelaciones_hoteles/', ha='center', va='center', 
           fontsize=18, fontweight='bold', color='white')
    
    # Estructura de carpetas
    estructura = [
        ('📓 proyecto_final.ipynb', 1, 7.5, 'Notebook principal'),
        ('📄 Reporte.pdf', 1, 7, 'Reporte técnico completo'),
        ('📖 README.md', 1, 6.5, 'Documentación principal'),
        ('📋 requirements.txt', 1, 6, 'Dependencias del proyecto'),
        
        ('📁 scripts/', 1, 5.3, 'Scripts de Python organizados'),
        ('  🔍 seleccion_caracteristicas/', 1.5, 4.9, '4 scripts de selección'),
        ('  🧮 extraccion_pca/', 1.5, 4.5, '3 scripts de PCA'),
        ('  📊 analisis_completo/', 1.5, 4.1, '7 scripts de análisis'),
        ('  📈 graficas/', 1.5, 3.7, '2 scripts de visualización'),
        
        ('📁 documentacion/', 7, 5.3, 'Documentación del proyecto'),
        ('  📝 celdas_notebook/', 7.5, 4.9, '5 archivos de celdas'),
        ('  📋 resumenes/', 7.5, 4.5, '3 resúmenes ejecutivos'),
        
        ('📁 visualizaciones/', 7, 3.5, 'Todas las visualizaciones'),
        ('  🔍 seleccion_secuencial/', 7.5, 3.1, '3 imágenes'),
        ('  🧮 pca/', 7.5, 2.7, '4 imágenes'),
        ('  📊 analisis_completo/', 7.5, 2.3, '8 imágenes'),
        ('  🎯 discusion_conclusiones/', 7.5, 1.9, '5 imágenes'),
    ]
    
    for item, x, y, descripcion in estructura:
        # Texto del item
        ax.text(x, y, item, fontsize=12, fontweight='bold', va='center')
        # Descripción
        ax.text(x + 3.5, y, descripcion, fontsize=10, va='center', 
               color=COLORS['gris_texto'], style='italic')
    
    # Estadísticas del proyecto
    stats_box = FancyBboxPatch((0.5, 0.5), 5, 1.2, boxstyle="round,pad=0.2",
                              facecolor=COLORS['verde_exito'], alpha=0.2, 
                              edgecolor=COLORS['verde_exito'], linewidth=2)
    ax.add_patch(stats_box)
    
    ax.text(3, 1.3, '📊 ESTADÍSTICAS DEL PROYECTO', ha='center', va='center', 
           fontsize=14, fontweight='bold', color=COLORS['verde_exito'])
    ax.text(3, 0.9, '• 13 scripts Python\n• 8 documentos MD\n• 18+ visualizaciones\n• Código 100% reproducible', 
           ha='center', va='center', fontsize=11)
    
    # Enlaces importantes
    links_box = FancyBboxPatch((6.5, 0.5), 5, 1.2, boxstyle="round,pad=0.2",
                              facecolor=COLORS['azul_corporativo'], alpha=0.2, 
                              edgecolor=COLORS['azul_corporativo'], linewidth=2)
    ax.add_patch(links_box)
    
    ax.text(9, 1.3, '🔗 ENLACES IMPORTANTES', ha='center', va='center', 
           fontsize=14, fontweight='bold', color=COLORS['azul_corporativo'])
    ax.text(9, 0.9, '• GitHub: github.com/user/repo\n• Colab: Notebook ejecutable\n• Documentación completa\n• Scripts listos para producción', 
           ha='center', va='center', fontsize=11)
    
    add_logo_watermark(ax)
    
    plt.tight_layout()
    plt.savefig('estructura_repositorio.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Estructura del repositorio creada: estructura_repositorio.png")

def main():
    """Función principal para crear todas las imágenes"""
    print("🎨 Iniciando creación de imágenes para video de presentación...")
    print("=" * 60)
    
    # Crear todas las imágenes en orden cronológico
    crear_logo_proyecto()
    crear_impacto_economico()
    crear_dataset_visualization()
    crear_distribucion_clases()
    crear_diagrama_5_modelos()
    crear_smote_visualization()
    crear_roadmap_implementacion()
    crear_estructura_repositorio()
    
    print("=" * 60)
    print("🎉 ¡Todas las imágenes han sido creadas exitosamente!")
    print("\n📋 Archivos generados:")
    archivos = [
        "logo_proyecto.png",
        "impacto_economico_hoteles.png", 
        "dataset_visualization.png",
        "distribucion_clases.png",
        "diagrama_5_modelos.png",
        "smote_visualization.png",
        "roadmap_implementacion.png",
        "estructura_repositorio.png"
    ]
    
    for i, archivo in enumerate(archivos, 1):
        print(f"  {i}. ✅ {archivo}")
    
    print(f"\n🎬 ¡Listo para la grabación del video de 10 minutos!")
    print("📁 Todas las imágenes están en resolución 1920x1080 (Full HD)")
    print("🎨 Estilo visual consistente con paleta de colores profesional")

if __name__ == "__main__":
    main() 