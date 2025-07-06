#!/usr/bin/env python3
"""
Train vs Validation vs Test Bar Chart
Three bars per metric showing performance across all data splits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Configuration
import matplotlib
matplotlib.use('Agg')

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12

def load_data():
    """Load and prepare data"""
    print("📊 Loading data...")
    
    df = pd.read_csv('datos/hotel_booking.csv')
    df = df.sample(frac=0.15, random_state=42)
    print(f"Dataset: {df.shape[0]} rows")
    
    # Basic cleaning
    df['children'] = df['children'].fillna(0)
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    df = df[df['adr'] >= 0]
    df = df[df['adults'] + df['children'] + df['babies'] > 0]
    df = df.drop(columns=['reservation_status', 'reservation_status_date'])
    
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    
    # Preprocessor
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    return X, y, preprocessor

def evaluate_model_all_splits(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model on train, validation, and test sets"""
    model.fit(X_train, y_train)
    
    results = {}
    datasets = {
        'Train': (X_train, y_train),
        'Validation': (X_val, y_val),
        'Test': (X_test, y_test)
    }
    
    for split_name, (X_split, y_split) in datasets.items():
        y_pred = model.predict(X_split)
        y_proba = model.predict_proba(X_split)[:, 1]
        
        results[split_name] = {
            'Accuracy': accuracy_score(y_split, y_pred),
            'Precision': precision_score(y_split, y_pred),
            'Recall': recall_score(y_split, y_pred),
            'F1-Score': f1_score(y_split, y_pred),
            'AUC-ROC': roc_auc_score(y_split, y_proba)
        }
    
    return results

def create_train_val_test_bars_chart(all_results):
    """Create train vs validation vs test bar chart"""
    
    models = list(all_results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('📊 Train vs Validation vs Test Performance Comparison', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Colors for each split
    colors = {
        'Train': '#3498db',      # Blue
        'Validation': '#e74c3c', # Red
        'Test': '#2ecc71'        # Green
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data for this metric
        train_scores = [all_results[model]['Train'][metric] for model in models]
        val_scores = [all_results[model]['Validation'][metric] for model in models]
        test_scores = [all_results[model]['Test'][metric] for model in models]
        
        # Set up bar positions
        x = np.arange(len(models))
        width = 0.25
        
        # Create bars
        bars1 = ax.bar(x - width, train_scores, width, label='Train', 
                      color=colors['Train'], alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, val_scores, width, label='Validation', 
                      color=colors['Validation'], alpha=0.8, edgecolor='black', linewidth=0.5)
        bars3 = ax.bar(x + width, test_scores, width, label='Test', 
                      color=colors['Test'], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Models', fontweight='bold', fontsize=12)
        ax.set_ylabel(metric, fontweight='bold', fontsize=12)
        ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits for better visualization
        if metric in ['Accuracy', 'F1-Score', 'AUC-ROC']:
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, max(max(train_scores), max(val_scores), max(test_scores)) * 1.1)
        
        # Add value labels on bars
        all_bars = [bars1, bars2, bars3]
        all_scores = [train_scores, val_scores, test_scores]
        
        for bars, scores in zip(all_bars, all_scores):
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=9)
    
    # Remove the extra subplot
    fig.delaxes(axes[5])
    
    # Add summary table in the last position
    ax_table = fig.add_subplot(2, 3, 6)
    ax_table.axis('off')
    
    # Create summary table
    table_data = []
    for model in models:
        train_f1 = all_results[model]['Train']['F1-Score']
        val_f1 = all_results[model]['Validation']['F1-Score']
        test_f1 = all_results[model]['Test']['F1-Score']
        
        # Calculate overfitting indicators
        train_val_diff = train_f1 - val_f1
        val_test_diff = abs(val_f1 - test_f1)
        
        # Status
        if train_val_diff < 0.05:
            status = '✅ Good'
        elif train_val_diff < 0.1:
            status = '⚠️ Moderate'
        else:
            status = '❌ Overfitting'
        
        table_data.append([
            model,
            f'{train_f1:.3f}',
            f'{val_f1:.3f}',
            f'{test_f1:.3f}',
            f'{train_val_diff:.3f}',
            status
        ])
    
    # Create table
    table = ax_table.table(
        cellText=table_data,
        colLabels=['Model', 'Train F1', 'Val F1', 'Test F1', 'Train-Val Δ', 'Status'],
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the status column
    for i in range(len(models)):
        status = table_data[i][5]
        if '✅' in status:
            table[(i+1, 5)].set_facecolor('#d4edda')  # Light green
        elif '⚠️' in status:
            table[(i+1, 5)].set_facecolor('#fff3cd')  # Light yellow
        else:
            table[(i+1, 5)].set_facecolor('#f8d7da')  # Light red
    
    # Style header
    for j in range(6):
        table[(0, j)].set_facecolor('#6c757d')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax_table.set_title('📋 Performance Summary', fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('visualizaciones/analisis_completo/train_val_test_bars_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("✅ Chart saved: train_val_test_bars_comparison.png")
    plt.close()

def main():
    """Main function"""
    print("🎯 GENERATING TRAIN vs VALIDATION vs TEST BAR CHART")
    print("=" * 60)
    
    # Load data
    X, y, preprocessor = load_data()
    
    # Split data: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Define models with optimized parameters
    models = {
        'Logistic Regression': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(random_state=42, C=1, solver='liblinear'))
        ]),
        'Random Forest': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=20))
        ]),
        'SVM': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', SVC(random_state=42, probability=True, C=10, kernel='rbf'))
        ]),
        'KNN': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', KNeighborsClassifier(n_neighbors=11, weights='distance'))
        ])
    }
    
    # Evaluate models
    all_results = {}
    print("\n📊 Evaluating models...")
    
    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            results = evaluate_model_all_splits(model, X_train, y_train, X_val, y_val, X_test, y_test)
            all_results[name] = results
            
            train_f1 = results['Train']['F1-Score']
            val_f1 = results['Validation']['F1-Score']
            test_f1 = results['Test']['F1-Score']
            print(f"    Train F1: {train_f1:.3f} | Val F1: {val_f1:.3f} | Test F1: {test_f1:.3f}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Create chart
    if all_results:
        print("\n📈 Generating chart...")
        create_train_val_test_bars_chart(all_results)
        print("\n✅ Chart generated successfully!")
        print("📁 Location: visualizaciones/analisis_completo/train_val_test_bars_comparison.png")
    else:
        print("❌ Could not evaluate models")

if __name__ == "__main__":
    main() 