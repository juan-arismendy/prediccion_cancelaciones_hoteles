#!/usr/bin/env python3
"""
Train vs Validation vs Test Chart Per Model
Each model gets its own subplot showing performance across all data splits
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
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Configuration
import matplotlib
matplotlib.use('Agg')

plt.style.use('default')
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 11

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

def create_per_model_chart(all_results):
    """Create chart with one subplot per model (5 models in 2x3 grid)"""
    
    models = list(all_results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    # Create figure with 2x3 grid for 5 models (one empty subplot)
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle('🎯 Train vs Validation vs Test Performance by Model (All 5 Models)', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Colors for each split
    colors = {
        'Train': '#3498db',      # Blue
        'Validation': '#e74c3c', # Red  
        'Test': '#2ecc71'        # Green
    }
    
    # Process each model
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        model_results = all_results[model_name]
        
        # Prepare data for this model
        train_scores = [model_results['Train'][metric] for metric in metrics]
        val_scores = [model_results['Validation'][metric] for metric in metrics]
        test_scores = [model_results['Test'][metric] for metric in metrics]
        
        # Set up bar positions
        x = np.arange(len(metrics))
        width = 0.25
        
        # Create bars for this model
        bars1 = ax.bar(x - width, train_scores, width, label='Train', 
                      color=colors['Train'], alpha=0.8, edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x, val_scores, width, label='Validation', 
                      color=colors['Validation'], alpha=0.8, edgecolor='black', linewidth=0.8)
        bars3 = ax.bar(x + width, test_scores, width, label='Test', 
                      color=colors['Test'], alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # Customize subplot
        ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title(f'{model_name}', fontweight='bold', fontsize=16, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
        
        # Add value labels on bars
        all_bars = [bars1, bars2, bars3]
        all_scores = [train_scores, val_scores, test_scores]
        
        for bars, scores in zip(all_bars, all_scores):
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{score:.3f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=9, rotation=0)
        
        # Add overfitting indicator
        train_f1 = model_results['Train']['F1-Score']
        val_f1 = model_results['Validation']['F1-Score']
        test_f1 = model_results['Test']['F1-Score']
        
        overfitting_diff = train_f1 - val_f1
        generalization_diff = abs(val_f1 - test_f1)
        
        # Status text
        if overfitting_diff < 0.05:
            status_color = 'green'
            status_text = '✅ Good Fit'
        elif overfitting_diff < 0.1:
            status_color = 'orange'
            status_text = '⚠️ Moderate Overfitting'
        else:
            status_color = 'red'
            status_text = '❌ Severe Overfitting'
        
        # Add status text box
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, 
                fontsize=11, fontweight='bold', color=status_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Add performance summary
        summary_text = f'Train→Val: {overfitting_diff:.3f}\nVal→Test: {generalization_diff:.3f}'
        ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, 
                fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    # Remove the extra subplot (6th position)
    if len(models) < 6:
        fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('visualizaciones/analisis_completo/train_val_test_per_model_complete.png', 
                dpi=300, bbox_inches='tight')
    print("✅ Chart saved: train_val_test_per_model_complete.png")
    plt.close()

def create_summary_comparison_chart(all_results):
    """Create an additional summary chart comparing key metrics across models"""
    
    models = list(all_results.keys())
    
    # Extract F1-Score and AUC-ROC for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('📊 All 5 Models Performance Summary: F1-Score & AUC-ROC', 
                 fontsize=18, fontweight='bold')
    
    # Colors
    colors = {
        'Train': '#3498db',
        'Validation': '#e74c3c', 
        'Test': '#2ecc71'
    }
    
    # F1-Score comparison
    train_f1 = [all_results[m]['Train']['F1-Score'] for m in models]
    val_f1 = [all_results[m]['Validation']['F1-Score'] for m in models]
    test_f1 = [all_results[m]['Test']['F1-Score'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax1.bar(x - width, train_f1, width, label='Train', 
                   color=colors['Train'], alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x, val_f1, width, label='Validation', 
                   color=colors['Validation'], alpha=0.8, edgecolor='black')
    bars3 = ax1.bar(x + width, test_f1, width, label='Test', 
                   color=colors['Test'], alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('F1-Score', fontweight='bold')
    ax1.set_title('F1-Score Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.05)
    
    # Add values on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # AUC-ROC comparison
    train_auc = [all_results[m]['Train']['AUC-ROC'] for m in models]
    val_auc = [all_results[m]['Validation']['AUC-ROC'] for m in models]
    test_auc = [all_results[m]['Test']['AUC-ROC'] for m in models]
    
    bars1 = ax2.bar(x - width, train_auc, width, label='Train', 
                   color=colors['Train'], alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x, val_auc, width, label='Validation', 
                   color=colors['Validation'], alpha=0.8, edgecolor='black')
    bars3 = ax2.bar(x + width, test_auc, width, label='Test', 
                   color=colors['Test'], alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Models', fontweight='bold')
    ax2.set_ylabel('AUC-ROC', fontweight='bold')
    ax2.set_title('AUC-ROC Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.05)
    
    # Add values on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizaciones/analisis_completo/train_val_test_summary_complete.png', 
                dpi=300, bbox_inches='tight')
    print("✅ Summary chart saved: train_val_test_summary_complete.png")
    plt.close()

def main():
    """Main function"""
    print("🎯 GENERATING TRAIN vs VALIDATION vs TEST CHART PER MODEL (ALL 5 MODELS)")
    print("=" * 80)
    
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
    
    # Define all 5 models with optimized parameters
    models = {
        'Logistic Regression': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(random_state=42, C=1, solver='liblinear'))
        ]),
        'KNN': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', KNeighborsClassifier(n_neighbors=11, weights='distance'))
        ]),
        'Random Forest': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=20))
        ]),
        'MLP': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', MLPClassifier(random_state=42, hidden_layer_sizes=(50,), 
                                       max_iter=1000, early_stopping=True, alpha=0.01))
        ]),
        'SVM': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', SVC(random_state=42, probability=True, C=10, kernel='rbf'))
        ])
    }
    
    # Evaluate models
    all_results = {}
    print("\n📊 Evaluating all 5 models...")
    
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
    
    # Create charts
    if all_results:
        print(f"\n📈 Generating charts for {len(all_results)} models...")
        create_per_model_chart(all_results)
        create_summary_comparison_chart(all_results)
        
        print("\n✅ Charts generated successfully!")
        print("📁 Location:")
        print("   - visualizaciones/analisis_completo/train_val_test_per_model_complete.png")
        print("   - visualizaciones/analisis_completo/train_val_test_summary_complete.png")
        
        print(f"\n🎯 Models included: {', '.join(all_results.keys())}")
    else:
        print("❌ Could not evaluate models")

if __name__ == "__main__":
    main() 