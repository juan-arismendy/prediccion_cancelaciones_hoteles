#!/usr/bin/env python3
"""
Sistema de Persistencia de Modelos - Versión Rápida
Optimizado para evitar warnings de timeout y problemas de memoria
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

class ModelPersistenceManager:
    """
    Gestor de persistencia de modelos para el proyecto de cancelaciones hoteleras
    """
    
    def __init__(self, base_path="saved_models"):
        """
        Inicializa el gestor de persistencia
        
        Args:
            base_path: Directorio base para guardar modelos y resultados
        """
        self.base_path = base_path
        self.models_path = os.path.join(base_path, "models")
        self.results_path = os.path.join(base_path, "results")
        self.metadata_path = os.path.join(base_path, "metadata")
        
        # Crear directorios si no existen
        for path in [self.models_path, self.results_path, self.metadata_path]:
            os.makedirs(path, exist_ok=True)
    
    def save_model(self, model, model_name, version=None):
        """
        Guarda un modelo entrenado
        
        Args:
            model: Modelo entrenado (pipeline completo)
            model_name: Nombre del modelo
            version: Versión del modelo (opcional)
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{model_name}_{version}.pkl"
        filepath = os.path.join(self.models_path, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Modelo {model_name} guardado en: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error guardando modelo {model_name}: {e}")
            return None
    
    def load_model(self, model_name, version=None):
        """
        Carga un modelo guardado
        
        Args:
            model_name: Nombre del modelo
            version: Versión específica (si None, carga la más reciente)
        """
        if version is None:
            # Buscar la versión más reciente
            files = [f for f in os.listdir(self.models_path) if f.startswith(model_name)]
            if not files:
                raise FileNotFoundError(f"No se encontraron modelos para {model_name}")
            
            # Ordenar por fecha y tomar el más reciente
            files.sort(reverse=True)
            filename = files[0]
        else:
            filename = f"{model_name}_{version}.pkl"
        
        filepath = os.path.join(self.models_path, filename)
        
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Modelo {model_name} cargado desde: {filepath}")
            return model
        except Exception as e:
            print(f"❌ Error cargando modelo {model_name}: {e}")
            return None
    
    def save_results(self, results, model_name, version=None):
        """
        Guarda resultados y métricas de evaluación
        
        Args:
            results: Diccionario con resultados del modelo
            model_name: Nombre del modelo
            version: Versión de los resultados
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{model_name}_results_{version}.json"
        filepath = os.path.join(self.results_path, filename)
        
        # Convertir numpy arrays a listas para serialización JSON
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"✅ Resultados de {model_name} guardados en: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error guardando resultados de {model_name}: {e}")
            return None
    
    def load_results(self, model_name, version=None):
        """
        Carga resultados guardados
        
        Args:
            model_name: Nombre del modelo
            version: Versión específica (si None, carga la más reciente)
        """
        if version is None:
            # Buscar la versión más reciente
            files = [f for f in os.listdir(self.results_path) if f.startswith(f"{model_name}_results")]
            if not files:
                raise FileNotFoundError(f"No se encontraron resultados para {model_name}")
            
            files.sort(reverse=True)
            filename = files[0]
        else:
            filename = f"{model_name}_results_{version}.json"
        
        filepath = os.path.join(self.results_path, filename)
        
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            print(f"✅ Resultados de {model_name} cargados desde: {filepath}")
            return results
        except Exception as e:
            print(f"❌ Error cargando resultados de {model_name}: {e}")
            return None
    
    def _make_json_serializable(self, obj):
        """
        Convierte objetos no serializables a JSON a formatos serializables
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Para objetos complejos, convertimos a string
        else:
            return obj
    
    def save_metadata(self, metadata, model_name, version=None):
        """
        Guarda metadatos del entrenamiento (hiperparámetros, configuración, etc.)
        
        Args:
            metadata: Diccionario con metadatos
            model_name: Nombre del modelo
            version: Versión de los metadatos
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{model_name}_metadata_{version}.json"
        filepath = os.path.join(self.metadata_path, filename)
        
        # Hacer serializable el metadata
        serializable_metadata = self._make_json_serializable(metadata)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
            print(f"✅ Metadatos de {model_name} guardados en: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error guardando metadatos de {model_name}: {e}")
            return None
    
    def load_metadata(self, model_name, version=None):
        """
        Carga metadatos guardados
        """
        if version is None:
            files = [f for f in os.listdir(self.metadata_path) if f.startswith(f"{model_name}_metadata")]
            if not files:
                raise FileNotFoundError(f"No se encontraron metadatos para {model_name}")
            
            files.sort(reverse=True)
            filename = files[0]
        else:
            filename = f"{model_name}_metadata_{version}.json"
        
        filepath = os.path.join(self.metadata_path, filename)
        
        try:
            with open(filepath, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadatos de {model_name} cargados desde: {filepath}")
            return metadata
        except Exception as e:
            print(f"❌ Error cargando metadatos de {model_name}: {e}")
            return None
    
    def save_complete_training_session(self, models_dict, results_dict, metadata_dict, session_name=None):
        """
        Guarda una sesión completa de entrenamiento
        
        Args:
            models_dict: Diccionario con modelos entrenados
            results_dict: Diccionario con resultados
            metadata_dict: Diccionario con metadatos
            session_name: Nombre de la sesión
        """
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_path = os.path.join(self.base_path, session_name)
        os.makedirs(session_path, exist_ok=True)
        
        # Guardar todos los modelos
        for model_name, model in models_dict.items():
            self.save_model(model, model_name, session_name)
        
        # Guardar todos los resultados
        for model_name, results in results_dict.items():
            self.save_results(results, model_name, session_name)
        
        # Guardar metadatos de la sesión
        session_metadata = {
            "session_name": session_name,
            "timestamp": datetime.now().isoformat(),
            "models_trained": list(models_dict.keys()),
            "metadata": metadata_dict
        }
        
        session_file = os.path.join(session_path, "session_info.json")
        with open(session_file, 'w') as f:
            json.dump(session_metadata, f, indent=2)
        
        print(f"✅ Sesión completa guardada: {session_name}")
        return session_name
    
    def load_complete_training_session(self, session_name):
        """
        Carga una sesión completa de entrenamiento
        """
        session_path = os.path.join(self.base_path, session_name)
        
        if not os.path.exists(session_path):
            raise FileNotFoundError(f"Sesión {session_name} no encontrada")
        
        # Cargar información de la sesión
        session_file = os.path.join(session_path, "session_info.json")
        with open(session_file, 'r') as f:
            session_info = json.load(f)
        
        # Cargar modelos
        models_dict = {}
        for model_name in session_info["models_trained"]:
            model = self.load_model(model_name, session_name)
            if model is not None:
                models_dict[model_name] = model
        
        # Cargar resultados
        results_dict = {}
        for model_name in session_info["models_trained"]:
            results = self.load_results(model_name, session_name)
            if results is not None:
                results_dict[model_name] = results
        
        print(f"✅ Sesión completa cargada: {session_name}")
        return models_dict, results_dict, session_info
    
    def list_available_sessions(self):
        """
        Lista todas las sesiones disponibles
        """
        sessions = []
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if os.path.isdir(item_path) and item.startswith("session_"):
                sessions.append(item)
        
        return sorted(sessions, reverse=True)
    
    def get_model_info(self, model_name):
        """
        Obtiene información sobre un modelo específico
        """
        model_files = [f for f in os.listdir(self.models_path) if f.startswith(model_name)]
        result_files = [f for f in os.listdir(self.results_path) if f.startswith(f"{model_name}_results")]
        metadata_files = [f for f in os.listdir(self.metadata_path) if f.startswith(f"{model_name}_metadata")]
        
        return {
            "model_files": sorted(model_files, reverse=True),
            "result_files": sorted(result_files, reverse=True),
            "metadata_files": sorted(metadata_files, reverse=True)
        }

def train_and_save_models_fast(X, y, session_name=None):
    """
    Entrena todos los modelos con configuración optimizada para evitar warnings
    
    Args:
        X: DataFrame con características
        y: Serie con variable objetivo
        session_name: Nombre de la sesión (opcional)
    """
    print("🚀 Iniciando entrenamiento rápido y guardado de modelos...")
    
    # Inicializar gestor de persistencia
    persistence_manager = ModelPersistenceManager()
    
    # Configuración de validación cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Identificar características numéricas y categóricas
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
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
    
    # Definir modelos con grids reducidos para evitar problemas de memoria
    models_config = {
        "LogisticRegression": {
            "pipeline": ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', LogisticRegression(random_state=42, solver='liblinear', max_iter=1000))
            ]),
            "param_grid": {
                'classifier__C': [0.1, 1, 10],  # Grid reducido
                'classifier__penalty': ['l2']  # Solo L2 para estabilidad
            }
        },
        "KNN": {
            "pipeline": ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', KNeighborsClassifier())
            ]),
            "param_grid": {
                'classifier__n_neighbors': [5, 7, 9],  # Grid reducido
                'classifier__weights': ['uniform'],  # Solo uniform para velocidad
                'classifier__p': [2]  # Solo distancia euclidiana
            }
        },
        "RandomForest": {
            "pipeline": ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            "param_grid": {
                'classifier__n_estimators': [100],  # Grid muy reducido
                'classifier__max_depth': [10, None],  # Solo 2 opciones
                'classifier__min_samples_leaf': [1, 2],  # Solo 2 opciones
                'classifier__max_features': ['sqrt']  # Solo sqrt
            }
        },
        "MLP": {
            "pipeline": ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', MLPClassifier(random_state=42, max_iter=500, early_stopping=True))
            ]),
            "param_grid": {
                'classifier__hidden_layer_sizes': [(50,)],  # Solo una opción
                'classifier__activation': ['relu'],
                'classifier__alpha': [0.001]  # Solo una opción
            }
        },
        "SVM": {
            "pipeline": ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', SVC(random_state=42, probability=True))
            ]),
            "param_grid": {
                'classifier__C': [1, 10],  # Grid reducido
                'classifier__kernel': ['rbf'],  # Solo RBF
                'classifier__gamma': ['scale']  # Solo scale
            }
        }
    }
    
    # Diccionarios para almacenar resultados
    trained_models = {}
    all_results = {}
    metadata = {
        "dataset_shape": X.shape,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "target_distribution": y.value_counts().to_dict(),
        "training_timestamp": datetime.now().isoformat(),
        "version": "fast"
    }
    
    # Entrenar cada modelo
    for model_name, config in models_config.items():
        print(f"\n--- Entrenando {model_name} (versión rápida) ---")
        
        try:
            # Grid search con configuración optimizada
            grid_search = GridSearchCV(
                config["pipeline"],
                config["param_grid"],
                cv=cv,
                scoring='f1',
                n_jobs=1,  # Procesamiento secuencial para evitar problemas
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            # Guardar el mejor modelo
            best_model = grid_search.best_estimator_
            trained_models[model_name] = best_model
            
            # Evaluar el modelo
            results = evaluate_model_with_cv(best_model, X, y, model_name)
            all_results[model_name] = results
            
            # Guardar metadatos específicos del modelo
            model_metadata = {
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "version": "fast"
            }
            persistence_manager.save_metadata(model_metadata, model_name)
            
            print(f"✅ {model_name} entrenado y guardado exitosamente")
            
        except Exception as e:
            print(f"❌ Error entrenando {model_name}: {e}")
            continue
    
    # Guardar sesión completa
    session_name = persistence_manager.save_complete_training_session(
        trained_models, all_results, metadata, session_name
    )
    
    print(f"\n🎉 Entrenamiento rápido completado. Sesión guardada: {session_name}")
    return session_name, trained_models, all_results

def evaluate_model_with_cv(model, X, y, model_name):
    """
    Evalúa un modelo usando validación cruzada
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    f1_scores, auc_roc_scores = [], []
    accuracy_scores, precision_scores, recall_scores = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        accuracy_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        auc_roc_scores.append(roc_auc_score(y_val, y_proba))
    
    return {
        'F1-Score': np.mean(f1_scores),
        'AUC-ROC': np.mean(auc_roc_scores),
        'Accuracy': np.mean(accuracy_scores),
        'Precision': np.mean(precision_scores),
        'Recall': np.mean(recall_scores),
        'F1-CI': np.std(f1_scores) * 2,
        'AUC-CI': np.std(auc_roc_scores) * 2
    }

def load_trained_models(session_name=None):
    """
    Carga modelos entrenados previamente
    
    Args:
        session_name: Nombre de la sesión (si None, carga la más reciente)
    """
    persistence_manager = ModelPersistenceManager()
    
    if session_name is None:
        available_sessions = persistence_manager.list_available_sessions()
        if not available_sessions:
            raise FileNotFoundError("No se encontraron sesiones guardadas")
        session_name = available_sessions[0]
        print(f"📂 Cargando sesión más reciente: {session_name}")
    
    try:
        models_dict, results_dict, session_info = persistence_manager.load_complete_training_session(session_name)
        print(f"✅ Sesión cargada exitosamente: {session_name}")
        return models_dict, results_dict, session_info
    except Exception as e:
        print(f"❌ Error cargando sesión: {e}")
        return None, None, None

def quick_model_access(model_name, session_name=None):
    """
    Acceso rápido a un modelo específico
    
    Args:
        model_name: Nombre del modelo
        session_name: Sesión específica (opcional)
    """
    persistence_manager = ModelPersistenceManager()
    
    try:
        model = persistence_manager.load_model(model_name, session_name)
        results = persistence_manager.load_results(model_name, session_name)
        metadata = persistence_manager.load_metadata(model_name, session_name)
        
        return model, results, metadata
    except Exception as e:
        print(f"❌ Error cargando modelo {model_name}: {e}")
        return None, None, None

# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 Sistema de Persistencia de Modelos - Versión Rápida")
    print("💡 Optimizado para evitar warnings de timeout y problemas de memoria")
    
    # Verificar si ya existen modelos guardados
    persistence_manager = ModelPersistenceManager()
    available_sessions = persistence_manager.list_available_sessions()
    
    if available_sessions:
        print(f"📁 Sesiones disponibles: {available_sessions}")
        print("💡 Para cargar modelos existentes, usa: load_trained_models()")
    else:
        print("📁 No se encontraron sesiones guardadas")
        print("💡 Para entrenar nuevos modelos rápidos, usa: train_and_save_models_fast(X, y)")
    
    print("\n📋 Funciones disponibles:")
    print("- train_and_save_models_fast(X, y, session_name=None): Entrena y guarda modelos (versión rápida)")
    print("- load_trained_models(session_name=None): Carga modelos guardados")
    print("- quick_model_access(model_name, session_name=None): Acceso rápido a un modelo")
    print("- persistence_manager.list_available_sessions(): Lista sesiones disponibles") 