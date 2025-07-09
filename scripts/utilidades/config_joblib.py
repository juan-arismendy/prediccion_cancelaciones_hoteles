#!/usr/bin/env python3
"""
Configuración para evitar warnings de joblib
"""

import os
import joblib
from sklearn.utils import parallel_backend

def configure_joblib_for_stability():
    """
    Configura joblib para evitar warnings de timeout
    """
    # Configurar variables de entorno para joblib
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
    os.environ['JOBLIB_MAX_NBYTES'] = '100M'  # Límite de memoria por worker
    
    print("✅ Configuración de joblib aplicada para estabilidad")

def get_optimized_gridsearch_params():
    """
    Retorna parámetros optimizados para GridSearchCV
    """
    return {
        'n_jobs': 2,  # Procesamiento limitado
        'verbose': 1,
        'cv': 5,
        'scoring': 'f1'
    }

# Configuración automática al importar
configure_joblib_for_stability() 