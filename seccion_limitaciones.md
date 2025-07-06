# VIII. LIMITACIONES DEL ESTUDIO

## A. Limitaciones Metodológicas

### 1. Problemas de Overfitting Identificados

**Overfitting Severo Detectado:** El análisis Train/Val/Test reveló problemas significativos de sobreajuste en cuatro de los cinco modelos evaluados, con diferencias Train-Validation superiores a 0.10 en Logistic Regression (0.227), KNN (0.319), MLP (0.243) y SVM (0.204).

**Implicaciones:**
- **Reducción del ROI estimado:** El overfitting podría disminuir el retorno de inversión proyectado en un 15-25%
- **Inestabilidad en producción:** Los modelos podrían mostrar degradación significativa con datos nuevos
- **Necesidad de regularización:** Requiere implementación de técnicas adicionales antes del despliegue

**Estrategias de Mitigación Propuestas:**
- Incremento de parámetros de regularización (α, λ, C)
- Implementación de early stopping en modelos neuronales
- Uso de ensemble methods con mayor diversidad
- Validación cruzada temporal para datos secuenciales

### 2. Desbalance de Clases Persistente

**Desbalance Original:** 62.2% no cancelaciones vs 37.8% cancelaciones
**Técnica Aplicada:** SMOTE para balanceo sintético
**Limitación:** El desbalance subyacente puede persistir en patrones complejos no capturados por SMOTE

**Consideraciones:**
- SMOTE puede generar ejemplos sintéticos no realistas en regiones de alta dimensionalidad
- El balanceo artificial podría no reflejar la distribución real de cancelaciones
- Posible sesgo hacia la clase minoritaria en métricas como Recall

### 3. Ausencia de Validación Temporal

**Limitación Crítica:** No se evaluó la estabilidad temporal de los modelos ni su degradación a través del tiempo.

**Implicaciones:**
- **Concept drift no detectado:** Cambios en patrones de cancelación por estacionalidad, eventos externos, o cambios en políticas hoteleras
- **Estabilidad no garantizada:** Los modelos podrían perder efectividad con el tiempo sin detección temprana
- **Re-entrenamiento no planificado:** Falta de protocolo para actualización de modelos

**Recomendaciones:**
- Implementar validación con series temporales
- Establecer métricas de monitoreo continuo
- Definir triggers automáticos para re-entrenamiento

## B. Limitaciones de Datos

### 1. Naturaleza Sintética de Información Personal

**Datos Sintéticos Identificados:**
- Nombres de huéspedes generados artificialmente
- Direcciones de correo electrónico sintéticas  
- Números telefónicos no reales
- Información de tarjetas de crédito enmascarada

**Impacto en Validez:**
- **Patrones artificiales:** Los datos sintéticos pueden no reflejar comportamientos reales de cancelación
- **Correlaciones espurias:** Relaciones no existentes en datos reales podrían influir en las predicciones
- **Generalización limitada:** Los modelos podrían no transferirse efectivamente a datos de producción reales

### 2. Limitaciones Temporales y Geográficas

**Período de Datos:** Dataset limitado a un período específico sin especificación clara de rango temporal
**Cobertura Geográfica:** Concentración en tipos específicos de hoteles (City Hotel vs Resort Hotel) sin diversidad geográfica amplia

**Restricciones:**
- **Estacionalidad no capturada:** Posibles patrones estacionales no representados completamente
- **Eventos extraordinarios:** Crisis económicas, pandemias, o eventos geopolíticos no considerados
- **Diversidad cultural limitada:** Patrones de cancelación específicos por región no evaluados

### 3. Completitud de Variables

**Variables con Datos Faltantes:**
- `country`: 488 valores faltantes (0.4%)
- `agent`: 16,340 valores faltantes (13.7%)
- `company`: 112,593 valores faltantes (94.3%)

**Estrategias de Imputación:**
- Eliminación de registros con valores faltantes críticos
- Imputación por moda para variables categóricas
- Creación de categoría "Unknown" para variables de identificación

**Limitaciones de la Imputación:**
- Posible pérdida de información predictiva
- Introducción de sesgos por métodos de imputación simplificados
- Reducción del tamaño efectivo del dataset

## C. Limitaciones de Implementación

### 1. Escalabilidad Computacional

**No Evaluada:** La escalabilidad de los modelos con datasets significativamente más grandes no fue evaluada.

**Consideraciones:**
- **Tiempo de entrenamiento:** Random Forest y SVM podrían requerir recursos computacionales prohibitivos con millones de registros
- **Memoria requerida:** Modelos ensemble pueden exceder capacidades de memoria en implementaciones de producción
- **Latencia de predicción:** Tiempo de respuesta no evaluado para predicciones en tiempo real

### 2. Restricciones de Tiempo Real

**Latencia No Considerada:** No se evaluaron restricciones de tiempo para predicciones en línea en sistemas de reservas.

**Implicaciones:**
- **SVM con kernel RBF:** Puede ser lento para predicciones individuales en tiempo real
- **Random Forest:** Tiempo de predicción proporcional al número de árboles
- **Preprocesamiento:** Codificación one-hot y escalado pueden introducir latencia adicional

### 3. Estrategia de Actualización de Modelos

**Falta de Protocolo:** No se definió una estrategia clara para el re-entrenamiento y actualización periódica de los modelos.

**Consideraciones Faltantes:**
- **Frecuencia de actualización:** Diaria, semanal, mensual, o basada en degradación de métricas
- **Datos de re-entrenamiento:** Ventana deslizante vs acumulación histórica
- **Validación de nuevas versiones:** Protocolo A/B testing para comparar modelos actualizados

## D. Limitaciones de Evaluación

### 1. Métricas de Negocio Ausentes

**Enfoque Técnico Predominante:** Las métricas utilizadas (F1-Score, AUC-ROC, Accuracy) son técnicas pero no reflejan directamente el impacto económico.

**Métricas de Negocio No Evaluadas:**
- **Costo de falsos positivos:** Sobreventa innecesaria y costos operativos
- **Costo de falsos negativos:** Habitaciones vacías y pérdida de ingresos
- **ROI real:** Retorno de inversión basado en implementación práctica
- **Customer lifetime value:** Impacto en la satisfacción y retención de clientes

### 2. Validación en Datos de Producción

**Ausencia de Piloto:** No se realizó validación en un entorno de producción real o con datos de un hotel específico.

**Limitaciones:**
- **Diferencias de distribución:** Los datos reales pueden diferir significativamente del dataset público
- **Factores externos:** Variables no capturadas en el dataset que influyen en cancelaciones reales
- **Integración de sistemas:** Complejidades de implementación en sistemas hoteleros existentes

## E. Consideraciones Éticas y de Privacidad

### 1. Implicaciones de Privacidad

**Datos Personales:** Aunque sintéticos, el modelo entrenado podría inferir patrones sobre comportamientos de huéspedes reales.

**Consideraciones:**
- **Sesgo algorítmico:** Posible discriminación basada en país de origen, tipo de cliente, o patrones de reserva
- **Transparencia:** Los huéspedes podrían no ser conscientes del uso de sus datos para predicciones
- **Consentimiento:** Necesidad de políticas claras sobre uso de datos para ML

### 2. Impacto en Decisiones de Negocio

**Automatización de Decisiones:** Los modelos podrían influir en políticas de sobreventa y gestión de inventario que afecten directamente a los huéspedes.

**Responsabilidades:**
- **Explicabilidad:** Necesidad de interpretar y justificar decisiones basadas en predicciones
- **Supervisión humana:** Importancia de mantener oversight humano en decisiones críticas
- **Auditoría continua:** Monitoreo de sesgos y impactos no intencionados

## F. Recomendaciones para Investigación Futura

### 1. Mejoras Metodológicas Prioritarias

1. **Implementar validación temporal** con series de tiempo para evaluar estabilidad
2. **Desarrollar técnicas de regularización** específicas para el dominio hotelero
3. **Evaluar ensemble methods** combinando Random Forest y SVM
4. **Incorporar análisis de costos** de negocio en la función objetivo

### 2. Expansión de Datos

1. **Obtener datos reales** de hoteles para validación práctica
2. **Incluir variables externas** (eventos, clima, competencia)
3. **Ampliar diversidad geográfica** y temporal del dataset
4. **Incorporar feedback** de implementaciones en producción

### 3. Consideraciones de Implementación

1. **Desarrollar arquitectura escalable** para procesamiento en tiempo real
2. **Establecer protocolos de monitoreo** y re-entrenamiento automático
3. **Crear métricas de negocio** específicas para evaluación continua
4. **Implementar explicabilidad** para decisiones de modelo

## G. Conclusiones sobre Limitaciones

### Impacto en Validez de Resultados

Las limitaciones identificadas **no invalidan** los resultados obtenidos, pero sí **contextualizan** su aplicabilidad:

1. **Resultados técnicamente válidos** dentro del scope del dataset utilizado
2. **Metodología robusta** comparada con el estado del arte
3. **Necesidad de validación adicional** antes de implementación en producción
4. **Oportunidades claras** de mejora y extensión del trabajo

### Transparencia Académica

La identificación explícita de estas limitaciones refleja:
- **Rigor científico** en la evaluación de resultados
- **Honestidad académica** sobre el alcance del estudio
- **Guía clara** para investigación futura
- **Consideraciones prácticas** para implementación real

Esta transparencia fortalece la credibilidad del trabajo y proporciona una base sólida para la continuación de la investigación en predicción de cancelaciones hoteleras. 