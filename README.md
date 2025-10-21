
# Bank Marketing – Flujo Orquestado con Prefect

Este proyecto implementa un flujo completo de Machine Learning para predecir si un cliente aceptará una oferta bancaria, usando **Prefect 2.0** para la orquestación y monitoreo del pipeline.

## Integrantes
- Nico Castañeda
- Alejandro Gómez
- Caren Piñeros

## ¿Qué hace el pipeline?

El flujo principal (`main_flow`) realiza automáticamente:

1. **Carga de datos** desde el repositorio UCI.
2. **Preprocesamiento**: codificación de variables categóricas, escalado, SMOTE para balanceo.
3. **Entrenamiento baseline** – Decision Tree.
4. **Tuning con búsqueda aleatoria y barras de progreso** de:
   - Red Neuronal (Keras + SciKeras)
   - Random Forest
   - XGBoost
5. **Evaluación en test** y comparación de métricas.
6. **Guardado automático en carpeta `artifacts/`** de:
   - Modelos entrenados
   - Mejores hiperparámetros
   - Resumen de métricas (`summary.json`)
7. **Log completo de la ejecución** almacenado por ti en `Log_prefect_flow.txt`.

---

## Cómo ejecutarlo

### 1. Clonar el repositorio y entrar a la carpeta
git clone https://github.com/NicoCastaneda/proyecto-ml.git
cd proyecto-ml

### 2. Crear entorno e instalar dependencias
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### 3. Iniciar Prefect
prefect orion start

### 4. Ejecutar el flujo principal
python main.py

🔹 Al finalizar, todos los modelos, métricas e hiperparámetros se guardan en la carpeta `artifacts/`.

🔹 El log completo de ejecución manual fue guardado por ti en:  
**`Log_prefect_flow.txt`**
