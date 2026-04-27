# Predicción de precios de viviendas

Proyecto de análisis de datos y machine learning enfocado en la predicción de precios de viviendas usando Python.

## Objetivo del proyecto

El objetivo de este proyecto es construir un modelo de machine learning capaz de predecir el precio de venta de una vivienda a partir de sus características.

El proyecto sigue una línea de trabajo de data science: carga de datos, limpieza, análisis exploratorio, preparación de variables, entrenamiento de modelos, evaluación e interpretación de la importancia de variables.

## Dataset

El proyecto utiliza un dataset extraído de Kaggle de precios de viviendas con variables numéricas y categóricas relacionadas con características de las propiedades.

## Tecnologías utilizadas

- Python
- pandas
- NumPy
- matplotlib
- scikit-learn
- Jupyter Notebook

## Cronología del proyecto

1. Carga y exploración inicial de datos.
2. Limpieza y preparación de datos.
3. Análisis exploratorio de datos.
4. Modelado.
5. Análisis de importancia de variables.
6. Conclusiones.

## Modelos utilizados

- Regresión lineal
- Random Forest Regressor

## Resultados

Primero se utilizó un modelo de regresión lineal como modelo base. Este modelo mostró muchas limitaciones, especialmente en viviendas de precio alto y en la presencia de errores grandes.

Posteriormente se entrenó un modelo Random Forest Regressor, que obtuvo mejores resultados, reduciendo errores grandes y mejorando el valor de R². Además, este modelo permitió analizar la importancia de las variables para entender mejor qué características influyen más en la predicción del precio.

## Conclusiones

El modelo Random Forest mostró un mejor rendimiento que la regresión lineal para este problema de regresión.

El proyecto demuestra un flujo completo inicial de machine learning, incluyendo limpieza de datos, análisis exploratorio, comparación de modelos e interpretación de resultados.

## Posibles mejoras futuras

- Aplicar validación cruzada.
- Optimizar hiperparámetros.
- Tratar outliers de forma más detallada.
- Probar modelos más avanzados como Gradient Boosting o XGBoost.
- Mejorar la ingeniería de variables.
