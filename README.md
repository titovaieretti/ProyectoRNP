# ProyectoFinalRedesNeuronales
Trabajo final de la materia "Redes Neuronales profundas" UTN-FRM - Roberto Vaieretti

# Clasificador de Cáncer de Piel con EfficientNet-B0

Este proyecto utiliza una red neuronal fine-tuned basada en EfficientNet-B0 para clasificar imágenes de lesiones cutáneas en 7 clases posibles.

## Estructura del Proyecto
 - app.py : Aplicacion Streamlit
 - Utils.py : Funciones necesarias para el funcionamiento de la aplicacion
 - model.ipynb : Notebook que contiene el proceso de creacion de datasets y entrenamiento de los modelos con fine tuning.
 - efficientnet_b0.pth : Red efficientnet_b0 con fine tuning entrenada
 - requirements.txt : Requerimientos necesarios para la ejecucion de la aplicacion

## Cómo ejecutar la aplicación
```bash
pip install -r requirements.txt
cd prod
streamlit run app.py

