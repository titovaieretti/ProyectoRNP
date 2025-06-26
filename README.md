# ProyectoFinalRedesNeuronales
Trabajo final de la materia "Redes Neuronales profundas" UTN-FRM - Roberto Vaieretti

# Clasificador de Cáncer de Piel con EfficientNet-B0

Este proyecto utiliza una red neuronal fine-tuned basada en EfficientNet-B0 para clasificar imágenes de lesiones cutáneas en 7 clases posibles.

## Estructura del Proyecto
- `data/`: datasets de entrenamiento y prueba.
- `dev/`: notebooks con el desarrollo del modelo (`.ipynb`).
- `prod/`: aplicación en Streamlit, modelo entrenado y scripts auxiliares.

## Cómo ejecutar la aplicación
```bash
pip install -r requirements.txt
streamlit run app.py

