import streamlit as st
import torch
from PIL import Image
import numpy as np
from utils import load_model, preprocess_image, CLASS_LABELS, obtener_recomendacion

DEVICE = torch.device("cpu")
MODEL_PATH = "efficientnet_b0.pth"

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH, DEVICE)

# Título y mensajes informativos
st.title("Clasificación de cáncer de piel - HAM10000")
st.markdown("""
⚠️ **IMPORTANTE:** Esta aplicación NO reemplaza el diagnóstico médico profesional. Ante cualquier duda, consulte a un dermatólogo.

---

## Introducción

La detección temprana del cáncer de piel puede salvar vidas. Los melanomas son tumores malignos que deben tratarse cuanto antes, mientras que muchas lesiones cutáneas (como los nevos o lunares) suelen ser benignas. Sin embargo, toda lesión que cambie de forma, color o tamaño debe ser evaluada por un profesional.

Esta aplicación utiliza una red neuronal para predecir la clase de una lesión de piel a partir de una imagen. Recuerde: **esto no reemplaza la consulta médica**.

---
""")
# Paneles interactivos con las descripciones de cada clase
DESCRIPCIONES_CLASES = {
    'Nevus melanocítico': "Lunar benigno. Los nevos melanocíticos son lesiones cutáneas benignas formadas por acumulación de melanocitos. Suelen ser inofensivos, pero cualquier cambio debe ser evaluado.",
    'Melanoma': "Tumor maligno de las células que producen melanina. Es el tipo más peligroso de cáncer de piel y requiere atención médica urgente.",
    'Lesiones benignas tipo queratosis': "Lesiones no cancerosas, como la queratosis seborreica, que pueden parecerse a verrugas o manchas rugosas.",
    'Carcinoma basocelular': "Cáncer de piel común y de crecimiento lento. Generalmente no se disemina, pero requiere tratamiento para evitar daño local.",
    'Queratosis actínica': "Lesión precancerosa causada por daño solar. Puede evolucionar a carcinoma si no se trata.",
    'Lesiones vasculares': "Incluye hemangiomas o malformaciones de los vasos sanguíneos en la piel. Generalmente benignas.",
    'Dermatofibroma': "Nódulo benigno en la piel, de color marrón o rojizo, habitualmente indoloro y sin gravedad."
}

st.markdown("### ¿Qué significa cada clase?")
for clase, descripcion in DESCRIPCIONES_CLASES.items():
    with st.expander(clase):
        st.write(descripcion)

st.write("Subí una imagen de una lesión para predecir su clase:")

uploaded_file = st.file_uploader("Elegí una imagen", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen cargada", use_column_width=True)

    model = get_model()
    tensor_img = preprocess_image(image)
    with torch.no_grad():
        output = model(tensor_img)
        probs = torch.nn.functional.softmax(output, dim=1)[0]  # 1D tensor
        probabilities = probs.cpu().numpy()  # numpy array

        pred_idx = np.argmax(probabilities)
        pred_class = list(CLASS_LABELS.keys())[pred_idx]
        pred_label = CLASS_LABELS[pred_class]

    st.success(f"**Predicción:** {pred_label} (`{pred_class}`)")
    st.markdown(f"### {obtener_recomendacion(pred_class)}")

    # Mostrar todas las probabilidades
    st.markdown("#### Probabilidades por clase:")
    # Mostrar como tabla
    prob_dict = {CLASS_LABELS[list(CLASS_LABELS.keys())[i]]: f"{probabilities[i]*100:.2f}%" for i in range(len(CLASS_LABELS))}
    st.table(prob_dict)




