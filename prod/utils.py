import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

CLASS_LABELS = {
    'nv': 'Nevus melanocítico',
    'mel': 'Melanoma',
    'bkl': 'Lesiones benignas tipo queratosis',
    'bcc': 'Carcinoma basocelular',
    'akiec': 'Queratosis actínica',
    'vasc': 'Lesiones vasculares',
    'df': 'Dermatofibroma'
}

def obtener_recomendacion(clase_predicha):
    if clase_predicha in ['mel', 'bcc', 'akiec']:
        return "⚠️ *Se recomienda visitar a un dermatólogo lo antes posible para una evaluación profesional.*"
    elif clase_predicha in ['nv', 'bkl', 'df', 'vasc']:
        return "🟢 *En general, esta lesión suele ser benigna, pero si nota cambios, consulte a su médico.*"
    else:
        return "❓ *No se pudo determinar la recomendación. Consulte a su médico ante dudas.*"

def load_model(model_path, device):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)




