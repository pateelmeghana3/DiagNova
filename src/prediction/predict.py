"""
DiagNova
Prediction Module

Author: Pateel Meghana
"""

import os
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from src.models.cnn_model import ChestCNN


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Model Path

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


model_path = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "chest_cnn.pth"
)


# Load Model

model = ChestCNN().to(device)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()


# Image Transform

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


# Classes

classes = [
    "NORMAL",
    "PNEUMONIA"
]


# Prediction Function

def predict_image(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)


    with torch.no_grad():

        outputs = model(image)

        probabilities = F.softmax(
            outputs,
            dim=1
        )


        confidence, predicted = torch.max(
            probabilities,
            1
        )


    predicted_class = classes[
        predicted.item()
    ]


    confidence_score = (
        confidence.item() * 100
    )


    return predicted_class, confidence_score