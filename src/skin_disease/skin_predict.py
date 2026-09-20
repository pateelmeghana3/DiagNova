"""
DiagNova
Skin Disease Prediction Module

Author: Pateel Meghana
"""

import os

import torch
from torchvision import transforms
from PIL import Image

from src.skin_disease.skin_model import SkinCNN


# --------------------------------------------------
# Class Names
# --------------------------------------------------

CLASS_NAMES = [
    "Actinic keratosis",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Vascular lesion"
]


# --------------------------------------------------
# Image Configuration
# --------------------------------------------------

IMAGE_SIZE = 224


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Model Path
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "skin_disease",
    "skin_cnn.pth"
)


# --------------------------------------------------
# Image Transformation
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = SkinCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_skin_image(uploaded_file):

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Apply preprocessing
    image_tensor = transform(
        image
    )


    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)


    # Prediction
    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            1
        )


    # Get class name
    prediction = CLASS_NAMES[
        predicted_class.item()
    ]


    # Convert confidence to percentage
    confidence = (
        confidence.item() * 100
    )


    return prediction, confidence