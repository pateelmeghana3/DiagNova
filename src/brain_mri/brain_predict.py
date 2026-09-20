"""
DiagNova
Brain MRI Prediction Module

Author: Pateel Meghana
"""

import os

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from src.brain_mri.brain_model import BrainCNN


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
    "brain_cnn.pth"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = BrainCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# --------------------------------------------------
# Image Transform
# Same preprocessing used during testing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# --------------------------------------------------
# Brain MRI Classes
# --------------------------------------------------

classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_brain_image(uploaded_file):

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Apply preprocessing
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Move to device
    image = image.to(device)

    # Prediction
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

    # Get predicted class
    predicted_class = classes[
        predicted.item()
    ]

    # Convert confidence to percentage
    confidence_score = (
        confidence.item() * 100
    )

    return predicted_class, confidence_score