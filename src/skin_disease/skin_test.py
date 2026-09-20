"""
DiagNova
Skin Disease CNN Testing

Author: Pateel Meghana
"""

import os
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.skin_disease.skin_model import SkinCNN
from src.skin_disease.skin_preprocess import test_loader


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("DiagNova - Skin Disease CNN Testing")
print("=" * 60)

print(f"\nDevice: {device}")


# --------------------------------------------------
# Project Paths
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


print(f"\nModel loaded from:")
print(MODEL_PATH)


# --------------------------------------------------
# Class Names
# --------------------------------------------------

class_names = test_loader.dataset.classes

print(f"\nClasses: {class_names}")

print(
    f"Test Images: "
    f"{len(test_loader.dataset)}"
)


# --------------------------------------------------
# Testing
# --------------------------------------------------

all_predictions = []
all_labels = []

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Predictions
        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# --------------------------------------------------
# Test Accuracy
# --------------------------------------------------

test_accuracy = (
    correct / total
) * 100


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"\nTest Accuracy : "
    f"{test_accuracy:.2f}%"
)


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names
    )
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

print("\nConfusion Matrix:")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(cm)


# --------------------------------------------------
# Testing Complete
# --------------------------------------------------

print("\n" + "=" * 60)
print("Skin Disease CNN Testing Completed")
print("=" * 60)