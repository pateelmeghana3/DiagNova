import os
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.models.cnn_model import ChestCNN
from src.preprocessing.preprocess import get_dataloaders


# -----------------------------
# Device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load Test Data
# -----------------------------

_, _, test_loader = get_dataloaders()


# -----------------------------
# Load Model
# -----------------------------

model = ChestCNN().to(device)

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

model_path = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "chest_cnn.pth"
)

model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()


# -----------------------------
# Evaluation
# -----------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())


# -----------------------------
# Metrics
# -----------------------------

accuracy = accuracy_score(all_labels, all_predictions)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"\nTest Accuracy : {accuracy * 100:.2f}%")

print("\nClassification Report\n")

print(classification_report(
    all_labels,
    all_predictions,
    target_names=["NORMAL", "PNEUMONIA"]
))

print("Confusion Matrix\n")

print(confusion_matrix(
    all_labels,
    all_predictions
))