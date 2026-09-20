import os
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.brain_mri.brain_resnet import BrainResNet


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DiagNova - Brain MRI ResNet18 Evaluation")
print("=" * 60)

print(f"Device: {device}")


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

TEST_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "brain_mri",
    "Testing"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "brain_mri",
    "brain_resnet18.pth"
)


# --------------------------------------------------
# Test Transform
# --------------------------------------------------

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Load Test Dataset
# --------------------------------------------------

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

classes = test_dataset.classes

print()
print(f"Testing Images : {len(test_dataset)}")

print("\nClasses:")

for i, cls in enumerate(classes):
    print(f"{i} -> {cls}")


# --------------------------------------------------
# Load ResNet18
# --------------------------------------------------

model = BrainResNet(
    num_classes=len(classes)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print()
print("=" * 60)
print("BRAIN MRI RESNET18 EVALUATION")
print("=" * 60)

print(
    f"\nTest Accuracy : {accuracy * 100:.2f}%"
)

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes
    )
)

print("Confusion Matrix\n")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

print("=" * 60)