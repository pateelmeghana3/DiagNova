"""
DiagNova
Brain MRI ResNet18 Training

Author: Pateel Meghana
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

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
print("DiagNova - Brain MRI ResNet18 Training")
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

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "brain_mri",
    "Training"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "brain_mri",
    "Testing"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "brain_mri"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "brain_resnet18.pth"
)


# --------------------------------------------------
# Image Transformations
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


# --------------------------------------------------
# Data Loaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

classes = train_dataset.classes

print()
print(f"Training Images : {len(train_dataset)}")
print(f"Testing Images  : {len(test_dataset)}")

print()
print("Classes:")

for i, cls in enumerate(classes):
    print(f"{i} -> {cls}")


# --------------------------------------------------
# Model
# --------------------------------------------------

model = BrainResNet(
    num_classes=len(classes)
)

model = model.to(device)


# --------------------------------------------------
# Loss Function
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = optim.Adam(
    model.model.fc.parameters(),
    lr=0.001
)


# --------------------------------------------------
# Training
# --------------------------------------------------

num_epochs = 10

print()
print("=" * 60)
print("Starting Training")
print("=" * 60)


for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    epoch_accuracy = (
        100 * correct / total
    )

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

print()
print("=" * 60)
print("Evaluating ResNet18")
print("=" * 60)

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


test_accuracy = (
    100 * correct / total
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)


# --------------------------------------------------
# Save Model
# --------------------------------------------------

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print()
print("=" * 60)
print("Training Completed Successfully!")
print("=" * 60)

print(
    f"Model saved at:\n{MODEL_PATH}"
)