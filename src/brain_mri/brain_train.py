"""
DiagNova
Brain MRI CNN Training

Author: Pateel Meghana
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from brain_model import BrainCNN
from brain_preprocess import (
    train_loader,
    test_loader
)

# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device : {device}\n")

# Model

model = BrainCNN().to(device)

# Loss Function

criterion = nn.CrossEntropyLoss()

# Optimizer

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Epochs

EPOCHS = 15

# Training Loop

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = (
        100 * correct / total
    )

    # Validation

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = (
        100 * val_correct / val_total
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Test Acc: {val_accuracy:.2f}%"
    )

# Save Model

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "saved_models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "brain_cnn.pth"
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\n===================================")
print("Brain CNN Model Saved Successfully")
print(MODEL_PATH)
print("===================================")