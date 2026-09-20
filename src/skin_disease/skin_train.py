"""
DiagNova
Skin Disease ResNet18 Transfer Learning

Author: Pateel Meghana
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, WeightedRandomSampler

from src.skin_disease.skin_resnet_model import SkinResNet18

from src.skin_disease.skin_preprocess import (
    train_dataset,
    val_dataset
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DiagNova - Skin Disease ResNet18 Training")
print("=" * 60)

print(f"\nDevice: {device}")


# --------------------------------------------------
# Class Information
# --------------------------------------------------

class_names = train_dataset.classes

num_classes = len(class_names)

print(f"\nNumber of Classes: {num_classes}")

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i} -> {name}")


# --------------------------------------------------
# Class Distribution
# --------------------------------------------------

class_counts = [0] * num_classes

for _, label in train_dataset.samples:
    class_counts[label] += 1


print("\nTraining Distribution:")

for i, count in enumerate(class_counts):
    print(
        f"{class_names[i]} : {count}"
    )


# --------------------------------------------------
# Weighted Random Sampler
# --------------------------------------------------

class_weights = [
    1.0 / count
    for count in class_counts
]


sample_weights = [
    class_weights[label]
    for _, label in train_dataset.samples
]


sample_weights = torch.DoubleTensor(
    sample_weights
)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# --------------------------------------------------
# Data Loaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = SkinResNet18(
    num_classes=num_classes
).to(device)


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
# Learning Rate Scheduler
# --------------------------------------------------

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# --------------------------------------------------
# Training Settings
# --------------------------------------------------

epochs = 10

best_val_accuracy = 0.0


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


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "skin_disease"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "skin_resnet18.pth"
)


# --------------------------------------------------
# Training Loop
# --------------------------------------------------

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(
            images
        )


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


    # --------------------------------------------------
    # Training Accuracy
    # --------------------------------------------------

    train_accuracy = (
        correct / total
    ) * 100


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(
                images
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)


            val_correct += (
                predicted == labels
            ).sum().item()


    # --------------------------------------------------
    # Validation Accuracy
    # --------------------------------------------------

    val_accuracy = (
        val_correct / val_total
    ) * 100


    # --------------------------------------------------
    # Average Loss
    # --------------------------------------------------

    average_loss = (
        running_loss /
        len(train_loader)
    )


    # --------------------------------------------------
    # Learning Rate Scheduler
    # --------------------------------------------------

    scheduler.step(
        val_accuracy
    )


    # --------------------------------------------------
    # Current Learning Rate
    # --------------------------------------------------

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------

    print(
        f"\nEpoch [{epoch + 1}/{epochs}]"
    )


    print(
        f"Loss           : "
        f"{average_loss:.4f}"
    )


    print(
        f"Train Accuracy : "
        f"{train_accuracy:.2f}%"
    )


    print(
        f"Val Accuracy   : "
        f"{val_accuracy:.2f}%"
    )


    print(
        f"Learning Rate  : "
        f"{current_lr:.6f}"
    )


    # --------------------------------------------------
    # Save Best Model
    # --------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print(
            "✓ Best ResNet18 model saved!"
        )


# --------------------------------------------------
# Training Complete
# --------------------------------------------------

print("\n" + "=" * 60)

print(
    "Skin Disease ResNet18 Training Completed"
)


print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.2f}%"
)


print(
    f"Model saved at:\n"
    f"{MODEL_PATH}"
)


print("=" * 60)