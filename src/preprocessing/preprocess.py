"""
DiagNova
Dataset Preprocessing Module

Author: Pateel Meghana
"""

import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# -----------------------------
# Configuration
# -----------------------------

IMAGE_SIZE = 224
BATCH_SIZE = 32

# Project Root Directory
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Dataset Paths
dataset_path = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "chest_xray"
)

train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
test_path = os.path.join(dataset_path, "test")


# -----------------------------
# Image Transformations
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


# -----------------------------
# DataLoaders
# -----------------------------

def get_dataloaders(batch_size=BATCH_SIZE):

    # Print paths for debugging
    print("Train Path:", train_path)
    print("Validation Path:", val_path)
    print("Test Path:", test_path)

    train_dataset = ImageFolder(train_path, transform=train_transform)
    val_dataset = ImageFolder(val_path, transform=test_transform)
    test_dataset = ImageFolder(test_path, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader