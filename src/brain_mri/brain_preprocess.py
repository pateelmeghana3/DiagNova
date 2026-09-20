"""
DiagNova
Brain MRI Preprocessing

Author: Pateel Meghana
"""

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset Paths

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

# Image Transformations

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Load Dataset

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

# Data Loaders

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Class Names

classes = train_dataset.classes

# Verify

if __name__ == "__main__":

    print("=" * 50)
    print("Brain MRI Dataset Loaded Successfully")
    print("=" * 50)

    print(f"Training Images : {len(train_dataset)}")
    print(f"Testing Images  : {len(test_dataset)}")

    print("\nClasses:")

    for i, cls in enumerate(classes):
        print(f"{i} -> {cls}")

    print("=" * 50)