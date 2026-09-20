"""
DiagNova
Skin Disease Preprocessing Module

Author: Pateel Meghana
"""

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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

DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "skin_disease",
    "final"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

VAL_DIR = os.path.join(
    DATASET_DIR,
    "val"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)


# --------------------------------------------------
# Image Transformations
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# --------------------------------------------------
# Load Datasets
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_test_transform
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

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Classes
# --------------------------------------------------

classes = train_dataset.classes


# --------------------------------------------------
# Class Distribution
# --------------------------------------------------

class_counts = [0] * len(classes)

for _, label in train_dataset.samples:
    class_counts[label] += 1


# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("DiagNova - Skin Disease Dataset")
    print("=" * 60)

    print(
        f"\nTraining Images   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation Images : "
        f"{len(val_dataset)}"
    )

    print(
        f"Testing Images    : "
        f"{len(test_dataset)}"
    )

    print("\nClasses and Training Distribution:")

    for i, cls in enumerate(classes):

        print(
            f"{i} -> {cls} "
            f"({class_counts[i]} images)"
        )

    print("\n" + "=" * 60)
    print("Skin Disease Dataset Loaded Successfully")
    print("=" * 60)