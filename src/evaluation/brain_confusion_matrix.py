import os
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.brain_mri.brain_model import BrainCNN


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
    "brain_cnn.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Transform
# --------------------------------------------------

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# --------------------------------------------------
# Dataset
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
# Predictions
# --------------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

matrix = confusion_matrix(
    all_labels,
    all_predictions
)

print("=" * 60)
print("BRAIN MRI CONFUSION MATRIX")
print("=" * 60)

print(matrix)

print("=" * 60)


# --------------------------------------------------
# Display
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=classes
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "Brain MRI - Custom CNN Confusion Matrix"
)

plt.tight_layout()


# --------------------------------------------------
# Save Image
# --------------------------------------------------

output_path = os.path.join(
    OUTPUT_DIR,
    "brain_cnn_confusion_matrix.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nConfusion matrix saved to:"
)

print(output_path)