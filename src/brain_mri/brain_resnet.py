"""
DiagNova
Brain MRI ResNet18 Model

Author: Pateel Meghana
"""

import torch.nn as nn
from torchvision import models


class BrainResNet(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Freeze pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final classification layer
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)