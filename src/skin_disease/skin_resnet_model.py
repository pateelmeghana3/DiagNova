"""
DiagNova
Skin Disease Transfer Learning Model

ResNet18 + Transfer Learning

Author: Pateel Meghana
"""

import torch.nn as nn
from torchvision import models


class SkinResNet18(nn.Module):

    def __init__(self, num_classes=8):

        super().__init__()

        # --------------------------------------------------
        # Load Pretrained ResNet18
        # --------------------------------------------------

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )


        # --------------------------------------------------
        # Freeze Early Layers
        # --------------------------------------------------

        for param in self.model.parameters():
            param.requires_grad = False


        # --------------------------------------------------
        # Replace Final Classification Layer
        # --------------------------------------------------

        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(

            nn.Dropout(0.4),

            nn.Linear(
                num_features,
                num_classes
            )
        )


    # --------------------------------------------------
    # Forward Pass
    # --------------------------------------------------

    def forward(self, x):

        return self.model(x)