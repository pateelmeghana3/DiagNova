import torch
import torch.nn as nn


class ChestCNN(nn.Module):

    def __init__(self):

        super(ChestCNN, self).__init__()

        # First Convolution Block
        self.conv_block1 = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        # Second Convolution Block
        self.conv_block2 = nn.Sequential(

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        # Flatten
        self.flatten = nn.Flatten()

        # Classifier
        self.classifier = nn.Sequential(

            nn.Linear(
                64 * 56 * 56,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        x = self.conv_block1(x)

        x = self.conv_block2(x)

        x = self.flatten(x)

        x = self.classifier(x)

        return x