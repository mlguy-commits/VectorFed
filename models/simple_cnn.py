"""
SimpleCNN for MNIST (10 classes) and FEMNIST (62 classes).

Architecture:
    Conv1 → ReLU → MaxPool
    Conv2 → ReLU → MaxPool
    FC1   → ReLU → Dropout
    FC2   → output (num_classes)

Input:  (batch, 1, 28, 28)
Output: (batch, num_classes)

Deliberately small for CPU training on VMs.
No BatchNorm (problematic in federated settings due to
local statistics not transferring well during aggregation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Lightweight CNN for grayscale 28×28 image classification.

    Works for both MNIST (num_classes=10) and
    FEMNIST (num_classes=62) by parameterizing the output layer.
    """

    def __init__(self, num_classes: int = 10):
        """
        Args:
            num_classes: Number of output classes
                         10  for MNIST
                         62  for FEMNIST
        """
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        # No BatchNorm: local statistics don't aggregate well in FL
        self.conv1 = nn.Conv2d(
            in_channels  = 1,
            out_channels = 32,
            kernel_size  = 3,
            padding      = 1,
        )
        self.conv2 = nn.Conv2d(
            in_channels  = 32,
            out_channels = 64,
            kernel_size  = 3,
            padding      = 1,
        )

        # MaxPool halves spatial dimensions each time
        # After conv1 + pool: (batch, 32, 14, 14)
        # After conv2 + pool: (batch, 64,  7,  7)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # Flattened size: 64 × 7 × 7 = 3136
        self.fc1     = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2     = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 1, 28, 28)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Block 1: conv → relu → pool
        x = self.pool(F.relu(self.conv1(x)))

        # Block 2: conv → relu → pool
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten: (batch, 64, 7, 7) → (batch, 3136)
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x