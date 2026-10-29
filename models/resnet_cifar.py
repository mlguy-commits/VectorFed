"""
ResNet-8 for CIFAR-10 (10 classes) and CIFAR-100 (100 classes).

Architecture:
    Initial conv (3→16)
    Layer 1: 2 residual blocks, 16 channels
    Layer 2: 2 residual blocks, 32 channels, stride 2
    Layer 3: 2 residual blocks, 64 channels, stride 2
    Global average pooling
    FC → num_classes

Input:  (batch, 3, 32, 32)
Output: (batch, num_classes)

Design choices for federated learning:
    - GroupNorm instead of BatchNorm
      BatchNorm uses local mini-batch statistics which
      do not transfer correctly during FL aggregation.
      GroupNorm uses channel group statistics which are
      independent of batch size and aggregate cleanly.
    - Lightweight (8 layers) for CPU training on VMs
    - Same architecture for CIFAR-10 and CIFAR-100
      (only final FC layer differs via num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Basic residual block with two 3×3 convolutions.

    Uses GroupNorm instead of BatchNorm for FL compatibility.
    Skip connection handles dimension mismatch via 1×1 conv.

    Structure:
        x → Conv → GN → ReLU → Conv → GN → (+x) → ReLU
                                              ↑
                                         shortcut
    """

    # Number of groups for GroupNorm
    # 8 groups works well for channels 16, 32, 64
    # (all divisible by 8)
    GN_GROUPS = 8

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int = 1,
    ):
        """
        Args:
            in_channels:  Number of input channels
            out_channels: Number of output channels
            stride:       Stride for first conv (2 = downsample)
        """
        super(ResidualBlock, self).__init__()

        # First conv + norm
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(
            num_groups   = self.GN_GROUPS,
            num_channels = out_channels,
        )

        # Second conv + norm
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(
            num_groups   = self.GN_GROUPS,
            num_channels = out_channels,
        )

        # Shortcut connection
        # If dimensions change (stride>1 or channel mismatch),
        # use 1×1 conv to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(self.GN_GROUPS, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor

        Returns:
            Output tensor with residual connection applied
        """
        # Main path
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

        # Add shortcut (residual connection)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet-8 for 32×32 RGB image classification.

    Works for both CIFAR-10 (num_classes=10) and
    CIFAR-100 (num_classes=100) by parameterizing output layer.

    Total parameters: ~270K (lightweight for CPU training)
    """

    def __init__(self, num_classes: int = 10):
        """
        Args:
            num_classes: Number of output classes
                         10  for CIFAR-10
                         100 for CIFAR-100
        """
        super(ResNetCIFAR, self).__init__()

        # Initial convolution: 3 → 16 channels
        # Keeps 32×32 spatial size (padding=1)
        self.conv_init = nn.Conv2d(
            in_channels  = 3,
            out_channels = 16,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            bias         = False,
        )
        self.gn_init = nn.GroupNorm(
            num_groups   = 8,
            num_channels = 16,
        )

        # Layer 1: 16 channels, 32×32, 2 blocks, no downsampling
        self.layer1 = self._make_layer(
            in_channels  = 16,
            out_channels = 16,
            num_blocks   = 2,
            stride       = 1,
        )

        # Layer 2: 16→32 channels, 16×16, 2 blocks, stride=2
        self.layer2 = self._make_layer(
            in_channels  = 16,
            out_channels = 32,
            num_blocks   = 2,
            stride       = 2,
        )

        # Layer 3: 32→64 channels, 8×8, 2 blocks, stride=2
        self.layer3 = self._make_layer(
            in_channels  = 32,
            out_channels = 64,
            num_blocks   = 2,
            stride       = 2,
        )

        # Global average pool: (batch, 64, 8, 8) → (batch, 64, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final classifier: 64 → num_classes
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(
        self,
        in_channels:  int,
        out_channels: int,
        num_blocks:   int,
        stride:       int,
    ) -> nn.Sequential:
        """
        Build a layer of residual blocks.

        First block handles stride and channel change.
        Remaining blocks maintain dimensions.

        Args:
            in_channels:  Input channels to this layer
            out_channels: Output channels from this layer
            num_blocks:   Number of residual blocks
            stride:       Stride for first block (downsampling)

        Returns:
            nn.Sequential containing all blocks
        """
        layers = []

        # First block: may change channels and/or spatial size
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks: same channels, stride=1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 3, 32, 32)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Initial conv
        x = F.relu(self.gn_init(self.conv_init(x)))

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global average pool + flatten
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc(x)

        return x