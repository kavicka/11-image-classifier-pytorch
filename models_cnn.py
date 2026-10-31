"""Models written by hand: a linear baseline, a plain CNN and a CIFAR ResNet-20.

Shapes in the comments are (B, C, H, W) for a 32×32 CIFAR batch.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Linear(nn.Module):
    """Baseline: one matrix from raw pixels to class scores. No notion of neighbouring pixels."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),  # bias is redundant right before BatchNorm
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """Three stages of conv-conv-pool. The depth at which plain stacking still trains without tricks."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),  # (B, 32, 32, 32)
            conv_block(32, 32),
            nn.MaxPool2d(2),  # (B, 32, 16, 16)
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),  # (B, 64, 8, 8)
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2),  # (B, 128, 4, 4)
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x.mean((2, 3)))  # global average pool: one number per channel


class ResidualBlock(nn.Module):
    """out = F(x) + x. The conv layers only learn the correction, so doing nothing is the easy default."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        # shapes differ when the block downsamples or adds channels: a 1×1 conv makes x addable
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet(nn.Module):
    """CIFAR ResNet from He et al. 2015: 6n+2 layers. n=3 gives ResNet-20 with 0.27 M parameters."""

    def __init__(self, n: int = 3, n_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.stage1 = self._stage(16, 16, n, stride=1)  # (B, 16, 32, 32)
        self.stage2 = self._stage(16, 32, n, stride=2)  # (B, 32, 16, 16)
        self.stage3 = self._stage(32, 64, n, stride=2)  # (B, 64, 8, 8)
        self.head = nn.Linear(64, n_classes)

    @staticmethod
    def _stage(c_in: int, c_out: int, n: int, stride: int) -> nn.Sequential:
        blocks = [ResidualBlock(c_in, c_out, stride)]
        blocks += [ResidualBlock(c_out, c_out) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.head(x.mean((2, 3)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
