"""Compact pre-activation SE-ResNet for 32x32 single-channel input.

The same class serves as student (default widths) and teacher (wider/deeper).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s[:, :, None, None]


class PreActSEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.se = SqueezeExcite(out_ch)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
            if stride != 1 or in_ch != out_ch
            else None
        )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        sc = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return self.se(out) + sc


class DevNet(nn.Module):
    def __init__(
        self,
        widths: tuple = (40, 80, 160),
        depths: tuple = (2, 2, 2),
        num_classes: int = 46,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Conv2d(1, widths[0], 3, 1, 1, bias=False)
        blocks, in_ch = [], widths[0]
        for stage, (w, d) in enumerate(zip(widths, depths)):
            for j in range(d):
                stride = 2 if (stage > 0 and j == 0) else 1
                blocks.append(PreActSEBlock(in_ch, w, stride))
                in_ch = w
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(in_ch)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = F.relu(self.bn(x)).mean(dim=(2, 3))
        return self.fc(self.drop(x))
