"""MallaNet architecture, ported verbatim from the author's repository
(github.com/sahajrajmalla/MallaNet, src/main.py) for baseline reproduction.

Used ONLY to run the published checkpoint for paired statistical comparison
(spec section 7). No architectural inheritance into DevNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.05):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        return out


class HFCLayer(nn.Module):
    def __init__(self, num_classes, D_b):
        super(HFCLayer, self).__init__()
        self.num_classes = num_classes
        self.V = nn.Parameter(torch.randn(num_classes, D_b))
        self.bn = nn.BatchNorm1d(num_classes * D_b)

    def forward(self, x):
        U_b = x.sum(dim=1)
        U_b_exp = U_b.unsqueeze(1)
        V_exp = self.V.unsqueeze(0)
        T_b = U_b_exp * V_exp
        batch_size = T_b.size(0)
        T_b_flat = T_b.view(batch_size, -1)
        T_b_bn = self.bn(T_b_flat)
        T_b_bn = T_b_bn.view(batch_size, self.num_classes, -1)
        T_b_relu = F.relu(T_b_bn)
        logits = T_b_relu.sum(dim=2)
        return logits


class MergingLayer(nn.Module):
    def __init__(self, num_branches=3):
        super(MergingLayer, self).__init__()
        self.w = nn.Parameter(torch.ones(num_branches) / num_branches)

    def forward(self, inputs):
        weights = F.softmax(self.w, dim=0)
        return sum(w * logit for w, logit in zip(weights, inputs))


class BMCNNBase(nn.Module):
    def __init__(self, dropout_rate=0.05):
        super(BMCNNBase, self).__init__()
        self.conv_block1 = nn.Sequential(
            ResidualBlock(1, 128, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(128, 128, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(128, 128, stride=1, dropout_rate=dropout_rate)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv_block2 = nn.Sequential(
            ResidualBlock(128, 256, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(256, 256, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(256, 256, stride=1, dropout_rate=dropout_rate)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv_block3 = nn.Sequential(
            ResidualBlock(256, 512, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(512, 512, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(512, 512, stride=1, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        x1 = self.conv_block1(x)
        x = self.pool1(x1)
        x2 = self.conv_block2(x)
        x = self.pool2(x2)
        x3 = self.conv_block3(x)
        return x1, x2, x3


class EnhancedBMCNNwHFCs(BMCNNBase):
    def __init__(self, num_classes=46, dropout_rate=0.05):
        super(EnhancedBMCNNwHFCs, self).__init__(dropout_rate)
        self.hfc1 = HFCLayer(num_classes, D_b=32*32)
        self.hfc2 = HFCLayer(num_classes, D_b=16*16)
        self.hfc3 = HFCLayer(num_classes, D_b=8*8)
        self.merging = MergingLayer(num_branches=3)

    def forward(self, x):
        x1, x2, x3 = super().forward(x)
        x1_reshaped = x1.view(x1.size(0), x1.size(1), -1)
        logit1 = self.hfc1(x1_reshaped)
        x2_reshaped = x2.view(x2.size(0), x2.size(1), -1)
        logit2 = self.hfc2(x2_reshaped)
        x3_reshaped = x3.view(x3.size(0), x3.size(1), -1)
        logit3 = self.hfc3(x3_reshaped)
        logits = self.merging((logit1, logit2, logit3))
        return logits
