"""
A simplified SRNet-style baseline (detection-only), for the comparison table
your paper needs (Section 7.1 of the spec: "you must compare against SRNet or
a close variant"). This is a faithful-in-spirit, parameter-light version of
SRNet's structure (Boroumand et al., "Deep Residual Network for Steganalysis
of Digital Images," TIFS 2019) -- not a line-for-line reproduction of their
exact layer widths, which is fine for a baseline as long as you report its
param count honestly (see net.py's pattern of printing param counts).

Structure: unpooled high-pass-residual conv layers (types 1) -> residual
blocks without pooling (type 2) -> residual blocks with pooling (type 3) ->
final conv + global pooling -> FC classifier. No high-pass filter bank at the
front here on purpose -- SRNet's whole point is that it learns its own
low-level filters from scratch, unlike Xu-Net/Zhu-Net which hand-initialize
with SRM kernels. That's a real architectural difference worth keeping for a
fair, literature-accurate baseline.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Type 1: conv-BN-ReLU, no residual connection."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlockNoPool(nn.Module):
    """Type 2: two conv-BN layers with a residual add, no spatial reduction."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class ResBlockPool(nn.Module):
    """Type 3: residual block with a strided conv shortcut for downsampling."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SRNetBaseline(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            ConvBlock(1, c),
            ConvBlock(c, c),
        )
        self.no_pool = nn.Sequential(
            ResBlockNoPool(c),
            ResBlockNoPool(c),
            ResBlockNoPool(c),
            ResBlockNoPool(c),
            ResBlockNoPool(c),
        )
        self.pool_stages = nn.Sequential(
            ResBlockPool(c, c * 2),
            ResBlockPool(c * 2, c * 4),
            ResBlockPool(c * 4, c * 8),
            ResBlockPool(c * 8, c * 8),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c * 8, 1)

    def forward(self, x):
        f = self.stem(x)
        f = self.no_pool(f)
        f = self.pool_stages(f)
        z = self.gap(f).flatten(1)
        return self.fc(z).squeeze(-1)  # detection-only, matches DetectionHead's logit convention


# quick check
_srnet = SRNetBaseline()
_n = sum(p.numel() for p in _srnet.parameters())
print(f"SRNetBaseline params: {_n:,} ({_n/1e6:.2f}M)")
print("output logits:", _srnet(torch.randn(2, 1, 256, 256)).shape)
