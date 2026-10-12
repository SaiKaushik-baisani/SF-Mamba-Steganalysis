import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """Global average pool -> FC -> single logit (cover=0 / stego=1)."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x):
        z = self.gap(x).flatten(1)
        return self.fc(z).squeeze(-1)  # (B,) raw logits -- use BCEWithLogitsLoss


class LocalizationHead(nn.Module):
    """Small decoder: upsample back to input resolution, 1x1 conv to 1 channel.
    Outputs raw logits (B, 1, H, W) -- use BCEWithLogitsLoss / Dice on sigmoid(out).
    """
    def __init__(self, in_channels: int, upsample_factor: int, mid_channels: int = 32):
        super().__init__()
        n_steps = max(1, (upsample_factor - 1).bit_length())  # e.g. factor=2 -> 1 step
        layers = []
        c = in_channels
        for _ in range(n_steps):
            layers += [
                nn.Conv2d(c, mid_channels, 3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ]
            c = mid_channels
        layers.append(nn.Conv2d(c, 1, kernel_size=1))
        self.decoder = nn.Sequential(*layers)
        self.upsample_factor = upsample_factor

    def forward(self, x, target_hw):
        out = self.decoder(x)
        if out.shape[-2:] != target_hw:
            out = nn.functional.interpolate(out, size=target_hw, mode="bilinear",
                                             align_corners=False)
        return out  # (B, 1, H, W) logits


# quick check
_fused = torch.randn(2, 96, 128, 128)
_det = DetectionHead(96)
_loc = LocalizationHead(96, upsample_factor=2)
print("Detection logits:", _det(_fused).shape)
print("Localization logits:", _loc(_fused, target_hw=(256, 256)).shape)
