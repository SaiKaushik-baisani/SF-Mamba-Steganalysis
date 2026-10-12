import torch
import torch.nn as nn

from src.models.residual_stem import ResidualStem


class ResBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()

        out_channels = channels * stride if stride > 1 else channels

        self.conv1 = nn.Conv2d(
            channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.GroupNorm(8, out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None

        if stride > 1 or channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    channels,
                    out_channels,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(8, out_channels),
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)

        out = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        out = self.bn2(
            self.conv2(out)
        )

        return self.relu(out + identity)

class LocalBranch(nn.Module):
    """
    4 residual blocks with one stride-2 downsampling step.
    BatchNorm has been replaced with GroupNorm for the
    small-batch steganalysis diagnostic.
    """

    def __init__(
        self,
        in_channels: int = 32,
        base_channels: int = 64,
        n_blocks: int = 4,
    ):
        super().__init__()

        blocks = [
            ResBlock(in_channels, stride=1),
            ResBlock(in_channels, stride=2),
        ]

        c = in_channels * 2

        for _ in range(n_blocks - 2):
            blocks.append(
                ResBlock(c, stride=1)
            )

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = c

    def forward(self, x):
        return self.blocks(x)


# quick check
_stem = ResidualStem(out_channels=32)
_branch = LocalBranch(in_channels=32)
_x = torch.randn(2, 1, 256, 256)
_f = _branch(_stem(_x))
print("LocalBranch output:", _f.shape, "channels:", _branch.out_channels)
