"""
Residual-aware gated fusion (your spec's replacement for SFAM/GFAM).

Rationale: SFAM/GFAM in the original SF-Mamba are built around semantic
foreground/background decoupling (buildings vs. roads vs. clutter), which has
no analog in steganalysis. This gate instead learns, per spatial location, how
much to trust local residual-CNN evidence vs. globally-aggregated evidence --
a much simpler and more directly justifiable mechanism for this task.
"""
import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, local_channels: int, global_channels: int, out_channels: int):
        super().__init__()
        self.local_proj = nn.Conv2d(local_channels, out_channels, 1, bias=False)
        self.global_proj = nn.Conv2d(global_channels, out_channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(local_channels + global_channels, out_channels, 1),
            nn.Sigmoid(),
        )
        self.bn = nn.GroupNorm(8, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, f_local, f_global):
        cat = torch.cat([f_local, f_global], dim=1)
        g = self.gate(cat)  # (B, out_channels, H, W) in (0,1)
        fused = g * self.local_proj(f_local) + (1 - g) * self.global_proj(f_global)
        return self.act(self.bn(fused))


# quick check
_m = GatedFusion(local_channels=64, global_channels=64, out_channels=96)
_fl = torch.randn(2, 64, 128, 128)
_fg = torch.randn(2, 64, 128, 128)
print("GatedFusion output:", _m(_fl, _fg).shape)
