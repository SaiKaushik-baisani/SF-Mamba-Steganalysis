import torch
import torch.nn as nn

from src.models.residual_stem import ResidualStem
from src.models.local_branch import LocalBranch
from src.models.mamba_branch import VSSBlock2D
from src.models.fusion import GatedFusion
from src.models.heads import DetectionHead, LocalizationHead

class SFMambaSteganalysis(nn.Module):
    """
    Full pipeline:
      input -> ResidualStem -> {LocalBranch, VSSBlock2D} -> GatedFusion
             -> DetectionHead
             -> LocalizationHead
    """
    def __init__(self,
                 stem_channels: int = 32,
                 local_channels: int = 64,
                 global_hidden: int = 64,
                 fused_channels: int = 96,
                 vss_pre_downsample: int = 8,
                 n_local_blocks: int = 4):
        super().__init__()
        self.stem = ResidualStem(out_channels=stem_channels)
        self.local_branch = LocalBranch(in_channels=stem_channels,
                                         n_blocks=n_local_blocks)
        # global branch runs on the stem output directly (not the downsampled
        # local-branch output) so it sees full-resolution residual evidence,
        # then internally downsamples for scan tractability (see vss_block.py)
        self.global_branch = VSSBlock2D(
            in_channels=stem_channels,
            hidden_dim=global_hidden,
            pre_downsample=vss_pre_downsample,
            n_blocks=4,
            d_state=16
        )
        # local branch downsamples the stem output by 2x once; make the
        # global branch output match that resolution before fusion
        self.local_downsample_factor = 2

        self.fusion = GatedFusion(local_channels=self.local_branch.out_channels,
                                   global_channels=global_hidden,
                                   out_channels=fused_channels)

        self.detection_head = DetectionHead(fused_channels)
        self.localization_head = LocalizationHead(fused_channels,
                                                    upsample_factor=self.local_downsample_factor)

    def forward(self, x):
        # x: (B, 1, H, W)
        H, W = x.shape[-2:]
        stem_out = self.stem(x)                      # (B, stem_c, H, W)

        f_local = self.local_branch(stem_out)         # (B, local_c, H/2, W/2)
        f_global_full = self.global_branch(stem_out)  # (B, global_hidden, H, W)
        f_global = nn.functional.interpolate(
            f_global_full, size=f_local.shape[-2:], mode="bilinear", align_corners=False
        )                                              # match local branch resolution

        fused = self.fusion(f_local, f_global)         # (B, fused_c, H/2, W/2)

        det_logits = self.detection_head(fused)                       # (B,)
        loc_logits = self.localization_head(fused, target_hw=(H, W))  # (B, 1, H, W)
        return det_logits, loc_logits


# quick check
_model = SFMambaSteganalysis()
_n_params = sum(p.numel() for p in _model.parameters())
print(f"Total params: {_n_params:,} ({_n_params/1e6:.2f}M)")
_x = torch.randn(2, 1, 256, 256)
_det, _loc = _model(_x)
print("Detection logits:", _det.shape)
print("Localization logits:", _loc.shape)
