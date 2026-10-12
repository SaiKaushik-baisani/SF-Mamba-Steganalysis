import torch

from src.models.residual_stem import ResidualStem
from src.models.local_branch import LocalBranch
from src.models.mamba_branch import VSSBlock2D
from src.models.fusion import GatedFusion
from src.datasets.dataset import SteganalysisDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=[0.4],
    image_size=256,
    seed=42,
)

# --------------------------------------------------
# Generate SAME-cover cover/stego pair
# --------------------------------------------------

cover_sample = None
stego_sample = None

while cover_sample is None or stego_sample is None:

    sample = dataset[0]

    if sample["y_det"].item() == 0:
        cover_sample = sample
    else:
        stego_sample = sample

cover = cover_sample["image"].unsqueeze(0).to(DEVICE)
stego = stego_sample["image"].unsqueeze(0).to(DEVICE)

print("Using device:", DEVICE)

# --------------------------------------------------
# Build branches
# --------------------------------------------------

stem = ResidualStem(
    out_channels=32
).to(DEVICE)

local_branch = LocalBranch(
    in_channels=32
).to(DEVICE)

global_branch = VSSBlock2D(
    in_channels=32,
    hidden_dim=64,
    pre_downsample=8,
    n_blocks=4,
    d_state=16
).to(DEVICE)

fusion = GatedFusion(
    local_channels=64,
    global_channels=64,
    out_channels=96
).to(DEVICE)

stem.eval()
local_branch.eval()
global_branch.eval()
fusion.eval()

# --------------------------------------------------
# Forward
# --------------------------------------------------

with torch.no_grad():

    cover_stem = stem(cover)
    stego_stem = stem(stego)

    cover_local = local_branch(cover_stem)
    stego_local = local_branch(stego_stem)

    cover_global = global_branch(cover_stem)
    stego_global = global_branch(stego_stem)

    # Match global resolution to local branch
    cover_global = torch.nn.functional.interpolate(
        cover_global,
        size=cover_local.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    stego_global = torch.nn.functional.interpolate(
        stego_global,
        size=stego_local.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    cover_fused = fusion(
        cover_local,
        cover_global
    )

    stego_fused = fusion(
        stego_local,
        stego_global
    )


# --------------------------------------------------
# Difference helper
# --------------------------------------------------

def report(name, a, b):

    diff = torch.abs(a - b)

    print()
    print(name)
    print("-" * 40)

    print("Shape:", tuple(a.shape))

    print(
        "Mean abs difference:",
        diff.mean().item()
    )

    print(
        "Max difference:",
        diff.max().item()
    )

    print(
        "Changed elements:",
        int((diff > 1e-6).sum().item())
    )

    print(
        "Total elements:",
        diff.numel()
    )

    print(
        "Fraction changed:",
        (diff > 1e-6).float().mean().item()
    )


# --------------------------------------------------
# Results
# --------------------------------------------------

report(
    "Local Branch",
    cover_local,
    stego_local
)

report(
    "SF-Mamba Global Branch",
    cover_global,
    stego_global
)

report(
    "Gated Fusion",
    cover_fused,
    stego_fused
)

print()
print("=" * 60)
print("FUSION SIGNAL TEST COMPLETE")
print("=" * 60)