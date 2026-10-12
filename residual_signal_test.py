import torch

from src.models.residual_stem import ResidualStem
from src.datasets.dataset import SteganalysisDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=[0.4],
    image_size=256,
    seed=42,
)

# Generate one cover and one stego from the SAME cover.
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

print("Cover shape:", cover.shape)
print("Stego shape:", stego.shape)

print(
    "Input mean abs difference:",
    torch.mean(torch.abs(cover - stego)).item()
)

print(
    "Input max difference:",
    torch.max(torch.abs(cover - stego)).item()
)

# --------------------------------------------------
# Residual stem
# --------------------------------------------------

stem = ResidualStem(
    out_channels=32,
).to(DEVICE)
stem.eval()

with torch.no_grad():

    cover_features = stem(cover)
    stego_features = stem(stego)

feature_diff = torch.abs(cover_features - stego_features)

print()
print("ResidualStem output shape:", cover_features.shape)

print(
    "Feature mean abs difference:",
    feature_diff.mean().item()
)

print(
    "Feature max difference:",
    feature_diff.max().item()
)

print(
    "Feature changed elements:",
    int((feature_diff > 1e-6).sum().item())
)

print(
    "Total feature elements:",
    feature_diff.numel()
)

print(
    "Fraction changed:",
    (feature_diff > 1e-6).float().mean().item()
)

print()
print("=" * 60)
print("RESIDUAL SIGNAL TEST COMPLETE")
print("=" * 60)
