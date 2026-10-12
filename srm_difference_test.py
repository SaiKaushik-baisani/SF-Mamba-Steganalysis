import torch

from src.models.residual_stem import ResidualStem
from src.datasets.dataset import SteganalysisDataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=[0.4],
    image_size=256,
    seed=42,
)

stem = ResidualStem(
    out_channels=32,
    learnable=False,
).to(DEVICE)

stem.eval()


print("SRM FILTER DIFFERENCE TEST")
print("=" * 60)


for idx in range(10):

    # -----------------------------
    # Generate cover
    # -----------------------------

    while True:
        sample = dataset[idx]

        if sample["y_det"].item() == 0:
            break

    cover = sample["image"].unsqueeze(0).to(DEVICE)


    # -----------------------------
    # Generate stego
    # -----------------------------

    while True:
        sample = dataset[idx]

        if sample["y_det"].item() == 1:
            break

    stego = sample["image"].unsqueeze(0).to(DEVICE)


    # -----------------------------
    # Raw image difference
    # -----------------------------

    image_diff = (
        stego - cover
    ).abs()


    # -----------------------------
    # Fixed SRM residuals
    # -----------------------------

    with torch.no_grad():

        cover_r = stem.hp_conv(cover)
        stego_r = stem.hp_conv(stego)

        cover_r = torch.clamp(
            cover_r,
            -stem.clip_value,
            stem.clip_value,
        )

        stego_r = torch.clamp(
            stego_r,
            -stem.clip_value,
            stem.clip_value,
        )

        residual_diff = (
            stego_r - cover_r
        ).abs()


    print(
        f"Pair {idx:2d}:"
    )

    print(
        f"  image mean diff    = "
        f"{image_diff.mean().item():.8f}"
    )

    print(
        f"  image max diff     = "
        f"{image_diff.max().item():.8f}"
    )

    print(
        f"  SRM mean diff      = "
        f"{residual_diff.mean().item():.8f}"
    )

    print(
        f"  SRM max diff       = "
        f"{residual_diff.max().item():.8f}"
    )

    print(
        f"  SRM changed values = "
        f"{(residual_diff > 0).sum().item()}"
    )

    print()