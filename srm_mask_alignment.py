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


print("SRM / HILL MASK ALIGNMENT")
print("=" * 60)


for idx in range(10):

    while True:
        sample = dataset[idx]

        if sample["y_det"].item() == 0:
            break

    cover = sample["image"].unsqueeze(0).to(DEVICE)

    while True:
        sample = dataset[idx]

        if sample["y_det"].item() == 1:
            break

    stego = sample["image"].unsqueeze(0).to(DEVICE)
    hard_mask = sample["hard_mask"].unsqueeze(0).to(DEVICE)


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

        # Collapse the 7 SRM filters.
        residual_map = residual_diff.mean(dim=1)


    mask = hard_mask.float().squeeze(1)

    # Pixels where SRM changed.
    changed = residual_map > 0

    # HILL modified pixels.
    modified = mask > 0

    overlap = (
        changed & modified
    ).sum().item()

    modified_count = modified.sum().item()

    changed_count = changed.sum().item()

    # Also allow a 1-pixel neighborhood around the
    # actual modified pixels.
    neighborhood = torch.nn.functional.max_pool2d(
    modified.float().unsqueeze(1),
    kernel_size=3,
    stride=1,
    padding=1,
).squeeze(1) > 0

    neighborhood_overlap = (
        changed & neighborhood
    ).sum().item()

    print(
        f"Pair {idx:2d}: "
        f"modified={modified_count:6d} | "
        f"SRM_changed={changed_count:6d} | "
        f"direct_overlap={overlap:6d} | "
        f"3x3_overlap={neighborhood_overlap:6d}"
    )