
import torch
from src.datasets.dataset import SteganalysisDataset

dataset = SteganalysisDataset(
    r"data\BOSSbase\split\train",
    payload_bpps=[0.1, 0.2, 0.3, 0.4],
    image_size=256,
    seed=42,
    deterministic=True,
)

print("Fixed-pair signal check\n")

for i in range(8):
    cover = dataset[2 * i]
    stego = dataset[2 * i + 1]

    c = cover["image"]
    s = stego["image"]
    mask = stego["hard_mask"].bool()

    # Compare the cover and stego at the exact modified locations.
    delta = s - c
    signed_changes = delta[mask]

    print(
        f"Pair {i}: "
        f"changed_pixels={int(mask.sum())} | "
        f"positive_changes={(signed_changes > 0).sum().item()} | "
        f"negative_changes={(signed_changes < 0).sum().item()} | "
        f"mean_signed_change={signed_changes.mean().item():.6f} | "
        f"mean_abs_change={signed_changes.abs().mean().item():.6f}"
    )
