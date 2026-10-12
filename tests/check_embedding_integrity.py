
import numpy as np

from src.datasets.dataset import SteganalysisDataset

COVER_DIR = r"data\BOSSbase\split\train"
PAIRS = 100
PAYLOAD = 0.4

dataset = SteganalysisDataset(
    cover_dir=COVER_DIR,
    payload_bpps=(PAYLOAD,),
    image_size=256,
    seed=42,
    deterministic=True,
)

total_changed = 0
total_pixels = 0
mask_mismatches = 0
boundary_mismatches = 0
changed_counts = []

for i in range(PAIRS):
    cover = dataset[2 * i]
    stego = dataset[2 * i + 1]

    # Convert normalized tensors back to their original pixel scale.
    c = np.rint((cover["image"].squeeze(0).numpy() + 1) * 127.5)
    s = np.rint((stego["image"].squeeze(0).numpy() + 1) * 127.5)
    mask = stego["hard_mask"].squeeze(0).numpy()

    actual = c != s
    changed = int(actual.sum())

    total_changed += changed
    total_pixels += actual.size
    changed_counts.append(changed)

    mask_mismatches += int(np.count_nonzero(actual != (mask > 0)))

    # Changes must not move pixels outside the valid grayscale range.
    boundary_mismatches += int(
        np.count_nonzero((c == 0) & (s != 0))
        + np.count_nonzero((c == 255) & (s != 255))
    )

print(f"Payload requested: {PAYLOAD} bpp")
print(f"Pairs checked: {PAIRS}")
print(f"Mean changed pixels per image: {np.mean(changed_counts):.1f}")
print(f"Minimum changed pixels: {min(changed_counts)}")
print(f"Maximum changed pixels: {max(changed_counts)}")
print(f"Overall changed-pixel fraction: {total_changed / total_pixels:.6f}")
print(f"Mask mismatches: {mask_mismatches}")
print(f"Boundary mismatches: {boundary_mismatches}")
