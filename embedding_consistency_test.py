import numpy as np

from src.datasets.dataset import SteganalysisDataset


TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=[0.4],
    image_size=256,
    seed=42,
)

idx = 0

print("Testing dataset index:", idx)
print()

samples = []

for i in range(5):
    sample = dataset[idx]

    image = sample["image"].numpy()
    label = float(sample["y_det"].item())
    hard_mask = sample["hard_mask"].numpy()
    soft_mask = sample["soft_mask"].numpy()

    samples.append(sample)

    print(
        f"Sample {i+1}: "
        f"label={label:.0f}, "
        f"hard_pixels={np.sum(hard_mask > 0)}, "
        f"soft_sum={soft_mask.sum():.2f}, "
        f"image_min={image.min():.3f}, "
        f"image_max={image.max():.3f}"
    )

print()
print("=" * 60)
print("PAIRWISE IMAGE DIFFERENCES")
print("=" * 60)

for i in range(1, len(samples)):
    a = samples[i - 1]["image"].numpy()
    b = samples[i]["image"].numpy()

    diff = np.abs(a - b)

    print(
        f"{i} -> {i+1}: "
        f"mean_abs_diff={diff.mean():.8f}, "
        f"max_diff={diff.max():.8f}, "
        f"changed_pixels={np.sum(diff > 1e-6)}"
    )

print()
print("=" * 60)
print("SAME-COVER DATASET TEST COMPLETE")
print("=" * 60)