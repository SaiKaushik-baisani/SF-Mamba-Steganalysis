import numpy as np
from PIL import Image

from src.steganography.hill import compute_hill_cost
from src.steganography.embedding import simulate_embedding


import glob

IMAGE_PATH = sorted(
    glob.glob(r"data\BOSSbase\split\train\*.pgm")
)[0]

rng = np.random.default_rng(42)

cover = np.array(
    Image.open(IMAGE_PATH).convert("L"),
    dtype=np.uint8,
)

rho = compute_hill_cost(cover)

stego, hard_mask, change, p_map = simulate_embedding(
    cover,
    rho,
    0.4,
    rng,
)

# Only consider pixels that are actually eligible for modification.
valid = (
    (cover > 0) &
    (cover < 255)
)

modified = (
    hard_mask.astype(bool) &
    valid
)

not_modified = (
    (~hard_mask.astype(bool)) &
    valid
)

modified_costs = rho[modified]
unmodified_costs = rho[not_modified]

print("=" * 60)
print("HILL COST / MODIFICATION ALIGNMENT")
print("=" * 60)

print("Image shape:", cover.shape)
print("Valid pixels:", int(valid.sum()))
print("Modified pixels:", int(modified.sum()))

print()
print("COST STATISTICS")
print("---------------")

print(
    "Modified cost:",
    "mean =", float(modified_costs.mean()),
    "median =", float(np.median(modified_costs)),
)

print(
    "Unmodified cost:",
    "mean =", float(unmodified_costs.mean()),
    "median =", float(np.median(unmodified_costs)),
)

print()
print("COST QUANTILES")
print("---------------")

for q in [0.1, 0.25, 0.5, 0.75, 0.9]:

    mc = np.quantile(modified_costs, q)
    uc = np.quantile(unmodified_costs, q)

    print(
        f"q={q:.2f} | "
        f"modified={mc:.6f} | "
        f"unmodified={uc:.6f}"
    )

print()
print("LOW-COST PIXEL TEST")
print("-------------------")

thresholds = [10, 25, 50, 100, 250, 500]

for threshold in thresholds:

    low_cost = (
        rho <= threshold
    ) & valid

    if low_cost.sum() == 0:
        continue

    modified_in_low_cost = (
        modified & low_cost
    ).sum()

    fraction_modified = (
        modified_in_low_cost /
        modified.sum()
    )

    fraction_available = (
        low_cost.sum() /
        valid.sum()
    )

    print(
        f"cost <= {threshold:6.1f} | "
        f"available={fraction_available:.4f} | "
        f"modified={fraction_modified:.4f}"
    )

print()
print("=" * 60)