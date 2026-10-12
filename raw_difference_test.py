import glob
import numpy as np
from PIL import Image

from src.steganography.hill import compute_hill_cost
from src.steganography.embedding import simulate_embedding


NUM_PAIRS = 10
PAYLOAD = 0.4

paths = sorted(
    glob.glob(r"data\BOSSbase\split\train\*.pgm")
)

rng = np.random.default_rng(42)

print("=" * 60)
print("RAW HILL DIFFERENCE TEST")
print("=" * 60)

for i, path in enumerate(paths[:NUM_PAIRS]):

    cover = np.array(
        Image.open(path).convert("L"),
        dtype=np.uint8
    )

    rho = compute_hill_cost(cover)

    stego, hard_mask, change, _ = simulate_embedding(
        cover,
        rho,
        PAYLOAD,
        rng
    )

    # Exact pixel difference
    diff = (
        stego.astype(np.int16)
        - cover.astype(np.int16)
    )

    changed = diff != 0

    # Only pixels that were actually modified
    changed_values = diff[changed]

    print()
    print(f"PAIR {i}")
    print("-" * 40)

    print("Changed pixels:", int(changed.sum()))

    print(
        "Mean |difference|:",
        float(np.abs(diff).mean())
    )

    print(
        "Max |difference|:",
        int(np.abs(diff).max())
    )

    print(
        "Negative changes:",
        int((changed_values == -1).sum())
    )

    print(
        "Positive changes:",
        int((changed_values == 1).sum())
    )

    print(
        "Other changes:",
        int(
            ((changed_values != -1) &
             (changed_values != 1)).sum()
        )
    )

    # Verify against the embedding mask
    mask_match = np.array_equal(
        changed.astype(np.uint8),
        hard_mask
    )

    print("Difference == hard_mask:", mask_match)

print()
print("=" * 60)