import glob
import numpy as np
from PIL import Image

from src.steganography.hill import compute_hill_cost
from src.steganography.embedding import simulate_embedding


NUM_PAIRS = 20
PAYLOAD = 0.4

paths = sorted(
    glob.glob(r"data\BOSSbase\split\train\*.pgm")
)

rng = np.random.default_rng(42)

print("=" * 60)
print("HILL MASK SANITY TEST")
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

    # --------------------------------------------------------
    # COVER
    # --------------------------------------------------------

    cover_mask_sum = 0

    # --------------------------------------------------------
    # STEGO
    # --------------------------------------------------------

    stego_mask_sum = int(hard_mask.sum())

    # --------------------------------------------------------
    # DIRECT IMAGE DIFFERENCE
    # --------------------------------------------------------

    diff = (
        stego.astype(np.int16)
        - cover.astype(np.int16)
    )

    diff_mask_sum = int(np.count_nonzero(diff))

    print(
        f"Pair {i:2d} | "
        f"cover_mask={cover_mask_sum:5d} | "
        f"stego_mask={stego_mask_sum:5d} | "
        f"diff={diff_mask_sum:5d}"
    )

print()
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("Every cover should have mask sum = 0.")
print("Every stego should have mask sum > 0.")
print("The stego mask sum should equal the image difference count.")
print("=" * 60)