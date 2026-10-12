import numpy as np

from src.datasets.dataset import SteganalysisDataset
from src.steganography.hill import compute_hill_cost
from src.steganography.embedding import (
    simulate_embedding,
    _solve_lambda,
    _capacity_bits,
)


TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=[0.4],
    image_size=256,
    seed=42,
)

TARGET_BPP = 0.4

print("HILL EMBEDDING STATISTICS")
print("=" * 70)

for idx in range(10):

    cover = np.array(
        __import__("PIL").Image.open(
            dataset.paths[idx]
        ).convert("L"),
        dtype=np.uint8,
    )

    rho = compute_hill_cost(cover)

    H, W = cover.shape

    target_bits = TARGET_BPP * H * W

    lam = _solve_lambda(
        rho,
        target_bits,
    )

    expected_capacity = _capacity_bits(
        rho,
        lam,
    )

    stego, mask, change, p_map = simulate_embedding(
        cover,
        rho,
        TARGET_BPP,
        dataset.rng,
    )

    n_minus = int((change == -1).sum())
    n_zero = int((change == 0).sum())
    n_plus = int((change == 1).sum())

    total = change.size

    actual_changed = n_minus + n_plus

    actual_change_rate = (
        actual_changed / total
    )

    # Empirical ternary entropy.
    probs = np.array([
        n_minus / total,
        n_zero / total,
        n_plus / total,
    ])

    empirical_entropy = 0.0

    for p in probs:
        if p > 0:
            empirical_entropy -= p * np.log2(p)

    print(f"Image {idx:2d}")
    print(f"  target bpp              : {TARGET_BPP:.6f}")
    print(f"  solved lambda           : {lam:.6f}")
    print(f"  expected capacity       : {expected_capacity:.3f} bits")
    print(f"  target bits             : {target_bits:.3f}")
    print(f"  actual changed pixels   : {actual_changed}")
    print(f"  actual change rate      : {actual_change_rate:.6f}")
    print(f"  -1 changes              : {n_minus}")
    print(f"   0 changes              : {n_zero}")
    print(f"  +1 changes              : {n_plus}")
    print(f"  empirical entropy       : {empirical_entropy:.6f}")
    print()