"""
Ternary (+-1) embedding simulator.

This is the standard way steganalysis research generates training data: rather
than running a real STC (syndrome-trellis code) encoder -- which requires an
actual message payload and is computationally heavier -- everyone in the field
(SRNet, Zhu-Net, and the original WOW/S-UNIWARD/HILL papers themselves) uses a
*simulator* that achieves the same rate-distortion trade-off that a real
near-optimal STC encoder would, without needing an actual STC implementation.
This is the correct, standard, published methodology -- not an approximation
you're introducing yourself.

How it works:
  Given a per-pixel cost rho(i,j) (from HILL, WOW, etc.) and a target payload
  alpha (bits per pixel), find a scalar "inverse temperature" lambda such that
  the expected total embedding capacity (in bits) at that lambda equals the
  target payload. Then, independently at each pixel, sample a ternary change
  in {-1, 0, +1} with probability (p, 1-2p, p) where
      p(i,j) = exp(-lambda * rho(i,j)) / (1 + 2*exp(-lambda * rho(i,j)))
  Higher-cost pixels get near-zero p (never touched); lower-cost pixels get p
  approaching 1/3 (heavily used) as payload increases.

This gives you, for free:
  - the stego image (cover + sampled change)
  - the EXACT hard modification mask (change != 0) -- ground truth localization
  - the soft per-pixel probability map, if you want soft localization targets
"""
import numpy as np


def _capacity_bits(rho: np.ndarray, lam: float) -> float:
    """Total expected embedding capacity (bits) at a given lambda."""
    p = np.exp(-lam * rho) / (1.0 + 2.0 * np.exp(-lam * rho))
    p = np.clip(p, 1e-12, (1.0 / 3.0) - 1e-12)  # keep entropy well-defined
    p0 = 1.0 - 2.0 * p
    # ternary entropy per pixel, in bits
    h = -2 * p * np.log2(p) - p0 * np.log2(p0)
    return float(h.sum())


def _solve_lambda(rho: np.ndarray, target_bits: float,
                   lam_lo: float = 1e-6, lam_hi: float = 500.0,
                   n_iters: int = 40) -> float:
    """Binary search for lambda such that capacity(lambda) == target_bits.
    Capacity is monotonically decreasing in lambda (higher lambda -> more
    conservative embedding -> lower capacity), so binary search is valid.
    """
    # make sure bracket is valid; expand hi if needed for very low payloads
    cap_lo = _capacity_bits(rho, lam_lo)
    if target_bits >= cap_lo:
        return lam_lo  # target payload exceeds max achievable capacity at lam_lo

    cap_hi = _capacity_bits(rho, lam_hi)
    tries = 0
    while cap_hi > target_bits and tries < 20:
        lam_hi *= 2
        cap_hi = _capacity_bits(rho, lam_hi)
        tries += 1

    for _ in range(n_iters):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        cap_mid = _capacity_bits(rho, lam_mid)
        if cap_mid > target_bits:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
    return 0.5 * (lam_lo + lam_hi)


def simulate_embedding(cover: np.ndarray, rho: np.ndarray, payload_bpp: float,
                        rng: np.random.Generator = None):
    """
    cover: 2D uint8/float array, grayscale cover image
    rho:   2D float array, same shape, cost map (e.g. from compute_hill_cost)
    payload_bpp: target payload in bits per pixel
    rng: numpy random Generator (pass one in for reproducibility)

    Returns:
      stego:      2D array, same dtype range as cover, modified image
      hard_mask:  2D uint8 array, 1 where pixel was changed, 0 otherwise
      change:     2D int array in {-1, 0, +1}, the actual signed change applied
      p_map:      2D float array, the per-pixel total change probability (2p),
                  useful as a SOFT localization target
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = cover.shape
    target_bits = payload_bpp * H * W
    lam = _solve_lambda(rho, target_bits)

    p = np.exp(-lam * rho) / (1.0 + 2.0 * np.exp(-lam * rho))
    p = np.clip(p, 0.0, 1.0 / 3.0)

    # sample change in {-1, 0, +1} per pixel with prob (p, 1-2p, p)
    u = rng.random(size=cover.shape)
    change = np.zeros_like(cover, dtype=np.int8)
    change[u < p] = -1
    change[(u >= p) & (u < 2 * p)] = +1
    # else: stays 0

    stego = cover.astype(np.int16) + change
    stego = np.clip(stego, 0, 255).astype(cover.dtype)

    hard_mask = (change != 0).astype(np.uint8)
    p_map = (2 * p).astype(np.float32)  # total probability of being touched

    return stego, hard_mask, change, p_map


if __name__ == "__main__":
    
    rng = np.random.default_rng(0)
    cover = rng.integers(0, 256, size=(64, 64)).astype(np.uint8)
    rho = compute_hill_cost(cover)

    for bpp in [0.1, 0.2, 0.4]:
        stego, mask, change, p_map = simulate_embedding(cover, rho, bpp, rng)
        actual_rate = mask.sum() / cover.size
        print(f"target bpp={bpp:.2f} -> fraction of pixels changed={actual_rate:.4f}, "
              f"max |change|={np.abs(change).max()}")
