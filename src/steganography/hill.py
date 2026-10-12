"""
HILL (HIgh-pass, Low-pass, Low-pass) cost function.
Reference: Li, Wang, Huang, Shi, "A New Cost Function for Spatial Image
Steganography," ICIP 2014.

Cost at each pixel is INVERSELY proportional to local texture complexity:
smooth/flat regions -> high cost (avoid embedding there, changes are visible
as statistical anomalies), textured regions -> low cost (embed here, changes
blend into existing noise).

Pipeline:
  1. High-pass filter the image with a small kernel (H1) to get a residual.
  2. Take the absolute value.
  3. Smooth twice with average filters of increasing size (L1 then L2) --
     this is what turns a single-pixel residual into a *neighborhood*
     texture-complexity estimate.
  4. Cost = 1 / (smoothed result), with a floor to avoid divide-by-zero.

This is the simplest of the standard content-adaptive cost functions
(WOW and S-UNIWARD use directional filter banks and are more involved --
same interface, swap this module out if/when you add them).
"""
import numpy as np
from scipy.signal import convolve2d


# The HILL paper's H1 kernel (a normalized Laplacian-of-Gaussian-like kernel)
H1 = np.array([[-1, 2, -1],
               [2, -4, 2],
               [-1, 2, -1]], dtype=np.float64)

L1 = np.ones((3, 3), dtype=np.float64) / 9.0
L2 = np.ones((15, 15), dtype=np.float64) / 225.0

_EPS = 1e-10


def compute_hill_cost(image: np.ndarray) -> np.ndarray:
    """
    image: 2D float array, grayscale, any consistent range (e.g. 0-255).
    returns: cost map, same shape, cost(i,j) > 0 for every pixel.
    """
    assert image.ndim == 2, "expects a single-channel (grayscale) image"
    img = image.astype(np.float64)

    residual = convolve2d(img, H1, mode="same", boundary="symm")
    abs_residual = np.abs(residual)

    smoothed = convolve2d(abs_residual, L1, mode="same", boundary="symm")
    smoothed = convolve2d(smoothed, L2, mode="same", boundary="symm")

    cost = 1.0 / (smoothed + _EPS)

    # Boundary pixels where the embedding simulator forbids saturated values
    # (0 or 255) get an effectively infinite cost so they're never selected.
    cost[img <= 0] = 1e13
    cost[img >= 255] = 1e13
    return cost


# quick check
_rng = np.random.default_rng(0)
_fake_img = _rng.integers(0, 256, size=(64, 64)).astype(np.float64)
_cost = compute_hill_cost(_fake_img)
print("cost map shape:", _cost.shape, "min/max:", _cost.min(), _cost.max())
