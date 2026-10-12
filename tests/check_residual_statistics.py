
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.dataset import SteganalysisDataset

COVER_DIR = r"data\BOSSbase\split\train"
PAIRS = 100
PAYLOAD = 0.4

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SteganalysisDataset(
    cover_dir=COVER_DIR,
    payload_bpps=(PAYLOAD,),
    image_size=256,
    seed=42,
    deterministic=True,
)

# Fixed high-pass residual filter
kernel = torch.tensor(
    [[0.0, -1.0, 0.0],
     [-1.0, 4.0, -1.0],
     [0.0, -1.0, 0.0]],
    dtype=torch.float32,
    device=device,
).view(1, 1, 3, 3)

cover_stats = []
stego_stats = []
delta_stats = []

for i in range(PAIRS):
    cover = dataset[2 * i]["image"].unsqueeze(0).to(device)
    stego = dataset[2 * i + 1]["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        cover_res = F.conv2d(cover, kernel, padding=1)
        stego_res = F.conv2d(stego, kernel, padding=1)

    cover_stats.append(cover_res.abs().mean().item())
    stego_stats.append(stego_res.abs().mean().item())
    delta_stats.append((stego_res - cover_res).abs().mean().item())

cover_stats = np.array(cover_stats)
stego_stats = np.array(stego_stats)
delta_stats = np.array(delta_stats)

print(f"Device: {device}")
print(f"Pairs: {PAIRS}")
print(f"Payload: {PAYLOAD}")
print()
print(f"Cover residual mean: {cover_stats.mean():.8f}")
print(f"Stego residual mean: {stego_stats.mean():.8f}")
print(f"Mean paired change: {(stego_stats - cover_stats).mean():.8f}")
print(f"Mean absolute residual difference: {delta_stats.mean():.8f}")
print(
    "Pairs with increased residual:",
    f"{np.sum(stego_stats > cover_stats)}/{PAIRS}",
)
print(
    "Pairs with decreased residual:",
    f"{np.sum(stego_stats < cover_stats)}/{PAIRS}",
)
print(
    "Identical residual statistics:",
    f"{np.sum(stego_stats == cover_stats)}/{PAIRS}",
)
