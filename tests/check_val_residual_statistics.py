
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.dataset import SteganalysisDataset

VAL_DIR = r"data\BOSSbase\split\val"
PAIRS = 100
PAYLOAD = 0.4

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SteganalysisDataset(
    cover_dir=VAL_DIR,
    payload_bpps=(PAYLOAD,),
    image_size=256,
    seed=42,
    deterministic=True,
)

kernel = torch.tensor(
    [[0.0, -1.0, 0.0],
     [-1.0, 4.0, -1.0],
     [0.0, -1.0, 0.0]],
    dtype=torch.float32,
    device=device,
).view(1, 1, 3, 3)

cover_stats = []
stego_stats = []

for i in range(PAIRS):
    cover = dataset[2 * i]["image"].unsqueeze(0).to(device)
    stego = dataset[2 * i + 1]["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        cover_res = F.conv2d(cover, kernel, padding=1)
        stego_res = F.conv2d(stego, kernel, padding=1)

    cover_stats.append(cover_res.abs().mean().item())
    stego_stats.append(stego_res.abs().mean().item())

cover_stats = np.array(cover_stats)
stego_stats = np.array(stego_stats)
delta = stego_stats - cover_stats

print(f"Validation pairs: {PAIRS}")
print(f"Payload: {PAYLOAD}")
print(f"Mean cover statistic: {cover_stats.mean():.8f}")
print(f"Mean stego statistic: {stego_stats.mean():.8f}")
print(f"Mean paired increase: {delta.mean():.8f}")
print(f"Median paired increase: {np.median(delta):.8f}")
print(f"Positive pairs: {(delta > 0).sum()}/{PAIRS}")
print(f"Negative pairs: {(delta < 0).sum()}/{PAIRS}")
print(f"Unchanged pairs: {(delta == 0).sum()}/{PAIRS}")
