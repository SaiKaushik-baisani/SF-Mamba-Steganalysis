
import torch
from src.datasets.dataset import SteganalysisDataset

dataset = SteganalysisDataset(
    r"data\BOSSbase\split\train",
    payload_bpps=[0.1, 0.2, 0.3, 0.4],
    image_size=256,
    seed=42,
    deterministic=True,
)

print("Checking 8 fixed cover-stego pairs\n")

for i in range(8):
    cover = dataset[2 * i]
    stego = dataset[2 * i + 1]

    c = cover["image"]
    s = stego["image"]
    mask = stego["hard_mask"].bool()

    delta = s - c

    print(
        f"Pair {i}: "
        f"labels=({cover['y_det'].item():.0f},"
        f"{stego['y_det'].item():.0f}) | "
        f"mask={int(mask.sum())} | "
        f"changed={(delta != 0).sum().item()} | "
        f"mean_abs_delta={delta.abs().mean().item():.7f} | "
        f"cover_std={c.std().item():.4f} | "
        f"stego_std={s.std().item():.4f} | "
        f"same_path={cover['path'] == stego['path']}"
    )

    assert cover["y_det"].item() == 0
    assert stego["y_det"].item() == 1
    assert cover["path"] == stego["path"]
    assert int(mask.sum()) == int((delta != 0).sum())
