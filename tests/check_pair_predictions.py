
import torch
import torch.nn as nn
from src.datasets.dataset import SteganalysisDataset

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.features(x).flatten(1)).squeeze(1)




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This deliberately uses a fresh, untrained model.
# We are checking the paired input/prediction path, not accuracy.
model = TinyCNN().to(DEVICE)
model.eval()

dataset = SteganalysisDataset(
    r"data\BOSSbase\split\val",
    payload_bpps=[0.1, 0.2, 0.3, 0.4],
    image_size=256,
    seed=42,
    deterministic=True,
)

with torch.no_grad():
    for i in range(0, 10, 2):
        cover = dataset[i]
        stego = dataset[i + 1]

        x = torch.stack([cover["image"], stego["image"]]).to(DEVICE)
        probs = torch.sigmoid(model(x)).cpu()

        pixel_diff = (
            cover["image"] - stego["image"]
        ).abs().mean().item()

        actual_diff = (
            cover["image"] != stego["image"]
        ).sum().item()

        mask_pixels = stego["hard_mask"].sum().item()

        cover_np = cover["image"].squeeze(0)
        stego_np = stego["image"].squeeze(0)
        mask = stego["hard_mask"].squeeze(0).bool()

        if mask.any():
            changed_values = (
                stego_np[mask] - cover_np[mask]
            ).abs()

            print(
                f"  mean_change_at_mask={changed_values.mean().item():.6f} | "
                f"min_change_at_mask={changed_values.min().item():.6f} | "
                f"max_change_at_mask={changed_values.max().item():.6f}"
            )

        print(
            f"  mask_pixels={int(mask_pixels)} | "
            f"actual_changed_pixels={int(actual_diff)}"
        )

        print(
            f"Pair {i // 2}: "
            f"same_source={cover['path'] == stego['path']} | "
            f"cover_prob={probs[0].item():.6f} | "
            f"stego_prob={probs[1].item():.6f} | "
            f"mean_pixel_diff={pixel_diff:.6f}"
        )
