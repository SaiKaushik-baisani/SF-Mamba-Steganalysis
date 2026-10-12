
import torch
import torch.nn as nn

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import get_dataloader
from src.training.losses import detection_loss

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = get_dataloader(
    r"data\BOSSbase\split\train",
    payload_bpps=[0.4],
    image_size=256,
    batch_size=2,
    num_workers=0,
    shuffle=False,
    seed=42,
    deterministic=True,
)

batch = next(iter(loader))
x = batch["image"].to(DEVICE)
y = batch["y_det"].to(DEVICE)

model = SFMambaSteganalysis().to(DEVICE).eval()

with torch.no_grad():
    stem = model.stem(x)
    local = model.local_branch(stem)
    global_full = model.global_branch(stem)

    global_match = torch.nn.functional.interpolate(
        global_full,
        size=local.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    fused = model.fusion(local, global_match)

    # Existing head
    existing_logits = model.detection_head(fused)

    # Alternative: preserve spatial information longer
    spatial_head = nn.Sequential(
        nn.Conv2d(96, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 1),
    ).to(DEVICE).eval()

    alternative_logits = spatial_head(fused).squeeze(-1)

    print("Labels:", y.cpu().tolist())
    print("Fused feature shape:", tuple(fused.shape))
    print("Existing logits:", existing_logits.cpu().tolist())
    print("Alternative logits:", alternative_logits.cpu().tolist())

    print(
        "Existing loss:",
        detection_loss(existing_logits, y).item(),
    )
    print(
        "Alternative loss (random head):",
        detection_loss(alternative_logits, y).item(),
    )
