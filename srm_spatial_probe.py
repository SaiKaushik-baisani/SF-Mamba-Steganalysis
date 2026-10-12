import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.residual_stem import ResidualStem
from src.datasets.dataset import SteganalysisDataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

NUM_PAIRS = 20
STEPS = 300
LR = 1e-3
BATCH_SIZE = 2


# ============================================================
# DATASET
# ============================================================

dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=[0.4],
    image_size=256,
    seed=42,
)


# ============================================================
# FIXED HILL DATASET
# ============================================================

images = []
labels = []

print("Generating fixed HILL dataset...")

for idx in range(NUM_PAIRS):

    while True:
        sample = dataset[idx]
        if sample["y_det"].item() == 0:
            break

    cover = sample["image"]

    while True:
        sample = dataset[idx]
        if sample["y_det"].item() == 1:
            break

    stego = sample["image"]

    images.append(cover)
    labels.append(0.0)

    images.append(stego)
    labels.append(1.0)


x = torch.stack(images).to(DEVICE)

y = torch.tensor(
    labels,
    dtype=torch.float32,
    device=DEVICE,
)

print("Dataset:", x.shape)
print("Labels:", y.cpu().tolist())


# ============================================================
# FIXED SRM FILTERS
# ============================================================

stem = ResidualStem(
    out_channels=32,
    learnable=False,
).to(DEVICE)

stem.eval()

for p in stem.parameters():
    p.requires_grad = False


# ============================================================
# EXTRACT SPATIAL SRM RESIDUALS
# ============================================================

features = []

with torch.no_grad():

    for start in range(0, len(x), BATCH_SIZE):

        xb = x[start:start + BATCH_SIZE]

        r = stem.hp_conv(xb)

        r = torch.clamp(
            r,
            -stem.clip_value,
            stem.clip_value,
        )

        features.append(r)


features = torch.cat(features, dim=0)

print("SRM spatial features:", features.shape)


# ============================================================
# TINY SPATIAL CNN
# ============================================================

class SpatialProbe(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
            stem.hp_conv.out_channels,
            16,
            kernel_size=3,
            padding=1,
            bias=False,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),

        )

        self.fc = nn.Linear(32, 1)


    def forward(self, x):

        x = self.net(x)

        x = x.flatten(1)

        return self.fc(x).squeeze(1)


model = SpatialProbe().to(DEVICE)


# ============================================================
# TRAIN
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.0,
)


num_samples = len(features)

model.train()

for step in range(STEPS):

    perm = torch.randperm(
        num_samples,
        device=DEVICE,
    )

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for start in range(
        0,
        num_samples,
        BATCH_SIZE,
    ):

        indices = perm[start:start + BATCH_SIZE]

        xb = features[indices]
        yb = y[indices]

        logits = model(xb)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            yb,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * len(indices)
        )

        with torch.no_grad():

            pred = (
                torch.sigmoid(logits) > 0.5
            ).float()

            total_correct += (
                pred == yb
            ).sum().item()

            total_count += len(indices)


    if step == 0 or (step + 1) % 25 == 0:

        print(
            f"Step {step + 1:3d}/{STEPS} | "
            f"loss={total_loss / num_samples:.6f} | "
            f"acc={total_correct / total_count:.3f}"
        )


# ============================================================
# FINAL EVALUATION
# ============================================================

model.eval()

with torch.no_grad():

    logits = model(features)

    probs = torch.sigmoid(logits)

    preds = (probs > 0.5).float()

    accuracy = (
        preds == y
    ).float().mean().item()


print()
print("=" * 60)
print("SRM SPATIAL PROBE COMPLETE")
print("=" * 60)

print(
    f"Final accuracy: {accuracy:.3f}"
)