import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.models.residual_stem import ResidualStem
from src.steganography.hill import compute_hill_cost
from src.steganography.embedding import simulate_embedding


NUM_PAIRS = 20
PAYLOAD = 0.4
EPOCHS = 100
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# ============================================================
# LOAD SRM STEM
# ============================================================

stem = ResidualStem(out_channels=32).to(DEVICE)
stem.eval()

# We don't train the SRM stem.
for p in stem.parameters():
    p.requires_grad = False


# ============================================================
# CREATE SRM-DIFFERENCE DATA
# ============================================================

paths = sorted(
    glob.glob(r"data\BOSSbase\split\train\*.pgm")
)

rng = np.random.default_rng(42)

samples = []
labels = []


with torch.no_grad():

    for path in paths[:NUM_PAIRS]:

        cover = np.array(
            Image.open(path).convert("L"),
            dtype=np.uint8
        )

        rho = compute_hill_cost(cover)

        stego, _, _, _ = simulate_embedding(
            cover,
            rho,
            PAYLOAD,
            rng
        )

        # ----------------------------------------------------
        # NORMALIZE EXACTLY LIKE THE REAL DATASET
        # ----------------------------------------------------

        cover_t = torch.from_numpy(
            cover.astype(np.float32) / 127.5 - 1.0
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        stego_t = torch.from_numpy(
            stego.astype(np.float32) / 127.5 - 1.0
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # SRM FEATURES
        # ----------------------------------------------------

        cover_srm = stem(cover_t)
        stego_srm = stem(stego_t)

        # Actual SRM feature difference
        srm_diff = stego_srm - cover_srm

        samples.append(
            srm_diff.squeeze(0).cpu()
        )

        labels.append(1.0)

        # ----------------------------------------------------
        # ZERO DIFFERENCE = COVER CONTROL
        # ----------------------------------------------------

        samples.append(
            torch.zeros_like(
                srm_diff.squeeze(0).cpu()
            )
        )

        labels.append(0.0)


x = torch.stack(samples)
y = torch.tensor(
    labels,
    dtype=torch.float32
)

print("Dataset:", x.shape)
print("Difference min:", x.min().item())
print("Difference max:", x.max().item())
print(
    "Nonzero values:",
    int(torch.count_nonzero(x))
)


# ============================================================
# TINY CNN
# ============================================================

class TinyCNN(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                in_channels,
                16,
                3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                16,
                32,
                3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(64, 1)


    def forward(self, x):

        x = self.net(x)

        x = x.flatten(1)

        return self.fc(x).squeeze(1)


model = TinyCNN(
    in_channels=x.shape[1]
).to(DEVICE)

x = x.to(DEVICE)
y = y.to(DEVICE)


# ============================================================
# TRAIN
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    logits = model(x)

    loss = criterion(
        logits,
        y
    )

    loss.backward()

    optimizer.step()

    with torch.no_grad():

        probs = torch.sigmoid(logits)

        preds = (
            probs >= 0.5
        ).float()

        acc = (
            preds == y
        ).float().mean().item()


    if epoch == 0 or (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} "
            f"loss={loss.item():.6f} "
            f"acc={acc:.3f}"
        )


# ============================================================
# FINAL
# ============================================================

model.eval()

with torch.no_grad():

    logits = model(x)

    probs = torch.sigmoid(logits)

    preds = (
        probs >= 0.5
    ).float()

    acc = (
        preds == y
    ).float().mean().item()


print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Accuracy:", acc)

for i in range(10):

    print(
        f"{i:2d} | "
        f"label={int(y[i].item())} | "
        f"prob={probs[i].item():.6f}"
    )

print("=" * 60)