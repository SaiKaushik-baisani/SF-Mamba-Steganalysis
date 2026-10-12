import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.steganography.hill import compute_hill_cost
from src.steganography.embedding import simulate_embedding


# ============================================================
# CONFIG
# ============================================================

NUM_PAIRS = 20
PAYLOAD = 0.4
EPOCHS = 100
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# ============================================================
# LOAD 20 COVER/STego PAIRS
# ============================================================

paths = sorted(glob.glob(r"data\BOSSbase\split\train\*.pgm"))

rng = np.random.default_rng(42)

samples = []
labels = []

for idx, path in enumerate(paths[:NUM_PAIRS]):

    cover = np.array(
        Image.open(path).convert("L"),
        dtype=np.uint8
    )

    rho = compute_hill_cost(cover)

    # Cover
    samples.append(cover.astype(np.float32) / 255.0)
    labels.append(0.0)

    # Stego
    stego, _, _, _ = simulate_embedding(
        cover,
        rho,
        PAYLOAD,
        rng
    )

    samples.append(stego.astype(np.float32) / 255.0)
    labels.append(1.0)


x = torch.tensor(
    np.stack(samples),
    dtype=torch.float32
).unsqueeze(1)

y = torch.tensor(
    labels,
    dtype=torch.float32
)

print("Dataset:", x.shape)
print("Labels:", y.tolist())


# ============================================================
# VERY SMALL CNN
# ============================================================

class TinyCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x).squeeze(1)


model = TinyCNN().to(DEVICE)

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

    loss = criterion(logits, y)

    loss.backward()

    optimizer.step()

    with torch.no_grad():

        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).float()

        acc = (preds == y).float().mean().item()

    if epoch == 0 or (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} "
            f"loss={loss.item():.6f} "
            f"acc={acc:.3f}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

model.eval()

with torch.no_grad():

    logits = model(x)

    probs = torch.sigmoid(logits)

    preds = (probs >= 0.5).float()

    acc = (preds == y).float().mean().item()


print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Accuracy:", acc)

print()
print("First 10 predictions:")

for i in range(min(10, len(y))):

    print(
        f"{i:2d} | "
        f"label={int(y[i].item())} | "
        f"prob={probs[i].item():.6f}"
    )

print("=" * 60)