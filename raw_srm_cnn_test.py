import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.models.residual_stem import SRM_KERNELS
from src.steganography.hill import compute_hill_cost
from src.steganography.embedding import simulate_embedding


NUM_PAIRS = 20
PAYLOAD = 0.4
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# ============================================================
# RAW FIXED SRM FILTER BANK
# ============================================================

srm = nn.Conv2d(
    1,
    SRM_KERNELS.shape[0],
    kernel_size=3,
    padding=1,
    bias=False
).to(DEVICE)

with torch.no_grad():
    srm.weight.copy_(
        torch.from_numpy(SRM_KERNELS).unsqueeze(1)
    )

srm.weight.requires_grad = False
srm.eval()

print("SRM channels:", SRM_KERNELS.shape[0])


# ============================================================
# CREATE DATA
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

        cover_t = torch.from_numpy(
            cover.astype(np.float32) / 127.5 - 1.0
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        stego_t = torch.from_numpy(
            stego.astype(np.float32) / 127.5 - 1.0
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        cover_srm = srm(cover_t)
        stego_srm = srm(stego_t)

        samples.append(
            cover_srm.squeeze(0).cpu()
        )
        labels.append(0.0)

        samples.append(
            stego_srm.squeeze(0).cpu()
        )
        labels.append(1.0)


X = torch.stack(samples)
Y = torch.tensor(
    labels,
    dtype=torch.float32
)

print("Dataset:", X.shape)
print("SRM min:", X.min().item())
print("SRM max:", X.max().item())


# ============================================================
# TINY CNN
# ============================================================

class RawSRMCNN(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.features = nn.Sequential(

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

        x = self.features(x)
        x = x.flatten(1)

        return self.fc(x).squeeze(1)


cnn = RawSRMCNN(
    in_channels=X.shape[1]
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    cnn.parameters(),
    lr=LR
)


# ============================================================
# TRAIN
# ============================================================

N = len(X)

for epoch in range(EPOCHS):

    cnn.train()

    permutation = torch.randperm(N)

    total_loss = 0.0
    correct = 0
    total = 0

    for start in range(0, N, BATCH_SIZE):

        idx = permutation[start:start + BATCH_SIZE]

        xb = X[idx].to(DEVICE)
        yb = Y[idx].to(DEVICE)

        optimizer.zero_grad()

        logits = cnn(xb)

        loss = criterion(
            logits,
            yb
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(yb)

        with torch.no_grad():

            preds = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            correct += (
                preds == yb
            ).sum().item()

            total += len(yb)

    if epoch == 0 or (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} "
            f"loss={total_loss/N:.6f} "
            f"acc={correct/total:.3f}"
        )


# ============================================================
# FINAL
# ============================================================

cnn.eval()

correct = 0
total = 0

all_probs = []
all_labels = []

with torch.no_grad():

    for start in range(0, N, BATCH_SIZE):

        xb = X[start:start + BATCH_SIZE].to(DEVICE)
        yb = Y[start:start + BATCH_SIZE].to(DEVICE)

        logits = cnn(xb)

        probs = torch.sigmoid(logits)

        preds = (
            probs >= 0.5
        ).float()

        correct += (
            preds == yb
        ).sum().item()

        total += len(yb)

        all_probs.append(probs.cpu())
        all_labels.append(yb.cpu())


probs = torch.cat(all_probs)
labels = torch.cat(all_labels)

print()
print("=" * 60)
print("RAW 13-CHANNEL SRM CNN")
print("=" * 60)

print("Final accuracy:", correct / total)

for i in range(10):

    print(
        f"{i:2d} | "
        f"label={int(labels[i].item())} | "
        f"prob={probs[i].item():.6f}"
    )

print("=" * 60)