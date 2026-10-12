import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.models.sf_mamba import SFMambaSteganalysis
from src.steganography.hill import compute_hill_cost
from src.steganography.embedding import simulate_embedding


NUM_PAIRS = 20
PAYLOAD = 0.4
EPOCHS = 100
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# ============================================================
# MODEL
# ============================================================

model = SFMambaSteganalysis().to(DEVICE)

# We only use the detection path.
# Localization is completely irrelevant for this test.

model.train()


# ============================================================
# DATA
# ============================================================

paths = sorted(
    glob.glob(r"data\BOSSbase\split\train\*.pgm")
)

rng = np.random.default_rng(42)

samples = []
labels = []


# ============================================================
# CREATE FIXED SRM RESIDUAL DATA
# ============================================================

# Use the model's actual SRM stem.
#
# IMPORTANT:
# ResidualStem expects a normalized image:
#
#       pixel / 127.5 - 1
#
# exactly like the real dataset.

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

    # --------------------------------------------------------
    # Extract SRM residuals ONLY
    # --------------------------------------------------------

    with torch.no_grad():

        cover_srm = model.stem(cover_t)

        stego_srm = model.stem(stego_t)

    samples.append(cover_srm.squeeze(0))
    labels.append(0.0)

    samples.append(stego_srm.squeeze(0))
    labels.append(1.0)


x = torch.stack(samples)
y = torch.tensor(
    labels,
    dtype=torch.float32
)

print("Dataset:", x.shape)

print(
    "SRM range:",
    x.min().item(),
    x.max().item()
)

print("Labels:", y.shape)


# ============================================================
# MOVE TO GPU
# ============================================================

# Keep the dataset on CPU.
# Move only the current batch to GPU during training.

# ============================================================
# IMPORTANT:
# CREATE A MODEL THAT STARTS AFTER THE SRM STEM
# ============================================================

class ResidualToDetection(nn.Module):

    def __init__(self, original_model):

        super().__init__()

        self.local_branch = original_model.local_branch

        self.global_branch = original_model.global_branch

        self.fusion = original_model.fusion

        self.detection_head = original_model.detection_head


    def forward(self, x):

        # ----------------------------------------------------
        # Local branch
        # ----------------------------------------------------

        local = self.local_branch(x)

        # ----------------------------------------------------
        # SF-Mamba global branch
        # ----------------------------------------------------

        global_full = self.global_branch(x)

        global_feat = torch.nn.functional.interpolate(
            global_full,
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        fused = self.fusion(
            local,
            global_feat
        )

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        return self.detection_head(fused)


classifier = ResidualToDetection(model).to(DEVICE)

# ============================================================
# TRAIN IN SMALL BATCHES
# ============================================================

BATCH_SIZE = 2

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=LR,
    weight_decay=0.0
)


N = len(x)

print("Training samples:", N)
print("Batch size:", BATCH_SIZE)


for epoch in range(EPOCHS):

    classifier.train()

    # Shuffle indices on CPU
    perm = torch.randperm(N)

    epoch_loss = 0.0
    correct = 0
    total = 0

    for start in range(0, N, BATCH_SIZE):

        idx = perm[start:start + BATCH_SIZE]

        # Only this tiny batch goes through SF-Mamba
        xb = x[idx].to(DEVICE)
        yb = y[idx].to(DEVICE)

        optimizer.zero_grad()

        logits = classifier(xb)

        loss = criterion(
            logits,
            yb
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item() * len(yb)

        with torch.no_grad():

            preds = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            correct += (
                preds == yb
            ).sum().item()

            total += len(yb)

        del xb, yb, logits, loss

    epoch_loss /= N
    epoch_acc = correct / total

    if epoch == 0 or (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} "
            f"loss={epoch_loss:.6f} "
            f"acc={epoch_acc:.3f}"
        )


# ============================================================
# FINAL EVALUATION
# ============================================================

classifier.eval()

all_probs = []
all_labels = []

with torch.no_grad():

    for start in range(0, N, BATCH_SIZE):

        xb = x[start:start + BATCH_SIZE].to(DEVICE)

        logits = classifier(xb)

        probs = torch.sigmoid(logits)

        all_probs.append(
            probs.cpu()
        )

        all_labels.append(
            y[start:start + BATCH_SIZE].cpu()
        )

probs = torch.cat(all_probs)
labels = torch.cat(all_labels)

preds = (
    probs >= 0.5
).float()

acc = (
    preds == labels
).float().mean().item()


print()
print("=" * 60)
print("SF-MAMBA ON FIXED SRM RESIDUALS")
print("=" * 60)

print("Final accuracy:", acc)

for i in range(min(10, N)):

    print(
        f"{i:2d} | "
        f"label={int(labels[i].item())} | "
        f"prob={probs[i].item():.6f}"
    )

print("=" * 60)

# ============================================================
# FINAL
# ============================================================

classifier.eval()

with torch.no_grad():

    logits = classifier(x)

    probs = torch.sigmoid(logits)

    preds = (
        probs >= 0.5
    ).float()

    acc = (
        preds == y
    ).float().mean().item()


print()
print("=" * 60)
print("SF-MAMBA ON FIXED SRM RESIDUALS")
print("=" * 60)

print("Final accuracy:", acc)

for i in range(10):

    print(
        f"{i:2d} | "
        f"label={int(y[i].item())} | "
        f"prob={probs[i].item():.6f}"
    )

print("=" * 60)