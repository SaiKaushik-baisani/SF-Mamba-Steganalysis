import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.models.sf_mamba import SFMambaSteganalysis
from src.steganography.hill import compute_hill_cost
from src.steganography.embedding import simulate_embedding


# ============================================================
# CONFIG
# ============================================================

NUM_PAIRS = 20
PAYLOAD = 0.4
EPOCHS = 300
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# ============================================================
# MODEL
# ============================================================

model = SFMambaSteganalysis().to(DEVICE)
model.eval()

for p in model.parameters():
    p.requires_grad = False


# ============================================================
# LOAD DATA + EXTRACT FUSION FEATURES
# ============================================================

paths = sorted(
    glob.glob(r"data\BOSSbase\split\train\*.pgm")
)

rng = np.random.default_rng(42)

features = []
labels = []


def get_fusion(x):

    stem = model.stem(x)

    local = model.local_branch(stem)

    global_full = model.global_branch(stem)

    global_feat = torch.nn.functional.interpolate(
        global_full,
        size=local.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    fused = model.fusion(
        local,
        global_feat
    )

    return fused


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

        cover_fused = get_fusion(cover_t)
        stego_fused = get_fusion(stego_t)

        # GAP representation
        cover_gap = torch.mean(
            cover_fused,
            dim=(2, 3)
        )

        stego_gap = torch.mean(
            stego_fused,
            dim=(2, 3)
        )

        features.append(
            cover_gap.squeeze(0).cpu()
        )
        labels.append(0.0)

        features.append(
            stego_gap.squeeze(0).cpu()
        )
        labels.append(1.0)


X = torch.stack(features).to(DEVICE)
Y = torch.tensor(
    labels,
    dtype=torch.float32
).to(DEVICE)


print("Feature shape:", X.shape)
print("Labels:", Y.shape)


# ============================================================
# LINEAR PROBE
# ============================================================

classifier = nn.Linear(
    X.shape[1],
    1
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=LR
)


# ============================================================
# TRAIN ONLY THE CLASSIFIER
# ============================================================

for epoch in range(EPOCHS):

    classifier.train()

    optimizer.zero_grad()

    logits = classifier(X).squeeze(1)

    loss = criterion(
        logits,
        Y
    )

    loss.backward()

    optimizer.step()

    with torch.no_grad():

        probs = torch.sigmoid(logits)

        preds = (
            probs >= 0.5
        ).float()

        acc = (
            preds == Y
        ).float().mean().item()


    if epoch == 0 or (epoch + 1) % 25 == 0:

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} "
            f"loss={loss.item():.6f} "
            f"acc={acc:.3f}"
        )


# ============================================================
# FINAL
# ============================================================

classifier.eval()

with torch.no_grad():

    logits = classifier(X).squeeze(1)

    probs = torch.sigmoid(logits)

    preds = (
        probs >= 0.5
    ).float()

    acc = (
        preds == Y
    ).float().mean().item()


print()
print("=" * 60)
print("FUSION GAP LINEAR PROBE")
print("=" * 60)

print("Final accuracy:", acc)

for i in range(10):

    print(
        f"{i:2d} | "
        f"label={int(Y[i].item())} | "
        f"prob={probs[i].item():.6f}"
    )

print("=" * 60)