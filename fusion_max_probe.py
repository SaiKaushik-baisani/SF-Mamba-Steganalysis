import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.residual_stem import ResidualStem
from src.models.local_branch import LocalBranch
from src.models.mamba_branch import VSSBlock2D
from src.models.fusion import GatedFusion
from src.datasets.dataset import SteganalysisDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

NUM_PAIRS = 20
BATCH_SIZE = 2
STEPS = 300
LR = 1e-3


dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=[0.4],
    image_size=256,
    seed=42,
)


# --------------------------------------------------
# Generate fixed dataset
# --------------------------------------------------

images = []
labels = []

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
    device=DEVICE
)


# --------------------------------------------------
# Feature extractor
# --------------------------------------------------

stem = ResidualStem(
    out_channels=32
).to(DEVICE)

local_branch = LocalBranch(
    in_channels=32
).to(DEVICE)

global_branch = VSSBlock2D(
    in_channels=32,
    hidden_dim=64,
    pre_downsample=8,
    n_blocks=4,
    d_state=16
).to(DEVICE)

fusion = GatedFusion(
    local_channels=64,
    global_channels=64,
    out_channels=96
).to(DEVICE)


for module in [
    stem,
    local_branch,
    global_branch,
    fusion
]:
    module.eval()

    for p in module.parameters():
        p.requires_grad = False


# --------------------------------------------------
# Extract MAX-pooled features
# --------------------------------------------------

features = []

with torch.no_grad():

    for start in range(0, len(x), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(x))

        xb = x[start:end]

        stem_out = stem(xb)

        local = local_branch(stem_out)

        global_feat = global_branch(stem_out)

        global_feat = F.interpolate(
            global_feat,
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        fused = fusion(
            local,
            global_feat
        )

        # IMPORTANT:
        # Keep the strongest spatial response
        # instead of averaging everything.
        pooled = fused.abs().amax(
            dim=(2, 3)
        )

        features.append(pooled)


features = torch.cat(features, dim=0)

print("Feature shape:", features.shape)


# --------------------------------------------------
# Linear probe
# --------------------------------------------------

probe = nn.Linear(
    features.shape[1],
    1
).to(DEVICE)

optimizer = torch.optim.AdamW(
    probe.parameters(),
    lr=LR,
    weight_decay=0.0
)


# --------------------------------------------------
# Train
# --------------------------------------------------

for step in range(STEPS):

    indices = torch.randint(
        0,
        len(features),
        (BATCH_SIZE,),
        device=DEVICE
    )

    xb = features[indices]

    yb = y[indices]

    logits = probe(xb).squeeze(1)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        yb
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    with torch.no_grad():

        probs = torch.sigmoid(logits)

        pred = (probs > 0.5).float()

        acc = (pred == yb).float().mean().item()

    if step == 0 or (step + 1) % 25 == 0:

        print(
            f"Step {step+1:3d}/{STEPS} | "
            f"loss={loss.item():.6f} | "
            f"batch_acc={acc:.3f}"
        )


# --------------------------------------------------
# Evaluate all 40
# --------------------------------------------------

probe.eval()

with torch.no_grad():

    logits = probe(features).squeeze(1)

    probs = torch.sigmoid(logits)

    preds = (probs > 0.5).float()

    accuracy = (
        preds == y
    ).float().mean().item()


print()
print("=" * 60)
print("FUSION MAX-POOL LINEAR PROBE COMPLETE")
print("=" * 60)

print("Final accuracy:", accuracy)