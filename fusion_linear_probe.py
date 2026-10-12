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


print("Using device:", DEVICE)


# --------------------------------------------------
# 1. Generate fixed HILL dataset
# --------------------------------------------------

dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=[0.4],
    image_size=256,
    seed=42,
)

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


# --------------------------------------------------
# 2. Build feature extractor
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


# Freeze feature extractor
stem.eval()
local_branch.eval()
global_branch.eval()
fusion.eval()

for module in [stem, local_branch, global_branch, fusion]:

    for parameter in module.parameters():
        parameter.requires_grad = False


# --------------------------------------------------
# 3. Extract fused features
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

        # Global average pooling
        pooled = fused.mean(dim=(2, 3))

        features.append(pooled)


features = torch.cat(features, dim=0)

print("Feature shape:", features.shape)

print(
    "Feature mean:",
    features.mean().item()
)

print(
    "Feature std:",
    features.std().item()
)


# --------------------------------------------------
# 4. Tiny linear probe
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
# 5. Train probe
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
# 6. Evaluate ALL samples
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
print("FUSION LINEAR PROBE COMPLETE")
print("=" * 60)

print("Final accuracy:", accuracy)

print()

for i in range(len(y)):

    print(
        f"{i:2d}: "
        f"label={int(y[i].item())} "
        f"prob={probs[i].item():.6f} "
        f"pred={int(preds[i].item())}"
    )