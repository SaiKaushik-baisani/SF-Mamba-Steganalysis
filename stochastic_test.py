import torch

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import SteganalysisDataset
from src.training.losses import detection_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

IMAGE_SIZE = 256
PAYLOADS = [0.4]

STEPS = 100
BATCH_SIZE = 2
LR = 1e-4

print("Using device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=PAYLOADS,
    image_size=IMAGE_SIZE,
    seed=42,
)


# ============================================================
# MODEL
# ============================================================

model = SFMambaSteganalysis().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)


# ============================================================
# FRESH STOCHASTIC SAMPLES EVERY STEP
# ============================================================

model.train()

for step in range(STEPS):

    images = []
    labels = []

    # Generate a completely fresh batch.
    for _ in range(BATCH_SIZE):

        idx = torch.randint(
            0,
            len(dataset),
            (1,)
        ).item()

        sample = dataset[idx]

        images.append(sample["image"])
        labels.append(sample["y_det"])

    x = torch.stack(images).to(DEVICE)
    y = torch.stack(labels).to(DEVICE)

    det_logits, _ = model(x)

    loss = detection_loss(det_logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(det_logits)
        pred = (probs > 0.5).float()
        acc = (pred == y).float().mean().item()

    if step == 0 or (step + 1) % 10 == 0:
        print(
            f"Step {step + 1:3d}/{STEPS} | "
            f"loss={loss.item():.6f} | "
            f"acc={acc:.2f} | "
            f"labels={y.tolist()}"
        )


print()
print("=" * 60)
print("STOCHASTIC DATA TEST COMPLETE")
print("=" * 60)