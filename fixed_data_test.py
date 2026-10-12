import torch

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import SteganalysisDataset
from src.training.losses import detection_loss


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

IMAGE_SIZE = 256
PAYLOADS = [0.1, 0.2, 0.3, 0.4]

NUM_SAMPLES = 100
STEPS = 300
LR = 1e-3

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
# CREATE FIXED DATASET
# ============================================================

images = []
labels = []

print("Creating fixed samples...")

for idx in range(NUM_SAMPLES):

    sample = dataset[idx]

    images.append(sample["image"])
    labels.append(sample["y_det"])


x = torch.stack(images).to(DEVICE)
y = torch.stack(labels).to(DEVICE)

print("Fixed dataset shape:", x.shape)

print(
    "Labels:",
    {
        "cover": int((y == 0).sum()),
        "stego": int((y == 1).sum()),
    }
)


# ============================================================
# MODEL
# ============================================================

model = SFMambaSteganalysis().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.0,
)


# ============================================================
# TRAIN ON FIXED DATASET
# ============================================================

model.train()

for step in range(STEPS):

    # Shuffle the fixed samples.
    # The actual images themselves never change.
    perm = torch.randperm(NUM_SAMPLES, device=DEVICE)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # Batch size = 2 to match normal training.
    for start in range(0, NUM_SAMPLES, 2):

        indices = perm[start:start + 2]

        xb = x[indices]
        yb = y[indices]

        det_logits, _ = model(xb)

        loss = detection_loss(det_logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(indices)

        with torch.no_grad():
            pred = (torch.sigmoid(det_logits) > 0.5).float()
            total_correct += (pred == yb).sum().item()
            total_count += len(indices)

    avg_loss = total_loss / NUM_SAMPLES
    accuracy = total_correct / total_count

    if step == 0 or (step + 1) % 25 == 0:
        print(
            f"Step {step + 1:3d}/{STEPS} | "
            f"loss={avg_loss:.6f} | "
            f"acc={accuracy:.3f}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

model.eval()

with torch.no_grad():

    correct = 0

    for start in range(0, NUM_SAMPLES, 2):

        xb = x[start:start + 2]
        yb = y[start:start + 2]

        logits, _ = model(xb)

        pred = (torch.sigmoid(logits) > 0.5).float()

        correct += (pred == yb).sum().item()

    final_acc = correct / NUM_SAMPLES


print()
print("=" * 60)
print("FIXED DATASET TEST COMPLETE")
print("=" * 60)

print(f"Final accuracy: {final_acc:.3f}")

if final_acc >= 0.95:
    print("[PASS] Model learns the fixed dataset.")
else:
    print("[FAIL] Model did not sufficiently learn the fixed dataset.")