import torch

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import SteganalysisDataset
from src.training.losses import detection_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

NUM_PAIRS = 20
STEPS = 500
LR = 1e-3

print("Using device:", DEVICE)

dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=[0.4],
    image_size=256,
    seed=42,
)

# --------------------------------------------------
# Generate a fixed dataset
# --------------------------------------------------

images = []
labels = []

print("Generating fixed HILL dataset...")

for idx in range(NUM_PAIRS):

    # Generate one cover
    while True:
        sample = dataset[idx]

        if sample["y_det"].item() == 0:
            break

    cover = sample["image"]

    # Generate one stego
    while True:
        sample = dataset[idx]

        if sample["y_det"].item() == 1:
            break

    stego = sample["image"]

    images.append(cover)
    labels.append(0.0)

    images.append(stego)
    labels.append(1.0)

    print(
        f"Pair {idx+1:2d}/{NUM_PAIRS}: "
        f"stego pixels={int(sample['hard_mask'].sum().item())}"
    )


x = torch.stack(images).to(DEVICE)

y = torch.tensor(
    labels,
    dtype=torch.float32,
    device=DEVICE,
)

print()
print("Fixed dataset shape:", x.shape)
print("Labels:", y.detach().cpu().tolist())

# --------------------------------------------------
# Model
# --------------------------------------------------

model = SFMambaSteganalysis().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.0,
)

model.train()

# --------------------------------------------------
# Train using small batches
# --------------------------------------------------

BATCH_SIZE = 2

num_samples = len(x)

for step in range(STEPS):

    # Shuffle the entire fixed dataset.
    # Every image is seen exactly once per epoch.
    perm = torch.randperm(
        num_samples,
        device=DEVICE,
    )

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for start in range(0, num_samples, BATCH_SIZE):

        indices = perm[start:start + BATCH_SIZE]

        xb = x[indices]
        yb = y[indices]

        det_logits, _ = model(xb)

        loss = detection_loss(
            det_logits,
            yb,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(indices)

        with torch.no_grad():

            probs = torch.sigmoid(det_logits)

            pred = (probs > 0.5).float()

            total_correct += (
                pred == yb
            ).sum().item()

            total_count += len(indices)

    avg_loss = total_loss / num_samples
    epoch_acc = total_correct / total_count

    if step == 0 or (step + 1) % 25 == 0:

        print(
            f"Step {step+1:3d}/{STEPS} | "
            f"loss={avg_loss:.6f} | "
            f"acc={epoch_acc:.3f}"
        )


# --------------------------------------------------
# Evaluate ALL 40 fixed images
# --------------------------------------------------

model.eval()

correct = 0
total = num_samples

with torch.no_grad():

    for start in range(0, num_samples, BATCH_SIZE):

        end = min(start + BATCH_SIZE, num_samples)

        xb = x[start:end]
        yb = y[start:end]

        det_logits, _ = model(xb)

        probs = torch.sigmoid(det_logits)

        pred = (probs > 0.5).float()

        correct += (pred == yb).sum().item()

final_acc = correct / total

print()
print("=" * 60)
print("SMALL FIXED HILL DATASET TEST COMPLETE")
print("=" * 60)

print(f"Final accuracy on all 40 samples: {final_acc:.3f}")

print()
print("=" * 60)
print("FINAL PREDICTIONS")
print("=" * 60)

model.eval()

all_probs = []
all_preds = []

with torch.no_grad():
    for start in range(0, num_samples, BATCH_SIZE):

        end = min(start + BATCH_SIZE, num_samples)

        xb = x[start:end]

        det_logits, _ = model(xb)

        probs = torch.sigmoid(det_logits)

        preds = (probs > 0.5).float()

        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

for i in range(num_samples):
    print(
        f"{i:2d}: "
        f"label={int(y[i].item())} "
        f"prob={all_probs[i]:.6f} "
        f"pred={int(all_preds[i])}"
    )