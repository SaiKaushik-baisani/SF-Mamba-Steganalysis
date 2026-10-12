import torch

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import SteganalysisDataset
from src.training.losses import detection_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

IMAGE_SIZE = 256
PAYLOADS = [0.4]

STEPS = 150
LR = 1e-4

print("Using device:", DEVICE)

# Dataset
dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=PAYLOADS,
    image_size=IMAGE_SIZE,
    seed=42,
)

# Pick ONE cover image.
# We deliberately use the same dataset index every time.
cover_idx = 0

print("Using fixed dataset index:", cover_idx)

model = SFMambaSteganalysis().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

model.train()

for step in range(STEPS):

    images = []
    labels = []

    # Always use the SAME cover index.
    #
    # The dataset still generates a fresh random
    # cover/stego decision and fresh HILL embedding
    # whenever __getitem__ is called.
    for _ in range(2):
        sample = dataset[cover_idx]

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
        pred = (torch.sigmoid(det_logits) > 0.5).float()
        acc = (pred == y).float().mean().item()

    if step == 0 or (step + 1) % 10 == 0:
        print(
            f"Step {step+1:3d}/{STEPS} | "
            f"loss={loss.item():.6f} | "
            f"acc={acc:.2f} | "
            f"labels={y.detach().cpu().tolist()}"
        )

print()
print("=" * 60)
print("SINGLE-COVER STOCHASTIC TEST COMPLETE")
print("=" * 60)