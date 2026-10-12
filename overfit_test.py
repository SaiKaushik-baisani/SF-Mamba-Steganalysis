import torch

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import get_dataloader
from src.training.losses import detection_loss


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

IMAGE_SIZE = 256
PAYLOADS = [0.1, 0.2, 0.3, 0.4]

BATCH_SIZE = 2

STEPS = 200
LR = 1e-3

print("Using device:", DEVICE)


# ============================================================
# DATA
# ============================================================

loader = get_dataloader(
    TRAIN_COVER_DIR,
    payload_bpps=PAYLOADS,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
)

# Find a batch containing both cover and stego
while True:
    batch = next(iter(loader))

    y = batch["y_det"]

    if y.unique().numel() == 2:
        break


x = batch["image"].to(DEVICE)
y = batch["y_det"].to(DEVICE)

print("Fixed batch labels:", y.tolist())


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
# OVERFIT ONE FIXED BATCH
# ============================================================

model.train()

for step in range(STEPS):

    det_logits, loc_logits = model(x)

    loss = detection_loss(det_logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(det_logits)
        pred = (probs > 0.5).float()
        acc = (pred == y).float().mean().item()

    if step == 0 or (step + 1) % 20 == 0:
        print(
            f"Step {step + 1:3d}/{STEPS} | "
            f"loss={loss.item():.6f} | "
            f"acc={acc:.2f} | "
            f"probs={[round(v, 4) for v in probs.tolist()]}"
        )


# ============================================================
# RESULT
# ============================================================

if acc == 1.0 and loss.item() < 0.1:
    print("\n[PASS] Model successfully overfit the fixed batch.")
else:
    print("\n[FAIL] Model did not overfit the fixed batch.")