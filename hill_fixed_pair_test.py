import torch

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import SteganalysisDataset
from src.training.losses import detection_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

print("Using device:", DEVICE)

# --------------------------------------------------
# 1. Generate ONE cover and ONE stego sample
# --------------------------------------------------

dataset = SteganalysisDataset(
    TRAIN_COVER_DIR,
    payload_bpps=[0.4],
    image_size=256,
    seed=42,
)

cover_idx = 0

# Keep asking until we get one cover and one stego.
cover_sample = None
stego_sample = None

while cover_sample is None or stego_sample is None:

    sample = dataset[cover_idx]

    if sample["y_det"].item() == 0:
        cover_sample = sample
    else:
        stego_sample = sample

print("Generated fixed cover/stego pair")

print(
    "Cover:",
    "label =", cover_sample["y_det"].item(),
    "hard pixels =", int(cover_sample["hard_mask"].sum().item())
)

print(
    "Stego:",
    "label =", stego_sample["y_det"].item(),
    "hard pixels =", int(stego_sample["hard_mask"].sum().item())
)

# --------------------------------------------------
# 2. Put the SAME two images into every batch
# --------------------------------------------------

x = torch.stack([
    cover_sample["image"],
    stego_sample["image"],
]).to(DEVICE)

y = torch.tensor(
    [0.0, 1.0],
    dtype=torch.float32,
    device=DEVICE,
)

print("Input shape:", x.shape)
print("Labels:", y.tolist())

# --------------------------------------------------
# 3. Train only on these exact two images
# --------------------------------------------------

model = SFMambaSteganalysis().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.0,
)

model.train()

STEPS = 200

for step in range(STEPS):

    det_logits, _ = model(x)

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
            f"Step {step+1:3d}/{STEPS} | "
            f"loss={loss.item():.6f} | "
            f"acc={acc:.2f} | "
            f"probs={probs.detach().cpu().tolist()}"
        )

print()
print("=" * 60)
print("HILL FIXED PAIR TEST COMPLETE")
print("=" * 60)