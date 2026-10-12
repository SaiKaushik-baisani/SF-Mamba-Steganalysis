import torch

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import SteganalysisDataset
from src.training.losses import detection_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

IMAGE_SIZE = 256
PAYLOADS = [0.1, 0.2, 0.3, 0.4]

BATCH_SIZE = 2
STEPS = 150

LEARNING_RATES = [1e-4, 3e-4, 1e-3]

print("Using device:", DEVICE)


for lr in LEARNING_RATES:

    print()
    print("=" * 60)
    print(f"TESTING LEARNING RATE: {lr}")
    print("=" * 60)

    dataset = SteganalysisDataset(
        TRAIN_COVER_DIR,
        payload_bpps=PAYLOADS,
        image_size=IMAGE_SIZE,
        seed=42,
    )

    model = SFMambaSteganalysis().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    model.train()

    losses = []

    for step in range(STEPS):

        images = []
        labels = []

        for _ in range(BATCH_SIZE):

            idx = torch.randint(
                0,
                len(dataset),
                (1,),
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

        losses.append(loss.item())

        with torch.no_grad():
            pred = (torch.sigmoid(det_logits) > 0.5).float()
            acc = (pred == y).float().mean().item()

        if step == 0 or (step + 1) % 25 == 0:
            print(
                f"Step {step + 1:3d}/{STEPS} | "
                f"loss={loss.item():.6f} | "
                f"acc={acc:.2f}"
            )

    print()
    print(
        f"LR {lr} final loss: {losses[-1]:.6f}"
    )


print()
print("=" * 60)
print("LEARNING RATE TEST COMPLETE")
print("=" * 60)