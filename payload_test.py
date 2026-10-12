import torch

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import SteganalysisDataset
from src.training.losses import detection_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

IMAGE_SIZE = 256

PAYLOADS_TO_TEST = [0.1, 0.2, 0.3, 0.4]

SAMPLES = 40
STEPS = 60
BATCH_SIZE = 2
LR = 1e-4


print("Using device:", DEVICE)


for payload in PAYLOADS_TO_TEST:

    print()
    print("=" * 60)
    print(f"TESTING PAYLOAD: {payload} bpp")
    print("=" * 60)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SteganalysisDataset(
        TRAIN_COVER_DIR,
        payload_bpps=[payload],
        image_size=IMAGE_SIZE,
        seed=42,
    )

    # --------------------------------------------------------
    # Create fixed balanced dataset
    # --------------------------------------------------------

    images = []
    labels = []

    idx = 0

    while len(images) < SAMPLES:

        sample = dataset[idx]

        # Force a balanced dataset:
        # alternate cover/stego generation.
        #
        # Dataset itself is stochastic, so keep sampling
        # until we have roughly equal numbers.

        if len(labels) == 0:
            desired_label = 0.0
        elif labels.count(0.0) > labels.count(1.0):
            desired_label = 1.0
        else:
            desired_label = 0.0

        if sample["y_det"].item() == desired_label:
            images.append(sample["image"])
            labels.append(sample["y_det"].item())

        idx += 1

    x = torch.stack(images).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.float32).to(DEVICE)

    print(
        "Dataset:",
        x.shape,
        "| cover:",
        int((y == 0).sum()),
        "| stego:",
        int((y == 1).sum()),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SFMambaSteganalysis().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.0,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.train()

    for step in range(STEPS):

        perm = torch.randperm(SAMPLES, device=DEVICE)

        total_loss = 0.0
        correct = 0

        for start in range(0, SAMPLES, BATCH_SIZE):

            ids = perm[start:start + BATCH_SIZE]

            xb = x[ids]
            yb = y[ids]

            logits, _ = model(xb)

            loss = detection_loss(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(ids)

            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                correct += (pred == yb).sum().item()

        avg_loss = total_loss / SAMPLES
        acc = correct / SAMPLES

        if step == 0 or (step + 1) % 20 == 0:
            print(
                f"Step {step + 1:3d}/{STEPS} | "
                f"loss={avg_loss:.6f} | "
                f"acc={acc:.3f}"
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits, _ = model(x)

        pred = (torch.sigmoid(logits) > 0.5).float()

        final_acc = (pred == y).float().mean().item()

    print(f"FINAL ACCURACY: {final_acc:.3f}")

    if final_acc >= 0.90:
        print("[PASS] Payload is learnable.")
    else:
        print("[WARNING] Payload did not reach 90% accuracy.")


print()
print("=" * 60)
print("ALL PAYLOAD TESTS COMPLETE")
print("=" * 60)