import os
import json
import time

import torch

from src.models.sf_mamba import SFMambaSteganalysis
from src.datasets.dataset import get_dataloader
from src.training.losses import total_loss


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# ============================================================
# TRAINING CONFIG
# ============================================================

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"
VAL_COVER_DIR = r"data\BOSSbase\split\val"

OUT_DIR = "runs/exp1"

IMAGE_SIZE = 256
PAYLOADS = [0.1, 0.2, 0.3, 0.4]

BATCH_SIZE = 2
NUM_WORKERS = 0

LR = 1e-4
WEIGHT_DECAY = 1e-4

EPOCHS = 2

STAGE1_EPOCHS = 999
STAGE2_EPOCHS = 20

STAGE2_ALPHA = 0.1
STAGE3_ALPHA = 0.3

EVAL_EVERY = 1
MAX_TRAIN_BATCHES = None  # None = full training; use 5 for a smoke test


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# MODEL
# ============================================================

model = SFMambaSteganalysis().to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())

print(f"Model params: {n_params:,} ({n_params / 1e6:.2f}M)")


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = get_dataloader(
    TRAIN_COVER_DIR,
    payload_bpps=PAYLOADS,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
)

val_loader = get_dataloader(
    VAL_COVER_DIR,
    payload_bpps=PAYLOADS,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    seed=42,
    deterministic=True,
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# ============================================================
# OPTIMIZER + SCHEDULER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)


# ============================================================
# TRAINING
# ============================================================

def alpha_for_epoch(epoch):
    if epoch < STAGE1_EPOCHS:
        return 0.0
    elif epoch < STAGE1_EPOCHS + STAGE2_EPOCHS:
        return STAGE2_ALPHA
    else:
        return STAGE3_ALPHA



def evaluate_quick(model, loader, device, max_batches=None):
    model.eval()

    n, correct = 0, 0
    dice_sum, dice_n = 0.0, 0

    with torch.no_grad():
        for i, batch in enumerate(loader):

            if max_batches and i >= max_batches:
                break

            x = batch["image"].to(device)
            y_det = batch["y_det"].to(device)
            hard_mask = batch["hard_mask"].to(device)

            det_logits, loc_logits = model(x)

            pred = (torch.sigmoid(det_logits) > 0.5).float()

            correct += (pred == y_det).sum().item()
            n += y_det.numel()

            stego_idx = (
                (y_det > 0.5)
                & (hard_mask.flatten(1).sum(1) > 0)
            )

            if stego_idx.any():

                probs = torch.sigmoid(loc_logits[stego_idx])
                target = hard_mask[stego_idx]

                inter = (
                    (probs * target)
                    .flatten(1)
                    .sum(1)
                )

                union = (
                    probs.flatten(1).sum(1)
                    + target.flatten(1).sum(1)
                )

                dice = (2 * inter + 1e-6) / (union + 1e-6)

                dice_sum += dice.sum().item()
                dice_n += dice.numel()

    model.train()

    return {
        "detection_acc": correct / max(n, 1),
        "localization_dice": dice_sum / max(dice_n, 1),
    }


best_val_acc = 0.0
log_path = os.path.join(OUT_DIR, "log.jsonl")


for epoch in range(EPOCHS):

    alpha = alpha_for_epoch(epoch)
    t0 = time.time()

    running = {
        "total": 0.0,
        "det": 0.0,
        "loc": 0.0,
        "n": 0,
    }

    model.train()

    for batch_idx, batch in enumerate(train_loader):

        if MAX_TRAIN_BATCHES is not None and batch_idx >= MAX_TRAIN_BATCHES:
            break

        # -------------------------
        # DATA → GPU
        # -------------------------
        t_data = time.time()

        x = batch["image"].to(DEVICE)
        y_det = batch["y_det"].to(DEVICE)
        hard_mask = batch["hard_mask"].to(DEVICE)
        soft_mask = batch["soft_mask"].to(DEVICE)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        data_time = time.time() - t_data

        # -------------------------
        # FORWARD
        # -------------------------
        t_forward = time.time()

        det_logits, loc_logits = model(x)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        forward_time = time.time() - t_forward

        # -------------------------
        # LOSS + BACKWARD
        # -------------------------
        t_backward = time.time()

        loss, parts = total_loss(
            det_logits,
            loc_logits,
            y_det,
            soft_mask,
            hard_mask,
            alpha=alpha,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        backward_time = time.time() - t_backward

        # -------------------------
        # PRINT FIRST 5 BATCHES
        # -------------------------
        if batch_idx < 5:
            print(
                f"Batch {batch_idx}: "
                f"data={data_time:.3f}s | "
                f"forward={forward_time:.3f}s | "
                f"backward={backward_time:.3f}s"
            )

        bs = x.size(0)

        running["total"] += loss.item() * bs
        running["det"] += parts["det"] * bs
        running["loc"] += parts["loc"] * bs
        running["n"] += bs

    scheduler.step()

    train_metrics = {
        k: running[k] / running["n"]
        for k in ["total", "det", "loc"]
    }

    log_entry = {
        "epoch": epoch,
        "alpha": alpha,
        "time_s": round(time.time() - t0, 1),
        **{
            f"train_{k}": v
            for k, v in train_metrics.items()
        },
    }

    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
        val_metrics = evaluate_quick(
            model,
            val_loader,
            DEVICE,
            max_batches=50
        )

        log_entry.update({
            f"val_{k}": v
            for k, v in val_metrics.items()
        })

        if val_metrics["detection_acc"] > best_val_acc:
            best_val_acc = val_metrics["detection_acc"]

            torch.save(
                model.state_dict(),
                os.path.join(OUT_DIR, "best_model.pt")
            )

    print(json.dumps(log_entry))

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

torch.save(
    model.state_dict(),
    os.path.join(OUT_DIR, "final_model.pt")
)

print("Training complete. Best val detection acc:", best_val_acc)