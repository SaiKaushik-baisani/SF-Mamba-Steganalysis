import torch
import numpy as np

from src.datasets.dataset import get_dataloader


# ============================================================
# CONFIG
# ============================================================

TRAIN_COVER_DIR = r"data\BOSSbase\split\train"

IMAGE_SIZE = 256
PAYLOADS = [0.1, 0.2, 0.3, 0.4]

BATCH_SIZE = 8

print("Starting dataset diagnostic...")


# ============================================================
# DATA LOADER
# ============================================================

loader = get_dataloader(
    TRAIN_COVER_DIR,
    payload_bpps=PAYLOADS,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
)


# ============================================================
# COLLECT MULTIPLE BATCHES
# ============================================================

for batch_number in range(5):

    batch = next(iter(loader))

    images = batch["image"]
    labels = batch["y_det"]
    hard_masks = batch["hard_mask"]
    soft_masks = batch["soft_mask"]

    print("\n" + "=" * 60)
    print(f"BATCH {batch_number + 1}")
    print("=" * 60)

    print("labels:", labels.tolist())

    print(
        "cover count:",
        int((labels == 0).sum()),
        "| stego count:",
        int((labels == 1).sum()),
    )

    # --------------------------------------------------------
    # Image statistics
    # --------------------------------------------------------

    print(
        "image min/max:",
        images.min().item(),
        images.max().item(),
    )

    print(
        "image mean/std:",
        images.mean().item(),
        images.std().item(),
    )

    # --------------------------------------------------------
    # Localization masks
    # --------------------------------------------------------

    hard_pixels = hard_masks.flatten(1).sum(dim=1)

    soft_values = soft_masks.flatten(1).sum(dim=1)

    print(
        "hard-mask pixels per sample:",
        hard_pixels.tolist(),
    )

    print(
        "soft-mask sum per sample:",
        soft_values.tolist(),
    )

    # --------------------------------------------------------
    # Check consistency
    # --------------------------------------------------------

    for i in range(BATCH_SIZE):

        if labels[i].item() == 0:
            if hard_pixels[i].item() != 0:
                print(
                    f"[WARNING] Sample {i}: cover has non-zero hard mask!"
                )

        if labels[i].item() == 1:
            if hard_pixels[i].item() == 0:
                print(
                    f"[WARNING] Sample {i}: stego has ZERO hard-mask pixels!"
                )

    # --------------------------------------------------------
    # Check for invalid values
    # --------------------------------------------------------

    if not torch.isfinite(images).all():
        print("[ERROR] Images contain NaN/Inf!")

    if not torch.isfinite(hard_masks).all():
        print("[ERROR] Hard masks contain NaN/Inf!")

    if not torch.isfinite(soft_masks).all():
        print("[ERROR] Soft masks contain NaN/Inf!")

    print("[OK] Batch completed.")


print("\n" + "=" * 60)
print("DATASET DIAGNOSTIC COMPLETE")
print("=" * 60)