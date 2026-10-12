"""
Standalone evaluation for SF-Mamba Steganalysis.

Reports:
- Detection: accuracy, balanced accuracy, ROC-AUC, precision, recall, F1
- Localization: pixel precision, pixel recall, pixel F1, Dice, IoU
- Efficiency: parameter count and inference time per image
- Per-payload evaluation through run_eval_all()
"""

import time
import json

import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


def compute_localization_metrics(probs, hard_mask, threshold=0.5):
    pred = (probs > threshold).float()

    tp = (pred * hard_mask).sum().item()
    fp = (pred * (1 - hard_mask)).sum().item()
    fn = ((1 - pred) * hard_mask).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "iou": iou,
        "dice": dice,
    }


@torch.no_grad()
def run_eval(model, loader, device, payload_filter=None):
    model.eval()

    y_true = []
    y_score = []

    loc_metrics_accum = []
    inference_times = []

    for batch in loader:
        x = batch["image"].to(device)
        y_det = batch["y_det"]
        hard_mask = batch["hard_mask"].to(device)

        t0 = time.time()

        det_logits, loc_logits = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_times.append(
            (time.time() - t0) / x.size(0)
        )

        y_true.extend(
            y_det.numpy().tolist()
        )

        y_score.extend(
            torch.sigmoid(det_logits)
            .cpu()
            .numpy()
            .tolist()
        )

        # Evaluate localization only on stego images
        stego_idx = (
            (y_det > 0.5)
            & (hard_mask.flatten(1).sum(1).cpu() > 0)
        )

        if stego_idx.any():
            probs = torch.sigmoid(
                loc_logits[stego_idx]
            )

            target = hard_mask[stego_idx]

            loc_metrics_accum.append(
                compute_localization_metrics(
                    probs,
                    target
                )
            )

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    y_pred = (y_score > 0.5).astype(float)

    # -------------------------
    # Detection metrics
    # -------------------------

    det_metrics = {
        "accuracy": float(
            (y_pred == y_true).mean()
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_score
            )
        )
        if len(set(y_true.tolist())) > 1
        else float("nan"),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
    }

    # -------------------------
    # Localization metrics
    # -------------------------

    if loc_metrics_accum:
        loc_metrics = {
            k: float(
                np.mean(
                    [m[k] for m in loc_metrics_accum]
                )
            )
            for k in loc_metrics_accum[0]
        }
    else:
        loc_metrics = {}

    # -------------------------
    # Efficiency metrics
    # -------------------------

    efficiency = {
        "params_M": (
            sum(
                p.numel()
                for p in model.parameters()
            ) / 1e6
        ),

        "ms_per_image": float(
            np.mean(inference_times) * 1000
        ),
    }

    return {
        "detection": det_metrics,
        "localization": loc_metrics,
        "efficiency": efficiency,
    }


def run_eval_all(
    checkpoint_path,
    test_cover_dir,
    image_size=256,
    batch_size=16,
    payloads=(0.1, 0.2, 0.3, 0.4),
    out_json=None,
):
    model = SFMambaSteganalysis().to(DEVICE)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    results = {}

    # -------------------------
    # Overall evaluation
    # -------------------------

    loader = get_dataloader(
        test_cover_dir,
        payload_bpps=list(payloads),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    results["overall"] = run_eval(
        model,
        loader,
        DEVICE
    )

    # -------------------------
    # Per-payload evaluation
    # -------------------------

    for bpp in payloads:

        loader = get_dataloader(
            test_cover_dir,
            payload_bpps=[bpp],
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        results[f"bpp_{bpp}"] = run_eval(
            model,
            loader,
            DEVICE
        )

    print(
        json.dumps(
            results,
            indent=2
        )
    )

    if out_json:
        with open(out_json, "w") as f:
            json.dump(
                results,
                f,
                indent=2
            )

    return results