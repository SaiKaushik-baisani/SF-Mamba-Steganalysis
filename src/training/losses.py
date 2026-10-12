"""
Loss functions for the joint detection + localization model.

L_total = L_det + alpha * L_loc
L_loc   = lambda_bce * WeightedBCE(pred, soft_target) + lambda_dice * Dice(pred, hard_target)

Why weighted BCE + Dice, not plain BCE:
At realistic payloads (0.1-0.4 bpp), only a few percent of pixels are ever
modified (see the simulator sanity check -- ~1-8% of pixels for 0.1-0.4 bpp).
Plain BCE on a >90%-negative pixel-classification problem collapses to
predicting "unchanged everywhere" and still gets a very low loss. Weighting
the positive class and adding a Dice term (which is scale-invariant to class
imbalance, unlike BCE) are both standard fixes for this in segmentation
literature, and directly necessary here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce_loss(logits: torch.Tensor, soft_target: torch.Tensor,
                       pos_weight: float = None) -> torch.Tensor:
    """
    logits: (B, 1, H, W) raw localization logits
    soft_target: (B, 1, H, W) in [0, 1] -- can be the exact hard mask or the
        p_map soft probability map from the simulator
    pos_weight: if None, computed per-batch from the target's positive rate
        (so you don't have to hand-tune it per payload level)
    """
    if pos_weight is None:
        pos_rate = soft_target.mean().clamp(min=1e-4, max=1 - 1e-4)
        pos_weight = ((1 - pos_rate) / pos_rate).detach()
    pos_weight_t = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, soft_target, pos_weight=pos_weight_t)


def dice_loss(logits: torch.Tensor, hard_target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = hard_target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def localization_loss(logits: torch.Tensor, soft_target: torch.Tensor, hard_target: torch.Tensor,
                       lambda_bce: float = 0.5, lambda_dice: float = 0.5) -> torch.Tensor:
    return (lambda_bce * weighted_bce_loss(logits, soft_target)
            + lambda_dice * dice_loss(logits, hard_target))


def detection_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)


def total_loss(det_logits, loc_logits, y_det, soft_mask, hard_mask, alpha: float = 0.3,
               lambda_bce: float = 0.5, lambda_dice: float = 0.5):
    l_det = detection_loss(det_logits, y_det)
    l_loc = localization_loss(loc_logits, soft_mask, hard_mask, lambda_bce, lambda_dice)
    return l_det + alpha * l_loc, {"det": l_det.item(), "loc": l_loc.item()}


# quick check
_det_logits = torch.randn(4)
_loc_logits = torch.randn(4, 1, 64, 64)
_y_det = torch.randint(0, 2, (4,)).float()
_hard_mask = (torch.rand(4, 1, 64, 64) < 0.05).float()  # sparse, realistic
_soft_mask = _hard_mask.clone()
_loss, _parts = total_loss(_det_logits, _loc_logits, _y_det, _soft_mask, _hard_mask)
print("total loss:", _loss.item(), _parts)
