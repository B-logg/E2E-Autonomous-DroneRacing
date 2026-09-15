"""GateNet losses (paper Appendix A, "Training setup").

Per output map:      L_i = Dice(y_i, y_hat_i) + 2 * BCE(y_i, y_hat_i)
Across the five:     L   = 4*L_0 + 2*L_1 + L_2 + L_3 + L_4

PAPER-AMBIGUITY (docs/paper_gaps.md A8): the appendix figure numbers the heads
`outc0` at the bottleneck and `outc4` at full resolution, which would make
`4*L_0` weight the *coarsest* map most -- the opposite of the stated intent,
"output-specific scaling factors to emphasize higher-resolution predictions".
We follow the intent: index 0 is the finest map.  `GateNet.forward` returns
finest-first to match.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE_WEIGHTS = (4.0, 2.0, 1.0, 1.0, 1.0)
BCE_WEIGHT = 2.0


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Soft Dice on probabilities, averaged over the batch."""
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.dim()))
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def downsample_to(target: torch.Tensor, size) -> torch.Tensor:
    """Area-average the full-resolution label down to a head's resolution.

    Area (not nearest) so a thin gate rail that falls between output pixels
    still leaves a signal instead of vanishing; re-thresholded low for the same
    reason."""
    if tuple(target.shape[-2:]) == tuple(size):
        return target
    return (F.adaptive_avg_pool2d(target, size) > 0.0).float()


def multiscale_loss(outputs: list[torch.Tensor], target: torch.Tensor):
    """`outputs` finest-first, `target` a full-resolution {0,1} mask (B,1,H,W)."""
    assert len(outputs) == len(SCALE_WEIGHTS), (len(outputs), len(SCALE_WEIGHTS))
    total = target.new_zeros(())
    per_head = []
    for out, w in zip(outputs, SCALE_WEIGHTS):
        tgt = downsample_to(target, out.shape[-2:])
        li = dice_loss(out, tgt) + BCE_WEIGHT * F.binary_cross_entropy_with_logits(out, tgt)
        per_head.append(li.detach())
        total = total + w * li
    return total, per_head


@torch.no_grad()
def iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Intersection over union of the full-resolution head -- the number to
    watch, since Dice/BCE keep falling long after the mask stops improving."""
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = tuple(range(1, pred.dim()))
    inter = (pred * target).sum(dims)
    union = ((pred + target) > 0).float().sum(dims)
    return torch.where(union > 0, inter / union, torch.ones_like(union)).mean()


class GateNetLoss(nn.Module):
    def forward(self, outputs, target):
        return multiscale_loss(outputs, target)
