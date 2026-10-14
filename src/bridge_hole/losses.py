from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def heatmap_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """CenterNet-style modified focal loss and diagnostics for heatmaps."""
    # Keep this entire calculation in float32. Under CUDA AMP, summing the
    # loss for all heatmap pixels in float16 can overflow to infinity.
    alpha, beta = 2.0, 4.0
    logits = logits.float()
    target = target.float()
    prediction = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    positive = target.eq(1.0).float()
    negative = target.lt(1.0).float()
    negative_weight = torch.pow(1.0 - target, beta)
    positive_loss = (
        -torch.log(prediction) * torch.pow(1.0 - prediction, alpha) * positive
    )
    negative_loss = (
        -torch.log(1.0 - prediction)
        * torch.pow(prediction, alpha)
        * negative_weight
        * negative
    )
    num_pos = positive.sum()
    if num_pos.item() > 0:
        denominator = num_pos
        heatmap = (positive_loss.sum() + negative_loss.sum()) / denominator
        positive_value = positive_loss.sum() / denominator
        negative_value = negative_loss.sum() / denominator
    else:
        # Pure-background samples have no positive center to normalize by.
        # Use the mean negative loss so such a sample remains finite and does
        # not dominate the batch loss merely because it has many pixels.
        denominator = negative.sum().clamp(min=1.0)
        positive_value = positive_loss.sum()
        negative_value = negative_loss.sum() / denominator
        heatmap = negative_value
    return {
        "heatmap": heatmap,
        "heatmap_positive": positive_value,
        "heatmap_negative": negative_value,
        "num_pos": num_pos.detach(),
        "pred_heatmap_max": prediction.detach().amax(),
        "pred_heatmap_mean": prediction.detach().mean(),
    }


class HoleLayoutLoss(nn.Module):
    def __init__(
        self,
        heatmap_weight: float,
        offset_weight: float,
        radius_weight: float,
        mask_weight: float = 0.5,
        density_weight: float = 0.1,
        count_weight: float = 0.1,
        geometry_weight: float = 0.5,
        region_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.radius_weight = radius_weight
        self.mask_weight = mask_weight
        self.density_weight = density_weight
        self.count_weight = count_weight
        self.geometry_weight = geometry_weight
        self.region_weight = region_weight

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        heatmap_values = heatmap_focal_loss(
            predictions["heatmap_logits"], targets["heatmap"]
        )
        heatmap = heatmap_values["heatmap"]
        class_mask = targets["center_mask"].float()
        offset_mask = class_mask.repeat_interleave(2, dim=1)
        offset_denominator = offset_mask.sum().clamp(min=1.0)
        radius_denominator = class_mask.sum().clamp(min=1.0)
        offset = (
            F.l1_loss(
                predictions["offset"].float(), targets["offset"].float(), reduction="none"
            )
            * offset_mask
        ).sum() / offset_denominator
        radius = (
            F.smooth_l1_loss(
                predictions["log_radius"].float(),
                targets["log_radius"].float(),
                reduction="none",
            )
            * class_mask
        ).sum() / radius_denominator
        mask_logits = predictions["mask_logits"].float()
        mask_target = targets["mask"].float()
        mask_bce = F.binary_cross_entropy_with_logits(mask_logits, mask_target)
        mask_prob = torch.sigmoid(mask_logits)
        dice_denominator = mask_prob.sum(dim=(-2, -1)) + mask_target.sum(dim=(-2, -1))
        dice = 1.0 - (
            2.0 * (mask_prob * mask_target).sum(dim=(-2, -1)) + 1.0
        ) / (dice_denominator + 1.0)
        mask = mask_bce + dice.mean()
        density = F.smooth_l1_loss(
            predictions["density"].float(), targets["density"].float()
        )
        count = F.smooth_l1_loss(
            predictions["count"].float(), targets["count"].float()
        )
        section_mask = targets["section_mask"].float()
        outside = (1.0 - section_mask).expand_as(predictions["heatmap_logits"])
        outside_denominator = outside.sum().clamp(min=1.0)
        geometry = (
            F.softplus(predictions["heatmap_logits"].float()) * outside
        ).sum() / outside_denominator
        region = F.cross_entropy(
            predictions["region_logits"].float(), targets["region_class"].long()
        )
        total = (
            self.heatmap_weight * heatmap
            + self.offset_weight * offset
            + self.mask_weight * mask
            + self.radius_weight * radius
            + self.density_weight * density
            + self.count_weight * count
            + self.geometry_weight * geometry
            + self.region_weight * region
        )
        return {
            "total": total,
            "heatmap": heatmap,
            "offset": offset,
            "mask": mask,
            "radius": radius,
            "density": density,
            "count": count,
            "geometry": geometry,
            "region": region,
            "heatmap_positive": heatmap_values["heatmap_positive"],
            "heatmap_negative": heatmap_values["heatmap_negative"],
            "num_pos": heatmap_values["num_pos"],
            "pred_heatmap_max": heatmap_values["pred_heatmap_max"],
            "pred_heatmap_mean": heatmap_values["pred_heatmap_mean"],
        }
