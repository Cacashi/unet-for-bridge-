"""Deterministic through-tendon candidate grids."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .rules import CLASS_NAMES, legal_center_mask, load_tendon_rules


def region_masks(section_mask: np.ndarray, bands: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    rows = np.flatnonzero(np.any(section_mask > 0, axis=1))
    output = {name: np.zeros_like(section_mask, dtype=np.uint8) for name in CLASS_NAMES}
    if not rows.size:
        return output
    y_min = int(rows[0])
    y_span = max(int(rows[-1] - rows[0]), 1)
    ratio = (np.arange(section_mask.shape[0], dtype=np.float32) - y_min) / float(y_span)
    claimed = np.zeros_like(section_mask, dtype=bool)
    for name in CLASS_NAMES:
        low, high = bands[name]
        mask = (section_mask > 0) & (ratio[:, None] >= float(low)) & (ratio[:, None] <= float(high))
        mask &= ~claimed
        output[name] = mask.astype(np.uint8)
        claimed |= mask
    return output


def make_grid(mask: np.ndarray, pitch_px: float, phase: Tuple[int, int] = (0, 0)) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return np.empty((0, 2), dtype=np.float32)
    gx = np.arange(float(phase[0]), float(xs.max()) + 1.0, float(pitch_px), dtype=np.float32)
    gy = np.arange(float(phase[1]), float(ys.max()) + 1.0, float(pitch_px), dtype=np.float32)
    if not gx.size or not gy.size:
        return np.empty((0, 2), dtype=np.float32)
    xx, yy = np.meshgrid(gx, gy)
    ix = np.rint(xx).astype(np.int32).clip(0, mask.shape[1] - 1)
    iy = np.rint(yy).astype(np.int32).clip(0, mask.shape[0] - 1)
    keep = mask[iy, ix] > 0
    return np.stack([xx[keep], yy[keep]], axis=1).astype(np.float32)


def best_phase(mask: np.ndarray, pitch_px: float) -> Tuple[int, int]:
    period = max(1, int(np.ceil(pitch_px)))
    best_phase_value = (0, 0)
    best_key = (-1, 0.0)
    for ox in range(0, period, 2):
        for oy in range(0, period, 2):
            points = make_grid(mask, pitch_px, (ox, oy))
            key = (len(points), -float(np.mean(points[:, 0] + points[:, 1])) if len(points) else 0.0)
            if key > best_key:
                best_key = key
                best_phase_value = (ox, oy)
    return best_phase_value


def grid_candidates(
    predictions: Dict[str, torch.Tensor],
    section_mask: np.ndarray,
    strand_counts: List[Optional[int]],
    meta: Dict[str, float],
    mm_per_pixel: float = 5.0,
    bands: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[Dict[str, float]]]:
    bands = bands or {"top": [0.0, 0.4], "web": [0.4, 0.6], "bottom": [0.6, 1.0]}
    regions = region_masks(section_mask, bands)
    probabilities = torch.sigmoid(predictions["heatmap_logits"])[0]
    if "region_logits" in predictions:
        probabilities = probabilities * torch.softmax(predictions["region_logits"], dim=1)[0, 1:]
    result: Dict[str, List[Dict[str, float]]] = {}
    rules = load_tendon_rules()
    scale = float(meta["scale"])
    pad_x, pad_y = float(meta["pad_x"]), float(meta["pad_y"])
    out_h, out_w = probabilities.shape[-2:]
    for class_index, (name, strand_count) in enumerate(zip(CLASS_NAMES, strand_counts)):
        if strand_count is None or strand_count not in rules:
            result[name] = []
            continue
        rule = rules[strand_count]
        pitch_px = rule["pass_pass_spacing_mm"] / float(mm_per_pixel)
        legal = legal_center_mask(section_mask, name, strand_count, mm_per_pixel, "pass")
        legal = (legal & (regions[name] > 0)).astype(np.uint8)
        phase = best_phase(legal, pitch_px)
        points = make_grid(legal, pitch_px, phase)
        candidates = []
        for x, y in points:
            canvas_x = float(x) * scale + pad_x
            canvas_y = float(y) * scale + pad_y
            model_x = canvas_x / (4.0 * max(out_w - 1, 1)) * 2.0 - 1.0
            model_y = canvas_y / (4.0 * max(out_h - 1, 1)) * 2.0 - 1.0
            sample_grid = torch.tensor([[[[model_x, model_y]]]], device=probabilities.device, dtype=probabilities.dtype)
            score = float(F.grid_sample(probabilities[class_index:class_index + 1].unsqueeze(0), sample_grid, mode="bilinear", align_corners=True).item())
            candidates.append({"x": float(x), "y": float(y), "r": rule["duct_diameter_mm"] / (2.0 * mm_per_pixel), "class": name, "score": score})
        result[name] = sorted(candidates, key=lambda item: item["score"], reverse=True)
    return result
