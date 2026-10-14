import math
from typing import Dict, List

import numpy as np
import torch


CLASS_INDEX = {"top": 0, "web": 1, "bottom": 2}


def _gaussian_radius(radius_grid: float) -> float:
    return max(radius_grid / 2.0, 1.0)


def _draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, sigma: float) -> None:
    radius = max(int(3.0 * sigma), 1)
    height, width = heatmap.shape
    left, right = min(cx, radius), min(width - cx - 1, radius)
    top, bottom = min(cy, radius), min(height - cy - 1, radius)
    if min(left, right, top, bottom) < 0:
        return
    size = 2 * radius + 1
    coords = np.arange(size, dtype=np.float32) - radius
    gaussian = np.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / (2.0 * sigma ** 2))
    patch = heatmap[cy - top : cy + bottom + 1, cx - left : cx + right + 1]
    kernel = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    np.maximum(patch, kernel, out=patch)


def _add_gaussian(heatmap: np.ndarray, cx: int, cy: int, sigma: float) -> None:
    """Add a Gaussian density contribution instead of taking a maximum."""
    radius = max(int(3.0 * sigma), 1)
    height, width = heatmap.shape
    left, right = min(cx, radius), min(width - cx - 1, radius)
    top, bottom = min(cy, radius), min(height - cy - 1, radius)
    if min(left, right, top, bottom) < 0:
        return
    size = 2 * radius + 1
    coords = np.arange(size, dtype=np.float32) - radius
    kernel = np.exp(
        -(coords[:, None] ** 2 + coords[None, :] ** 2) / (2.0 * sigma ** 2)
    )
    patch = heatmap[cy - top : cy + bottom + 1, cx - left : cx + right + 1]
    patch += kernel[radius - top : radius + bottom + 1, radius - left : radius + right + 1]


def _draw_disk(mask: np.ndarray, cx: int, cy: int, radius: float) -> None:
    radius = max(float(radius), 1.0)
    x0, x1 = max(0, int(cx - radius)), min(mask.shape[1] - 1, int(cx + radius))
    y0, y1 = max(0, int(cy - radius)), min(mask.shape[0] - 1, int(cy + radius))
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask[y0 : y1 + 1, x0 : x1 + 1][(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] = 1.0


def encode_targets(
    holes: List[Dict[str, float]], image_height: int, image_width: int, stride: int
) -> Dict[str, torch.Tensor]:
    out_h, out_w = image_height // stride, image_width // stride
    heatmap = np.zeros((3, out_h, out_w), dtype=np.float32)
    mask = np.zeros((3, out_h, out_w), dtype=np.float32)
    density = np.zeros((3, out_h, out_w), dtype=np.float32)
    offset = np.zeros((6, out_h, out_w), dtype=np.float32)
    log_radius = np.zeros((3, out_h, out_w), dtype=np.float32)
    center_mask = np.zeros((3, out_h, out_w), dtype=np.float32)
    for hole in holes:
        class_index = CLASS_INDEX[hole["class"]]
        x_grid = float(hole["x"]) / stride
        y_grid = float(hole["y"]) / stride
        gx, gy = int(math.floor(x_grid)), int(math.floor(y_grid))
        if not (0 <= gx < out_w and 0 <= gy < out_h):
            continue
        radius_grid = max(float(hole["r"]) / stride, 1e-3)
        _draw_gaussian(heatmap[class_index], gx, gy, _gaussian_radius(radius_grid))
        _draw_disk(mask[class_index], gx, gy, radius_grid)
        _add_gaussian(density[class_index], gx, gy, _gaussian_radius(radius_grid))
        offset[class_index * 2, gy, gx] = x_grid - gx
        offset[class_index * 2 + 1, gy, gx] = y_grid - gy
        log_radius[class_index, gy, gx] = math.log(radius_grid)
        center_mask[class_index, gy, gx] = 1.0
    return {
        "heatmap": torch.from_numpy(heatmap),
        "mask": torch.from_numpy(mask),
        "density": torch.from_numpy(density),
        "offset": torch.from_numpy(offset),
        "log_radius": torch.from_numpy(log_radius),
        "center_mask": torch.from_numpy(center_mask),
    }
