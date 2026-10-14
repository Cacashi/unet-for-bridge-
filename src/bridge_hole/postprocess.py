import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


CLASS_NAMES = ["top", "web", "bottom"]
CLASS_COLORS = {"top": (255, 0, 0), "web": (0, 255, 0), "bottom": (0, 0, 255)}


class ConstraintDecodeError(RuntimeError):
    """Raised when constrained decoding cannot satisfy a requested count.

    ``partial`` contains the points that passed all constraints before the
    failure.  Keeping it on the exception lets inference save a diagnostic
    image without silently treating an incomplete result as valid.
    """

    def __init__(self, message: str, partial: Optional[List[Dict[str, float]]] = None):
        super().__init__(message)
        self.partial = list(partial or [])


def _restore_coordinates(hole: Dict[str, float], meta: Dict[str, float]) -> Dict[str, float]:
    restored = dict(hole)
    scale = float(meta["scale"])
    restored["x"] = (hole["x"] - float(meta["pad_x"])) / scale
    restored["y"] = (hole["y"] - float(meta["pad_y"])) / scale
    restored["r"] = hole["r"] / scale
    return restored


def decode_candidates(
    predictions: Dict[str, torch.Tensor],
    counts: List[int],
    stride: int,
    multiplier: int,
    min_score: float,
    meta: Optional[Dict[str, float]] = None,
    use_region_gate: bool = False,
) -> Dict[str, List[Dict[str, float]]]:
    heatmap = torch.sigmoid(predictions["heatmap_logits"])[0]
    if use_region_gate and "region_logits" in predictions:
        region_probability = torch.softmax(predictions["region_logits"], dim=1)[0, 1:]
        heatmap = heatmap * region_probability
    pooled = F.max_pool2d(heatmap.unsqueeze(0), 3, stride=1, padding=1)[0]
    heatmap = heatmap * heatmap.eq(pooled)
    offsets = predictions["offset"][0]
    log_radius = predictions["log_radius"][0]
    result = {}
    width = heatmap.shape[2]
    for class_index, class_name in enumerate(CLASS_NAMES):
        requested = max(int(counts[class_index]), 0)
        candidate_count = min(
            max(requested * multiplier, requested, 1),
            heatmap[class_index].numel(),
        )
        scores, indices = torch.topk(heatmap[class_index].flatten(), candidate_count)
        candidates = []
        for score_tensor, index_tensor in zip(scores, indices):
            score = float(score_tensor.item())
            if score < min_score:
                continue
            index = int(index_tensor.item())
            gy, gx = divmod(index, width)
            dx = float(offsets[class_index * 2, gy, gx].item())
            dy = float(offsets[class_index * 2 + 1, gy, gx].item())
            radius = math.exp(float(log_radius[class_index, gy, gx].item())) * stride
            hole = {
                "x": (gx + dx) * stride,
                "y": (gy + dy) * stride,
                "r": radius,
                "class": class_name,
                "score": score,
            }
            candidates.append(_restore_coordinates(hole, meta) if meta else hole)
        result[class_name] = candidates
    return result


def _overlaps(
    candidate: Dict[str, float],
    selected: List[Dict[str, float]],
    clearance: float,
) -> bool:
    for other in selected:
        distance = math.hypot(candidate["x"] - other["x"], candidate["y"] - other["y"])
        if distance < candidate["r"] + other["r"] + clearance:
            return True
    return False


def constrained_select(
    candidates: Dict[str, List[Dict[str, float]]],
    counts: List[int],
    image_size: Tuple[int, int],
    min_clearance: float = 0.0,
    class_bands: Optional[Dict[str, List[float]]] = None,
    section_mask: Optional[np.ndarray] = None,
    boundary_tolerance: float = 0.0,
) -> List[Dict[str, float]]:
    height, width = image_size
    distance_to_boundary = None
    section_y_min, section_y_span = 0, max(height - 1, 1)
    if section_mask is not None:
        if section_mask.shape != (height, width):
            raise ValueError(
                "section_mask shape %s does not match image size %s"
                % (section_mask.shape, (height, width))
            )
        distance_to_boundary = cv2.distanceTransform(
            (section_mask > 0).astype(np.uint8), cv2.DIST_L2, 5
        )
        occupied_rows = np.flatnonzero(np.any(section_mask > 0, axis=1))
        if occupied_rows.size:
            section_y_min = int(occupied_rows[0])
            section_y_span = max(int(occupied_rows[-1]) - section_y_min, 1)
    selected = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        accepted = []
        for hole in candidates[class_name]:
            if hole["x"] - hole["r"] < 0 or hole["x"] + hole["r"] >= width:
                continue
            if hole["y"] - hole["r"] < 0 or hole["y"] + hole["r"] >= height:
                continue
            if distance_to_boundary is not None:
                px = int(round(hole["x"]))
                py = int(round(hole["y"]))
                if not (0 <= px < width and 0 <= py < height):
                    continue
                required_clearance = max(
                    float(hole["r"])
                    + float(min_clearance)
                    - float(boundary_tolerance),
                    0.0,
                )
                if float(distance_to_boundary[py, px]) < required_clearance:
                    continue
            if class_bands is not None:
                low, high = class_bands[class_name]
                # Bands are relative to the actual section bounding box, not
                # the mostly-white image canvas.
                ratio = (hole["y"] - section_y_min) / float(section_y_span)
                if not (low <= ratio <= high):
                    continue
            if _overlaps(hole, selected + accepted, min_clearance):
                continue
            accepted.append(hole)
            if len(accepted) == int(counts[class_index]):
                break
        if len(accepted) != int(counts[class_index]):
            raise ConstraintDecodeError(
                "%s requires %d holes, but only %d legal candidates were found"
                % (class_name, int(counts[class_index]), len(accepted)),
                partial=selected + accepted,
            )
        selected.extend(accepted)
    return selected


def mirror_holes(
    holes: List[Dict[str, float]], axis_x: float, centerline_tolerance: float
) -> List[Dict[str, float]]:
    mirrored = list(holes)
    for hole in holes:
        if abs(hole["x"] - axis_x) <= centerline_tolerance:
            continue
        item = dict(hole)
        item["x"] = 2.0 * axis_x - hole["x"]
        mirrored.append(item)
    return mirrored


def validate_no_overlap(holes: List[Dict[str, float]], clearance: float = 0.0) -> None:
    for index, hole in enumerate(holes):
        if _overlaps(hole, holes[:index], clearance):
            raise ConstraintDecodeError("Mirrored full-section layout contains a hole collision")


def render_holes(input_path: str, holes: List[Dict[str, float]], output_path: str) -> None:
    image = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for hole in holes:
        x, y, radius = hole["x"], hole["y"], hole["r"]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=CLASS_COLORS[hole["class"]],
        )
    image.save(output_path)


def render_hole_overlay(
    background_path: str,
    holes: List[Dict[str, float]],
    output_path: str,
    prediction_alpha: int = 90,
) -> None:
    """Overlay pale prediction markers on the supplied ground-truth image."""
    background = Image.open(background_path).convert("RGBA")
    layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    alpha = max(0, min(int(prediction_alpha), 255))
    for hole in holes:
        x, y, radius = hole["x"], hole["y"], hole["r"]
        color = CLASS_COLORS[hole["class"]]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(*color, alpha),
            outline=(*color, min(180, alpha + 50)),
            width=max(1, int(round(radius * 0.12))),
        )
    Image.alpha_composite(background, layer).convert("RGB").save(output_path)


def render_class_masks(
    image_size: Tuple[int, int], holes: List[Dict[str, float]], output_dir: str, sample: str
) -> None:
    height, width = image_size
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = {"top": "red", "web": "green", "bottom": "blue"}
    for class_name in CLASS_NAMES:
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        for hole in holes:
            if hole["class"] != class_name:
                continue
            x, y, radius = hole["x"], hole["y"], hole["r"]
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=CLASS_COLORS[class_name],
            )
        image.save(destination / (sample + "_" + suffixes[class_name] + ".png"))
