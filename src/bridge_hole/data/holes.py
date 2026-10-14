from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


COLORS = {
    "top": (255, 0, 0),
    "web": (0, 255, 0),
    "bottom": (0, 0, 255),
}


def extract_holes(mask_path: str, class_name: str) -> List[Dict[str, float]]:
    bgr = cv2.imread(str(Path(mask_path)), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(mask_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    color = np.asarray(COLORS[class_name], dtype=np.uint8)
    binary = np.all(rgb == color, axis=2).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    holes = []
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels[y0 : y0 + height, x0 : x0 + width] == label
        ys, xs = np.nonzero(component)
        points = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)
        (_, _), enclosing_radius = cv2.minEnclosingCircle(points)
        equivalent_radius = float(np.sqrt(area / np.pi))
        radius = 0.5 * (float(enclosing_radius) + equivalent_radius)
        holes.append(
            {
                "x": float(centroids[label][0]),
                "y": float(centroids[label][1]),
                "r": max(radius, 0.5),
                "class": class_name,
            }
        )
    return holes


def extract_sample_holes(output_dir: str, sample: str) -> List[Dict[str, float]]:
    suffix = {"top": "red", "web": "green", "bottom": "blue"}
    holes = []
    for class_name, color_name in suffix.items():
        path = Path(output_dir) / (sample + "_" + color_name + ".png")
        holes.extend(extract_holes(str(path), class_name))
    return holes
