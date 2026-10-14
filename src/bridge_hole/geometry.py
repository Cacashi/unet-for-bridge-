from typing import Dict, List, Tuple

import cv2
import numpy as np


def build_section_mask(
    image: np.ndarray,
    line_threshold: int = 200,
    close_kernel: int = 3,
) -> np.ndarray:
    """Fill material regions in a half-section line drawing.

    The symmetry-side boundary of these half sections is the left image edge.
    Starting from the known exterior at the right edge, an even/odd scan-line
    fill toggles material whenever it crosses a dark outline run. This also
    excludes cavities bounded by inner contours.
    """
    if image.ndim != 2:
        raise ValueError("build_section_mask expects a grayscale image")
    if close_kernel < 1 or close_kernel % 2 == 0:
        raise ValueError("close_kernel must be a positive odd integer")

    line = (image <= int(line_threshold)).astype(np.uint8)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, kernel)

    previous_is_line = np.pad(line[:, :-1], ((0, 0), (1, 0)), constant_values=0)
    run_starts = (line == 1) & (previous_is_line == 0)
    crossings_to_right = np.flip(
        np.cumsum(np.flip(run_starts, axis=1), axis=1), axis=1
    )
    return ((crossings_to_right % 2 == 1) & (line == 0)).astype(np.uint8)


def resize_binary_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to ``(height, width)`` without soft labels."""
    height, width = size
    resized = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_AREA
    )
    return (resized >= 0.5).astype(np.float32)


def holes_have_valid_centres(
    mask: np.ndarray, holes: List[Dict[str, float]]
) -> bool:
    """Return whether every labelled centre lies in section material."""
    height, width = mask.shape
    for hole in holes:
        x, y = int(round(float(hole["x"]))), int(round(float(hole["y"])))
        if not (0 <= x < width and 0 <= y < height and mask[y, x] > 0):
            return False
    return True
