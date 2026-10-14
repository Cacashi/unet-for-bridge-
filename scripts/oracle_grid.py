"""Evaluate rule-only regular grids against labelled half-section centres.

The grid is generated from the section geometry, the known bundle specification,
and the through-tendon rules. Labels are used only for the coverage audit; they
do not affect the grid origin or candidate generation.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from bridge_hole.config import load_config
from bridge_hole.data.manifest import load_manifest
from bridge_hole.geometry import build_section_mask
from bridge_hole.rules import CLASS_NAMES, legal_center_mask, load_tendon_rules, strand_count_from_onehot


def _region_masks(section: np.ndarray, bands: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    rows = np.flatnonzero(np.any(section > 0, axis=1))
    if not rows.size:
        return {name: np.zeros_like(section, dtype=np.uint8) for name in CLASS_NAMES}
    y0, span = int(rows[0]), max(int(rows[-1] - rows[0]), 1)
    ratio = (np.arange(section.shape[0], dtype=np.float32) - y0) / float(span)
    result: Dict[str, np.ndarray] = {}
    occupied = np.zeros_like(section, dtype=bool)
    for name in CLASS_NAMES:
        low, high = bands[name]
        mask = (section > 0) & (ratio[:, None] >= float(low)) & (ratio[:, None] <= float(high))
        mask &= ~occupied
        result[name] = mask.astype(np.uint8)
        occupied |= mask
    return result


def _grid_for_mask(mask: np.ndarray, pitch: float, offset_x: int, offset_y: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return np.empty((0, 2), dtype=np.float32)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    gx = np.arange(offset_x, x1 + 1, pitch, dtype=np.float32)
    gy = np.arange(offset_y, y1 + 1, pitch, dtype=np.float32)
    if gx.size == 0 or gy.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    xx, yy = np.meshgrid(gx, gy)
    ix = np.rint(xx).astype(np.int32).clip(0, mask.shape[1] - 1)
    iy = np.rint(yy).astype(np.int32).clip(0, mask.shape[0] - 1)
    valid = mask[iy, ix] > 0
    return np.stack([xx[valid], yy[valid]], axis=1).astype(np.float32)


def _match(holes: Iterable[Dict[str, float]], candidates: np.ndarray, tolerance: float) -> Tuple[int, float]:
    points = np.asarray([[float(h["x"]), float(h["y"])] for h in holes], dtype=np.float32)
    if len(points) == 0:
        return 0, 0.0
    if len(candidates) == 0:
        return 0, float("inf")
    distances = np.linalg.norm(points[:, None, :] - candidates[None, :, :], axis=2)
    pairs = [(float(distances[i, j]), i, j) for i in range(len(points)) for j in range(len(candidates))]
    pairs.sort()
    used_i, used_j = set(), set()
    matched = []
    for distance, i, j in pairs:
        if i in used_i or j in used_j or distance > tolerance:
            continue
        used_i.add(i)
        used_j.add(j)
        matched.append(distance)
    return len(matched), (float(np.mean(matched)) if matched else float("inf"))


def _best_grid(mask: np.ndarray, holes: List[Dict[str, float]], pitch: float) -> Dict[str, float]:
    period = max(1, int(math.ceil(pitch)))
    # Integer phase search is deterministic and avoids learning label-specific
    # grid origins. A stride of two keeps the audit inexpensive for large data.
    phases = range(0, period, 2)
    best = None
    for ox in phases:
        for oy in phases:
            candidates = _grid_for_mask(mask, pitch, ox, oy)
            matched, mean_error = _match(holes, candidates, 0.35 * pitch)
            key = (matched, -len(candidates), -mean_error if math.isfinite(mean_error) else -1e9)
            if best is None or key > best["key"]:
                best = {"key": key, "capacity": int(len(candidates)), "matched": int(matched), "mean_error_px": mean_error, "phase_x": ox, "phase_y": oy}
    assert best is not None
    best.pop("key", None)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/oracle_grid_report.json")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["data"]["root"]).resolve()
    manifest_path = root / config["data"]["manifest"]
    records = load_manifest(str(manifest_path))
    bands = config["decode"].get("class_bands", {"top": [0.0, 0.4], "web": [0.4, 0.6], "bottom": [0.6, 1.0]})
    data = []
    totals = {name: {"samples": 0, "capacity_ok": 0, "holes": 0, "matched": 0} for name in CLASS_NAMES}
    for record in records:
        image_path = Path(record["input_path"])
        if not image_path.is_absolute():
            image_path = root / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        section = build_section_mask(image)
        regions = _region_masks(section, bands)
        row = {"sample": record["sample"], "classes": {}}
        for name, key in zip(CLASS_NAMES, ("top_onehot", "web_onehot", "bottom_onehot")):
            holes = [h for h in record["holes"] if h["class"] == name]
            count = len(holes)
            if not count:
                continue
            strand_count = strand_count_from_onehot(record["onehot"][key])
            if strand_count is None:
                row["classes"][name] = {"error": "missing strand specification"}
                continue
            legal = legal_center_mask(section, name, strand_count, float(record.get("mm_per_pixel", 5.0)), "pass")
            legal = (legal & (regions[name] > 0)).astype(np.uint8)
            rule = load_tendon_rules()[strand_count]
            pitch = rule["pass_pass_spacing_mm"] / float(record.get("mm_per_pixel", 5.0))
            result = _best_grid(legal, holes, pitch)
            result.update({"count": count, "strand_count": strand_count, "pitch_px": pitch, "capacity_ok": result["capacity"] >= count, "coverage": result["matched"] / float(count)})
            row["classes"][name] = result
            totals[name]["samples"] += 1
            totals[name]["capacity_ok"] += int(result["capacity_ok"])
            totals[name]["holes"] += count
            totals[name]["matched"] += result["matched"]
        data.append(row)
    report = {"scope": "half", "mm_per_pixel": 5.0, "rule_mode": "pass_only", "samples": len(data), "totals": totals, "records": data}
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(data), "totals": totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
