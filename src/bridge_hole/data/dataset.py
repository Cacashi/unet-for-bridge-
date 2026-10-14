import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .manifest import load_manifest
from .position import position_group
from .splits import load_splits
from .targets import encode_targets
from ..geometry import build_section_mask, resize_binary_mask
from ..rules import strand_count_from_onehot


def _resolve_manifest_path(value: str, root: Path) -> str:
    """Resolve old absolute manifest paths against the current project root."""
    path = Path(value)
    if path.exists():
        return str(path)
    normalized = str(value).replace("\\", "/")
    for marker in ("half-section/input/", "half-section/output/"):
        position = normalized.lower().find(marker)
        if position >= 0:
            return str(root / Path(normalized[position:]))
    return str(root / path.name)


def _letterbox(
    image: np.ndarray,
    holes: List[Dict[str, float]],
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, List[Dict[str, float]], Dict[str, float]]:
    target_h, target_w = target_size
    height, width = image.shape
    scale = min(target_w / float(width), target_h / float(height))
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    transformed = []
    for hole in holes:
        item = dict(hole)
        item["x"] = float(hole["x"]) * scale + pad_x
        item["y"] = float(hole["y"]) * scale + pad_y
        item["r"] = float(hole["r"]) * scale
        transformed.append(item)
    meta = {
        "scale": scale,
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "original_width": float(width),
        "original_height": float(height),
    }
    return canvas, transformed, meta


def _augment_geometry(
    image: np.ndarray,
    holes: List[Dict[str, float]],
    scale_range: Tuple[float, float],
    translate_px: float,
    rotate_deg: float,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    height, width = image.shape
    scale = float(np.random.uniform(*scale_range))
    angle = float(np.random.uniform(-rotate_deg, rotate_deg))
    tx = float(np.random.uniform(-translate_px, translate_px))
    ty = float(np.random.uniform(-translate_px, translate_px))
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[:, 2] += (tx, ty)
    transformed_image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderValue=255,
    )
    transformed_holes = []
    for hole in holes:
        point = np.asarray([hole["x"], hole["y"], 1.0], dtype=np.float32)
        x, y = matrix @ point
        item = dict(hole)
        item["x"] = float(x)
        item["y"] = float(y)
        item["r"] = float(hole["r"]) * scale
        transformed_holes.append(item)
    return transformed_image, transformed_holes


def _build_region_mask(
    section_mask: np.ndarray,
    bands: Dict[str, List[float]],
    class_names: Tuple[str, ...] = ("top", "web", "bottom"),
) -> np.ndarray:
    """Build non-leaking allowed-region targets from section geometry only."""
    output = np.zeros((len(class_names), *section_mask.shape), dtype=np.float32)
    occupied_rows = np.flatnonzero(np.any(section_mask > 0, axis=1))
    if occupied_rows.size == 0:
        return output
    y_min = int(occupied_rows[0])
    y_span = max(int(occupied_rows[-1]) - y_min, 1)
    yy = (np.arange(section_mask.shape[0], dtype=np.float32) - y_min) / float(y_span)
    for index, class_name in enumerate(class_names):
        low, high = (float(v) for v in bands[class_name])
        rows = (yy >= low) & (yy <= high)
        output[index] = section_mask.astype(np.float32) * rows[:, None].astype(np.float32)
    # Region targets must be mutually exclusive. Later channels are assigned
    # only where an earlier channel has not already claimed the pixel.
    claimed = np.zeros(section_mask.shape, dtype=bool)
    for index in range(len(class_names)):
        output[index] *= (~claimed).astype(np.float32)
        claimed |= output[index] > 0
    return output


class HoleLayoutDataset(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        split: str,
        domain: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
    ) -> None:
        self.config = config
        root = Path(config["data"].get("root", ".")).resolve()
        records = load_manifest(config["data"]["manifest"])
        for record in records:
            for key in ("input_path", "red_path", "green_path", "blue_path"):
                if key in record:
                    record[key] = _resolve_manifest_path(record[key], root)
        # Older JSON files can store group IDs as numbers instead of strings.
        split_groups = {str(group).strip() for group in load_splits(config["data"]["splits"])[split]}
        if isinstance(domain, str):
            allowed_domains = {domain.strip().upper()}
        elif domain is None:
            allowed_domains = None
        else:
            allowed_domains = {str(value).strip().upper() for value in domain}
        selected_position = config["data"].get("position_group", "all")
        if selected_position not in ("all", "zero", "nonzero"):
            raise ValueError("data.position_group must be all, zero or nonzero")
        split_records = [
            record
            for record in records
            if str(record["group_id"]).strip() in split_groups
        ]
        domain_records = [
            record for record in split_records
            if allowed_domains is None or str(record["domain"]).strip().upper() in allowed_domains
        ]
        self.records = [
            record for record in domain_records
            if selected_position == "all" or position_group(record["sample"]) == selected_position
        ]
        self.selection_summary = {
            "manifest": str(config["data"]["manifest"]),
            "splits": str(config["data"]["splits"]),
            "split": split,
            "domains": sorted(allowed_domains) if allowed_domains is not None else "all",
            "position_group": selected_position,
            "manifest_records": len(records),
            "after_split": len(split_records),
            "after_domain": len(domain_records),
            "after_position": len(self.records),
            "domain_position_counts": dict(Counter(position_group(record["sample"]) for record in domain_records)),
            "unmatched_manifest_groups": sorted({str(record["group_id"]).strip() for record in records} - split_groups),
        }
        if not self.records:
            raise ValueError(
                "No samples remain after dataset filtering: "
                + json.dumps(self.selection_summary, ensure_ascii=False)
                + ". Rebuild the finetuning manifest with "
                "python scripts/prepare_data.py --config configs/finetune_a_position_base.yaml; "
                "check the reported manifest/split/domain/position counts. "
                "With require_combine=true, the configured output directory must contain "
                "the retained *_combine.png files. Do not randomly replace the pretrained splits."
            )
        self.image_size = tuple(int(v) for v in config["data"]["image_size"])
        self.stride = int(config["model"]["output_stride"])
        augmentation = config["data"].get("augmentation", {})
        self.augment = split == "train" and bool(augmentation.get("enabled", False))
        self.augmentation = augmentation
        self.continuous_keys = config["features"]["continuous"]
        self.onehot_keys = config["features"]["onehot"]
        with Path(config["data"]["feature_stats"]).open("r", encoding="utf-8") as handle:
            self.feature_stats = json.load(handle)

    def __len__(self) -> int:
        return len(self.records)

    def _tabular(self, record: Dict[str, Any]) -> torch.Tensor:
        values = []
        for key in self.continuous_keys:
            value = float(record["continuous"][key])
            stats = self.feature_stats[key]
            values.append((value - float(stats["mean"])) / float(stats["std"]))
        for key in self.onehot_keys:
            values.extend(float(v) for v in record["onehot"][key])
        return torch.tensor(values, dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = np.asarray(Image.open(record["input_path"]).convert("L"), dtype=np.uint8)
        image, holes, meta = _letterbox(image, record["holes"], self.image_size)
        if self.augment:
            image, holes = _augment_geometry(
                image,
                holes,
                tuple(float(v) for v in self.augmentation.get("scale_range", [0.98, 1.02])),
                float(self.augmentation.get("translate_px", 4.0)),
                float(self.augmentation.get("rotate_deg", 1.0)),
            )
        section_mask = build_section_mask(image)
        outline = (255.0 - image.astype(np.float32)) / 255.0
        image_tensor = torch.from_numpy(
            np.stack([outline, section_mask.astype(np.float32)], axis=0)
        )
        targets = encode_targets(holes, self.image_size[0], self.image_size[1], self.stride)
        targets["section_mask"] = torch.from_numpy(
            resize_binary_mask(
                section_mask,
                (self.image_size[0] // self.stride, self.image_size[1] // self.stride),
            )
        ).unsqueeze(0)
        region_bands = self.config.get("decode", {}).get(
            "class_bands",
            {"top": [0.0, 0.40], "web": [0.0, 0.80], "bottom": [0.60, 1.0]},
        )
        region_mask = _build_region_mask(section_mask, region_bands)
        region_class = np.zeros(section_mask.shape, dtype=np.int64)
        for class_index in range(region_mask.shape[0]):
            region_class[region_mask[class_index] > 0.5] = class_index + 1
        targets["region_mask"] = torch.from_numpy(
            np.stack(
                [
                    resize_binary_mask(
                        channel,
                        (self.image_size[0] // self.stride, self.image_size[1] // self.stride),
                    )
                    for channel in region_mask
                ],
                axis=0,
            )
        )
        targets["region_class"] = torch.from_numpy(
            cv2.resize(
                region_class,
                (self.image_size[1] // self.stride, self.image_size[0] // self.stride),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)
        )
        # Always derive supervision counts from the half-section label holes.
        # This also makes old manifests safe: their ``counts`` field stored
        # full-section workbook counts (for example 34 instead of 17).
        label_counts = Counter(hole["class"] for hole in holes)
        half_count_values = [
            int(label_counts["top"]),
            int(label_counts["web"]),
            int(label_counts["bottom"]),
        ]
        targets["count"] = torch.tensor(half_count_values, dtype=torch.float32)
        return {
            "image": image_tensor,
            "tabular": self._tabular(record),
            "targets": targets,
            "counts": torch.tensor(
                half_count_values,
                dtype=torch.long,
            ),
            "strand_counts": torch.tensor(
                [strand_count_from_onehot(record["onehot"][key]) or 0
                 for key in ("top_onehot", "web_onehot", "bottom_onehot")],
                dtype=torch.long,
            ),
            "mm_per_pixel": float(record.get("mm_per_pixel", self.config["data"].get("mm_per_pixel", 5.0))),
            "sample": record["sample"],
            "group_id": record["group_id"],
            "meta": meta,
            "input_path": record["input_path"],
        }
