import ast
import ast
from collections import Counter
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List

import cv2
import openpyxl
from tqdm import tqdm

from ..geometry import build_section_mask, holes_have_valid_centres
from .holes import extract_sample_holes


def count_holes(holes: List[Dict[str, float]]) -> Dict[str, int]:
    values = Counter(hole["class"] for hole in holes)
    return {
        "top": int(values["top"]),
        "web": int(values["web"]),
        "bottom": int(values["bottom"]),
        "total": int(len(holes)),
    }


def _parse_onehot(value: Any, length: int) -> List[float]:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    elif isinstance(value, (list, tuple)):
        parsed = value
    else:
        parsed = []
    result = [float(v) for v in parsed]
    if len(result) != length:
        raise ValueError("One-hot length is %d, expected %d" % (len(result), length))
    return result


def _row_value(row: Dict[str, Any], *keys: str) -> Any:
    """Return the first present field, supporting old and normalized workbooks."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError("None of the workbook columns exist: " + ", ".join(keys))


def _parse_onehot_compat(row: Dict[str, Any], key: str, length: int) -> List[float]:
    value = row.get(key)
    if value is not None:
        return _parse_onehot(value, length)

    # The normalized data_train workbook stores active strand counts (7..27)
    # as a compact string such as ``9`` or ``9,12`` instead of the original
    # vector representation.
    specs_key = key.replace("_onehot", "_onehot_specs")
    specs = row.get(specs_key)
    if specs is None or str(specs).strip() == "":
        return [0.0] * length
    active = [int(part.strip()) for part in str(specs).split(",") if part.strip()]
    result = [0.0] * length
    for count in active:
        index = count - 7
        if not 0 <= index < length:
            raise ValueError(
                "One-hot strand count %d maps outside a vector of length %d" % (count, length)
            )
        result[index] = 1.0
    return result


def _find_domain_column(headers: List[str], rows: List[tuple]) -> str:
    for index, header in enumerate(headers):
        values = {str(row[index]).strip().upper() for row in rows if row[index] is not None}
        if values and values.issubset({"A", "B"}):
            return header
    raise ValueError("Cannot find the A/B domain column")


def build_manifest(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_cfg = config["data"]
    root = Path(data_cfg["root"]).resolve()
    workbook_path = root / data_cfg["workbook"]
    workbook = openpyxl.load_workbook(str(workbook_path), read_only=True, data_only=True)
    worksheet = workbook[data_cfg["worksheet"]]
    rows = list(worksheet.iter_rows(values_only=True))
    headers = [str(v) for v in rows[0]]
    body = [row for row in rows[1:] if any(v is not None for v in row)]
    domain_column = _find_domain_column(headers, body)
    length = int(config["features"]["onehot_length"])
    records = []
    excluded_samples = {str(sample) for sample in data_cfg.get("excluded_samples", [])}
    for raw in tqdm(body, desc="Building manifest", unit="sample", total=len(body)):
        row = dict(zip(headers, raw))
        sample = str(_row_value(row, "sample", "sample_id"))
        # Exclude known-bad samples before opening their label masks. This also
        # lets a stale workbook be used safely after invalid files are removed.
        if sample in excluded_samples:
            continue
        output_dir = root / data_cfg["output_dir"]
        if data_cfg.get("require_combine", False) and not (
            output_dir / (sample + "_combine.png")
        ).is_file():
            continue
        input_path = root / data_cfg["input_dir"] / (sample + ".png")
        if not input_path.is_file():
            raise FileNotFoundError(
                "Input image for %s is missing: %s" % (sample, input_path)
            )
        holes = extract_sample_holes(str(output_dir), sample)
        if bool(data_cfg.get("exclude_invalid_geometry", True)):
            input_image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
            if input_image is None:
                raise FileNotFoundError(input_path)
            section_mask = build_section_mask(input_image)
            if not holes_have_valid_centres(section_mask, holes):
                warnings.warn(
                    "%s was excluded because at least one labelled centre is outside "
                    "the section" % sample,
                    stacklevel=2,
                )
                continue
        half_counts = count_holes(holes)
        full_counts = {
            "top": int(_row_value(row, "red_components", "full_top_component_count")),
            "web": int(_row_value(row, "green_components", "full_web_component_count")),
            "bottom": int(_row_value(row, "blue_components", "full_bottom_component_count")),
            "total": int(_row_value(row, "holes_count", "full_hole_count")),
        }
        group_column = data_cfg.get("group_column", "id")
        record = {
            "sample": sample,
            "group_id": str(_row_value(row, group_column, "group_id", "id")),
            "domain": str(row[domain_column]).strip().upper(),
            # Keep manifests portable across machines. Dataset loading still
            # accepts older manifests containing absolute paths.
            "input_path": str(input_path.relative_to(root)),
            "red_path": str((output_dir / (sample + "_red.png")).relative_to(root)),
            "green_path": str((output_dir / (sample + "_green.png")).relative_to(root)),
            "blue_path": str((output_dir / (sample + "_blue.png")).relative_to(root)),
            # The masks are half sections, while the workbook stores full-
            # section design counts. Never supervise a half heatmap with full
            # counts; retain both scopes explicitly.
            "counts": half_counts,
            "half_counts": half_counts,
            "count_scope": "half",
            "mm_per_pixel": float(data_cfg.get("mm_per_pixel", 5.0)),
            "rule_mode": str(data_cfg.get("rule_mode", "pass_only")),
            # Filled from the longitudinal tendon schedule when that table is
            # available. Do not synthesize tendon IDs from anonymous centres.
            "tendons": None,
            "half_top_count": int(half_counts["top"]),
            "half_web_count": int(half_counts["web"]),
            "half_bottom_count": int(half_counts["bottom"]),
            "full_counts": full_counts,
            "continuous": {
                key: (
                    float(half_counts["top"])
                    if key == "half_top_count"
                    else float(half_counts["web"])
                    if key == "half_web_count"
                    else float(half_counts["bottom"])
                    if key == "half_bottom_count"
                    else float(row[key])
                )
                for key in config["features"]["continuous"]
            },
            "onehot": {
                key: _parse_onehot_compat(row, key, length)
                for key in config["features"]["onehot"]
            },
            "holes": holes,
        }
        records.append(record)
    return records


def save_manifest(records: List[Dict[str, Any]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_manifest(path: str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_feature_stats(
    records: List[Dict[str, Any]], train_groups: List[str], keys: List[str], path: str
) -> None:
    allowed = set(train_groups)
    train_records = [record for record in records if record["group_id"] in allowed]
    stats = {}
    for key in keys:
        values = [float(record["continuous"][key]) for record in train_records]
        mean = sum(values) / max(len(values), 1)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        stats[key] = {"mean": mean, "std": max(math.sqrt(variance), 1e-6)}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
