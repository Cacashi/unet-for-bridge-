"""Railway tendon spacing rules and pixel-space legal masks."""

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, Optional

import cv2
import numpy as np


CLASS_NAMES = ("top", "web", "bottom")


@lru_cache(maxsize=1)
def load_tendon_rules(xml_path: Optional[str] = None) -> Dict[int, Dict[str, float]]:
    path = Path(xml_path) if xml_path else Path(__file__).resolve().parents[2] / "ruler" / "铁路钢束空间参数库.xml"
    root = ET.parse(path).getroot()
    result: Dict[int, Dict[str, float]] = {}
    for node in root.findall("./tendon_specifications/tendon_specification"):
        count = int(node.attrib["strand_count"])
        duct = node.find("duct")
        spacing = node.find("spacing")
        jack = node.find("tensioning_jack")
        edge = node.find("anchor_edge")
        assert duct is not None and spacing is not None and jack is not None and edge is not None
        result[count] = {
            "strand_count": count,
            "bundle_spec": node.attrib["bundle_spec"],
            "anchor_model": node.attrib["anchor_model"],
            "duct_diameter_mm": float(duct.attrib["inner_diameter_mm"]),
            "pass_top_web_edge_mm": float(duct.attrib["top_web_edge_mm"]),
            "pass_bottom_edge_mm": float(duct.attrib["bottom_slab_edge_mm"]),
            "pass_pass_spacing_mm": float(spacing.attrib["pass_to_pass_center_mm"]),
            "anchor_anchor_spacing_mm": float(spacing.attrib["anchor_to_anchor_center_mm"]),
            "anchor_pass_spacing_mm": float(spacing.attrib["anchor_to_pass_center_mm"]),
            "jack_top_edge_mm": float(jack.attrib["top_slab_control_edge_mm"]),
            "jack_bottom_edge_mm": float(jack.attrib["bottom_slab_control_edge_mm"]),
            "anchor_top_bottom_edge_mm": float(edge.attrib["top_bottom_slab_mm"]),
            "anchor_web_edge_mm": float(edge.attrib["web_mm"]),
        }
    return result


def strand_count_from_onehot(onehot: Iterable[float]) -> Optional[int]:
    values = list(float(v) for v in onehot)
    if not values or max(values) <= 0.5:
        return None
    return 7 + int(np.argmax(values))


def boundary_requirement_mm(rule: Dict[str, float], class_name: str, role: str = "pass") -> float:
    if role == "pass":
        return rule["pass_bottom_edge_mm"] if class_name == "bottom" else rule["pass_top_web_edge_mm"]
    return rule["anchor_top_bottom_edge_mm"] if class_name in ("top", "bottom") else rule["anchor_web_edge_mm"]


def pair_spacing_mm(rule_a: Dict[str, float], role_a: str, rule_b: Dict[str, float], role_b: str) -> float:
    if role_a == "pass" and role_b == "pass":
        return max(rule_a["pass_pass_spacing_mm"], rule_b["pass_pass_spacing_mm"])
    if role_a == "anchor" and role_b == "anchor":
        return max(rule_a["anchor_anchor_spacing_mm"], rule_b["anchor_anchor_spacing_mm"])
    return max(rule_a["anchor_pass_spacing_mm"], rule_b["anchor_pass_spacing_mm"])


def legal_center_mask(
    section_mask: np.ndarray,
    class_name: str,
    strand_count: int,
    mm_per_pixel: float = 5.0,
    role: str = "pass",
    tolerance_mm: float = 0.0,
) -> np.ndarray:
    rules = load_tendon_rules()
    if strand_count not in rules:
        raise KeyError(f"Unsupported strand count: {strand_count}")
    distance_px = cv2.distanceTransform((section_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    required_px = (boundary_requirement_mm(rules[strand_count], class_name, role) + tolerance_mm) / float(mm_per_pixel)
    return ((section_mask > 0) & (distance_px >= required_px)).astype(np.uint8)


def required_spacing_px(
    strand_count_a: int,
    role_a: str,
    strand_count_b: int,
    role_b: str,
    mm_per_pixel: float = 5.0,
) -> float:
    rules = load_tendon_rules()
    return pair_spacing_mm(rules[strand_count_a], role_a, rules[strand_count_b], role_b) / float(mm_per_pixel)
