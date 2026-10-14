#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:45:30 2026

@author: elincrayon
"""

from pathlib import Path
import xml.etree.ElementTree as ET

DATABASE_PATH = Path("铁路钢束空间参数库.xml")


def get_tendon_parameters(bundle_spec: str) -> dict:
    """根据钢束规格读取空间构造参数。"""

    tree = ET.parse(DATABASE_PATH)
    root = tree.getroot()

    xpath = (
        ".//tendon_specification"
        f"[@bundle_spec='{bundle_spec}']"
    )
    node = root.find(xpath)

    if node is None:
        raise ValueError(f"未找到钢束规格：{bundle_spec}")

    duct = node.find("duct")
    spacing = node.find("spacing")
    jack = node.find("tensioning_jack")
    edge = node.find("anchor_edge")

    return {
        "bundle_spec": node.get("bundle_spec"),
        "anchor_model": node.get("anchor_model"),
        "strand_count": int(node.get("strand_count")),
        "metal_duct_inner_diameter": float(
            duct.get("inner_diameter_mm")
        ),
        "duct_top_web_edge_distance": float(
            duct.get("top_web_edge_mm")
        ),
        "duct_bottom_edge_distance": float(
            duct.get("bottom_slab_edge_mm")
        ),
        "pass_to_pass_center_spacing": float(
            spacing.get("pass_to_pass_center_mm")
        ),
        "anchor_to_anchor_center_spacing": float(
            spacing.get("anchor_to_anchor_center_mm")
        ),
        "anchor_to_pass_center_spacing": float(
            spacing.get("anchor_to_pass_center_mm")
        ),
        "jack_top_edge_distance": float(
            jack.get("top_slab_control_edge_mm")
        ),
        "jack_bottom_edge_distance": float(
            jack.get("bottom_slab_control_edge_mm")
        ),
        "anchor_top_bottom_edge_distance": float(
            edge.get("top_bottom_slab_mm")
        ),
        "anchor_web_edge_distance": float(
            edge.get("web_mm")
        ),
    }


params = get_tendon_parameters("12-φ15.2")
print(params)
