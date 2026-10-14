"""Build a normalized JSON intermediate for the data_train workbook."""

import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "data_train_build.json"
EXCLUDED_SAMPLES = {"0_7_-51"}  # labelled centres lie outside the section geometry


def numeric(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def integer(value):
    value = numeric(value)
    return None if value is None else int(round(value))


def half_count(value):
    value = integer(value)
    return None if value is None else int(math.ceil(value / 2.0))


def spec_set(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("��", "-").replace("，", ",")
    parts = re.findall(r"\d+", text)
    return ",".join(dict.fromkeys(parts))


def singleton_spec(value):
    values = spec_set(value)
    return int(values) if values.isdigit() else None


def onehot_specs(value):
    if pd.isna(value):
        return ""
    parts = [float(v.strip()) for v in str(value).strip("[]").split(",") if v.strip()]
    active = [str(7 + i) for i, v in enumerate(parts) if v > 0.5]
    return ",".join(active)


def load_rules(path: Path):
    # This is the third worksheet in design_parameter.xlsx. The source uses a
    # multi-row Chinese header, so the stable anchor is the OVM.M15-* model id.
    raw = pd.read_excel(path, sheet_name=2, header=None)
    rows = {}
    for _, row in raw.iterrows():
        anchor = row.iloc[1] if len(row) > 1 else None
        if pd.isna(anchor) or not str(anchor).startswith("OVM.M15-"):
            continue
        try:
            count = int(str(anchor).split("-")[-1])
        except ValueError:
            continue
        rows[count] = {
            "strand_count": count,
            "bundle_spec": f"{count}-φ15.2",
            "anchor_model": str(anchor),
            "anchor_top_bottom_edge_mm": numeric(row.iloc[5]),
            "duct_inner_diameter_mm": numeric(row.iloc[6]),
            "pass_pass_spacing_mm": numeric(row.iloc[8]),
            "pass_top_web_edge_mm": numeric(row.iloc[9]),
            "pass_bottom_edge_mm": numeric(row.iloc[10]),
            "jack_top_edge_mm": numeric(row.iloc[13]),
            "jack_bottom_edge_mm": numeric(row.iloc[14]),
            "anchor_anchor_spacing_mm": numeric(row.iloc[18]),
            "anchor_web_edge_mm": numeric(row.iloc[22]),
            "anchor_pass_spacing_mm": numeric(row.iloc[24]),
            "source_sheet": "钢束空间(铁路) 加大波纹管内径",
        }
    return rows


def add_rule_fields(target, prefix, raw_spec, rules):
    values = spec_set(raw_spec)
    target[f"{prefix}_strand_specs"] = values
    target[f"{prefix}_single_strand_count"] = singleton_spec(raw_spec)
    target[f"{prefix}_rule_lookup_status"] = "none" if not values else ("single_spec" if "," not in values else "mixed_spec_requires_tendon_level")
    rule = rules.get(singleton_spec(raw_spec))
    for key in (
        "duct_inner_diameter_mm",
        "pass_top_web_edge_mm",
        "pass_bottom_edge_mm",
        "pass_pass_spacing_mm",
        "anchor_top_bottom_edge_mm",
        "anchor_web_edge_mm",
        "anchor_anchor_spacing_mm",
        "anchor_pass_spacing_mm",
        "jack_top_edge_mm",
        "jack_bottom_edge_mm",
    ):
        target[f"{prefix}_{key}"] = None if rule is None else rule[key]


def main():
    overlap = pd.read_excel(ROOT / "overlap_summary.xlsx")
    holes = pd.read_excel(ROOT / "hole_stats_AB_hole.xlsx")
    report = pd.read_excel(ROOT / "dataset_report.xlsx")
    rules = load_rules(ROOT / "design_parameter.xlsx")

    overlap = overlap.rename(columns={"Unnamed: 4": "unused_overlap_column"})
    overlap = overlap.rename(columns={"id": "group_id", "id_num": "sample_order"})
    holes = holes.drop(columns=["id", "id_num"], errors="ignore")
    report = report.drop(columns=["id"], errors="ignore")
    domain_candidates = []
    for column in holes.columns:
        values = set(str(value).strip().upper() for value in holes[column].dropna().tolist())
        if values and values.issubset({"A", "B"}):
            domain_candidates.append(column)
    domain_column = domain_candidates[0] if domain_candidates else None
    overlap_clean = overlap[pd.to_numeric(overlap["overlap"], errors="coerce") == 0].copy()
    merged = overlap_clean.merge(holes, on="sample", how="left", validate="one_to_one")
    merged = merged.merge(report, on="sample", how="left", validate="one_to_one", suffixes=("", "_report"))
    merged = merged[~merged["sample"].astype(str).isin(EXCLUDED_SAMPLES)].copy()

    records = []
    for _, row in merged.iterrows():
        out = {
            "sample_id": str(row["sample"]),
            "group_id": integer(row["group_id"]),
            "sample_order": integer(row["sample_order"]),
            "dataset_domain": str(row[domain_column]) if domain_column and not pd.isna(row[domain_column]) else "",
            "overlap_flag": integer(row["overlap"]),
            "overlap_status": "clean_overlap_0",
            "count_scope": "half_section",
            "mm_per_pixel": 5.0,
            "file_name": str(row["file_name"]),
            "section_num": integer(row["section_num"]),
            "distance": numeric(row["distance"]),
            "side_distance_ratio": numeric(row["side_distance_ratio"]),
            "full_hole_count": integer(row["holes_count"]),
            "half_hole_count": half_count(row["holes_count"]),
            "full_top_component_count": integer(row["red_components"]),
            "full_web_component_count": integer(row["green_components"]),
            "full_bottom_component_count": integer(row["blue_components"]),
            "half_top_component_count": half_count(row["red_components"]),
            "half_web_component_count": half_count(row["green_components"]),
            "half_bottom_component_count": half_count(row["blue_components"]),
            "full_top_region_count": integer(row["red_regions"]),
            "full_web_region_count": integer(row["green_regions"]),
            "full_bottom_region_count": integer(row["blue_regions"]),
            "half_top_region_count": half_count(row["red_regions"]),
            "half_web_region_count": half_count(row["green_regions"]),
            "half_bottom_region_count": half_count(row["blue_regions"]),
            "full_top_entity_count": integer(row["red_entities"]),
            "full_web_entity_count": integer(row["green_entities"]),
            "full_bottom_entity_count": integer(row["blue_entities"]),
            "geometry_center_x_cm": numeric(row["center_x_cm"]),
            "geometry_center_y_cm": numeric(row["center_y_cm"]),
            "geometry_bbox_width_cm": numeric(row["bbox_width_cm"]),
            "geometry_bbox_height_cm": numeric(row["bbox_height_cm"]),
            "geometry_window_width_cm": numeric(row["window_width_cm"]),
            "geometry_window_height_cm": numeric(row["window_height_cm"]),
            "clipped_x": integer(row["clipped_x"]),
            "clipped_y": integer(row["clipped_y"]),
            "label_quality_status": str(row["status"]),
            "full_red_count_mismatch": numeric(row["red_count_mismatch"]),
            "full_green_count_mismatch": numeric(row["green_count_mismatch"]),
            "full_blue_count_mismatch": numeric(row["blue_count_mismatch"]),
            "rule_mode": "pass_only",
        }
        for prefix, source_col, onehot_col in (
            ("top", "top", "top_onehot"),
            ("web", "web", "web_onehot"),
            ("bottom", "bottom", "bottom_onehot"),
        ):
            add_rule_fields(out, prefix, row[source_col], rules)
            out[f"{prefix}_onehot_specs"] = onehot_specs(row[onehot_col])
        records.append(out)

    payload = {
        "metadata": {
            "row_count": len(records),
            "filter": "overlap == 0; exclude invalid geometry samples",
            "excluded_samples": sorted(EXCLUDED_SAMPLES),
            "count_scope": "half_section",
            "half_conversion": "ceil(full_count / 2), preserving a centreline item in the half-section",
            "mm_per_pixel": 5.0,
            "sources": ["overlap_summary.xlsx", "hole_stats_AB_hole.xlsx", "design_parameter.xlsx", "dataset_report.xlsx"],
        },
        "records": records,
        "rules": list(rules.values()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metadata"], ensure_ascii=False))


if __name__ == "__main__":
    main()
