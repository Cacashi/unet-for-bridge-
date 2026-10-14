"""Build the cleaned A-domain finetuning table with the same enrichment as data_train.

Surviving ``*_combine.png`` files under ``half-section/output/a_finetune`` are
the manual-review allowlist. The overlap workbook is deliberately not used.
"""

import json
from pathlib import Path

import pandas as pd

from build_data_train_intermediate import (
    add_rule_fields,
    half_count,
    integer,
    load_rules,
    numeric,
    onehot_specs,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "data_finetune_build.json"


def main():
    holes = pd.read_excel(ROOT / "hole_stats_AB_hole.xlsx")
    report = pd.read_excel(ROOT / "dataset_report.xlsx").drop(columns=["id"], errors="ignore")
    rules = load_rules(ROOT / "design_parameter.xlsx")

    domain_candidates = []
    for column in holes.columns:
        values = {
            str(value).strip().upper()
            for value in holes[column].dropna().tolist()
        }
        if values and values.issubset({"A", "B"}):
            domain_candidates.append(column)
    if len(domain_candidates) != 1:
        raise ValueError(
            "Expected one A/B domain column, found: %s" % domain_candidates
        )
    domain_column = domain_candidates[0]
    holes = holes[
        holes[domain_column].astype(str).str.strip().str.upper() == "A"
    ].copy()
    combine_dir = ROOT / "half-section" / "output" / "a_finetune"
    kept_samples = {
        path.name[: -len("_combine.png")]
        for path in combine_dir.glob("*_combine.png")
    }
    excluded_samples = sorted(
        set(holes["sample"].astype(str)) - kept_samples
    )
    holes = holes[holes["sample"].astype(str).isin(kept_samples)].copy()
    if holes["sample"].duplicated().any():
        raise ValueError("A-domain source contains duplicate sample IDs")

    merged = holes.merge(
        report,
        on="sample",
        how="left",
        validate="one_to_one",
        suffixes=("", "_report"),
        indicator=True,
    )
    unmatched_reports = int((merged["_merge"] != "both").sum())
    merged = merged.drop(columns=["_merge"])

    records = []
    for _, row in merged.iterrows():
        out = {
            "sample_id": str(row["sample"]),
            "group_id": integer(row["id"]),
            "sample_order": integer(row["id_num"]),
            "dataset_domain": "A",
            "overlap_flag": None,
            "overlap_status": "not_checked",
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
            "filter": "dataset_domain == A; surviving a_finetune combine files; overlap not considered",
            "excluded_after_manual_review": excluded_samples,
            "excluded_after_manual_review_count": len(excluded_samples),
            "overlap_report_matches": None,
            "unmatched_dataset_report_rows": unmatched_reports,
            "count_scope": "half_section",
            "half_conversion": "ceil(full_count / 2), preserving a centreline item in the half-section",
            "mm_per_pixel": 5.0,
            "sources": [
                "hole_stats_AB_hole.xlsx",
                "dataset_report.xlsx",
                "design_parameter.xlsx",
            ],
        },
        "records": records,
        "rules": list(rules.values()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metadata"], ensure_ascii=False))


if __name__ == "__main__":
    main()
