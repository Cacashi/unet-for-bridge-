import argparse
from pathlib import Path

from bridge_hole.config import load_config
from bridge_hole.data.manifest import build_manifest, save_feature_stats, save_manifest
from bridge_hole.data.splits import load_splits, make_group_splits, save_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["data"]["root"]).resolve()
    config["data"]["manifest"] = str(root / config["data"]["manifest"])
    config["data"]["splits"] = str(root / config["data"]["splits"])
    config["data"]["feature_stats"] = str(root / config["data"]["feature_stats"])
    records = build_manifest(config)
    if not records:
        output_dir = root / config["data"]["output_dir"]
        if config["data"].get("require_combine", False):
            detail = "retained combine files in %s: %d" % (
                output_dir,
                sum(1 for _ in output_dir.glob("*_combine.png")),
            )
        else:
            detail = "input_dir=%s, output_dir=%s, excluded_samples=%s" % (
                root / config["data"]["input_dir"],
                output_dir,
                config["data"].get("excluded_samples", []),
            )
        raise ValueError(
            "Manifest has no samples. %s. Check workbook sample IDs and the "
            "configured image directories. Existing manifest was not overwritten."
            % detail
        )
    if config["data"].get("reuse_splits", False):
        splits = load_splits(config["data"]["splits"])
        known_groups = {str(group) for groups in splits.values() for group in groups}
        missing = {str(record["group_id"]).strip() for record in records} - known_groups
        if missing:
            raise ValueError("Finetuning groups absent from pretrained splits: %s" % sorted(missing))
    else:
        splits = make_group_splits(
            records,
            config["data"]["split_ratios"],
            int(config["seed"]),
            deduplicate_content=bool(config["data"].get("deduplicate_content", True)),
        )
        save_splits(splits, config["data"]["splits"])
    if config["data"].get("reuse_feature_stats", False):
        if not Path(config["data"]["feature_stats"]).is_file():
            raise FileNotFoundError(config["data"]["feature_stats"])
    else:
        save_feature_stats(
            records,
            splits["train"],
            config["features"]["continuous"],
            config["data"]["feature_stats"],
        )
    save_manifest(records, config["data"]["manifest"])
    print("manifest:", config["data"]["manifest"])
    print("splits:", config["data"]["splits"])
    print("feature stats:", config["data"]["feature_stats"])


if __name__ == "__main__":
    main()
