import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bridge_hole.config import load_config
from bridge_hole.data.dataset import HoleLayoutDataset
from bridge_hole.data.position import position_group
from bridge_hole.engine import build_model
from bridge_hole.geometry import build_section_mask
from bridge_hole.postprocess import (
    ConstraintDecodeError,
    constrained_select,
    decode_candidates,
    render_class_masks,
    render_hole_overlay,
    render_holes,
)
from bridge_hole.grid import grid_candidates
from bridge_hole.rules import CLASS_NAMES, load_tendon_rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Single-model checkpoint (legacy mode)")
    parser.add_argument("--checkpoint-zero", help="Override the zero-position checkpoint")
    parser.add_argument("--checkpoint-nonzero", help="Override the nonzero-position checkpoint")
    parser.add_argument("--sample", help="Sample id; omit when using --all")
    parser.add_argument(
        "--all", action="store_true", help="Predict the selected split (using configured test quotas, if any)"
    )
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    if args.all == (args.sample is not None):
        parser.error("provide exactly one of --sample SAMPLE or --all")
    config = load_config(args.config)
    project_root = Path(config["_config_path"]).resolve().parent.parent
    root = Path(config["data"]["root"]).resolve()
    config["data"]["manifest"] = str(root / config["data"]["manifest"])
    config["data"]["splits"] = str(root / config["data"]["splits"])
    config["data"]["feature_stats"] = str(root / config["data"]["feature_stats"])
    dataset = HoleLayoutDataset(config, args.split, domain=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.all:
        samples = [record["sample"] for record in dataset.records]
        quotas = config.get("prediction", {}).get("test_sample_counts") if args.split == "test" else None
        if quotas is not None:
            seed = str(config["prediction"].get("test_selection_seed", config["seed"]))
            selected = []
            for group in ("zero", "nonzero"):
                candidates = sorted(
                    sample for sample in samples if position_group(sample) == group
                )
                count = int(quotas[group])
                if count < 0 or len(candidates) < count:
                    raise ValueError(
                        f"Test group '{group}' requires {count} samples, "
                        f"but only {len(candidates)} are available in this manifest/split."
                    )
                ranked = sorted(
                    candidates,
                    key=lambda sample: hashlib.sha256(
                        f"{seed}:{sample}".encode("utf-8")
                    ).hexdigest(),
                )
                selected.extend(ranked[:count])
                print(f"Test group '{group}': {len(candidates)} eligible, "
                      f"selected {count}: {ranked[:count]}")
            samples = selected
    else:
        samples = [args.sample]
    if not samples:
        raise RuntimeError(f"No samples found in split '{args.split}'.")
    print(f"Predicting {len(samples)} sample(s) from split '{args.split}'.")
    routes = config.get("prediction", {}).get("position_models")
    if routes:
        if args.checkpoint:
            parser.error("position routing uses --checkpoint-zero / --checkpoint-nonzero")
        # Load one model at a time, avoiding two simultaneous GPU allocations.
        for group in ("zero", "nonzero"):
            group_samples = [sample for sample in samples if position_group(sample) == group]
            if not group_samples:
                continue
            route = routes[group]
            model_config = load_config(str(project_root / route["config"]))
            override = args.checkpoint_zero if group == "zero" else args.checkpoint_nonzero
            checkpoint_value = Path(override or route["checkpoint"])
            checkpoint = str(
                checkpoint_value
                if checkpoint_value.is_absolute()
                else project_root / checkpoint_value
            )
            model = build_model(model_config).to(device)
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state["model"])
            del state
            model.eval()
            print(f"Position group '{group}': {len(group_samples)} sample(s), {checkpoint}")
            for sample in group_samples:
                predict_sample(
                    config, dataset, sample, model, device, root, args.split,
                    output_group=group, checkpoint=checkpoint,
                )
            del model
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for single-model prediction")
        if args.checkpoint_zero or args.checkpoint_nonzero:
            parser.error("position checkpoint overrides require a position-routing config")
        model = build_model(config).to(device)
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path = (
            checkpoint_path
            if checkpoint_path.is_absolute()
            else project_root / checkpoint_path
        )
        state = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(state["model"])
        model.eval()
        for sample in samples:
            predict_sample(config, dataset, sample, model, device, root, args.split)


def predict_sample(
    config, dataset, sample, model, device, root, split,
    output_group=None, checkpoint=None,
) -> None:
    matches = [i for i, record in enumerate(dataset.records) if record["sample"] == sample]
    if not matches:
        raise ValueError(
            f"Sample '{sample}' was not found in split '{split}'."
        )
    batch = dataset[matches[0]]
    with torch.no_grad():
        predictions = model(
            batch["image"].unsqueeze(0).to(device),
            batch["tabular"].unsqueeze(0).to(device),
        )
    # Keep label counts as the requested decode counts, per the selected
    # evaluation protocol. Model count-head outputs are reported separately.
    raw_counts = predictions["count"][0].detach().cpu().tolist()
    counts = [
        int(v)
        for v in batch["targets"]["center_mask"].sum(dim=(1, 2)).tolist()
    ]
    strand_counts = [int(value) or None for value in batch["strand_counts"].tolist()]
    mm_per_pixel = float(
        batch.get("mm_per_pixel", config["data"].get("mm_per_pixel", 5.0))
    )
    original_size = (
        int(batch["meta"]["original_height"]),
        int(batch["meta"]["original_width"]),
    )
    original_image = np.asarray(Image.open(batch["input_path"]).convert("L"), dtype=np.uint8)
    section_mask = build_section_mask(original_image)
    if str(config["decode"].get("mode", "heatmap")) == "grid":
        candidates = grid_candidates(
            predictions,
            section_mask,
            strand_counts,
            batch["meta"],
            mm_per_pixel,
            config["decode"].get("class_bands"),
        )
    else:
        candidates = decode_candidates(
            predictions,
            counts,
            int(config["model"]["output_stride"]),
            int(config["decode"]["candidate_multiplier"]),
            float(config["decode"]["min_score"]),
            batch["meta"],
            bool(config["decode"].get("use_region_gate", True)),
        )
    # Use the engineering XML duct diameter for every decoder: radius in px
    # is inner_diameter_mm / 2 / mm_per_pixel.
    rules = load_tendon_rules()
    for class_index, class_name in enumerate(CLASS_NAMES):
        if not counts[class_index]:
            continue
        strand_count = strand_counts[class_index]
        if strand_count not in rules:
            raise ValueError(
                f"Cannot apply XML radius for {class_name}: missing or unsupported "
                f"strand count {strand_count}."
            )
        xml_radius = rules[strand_count]["duct_diameter_mm"] / (2.0 * mm_per_pixel)
        for candidate in candidates[class_name]:
            candidate["r"] = xml_radius
    bands = config["decode"]["class_bands"] if config["decode"]["enforce_class_bands"] else None
    decode_error = None
    try:
        holes = constrained_select(
            candidates,
            counts,
            original_size,
            float(config["decode"]["min_clearance"]),
            bands,
            section_mask,
            float(config["decode"].get("boundary_tolerance", 2.0)),
        )
    except ConstraintDecodeError as error:
        # Preserve the legal points found before the first unsatisfied class.
        # This is intentionally marked as partial in the output metadata.
        decode_error = str(error)
        holes = error.partial
        print("WARNING: partial decode; " + decode_error)
    # The dataset and decoder operate in half-section coordinates.  Do not
    # mirror here: a full-section canvas requires a separate, explicitly
    # prepared geometry and must not be inferred from a half-section sample.
    output_dir = root / "outputs" / "predictions"
    output_subdir = config.get("prediction", {}).get("output_subdir")
    if output_subdir:
        output_dir = output_dir / output_subdir
    if output_group is not None:
        output_dir = output_dir / output_group
    output_dir = output_dir / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    true_combine = root / config["data"]["output_dir"] / (sample + "_combine.png")
    if not true_combine.is_file():
        raise FileNotFoundError(
            "Ground-truth combine image is required for prediction comparison: "
            + str(true_combine)
        )
    # Keep the conventional *_combine.png name for the ground-truth image;
    # downstream image analysis tools treat that suffix as the reference.
    true_combine_output = output_dir / (sample + "_combine.png")
    shutil.copy2(true_combine, true_combine_output)
    with (output_dir / (sample + "_holes.json")).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample": sample,
                "position_group": position_group(sample),
                "checkpoint": checkpoint,
                "scope": "half",
                "rule_mode": config["data"].get("rule_mode", "pass_only"),
                "anchor_constraints_enabled": bool(
                    config["data"].get("enable_anchor_constraints", False)
                ),
                "target_half_counts": dict(zip(CLASS_NAMES, counts)),
                "predicted_half_counts": {
                    name: float(value) for name, value in zip(CLASS_NAMES, raw_counts)
                },
                "selection_half_counts": dict(zip(CLASS_NAMES, counts)),
                "selection_color_counts": {
                    "red": counts[0],
                    "green": counts[1],
                    "blue": counts[2],
                },
                "decode_status": "partial" if decode_error else "complete",
                "decode_error": decode_error,
                "decoded_counts": {
                    name: sum(1 for hole in holes if hole["class"] == name)
                    for name in ("top", "web", "bottom")
                },
                "holes": holes,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    np.savez_compressed(
        output_dir / (sample + "_raw_heads.npz"),
        heatmap=torch.sigmoid(predictions["heatmap_logits"])[0].cpu().numpy(),
        offset=predictions["offset"][0].cpu().numpy(),
        log_radius=predictions["log_radius"][0].cpu().numpy(),
        density=predictions["density"][0].cpu().numpy(),
        count=predictions["count"][0].cpu().numpy(),
        region_probability=torch.softmax(predictions["region_logits"], dim=1)[0].cpu().numpy(),
        section_mask=section_mask,
    )
    render_class_masks(original_size, holes, str(output_dir), sample)
    render_holes(
        batch["input_path"], holes,
        str(output_dir / (sample + "_predicted_combine.png")),
    )
    render_hole_overlay(
        str(true_combine),
        holes,
        str(output_dir / (sample + "_overlap.png")),
        prediction_alpha=int(config.get("prediction", {}).get("overlay_alpha", 90)),
    )
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
