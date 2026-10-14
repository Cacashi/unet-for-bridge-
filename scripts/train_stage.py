import argparse
from pathlib import Path

from bridge_hole.config import load_config
from bridge_hole.engine import train_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    project_root = Path(config["_config_path"]).resolve().parent.parent
    data_root = Path(config["data"]["root"]).resolve()
    config["data"]["manifest"] = str(data_root / config["data"]["manifest"])
    config["data"]["splits"] = str(data_root / config["data"]["splits"])
    config["data"]["feature_stats"] = str(data_root / config["data"]["feature_stats"])
    checkpoint = config["train"].get("checkpoint")
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        config["train"]["checkpoint"] = str(
            checkpoint_path if checkpoint_path.is_absolute() else project_root / checkpoint_path
        )
    output_dir = Path(config["train"]["output_dir"])
    config["train"]["output_dir"] = str(
        output_dir if output_dir.is_absolute() else project_root / output_dir
    )
    train_stage(config)


if __name__ == "__main__":
    main()
