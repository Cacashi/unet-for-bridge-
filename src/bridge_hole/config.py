from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base_name = config.pop("base", None)
    if base_name:
        base = load_config(str(config_path.parent / base_name))
        config = _merge(base, config)
    config["_config_path"] = str(config_path)
    return config

