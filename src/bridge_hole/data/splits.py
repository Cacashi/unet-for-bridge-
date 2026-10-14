import json
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


class _UnionFind:
    def __init__(self, values: List[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _file_digest(path: Any) -> Optional[str]:
    if not path:
        return None
    try:
        digest = hashlib.md5(usedforsecurity=False)
        with Path(str(path)).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, TypeError):
        return None


def _split_units(records: List[Dict[str, Any]], deduplicate_content: bool) -> List[List[str]]:
    groups = sorted({str(record["group_id"]) for record in records})
    uf = _UnionFind(groups)
    if deduplicate_content:
        signatures: Dict[str, str] = {}
        for record in records:
            group = str(record["group_id"])
            # Use the rendered input as the primary identity. Individual
            # class masks are intentionally not used here: empty class masks
            # are identical across many unrelated samples and would merge
            # the entire dataset into one split.
            digest = _file_digest(record.get("input_path"))
            if digest is not None:
                previous = signatures.get(digest)
                if previous is None:
                    signatures[digest] = group
                else:
                    uf.union(group, previous)
    units: Dict[str, List[str]] = {}
    for group in groups:
        units.setdefault(uf.find(group), []).append(group)
    return sorted((sorted(unit) for unit in units.values()), key=lambda unit: unit[0])


def make_group_splits(
    records: List[Dict[str, Any]],
    ratios: List[float],
    seed: int,
    deduplicate_content: bool = True,
) -> Dict[str, List[str]]:
    units = _split_units(records, deduplicate_content)
    rng = random.Random(seed)
    rng.shuffle(units)
    train_end = int(round(len(units) * ratios[0]))
    val_end = train_end + int(round(len(units) * ratios[1]))
    train_units, val_units, test_units = units[:train_end], units[train_end:val_end], units[val_end:]
    return {
        "train": sorted(group for unit in train_units for group in unit),
        "val": sorted(group for unit in val_units for group in unit),
        "test": sorted(group for unit in test_units for group in unit),
    }


def save_splits(splits: Dict[str, List[str]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, ensure_ascii=False, indent=2)


def load_splits(path: str) -> Dict[str, List[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
