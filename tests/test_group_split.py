from bridge_hole.data.splits import make_group_splits


def test_group_split_has_no_overlap():
    records = [{"group_id": str(index // 2)} for index in range(20)]
    splits = make_group_splits(records, [0.7, 0.15, 0.15], 7)
    train, val, test = map(set, (splits["train"], splits["val"], splits["test"]))
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_content_duplicates_stay_in_one_split(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"same-input")
    second.write_bytes(b"same-input")
    records = [
        {"group_id": "a", "input_path": str(first)},
        {"group_id": "b", "input_path": str(second)},
        {"group_id": "c", "input_path": str(tmp_path / "other.png")},
    ]
    (tmp_path / "other.png").write_bytes(b"different-input")
    splits = make_group_splits(records, [0.67, 0.165, 0.165], 3)
    locations = {
        split: group
        for split, groups in splits.items()
        for group in groups
    }
    assert locations["a"] == locations["b"]
