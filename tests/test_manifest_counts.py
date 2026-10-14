from bridge_hole.data.manifest import count_holes


def test_half_counts_are_derived_from_label_holes():
    holes = [
        {"class": "top"},
        {"class": "top"},
        {"class": "web"},
    ]
    counts = count_holes(holes)

    assert counts == {"top": 2, "web": 1, "bottom": 0, "total": 3}
