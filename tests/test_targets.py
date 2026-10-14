from bridge_hole.data.targets import encode_targets


def test_target_shapes():
    target = encode_targets(
        [{"x": 20.5, "y": 40.25, "r": 4.0, "class": "top"}],
        image_height=128,
        image_width=64,
        stride=4,
    )
    assert tuple(target["heatmap"].shape) == (3, 32, 16)
    assert tuple(target["offset"].shape) == (6, 32, 16)
    assert tuple(target["log_radius"].shape) == (3, 32, 16)
