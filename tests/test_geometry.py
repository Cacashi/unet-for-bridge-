import numpy as np

from bridge_hole.geometry import build_section_mask, holes_have_valid_centres
from bridge_hole.postprocess import constrained_select


def test_scanline_mask_fills_material_and_excludes_cavity():
    image = np.full((12, 20), 255, dtype=np.uint8)
    # Half-section material spans x=0..14. The second pair of boundaries
    # creates a cavity between x=5 and x=9 on the middle rows.
    image[2, :15] = 0
    image[9, :15] = 0
    image[2:10, 14] = 0
    image[4:8, 5] = 0
    image[4:8, 9] = 0

    mask = build_section_mask(image, close_kernel=1)

    assert mask[3, 2] == 1
    assert mask[5, 7] == 0
    assert mask[5, 12] == 1
    assert mask[5, 18] == 0


def test_constrained_select_rejects_circle_outside_section():
    section_mask = np.zeros((30, 30), dtype=np.uint8)
    section_mask[5:25, 5:25] = 1
    candidates = {
        "top": [
            {"x": 2.0, "y": 2.0, "r": 1.0, "class": "top", "score": 0.9},
            {"x": 15.0, "y": 15.0, "r": 2.0, "class": "top", "score": 0.8},
        ],
        "web": [],
        "bottom": [],
    }

    selected = constrained_select(
        candidates,
        [1, 0, 0],
        (30, 30),
        section_mask=section_mask,
    )

    assert len(selected) == 1
    assert selected[0]["x"] == 15.0


def test_invalid_label_centre_is_detected():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1

    assert holes_have_valid_centres(mask, [{"x": 4.0, "y": 4.0}])
    assert not holes_have_valid_centres(mask, [{"x": 9.0, "y": 9.0}])
