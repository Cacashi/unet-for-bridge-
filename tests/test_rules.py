from bridge_hole.rules import load_tendon_rules, pair_spacing_mm, strand_count_from_onehot


def test_onehot_maps_to_strand_count():
    values = [0.0] * 21
    values[5] = 1.0
    assert strand_count_from_onehot(values) == 12


def test_m15_12_through_rules():
    rule = load_tendon_rules()[12]
    assert rule["pass_pass_spacing_mm"] == 174.6
    assert rule["pass_top_web_edge_mm"] == 145.5
    assert rule["pass_bottom_edge_mm"] == 108.5
    assert pair_spacing_mm(rule, "pass", rule, "pass") == 174.6
