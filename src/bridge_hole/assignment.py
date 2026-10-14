"""Role-aware global assignment for rule-grid candidates.

The first pass has no tendon schedule and uses ordinary top-n selection. Once
the longitudinal design table supplies tendon IDs and roles, this module
assigns each tendon to a unique candidate while enforcing role-dependent
centre distances.
"""

import math
from typing import Dict, List, Sequence

from .rules import pair_spacing_mm, load_tendon_rules


class AssignmentError(RuntimeError):
    pass


def assign_tendons(
    tendons: Sequence[Dict[str, object]],
    candidates: Dict[str, List[Dict[str, float]]],
    mm_per_pixel: float = 5.0,
    beam_width: int = 512,
    enable_anchor_constraints: bool = False,
) -> List[Dict[str, float]]:
    """Assign scheduled tendons to unique scored candidates.

    This deterministic beam search is intended for the moderate candidate
    sets produced after grid generation. It enforces the XML spacing rules for
    pass/pass, anchor/anchor, and anchor/pass pairs. Boundary legality is
    expected to have been applied during candidate generation.
    """
    rules = load_tendon_rules()
    ordered = sorted(
        tendons,
        key=lambda item: (0 if str(item.get("role", "pass")) == "anchor" else 1, str(item.get("tendon_id", ""))),
    )
    # (score, assignments, used candidate identities)
    beam = [(0.0, [], set())]
    for tendon in ordered:
        name = str(tendon["class"])
        strand_count = int(tendon["strand_count"])
        requested_role = str(tendon.get("role", "pass"))
        role = requested_role if enable_anchor_constraints else "pass"
        options = candidates.get(name, [])
        if not options:
            raise AssignmentError(f"No candidates for tendon class {name}")
        expanded = []
        for total_score, assigned, used in beam:
            for index, candidate in enumerate(options):
                identity = (name, index)
                if identity in used:
                    continue
                valid = True
                for previous_tendon, previous_candidate in assigned:
                    required = pair_spacing_mm(
                        rules[strand_count], role,
                        rules[int(previous_tendon["strand_count"])],
                        str(previous_tendon.get("role", "pass")) if enable_anchor_constraints else "pass",
                    ) / float(mm_per_pixel)
                    distance = math.hypot(
                        float(candidate["x"]) - float(previous_candidate["x"]),
                        float(candidate["y"]) - float(previous_candidate["y"]),
                    )
                    if distance + 1e-6 < required:
                        valid = False
                        break
                if valid:
                    expanded.append(
                        (
                            total_score + float(candidate.get("score", 0.0)),
                            assigned + [(tendon, candidate)],
                            used | {identity},
                        )
                    )
        if not expanded:
            raise AssignmentError(f"No legal assignment for tendon {tendon.get('tendon_id', '?')}")
        expanded.sort(key=lambda state: state[0], reverse=True)
        beam = expanded[: max(int(beam_width), 1)]
    best = max(beam, key=lambda state: state[0])
    output = []
    for tendon, candidate in best[1]:
        item = dict(candidate)
        item["tendon_id"] = str(tendon["tendon_id"])
        item["role"] = str(tendon.get("role", "pass")) if enable_anchor_constraints else "pass"
        item["strand_count"] = int(tendon["strand_count"])
        output.append(item)
    return output
