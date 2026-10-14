"""Classify section positions using the final numeric sample-ID field."""

from decimal import Decimal, InvalidOperation


def position_group(sample: str) -> str:
    try:
        prefix, suffix = str(sample).rsplit("_", 1)
        distance = Decimal(suffix)
    except (ValueError, InvalidOperation) as error:
        raise ValueError("Invalid section sample ID: %r" % sample) from error
    if not prefix or not distance.is_finite():
        raise ValueError("Invalid section sample ID: %r" % sample)
    return "zero" if distance == 0 else "nonzero"
