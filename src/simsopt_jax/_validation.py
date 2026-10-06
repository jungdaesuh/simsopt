"""Shared scalar contracts for host preparation boundaries."""

from numbers import Integral


def is_integral(value: object) -> bool:
    """Accept Python and NumPy integers, excluding booleans."""
    return isinstance(value, Integral) and not isinstance(value, bool)
