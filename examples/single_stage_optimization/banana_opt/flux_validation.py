from __future__ import annotations


def validate_normalized_toroidal_flux(
    value: float,
    *,
    field_name: str = "toroidal_flux",
) -> float:
    flux = float(value)
    if not 0.0 <= flux <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1 inclusive, got {value!r}."
        )
    return flux
