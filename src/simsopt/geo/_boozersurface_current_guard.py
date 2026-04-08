from __future__ import annotations


def _coils_from_attrs(biotsavart, *, coil_attrs: tuple[str, ...]):
    for attr_name in coil_attrs:
        coils = getattr(biotsavart, attr_name, None)
        if coils is not None:
            return coils
    return None


def coil_currents_are_fixed(biotsavart, *, coil_attrs: tuple[str, ...]) -> bool:
    coils = _coils_from_attrs(biotsavart, coil_attrs=coil_attrs)
    if coils is None:
        return True
    return all(coil.current.dofs.all_fixed() for coil in coils)


def require_fixed_currents_for_none_G(
    biotsavart,
    *,
    component: str,
    coil_attrs: tuple[str, ...],
) -> None:
    if coil_currents_are_fixed(biotsavart, coil_attrs=coil_attrs):
        return
    raise ValueError(
        f"{component} requires fixed coil currents when G=None to avoid "
        "incorrect coil gradients."
    )
