from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping

import numpy as np

from banana_opt.hardware_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
    validate_tf_current_limit,
)
from banana_opt.hardware_constraint_schema import (
    hardware_constraint_penalty_box_bound_names,
    resolve_penalty_box_bound_threshold,
)


MU0 = 4.0e-7 * np.pi
MU0_OVER_2PI = 2.0e-7

CurrentInputSource = Literal[
    "physical_A",
    "raw_boozer_I",
    "default_zero",
    "artifact_default_A",
]
FiniteCurrentModeSource = Literal["artifact_metadata", "legacy_assumed_default"]
BoozerCurrentConvention = Literal["mu0_over_2pi", "mu0"]
FiniteCurrentMode = Literal["boozer_surrogate", "wataru_proxy_field"]
EffectiveCurrentMode = Literal["vacuum", "boozer_surrogate", "wataru_proxy_field"]
CURRENT_MODE_ZERO_TOL = 1e-12
DEFAULT_FINITE_CURRENT_MODE: FiniteCurrentMode = "wataru_proxy_field"
FINITE_CURRENT_MODE_SOURCE_ARTIFACT_METADATA: FiniteCurrentModeSource = (
    "artifact_metadata"
)
FINITE_CURRENT_MODE_SOURCE_LEGACY_ASSUMED_DEFAULT: FiniteCurrentModeSource = (
    "legacy_assumed_default"
)

__all__ = [
    "BANANA_CURRENT_HARD_LIMIT_A",
    "BoozerCurrentConvention",
    "CURRENT_MODE_ZERO_TOL",
    "CurrentInputSource",
    "DEFAULT_FINITE_CURRENT_MODE",
    "EffectiveCurrentMode",
    "FiniteCurrentMode",
    "FiniteCurrentModeSource",
    "FINITE_CURRENT_MODE_SOURCE_ARTIFACT_METADATA",
    "FINITE_CURRENT_MODE_SOURCE_LEGACY_ASSUMED_DEFAULT",
    "MU0",
    "MU0_OVER_2PI",
    "PenaltyBoxBoundHandler",
    "PlasmaCurrentSettings",
    "apply_banana_current_upper_bound",
    "apply_penalty_traversal_forbidden_box_bounds",
    "banana_current_exceeds_limit",
    "boozer_I_to_physical_current_A",
    "infer_uniform_coil_current_A",
    "physical_current_to_boozer_I",
    "resolve_boozer_current_convention",
    "resolve_finite_current_mode",
    "resolve_effective_current_mode",
    "resolve_penalty_traversal_forbidden_box_bounds",
    "resolve_loaded_tf_current_A",
    "resolve_plasma_current_settings",
    "resolve_plasma_current_settings_for_num_surfaces",
    "resolve_single_surface_plasma_current_settings",
    "unwrap_current_optimizable",
]


@dataclass(frozen=True)
class PlasmaCurrentSettings:
    boozer_I: float
    plasma_current_A: float
    input_source: CurrentInputSource
    boozer_current_convention: BoozerCurrentConvention
    mode: FiniteCurrentMode
    effective_mode: EffectiveCurrentMode


@dataclass(frozen=True)
class PenaltyBoxBoundHandler:
    apply_bound: Callable[[object, float], None]
    exceeds_limit: Callable[[float, float], bool]


_BOOZER_CURRENT_SCALE_BY_CONVENTION: Mapping[BoozerCurrentConvention, float] = {
    "mu0_over_2pi": MU0_OVER_2PI,
    "mu0": MU0,
}

_BOOZER_CURRENT_CONVENTION_BY_MODE: Mapping[
    FiniteCurrentMode,
    BoozerCurrentConvention,
] = {
    # SIMSOPT's BoozerSurface residual is written in normalized angles, so the
    # code-facing current function carries the 2π from the angle change of
    # variables. Physical enclosed current in amperes therefore maps to μ0*I_A
    # at the API boundary, not μ0/(2π)*I_A.
    "boozer_surrogate": "mu0",
    # Wataru confirmed his workflow intentionally uses BoozerSurface(..., I=μ0*I_A)
    # with no extra 2π factor. This matches the normalized-angle SIMSOPT API.
    "wataru_proxy_field": "mu0",
}


def _validated_finite_current_mode(mode: str) -> FiniteCurrentMode:
    if mode == "boozer_surrogate":
        return "boozer_surrogate"
    if mode == "wataru_proxy_field":
        return "wataru_proxy_field"
    raise ValueError(f"Unsupported finite-current mode {mode!r}.")


def resolve_boozer_current_convention(
    finite_current_mode: FiniteCurrentMode,
) -> BoozerCurrentConvention:
    """Return the BoozerSurface I normalization owned by the selected workflow."""
    return _BOOZER_CURRENT_CONVENTION_BY_MODE[finite_current_mode]


def physical_current_to_boozer_I(
    plasma_current_A: float,
    *,
    convention: BoozerCurrentConvention = "mu0",
) -> float:
    return _BOOZER_CURRENT_SCALE_BY_CONVENTION[convention] * float(plasma_current_A)


def boozer_I_to_physical_current_A(
    boozer_I: float,
    *,
    convention: BoozerCurrentConvention = "mu0",
) -> float:
    return float(boozer_I) / _BOOZER_CURRENT_SCALE_BY_CONVENTION[convention]


def resolve_finite_current_mode(
    requested_mode: FiniteCurrentMode | None,
    *,
    artifact_mode: str | None = None,
    artifact_mode_source: FiniteCurrentModeSource | None = None,
) -> FiniteCurrentMode:
    if artifact_mode in {None, ""}:
        if requested_mode is None:
            return DEFAULT_FINITE_CURRENT_MODE
        return _validated_finite_current_mode(requested_mode)
    normalized_artifact_mode = _validated_finite_current_mode(str(artifact_mode))
    if requested_mode is None:
        return normalized_artifact_mode
    if requested_mode != normalized_artifact_mode:
        if artifact_mode_source == FINITE_CURRENT_MODE_SOURCE_LEGACY_ASSUMED_DEFAULT:
            raise ValueError(
                "Requested finite-current mode "
                f"{requested_mode!r} does not match the donor artifact mode "
                f"{normalized_artifact_mode!r}. The donor artifact recorded no "
                "finite-current mode, so that value was assumed as the legacy "
                "default during upgrade."
            )
        raise ValueError(
            "Requested finite-current mode "
            f"{requested_mode!r} does not match the donor artifact mode "
            f"{normalized_artifact_mode!r}."
        )
    return requested_mode


def resolve_effective_current_mode(
    boozer_I: float,
    *,
    finite_current_mode: FiniteCurrentMode = DEFAULT_FINITE_CURRENT_MODE,
) -> EffectiveCurrentMode:
    if abs(float(boozer_I)) <= CURRENT_MODE_ZERO_TOL:
        return "vacuum"
    return finite_current_mode


def _build_plasma_current_settings(
    *,
    boozer_I: float,
    plasma_current_A: float,
    input_source: CurrentInputSource,
    boozer_current_convention: BoozerCurrentConvention,
    finite_current_mode: FiniteCurrentMode,
) -> PlasmaCurrentSettings:
    return PlasmaCurrentSettings(
        boozer_I=float(boozer_I),
        plasma_current_A=float(plasma_current_A),
        input_source=input_source,
        boozer_current_convention=boozer_current_convention,
        mode=finite_current_mode,
        effective_mode=resolve_effective_current_mode(
            boozer_I,
            finite_current_mode=finite_current_mode,
        ),
    )


def unwrap_current_optimizable(current):
    scale = 1.0
    current_optimizable = current
    while hasattr(current_optimizable, "current_to_scale") and hasattr(
        current_optimizable,
        "scale",
    ):
        scale *= float(current_optimizable.scale)
        current_optimizable = current_optimizable.current_to_scale
    if not hasattr(current_optimizable, "local_lower_bounds") or not hasattr(
        current_optimizable,
        "local_upper_bounds",
    ):
        raise TypeError("Current does not expose local bounds.")
    return current_optimizable, scale


def apply_banana_current_upper_bound(current, banana_current_max_A):
    current_optimizable, scale = unwrap_current_optimizable(current)
    if scale == 0.0:
        raise ValueError("Banana current scale must be non-zero to apply a bound.")
    lower_bounds = np.asarray(current_optimizable.local_lower_bounds, dtype=float).copy()
    upper_bounds = np.asarray(current_optimizable.local_upper_bounds, dtype=float).copy()
    scaled_magnitude_bound = float(banana_current_max_A) / abs(scale)
    lower_bounds[0] = max(lower_bounds[0], -scaled_magnitude_bound)
    upper_bounds[0] = min(upper_bounds[0], scaled_magnitude_bound)
    current_optimizable.local_lower_bounds = lower_bounds
    current_optimizable.local_upper_bounds = upper_bounds


def banana_current_exceeds_limit(current_A: float, banana_current_max_A: float) -> bool:
    return abs(float(current_A)) > float(banana_current_max_A)


_PENALTY_BOX_BOUND_HANDLERS: Mapping[str, PenaltyBoxBoundHandler] = {
    "banana_current": PenaltyBoxBoundHandler(
        apply_bound=apply_banana_current_upper_bound,
        exceeds_limit=banana_current_exceeds_limit,
    ),
}


def _penalty_box_bound_handler(name: str) -> PenaltyBoxBoundHandler:
    try:
        return _PENALTY_BOX_BOUND_HANDLERS[name]
    except KeyError as exc:
        raise KeyError(
            f"No penalty box-bound handler registered for hardware constraint {name!r}."
        ) from exc


def resolve_penalty_traversal_forbidden_box_bounds(
    requested_thresholds: Mapping[str, float | None],
) -> dict[str, float]:
    # Only penalty-search box bounds need runtime handlers here. ALM and
    # artifact enforcement consume the schema through separate paths.
    return {
        name: resolve_penalty_box_bound_threshold(
            name,
            requested_threshold=requested_thresholds.get(name),
        )
        for name in hardware_constraint_penalty_box_bound_names(
            traversal_policy="forbidden",
        )
    }


def apply_penalty_traversal_forbidden_box_bounds(
    *,
    bound_targets: Mapping[str, object],
    requested_thresholds: Mapping[str, float | None],
    seed_values: Mapping[str, float | None] | None = None,
    validate_seed: bool = False,
    seed_context: str = "Loaded seed",
) -> dict[str, float]:
    resolved_thresholds = resolve_penalty_traversal_forbidden_box_bounds(
        requested_thresholds,
    )
    applied_thresholds: dict[str, float] = {}
    for name, threshold in resolved_thresholds.items():
        target = bound_targets.get(name)
        if target is None:
            raise KeyError(
                f"Missing penalty box-bound target for hardware constraint {name!r}."
            )
        handler = _penalty_box_bound_handler(name)
        if validate_seed and seed_values is not None:
            seed_value = seed_values.get(name)
            if (
                seed_value is not None
                and handler.exceeds_limit(float(seed_value), threshold)
            ):
                raise ValueError(
                    f"{seed_context} {name}={float(seed_value):.6f} exceeds the "
                    f"traversal-forbidden penalty box bound {threshold:.6f}."
                )
        handler.apply_bound(target, threshold)
        applied_thresholds[name] = threshold
    return applied_thresholds


def infer_uniform_coil_current_A(coils) -> float | None:
    if not coils:
        return None
    coil_currents = np.asarray([coil.current.get_value() for coil in coils], dtype=float)
    if np.allclose(coil_currents, coil_currents[0], rtol=0.0, atol=1.0e-12):
        return float(coil_currents[0])
    return None


def resolve_loaded_tf_current_A(recorded_tf_current_A, tf_coils) -> float:
    realized_tf_current_A = infer_uniform_coil_current_A(tf_coils)
    if realized_tf_current_A is None:
        raise ValueError(
            "Loaded Stage 2 TF coils do not share a uniform fixed current; cannot "
            "validate the seed current contract."
        )
    if recorded_tf_current_A is not None and not np.isclose(
        realized_tf_current_A,
        float(recorded_tf_current_A),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Loaded Stage 2 TF coil current "
            f"{realized_tf_current_A:.6f} A does not match the artifact metadata "
            f"TF_CURRENT_A={float(recorded_tf_current_A):.6f} A."
        )
    return validate_tf_current_limit(realized_tf_current_A)


def resolve_plasma_current_settings(
    *,
    raw_boozer_I: float | None,
    plasma_current_A: float | None,
    finite_current_mode: FiniteCurrentMode = DEFAULT_FINITE_CURRENT_MODE,
    default_plasma_current_A: float = 0.0,
) -> PlasmaCurrentSettings:
    boozer_current_convention = resolve_boozer_current_convention(finite_current_mode)
    if plasma_current_A is not None:
        if raw_boozer_I is not None:
            raise ValueError("Cannot use --plasma-current-A together with --boozer-I")
        resolved_boozer_I = physical_current_to_boozer_I(
            plasma_current_A,
            convention=boozer_current_convention,
        )
        return _build_plasma_current_settings(
            boozer_I=resolved_boozer_I,
            plasma_current_A=plasma_current_A,
            input_source="physical_A",
            boozer_current_convention=boozer_current_convention,
            finite_current_mode=finite_current_mode,
        )
    if raw_boozer_I is not None:
        resolved_boozer_I = float(raw_boozer_I)
        return _build_plasma_current_settings(
            boozer_I=resolved_boozer_I,
            plasma_current_A=boozer_I_to_physical_current_A(
                raw_boozer_I,
                convention=boozer_current_convention,
            ),
            input_source="raw_boozer_I",
            boozer_current_convention=boozer_current_convention,
            finite_current_mode=finite_current_mode,
        )

    resolved_default_plasma_current_A = float(default_plasma_current_A)
    resolved_default_boozer_I = physical_current_to_boozer_I(
        resolved_default_plasma_current_A,
        convention=boozer_current_convention,
    )
    return _build_plasma_current_settings(
        boozer_I=resolved_default_boozer_I,
        plasma_current_A=resolved_default_plasma_current_A,
        input_source=(
            "default_zero"
            if abs(resolved_default_plasma_current_A) <= CURRENT_MODE_ZERO_TOL
            else "artifact_default_A"
        ),
        boozer_current_convention=boozer_current_convention,
        finite_current_mode=finite_current_mode,
    )


def resolve_single_surface_plasma_current_settings(
    *,
    raw_boozer_I: float | None,
    plasma_current_A: float | None,
    default_plasma_current_A: float = 0.0,
) -> PlasmaCurrentSettings:
    """Resolve the single-surface Boozer current using Wataru's contract.

    User-facing current input remains physical plasma current in amperes. The
    solver-facing BoozerSurface ``I`` parameter is then derived as ``mu0 * I_A``.
    A raw Boozer-current value is still allowed as an explicit expert override.
    Single-surface mode also locks the recorded provenance mode to the Wataru
    proxy-field contract even though the numerical current convention is
    identical to the generic multisurface path today.
    """
    return resolve_plasma_current_settings(
        raw_boozer_I=raw_boozer_I,
        plasma_current_A=plasma_current_A,
        finite_current_mode=DEFAULT_FINITE_CURRENT_MODE,
        default_plasma_current_A=default_plasma_current_A,
    )


def resolve_plasma_current_settings_for_num_surfaces(
    *,
    raw_boozer_I: float | None,
    plasma_current_A: float | None,
    finite_current_mode: FiniteCurrentMode = DEFAULT_FINITE_CURRENT_MODE,
    default_plasma_current_A: float = 0.0,
    num_surfaces: int | None = 1,
    requested_finite_current_mode: FiniteCurrentMode | None = None,
) -> PlasmaCurrentSettings:
    resolved_num_surfaces = 1 if num_surfaces is None else int(num_surfaces)
    if resolved_num_surfaces == 1:
        if requested_finite_current_mode not in {
            None,
            "",
            DEFAULT_FINITE_CURRENT_MODE,
        }:
            raise ValueError(
                "Single-surface mode is locked to "
                f"{DEFAULT_FINITE_CURRENT_MODE!r}; remove --finite-current-mode or "
                f"set it to {DEFAULT_FINITE_CURRENT_MODE!r}."
            )
        return resolve_single_surface_plasma_current_settings(
            raw_boozer_I=raw_boozer_I,
            plasma_current_A=plasma_current_A,
            default_plasma_current_A=default_plasma_current_A,
        )
    return resolve_plasma_current_settings(
        raw_boozer_I=raw_boozer_I,
        plasma_current_A=plasma_current_A,
        finite_current_mode=finite_current_mode,
        default_plasma_current_A=default_plasma_current_A,
    )
