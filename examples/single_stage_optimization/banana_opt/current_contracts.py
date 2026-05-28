from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Mapping

import numpy as np

from banana_opt.hardware_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
    validate_tf_current_limit,
)
from banana_opt.hardware_constraint_schema import (
    hardware_constraint_penalty_box_bound_names,
    resolve_penalty_box_bound_threshold,
)
from banana_opt.surface_mode_contracts import (
    PUBLISHED_MULTISURFACE,
    SINGLE_SURFACE,
    SurfaceModeContract,
)

if TYPE_CHECKING:
    from simsopt.field.coil import Coil


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
G0Policy = Literal["signed_explicit_tf_current"]
ProxyPlacementPolicy = Literal[
    "none",
    "vmec_axis_zeroth_coefficients",
    "surface_major_radius_z0",
]
ProxyVfCurrentScalarPolicy = Literal[
    "none",
    "nonnegative_magnitude",
    "signed_physical_scalar",
]
VfCurrentSignPolicy = Literal[
    "none",
    "template_sign_vf_current_scalar",
    "template_sign_abs_proxy_current",
]
VfCurrentMutability = Literal[
    "none",
    "independent_fixed_current",
    "shared_unfixed_scaled_current",
]
BananaReplayPolicy = Literal[
    "stage2_cli_seed_current",
    "flip_banana_parent_suffix_env_BANANA_CURRENT_SIGN_BANANA_I_FIXED_S2",
]
FiniteCurrentMode = Literal[
    "vacuum",
    "boozer_surrogate",
    "wataru_proxy_field",
    "jhalpern30_proxy_field",
]
EffectiveCurrentMode = Literal[
    "vacuum",
    "boozer_surrogate",
    "wataru_proxy_field",
    "jhalpern30_proxy_field",
]
CURRENT_MODE_ZERO_TOL = 1e-12
DEFAULT_FINITE_CURRENT_MODE: FiniteCurrentMode = "wataru_proxy_field"
FINITE_CURRENT_MODE_CHOICES: tuple[FiniteCurrentMode, ...] = (
    "vacuum",
    "boozer_surrogate",
    DEFAULT_FINITE_CURRENT_MODE,
    "jhalpern30_proxy_field",
)
HBT_PROXY_VF_CURRENT_RATIO = 1.0 / 6.5
HBT_PROXY_VF_CURRENT_TOL_A = 1.0e-9
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
    "FINITE_CURRENT_MODE_CHOICES",
    "FiniteCurrentModeSource",
    "FINITE_CURRENT_MODE_SOURCE_ARTIFACT_METADATA",
    "FINITE_CURRENT_MODE_SOURCE_LEGACY_ASSUMED_DEFAULT",
    "BananaReplayPolicy",
    "G0Policy",
    "HBT_PROXY_VF_CURRENT_RATIO",
    "MU0",
    "MU0_OVER_2PI",
    "PenaltyBoxBoundHandler",
    "PlasmaCurrentSettings",
    "ProxyPlacementPolicy",
    "ProxyVfCurrentScalarPolicy",
    "VFCoilBuildResult",
    "VfCurrentMutability",
    "VfCurrentSignPolicy",
    "apply_banana_current_seed_sign_box_bound",
    "apply_banana_current_upper_bound",
    "apply_penalty_traversal_forbidden_box_bounds",
    "apply_vf_current_upper_bound",
    "banana_current_exceeds_limit",
    "boozer_I_to_physical_current_A",
    "infer_uniform_coil_current_A",
    "physical_current_to_boozer_I",
    "resolve_boozer_current_convention",
    "resolve_finite_current_mode",
    "resolve_effective_current_mode",
    "resolve_jhalpern30_fresh_vf_current_A",
    "validate_hbt_proxy_vf_current_convention",
    "validate_jhalpern30_proxy_vf_current_convention",
    "validate_proxy_vf_current_convention_for_mode",
    "validate_signed_proxy_vf_current_convention",
    "vf_current_exceeds_limit",
    "resolve_penalty_traversal_forbidden_box_bounds",
    "resolve_loaded_tf_current_A",
    "resolve_plasma_current_settings",
    "resolve_plasma_current_settings_for_num_surfaces",
    "resolve_plasma_current_settings_for_surface_mode",
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
class VFCoilBuildResult:
    coils: list[Coil]
    current_control: object | None


@dataclass(frozen=True)
class PenaltyBoxBoundHandler:
    apply_bound: Callable[[object, float], None]
    exceeds_limit: Callable[[float, float], bool]
    apply_seed_sign_bound: Callable[[object, float, float], None] | None = None


_BOOZER_CURRENT_SCALE_BY_CONVENTION: Mapping[BoozerCurrentConvention, float] = {
    "mu0_over_2pi": MU0_OVER_2PI,
    "mu0": MU0,
}

_BOOZER_CURRENT_CONVENTION_BY_MODE: Mapping[
    FiniteCurrentMode,
    BoozerCurrentConvention,
] = {
    "vacuum": "mu0",
    # SIMSOPT's BoozerSurface residual is written in normalized angles, so the
    # code-facing current function carries the 2π from the angle change of
    # variables. Physical enclosed current in amperes therefore maps to μ0*I_A
    # at the API boundary, not μ0/(2π)*I_A.
    "boozer_surrogate": "mu0",
    # Wataru confirmed his workflow intentionally uses BoozerSurfaceFiniteI(..., I=μ0*I_A)
    # with no extra 2π factor. This matches the normalized-angle SIMSOPT API.
    # (Originally written as BoozerSurface(..., I=...); the I= kwarg moved to the
    # examples-side BoozerSurfaceFiniteI wrapper during the finite-I refactor.)
    "wataru_proxy_field": "mu0",
    # jhalpern30 also passes BoozerSurface(..., I=mu0*proxy_current_A). The
    # compatibility mode keeps that parameter separate from fresh-run G0.
    "jhalpern30_proxy_field": "mu0",
}


def _validated_finite_current_mode(mode: str) -> FiniteCurrentMode:
    if mode == "vacuum":
        return "vacuum"
    if mode == "boozer_surrogate":
        return "boozer_surrogate"
    if mode == "wataru_proxy_field":
        return "wataru_proxy_field"
    if mode == "jhalpern30_proxy_field":
        return "jhalpern30_proxy_field"
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


def validate_hbt_proxy_vf_current_convention(
    *,
    proxy_plasma_current_A: float,
    vf_current_A: float,
) -> tuple[float, float]:
    """Validate Wataru's HBT proxy/VF current convention.

    The HBT notes define the proxy current as non-negative and the VF current
    as ``proxy / 6.5``. This contract intentionally does not derive proxy or VF
    signs from the TF-current sign.
    """
    proxy_current_A = float(proxy_plasma_current_A)
    resolved_vf_current_A = float(vf_current_A)
    if proxy_current_A < 0.0:
        raise ValueError("HBT proxy plasma current must be non-negative.")
    if resolved_vf_current_A < 0.0:
        raise ValueError("HBT VF current must be non-negative.")
    return _validate_proxy_vf_current_ratio(
        proxy_current_A,
        resolved_vf_current_A,
        context="HBT proxy/VF convention",
        requirement="--vf-current-A = --proxy-plasma-current-A / 6.5",
    )


def _validate_proxy_vf_current_ratio(
    proxy_current_A: float,
    resolved_vf_current_A: float,
    *,
    context: str,
    requirement: str,
) -> tuple[float, float]:
    if not np.isfinite(proxy_current_A) or not np.isfinite(resolved_vf_current_A):
        raise ValueError(f"{context} requires finite proxy and VF currents.")
    expected_vf_current_A = proxy_current_A * HBT_PROXY_VF_CURRENT_RATIO
    if not np.isclose(
        resolved_vf_current_A,
        expected_vf_current_A,
        rtol=1.0e-12,
        atol=HBT_PROXY_VF_CURRENT_TOL_A,
    ):
        raise ValueError(f"{context} requires {requirement}.")
    return proxy_current_A, resolved_vf_current_A


def validate_signed_proxy_vf_current_convention(
    *,
    proxy_plasma_current_A: float,
    vf_current_A: float,
) -> tuple[float, float]:
    proxy_current_A = float(proxy_plasma_current_A)
    resolved_vf_current_A = float(vf_current_A)
    if not np.isfinite(proxy_current_A) or not np.isfinite(resolved_vf_current_A):
        raise ValueError("signed proxy/VF convention requires finite currents.")
    return proxy_current_A, resolved_vf_current_A


def validate_jhalpern30_proxy_vf_current_convention(
    *,
    proxy_plasma_current_A: float,
    vf_current_A: float,
) -> tuple[float, float]:
    """Validate the signed proxy/VF scalar contract for jhalpern30 replay.

    Historical jhalpern30 runs allowed signed proxy current. The scalar
    ``VF_CURRENT_A`` starts from ``proxy / 6.5`` by default, but the jhalpern
    VF profile is shared and mutable, so optimized artifacts may carry a final
    signed VF scalar that no longer equals the proxy ratio.
    """
    if float(proxy_plasma_current_A) == 0.0:
        raise ValueError(
            "jhalpern30 proxy/VF convention requires non-zero PROXY_PLASMA_CURRENT_A."
        )
    return validate_signed_proxy_vf_current_convention(
        proxy_plasma_current_A=proxy_plasma_current_A,
        vf_current_A=vf_current_A,
    )


def resolve_jhalpern30_fresh_vf_current_A(
    *,
    proxy_plasma_current_A: float,
    requested_vf_current_A: float | None = None,
) -> float:
    """Resolve jhalpern30 fresh-run VF current from the signed proxy current."""
    proxy_current_A = float(proxy_plasma_current_A)
    derived_vf_current_A = proxy_current_A * HBT_PROXY_VF_CURRENT_RATIO
    validate_jhalpern30_proxy_vf_current_convention(
        proxy_plasma_current_A=proxy_current_A,
        vf_current_A=derived_vf_current_A,
    )
    if requested_vf_current_A is None:
        return derived_vf_current_A
    requested_vf_current_A = float(requested_vf_current_A)
    if not np.isfinite(requested_vf_current_A):
        raise ValueError("fresh jhalpern30 --vf-current-A must be finite.")
    if not np.isclose(
        requested_vf_current_A,
        derived_vf_current_A,
        rtol=1.0e-12,
        atol=HBT_PROXY_VF_CURRENT_TOL_A,
    ):
        raise ValueError(
            "fresh jhalpern30 derives --vf-current-A from "
            "--proxy-plasma-current-A / 6.5; omit --vf-current-A or pass the "
            "derived value. Explicit jhalpern30 VF retargeting is only "
            "supported for seeded current traversal."
        )
    return derived_vf_current_A


def validate_proxy_vf_current_convention_for_mode(
    finite_current_mode: FiniteCurrentMode,
    *,
    proxy_plasma_current_A: float,
    vf_current_A: float,
) -> tuple[float, float]:
    if finite_current_mode == "vacuum":
        if (
            abs(float(proxy_plasma_current_A)) > CURRENT_MODE_ZERO_TOL
            or abs(float(vf_current_A)) > CURRENT_MODE_ZERO_TOL
        ):
            raise ValueError(
                "vacuum finite-current mode requires zero proxy and VF currents."
            )
        return 0.0, 0.0
    if finite_current_mode == "jhalpern30_proxy_field":
        return validate_jhalpern30_proxy_vf_current_convention(
            proxy_plasma_current_A=proxy_plasma_current_A,
            vf_current_A=vf_current_A,
        )
    return validate_hbt_proxy_vf_current_convention(
        proxy_plasma_current_A=proxy_plasma_current_A,
        vf_current_A=vf_current_A,
    )


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


def _apply_scaled_symmetric_current_bound(
    current, current_max_A, *, label: str
) -> None:
    current_optimizable, scale = unwrap_current_optimizable(current)
    if scale == 0.0:
        raise ValueError(f"{label} current scale must be non-zero to apply a bound.")
    lower_bounds = np.asarray(
        current_optimizable.local_lower_bounds, dtype=float
    ).copy()
    upper_bounds = np.asarray(
        current_optimizable.local_upper_bounds, dtype=float
    ).copy()
    scaled_magnitude_bound = float(current_max_A) / abs(scale)
    lower_bounds[0] = max(lower_bounds[0], -scaled_magnitude_bound)
    upper_bounds[0] = min(upper_bounds[0], scaled_magnitude_bound)
    current_optimizable.local_lower_bounds = lower_bounds
    current_optimizable.local_upper_bounds = upper_bounds


def apply_banana_current_upper_bound(current, banana_current_max_A):
    _apply_scaled_symmetric_current_bound(
        current,
        banana_current_max_A,
        label="Banana",
    )


def apply_banana_current_seed_sign_box_bound(
    current,
    banana_current_max_A,
    seed_current_A,
):
    seed_current_A = float(seed_current_A)
    if seed_current_A == 0.0:
        raise ValueError("Banana current seed must be non-zero to preserve its sign.")
    current_optimizable, scale = unwrap_current_optimizable(current)
    if scale == 0.0:
        raise ValueError("Banana current scale must be non-zero to apply a bound.")
    lower_bounds = np.asarray(
        current_optimizable.local_lower_bounds, dtype=float
    ).copy()
    upper_bounds = np.asarray(
        current_optimizable.local_upper_bounds, dtype=float
    ).copy()
    scaled_magnitude_bound = float(banana_current_max_A) / abs(scale)
    seed_sign_matches_scale = np.sign(seed_current_A) == np.sign(scale)
    if seed_sign_matches_scale:
        lower_bounds[0] = max(lower_bounds[0], 0.0)
        upper_bounds[0] = min(upper_bounds[0], scaled_magnitude_bound)
    else:
        lower_bounds[0] = max(lower_bounds[0], -scaled_magnitude_bound)
        upper_bounds[0] = min(upper_bounds[0], 0.0)
    if lower_bounds[0] > upper_bounds[0]:
        raise ValueError("Banana current sign-preserving bounds are inconsistent.")
    current_optimizable.local_lower_bounds = lower_bounds
    current_optimizable.local_upper_bounds = upper_bounds


def banana_current_exceeds_limit(current_A: float, banana_current_max_A: float) -> bool:
    return abs(float(current_A)) > float(banana_current_max_A)


def apply_vf_current_upper_bound(current, vf_current_max_A):
    _apply_scaled_symmetric_current_bound(
        current,
        vf_current_max_A,
        label="VF",
    )


def vf_current_exceeds_limit(current_A: float, vf_current_max_A: float) -> bool:
    return abs(float(current_A)) > float(vf_current_max_A)


_PENALTY_BOX_BOUND_HANDLERS: Mapping[str, PenaltyBoxBoundHandler] = {
    "banana_current": PenaltyBoxBoundHandler(
        apply_bound=apply_banana_current_upper_bound,
        exceeds_limit=banana_current_exceeds_limit,
        apply_seed_sign_bound=apply_banana_current_seed_sign_box_bound,
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
    *,
    allow_offspec_threshold_names: frozenset[str] = frozenset(),
) -> dict[str, float]:
    # Only penalty-search box bounds need runtime handlers here. ALM and
    # artifact enforcement consume the schema through separate paths.
    resolved_thresholds: dict[str, float] = {}
    for name in hardware_constraint_penalty_box_bound_names(
        traversal_policy="forbidden",
    ):
        requested_threshold = requested_thresholds.get(name)
        if requested_threshold is not None and name in allow_offspec_threshold_names:
            resolved_thresholds[name] = float(requested_threshold)
            continue
        resolved_thresholds[name] = resolve_penalty_box_bound_threshold(
            name,
            requested_threshold=requested_threshold,
        )
    return resolved_thresholds


def apply_penalty_traversal_forbidden_box_bounds(
    *,
    bound_targets: Mapping[str, object],
    requested_thresholds: Mapping[str, float | None],
    seed_values: Mapping[str, float | None] | None = None,
    validate_seed: bool = False,
    seed_context: str = "Loaded seed",
    allow_offspec_threshold_names: frozenset[str] = frozenset(),
    preserve_seed_sign_names: frozenset[str] = frozenset(),
) -> dict[str, float]:
    resolved_thresholds = resolve_penalty_traversal_forbidden_box_bounds(
        requested_thresholds,
        allow_offspec_threshold_names=allow_offspec_threshold_names,
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
            if seed_value is not None and handler.exceeds_limit(
                float(seed_value), threshold
            ):
                raise ValueError(
                    f"{seed_context} {name}={float(seed_value):.6f} exceeds the "
                    f"traversal-forbidden penalty box bound {threshold:.6f}."
                )
        if name in preserve_seed_sign_names:
            if seed_values is None or seed_values.get(name) is None:
                raise ValueError(
                    f"{seed_context} {name} is required to preserve the seed sign."
                )
            if handler.apply_seed_sign_bound is None:
                raise ValueError(
                    f"No sign-preserving penalty box-bound handler registered for "
                    f"hardware constraint {name!r}."
                )
            handler.apply_seed_sign_bound(target, threshold, float(seed_values[name]))
        else:
            handler.apply_bound(target, threshold)
        applied_thresholds[name] = threshold
    return applied_thresholds


def infer_uniform_coil_current_A(coils) -> float | None:
    if not coils:
        return None
    coil_currents = np.asarray(
        [coil.current.get_value() for coil in coils], dtype=float
    )
    if np.allclose(coil_currents, coil_currents[0], rtol=0.0, atol=1.0e-12):
        return float(coil_currents[0])
    return None


def resolve_loaded_tf_current_A(
    recorded_tf_current_A,
    tf_coils,
    *,
    allow_offspec_current_contract: bool = False,
) -> float:
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
    if allow_offspec_current_contract:
        if not np.isfinite(realized_tf_current_A) or realized_tf_current_A == 0.0:
            raise ValueError(
                "Loaded Stage 2 TF coil current must be finite and non-zero."
            )
        return float(realized_tf_current_A)
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
        if (
            finite_current_mode == "vacuum"
            and abs(float(plasma_current_A)) > CURRENT_MODE_ZERO_TOL
        ):
            raise ValueError("vacuum finite-current mode requires zero plasma current.")
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
        if (
            finite_current_mode == "vacuum"
            and abs(resolved_boozer_I) > CURRENT_MODE_ZERO_TOL
        ):
            raise ValueError("vacuum finite-current mode requires zero Boozer I.")
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
    if (
        finite_current_mode == "vacuum"
        and abs(resolved_default_plasma_current_A) > CURRENT_MODE_ZERO_TOL
    ):
        raise ValueError(
            "vacuum finite-current mode requires zero default plasma current."
        )
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
    finite_current_mode: FiniteCurrentMode = DEFAULT_FINITE_CURRENT_MODE,
    default_plasma_current_A: float = 0.0,
) -> PlasmaCurrentSettings:
    """Resolve the single-surface Boozer current for an explicit replay mode.

    User-facing current input remains physical plasma current in amperes. The
    solver-facing BoozerSurface ``I`` parameter is then derived as ``mu0 * I_A``.
    A raw Boozer-current value is still allowed as an explicit expert override.
    """
    return resolve_plasma_current_settings(
        raw_boozer_I=raw_boozer_I,
        plasma_current_A=plasma_current_A,
        finite_current_mode=finite_current_mode,
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
            "vacuum",
            DEFAULT_FINITE_CURRENT_MODE,
            "jhalpern30_proxy_field",
        }:
            raise ValueError(
                "Single-surface mode is locked to "
                f"'vacuum', {DEFAULT_FINITE_CURRENT_MODE!r}, or "
                "'jhalpern30_proxy_field'; "
                "remove --finite-current-mode or select one of those replay modes."
            )
        if finite_current_mode == "vacuum":
            single_surface_mode = "vacuum"
        elif finite_current_mode == "jhalpern30_proxy_field":
            single_surface_mode = "jhalpern30_proxy_field"
        else:
            single_surface_mode = DEFAULT_FINITE_CURRENT_MODE
        return resolve_single_surface_plasma_current_settings(
            raw_boozer_I=raw_boozer_I,
            plasma_current_A=plasma_current_A,
            finite_current_mode=single_surface_mode,
            default_plasma_current_A=default_plasma_current_A,
        )
    return resolve_plasma_current_settings(
        raw_boozer_I=raw_boozer_I,
        plasma_current_A=plasma_current_A,
        finite_current_mode=finite_current_mode,
        default_plasma_current_A=default_plasma_current_A,
    )


def resolve_plasma_current_settings_for_surface_mode(
    *,
    raw_boozer_I: float | None,
    plasma_current_A: float | None,
    finite_current_mode: FiniteCurrentMode = DEFAULT_FINITE_CURRENT_MODE,
    default_plasma_current_A: float = 0.0,
    surface_mode_contract: SurfaceModeContract,
    requested_finite_current_mode: FiniteCurrentMode | None = None,
) -> PlasmaCurrentSettings:
    if surface_mode_contract.mode == PUBLISHED_MULTISURFACE:
        # published_multisurface v1 is vacuum-locked. It accepts legacy/default
        # mode tokens as CLI compatibility aliases, then normalizes telemetry and
        # downstream settings to the plain vacuum finite-current mode.
        if requested_finite_current_mode not in {
            None,
            "",
            "vacuum",
            DEFAULT_FINITE_CURRENT_MODE,
        }:
            raise ValueError(
                "published_multisurface v1 is vacuum-locked; remove "
                "--finite-current-mode or set it to "
                f"'vacuum' or {DEFAULT_FINITE_CURRENT_MODE!r}."
            )
        if finite_current_mode not in {"vacuum", DEFAULT_FINITE_CURRENT_MODE}:
            raise ValueError(
                "published_multisurface v1 is vacuum-locked and cannot inherit "
                f"finite-current donor mode {finite_current_mode!r}."
            )
        if raw_boozer_I is not None:
            raise ValueError(
                "published_multisurface v1 is vacuum-locked and rejects raw "
                "--boozer-I overrides."
            )
        if (
            plasma_current_A is not None
            and abs(float(plasma_current_A)) > CURRENT_MODE_ZERO_TOL
        ):
            raise ValueError(
                "published_multisurface v1 is vacuum-locked and rejects nonzero "
                "--plasma-current-A."
            )
        if abs(float(default_plasma_current_A)) > CURRENT_MODE_ZERO_TOL:
            raise ValueError(
                "published_multisurface v1 is vacuum-locked and rejects nonzero "
                "artifact default plasma current."
            )
        return resolve_single_surface_plasma_current_settings(
            raw_boozer_I=None,
            plasma_current_A=plasma_current_A,
            finite_current_mode="vacuum",
            default_plasma_current_A=0.0,
        )
    if surface_mode_contract.mode == SINGLE_SURFACE:
        return resolve_plasma_current_settings_for_num_surfaces(
            raw_boozer_I=raw_boozer_I,
            plasma_current_A=plasma_current_A,
            finite_current_mode=finite_current_mode,
            default_plasma_current_A=default_plasma_current_A,
            num_surfaces=1,
            requested_finite_current_mode=requested_finite_current_mode,
        )
    return resolve_plasma_current_settings(
        raw_boozer_I=raw_boozer_I,
        plasma_current_A=plasma_current_A,
        finite_current_mode=finite_current_mode,
        default_plasma_current_A=default_plasma_current_A,
    )
