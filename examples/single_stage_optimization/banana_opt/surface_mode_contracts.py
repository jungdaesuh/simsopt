from __future__ import annotations

from dataclasses import dataclass
import warnings


SINGLE_SURFACE = "single_surface"
PUBLISHED_MULTISURFACE = "published_multisurface"
EXPERIMENTAL_MULTISURFACE = "experimental_multisurface"
SURFACE_MODE_CHOICES = (
    SINGLE_SURFACE,
    PUBLISHED_MULTISURFACE,
    EXPERIMENTAL_MULTISURFACE,
)
SURFACE_MODE_VERSION_V1 = 1
DEFAULT_INNER_SURFACE_RATIO = 0.8

SURFACE_MODE_SOURCE_EXPLICIT_CLI = "explicit_cli"
SURFACE_MODE_SOURCE_LEGACY_NUM_SURFACES_MAPPING = "legacy_num_surfaces_mapping"
SURFACE_MODE_SOURCE_WRAPPER_DEFINED = "wrapper_defined"

SURFACE_STACK_POLICY_SINGLE_SURFACE_DIRECT = "single_surface_direct"
SURFACE_STACK_POLICY_PUBLISHED_FIXED_STACK = "published_fixed_stack"
SURFACE_STACK_POLICY_EXPERIMENTAL_CONTINUATION_STACK = (
    "experimental_continuation_stack"
)

_PUBLISHED_LABEL_FRACTIONS_V1 = (0.6, 0.8, 1.0)


@dataclass(frozen=True)
class SurfaceModeContract:
    mode: str
    version: int
    source: str
    label_fractions: tuple[float, ...]
    weights: tuple[float, ...]
    stack_policy: str
    physics_contract: str
    legacy_num_surfaces: int | None
    legacy_inner_surface_ratio: float | None

    @property
    def num_surfaces(self) -> int:
        return len(self.label_fractions)


def _validate_surface_mode_name(mode: str) -> str:
    if mode not in SURFACE_MODE_CHOICES:
        raise ValueError(
            "Unsupported --surface-mode "
            f"{mode!r}. Expected one of {SURFACE_MODE_CHOICES!r}."
        )
    return mode


def _validate_inner_surface_ratio(inner_surface_ratio: float | None) -> float:
    if inner_surface_ratio is None:
        raise ValueError(
            "--inner-surface-ratio is required for experimental_multisurface"
        )
    ratio = float(inner_surface_ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError(
            "--inner-surface-ratio must be between 0 and 1 when the "
            "experimental_multisurface contract is selected"
        )
    return ratio


def resolve_surface_mode(
    *,
    requested_surface_mode: str | None,
    legacy_num_surfaces: int | None,
    warn_on_legacy_mapping: bool = True,
) -> tuple[str, str]:
    if requested_surface_mode is not None:
        return (
            _validate_surface_mode_name(str(requested_surface_mode)),
            SURFACE_MODE_SOURCE_EXPLICIT_CLI,
        )

    resolved_legacy_num_surfaces = 1 if legacy_num_surfaces is None else int(
        legacy_num_surfaces
    )
    if resolved_legacy_num_surfaces == 1:
        mode = SINGLE_SURFACE
    elif resolved_legacy_num_surfaces == 2:
        mode = EXPERIMENTAL_MULTISURFACE
    else:
        raise ValueError(
            "Legacy --num-surfaces currently supports only 1 or 2. "
            f"Received {resolved_legacy_num_surfaces}."
        )

    if warn_on_legacy_mapping:
        warnings.warn(
            "Legacy --num-surfaces surface selection is deprecated; prefer "
            "--surface-mode.",
            DeprecationWarning,
            stacklevel=3,
        )
    return mode, SURFACE_MODE_SOURCE_LEGACY_NUM_SURFACES_MAPPING


def resolve_surface_label_fractions(
    mode: str,
    *,
    legacy_inner_surface_ratio: float | None,
) -> tuple[float, ...]:
    resolved_mode = _validate_surface_mode_name(mode)
    if resolved_mode == SINGLE_SURFACE:
        return (1.0,)
    if resolved_mode == PUBLISHED_MULTISURFACE:
        return _PUBLISHED_LABEL_FRACTIONS_V1
    return (_validate_inner_surface_ratio(legacy_inner_surface_ratio), 1.0)


def resolve_surface_weights(mode: str) -> tuple[float, ...]:
    resolved_mode = _validate_surface_mode_name(mode)
    if resolved_mode == SINGLE_SURFACE:
        return (1.0,)
    if resolved_mode == PUBLISHED_MULTISURFACE:
        return (1.0, 1.0, 1.0)
    return (1.0, 1.0)


def resolve_surface_stack_policy(mode: str) -> str:
    resolved_mode = _validate_surface_mode_name(mode)
    if resolved_mode == SINGLE_SURFACE:
        return SURFACE_STACK_POLICY_SINGLE_SURFACE_DIRECT
    if resolved_mode == PUBLISHED_MULTISURFACE:
        return SURFACE_STACK_POLICY_PUBLISHED_FIXED_STACK
    return SURFACE_STACK_POLICY_EXPERIMENTAL_CONTINUATION_STACK


def resolve_surface_physics_contract(mode: str) -> str:
    resolved_mode = _validate_surface_mode_name(mode)
    if resolved_mode == SINGLE_SURFACE:
        return (
            "single outer surface; QS, Boozer residual, iota, and volume all use "
            "the outer surface"
        )
    if resolved_mode == PUBLISHED_MULTISURFACE:
        return (
            "fixed multisurface stack; QS and Boozer residual aggregate across all "
            "configured surfaces; outer-surface iota and volume only in v1"
        )
    return (
        "custom two-surface stack; QS and Boozer residual aggregate across both "
        "surfaces; iota and volume stay outer-only; continuation-gated search policy"
    )


def resolve_surface_mode_inner_surface_ratio(
    contract: SurfaceModeContract,
    *,
    fallback_inner_surface_ratio: float | None,
) -> float:
    if contract.mode == EXPERIMENTAL_MULTISURFACE:
        return float(contract.label_fractions[0])
    if fallback_inner_surface_ratio is None:
        return DEFAULT_INNER_SURFACE_RATIO
    return float(fallback_inner_surface_ratio)


def build_surface_mode_contract(
    *,
    requested_surface_mode: str | None,
    legacy_num_surfaces: int | None,
    legacy_inner_surface_ratio: float | None,
    surface_mode_source: str | None = None,
    warn_on_legacy_mapping: bool = True,
) -> SurfaceModeContract:
    if surface_mode_source is None:
        mode, source = resolve_surface_mode(
            requested_surface_mode=requested_surface_mode,
            legacy_num_surfaces=legacy_num_surfaces,
            warn_on_legacy_mapping=warn_on_legacy_mapping,
        )
    else:
        if requested_surface_mode is None:
            raise ValueError(
                "surface_mode_source override requires an explicit surface mode"
            )
        mode = _validate_surface_mode_name(str(requested_surface_mode))
        source = surface_mode_source

    legacy_ratio = (
        float(legacy_inner_surface_ratio)
        if legacy_inner_surface_ratio is not None
        else None
    )
    return SurfaceModeContract(
        mode=mode,
        version=SURFACE_MODE_VERSION_V1,
        source=source,
        label_fractions=resolve_surface_label_fractions(
            mode,
            legacy_inner_surface_ratio=legacy_ratio,
        ),
        weights=resolve_surface_weights(mode),
        stack_policy=resolve_surface_stack_policy(mode),
        physics_contract=resolve_surface_physics_contract(mode),
        legacy_num_surfaces=(
            None
            if source != SURFACE_MODE_SOURCE_LEGACY_NUM_SURFACES_MAPPING
            else (1 if legacy_num_surfaces is None else int(legacy_num_surfaces))
        ),
        legacy_inner_surface_ratio=(
            None
            if source != SURFACE_MODE_SOURCE_LEGACY_NUM_SURFACES_MAPPING
            else legacy_ratio
        ),
    )


def build_surface_mode_metadata(contract: SurfaceModeContract) -> dict[str, object]:
    return {
        "SURFACE_MODE": contract.mode,
        "SURFACE_MODE_VERSION": contract.version,
        "SURFACE_MODE_SOURCE": contract.source,
        "SURFACE_LABEL_FRACTIONS": [float(value) for value in contract.label_fractions],
        "SURFACE_WEIGHTS": [float(value) for value in contract.weights],
        "SURFACE_STACK_POLICY": contract.stack_policy,
        "SURFACE_PHYSICS_CONTRACT": contract.physics_contract,
        "LEGACY_NUM_SURFACES": contract.legacy_num_surfaces,
        "LEGACY_INNER_SURFACE_RATIO": contract.legacy_inner_surface_ratio,
    }


def _resolve_mode_name(mode_or_contract: str | SurfaceModeContract) -> str:
    if isinstance(mode_or_contract, SurfaceModeContract):
        return mode_or_contract.mode
    return _validate_surface_mode_name(str(mode_or_contract))


def surface_mode_supports_alm(mode_or_contract: str | SurfaceModeContract) -> bool:
    return _resolve_mode_name(mode_or_contract) == SINGLE_SURFACE


def surface_mode_supports_boozer_stage_refinement(
    mode_or_contract: str | SurfaceModeContract,
) -> bool:
    return _resolve_mode_name(mode_or_contract) == SINGLE_SURFACE


def surface_mode_supports_topology_gate(
    mode_or_contract: str | SurfaceModeContract,
) -> bool:
    return _resolve_mode_name(mode_or_contract) == EXPERIMENTAL_MULTISURFACE


def surface_mode_runtime_supported(
    mode_or_contract: str | SurfaceModeContract,
) -> bool:
    return _resolve_mode_name(mode_or_contract) != PUBLISHED_MULTISURFACE


def validate_surface_mode_runtime_support(contract: SurfaceModeContract) -> None:
    if surface_mode_runtime_supported(contract):
        return
    raise ValueError(
        "published_multisurface is recognized but not implemented yet in the "
        "current single-stage runtime. Use single_surface or "
        "experimental_multisurface for now."
    )
