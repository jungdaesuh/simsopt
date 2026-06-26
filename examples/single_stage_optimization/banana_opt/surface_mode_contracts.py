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
SURFACE_STACK_POLICY_EXPERIMENTAL_CONTINUATION_STACK = "experimental_continuation_stack"

SURFACE_COUNT_POLICY_SINGLE_SURFACE = "single_surface"
SURFACE_COUNT_POLICY_EXPERIMENTAL_TWO_SURFACES = "experimental_two_surfaces"
SURFACE_COUNT_POLICY_PUBLISHED_FIXED_STACK_V1 = "published_fixed_stack_v1"

FINAL_REFINEMENT_POLICY_BOOZER_STAGE = "boozer_stage_refinement"
FINAL_REFINEMENT_POLICY_UNSUPPORTED = "unsupported"

CURRENT_POLICY_INHERIT_STAGE2 = "inherit_stage2"
CURRENT_POLICY_VACUUM_LOCKED = "vacuum_locked"

TOPOLOGY_POLICY_UNSUPPORTED = "unsupported"
TOPOLOGY_POLICY_SEARCH_GATE = "search_gate"

SURFACE_MODE_TELEMETRY_SCHEMA_VERSION_V1 = 1

PRODUCTION_SUPPORT_LEVEL_PRODUCTION = "production"
PRODUCTION_SUPPORT_LEVEL_PUBLISHED_V1 = "published_v1"
PRODUCTION_SUPPORT_LEVEL_EXPERIMENTAL = "experimental"

_PUBLISHED_LABEL_FRACTIONS_V1 = (0.6, 0.8, 1.0)
_SINGLE_SURFACE_NAMES = ("outer",)
_EXPERIMENTAL_SURFACE_NAMES = ("inner", "outer")
# Published surface names are derived from the surface count via
# published_surface_names(); the v1 default 3-surface stack reproduces
# ("inner0", "inner1", "outer") byte-identically.

# Named interior-covering presets for the published multisurface stack. Each
# preset is a strictly increasing tuple of normalized-flux *fractions* of the
# outer seed label whose final entry is 1.0 (the outer surface always carries
# the iota/volume label). The default 3-shell stack only constrains QS at
# s >= 0.36 (fraction 0.6 of the outer label); the interior-covering presets
# add inner shells that march toward the core (down to fraction 0.2) so the
# stack better approximates the 8-9 nested surfaces used by Giuliani et al.
# These are opt-in: callers must select a preset (or pass explicit fractions)
# for anything other than the default v1 stack.
PUBLISHED_PRESET_DEFAULT_V1 = "default_v1"
PUBLISHED_PRESET_INTERIOR_COVERING = "interior_covering"
PUBLISHED_PRESET_INTERIOR_COVERING_DEEP = "interior_covering_deep"
PUBLISHED_PRESET_EDGE_DELIVERED_IOTA_LANE = "edge_delivered_iota_lane"
_PUBLISHED_PRESET_FRACTIONS = {
    PUBLISHED_PRESET_DEFAULT_V1: _PUBLISHED_LABEL_FRACTIONS_V1,
    PUBLISHED_PRESET_INTERIOR_COVERING: (0.4, 0.6, 0.8, 1.0),
    PUBLISHED_PRESET_INTERIOR_COVERING_DEEP: (0.2, 0.4, 0.6, 0.8, 1.0),
    PUBLISHED_PRESET_EDGE_DELIVERED_IOTA_LANE: (0.70, 0.85, 0.95, 1.0),
}
PUBLISHED_PRESET_CHOICES = tuple(_PUBLISHED_PRESET_FRACTIONS)


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
    requires_inner_surface_ratio: bool
    surface_count_policy: str
    final_refinement_policy: str
    current_policy: str
    topology_policy: str
    telemetry_schema_version: int
    production_support_level: str

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


def validate_published_label_fractions(
    label_fractions: tuple[float, ...],
) -> tuple[float, ...]:
    """Validate an opt-in published multisurface label-fraction stack.

    The stack must be a non-empty, strictly increasing tuple of normalized-flux
    fractions in (0, 1] whose outermost entry is exactly 1.0 (the outer surface
    always carries the iota/volume label). Anything else is fail-closed because
    the downstream BoozerSurface stack derives strictly volume-ordered target
    volumes from these fractions and the outer-anchored continuation/G-lock
    invariants depend on the outermost surface being the seed label.
    """
    fractions = tuple(float(value) for value in label_fractions)
    if len(fractions) < 1:
        raise ValueError(
            "published label fractions must contain at least one surface"
        )
    if any(not (0.0 < value <= 1.0) for value in fractions):
        raise ValueError(
            "published label fractions must each lie in (0, 1]; "
            f"received {fractions!r}"
        )
    if any(
        right <= left for left, right in zip(fractions[:-1], fractions[1:])
    ):
        raise ValueError(
            "published label fractions must be strictly increasing from inner "
            f"to outer; received {fractions!r}"
        )
    if fractions[-1] != 1.0:
        raise ValueError(
            "the outermost published label fraction must be exactly 1.0 so the "
            f"outer surface carries the iota/volume label; received {fractions!r}"
        )
    return fractions


def resolve_published_preset_label_fractions(preset: str) -> tuple[float, ...]:
    """Map a named interior-covering preset to its validated label fractions."""
    if preset not in _PUBLISHED_PRESET_FRACTIONS:
        raise ValueError(
            f"Unsupported published surface preset {preset!r}. Expected one of "
            f"{PUBLISHED_PRESET_CHOICES!r}."
        )
    return validate_published_label_fractions(_PUBLISHED_PRESET_FRACTIONS[preset])


def resolve_published_label_fractions(
    *,
    published_label_fractions: tuple[float, ...] | None,
    published_surface_preset: str | None,
) -> tuple[float, ...]:
    """Resolve the published stack fractions from the opt-in knobs.

    Returns the default v1 stack byte-identically when neither knob is set.
    Explicit fractions take precedence over a named preset; supplying both an
    explicit stack and a non-default preset is rejected to keep the resolved
    stack single-source-of-truth.
    """
    if published_label_fractions is not None:
        if (
            published_surface_preset is not None
            and published_surface_preset != PUBLISHED_PRESET_DEFAULT_V1
        ):
            raise ValueError(
                "Pass either explicit published label fractions or a named "
                "preset, not both."
            )
        return validate_published_label_fractions(published_label_fractions)
    if published_surface_preset is not None:
        return resolve_published_preset_label_fractions(published_surface_preset)
    return _PUBLISHED_LABEL_FRACTIONS_V1


def published_surface_names(num_surfaces: int) -> tuple[str, ...]:
    """Generate volume-ordered names (inner0..innerN-2, outer) for N surfaces.

    The default v1 stack (3 surfaces) reproduces ("inner0", "inner1", "outer")
    byte-identically.
    """
    if num_surfaces < 1:
        raise ValueError("published stack must contain at least one surface")
    inner_names = tuple(f"inner{index}" for index in range(num_surfaces - 1))
    return inner_names + ("outer",)


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

    resolved_legacy_num_surfaces = (
        1 if legacy_num_surfaces is None else int(legacy_num_surfaces)
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
    published_label_fractions: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    resolved_mode = _validate_surface_mode_name(mode)
    if resolved_mode == SINGLE_SURFACE:
        return (1.0,)
    if resolved_mode == PUBLISHED_MULTISURFACE:
        if published_label_fractions is None:
            return _PUBLISHED_LABEL_FRACTIONS_V1
        return validate_published_label_fractions(published_label_fractions)
    return (_validate_inner_surface_ratio(legacy_inner_surface_ratio), 1.0)


def resolve_surface_weights(
    mode: str,
    *,
    published_label_fractions: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    resolved_mode = _validate_surface_mode_name(mode)
    if resolved_mode == SINGLE_SURFACE:
        return (1.0,)
    if resolved_mode == PUBLISHED_MULTISURFACE:
        fractions = resolve_surface_label_fractions(
            resolved_mode,
            legacy_inner_surface_ratio=None,
            published_label_fractions=published_label_fractions,
        )
        return tuple(1.0 for _ in fractions)
    return (1.0, 1.0)


def resolve_surface_stack_policy(mode: str) -> str:
    resolved_mode = _validate_surface_mode_name(mode)
    if resolved_mode == SINGLE_SURFACE:
        return SURFACE_STACK_POLICY_SINGLE_SURFACE_DIRECT
    if resolved_mode == PUBLISHED_MULTISURFACE:
        return SURFACE_STACK_POLICY_PUBLISHED_FIXED_STACK
    return SURFACE_STACK_POLICY_EXPERIMENTAL_CONTINUATION_STACK


def resolve_surface_physics_contract(
    mode: str,
    *,
    published_label_fractions: tuple[float, ...] | None = None,
) -> str:
    resolved_mode = _validate_surface_mode_name(mode)
    if resolved_mode == SINGLE_SURFACE:
        return (
            "single outer surface; QS, Boozer residual, iota, and volume all use "
            "the outer surface"
        )
    if resolved_mode == PUBLISHED_MULTISURFACE:
        fractions = resolve_surface_label_fractions(
            resolved_mode,
            legacy_inner_surface_ratio=None,
            published_label_fractions=published_label_fractions,
        )
        # Deviation note: the v1 default constrains QS at three mid-to-edge
        # shells (label fractions 0.6/0.8/1.0 -> s ~ 0.36/0.64/1.0), coarser
        # than the 8-9 nested surfaces of Giuliani et al. The stack depth is now
        # configurable (interior-covering presets / explicit fractions) to add
        # inner shells that reach toward the core; the default stays the v1
        # 3-shell stack. iota and volume remain outer-only regardless of depth.
        return (
            "vacuum-locked fixed multisurface stack over "
            f"{len(fractions)} configurable nested surfaces (label fractions "
            f"{tuple(float(value) for value in fractions)!r}); QS and Boozer "
            "residual aggregate across all configured surfaces; outer-surface "
            "iota and volume only"
        )
    return (
        "custom two-surface stack; QS and Boozer residual aggregate across both "
        "surfaces; iota and volume stay outer-only; continuation-gated search policy"
    )


def resolve_surface_mode_requires_inner_surface_ratio(mode: str) -> bool:
    return _validate_surface_mode_name(mode) == EXPERIMENTAL_MULTISURFACE


def resolve_surface_count_policy(mode: str) -> str:
    resolved_mode = _validate_surface_mode_name(mode)
    if resolved_mode == SINGLE_SURFACE:
        return SURFACE_COUNT_POLICY_SINGLE_SURFACE
    if resolved_mode == PUBLISHED_MULTISURFACE:
        return SURFACE_COUNT_POLICY_PUBLISHED_FIXED_STACK_V1
    return SURFACE_COUNT_POLICY_EXPERIMENTAL_TWO_SURFACES


def resolve_final_refinement_policy(mode: str) -> str:
    if _validate_surface_mode_name(mode) == SINGLE_SURFACE:
        return FINAL_REFINEMENT_POLICY_BOOZER_STAGE
    return FINAL_REFINEMENT_POLICY_UNSUPPORTED


def resolve_current_policy(mode: str) -> str:
    if _validate_surface_mode_name(mode) == PUBLISHED_MULTISURFACE:
        return CURRENT_POLICY_VACUUM_LOCKED
    return CURRENT_POLICY_INHERIT_STAGE2


def resolve_topology_policy(mode: str) -> str:
    if _validate_surface_mode_name(mode) == SINGLE_SURFACE:
        return TOPOLOGY_POLICY_UNSUPPORTED
    return TOPOLOGY_POLICY_SEARCH_GATE


def resolve_production_support_level(mode: str) -> str:
    resolved_mode = _validate_surface_mode_name(mode)
    if resolved_mode == SINGLE_SURFACE:
        return PRODUCTION_SUPPORT_LEVEL_PRODUCTION
    if resolved_mode == PUBLISHED_MULTISURFACE:
        return PRODUCTION_SUPPORT_LEVEL_PUBLISHED_V1
    return PRODUCTION_SUPPORT_LEVEL_EXPERIMENTAL


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
    published_label_fractions: tuple[float, ...] | None = None,
    published_surface_preset: str | None = None,
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

    # The published-stack depth knobs are opt-in and only meaningful for the
    # published multisurface contract. Resolving (and validating) them here once
    # keeps the label fractions / weights / physics contract single-source. For
    # any other mode they must be unset so a misrouted knob fails closed rather
    # than silently changing a single-surface or experimental run.
    if mode == PUBLISHED_MULTISURFACE:
        effective_published_fractions = resolve_published_label_fractions(
            published_label_fractions=published_label_fractions,
            published_surface_preset=published_surface_preset,
        )
    else:
        if published_label_fractions is not None or (
            published_surface_preset is not None
            and published_surface_preset != PUBLISHED_PRESET_DEFAULT_V1
        ):
            raise ValueError(
                "published surface-depth knobs only apply to "
                f"surface mode {PUBLISHED_MULTISURFACE!r}; received them for "
                f"{mode!r}."
            )
        effective_published_fractions = None

    return SurfaceModeContract(
        mode=mode,
        version=SURFACE_MODE_VERSION_V1,
        source=source,
        label_fractions=resolve_surface_label_fractions(
            mode,
            legacy_inner_surface_ratio=legacy_ratio,
            published_label_fractions=effective_published_fractions,
        ),
        weights=resolve_surface_weights(
            mode,
            published_label_fractions=effective_published_fractions,
        ),
        stack_policy=resolve_surface_stack_policy(mode),
        physics_contract=resolve_surface_physics_contract(
            mode,
            published_label_fractions=effective_published_fractions,
        ),
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
        requires_inner_surface_ratio=resolve_surface_mode_requires_inner_surface_ratio(
            mode
        ),
        surface_count_policy=resolve_surface_count_policy(mode),
        final_refinement_policy=resolve_final_refinement_policy(mode),
        current_policy=resolve_current_policy(mode),
        topology_policy=resolve_topology_policy(mode),
        telemetry_schema_version=SURFACE_MODE_TELEMETRY_SCHEMA_VERSION_V1,
        production_support_level=resolve_production_support_level(mode),
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
        "REQUIRES_INNER_SURFACE_RATIO": contract.requires_inner_surface_ratio,
        "SURFACE_COUNT_POLICY": contract.surface_count_policy,
        "FINAL_REFINEMENT_POLICY": contract.final_refinement_policy,
        "CURRENT_POLICY": contract.current_policy,
        "TOPOLOGY_POLICY": contract.topology_policy,
        "SURFACE_MODE_TELEMETRY_SCHEMA_VERSION": contract.telemetry_schema_version,
        "PRODUCTION_SUPPORT_LEVEL": contract.production_support_level,
    }


def surface_mode_surface_names(contract: SurfaceModeContract) -> tuple[str, ...]:
    resolved_mode = _validate_surface_mode_name(contract.mode)
    if resolved_mode == SINGLE_SURFACE:
        names = _SINGLE_SURFACE_NAMES
    elif resolved_mode == PUBLISHED_MULTISURFACE:
        names = published_surface_names(contract.num_surfaces)
    else:
        names = _EXPERIMENTAL_SURFACE_NAMES
    if len(names) != contract.num_surfaces:
        raise ValueError(
            f"Surface mode {contract.mode!r} defines {len(names)} names for "
            f"{contract.num_surfaces} label fractions."
        )
    return names


def _resolve_mode_name(mode_or_contract: str | SurfaceModeContract) -> str:
    if isinstance(mode_or_contract, SurfaceModeContract):
        return mode_or_contract.mode
    return _validate_surface_mode_name(str(mode_or_contract))


def surface_mode_supports_alm(mode_or_contract: str | SurfaceModeContract) -> bool:
    return _resolve_mode_name(mode_or_contract) in {
        SINGLE_SURFACE,
        PUBLISHED_MULTISURFACE,
        EXPERIMENTAL_MULTISURFACE,
    }


def surface_mode_supports_boozer_stage_refinement(
    mode_or_contract: str | SurfaceModeContract,
) -> bool:
    if isinstance(mode_or_contract, SurfaceModeContract):
        return (
            not _surface_mode_runtime_support_errors(mode_or_contract)
            and mode_or_contract.final_refinement_policy
            == FINAL_REFINEMENT_POLICY_BOOZER_STAGE
        )
    return (
        resolve_final_refinement_policy(_resolve_mode_name(mode_or_contract))
        == FINAL_REFINEMENT_POLICY_BOOZER_STAGE
    )


def surface_mode_supports_topology_gate(
    mode_or_contract: str | SurfaceModeContract,
) -> bool:
    if isinstance(mode_or_contract, SurfaceModeContract):
        return (
            not _surface_mode_runtime_support_errors(mode_or_contract)
            and mode_or_contract.topology_policy == TOPOLOGY_POLICY_SEARCH_GATE
        )
    return (
        resolve_topology_policy(_resolve_mode_name(mode_or_contract))
        == TOPOLOGY_POLICY_SEARCH_GATE
    )


def _surface_mode_runtime_support_errors(contract: SurfaceModeContract) -> list[str]:
    mode = _validate_surface_mode_name(contract.mode)
    errors: list[str] = []

    # The published stack is opt-in configurable in depth, so its expected
    # shape (count, weights, physics contract) is derived from the contract's
    # own label fractions rather than the fixed default. For every other mode
    # the expected shape comes from the mode resolvers exactly as before, so a
    # contract whose fractions/weights drift from the mode default still fails
    # closed. A published contract whose fractions are not a valid strictly
    # increasing outer-anchored nested stack also fails closed below.
    if mode == PUBLISHED_MULTISURFACE:
        try:
            published_fractions = validate_published_label_fractions(
                contract.label_fractions
            )
        except ValueError as exc:
            # Record the root nested-stack violation and derive the expected
            # shape from the contract's own fields so this validator stays
            # non-raising; the appended error already fails the contract closed.
            errors.append(f"published label_fractions invalid: {exc}")
            expected_weights = contract.weights
            expected_physics_contract = contract.physics_contract
        else:
            expected_weights = resolve_surface_weights(
                mode, published_label_fractions=published_fractions
            )
            expected_physics_contract = resolve_surface_physics_contract(
                mode, published_label_fractions=published_fractions
            )
    else:
        expected_weights = resolve_surface_weights(mode)
        expected_physics_contract = resolve_surface_physics_contract(mode)
    expected_surface_count = len(expected_weights)

    if contract.version != SURFACE_MODE_VERSION_V1:
        errors.append(
            f"version={contract.version!r} is not {SURFACE_MODE_VERSION_V1!r}"
        )
    if contract.num_surfaces != expected_surface_count:
        errors.append(
            f"num_surfaces={contract.num_surfaces!r} does not match "
            f"{expected_surface_count!r} for {mode!r}"
        )
    if len(contract.weights) != contract.num_surfaces:
        errors.append(
            f"weights length {len(contract.weights)!r} does not match "
            f"num_surfaces={contract.num_surfaces!r}"
        )
    if contract.weights != expected_weights:
        errors.append(
            f"weights={contract.weights!r} does not match "
            f"{expected_weights!r}"
        )
    if contract.stack_policy != resolve_surface_stack_policy(mode):
        errors.append(
            f"stack_policy={contract.stack_policy!r} does not match "
            f"{resolve_surface_stack_policy(mode)!r}"
        )
    if contract.physics_contract != expected_physics_contract:
        errors.append("physics_contract does not match the selected surface mode")
    if (
        contract.requires_inner_surface_ratio
        != resolve_surface_mode_requires_inner_surface_ratio(mode)
    ):
        errors.append(
            "requires_inner_surface_ratio does not match the selected surface mode"
        )
    if contract.surface_count_policy != resolve_surface_count_policy(mode):
        errors.append(
            f"surface_count_policy={contract.surface_count_policy!r} does not "
            f"match {resolve_surface_count_policy(mode)!r}"
        )
    if contract.final_refinement_policy != resolve_final_refinement_policy(mode):
        errors.append(
            f"final_refinement_policy={contract.final_refinement_policy!r} does "
            f"not match {resolve_final_refinement_policy(mode)!r}"
        )
    if contract.current_policy != resolve_current_policy(mode):
        errors.append(
            f"current_policy={contract.current_policy!r} does not match "
            f"{resolve_current_policy(mode)!r}"
        )
    if contract.topology_policy != resolve_topology_policy(mode):
        errors.append(
            f"topology_policy={contract.topology_policy!r} does not match "
            f"{resolve_topology_policy(mode)!r}"
        )
    if contract.telemetry_schema_version != SURFACE_MODE_TELEMETRY_SCHEMA_VERSION_V1:
        errors.append(
            "telemetry_schema_version does not match "
            f"{SURFACE_MODE_TELEMETRY_SCHEMA_VERSION_V1!r}"
        )
    if contract.production_support_level != resolve_production_support_level(mode):
        errors.append(
            f"production_support_level={contract.production_support_level!r} "
            f"does not match {resolve_production_support_level(mode)!r}"
        )

    if mode == EXPERIMENTAL_MULTISURFACE:
        if contract.num_surfaces == 2:
            inner_fraction, outer_fraction = contract.label_fractions
            if not (0.0 < float(inner_fraction) < 1.0):
                errors.append(
                    "experimental_multisurface requires an inner label fraction "
                    "between 0 and 1"
                )
            if float(outer_fraction) != 1.0:
                errors.append(
                    "experimental_multisurface requires the outer label fraction "
                    "to be 1.0"
                )
    elif mode == PUBLISHED_MULTISURFACE:
        # Published label fractions are configurable; their strictly increasing
        # outer-anchored nested-stack validity is enforced above via
        # validate_published_label_fractions, so no fixed-default equality
        # check is applied here.
        pass
    else:
        expected_label_fractions = resolve_surface_label_fractions(
            mode,
            legacy_inner_surface_ratio=None,
        )
        if contract.label_fractions != expected_label_fractions:
            errors.append(
                f"label_fractions={contract.label_fractions!r} does not match "
                f"{expected_label_fractions!r}"
            )

    if not errors:
        surface_mode_surface_names(contract)
    return errors


def surface_mode_runtime_supported(
    mode_or_contract: str | SurfaceModeContract,
) -> bool:
    if not isinstance(mode_or_contract, SurfaceModeContract):
        _resolve_mode_name(mode_or_contract)
        return True
    return not _surface_mode_runtime_support_errors(mode_or_contract)


def validate_surface_mode_runtime_support(contract: SurfaceModeContract) -> None:
    errors = _surface_mode_runtime_support_errors(contract)
    if errors:
        raise ValueError(
            "Unsupported surface-mode runtime contract for "
            f"{contract.mode!r}: " + "; ".join(errors)
        )
