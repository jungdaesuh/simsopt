from __future__ import annotations

from types import MappingProxyType

AlmCliField = tuple[str, type, int | float]

# Single-stage and baseline-sweep ALM defaults. Stage 2 derives its command,
# artifact-path, parser, and config defaults from this tuple plus explicit
# per-regime overrides below.
SINGLE_STAGE_ALM_CLI_FIELDS: tuple[AlmCliField, ...] = (
    ("max_outer_iters", int, 10),
    ("penalty_init", float, 1.0),
    ("penalty_scale", float, 10.0),
    ("penalty_max", float, 1.0e8),
    ("feas_tol", float, 1.0e-6),
    ("stationarity_tol", float, 1.0e-6),
    ("trust_radius_init", float, 0.05),
    ("trust_radius_min", float, 1.0e-4),
    ("trust_radius_shrink", float, 0.5),
    ("trust_radius_grow", float, 1.5),
    ("max_inner_attempts", int, 4),
    ("max_subproblem_continuations", int, 20),
    ("distance_smoothing", float, 0.005),
    ("curvature_smoothing", float, 0.05),
)

STAGE2_ALM_DEFAULT_OVERRIDES = MappingProxyType(
    {
        "penalty_init": 0.1,
        "penalty_scale": 2.0,
        "curvature_smoothing": 0.25,
    }
)
STAGE2_ALM_CLI_FIELDS: tuple[AlmCliField, ...] = tuple(
    (
        suffix,
        value_type,
        STAGE2_ALM_DEFAULT_OVERRIDES.get(suffix, default),
    )
    for suffix, value_type, default in SINGLE_STAGE_ALM_CLI_FIELDS
)
SINGLE_STAGE_ALM_DEFAULTS = MappingProxyType(
    {suffix: default for suffix, _value_type, default in SINGLE_STAGE_ALM_CLI_FIELDS}
)
STAGE2_ALM_DEFAULTS = MappingProxyType(
    {suffix: default for suffix, _value_type, default in STAGE2_ALM_CLI_FIELDS}
)


def single_stage_alm_default(suffix: str) -> int | float:
    return SINGLE_STAGE_ALM_DEFAULTS[suffix]


def stage2_alm_default(suffix: str) -> int | float:
    return STAGE2_ALM_DEFAULTS[suffix]
