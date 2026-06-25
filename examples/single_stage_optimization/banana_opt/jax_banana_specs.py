from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Callable, Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.core.banana import (
    BananaObjectiveSpec,
    make_banana_objective_spec,
    make_banana_system_spec,
)
from simsopt_jax.core.field import grouped_coil_set_spec_from_coil_specs
from simsopt_jax.core.objectives_flux import fixed_surface_flux_specs_from_surface
from simsopt_jax.core.specs import CurveSpec, make_coil_symmetry_spec
from simsopt_jax_adapters.field._coil_graph import (
    _unwrap_coil_curve_and_current_objects,
)
from simsopt_jax_adapters.geo.surface_objectives import (
    _surface_spec_from_surface,
    make_traceable_solved_state_value_and_grad,
)
from simsopt_jax_adapters.objectives.stage2_target import _coil_spec_from_coil

from .jax_banana_types import (
    BANANA_IDX,
    BANANA_TARGETS,
    HARDWARE_LIMITS,
    HBT_BANANA_WS,
    KA_TO_A,
    N_BANANA,
    PROXY_IDX,
    SingleStageWeights,
    Stage2Weights,
    TF_IDX,
    VF_IDX,
)

BananaStage = Literal["stage2", "single_stage"]
BananaStageWeights = Stage2Weights | SingleStageWeights


class BananaSpecCurve(Protocol):
    """Host curve contract needed to freeze one CWS banana owner."""

    order: int

    def get_dofs(self) -> object: ...

    def to_spec(self) -> CurveSpec: ...


class BananaSpecCurrent(Protocol):
    """Host current contract after unwrapping scale-only current wrappers."""

    @property
    def local_x(self) -> object: ...

    @property
    def local_full_x(self) -> object: ...


class BananaSpecCoil(Protocol):
    """Host coil contract consumed by the banana frozen-spec builder."""

    curve: object
    current: object


class BananaSpecBiotSavart(Protocol):
    """Host Biot-Savart contract for slicing the banana coil block."""

    coils: Sequence[BananaSpecCoil]
    x: object


class BananaSpecSurface(Protocol):
    """Host surface contract needed for frozen coil-surface penalties."""

    def gamma(self) -> object: ...

    def normal(self) -> object: ...


class BananaSpecBoozerSurface(Protocol):
    """Host Boozer contract needed to snapshot explicit solved-state inputs."""

    def get_adjoint_runtime_state(self) -> object: ...

    def _pack_decision_vector(
        self,
        iota: object,
        G: object,
        *,
        sdofs: object,
    ) -> object: ...


class BananaSingleStageSolvedStateTarget(NamedTuple):
    """Compiled single-stage target lane plus explicit solved Boozer inputs."""

    value_and_grad: Callable[[object, object, object], tuple[object, object]]
    coil_dofs: jax.Array
    solved_x: jax.Array
    linear_solve_factors: object


class _SurfaceSpecProvider:
    def __init__(self, surface_spec: object):
        self._surface_spec = surface_spec

    def surface_spec(self) -> object:
        return self._surface_spec


def _as_float64_vector(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape((-1,))


def _current_partition(current: BananaSpecCurrent) -> tuple[int, np.ndarray, np.ndarray]:
    free_current = _as_float64_vector(current.local_x)
    full_current = _as_float64_vector(current.local_full_x)
    if full_current.size != 1:
        raise ValueError("banana specs require one scalar base current DOF")
    if free_current.size not in (0, 1):
        raise ValueError("banana specs support 0 or 1 free base current DOF")
    if free_current.size == 1:
        return 1, np.asarray((), dtype=np.float64), free_current
    return 0, full_current, free_current


def _stage_weight(weight_source: BananaStageWeights, name: str) -> float:
    return float(getattr(weight_source, name))


def _surface_geometry(
    surface: BananaSpecSurface | None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if surface is None:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty, 0, 0
    gamma = np.asarray(surface.gamma(), dtype=np.float64)
    normal = np.asarray(surface.normal(), dtype=np.float64)
    surface_nphi = int(gamma.shape[0]) if gamma.ndim >= 3 else 0
    surface_ntheta = int(gamma.shape[1]) if gamma.ndim >= 3 else 0
    return (
        gamma.reshape((-1, 3)),
        normal.reshape((-1, 3)),
        surface_nphi,
        surface_ntheta,
    )


def _fixed_surface_flux_spec(surface: BananaSpecSurface, definition: str):
    _field_eval_spec, flux_spec = fixed_surface_flux_specs_from_surface(
        _SurfaceSpecProvider(_surface_spec_from_surface(surface)),
        definition=definition,
    )
    return flux_spec


def _require_zero_boozer_I(boozer_I: float) -> None:
    if float(boozer_I) != 0.0:
        raise ValueError(
            "JAX banana single-stage solved-state target only supports "
            "zero enclosed Boozer current (boozer_I == 0.0); finite-I Boozer "
            "requires the G_eff = G + iota * I residual and adjoint chain-rule port."
        )


def build_banana_single_stage_solved_state_target(
    boozer_surface: BananaSpecBoozerSurface,
    biotsavart: BananaSpecBiotSavart,
    iota_target: object,
    *,
    outer_objective_config: object,
    success_filter: object = None,
    boozer_I: float = 0.0,
) -> BananaSingleStageSolvedStateTarget:
    """Freeze the single-stage Boozer target lane behind explicit solved state.

    The returned callable evaluates the Boozer residual, iota target penalty,
    non-quasisymmetry ratio, and configured single-stage hardware terms from
    ``(coil_dofs, solved_x, linear_solve_factors)``. Boozer solving remains a
    separate host/device step; this helper only packages the already-solved
    state needed by the compiled value-and-gradient kernel.
    """

    _require_zero_boozer_I(boozer_I)
    adjoint_state = boozer_surface.get_adjoint_runtime_state()
    solved_state = adjoint_state.solved_state
    solved_x = boozer_surface._pack_decision_vector(
        solved_state.iota,
        solved_state.G,
        sdofs=solved_state.sdofs,
    )
    return BananaSingleStageSolvedStateTarget(
        value_and_grad=make_traceable_solved_state_value_and_grad(
            boozer_surface,
            biotsavart,
            iota_target,
            outer_objective_config=outer_objective_config,
            success_filter=success_filter,
        ),
        coil_dofs=jnp.asarray(np.asarray(biotsavart.x.copy(), dtype=np.float64)),
        solved_x=jnp.asarray(solved_x, dtype=jnp.float64),
        linear_solve_factors=adjoint_state.linear_solve_factors,
    )


def _banana_symmetry_specs_from_coils(
    banana_coils: Sequence[BananaSpecCoil],
    *,
    base_curve: object,
    base_current: object,
) -> tuple[object, ...]:
    symmetry_specs = []
    for coil in banana_coils:
        curve, rotmat, current, scale = _unwrap_coil_curve_and_current_objects(
            coil.curve,
            coil.current,
        )
        if curve is not base_curve:
            raise ValueError("banana symmetry coils must share one base curve")
        if current is not base_current:
            raise ValueError("banana symmetry coils must share one base current")
        symmetry_specs.append(make_coil_symmetry_spec(rotmat=rotmat, scale=scale))
    return tuple(symmetry_specs)


def build_banana_local_objective_spec_from_coils(
    banana_coils: Sequence[BananaSpecCoil],
    *,
    stage: BananaStage,
    weights: BananaStageWeights,
    tf_current: BananaSpecCurrent | None = None,
    tf_current_scale: float = 0.0,
    surface: BananaSpecSurface | None = None,
    include_current_penalty: bool,
    include_min_length: bool,
    include_width: bool,
    include_coil_surface_distance: bool | None = None,
    include_squared_flux: bool = False,
    fixed_flux_coils: Sequence[BananaSpecCoil] = (),
    fixed_auxiliary_coil_roles: Sequence[str] = (),
    squared_flux_definition: str = "normalized",
) -> tuple[BananaObjectiveSpec, np.ndarray]:
    """Freeze banana local geometry/current terms into a spec and decision vector."""

    if not banana_coils:
        raise ValueError("banana specs require at least one banana coil")
    include_coil_surface_term = (
        surface is not None
        if include_coil_surface_distance is None
        else bool(include_coil_surface_distance)
    )
    if include_coil_surface_term and surface is None:
        raise ValueError("coil-surface banana specs require a surface")
    if include_squared_flux and surface is None:
        raise ValueError("squared-flux banana specs require a surface")

    base_curve, _rotmat, base_current, _scale = _unwrap_coil_curve_and_current_objects(
        banana_coils[0].curve,
        banana_coils[0].current,
    )
    curve = base_curve
    current = base_current
    curve_dofs = _as_float64_vector(curve.get_dofs())
    current_dof_count, fixed_current_dofs, current_dofs = _current_partition(current)
    tf_current_dof_count = 0
    fixed_tf_current_dofs = np.asarray((), dtype=np.float64)
    tf_current_dofs = np.asarray((), dtype=np.float64)
    if tf_current is not None:
        (
            tf_current_dof_count,
            fixed_tf_current_dofs,
            tf_current_dofs,
        ) = _current_partition(tf_current)
    system_spec = make_banana_system_spec(
        curve=curve.to_spec(),
        symmetries=_banana_symmetry_specs_from_coils(
            banana_coils,
            base_curve=base_curve,
            base_current=base_current,
        ),
        fixed_current_dofs=fixed_current_dofs,
        current_dof_count=current_dof_count,
        fixed_tf_current_dofs=fixed_tf_current_dofs,
        tf_current_dof_count=tf_current_dof_count,
        tf_current_scale=tf_current_scale,
        curve_dof_count=int(curve_dofs.size),
    )
    decision_vector = np.concatenate((tf_current_dofs, current_dofs, curve_dofs), axis=0)
    surface_gamma, surface_normal, surface_nphi, surface_ntheta = _surface_geometry(
        surface
    )
    coil_surface_distance_weight = _stage_weight(weights, "csdist")
    current_weight = _stage_weight(weights, "currents")
    fixed_flux_coil_set = grouped_coil_set_spec_from_coil_specs(
        tuple(_coil_spec_from_coil(coil) for coil in fixed_flux_coils)
    )
    fixed_auxiliary_coil_set = grouped_coil_set_spec_from_coil_specs(
        tuple(_coil_spec_from_coil(coil) for coil in fixed_flux_coils)
    )
    auxiliary_roles = tuple(str(role) for role in fixed_auxiliary_coil_roles)
    if not auxiliary_roles and fixed_flux_coils:
        auxiliary_roles = tuple("auxiliary" for _coil in fixed_flux_coils)
    if len(auxiliary_roles) != len(fixed_flux_coils):
        raise ValueError("fixed auxiliary coil roles must match fixed flux coils")
    flux_spec = (
        _fixed_surface_flux_spec(surface, squared_flux_definition)
        if include_squared_flux and surface is not None
        else None
    )
    system_spec = replace(
        system_spec,
        fixed_auxiliary_coil_set=fixed_auxiliary_coil_set,
        fixed_auxiliary_coil_roles=auxiliary_roles,
    )
    objective_spec = make_banana_objective_spec(
        system=system_spec,
        stage=stage,
        length_max_threshold=HARDWARE_LIMITS.max_length,
        length_min_threshold=0.5 * HARDWARE_LIMITS.max_length,
        curvature_threshold=HARDWARE_LIMITS.max_curvature,
        curvature_p=HARDWARE_LIMITS.banana_curv_p,
        poloidal_major_radius=HBT_BANANA_WS.major_radius,
        poloidal_theta_target=np.deg2rad(BANANA_TARGETS.poloidal_deg),
        width_major_radius=HBT_BANANA_WS.major_radius,
        width_minor_radius=HBT_BANANA_WS.minor_radius,
        width_max_threshold=BANANA_TARGETS.width_max,
        width_min_threshold=BANANA_TARGETS.width_min,
        self_distance_minimum=BANANA_TARGETS.self_distance_min,
        coil_coil_distance_minimum=HARDWARE_LIMITS.min_ccdist,
        coil_surface_distance_minimum=HARDWARE_LIMITS.min_csdist,
        tf_current_max_threshold=max(
            abs(limit) for limit in HARDWARE_LIMITS.tf_current_ka_limits
        )
        * KA_TO_A,
        banana_current_max_threshold=max(
            abs(limit) for limit in HARDWARE_LIMITS.banana_current_ka_limits
        )
        * KA_TO_A,
        squared_flux_weight=float(getattr(weights, "sqflux", 0.0)),
        length_weight=_stage_weight(weights, "length"),
        curvature_weight=_stage_weight(weights, "curvature"),
        poloidal_weight=_stage_weight(weights, "poloidal"),
        width_weight=_stage_weight(weights, "width"),
        self_distance_weight=_stage_weight(weights, "selfint"),
        coil_coil_distance_weight=_stage_weight(weights, "ccdist"),
        coil_surface_distance_weight=coil_surface_distance_weight,
        tf_current_weight=current_weight,
        banana_current_weight=current_weight,
        surface_gamma=surface_gamma,
        surface_normal=surface_normal,
        surface_nphi=surface_nphi,
        surface_ntheta=surface_ntheta,
        fixed_flux_coil_set=fixed_flux_coil_set,
        flux=flux_spec,
        self_distance_neighbor_skip=max(3, int(1.5 * int(curve.order))),
        include_min_length=include_min_length,
        include_width=include_width,
        include_coil_surface_distance=include_coil_surface_term
        and coil_surface_distance_weight != 0.0,
        include_squared_flux=include_squared_flux,
        include_tf_current_penalty=include_current_penalty and tf_current is not None,
        include_current_penalty=include_current_penalty,
    )
    return objective_spec, decision_vector


def build_banana_local_objective_spec_from_biotsavart(
    biotsavart: BananaSpecBiotSavart,
    *,
    stage: BananaStage,
    weights: BananaStageWeights,
    surface: BananaSpecSurface | None = None,
    include_current_penalty: bool,
    include_min_length: bool,
    include_width: bool,
    include_coil_surface_distance: bool | None = None,
    include_squared_flux: bool = False,
    squared_flux_definition: str = "normalized",
) -> tuple[BananaObjectiveSpec, np.ndarray]:
    """Freeze the canonical banana coil block from a host Biot-Savart object."""

    banana_coils = biotsavart.coils[BANANA_IDX : BANANA_IDX + N_BANANA]
    if len(banana_coils) != N_BANANA:
        raise ValueError("canonical banana Biot-Savart block is incomplete")
    fixed_flux_coils = (
        tuple(biotsavart.coils[:BANANA_IDX])
        + tuple(biotsavart.coils[BANANA_IDX + N_BANANA :])
        if include_squared_flux
        else ()
    )
    fixed_auxiliary_coil_roles = (
        tuple("tf" for _coil in biotsavart.coils[:BANANA_IDX])
        + tuple("proxy" for _coil in biotsavart.coils[PROXY_IDX:VF_IDX])
        + tuple("vf" for _coil in biotsavart.coils[VF_IDX:])
        if include_squared_flux
        else ()
    )
    _tf_curve, _tf_rotmat, tf_current, tf_current_scale = (
        _unwrap_coil_curve_and_current_objects(
            biotsavart.coils[TF_IDX].curve,
            biotsavart.coils[TF_IDX].current,
        )
    )
    return build_banana_local_objective_spec_from_coils(
        banana_coils,
        stage=stage,
        weights=weights,
        tf_current=tf_current,
        tf_current_scale=tf_current_scale,
        surface=surface,
        include_current_penalty=include_current_penalty,
        include_min_length=include_min_length,
        include_width=include_width,
        include_coil_surface_distance=include_coil_surface_distance,
        include_squared_flux=include_squared_flux,
        fixed_flux_coils=fixed_flux_coils,
        fixed_auxiliary_coil_roles=fixed_auxiliary_coil_roles,
        squared_flux_definition=squared_flux_definition,
    )
