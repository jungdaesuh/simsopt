"""Pure JAX banana-coil specs and local objective kernels.

The module owns the frozen-spec boundary for banana geometry and local
hardware-style penalties. Host loaders and SIMSOPT ``Optimizable`` objects
should build these specs, then compiled code should consume only a spec plus a
flat decision vector.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._pairwise_reductions import (
    _chunk_rows,
    _masked_pairwise_distances,
    _resolve_pairwise_penalty_chunk_size,
    _use_dense_pairwise_path,
)

from ._math_utils import as_jax_float64 as _as_float64_array
from .curve_geometry import curve_geometry_from_dofs
from .curve_kernels import (
    curve_curve_distance_penalty_pure,
    curve_length_from_incremental_arclength_pure,
    curve_surface_distance_penalty_pure,
    curvature_p_norm_from_kappa_pure as banana_curvature_p_norm_from_kappa_pure,
    incremental_arclength_pure,
    kappa_pure,
)
from .field import grouped_biot_savart_B_from_spec, grouped_coil_set_spec_from_lists
from .objectives_flux import fixed_surface_flux_integral_from_B
from .specs import (
    CoilSymmetrySpec,
    CurveSpec,
    FixedSurfaceFluxSpec,
    GroupedCoilSetSpec,
    _register_jax_spec,
    apply_coil_symmetry,
    make_fixed_surface_flux_spec,
    make_grouped_coil_set_spec,
)

__all__ = [
    "BananaBoozerSolveSpec",
    "BananaDecision",
    "BananaDecisionSpec",
    "BananaGeometry",
    "BananaLocalTerms",
    "BananaObjectiveSpec",
    "BananaQuadratureSpec",
    "BananaSystemSpec",
    "banana_coil_coil_distance_pure",
    "banana_coil_surface_distance_pure",
    "banana_hardware_keepout_point_cloud_curve_pure",
    "banana_hardware_keepout_point_cloud_pure",
    "banana_current_magnitude_penalty",
    "banana_curve_length_pure",
    "banana_curvature_p_norm_from_kappa_pure",
    "banana_curvature_p_norm_pure",
    "banana_decision_from_dofs",
    "banana_geometry_from_dofs",
    "banana_local_terms",
    "banana_local_value",
    "banana_local_value_and_grad",
    "banana_lp_abs_hinge_pure",
    "banana_pack_rotation_fold_pure",
    "banana_pack_twist_strain_pure",
    "banana_poloidal_extent_pure",
    "banana_projected_reach_pure",
    "banana_projected_ellipse_width_pure",
    "banana_quadratic_penalty",
    "banana_rotation_aware_curvature_excess_pure",
    "banana_self_distance_mask",
    "banana_self_distance_pure",
    "banana_stage2_terms",
    "banana_stage2_value",
    "banana_stage2_value_and_grad",
    "banana_symmetry_geometry_from_arrays",
    "banana_swept_channel_surface_points",
    "banana_torsional_strain_pure",
    "make_banana_objective_spec",
    "make_banana_system_spec",
]

PenaltyMode = Literal["identity", "max", "min", "two-sided"]

_SWEPT_SURFACE_QUADRATURE = np.array(
    [
        *(
            (sweep, width, depth)
            for sweep in (-1.0, 1.0)
            for width in (-1.0, 1.0)
            for depth in (-1.0, 1.0)
        ),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        *(
            (sweep, width, 0.0)
            for sweep in (-1.0, 1.0)
            for width in (-1.0, 1.0)
        ),
        *(
            (sweep, 0.0, depth)
            for sweep in (-1.0, 1.0)
            for depth in (-1.0, 1.0)
        ),
        *(
            (0.0, width, depth)
            for width in (-1.0, 1.0)
            for depth in (-1.0, 1.0)
        ),
    ],
    dtype=np.float64,
)


class BananaDecision(NamedTuple):
    """Canonical split of the banana decision vector."""

    tf_current_dofs: jax.Array
    current_dofs: jax.Array
    curve_dofs: jax.Array
    flat_dofs: jax.Array


class BananaGeometry(NamedTuple):
    """Base and symmetry-expanded banana geometry for one optimizer state."""

    base_gamma: jax.Array
    base_gammadash: jax.Array
    base_gammadashdash: jax.Array
    gammas: jax.Array
    gammadashs: jax.Array
    currents: jax.Array
    tf_current_dof: jax.Array
    tf_current: jax.Array
    base_current_dof: jax.Array


class BananaLocalTerms(NamedTuple):
    """Stable local banana term order before applying term weights."""

    squared_flux: jax.Array
    length_max: jax.Array
    length_min: jax.Array
    curvature: jax.Array
    poloidal_extent: jax.Array
    width_max: jax.Array
    width_min: jax.Array
    self_distance: jax.Array
    coil_coil_distance: jax.Array
    coil_surface_distance: jax.Array
    tf_current_max: jax.Array
    banana_current_max: jax.Array


@_register_jax_spec(
    data_fields=(),
    meta_fields=("current_dof_count", "curve_dof_count", "tf_current_dof_count"),
)
class BananaDecisionSpec:
    """Static partition for ``[tf_current?, banana_current?, curve_dofs...]`` vectors."""

    current_dof_count: int
    curve_dof_count: int
    tf_current_dof_count: int = 0


@_register_jax_spec(
    data_fields=("iota", "G"),
    meta_fields=(
        "enabled",
        "solver",
        "mpol",
        "ntor",
        "nfp",
        "nphi",
        "ntheta",
    ),
)
class BananaBoozerSolveSpec:
    """Explicit optional Boozer state/config carried outside mutable objects."""

    iota: jax.Array
    G: jax.Array
    enabled: bool
    solver: str
    mpol: int
    ntor: int
    nfp: int
    nphi: int
    ntheta: int


@_register_jax_spec(
    data_fields=(),
    meta_fields=(
        "curve_point_count",
        "surface_sample_count",
        "surface_nphi",
        "surface_ntheta",
        "flux_point_count",
        "flux_nphi",
        "flux_ntheta",
    ),
)
class BananaQuadratureSpec:
    """Static sample counts that define compiled banana objective shapes."""

    curve_point_count: int
    surface_sample_count: int
    surface_nphi: int
    surface_ntheta: int
    flux_point_count: int
    flux_nphi: int
    flux_ntheta: int


@_register_jax_spec(
    data_fields=(
        "curve",
        "symmetries",
        "fixed_current_dofs",
        "fixed_tf_current_dofs",
        "tf_current_scale",
        "fixed_auxiliary_coil_set",
        "boozer",
    ),
    meta_fields=("decision", "fixed_auxiliary_coil_roles"),
)
class BananaSystemSpec:
    """Frozen banana geometry state for local banana kernels."""

    curve: CurveSpec
    symmetries: tuple[CoilSymmetrySpec, ...]
    fixed_current_dofs: jax.Array
    fixed_tf_current_dofs: jax.Array
    tf_current_scale: jax.Array
    fixed_auxiliary_coil_set: GroupedCoilSetSpec
    boozer: BananaBoozerSolveSpec
    decision: BananaDecisionSpec
    fixed_auxiliary_coil_roles: tuple[str, ...]


@_register_jax_spec(
    data_fields=(
        "system",
        "self_distance_mask",
        "surface_gamma",
        "surface_normal",
        "fixed_flux_coil_set",
        "flux",
    ),
    meta_fields=(
        "quadrature",
        "stage",
        "length_max_mode",
        "length_max_threshold",
        "length_min_mode",
        "length_min_threshold",
        "curvature_threshold",
        "curvature_p",
        "poloidal_major_radius",
        "poloidal_z_position",
        "poloidal_theta_target",
        "poloidal_exponent",
        "width_major_radius",
        "width_minor_radius",
        "width_z_position",
        "width_scale",
        "width_epsilon",
        "width_max_mode",
        "width_max_threshold",
        "width_min_mode",
        "width_min_threshold",
        "self_distance_minimum",
        "self_distance_normalize",
        "coil_coil_distance_minimum",
        "coil_surface_distance_minimum",
        "tf_current_max_mode",
        "tf_current_max_threshold",
        "banana_current_max_mode",
        "banana_current_max_threshold",
        "include_min_length",
        "include_width",
        "include_coil_surface_distance",
        "include_squared_flux",
        "include_tf_current_penalty",
        "include_current_penalty",
        "squared_flux_weight",
        "length_weight",
        "curvature_weight",
        "poloidal_weight",
        "width_weight",
        "self_distance_weight",
        "coil_coil_distance_weight",
        "coil_surface_distance_weight",
        "tf_current_weight",
        "banana_current_weight",
    ),
)
class BananaObjectiveSpec:
    """Frozen local banana objective config with static weights and gates."""

    system: BananaSystemSpec
    self_distance_mask: jax.Array
    surface_gamma: jax.Array
    surface_normal: jax.Array
    fixed_flux_coil_set: GroupedCoilSetSpec
    flux: FixedSurfaceFluxSpec
    quadrature: BananaQuadratureSpec
    stage: str
    length_max_mode: PenaltyMode
    length_max_threshold: float
    length_min_mode: PenaltyMode
    length_min_threshold: float
    curvature_threshold: float
    curvature_p: int
    poloidal_major_radius: float
    poloidal_z_position: float
    poloidal_theta_target: float
    poloidal_exponent: int
    width_major_radius: float
    width_minor_radius: float
    width_z_position: float
    width_scale: float
    width_epsilon: float
    width_max_mode: PenaltyMode
    width_max_threshold: float
    width_min_mode: PenaltyMode
    width_min_threshold: float
    self_distance_minimum: float
    self_distance_normalize: bool
    coil_coil_distance_minimum: float
    coil_surface_distance_minimum: float
    tf_current_max_mode: PenaltyMode
    tf_current_max_threshold: float
    banana_current_max_mode: PenaltyMode
    banana_current_max_threshold: float
    include_min_length: bool
    include_width: bool
    include_coil_surface_distance: bool
    include_squared_flux: bool
    include_tf_current_penalty: bool
    include_current_penalty: bool
    squared_flux_weight: float
    length_weight: float
    curvature_weight: float
    poloidal_weight: float
    width_weight: float
    self_distance_weight: float
    coil_coil_distance_weight: float
    coil_surface_distance_weight: float
    tf_current_weight: float
    banana_current_weight: float


def make_banana_system_spec(
    *,
    curve: CurveSpec,
    symmetries: object,
    fixed_current_dofs: object = (0.0,),
    current_dof_count: Literal[0, 1] = 1,
    fixed_tf_current_dofs: object = (),
    tf_current_dof_count: Literal[0, 1] = 0,
    tf_current_scale: object = 0.0,
    fixed_auxiliary_coil_set: GroupedCoilSetSpec | None = None,
    fixed_auxiliary_coil_roles: object = (),
    boozer_iota: object = (),
    boozer_G: object = (),
    boozer_enabled: bool = False,
    boozer_solver: str = "",
    boozer_mpol: int = 0,
    boozer_ntor: int = 0,
    boozer_nfp: int = 0,
    boozer_nphi: int = 0,
    boozer_ntheta: int = 0,
    curve_dof_count: int | None = None,
) -> BananaSystemSpec:
    """Build a banana system spec using the Stage 2 current-first DOF layout."""

    if int(current_dof_count) not in (0, 1):
        raise ValueError("banana system specs support 0 or 1 active current DOF")
    if int(tf_current_dof_count) not in (0, 1):
        raise ValueError("banana system specs support 0 or 1 active TF current DOF")
    resolved_curve_dof_count = (
        int(curve.dofs.shape[0]) if curve_dof_count is None else int(curve_dof_count)
    )
    if fixed_auxiliary_coil_set is None:
        fixed_auxiliary_coil_set = make_grouped_coil_set_spec(())
    return BananaSystemSpec(
        curve=curve,
        symmetries=tuple(symmetries),
        fixed_current_dofs=_as_float64_array(fixed_current_dofs).reshape((-1,)),
        fixed_tf_current_dofs=_as_float64_array(fixed_tf_current_dofs).reshape((-1,)),
        tf_current_scale=_as_float64_array([tf_current_scale]).reshape(()),
        fixed_auxiliary_coil_set=fixed_auxiliary_coil_set,
        boozer=BananaBoozerSolveSpec(
            iota=_as_float64_array(boozer_iota).reshape((-1,)),
            G=_as_float64_array(boozer_G).reshape((-1,)),
            enabled=bool(boozer_enabled),
            solver=str(boozer_solver),
            mpol=int(boozer_mpol),
            ntor=int(boozer_ntor),
            nfp=int(boozer_nfp),
            nphi=int(boozer_nphi),
            ntheta=int(boozer_ntheta),
        ),
        decision=BananaDecisionSpec(
            current_dof_count=int(current_dof_count),
            curve_dof_count=resolved_curve_dof_count,
            tf_current_dof_count=int(tf_current_dof_count),
        ),
        fixed_auxiliary_coil_roles=tuple(
            str(role) for role in fixed_auxiliary_coil_roles
        ),
    )


def make_banana_objective_spec(
    *,
    system: BananaSystemSpec,
    stage: str,
    length_max_threshold: float,
    length_min_threshold: float,
    curvature_threshold: float,
    curvature_p: int,
    poloidal_major_radius: float,
    poloidal_theta_target: float,
    width_major_radius: float,
    width_minor_radius: float,
    width_max_threshold: float,
    width_min_threshold: float,
    self_distance_minimum: float,
    coil_coil_distance_minimum: float,
    banana_current_max_threshold: float,
    length_weight: float,
    curvature_weight: float,
    poloidal_weight: float,
    width_weight: float,
    self_distance_weight: float,
    coil_coil_distance_weight: float,
    banana_current_weight: float,
    coil_surface_distance_minimum: float = 0.0,
    coil_surface_distance_weight: float = 0.0,
    squared_flux_weight: float = 0.0,
    tf_current_max_threshold: float = 0.0,
    tf_current_weight: float = 0.0,
    surface_gamma: object = (),
    surface_normal: object = (),
    surface_nphi: int = 0,
    surface_ntheta: int = 0,
    fixed_flux_coil_set: GroupedCoilSetSpec | None = None,
    flux: FixedSurfaceFluxSpec | None = None,
    poloidal_z_position: float = 0.0,
    poloidal_exponent: int = 4,
    width_z_position: float = 0.0,
    width_scale: float = 2.0 * np.sqrt(2.0),
    width_epsilon: float = 1.0e-20,
    self_distance_neighbor_skip: int = 3,
    self_distance_normalize: bool = False,
    include_min_length: bool = True,
    include_width: bool = True,
    include_coil_surface_distance: bool = False,
    include_squared_flux: bool = False,
    include_tf_current_penalty: bool = False,
    include_current_penalty: bool = True,
    length_max_mode: PenaltyMode = "max",
    length_min_mode: PenaltyMode = "min",
    width_max_mode: PenaltyMode = "max",
    width_min_mode: PenaltyMode = "min",
    tf_current_max_mode: PenaltyMode = "max",
    banana_current_max_mode: PenaltyMode = "max",
) -> BananaObjectiveSpec:
    """Build a frozen local-objective spec from explicit thresholds and weights."""

    point_count = int(system.curve.quadpoints.shape[0])
    surface_gamma_array = _as_float64_array(surface_gamma).reshape((-1, 3))
    surface_normal_array = _as_float64_array(surface_normal).reshape((-1, 3))
    if bool(include_coil_surface_distance):
        if surface_gamma_array.shape[0] == 0 or surface_normal_array.shape[0] == 0:
            raise ValueError("coil-surface banana specs require frozen surface samples")
        if surface_gamma_array.shape != surface_normal_array.shape:
            raise ValueError("surface gamma and normal samples must have matching shape")
    if fixed_flux_coil_set is None:
        fixed_flux_coil_set = make_grouped_coil_set_spec(())
    if flux is None:
        flux = make_fixed_surface_flux_spec(
            points=np.empty((0, 3), dtype=np.float64),
            normal=np.empty((0, 0, 3), dtype=np.float64),
            target=np.empty((0, 0, 3), dtype=np.float64),
            definition="normalized",
        )
    if bool(include_squared_flux) and int(flux.points.shape[0]) == 0:
        raise ValueError("squared-flux banana specs require frozen flux samples")
    flux_normal_shape = tuple(flux.normal.shape)
    flux_nphi = int(flux_normal_shape[0]) if len(flux_normal_shape) >= 1 else 0
    flux_ntheta = int(flux_normal_shape[1]) if len(flux_normal_shape) >= 2 else 0
    return BananaObjectiveSpec(
        system=system,
        self_distance_mask=banana_self_distance_mask(
            point_count,
            int(self_distance_neighbor_skip),
        ),
        surface_gamma=surface_gamma_array,
        surface_normal=surface_normal_array,
        fixed_flux_coil_set=fixed_flux_coil_set,
        flux=flux,
        quadrature=BananaQuadratureSpec(
            curve_point_count=point_count,
            surface_sample_count=int(surface_gamma_array.shape[0]),
            surface_nphi=int(surface_nphi),
            surface_ntheta=int(surface_ntheta),
            flux_point_count=int(flux.points.shape[0]),
            flux_nphi=flux_nphi,
            flux_ntheta=flux_ntheta,
        ),
        stage=str(stage),
        length_max_mode=length_max_mode,
        length_max_threshold=float(length_max_threshold),
        length_min_mode=length_min_mode,
        length_min_threshold=float(length_min_threshold),
        curvature_threshold=float(curvature_threshold),
        curvature_p=int(curvature_p),
        poloidal_major_radius=float(poloidal_major_radius),
        poloidal_z_position=float(poloidal_z_position),
        poloidal_theta_target=float(poloidal_theta_target),
        poloidal_exponent=int(poloidal_exponent),
        width_major_radius=float(width_major_radius),
        width_minor_radius=float(width_minor_radius),
        width_z_position=float(width_z_position),
        width_scale=float(width_scale),
        width_epsilon=float(width_epsilon),
        width_max_mode=width_max_mode,
        width_max_threshold=float(width_max_threshold),
        width_min_mode=width_min_mode,
        width_min_threshold=float(width_min_threshold),
        self_distance_minimum=float(self_distance_minimum),
        self_distance_normalize=bool(self_distance_normalize),
        coil_coil_distance_minimum=float(coil_coil_distance_minimum),
        coil_surface_distance_minimum=float(coil_surface_distance_minimum),
        tf_current_max_mode=tf_current_max_mode,
        tf_current_max_threshold=float(tf_current_max_threshold),
        banana_current_max_mode=banana_current_max_mode,
        banana_current_max_threshold=float(banana_current_max_threshold),
        include_min_length=bool(include_min_length),
        include_width=bool(include_width),
        include_coil_surface_distance=bool(include_coil_surface_distance),
        include_squared_flux=bool(include_squared_flux),
        include_tf_current_penalty=bool(include_tf_current_penalty),
        include_current_penalty=bool(include_current_penalty),
        squared_flux_weight=float(squared_flux_weight),
        length_weight=float(length_weight),
        curvature_weight=float(curvature_weight),
        poloidal_weight=float(poloidal_weight),
        width_weight=float(width_weight),
        self_distance_weight=float(self_distance_weight),
        coil_coil_distance_weight=float(coil_coil_distance_weight),
        coil_surface_distance_weight=float(coil_surface_distance_weight),
        tf_current_weight=float(tf_current_weight),
        banana_current_weight=float(banana_current_weight),
    )


def banana_decision_from_dofs(
    spec: BananaDecisionSpec,
    decision_vector: jax.Array,
) -> BananaDecision:
    """Split the current-first banana decision vector into typed leaves."""

    flat_dofs = jnp.asarray(decision_vector, dtype=jnp.float64).reshape((-1,))
    tf_current_count = int(spec.tf_current_dof_count)
    current_count = int(spec.current_dof_count)
    curve_count = int(spec.curve_dof_count)
    expected_count = tf_current_count + current_count + curve_count
    if int(flat_dofs.shape[0]) != expected_count:
        raise ValueError(
            "banana decision vector length must match "
            f"{expected_count} entries, got {int(flat_dofs.shape[0])}"
        )
    tf_current_dofs = jax.lax.slice_in_dim(flat_dofs, 0, tf_current_count, axis=0)
    current_start = tf_current_count
    current_dofs = jax.lax.slice_in_dim(
        flat_dofs,
        current_start,
        current_start + current_count,
        axis=0,
    )
    curve_start = tf_current_count + current_count
    curve_dofs = jax.lax.slice_in_dim(
        flat_dofs,
        curve_start,
        curve_start + curve_count,
        axis=0,
    )
    return BananaDecision(
        tf_current_dofs=tf_current_dofs,
        current_dofs=current_dofs,
        curve_dofs=curve_dofs,
        flat_dofs=jnp.concatenate((tf_current_dofs, current_dofs, curve_dofs), axis=0),
    )


def banana_symmetry_geometry_from_arrays(
    base_gamma: jax.Array,
    base_gammadash: jax.Array,
    rotmats: jax.Array,
    current_scales: jax.Array,
    current_dof: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply dense rotation/current-scale arrays to one base banana curve."""

    def _apply_one(rotmat, current_scale):
        return (
            base_gamma @ rotmat,
            base_gammadash @ rotmat,
            current_dof * current_scale,
        )

    return jax.vmap(_apply_one, in_axes=(0, 0))(rotmats, current_scales)


def banana_geometry_from_dofs(
    spec: BananaSystemSpec,
    decision_vector: jax.Array,
) -> BananaGeometry:
    """Evaluate base and symmetry-expanded banana geometry from frozen specs."""

    decision = banana_decision_from_dofs(spec.decision, decision_vector)
    base_gamma, base_gammadash, base_gammadashdash = curve_geometry_from_dofs(
        spec.curve,
        decision.curve_dofs,
    )
    if int(spec.decision.current_dof_count) == 0:
        base_current_dof = jnp.sum(spec.fixed_current_dofs)
    else:
        base_current_dof = jnp.sum(
            jax.lax.slice_in_dim(decision.current_dofs, 0, 1, axis=0)
        )
    if int(spec.decision.tf_current_dof_count) == 0:
        tf_current_dof = jnp.sum(spec.fixed_tf_current_dofs)
    else:
        tf_current_dof = jnp.sum(
            jax.lax.slice_in_dim(decision.tf_current_dofs, 0, 1, axis=0)
        )
    tf_current = tf_current_dof * spec.tf_current_scale

    gammas = []
    gammadashs = []
    currents = []
    for symmetry in spec.symmetries:
        gamma, gammadash, current = apply_coil_symmetry(
            base_gamma,
            base_gammadash,
            base_current_dof,
            symmetry,
        )
        gammas.append(gamma)
        gammadashs.append(gammadash)
        currents.append(current)

    return BananaGeometry(
        base_gamma=base_gamma,
        base_gammadash=base_gammadash,
        base_gammadashdash=base_gammadashdash,
        gammas=jnp.stack(tuple(gammas), axis=0),
        gammadashs=jnp.stack(tuple(gammadashs), axis=0),
        currents=jnp.stack(tuple(currents), axis=0),
        tf_current_dof=tf_current_dof,
        tf_current=tf_current,
        base_current_dof=base_current_dof,
    )


def banana_curve_length_pure(gammadash: jax.Array) -> jax.Array:
    """Return the same mean incremental arclength normalization as CurveLengthJAX."""

    return curve_length_from_incremental_arclength_pure(
        incremental_arclength_pure(gammadash)
    )


def banana_curvature_p_norm_pure(
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    p: jax.Array | float,
    desired_kappa: jax.Array | float,
) -> jax.Array:
    return banana_curvature_p_norm_from_kappa_pure(
        kappa_pure(gammadash, gammadashdash),
        gammadash,
        p,
        desired_kappa,
    )


def banana_poloidal_extent_pure(
    gamma: jax.Array,
    gammadash: jax.Array,
    major_radius: float,
    z_position: float,
    theta_target: float,
    exponent: int,
) -> jax.Array:
    radius = jnp.linalg.norm(gamma[:, :2], axis=-1)
    theta_in = jnp.arctan2(gamma[:, 2] - z_position, -(radius - major_radius))
    arclength = jnp.linalg.norm(gammadash, axis=-1)
    excess = jnp.maximum(jnp.abs(theta_in) - theta_target, 0.0)
    return (1.0 / exponent) * jnp.mean((excess**exponent) * arclength)


def banana_projected_ellipse_width_pure(
    gamma: jax.Array,
    gammadash: jax.Array,
    major_radius: float,
    minor_radius: float,
    z_position: float,
    scale: float,
    epsilon: float,
) -> jax.Array:
    radius = jnp.linalg.norm(gamma[:, :2], axis=-1)
    phi = jnp.arctan2(gamma[:, 1], gamma[:, 0])
    theta = jnp.arctan2(gamma[:, 2] - z_position, -(radius - major_radius))
    phi_ref = jnp.arctan2(jnp.mean(jnp.sin(phi)), jnp.mean(jnp.cos(phi)))
    dphi = jnp.mod(phi - phi_ref + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    projected = jnp.stack([major_radius * dphi, minor_radius * theta], axis=-1)
    dl = jnp.linalg.norm(gammadash, axis=-1)
    weights = dl / jnp.sum(dl)
    center = jnp.sum(weights[:, None] * projected, axis=0)
    centered = projected - center
    cov = (weights[:, None] * centered).T @ centered
    cov_xx = cov[0, 0]
    cov_xy = 0.5 * (cov[0, 1] + cov[1, 0])
    cov_yy = cov[1, 1]
    trace = cov_xx + cov_yy
    discriminant = jnp.maximum((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy**2, 0.0)
    lambda_minor = 0.5 * (trace - jnp.sqrt(discriminant))
    return scale * jnp.sqrt(jnp.maximum(lambda_minor, epsilon))


def banana_lp_abs_hinge_pure(
    values: jax.Array,
    gammadash: jax.Array,
    p: int | float,
    threshold: float,
) -> jax.Array:
    """SIMSOPT ``Lp_torsion_pure`` scale for signed finite-build quantities."""

    values = jnp.asarray(values, dtype=jnp.float64)
    gammadash = jnp.asarray(gammadash, dtype=jnp.float64)
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    threshold_jax = jnp.asarray(threshold, dtype=values.dtype)
    return (1.0 / p) * jnp.mean(
        jnp.maximum(jnp.abs(values) - threshold_jax, 0.0) ** p * arc_length
    )


def banana_pack_rotation_fold_pure(
    frame_binormal_curvature: jax.Array,
    gammadash: jax.Array,
    p: int | float,
    threshold: float,
) -> jax.Array:
    """Fold/geodesic-curvature pack metric from explicit framed-curve arrays."""

    return banana_lp_abs_hinge_pure(
        frame_binormal_curvature,
        gammadash,
        p,
        threshold,
    )


def banana_torsional_strain_pure(
    frame_torsion: jax.Array,
    width: float,
) -> jax.Array:
    """HTS tape torsional strain used by ``LPTorsionalStrainPenalty``."""

    torsion = jnp.asarray(frame_torsion, dtype=jnp.float64)
    width_jax = jnp.asarray(width, dtype=torsion.dtype)
    return torsion**2 * width_jax**2 / 12.0


def banana_pack_twist_strain_pure(
    frame_torsion: jax.Array,
    gammadash: jax.Array,
    width: float,
    p: int | float,
    threshold: float,
) -> jax.Array:
    """Pack twist-strain metric matching ``LPTorsionalStrainPenalty.J``."""

    return banana_lp_abs_hinge_pure(
        banana_torsional_strain_pure(frame_torsion, width),
        gammadash,
        p,
        threshold,
    )


def banana_projected_reach_pure(
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    frame_normal: jax.Array,
    frame_binormal: jax.Array,
    half_normal_width: float,
    half_binormal_width: float,
) -> jax.Array:
    """Project the rotated finite-build pack half-extent into the bend plane."""

    gammadash = jnp.asarray(gammadash, dtype=jnp.float64)
    gammadashdash = jnp.asarray(gammadashdash, dtype=jnp.float64)
    frame_normal = jnp.asarray(frame_normal, dtype=jnp.float64)
    frame_binormal = jnp.asarray(frame_binormal, dtype=jnp.float64)
    half_normal = jnp.asarray(half_normal_width, dtype=gammadash.dtype)
    half_binormal = jnp.asarray(half_binormal_width, dtype=gammadash.dtype)
    tangent_norm = jnp.linalg.norm(gammadash, axis=1)
    tangent = gammadash / tangent_norm[:, None]
    curvature_vector = (
        gammadashdash - jnp.sum(gammadashdash * tangent, axis=1)[:, None] * tangent
    )
    bend_norm = jnp.linalg.norm(curvature_vector, axis=1)
    safe_bend_norm = jnp.where(bend_norm > 0.0, bend_norm, 1.0)
    bend_direction = jnp.where(
        (bend_norm > 0.0)[:, None],
        curvature_vector / safe_bend_norm[:, None],
        jnp.zeros_like(curvature_vector),
    )
    normal_component = jnp.sum(bend_direction * frame_normal, axis=1)
    binormal_component = jnp.sum(bend_direction * frame_binormal, axis=1)
    return (
        half_normal * jnp.abs(normal_component)
        + half_binormal * jnp.abs(binormal_component)
    )


def banana_rotation_aware_curvature_excess_pure(
    kappa: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    frame_normal: jax.Array,
    frame_binormal: jax.Array,
    p: int | float,
    margin: float,
    half_normal_width: float,
    half_binormal_width: float,
) -> jax.Array:
    """Rotation-aware finite-build curvature-excess metric for pack frames."""

    kappa = jnp.asarray(kappa, dtype=jnp.float64)
    gammadash = jnp.asarray(gammadash, dtype=jnp.float64)
    margin_jax = jnp.asarray(margin, dtype=kappa.dtype)
    reach = banana_projected_reach_pure(
        gammadash,
        gammadashdash,
        frame_normal,
        frame_binormal,
        half_normal_width,
        half_binormal_width,
    )
    cap = 1.0 / (margin_jax + reach)
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    excess = jnp.maximum(jnp.abs(kappa) - cap, 0.0) ** p
    return jnp.sum(excess * arc_length) / jnp.sum(arc_length)


def _normalize_rows(vectors: jax.Array) -> jax.Array:
    return vectors / jnp.clip(jnp.linalg.norm(vectors, axis=-1, keepdims=True), 1e-30)


def _banana_bracket_frame(
    gamma: jax.Array,
    winding_major_radius: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    next_gamma = jnp.roll(gamma, -1, axis=0)
    previous_gamma = jnp.roll(gamma, 1, axis=0)
    tangent = _normalize_rows(next_gamma - previous_gamma)
    phi = jnp.arctan2(gamma[:, 1], gamma[:, 0])
    winding_radius = jnp.asarray(winding_major_radius, dtype=gamma.dtype)
    axis = jnp.stack(
        [
            winding_radius * jnp.cos(phi),
            winding_radius * jnp.sin(phi),
            jnp.zeros_like(phi),
        ],
        axis=1,
    )
    radial_seed = _normalize_rows(gamma - axis)
    tangential = _normalize_rows(jnp.cross(tangent, radial_seed))
    radial = _normalize_rows(jnp.cross(tangential, tangent))
    half_segment = 0.25 * jnp.linalg.norm(next_gamma - previous_gamma, axis=1)
    return tangent, tangential, radial, half_segment


def banana_swept_channel_surface_points(
    gamma: jax.Array,
    half_binormal_width: float,
    half_radial_depth: float,
    winding_major_radius: float,
) -> jax.Array:
    """Sample the swept Type-KK U-channel envelope in the viewer-matched frame."""

    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    tangent, tangential, radial, half_segment = _banana_bracket_frame(
        gamma,
        winding_major_radius,
    )
    coeffs = jnp.asarray(_SWEPT_SURFACE_QUADRATURE, dtype=gamma.dtype)
    sweep = coeffs[:, 0]
    width = coeffs[:, 1]
    depth = coeffs[:, 2]
    samples = (
        gamma[:, None, :]
        + sweep[None, :, None] * half_segment[:, None, None] * tangent[:, None, :]
        + width[None, :, None] * half_binormal_width * tangential[:, None, :]
        + depth[None, :, None] * half_radial_depth * radial[:, None, :]
    )
    return samples.reshape((-1, 3))


def _banana_hardware_keepout_curve_chunk_pure(
    gamma: jax.Array,
    hardware_points: jax.Array,
    point_valid: jax.Array,
    point_weight: float,
    half_binormal_width: float,
    half_radial_depth: float,
    margin: float,
    winding_major_radius: float,
) -> jax.Array:
    tangent, tangential, radial, half_segment = _banana_bracket_frame(
        gamma,
        winding_major_radius,
    )
    points = jnp.asarray(hardware_points, dtype=gamma.dtype)
    valid = jnp.asarray(point_valid)
    margin_jax = jnp.asarray(margin, dtype=gamma.dtype)
    point_weight_jax = jnp.asarray(point_weight, dtype=gamma.dtype)
    zero = jnp.asarray(0.0, dtype=gamma.dtype)
    delta = points[None, :, :] - gamma[:, None, :]
    along_sweep = jnp.abs(jnp.sum(delta * tangent[:, None, :], axis=2))
    along_width = jnp.abs(jnp.sum(delta * tangential[:, None, :], axis=2))
    along_depth = jnp.abs(jnp.sum(delta * radial[:, None, :], axis=2))
    sweep_excess = jnp.maximum(along_sweep - half_segment[:, None], zero)
    width_excess = jnp.maximum(along_width - half_binormal_width, zero)
    depth_excess = jnp.maximum(along_depth - half_radial_depth, zero)
    box_distance_sq = sweep_excess**2 + width_excess**2 + depth_excess**2
    solid_distance_sq = jnp.min(box_distance_sq, axis=0)
    safe_distance_sq = jnp.where(solid_distance_sq > zero, solid_distance_sq, 1.0)
    solid_distance = jnp.where(
        solid_distance_sq > zero,
        jnp.sqrt(safe_distance_sq),
        zero,
    )
    violation = jnp.maximum(margin_jax - solid_distance, zero) / margin_jax
    weighted = point_weight_jax * violation**2
    return jnp.sum(jnp.where(valid, weighted, zero)) / margin_jax**2


def banana_hardware_keepout_point_cloud_curve_pure(
    gamma: jax.Array,
    hardware_points: jax.Array,
    point_weight: float,
    half_binormal_width: float,
    half_radial_depth: float,
    margin: float,
    winding_major_radius: float,
) -> jax.Array:
    """Point-cloud hardware keepout metric for one banana curve."""

    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    points = jnp.asarray(hardware_points, dtype=jnp.float64)
    valid = jnp.ones((points.shape[0],), dtype=bool)
    return _banana_hardware_keepout_curve_chunk_pure(
        gamma,
        points,
        valid,
        point_weight,
        half_binormal_width,
        half_radial_depth,
        margin,
        winding_major_radius,
    )


def banana_hardware_keepout_point_cloud_pure(
    gammas: jax.Array,
    hardware_points: jax.Array,
    point_weight: float,
    half_binormal_width: float,
    half_radial_depth: float,
    margin: float,
    winding_major_radius: float,
) -> jax.Array:
    """Sum point-cloud hardware keepout over symmetry-expanded banana curves."""

    gammas = jnp.asarray(gammas, dtype=jnp.float64)
    points = jnp.asarray(hardware_points, dtype=jnp.float64)
    zero = jnp.asarray(0.0, dtype=gammas.dtype)
    point_count = int(points.shape[0])
    chunk_size = _resolve_pairwise_penalty_chunk_size()

    def _curve_value(curve_gamma: jax.Array) -> jax.Array:
        if _use_dense_pairwise_path(int(curve_gamma.shape[0]), point_count, chunk_size):
            return banana_hardware_keepout_point_cloud_curve_pure(
                curve_gamma,
                points,
                point_weight,
                half_binormal_width,
                half_radial_depth,
                margin,
                winding_major_radius,
            )
        point_chunks, point_valid = _chunk_rows(points, chunk_size)

        def _scan_point_chunks(total: jax.Array, chunk_inputs) -> tuple[jax.Array, None]:
            point_chunk, valid_chunk = chunk_inputs
            value = _banana_hardware_keepout_curve_chunk_pure(
                curve_gamma,
                point_chunk,
                valid_chunk,
                point_weight,
                half_binormal_width,
                half_radial_depth,
                margin,
                winding_major_radius,
            )
            return total + value, None

        total, _ = jax.lax.scan(
            jax.checkpoint(_scan_point_chunks),
            zero,
            (point_chunks, point_valid),
        )
        return total

    total = zero
    for coil_index in range(int(gammas.shape[0])):
        total = total + _curve_value(gammas[coil_index])
    return total


def banana_self_distance_mask(point_count: int, neighbor_skip: int) -> jax.Array:
    indices = np.arange(int(point_count), dtype=np.int32)
    distance = np.abs(indices[:, None] - indices[None, :])
    periodic_distance = np.minimum(distance, int(point_count) - distance)
    return _as_float64_array((periodic_distance > int(neighbor_skip)).astype(np.float64))


def banana_self_distance_pure(
    gamma: jax.Array,
    gammadash: jax.Array,
    minimum_distance: float,
    mask: jax.Array,
    normalize: bool,
) -> jax.Array:
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    gammadash = jnp.asarray(gammadash, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=jnp.float64)
    minimum_distance_jax = jnp.asarray(minimum_distance, dtype=gamma.dtype)
    zero = jnp.asarray(0.0, dtype=gamma.dtype)
    row_count = int(gamma.shape[0])
    chunk_size = _resolve_pairwise_penalty_chunk_size()
    if _use_dense_pairwise_path(row_count, row_count, chunk_size):
        dist_sq = jnp.sum((gamma[:, None, :] - gamma[None, :, :]) ** 2, axis=2)
        safe_dist_sq = jnp.where(dist_sq > zero, dist_sq, 1.0)
        distances = jnp.where(dist_sq > zero, jnp.sqrt(safe_dist_sq), zero)
        arclength = jnp.linalg.norm(gammadash, axis=1)
        arc_weights = arclength[:, None] * arclength[None, :]
        violation = jnp.maximum(minimum_distance_jax - distances, zero) ** 2
        total = 0.5 * jnp.sum(mask * arc_weights * violation)
        if normalize:
            return total / (gamma.shape[0] ** 2)
        return total

    arclength = jnp.linalg.norm(gammadash, axis=1)
    gamma_chunks, gamma_valid = _chunk_rows(gamma, chunk_size)
    arclength_chunks, _ = _chunk_rows(arclength, chunk_size)
    chunk_count = int(gamma_chunks.shape[0])
    padded_count = chunk_count * chunk_size
    pad_count = padded_count - row_count
    padded_mask = jnp.pad(mask, ((0, pad_count), (0, pad_count)))
    mask_blocks = padded_mask.reshape(
        chunk_count,
        chunk_size,
        chunk_count,
        chunk_size,
    ).transpose((0, 2, 1, 3))

    def _scan_row_chunks(total, row_inputs):
        row_gamma, row_arclength, row_valid, row_mask_blocks = row_inputs

        def _scan_col_chunks(row_total, col_inputs):
            col_gamma, col_arclength, col_valid, mask_block = col_inputs
            valid = row_valid[:, None] & col_valid[None, :] & (mask_block > zero)
            distances = _masked_pairwise_distances(
                row_gamma,
                col_gamma,
                valid,
                minimum_distance_jax,
            )
            arc_weights = row_arclength[:, None] * col_arclength[None, :]
            safe_distances = jnp.where(valid, distances, minimum_distance_jax)
            violation = jnp.maximum(minimum_distance_jax - safe_distances, zero) ** 2
            block_total = jnp.sum(jnp.where(valid, arc_weights * violation, zero))
            return row_total + block_total, None

        total, _ = jax.lax.scan(
            jax.checkpoint(_scan_col_chunks),
            total,
            (gamma_chunks, arclength_chunks, gamma_valid, row_mask_blocks),
        )
        return total, None

    total, _ = jax.lax.scan(
        _scan_row_chunks,
        zero,
        (gamma_chunks, arclength_chunks, gamma_valid, mask_blocks),
    )
    total = 0.5 * total
    if normalize:
        return total / (gamma.shape[0] ** 2)
    return total


def banana_coil_coil_distance_pure(
    gammas: jax.Array,
    gammadashs: jax.Array,
    minimum_distance: float,
) -> jax.Array:
    """Sum CurveCurveDistanceJAX penalties over unique banana-coil pairs."""

    zero = jnp.asarray(0.0, dtype=gammas.dtype)
    total = zero
    for coil_index in range(int(gammas.shape[0])):
        for other_index in range(coil_index):
            total = total + curve_curve_distance_penalty_pure(
                gammas[coil_index],
                gammadashs[coil_index],
                gammas[other_index],
                gammadashs[other_index],
                minimum_distance,
            )
    return total


def banana_coil_surface_distance_pure(
    gammas: jax.Array,
    gammadashs: jax.Array,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    minimum_distance: float,
) -> jax.Array:
    """Sum CurveSurfaceDistanceJAX penalties over all banana coils."""

    zero = jnp.asarray(0.0, dtype=gammas.dtype)
    total = zero
    for coil_index in range(int(gammas.shape[0])):
        total = total + curve_surface_distance_penalty_pure(
            gammas[coil_index],
            gammadashs[coil_index],
            surface_gamma,
            surface_normal,
            minimum_distance,
        )
    return total


def banana_quadratic_penalty(
    value: jax.Array,
    threshold: float,
    mode: PenaltyMode,
) -> jax.Array:
    diff = value - jnp.asarray(threshold, dtype=value.dtype)
    if mode == "max":
        return 0.5 * jnp.maximum(diff, 0.0) ** 2
    if mode == "min":
        return 0.5 * jnp.minimum(diff, 0.0) ** 2
    if mode in ("identity", "two-sided"):
        return 0.5 * diff**2
    raise ValueError(f"unsupported banana penalty mode: {mode}")


def banana_current_magnitude_penalty(
    current: jax.Array,
    threshold: float,
    mode: PenaltyMode = "max",
) -> jax.Array:
    return banana_quadratic_penalty(jnp.abs(current), threshold, mode)


def _banana_squared_flux_pure(
    geometry: BananaGeometry,
    fixed_flux_coil_set: GroupedCoilSetSpec,
    flux: FixedSurfaceFluxSpec,
) -> jax.Array:
    dynamic_coil_spec = grouped_coil_set_spec_from_lists(
        geometry.gammas,
        geometry.gammadashs,
        geometry.currents,
    )
    fixed_field = grouped_biot_savart_B_from_spec(flux.points, fixed_flux_coil_set)
    dynamic_field = grouped_biot_savart_B_from_spec(flux.points, dynamic_coil_spec)
    return fixed_surface_flux_integral_from_B(fixed_field + dynamic_field, flux)


def banana_local_terms(
    spec: BananaObjectiveSpec,
    decision_vector: jax.Array,
) -> BananaLocalTerms:
    geometry = banana_geometry_from_dofs(spec.system, decision_vector)
    zero = jnp.asarray(0.0, dtype=geometry.base_gamma.dtype)
    squared_flux = (
        _banana_squared_flux_pure(geometry, spec.fixed_flux_coil_set, spec.flux)
        if spec.include_squared_flux
        else zero
    )
    curve_length = banana_curve_length_pure(geometry.base_gammadash)
    length_max = banana_quadratic_penalty(
        curve_length,
        spec.length_max_threshold,
        spec.length_max_mode,
    )
    length_min = (
        banana_quadratic_penalty(
            curve_length,
            spec.length_min_threshold,
            spec.length_min_mode,
        )
        if spec.include_min_length
        else zero
    )
    curvature = banana_curvature_p_norm_pure(
        geometry.base_gammadash,
        geometry.base_gammadashdash,
        spec.curvature_p,
        spec.curvature_threshold,
    )
    poloidal_extent = banana_poloidal_extent_pure(
        geometry.base_gamma,
        geometry.base_gammadash,
        spec.poloidal_major_radius,
        spec.poloidal_z_position,
        spec.poloidal_theta_target,
        spec.poloidal_exponent,
    )
    width = (
        banana_projected_ellipse_width_pure(
            geometry.base_gamma,
            geometry.base_gammadash,
            spec.width_major_radius,
            spec.width_minor_radius,
            spec.width_z_position,
            spec.width_scale,
            spec.width_epsilon,
        )
        if spec.include_width
        else zero
    )
    width_max = (
        banana_quadratic_penalty(width, spec.width_max_threshold, spec.width_max_mode)
        if spec.include_width
        else zero
    )
    width_min = (
        banana_quadratic_penalty(width, spec.width_min_threshold, spec.width_min_mode)
        if spec.include_width
        else zero
    )
    self_distance = banana_self_distance_pure(
        geometry.base_gamma,
        geometry.base_gammadash,
        spec.self_distance_minimum,
        spec.self_distance_mask,
        spec.self_distance_normalize,
    )
    coil_coil_distance = banana_coil_coil_distance_pure(
        geometry.gammas,
        geometry.gammadashs,
        spec.coil_coil_distance_minimum,
    )
    coil_surface_distance = (
        banana_coil_surface_distance_pure(
            geometry.gammas,
            geometry.gammadashs,
            spec.surface_gamma,
            spec.surface_normal,
            spec.coil_surface_distance_minimum,
        )
        if spec.include_coil_surface_distance
        else zero
    )
    banana_current_max = (
        banana_current_magnitude_penalty(
            jnp.max(jnp.abs(geometry.currents)),
            spec.banana_current_max_threshold,
            spec.banana_current_max_mode,
        )
        if spec.include_current_penalty
        else zero
    )
    tf_current_max = (
        banana_current_magnitude_penalty(
            geometry.tf_current,
            spec.tf_current_max_threshold,
            spec.tf_current_max_mode,
        )
        if spec.include_tf_current_penalty
        else zero
    )
    return BananaLocalTerms(
        squared_flux=squared_flux,
        length_max=length_max,
        length_min=length_min,
        curvature=curvature,
        poloidal_extent=poloidal_extent,
        width_max=width_max,
        width_min=width_min,
        self_distance=self_distance,
        coil_coil_distance=coil_coil_distance,
        coil_surface_distance=coil_surface_distance,
        tf_current_max=tf_current_max,
        banana_current_max=banana_current_max,
    )


def banana_local_value(
    spec: BananaObjectiveSpec,
    decision_vector: jax.Array,
) -> jax.Array:
    terms = banana_local_terms(spec, decision_vector)
    return (
        spec.squared_flux_weight * terms.squared_flux
        + spec.length_weight * (terms.length_max + terms.length_min)
        + spec.curvature_weight * terms.curvature
        + spec.poloidal_weight * terms.poloidal_extent
        + spec.width_weight * (terms.width_max + terms.width_min)
        + spec.self_distance_weight * terms.self_distance
        + spec.coil_coil_distance_weight * terms.coil_coil_distance
        + spec.coil_surface_distance_weight * terms.coil_surface_distance
        + spec.tf_current_weight * terms.tf_current_max
        + spec.banana_current_weight * terms.banana_current_max
    )


def banana_stage2_terms(
    spec: BananaObjectiveSpec,
    decision_vector: jax.Array,
) -> BananaLocalTerms:
    """Named Stage 2 facade for the frozen local banana term kernel."""

    return banana_local_terms(spec, decision_vector)


def banana_stage2_value(
    spec: BananaObjectiveSpec,
    decision_vector: jax.Array,
) -> jax.Array:
    """Named Stage 2 facade for the frozen local banana value kernel."""

    return banana_local_value(spec, decision_vector)


banana_stage2_value_and_grad = jax.jit(jax.value_and_grad(banana_stage2_value, argnums=1))
banana_local_value_and_grad = banana_stage2_value_and_grad
