"""Frozen physics contract for the NCSX single-stage full-space problem.

This module owns only invariant problem semantics. Runtime bootstrap values and
optimizer policy live in the adapter and solver layers respectively.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp

from simsopt_jax.core.curve_geometry import curve_length_from_spec
from simsopt_jax.core.field import (
    coil_set_spec_from_dof_extraction_spec,
    coil_specs_from_dof_extraction_spec,
    grouped_biot_savart_B_from_spec,
)
from simsopt_jax.core.quasisymmetry import (
    non_quasi_symmetric_residual_primitives,
)
from simsopt_jax.core.specs import (
    CoilSetDofExtractionSpec,
    GroupedCoilSetSpec,
    SurfaceXYZTensorFourierSpec,
)
from simsopt_jax.core.surface_dofs import (
    surface_gamma_tangents_from_dofs,
)
from simsopt_jax.core.surface_integrals import (
    surface_major_radius,
    surface_volume,
)
from simsopt_jax.geo.boozer_residual import (
    boozer_residual_scalar_from_vector,
    boozer_residual_vector,
    select_boozer_residual_mask,
)

SCHEMA_VERSION = "single-stage-fullspace-physics-v1"
EXAMPLE_ID = "single-stage-boozer-vacuum-fullspace-ncsx-v1"
AUTHORITATIVE_DTYPE = "float64"


class TermClassification(StrEnum):
    OBJECTIVE = "OBJECTIVE"
    OBJECTIVE_PENALTY = "OBJECTIVE_PENALTY"
    EQUALITY = "EQUALITY"
    FIXED_STATE = "FIXED_STATE"
    INACTIVE = "INACTIVE"


ReductionName: TypeAlias = Literal[
    "none",
    "normalized_sum_squares",
    "surface_weighted_ratio",
    "scalar_identity",
    "scalar_one_sided_max",
    "masked_vector",
    "signed_scalar",
]


@dataclass(frozen=True, slots=True)
class SourceRef:
    path: str
    sha256: str
    symbol: str


@dataclass(frozen=True, slots=True)
class TermTolerance:
    rtol: float
    atol: float


@dataclass(frozen=True, slots=True)
class TermLedgerRow:
    term_id: str
    classification: TermClassification
    fullspace_name: str
    expression: str
    sign: str
    target: str | None
    weight: float
    dtype: str
    reduction: ReductionName
    tolerance: TermTolerance
    source: SourceRef


@dataclass(frozen=True, slots=True)
class JointLayoutContract:
    ordering: tuple[str, ...]
    coil_dof_count: int
    surface_dof_count: int
    scalar_dof_count: int
    total_dof_count: int
    equality_count: int
    fixed_first_base_current: bool


@dataclass(frozen=True, slots=True)
class FullSpaceState:
    """Dynamic joint coordinates in the one canonical block order."""

    coil_dofs: jax.Array
    surface_dofs: jax.Array
    iota: jax.Array
    G: jax.Array


jax.tree_util.register_dataclass(
    FullSpaceState,
    data_fields=["coil_dofs", "surface_dofs", "iota", "G"],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class FullSpaceLayout:
    """Static pack/unpack contract for the 716-coordinate joint vector."""

    coil_dof_count: int
    surface_dof_count: int

    @property
    def total_dof_count(self) -> int:
        return self.coil_dof_count + self.surface_dof_count + 2

    def pack(self, state: FullSpaceState) -> jax.Array:
        blocks = (
            jnp.asarray(state.coil_dofs),
            jnp.asarray(state.surface_dofs),
            jnp.reshape(jnp.asarray(state.iota), (1,)),
            jnp.reshape(jnp.asarray(state.G), (1,)),
        )
        expected_shapes = (
            (self.coil_dof_count,),
            (self.surface_dof_count,),
            (1,),
            (1,),
        )
        if tuple(block.shape for block in blocks) != expected_shapes:
            raise ValueError("full-space state shapes do not match the frozen layout")
        if any(block.dtype != jnp.float64 for block in blocks):
            raise TypeError("full-space coordinates must all use float64")
        return jnp.concatenate(blocks)

    def unpack(self, z: jax.Array) -> FullSpaceState:
        vector = jnp.asarray(z)
        if vector.shape != (self.total_dof_count,):
            raise ValueError("joint vector shape does not match the frozen layout")
        if vector.dtype != jnp.float64:
            raise TypeError("joint vector must use float64")
        surface_start = self.coil_dof_count
        iota_index = surface_start + self.surface_dof_count
        return FullSpaceState(
            coil_dofs=vector[:surface_start],
            surface_dofs=vector[surface_start:iota_index],
            iota=vector[iota_index],
            G=vector[iota_index + 1],
        )


@dataclass(frozen=True, slots=True)
class FullSpaceObjectiveConfig:
    """Dynamic targets and static choices for the frozen physical objective."""

    iota_target: jax.Array
    major_radius_target: jax.Array
    length_target: jax.Array
    volume_target: jax.Array
    non_qs_weight: jax.Array
    residual_weight: jax.Array
    iota_weight: jax.Array
    major_radius_weight: jax.Array
    length_weight: jax.Array
    non_qs_axis: int
    weight_inv_modB: bool
    length_coil_indices: tuple[int, ...]


jax.tree_util.register_dataclass(
    FullSpaceObjectiveConfig,
    data_fields=[
        "iota_target",
        "major_radius_target",
        "length_target",
        "volume_target",
        "non_qs_weight",
        "residual_weight",
        "iota_weight",
        "major_radius_weight",
        "length_weight",
    ],
    meta_fields=["non_qs_axis", "weight_inv_modB", "length_coil_indices"],
)


@dataclass(frozen=True, slots=True)
class FullSpaceProblem:
    """Immutable arrays/specs needed by the pure full-space evaluator."""

    coil_dof_extraction: CoilSetDofExtractionSpec
    exact_surface_template: SurfaceXYZTensorFourierSpec
    label_surface_template: SurfaceXYZTensorFourierSpec
    non_qs_surface_template: SurfaceXYZTensorFourierSpec
    exact_mask_indices: jax.Array
    config: FullSpaceObjectiveConfig
    layout: FullSpaceLayout


jax.tree_util.register_dataclass(
    FullSpaceProblem,
    data_fields=[
        "coil_dof_extraction",
        "exact_surface_template",
        "label_surface_template",
        "non_qs_surface_template",
        "exact_mask_indices",
        "config",
    ],
    meta_fields=["layout"],
)


@dataclass(frozen=True, slots=True)
class FullSpaceRawTerms:
    non_qs: jax.Array
    residual: jax.Array
    iota: jax.Array
    major_radius: jax.Array
    length: jax.Array


jax.tree_util.register_dataclass(
    FullSpaceRawTerms,
    data_fields=["non_qs", "residual", "iota", "major_radius", "length"],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class FullSpaceConstraints:
    boozer: jax.Array
    volume: jax.Array


jax.tree_util.register_dataclass(
    FullSpaceConstraints,
    data_fields=["boozer", "volume"],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class FullSpaceObservables:
    iota: jax.Array
    G: jax.Array
    volume: jax.Array
    major_radius: jax.Array
    total_length: jax.Array
    non_qs_ratio: jax.Array
    boozer_residual_scalar: jax.Array
    boozer_residual_rms: jax.Array


jax.tree_util.register_dataclass(
    FullSpaceObservables,
    data_fields=[
        "iota",
        "G",
        "volume",
        "major_radius",
        "total_length",
        "non_qs_ratio",
        "boozer_residual_scalar",
        "boozer_residual_rms",
    ],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class FullSpaceEvaluation:
    raw_terms: FullSpaceRawTerms
    weighted_total: jax.Array
    constraints: FullSpaceConstraints
    observables: FullSpaceObservables


jax.tree_util.register_dataclass(
    FullSpaceEvaluation,
    data_fields=["raw_terms", "weighted_total", "constraints", "observables"],
    meta_fields=[],
)


class FullSpaceObjectiveResiduals(NamedTuple):
    """Five canonical least-squares blocks in frozen objective order."""

    non_qs: jax.Array
    boozer: jax.Array
    iota: jax.Array
    major_radius: jax.Array
    length: jax.Array


class FullSpaceResidualEvaluation(NamedTuple):
    """Authoritative evaluation paired with route-local GN residuals."""

    evaluation: FullSpaceEvaluation
    objective_residuals: FullSpaceObjectiveResiduals
    objective_residual_vector: jax.Array
    residual_valid: jax.Array


@dataclass(frozen=True, slots=True)
class FullSpaceFeasibilityPrimitives:
    """Raw equality residuals and device-side feasibility diagnostics."""

    residual: jax.Array
    infinity_norm: jax.Array
    all_finite: jax.Array


jax.tree_util.register_dataclass(
    FullSpaceFeasibilityPrimitives,
    data_fields=["residual", "infinity_norm", "all_finite"],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class FullSpaceKKTPrimitives:
    """Direct full-space equality-KKT quantities, without a certificate verdict."""

    objective_value: jax.Array
    constraint_residual: jax.Array
    stationarity_residual: jax.Array
    primal_feasibility_inf: jax.Array
    stationarity_inf: jax.Array
    all_finite: jax.Array


jax.tree_util.register_dataclass(
    FullSpaceKKTPrimitives,
    data_fields=[
        "objective_value",
        "constraint_residual",
        "stationarity_residual",
        "primal_feasibility_inf",
        "stationarity_inf",
        "all_finite",
    ],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class ProblemInvariantContract:
    schema_version: str
    example_id: str
    dtype: str
    nfp: int
    exact_grid_shape: tuple[int, int]
    non_qs_grid_shape: tuple[int, int]
    stellsym: bool
    optimize_G: bool
    weight_inv_modB: bool
    exact_newton_maxiter: int
    exact_newton_tolerance: float
    branch_rule: str
    layout: JointLayoutContract
    terms: tuple[TermLedgerRow, ...]


_NATIVE_EXAMPLE = SourceRef(
    path="examples/3_Advanced/single_stage_boozer_vacuum_optimization.py",
    sha256="9e01cb1f51c0354850c1da8148d87f61aa3a2f0fbba78a8aa8816352cd5d4041",
    symbol="objective construction in solve()",
)
_JAX_EXAMPLE = SourceRef(
    path="examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py",
    sha256="d0828b93afae28702cce9a20ab70bcc73b1486c971f9d7aa55272629d53a06a6",
    symbol="solve() and _objective_config()",
)
_SURFACE_TERMS = SourceRef(
    path="src/simsopt_jax_adapters/geo/surface_objectives.py",
    sha256="17bb02df157246da23556b20771ddae625d891f7fc630de7efd4d1d8899ad9d7",
    symbol="_traceable_single_stage_outer_term_values",
)
_BOOZER_CORE = SourceRef(
    path="src/simsopt_jax/geo/boozer_residual.py",
    sha256="52dcfe2a5fec45ad92872ed3a0e44bddbc478a7cf7afa162fa5d206a38ac6f0c",
    symbol="boozer_residual_vector and boozer_residual_scalar",
)
_BOOZER_EXACT = SourceRef(
    path="src/simsopt_jax_adapters/geo/boozer_surface.py",
    sha256="f0fb4ca3c6345cf07361cb32e9eab93f9520690ce701256e46cb938766823863",
    symbol="exact masked residual and signed volume label",
)

_SCALAR_TOLERANCE = TermTolerance(rtol=1.0e-12, atol=1.0e-15)
_VECTOR_TOLERANCE = TermTolerance(rtol=1.0e-10, atol=1.0e-12)


TERM_LEDGER = (
    TermLedgerRow(
        "non_qs",
        TermClassification.OBJECTIVE,
        "Phi_non_qs",
        "sum(dS*(abs_B-B_QS)^2)/sum(dS*B_QS^2)",
        "+",
        None,
        1.0,
        AUTHORITATIVE_DTYPE,
        "surface_weighted_ratio",
        _SCALAR_TOLERANCE,
        _SURFACE_TERMS,
    ),
    TermLedgerRow(
        "boozer_residual_objective",
        TermClassification.OBJECTIVE,
        "Phi_residual",
        "0.5*sum(r_full^2)/(3*nphi*ntheta)",
        "+",
        "zero",
        1.0,
        AUTHORITATIVE_DTYPE,
        "normalized_sum_squares",
        _SCALAR_TOLERANCE,
        _BOOZER_CORE,
    ),
    TermLedgerRow(
        "iota_penalty",
        TermClassification.OBJECTIVE_PENALTY,
        "Phi_iota",
        "0.5*(iota-iota_target)^2",
        "+",
        "bootstrap iota",
        1.0,
        AUTHORITATIVE_DTYPE,
        "scalar_identity",
        _SCALAR_TOLERANCE,
        _NATIVE_EXAMPLE,
    ),
    TermLedgerRow(
        "major_radius_penalty",
        TermClassification.OBJECTIVE_PENALTY,
        "Phi_major_radius",
        "0.5*(major_radius-major_radius_target)^2",
        "+",
        "bootstrap major radius",
        1.0,
        AUTHORITATIVE_DTYPE,
        "scalar_identity",
        _SCALAR_TOLERANCE,
        _SURFACE_TERMS,
    ),
    TermLedgerRow(
        "length_penalty",
        TermClassification.OBJECTIVE_PENALTY,
        "Phi_length",
        "0.5*max(total_length-length_target,0)^2",
        "+",
        "initial total base-curve length",
        1.0,
        AUTHORITATIVE_DTYPE,
        "scalar_one_sided_max",
        _SCALAR_TOLERANCE,
        _SURFACE_TERMS,
    ),
    TermLedgerRow(
        "boozer_equality",
        TermClassification.EQUALITY,
        "F_Boozer",
        "r_full[exact_mask_indices]",
        "G*B-abs_B^2*(x_phi+iota*x_theta)",
        "zero",
        1.0,
        AUTHORITATIVE_DTYPE,
        "masked_vector",
        _VECTOR_TOLERANCE,
        _BOOZER_EXACT,
    ),
    TermLedgerRow(
        "volume_equality",
        TermClassification.EQUALITY,
        "F_volume",
        "signed_volume(surface)-volume_target",
        "+",
        "pre-solve signed volume",
        1.0,
        AUTHORITATIVE_DTYPE,
        "signed_scalar",
        _VECTOR_TOLERANCE,
        _BOOZER_EXACT,
    ),
    TermLedgerRow(
        "first_base_current",
        TermClassification.FIXED_STATE,
        "fixed_first_base_current",
        "base_currents[0].fix_all()",
        "none",
        None,
        0.0,
        AUTHORITATIVE_DTYPE,
        "none",
        _SCALAR_TOLERANCE,
        _NATIVE_EXAMPLE,
    ),
    TermLedgerRow(
        "inactive_hardware_terms",
        TermClassification.INACTIVE,
        "inactive_hardware_terms",
        "curvature=curve_curve=curve_surface=surface_vessel=0",
        "none",
        None,
        0.0,
        AUTHORITATIVE_DTYPE,
        "none",
        _SCALAR_TOLERANCE,
        _JAX_EXAMPLE,
    ),
)


FROZEN_LAYOUT = FullSpaceLayout(coil_dof_count=461, surface_dof_count=253)


FROZEN_PROBLEM_CONTRACT = ProblemInvariantContract(
    schema_version=SCHEMA_VERSION,
    example_id=EXAMPLE_ID,
    dtype=AUTHORITATIVE_DTYPE,
    nfp=3,
    exact_grid_shape=(13, 13),
    non_qs_grid_shape=(40, 40),
    stellsym=True,
    optimize_G=True,
    weight_inv_modB=False,
    exact_newton_maxiter=20,
    exact_newton_tolerance=1.0e-13,
    branch_rule=(
        "continue the root reached by the authoritative exact bootstrap and "
        "reproduce it with the authoritative exact solve seeded at candidate y"
    ),
    layout=JointLayoutContract(
        ordering=("coil_dofs", "surface_dofs", "iota", "G"),
        coil_dof_count=461,
        surface_dof_count=253,
        scalar_dof_count=2,
        total_dof_count=716,
        equality_count=255,
        fixed_first_base_current=True,
    ),
    terms=TERM_LEDGER,
)


def _surface_field(
    surface_dofs: jax.Array,
    surface_template: SurfaceXYZTensorFourierSpec,
    coil_set: GroupedCoilSetSpec,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    gamma, xphi, xtheta = surface_gamma_tangents_from_dofs(
        surface_template,
        surface_dofs,
    )
    nphi, ntheta = gamma.shape[:2]
    magnetic_field = grouped_biot_savart_B_from_spec(
        gamma.reshape((-1, 3)),
        coil_set,
    ).reshape((nphi, ntheta, 3))
    return gamma, xphi, xtheta, magnetic_field


def _non_qs_quantities(
    surface_dofs: jax.Array,
    surface_template: SurfaceXYZTensorFourierSpec,
    coil_set: GroupedCoilSetSpec,
    *,
    axis: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    _gamma, xphi, xtheta, magnetic_field = _surface_field(
        surface_dofs,
        surface_template,
        coil_set,
    )
    return non_quasi_symmetric_residual_primitives(
        xphi,
        xtheta,
        magnetic_field,
        axis=axis,
    )


def _selected_total_length(
    coil_dofs: jax.Array,
    problem: FullSpaceProblem,
) -> jax.Array:
    coil_specs = coil_specs_from_dof_extraction_spec(
        problem.coil_dof_extraction,
        coil_dofs,
    )
    lengths = tuple(
        curve_length_from_spec(coil_specs[index].curve)
        for index in problem.config.length_coil_indices
    )
    return sum(lengths[1:], start=lengths[0])


def _evaluate_fullspace_impl(
    z: jax.Array,
    problem: FullSpaceProblem,
    *,
    include_objective_residuals: bool,
) -> tuple[FullSpaceEvaluation, FullSpaceObjectiveResiduals, jax.Array]:
    state = problem.layout.unpack(z)
    coil_set = coil_set_spec_from_dof_extraction_spec(
        problem.coil_dof_extraction,
        state.coil_dofs,
    )
    exact_gamma, exact_xphi, exact_xtheta, exact_B = _surface_field(
        state.surface_dofs,
        problem.exact_surface_template,
        coil_set,
    )
    full_boozer_residual = boozer_residual_vector(
        state.G,
        state.iota,
        exact_B,
        exact_xphi,
        exact_xtheta,
        weight_inv_modB=problem.config.weight_inv_modB,
    )
    residual_objective = boozer_residual_scalar_from_vector(
        full_boozer_residual,
        nphi=exact_B.shape[0],
        ntheta=exact_B.shape[1],
    )
    (
        non_qs,
        non_qs_differential_area,
        non_quasi_symmetric_B,
        non_qs_denominator,
    ) = _non_qs_quantities(
        state.surface_dofs,
        problem.non_qs_surface_template,
        coil_set,
        axis=problem.config.non_qs_axis,
    )
    major_radius = surface_major_radius(
        exact_gamma,
        exact_xphi,
        exact_xtheta,
    )
    total_length = _selected_total_length(state.coil_dofs, problem)
    half = jnp.asarray(0.5, dtype=z.dtype)
    zero = jnp.asarray(0.0, dtype=z.dtype)
    iota_penalty = half * (state.iota - problem.config.iota_target) ** 2
    major_radius_penalty = (
        half * (major_radius - problem.config.major_radius_target) ** 2
    )
    length_excess = jnp.maximum(total_length - problem.config.length_target, zero)
    length_penalty = half * length_excess**2
    raw_terms = FullSpaceRawTerms(
        non_qs=non_qs,
        residual=residual_objective,
        iota=iota_penalty,
        major_radius=major_radius_penalty,
        length=length_penalty,
    )
    weighted_total = (
        problem.config.non_qs_weight * raw_terms.non_qs
        + problem.config.residual_weight * raw_terms.residual
        + problem.config.iota_weight * raw_terms.iota
        + problem.config.major_radius_weight * raw_terms.major_radius
        + problem.config.length_weight * raw_terms.length
    )
    label_gamma, label_xphi, label_xtheta = surface_gamma_tangents_from_dofs(
        problem.label_surface_template,
        state.surface_dofs,
    )
    signed_volume = surface_volume(
        label_gamma,
        jnp.cross(label_xphi, label_xtheta),
    )
    constraints = FullSpaceConstraints(
        boozer=select_boozer_residual_mask(
            full_boozer_residual,
            problem.exact_mask_indices,
        ),
        volume=signed_volume - problem.config.volume_target,
    )
    observables = FullSpaceObservables(
        iota=state.iota,
        G=state.G,
        volume=signed_volume,
        major_radius=major_radius,
        total_length=total_length,
        non_qs_ratio=non_qs,
        boozer_residual_scalar=residual_objective,
        boozer_residual_rms=jnp.sqrt(jnp.mean(full_boozer_residual**2)),
    )
    evaluation = FullSpaceEvaluation(
        raw_terms=raw_terms,
        weighted_total=weighted_total,
        constraints=constraints,
        observables=observables,
    )
    if include_objective_residuals:
        config = problem.config
        boozer_count = jnp.asarray(full_boozer_residual.size, dtype=z.dtype)
        objective_residuals = FullSpaceObjectiveResiduals(
            non_qs=(
                jnp.sqrt(
                    2.0
                    * config.non_qs_weight
                    * non_qs_differential_area
                    / non_qs_denominator
                )
                * non_quasi_symmetric_B
            ).ravel(),
            boozer=(
                jnp.sqrt(config.residual_weight / boozer_count) * full_boozer_residual
            ),
            iota=jnp.reshape(
                jnp.sqrt(config.iota_weight) * (state.iota - config.iota_target),
                (1,),
            ),
            major_radius=jnp.reshape(
                jnp.sqrt(config.major_radius_weight)
                * (major_radius - config.major_radius_target),
                (1,),
            ),
            length=jnp.reshape(
                jnp.sqrt(config.length_weight) * length_excess,
                (1,),
            ),
        )
        residual_vector = flatten_fullspace_objective_residuals(objective_residuals)
        weights = jnp.stack(
            (
                config.non_qs_weight,
                config.residual_weight,
                config.iota_weight,
                config.major_radius_weight,
                config.length_weight,
            )
        )
        residual_valid = (
            jnp.all(jnp.isfinite(weights))
            & jnp.all(weights >= zero)
            & jnp.isfinite(non_qs_denominator)
            & (non_qs_denominator > zero)
            & jnp.all(jnp.isfinite(residual_vector))
        )
    else:
        empty = jnp.empty((0,), dtype=z.dtype)
        objective_residuals = FullSpaceObjectiveResiduals(
            non_qs=empty,
            boozer=empty,
            iota=empty,
            major_radius=empty,
            length=empty,
        )
        residual_valid = jnp.asarray(True)
    return evaluation, objective_residuals, residual_valid


def evaluate_fullspace(z: jax.Array, problem: FullSpaceProblem) -> FullSpaceEvaluation:
    """Evaluate the unchanged objective, exact equalities, and observables."""

    evaluation, _residuals, _residual_valid = _evaluate_fullspace_impl(
        z,
        problem,
        include_objective_residuals=False,
    )
    return evaluation


def flatten_fullspace_objective_residuals(
    residuals: FullSpaceObjectiveResiduals,
) -> jax.Array:
    """Flatten objective residuals in their frozen block order."""

    return jnp.concatenate(
        (
            residuals.non_qs,
            residuals.boozer,
            residuals.iota,
            residuals.major_radius,
            residuals.length,
        )
    )


def evaluate_fullspace_with_objective_residuals(
    z: jax.Array,
    problem: FullSpaceProblem,
) -> FullSpaceResidualEvaluation:
    """Evaluate authoritative values and GN residuals in one physics traversal."""

    evaluation, residuals, residual_valid = _evaluate_fullspace_impl(
        z,
        problem,
        include_objective_residuals=True,
    )
    return FullSpaceResidualEvaluation(
        evaluation=evaluation,
        objective_residuals=residuals,
        objective_residual_vector=flatten_fullspace_objective_residuals(residuals),
        residual_valid=residual_valid,
    )


def fullspace_objective_residual_vector(
    z: jax.Array,
    problem: FullSpaceProblem,
) -> jax.Array:
    """Return the canonical flat residual used only for GN curvature."""

    return evaluate_fullspace_with_objective_residuals(
        z,
        problem,
    ).objective_residual_vector


def fullspace_value(z: jax.Array, problem: FullSpaceProblem) -> jax.Array:
    return evaluate_fullspace(z, problem).weighted_total


def fullspace_constraints(
    z: jax.Array, problem: FullSpaceProblem
) -> FullSpaceConstraints:
    return evaluate_fullspace(z, problem).constraints


def flatten_fullspace_constraints(constraints: FullSpaceConstraints) -> jax.Array:
    return jnp.concatenate((constraints.boozer, jnp.reshape(constraints.volume, (1,))))


def fullspace_value_and_grad(
    z: jax.Array,
    problem: FullSpaceProblem,
) -> tuple[jax.Array, jax.Array]:
    return jax.value_and_grad(fullspace_value)(z, problem)


def fullspace_constraint_vector(
    z: jax.Array,
    problem: FullSpaceProblem,
) -> jax.Array:
    """Return the canonical flat equality vector in Boozer-then-volume order."""

    return flatten_fullspace_constraints(fullspace_constraints(z, problem))


def fullspace_constraint_jvp(
    z: jax.Array,
    tangent: jax.Array,
    problem: FullSpaceProblem,
) -> jax.Array:
    """Apply the equality Jacobian to one joint-coordinate tangent."""

    _, product = jax.jvp(
        lambda candidate: fullspace_constraint_vector(candidate, problem),
        (z,),
        (tangent,),
    )
    return product


def fullspace_constraint_vjp(
    z: jax.Array,
    cotangent: jax.Array,
    problem: FullSpaceProblem,
) -> jax.Array:
    """Apply the transposed equality Jacobian to one equality cotangent."""

    _, pullback = jax.vjp(
        lambda candidate: fullspace_constraint_vector(candidate, problem),
        z,
    )
    return pullback(cotangent)[0]


def fullspace_feasibility_primitives(
    z: jax.Array,
    problem: FullSpaceProblem,
) -> FullSpaceFeasibilityPrimitives:
    """Compute pure device-side quantities needed by the feasibility certificate."""

    residual = fullspace_constraint_vector(z, problem)
    return FullSpaceFeasibilityPrimitives(
        residual=residual,
        infinity_norm=jnp.max(jnp.abs(residual)),
        all_finite=jnp.all(jnp.isfinite(z)) & jnp.all(jnp.isfinite(residual)),
    )


def fullspace_kkt_primitives(
    z: jax.Array,
    equality_multipliers: jax.Array,
    problem: FullSpaceProblem,
) -> FullSpaceKKTPrimitives:
    """Compute direct joint KKT residuals with one fused Lagrangian pullback."""

    def lagrangian_with_aux(
        candidate: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        evaluation = evaluate_fullspace(candidate, problem)
        constraints = flatten_fullspace_constraints(evaluation.constraints)
        lagrangian = evaluation.weighted_total + jnp.vdot(
            equality_multipliers,
            constraints,
        )
        return lagrangian, (evaluation.weighted_total, constraints)

    (_, (objective_value, constraint_residual)), stationarity_residual = (
        jax.value_and_grad(lagrangian_with_aux, has_aux=True)(z)
    )
    primal_feasibility_inf = jnp.max(jnp.abs(constraint_residual))
    stationarity_inf = jnp.max(jnp.abs(stationarity_residual))
    all_finite = (
        jnp.all(jnp.isfinite(z))
        & jnp.all(jnp.isfinite(equality_multipliers))
        & jnp.isfinite(objective_value)
        & jnp.all(jnp.isfinite(constraint_residual))
        & jnp.all(jnp.isfinite(stationarity_residual))
    )
    return FullSpaceKKTPrimitives(
        objective_value=objective_value,
        constraint_residual=constraint_residual,
        stationarity_residual=stationarity_residual,
        primal_feasibility_inf=primal_feasibility_inf,
        stationarity_inf=stationarity_inf,
        all_finite=all_finite,
    )


def frozen_problem_contract_payload() -> dict[str, object]:
    """Return the immutable physics ledger as a JSON-compatible payload."""

    return asdict(FROZEN_PROBLEM_CONTRACT)
