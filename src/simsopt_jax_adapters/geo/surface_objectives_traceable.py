"""
Traceable JAX runtime bundles for single-stage surface objectives.

This module owns the pure-JAX runtime/cache/custom-VJP path that is shared by
single-stage target optimization and diagnostics. The legacy
``surfaceobjectives_jax`` module re-exports these builders for import-path
compatibility with existing callers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import partial
from threading import Lock
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec as P
from simsopt.geo.curve import incremental_arclength_pure, kappa_pure
from simsopt_jax.backend import get_backend_policy
from simsopt_jax.backend.dtypes import runtime_device_put as _runtime_device_put
from simsopt_jax.core._device_scalars import staged_like as _staged_like
from simsopt_jax.core._math_utils import (
    as_compute_array as _as_compute_array,
)
from simsopt_jax.core._math_utils import (
    as_jax_float64 as _as_jax_float64,
)
from simsopt_jax.core._math_utils import (
    as_runtime_float64 as _as_runtime_float64,
)
from simsopt_jax.core.curve_geometry import curve_geometry_from_spec
from simsopt_jax.core.field import (
    coil_set_spec_from_dof_extraction_spec,
    coil_specs_from_dof_extraction_spec,
    grouped_biot_savart_B_from_spec,
)
from simsopt_jax.core.sharding import (
    inspect_array_sharding_summary,
    maybe_shard_seed_batch_inputs,
    seed_batch_sharding_config,
)
from simsopt_jax.geo._pairwise_reductions import (
    pairwise_min_distance_batched_pure,
    pairwise_min_distance_pure,
)
from simsopt_jax.geo.boozer_residual import _surface_geometry_from_dofs
from simsopt_jax.geo.optimizers import adjoint_linear_solve as _adjoint_linear_solve
from simsopt_jax.geo.optimizers import dense_ir as _dense_ir
from simsopt_jax.geo.optimizers import linear_solve as _linear_solve
from simsopt_jax.geo.optimizers._shared import (
    cast_floating_tree,
    mark_cacheable_jit_value_and_grad,
)
from simsopt_jax.geo.surface_fourier import surface_volume
from simsopt_jax.numerical_policy import (
    MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS,
    CertificateProbeAuthority,
    CertificateProbeEvidence,
    CertificateProbeKeyData,
    resolve_certificate_probe_authority,
)
from simsopt_jax.runtime.host_boundary import (
    host_array as _host_array,
)
from simsopt_jax.runtime.host_boundary import (
    host_bool as _host_bool,
)
from simsopt_jax.runtime.host_boundary import (
    host_inf_norm as _host_inf_norm,
)
from simsopt_jax.runtime.host_boundary import (
    host_int as _host_int,
)
from simsopt_jax.runtime.host_boundary import (
    host_scalar as _host_scalar,
)
from simsopt_jax.runtime.host_boundary import (
    runtime_certificate_probe_key as _runtime_certificate_probe_key,
)

from simsopt_jax_adapters.geo.curve_objectives import curve_length_pure
from simsopt_jax_adapters.geo.factor_handoff_identity import (
    ExactFactorHandoff,
    build_exact_factor_handoff,
    require_exact_factor_handoff,
)

from .boozer_surface import (
    _ONDEVICE_OPTIMIZER_METHODS,
    _boozer_exact_residual,
    _make_boozer_penalty_objective_closure,
    _make_boozer_penalty_residual_closure,
)


@dataclass
class _LazyCompiledGradientState:
    total_gradient: Callable | None = None
    total_gradient_with_key: Callable | None = None
    value_and_grad: Callable | None = None
    gradient_lock: Lock = dataclass_field(default_factory=Lock)
    keyed_gradient_lock: Lock = dataclass_field(default_factory=Lock)
    value_and_grad_lock: Lock = dataclass_field(default_factory=Lock)


@dataclass
class _HostReportingState:
    baseline_metrics: dict[bool, dict[str, object]] = dataclass_field(
        default_factory=dict
    )
    resolved_reporting_metrics: Callable | None = None


@dataclass
class _BaselineValueAndGradState:
    value_and_grad: tuple[float, np.ndarray] | None = None
    lock: Lock = dataclass_field(default_factory=Lock)


def surface_to_surface_shortest_distance_pure(gamma1, gamma2):
    gamma1 = _as_jax_float64(gamma1).reshape((-1, 3))
    gamma2 = _as_jax_float64(gamma2).reshape((-1, 3))
    return pairwise_min_distance_pure(gamma1, gamma2)


__all__ = [
    "AcceptedIncumbentHostValueAndGrad",
    "TraceableObjectiveCandidateEvaluation",
    "TraceableObjectiveCertifiedSeededValueAndGrad",
    "TraceableObjectiveIncumbentEvaluation",
    "TraceableObjectiveInnerState",
    "TraceableObjectiveSeededValueAndGrad",
    "TraceableObjectiveSession",
    "TraceableObjectiveSolvedPair",
    "TraceableObjectiveTrialResult",
    "diagnose_traceable_objective_runtime",
    "make_traceable_objective",
    "make_traceable_objective_certified_seeded_value_and_grad",
    "make_traceable_objective_profile_suite",
    "make_traceable_objective_runtime_bundle",
    "make_traceable_objective_seeded_value_and_grad",
    "make_traceable_objective_session",
    "make_traceable_objective_solved_pair",
    "make_traceable_objective_value_and_grad",
    "make_traceable_single_stage_alm_runtime_bundle",
    "make_traceable_solved_state_value_and_grad",
    "traceable_forward_result_outer_raw_terms",
]


def _traceable_iota_from_x_inner(x_inner, optimize_G):
    """Extract iota from the inner decision vector."""
    _, iota, _ = _split_x_inner_runtime(x_inner, optimize_G)
    return iota


def _traceable_iota_target_penalty(x_inner, *, optimize_G, iota_target):
    """Quadratic iota-target penalty at an explicit inner state."""
    iota = _traceable_iota_from_x_inner(x_inner, optimize_G)
    half = _runtime_float64_scalar(0.5, reference=iota)
    iota_target_jax = _runtime_float64_scalar(iota_target, reference=iota)
    delta = iota - iota_target_jax
    return half * (delta * delta)


_TRACEABLE_SURFACE_GEOMETRY_KEYS = (
    "quadpoints_phi",
    "quadpoints_theta",
    "mpol",
    "ntor",
    "nfp",
    "stellsym",
    "scatter_indices",
    "surface_kind",
)

_TRACEABLE_LABEL_GEOMETRY_KEYS = (
    "label_quadpoints_phi",
    "label_quadpoints_theta",
    "label_mpol",
    "label_ntor",
    "label_nfp",
    "label_stellsym",
    "label_scatter_indices",
    "label_surface_kind",
)

_TRACEABLE_NEWTON_TRACE_KEYS = (
    "newton_trace_active",
    "newton_trace_step_accepted",
    "newton_trace_linear_solve_success",
    "newton_trace_linear_residual_relative",
    "newton_trace_linear_residual_norm",
    "newton_trace_linear_residual_scale",
    "newton_trace_linear_requested_tolerance",
    "newton_trace_linear_effective_tolerance",
    "newton_trace_linear_live_operator_certificate",
    "newton_trace_linear_factorization_dtype_bits",
    "newton_trace_linear_factor_application_dtype_bits",
    "newton_trace_linear_residual_dtype_bits",
    "newton_trace_certificate_value_dtype_bits",
    "newton_trace_certificate_gradient_dtype_bits",
)


class TraceableObjectiveInnerState(NamedTuple):
    """Immutable accepted Boozer state supplied explicitly to candidate solves."""

    coil_dofs: jax.Array
    solved_x: jax.Array
    objective_value: jax.Array
    eligible: jax.Array


class TraceableObjectiveIncumbentEvaluation(NamedTuple):
    """Device-resident value, gradient, and candidate continuation state."""

    value: jax.Array
    gradient: jax.Array
    candidate_inner_state: TraceableObjectiveInnerState


class AcceptedIncumbentHostValueAndGrad:
    """Host boundary that promotes Boozer continuation only after acceptance.

    Every evaluation since the last accepted step stays addressable by
    parameter identity: a line search may accept an earlier trial after
    evaluating later ones. Acceptance clears the pending set, so it is
    bounded by the evaluations of one outer iteration.

    Candidates are promotable only against the incumbent generation they
    were evaluated from: an evaluation that overlaps an acceptance is
    discarded at insert, so a later ``accept`` of it fails closed instead
    of promoting a continuation state derived from a stale incumbent.
    """

    __slots__ = ("_compiled_evaluate", "_generation", "_incumbent", "_lock", "_pending")

    def __init__(self, compiled_evaluate: Callable, initial_state):
        self._compiled_evaluate = compiled_evaluate
        self._generation = 0
        self._incumbent = initial_state
        self._pending: dict[str, TraceableObjectiveIncumbentEvaluation] = {}
        self._lock = Lock()

    @staticmethod
    def _parameter_sha256(parameters: np.ndarray) -> str:
        canonical = np.ascontiguousarray(parameters, dtype=np.dtype("<f8")).reshape(-1)
        return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()

    @property
    def current_inner_state(self) -> TraceableObjectiveInnerState:
        with self._lock:
            return self._incumbent

    def value_and_grad(self, parameters) -> tuple[float, np.ndarray]:
        canonical = np.ascontiguousarray(parameters, dtype=np.dtype("<f8")).reshape(-1)
        # Explicit H2D: host optimizer parameters cross the device boundary
        # here, and guarded runs (JAX_TRANSFER_GUARD=disallow) forbid the
        # implicit conversion.
        candidate = _runtime_device_put(canonical, dtype=jnp.float64)
        with self._lock:
            incumbent = self._incumbent
            generation = self._generation
        evaluation = self._compiled_evaluate(candidate, incumbent)
        jax.block_until_ready(evaluation)
        with self._lock:
            if generation == self._generation:
                self._pending[self._parameter_sha256(canonical)] = evaluation
        return (
            float(_host_scalar(evaluation.value, dtype=np.float64)),
            _host_array(evaluation.gradient, dtype=np.float64),
        )

    def __call__(self, parameters) -> tuple[float, np.ndarray]:
        return self.value_and_grad(parameters)

    def accept(self, parameters) -> None:
        parameter_sha256 = self._parameter_sha256(np.asarray(parameters))
        with self._lock:
            evaluation = self._pending.get(parameter_sha256)
            if evaluation is None:
                raise RuntimeError(
                    "accepted parameters do not match any pending candidate "
                    "evaluation"
                )
            if not _host_bool(evaluation.candidate_inner_state.eligible):
                raise RuntimeError("cannot promote an uncertified Boozer candidate")
            self._incumbent = evaluation.candidate_inner_state
            self._generation += 1
            self._pending.clear()


class TraceableObjectiveTrialResult(NamedTuple):
    """Immutable host diagnostic for one forward solve and its actual adjoint."""

    raw_objective_value: float
    filtered_objective_value: float
    gradient: np.ndarray
    gradient_source: str
    gradient_is_finite: bool
    gradient_inf_norm: float
    predictor_success: bool
    primal_success: bool
    actual_adjoint_success: bool
    adjoint_linear_solve_available: bool
    newton_success: bool
    newton_iterations: int
    newton_attempted_iterations: int | None
    newton_stop_reason_code: int | None
    newton_last_linear_solve_success: bool | None
    inner_penalty_residual_l2: float | None
    final_gradient_inf_norm: float | None
    newton_trace_active: np.ndarray
    newton_trace_step_accepted: np.ndarray
    newton_trace_linear_solve_success: np.ndarray
    newton_trace_linear_residual_relative: np.ndarray
    newton_trace_linear_residual_norm: np.ndarray
    newton_trace_linear_residual_scale: np.ndarray
    newton_trace_linear_requested_tolerance: np.ndarray
    newton_trace_linear_effective_tolerance: np.ndarray
    newton_trace_linear_live_operator_certificate: np.ndarray
    newton_trace_linear_factorization_dtype_bits: np.ndarray
    newton_trace_linear_factor_application_dtype_bits: np.ndarray
    newton_trace_linear_residual_dtype_bits: np.ndarray
    newton_trace_certificate_value_dtype_bits: np.ndarray
    newton_trace_certificate_gradient_dtype_bits: np.ndarray
    newton_trace_active_present: bool
    newton_trace_step_accepted_present: bool
    newton_trace_linear_solve_success_present: bool
    newton_trace_linear_residual_relative_present: bool
    newton_trace_linear_residual_norm_present: bool
    newton_trace_linear_residual_scale_present: bool
    newton_trace_linear_requested_tolerance_present: bool
    newton_trace_linear_effective_tolerance_present: bool
    newton_trace_linear_live_operator_certificate_present: bool
    newton_trace_linear_factorization_dtype_bits_present: bool
    newton_trace_linear_factor_application_dtype_bits_present: bool
    newton_trace_linear_residual_dtype_bits_present: bool
    newton_trace_certificate_value_dtype_bits_present: bool
    newton_trace_certificate_gradient_dtype_bits_present: bool
    newton_linear_solve_backend_code: int
    newton_linear_solve_backend_code_present: bool
    dense_hessian_bytes: int
    dense_hessian_bytes_present: bool
    max_dense_hessian_bytes: int
    max_dense_hessian_bytes_present: bool
    candidate_inner_state: TraceableObjectiveInnerState


_TRACEABLE_LABEL_OBJECTIVE_KEYS = (
    "targetlabel",
    "constraint_weight",
    "label_type",
    "phi_idx",
)

_TRACEABLE_INNER_OBJECTIVE_KEYS = (
    *_TRACEABLE_SURFACE_GEOMETRY_KEYS,
    *_TRACEABLE_LABEL_GEOMETRY_KEYS,
    *_TRACEABLE_LABEL_OBJECTIVE_KEYS,
    "optimize_G",
    "weight_inv_modB",
)

_TRACEABLE_TOTAL_OBJECTIVE_KEYS = (
    *_TRACEABLE_INNER_OBJECTIVE_KEYS,
    "iota_target",
    "surface_quadpoints_phi",
    "surface_quadpoints_theta",
    "coil_dof_extraction_spec",
    "outer_objective_config",
)

_TRACEABLE_EXACT_RESIDUAL_KEYS = (
    "exact_quadpoints_phi",
    "exact_quadpoints_theta",
    "mpol",
    "ntor",
    "nfp",
    "stellsym",
    "scatter_indices",
    "surface_kind",
    *_TRACEABLE_LABEL_GEOMETRY_KEYS,
    "targetlabel",
    "label_type",
    "phi_idx",
    "mask_indices",
    "stellsym_surface",
    "weight_inv_modB",
)


def _traceable_inner_objective_kwargs(objective_kwargs):
    """Select the LS inner-objective kwargs from the full traceable contract."""
    return {key: objective_kwargs[key] for key in _TRACEABLE_INNER_OBJECTIVE_KEYS}


def _traceable_total_objective_kwargs(objective_kwargs):
    """Select the scalar total-objective kwargs from the full traceable contract."""
    return {key: objective_kwargs[key] for key in _TRACEABLE_TOTAL_OBJECTIVE_KEYS}


def _traceable_exact_residual_kwargs(objective_kwargs):
    """Select the exact-residual kwargs from the full traceable contract."""
    exact_kwargs = {
        key: objective_kwargs[key] for key in _TRACEABLE_EXACT_RESIDUAL_KEYS
    }
    exact_kwargs["quadpoints_phi"] = exact_kwargs.pop("exact_quadpoints_phi")
    exact_kwargs["quadpoints_theta"] = exact_kwargs.pop("exact_quadpoints_theta")
    return exact_kwargs


def _traceable_total_objective(
    x_inner,
    coil_dofs,
    coil_set_spec,
    *,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    label_quadpoints_phi,
    label_quadpoints_theta,
    label_mpol,
    label_ntor,
    label_nfp,
    label_stellsym,
    label_scatter_indices,
    label_surface_kind,
    optimize_G,
    weight_inv_modB,
    constraint_weight,
    targetlabel,
    label_type,
    phi_idx,
    iota_target,
    surface_quadpoints_phi,
    surface_quadpoints_theta,
    coil_dof_extraction_spec,
    outer_objective_config,
):
    """Pure single-stage objective evaluated at an explicit inner state."""
    if outer_objective_config is not None:
        return _traceable_full_single_stage_outer_objective(
            x_inner,
            coil_dofs,
            coil_set_spec,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            scatter_indices=scatter_indices,
            surface_kind=surface_kind,
            label_quadpoints_phi=label_quadpoints_phi,
            label_quadpoints_theta=label_quadpoints_theta,
            label_mpol=label_mpol,
            label_ntor=label_ntor,
            label_nfp=label_nfp,
            label_stellsym=label_stellsym,
            label_scatter_indices=label_scatter_indices,
            label_surface_kind=label_surface_kind,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
            constraint_weight=constraint_weight,
            targetlabel=targetlabel,
            label_type=label_type,
            phi_idx=phi_idx,
            iota_target=iota_target,
            surface_quadpoints_phi=surface_quadpoints_phi,
            surface_quadpoints_theta=surface_quadpoints_theta,
            coil_dof_extraction_spec=coil_dof_extraction_spec,
            outer_objective_config=outer_objective_config,
        )
    J_boozer = _boozer_residual_J_of_x_inner(
        x_inner,
        coil_set_spec=coil_set_spec,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
        label_quadpoints_phi=label_quadpoints_phi,
        label_quadpoints_theta=label_quadpoints_theta,
        label_mpol=label_mpol,
        label_ntor=label_ntor,
        label_nfp=label_nfp,
        label_stellsym=label_stellsym,
        label_scatter_indices=label_scatter_indices,
        label_surface_kind=label_surface_kind,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
        constraint_weight=constraint_weight,
        targetlabel=targetlabel,
        label_type=label_type,
        phi_idx=phi_idx,
    )
    return J_boozer + _traceable_iota_target_penalty(
        x_inner,
        optimize_G=optimize_G,
        iota_target=iota_target,
    )


def _evaluate_traceable_total_objective(
    x_inner,
    coil_dofs,
    coil_set_spec,
    objective_kwargs,
):
    """Evaluate the full traceable scalar objective from packed kwargs."""
    return _traceable_total_objective(
        x_inner,
        coil_dofs,
        coil_set_spec,
        **_traceable_total_objective_kwargs(objective_kwargs),
    )


def _traceable_weighted_terms_total(weighted_terms):
    """Sum weighted outer terms in their canonical specification order."""
    total = None
    for term_name, _weight_key in _TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS:
        term_value = weighted_terms[term_name]
        total = term_value if total is None else total + term_value
    return total


def _evaluate_traceable_total_objective_with_raw_terms(
    x_inner,
    coil_dofs,
    coil_set_spec,
    objective_kwargs,
):
    """Return the total objective and reusable raw outer terms when available."""
    outer_objective_config = objective_kwargs.get("outer_objective_config")
    if outer_objective_config is None:
        return (
            _evaluate_traceable_total_objective(
                x_inner,
                coil_dofs,
                coil_set_spec,
                objective_kwargs,
            ),
            None,
        )
    raw_terms = _traceable_single_stage_outer_term_values(
        x_inner,
        coil_dofs,
        coil_set_spec,
        **_traceable_total_objective_kwargs(objective_kwargs),
    )
    weighted_terms = _traceable_weighted_single_stage_outer_term_values(
        raw_terms,
        outer_objective_config=outer_objective_config,
    )
    return _traceable_weighted_terms_total(weighted_terms), raw_terms


def _traceable_directional_inner_stationarity(
    x_inner,
    tangent,
    coil_set_spec,
    **objective_kwargs,
):
    """Directional inner stationarity without materializing the full gradient."""
    runtime_objective_kwargs = _traceable_runtime_deviceify_tree(objective_kwargs)
    inner_objective = _make_boozer_penalty_objective_closure(
        coil_set_spec=coil_set_spec,
        decision_split_mode="jvp",
        **runtime_objective_kwargs,
    )
    return jax.jvp(inner_objective, (x_inner,), (tangent,))[1]


def _traceable_directional_exact_residual(
    x_inner,
    tangent,
    coil_set_spec,
    objective_kwargs,
):
    """Contract the exact inner equation with an adjoint tangent."""
    residual = _boozer_exact_residual(
        x_inner,
        coil_set_spec=coil_set_spec,
        **_traceable_exact_residual_kwargs(objective_kwargs),
    )
    return jnp.vdot(tangent, residual)


def _traceable_inner_stationarity_coil_jvp(
    x_inner,
    coil_dofs,
    coil_dofs_tangent,
    coil_set_spec_from_dofs,
    **objective_kwargs,
):
    """Directional coil derivative of inner stationarity without a basis map."""
    runtime_objective_kwargs = _traceable_runtime_deviceify_tree(objective_kwargs)

    def coil_directional_inner_objective(current_x_inner):
        def inner_objective_of_coils(current_coil_dofs):
            inner_objective = _make_boozer_penalty_objective_closure(
                coil_set_spec=coil_set_spec_from_dofs(current_coil_dofs),
                decision_split_mode="jvp",
                **runtime_objective_kwargs,
            )
            return inner_objective(current_x_inner)

        return jax.jvp(
            inner_objective_of_coils,
            (coil_dofs,),
            (coil_dofs_tangent,),
        )[1]

    return _strict_scalar_grad(coil_directional_inner_objective, x_inner)


def _traceable_directional_inner_objective(
    x_inner,
    tangent,
    coil_set_spec,
    **objective_kwargs,
):
    """Directional derivative of the LS inner objective at an explicit state."""
    return _traceable_directional_inner_stationarity(
        x_inner,
        tangent,
        coil_set_spec,
        **objective_kwargs,
    )


def _traceable_non_dense_adjoint_selected() -> bool:
    """Return whether the explicit adjoint route is matrix-free."""
    return _adjoint_linear_solve._ADJOINT_LINEAR_SOLVER in ("cg", "lsmr_j")


def _traceable_residual_jacobian_adjoint_selected() -> bool:
    """Return whether the adjoint route acts on the residual Jacobian."""
    return _adjoint_linear_solve._ADJOINT_LINEAR_SOLVER == "lsmr_j"


def _traceable_solve_hessian_linearization(
    booz_jax,
    solved_x,
    rhs,
    coil_set_spec,
    objective_kwargs,
    *,
    linear_solve_factors,
    linear_solve_tol,
    linear_solve_stab,
    transpose,
    certificate_probe_key=None,
):
    explicit_adjoint = transpose and _traceable_non_dense_adjoint_selected()
    residual_jacobian_adjoint = (
        explicit_adjoint and _traceable_residual_jacobian_adjoint_selected()
    )
    objective_fn = _make_boozer_penalty_objective_closure(
        coil_set_spec=coil_set_spec,
        decision_split_mode="jvp",
        **_traceable_inner_objective_kwargs(objective_kwargs),
    )
    if linear_solve_factors is not None and not explicit_adjoint:
        live_x = _as_jax_float64(solved_x)
        live_rhs = _as_jax_float64(rhs)
        hessian_operator = _adjoint_linear_solve._hessian_linear_operator(
            objective_fn,
            live_x,
            stab=float(linear_solve_stab),
        )
        live_matvec = (
            hessian_operator["transpose_matvec"]
            if transpose
            else hessian_operator["matvec"]
        )
        return _traceable_solve_plu_linearization(
            linear_solve_factors,
            live_rhs,
            live_matvec=live_matvec,
            linear_solve_tol=linear_solve_tol,
            transpose=transpose,
        )

    # `_traceable_result_linear_solve_factors` deliberately returns ``None`` on
    # the LS runtime lane so adjoint solves stay matrix-free. The default path
    # uses the pure-JAX Hessian operator solve; the explicit ``lsmr_j`` selector
    # supplies the residual-J closure to the same solver seam. Both remain fully
    # traceable under JIT and do not call a live host solver. Removing this path
    # would force every LS warm-start and adjoint solve to surface
    # ``success=False`` and emit NaN gradients (verified by
    # ``test_runtime_bundle_allows_strict_transfer_guard`` /
    # ``test_runtime_bundle_host_wrappers_allow_host_inputs_under_strict_transfer_guard``).
    residual_kwargs = {}
    if residual_jacobian_adjoint:
        residual_kwargs["residual_fn"] = _make_boozer_penalty_residual_closure(
            coil_set_spec=coil_set_spec,
            decision_split_mode="jvp",
            **_traceable_inner_objective_kwargs(objective_kwargs),
        )
    linear_solver = (
        _adjoint_linear_solve._ADJOINT_LINEAR_SOLVER if transpose else "dense"
    )
    policy = get_backend_policy()
    mixed_dense_ir = (
        certificate_probe_key is not None
        and linear_solver == "dense"
        and np.dtype(policy.compute_dtype) == np.dtype(np.float32)
        and np.dtype(policy.runtime_dtype) == np.dtype(np.float64)
    )
    if mixed_dense_ir:
        proposal_dtype = np.dtype(policy.compute_dtype)
        proposal_coil_set_spec = cast_floating_tree(
            coil_set_spec,
            proposal_dtype,
        )
        proposal_objective_kwargs = cast_floating_tree(
            _traceable_inner_objective_kwargs(objective_kwargs),
            proposal_dtype,
        )
        residual_kwargs["proposal_objective_fn"] = (
            _make_boozer_penalty_objective_closure(
                coil_set_spec=proposal_coil_set_spec,
                decision_split_mode="jvp",
                infer_optimizer_state_dtype=True,
                **proposal_objective_kwargs,
            )
        )
        residual_kwargs["certificate_probe_key"] = certificate_probe_key
    return _adjoint_linear_solve._solve_hessian_least_squares_system_with_status(
        objective_fn,
        solved_x,
        rhs,
        stab=float(linear_solve_stab),
        tol=linear_solve_tol,
        solver=linear_solver,
        **residual_kwargs,
    )


def _traceable_plu_unpack_triple(linear_solve_factors):
    """Return ``(P, L, U)`` from a 3- or 5-tuple ``linear_solve_factors``.

    Phase 2 (``docs/parity_scientific_equivalence_contract_2026-05-09.md``
    §5.3) packs the LS forward path's ``(lu, piv)`` factors alongside
    the public ``(P, L, U)`` triple as a 5-tuple
    ``(P, L, U, lu, piv)`` so that the byte-shared dispatch can route
    forward and adjoint solves through ``jsp_linalg.lu_solve``. The
    legacy 3-tuple form remains the supported triangular fallback.
    """
    return linear_solve_factors[0], linear_solve_factors[1], linear_solve_factors[2]


def _traceable_plu_unpack_lu_piv(linear_solve_factors):
    """Return ``(lu, piv)`` from a 5-tuple ``linear_solve_factors``, else ``None``."""
    factor_count = len(linear_solve_factors)
    assert factor_count in (3, 5), (
        "linear_solve_factors must be (P, L, U) or (P, L, U, lu, piv)"
    )
    if factor_count == 5:
        return linear_solve_factors[3], linear_solve_factors[4]
    return None


def _traceable_plu_matvec(linear_solve_factors, vector, *, transpose):
    P, L, U = _traceable_plu_unpack_triple(linear_solve_factors)
    if transpose:
        return U.T @ (L.T @ (P.T @ vector))
    return P @ (L @ (U @ vector))


def _traceable_plu_matrix(linear_solve_factors):
    P, L, U = _traceable_plu_unpack_triple(linear_solve_factors)
    return P @ (L @ U)


_TRACEABLE_SUPPLIED_FACTOR_MAX_CORRECTIONS = MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS
_TRACEABLE_SUPPLIED_FACTOR_FP64_REBUILD_BUDGET = 1


class _TraceablePLURefinement(NamedTuple):
    """One adaptive refinement attempt against an externally supplied operator."""

    solution: jax.Array
    residual: jax.Array
    status: _linear_solve._LinearSolveStatus
    residual_relative_trace: jax.Array
    contraction_ratio_trace: jax.Array
    residual_relative_trace_length: jax.Array
    contraction_finite: jax.Array
    contraction_monotone: jax.Array
    stagnated: jax.Array


class _TraceableSuppliedFactorSolveStatus(NamedTuple):
    """Solve status plus live-operator supplied-factor certificate metrics."""

    success: jax.Array
    residual: jax.Array
    residual_relative: jax.Array
    iterations: jax.Array
    residual_scale: jax.Array
    requested_tolerance: jax.Array
    effective_tolerance: jax.Array
    supplied_factor_residual_relative_trace: jax.Array
    supplied_factor_residual_relative_trace_length: jax.Array
    supplied_factor_contraction_ratio_trace: jax.Array
    supplied_factor_contraction_finite: jax.Array
    supplied_factor_contraction_monotone: jax.Array
    supplied_factor_stagnated: jax.Array
    fp64_rebuild_count: jax.Array
    fp64_rebuild_residual_relative_trace: jax.Array
    fp64_rebuild_residual_relative_trace_length: jax.Array


def _traceable_apply_plu_inverse(
    linear_solve_factors,
    rhs,
    *,
    transpose,
):
    """Apply supplied factors as an approximate inverse in their native dtype."""
    lu_piv = _traceable_plu_unpack_lu_piv(linear_solve_factors)
    if lu_piv is not None:
        lu, piv = lu_piv
        factor_rhs = jnp.asarray(rhs, dtype=lu.dtype)
        solution = jsp_linalg.lu_solve(
            (lu, piv),
            factor_rhs,
            trans=1 if transpose else 0,
        )
    else:
        P, L, U = _traceable_plu_unpack_triple(linear_solve_factors)
        factor_rhs = jnp.asarray(rhs, dtype=U.dtype)
        if transpose:
            y = jsp_linalg.solve_triangular(U.T, factor_rhs, lower=True)
            z = jsp_linalg.solve_triangular(L.T, y, lower=False)
            solution = P @ z
        else:
            y = jsp_linalg.solve_triangular(L, P.T @ factor_rhs, lower=True)
            solution = jsp_linalg.solve_triangular(U, y, lower=False)
    return jnp.asarray(solution, dtype=rhs.dtype)


def _traceable_refine_plu_linearization(
    linear_solve_factors,
    rhs,
    *,
    live_matvec,
    linear_solve_tol,
    transpose,
):
    """Adaptively refine a supplied inverse against the live FP64 operator."""
    rhs = _as_jax_float64(rhs)
    solution = _traceable_apply_plu_inverse(
        linear_solve_factors,
        rhs,
        transpose=transpose,
    )
    residual = rhs - _as_jax_float64(live_matvec(solution))
    status = _linear_solve._linear_solve_status(
        solution,
        residual,
        rhs,
        tol=linear_solve_tol,
        iterations=0,
    )
    trace_dtype = rhs.dtype
    residual_relative_trace = (
        jnp.full(
            (_TRACEABLE_SUPPLIED_FACTOR_MAX_CORRECTIONS + 1,),
            jnp.asarray(jnp.nan, dtype=trace_dtype),
            dtype=trace_dtype,
        )
        .at[0]
        .set(status.residual_relative)
    )
    contraction_ratio_trace = jnp.full(
        (_TRACEABLE_SUPPLIED_FACTOR_MAX_CORRECTIONS,),
        jnp.asarray(jnp.nan, dtype=trace_dtype),
        dtype=trace_dtype,
    )
    initial_finite = _linear_solve._linear_solve_finite(
        solution,
        residual,
    ) & jnp.isfinite(status.residual_relative)
    initial_state = _TraceablePLURefinement(
        solution=solution,
        residual=residual,
        status=status,
        residual_relative_trace=residual_relative_trace,
        contraction_ratio_trace=contraction_ratio_trace,
        residual_relative_trace_length=jnp.asarray(1, dtype=jnp.int32),
        contraction_finite=initial_finite,
        contraction_monotone=jnp.asarray(True, dtype=jnp.bool_),
        stagnated=jnp.asarray(False, dtype=jnp.bool_),
    )
    unit_roundoff = jnp.asarray(
        np.finfo(np.float64).eps / 2.0,
        dtype=trace_dtype,
    )

    def refinement_active(state):
        correction_count = state.residual_relative_trace_length - 1
        return (
            state.contraction_finite
            & state.contraction_monotone
            & ~state.stagnated
            & ~state.status.success
            & (correction_count < _TRACEABLE_SUPPLIED_FACTOR_MAX_CORRECTIONS)
        )

    def refine_once(state):
        correction = _traceable_apply_plu_inverse(
            linear_solve_factors,
            state.residual,
            transpose=transpose,
        )
        candidate_solution = state.solution + correction
        candidate_residual = rhs - _as_jax_float64(live_matvec(candidate_solution))
        candidate_status = _linear_solve._linear_solve_status(
            candidate_solution,
            candidate_residual,
            rhs,
            tol=linear_solve_tol,
            iterations=state.residual_relative_trace_length,
        )
        previous_relative = state.status.residual_relative
        candidate_relative = candidate_status.residual_relative
        ratio = candidate_relative / jnp.maximum(previous_relative, unit_roundoff)
        candidate_finite = (
            _linear_solve._linear_solve_finite(
                candidate_solution,
                candidate_residual,
            )
            & jnp.isfinite(candidate_relative)
            & jnp.isfinite(ratio)
        )
        monotone = candidate_relative < previous_relative
        improvement = previous_relative - candidate_relative
        stagnation_floor = unit_roundoff * jnp.maximum(
            previous_relative,
            jnp.asarray(1.0, dtype=trace_dtype),
        )
        stagnated = candidate_finite & monotone & (improvement <= stagnation_floor)
        accept_candidate = candidate_finite & monotone
        solution, residual, accepted_status = lax.cond(
            accept_candidate,
            lambda _: (candidate_solution, candidate_residual, candidate_status),
            lambda _: (state.solution, state.residual, state.status),
            operand=None,
        )
        trace_index = state.residual_relative_trace_length
        ratio_index = trace_index - 1
        return _TraceablePLURefinement(
            solution=solution,
            residual=residual,
            status=accepted_status,
            residual_relative_trace=state.residual_relative_trace.at[trace_index].set(
                candidate_relative
            ),
            contraction_ratio_trace=state.contraction_ratio_trace.at[ratio_index].set(
                ratio
            ),
            residual_relative_trace_length=trace_index + 1,
            contraction_finite=state.contraction_finite & candidate_finite,
            contraction_monotone=state.contraction_monotone & monotone,
            stagnated=state.stagnated | stagnated,
        )

    refined = lax.while_loop(refinement_active, refine_once, initial_state)
    correction_count = refined.residual_relative_trace_length - 1
    return refined._replace(
        status=refined.status._replace(iterations=correction_count),
    )


def _traceable_solve_plu_linearization(
    linear_solve_factors,
    rhs,
    *,
    live_matvec,
    linear_solve_tol,
    transpose,
):
    """Use supplied PLU provisionally and certify it on the live FP64 operator."""
    rhs = _as_jax_float64(rhs)
    supplied = _traceable_refine_plu_linearization(
        linear_solve_factors,
        rhs,
        live_matvec=live_matvec,
        linear_solve_tol=linear_solve_tol,
        transpose=transpose,
    )

    def rebuild_live_fp64(_):
        live_matrix = _linear_solve._dense_square_operator_matrix(
            live_matvec,
            rhs,
            matrix_dtype=np.float64,
            sweep_dtype=np.float64,
        )
        live_lu_piv = jsp_linalg.lu_factor(live_matrix)
        live_plu = _linear_solve._plu_from_lu_piv(live_lu_piv)
        rebuilt_factors = (
            live_plu[0],
            live_plu[1],
            live_plu[2],
            live_lu_piv[0],
            live_lu_piv[1],
        )
        rebuilt = _traceable_refine_plu_linearization(
            rebuilt_factors,
            rhs,
            live_matvec=live_matvec,
            linear_solve_tol=linear_solve_tol,
            transpose=False,
        )
        solve_safe = _linear_solve._dense_matrix_solve_numerically_safe(
            live_matrix,
            rebuilt.solution,
            rhs,
            tol=linear_solve_tol,
            lu_piv=live_lu_piv,
            solve_dtype=rhs.dtype,
        )
        small_solution_success = (
            _linear_solve._dense_matrix_solve_small_solution_success(
                rebuilt.solution,
                rhs,
                tol=linear_solve_tol,
            )
        )
        rebuilt_status = rebuilt.status._replace(
            success=rebuilt.status.success & (solve_safe | small_solution_success),
            iterations=supplied.status.iterations + rebuilt.status.iterations,
        )
        return rebuilt.solution, rebuilt_status, rebuilt

    def keep_supplied(_):
        empty_trace = jnp.full_like(
            supplied.residual_relative_trace,
            jnp.asarray(jnp.nan, dtype=rhs.dtype),
        )
        empty_ratio_trace = jnp.full_like(
            supplied.contraction_ratio_trace,
            jnp.asarray(jnp.nan, dtype=rhs.dtype),
        )
        unused_rebuild = _TraceablePLURefinement(
            solution=supplied.solution,
            residual=supplied.residual,
            status=supplied.status,
            residual_relative_trace=empty_trace,
            contraction_ratio_trace=empty_ratio_trace,
            residual_relative_trace_length=jnp.asarray(0, dtype=jnp.int32),
            contraction_finite=jnp.asarray(True, dtype=jnp.bool_),
            contraction_monotone=jnp.asarray(True, dtype=jnp.bool_),
            stagnated=jnp.asarray(False, dtype=jnp.bool_),
        )
        return supplied.solution, supplied.status, unused_rebuild

    solution, final_status, rebuilt = lax.cond(
        supplied.status.success,
        keep_supplied,
        rebuild_live_fp64,
        operand=None,
    )
    rebuild_count = (
        jnp.asarray(
            _TRACEABLE_SUPPLIED_FACTOR_FP64_REBUILD_BUDGET,
            dtype=jnp.int32,
        )
        * ~supplied.status.success
    )
    status = _TraceableSuppliedFactorSolveStatus(
        success=final_status.success,
        residual=final_status.residual,
        residual_relative=final_status.residual_relative,
        iterations=final_status.iterations,
        residual_scale=final_status.residual_scale,
        requested_tolerance=final_status.requested_tolerance,
        effective_tolerance=final_status.effective_tolerance,
        supplied_factor_residual_relative_trace=supplied.residual_relative_trace,
        supplied_factor_residual_relative_trace_length=(
            supplied.residual_relative_trace_length
        ),
        supplied_factor_contraction_ratio_trace=supplied.contraction_ratio_trace,
        supplied_factor_contraction_finite=supplied.contraction_finite,
        supplied_factor_contraction_monotone=supplied.contraction_monotone,
        supplied_factor_stagnated=supplied.stagnated,
        fp64_rebuild_count=rebuild_count,
        fp64_rebuild_residual_relative_trace=rebuilt.residual_relative_trace,
        fp64_rebuild_residual_relative_trace_length=(
            rebuilt.residual_relative_trace_length
        ),
    )
    return solution, status


def _traceable_solve_exact_linearization(
    solved_x,
    rhs,
    coil_set_spec,
    objective_kwargs,
    *,
    linear_solve_tol,
    transpose,
):
    def residual_fn(x_inner):
        return _boozer_exact_residual(
            x_inner,
            coil_set_spec=coil_set_spec,
            **_traceable_exact_residual_kwargs(objective_kwargs),
        )

    return _linear_solve._solve_jacobian_system_with_status(
        residual_fn,
        solved_x,
        rhs,
        transpose=transpose,
        tol=linear_solve_tol,
        max_refinement_steps=(
            _adjoint_linear_solve._EXACT_JACOBIAN_OPERATOR_GMRES_REFINEMENT_STEPS
        ),
    )


def _traceable_solve_linearization(
    booz_jax,
    solved_x,
    rhs,
    coil_set_spec,
    objective_kwargs,
    *,
    linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    transpose,
    certificate_probe_key=None,
):
    if linearization_kind == "hessian":
        return _traceable_solve_hessian_linearization(
            booz_jax,
            solved_x,
            rhs,
            coil_set_spec,
            objective_kwargs,
            linear_solve_factors=linear_solve_factors,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            transpose=transpose,
            certificate_probe_key=certificate_probe_key,
        )
    if linearization_kind == "exact_jacobian":
        return _traceable_solve_exact_linearization(
            solved_x,
            rhs,
            coil_set_spec,
            objective_kwargs,
            linear_solve_tol=linear_solve_tol,
            transpose=transpose,
        )
    raise ValueError(
        f"Unsupported traceable linearization kind {linearization_kind!r}."
    )


def _pack_traceable_forward_result(
    *,
    value,
    raw_value=None,
    x,
    sdofs,
    iota,
    G,
    linear_solve_factors,
    success,
    predictor_success=None,
    primal_success,
    adjoint_linear_solve_available,
    newton_success=None,
    newton_iterations=None,
    newton_attempted_iterations=None,
    newton_stop_reason_code=None,
    newton_last_linear_solve_success=None,
    inner_penalty_residual_l2=None,
    final_gradient_inf_norm=None,
    newton_trace_capacity: int,
    newton_trace_active: jax.Array | None = None,
    newton_trace_step_accepted: jax.Array | None = None,
    newton_trace_linear_solve_success: jax.Array | None = None,
    newton_trace_linear_residual_relative: jax.Array | None = None,
    newton_trace_linear_residual_norm: jax.Array | None = None,
    newton_trace_linear_residual_scale: jax.Array | None = None,
    newton_trace_linear_requested_tolerance: jax.Array | None = None,
    newton_trace_linear_effective_tolerance: jax.Array | None = None,
    newton_trace_linear_live_operator_certificate: jax.Array | None = None,
    newton_trace_linear_factorization_dtype_bits: jax.Array | None = None,
    newton_trace_linear_factor_application_dtype_bits: jax.Array | None = None,
    newton_trace_linear_residual_dtype_bits: jax.Array | None = None,
    newton_trace_certificate_value_dtype_bits: jax.Array | None = None,
    newton_trace_certificate_gradient_dtype_bits: jax.Array | None = None,
    newton_trace_presence: Mapping[str, jax.Array | None] | None = None,
    newton_linear_solve_backend_code=None,
    dense_hessian_bytes=None,
    max_dense_hessian_bytes=None,
    outer_raw_terms=None,
):
    """Return the normalized traceable forward-result contract."""
    missing_bool = _staged_like(value, False, dtype=jnp.bool_)
    missing_int = _staged_like(value, np.iinfo(np.int32).min, dtype=jnp.int32)
    missing_float = _staged_like(value, np.nan, dtype=jnp.float64)

    def newton_trace_field(
        trace_value: jax.Array | None,
        *,
        dtype: jnp.dtype,
        missing_value: jax.Array,
    ) -> jax.Array:
        if trace_value is None:
            return jnp.full(
                (newton_trace_capacity,),
                missing_value,
                dtype=dtype,
            )
        trace = jnp.asarray(trace_value, dtype=dtype)
        if trace.ndim != 1 or trace.shape[0] > newton_trace_capacity:
            raise ValueError(
                "Newton trace must be one-dimensional and fit its static capacity."
            )
        return jnp.pad(
            trace,
            (0, newton_trace_capacity - trace.shape[0]),
            constant_values=missing_value,
        )

    def newton_bool_trace_field(trace_value: jax.Array | None) -> jax.Array:
        return newton_trace_field(
            trace_value,
            dtype=jnp.dtype(jnp.bool_),
            missing_value=missing_bool,
        )

    def newton_float_trace_field(trace_value: jax.Array | None) -> jax.Array:
        return newton_trace_field(
            trace_value,
            dtype=jnp.dtype(jnp.float64),
            missing_value=missing_float,
        )

    def newton_int_trace_field(trace_value: jax.Array | None) -> jax.Array:
        return newton_trace_field(
            trace_value,
            dtype=jnp.dtype(jnp.int32),
            missing_value=missing_int,
        )

    backend_code_present = newton_linear_solve_backend_code is not None
    trace_values = {
        "newton_trace_active": newton_trace_active,
        "newton_trace_step_accepted": newton_trace_step_accepted,
        "newton_trace_linear_solve_success": newton_trace_linear_solve_success,
        "newton_trace_linear_residual_relative": (
            newton_trace_linear_residual_relative
        ),
        "newton_trace_linear_residual_norm": newton_trace_linear_residual_norm,
        "newton_trace_linear_residual_scale": newton_trace_linear_residual_scale,
        "newton_trace_linear_requested_tolerance": (
            newton_trace_linear_requested_tolerance
        ),
        "newton_trace_linear_effective_tolerance": (
            newton_trace_linear_effective_tolerance
        ),
        "newton_trace_linear_live_operator_certificate": (
            newton_trace_linear_live_operator_certificate
        ),
        "newton_trace_linear_factorization_dtype_bits": (
            newton_trace_linear_factorization_dtype_bits
        ),
        "newton_trace_linear_factor_application_dtype_bits": (
            newton_trace_linear_factor_application_dtype_bits
        ),
        "newton_trace_linear_residual_dtype_bits": (
            newton_trace_linear_residual_dtype_bits
        ),
        "newton_trace_certificate_value_dtype_bits": (
            newton_trace_certificate_value_dtype_bits
        ),
        "newton_trace_certificate_gradient_dtype_bits": (
            newton_trace_certificate_gradient_dtype_bits
        ),
    }
    packed = {
        "value": value,
        "raw_value": value if raw_value is None else raw_value,
        "x": x,
        "sdofs": sdofs,
        "iota": iota,
        "G": G,
        "linear_solve_factors": linear_solve_factors,
        "success": success,
        "predictor_success": (
            success if predictor_success is None else predictor_success
        ),
        "primal_success": primal_success,
        "adjoint_linear_solve_available": adjoint_linear_solve_available,
        "newton_success": success if newton_success is None else newton_success,
        "newton_iterations": (
            missing_int if newton_iterations is None else newton_iterations
        ),
        "newton_attempted_iterations": (
            missing_int
            if newton_attempted_iterations is None
            else newton_attempted_iterations
        ),
        "newton_stop_reason_code": (
            missing_int if newton_stop_reason_code is None else newton_stop_reason_code
        ),
        "newton_last_linear_solve_success": (
            missing_bool
            if newton_last_linear_solve_success is None
            else newton_last_linear_solve_success
        ),
        "inner_penalty_residual_l2": (
            missing_float
            if inner_penalty_residual_l2 is None
            else inner_penalty_residual_l2
        ),
        "final_gradient_inf_norm": (
            missing_float
            if final_gradient_inf_norm is None
            else final_gradient_inf_norm
        ),
        "newton_trace_active": newton_bool_trace_field(newton_trace_active),
        "newton_trace_step_accepted": newton_bool_trace_field(
            newton_trace_step_accepted
        ),
        "newton_trace_linear_solve_success": newton_bool_trace_field(
            newton_trace_linear_solve_success
        ),
        "newton_trace_linear_residual_relative": newton_float_trace_field(
            newton_trace_linear_residual_relative
        ),
        "newton_trace_linear_residual_norm": newton_float_trace_field(
            newton_trace_linear_residual_norm
        ),
        "newton_trace_linear_residual_scale": newton_float_trace_field(
            newton_trace_linear_residual_scale
        ),
        "newton_trace_linear_requested_tolerance": newton_float_trace_field(
            newton_trace_linear_requested_tolerance
        ),
        "newton_trace_linear_effective_tolerance": newton_float_trace_field(
            newton_trace_linear_effective_tolerance
        ),
        "newton_trace_linear_live_operator_certificate": (
            newton_bool_trace_field(newton_trace_linear_live_operator_certificate)
        ),
        "newton_trace_linear_factorization_dtype_bits": newton_int_trace_field(
            newton_trace_linear_factorization_dtype_bits
        ),
        "newton_trace_linear_factor_application_dtype_bits": (
            newton_int_trace_field(newton_trace_linear_factor_application_dtype_bits)
        ),
        "newton_trace_linear_residual_dtype_bits": newton_int_trace_field(
            newton_trace_linear_residual_dtype_bits
        ),
        "newton_trace_certificate_value_dtype_bits": newton_int_trace_field(
            newton_trace_certificate_value_dtype_bits
        ),
        "newton_trace_certificate_gradient_dtype_bits": newton_int_trace_field(
            newton_trace_certificate_gradient_dtype_bits
        ),
        "newton_linear_solve_backend_code": _runtime_int32_scalar(
            0
            if newton_linear_solve_backend_code is None
            else newton_linear_solve_backend_code
        ),
        "newton_linear_solve_backend_code_present": _runtime_bool(backend_code_present),
    }
    for key in _TRACEABLE_NEWTON_TRACE_KEYS:
        explicit_presence = (
            None if newton_trace_presence is None else newton_trace_presence.get(key)
        )
        packed[f"{key}_present"] = _staged_like(
            value,
            trace_values[key] is not None
            if explicit_presence is None
            else explicit_presence,
            dtype=jnp.bool_,
        )
    packed["outer_raw_terms_present"] = _runtime_bool(outer_raw_terms is not None)
    missing_int64 = _staged_like(
        value,
        np.iinfo(np.int64).min,
        dtype=jnp.int64,
    )
    packed["dense_hessian_bytes"] = (
        missing_int64
        if dense_hessian_bytes is None
        else _staged_like(value, dense_hessian_bytes, dtype=jnp.int64)
    )
    packed["dense_hessian_bytes_present"] = _runtime_bool(
        dense_hessian_bytes is not None
    )
    packed["max_dense_hessian_bytes"] = (
        missing_int64
        if max_dense_hessian_bytes is None
        else _staged_like(value, max_dense_hessian_bytes, dtype=jnp.int64)
    )
    packed["max_dense_hessian_bytes_present"] = _runtime_bool(
        max_dense_hessian_bytes is not None
    )
    for term_name, _weight_key in _TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS:
        packed[f"outer_raw_term_{term_name}"] = (
            _runtime_float64_scalar(np.nan, reference=value)
            if outer_raw_terms is None
            else jnp.asarray(outer_raw_terms[term_name], dtype=jnp.float64)
        )
    return packed


def traceable_forward_result_outer_raw_terms(forward_result):
    """Return optional raw outer-term data carried by a forward result."""
    if "outer_raw_terms_present" not in forward_result:
        return None
    return (
        forward_result["outer_raw_terms_present"],
        {
            term_name: forward_result[f"outer_raw_term_{term_name}"]
            for term_name, _weight_key in _TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS
        },
    )


def _traceable_result_linear_solve_factors(solve_result, linearization_kind):
    """Return factors carried by traceable autodiff state, if this kind uses them.

    Dense factor snapshots remain reporting/parity artifacts on the JAX
    on-device LS lane. The runtime target objective keeps adjoint solves
    operator-backed so compiled value/grad paths do not stage dense LU solves.
    """
    del solve_result, linearization_kind


def _build_linear_solve_factors_from_res(res):
    """Return the public ``linear_solve_factors`` tuple for a solved ``res``.

    The compatibility-lane ``run_code()`` result dict stores the public
    ``(P, L, U)`` triple under ``res["PLU"]`` and the Phase 2 packed
    factors under ``res["LU_PIV"]``. When both are present the linear
    solve helpers receive the 5-tuple ``(P, L, U, lu, piv)`` so adjoint
    solves consume the same packed factor bytes as the forward solve.

    Host drivers that cross an evaluation boundary with these factors must
    seal them via :func:`_host_seal_linear_solve_factors_for_reuse` /
    :func:`build_exact_factor_handoff` and release them only through
    :func:`_host_require_exact_factor_handoff` /
    :func:`require_exact_factor_handoff`. This helper still returns the raw
    factor tuple for in-process JIT paths that certify against the live
    operator; it is not a substitute for the integrity-receipt consumer API.
    """
    plu = res.get("PLU")
    if plu is None:
        return None
    lu_piv = res.get("LU_PIV")
    if lu_piv is None:
        return plu
    return (plu[0], plu[1], plu[2], lu_piv[0], lu_piv[1])


def _host_seal_linear_solve_factors_for_reuse(
    factors,
    *,
    state,
    coil_dofs,
    solved_x,
    producer_graph_sha256,
):
    """Seal exported factors for cross-evaluation reuse (producer boundary).

    ``None`` stays ``None`` so reuse remains default-off when no factors were
    produced. An already-sealed :class:`ExactFactorHandoff` is returned as-is.
    Raw factor trees are sealed into an atomic integrity receipt; callers cannot
    re-affiliate unaffiliated factors after the fact without going through
    :func:`build_exact_factor_handoff`.
    """
    if factors is None:
        return None
    if isinstance(factors, ExactFactorHandoff):
        return factors
    return build_exact_factor_handoff(
        state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        factors=factors,
        producer_graph_sha256=producer_graph_sha256,
    )


def _host_seal_linear_solve_factors_from_res(
    res,
    *,
    state,
    coil_dofs,
    solved_x,
    producer_graph_sha256,
):
    """Extract public factors from a solved ``res`` and seal them for reuse."""
    return _host_seal_linear_solve_factors_for_reuse(
        _build_linear_solve_factors_from_res(res),
        state=state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        producer_graph_sha256=producer_graph_sha256,
    )


def _host_require_exact_factor_handoff(
    handoff,
    *,
    state,
    coil_dofs,
    solved_x,
    producer_graph_sha256,
):
    """Host factor-consumer boundary: release factors only after re-seal.

    This is the Python-side gate for factors that leave a producer evaluation
    and re-enter a consumer path. JIT-internal PLU refine paths that certify
    via the live operator are unchanged and do not call this helper.
    """
    return require_exact_factor_handoff(
        handoff,
        state=state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        producer_graph_sha256=producer_graph_sha256,
    )


def _host_unwrap_linear_solve_factors_for_consumption(
    factors_or_handoff,
    *,
    state,
    coil_dofs,
    solved_x,
    producer_graph_sha256,
):
    """Release sealed factors at the host handoff gate; reject raw trees.

    Host gate contract (``host_factor_handoff_gate=True``):

    * ``None`` — reuse-off / no factors; pass through.
    * :class:`ExactFactorHandoff` — re-seal integrity receipt, then release.
    * raw unaffiliated factor trees — fail closed (must seal at the producer).

    JIT-internal callers keep ``host_factor_handoff_gate=False`` and never enter
    this helper, so live-operator-certified in-process PLU refine paths remain
    numerically unchanged.
    """
    if factors_or_handoff is None:
        return None
    if isinstance(factors_or_handoff, ExactFactorHandoff):
        return _host_require_exact_factor_handoff(
            factors_or_handoff,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256=producer_graph_sha256,
        )
    raise TypeError(
        "Host factor-handoff gate rejects unaffiliated raw factors; seal with "
        "ExactFactorHandoff / build_exact_factor_handoff_for before "
        "cross-evaluation reuse."
    )


def _resolve_traceable_solved_state(
    booz_jax,
    solve_result,
    *,
    optimize_G,
    coil_set_spec,
):
    """Return solved ``(sdofs, iota, G)`` even when the solve only returns ``x``."""
    if (
        "sdofs" in solve_result
        and "iota" in solve_result
        and ("G" in solve_result or not optimize_G)
    ):
        return (
            solve_result["sdofs"],
            solve_result["iota"],
            solve_result["G"],
        )
    return booz_jax._unpack_decision_vector_jax(
        solve_result["x"],
        optimize_G,
        coil_set_spec=coil_set_spec,
    )


def _traceable_general_forward_result(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    objective_coil_dofs=None,
    certificate_coil_set_spec=None,
    certificate_coil_set_spec_from_dofs=None,
    baseline_x,
    baseline_value,
    baseline_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    optimize_G,
    baseline_coil_dofs,
    baseline_certificate_coil_dofs=None,
    predictor_kind,
    objective_kwargs,
    success_filter,
    newton_trace_capacity: int,
):
    """Run the general traceable inner solve without the baseline fast path."""
    objective_coil_dofs = (
        coil_dofs if objective_coil_dofs is None else objective_coil_dofs
    )
    coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
    certificate_coil_set_spec = (
        coil_set_spec
        if certificate_coil_set_spec is None
        else certificate_coil_set_spec
    )
    warmstart_x, warmstart_linear_solve_success = _traceable_predict_warmstart_x(
        booz_jax,
        coil_set_spec_from_dofs,
        certificate_coil_set_spec_from_dofs=(certificate_coil_set_spec_from_dofs),
        baseline_certificate_coil_dofs=baseline_certificate_coil_dofs,
        coil_dofs=coil_dofs,
        baseline_coil_dofs=baseline_coil_dofs,
        baseline_x=baseline_x,
        baseline_linear_solve_factors=baseline_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        predictor_kind=predictor_kind,
        objective_kwargs=objective_kwargs,
    )

    def _run_traceable_solve(initial_x, predictor_success):
        warmstart_sdofs, warmstart_iota, warmstart_G = (
            booz_jax._unpack_decision_vector_jax(
                initial_x,
                optimize_G,
                coil_set_spec=coil_set_spec,
            )
        )
        solve_result = booz_jax.run_code_traceable(
            coil_set_spec,
            warmstart_sdofs,
            warmstart_iota,
            warmstart_G,
            certificate_coil_source=certificate_coil_set_spec,
            materialize_dense_linearization=False,
        )
        solved_sdofs, solved_iota, solved_G = _resolve_traceable_solved_state(
            booz_jax,
            solve_result,
            optimize_G=optimize_G,
            coil_set_spec=certificate_coil_set_spec,
        )
        primal_success = solve_result["primal_success"]
        adjoint_linear_solve_available = solve_result["adjoint_linear_solve_available"]
        success = primal_success
        if success_filter is not None:
            success = success & jax.lax.cond(
                primal_success,
                lambda _: success_filter(objective_coil_dofs, solve_result["x"]),
                lambda _: _runtime_bool(False),
                operand=None,
            )
        objective_value, outer_raw_terms = (
            _evaluate_traceable_total_objective_with_raw_terms(
                solve_result["x"],
                objective_coil_dofs,
                certificate_coil_set_spec,
                objective_kwargs,
            )
        )
        filtered_objective_value = jax.lax.cond(
            success,
            lambda _: objective_value,
            lambda _: _traceable_rejected_objective_value(
                objective_value,
                baseline_value,
            ),
            operand=None,
        )
        return _pack_traceable_forward_result(
            value=filtered_objective_value,
            raw_value=objective_value,
            x=solve_result["x"],
            sdofs=solved_sdofs,
            iota=solved_iota,
            G=solved_G,
            linear_solve_factors=_traceable_result_linear_solve_factors(
                solve_result,
                linearization_kind,
            ),
            success=success,
            predictor_success=predictor_success,
            primal_success=primal_success,
            adjoint_linear_solve_available=adjoint_linear_solve_available,
            newton_success=solve_result.get("success", primal_success),
            newton_iterations=solve_result.get(
                "newton_iter",
                solve_result.get("nit", _runtime_int32_scalar(0)),
            ),
            newton_attempted_iterations=solve_result.get("newton_attempted_iterations"),
            newton_stop_reason_code=solve_result.get("newton_stop_reason_code"),
            newton_last_linear_solve_success=solve_result.get(
                "newton_last_linear_solve_success"
            ),
            inner_penalty_residual_l2=solve_result.get("inner_penalty_residual_l2"),
            final_gradient_inf_norm=solve_result.get("final_gradient_inf_norm"),
            newton_trace_capacity=newton_trace_capacity,
            newton_trace_active=solve_result.get("newton_trace_active"),
            newton_trace_step_accepted=solve_result.get("newton_trace_step_accepted"),
            newton_trace_linear_solve_success=solve_result.get(
                "newton_trace_linear_solve_success"
            ),
            newton_trace_linear_residual_relative=solve_result.get(
                "newton_trace_linear_residual_relative"
            ),
            newton_trace_linear_residual_norm=solve_result.get(
                "newton_trace_linear_residual_norm"
            ),
            newton_trace_linear_residual_scale=solve_result.get(
                "newton_trace_linear_residual_scale"
            ),
            newton_trace_linear_requested_tolerance=solve_result.get(
                "newton_trace_linear_requested_tolerance"
            ),
            newton_trace_linear_effective_tolerance=solve_result.get(
                "newton_trace_linear_effective_tolerance"
            ),
            newton_trace_linear_live_operator_certificate=solve_result.get(
                "newton_trace_linear_live_operator_certificate"
            ),
            newton_trace_linear_factorization_dtype_bits=solve_result.get(
                "newton_trace_linear_factorization_dtype_bits"
            ),
            newton_trace_linear_factor_application_dtype_bits=solve_result.get(
                "newton_trace_linear_factor_application_dtype_bits"
            ),
            newton_trace_linear_residual_dtype_bits=solve_result.get(
                "newton_trace_linear_residual_dtype_bits"
            ),
            newton_trace_certificate_value_dtype_bits=solve_result.get(
                "newton_trace_certificate_value_dtype_bits"
            ),
            newton_trace_certificate_gradient_dtype_bits=solve_result.get(
                "newton_trace_certificate_gradient_dtype_bits"
            ),
            newton_trace_presence={
                key: solve_result.get(f"{key}_present")
                for key in _TRACEABLE_NEWTON_TRACE_KEYS
            },
            newton_linear_solve_backend_code=solve_result.get(
                "newton_linear_solve_backend_code"
            ),
            dense_hessian_bytes=solve_result.get("dense_hessian_bytes"),
            max_dense_hessian_bytes=solve_result.get("max_dense_hessian_bytes"),
            outer_raw_terms=outer_raw_terms,
        )

    if linearization_kind != "exact_jacobian":
        return _run_traceable_solve(
            warmstart_x,
            warmstart_linear_solve_success,
        )

    return lax.cond(
        warmstart_linear_solve_success,
        lambda _: _run_traceable_solve(
            warmstart_x,
            _runtime_bool(True),
        ),
        lambda _: _run_traceable_solve(
            baseline_x,
            _runtime_bool(False),
        ),
        operand=None,
    )


def _traceable_forward_result(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    objective_coil_dofs=None,
    certificate_coil_set_spec=None,
    certificate_coil_set_spec_from_dofs=None,
    baseline_x,
    baseline_value,
    baseline_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    optimize_G,
    baseline_coil_dofs,
    baseline_objective_coil_dofs=None,
    predictor_kind,
    objective_kwargs,
    success_filter,
    newton_trace_capacity: int,
):
    """Run the pure traceable inner solve and return value plus solver data."""
    objective_coil_dofs = (
        coil_dofs if objective_coil_dofs is None else objective_coil_dofs
    )
    baseline_objective_coil_dofs = (
        baseline_coil_dofs
        if baseline_objective_coil_dofs is None
        else baseline_objective_coil_dofs
    )
    same_coils = jnp.all(objective_coil_dofs == baseline_objective_coil_dofs)

    def baseline_case(_):
        baseline_sdofs, baseline_iota, baseline_G = _split_x_inner_runtime(
            baseline_x,
            optimize_G,
        )
        return _pack_traceable_forward_result(
            # The exact baseline state must return the solved reference objective so
            # the outer optimizer can obtain a real descent direction even when the
            # seed is hardware-invalid. Candidate (non-baseline) states remain
            # subject to the hard success filter below.
            value=baseline_value,
            raw_value=baseline_value,
            x=baseline_x,
            sdofs=baseline_sdofs,
            iota=baseline_iota,
            G=baseline_G,
            linear_solve_factors=baseline_linear_solve_factors,
            success=_runtime_bool(True),
            predictor_success=_runtime_bool(True),
            primal_success=_runtime_bool(True),
            adjoint_linear_solve_available=_runtime_bool(False),
            newton_success=_runtime_bool(True),
            newton_iterations=_runtime_int32_scalar(0),
            newton_trace_capacity=newton_trace_capacity,
        )

    def general_case(_):
        return _traceable_general_forward_result(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            objective_coil_dofs=objective_coil_dofs,
            certificate_coil_set_spec=certificate_coil_set_spec,
            certificate_coil_set_spec_from_dofs=(certificate_coil_set_spec_from_dofs),
            baseline_x=baseline_x,
            baseline_value=baseline_value,
            baseline_linear_solve_factors=baseline_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            optimize_G=optimize_G,
            baseline_coil_dofs=baseline_coil_dofs,
            baseline_certificate_coil_dofs=baseline_objective_coil_dofs,
            predictor_kind=predictor_kind,
            objective_kwargs=objective_kwargs,
            success_filter=success_filter,
            newton_trace_capacity=newton_trace_capacity,
        )

    return jax.lax.cond(same_coils, baseline_case, general_case, operand=None)


def _traceable_total_gradient(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    solved_x,
    solved_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    objective_kwargs,
):
    """Implicit total derivative of the pure traceable objective."""
    return _traceable_objective_gradient_parts(
        booz_jax,
        coil_set_spec_from_dofs,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        solved_linear_solve_factors=solved_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        objective_kwargs=objective_kwargs,
    )[2]


def _traceable_adjoint_gradient_or_nan(gradient, linear_solve_success):
    """Surface adjoint-solve failures as non-finite gradients, not fallbacks."""
    failure_gradient = _traceable_adjoint_fail_gradient_like(gradient)
    success = jnp.asarray(linear_solve_success)
    return jax.tree.map(
        lambda valid_gradient, invalid_gradient: jnp.where(
            success,
            valid_gradient,
            invalid_gradient,
        ),
        gradient,
        failure_gradient,
    )


def _traceable_total_gradient_with_status(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    solved_x,
    solved_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    objective_kwargs,
    scalar_objective_fn=None,
    certificate_probe_key=None,
):
    _, _, total_grad, linear_solve_success, _ = _traceable_objective_gradient_parts(
        booz_jax,
        coil_set_spec_from_dofs,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        solved_linear_solve_factors=solved_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        objective_kwargs=objective_kwargs,
        scalar_objective_fn=scalar_objective_fn,
        certificate_probe_key=certificate_probe_key,
    )
    return total_grad, linear_solve_success


def _traceable_total_gradient_with_trust(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    solved_x,
    solved_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    objective_kwargs,
    certificate_probe_key,
):
    """Return the total gradient and its mixed dense-IR certificate evidence."""
    _, _, total_grad, linear_solve_success, trust = _traceable_objective_gradient_parts(
        booz_jax,
        coil_set_spec_from_dofs,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        solved_linear_solve_factors=solved_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        objective_kwargs=objective_kwargs,
        certificate_probe_key=certificate_probe_key,
    )
    return total_grad, linear_solve_success, trust


def _traceable_adjoint_rhs_exactly_zero(rhs):
    """Test exact zero entirely on device without staging a scalar constant."""
    rhs = jnp.asarray(rhs)
    return jnp.logical_not(jnp.any(rhs))


def _traceable_objective_gradient_parts(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    solved_x,
    solved_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    objective_kwargs,
    term_name=None,
    scalar_objective_fn=None,
    certificate_probe_key=None,
):
    """Return FP64-certificate gradients for one traceable objective."""
    if scalar_objective_fn is not None and term_name is not None:
        raise ValueError(
            "scalar_objective_fn and term_name are mutually exclusive traceable "
            "gradient selectors."
        )

    coil_dofs = _as_jax_float64(coil_dofs)
    solved_x = _as_jax_float64(solved_x)
    inactive_mixed_dense_ir_trust = _dense_ir._inactive_mixed_dense_ir_trust_telemetry(
        solved_x
    )

    def _evaluate_objective(x_inner, current_coil_dofs, coil_set_spec):
        if scalar_objective_fn is not None:
            return scalar_objective_fn(
                x_inner,
                current_coil_dofs,
                coil_set_spec,
                objective_kwargs=objective_kwargs,
            )
        if term_name is None:
            return _evaluate_traceable_total_objective(
                x_inner,
                current_coil_dofs,
                coil_set_spec,
                objective_kwargs,
            )
        return _evaluate_traceable_weighted_single_stage_outer_term(
            term_name,
            x_inner,
            current_coil_dofs,
            coil_set_spec,
            objective_kwargs,
        )

    def _evaluate_objective_of_coils(current_coil_dofs):
        return _evaluate_objective(
            solved_x,
            current_coil_dofs,
            coil_set_spec_from_dofs(current_coil_dofs),
        )

    coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
    depends_on_x_inner = True
    depends_on_coil_dofs = True
    if scalar_objective_fn is None:
        depends_on_x_inner, depends_on_coil_dofs = (
            _traceable_single_stage_effective_dependency_flags(
                term_name,
                objective_kwargs=objective_kwargs,
            )
        )

    direct_grad = None
    if not depends_on_x_inner:
        dJ_dx = _runtime_zeros_like(solved_x)
        adjoint = _runtime_zeros_like(solved_x)
        linear_solve_success = _runtime_bool(True)
    else:
        if depends_on_coil_dofs:

            def _evaluate_objective_of_x_and_coils(
                current_x_inner,
                current_coil_dofs,
            ):
                return _evaluate_objective(
                    current_x_inner,
                    current_coil_dofs,
                    coil_set_spec_from_dofs(current_coil_dofs),
                )

            objective_value, pullback = jax.vjp(
                _evaluate_objective_of_x_and_coils,
                solved_x,
                coil_dofs,
            )
            dJ_dx, direct_grad = pullback(
                _explicit_scalar_pullback_seed(objective_value)
            )
        else:
            dJ_dx = _strict_scalar_grad(
                lambda x: _evaluate_objective(x, coil_dofs, coil_set_spec),
                solved_x,
            )

        def zero_adjoint(_):
            return (
                _runtime_zeros_like(solved_x),
                _runtime_bool(True),
                inactive_mixed_dense_ir_trust,
            )

        def solve_adjoint(_):
            adjoint_value, linear_solve_status = _traceable_solve_linearization(
                booz_jax,
                solved_x,
                dJ_dx,
                coil_set_spec,
                objective_kwargs,
                linear_solve_factors=solved_linear_solve_factors,
                linearization_kind=linearization_kind,
                linear_solve_tol=linear_solve_tol,
                linear_solve_stab=linear_solve_stab,
                transpose=True,
                certificate_probe_key=certificate_probe_key,
            )
            adjoint_value = jnp.asarray(adjoint_value, dtype=solved_x.dtype)
            trust = (
                linear_solve_status.trust
                if isinstance(
                    linear_solve_status,
                    _dense_ir._MixedDenseIrSolveStatus,
                )
                else inactive_mixed_dense_ir_trust
            )
            return (
                adjoint_value,
                _linear_solve._linear_solve_status_success(linear_solve_status),
                trust,
            )

        adjoint, linear_solve_success, mixed_dense_ir_trust = lax.cond(
            _traceable_adjoint_rhs_exactly_zero(dJ_dx),
            zero_adjoint,
            solve_adjoint,
            operand=None,
        )

    if not depends_on_x_inner:
        mixed_dense_ir_trust = inactive_mixed_dense_ir_trust

    if not depends_on_coil_dofs:
        # Some diagnostic terms depend only on the solved inner state, so
        # their explicit coil derivative is exactly zero. Avoid autodiff on
        # these constant-in-coils scalars under strict transfer guard because
        # null tangent paths instantiate host scalar zeros.
        direct_grad = _runtime_zeros_like(coil_dofs)
    elif direct_grad is None:
        direct_grad = _strict_scalar_grad(_evaluate_objective_of_coils, coil_dofs)

    if not depends_on_x_inner:
        implicit_grad = _runtime_zeros_like(coil_dofs)
        return (
            direct_grad,
            implicit_grad,
            direct_grad,
            linear_solve_success,
            mixed_dense_ir_trust,
        )

    if linearization_kind == "exact_jacobian":

        def directional_stationarity_of_coils(current_coil_dofs):
            return _traceable_directional_exact_residual(
                solved_x,
                adjoint,
                coil_set_spec_from_dofs(current_coil_dofs),
                objective_kwargs,
            )

    else:
        inner_objective_kwargs = _traceable_inner_objective_kwargs(objective_kwargs)

        def directional_stationarity_of_coils(current_coil_dofs):
            return _traceable_directional_inner_stationarity(
                solved_x,
                adjoint,
                coil_set_spec_from_dofs(current_coil_dofs),
                **inner_objective_kwargs,
            )

    implicit_grad = _strict_scalar_grad(
        directional_stationarity_of_coils,
        coil_dofs,
    )
    total_grad = _traceable_adjoint_gradient_or_nan(
        direct_grad - implicit_grad,
        linear_solve_success,
    )
    return (
        direct_grad,
        implicit_grad,
        total_grad,
        linear_solve_success,
        mixed_dense_ir_trust,
    )


def _traceable_predict_warmstart_result_from_anchor(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    certificate_coil_set_spec_from_dofs=None,
    anchor_certificate_coil_dofs=None,
    coil_dofs,
    anchor_coil_dofs,
    anchor_x,
    anchor_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    predictor_kind,
    objective_kwargs,
    predictor_coil_use_compute_dtype=True,
    predictor_state_use_compute_dtype=False,
):
    """Predict a warm start and retain its linear-solve certificate."""
    anchor_certificate_coil_dofs = _as_jax_float64(
        anchor_coil_dofs
        if anchor_certificate_coil_dofs is None
        else anchor_certificate_coil_dofs
    )
    predictor_anchor_coil_dofs = (
        _as_compute_array(anchor_coil_dofs)
        if predictor_coil_use_compute_dtype
        else _as_jax_float64(anchor_coil_dofs)
    )
    predictor_coil_dofs = (
        _as_compute_array(coil_dofs)
        if predictor_coil_use_compute_dtype
        else _as_jax_float64(coil_dofs)
    )
    delta = predictor_coil_dofs - predictor_anchor_coil_dofs

    if predictor_kind == "exact":
        exact_residual_kwargs = _traceable_exact_residual_kwargs(objective_kwargs)

        def anchor_residual_of_coils(cd):
            return _boozer_exact_residual(
                anchor_x,
                coil_set_spec=coil_set_spec_from_dofs(cd),
                **exact_residual_kwargs,
            )

        forcing = jax.jvp(
            anchor_residual_of_coils,
            (predictor_anchor_coil_dofs,),
            (delta,),
        )[1]
    else:
        inner_objective_kwargs = _traceable_inner_objective_kwargs(objective_kwargs)
        forcing = _traceable_inner_stationarity_coil_jvp(
            _as_compute_array(anchor_x)
            if predictor_state_use_compute_dtype
            else anchor_x,
            predictor_anchor_coil_dofs,
            delta,
            coil_set_spec_from_dofs,
            **inner_objective_kwargs,
        )

    live_coil_set_spec_from_dofs = (
        coil_set_spec_from_dofs
        if certificate_coil_set_spec_from_dofs is None
        else certificate_coil_set_spec_from_dofs
    )
    dx, linear_solve_status = _traceable_solve_linearization(
        booz_jax,
        _as_jax_float64(anchor_x),
        _as_jax_float64(-forcing),
        live_coil_set_spec_from_dofs(anchor_certificate_coil_dofs),
        objective_kwargs,
        linear_solve_factors=anchor_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        transpose=False,
    )
    linear_solve_success = _linear_solve._linear_solve_status_success(
        linear_solve_status
    )
    dx = jnp.asarray(dx, dtype=anchor_x.dtype)
    predicted_x = anchor_x + dx
    preserve_failed_predictor = (
        predictor_kind == "exact" or linearization_kind == "exact_jacobian"
    )
    if preserve_failed_predictor:
        return forcing, predicted_x, linear_solve_status, linear_solve_success
    return (
        forcing,
        lax.cond(
            linear_solve_success,
            lambda _: predicted_x,
            lambda _: anchor_x,
            operand=None,
        ),
        linear_solve_status,
        linear_solve_success,
    )


def _traceable_predict_warmstart_from_anchor(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    certificate_coil_set_spec_from_dofs=None,
    anchor_certificate_coil_dofs=None,
    coil_dofs,
    anchor_coil_dofs,
    anchor_x,
    anchor_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    predictor_kind,
    objective_kwargs,
):
    """Predict a warm start from one caller-authorized solved anchor."""
    _, predicted_x, _, linear_solve_success = (
        _traceable_predict_warmstart_result_from_anchor(
            booz_jax,
            coil_set_spec_from_dofs,
            certificate_coil_set_spec_from_dofs=(certificate_coil_set_spec_from_dofs),
            anchor_certificate_coil_dofs=anchor_certificate_coil_dofs,
            coil_dofs=coil_dofs,
            anchor_coil_dofs=anchor_coil_dofs,
            anchor_x=anchor_x,
            anchor_linear_solve_factors=anchor_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            predictor_kind=predictor_kind,
            objective_kwargs=objective_kwargs,
        )
    )
    return predicted_x, linear_solve_success


def _traceable_select_predictor_linear_solve_factors(
    anchor_eligible,
    *,
    baseline_linear_solve_factors,
    anchor_linear_solve_factors,
):
    """Select anchor factors only after the caller authorizes the anchor."""
    if baseline_linear_solve_factors is None:
        return None
    return jax.tree.map(
        lambda anchor_value, baseline_value: lax.select(
            anchor_eligible,
            anchor_value,
            baseline_value,
        ),
        anchor_linear_solve_factors,
        baseline_linear_solve_factors,
    )


def _traceable_predict_warmstart_x(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    certificate_coil_set_spec_from_dofs=None,
    baseline_certificate_coil_dofs=None,
    coil_dofs,
    baseline_coil_dofs,
    baseline_x,
    baseline_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    predictor_kind,
    objective_kwargs,
):
    """Predict a coil-dependent warm start from the immutable baseline."""
    return _traceable_predict_warmstart_from_anchor(
        booz_jax,
        coil_set_spec_from_dofs,
        certificate_coil_set_spec_from_dofs=(certificate_coil_set_spec_from_dofs),
        anchor_certificate_coil_dofs=baseline_certificate_coil_dofs,
        coil_dofs=coil_dofs,
        anchor_coil_dofs=baseline_coil_dofs,
        anchor_x=baseline_x,
        anchor_linear_solve_factors=baseline_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        predictor_kind=predictor_kind,
        objective_kwargs=objective_kwargs,
    )


def _build_traceable_objective_cache_state(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    require_ondevice_inner=True,
):
    """Return the cheap traceable-runtime state needed for cache-key checks."""
    objective_method = None
    if booz_jax.boozer_type == "ls":
        objective_method = booz_jax._resolve_optimizer_method()
        if (
            require_ondevice_inner
            and objective_method not in _ONDEVICE_OPTIMIZER_METHODS
        ):
            raise RuntimeError(
                "make_traceable_objective() requires an on-device optimizer method; "
                f"got {objective_method!r}."
            )

    solved_state = _resolved_boozer_solved_runtime_state(booz_jax)
    warmstart_sdofs = solved_state.sdofs
    warmstart_iota = solved_state.iota
    warmstart_G = solved_state.G

    baseline_coil_dofs = _as_jax_float64(bs_jax.x.copy())
    coil_dof_extraction_spec = _traceable_runtime_hostify_tree(
        bs_jax.coil_dof_extraction_spec()
    )
    coil_layout_signature = _traceable_contract_tree_signature(coil_dof_extraction_spec)
    coil_set_spec_from_dofs = lambda coil_dofs: coil_set_spec_from_dof_extraction_spec(
        coil_dof_extraction_spec,
        coil_dofs,
    )
    optimize_G = warmstart_G is not None
    predictor_kind = booz_jax.boozer_type
    solve_quadpoints_phi = _as_jax_float64(booz_jax.quadpoints_phi)
    solve_quadpoints_theta = _as_jax_float64(booz_jax.quadpoints_theta)
    label_quadpoints_phi = _as_jax_float64(booz_jax.label_quadpoints_phi)
    label_quadpoints_theta = _as_jax_float64(booz_jax.label_quadpoints_theta)
    exact_quadpoints_phi, exact_quadpoints_theta, mask_indices = (
        _canonicalize_traceable_exact_quadrature(booz_jax)
    )
    objective_kwargs = {
        "quadpoints_phi": solve_quadpoints_phi,
        "quadpoints_theta": solve_quadpoints_theta,
        "mpol": booz_jax.mpol,
        "ntor": booz_jax.ntor,
        "nfp": booz_jax.nfp,
        "stellsym": booz_jax.stellsym,
        "scatter_indices": booz_jax.scatter_indices,
        "surface_kind": booz_jax._surface_geometry_kind,
        "label_quadpoints_phi": label_quadpoints_phi,
        "label_quadpoints_theta": label_quadpoints_theta,
        "label_mpol": booz_jax.label_mpol,
        "label_ntor": booz_jax.label_ntor,
        "label_nfp": booz_jax.label_nfp,
        "label_stellsym": booz_jax.label_stellsym,
        "label_scatter_indices": booz_jax.label_scatter_indices,
        "label_surface_kind": booz_jax._label_surface_geometry_kind,
        "optimize_G": optimize_G,
        "weight_inv_modB": solved_state.weight_inv_modB,
        "constraint_weight": booz_jax.constraint_weight,
        "targetlabel": booz_jax.targetlabel,
        "label_type": booz_jax.label_type,
        "phi_idx": booz_jax.phi_idx,
        "iota_target": _as_jax_float64(iota_target),
        "exact_quadpoints_phi": exact_quadpoints_phi,
        "exact_quadpoints_theta": exact_quadpoints_theta,
        "surface_quadpoints_phi": _as_jax_float64(booz_jax.surface.quadpoints_phi),
        "surface_quadpoints_theta": _as_jax_float64(booz_jax.surface.quadpoints_theta),
        "coil_dof_extraction_spec": coil_dof_extraction_spec,
        "outer_objective_config": outer_objective_config,
        "mask_indices": mask_indices,
        "stellsym_surface": booz_jax.stellsym,
    }
    linearization_kind = booz_jax.res["linearization_kind"]
    baseline_linear_solve_factors = None
    linear_solve_tol = booz_jax._linear_solve_tolerance()
    linear_solve_stab = float(booz_jax.options.get("newton_stab", 0.0))
    newton_trace_capacity = booz_jax.traceable_newton_trace_capacity(objective_method)

    return {
        "objective_kwargs": objective_kwargs,
        "warmstart_sdofs": warmstart_sdofs,
        "warmstart_iota": warmstart_iota,
        "warmstart_G": warmstart_G,
        "baseline_linear_solve_factors": baseline_linear_solve_factors,
        "baseline_coil_dofs": baseline_coil_dofs,
        "coil_dof_extraction_spec": coil_dof_extraction_spec,
        "coil_set_spec_from_dofs": coil_set_spec_from_dofs,
        "solve_state_token": booz_jax._traceable_solve_state_token,
        "coil_dof_state_token": bs_jax._coil_dof_state_token,
        "coil_layout_signature": coil_layout_signature,
        "optimize_G": optimize_G,
        "predictor_kind": predictor_kind,
        "objective_method": objective_method,
        "linearization_kind": linearization_kind,
        "linear_solve_tol": linear_solve_tol,
        "linear_solve_stab": linear_solve_stab,
        "newton_trace_capacity": newton_trace_capacity,
    }


def _materialize_traceable_objective_state(booz_jax, bs_jax, cache_state):
    """Add baseline value/runtime constants after the runtime cache misses."""
    baseline_x = booz_jax._pack_decision_vector(
        cache_state["warmstart_iota"],
        cache_state["warmstart_G"],
        sdofs=cache_state["warmstart_sdofs"],
    )
    baseline_coil_dofs = cache_state["baseline_coil_dofs"]
    objective_kwargs = cache_state["objective_kwargs"]

    baseline_value = _evaluate_traceable_total_objective(
        baseline_x,
        baseline_coil_dofs,
        bs_jax.coil_set_spec_from_dofs(baseline_coil_dofs),
        objective_kwargs,
    )
    return {
        "objective_kwargs": _traceable_runtime_hostify_tree(objective_kwargs),
        "baseline_x": _traceable_runtime_hostify_tree(baseline_x),
        "baseline_value": _traceable_runtime_hostify_tree(baseline_value),
        "baseline_linear_solve_factors": _traceable_runtime_hostify_tree(
            cache_state["baseline_linear_solve_factors"]
        ),
        "baseline_coil_dofs": _traceable_runtime_hostify_tree(baseline_coil_dofs),
        "coil_dof_extraction_spec": cache_state["coil_dof_extraction_spec"],
        "coil_set_spec_from_dofs": cache_state["coil_set_spec_from_dofs"],
        "solve_state_token": cache_state["solve_state_token"],
        "coil_dof_state_token": cache_state["coil_dof_state_token"],
        "coil_layout_signature": cache_state["coil_layout_signature"],
        "optimize_G": cache_state["optimize_G"],
        "predictor_kind": cache_state["predictor_kind"],
        "objective_method": cache_state["objective_method"],
        "linearization_kind": cache_state["linearization_kind"],
        "linear_solve_tol": cache_state["linear_solve_tol"],
        "linear_solve_stab": cache_state["linear_solve_stab"],
        "newton_trace_capacity": cache_state["newton_trace_capacity"],
    }


def _build_traceable_objective_state(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
):
    """Return the shared state used by the traceable objective builders.

    This setup reads the solved mutable object state once, computes the solved
    baseline objective in JAX, then explicitly hostifies the captured runtime
    constants before building the compiled target-lane closures. The resulting
    closures stay pure in the hot path without capturing device-backed arrays
    that would trip strict transfer-guard lowering.
    """
    cache_state = _build_traceable_objective_cache_state(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
    )
    return _materialize_traceable_objective_state(booz_jax, bs_jax, cache_state)


def _traceable_runtime_reject_host_input(coil_dofs, entrypoint_name):
    if isinstance(coil_dofs, (np.ndarray, np.generic, list, tuple, float, int)):
        raise RuntimeError(
            f"{entrypoint_name} requires a JAX array. Host inputs must enter "
            "through an explicit staging boundary; transfer_guard=disallow "
            "rejects implicit host-to-device transfer."
        )
    return coil_dofs


def _make_traceable_runtime_jax_array_boundary(compiled_callable, entrypoint_name):
    def boundary(coil_dofs):
        return compiled_callable(
            _traceable_runtime_reject_host_input(coil_dofs, entrypoint_name)
        )

    return boundary


def _build_traceable_objective_compiled_bundle_from_state(
    booz_jax,
    state,
    *,
    success_filter=None,
    general_only_forward=False,
):
    """Build shared compiled closures for one traceable single-stage state."""
    objective_kwargs = state["objective_kwargs"]
    baseline_x = state["baseline_x"]
    baseline_value = state["baseline_value"]
    baseline_linear_solve_factors = state["baseline_linear_solve_factors"]
    baseline_coil_dofs = state["baseline_coil_dofs"]
    optimize_G = state["optimize_G"]
    predictor_kind = state["predictor_kind"]
    linearization_kind = state["linearization_kind"]
    linear_solve_tol = state["linear_solve_tol"]
    linear_solve_stab = state["linear_solve_stab"]
    newton_trace_capacity = state["newton_trace_capacity"]
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]

    def _general_forward_result_for(
        coil_dofs,
        anchor_coil_dofs,
        anchor_x,
        anchor_value,
    ):
        objective_coil_dofs = _as_jax_float64(coil_dofs)
        proposal_coil_dofs = _as_compute_array(objective_coil_dofs)
        proposal_anchor_coil_dofs = _as_compute_array(anchor_coil_dofs)
        certificate_coil_set_spec = coil_set_spec_from_dofs(objective_coil_dofs)
        return _traceable_general_forward_result(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=proposal_coil_dofs,
            objective_coil_dofs=objective_coil_dofs,
            certificate_coil_set_spec=certificate_coil_set_spec,
            certificate_coil_set_spec_from_dofs=coil_set_spec_from_dofs,
            baseline_x=anchor_x,
            baseline_value=_as_jax_float64(anchor_value),
            baseline_linear_solve_factors=baseline_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            optimize_G=optimize_G,
            baseline_coil_dofs=proposal_anchor_coil_dofs,
            baseline_certificate_coil_dofs=anchor_coil_dofs,
            predictor_kind=predictor_kind,
            objective_kwargs=objective_kwargs,
            success_filter=success_filter,
            newton_trace_capacity=newton_trace_capacity,
        )

    def _forward_result_for(coil_dofs):
        if general_only_forward:
            return _general_forward_result_for(
                coil_dofs,
                baseline_coil_dofs,
                baseline_x,
                baseline_value,
            )
        objective_coil_dofs = _as_jax_float64(coil_dofs)
        proposal_coil_dofs = _as_compute_array(objective_coil_dofs)
        proposal_baseline_coil_dofs = _as_compute_array(baseline_coil_dofs)
        certificate_coil_set_spec = coil_set_spec_from_dofs(objective_coil_dofs)
        return _traceable_forward_result(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=proposal_coil_dofs,
            objective_coil_dofs=objective_coil_dofs,
            certificate_coil_set_spec=certificate_coil_set_spec,
            certificate_coil_set_spec_from_dofs=coil_set_spec_from_dofs,
            baseline_x=baseline_x,
            baseline_value=_as_jax_float64(baseline_value),
            baseline_linear_solve_factors=baseline_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            optimize_G=optimize_G,
            baseline_coil_dofs=proposal_baseline_coil_dofs,
            baseline_objective_coil_dofs=baseline_coil_dofs,
            predictor_kind=predictor_kind,
            objective_kwargs=objective_kwargs,
            success_filter=success_filter,
            newton_trace_capacity=newton_trace_capacity,
        )

    def _forward_result_from_anchor_for(
        coil_dofs,
        incumbent: TraceableObjectiveInnerState,
    ):
        anchor_eligible = jnp.asarray(incumbent.eligible, dtype=jnp.bool_)
        anchor_coil_dofs = lax.select(
            anchor_eligible,
            _as_jax_float64(incumbent.coil_dofs),
            _as_jax_float64(baseline_coil_dofs),
        )
        anchor_x = lax.select(
            anchor_eligible,
            _as_jax_float64(incumbent.solved_x),
            _as_jax_float64(baseline_x),
        )
        anchor_value = lax.select(
            anchor_eligible,
            _as_jax_float64(incumbent.objective_value),
            _as_jax_float64(baseline_value),
        )
        return _general_forward_result_for(
            coil_dofs,
            anchor_coil_dofs,
            anchor_x,
            anchor_value,
        )

    jitted_forward_result_for = jax.jit(_forward_result_for)
    jitted_forward_result_from_anchor_for = jax.jit(
        _forward_result_from_anchor_for
    )

    def _total_gradient_for(coil_dofs, solved_x, solved_linear_solve_factors):
        return _traceable_total_gradient_with_status(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_linear_solve_factors=_traceable_runtime_deviceify_tree(
                solved_linear_solve_factors
            ),
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            objective_kwargs=objective_kwargs,
        )

    def _build_compiled_total_gradient_for():
        return jax.jit(_total_gradient_for)

    def _total_gradient_for_with_certificate_key(
        coil_dofs,
        solved_x,
        solved_linear_solve_factors,
        certificate_probe_key,
    ):
        return _traceable_total_gradient_with_trust(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_linear_solve_factors=_traceable_runtime_deviceify_tree(
                solved_linear_solve_factors
            ),
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            objective_kwargs=objective_kwargs,
            certificate_probe_key=certificate_probe_key,
        )

    def _build_compiled_total_gradient_for_with_certificate_key():
        return jax.jit(_total_gradient_for_with_certificate_key)

    def _build_value_and_grad_for(compiled_total_gradient_for):
        def _value_and_grad_for(coil_dofs):
            result = jitted_forward_result_for(coil_dofs)

            def _total_gradient_at(
                candidate_coil_dofs,
                candidate_x,
                linear_solve_factors,
            ):
                return compiled_total_gradient_for(
                    candidate_coil_dofs,
                    candidate_x,
                    linear_solve_factors,
                )

            def _accepted_candidate_gradient(_):
                return _total_gradient_at(
                    coil_dofs,
                    result["x"],
                    result["linear_solve_factors"],
                )

            def _rejected_candidate_gradient(_):
                return lax.cond(
                    result["primal_success"],
                    lambda _: _total_gradient_at(
                        coil_dofs,
                        result["x"],
                        result["linear_solve_factors"],
                    ),
                    lambda _: _total_gradient_at(
                        baseline_coil_dofs,
                        baseline_x,
                        baseline_linear_solve_factors,
                    ),
                    operand=None,
                )

            grad, linear_solve_success = lax.cond(
                result["success"],
                _accepted_candidate_gradient,
                _rejected_candidate_gradient,
                operand=None,
            )
            return result["value"], _traceable_adjoint_gradient_or_nan(
                grad,
                linear_solve_success,
            )

        return _value_and_grad_for

    jit_kwargs = {}
    if get_backend_policy().supports_buffer_donation:
        jit_kwargs["donate_argnums"] = (0,)

    def _build_compiled_value_and_grad_for(compiled_total_gradient_for):
        jitted_value_and_grad_for = jax.jit(
            _build_value_and_grad_for(compiled_total_gradient_for),
            **jit_kwargs,
        )
        return mark_cacheable_jit_value_and_grad(
            _make_traceable_runtime_jax_array_boundary(
                jitted_value_and_grad_for,
                "compiled_value_and_grad_for",
            )
        )

    compiled_forward_result_for = _make_traceable_runtime_jax_array_boundary(
        jitted_forward_result_for,
        "compiled_forward_result_for",
    )
    if general_only_forward:
        lazy_state = _LazyCompiledGradientState()

        def _lazy_compiled_total_gradient_for(*args):
            if lazy_state.total_gradient is None:
                with lazy_state.gradient_lock:
                    if lazy_state.total_gradient is None:
                        lazy_state.total_gradient = _build_compiled_total_gradient_for()
            return lazy_state.total_gradient(*args)

        def _lazy_compiled_total_gradient_for_with_certificate_key(*args):
            if lazy_state.total_gradient_with_key is None:
                with lazy_state.keyed_gradient_lock:
                    if lazy_state.total_gradient_with_key is None:
                        lazy_state.total_gradient_with_key = (
                            _build_compiled_total_gradient_for_with_certificate_key()
                        )
            return lazy_state.total_gradient_with_key(*args)

        def _lazy_compiled_value_and_grad_for(coil_dofs):
            if lazy_state.value_and_grad is None:
                with lazy_state.value_and_grad_lock:
                    if lazy_state.value_and_grad is None:
                        if lazy_state.total_gradient is None:
                            with lazy_state.gradient_lock:
                                if lazy_state.total_gradient is None:
                                    lazy_state.total_gradient = (
                                        _build_compiled_total_gradient_for()
                                    )
                        lazy_state.value_and_grad = _build_compiled_value_and_grad_for(
                            lazy_state.total_gradient
                        )
            return lazy_state.value_and_grad(coil_dofs)

        compiled_total_gradient_for = _lazy_compiled_total_gradient_for
        compiled_total_gradient_for_with_certificate_key = (
            _lazy_compiled_total_gradient_for_with_certificate_key
        )
        compiled_value_and_grad_for = mark_cacheable_jit_value_and_grad(
            _lazy_compiled_value_and_grad_for
        )
    else:
        lazy_state = None
        compiled_total_gradient_for = _build_compiled_total_gradient_for()
        compiled_total_gradient_for_with_certificate_key = (
            _build_compiled_total_gradient_for_with_certificate_key()
        )
        compiled_value_and_grad_for = _build_compiled_value_and_grad_for(
            compiled_total_gradient_for
        )

    return {
        "state": state,
        "compiled_forward_result_for": compiled_forward_result_for,
        "compiled_forward_result_from_anchor_for": (
            jitted_forward_result_from_anchor_for
        ),
        "compiled_total_gradient_for": compiled_total_gradient_for,
        "compiled_total_gradient_for_with_certificate_key": (
            compiled_total_gradient_for_with_certificate_key
        ),
        "compiled_value_and_grad_for": compiled_value_and_grad_for,
        "lazy_gradient_state": lazy_state,
    }


def _build_traceable_solved_state_value_and_grad_from_state(booz_jax, state):
    """Compile value/adjoint kernels for caller-provided solved Boozer states."""
    objective_kwargs = state["objective_kwargs"]
    linearization_kind = state["linearization_kind"]
    linear_solve_tol = state["linear_solve_tol"]
    linear_solve_stab = state["linear_solve_stab"]
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]

    def _solved_state_value_and_grad_for(
        coil_dofs,
        solved_x,
        solved_linear_solve_factors,
    ):
        coil_dofs = _as_jax_float64(coil_dofs)
        solved_x = _as_jax_float64(solved_x)
        coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
        value = _evaluate_traceable_total_objective(
            solved_x,
            coil_dofs,
            coil_set_spec,
            objective_kwargs,
        )
        grad, linear_solve_success = _traceable_total_gradient_with_status(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_linear_solve_factors=_traceable_runtime_deviceify_tree(
                solved_linear_solve_factors
            ),
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            objective_kwargs=objective_kwargs,
        )
        return value, _traceable_adjoint_gradient_or_nan(
            grad,
            linear_solve_success,
        )

    return jax.jit(_solved_state_value_and_grad_for)


def _build_traceable_solved_state_value_and_grad_from_compiled_bundle(
    compiled_bundle,
    *,
    producer_graph_sha256=None,
    host_factor_handoff_gate=False,
):
    """Build solved-state value/grad while reusing the bundle's adjoint kernel.

    When ``host_factor_handoff_gate`` is true, the returned callable is a host
    Python wrapper that accepts only ``None`` or a sealed
    :class:`ExactFactorHandoff` and re-requires the integrity receipt before
    entering the jitted device kernel. Unaffiliated raw factor trees fail
    closed at this host boundary. JIT-internal callers keep
    ``host_factor_handoff_gate=False`` so raw factor trees and live-operator
    certification paths remain numerically unchanged.
    """
    state = compiled_bundle["state"]
    objective_kwargs = state["objective_kwargs"]
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]
    compiled_total_gradient_for = compiled_bundle["compiled_total_gradient_for"]

    def _solved_state_value_and_grad_for(
        coil_dofs,
        solved_x,
        solved_linear_solve_factors,
    ):
        coil_dofs = _as_jax_float64(coil_dofs)
        solved_x = _as_jax_float64(solved_x)
        coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
        value = _evaluate_traceable_total_objective(
            solved_x,
            coil_dofs,
            coil_set_spec,
            objective_kwargs,
        )
        grad, linear_solve_success = compiled_total_gradient_for(
            coil_dofs,
            solved_x,
            _traceable_runtime_deviceify_tree(solved_linear_solve_factors),
        )
        return value, _traceable_adjoint_gradient_or_nan(
            grad,
            linear_solve_success,
        )

    compiled_value_and_grad = jax.jit(_solved_state_value_and_grad_for)
    if not host_factor_handoff_gate:
        return compiled_value_and_grad
    if producer_graph_sha256 is None:
        raise ValueError(
            "host_factor_handoff_gate requires producer_graph_sha256 for integrity "
            "receipt re-validation."
        )

    def _host_value_and_grad_from_solved(
        coil_dofs,
        solved_x,
        solved_linear_solve_factors,
    ):
        factors = _host_unwrap_linear_solve_factors_for_consumption(
            solved_linear_solve_factors,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256=producer_graph_sha256,
        )
        return compiled_value_and_grad(coil_dofs, solved_x, factors)

    return _host_value_and_grad_from_solved


def _traceable_runtime_option_signature(booz_jax, state):
    """Capture the solver options that affect traceable runtime compilation."""
    option_state = {
        key: booz_jax.options.get(key) for key in _TRACEABLE_RUNTIME_OPTION_KEYS
    }
    if state["predictor_kind"] == "ls":
        option_state["optimizer_options"] = booz_jax._collect_optimizer_options(
            method=state["objective_method"]
        )
    else:
        option_state["optimizer_options"] = {}
    return _traceable_cache_tree_signature(option_state)


@dataclass(frozen=True)
class _TraceableCallableSignature:
    callback: object

    def __eq__(self, other):
        return (
            isinstance(other, _TraceableCallableSignature)
            and self.callback is other.callback
        )

    def __hash__(self):
        return object.__hash__(self.callback)


def _traceable_success_filter_signature(success_filter):
    """Return the runtime-cache signature for one optional success filter."""

    if success_filter is None:
        return None
    signature = getattr(success_filter, "_traceable_runtime_cache_signature", None)
    if signature is not None:
        return ("structural", signature)
    return ("callable", _TraceableCallableSignature(success_filter))


@dataclass(frozen=True)
class _TraceableRuntimeCacheKey:
    solve_state_token: int
    coil_dof_state_token: int
    coil_layout_signature: object
    optimize_G: bool
    predictor_kind: str
    objective_contract_signature: object
    option_signature: object
    success_filter_signature: object
    precision_signature: tuple[str, str]


def _traceable_runtime_cache_key(booz_jax, state, *, success_filter=None):
    """Return a stable cache key for one compiled traceable runtime state.

    The key is derived from the immutable runtime state captured by the cheap
    ``_build_traceable_objective_cache_state`` path. Large solved baseline
    arrays are represented by explicit solve/coil state tokens instead of
    value-hashing their contents on every lookup; the coil reconstruction
    layout is signed structurally because the compiled bundle closes over that
    layout.
    """
    objective_kwargs = state["objective_kwargs"]
    policy = get_backend_policy()
    return _TraceableRuntimeCacheKey(
        solve_state_token=state["solve_state_token"],
        coil_dof_state_token=state["coil_dof_state_token"],
        coil_layout_signature=state["coil_layout_signature"],
        optimize_G=state["optimize_G"],
        predictor_kind=state["predictor_kind"],
        objective_contract_signature=_traceable_contract_tree_signature(
            objective_kwargs
        ),
        option_signature=_traceable_runtime_option_signature(booz_jax, state),
        success_filter_signature=_traceable_success_filter_signature(success_filter),
        precision_signature=(
            np.dtype(policy.compute_dtype).str,
            np.dtype(policy.runtime_dtype).str,
        ),
    )


def _build_traceable_runtime_entry(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
):
    """Build one isolated traceable runtime entry for an owning session."""
    cache_state = _build_traceable_objective_cache_state(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
    )
    cache_key = _traceable_runtime_cache_key(
        booz_jax,
        cache_state,
        success_filter=success_filter,
    )
    state = _materialize_traceable_objective_state(booz_jax, bs_jax, cache_state)
    compiled_bundle = _build_traceable_objective_compiled_bundle_from_state(
        booz_jax,
        state,
        success_filter=success_filter,
    )
    objective = _make_traceable_objective_from_compiled_bundle(compiled_bundle)
    return TraceableObjectiveSession(
        cache_key=cache_key,
        success_filter=success_filter,
        compiled_bundle=compiled_bundle,
        booz_jax=booz_jax,
        objective=objective,
        batched_value_and_grad=_make_traceable_batched_value_and_grad_pipeline(
            compiled_bundle["compiled_value_and_grad_for"]
        ),
    )


def _build_traceable_solved_state_value_and_grad_entry(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
):
    """Build an isolated host-solve/compiled-gradient bridge."""
    cache_state = _build_traceable_objective_cache_state(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        require_ondevice_inner=False,
    )
    cache_key = _traceable_runtime_cache_key(
        booz_jax,
        cache_state,
        success_filter=success_filter,
    )
    state = _materialize_traceable_objective_state(booz_jax, bs_jax, cache_state)
    session = TraceableObjectiveSession(
        cache_key=cache_key,
        success_filter=success_filter,
        compiled_bundle={"state": state},
        booz_jax=booz_jax,
    )
    session.solved_state_value_and_grad = (
        _build_traceable_solved_state_value_and_grad_from_state(booz_jax, state)
    )
    return session


def _ensure_traceable_runtime_reporting_metrics(runtime_entry):
    """Materialize the pure reporting-metrics selector on demand."""
    if runtime_entry["reporting_metrics"] is None:
        runtime_entry["reporting_metrics"] = _make_traceable_reporting_metrics_bundle(
            runtime_entry["compiled_bundle"]
        )
    return runtime_entry


def _ensure_traceable_runtime_reporting_metrics_from_solution(runtime_entry):
    """Materialize solved-state reporting metrics that do not replay the solve."""
    if runtime_entry["reporting_metrics_from_solution"] is None:
        runtime_entry["reporting_metrics_from_solution"] = (
            _make_traceable_reporting_metrics_from_solution_bundle(
                runtime_entry["compiled_bundle"]
            )
        )
    return runtime_entry


def _make_traceable_lazy_reporting_metrics_boundary(runtime_entry):
    """Build a public reporting-metrics boundary that resolves lazily."""

    def reporting_metrics_for(coil_dofs, *, include_distance_metrics=True):
        reporting_metrics = _ensure_traceable_runtime_reporting_metrics(runtime_entry)[
            "reporting_metrics"
        ]
        return reporting_metrics(
            _as_jax_float64(coil_dofs),
            include_distance_metrics=include_distance_metrics,
        )

    return reporting_metrics_for


def _make_traceable_lazy_reporting_metrics_from_solution_boundary(runtime_entry):
    """Build a public reporting boundary for an explicit solved decision vector."""

    def reporting_metrics_from_solution_for(
        coil_dofs,
        solved_x,
        solver_success,
        *,
        include_distance_metrics=True,
        outer_raw_terms=None,
    ):
        reporting_metrics_from_solution = (
            _ensure_traceable_runtime_reporting_metrics_from_solution(runtime_entry)[
                "reporting_metrics_from_solution"
            ]
        )
        staged_coil_dofs = _as_jax_float64(coil_dofs)
        staged_solved_x = _as_jax_float64(solved_x)
        staged_solver_success = _staged_like(
            staged_solved_x,
            solver_success,
            dtype=np.bool_,
        )
        staged_outer_raw_terms = None
        if outer_raw_terms is not None:
            raw_terms_present, raw_terms = outer_raw_terms
            staged_outer_raw_terms = (
                _staged_like(
                    staged_solved_x,
                    raw_terms_present,
                    dtype=np.bool_,
                ),
                jax.tree.map(
                    lambda raw_term: _staged_like(staged_solved_x, raw_term),
                    raw_terms,
                ),
            )
        return reporting_metrics_from_solution(
            staged_coil_dofs,
            staged_solved_x,
            staged_solver_success,
            include_distance_metrics=include_distance_metrics,
            outer_raw_terms=staged_outer_raw_terms,
        )

    return reporting_metrics_from_solution_for


def _ensure_traceable_runtime_public_boundaries(runtime_entry):
    """Materialize stable public runtime-bundle boundaries on demand."""
    if runtime_entry["public_objective"] is None:
        runtime_entry["public_objective"] = _make_traceable_objective_boundary(
            runtime_entry["objective"]
        )
    if runtime_entry["public_value_and_grad"] is None:
        runtime_entry["public_value_and_grad"] = (
            _make_traceable_value_and_grad_boundary(
                runtime_entry["compiled_bundle"]["compiled_value_and_grad_for"]
            )
        )
    if runtime_entry["public_batched_value_and_grad"] is None:
        runtime_entry["public_batched_value_and_grad"] = (
            _make_traceable_batched_value_and_grad_boundary(
                runtime_entry["batched_value_and_grad"]
            )
        )
    if runtime_entry["public_forward_result"] is None:
        runtime_entry["public_forward_result"] = (
            _make_traceable_forward_result_boundary(
                runtime_entry["compiled_bundle"]["compiled_forward_result_for"],
            )
        )
    if runtime_entry["public_reporting_metrics"] is None:
        runtime_entry["public_reporting_metrics"] = (
            _make_traceable_lazy_reporting_metrics_boundary(runtime_entry)
        )
    if runtime_entry["public_reporting_metrics_from_solution"] is None:
        runtime_entry["public_reporting_metrics_from_solution"] = (
            _make_traceable_lazy_reporting_metrics_from_solution_boundary(runtime_entry)
        )
    return runtime_entry


def _ensure_traceable_runtime_optimizer_compiled_bundle(runtime_entry, booz_jax):
    optimizer_compiled_bundle = runtime_entry.get("optimizer_compiled_bundle")
    if optimizer_compiled_bundle is None:
        state = runtime_entry["compiled_bundle"]["state"]
        optimizer_compiled_bundle = (
            _build_traceable_objective_compiled_bundle_from_state(
                booz_jax,
                state,
                success_filter=runtime_entry.get("success_filter"),
                general_only_forward=True,
            )
        )
        runtime_entry["optimizer_compiled_bundle"] = optimizer_compiled_bundle
    return optimizer_compiled_bundle


def _ensure_traceable_runtime_optimizer_value_and_grad(runtime_entry, booz_jax):
    optimizer_value_and_grad = runtime_entry.get("optimizer_value_and_grad")
    if optimizer_value_and_grad is None:
        optimizer_compiled_bundle = _ensure_traceable_runtime_optimizer_compiled_bundle(
            runtime_entry,
            booz_jax,
        )
        optimizer_value_and_grad = _make_traceable_value_and_grad_boundary(
            optimizer_compiled_bundle["compiled_value_and_grad_for"]
        )
        runtime_entry["optimizer_value_and_grad"] = optimizer_value_and_grad
    return optimizer_value_and_grad


def _producer_graph_sha256_for_traceable_state(state):
    """State-contract graph id for host integrity receipts (R06).

    StableHLO-based producer-graph hashing is deferred; this binds the compiled-
    bundle configuration without changing JIT-internal PLU refine paths.
    """
    return hashlib.sha256(
        repr(
            _traceable_cache_tree_signature(
                {
                    "linearization_kind": state["linearization_kind"],
                    "linear_solve_tol": state["linear_solve_tol"],
                    "linear_solve_stab": state["linear_solve_stab"],
                    "objective_kwargs": state["objective_kwargs"],
                    "optimize_G": state["optimize_G"],
                    "predictor_kind": state["predictor_kind"],
                }
            )
        ).encode("utf-8")
    ).hexdigest()


def _build_traceable_optimizer_solved_pair(optimizer_compiled_bundle):
    """Build the gated decomposed solved-pair from one optimizer compiled bundle.

    Both halves share ``optimizer_compiled_bundle`` so ``solve_fn`` outputs are
    exactly what ``value_grad_from_solved`` consumes after host seal/require.
    Cross-evaluation factor reuse is default-off (``None`` factors pass through);
    when factors are exported they must travel as :class:`ExactFactorHandoff`.
    Host gate stays on via ``host_factor_handoff_gate=True``.
    """
    state = optimizer_compiled_bundle["state"]
    producer_graph_sha256 = _producer_graph_sha256_for_traceable_state(state)

    def build_exact_factor_handoff_for(coil_dofs, solved_x, factors):
        """Producer: seal raw factors (or pass through None / sealed handoff)."""
        return _host_seal_linear_solve_factors_for_reuse(
            factors,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256=producer_graph_sha256,
        )

    def seal_res_factors_for_reuse(res, *, coil_dofs, solved_x):
        """Producer: extract public factors from a solved res and seal them."""
        return _host_seal_linear_solve_factors_from_res(
            res,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256=producer_graph_sha256,
        )

    def require_exact_factor_handoff_for(handoff, *, coil_dofs, solved_x):
        """Consumer: release factors only after integrity re-seal."""
        return _host_require_exact_factor_handoff(
            handoff,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256=producer_graph_sha256,
        )

    return TraceableObjectiveSolvedPair(
        solve_fn=optimizer_compiled_bundle["compiled_forward_result_for"],
        value_grad_from_solved=(
            _build_traceable_solved_state_value_and_grad_from_compiled_bundle(
                optimizer_compiled_bundle,
                producer_graph_sha256=producer_graph_sha256,
                host_factor_handoff_gate=True,
            )
        ),
        build_exact_factor_handoff_for=build_exact_factor_handoff_for,
        require_exact_factor_handoff_for=require_exact_factor_handoff_for,
        seal_res_factors_for_reuse=seal_res_factors_for_reuse,
    )


def _ensure_traceable_runtime_optimizer_solved_pair(runtime_entry, booz_jax):
    """Residual dict-entry materializer for the decomposed solved pair.

    Prefer :meth:`TraceableObjectiveSession.solved_pair` for new call sites.
    This path remains for residual ``runtime_entry`` consumers that still store
    lazy fields on the booz-attached cache dict (not yet session-owned).
    """
    optimizer_solved_pair = runtime_entry.get("optimizer_solved_pair")
    if optimizer_solved_pair is None:
        optimizer_compiled_bundle = _ensure_traceable_runtime_optimizer_compiled_bundle(
            runtime_entry,
            booz_jax,
        )
        optimizer_solved_pair = _build_traceable_optimizer_solved_pair(
            optimizer_compiled_bundle
        )
        runtime_entry["optimizer_solved_pair"] = optimizer_solved_pair
    return optimizer_solved_pair


class TraceableObjectiveCandidateEvaluation(NamedTuple):
    """Device-resident forward, gradient, and state for one candidate."""

    forward_result: dict[str, object]
    gradient: jax.Array
    actual_adjoint_success: jax.Array
    gradient_source: str
    candidate_inner_state: TraceableObjectiveInnerState


def _build_candidate_evaluation_core(
    compiled_bundle,
    *,
    fallback_gradient_on_primal_failure: bool = True,
):
    """Build the single owner of forward, gradient routing, and eligibility.

    The core calls the owned compiled forward exactly once, applies the caller's
    fail-closed gradient-fallback policy, and certifies the candidate inner state
    device-side. Only the host branching the routing needs is synchronized.
    """
    state = compiled_bundle["state"]
    compiled_forward_result_for = compiled_bundle["compiled_forward_result_for"]
    compiled_forward_result_from_anchor_for = compiled_bundle.get(
        "compiled_forward_result_from_anchor_for"
    )
    compiled_total_gradient_for = compiled_bundle["compiled_total_gradient_for"]
    baseline_coil_dofs = _as_jax_float64(state["baseline_coil_dofs"])
    baseline_x = _as_jax_float64(state["baseline_x"])
    baseline_linear_solve_factors = _traceable_runtime_deviceify_tree(
        state["baseline_linear_solve_factors"]
    )

    def evaluate_candidate(
        candidate_coil_dofs,
        incumbent: TraceableObjectiveInnerState | None,
    ) -> TraceableObjectiveCandidateEvaluation:
        if incumbent is None:
            forward_result = compiled_forward_result_for(candidate_coil_dofs)
            fallback_coil_dofs = baseline_coil_dofs
            fallback_x = baseline_x
            fallback_gradient_source = "baseline"
        else:
            if compiled_forward_result_from_anchor_for is None:
                raise RuntimeError(
                    "explicit incumbent evaluation requires the anchored forward "
                    "program"
                )
            forward_result = compiled_forward_result_from_anchor_for(
                candidate_coil_dofs,
                incumbent,
            )
            incumbent_eligible = _host_bool(incumbent.eligible)
            fallback_coil_dofs = (
                incumbent.coil_dofs if incumbent_eligible else baseline_coil_dofs
            )
            fallback_x = incumbent.solved_x if incumbent_eligible else baseline_x
            fallback_gradient_source = (
                "incumbent" if incumbent_eligible else "baseline"
            )
        candidate_gradient_source = _host_bool(forward_result["success"]) or _host_bool(
            forward_result["primal_success"]
        )
        if candidate_gradient_source:
            gradient_coil_dofs = candidate_coil_dofs
            gradient_x = forward_result["x"]
            gradient_linear_solve_factors = forward_result["linear_solve_factors"]
            gradient_source = "candidate"
        elif not fallback_gradient_on_primal_failure:
            gradient = _traceable_adjoint_fail_gradient_like(candidate_coil_dofs)
            actual_adjoint_success = _runtime_device_put(np.bool_(False))
            gradient_source = "unavailable"
        else:
            gradient_coil_dofs = fallback_coil_dofs
            gradient_x = fallback_x
            gradient_linear_solve_factors = baseline_linear_solve_factors
            gradient_source = fallback_gradient_source

        if candidate_gradient_source or fallback_gradient_on_primal_failure:
            raw_gradient, actual_adjoint_success = compiled_total_gradient_for(
                gradient_coil_dofs,
                gradient_x,
                gradient_linear_solve_factors,
            )
            gradient = _traceable_adjoint_gradient_or_nan(
                raw_gradient,
                actual_adjoint_success,
            )
        candidate_inner_state = TraceableObjectiveInnerState(
            coil_dofs=candidate_coil_dofs,
            solved_x=forward_result["x"],
            objective_value=forward_result["raw_value"],
            eligible=(
                jnp.asarray(forward_result["success"], dtype=jnp.bool_)
                & jnp.asarray(actual_adjoint_success, dtype=jnp.bool_)
                & jnp.all(jnp.isfinite(jnp.asarray(gradient)))
                & jnp.isfinite(jnp.asarray(forward_result["raw_value"]))
            ),
        )
        return TraceableObjectiveCandidateEvaluation(
            forward_result=dict(forward_result),
            gradient=gradient,
            actual_adjoint_success=actual_adjoint_success,
            gradient_source=gradient_source,
            candidate_inner_state=candidate_inner_state,
        )

    return evaluate_candidate


def _build_accepted_incumbent_evaluator(compiled_bundle):
    """Build the lean compiled evaluate for the accepted-incumbent controller.

    Returns device-resident value, gradient, and candidate state without the
    diagnostic trace materialization the trial evaluator performs.
    """
    evaluate_candidate = _build_candidate_evaluation_core(compiled_bundle)

    def compiled_evaluate(
        candidate_coil_dofs,
        incumbent: TraceableObjectiveInnerState,
    ) -> TraceableObjectiveIncumbentEvaluation:
        core = evaluate_candidate(candidate_coil_dofs, incumbent)
        return TraceableObjectiveIncumbentEvaluation(
            value=jnp.asarray(core.forward_result["value"], dtype=jnp.float64),
            gradient=jnp.asarray(core.gradient, dtype=jnp.float64),
            candidate_inner_state=core.candidate_inner_state,
        )

    return compiled_evaluate


def _build_traceable_trial_evaluator(compiled_bundle):
    """Build the opt-in host diagnostic for one forward/adjoint evaluation.

    Hostifies the shared candidate-evaluation core plus the full Newton trace
    telemetry; production optimization uses the lean incumbent evaluator
    instead.
    """
    evaluate_candidate = _build_candidate_evaluation_core(compiled_bundle)

    def evaluate_trial(
        coil_dofs,
        incumbent: TraceableObjectiveInnerState | None = None,
    ) -> TraceableObjectiveTrialResult:
        candidate_coil_dofs = _as_jax_float64(coil_dofs)
        core = evaluate_candidate(candidate_coil_dofs, incumbent)
        forward_result = core.forward_result
        actual_adjoint_success = core.actual_adjoint_success
        gradient_source = core.gradient_source
        candidate_inner_state = core.candidate_inner_state
        gradient = _host_array(core.gradient, dtype=np.float64)
        gradient_is_finite = bool(np.all(np.isfinite(gradient)))
        gradient_inf_norm = (
            float(np.max(np.abs(gradient))) if gradient_is_finite else float("nan")
        )
        raw_objective_value = float(
            _host_scalar(forward_result["raw_value"], dtype=np.float64)
        )

        def trace_array(key: str) -> np.ndarray:
            if key in {
                "newton_trace_active",
                "newton_trace_step_accepted",
                "newton_trace_linear_solve_success",
                "newton_trace_linear_live_operator_certificate",
            }:
                dtype = np.bool_
            elif "dtype_bits" in key:
                dtype = np.int32
            else:
                dtype = np.float64
            return _host_array(forward_result[key], dtype=dtype)

        def trace_present(key: str) -> bool:
            return _host_bool(forward_result[f"{key}_present"])

        missing_int = int(np.iinfo(np.int32).min)
        newton_attempted_iterations = _host_int(
            forward_result["newton_attempted_iterations"]
        )
        newton_stop_reason_code = _host_int(
            forward_result["newton_stop_reason_code"]
        )
        detailed_newton_telemetry_present = not (
            newton_attempted_iterations == missing_int
            and newton_stop_reason_code == missing_int
        )
        inner_penalty_residual_l2 = float(
            _host_scalar(
                forward_result["inner_penalty_residual_l2"],
                dtype=np.float64,
            )
        )
        final_gradient_inf_norm = float(
            _host_scalar(
                forward_result["final_gradient_inf_norm"],
                dtype=np.float64,
            )
        )

        return TraceableObjectiveTrialResult(
            raw_objective_value=raw_objective_value,
            filtered_objective_value=float(
                _host_scalar(forward_result["value"], dtype=np.float64)
            ),
            gradient=gradient,
            gradient_source=gradient_source,
            gradient_is_finite=gradient_is_finite,
            gradient_inf_norm=gradient_inf_norm,
            predictor_success=_host_bool(forward_result["predictor_success"]),
            primal_success=_host_bool(forward_result["primal_success"]),
            actual_adjoint_success=_host_bool(actual_adjoint_success),
            adjoint_linear_solve_available=_host_bool(
                forward_result["adjoint_linear_solve_available"]
            ),
            newton_success=_host_bool(forward_result["newton_success"]),
            newton_iterations=_host_int(forward_result["newton_iterations"]),
            newton_attempted_iterations=(
                newton_attempted_iterations
                if detailed_newton_telemetry_present
                else None
            ),
            newton_stop_reason_code=(
                newton_stop_reason_code if detailed_newton_telemetry_present else None
            ),
            newton_last_linear_solve_success=(
                _host_bool(forward_result["newton_last_linear_solve_success"])
                if detailed_newton_telemetry_present
                else None
            ),
            inner_penalty_residual_l2=(
                inner_penalty_residual_l2
                if np.isfinite(inner_penalty_residual_l2)
                else None
            ),
            final_gradient_inf_norm=(
                final_gradient_inf_norm
                if np.isfinite(final_gradient_inf_norm)
                else None
            ),
            newton_trace_active=trace_array("newton_trace_active"),
            newton_trace_step_accepted=trace_array("newton_trace_step_accepted"),
            newton_trace_linear_solve_success=trace_array(
                "newton_trace_linear_solve_success"
            ),
            newton_trace_linear_residual_relative=trace_array(
                "newton_trace_linear_residual_relative"
            ),
            newton_trace_linear_residual_norm=trace_array(
                "newton_trace_linear_residual_norm"
            ),
            newton_trace_linear_residual_scale=trace_array(
                "newton_trace_linear_residual_scale"
            ),
            newton_trace_linear_requested_tolerance=trace_array(
                "newton_trace_linear_requested_tolerance"
            ),
            newton_trace_linear_effective_tolerance=trace_array(
                "newton_trace_linear_effective_tolerance"
            ),
            newton_trace_linear_live_operator_certificate=trace_array(
                "newton_trace_linear_live_operator_certificate"
            ),
            newton_trace_linear_factorization_dtype_bits=trace_array(
                "newton_trace_linear_factorization_dtype_bits"
            ),
            newton_trace_linear_factor_application_dtype_bits=trace_array(
                "newton_trace_linear_factor_application_dtype_bits"
            ),
            newton_trace_linear_residual_dtype_bits=trace_array(
                "newton_trace_linear_residual_dtype_bits"
            ),
            newton_trace_certificate_value_dtype_bits=trace_array(
                "newton_trace_certificate_value_dtype_bits"
            ),
            newton_trace_certificate_gradient_dtype_bits=trace_array(
                "newton_trace_certificate_gradient_dtype_bits"
            ),
            newton_trace_active_present=trace_present("newton_trace_active"),
            newton_trace_step_accepted_present=trace_present(
                "newton_trace_step_accepted"
            ),
            newton_trace_linear_solve_success_present=trace_present(
                "newton_trace_linear_solve_success"
            ),
            newton_trace_linear_residual_relative_present=trace_present(
                "newton_trace_linear_residual_relative"
            ),
            newton_trace_linear_residual_norm_present=trace_present(
                "newton_trace_linear_residual_norm"
            ),
            newton_trace_linear_residual_scale_present=trace_present(
                "newton_trace_linear_residual_scale"
            ),
            newton_trace_linear_requested_tolerance_present=trace_present(
                "newton_trace_linear_requested_tolerance"
            ),
            newton_trace_linear_effective_tolerance_present=trace_present(
                "newton_trace_linear_effective_tolerance"
            ),
            newton_trace_linear_live_operator_certificate_present=trace_present(
                "newton_trace_linear_live_operator_certificate"
            ),
            newton_trace_linear_factorization_dtype_bits_present=trace_present(
                "newton_trace_linear_factorization_dtype_bits"
            ),
            newton_trace_linear_factor_application_dtype_bits_present=trace_present(
                "newton_trace_linear_factor_application_dtype_bits"
            ),
            newton_trace_linear_residual_dtype_bits_present=trace_present(
                "newton_trace_linear_residual_dtype_bits"
            ),
            newton_trace_certificate_value_dtype_bits_present=trace_present(
                "newton_trace_certificate_value_dtype_bits"
            ),
            newton_trace_certificate_gradient_dtype_bits_present=trace_present(
                "newton_trace_certificate_gradient_dtype_bits"
            ),
            newton_linear_solve_backend_code=_host_int(
                forward_result["newton_linear_solve_backend_code"]
            ),
            newton_linear_solve_backend_code_present=_host_bool(
                forward_result["newton_linear_solve_backend_code_present"]
            ),
            dense_hessian_bytes=_host_int(forward_result["dense_hessian_bytes"]),
            dense_hessian_bytes_present=_host_bool(
                forward_result["dense_hessian_bytes_present"]
            ),
            max_dense_hessian_bytes=_host_int(
                forward_result["max_dense_hessian_bytes"]
            ),
            max_dense_hessian_bytes_present=_host_bool(
                forward_result["max_dense_hessian_bytes_present"]
            ),
            candidate_inner_state=candidate_inner_state,
        )

    return evaluate_trial


def _ensure_traceable_runtime_seeded_value_and_grad(
    runtime_entry,
    booz_jax,
):
    """Materialize the deterministic compatibility seed and cache its result."""
    state = runtime_entry["compiled_bundle"]["state"]
    seeded_value_and_grad = runtime_entry.get("seeded_value_and_grad")
    if seeded_value_and_grad is not None:
        return seeded_value_and_grad

    seeded_compiled_bundle = _ensure_traceable_runtime_optimizer_compiled_bundle(
        runtime_entry,
        booz_jax,
    )
    baseline_coil_dofs = _traceable_runtime_deviceify_tree(state["baseline_coil_dofs"])
    baseline_x = _traceable_runtime_deviceify_tree(state["baseline_x"])
    baseline_value = _traceable_runtime_deviceify_tree(state["baseline_value"])
    baseline_linear_solve_factors = _traceable_runtime_deviceify_tree(
        state["baseline_linear_solve_factors"]
    )
    baseline_gradient, baseline_linear_solve_success = seeded_compiled_bundle[
        "compiled_total_gradient_for"
    ](
        baseline_coil_dofs,
        baseline_x,
        baseline_linear_solve_factors,
    )
    mixed_dense_ir_trust = _dense_ir._inactive_mixed_dense_ir_trust_telemetry(
        baseline_x
    )
    baseline_gradient = _traceable_adjoint_gradient_or_nan(
        baseline_gradient,
        baseline_linear_solve_success,
    )
    seeded_value_and_grad = TraceableObjectiveCertifiedSeededValueAndGrad(
        seeded_value_and_grad=TraceableObjectiveSeededValueAndGrad(
            value_and_grad=_ensure_traceable_runtime_optimizer_value_and_grad(
                runtime_entry,
                booz_jax,
            ),
            optimizer_initial_value_and_grad=(
                baseline_value,
                baseline_gradient,
            ),
        ),
        certificate_probe_authority=None,
        certificate_probe_evidence=None,
        mixed_dense_ir_trust=mixed_dense_ir_trust,
    )
    runtime_entry["seeded_compiled_bundle"] = seeded_compiled_bundle
    runtime_entry["seeded_value_and_grad"] = seeded_value_and_grad
    return seeded_value_and_grad


def _mixed_certificate_probe_evidence(
    authority: CertificateProbeAuthority,
    trust: _dense_ir._MixedDenseIrTrustTelemetry,
) -> CertificateProbeEvidence | None:
    """Bind active device trust to its exact host challenge and fallback decision."""
    if not _host_bool(trust.active):
        return None
    observed_words = _host_array(
        trust.certificate_probe_key_data,
        dtype=np.uint32,
    )
    evidence = CertificateProbeEvidence(
        authority=authority,
        observed_key_data=CertificateProbeKeyData(
            int(observed_words[0]),
            int(observed_words[1]),
        ),
        active=True,
        proposal_trusted=_host_bool(trust.proposal_trusted),
        fp64_rebuild_count=_host_int(trust.fp64_rebuild_count),
        fallback_attempted=_host_bool(trust.fallback.attempted),
        fallback_success=_host_bool(trust.fallback.success),
    )
    evidence.require_valid_for_mixed()
    return evidence


def _make_traceable_runtime_certified_seeded_value_and_grad(
    runtime_entry,
    booz_jax,
    *,
    certificate_probe_key_data: CertificateProbeKeyData | None = None,
):
    """Evaluate one uncached mixed certificate from fresh or replay authority."""
    state = runtime_entry["compiled_bundle"]["state"]
    policy = get_backend_policy()
    mixed_dense_ir_enabled = np.dtype(policy.compute_dtype) == np.dtype(
        np.float32
    ) and np.dtype(policy.runtime_dtype) == np.dtype(np.float64)
    if not mixed_dense_ir_enabled or state["baseline_linear_solve_factors"] is not None:
        return _ensure_traceable_runtime_seeded_value_and_grad(runtime_entry, booz_jax)

    authority = resolve_certificate_probe_authority(certificate_probe_key_data)
    seeded_compiled_bundle = _ensure_traceable_runtime_optimizer_compiled_bundle(
        runtime_entry,
        booz_jax,
    )
    baseline_coil_dofs = _traceable_runtime_deviceify_tree(state["baseline_coil_dofs"])
    baseline_x = _traceable_runtime_deviceify_tree(state["baseline_x"])
    baseline_value = _traceable_runtime_deviceify_tree(state["baseline_value"])
    certificate_probe_key = _runtime_certificate_probe_key(authority.key_data)
    (
        baseline_gradient,
        baseline_linear_solve_success,
        mixed_dense_ir_trust,
    ) = seeded_compiled_bundle["compiled_total_gradient_for_with_certificate_key"](
        baseline_coil_dofs,
        baseline_x,
        None,
        certificate_probe_key,
    )
    baseline_gradient = _traceable_adjoint_gradient_or_nan(
        baseline_gradient,
        baseline_linear_solve_success,
    )
    evidence = _mixed_certificate_probe_evidence(authority, mixed_dense_ir_trust)
    observed_authority = authority if evidence is not None else None
    return TraceableObjectiveCertifiedSeededValueAndGrad(
        seeded_value_and_grad=TraceableObjectiveSeededValueAndGrad(
            value_and_grad=_ensure_traceable_runtime_optimizer_value_and_grad(
                runtime_entry,
                booz_jax,
            ),
            optimizer_initial_value_and_grad=(baseline_value, baseline_gradient),
        ),
        certificate_probe_authority=observed_authority,
        certificate_probe_evidence=evidence,
        mixed_dense_ir_trust=mixed_dense_ir_trust,
    )


def _make_traceable_lazy_host_reporting_metrics(runtime_entry):
    """Build a host-normalized reporting wrapper that resolves lazily."""
    compiled_bundle = runtime_entry["compiled_bundle"]
    state = compiled_bundle["state"]
    baseline_coil_dofs = np.asarray(state["baseline_coil_dofs"], dtype=np.float64)
    baseline_coil_dofs_jax = _traceable_runtime_deviceify_tree(
        state["baseline_coil_dofs"]
    )
    baseline_x = _traceable_runtime_deviceify_tree(state["baseline_x"])
    reporting_state = _HostReportingState()
    runtime_entry["host_reporting_state"] = reporting_state

    def _baseline_reporting_metrics(*, include_distance_metrics):
        include_distances = bool(include_distance_metrics)
        cached_metrics = reporting_state.baseline_metrics.get(include_distances)
        if cached_metrics is None:
            metrics = _traceable_reporting_metrics_from_solution(
                state["objective_kwargs"],
                state["coil_set_spec_from_dofs"],
                coil_dofs=baseline_coil_dofs_jax,
                solved_x=baseline_x,
                solver_success=_runtime_bool(True),
                optimize_G=bool(state["optimize_G"]),
                include_distance_metrics=include_distances,
            )
            with jax.transfer_guard_device_to_host("allow"):
                cached_metrics = _hostify_traceable_reporting_metrics(
                    metrics,
                    include_distance_metrics=include_distances,
                )
            reporting_state.baseline_metrics[include_distances] = cached_metrics
        return dict(cached_metrics)

    def host_reporting_metrics(coil_dofs, *, include_distance_metrics=True):
        if _host_input_matches_baseline(coil_dofs, baseline_coil_dofs):
            return _baseline_reporting_metrics(
                include_distance_metrics=include_distance_metrics
            )
        if reporting_state.resolved_reporting_metrics is None:
            reporting_metrics = _ensure_traceable_runtime_reporting_metrics(
                runtime_entry
            )["reporting_metrics"]
            reporting_state.resolved_reporting_metrics = (
                _make_traceable_host_reporting_metrics(reporting_metrics)
            )
        return reporting_state.resolved_reporting_metrics(
            coil_dofs,
            include_distance_metrics=include_distance_metrics,
        )

    return host_reporting_metrics


def _ensure_traceable_runtime_host_wrappers(runtime_entry, booz_jax):
    """Materialize host-boundary wrappers for one cached runtime entry on demand."""
    if (
        runtime_entry["host_objective"] is None
        or runtime_entry["host_value_and_grad"] is None
        or runtime_entry["host_reporting_metrics"] is None
    ):
        compiled_bundle = runtime_entry["compiled_bundle"]
        state = compiled_bundle["state"]
        baseline_coil_dofs = np.asarray(state["baseline_coil_dofs"], dtype=np.float64)
        baseline_value = float(np.asarray(state["baseline_value"], dtype=np.float64))
        runtime_entry["host_objective"] = _make_traceable_host_objective(
            runtime_entry["objective"],
            baseline_coil_dofs=baseline_coil_dofs,
            baseline_return=baseline_value,
        )
        baseline_state = _BaselineValueAndGradState()
        runtime_entry["baseline_value_and_grad_state"] = baseline_state

        def compute_baseline_value_and_grad():
            baseline_coil_dofs_jax = _traceable_runtime_deviceify_tree(
                state["baseline_coil_dofs"]
            )
            baseline_x = _traceable_runtime_deviceify_tree(state["baseline_x"])
            baseline_linear_solve_factors = _traceable_runtime_deviceify_tree(
                state["baseline_linear_solve_factors"]
            )
            with jax.transfer_guard_host_to_device("allow"):
                baseline_gradient, baseline_linear_solve_success = compiled_bundle[
                    "compiled_total_gradient_for"
                ](
                    baseline_coil_dofs_jax,
                    baseline_x,
                    baseline_linear_solve_factors,
                )
            baseline_gradient = _traceable_adjoint_gradient_or_nan(
                baseline_gradient,
                baseline_linear_solve_success,
            )
            with jax.transfer_guard_device_to_host("allow"):
                return (
                    baseline_value,
                    _host_array(
                        baseline_gradient,
                        dtype=np.float64,
                    ),
                )

        def baseline_value_and_grad_return():
            if baseline_state.value_and_grad is None:
                with baseline_state.lock:
                    if baseline_state.value_and_grad is None:
                        baseline_state.value_and_grad = (
                            compute_baseline_value_and_grad()
                        )
            baseline_value_for_value_and_grad, baseline_gradient = (
                baseline_state.value_and_grad
            )
            return baseline_value_for_value_and_grad, baseline_gradient.copy()

        runtime_entry["host_value_and_grad"] = _make_traceable_host_value_and_grad(
            compiled_bundle["compiled_value_and_grad_for"],
            baseline_coil_dofs=baseline_coil_dofs,
            baseline_return=baseline_value_and_grad_return,
        )
        runtime_entry["host_reporting_metrics"] = (
            _make_traceable_lazy_host_reporting_metrics(runtime_entry)
        )
    return runtime_entry


def _make_traceable_objective_from_compiled_bundle(compiled_bundle):
    """Build the scalar custom-VJP target-lane objective from one compiled bundle."""
    compiled_forward_result_for = compiled_bundle["compiled_forward_result_for"]
    compiled_total_gradient_for = compiled_bundle["compiled_total_gradient_for"]
    state = compiled_bundle["state"]
    baseline_coil_dofs = state["baseline_coil_dofs"]
    baseline_x = state["baseline_x"]
    baseline_linear_solve_factors = _traceable_runtime_deviceify_tree(
        state["baseline_linear_solve_factors"]
    )

    @jax.custom_vjp
    def f(coil_dofs):
        coil_dofs = _as_jax_float64(coil_dofs)
        return compiled_forward_result_for(coil_dofs)["value"]

    def f_fwd(coil_dofs):
        coil_dofs = _as_jax_float64(coil_dofs)
        result = compiled_forward_result_for(coil_dofs)
        # Phase 2 (docs/parity_scientific_equivalence_contract_2026-05-09.md
        # §5.3 / §6): stop_gradient on the cached factor state so the
        # IFT adjoint backward pass cannot retrace into the linear-solve
        # factorization graph.
        return result["value"], (
            coil_dofs,
            lax.stop_gradient(result["x"]),
            jax.tree.map(lax.stop_gradient, result["linear_solve_factors"]),
            result["success"],
            result["primal_success"],
        )

    def f_bwd(saved_state, cotangent):
        (
            coil_dofs,
            solved_x,
            solved_linear_solve_factors,
            success,
            primal_success,
        ) = saved_state
        solved_linear_solve_factors = _traceable_runtime_deviceify_tree(
            solved_linear_solve_factors
        )

        def _gradient_or_nan(candidate_coil_dofs, candidate_x, linear_solve_factors):
            grad, linear_solve_success = compiled_total_gradient_for(
                candidate_coil_dofs,
                candidate_x,
                linear_solve_factors,
            )
            return _traceable_adjoint_gradient_or_nan(grad, linear_solve_success)

        def _accepted_candidate_gradient(_):
            return _gradient_or_nan(
                coil_dofs,
                solved_x,
                solved_linear_solve_factors,
            )

        def _rejected_candidate_gradient(_):
            return lax.cond(
                primal_success,
                lambda _: _gradient_or_nan(
                    coil_dofs,
                    solved_x,
                    solved_linear_solve_factors,
                ),
                lambda _: _gradient_or_nan(
                    baseline_coil_dofs,
                    baseline_x,
                    baseline_linear_solve_factors,
                ),
                operand=None,
            )

        grad = lax.cond(
            success,
            _accepted_candidate_gradient,
            _rejected_candidate_gradient,
            operand=None,
        )
        return (_as_runtime_float64(cotangent, reference=grad) * grad,)

    f.defvjp(f_fwd, f_bwd)

    # Keep the pure runtime entrypoint on a real JIT boundary so transfer_guard
    # rejects implicit host inputs consistently with the other runtime-bundle
    # callables. Explicit host materialization belongs on the host wrapper.
    # NOTE: donate_argnums is intentionally omitted here. ``f`` returns a
    # scalar, so XLA cannot reuse the ``coil_dofs`` buffer for the output;
    # adding donation triggers "Some donated buffers were not usable"
    # warnings without freeing memory. Buffer donation lives on
    # ``_value_and_grad_for`` instead, whose ``grad`` output has the same
    # shape as ``coil_dofs``.
    return jax.jit(f)


def _host_input_matches_baseline(coil_dofs, baseline_coil_dofs):
    """Return whether a host input exactly matches the cached baseline state."""
    if coil_dofs is baseline_coil_dofs:
        return True
    if not isinstance(
        coil_dofs,
        (np.ndarray, np.generic, list, tuple, float, int),
    ):
        return False
    host_coil_dofs = np.asarray(coil_dofs, dtype=np.float64)
    return host_coil_dofs.shape == baseline_coil_dofs.shape and np.array_equal(
        host_coil_dofs, baseline_coil_dofs
    )


def _host_boundary_with_baseline_peel(
    host_callable,
    baseline_coil_dofs,
    baseline_return,
):
    """Skip the traced host boundary when host inputs equal the solved baseline."""
    baseline_host = np.asarray(baseline_coil_dofs, dtype=np.float64)

    def wrapped(coil_dofs, *args, **kwargs):
        if _host_input_matches_baseline(coil_dofs, baseline_host):
            if callable(baseline_return):
                return baseline_return(*args, **kwargs)
            return baseline_return
        return host_callable(coil_dofs, *args, **kwargs)

    return wrapped


def _make_traceable_host_objective(
    pure_objective,
    *,
    baseline_coil_dofs=None,
    baseline_return=None,
):
    """Build a host-normalized scalar wrapper around the pure JAX objective."""

    def host_objective(coil_dofs):
        return float(
            _host_scalar(
                pure_objective(_as_jax_float64(coil_dofs)),
                dtype=np.float64,
            )
        )

    if baseline_coil_dofs is None:
        return host_objective
    return _host_boundary_with_baseline_peel(
        host_objective,
        baseline_coil_dofs,
        baseline_return,
    )


def _make_traceable_objective_boundary(pure_objective):
    """Build the public pure-JAX scalar entrypoint for one runtime bundle."""

    def objective(coil_dofs):
        return pure_objective(_as_jax_float64(coil_dofs))

    return objective


def _make_traceable_forward_result_boundary(compiled_forward_result_for):
    """Build the public pure-JAX forward-result entrypoint for one runtime bundle."""

    def forward_result(coil_dofs):
        result = compiled_forward_result_for(_as_jax_float64(coil_dofs))
        if "dense_plu" in result:
            return result
        linear_solve_factors = result["linear_solve_factors"]
        return dict(
            result,
            dense_plu=linear_solve_factors,
            linear_solve_backend="operator",
            dense_linear_solve_factors_available=linear_solve_factors is not None,
        )

    return forward_result


def _make_traceable_host_value_and_grad(
    compiled_value_and_grad_for,
    *,
    baseline_coil_dofs=None,
    baseline_return=None,
):
    """Build a host-normalized wrapper around the fused JAX value/grad callable.

    The underlying value/grad JIT donates argnum 0, so we materialize a fresh
    JAX buffer before forwarding to keep the caller's input array intact.
    """

    def host_value_and_grad(coil_dofs):
        value, grad = compiled_value_and_grad_for(_as_jax_float64(coil_dofs).copy())
        return (
            float(_host_scalar(value, dtype=np.float64)),
            _host_array(grad, dtype=np.float64),
        )

    if baseline_coil_dofs is None:
        return host_value_and_grad
    return _host_boundary_with_baseline_peel(
        host_value_and_grad,
        baseline_coil_dofs,
        baseline_return,
    )


def _make_traceable_value_and_grad_boundary(compiled_value_and_grad_for):
    """Build the public pure-JAX value/grad entrypoint for one runtime bundle.

    This is the explicit host-to-device staging seam for callers that still
    hold coil DOFs as NumPy arrays during setup or test harness construction.
    Under JAX transfer-guard ``disallow``, explicit staging is allowed while
    implicit transfers are not, so keep this entrypoint aligned with the scalar
    objective and reporting-metrics boundaries.

    The underlying value/grad JIT donates argnum 0, so we materialize a fresh
    JAX buffer before forwarding to keep the caller's input array intact.
    """

    def value_and_grad(coil_dofs):
        return compiled_value_and_grad_for(_as_jax_float64(coil_dofs).copy())

    return mark_cacheable_jit_value_and_grad(value_and_grad)


def _traceable_reporting_metrics_from_solution(
    objective_kwargs,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    solved_x,
    solver_success,
    optimize_G,
    include_distance_metrics,
    raw_terms=None,
    raw_terms_present=None,
):
    """Compute reporting metrics for one explicit solved state."""
    outer_objective_config = objective_kwargs["outer_objective_config"]
    if outer_objective_config is None:
        raise RuntimeError(
            "Traceable reporting metrics require the full single-stage outer objective."
        )

    coil_dof_extraction_spec = objective_kwargs["coil_dof_extraction_spec"]
    optimized_coil_index = int(outer_objective_config["optimized_coil_index"])
    coil_set_spec = coil_set_spec_from_dofs(coil_dofs)

    def compute_raw_terms():
        return _traceable_single_stage_outer_term_values(
            solved_x,
            coil_dofs,
            coil_set_spec,
            **_traceable_total_objective_kwargs(objective_kwargs),
        )

    if raw_terms is None:
        raw_terms = compute_raw_terms()
    elif raw_terms_present is not None:
        raw_terms = lax.cond(
            jnp.asarray(raw_terms_present, dtype=bool),
            lambda _: raw_terms,
            lambda _: compute_raw_terms(),
            operand=None,
        )
    sdofs, iota, G = _split_x_inner_runtime(solved_x, optimize_G)
    surface_gamma, xphi, xtheta = _surface_geometry_from_dofs(
        sdofs,
        objective_kwargs["surface_quadpoints_phi"],
        objective_kwargs["surface_quadpoints_theta"],
        objective_kwargs["mpol"],
        objective_kwargs["ntor"],
        objective_kwargs["nfp"],
        objective_kwargs["stellsym"],
        objective_kwargs["scatter_indices"],
        surface_kind=objective_kwargs["surface_kind"],
    )
    surface_normal = jnp.cross(xphi, xtheta)
    nphi, ntheta = surface_gamma.shape[:2]
    surface_points = surface_gamma.reshape(-1, 3)
    surface_B = grouped_biot_savart_B_from_spec(
        surface_points,
        coil_set_spec,
    ).reshape(nphi, ntheta, 3)
    surface_normal_norm = jnp.sqrt(jnp.sum(surface_normal * surface_normal, axis=-1))
    surface_unit_normal = surface_normal / surface_normal_norm[:, :, None]
    surface_B_normal = jnp.sum(surface_B * surface_unit_normal, axis=-1)
    surface_B_norm = jnp.sqrt(jnp.sum(surface_B * surface_B, axis=-1))
    surface_area = surface_normal_norm / surface_normal_norm.size
    field_error = jnp.sum(
        jnp.abs(surface_B_normal / surface_B_norm) * surface_area
    ) / jnp.sum(surface_area)
    coil_specs = coil_specs_from_dof_extraction_spec(
        coil_dof_extraction_spec,
        coil_dofs,
    )
    optimized_coil_curve_spec = coil_specs[optimized_coil_index].curve
    optimized_current = jnp.abs(
        _take_runtime_scalar(coil_specs[optimized_coil_index].current.value, 0)
    )
    _gamma, optimized_gammadash, optimized_gammadashdash = curve_geometry_from_spec(
        optimized_coil_curve_spec
    )
    coil_length = curve_length_pure(incremental_arclength_pure(optimized_gammadash))
    max_curvature = jnp.max(kappa_pure(optimized_gammadash, optimized_gammadashdash))
    inf = _runtime_float64_scalar(np.inf, reference=surface_gamma)
    curve_curve_min_dist = inf
    curve_surface_min_dist = inf
    surface_vessel_min_dist = inf
    if include_distance_metrics:
        vessel_gamma = _runtime_float64_array(
            outer_objective_config["vessel_gamma"],
            reference=surface_gamma,
        ).reshape((-1, 3))
        surface_gamma_flat = surface_gamma.reshape((-1, 3))
        curve_stacks = _curve_stacks_from_grouped_spec(coil_set_spec)
        curve_surface_min_dist = jnp.minimum(
            curve_surface_min_dist,
            pairwise_min_distance_batched_pure(
                _curve_surface_point_pair_batches_from_stacks(
                    curve_stacks,
                    surface_gamma_flat,
                )
            ),
        )
        curve_curve_min_dist = jnp.minimum(
            curve_curve_min_dist,
            pairwise_min_distance_batched_pure(
                _curve_curve_point_pair_batches_from_stacks(curve_stacks)
            ),
        )
        surface_vessel_min_dist = surface_to_surface_shortest_distance_pure(
            surface_gamma,
            vessel_gamma,
        )
    return {
        "solver_success": solver_success,
        "has_G": jnp.asarray(optimize_G, dtype=bool),
        "final_G": G if G is not None else _runtime_float64_scalar(0.0, reference=iota),
        "final_non_qs": raw_terms["non_qs"],
        "final_boozer_residual": raw_terms["residual"],
        "final_iota_penalty": raw_terms["iota"],
        "final_major_radius_penalty": raw_terms["major_radius"],
        "final_length_penalty": raw_terms["length"],
        "final_curve_curve_penalty": raw_terms["curve_curve"],
        "final_curve_surface_penalty": raw_terms["curve_surface"],
        "final_surface_vessel_penalty": raw_terms["surface_vessel"],
        "final_curvature_penalty": raw_terms["curvature"],
        "coil_length": coil_length,
        "max_curvature": max_curvature,
        "optimized_coil_current_A": optimized_current,
        "field_error": field_error,
        "curve_curve_min_dist": curve_curve_min_dist,
        "curve_surface_min_dist": curve_surface_min_dist,
        "surface_vessel_min_dist": surface_vessel_min_dist,
        "final_volume": surface_volume(surface_gamma, surface_normal),
        "final_iota": iota,
    }


def _traceable_reporting_metrics_context(compiled_bundle):
    """Return the immutable state needed by reporting-metrics closures."""
    state = compiled_bundle["state"]
    return (
        state["objective_kwargs"],
        bool(state["optimize_G"]),
        state["coil_set_spec_from_dofs"],
    )


def _make_traceable_reporting_metrics(compiled_bundle, *, include_distance_metrics):
    """Build a pure solved-state reporting summary for one compiled runtime bundle."""
    compiled_forward_result_for = compiled_bundle["compiled_forward_result_for"]
    (
        objective_kwargs,
        optimize_G,
        coil_set_spec_from_dofs,
    ) = _traceable_reporting_metrics_context(compiled_bundle)

    def reporting_metrics(coil_dofs):
        coil_dofs = _as_jax_float64(coil_dofs)
        forward_result = compiled_forward_result_for(coil_dofs)
        return _traceable_reporting_metrics_from_solution(
            objective_kwargs,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=forward_result["x"],
            solver_success=forward_result["success"],
            optimize_G=optimize_G,
            include_distance_metrics=include_distance_metrics,
        )

    return jax.jit(reporting_metrics)


def _make_traceable_reporting_metrics_from_solution(
    compiled_bundle, *, include_distance_metrics
):
    """Build reporting metrics for a caller-provided solved Boozer state."""
    (
        objective_kwargs,
        optimize_G,
        coil_set_spec_from_dofs,
    ) = _traceable_reporting_metrics_context(compiled_bundle)

    def reporting_metrics_from_solution(coil_dofs, solved_x, solver_success):
        coil_dofs = _as_jax_float64(coil_dofs)
        solved_x = _as_jax_float64(solved_x)
        return _traceable_reporting_metrics_from_solution(
            objective_kwargs,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solver_success=jnp.asarray(solver_success, dtype=bool),
            optimize_G=optimize_G,
            include_distance_metrics=include_distance_metrics,
        )

    return jax.jit(reporting_metrics_from_solution)


def _make_traceable_reporting_metrics_from_solution_with_raw_terms(
    compiled_bundle, *, include_distance_metrics
):
    """Build solved-state reporting metrics that can reuse raw outer terms."""
    (
        objective_kwargs,
        optimize_G,
        coil_set_spec_from_dofs,
    ) = _traceable_reporting_metrics_context(compiled_bundle)

    def reporting_metrics_from_solution(
        coil_dofs,
        solved_x,
        solver_success,
        raw_terms_present,
        raw_terms,
    ):
        coil_dofs = _as_jax_float64(coil_dofs)
        solved_x = _as_jax_float64(solved_x)
        return _traceable_reporting_metrics_from_solution(
            objective_kwargs,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solver_success=jnp.asarray(solver_success, dtype=bool),
            optimize_G=optimize_G,
            include_distance_metrics=include_distance_metrics,
            raw_terms=raw_terms,
            raw_terms_present=raw_terms_present,
        )

    return jax.jit(reporting_metrics_from_solution)


def _make_traceable_reporting_metrics_bundle(compiled_bundle):
    """Build the pure reporting-metrics selector for one compiled bundle."""
    reporting_metrics = _make_traceable_reporting_metrics(
        compiled_bundle,
        include_distance_metrics=True,
    )
    reporting_metrics_without_distances = _make_traceable_reporting_metrics(
        compiled_bundle,
        include_distance_metrics=False,
    )

    def reporting_metrics_for(coil_dofs, *, include_distance_metrics=True):
        selected_reporting_metrics = (
            reporting_metrics
            if include_distance_metrics
            else reporting_metrics_without_distances
        )
        return selected_reporting_metrics(coil_dofs)

    return reporting_metrics_for


def _make_traceable_reporting_metrics_from_solution_bundle(compiled_bundle):
    """Build the solved-state reporting selector for one compiled bundle."""
    reporting_metrics_from_solution = _make_traceable_reporting_metrics_from_solution(
        compiled_bundle,
        include_distance_metrics=True,
    )
    reporting_metrics_from_solution_without_distances = (
        _make_traceable_reporting_metrics_from_solution(
            compiled_bundle,
            include_distance_metrics=False,
        )
    )
    reporting_metrics_from_solution_with_raw_terms = (
        _make_traceable_reporting_metrics_from_solution_with_raw_terms(
            compiled_bundle,
            include_distance_metrics=True,
        )
    )
    reporting_metrics_from_solution_without_distances_with_raw_terms = (
        _make_traceable_reporting_metrics_from_solution_with_raw_terms(
            compiled_bundle,
            include_distance_metrics=False,
        )
    )

    def reporting_metrics_from_solution_for(
        coil_dofs,
        solved_x,
        solver_success,
        *,
        include_distance_metrics=True,
        outer_raw_terms=None,
    ):
        if outer_raw_terms is None:
            selected_reporting_metrics = (
                reporting_metrics_from_solution
                if include_distance_metrics
                else reporting_metrics_from_solution_without_distances
            )
            return selected_reporting_metrics(coil_dofs, solved_x, solver_success)
        raw_terms_present, raw_terms = outer_raw_terms
        selected_reporting_metrics = (
            reporting_metrics_from_solution_with_raw_terms
            if include_distance_metrics
            else reporting_metrics_from_solution_without_distances_with_raw_terms
        )
        return selected_reporting_metrics(
            coil_dofs,
            solved_x,
            solver_success,
            raw_terms_present,
            raw_terms,
        )

    return reporting_metrics_from_solution_for


def _hostify_traceable_reporting_metrics(metrics, *, include_distance_metrics):
    """Materialize one traceable reporting-metrics dict on the host."""
    float_metric_names = (
        "final_non_qs",
        "final_boozer_residual",
        "final_iota_penalty",
        "final_major_radius_penalty",
        "final_length_penalty",
        "final_curve_curve_penalty",
        "final_curve_surface_penalty",
        "final_surface_vessel_penalty",
        "final_curvature_penalty",
        "coil_length",
        "max_curvature",
        "optimized_coil_current_A",
        "field_error",
        "final_volume",
        "final_iota",
    )
    distance_metric_names = (
        "curve_curve_min_dist",
        "curve_surface_min_dist",
        "surface_vessel_min_dist",
    )
    has_G = _host_bool(metrics["has_G"])
    host_metrics = {
        "solver_success": _host_bool(metrics["solver_success"]),
        "final_G": None
        if not has_G
        else float(_host_scalar(metrics["final_G"], dtype=np.float64)),
    }
    for metric_name in float_metric_names:
        host_metrics[metric_name] = float(
            _host_scalar(metrics[metric_name], dtype=np.float64)
        )
    for metric_name in distance_metric_names:
        host_metrics[metric_name] = (
            None
            if not include_distance_metrics
            else float(_host_scalar(metrics[metric_name], dtype=np.float64))
        )
    return host_metrics


def _make_traceable_host_reporting_metrics(reporting_metrics):
    """Build a host-normalized solved-state reporting summary wrapper."""

    def host_reporting_metrics(coil_dofs, *, include_distance_metrics=True):
        metrics = reporting_metrics(
            _as_jax_float64(coil_dofs),
            include_distance_metrics=include_distance_metrics,
        )
        return _hostify_traceable_reporting_metrics(
            metrics,
            include_distance_metrics=include_distance_metrics,
        )

    return host_reporting_metrics


def _make_traceable_batched_value_and_grad_pipeline(compiled_value_and_grad_for):
    """Build a scalar-equivalent batched ``(value, grad)`` pipeline for seed scoring."""

    def _batched_value_and_grad_for(coil_dofs_batch):
        coil_dofs_batch = _as_jax_float64(coil_dofs_batch)
        config = seed_batch_sharding_config(coil_dofs_batch)
        if config is not None:
            (coil_dofs_batch,) = maybe_shard_seed_batch_inputs(
                coil_dofs_batch,
                config=config,
            )

            @partial(
                jax.shard_map,
                mesh=config.mesh,
                in_specs=(P(config.axis_name, None),),
                out_specs=(P(config.axis_name), P(config.axis_name, None)),
                check_vma=True,
            )
            def score_seed_shard(coil_dofs_block):
                return lax.map(compiled_value_and_grad_for, coil_dofs_block)

            return score_seed_shard(coil_dofs_batch)

        return lax.map(compiled_value_and_grad_for, coil_dofs_batch)

    return mark_cacheable_jit_value_and_grad(jax.jit(_batched_value_and_grad_for))


def _make_traceable_batched_value_and_grad_boundary(batched_value_and_grad):
    """Build the public pure-JAX batched value/grad entrypoint."""

    def batched_value_and_grad_for(coil_dofs_batch):
        return batched_value_and_grad(_as_jax_float64(coil_dofs_batch))

    return batched_value_and_grad_for


def _classify_nonfinite_scalar(host_value):
    """Classify one non-finite scalar for compact diagnostics."""
    if np.isnan(host_value):
        return "nan"
    if np.isposinf(host_value):
        return "+inf"
    if np.isneginf(host_value):
        return "-inf"
    return None


def _summarize_traceable_scalar(value):
    """Return a compact host summary for one scalar JAX value."""
    host_value = float(_host_scalar(value, dtype=np.float64))
    finite = bool(np.isfinite(host_value))
    return {
        "value": host_value if finite else None,
        "finite": finite,
        "classification": None if finite else _classify_nonfinite_scalar(host_value),
    }


def _summarize_traceable_gradient(gradient):
    """Return a compact host summary for one gradient vector."""
    host_gradient = _host_array(gradient, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(host_gradient)
    all_finite = bool(np.all(finite_mask))
    first_nonfinite_index = None
    if not all_finite:
        first_nonfinite_index = int(np.flatnonzero(~finite_mask)[0])
    return {
        "all_finite": all_finite,
        "inf_norm": float(_host_inf_norm(gradient)) if all_finite else None,
        "size": int(host_gradient.size),
        "nonfinite_count": int(host_gradient.size - int(np.count_nonzero(finite_mask))),
        "first_nonfinite_index": first_nonfinite_index,
    }


def _summarize_traceable_linear_solve_status(status):
    return {
        "residual": _summarize_traceable_scalar(status.residual),
        "residual_relative": _summarize_traceable_scalar(status.residual_relative),
        "iterations": _linear_solve._linear_solve_iterations_host_value(
            status.iterations
        ),
    }


def _traceable_term_adjoint_solve_report(
    booz_jax,
    coil_set_spec_from_dofs,
    *,
    coil_dofs,
    solved_x,
    solved_linear_solve_factors,
    linearization_kind,
    linear_solve_tol,
    linear_solve_stab,
    objective_kwargs,
    term_name,
):
    depends_on_x_inner, _ = _traceable_single_stage_effective_dependency_flags(
        term_name,
        objective_kwargs=objective_kwargs,
    )
    if not depends_on_x_inner:
        return None

    coil_set_spec = coil_set_spec_from_dofs(coil_dofs)

    def objective_of_x(current_x):
        return _evaluate_traceable_weighted_single_stage_outer_term(
            term_name,
            current_x,
            coil_dofs,
            coil_set_spec,
            objective_kwargs,
        )

    rhs = _strict_scalar_grad(objective_of_x, solved_x)
    adjoint, status = _traceable_solve_linearization(
        booz_jax,
        solved_x,
        rhs,
        coil_set_spec,
        objective_kwargs,
        linear_solve_factors=solved_linear_solve_factors,
        linearization_kind=linearization_kind,
        linear_solve_tol=linear_solve_tol,
        linear_solve_stab=linear_solve_stab,
        transpose=True,
    )
    success = _linear_solve._linear_solve_status_success(status)
    report = {
        "success": _host_bool(success),
        **_summarize_traceable_linear_solve_status(status),
        "rhs_norm": _summarize_traceable_scalar(jnp.linalg.norm(rhs)),
        "solution_norm": _summarize_traceable_scalar(jnp.linalg.norm(adjoint)),
        "solution": _summarize_traceable_gradient(adjoint),
    }
    if solved_linear_solve_factors is not None:
        residual = rhs - _traceable_plu_matvec(
            solved_linear_solve_factors,
            adjoint,
            transpose=True,
        )
        residual_norm = jnp.linalg.norm(residual)
        residual_tol = _linear_solve._linear_solve_residual_tolerance(
            rhs,
            linear_solve_tol,
        )
        matrix = _traceable_plu_matrix(solved_linear_solve_factors)
        report["plu"] = {
            "matrix_norm": _summarize_traceable_scalar(jnp.linalg.norm(matrix)),
            "residual_tolerance": _summarize_traceable_scalar(residual_tol),
            "residual_norm": _summarize_traceable_scalar(residual_norm),
            "relative_residual": _summarize_traceable_scalar(
                _linear_solve._relative_residual_norm(residual, rhs)
            ),
        }
    elif linearization_kind == "hessian":
        objective_fn = _make_boozer_penalty_objective_closure(
            coil_set_spec=coil_set_spec,
            decision_split_mode="jvp",
            **_traceable_inner_objective_kwargs(objective_kwargs),
        )
        hvp_fn = _linear_solve._hessian_vector_product_fn(objective_fn)
        candidate_stab = float(linear_solve_stab)
        residual_kwargs = {}
        if _adjoint_linear_solve._ADJOINT_LINEAR_SOLVER == "lsmr_j":
            residual_kwargs["residual_fn"] = _make_boozer_penalty_residual_closure(
                coil_set_spec=coil_set_spec,
                decision_split_mode="jvp",
                **_traceable_inner_objective_kwargs(objective_kwargs),
            )
        solution, attempt_status = (
            _adjoint_linear_solve._solve_hessian_least_squares_system_with_status(
                objective_fn,
                solved_x,
                rhs,
                stab=candidate_stab,
                tol=linear_solve_tol,
                **residual_kwargs,
            )
        )
        attempt_success = _linear_solve._linear_solve_status_success(attempt_status)
        hessian_operator = _adjoint_linear_solve._hessian_linear_operator(
            objective_fn,
            solved_x,
            stab=candidate_stab,
        )
        residual_tol = _linear_solve._linear_solve_residual_tolerance(
            rhs,
            linear_solve_tol,
        )
        residual = rhs - hessian_operator["matvec"](solution)
        residual_norm = jnp.linalg.norm(residual)
        report["hessian_least_squares_operator"] = {
            "attempts": [
                {
                    "stab": candidate_stab,
                    "success": _host_bool(attempt_success),
                    **_summarize_traceable_linear_solve_status(attempt_status),
                    "solution": _summarize_traceable_gradient(solution),
                    "solution_norm": _summarize_traceable_scalar(
                        jnp.linalg.norm(solution)
                    ),
                    "primal_residual_tolerance": _summarize_traceable_scalar(
                        residual_tol
                    ),
                    "primal_residual_norm": _summarize_traceable_scalar(residual_norm),
                    "primal_relative_residual": _summarize_traceable_scalar(
                        _linear_solve._relative_residual_norm(residual, rhs)
                    ),
                }
            ]
        }
        solution, attempt_status = (
            _adjoint_linear_solve._solve_hessian_system_with_status(
                objective_fn,
                solved_x,
                rhs,
                stab=candidate_stab,
                tol=linear_solve_tol,
            )
        )
        attempt_success = _linear_solve._linear_solve_status_success(attempt_status)
        residual = rhs - (
            hvp_fn(solved_x, solution)
            + _runtime_float64_scalar(candidate_stab, reference=solution) * solution
        )
        residual_norm = jnp.linalg.norm(residual)
        report["hessian_operator"] = {
            "attempts": [
                {
                    "stab": candidate_stab,
                    "success": _host_bool(attempt_success),
                    **_summarize_traceable_linear_solve_status(attempt_status),
                    "solution": _summarize_traceable_gradient(solution),
                    "solution_norm": _summarize_traceable_scalar(
                        jnp.linalg.norm(solution)
                    ),
                    "residual_tolerance": _summarize_traceable_scalar(residual_tol),
                    "residual_norm": _summarize_traceable_scalar(residual_norm),
                    "relative_residual": _summarize_traceable_scalar(
                        _linear_solve._relative_residual_norm(residual, rhs)
                    ),
                }
            ]
        }
    return report


def diagnose_traceable_objective_runtime(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
):
    """Return a compact baseline diagnostic report for the target-lane runtime."""
    _traceable_diag_progress("resolve_runtime_entry")
    runtime_entry = _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    compiled_bundle = runtime_entry["compiled_bundle"]
    state = compiled_bundle["state"]
    objective_kwargs = _traceable_runtime_deviceify_tree(state["objective_kwargs"])
    if objective_kwargs["outer_objective_config"] is None:
        raise RuntimeError(
            "Traceable runtime diagnosis requires the full single-stage outer objective."
        )

    _traceable_diag_progress("deviceify_baseline_state")
    baseline_coil_dofs = _traceable_runtime_deviceify_tree(state["baseline_coil_dofs"])
    baseline_x = _traceable_runtime_deviceify_tree(state["baseline_x"])
    baseline_value = _traceable_runtime_deviceify_tree(state["baseline_value"])
    baseline_linear_solve_factors = _traceable_runtime_deviceify_tree(
        state["baseline_linear_solve_factors"]
    )
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]
    baseline_coil_set_spec = coil_set_spec_from_dofs(baseline_coil_dofs)
    optimize_G = bool(state["optimize_G"])
    baseline_sdofs, baseline_iota, baseline_G = _split_x_inner_runtime(
        baseline_x,
        optimize_G,
    )
    baseline_success = _traceable_runtime_deviceify_tree(np.asarray(True, dtype=bool))
    # The gradient diagnosis always evaluates the cached solved baseline. Peel
    # that state directly here so this host-side diagnostic does not spend
    # minutes compiling the full coil-dependent forward-result JIT before it
    # even reaches the actual baseline objective/gradient checks.
    forward_result = _pack_traceable_forward_result(
        value=baseline_value,
        x=baseline_x,
        sdofs=baseline_sdofs,
        iota=baseline_iota,
        G=baseline_G,
        linear_solve_factors=baseline_linear_solve_factors,
        success=baseline_success,
        primal_success=baseline_success,
        adjoint_linear_solve_available=baseline_success,
        newton_trace_capacity=state["newton_trace_capacity"],
    )
    _traceable_diag_progress("baseline_total_gradient")
    total_value = baseline_value
    total_gradient, total_linear_solve_success = _traceable_total_gradient_with_status(
        booz_jax,
        coil_set_spec_from_dofs,
        coil_dofs=baseline_coil_dofs,
        solved_x=baseline_x,
        solved_linear_solve_factors=baseline_linear_solve_factors,
        linearization_kind=state["linearization_kind"],
        linear_solve_tol=state["linear_solve_tol"],
        linear_solve_stab=state["linear_solve_stab"],
        objective_kwargs=objective_kwargs,
    )
    del total_linear_solve_success
    _traceable_diag_progress("raw_term_values")
    raw_terms = _traceable_single_stage_outer_term_values(
        baseline_x,
        baseline_coil_dofs,
        baseline_coil_set_spec,
        **_traceable_total_objective_kwargs(objective_kwargs),
    )
    weighted_terms = _traceable_weighted_single_stage_outer_term_values(
        raw_terms,
        outer_objective_config=objective_kwargs["outer_objective_config"],
    )
    report = {
        "baseline_success": _host_bool(forward_result["success"]),
        "total": {
            "value": _summarize_traceable_scalar(total_value),
            "grad": _summarize_traceable_gradient(total_gradient),
        },
        "terms": {},
    }
    nonfinite_terms = []
    for term_name, weight_key in _TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS:
        _traceable_diag_progress(f"term_gradient:{term_name}")
        (
            direct_grad,
            implicit_grad,
            term_total_grad,
            linear_solve_success,
            _mixed_dense_ir_trust,
        ) = _traceable_objective_gradient_parts(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=baseline_coil_dofs,
            solved_x=baseline_x,
            solved_linear_solve_factors=baseline_linear_solve_factors,
            linearization_kind=state["linearization_kind"],
            linear_solve_tol=state["linear_solve_tol"],
            linear_solve_stab=state["linear_solve_stab"],
            objective_kwargs=objective_kwargs,
            term_name=term_name,
        )
        term_report = {
            "weight": float(
                _host_scalar(
                    objective_kwargs["outer_objective_config"].get(weight_key, 0.0)
                )
            ),
            "raw_value": _summarize_traceable_scalar(raw_terms[term_name]),
            "weighted_value": _summarize_traceable_scalar(weighted_terms[term_name]),
            "direct_grad": _summarize_traceable_gradient(direct_grad),
            "implicit_grad": _summarize_traceable_gradient(implicit_grad),
            "total_grad": _summarize_traceable_gradient(term_total_grad),
            "linear_solve_success": _host_bool(linear_solve_success),
        }
        issues = []
        if not term_report["raw_value"]["finite"]:
            issues.append("raw_value")
        if not term_report["weighted_value"]["finite"]:
            issues.append("weighted_value")
        if not term_report["direct_grad"]["all_finite"]:
            issues.append("direct_grad")
        if not term_report["implicit_grad"]["all_finite"]:
            issues.append("implicit_grad")
        if not term_report["total_grad"]["all_finite"]:
            issues.append("total_grad")
        if not term_report["linear_solve_success"]:
            term_report["adjoint_solve"] = _traceable_term_adjoint_solve_report(
                booz_jax,
                coil_set_spec_from_dofs,
                coil_dofs=baseline_coil_dofs,
                solved_x=baseline_x,
                solved_linear_solve_factors=baseline_linear_solve_factors,
                linearization_kind=state["linearization_kind"],
                linear_solve_tol=state["linear_solve_tol"],
                linear_solve_stab=state["linear_solve_stab"],
                objective_kwargs=objective_kwargs,
                term_name=term_name,
            )
        term_report["issues"] = issues
        report["terms"][term_name] = term_report
        if issues:
            nonfinite_terms.append(term_name)
    report["nonfinite_terms"] = nonfinite_terms
    report["first_nonfinite_term"] = nonfinite_terms[0] if nonfinite_terms else None
    report["all_finite"] = bool(
        report["total"]["value"]["finite"]
        and report["total"]["grad"]["all_finite"]
        and not nonfinite_terms
    )
    _traceable_diag_progress("report_complete")
    return report


def make_traceable_objective(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
    session: TraceableObjectiveSession | None = None,
):
    """Build a pure function ``f(coil_dofs) -> scalar`` for single-stage optimization.

    The returned closure:

    * **Forward**: re-solves the inner Boozer problem from a coil-dependent
      linearized warm-start predictor and returns the exact
      single-stage scalar objective
      ``BoozerResidualJAX + 0.5 * (iota - iota_target)^2``.
    * **No object mutation**: coil geometry is reconstructed directly from
      the explicit ``coil_dofs`` vector, so the traced objective does not
      touch ``bs_jax.x``, ``booz_jax.res``, or descendant Optimizable caches.
    * **No callback seam**: the traced path stays inside JAX primitives;
      there is no ``jax.pure_callback`` bridge back into the stateful
      ``run_code()`` implementation.
    * **Backward**: uses the same implicit-differentiation structure as the
      validated object path, but expressed entirely with pure JAX arrays.

    Args:
        booz_jax: solved :class:`BoozerSurfaceJAX`.
        bs_jax:   :class:`BiotSavartJAX` providing coil geometry.
        iota_target: scalar target iota for the quadratic penalty.
        outer_objective_config: optional structured config enabling the full
            single-stage outer objective. When omitted, the historical traced
            objective remains ``BoozerResidualJAX + 0.5 * (iota-iota_target)^2``.

    Returns:
        ``f(coil_dofs) -> jax.Array`` — traceable scalar objective.

        This is the pure-JAX optimizer contract used by the single-stage
        ondevice lane. Callers that need Python/NumPy materialization should
        use :func:`make_traceable_objective_runtime_bundle` with
        ``include_host_wrappers=True`` and the returned ``host_objective`` /
        ``host_value_and_grad`` wrappers instead of coercing this traced scalar
        directly.
    """
    return _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
        session=session,
    )["objective"]


class TraceableObjectiveSeededValueAndGrad(NamedTuple):
    """Explicit cached baseline seed plus the target-lane value/grad callable."""

    value_and_grad: callable
    optimizer_initial_value_and_grad: tuple[jax.Array, jax.Array]


class TraceableObjectiveCertifiedSeededValueAndGrad(NamedTuple):
    """Seeded value/gradient plus replayable mixed certificate authority."""

    seeded_value_and_grad: TraceableObjectiveSeededValueAndGrad
    certificate_probe_authority: CertificateProbeAuthority | None
    certificate_probe_evidence: CertificateProbeEvidence | None
    mixed_dense_ir_trust: _dense_ir._MixedDenseIrTrustTelemetry


def make_traceable_objective_seeded_value_and_grad(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
    session: TraceableObjectiveSession | None = None,
):
    """Build the explicit cached baseline value/gradient contract.

    The normal ondevice optimizer path should use
    :func:`make_traceable_objective_value_and_grad` and let the optimizer evaluate
    its own first ``(value, grad)`` at ``x0``. This helper is only for callers
    that intentionally need the cached baseline seed as data.
    """
    runtime_entry = _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
        session=session,
    )
    certified_seed = _ensure_traceable_runtime_seeded_value_and_grad(
        runtime_entry,
        booz_jax,
    )
    return certified_seed.seeded_value_and_grad


def make_traceable_objective_certified_seeded_value_and_grad(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
    certificate_probe_key_data: CertificateProbeKeyData | None = None,
    session: TraceableObjectiveSession | None = None,
):
    """Build a seeded value/gradient with explicit fresh-or-replay authority."""
    runtime_entry = _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
        session=session,
    )
    return _make_traceable_runtime_certified_seeded_value_and_grad(
        runtime_entry,
        booz_jax,
        certificate_probe_key_data=certificate_probe_key_data,
    )


def make_traceable_objective_value_and_grad(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
    session: TraceableObjectiveSession | None = None,
):
    """Build a pure-JAX function ``f(coil_dofs) -> (value, grad)`` for ondevice L-BFGS.

    This is the fused outer-optimizer objective contract for the single-stage
    ondevice target lane. It uses the general forward path even at the baseline
    DOFs so the first optimizer evaluation is produced by the same callable as
    later trial points instead of by a cached baseline-gradient seed.

    For host-normalized outputs, use
    ``make_traceable_objective_runtime_bundle(include_host_wrappers=True)``
    and call ``runtime_bundle["host_value_and_grad"]``.
    """
    runtime_entry = _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
        session=session,
    )
    return _ensure_traceable_runtime_optimizer_value_and_grad(runtime_entry, booz_jax)


def make_traceable_solved_state_value_and_grad(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
):
    """Build ``(coil_dofs, solved_x, factors) -> (value, grad)``.

    This is the host-controlled Boozer bridge: Python owns the nonlinear solve
    and passes the solved state into a bounded compiled value/adjoint kernel.
    It intentionally does not build or call the traceable forward-solve graph.
    """
    runtime_entry = _build_traceable_solved_state_value_and_grad_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    return runtime_entry["value_and_grad"]


class TraceableObjectiveSolvedPair(NamedTuple):
    """Decomposed outer objective: a forward solve plus a solved-state value/grad.

    ``solve_fn(coil_dofs) -> forward_result`` runs the device-traceable forward
    Boozer solve and returns the normalized forward-result mapping (keys include
    ``"x"`` the solved decision vector, ``"linear_solve_factors"``, and
    ``"success"``). ``value_grad_from_solved(coil_dofs, solved_x,
    solved_linear_solve_factors) -> (value, grad)`` evaluates the objective and the
    IFT adjoint gradient from an already-solved state, performing no forward solve.

    Cross-evaluation factor-reuse routing (host gate, always on for this pair):

    * **Producer** — seal with ``build_exact_factor_handoff_for(coil_dofs, x,
      factors)`` or ``seal_res_factors_for_reuse(res, coil_dofs=..., solved_x=...)``.
      ``None`` factors stay ``None`` (reuse default-off).
    * **Consumer** — pass the sealed :class:`ExactFactorHandoff` (or ``None``)
      into ``value_grad_from_solved``. The host gate re-requires the integrity
      receipt before device consumption and rejects unaffiliated raw factor
      trees. JIT-internal PLU refine paths are separate and keep raw trees
      under live-operator certification.

    Together they are a faithful split of the fused
    :func:`make_traceable_objective_value_and_grad` callable -- both halves are
    built from one shared compiled-bundle state, so ``forward_result["x"]`` /
    sealed ``forward_result["linear_solve_factors"]`` are exactly what
    ``value_grad_from_solved`` consumes after the host seal/require boundary. A
    host driver that consumes this pair owns the optimizer loop and line search,
    so the per-step jit never encloses the forward solve (the macro-step
    breadth-exclusion gate).

    Parity note: the fused callable gates on ``forward_result["success"]`` and
    falls back to the baseline state for the gradient on solver failure. A host
    driver MUST replicate that branch (inspect ``forward_result["success"]`` and
    pick the baseline-vs-candidate solved state before calling
    ``value_grad_from_solved``) to match the fused path's failure handling.

    Prefer constructing this pair via :class:`TraceableObjectiveSession` so the
    owning session, not ambient ``booz_jax`` attributes, holds mutable caches.
    """

    solve_fn: Callable
    value_grad_from_solved: Callable
    build_exact_factor_handoff_for: Callable | None = None
    require_exact_factor_handoff_for: Callable | None = None
    seal_res_factors_for_reuse: Callable | None = None


class TraceableObjectiveSession:
    """Owned session for traceable objective runners and mutable runtime state.

    Owns compiled/traceable runners and the mutable per-session caches that used
    to live as ambient attributes on ``booz_jax`` or as closed-over nonlocal
    state. Two sessions never share those caches: mutations on session A cannot
    affect session B.

    Configuration needed by the solve / value_grad pair is captured in
    ``compiled_bundle["state"]`` at construction. Lazy fields are explicit
    instance attributes (not dynamic ``setattr`` bags).

    Fused, decomposed, reporting, host-wrapper, profile, and ALM entrypoints all
    derive from the session-owned ``runtime_entry``. No cache state is attached
    to ``booz_jax``.
    """

    __slots__ = (
        "_materialize_lock",
        "_optimizer_compiled_bundle",
        "_optimizer_solved_pair",
        "_trial_evaluator",
        "alm_runtime_bundles",
        "baseline_value_and_grad_state",
        "batched_value_and_grad",
        "booz_jax",
        "cache_key",
        "compiled_bundle",
        "host_objective",
        "host_reporting_metrics",
        "host_reporting_state",
        "host_value_and_grad",
        "objective",
        "optimizer_value_and_grad",
        "profile_suite",
        "public_batched_value_and_grad",
        "public_forward_result",
        "public_objective",
        "public_reporting_metrics",
        "public_reporting_metrics_from_solution",
        "public_value_and_grad",
        "reporting_metrics",
        "reporting_metrics_from_solution",
        "seeded_compiled_bundle",
        "seeded_value_and_grad",
        "solved_state_value_and_grad",
        "success_filter",
    )

    def __init__(
        self,
        *,
        cache_key,
        success_filter,
        compiled_bundle,
        booz_jax,
        optimizer_compiled_bundle=None,
        objective=None,
        batched_value_and_grad=None,
    ):
        self.cache_key = cache_key
        self.success_filter = success_filter
        self.compiled_bundle = compiled_bundle
        self.booz_jax = booz_jax
        self.objective = objective
        self.batched_value_and_grad = batched_value_and_grad
        self.reporting_metrics = None
        self.reporting_metrics_from_solution = None
        self.public_objective = None
        self.public_value_and_grad = None
        self.public_batched_value_and_grad = None
        self.public_forward_result = None
        self.public_reporting_metrics = None
        self.public_reporting_metrics_from_solution = None
        self.host_objective = None
        self.host_value_and_grad = None
        self.host_reporting_metrics = None
        self.profile_suite = None
        self.optimizer_value_and_grad = None
        self.seeded_compiled_bundle = None
        self.seeded_value_and_grad = None
        self.alm_runtime_bundles = {}
        self.host_reporting_state = None
        self.baseline_value_and_grad_state = None
        self.solved_state_value_and_grad = None
        self._optimizer_compiled_bundle = optimizer_compiled_bundle
        self._optimizer_solved_pair = None
        self._trial_evaluator = None
        self._materialize_lock = Lock()

    @property
    def runtime_entry(self):
        """Compatibility view: the session itself is the sole runtime entry."""
        return self

    def __getitem__(self, key):
        if key == "state":
            return self.compiled_bundle["state"]
        if key == "value_and_grad":
            return self.solved_state_value_and_grad
        if key == "optimizer_compiled_bundle":
            return self._optimizer_compiled_bundle
        if key == "optimizer_solved_pair":
            return self._optimizer_solved_pair
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key == "optimizer_compiled_bundle":
            self._optimizer_compiled_bundle = value
            return
        if key == "optimizer_solved_pair":
            self._optimizer_solved_pair = value
            return
        setattr(self, key, value)

    def get(self, key, default=None):
        if key == "optimizer_compiled_bundle":
            value = self._optimizer_compiled_bundle
        elif key == "optimizer_solved_pair":
            value = self._optimizer_solved_pair
        elif key == "state":
            value = self.compiled_bundle["state"]
        elif key == "value_and_grad":
            value = self.solved_state_value_and_grad
        else:
            value = getattr(self, key, default)
        return default if value is None else value

    def __contains__(self, key):
        return self.get(key) is not None

    @classmethod
    def from_optimizer_compiled_bundle(
        cls,
        optimizer_compiled_bundle,
        *,
        booz_jax=None,
        cache_key=None,
        success_filter=None,
        compiled_bundle=None,
    ):
        """Build a session that already owns the optimizer compiled bundle.

        Used by focused unit tests and residual bridges that materialize the
        general-only-forward optimizer bundle without re-entering the factory.
        """
        if compiled_bundle is None:
            compiled_bundle = optimizer_compiled_bundle
        return cls(
            cache_key=cache_key,
            success_filter=success_filter,
            compiled_bundle=compiled_bundle,
            booz_jax=booz_jax,
            optimizer_compiled_bundle=optimizer_compiled_bundle,
        )

    def ensure_optimizer_compiled_bundle(self):
        """Return the general-only-forward optimizer compiled bundle (session-owned)."""
        if self._optimizer_compiled_bundle is not None:
            return self._optimizer_compiled_bundle
        with self._materialize_lock:
            if self._optimizer_compiled_bundle is not None:
                return self._optimizer_compiled_bundle
            if self.booz_jax is None:
                raise RuntimeError(
                    "TraceableObjectiveSession cannot build an optimizer compiled "
                    "bundle without booz_jax; pass optimizer_compiled_bundle at "
                    "construction or construct via make_traceable_objective_session()."
                )
            state = self.compiled_bundle["state"]
            self._optimizer_compiled_bundle = (
                _build_traceable_objective_compiled_bundle_from_state(
                    self.booz_jax,
                    state,
                    success_filter=self.success_filter,
                    general_only_forward=True,
                )
            )
            return self._optimizer_compiled_bundle

    def solved_pair(self) -> TraceableObjectiveSolvedPair:
        """Return the session-owned ``(solve_fn, value_grad_from_solved)`` pair.

        Materializes once per session. R06 host factor-handoff gates remain on
        ``value_grad_from_solved`` and the pair's handoff builders.
        """
        if self._optimizer_solved_pair is not None:
            return self._optimizer_solved_pair
        with self._materialize_lock:
            if self._optimizer_solved_pair is not None:
                return self._optimizer_solved_pair
            optimizer_compiled_bundle = self._optimizer_compiled_bundle
            if optimizer_compiled_bundle is None:
                if self.booz_jax is None:
                    raise RuntimeError(
                        "TraceableObjectiveSession cannot build an optimizer "
                        "compiled bundle without booz_jax; pass "
                        "optimizer_compiled_bundle at construction or construct "
                        "via make_traceable_objective_session()."
                    )
                state = self.compiled_bundle["state"]
                optimizer_compiled_bundle = (
                    _build_traceable_objective_compiled_bundle_from_state(
                        self.booz_jax,
                        state,
                        success_filter=self.success_filter,
                        general_only_forward=True,
                    )
                )
                self._optimizer_compiled_bundle = optimizer_compiled_bundle
            self._optimizer_solved_pair = _build_traceable_optimizer_solved_pair(
                optimizer_compiled_bundle
            )
            return self._optimizer_solved_pair

    def trial_evaluator(self) -> Callable:
        """Return the cached opt-in evaluator for immutable Boozer trial records."""
        if self._trial_evaluator is not None:
            return self._trial_evaluator
        with self._materialize_lock:
            if self._trial_evaluator is not None:
                return self._trial_evaluator
            optimizer_compiled_bundle = self._optimizer_compiled_bundle
            if optimizer_compiled_bundle is None:
                if self.booz_jax is None:
                    raise RuntimeError(
                        "TraceableObjectiveSession cannot build a trial evaluator "
                        "without booz_jax; pass optimizer_compiled_bundle at "
                        "construction or construct via "
                        "make_traceable_objective_session()."
                    )
                optimizer_compiled_bundle = (
                    _build_traceable_objective_compiled_bundle_from_state(
                        self.booz_jax,
                        self.compiled_bundle["state"],
                        success_filter=self.success_filter,
                        general_only_forward=True,
                    )
                )
                self._optimizer_compiled_bundle = optimizer_compiled_bundle
            self._trial_evaluator = _build_traceable_trial_evaluator(
                optimizer_compiled_bundle
            )
            return self._trial_evaluator

    def evaluate_candidate_from_anchor(
        self,
        parameters: np.ndarray,
        incumbent_state: TraceableObjectiveInnerState,
    ) -> TraceableObjectiveCandidateEvaluation:
        """Evaluate one candidate from an explicit continuation anchor.

        Runs the shared forward/gradient core once and returns its device-resident
        forward state and candidate gradient. A fallback gradient is unavailable.
        """
        if not _host_bool(incumbent_state.eligible):
            raise RuntimeError("candidate evaluation requires an eligible anchor")
        canonical = np.ascontiguousarray(
            parameters,
            dtype=np.dtype("<f8"),
        ).reshape(-1)
        candidate = _runtime_device_put(canonical, dtype=jnp.float64)
        evaluate_candidate = _build_candidate_evaluation_core(
            self.ensure_optimizer_compiled_bundle(),
            fallback_gradient_on_primal_failure=False,
        )
        evaluation = evaluate_candidate(candidate, incumbent_state)
        jax.block_until_ready(evaluation)
        return evaluation

    def accepted_incumbent_host_value_and_grad(
        self,
    ) -> AcceptedIncumbentHostValueAndGrad:
        """Mint an isolated accepted-incumbent controller seeded from baseline.

        Each call returns a fresh controller; controllers never share incumbent
        or pending state. Promotion happens only through ``accept``, which the
        host driver must call from its post-acceptance callback.
        """
        bundle = self.ensure_optimizer_compiled_bundle()
        state = bundle["state"]
        initial_state = TraceableObjectiveInnerState(
            coil_dofs=_as_jax_float64(state["baseline_coil_dofs"]),
            solved_x=_as_jax_float64(state["baseline_x"]),
            objective_value=jnp.asarray(state["baseline_value"], dtype=jnp.float64),
            eligible=_runtime_device_put(np.bool_(True)),
        )
        return AcceptedIncumbentHostValueAndGrad(
            _build_accepted_incumbent_evaluator(bundle),
            initial_state,
        )

    def clear_solved_pair_cache(self):
        """Drop the session-local solved-pair cache (isolation / reset helper)."""
        with self._materialize_lock:
            self._optimizer_solved_pair = None


def make_traceable_objective_session(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
):
    """Build an owned :class:`TraceableObjectiveSession` for decomposed objectives.

    The session owns compiled runners and mutable caches. It does **not** attach
    cache state to ``booz_jax``; callers that need isolation construct one session
    per optimizer run. Public solved-pair entrypoints construct a session
    internally and return its pair.
    """
    runtime_entry = _build_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    return runtime_entry


def _traceable_runtime_entry_from_session(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
):
    """Return the runtime entry retained by a fresh explicit owner session."""
    return make_traceable_objective_session(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    ).runtime_entry


def _get_cached_traceable_runtime_entry(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
    session: TraceableObjectiveSession | None = None,
):
    """Resolve an explicitly retained session or build an isolated one-shot entry."""
    if session is not None:
        return session.runtime_entry
    return _traceable_runtime_entry_from_session(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )


def make_traceable_objective_solved_pair(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    success_filter=None,
    session: TraceableObjectiveSession | None = None,
):
    """Build the decomposed ``(solve_fn, value_grad_from_solved)`` outer objective.

    This is the host-driven counterpart to
    :func:`make_traceable_objective_value_and_grad`: rather than a single fused
    ``f(coil_dofs) -> (value, grad)`` whose jit encloses the forward Boozer solve,
    it returns a :class:`TraceableObjectiveSolvedPair` whose two halves share one
    compiled-bundle state. The optimizer can then run the forward solve and the
    solved-state value/adjoint as separate device kernels under a host-owned loop.

    Thin public adapter: constructs a :class:`TraceableObjectiveSession` and
    returns its session-owned pair. Prefer :func:`make_traceable_objective_session`
    when the host needs isolatable ownership of caches and runners.

    See :class:`TraceableObjectiveSolvedPair` for the contract and the failure-mode
    parity note the host driver must honor.
    """
    if session is None:
        session = make_traceable_objective_session(
            booz_jax,
            bs_jax,
            iota_target,
            outer_objective_config=outer_objective_config,
            success_filter=success_filter,
        )
    return session.solved_pair()


def _make_traceable_forward_value_pipeline(compiled_forward_result_for):
    def _forward_value_for(coil_dofs):
        return compiled_forward_result_for(coil_dofs)["value"]

    return jax.jit(_forward_value_for)


def _make_traceable_field_eval_sharding_pipeline(field_at_solution_for):
    compiled_field_at_solution_for = jax.jit(field_at_solution_for)

    def _field_eval_sharding(coil_dofs):
        return inspect_array_sharding_summary(compiled_field_at_solution_for(coil_dofs))

    return _field_eval_sharding


def _make_traceable_objective_profile_suite_from_compiled_bundle(
    compiled_bundle,
    booz_jax,
    bs_jax,
    *,
    value_and_grad_pipeline=None,
    batched_value_and_grad_pipeline=None,
):
    """Build profiling closures from the shared traceable runtime bundle."""
    state = compiled_bundle["state"]
    objective_kwargs = state["objective_kwargs"]
    baseline_coil_dofs = state["baseline_coil_dofs"]
    baseline_x = state["baseline_x"]
    baseline_linear_solve_factors = state["baseline_linear_solve_factors"]
    optimize_G = state["optimize_G"]
    predictor_kind = state["predictor_kind"]
    linearization_kind = state["linearization_kind"]
    linear_solve_tol = state["linear_solve_tol"]
    linear_solve_stab = state["linear_solve_stab"]
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]
    compiled_forward_result_for = compiled_bundle["compiled_forward_result_for"]
    resolved_value_and_grad_pipeline = (
        compiled_bundle["compiled_value_and_grad_for"]
        if value_and_grad_pipeline is None
        else value_and_grad_pipeline
    )
    resolved_batched_value_and_grad_pipeline = (
        _make_traceable_batched_value_and_grad_pipeline(
            compiled_bundle["compiled_value_and_grad_for"]
        )
        if batched_value_and_grad_pipeline is None
        else batched_value_and_grad_pipeline
    )

    def _warmstart_for(coil_dofs):
        warmstart_x, warmstart_linear_solve_success = _traceable_predict_warmstart_x(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            baseline_coil_dofs=baseline_coil_dofs,
            baseline_x=baseline_x,
            baseline_linear_solve_factors=baseline_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            predictor_kind=predictor_kind,
            objective_kwargs=objective_kwargs,
        )
        return {
            "x": warmstart_x,
            "success": warmstart_linear_solve_success,
        }

    def _current_incumbent_warmstart_for(
        coil_dofs,
        anchor_coil_dofs,
        anchor_x,
        anchor_linear_solve_factors,
        anchor_eligible,
    ):
        anchor_eligible = jnp.asarray(anchor_eligible, dtype=bool)
        selected_anchor_coil_dofs = lax.select(
            anchor_eligible,
            _as_jax_float64(anchor_coil_dofs),
            baseline_coil_dofs,
        )
        selected_anchor_x = lax.select(
            anchor_eligible,
            _as_jax_float64(anchor_x),
            baseline_x,
        )
        selected_anchor_linear_solve_factors = (
            _traceable_select_predictor_linear_solve_factors(
                anchor_eligible,
                baseline_linear_solve_factors=baseline_linear_solve_factors,
                anchor_linear_solve_factors=anchor_linear_solve_factors,
            )
        )
        warmstart_x, warmstart_linear_solve_success = (
            _traceable_predict_warmstart_from_anchor(
                booz_jax,
                coil_set_spec_from_dofs,
                coil_dofs=coil_dofs,
                anchor_coil_dofs=selected_anchor_coil_dofs,
                anchor_x=selected_anchor_x,
                anchor_linear_solve_factors=selected_anchor_linear_solve_factors,
                linearization_kind=linearization_kind,
                linear_solve_tol=linear_solve_tol,
                linear_solve_stab=linear_solve_stab,
                predictor_kind=predictor_kind,
                objective_kwargs=objective_kwargs,
            )
        )
        return {
            "x": warmstart_x,
            "success": warmstart_linear_solve_success,
            "anchor_used": anchor_eligible,
        }

    def _solve_for(coil_dofs):
        coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
        warmstart = _warmstart_for(coil_dofs)
        warmstart_x = warmstart["x"]
        warmstart_linear_solve_success = warmstart["success"]

        def _run_traceable_solve(_):
            warmstart_sdofs, warmstart_iota, warmstart_G = (
                booz_jax._unpack_decision_vector_jax(
                    warmstart_x,
                    optimize_G,
                    coil_set_spec=coil_set_spec,
                )
            )
            solve_result = booz_jax.run_code_traceable(
                coil_set_spec,
                warmstart_sdofs,
                warmstart_iota,
                warmstart_G,
                materialize_dense_linearization=False,
            )
            solved_sdofs, solved_iota, solved_G = _resolve_traceable_solved_state(
                booz_jax,
                solve_result,
                optimize_G=optimize_G,
                coil_set_spec=coil_set_spec,
            )
            return {
                "x": solve_result["x"],
                "sdofs": solved_sdofs,
                "iota": solved_iota,
                "G": solved_G,
                "fun": solve_result["fun"],
                "linear_solve_factors": _traceable_result_linear_solve_factors(
                    solve_result,
                    linearization_kind,
                ),
                "success": solve_result["success"],
                "nit": _runtime_int32_scalar(solve_result["nit"]),
            }

        if linearization_kind != "exact_jacobian":
            return _run_traceable_solve(None)

        def _warmstart_failure(_):
            warmstart_sdofs, warmstart_iota, warmstart_G = (
                booz_jax._unpack_decision_vector_jax(
                    warmstart_x,
                    optimize_G,
                    coil_set_spec=coil_set_spec,
                )
            )
            warmstart_fun = _evaluate_traceable_total_objective(
                warmstart_x,
                coil_dofs,
                coil_set_spec,
                objective_kwargs,
            )
            return {
                "x": warmstart_x,
                "sdofs": warmstart_sdofs,
                "iota": warmstart_iota,
                "G": warmstart_G,
                "fun": warmstart_fun,
                "linear_solve_factors": None,
                "success": _runtime_bool(False),
                "nit": _runtime_int32_scalar(0),
            }

        return lax.cond(
            warmstart_linear_solve_success,
            _run_traceable_solve,
            _warmstart_failure,
            operand=None,
        )

    def _surface_geometry_for(solved_x):
        sdofs, _, _ = _split_x_inner_runtime(solved_x, optimize_G)
        return _surface_geometry_from_dofs(
            sdofs,
            objective_kwargs["quadpoints_phi"],
            objective_kwargs["quadpoints_theta"],
            objective_kwargs["mpol"],
            objective_kwargs["ntor"],
            objective_kwargs["nfp"],
            objective_kwargs["stellsym"],
            objective_kwargs["scatter_indices"],
            surface_kind=objective_kwargs["surface_kind"],
        )

    def _field_for(coil_dofs, solved_x):
        coil_set_spec = coil_set_spec_from_dofs(coil_dofs)
        gamma, _, _ = _surface_geometry_for(solved_x)
        points = gamma.reshape(-1, 3)
        return grouped_biot_savart_B_from_spec(points, coil_set_spec)

    def _field_at_solution_for(coil_dofs):
        return _field_for(coil_dofs, _solve_for(coil_dofs)["x"])

    def _solved_total_objective_for(coil_dofs, solved_x):
        return _evaluate_traceable_total_objective(
            solved_x,
            coil_dofs,
            coil_set_spec_from_dofs(coil_dofs),
            objective_kwargs,
        )

    def _total_gradient_for(coil_dofs, solved_x, solved_linear_solve_factors):
        return _traceable_total_gradient(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_linear_solve_factors=solved_linear_solve_factors,
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            objective_kwargs=objective_kwargs,
        )

    compiled_forward_value_for = _make_traceable_forward_value_pipeline(
        compiled_forward_result_for
    )
    compiled_warmstart_for = jax.jit(_warmstart_for)
    compiled_current_incumbent_warmstart_for = jax.jit(_current_incumbent_warmstart_for)
    compiled_inner_solve_for = jax.jit(_solve_for)
    compiled_surface_geometry_for = jax.jit(_surface_geometry_for)
    compiled_field_for = jax.jit(_field_for)
    compiled_field_eval_sharding = _make_traceable_field_eval_sharding_pipeline(
        _field_at_solution_for
    )
    compiled_solved_total_objective_for = jax.jit(_solved_total_objective_for)
    compiled_solved_total_gradient_for = jax.jit(_total_gradient_for)

    return {
        "forward_result": compiled_forward_result_for,
        "forward_value": compiled_forward_value_for,
        "warmstart_predict": compiled_warmstart_for,
        "current_incumbent_warmstart_predict": (
            compiled_current_incumbent_warmstart_for
        ),
        "inner_solve": compiled_inner_solve_for,
        "surface_geometry": compiled_surface_geometry_for,
        "field_eval": compiled_field_for,
        "field_eval_sharding": compiled_field_eval_sharding,
        "solved_total_objective": compiled_solved_total_objective_for,
        "solved_total_gradient": compiled_solved_total_gradient_for,
        "value_and_grad_pipeline": resolved_value_and_grad_pipeline,
        "batched_value_and_grad_pipeline": resolved_batched_value_and_grad_pipeline,
    }


def make_traceable_objective_runtime_bundle(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    include_profile_suite=False,
    include_host_wrappers=False,
    outer_objective_config=None,
    success_filter=None,
    session: TraceableObjectiveSession | None = None,
):
    """Build the shared runtime bundle for the target single-stage objective path.

    Pass an explicit ``session`` to retain compiled runners and lazily built
    profile/host boundaries across calls. Without one, this factory builds an
    isolated one-shot session. Rebuild the session after changing captured
    inputs; an existing session does not retarget itself.

    Returned keys:

    ``objective``
        Pure JAX scalar callable returning a 0-d ``jax.Array``.
    ``value_and_grad``
        Pure JAX callable returning ``(0-d jax.Array, grad jax.Array)``.
    ``batched_value_and_grad``
        Pure JAX callable returning batched ``(value, grad)`` outputs for a
        ``(batch, dof)`` seed array.
    ``forward_result``
        Pure JAX callable returning the traceable inner-solve result used by
        the target-lane objective, including the solved decision vector,
        unpacked Boozer state, and success flags.
    ``reporting_metrics``
        Pure JAX callable returning the solved-state reporting scalars used by
        the single-stage example. Callers that need Python/NumPy materialization
        can host-normalize this explicit boundary themselves, or request the
        companion ``host_reporting_metrics`` wrapper. This entrypoint resolves
        lazily and requires ``outer_objective_config`` when invoked.
    ``reporting_metrics_from_solution``
        Pure JAX callable returning the same reporting scalars from a caller-
        provided packed solved state, avoiding a second forward solve.
    ``host_objective``
        Optional host-normalized callable returning a Python ``float`` when
        ``include_host_wrappers=True``.
    ``host_value_and_grad``
        Optional host-normalized callable returning ``(float, np.ndarray)``
        when ``include_host_wrappers=True``.
    ``host_reporting_metrics``
        Optional host-normalized callable returning the final solved-state
        reporting scalars used by the single-stage example when
        ``include_host_wrappers=True``.
    ``profile_suite``
        Optional profiled pure-JAX closures when ``include_profile_suite=True``.
    """
    runtime_entry = _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
        session=session,
    )
    _ensure_traceable_runtime_public_boundaries(runtime_entry)
    runtime_bundle = {
        "objective": runtime_entry["public_objective"],
        "value_and_grad": runtime_entry["public_value_and_grad"],
        "batched_value_and_grad": runtime_entry["public_batched_value_and_grad"],
        "forward_result": runtime_entry["public_forward_result"],
        "reporting_metrics": runtime_entry["public_reporting_metrics"],
        "reporting_metrics_from_solution": runtime_entry[
            "public_reporting_metrics_from_solution"
        ],
    }
    if include_host_wrappers:
        _ensure_traceable_runtime_host_wrappers(runtime_entry, booz_jax)
        runtime_bundle.update(
            {
                "host_objective": runtime_entry["host_objective"],
                "host_value_and_grad": runtime_entry["host_value_and_grad"],
                "host_reporting_metrics": runtime_entry["host_reporting_metrics"],
            }
        )
    if not include_profile_suite:
        return runtime_bundle
    compiled_bundle = runtime_entry["compiled_bundle"]
    if runtime_entry["profile_suite"] is None:
        runtime_entry["profile_suite"] = (
            _make_traceable_objective_profile_suite_from_compiled_bundle(
                compiled_bundle,
                booz_jax,
                bs_jax,
                value_and_grad_pipeline=runtime_entry["public_value_and_grad"],
                batched_value_and_grad_pipeline=runtime_entry[
                    "public_batched_value_and_grad"
                ],
            )
        )
    runtime_bundle["profile_suite"] = runtime_entry["profile_suite"]
    return runtime_bundle


def make_traceable_single_stage_alm_runtime_bundle(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config,
    alm_config,
    success_filter=None,
    session: TraceableObjectiveSession | None = None,
):
    """Build the pure-JAX single-stage ALM runtime bundle for the inner solve.

    The returned bundle keeps the hot ALM subproblem entirely in JAX and is
    intended for ``backend='jax', optimizer_backend='ondevice'`` single-stage
    ALM inner solves. Host-side accepted-step reporting and artifact shaping
    remain outside this bundle on explicit Python boundaries. ``success_filter``
    is optional and can reject solved states, for example when the host ALM
    reference lane would treat a self-intersecting surface as an immediate
    failure.
    """
    if outer_objective_config is None:
        raise ValueError(
            "make_traceable_single_stage_alm_runtime_bundle() requires "
            "outer_objective_config."
        )
    normalized_alm_config = _traceable_runtime_hostify_tree(dict(alm_config))
    runtime_entry = _get_cached_traceable_runtime_entry(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
        session=session,
    )
    alm_cache_key = _traceable_contract_tree_signature(normalized_alm_config)
    cached_bundle = runtime_entry["alm_runtime_bundles"].get(alm_cache_key)
    if cached_bundle is not None:
        return cached_bundle

    compiled_bundle = runtime_entry["compiled_bundle"]
    state = compiled_bundle["state"]
    objective_kwargs = dict(
        state["objective_kwargs"],
        outer_objective_config=_traceable_runtime_hostify_tree(outer_objective_config),
    )
    coil_set_spec_from_dofs = state["coil_set_spec_from_dofs"]
    compiled_forward_result_for = compiled_bundle["compiled_forward_result_for"]
    baseline_coil_dofs = state["baseline_coil_dofs"]
    baseline_x = state["baseline_x"]
    linearization_kind = state["linearization_kind"]
    linear_solve_tol = state["linear_solve_tol"]
    linear_solve_stab = state["linear_solve_stab"]
    constraint_names = tuple(
        str(name) for name in normalized_alm_config["constraint_names"]
    )
    normalized_alm_config["constraint_names"] = constraint_names

    def _alm_total_for(coil_dofs, solved_x, multipliers, penalty):
        return _traceable_single_stage_alm_evaluation(
            solved_x,
            coil_dofs,
            coil_set_spec_from_dofs(coil_dofs),
            objective_kwargs=objective_kwargs,
            alm_config=normalized_alm_config,
            multipliers=multipliers,
            penalty=penalty,
        )["total"]

    def _failure_evaluation(forward_result, multipliers, penalty):
        baseline_total = _alm_total_for(
            baseline_coil_dofs,
            baseline_x,
            multipliers,
            penalty,
        )
        failure_total = _traceable_rejected_objective_value(
            forward_result["value"],
            baseline_total,
        )
        constraint_values = jnp.broadcast_to(
            failure_total,
            (len(constraint_names),),
        )
        return {
            "total": failure_total,
            "base_total": failure_total,
            "physics_total": failure_total,
            "constraint_values": constraint_values,
            "feasibility_values": constraint_values,
            "x": forward_result["x"],
            "linear_solve_factors": forward_result["linear_solve_factors"],
            "success": forward_result["success"],
        }

    def _normalize_runtime_inputs(coil_dofs, multipliers, penalty):
        return (
            _as_jax_float64(coil_dofs),
            _as_jax_float64(multipliers),
            _as_jax_float64(penalty),
        )

    def _alm_evaluation_for(coil_dofs, multipliers, penalty):
        coil_dofs, multipliers, penalty = _normalize_runtime_inputs(
            coil_dofs,
            multipliers,
            penalty,
        )
        forward_result = compiled_forward_result_for(coil_dofs)

        def _success(_):
            evaluation = _traceable_single_stage_alm_evaluation(
                forward_result["x"],
                coil_dofs,
                coil_set_spec_from_dofs(coil_dofs),
                objective_kwargs=objective_kwargs,
                alm_config=normalized_alm_config,
                multipliers=multipliers,
                penalty=penalty,
            )
            evaluation["x"] = forward_result["x"]
            evaluation["linear_solve_factors"] = forward_result["linear_solve_factors"]
            evaluation["success"] = forward_result["success"]
            return evaluation

        return jax.lax.cond(
            forward_result["success"],
            _success,
            lambda _: _failure_evaluation(forward_result, multipliers, penalty),
            operand=None,
        )

    compiled_evaluation_for = jax.jit(_alm_evaluation_for)

    def _alm_total_gradient_for(
        coil_dofs,
        solved_x,
        solved_linear_solve_factors,
        multipliers,
        penalty,
    ):
        def _scalar_objective_fn(
            x_inner,
            current_coil_dofs,
            coil_set_spec,
            *,
            objective_kwargs,
        ):
            return _traceable_single_stage_alm_evaluation(
                x_inner,
                current_coil_dofs,
                coil_set_spec,
                objective_kwargs=objective_kwargs,
                alm_config=normalized_alm_config,
                multipliers=multipliers,
                penalty=penalty,
            )["total"]

        return _traceable_total_gradient_with_status(
            booz_jax,
            coil_set_spec_from_dofs,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solved_linear_solve_factors=_traceable_runtime_deviceify_tree(
                solved_linear_solve_factors
            ),
            linearization_kind=linearization_kind,
            linear_solve_tol=linear_solve_tol,
            linear_solve_stab=linear_solve_stab,
            objective_kwargs=objective_kwargs,
            scalar_objective_fn=_scalar_objective_fn,
        )

    compiled_total_gradient_for = jax.jit(_alm_total_gradient_for)

    def _alm_failure_gradient_for(multipliers, penalty):
        def _objective_of_coils(current_coil_dofs):
            return _traceable_single_stage_alm_evaluation(
                baseline_x,
                current_coil_dofs,
                coil_set_spec_from_dofs(current_coil_dofs),
                objective_kwargs=objective_kwargs,
                alm_config=normalized_alm_config,
                multipliers=multipliers,
                penalty=penalty,
            )["total"]

        return _strict_scalar_grad(_objective_of_coils, baseline_coil_dofs)

    compiled_failure_gradient_for = jax.jit(_alm_failure_gradient_for)

    @jax.custom_vjp
    def _objective(coil_dofs, multipliers, penalty):
        return compiled_evaluation_for(coil_dofs, multipliers, penalty)["total"]

    def _objective_fwd(coil_dofs, multipliers, penalty):
        evaluation = compiled_evaluation_for(coil_dofs, multipliers, penalty)
        # Phase 2 (docs/parity_scientific_equivalence_contract_2026-05-09.md
        # §5.3 / §6): stop_gradient on the cached factor state so the
        # IFT adjoint backward pass cannot retrace into the linear-solve
        # factorization graph.
        return evaluation["total"], (
            coil_dofs,
            lax.stop_gradient(evaluation["x"]),
            jax.tree.map(lax.stop_gradient, evaluation["linear_solve_factors"]),
            evaluation["success"],
            multipliers,
            penalty,
        )

    def _objective_bwd(saved_state, cotangent):
        (
            coil_dofs,
            solved_x,
            solved_linear_solve_factors,
            success,
            multipliers,
            penalty,
        ) = saved_state
        solved_linear_solve_factors = _traceable_runtime_deviceify_tree(
            solved_linear_solve_factors
        )

        def _accepted_candidate_gradient(_):
            grad, linear_solve_success = compiled_total_gradient_for(
                coil_dofs,
                solved_x,
                solved_linear_solve_factors,
                multipliers,
                penalty,
            )
            return _traceable_adjoint_gradient_or_nan(grad, linear_solve_success)

        def _rejected_candidate_gradient(_):
            return compiled_failure_gradient_for(multipliers, penalty)

        grad = lax.cond(
            success,
            _accepted_candidate_gradient,
            _rejected_candidate_gradient,
            operand=None,
        )
        multipliers_bar = _runtime_zeros_like(multipliers)
        penalty_bar = _runtime_float64_scalar(0.0, reference=grad)
        return (
            _as_runtime_float64(cotangent, reference=grad) * grad,
            multipliers_bar,
            penalty_bar,
        )

    _objective.defvjp(_objective_fwd, _objective_bwd)
    compiled_objective = jax.jit(_objective)

    def objective(coil_dofs, multipliers, penalty):
        coil_dofs, multipliers, penalty = _normalize_runtime_inputs(
            coil_dofs,
            multipliers,
            penalty,
        )
        return compiled_objective(
            coil_dofs,
            multipliers,
            penalty,
        )

    def evaluate(coil_dofs, multipliers, penalty):
        coil_dofs, multipliers, penalty = _normalize_runtime_inputs(
            coil_dofs,
            multipliers,
            penalty,
        )
        return compiled_evaluation_for(
            coil_dofs,
            multipliers,
            penalty,
        )

    @mark_cacheable_jit_value_and_grad
    @jax.jit
    def value_and_grad(coil_dofs, multipliers, penalty):
        return jax.value_and_grad(_objective, argnums=0)(
            coil_dofs,
            multipliers,
            penalty,
        )

    def public_value_and_grad(coil_dofs, multipliers, penalty):
        coil_dofs, multipliers, penalty = _normalize_runtime_inputs(
            coil_dofs,
            multipliers,
            penalty,
        )
        return value_and_grad(
            coil_dofs,
            multipliers,
            penalty,
        )

    alm_runtime_bundle = {
        "objective": objective,
        "evaluate": evaluate,
        "value_and_grad": public_value_and_grad,
        "constraint_names": constraint_names,
    }
    runtime_entry["alm_runtime_bundles"][alm_cache_key] = alm_runtime_bundle
    return alm_runtime_bundle


def make_traceable_objective_profile_suite(
    booz_jax,
    bs_jax,
    iota_target,
    *,
    outer_objective_config=None,
    session: TraceableObjectiveSession | None = None,
):
    """Build profiled pure-JAX closures for the target single-stage objective path."""
    return make_traceable_objective_runtime_bundle(
        booz_jax,
        bs_jax,
        iota_target,
        include_profile_suite=True,
        outer_objective_config=outer_objective_config,
        session=session,
    )["profile_suite"]


# Import the helper layer after this module defines its re-exported names.
# That keeps direct imports of this sibling module from cycling through
# ``surfaceobjectives_jax`` before the compatibility symbols exist.
from .surface_objectives import (
    _TRACEABLE_RUNTIME_OPTION_KEYS,
    _TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS,
    _boozer_residual_J_of_x_inner,
    _canonicalize_traceable_exact_quadrature,
    _curve_curve_point_pair_batches_from_stacks,
    _curve_stacks_from_grouped_spec,
    _curve_surface_point_pair_batches_from_stacks,
    _evaluate_traceable_weighted_single_stage_outer_term,
    _explicit_scalar_pullback_seed,
    _resolved_boozer_solved_runtime_state,
    _runtime_bool,
    _runtime_float64_array,
    _runtime_float64_scalar,
    _runtime_int32_scalar,
    _runtime_zeros_like,
    _split_x_inner_runtime,
    _strict_scalar_grad,
    _take_runtime_scalar,
    _traceable_adjoint_fail_gradient_like,
    _traceable_cache_tree_signature,
    _traceable_contract_tree_signature,
    _traceable_diag_progress,
    _traceable_full_single_stage_outer_objective,
    _traceable_rejected_objective_value,
    _traceable_runtime_deviceify_tree,
    _traceable_runtime_hostify_tree,
    _traceable_single_stage_alm_evaluation,
    _traceable_single_stage_effective_dependency_flags,
    _traceable_single_stage_outer_term_values,
    _traceable_weighted_single_stage_outer_term_values,
)
