"""
Lane-aware JAX Boozer surface solver.

The public/reference lane still permits host-side SciPy minimization via
``optimizer_backend="scipy"``. The target lane uses
``optimizer_backend="ondevice"`` for JAX-resident execution.

This module owns the LS/exact solver routing contract. Only the
``ondevice`` backend is intended to represent the eventual target optimizer
lane, not a claim that the full workflow is already production-complete.

Architecture (per M0 contract §5-§6):
  - Adapter pattern: ``BoozerSurfaceJAX`` inherits ``Optimizable`` and
    mirrors the CPU ``BoozerSurface`` public API.
  - The outer ``Optimizable`` dependency graph and ``need_to_run_code``
    dirty-flag semantics are preserved.
  - The reference lane may still cross the host/device boundary inside the LS
    optimizer loop; that boundary is isolated to the reference-only optimizer
    module.

Builds on M3's composed derivative path:
  - ``_surface_geometry_from_dofs()`` for surface DOFs → geometry (SSOT)
  - ``boozer_residual_scalar()`` for the forward residual
  - ``boozer_residual_vector()`` for the exact Newton residual vector
  - ``boozer_residual_coil_vjp()`` for outer-path coil sensitivities
  - ``jax.grad`` / ``jax.hessian`` / ``jax.jacfwd`` for all derivatives
"""

import hashlib
import inspect
from dataclasses import dataclass, field
from functools import partial
from itertools import count

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import scipy.linalg

from ..backend import (
    get_backend_config,
    get_backend_policy,
    is_parity_mode,
    raise_if_strict_jax_fallback,
    warn_if_jax_fallback,
)
from .._core.jax_host_boundary import (
    host_all_finite as _host_all_finite,
    host_array as _host_numpy,
    host_inf_norm as _host_inf_norm,
    host_scalar as _host_scalar,
    host_tree as _hostify_tree,
)
from ..jax_core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_jax_int32 as _as_jax_int32,
    as_runtime_float64 as _as_runtime_float64,
    concat_jax_float64 as _concat_jax_float64,
    runtime_device_put,
)
from .._core.optimizable import Optimizable
from .surfacerzfourier import SurfaceRZFourier
from .surfacexyzfourier import SurfaceXYZFourier
from .surfacexyztensorfourier import SurfaceXYZTensorFourier


from .surface_fourier_jax import (
    stellsym_scatter_indices,
)
from ._surface_stellsym import (
    compute_stellsym_mask_indices_for_grid,
)
from ._boozersurface_current_guard import (
    guard_none_G_coil_gradient_callback as _guard_none_G_coil_gradient_callback,
    require_fixed_currents_for_none_G as _require_fixed_currents_for_none_G,
)
from ..jax_core.field import (
    _evaluate_grouped_field_group,
    grouped_biot_savart_A_from_inputs,
    grouped_biot_savart_A_from_spec,
    grouped_biot_savart_B_and_dB_from_spec,
    grouped_biot_savart_B_from_inputs,
    grouped_biot_savart_B_from_spec,
    grouped_biot_savart_dA_by_dX_from_spec,
    grouped_coil_currents_from_inputs,
    grouped_coil_currents_from_spec,
    grouped_coil_index_lists_from_spec,
    grouped_coil_set_spec_from_inputs,
    grouped_coil_set_spec_from_source,
    grouped_field_data_from_spec,
    grouped_field_inputs_from_spec,
)
from ..jax_core.biotsavart import (
    biot_savart_A,
    biot_savart_B_and_dB,
    biot_savart_dA_by_dX,
)
from ..jax_core.specs import GroupedCoilSetSpec
from ..jax_core.sharding import place_active_replicated
from .boozer_residual_jax import (
    _split_decision_vector as _split_boozer_decision_vector,
    boozer_residual_scalar_and_grad_cpu_ordered,
    boozer_residual_scalar,
    boozer_residual_vector,
    _surface_geometry_from_dofs,
)
from .label_constraints_jax import (
    area_jax,
    volume_jax,
    toroidal_flux_jax,
    compute_G_from_currents,
)
from . import optimizer_jax as _optimizer_jax
from .optimizer_jax import (
    VALID_LEAST_SQUARES_ALGORITHMS,
    VALID_OPTIMIZER_BACKENDS,
    jax_least_squares_optimistix,
    levenberg_marquardt_minpack_traceable,
    levenberg_marquardt_traceable,
    newton_exact,
    newton_exact_traceable,
    newton_polish,
    newton_polish_traceable,
    reference_least_squares,
    reference_minimize,
    require_target_backend_x64,
    resolve_reference_least_squares_optimizer_method,
    resolve_target_least_squares_optimizer_method,
    target_least_squares,
    target_minimize,
)

_TRACEABLE_SOLVE_STATE_TOKEN_COUNTER = count()


def _new_traceable_solve_state_token() -> int:
    return next(_TRACEABLE_SOLVE_STATE_TOKEN_COUNTER)


SOLVE_QUALITY_LS_FIELDS: tuple[str, ...] = (
    "ls_hessian_symmetry_rel",
    "ls_hessian_action_max_rel",
    "ls_newton_linear_residual_rel",
    "ls_newton_step_abs_diff_rel",
    "ls_factorization_backend",
    "ls_condition_estimate",
)


SOLVE_QUALITY_EXACT_FIELDS: tuple[str, ...] = (
    "exact_jacobian_action_max_rel",
    "exact_newton_linear_residual_rel",
    "exact_refinement_correction_rel",
    "exact_adjoint_solve_residual_rel",
    "exact_factorization_backend",
    "exact_condition_estimate",
)


_BOOZER_LS_SOLVE_QUALITY_RESULT_KEYS = frozenset(SOLVE_QUALITY_LS_FIELDS)
_BOOZER_EXACT_SOLVE_QUALITY_RESULT_KEYS = frozenset(SOLVE_QUALITY_EXACT_FIELDS)


# Per docs/parity_scientific_equivalence_contract_2026-05-09.md §3.2: exact
# Newton solves the linearization through the operator GMRES seam in
# ``simsopt.geo.optimizer_jax`` (``_run_operator_gmres``); dense PLU storage
# is reporting metadata only.
EXACT_FACTORIZATION_BACKEND: str = "operator-gmres"


__all__ = [
    "BoozerSurfaceJAX",
    "EXACT_FACTORIZATION_BACKEND",
    "SOLVE_QUALITY_EXACT_FIELDS",
    "SOLVE_QUALITY_LS_FIELDS",
    "build_boozer_surface_runtime_state",
]


@dataclass(frozen=True)
class _BoozerResultSchema:
    required_keys: frozenset
    forbidden_keys: frozenset = field(default_factory=frozenset)


_BOOZER_SOLVER_RESULT_CORE_KEYS = frozenset(
    {"success", "G", "s", "iota", "weight_inv_modB", "type"}
)
_BOOZER_RUNTIME_RESULT_KEYS = frozenset(
    {"sdofs", "primal_success", "adjoint_linear_solve_available"}
)
_BOOZER_LINEARIZED_RESULT_KEYS = frozenset(
    {
        "linearization_kind",
        "linear_solve_backend",
        "dense_linear_solve_factors_available",
        "linearization_residency",
    }
)
_BOOZER_HESSIAN_REPORTING_RESULT_KEYS = frozenset(
    {
        "hessian_materialized",
        "dense_hessian_shape",
        "dense_hessian_bytes",
        "max_dense_hessian_bytes",
        "dense_newton_steps_materialized",
        "dense_newton_steps_message",
        "newton_iter",
        "final_gradient_norm",
        "final_gradient_inf_norm",
        "iterative_refinement_ran",
        "final_step_iterative_refinement_ran",
        "dense_refinement_ran",
        "final_step_dense_refinement_ran",
        "failure_category",
        "failure_stage",
        "message",
    }
)
_BOOZER_EXACT_REPORTING_RESULT_KEYS = frozenset(
    {
        "jacobian_materialized",
        "dense_jacobian_shape",
        "dense_jacobian_bytes",
        "max_dense_jacobian_bytes",
        "failure_category",
        "failure_stage",
        "message",
    }
)
_BOOZER_TRACEABLE_RESULT_KEYS = frozenset(
    {
        "x",
        "sdofs",
        "iota",
        "G",
        "fun",
        "plu",
        "nit",
        "success",
        "primal_success",
        "adjoint_linear_solve_available",
        "linearization_kind",
        "linear_solve_backend",
        "dense_linear_solve_factors_available",
        "type",
        "weight_inv_modB",
    }
)
# Lowercase ``lu_piv`` is the private traceable companion to the public
# lowercase ``plu`` triple (see
# docs/parity_scientific_equivalence_contract_2026-05-09.md §5.3).
# Traceable consumers may still receive only the legacy ``(P, L, U)``
# triple, so lowercase ``lu_piv`` is neither required nor forbidden.
# Uppercase ``LU_PIV`` is a public result-dict key and remains forbidden
# on the traceable result schema alongside uppercase ``PLU``.
_BOOZER_TRACEABLE_FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "PLU",
        "LU_PIV",
        "s",
        "info",
        "vjp",
        "vjp_groups",
        "solve_generation",
        "mask",
        "lm",
    }
)

_BOOZER_RESULT_SCHEMAS = {
    "lbfgs": _BoozerResultSchema(
        required_keys=_BOOZER_SOLVER_RESULT_CORE_KEYS
        | _BOOZER_RUNTIME_RESULT_KEYS
        | _BOOZER_LS_SOLVE_QUALITY_RESULT_KEYS
        | frozenset(
            {
                "fun",
                "gradient",
                "iter",
                "info",
                "optimizer_method",
            }
        ),
        forbidden_keys=frozenset(
            {
                "residual",
                "jacobian",
                "hessian",
                "PLU",
                "plu",
                "vjp",
                "vjp_groups",
                "linearization_kind",
                "linear_solve_backend",
                "dense_linear_solve_factors_available",
                "mask",
                "lm",
            }
        ),
    ),
    "ls_manual": _BoozerResultSchema(
        required_keys=_BOOZER_SOLVER_RESULT_CORE_KEYS
        | _BOOZER_RUNTIME_RESULT_KEYS
        | _BOOZER_LS_SOLVE_QUALITY_RESULT_KEYS
        | frozenset({"residual", "gradient", "jacobian", "optimizer_method"}),
        forbidden_keys=frozenset(
            {
                "info",
                "fun",
                "hessian",
                "PLU",
                "plu",
                "vjp",
                "vjp_groups",
                "linearization_kind",
                "linear_solve_backend",
                "dense_linear_solve_factors_available",
                "mask",
                "lm",
            }
        ),
    ),
    "ls_lm": _BoozerResultSchema(
        required_keys=_BOOZER_SOLVER_RESULT_CORE_KEYS
        | _BOOZER_RUNTIME_RESULT_KEYS
        | _BOOZER_LS_SOLVE_QUALITY_RESULT_KEYS
        | frozenset({"info", "residual", "gradient", "jacobian", "optimizer_method"}),
        forbidden_keys=frozenset(
            {
                "fun",
                "hessian",
                "PLU",
                "plu",
                "vjp",
                "vjp_groups",
                "linearization_kind",
                "linear_solve_backend",
                "dense_linear_solve_factors_available",
                "mask",
                "lm",
            }
        ),
    ),
    # ``LU_PIV`` is the Phase 2 packed-factor companion to the public
    # ``PLU`` triple (see
    # docs/parity_scientific_equivalence_contract_2026-05-09.md §5.3).
    # It is optional metadata: schema consumers that supply only the
    # legacy ``PLU`` triple without packed factors stay supported, so
    # ``LU_PIV`` is intentionally absent from both required and
    # forbidden sets.
    "newton": _BoozerResultSchema(
        required_keys=_BOOZER_SOLVER_RESULT_CORE_KEYS
        | _BOOZER_RUNTIME_RESULT_KEYS
        | _BOOZER_LINEARIZED_RESULT_KEYS
        | _BOOZER_HESSIAN_REPORTING_RESULT_KEYS
        | _BOOZER_LS_SOLVE_QUALITY_RESULT_KEYS
        | frozenset(
            {
                "residual",
                "jacobian",
                "hessian",
                "iter",
                "PLU",
                "vjp",
                "vjp_groups",
                "optimizer_method",
                "solve_generation",
                "fun",
            }
        ),
        forbidden_keys=frozenset({"plu", "mask", "lm", "info"}),
    ),
    "exact": _BoozerResultSchema(
        required_keys=_BOOZER_SOLVER_RESULT_CORE_KEYS
        | _BOOZER_RUNTIME_RESULT_KEYS
        | _BOOZER_LINEARIZED_RESULT_KEYS
        | _BOOZER_EXACT_REPORTING_RESULT_KEYS
        | _BOOZER_EXACT_SOLVE_QUALITY_RESULT_KEYS
        | frozenset(
            {
                "residual",
                "fun",
                "jacobian",
                "iter",
                "PLU",
                "mask",
                "vjp",
                "vjp_groups",
                "solve_generation",
            }
        ),
        forbidden_keys=frozenset(
            {"plu", "hessian", "gradient", "optimizer_method", "lm", "info"}
        ),
    ),
    "exact_constraints": _BoozerResultSchema(
        required_keys=_BOOZER_SOLVER_RESULT_CORE_KEYS
        | _BOOZER_RUNTIME_RESULT_KEYS
        | _BOOZER_EXACT_SOLVE_QUALITY_RESULT_KEYS
        | frozenset({"residual", "jacobian", "iter", "lm"}),
        forbidden_keys=frozenset(
            {
                "fun",
                "gradient",
                "hessian",
                "PLU",
                "plu",
                "vjp",
                "vjp_groups",
                "linearization_kind",
                "linear_solve_backend",
                "dense_linear_solve_factors_available",
                "mask",
                "optimizer_method",
                "info",
            }
        ),
    ),
    "traceable": _BoozerResultSchema(
        required_keys=_BOOZER_TRACEABLE_RESULT_KEYS,
        forbidden_keys=_BOOZER_TRACEABLE_FORBIDDEN_RESULT_KEYS,
    ),
    "traceable_exact": _BoozerResultSchema(
        required_keys=_BOOZER_TRACEABLE_RESULT_KEYS
        | _BOOZER_EXACT_REPORTING_RESULT_KEYS
        | _BOOZER_EXACT_SOLVE_QUALITY_RESULT_KEYS
        | frozenset({"residual", "jacobian"}),
        forbidden_keys=_BOOZER_TRACEABLE_FORBIDDEN_RESULT_KEYS
        | frozenset({"grad", "hessian", "optimizer_method"}),
    ),
    "traceable_ls": _BoozerResultSchema(
        required_keys=_BOOZER_TRACEABLE_RESULT_KEYS
        | _BOOZER_HESSIAN_REPORTING_RESULT_KEYS
        | _BOOZER_LS_SOLVE_QUALITY_RESULT_KEYS
        | frozenset({"grad", "hessian", "optimizer_method"}),
        forbidden_keys=_BOOZER_TRACEABLE_FORBIDDEN_RESULT_KEYS
        | frozenset({"residual", "jacobian"}),
    ),
}


@dataclass(frozen=True)
class _BoozerPenaltyOptimizerState:
    surface_dofs: jax.Array
    iota: jax.Array


@dataclass(frozen=True)
class _BoozerPenaltyOptimizerStateWithG:
    surface_dofs: jax.Array
    iota: jax.Array
    G: jax.Array


jax.tree_util.register_dataclass(
    _BoozerPenaltyOptimizerState,
    data_fields=["surface_dofs", "iota"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    _BoozerPenaltyOptimizerStateWithG,
    data_fields=["surface_dofs", "iota", "G"],
    meta_fields=[],
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BoozerSurfaceRuntimeState:
    quadpoints_phi: jax.Array
    quadpoints_theta: jax.Array
    scatter_indices: jax.Array | None
    mpol: int = field(metadata={"static": True})
    ntor: int = field(metadata={"static": True})
    nfp: int = field(metadata={"static": True})
    stellsym: bool = field(metadata={"static": True})
    surface_kind: str = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BoozerSolvedRuntimeState:
    """Immutable solved-state summary for pure/runtime Boozer consumers.

    This deliberately carries only the solved state that pure outer-objective
    code needs as its source of truth. Legacy mutable solve artifacts such as
    ``PLU`` or callback hooks remain in ``self.res`` on the compatibility lane.
    """

    sdofs: jax.Array
    iota: jax.Array
    G: jax.Array | None
    weight_inv_modB: bool = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BoozerPenaltyGeometry:
    gamma: jax.Array
    xphi: jax.Array
    xtheta: jax.Array


def _place_active_replicated_geometry(
    geometry: _BoozerPenaltyGeometry,
) -> _BoozerPenaltyGeometry:
    return _BoozerPenaltyGeometry(
        gamma=place_active_replicated(geometry.gamma),
        xphi=place_active_replicated(geometry.xphi),
        xtheta=place_active_replicated(geometry.xtheta),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BoozerForwardLocalFieldTerms:
    B: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BoozerForwardToroidalFluxFieldTerms:
    B: jax.Array
    A: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BoozerLocalFieldTerms:
    B: jax.Array
    dB_dX: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BoozerToroidalFluxFieldTerms:
    B: jax.Array
    dB_dX: jax.Array
    A: jax.Array
    dA_dX: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BoozerPenaltyVectorizedInputs:
    """Boundary inputs to ``boozer_residual_scalar_and_grad_cpu_ordered``.

    The same arrays are produced by both
    ``BoozerSurface._boozer_penalty_vectorized_inputs`` (CPU C++ path) and
    ``_boozer_penalty_value_and_grad_inputs_cpu_ordered`` (JAX path). The
    bit-identity census materializes JAX arrays via ``jax.device_get`` and
    compares the float64 byte representations name-for-name.
    """

    G_value: jax.Array
    iota: jax.Array
    B: jax.Array
    dB_dX: jax.Array
    xphi: jax.Array
    xtheta: jax.Array
    dx_ds: jax.Array
    dxphi_ds: jax.Array
    dxtheta_ds: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BoozerPenaltyParams:
    iota: jax.Array
    G: jax.Array
    targetlabel: jax.Array
    constraint_weight: jax.Array
    label_type: str = field(metadata={"static": True})
    phi_idx: int = field(metadata={"static": True})
    weight_inv_modB: bool = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BoozerLSGroupedVJPSnapshot:
    quadpoints_phi: jax.Array
    quadpoints_theta: jax.Array
    scatter_indices: jax.Array | None
    surface_dofs: jax.Array
    x: jax.Array
    iota: jax.Array
    G: jax.Array
    field_terms: _BoozerLocalFieldTerms | _BoozerToroidalFluxFieldTerms
    coil_set_spec: GroupedCoilSetSpec
    geometry: _BoozerPenaltyGeometry
    label_geometry: _BoozerPenaltyGeometry
    mpol: int = field(metadata={"static": True})
    ntor: int = field(metadata={"static": True})
    nfp: int = field(metadata={"static": True})
    stellsym: bool = field(metadata={"static": True})
    label_type: str = field(metadata={"static": True})
    phi_idx: int = field(metadata={"static": True})
    surface_kind: str = field(metadata={"static": True})
    label_mpol: int = field(metadata={"static": True})
    label_ntor: int = field(metadata={"static": True})
    label_nfp: int = field(metadata={"static": True})
    label_stellsym: bool = field(metadata={"static": True})
    label_surface_kind: str = field(metadata={"static": True})
    label_quadpoints_phi: jax.Array
    label_quadpoints_theta: jax.Array
    label_scatter_indices: jax.Array | None
    constraint_weight: float = field(metadata={"static": True})
    targetlabel: float = field(metadata={"static": True})
    optimize_G: bool = field(metadata={"static": True})
    weight_inv_modB: bool = field(metadata={"static": True})
    coil_indices: tuple[tuple[int, ...], ...] = field(metadata={"static": True})
    solver_generation: int = field(metadata={"static": True})


@dataclass(frozen=True)
class _BoozerAdjointRuntimeState:
    """Immutable adjoint-state summary for wrapper gradient consumers."""

    solved_state: _BoozerSolvedRuntimeState
    linearization_kind: str
    decision_size: int
    dtype: object
    apply_forward: callable
    apply_transpose: callable
    solve_forward: callable
    solve_transpose: callable
    solve_forward_with_status: callable
    solve_transpose_with_status: callable
    stream_group_vjps: callable
    linear_solve_backend: str = "operator"
    dense_linear_solve_factors_available: bool = False
    linear_solve_factors: tuple[jax.Array, jax.Array, jax.Array] | None = None
    linearization_residency: str = "device"

    @property
    def plu(self):
        """Dense-factor compatibility alias for legacy readers."""
        return self.linear_solve_factors


def _surface_geometry_kind(surface) -> str:
    if isinstance(surface, SurfaceRZFourier):
        return "rzfourier"
    if isinstance(surface, SurfaceXYZFourier):
        return "xyzfourier"
    if isinstance(surface, SurfaceXYZTensorFourier):
        return "xyztensorfourier"

    explicit_kind = getattr(surface, "jax_surface_kind", None)
    if explicit_kind in {"generic", "rzfourier", "xyzfourier", "xyztensorfourier"}:
        return explicit_kind
    if explicit_kind is not None:
        raise ValueError(
            f"Unsupported BoozerSurfaceJAX jax_surface_kind {explicit_kind!r}."
        )

    raise TypeError(
        f"Unsupported BoozerSurfaceJAX surface type {type(surface).__name__!r}. "
        "Supported surfaces must be real SurfaceRZFourier, SurfaceXYZFourier, "
        "SurfaceXYZTensorFourier, or expose an explicit jax_surface_kind."
    )


def _is_exact_surface_xyz_tensor_fourier(surface_kind: str) -> bool:
    return surface_kind == "xyztensorfourier"


def build_boozer_surface_runtime_state(surface) -> _BoozerSurfaceRuntimeState:
    """Snapshot the immutable surface metadata required by JAX Boozer solves."""
    scatter_indices = None
    if surface.stellsym:
        geometry_kind = _surface_geometry_kind(surface)
        if geometry_kind in {"generic", "xyztensorfourier"}:
            scatter_indices = _generic_surface_scatter_operator(
                surface.mpol,
                surface.ntor,
            )
        else:
            scatter_indices = _as_jax_int32(
                stellsym_scatter_indices(surface.mpol, surface.ntor)
            )
    return _BoozerSurfaceRuntimeState(
        quadpoints_phi=_as_jax_float64(surface.quadpoints_phi),
        quadpoints_theta=_as_jax_float64(surface.quadpoints_theta),
        scatter_indices=scatter_indices,
        mpol=int(surface.mpol),
        ntor=int(surface.ntor),
        nfp=int(surface.nfp),
        stellsym=bool(surface.stellsym),
        surface_kind=_surface_geometry_kind(surface),
    )


def _surface_dofs_fingerprint_from_dofs(dofs) -> tuple[str, tuple[int, ...], str]:
    array = np.ascontiguousarray(np.asarray(jax.device_get(dofs), dtype=np.float64))
    return (
        str(array.dtype),
        tuple(int(dim) for dim in array.shape),
        hashlib.blake2b(array.tobytes(), digest_size=16).hexdigest(),
    )


def _require_boozer_vjp_callback_signature(callback, *, callback_name: str):
    """Fail fast when a result-dict VJP hook cannot accept the public contract."""
    if callback is None:
        return None
    try:
        inspect.signature(callback).bind(object(), object(), object(), object())
    except TypeError as exc:
        raise TypeError(
            f"BoozerSurfaceJAX result callback {callback_name!r} must accept "
            "(lm, booz_surf, iota, G)."
        ) from exc
    return callback


def _guard_result_callback_with_check(callback, *, callback_name: str, check):
    if callback is None:
        return None

    if callback_name != "vjp_groups":

        def guarded(*args, **kwargs):
            check()
            return callback(*args, **kwargs)

        return guarded

    def guarded(*args, **kwargs):
        def stream():
            check()
            yield from callback(*args, **kwargs)

        return stream()

    return guarded


def _guard_solver_callback_freshness(
    callback,
    *,
    booz_surf,
    solve_generation: int,
    callback_name: str,
):
    """Reject stale result callbacks after the Boozer solve state changes."""

    def raise_if_stale():
        current_generation = getattr(booz_surf, "_solver_generation", None)
        if booz_surf.need_to_run_code or current_generation != solve_generation:
            raise RuntimeError(
                f"BoozerSurfaceJAX result callback {callback_name!r} is stale "
                f"(expected generation {solve_generation}, got {current_generation}). "
                "Re-run boozer_surface.run_code(...) before requesting adjoints."
            )

    return _guard_result_callback_with_check(
        callback,
        callback_name=callback_name,
        check=raise_if_stale,
    )


def _guard_result_callback_surface_dofs(callback, *, booz_surf, callback_name: str):
    def raise_if_drifted():
        booz_surf._raise_if_surface_dofs_drifted(callback_name=callback_name)

    return _guard_result_callback_with_check(
        callback,
        callback_name=callback_name,
        check=raise_if_drifted,
    )


def _advance_solver_generation(booz_surf) -> int:
    solve_generation = booz_surf._solver_generation + 1
    booz_surf._solver_generation = solve_generation
    booz_surf._traceable_solve_state_token = _new_traceable_solve_state_token()
    booz_surf._traceable_runtime_entry_cache = None
    return solve_generation


def _prepare_result_callback(
    callback,
    *,
    booz_surf,
    solve_generation: int,
    callback_name: str,
    G_provided: bool,
    freshness_guard: bool,
):
    callback = _guard_none_G_coil_gradient_callback(
        callback,
        biotsavart=booz_surf.biotsavart,
        component="BoozerSurfaceJAX",
        coil_attrs=("coils",),
        G_provided=G_provided,
    )
    callback = _require_boozer_vjp_callback_signature(
        callback,
        callback_name=callback_name,
    )
    callback = _guard_result_callback_surface_dofs(
        callback,
        booz_surf=booz_surf,
        callback_name=callback_name,
    )
    if freshness_guard:
        callback = _guard_solver_callback_freshness(
            callback,
            booz_surf=booz_surf,
            solve_generation=solve_generation,
            callback_name=callback_name,
        )
    return callback


def _as_boozer_penalty_optimizer_state(x, *, optimize_G):
    if optimize_G:
        if isinstance(x, _BoozerPenaltyOptimizerStateWithG):
            return _BoozerPenaltyOptimizerStateWithG(
                surface_dofs=_as_jax_float64(x.surface_dofs),
                iota=_as_jax_float64(x.iota),
                G=_as_jax_float64(x.G),
            )
    elif isinstance(x, _BoozerPenaltyOptimizerState):
        return _BoozerPenaltyOptimizerState(
            surface_dofs=_as_jax_float64(x.surface_dofs),
            iota=_as_jax_float64(x.iota),
        )

    x_jax = _as_jax_float64(x)
    sdofs, iota, G = _split_decision_vector_jax(x_jax, optimize_G=optimize_G)
    if optimize_G:
        return _BoozerPenaltyOptimizerStateWithG(
            surface_dofs=sdofs,
            iota=iota,
            G=G,
        )
    return _BoozerPenaltyOptimizerState(
        surface_dofs=sdofs,
        iota=iota,
    )


def _traceable_array_signature(array):
    """Return a value-based signature for traced static array-like inputs."""
    if array is None:
        return None
    if isinstance(array, jax.Array):
        array_np = np.asarray(jax.device_get(array))
    else:
        array_np = np.asarray(array)
    return (
        str(array_np.dtype),
        tuple(int(dim) for dim in array_np.shape),
        array_np.tobytes(),
    )


def _runtime_cache_leaf_signature(leaf):
    if isinstance(leaf, (jax.Array, np.ndarray)):
        array = np.asarray(jax.device_get(leaf))
        return (
            "array",
            str(array.dtype),
            tuple(int(dim) for dim in array.shape),
            hashlib.blake2b(array.tobytes(), digest_size=16).hexdigest(),
        )
    if isinstance(leaf, np.generic):
        return ("numpy_scalar", str(leaf.dtype), leaf.item())
    if isinstance(leaf, (str, int, float, bool, type(None))):
        return ("scalar", leaf)
    return ("repr", type(leaf).__qualname__, repr(leaf))


def _runtime_cache_tree_signature(tree):
    leaves, treedef = jax.tree.flatten(tree)
    return (
        "tree",
        repr(treedef),
        tuple(_runtime_cache_leaf_signature(leaf) for leaf in leaves),
    )


def _boozer_penalty_optimizer_state_to_vector(x, *, optimize_G):
    optimizer_state = _as_boozer_penalty_optimizer_state(x, optimize_G=optimize_G)
    if optimize_G:
        return _concat_jax_float64(
            optimizer_state.surface_dofs,
            [optimizer_state.iota, optimizer_state.G],
        )
    return _concat_jax_float64(
        optimizer_state.surface_dofs,
        [optimizer_state.iota],
    )


def _split_decision_vector_jax(x, *, optimize_G):
    return _split_boozer_decision_vector(x, optimize_G=optimize_G)


def _generic_surface_scatter_operator(mpol: int, ntor: int):
    positions = np.asarray(stellsym_scatter_indices(mpol, ntor), dtype=np.int32)
    n_per_coord = int((2 * mpol + 1) * (2 * ntor + 1))
    operator = np.zeros((3 * n_per_coord, positions.size), dtype=np.float64)
    operator[positions, np.arange(positions.size)] = 1.0
    return _as_jax_float64(operator)


def _cross_product(left, right):
    return jnp.cross(left, right, axis=-1)


def _cross_product_cpu_ordered(left, right):
    return jnp.stack(
        (
            left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],
            left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2],
            left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0],
        ),
        axis=-1,
    )


def _surface_volume_cpu_ordered(gamma, normal):
    nphi, ntheta = gamma.shape[:2]
    zero = jnp.sum(gamma, dtype=gamma.dtype) - jnp.sum(gamma, dtype=gamma.dtype)
    one_third = _as_runtime_float64(1.0 / 3.0, reference=gamma)

    def point_body(flat_index, volume):
        i = flat_index // ntheta
        j = flat_index - i * ntheta
        point = (
            gamma[i, j, 0] * normal[i, j, 0]
            + gamma[i, j, 1] * normal[i, j, 1]
            + gamma[i, j, 2] * normal[i, j, 2]
        )
        return volume + one_third * point

    volume = jax.lax.fori_loop(0, nphi * ntheta, point_body, zero)
    return volume / _as_runtime_float64(nphi * ntheta, reference=gamma)


def _select_axis0(array, index: int):
    selector = np.zeros(int(array.shape[0]), dtype=np.float64)
    selector[int(index)] = 1.0
    return jnp.tensordot(
        _as_runtime_float64(selector, reference=array),
        jnp.asarray(array),
        axes=((0,), (0,)),
    )


def _surface_sample_z(gamma):
    sample = _select_axis0(_select_axis0(gamma, 0), 0)
    return _select_axis0(sample, 2)


def _replace_group_coil_array(coil_arrays, group_index, group_array):
    grouped_arrays = list(coil_arrays)
    grouped_arrays[group_index] = group_array
    return grouped_arrays


def _yield_group_vjps(lm, group_runners, coil_arrays, coil_indices):
    for group_runner, group_array, group_index_list in zip(
        group_runners,
        coil_arrays,
        coil_indices,
    ):
        _, vjp_fn = jax.vjp(group_runner, group_array)
        yield vjp_fn(lm)[0], group_index_list


def _extract_grouped_coil_set_spec(biotsavart):
    """Return the immutable grouped-coil spec for a biotsavart-like object.

    ``BoozerSurfaceJAX`` now requires its field adapter to expose explicit
    immutable grouped-coil state through ``coil_set_spec()``. Hidden grouped
    extractors and raw ``_coils`` snapshots are no longer accepted here.
    """
    coil_set_spec = getattr(biotsavart, "coil_set_spec", None)
    if coil_set_spec is None or not callable(coil_set_spec):
        raise AttributeError(
            "BoozerSurfaceJAX requires a biotsavart object that provides "
            "coil_set_spec() for explicit immutable grouped-coil state. "
            "Hidden _extract_coil_data_grouped() and _coils compatibility seams "
            "are no longer supported."
        )
    return grouped_coil_set_spec_from_source(coil_set_spec())


def _grouped_coil_currents(*, coil_arrays=None, coil_set_spec=None):
    if coil_set_spec is not None:
        return grouped_coil_currents_from_spec(coil_set_spec)
    return grouped_coil_currents_from_inputs(coil_arrays)


def _resolved_coil_set_spec(default_spec, *, coil_arrays=None, coil_set_spec=None):
    if coil_set_spec is not None:
        return coil_set_spec
    if coil_arrays is not None:
        return grouped_coil_set_spec_from_inputs(coil_arrays)
    return default_spec


def _grouped_biot_savart_B_points(points, *, coil_arrays=None, coil_set_spec=None):
    if coil_set_spec is not None:
        return grouped_biot_savart_B_from_spec(points, coil_set_spec)
    return grouped_biot_savart_B_from_inputs(points, coil_arrays)


def _grouped_biot_savart_A_points(points, *, coil_arrays=None, coil_set_spec=None):
    if coil_set_spec is not None:
        return grouped_biot_savart_A_from_spec(points, coil_set_spec)
    return grouped_biot_savart_A_from_inputs(points, coil_arrays)


def _geometry_from_x(
    x,
    *,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    optimize_G,
):
    optimizer_state = _as_boozer_penalty_optimizer_state(x, optimize_G=optimize_G)
    geometry = _geometry_from_surface_dofs(
        optimizer_state.surface_dofs,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
    )
    return geometry, optimizer_state


def _geometry_from_surface_dofs(
    surface_dofs,
    *,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    parity_policy: str = "production",
):
    if parity_policy == "cpu_ordered" and surface_kind in (
        "generic",
        "xyztensorfourier",
    ):
        from .surface_fourier_jax_cpu_ordered import (  # noqa: PLC0415
            surface_gamma_from_dofs_cpu_ordered,
            surface_gammadash1_from_dofs_cpu_ordered,
            surface_gammadash2_from_dofs_cpu_ordered,
        )

        gamma = surface_gamma_from_dofs_cpu_ordered(
            surface_dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices=scatter_indices,
        )
        xphi = surface_gammadash1_from_dofs_cpu_ordered(
            surface_dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices=scatter_indices,
        )
        xtheta = surface_gammadash2_from_dofs_cpu_ordered(
            surface_dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices=scatter_indices,
        )
        return _place_active_replicated_geometry(
            _BoozerPenaltyGeometry(gamma=gamma, xphi=xphi, xtheta=xtheta)
        )
    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        surface_dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        surface_kind=surface_kind,
    )
    return _place_active_replicated_geometry(
        _BoozerPenaltyGeometry(gamma=gamma, xphi=xphi, xtheta=xtheta)
    )


def _penalty_params(
    *,
    iota,
    G_value,
    targetlabel,
    constraint_weight,
    label_type,
    phi_idx,
    weight_inv_modB,
):
    return _BoozerPenaltyParams(
        iota=_as_jax_float64(iota),
        G=_as_jax_float64(G_value),
        targetlabel=_as_jax_float64(targetlabel),
        constraint_weight=_as_jax_float64(constraint_weight),
        label_type=label_type,
        phi_idx=phi_idx,
        weight_inv_modB=weight_inv_modB,
    )


def _field_shape_from_geometry(geometry: _BoozerPenaltyGeometry):
    return tuple(int(dim) for dim in geometry.gamma.shape[:2])


def _field_points_from_geometry(geometry: _BoozerPenaltyGeometry):
    return geometry.gamma.reshape(-1, 3)


def _reshape_vector_field(field_values, field_shape):
    return field_values.reshape(field_shape + (3,))


def _reshape_field_jacobian(field_values, field_shape):
    return field_values.reshape(field_shape + (3, 3))


def _forward_field_terms_for_local_label(
    points,
    field_shape,
    *,
    label_points,
    label_field_shape,
    coil_arrays=None,
    coil_set_spec=None,
):
    del label_points, label_field_shape
    return _BoozerForwardLocalFieldTerms(
        B=_reshape_vector_field(
            _grouped_biot_savart_B_points(
                points,
                coil_arrays=coil_arrays,
                coil_set_spec=coil_set_spec,
            ),
            field_shape,
        )
    )


def _forward_field_terms_for_toroidal_flux(
    points,
    field_shape,
    *,
    label_points,
    label_field_shape,
    coil_arrays=None,
    coil_set_spec=None,
):
    return _BoozerForwardToroidalFluxFieldTerms(
        B=_reshape_vector_field(
            _grouped_biot_savart_B_points(
                points,
                coil_arrays=coil_arrays,
                coil_set_spec=coil_set_spec,
            ),
            field_shape,
        ),
        A=_reshape_vector_field(
            _grouped_biot_savart_A_points(
                label_points,
                coil_arrays=coil_arrays,
                coil_set_spec=coil_set_spec,
            ),
            label_field_shape,
        ),
    )


def _field_terms_for_local_label(
    points,
    field_shape,
    *,
    label_points,
    label_field_shape,
    coil_set_spec=None,
    group=None,
    parity_policy: str = "production",
):
    del label_points, label_field_shape
    if parity_policy == "cpu_ordered":
        from ..jax_core.biotsavart_cpu_ordered import (  # noqa: PLC0415
            biot_savart_B_and_dB_cpu_ordered,
        )
        from ..jax_core.field import grouped_field_inputs_from_spec  # noqa: PLC0415

        if group is not None:
            B, dB_dX = biot_savart_B_and_dB_cpu_ordered(points, *group)
        else:
            inputs = grouped_field_inputs_from_spec(coil_set_spec)
            B = None
            dB_dX = None
            for gammas, gammadashs, currents in inputs:
                B_g, dB_g = biot_savart_B_and_dB_cpu_ordered(
                    points, gammas, gammadashs, currents
                )
                B = B_g if B is None else B + B_g
                dB_dX = dB_g if dB_dX is None else dB_dX + dB_g
    elif group is None:
        B, dB_dX = grouped_biot_savart_B_and_dB_from_spec(points, coil_set_spec)
    else:
        (B, dB_dX), _config = _evaluate_grouped_field_group(
            points,
            *group,
            biot_savart_B_and_dB,
        )
    return _BoozerLocalFieldTerms(
        B=_reshape_vector_field(B, field_shape),
        dB_dX=_reshape_field_jacobian(dB_dX, field_shape),
    )


def _field_terms_for_toroidal_flux(
    points,
    field_shape,
    *,
    label_points,
    label_field_shape,
    coil_set_spec=None,
    group=None,
):
    local_terms = _field_terms_for_local_label(
        points,
        field_shape,
        label_points=label_points,
        label_field_shape=label_field_shape,
        coil_set_spec=coil_set_spec,
        group=group,
    )
    if group is None:
        A = grouped_biot_savart_A_from_spec(label_points, coil_set_spec)
        dA_dX = grouped_biot_savart_dA_by_dX_from_spec(label_points, coil_set_spec)
    else:
        A, _config = _evaluate_grouped_field_group(
            label_points,
            *group,
            biot_savart_A,
        )
        dA_dX, _config = _evaluate_grouped_field_group(
            label_points,
            *group,
            biot_savart_dA_by_dX,
        )
    return _BoozerToroidalFluxFieldTerms(
        B=local_terms.B,
        dB_dX=local_terms.dB_dX,
        A=_reshape_vector_field(A, label_field_shape),
        dA_dX=_reshape_field_jacobian(dA_dX, label_field_shape),
    )


def _surface_geometry_and_derivatives_from_dofs(
    surface_dofs,
    *,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    parity_policy: str = "production",
):
    """Evaluate geometry and its DOF Jacobian.

    Args:
        parity_policy:
            * ``"production"`` (default) — use the matmul/einsum hot path and
              ``jax.jacfwd`` for the coefficient Jacobians. Same behavior the
              JAX backend has shipped since M3.
            * ``"cpu_ordered"`` — route through
              :mod:`simsopt.geo.surface_fourier_jax_cpu_ordered`. Mirrors the
              C++ ``surfacexyztensorfourier.h`` accumulation order; used by
              the strict bit-identity census ladder
              (``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md``
              Phase 2). Today this kernel set is implemented for
              ``surface_kind in {"generic", "xyztensorfourier"}``; other
              surface kinds fall back to ``"production"`` and the census will
              keep flagging the residual drift.
    """
    if parity_policy == "cpu_ordered" and surface_kind in (
        "generic",
        "xyztensorfourier",
    ):
        return _surface_geometry_and_derivatives_cpu_ordered(
            surface_dofs,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            scatter_indices=scatter_indices,
        )

    def geometry_arrays(sdofs):
        return _surface_geometry_from_dofs(
            sdofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            surface_kind=surface_kind,
        )

    gamma, xphi, xtheta = geometry_arrays(surface_dofs)
    dgamma, dxphi, dxtheta = jax.jacfwd(geometry_arrays)(surface_dofs)
    return (
        _place_active_replicated_geometry(
            _BoozerPenaltyGeometry(gamma=gamma, xphi=xphi, xtheta=xtheta)
        ),
        _place_active_replicated_geometry(
            _BoozerPenaltyGeometry(gamma=dgamma, xphi=dxphi, xtheta=dxtheta)
        ),
    )


def _surface_geometry_and_derivatives_cpu_ordered(
    surface_dofs,
    *,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
):
    """CPU-ordered geometry + analytic-Jacobian path for parity census.

    Geometry values come from the operator-for-operator C++ twins in
    :mod:`simsopt.geo.surface_fourier_jax_cpu_ordered`; the coefficient
    Jacobians come from the analytic ``dgamma_by_dcoeff`` family rather than
    ``jax.jacfwd`` (per the plan §8 risk register: ``jacfwd`` over the parity
    twins may still pick a derivative arithmetic order that differs from the
    C++ analytic kernels).
    """
    from .surface_fourier_jax_cpu_ordered import (  # noqa: PLC0415 - parity-only
        surface_gamma_from_dofs_cpu_ordered,
        surface_gammadash1_from_dofs_cpu_ordered,
        surface_gammadash2_from_dofs_cpu_ordered,
        dgamma_by_dcoeff_cpu_ordered,
        dgammadash1_by_dcoeff_cpu_ordered,
        dgammadash2_by_dcoeff_cpu_ordered,
    )

    gamma = surface_gamma_from_dofs_cpu_ordered(
        surface_dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices=scatter_indices,
    )
    xphi = surface_gammadash1_from_dofs_cpu_ordered(
        surface_dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices=scatter_indices,
    )
    xtheta = surface_gammadash2_from_dofs_cpu_ordered(
        surface_dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices=scatter_indices,
    )
    dgamma = dgamma_by_dcoeff_cpu_ordered(
        quadpoints_phi,
        quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
    )
    dxphi = dgammadash1_by_dcoeff_cpu_ordered(
        quadpoints_phi,
        quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
    )
    dxtheta = dgammadash2_by_dcoeff_cpu_ordered(
        quadpoints_phi,
        quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
    )
    return (
        _place_active_replicated_geometry(
            _BoozerPenaltyGeometry(gamma=gamma, xphi=xphi, xtheta=xtheta)
        ),
        _place_active_replicated_geometry(
            _BoozerPenaltyGeometry(gamma=dgamma, xphi=dxphi, xtheta=dxtheta)
        ),
    )


def _select_forward_field_terms_builder(label_type):
    if label_type == "toroidal_flux":
        return _forward_field_terms_for_toroidal_flux
    return _forward_field_terms_for_local_label


def _select_structured_field_terms_builder(label_type):
    if label_type == "toroidal_flux":
        return _field_terms_for_toroidal_flux
    return _field_terms_for_local_label


def _label_from_geometry_and_field_terms(
    label_geometry: _BoozerPenaltyGeometry,
    field_terms,
    params: _BoozerPenaltyParams,
):
    normal = _cross_product(label_geometry.xphi, label_geometry.xtheta)
    if params.label_type == "volume":
        return volume_jax(label_geometry.gamma, normal)
    if params.label_type == "area":
        return area_jax(normal)
    ntheta = label_geometry.gamma.shape[1]
    return toroidal_flux_jax(
        _select_axis0(field_terms.A, params.phi_idx),
        _select_axis0(label_geometry.xtheta, params.phi_idx),
        ntheta,
    )


def _label_value_from_surface_dofs(
    surface_dofs,
    *,
    coil_arrays=None,
    coil_set_spec=None,
    label_quadpoints_phi,
    label_quadpoints_theta,
    label_mpol,
    label_ntor,
    label_nfp,
    label_stellsym,
    label_scatter_indices,
    label_surface_kind,
    label_type,
    phi_idx,
    parity_policy: str = "production",
):
    label_geometry = _geometry_from_surface_dofs(
        surface_dofs,
        quadpoints_phi=label_quadpoints_phi,
        quadpoints_theta=label_quadpoints_theta,
        mpol=label_mpol,
        ntor=label_ntor,
        nfp=label_nfp,
        stellsym=label_stellsym,
        scatter_indices=label_scatter_indices,
        surface_kind=label_surface_kind,
        parity_policy=parity_policy,
    )
    return _compute_label(
        label_type,
        label_geometry,
        phi_idx,
        _field_points_from_geometry(label_geometry),
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
        parity_policy=parity_policy,
    )


def _boozer_penalty_value_and_grad_inputs_cpu_ordered(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind,
    optimize_G,
    parity_policy: str = "production",
):
    """Materialize the inputs consumed by
    :func:`boozer_residual_scalar_and_grad_cpu_ordered`.

    The same factoring backs both the production
    ``_boozer_penalty_value_and_grad_cpu_ordered`` and the bit-identity
    census reproducer; see
    ``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`` Phase 1.

    The returned ``_BoozerPenaltyVectorizedInputs`` is a JAX pytree, so this
    helper is safe to call inside a traced computation. The census layer
    materializes the arrays with ``jax.device_get`` after the trace.

    Args:
        parity_policy: ``"production"`` (default) routes through the matmul /
            ``jax.jacfwd`` hot path; ``"cpu_ordered"`` selects the C++-ordered
            surface kernels in ``surface_fourier_jax_cpu_ordered``. Phase 2
            of the bit-identity plan exercises ``"cpu_ordered"``.

    Returns:
        Tuple of (optimizer_state, geometry, geometry_derivative, inputs).

        ``inputs`` is the boundary record fed to
        ``boozer_residual_scalar_and_grad_cpu_ordered``;
        ``geometry`` and ``geometry_derivative`` are returned alongside
        because the production caller still needs ``geometry.gamma``,
        ``geometry_derivative.gamma`` for the rz-axis penalty term.
    """
    optimizer_state = _as_boozer_penalty_optimizer_state(x, optimize_G=optimize_G)
    geometry, geometry_derivative = _surface_geometry_and_derivatives_from_dofs(
        optimizer_state.surface_dofs,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
        parity_policy=parity_policy,
    )
    G_value = (
        optimizer_state.G
        if optimize_G
        else compute_G_from_currents(
            _grouped_coil_currents(coil_arrays=coil_arrays, coil_set_spec=coil_set_spec)
        )
    )
    field_terms = _field_terms_for_local_label(
        _field_points_from_geometry(geometry),
        _field_shape_from_geometry(geometry),
        label_points=None,
        label_field_shape=None,
        coil_set_spec=coil_set_spec,
        parity_policy=parity_policy,
    )
    inputs = _BoozerPenaltyVectorizedInputs(
        G_value=G_value,
        iota=optimizer_state.iota,
        B=field_terms.B,
        dB_dX=field_terms.dB_dX,
        xphi=geometry.xphi,
        xtheta=geometry.xtheta,
        dx_ds=geometry_derivative.gamma,
        dxphi_ds=geometry_derivative.xphi,
        dxtheta_ds=geometry_derivative.xtheta,
    )
    return optimizer_state, geometry, geometry_derivative, inputs


def _boozer_penalty_value_and_grad_cpu_ordered(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
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
    targetlabel,
    constraint_weight,
    label_type,
    phi_idx,
    optimize_G,
    weight_inv_modB,
    parity_policy: str = "production",
):
    optimizer_state, geometry, geometry_derivative, inputs = (
        _boozer_penalty_value_and_grad_inputs_cpu_ordered(
            x,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            scatter_indices=scatter_indices,
            surface_kind=surface_kind,
            optimize_G=optimize_G,
            parity_policy=parity_policy,
        )
    )
    value, gradient = boozer_residual_scalar_and_grad_cpu_ordered(
        inputs.G_value,
        inputs.iota,
        inputs.B,
        inputs.dB_dX,
        inputs.xphi,
        inputs.xtheta,
        inputs.dx_ds,
        inputs.dxphi_ds,
        inputs.dxtheta_ds,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )

    label_value, label_gradient = jax.value_and_grad(_label_value_from_surface_dofs)(
        optimizer_state.surface_dofs,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
        label_quadpoints_phi=label_quadpoints_phi,
        label_quadpoints_theta=label_quadpoints_theta,
        label_mpol=label_mpol,
        label_ntor=label_ntor,
        label_nfp=label_nfp,
        label_stellsym=label_stellsym,
        label_scatter_indices=label_scatter_indices,
        label_surface_kind=label_surface_kind,
        label_type=label_type,
        phi_idx=phi_idx,
        parity_policy=parity_policy,
    )
    constraint_weight = _as_jax_float64(constraint_weight)
    weight_sqrt = jnp.sqrt(constraint_weight)
    rl = weight_sqrt * (label_value - _as_jax_float64(targetlabel))
    rz = weight_sqrt * _surface_sample_z(geometry.gamma)
    value = value + _as_jax_float64(0.5) * rl * rl + _as_jax_float64(0.5) * rz * rz

    surface_size = optimizer_state.surface_dofs.shape[0]
    drl = weight_sqrt * label_gradient
    drz = weight_sqrt * geometry_derivative.gamma[0, 0, 2, :]
    surface_gradient = gradient[:surface_size] + rl * drl + rz * drz
    return value, _concat_jax_float64(surface_gradient, gradient[surface_size:])


def _penalty_from_geometry_and_field_terms(
    geometry: _BoozerPenaltyGeometry,
    label_geometry: _BoozerPenaltyGeometry,
    field_terms,
    params: _BoozerPenaltyParams,
    *,
    boozer_reduction_mode="default",
):
    J_boozer = boozer_residual_scalar(
        params.G,
        params.iota,
        field_terms.B,
        geometry.xphi,
        geometry.xtheta,
        weight_inv_modB=params.weight_inv_modB,
        reduction_mode=boozer_reduction_mode,
    )
    label_val = _label_from_geometry_and_field_terms(
        label_geometry,
        field_terms,
        params,
    )
    gamma_axis_z = _surface_sample_z(geometry.gamma)
    half = _as_jax_float64(0.5)
    label_delta = label_val - params.targetlabel
    J_label = half * params.constraint_weight * label_delta * label_delta
    J_z = half * params.constraint_weight * gamma_axis_z * gamma_axis_z
    return J_boozer + J_label + J_z


def _compute_label(
    label_type,
    label_geometry: _BoozerPenaltyGeometry,
    phi_idx,
    label_points,
    coil_arrays=None,
    coil_set_spec=None,
    parity_policy: str = "production",
):
    """Compute the label value (volume, area, or toroidal flux).

    Shared by penalty objective, exact residual, and residual vector.
    """
    if label_type == "volume":
        normal = (
            _cross_product_cpu_ordered(label_geometry.xphi, label_geometry.xtheta)
            if parity_policy == "cpu_ordered"
            else _cross_product(label_geometry.xphi, label_geometry.xtheta)
        )
        if parity_policy == "cpu_ordered":
            return _surface_volume_cpu_ordered(label_geometry.gamma, normal)
        return volume_jax(label_geometry.gamma, normal)
    normal = _cross_product(label_geometry.xphi, label_geometry.xtheta)
    if label_type == "area":
        return area_jax(normal)
    ntheta = label_geometry.gamma.shape[1]
    A = _grouped_biot_savart_A_points(
        label_points,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    A = A.reshape(label_geometry.gamma.shape)
    return toroidal_flux_jax(
        _select_axis0(A, phi_idx),
        _select_axis0(label_geometry.xtheta, phi_idx),
        ntheta,
    )


def _compute_label_and_axis_z(
    *,
    geometry: _BoozerPenaltyGeometry,
    label_geometry: _BoozerPenaltyGeometry,
    label_points,
    label_type,
    phi_idx,
    coil_arrays=None,
    coil_set_spec=None,
):
    label_value = _compute_label(
        label_type,
        label_geometry,
        phi_idx,
        label_points,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    return label_value, _surface_sample_z(geometry.gamma)


def _boozer_penalty_objective(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
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
    targetlabel,
    constraint_weight,
    label_type,
    phi_idx,
    optimize_G,
    weight_inv_modB,
    boozer_reduction_mode="default",
):
    """Scalarized penalty objective for the BoozerLS inner solve.

    Extends M3's ``boozer_penalty_composed`` with label and z-constraints.

    Pure function: ``x → scalar``.  JAX autodiff gives gradient and
    Hessian for free.

    The optimizer state may be either the historical flat decision vector
    ``[surface_dofs, iota]`` / ``[surface_dofs, iota, G]`` or the structured
    Boozer penalty optimizer pytree that carries the same fields explicitly.
    """
    geometry, optimizer_state = _geometry_from_x(
        x,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
        optimize_G=optimize_G,
    )
    label_geometry = _geometry_from_surface_dofs(
        optimizer_state.surface_dofs,
        quadpoints_phi=label_quadpoints_phi,
        quadpoints_theta=label_quadpoints_theta,
        mpol=label_mpol,
        ntor=label_ntor,
        nfp=label_nfp,
        stellsym=label_stellsym,
        scatter_indices=label_scatter_indices,
        surface_kind=label_surface_kind,
    )
    if optimize_G:
        G = optimizer_state.G
    else:
        G = compute_G_from_currents(
            _grouped_coil_currents(
                coil_arrays=coil_arrays,
                coil_set_spec=coil_set_spec,
            )
        )
    params = _penalty_params(
        iota=optimizer_state.iota,
        G_value=G,
        targetlabel=targetlabel,
        constraint_weight=constraint_weight,
        label_type=label_type,
        phi_idx=phi_idx,
        weight_inv_modB=weight_inv_modB,
    )
    points = _field_points_from_geometry(geometry)
    label_points = _field_points_from_geometry(label_geometry)
    field_terms = _select_forward_field_terms_builder(label_type)(
        points,
        _field_shape_from_geometry(geometry),
        label_points=label_points,
        label_field_shape=_field_shape_from_geometry(label_geometry),
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    return _penalty_from_geometry_and_field_terms(
        geometry,
        label_geometry,
        field_terms,
        params,
        boozer_reduction_mode=boozer_reduction_mode,
    )


def _boozer_penalty_residual_vector(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
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
    targetlabel,
    constraint_weight,
    label_type,
    phi_idx,
    optimize_G,
    weight_inv_modB,
):
    optimizer_state = _as_boozer_penalty_optimizer_state(x, optimize_G=optimize_G)
    G_value = (
        optimizer_state.G
        if optimize_G
        else compute_G_from_currents(
            _grouped_coil_currents(coil_arrays=coil_arrays, coil_set_spec=coil_set_spec)
        )
    )
    geometry = _geometry_from_surface_dofs(
        optimizer_state.surface_dofs,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
    )
    gamma = geometry.gamma
    xphi = geometry.xphi
    xtheta = geometry.xtheta
    label_geometry = _geometry_from_surface_dofs(
        optimizer_state.surface_dofs,
        quadpoints_phi=label_quadpoints_phi,
        quadpoints_theta=label_quadpoints_theta,
        mpol=label_mpol,
        ntor=label_ntor,
        nfp=label_nfp,
        stellsym=label_stellsym,
        scatter_indices=label_scatter_indices,
        surface_kind=label_surface_kind,
    )
    nphi, ntheta = int(gamma.shape[0]), int(gamma.shape[1])
    points = gamma.reshape(-1, 3)
    B = _grouped_biot_savart_B_points(
        points,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    ).reshape(nphi, ntheta, 3)

    r_boozer_raw = boozer_residual_vector(
        G_value,
        optimizer_state.iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=weight_inv_modB,
    )
    num_res = _as_jax_float64(3 * nphi * ntheta)
    r_boozer = r_boozer_raw / jnp.sqrt(num_res)

    constraint_weight = constraint_weight if constraint_weight is not None else 1.0
    constraint_weight = _as_jax_float64(constraint_weight)
    label_value, gamma_axis_z = _compute_label_and_axis_z(
        geometry=geometry,
        label_geometry=label_geometry,
        label_points=_field_points_from_geometry(label_geometry),
        label_type=label_type,
        phi_idx=phi_idx,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    weight_sqrt = jnp.sqrt(constraint_weight)
    rl = weight_sqrt * (label_value - _as_jax_float64(targetlabel))
    rz = weight_sqrt * gamma_axis_z

    return _concat_jax_float64(r_boozer, [rl, rz])


def _boozer_exact_residual(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
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
    targetlabel,
    label_type,
    phi_idx,
    mask_indices,
    stellsym_surface,
    weight_inv_modB,
):
    """Route to the stellsym-specialized exact residual implementation.

    ``stellsym_surface`` changes the residual length because the axis
    constraint is only present on the non-stellsym branch. Callers must bind
    that flag at closure-construction time so each compiled trace sees one
    fixed output shape.
    """
    residual_fn = _select_exact_residual_fn(stellsym_surface)
    return residual_fn(
        x,
        coil_arrays=coil_arrays,
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
        targetlabel=targetlabel,
        label_type=label_type,
        phi_idx=phi_idx,
        mask_indices=mask_indices,
        weight_inv_modB=weight_inv_modB,
    )


def _select_exact_residual_fn(stellsym_surface):
    """Select the exact-residual implementation for a fixed surface symmetry.

    The selected callable becomes part of the surrounding compiled closure, so
    ``stellsym_surface`` is a compile-time specialization choice rather than a
    dynamic traced branch.
    """
    if stellsym_surface:
        return _boozer_exact_residual_stellsym
    return _boozer_exact_residual_nonstellsym


def _boozer_exact_residual_impl(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
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
    targetlabel,
    label_type,
    phi_idx,
    mask_indices,
    weight_inv_modB,
    include_axis_constraint,
):
    """Residual vector for the BoozerExact Newton system.

    Extends M3's ``boozer_residual_vector`` with masking and constraint
    equations (label, z-coordinate).

    Returns: (n_eq,) residual vector where ``r(x) = 0`` at the solution.
    The decision vector is always ``x = [surface_dofs, iota, G]``.
    """
    sdofs, iota, G = _split_decision_vector_jax(x, optimize_G=True)

    geometry = _geometry_from_surface_dofs(
        sdofs,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        surface_kind=surface_kind,
    )
    gamma = geometry.gamma
    xphi = geometry.xphi
    xtheta = geometry.xtheta
    label_geometry = _geometry_from_surface_dofs(
        sdofs,
        quadpoints_phi=label_quadpoints_phi,
        quadpoints_theta=label_quadpoints_theta,
        mpol=label_mpol,
        ntor=label_ntor,
        nfp=label_nfp,
        stellsym=label_stellsym,
        scatter_indices=label_scatter_indices,
        surface_kind=label_surface_kind,
    )
    nphi, ntheta = gamma.shape[:2]

    points = gamma.reshape(-1, 3)
    B = _grouped_biot_savart_B_points(
        points,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    B = B.reshape(nphi, ntheta, 3)

    r_flat = boozer_residual_vector(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=weight_inv_modB,
    )
    r_masked = r_flat[mask_indices]

    label_val, gamma_axis_z = _compute_label_and_axis_z(
        geometry=geometry,
        label_geometry=label_geometry,
        label_points=_field_points_from_geometry(label_geometry),
        label_type=label_type,
        phi_idx=phi_idx,
        coil_arrays=coil_arrays,
        coil_set_spec=coil_set_spec,
    )
    r_label = label_val - _as_jax_float64(targetlabel)

    if include_axis_constraint:
        residual_tail = _as_jax_float64([r_label, gamma_axis_z])
    else:
        residual_tail = _as_jax_float64([r_label])
    return _concat_jax_float64(r_masked, residual_tail)


def _boozer_exact_residual_stellsym(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
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
    targetlabel,
    label_type,
    phi_idx,
    mask_indices,
    weight_inv_modB,
):
    return _boozer_exact_residual_impl(
        x,
        coil_arrays=coil_arrays,
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
        targetlabel=targetlabel,
        label_type=label_type,
        phi_idx=phi_idx,
        mask_indices=mask_indices,
        weight_inv_modB=weight_inv_modB,
        include_axis_constraint=False,
    )


def _boozer_exact_residual_nonstellsym(
    x,
    *,
    coil_arrays=None,
    coil_set_spec=None,
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
    targetlabel,
    label_type,
    phi_idx,
    mask_indices,
    weight_inv_modB,
):
    return _boozer_exact_residual_impl(
        x,
        coil_arrays=coil_arrays,
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
        targetlabel=targetlabel,
        label_type=label_type,
        phi_idx=phi_idx,
        mask_indices=mask_indices,
        weight_inv_modB=weight_inv_modB,
        include_axis_constraint=True,
    )


def _boozer_exact_coil_vjp(lm, booz_surf, iota, G):
    """JAX VJP for the exact path.

    Replaces CPU ``boozer_surface_dexactresidual_dcoils_dcurrents_vjp``.

    Differentiates the FULL exact residual vector (Boozer + label + z)
    w.r.t. coil geometry and currents via ``jax.vjp``.  This correctly
    includes the label derivative term that the CPU code adds explicitly.

    Args:
        lm: (n_eq,) adjoint vector from the outer implicit-function solve.
        booz_surf: ``BoozerSurfaceJAX`` instance.
        iota: rotational transform at the solution.
        G: Boozer G at the solution.

    Returns:
        (d_coil_arrays,), coil_indices — grouped cotangents and index list.
        ``d_coil_arrays`` is a list of ``(d_g, d_gd, d_c)`` tuples matching
        the coil_arrays pytree structure.
    """
    sdofs = booz_surf._get_cached_surface_dofs()
    x = _concat_jax_float64(sdofs, [iota, G])
    mask_indices = booz_surf._compute_stellsym_mask_indices()

    coil_arrays = booz_surf._coil_arrays
    coil_indices = booz_surf._coil_index_lists

    residual_fn = _select_exact_residual_fn(booz_surf.stellsym)

    def residual_of_coils(ca):
        return residual_fn(
            x,
            coil_arrays=ca,
            quadpoints_phi=booz_surf.quadpoints_phi,
            quadpoints_theta=booz_surf.quadpoints_theta,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            surface_kind=booz_surf._surface_geometry_kind,
            label_quadpoints_phi=booz_surf.label_quadpoints_phi,
            label_quadpoints_theta=booz_surf.label_quadpoints_theta,
            label_mpol=booz_surf.label_mpol,
            label_ntor=booz_surf.label_ntor,
            label_nfp=booz_surf.label_nfp,
            label_stellsym=booz_surf.label_stellsym,
            label_scatter_indices=booz_surf.label_scatter_indices,
            label_surface_kind=booz_surf._label_surface_geometry_kind,
            targetlabel=booz_surf.targetlabel,
            label_type=booz_surf.label_type,
            phi_idx=booz_surf.phi_idx,
            mask_indices=mask_indices,
            weight_inv_modB=booz_surf.options["weight_inv_modB"],
        )

    _, vjp_fn = jax.vjp(residual_of_coils, coil_arrays)
    (d_coil_arrays,) = vjp_fn(lm)
    return d_coil_arrays, coil_indices


def _boozer_exact_coil_vjp_groups(lm, booz_surf, iota, G):
    """Yield exact-solve coil VJPs one grouped coil block at a time."""
    yield from _build_exact_group_vjp_callback(booz_surf, iota, G)(
        lm,
        booz_surf,
        iota,
        G,
    )


def _build_exact_group_vjp_callback(booz_surf, iota, G):
    """Build stable exact-solve group runners for repeated streaming VJPs."""
    sdofs = booz_surf._get_cached_surface_dofs()
    x = _concat_jax_float64(sdofs, [iota, G])
    mask_indices = booz_surf._compute_stellsym_mask_indices()

    coil_arrays = booz_surf._coil_arrays
    coil_indices = booz_surf._coil_index_lists

    group_runners = tuple(
        _make_exact_group_runner(
            x,
            coil_arrays,
            booz_surf,
            mask_indices,
            group_index,
        )
        for group_index in range(len(coil_arrays))
    )

    def vjp_groups(lm, _booz_surf, _iota, _G):
        yield from _yield_group_vjps(lm, group_runners, coil_arrays, coil_indices)

    return vjp_groups


def _make_exact_group_runner(x, coil_arrays, booz_surf, mask_indices, group_index):
    residual_fn = _select_exact_residual_fn(booz_surf.stellsym)

    def residual_of_group(group_array):
        return residual_fn(
            x,
            coil_arrays=_replace_group_coil_array(
                coil_arrays,
                group_index,
                group_array,
            ),
            quadpoints_phi=booz_surf.quadpoints_phi,
            quadpoints_theta=booz_surf.quadpoints_theta,
            mpol=booz_surf.mpol,
            ntor=booz_surf.ntor,
            nfp=booz_surf.nfp,
            stellsym=booz_surf.stellsym,
            scatter_indices=booz_surf.scatter_indices,
            surface_kind=booz_surf._surface_geometry_kind,
            label_quadpoints_phi=booz_surf.label_quadpoints_phi,
            label_quadpoints_theta=booz_surf.label_quadpoints_theta,
            label_mpol=booz_surf.label_mpol,
            label_ntor=booz_surf.label_ntor,
            label_nfp=booz_surf.label_nfp,
            label_stellsym=booz_surf.label_stellsym,
            label_scatter_indices=booz_surf.label_scatter_indices,
            label_surface_kind=booz_surf._label_surface_geometry_kind,
            targetlabel=booz_surf.targetlabel,
            label_type=booz_surf.label_type,
            phi_idx=booz_surf.phi_idx,
            mask_indices=mask_indices,
            weight_inv_modB=booz_surf.options["weight_inv_modB"],
        )

    return residual_of_group


def _boozer_ls_coil_vjp(lm, booz_surf, iota, G, weight_inv_modB=True):
    """JAX VJP for the LS penalty path.

    Replaces CPU ``boozer_surface_dlsqgrad_dcoils_vjp``.

    Differentiates the penalty objective GRADIENT w.r.t. coil geometry
    and currents.  This captures all terms (Boozer residual + label +
    z-constraint) because the composed objective includes them.

    Args:
        lm: (n,) adjoint vector (same shape as decision vector).
        booz_surf: ``BoozerSurfaceJAX`` instance.
        iota: rotational transform at the solution.
        G: Boozer G at the solution.
        weight_inv_modB: residual weighting flag.

    Returns:
        (d_coil_arrays,), coil_indices — grouped cotangents and index list.
        ``d_coil_arrays`` is a list of ``(d_g, d_gd, d_c)`` tuples matching
        the coil_arrays pytree structure.
    """
    x, optimize_G = _ls_decision_vector(booz_surf, iota, G)

    coil_arrays = booz_surf._coil_arrays
    coil_indices = booz_surf._coil_index_lists

    def directional_objective_of_coils(ca):
        return _ls_penalty_directional_objective(
            x,
            lm,
            ca,
            booz_surf,
            optimize_G,
            weight_inv_modB,
        )

    d_coil_arrays = jax.grad(directional_objective_of_coils)(coil_arrays)
    return d_coil_arrays, coil_indices


def _boozer_ls_coil_vjp_groups(lm, booz_surf, iota, G, weight_inv_modB=True):
    """Yield LS-path coil VJPs one grouped coil block at a time."""
    yield from _build_ls_group_vjp_callback(
        booz_surf,
        iota,
        G,
        solve_generation=getattr(booz_surf, "_solver_generation", 0),
        weight_inv_modB=weight_inv_modB,
    )(
        lm,
        booz_surf,
        iota,
        G,
    )


def _build_ls_group_vjp_callback(
    booz_surf,
    iota,
    G,
    *,
    solve_generation: int,
    weight_inv_modB=True,
):
    """Build stable LS group runners for repeated streaming VJPs."""
    snapshot = _build_ls_grouped_vjp_snapshot(
        booz_surf,
        iota,
        G,
        solve_generation=solve_generation,
        weight_inv_modB=weight_inv_modB,
    )
    coil_arrays = grouped_field_inputs_from_spec(snapshot.coil_set_spec)

    def vjp_groups(lm, _booz_surf, _iota, _G):
        current_generation = getattr(booz_surf, "_solver_generation", None)
        if (
            booz_surf.need_to_run_code
            or current_generation != snapshot.solver_generation
        ):
            raise RuntimeError(
                "BoozerSurfaceJAX LS grouped VJP callback is stale; "
                "re-run boozer_surface.run_code(...) before requesting adjoints."
            )
        bar_field_terms, bar_G = _ls_field_term_cotangents(snapshot, lm)
        field_terms_builder = _select_structured_field_terms_builder(
            snapshot.label_type
        )
        points = _field_points_from_geometry(snapshot.geometry)
        field_shape = _field_shape_from_geometry(snapshot.geometry)
        label_points = _field_points_from_geometry(snapshot.label_geometry)
        label_field_shape = _field_shape_from_geometry(snapshot.label_geometry)
        for group_array, group_index_tuple in zip(coil_arrays, snapshot.coil_indices):

            def field_terms_of_group(group):
                return field_terms_builder(
                    points,
                    field_shape,
                    label_points=label_points,
                    label_field_shape=label_field_shape,
                    group=group,
                )

            _, pullback = jax.vjp(field_terms_of_group, group_array)
            d_group = pullback(bar_field_terms)[0]
            d_group = _add_G_current_cotangent(
                d_group,
                group_array,
                bar_G,
                optimize_G=snapshot.optimize_G,
            )
            yield d_group, list(group_index_tuple)

    return vjp_groups


def _build_ls_grouped_vjp_snapshot(
    booz_surf,
    iota,
    G,
    *,
    solve_generation: int,
    weight_inv_modB=True,
):
    """Freeze every value needed by an LS grouped-VJP callback."""
    x, optimize_G = _ls_decision_vector(booz_surf, iota, G)
    coil_set_spec = booz_surf.coil_set_spec
    geometry, optimizer_state = _geometry_from_x(
        x,
        quadpoints_phi=booz_surf.quadpoints_phi,
        quadpoints_theta=booz_surf.quadpoints_theta,
        mpol=booz_surf.mpol,
        ntor=booz_surf.ntor,
        nfp=booz_surf.nfp,
        stellsym=booz_surf.stellsym,
        scatter_indices=booz_surf.scatter_indices,
        surface_kind=booz_surf._surface_geometry_kind,
        optimize_G=optimize_G,
    )
    coil_currents = _grouped_coil_currents(coil_set_spec=coil_set_spec)
    G_value = (
        optimizer_state.G if optimize_G else compute_G_from_currents(coil_currents)
    )
    label_geometry = _geometry_from_surface_dofs(
        optimizer_state.surface_dofs,
        quadpoints_phi=booz_surf.label_quadpoints_phi,
        quadpoints_theta=booz_surf.label_quadpoints_theta,
        mpol=booz_surf.label_mpol,
        ntor=booz_surf.label_ntor,
        nfp=booz_surf.label_nfp,
        stellsym=booz_surf.label_stellsym,
        scatter_indices=booz_surf.label_scatter_indices,
        surface_kind=booz_surf._label_surface_geometry_kind,
    )
    field_terms = _select_structured_field_terms_builder(booz_surf.label_type)(
        _field_points_from_geometry(geometry),
        _field_shape_from_geometry(geometry),
        label_points=_field_points_from_geometry(label_geometry),
        label_field_shape=_field_shape_from_geometry(label_geometry),
        coil_set_spec=coil_set_spec,
    )
    return _BoozerLSGroupedVJPSnapshot(
        quadpoints_phi=booz_surf.quadpoints_phi,
        quadpoints_theta=booz_surf.quadpoints_theta,
        scatter_indices=booz_surf.scatter_indices,
        surface_dofs=optimizer_state.surface_dofs,
        x=x,
        iota=optimizer_state.iota,
        G=G_value,
        field_terms=field_terms,
        coil_set_spec=coil_set_spec,
        geometry=geometry,
        label_geometry=label_geometry,
        mpol=booz_surf.mpol,
        ntor=booz_surf.ntor,
        nfp=booz_surf.nfp,
        stellsym=booz_surf.stellsym,
        label_type=booz_surf.label_type,
        phi_idx=booz_surf.phi_idx,
        surface_kind=booz_surf._surface_geometry_kind,
        label_mpol=booz_surf.label_mpol,
        label_ntor=booz_surf.label_ntor,
        label_nfp=booz_surf.label_nfp,
        label_stellsym=booz_surf.label_stellsym,
        label_surface_kind=booz_surf._label_surface_geometry_kind,
        label_quadpoints_phi=booz_surf.label_quadpoints_phi,
        label_quadpoints_theta=booz_surf.label_quadpoints_theta,
        label_scatter_indices=booz_surf.label_scatter_indices,
        constraint_weight=booz_surf.constraint_weight,
        targetlabel=booz_surf.targetlabel,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
        coil_indices=tuple(tuple(indices) for indices in booz_surf._coil_index_lists),
        solver_generation=solve_generation,
    )


def _geometry_tangent_from_decision_tangent(
    snapshot: _BoozerLSGroupedVJPSnapshot,
    tangent_state,
):
    def geometry_of_surface_dofs(surface_dofs):
        return _geometry_from_surface_dofs(
            surface_dofs,
            quadpoints_phi=snapshot.quadpoints_phi,
            quadpoints_theta=snapshot.quadpoints_theta,
            mpol=snapshot.mpol,
            ntor=snapshot.ntor,
            nfp=snapshot.nfp,
            stellsym=snapshot.stellsym,
            scatter_indices=snapshot.scatter_indices,
            surface_kind=snapshot.surface_kind,
        )

    _, geometry_tangent = jax.jvp(
        geometry_of_surface_dofs,
        (snapshot.surface_dofs,),
        (tangent_state.surface_dofs,),
    )
    return geometry_tangent


def _label_geometry_tangent_from_decision_tangent(
    snapshot: _BoozerLSGroupedVJPSnapshot,
    tangent_state,
):
    def geometry_of_surface_dofs(surface_dofs):
        return _geometry_from_surface_dofs(
            surface_dofs,
            quadpoints_phi=snapshot.label_quadpoints_phi,
            quadpoints_theta=snapshot.label_quadpoints_theta,
            mpol=snapshot.label_mpol,
            ntor=snapshot.label_ntor,
            nfp=snapshot.label_nfp,
            stellsym=snapshot.label_stellsym,
            scatter_indices=snapshot.label_scatter_indices,
            surface_kind=snapshot.label_surface_kind,
        )

    _, geometry_tangent = jax.jvp(
        geometry_of_surface_dofs,
        (snapshot.surface_dofs,),
        (tangent_state.surface_dofs,),
    )
    return geometry_tangent


def _spatial_field_dot(field_jacobian, geometry_tangent: _BoozerPenaltyGeometry):
    return jnp.einsum("...j,...jl->...l", geometry_tangent.gamma, field_jacobian)


def _forward_field_values_from_structured_terms(field_terms):
    if isinstance(field_terms, _BoozerToroidalFluxFieldTerms):
        return _BoozerForwardToroidalFluxFieldTerms(B=field_terms.B, A=field_terms.A)
    return _BoozerForwardLocalFieldTerms(B=field_terms.B)


def _forward_field_tangent_from_structured_terms(
    field_terms,
    geometry_tangent: _BoozerPenaltyGeometry,
    label_geometry_tangent: _BoozerPenaltyGeometry,
):
    B_dot = _spatial_field_dot(field_terms.dB_dX, geometry_tangent)
    if isinstance(field_terms, _BoozerToroidalFluxFieldTerms):
        return _BoozerForwardToroidalFluxFieldTerms(
            B=B_dot,
            A=_spatial_field_dot(field_terms.dA_dX, label_geometry_tangent),
        )
    return _BoozerForwardLocalFieldTerms(B=B_dot)


def _penalty_params_tangent_from_decision_tangent(
    params: _BoozerPenaltyParams,
    tangent_state,
):
    if isinstance(tangent_state, _BoozerPenaltyOptimizerStateWithG):
        G_dot = tangent_state.G
    else:
        G_dot = params.G - params.G
    return _BoozerPenaltyParams(
        iota=tangent_state.iota,
        G=G_dot,
        targetlabel=params.targetlabel - params.targetlabel,
        constraint_weight=params.constraint_weight - params.constraint_weight,
        label_type=params.label_type,
        phi_idx=params.phi_idx,
        weight_inv_modB=params.weight_inv_modB,
    )


def _ls_directional_from_field_terms(
    snapshot: _BoozerLSGroupedVJPSnapshot,
    field_terms,
    tangent,
    *,
    G_value=None,
):
    tangent_state = _as_boozer_penalty_optimizer_state(
        tangent,
        optimize_G=snapshot.optimize_G,
    )
    G_value = snapshot.G if G_value is None else G_value
    params = _penalty_params(
        iota=snapshot.iota,
        G_value=G_value,
        targetlabel=snapshot.targetlabel,
        constraint_weight=snapshot.constraint_weight,
        label_type=snapshot.label_type,
        phi_idx=snapshot.phi_idx,
        weight_inv_modB=snapshot.weight_inv_modB,
    )
    geometry_tangent = _geometry_tangent_from_decision_tangent(snapshot, tangent_state)
    label_geometry_tangent = _label_geometry_tangent_from_decision_tangent(
        snapshot,
        tangent_state,
    )
    field_values = _forward_field_values_from_structured_terms(field_terms)
    field_tangent = _forward_field_tangent_from_structured_terms(
        field_terms,
        geometry_tangent,
        label_geometry_tangent,
    )
    params_tangent = _penalty_params_tangent_from_decision_tangent(
        params,
        tangent_state,
    )

    def reduced(geometry, label_geometry, values, reduced_params):
        return _penalty_from_geometry_and_field_terms(
            geometry,
            label_geometry,
            values,
            reduced_params,
        )

    _, directional = jax.jvp(
        reduced,
        (snapshot.geometry, snapshot.label_geometry, field_values, params),
        (
            geometry_tangent,
            label_geometry_tangent,
            field_tangent,
            params_tangent,
        ),
    )
    return directional


def _ls_field_term_cotangents(snapshot: _BoozerLSGroupedVJPSnapshot, tangent):
    unit_cotangent = _as_jax_float64(1.0)
    if snapshot.optimize_G:
        _, field_pullback = jax.vjp(
            lambda terms: _ls_directional_from_field_terms(snapshot, terms, tangent),
            snapshot.field_terms,
        )
        return field_pullback(unit_cotangent)[0], None
    _, field_pullback = jax.vjp(
        lambda terms, G_value: _ls_directional_from_field_terms(
            snapshot,
            terms,
            tangent,
            G_value=G_value,
        ),
        snapshot.field_terms,
        snapshot.G,
    )
    bar_field_terms, bar_G = field_pullback(unit_cotangent)
    return bar_field_terms, bar_G


def _add_G_current_cotangent(d_group, group_array, bar_G, *, optimize_G):
    if optimize_G:
        return d_group
    d_gammas, d_gammadashs, d_currents = d_group
    current_sign = jnp.where(
        group_array[2] < 0.0,
        -jnp.ones_like(group_array[2]),
        jnp.ones_like(group_array[2]),
    )
    dG_dcurrents = bar_G * (4.0 * jnp.pi * 1e-7) * current_sign
    return d_gammas, d_gammadashs, d_currents + dG_dcurrents


def _ls_decision_vector(booz_surf, iota, G):
    optimize_G = G is not None
    sdofs = booz_surf._get_cached_surface_dofs()
    if optimize_G:
        x = _concat_jax_float64(sdofs, [iota, G])
    else:
        x = _concat_jax_float64(sdofs, [iota])
    return x, optimize_G


def _ls_penalty_directional_objective(
    x,
    tangent,
    coil_arrays,
    booz_surf,
    optimize_G,
    weight_inv_modB,
):
    return _directional_derivative(
        _make_ls_penalty_objective(
            booz_surf,
            coil_arrays,
            optimize_G,
            weight_inv_modB,
        ),
        x,
        tangent,
    )


def _make_ls_penalty_objective(
    booz_surf,
    coil_arrays,
    optimize_G,
    weight_inv_modB,
):
    return _make_boozer_penalty_objective_closure(
        coil_arrays=coil_arrays,
        quadpoints_phi=booz_surf.quadpoints_phi,
        quadpoints_theta=booz_surf.quadpoints_theta,
        mpol=booz_surf.mpol,
        ntor=booz_surf.ntor,
        nfp=booz_surf.nfp,
        stellsym=booz_surf.stellsym,
        scatter_indices=booz_surf.scatter_indices,
        surface_kind=booz_surf._surface_geometry_kind,
        label_quadpoints_phi=booz_surf.label_quadpoints_phi,
        label_quadpoints_theta=booz_surf.label_quadpoints_theta,
        label_mpol=booz_surf.label_mpol,
        label_ntor=booz_surf.label_ntor,
        label_nfp=booz_surf.label_nfp,
        label_stellsym=booz_surf.label_stellsym,
        label_scatter_indices=booz_surf.label_scatter_indices,
        label_surface_kind=booz_surf._label_surface_geometry_kind,
        targetlabel=booz_surf.targetlabel,
        constraint_weight=booz_surf.constraint_weight,
        label_type=booz_surf.label_type,
        phi_idx=booz_surf.phi_idx,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )


def _make_boozer_penalty_closure(fn, **kwargs):
    """Generic closure builder for Boozer penalty functions.

    Captures all surface/coil keyword arguments and returns a unary
    ``fn(xx, **kwargs)`` closure suitable for JIT tracing.
    """

    def _closure(xx):
        return fn(xx, **kwargs)

    return _closure


def _make_boozer_penalty_objective_closure(**kwargs):
    return _make_boozer_penalty_closure(_boozer_penalty_objective, **kwargs)


def _make_boozer_penalty_residual_closure(**kwargs):
    return _make_boozer_penalty_closure(_boozer_penalty_residual_vector, **kwargs)


def _directional_derivative(objective, x, tangent):
    _, directional = jax.jvp(objective, (x,), (tangent,))
    return directional


def _traceable_plu_or_dummy(matrix, *, finite):
    """Build PLU factors only for finite matrices on traceable paths.

    The dummy path returns NaN-filled factors so that any accidental solve
    against a failed-solve PLU propagates NaN visibly rather than silently
    returning zeros (which could be mistaken for a valid zero gradient).
    """

    def compute_plu(mat):
        return jax.scipy.linalg.lu(mat)

    def dummy_plu(mat):
        nan_fill = jnp.full_like(mat, jnp.nan)
        return nan_fill, nan_fill, nan_fill

    if isinstance(finite, (bool, np.bool_)):
        return compute_plu(matrix) if bool(finite) else dummy_plu(matrix)
    if (
        isinstance(finite, jax.Array)
        and finite.shape == ()
        and not isinstance(finite, jax.core.Tracer)
    ):
        finite_value = bool(np.asarray(jax.device_get(finite)))
        return compute_plu(matrix) if finite_value else dummy_plu(matrix)

    return jax.lax.cond(
        jnp.asarray(finite, dtype=jnp.bool_), compute_plu, dummy_plu, matrix
    )


def _traceable_lu_piv_or_dummy(matrix, *, finite):
    """Factor a finite matrix once via ``lu_factor`` for shared dispatch.

    Phase 2 traceable analog of :func:`_traceable_plu_or_dummy` for the
    Phase 2 factor-once contract. Returns ``(lu, piv)`` packed factors;
    the public ``(P, L, U)`` triple is derived FROM THE SAME factors via
    :func:`_optimizer_jax._plu_from_lu_piv` so the bytes are
    bit-identical by construction. Failed-solve states materialize NaN
    factors with a placeholder pivot vector so accidental downstream
    solves propagate NaN visibly.
    """

    def compute(mat):
        return _optimizer_jax._factor_dense_hessian(mat, optimizer_backend="ondevice")

    def dummy(mat):
        nan_fill = jnp.full_like(mat, jnp.nan)
        nan_piv = jnp.zeros((mat.shape[0],), dtype=jnp.int32)
        return nan_fill, nan_piv

    if isinstance(finite, (bool, np.bool_)):
        return compute(matrix) if bool(finite) else dummy(matrix)
    if (
        isinstance(finite, jax.Array)
        and finite.shape == ()
        and not isinstance(finite, jax.core.Tracer)
    ):
        finite_value = bool(np.asarray(jax.device_get(finite)))
        return compute(matrix) if finite_value else dummy(matrix)

    return jax.lax.cond(jnp.asarray(finite, dtype=jnp.bool_), compute, dummy, matrix)


def _exact_newton_reporting_fields(result):
    return {
        "message": result.get("message"),
        "failure_category": result.get("failure_category"),
        "failure_stage": result.get("failure_stage"),
        "jacobian_materialized": result.get("jacobian_materialized"),
        "dense_jacobian_shape": result.get("dense_jacobian_shape"),
        "dense_jacobian_bytes": result.get("dense_jacobian_bytes"),
        "max_dense_jacobian_bytes": result.get("max_dense_jacobian_bytes"),
    }


def _none_solve_quality_fields(field_names: tuple[str, ...]) -> dict[str, None]:
    """Return ``None`` placeholders for solve-quality reporting fields.

    Used by result-dict finalizers that do not materialize a Hessian /
    Jacobian (e.g. BFGS, manual LS, LM, exact constraint Newton). The keys
    must always be present so consumers iterating ``res.keys()`` see a
    stable schema across all ``type``-tagged result dicts.
    """
    return dict.fromkeys(field_names, None)


def _ls_hessian_symmetry_rel(H) -> float | None:
    """Return ``‖H − H.T‖_F / ‖H‖_F`` for the LS solve-quality ladder.

    Per ``docs/parity_scientific_equivalence_contract_2026-05-09.md`` §3.1.
    Returns ``None`` when ``H`` is unavailable or has zero Frobenius norm;
    a non-finite norm propagates so the parity arbiter can flag a
    NaN/Inf-laden Hessian instead of treating it as "field unavailable".
    """
    if H is None:
        return None
    norms = np.asarray(
        jax.device_get(
            jnp.stack(
                [
                    jnp.linalg.norm(H, ord="fro"),
                    jnp.linalg.norm(H - H.T, ord="fro"),
                ]
            )
        )
    )
    H_norm = float(norms[0])
    if not np.isfinite(H_norm):
        return H_norm
    if H_norm == 0.0:
        return None
    return float(norms[1]) / H_norm


def _dense_condition_estimate_or_none(matrix, *, lu_piv=None):
    """Return the Hager-Higham 1-norm κ̂ of ``matrix`` (or ``None``).

    Pass ``lu_piv = (lu, piv)`` whenever the caller already holds the
    factorization of ``matrix`` so the Hager-Higham iteration consumes
    those packed factors via ``jsp_linalg.lu_solve`` rather than
    refactorizing ``matrix`` for every inner solve.
    """
    if matrix is None:
        return None
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    estimate = _optimizer_jax._dense_matrix_condition_estimate(
        matrix,
        lu_piv=lu_piv,
    )
    if isinstance(estimate, jax.core.Tracer):
        return estimate
    if isinstance(estimate, jax.Array):
        return float(_host_scalar(estimate))
    return float(estimate)


def _ls_factorization_backend(
    H,
    *,
    optimizer_backend: str | None,
    shared_dispatch: bool = False,
) -> str | None:
    """Return the LS factorization backend string for the result dict.

    Per ``docs/parity_scientific_equivalence_contract_2026-05-09.md`` §3.1.
    Returns ``None`` when ``H`` is absent. When ``shared_dispatch`` is
    ``True`` the Phase 2 factor-once dispatch is active: the forward and
    adjoint solves consume the same packed ``(lu, piv)`` factor bytes by
    construction (see §5.3), and the field reports
    ``"dense-plu-shared"``.

    The ``optimizer_backend == "scipy"`` branch reports ``lapack-dgetrf``
    because ``scipy.linalg.lu_factor`` materializes ``np.asarray(H)`` on
    the host before calling LAPACK ``dgetrf``. Other backends route
    through ``jax.scipy.linalg.lu_factor`` on ``H``'s device; the device
    platform determines the underlying factorization (LAPACK on CPU,
    cuSOLVER ``cusolverDnDgetrf`` on CUDA).
    """
    if H is None:
        return None
    if shared_dispatch:
        return "dense-plu-shared"
    if optimizer_backend == "scipy":
        return "lapack-dgetrf"
    platform = str(H.device.platform).lower()
    if platform in {"gpu", "cuda"}:
        return "cusolver-getrf-ffi"
    return "lapack-dgetrf"


def _ls_shared_lu_piv_dispatch(optimizer_backend: str | None, lu_piv) -> bool:
    return optimizer_backend != "scipy" and lu_piv is not None


def _ls_linear_solve_backend(
    *,
    optimizer_backend: str | None,
    plu_available: bool,
    shared_lu_piv_dispatch: bool,
) -> str:
    if shared_lu_piv_dispatch:
        return "dense-plu-shared"
    if optimizer_backend == "scipy" and plu_available:
        return "dense-plu"
    return "operator"


def _ls_factor_once_dispatch_eligible(H, *, max_dense_jacobian_bytes) -> bool:
    """Return whether ``decision_size**2 * 8 <= max_dense_jacobian_bytes``.

    Phase 2 (`docs/parity_scientific_equivalence_contract_2026-05-09.md`
    §5.3) shares ``(lu, piv)`` between LS forward and adjoint solves only
    when the dense factor fits inside the byte budget. Above the budget,
    the runtime stays on the existing operator-only adjoint path so the
    CLAUDE.md exact-lane scaling-limit guarantees keep holding for large
    problems. ``H`` must already be a real materialized matrix; ``None``
    short-circuits to ``False``.

    Budget-key contract:
    - ``max_dense_linearization_bytes`` (`_DEFAULT_OPTIONS_LS`) gates
      whether the LS Hessian is materialized in the first place by the
      Newton-polish runner; if that gate fails, ``H`` is ``None`` and
      this helper short-circuits without consulting
      ``max_dense_jacobian_bytes``.
    - ``max_dense_jacobian_bytes`` (this argument) is the
      *shared* dense-factor byte budget across the LS factor-once path
      and the Exact Jacobian materialization path. Operators that want
      to allow large LS Hessians but disable LS factor-once dispatch
      can set ``max_dense_jacobian_bytes`` below
      ``max_dense_linearization_bytes`` and the runtime uses
      operator-only LS adjoints while still emitting the dense Hessian
      for reporting. Both options default to 512 MB.
    """
    if H is None:
        return False
    if max_dense_jacobian_bytes is None:
        return True
    n = int(H.shape[0])
    itemsize = int(np.dtype(np.float64).itemsize)
    return n * n * itemsize <= int(max_dense_jacobian_bytes)


_DEFAULT_MAX_DENSE_JACOBIAN_BYTES = 512 * 1024 * 1024


_DEFAULT_OPTIONS_LS = {
    "verbose": True,
    "bfgs_tol": 1e-10,
    "bfgs_maxiter": 1500,
    "optimizer_backend": "scipy",
    "limited_memory": False,
    "newton_tol": 1e-11,
    "newton_maxiter": 40,
    "newton_stab": 0.0,
    "weight_inv_modB": True,
    "materialize_dense_linearization": None,
    "max_dense_linearization_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES,
    "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES,
    "record_scipy_callback_trace": False,
}

_DEFAULT_OPTIONS_EXACT = {
    "verbose": True,
    "newton_tol": 1e-13,
    "newton_maxiter": 40,
    "weight_inv_modB": False,
    "max_dense_jacobian_bytes": _DEFAULT_MAX_DENSE_JACOBIAN_BYTES,
}


def _dense_jacobian_default_for_backend():
    return get_backend_policy().max_dense_jacobian_bytes


def _nan_tree_like(tree):
    return jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.nan), tree)


def _solve_with_nan_on_failure(solution, success):
    return jax.lax.cond(
        jnp.asarray(success, dtype=jnp.bool_),
        lambda value: value,
        _nan_tree_like,
        solution,
    )


def _host_linearization_device():
    return jax.devices("cpu")[0]


def _place_linearization_factors_for_residency(factors, residency):
    if factors is None or residency == "device":
        return factors
    device = _host_linearization_device()
    return jax.tree.map(
        lambda factor: runtime_device_put(factor, device=device), factors
    )


def _solver_option_defaults(boozer_type, user_options):
    defaults = dict(
        _DEFAULT_OPTIONS_LS if boozer_type == "ls" else _DEFAULT_OPTIONS_EXACT
    )
    dense_default = _dense_jacobian_default_for_backend()
    if "max_dense_jacobian_bytes" not in user_options:
        defaults["max_dense_jacobian_bytes"] = dense_default
    if boozer_type == "ls" and "max_dense_linearization_bytes" not in user_options:
        defaults["max_dense_linearization_bytes"] = dense_default
    return defaults


# Options only meaningful for the target/private optimizer backend.
_PRIVATE_OPTIMIZER_OPTIONS = frozenset(
    {
        "force_ondevice_limited_memory",
        "line_search_maxiter",
    }
)

# Options shared by the public SciPy L-BFGS lane and the private L-BFGS lanes.
_LBFGS_TUNING_OPTIONS = frozenset({"maxcor", "ftol", "maxfun", "maxls"})
_LM_TUNING_OPTION_KEYS = ("ftol", "xtol", "gtol")
_LM_TUNING_OPTIONS = frozenset(_LM_TUNING_OPTION_KEYS)
_SCIPY_TRACE_OPTIONS = frozenset({"record_scipy_callback_trace"})

# Callback options accepted by all backends.
_CALLBACK_OPTIONS = frozenset({"stage_callback", "progress_callback"})
_LINEARIZATION_RESIDENCY_VALUES = frozenset({"device", "host"})
_ONDEVICE_LEAST_SQUARES_METHODS = _optimizer_jax._TARGET_LEAST_SQUARES_METHODS
_LEAST_SQUARES_METHODS = frozenset({"lm"}) | _ONDEVICE_LEAST_SQUARES_METHODS
_ONDEVICE_OPTIMIZER_METHODS = (
    frozenset({"bfgs-ondevice", "lbfgs-ondevice"}) | _ONDEVICE_LEAST_SQUARES_METHODS
)
_LS_DYNAMIC_OPTION_KEYS = frozenset(
    {"least_squares_algorithm", "materialize_dense_linearization"}
)

_ALLOWED_OPTIONS_LS = (
    frozenset(_DEFAULT_OPTIONS_LS)
    | {"linearization_residency"}
    | _LS_DYNAMIC_OPTION_KEYS
    | _PRIVATE_OPTIMIZER_OPTIONS
    | _LBFGS_TUNING_OPTIONS
    | _LM_TUNING_OPTIONS
    | _SCIPY_TRACE_OPTIONS
    | _CALLBACK_OPTIONS
)
_ALLOWED_OPTIONS_EXACT = frozenset(_DEFAULT_OPTIONS_EXACT) | {
    "optimizer_backend",
    "linearization_residency",
    "stage_callback",
}


def default_least_squares_algorithm_for_backend(optimizer_backend):
    del optimizer_backend
    return "quasi-newton"


def default_materialize_dense_linearization_for_backend(optimizer_backend):
    del optimizer_backend
    return True


class _BoozerSolverOptions(dict):
    """Mutable solver options with byte-capped dense-finalization defaults."""

    def __init__(
        self,
        values,
        *,
        materialize_dense_linearization_explicit=False,
    ):
        super().__init__(values)
        self.materialize_dense_linearization_explicit = (
            materialize_dense_linearization_explicit
        )

    def __setitem__(self, key, value):
        if key == "materialize_dense_linearization":
            self.materialize_dense_linearization_explicit = True
        super().__setitem__(key, value)
        if key == "optimizer_backend":
            self._refresh_backend_dense_linearization_default(value)

    def update(self, other=(), /, **kwargs):
        for key, value in dict(other, **kwargs).items():
            self[key] = value

    def _refresh_backend_dense_linearization_default(self, optimizer_backend):
        if self.materialize_dense_linearization_explicit:
            return
        if "materialize_dense_linearization" not in self:
            return
        super().__setitem__(
            "materialize_dense_linearization",
            default_materialize_dense_linearization_for_backend(optimizer_backend),
        )


def _default_ls_optimizer_backend() -> str:
    return get_backend_policy().default_optimizer_backend


def _normalize_solver_options(raw_options, boozer_type):
    """Validate and normalize constructor options for a Boozer solve mode."""
    if "bfgs_method" in raw_options:
        raise ValueError(
            "BoozerSurfaceJAX option 'bfgs_method' was removed. "
            "Use 'optimizer_backend' with one of: auto, scipy, ondevice."
        )

    allowed_option_keys = (
        _ALLOWED_OPTIONS_LS if boozer_type == "ls" else _ALLOWED_OPTIONS_EXACT
    )
    unknown_option_keys = sorted(set(raw_options) - allowed_option_keys)
    if unknown_option_keys:
        unknown_keys = ", ".join(repr(key) for key in unknown_option_keys)
        raise ValueError(f"Unknown BoozerSurfaceJAX option(s): {unknown_keys}.")

    optimizer_backend = raw_options.get("optimizer_backend")
    if (
        optimizer_backend is not None
        and optimizer_backend not in VALID_OPTIMIZER_BACKENDS
    ):
        raise ValueError("optimizer_backend must be one of: auto, scipy, ondevice.")
    effective_optimizer_backend = _optimizer_jax.resolve_optimizer_backend(
        optimizer_backend
    )
    linearization_residency = raw_options.get(
        "linearization_residency",
        get_backend_policy().default_residency,
    )
    if linearization_residency not in _LINEARIZATION_RESIDENCY_VALUES:
        allowed = ", ".join(sorted(_LINEARIZATION_RESIDENCY_VALUES))
        raise ValueError(f"linearization_residency must be one of: {allowed}.")
    least_squares_algorithm = raw_options.get("least_squares_algorithm")
    if (
        least_squares_algorithm is not None
        and least_squares_algorithm not in VALID_LEAST_SQUARES_ALGORITHMS
    ):
        allowed = ", ".join(sorted(VALID_LEAST_SQUARES_ALGORITHMS))
        raise ValueError(f"least_squares_algorithm must be one of: {allowed}.")
    if is_parity_mode() and float(raw_options.get("newton_stab", 0.0)) != 0.0:
        raise ValueError(
            "BoozerSurfaceJAX parity mode requires newton_stab=0.0 so "
            "linear residuals are checked against the undamped operator."
        )

    if boozer_type == "ls":
        private_keys = sorted(set(raw_options) & _PRIVATE_OPTIMIZER_OPTIONS)
        if private_keys and effective_optimizer_backend == "scipy":
            keys_str = ", ".join(repr(k) for k in private_keys)
            raise ValueError(
                f"Private optimizer option(s) {keys_str} require "
                "optimizer_backend='ondevice'."
            )

    normalized_options = dict(raw_options)
    if boozer_type == "ls":
        normalized_options["optimizer_backend"] = effective_optimizer_backend
        if "least_squares_algorithm" not in normalized_options:
            normalized_options["least_squares_algorithm"] = (
                default_least_squares_algorithm_for_backend(
                    normalized_options["optimizer_backend"]
                )
            )
        if normalized_options.get("materialize_dense_linearization") is None:
            normalized_options["materialize_dense_linearization"] = (
                default_materialize_dense_linearization_for_backend(
                    normalized_options["optimizer_backend"]
                )
            )
        if (
            normalized_options["optimizer_backend"] == "ondevice"
            and normalized_options["least_squares_algorithm"] == "optimistix-lm"
        ):
            callback_keys = sorted(set(normalized_options) & _CALLBACK_OPTIONS)
            if callback_keys:
                keys_str = ", ".join(repr(k) for k in callback_keys)
                raise ValueError(
                    f"BoozerSurfaceJAX option(s) {keys_str} are incompatible "
                    "with least_squares_algorithm='optimistix-lm'. Use "
                    "least_squares_algorithm='lm' for callback-instrumented "
                    "on-device LM runs."
                )
            # Missing keys use lane defaults so explicit default values pass.
            tuning_keys = _optimizer_jax._optimistix_lm_nondefault_tuning_options(
                normalized_options.get(
                    "ftol",
                    _optimizer_jax._OPTIMISTIX_LM_DEFAULT_FTOL,
                ),
                normalized_options.get(
                    "xtol",
                    _optimizer_jax._OPTIMISTIX_LM_DEFAULT_XTOL,
                ),
                normalized_options.get("gtol"),
            )
            if tuning_keys:
                keys_str = ", ".join(repr(k) for k in tuning_keys)
                raise ValueError(
                    f"BoozerSurfaceJAX option(s) {keys_str} are incompatible "
                    "with least_squares_algorithm='optimistix-lm'. "
                    "optimistix-lm uses the solver 'tol' as the single "
                    "Optimistix/Lineax convergence tolerance."
                )
    if boozer_type == "exact":
        normalized_options.pop("optimizer_backend", None)
    normalized_options.setdefault("linearization_residency", linearization_residency)
    return normalized_options


class BoozerSurfaceJAX(Optimizable):
    """JAX-native Boozer surface solver.

    Mirrors the CPU ``BoozerSurface`` API — inherits ``Optimizable``,
    carries ``self.label``, and returns result dicts with ``vjp`` hooks.
    The object wrapper is intentionally stateful and should be treated as
    thread-confined: ``run_code()``, ``recompute_bell()``, and related helpers
    mutate ``self.res``, ``self.surface``, ``self.need_to_run_code``, and the
    cached grouped-coil data. Use ``run_code_traceable()`` plus immutable coil
    specs/arrays when you need a pure array contract for the target ondevice
    lane.

    This class therefore sits at the current architecture boundary:
    immutable grouped-coil specs feed the traceable JAX kernels, while the
    public wrapper still owns mutable solve state and flat decision-vector
    orchestration for compatibility with the existing outer optimizer stack.

    Args:
        biotsavart: ``BiotSavartJAX`` instance (or any adapter exposing
            ``coil_set_spec()`` for explicit immutable grouped-coil state).
        surface: CPU ``SurfaceXYZTensorFourier`` instance.
        label: An ``Optimizable`` that computes a flux surface label
            (e.g. ``Volume``, ``ToroidalFlux``).  Stored as ``self.label``
            for downstream consumers that call ``boozer_surface.label.J()``.
        targetlabel: target value for the label constraint.
        constraint_weight: penalty weight.  If ``None``, BoozerExact
            path is used; otherwise BoozerLS.
        options: dict of solver options (see ``_DEFAULT_OPTIONS_*``).
            For LS solves, the omitted ``optimizer_backend`` default follows the
            active simsopt backend policy. ``optimizer_backend="scipy"`` remains
            the trusted CPU/reference lane and ``"ondevice"`` is the target
            on-device lane.
            ``record_scipy_callback_trace=True`` records every SciPy adapter
            objective evaluation on the SciPy reference lane only.
            ``least_squares_algorithm="quasi-newton"``
            preserves the historical BFGS/L-BFGS route; ``"lm"`` enables the
            residual-vector Levenberg-Marquardt route on supported backends.
        surface_runtime_state: optional immutable surface-metadata snapshot.
            When provided, traceable and exact JAX solver paths use this cached
            state instead of querying the live surface object for quadrature,
            symmetry-mask construction, or scatter metadata.
    """

    supports_explicit_surface_warm_start = True

    def __init__(
        self,
        biotsavart,
        surface,
        label,
        targetlabel,
        constraint_weight=None,
        options=None,
        surface_runtime_state=None,
    ):
        super().__init__(depends_on=[biotsavart])

        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = float(targetlabel)
        self.constraint_weight = constraint_weight
        self.need_to_run_code = True
        self.res = None
        self._solver_generation = 0
        self._traceable_solve_state_token = _new_traceable_solve_state_token()
        self._traceable_runtime_entry_cache = None

        # Determine solver type
        self.boozer_type = "ls" if constraint_weight is not None else "exact"

        # Infer label_type from the label object.
        # Only Volume, Area, and ToroidalFlux have JAX-native implementations.
        label_cls = type(label).__name__
        if "Volume" in label_cls:
            self.label_type = "volume"
        elif "Area" in label_cls:
            self.label_type = "area"
        elif "ToroidalFlux" in label_cls:
            self.label_type = "toroidal_flux"
        else:
            raise ValueError(
                f"Unsupported label type {label_cls!r} for BoozerSurfaceJAX. "
                "Supported: Volume, Area, ToroidalFlux."
            )

        user_options = dict(options or {})
        materialize_dense_linearization_explicit = (
            self.boozer_type == "ls"
            and user_options.get("materialize_dense_linearization") is not None
        )
        raw_options = _normalize_solver_options(
            user_options,
            self.boozer_type,
        )
        defaults = _solver_option_defaults(self.boozer_type, user_options)
        self.options = _BoozerSolverOptions(
            {**defaults, **raw_options},
            materialize_dense_linearization_explicit=(
                materialize_dense_linearization_explicit
            ),
        )
        if self.boozer_type == "ls":
            if self.options["optimizer_backend"] not in VALID_OPTIMIZER_BACKENDS:
                raise ValueError(
                    "optimizer_backend must be one of: auto, scipy, ondevice."
                )

        runtime_state = (
            surface_runtime_state
            if surface_runtime_state is not None
            else build_boozer_surface_runtime_state(surface)
        )
        label_surface = label.surface
        label_runtime_state = (
            runtime_state
            if label_surface is surface
            else build_boozer_surface_runtime_state(label_surface)
        )

        # --- Extract immutable surface metadata once; keep DOFs cached locally ---
        self._surface_runtime_state = runtime_state
        self._label_surface_runtime_state = label_runtime_state
        self._store_surface_dofs(surface.get_dofs())
        self._exact_mask_indices = None
        self.mpol = runtime_state.mpol
        self.ntor = runtime_state.ntor
        self.nfp = runtime_state.nfp
        self.stellsym = runtime_state.stellsym
        self.quadpoints_phi = runtime_state.quadpoints_phi
        self.quadpoints_theta = runtime_state.quadpoints_theta
        self._surface_geometry_kind = runtime_state.surface_kind
        self.scatter_indices = runtime_state.scatter_indices
        self.label_mpol = label_runtime_state.mpol
        self.label_ntor = label_runtime_state.ntor
        self.label_nfp = label_runtime_state.nfp
        self.label_stellsym = label_runtime_state.stellsym
        self.label_quadpoints_phi = label_runtime_state.quadpoints_phi
        self.label_quadpoints_theta = label_runtime_state.quadpoints_theta
        self._label_surface_geometry_kind = label_runtime_state.surface_kind
        self.label_scatter_indices = label_runtime_state.scatter_indices

        # Toroidal flux phi index (first phi point by default)
        self.phi_idx = 0

        self._traceable_penalty_objective_cache = {}
        self._traceable_penalty_residual_cache = {}
        self._traceable_exact_residual_cache = {}
        self._reference_penalty_objective_cache = {}
        self._reference_penalty_value_and_grad_cache = {}
        self._reference_penalty_residual_cache = {}

        # Coil data (extracted once, updated via _refresh_coil_data)
        self._refresh_coil_data()

    @property
    def _coil_arrays(self):
        """Coil geometry tuples ``(gammas, gammadashs, currents)`` without index lists."""
        return list(grouped_field_inputs_from_spec(self.coil_set_spec))

    @property
    def surface_runtime_state(self):
        """Immutable surface metadata snapshot used by the JAX solver lanes."""
        return self._surface_runtime_state

    def get_solved_runtime_state(self):
        """Return the last successful solved-state summary without host rereads."""
        if self.need_to_run_code:
            raise RuntimeError(
                "BoozerSurfaceJAX solve state is stale. Re-run "
                "boozer_surface.run_code(...) before requesting a runtime summary."
            )
        if self.res is None or not bool(self.res["primal_success"]):
            raise RuntimeError(
                "BoozerSurfaceJAX has no successful solve state. "
                "Call boozer_surface.run_code(...) before requesting a solved "
                "runtime summary."
            )
        G = self.res["G"]
        solved_G = None if G is None else _as_jax_float64(G)
        return _BoozerSolvedRuntimeState(
            sdofs=_as_jax_float64(self.res["sdofs"]),
            iota=_as_jax_float64(self.res["iota"]),
            G=solved_G,
            weight_inv_modB=bool(self.res["weight_inv_modB"]),
        )

    def _linear_solve_tolerance(self):
        if self.boozer_type == "exact":
            return min(1e-10, max(float(self.options["newton_tol"]) * 0.1, 1e-14))
        return min(
            1e-10,
            max(
                min(
                    float(self.options["bfgs_tol"]),
                    float(self.options["newton_tol"]),
                )
                * 0.1,
                1e-14,
            ),
        )

    def _resolved_linearization_kind(self):
        if self.boozer_type == "exact":
            return "exact_jacobian"
        return self.res["linearization_kind"]

    def _build_runtime_linear_solve_callbacks(self, solved_state):
        linearization_kind = self._resolved_linearization_kind()
        optimize_G = solved_state.G is not None
        x = self._pack_decision_vector(
            solved_state.iota,
            solved_state.G,
            sdofs=solved_state.sdofs,
        )
        tol = self._linear_solve_tolerance()
        tol_host = float(tol)
        compute_device = x.device

        def stage_linearization_factor(factor, *, dtype=None):
            array = (
                jnp.asarray(factor)
                if dtype is None
                else jnp.asarray(factor, dtype=dtype)
            )
            return runtime_device_put(array, device=compute_device)

        def pack_callbacks(
            apply_forward,
            apply_transpose,
            solve_forward,
            solve_transpose=None,
            solve_forward_with_status=None,
            solve_transpose_with_status=None,
            linear_solve_backend="operator",
            linear_solve_factors=None,
        ):
            def _with_nan_status(solver):
                def wrapped(rhs):
                    solution, success = solver(rhs)
                    return _solve_with_nan_on_failure(solution, success), success

                return wrapped

            if solve_transpose is None:
                solve_transpose = solve_forward
            if solve_transpose_with_status is None:
                solve_transpose_with_status = solve_forward_with_status
            solve_forward_with_status = _with_nan_status(solve_forward_with_status)
            solve_transpose_with_status = _with_nan_status(solve_transpose_with_status)
            if linear_solve_factors is not None:
                linear_solve_factors = jax.tree.map(
                    lambda factor: stage_linearization_factor(factor, dtype=x.dtype),
                    linear_solve_factors,
                )
            return (
                linearization_kind,
                x.shape[0],
                x.dtype,
                apply_forward,
                apply_transpose,
                solve_forward,
                solve_transpose,
                solve_forward_with_status,
                solve_transpose_with_status,
                linear_solve_backend,
                linear_solve_factors,
            )

        if linearization_kind == "hessian" and _ls_shared_lu_piv_dispatch(
            self.options["optimizer_backend"],
            self.res.get("LU_PIV"),
        ):
            # Phase 2 factor-once dispatch (see
            # docs/parity_scientific_equivalence_contract_2026-05-09.md
            # §5.3): forward and adjoint solves consume the same packed
            # ``(lu, piv)`` so their factor bytes are bit-identical by
            # construction. The public PLU triple stays load-bearing for
            # the ``linear_solve_factors`` reporting field. The
            # ``optimizer_backend == "scipy"`` lane is intentionally
            # excluded — it routes through the scipy host-LAPACK block
            # below to preserve C++-oracle byte parity for the
            # ``cpp_compatible_probe`` LS skeleton (which depends on
            # ``scipy.linalg.solve_triangular`` host-resident factor
            # consumption rather than the device ``lu_solve`` getrs
            # call). ``apply_forward`` / ``apply_transpose`` use the
            # device-resident Hessian directly so the JAX/CUDA LS lane
            # never crosses host on the runtime callback path.
            lu_piv = self.res["LU_PIV"]
            lu_device = stage_linearization_factor(lu_piv[0], dtype=x.dtype)
            piv_device = stage_linearization_factor(lu_piv[1], dtype=jnp.int32)
            H_dev = stage_linearization_factor(self.res["hessian"], dtype=x.dtype)

            def apply_forward(rhs):
                return H_dev @ jnp.asarray(rhs, dtype=x.dtype)

            def apply_transpose(rhs):
                return H_dev.T @ jnp.asarray(rhs, dtype=x.dtype)

            def solve_forward(rhs):
                return _optimizer_jax._lu_solve_dense_hessian(
                    (lu_device, piv_device),
                    jnp.asarray(rhs, dtype=x.dtype),
                    transpose=False,
                )

            def solve_transpose(rhs):
                return _optimizer_jax._lu_solve_dense_hessian(
                    (lu_device, piv_device),
                    jnp.asarray(rhs, dtype=x.dtype),
                    transpose=True,
                )

            def solve_forward_with_status(rhs):
                solved = solve_forward(rhs)
                return solved, jnp.all(jnp.isfinite(solved))

            def solve_transpose_with_status(rhs):
                solved = solve_transpose(rhs)
                return solved, jnp.all(jnp.isfinite(solved))

            return pack_callbacks(
                apply_forward,
                apply_transpose,
                solve_forward,
                solve_transpose,
                solve_forward_with_status=solve_forward_with_status,
                solve_transpose_with_status=solve_transpose_with_status,
                linear_solve_backend="dense-plu-shared",
                linear_solve_factors=tuple(
                    jnp.asarray(factor, dtype=x.dtype) for factor in self.res["PLU"]
                ),
            )

        if (
            linearization_kind == "hessian"
            and self.options["optimizer_backend"] == "scipy"
            and self.res.get("PLU") is not None
        ):
            P_host, L_host, U_host = (
                np.asarray(factor, dtype=np.float64) for factor in self.res["PLU"]
            )
            H_host = P_host @ L_host @ U_host

            def apply_forward(rhs):
                return jnp.asarray(
                    H_host @ np.asarray(rhs, dtype=np.float64), dtype=x.dtype
                )

            def apply_transpose(rhs):
                return jnp.asarray(
                    H_host.T @ np.asarray(rhs, dtype=np.float64),
                    dtype=x.dtype,
                )

            def solve_forward(rhs):
                rhs_host = np.asarray(rhs, dtype=np.float64)
                y = scipy.linalg.solve_triangular(
                    L_host,
                    P_host.T @ rhs_host,
                    lower=True,
                )
                solved = scipy.linalg.solve_triangular(U_host, y, lower=False)
                return jnp.asarray(solved, dtype=x.dtype)

            def solve_transpose(rhs):
                rhs_host = np.asarray(rhs, dtype=np.float64)
                y = scipy.linalg.solve_triangular(U_host.T, rhs_host, lower=True)
                z = scipy.linalg.solve_triangular(L_host.T, y, lower=False)
                solved = P_host @ z
                return jnp.asarray(solved, dtype=x.dtype)

            def solve_forward_with_status(rhs):
                solved = solve_forward(rhs)
                return solved, jnp.all(jnp.isfinite(solved))

            def solve_transpose_with_status(rhs):
                solved = solve_transpose(rhs)
                return solved, jnp.all(jnp.isfinite(solved))

            return pack_callbacks(
                apply_forward,
                apply_transpose,
                solve_forward,
                solve_transpose,
                solve_forward_with_status=solve_forward_with_status,
                solve_transpose_with_status=solve_transpose_with_status,
                linear_solve_backend="dense-plu",
                linear_solve_factors=tuple(
                    jnp.asarray(factor, dtype=x.dtype) for factor in self.res["PLU"]
                ),
            )

        # On the supported JAX/CUDA LS lane, runtime adjoints stay
        # operator-backed even if a dense Hessian/PLU was materialized for
        # diagnostics or parity probes.
        if linearization_kind == "least_squares_normal":
            residual_fn = self._make_penalty_residual_with(
                optimize_G,
                solved_state.weight_inv_modB,
                coil_set_spec=self.coil_set_spec,
                hostify_inputs=False,
            )
            operator = _optimizer_jax._least_squares_normal_operator(residual_fn, x)

            def solve_forward(rhs):
                return _optimizer_jax._solve_least_squares_normal_system(
                    residual_fn,
                    x,
                    rhs,
                    tol=tol_host,
                )

            def solve_forward_with_status(rhs):
                return _optimizer_jax._solve_least_squares_normal_system_with_status(
                    residual_fn,
                    x,
                    rhs,
                    tol=tol_host,
                )

            return pack_callbacks(
                operator["matvec"],
                operator["transpose_matvec"],
                solve_forward,
                solve_forward_with_status=solve_forward_with_status,
                solve_transpose_with_status=solve_forward_with_status,
            )

        if linearization_kind == "hessian":
            objective_fn = self._make_penalty_objective_with(
                optimize_G,
                solved_state.weight_inv_modB,
                coil_set_spec=self.coil_set_spec,
                hostify_inputs=False,
            )
            hvp_fn = _optimizer_jax._hessian_vector_product_fn(objective_fn)
            stab = float(self.options["newton_stab"])

            def apply_forward(rhs):
                rhs = jnp.asarray(rhs)
                stab_value = jnp.asarray(stab, dtype=rhs.dtype)

                def apply_column(column):
                    return hvp_fn(x, column) + stab_value * column

                return _optimizer_jax._apply_column_batched_operator(
                    apply_column,
                    rhs,
                )

            def solve_forward(rhs):
                return _optimizer_jax._solve_hessian_system(
                    objective_fn,
                    x,
                    rhs,
                    stab=stab,
                    tol=tol_host,
                )

            def solve_forward_with_status(rhs):
                return _optimizer_jax._solve_hessian_system_with_status(
                    objective_fn,
                    x,
                    rhs,
                    stab=stab,
                    tol=tol_host,
                )

            return pack_callbacks(
                apply_forward,
                apply_forward,
                solve_forward,
                solve_forward_with_status=solve_forward_with_status,
                solve_transpose_with_status=solve_forward_with_status,
            )

        if linearization_kind == "exact_jacobian":
            residual_fn = self._make_exact_residual(
                self._compute_stellsym_mask_indices()
            )
            operator = _optimizer_jax._jacobian_linear_operator(residual_fn, x)

            def solve_jacobian_system_with_status(rhs, *, transpose):
                return _optimizer_jax._solve_jacobian_system_with_status(
                    residual_fn,
                    x,
                    rhs,
                    transpose=transpose,
                    tol=tol_host,
                )

            def solve_jacobian_system(rhs, *, transpose):
                return _optimizer_jax._solve_jacobian_system(
                    residual_fn,
                    x,
                    rhs,
                    transpose=transpose,
                    tol=tol_host,
                )

            def solve_forward(rhs):
                return solve_jacobian_system(rhs, transpose=False)

            def solve_transpose(rhs):
                return solve_jacobian_system(rhs, transpose=True)

            def solve_forward_with_status(rhs):
                return solve_jacobian_system_with_status(rhs, transpose=False)

            def solve_transpose_with_status(rhs):
                return solve_jacobian_system_with_status(rhs, transpose=True)

            return pack_callbacks(
                operator["matvec"],
                operator["transpose_matvec"],
                solve_forward,
                solve_transpose,
                solve_forward_with_status=solve_forward_with_status,
                solve_transpose_with_status=solve_transpose_with_status,
            )

        raise RuntimeError(
            f"Unsupported BoozerSurfaceJAX linearization kind {linearization_kind!r}."
        )

    def get_adjoint_runtime_state(self):
        """Return the last successful adjoint-state summary for wrapper gradients."""
        solved_state = self.get_solved_runtime_state()
        if (
            not bool(self.res["adjoint_linear_solve_available"])
            or self.res["vjp_groups"] is None
        ):
            raise RuntimeError(
                "BoozerSurfaceJAX has no valid adjoint state. "
                "Call boozer_surface.run_code(...) before requesting adjoints."
            )

        vjp_groups = self.res["vjp_groups"]

        def stream_group_vjps(adjoint):
            yield from vjp_groups(adjoint, self, solved_state.iota, solved_state.G)

        (
            linearization_kind,
            decision_size,
            dtype,
            apply_forward,
            apply_transpose,
            solve_forward,
            solve_transpose,
            solve_forward_with_status,
            solve_transpose_with_status,
            linear_solve_backend,
            linear_solve_factors,
        ) = self._build_runtime_linear_solve_callbacks(solved_state)

        return _BoozerAdjointRuntimeState(
            solved_state=solved_state,
            linearization_kind=linearization_kind,
            decision_size=decision_size,
            dtype=dtype,
            apply_forward=apply_forward,
            apply_transpose=apply_transpose,
            solve_forward=solve_forward,
            solve_transpose=solve_transpose,
            solve_forward_with_status=solve_forward_with_status,
            solve_transpose_with_status=solve_transpose_with_status,
            stream_group_vjps=stream_group_vjps,
            linear_solve_backend=linear_solve_backend,
            dense_linear_solve_factors_available=bool(
                self.res["dense_linear_solve_factors_available"]
            ),
            linear_solve_factors=linear_solve_factors,
            linearization_residency=str(
                self.res.get("linearization_residency", "device")
            ),
        )

    @property
    def _coil_index_lists(self):
        """Per-group coil index lists from ``coil_groups``."""
        return list(grouped_coil_index_lists_from_spec(self.coil_set_spec))

    def recompute_bell(self, parent=None):
        """Mark solver as needing re-execution (dirty flag)."""
        self.need_to_run_code = True

    def _validate_none_G_precondition(self, G):
        if G is not None:
            return
        _require_fixed_currents_for_none_G(
            self.biotsavart,
            component="BoozerSurfaceJAX",
            coil_attrs=("coils",),
        )

    def _refresh_coil_data(self):
        """Extract coil geometry and currents as JAX arrays.

        Groups coils by quadrature point count so that coils with
        different ``num_quad_points`` can coexist without crashing
        on array stacking.
        """
        self.coil_set_spec = _extract_grouped_coil_set_spec(self.biotsavart)
        self.coil_groups = list(grouped_field_data_from_spec(self.coil_set_spec))
        self.coil_currents = grouped_coil_currents_from_spec(self.coil_set_spec)
        self._reference_penalty_objective_cache.clear()
        self._reference_penalty_value_and_grad_cache.clear()
        self._reference_penalty_residual_cache.clear()

    def _emit_stage_callback(
        self,
        label: str,
        **extra: float | str | None,
    ) -> None:
        callback = self.options.get("stage_callback")
        if callback is not None:
            callback(label, **extra)

    def _solver_diagnostics_payload(
        self,
        result,
        *,
        gradient_key: str,
        residual_key: str | None = None,
    ) -> dict[str, float]:
        payload = {
            "objective": float(_host_scalar(result["fun"])),
        }
        gradient = result.get(gradient_key)
        if gradient is not None:
            payload["grad_inf"] = float(_host_inf_norm(gradient))
        if residual_key is not None:
            residual = result.get(residual_key)
            if residual is not None:
                payload["residual_inf"] = float(_host_inf_norm(residual))
        return payload

    def _make_solver_progress_callback(self, method: str):
        stage_callback = self.options.get("stage_callback")
        if stage_callback is None:
            return None

        def emit_progress(iteration: int, fun_value: float, grad_inf: float) -> None:
            if iteration <= 5 or iteration % 25 == 0:
                stage_callback(
                    "boozer_ls_progress",
                    iteration=float(iteration),
                    objective=float(fun_value),
                    grad_inf=float(grad_inf),
                    method=method,
                )

        return emit_progress

    def _make_newton_progress_callback(self):
        stage_callback = self.options.get("stage_callback")
        if stage_callback is None:
            return None

        def emit_progress(iteration: int, fun_value: float, grad_norm: float) -> None:
            stage_callback(
                "boozer_newton_progress",
                iteration=float(iteration),
                objective=float(fun_value),
                grad_norm=float(grad_norm),
            )

        return emit_progress

    def _resolve_newton_progress_callback(self, method: str):
        if method in _ONDEVICE_OPTIMIZER_METHODS:
            return None
        return self._make_newton_progress_callback()

    def _store_surface_dofs(self, dofs):
        self._surface_dofs = _as_jax_float64(dofs)
        self._surface_dofs_fingerprint = _surface_dofs_fingerprint_from_dofs(
            self._surface_dofs
        )
        return self._surface_dofs

    def _raise_if_surface_dofs_drifted(self, *, callback_name: str | None = None):
        if (
            _surface_dofs_fingerprint_from_dofs(self.surface.get_dofs())
            == self._surface_dofs_fingerprint
        ):
            return
        callback_text = (
            ""
            if callback_name is None
            else f" before result callback {callback_name!r}"
        )
        raise RuntimeError(
            "BoozerSurfaceJAX cached surface DOFs are stale"
            f"{callback_text}: the live surface DOFs changed since the last "
            "BoozerSurfaceJAX synchronization. Re-run run_code(...) or rebuild "
            "the BoozerSurfaceJAX instance before requesting adjoints."
        )

    def _get_surface_dofs(self):
        """Get current surface DOFs as a JAX array and refresh the cache."""
        return self._store_surface_dofs(self.surface.get_dofs())

    def _get_cached_surface_dofs(self):
        """Get synchronized surface DOFs after validating the live host surface."""
        self._raise_if_surface_dofs_drifted()
        return self._surface_dofs

    def _set_surface_dofs(self, dofs_jax):
        """Update cached DOFs and mirror them back to the live surface."""
        self._store_surface_dofs(dofs_jax)
        self.surface.set_dofs(_host_numpy(self._surface_dofs))

    def _pack_decision_vector(self, iota, G, sdofs=None):
        """Pack [surface_dofs, iota] or [surface_dofs, iota, G]."""
        if sdofs is None:
            sdofs = self._get_surface_dofs()
        if G is not None:
            return _concat_jax_float64(sdofs, [iota, G])
        return _concat_jax_float64(sdofs, [iota])

    def _make_penalty_optimizer_state(self, iota, G, *, sdofs=None):
        if sdofs is None:
            sdofs = self._get_surface_dofs()
        if G is None:
            return _BoozerPenaltyOptimizerState(
                surface_dofs=_as_jax_float64(sdofs),
                iota=_as_jax_float64(iota),
            )
        return _BoozerPenaltyOptimizerStateWithG(
            surface_dofs=_as_jax_float64(sdofs),
            iota=_as_jax_float64(iota),
            G=_as_jax_float64(G),
        )

    def _unpack_decision_vector(self, x, optimize_G):
        """Unpack decision vector → (sdofs, iota, G_or_None)."""
        sdofs, iota, G = _split_decision_vector_jax(x, optimize_G=optimize_G)
        if optimize_G:
            return _host_numpy(sdofs), float(_host_scalar(iota)), float(_host_scalar(G))
        return _host_numpy(sdofs), float(_host_scalar(iota)), None

    def _unpack_penalty_optimizer_state(self, x, optimize_G):
        optimizer_state = _as_boozer_penalty_optimizer_state(x, optimize_G=optimize_G)
        if optimize_G:
            return (
                _host_numpy(optimizer_state.surface_dofs),
                float(_host_scalar(optimizer_state.iota)),
                float(_host_scalar(optimizer_state.G)),
            )
        return (
            _host_numpy(optimizer_state.surface_dofs),
            float(_host_scalar(optimizer_state.iota)),
            None,
        )

    def _unpack_decision_vector_jax(
        self,
        x,
        optimize_G,
        coil_set_spec=None,
        coil_arrays=None,
    ):
        """JAX-array version of ``_unpack_decision_vector``."""
        sdofs, iota, G = _split_decision_vector_jax(x, optimize_G=optimize_G)
        if optimize_G:
            return sdofs, iota, G
        G = compute_G_from_currents(
            _grouped_coil_currents(
                coil_arrays=coil_arrays,
                coil_set_spec=_resolved_coil_set_spec(
                    self.coil_set_spec,
                    coil_arrays=coil_arrays,
                    coil_set_spec=coil_set_spec,
                ),
            )
        )
        return sdofs, iota, G

    def _make_penalty_objective_with(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
        coil_set_spec=None,
        coil_arrays=None,
        *,
        hostify_inputs=True,
        boozer_reduction_mode="default",
    ):
        """Build penalty objective with explicit overrides."""
        resolved_coil_set_spec = _resolved_coil_set_spec(
            self.coil_set_spec,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        resolved_constraint_weight = self._resolve_constraint_weight(constraint_weight)
        if hostify_inputs:
            resolved_coil_set_spec = _hostify_tree(resolved_coil_set_spec)
            key = self._reference_penalty_cache_key(
                optimize_G,
                weight_inv_modB,
                resolved_constraint_weight,
                resolved_coil_set_spec,
                boozer_reduction_mode=boozer_reduction_mode,
            )
            objective_fn = self._reference_penalty_objective_cache.get(key)
            if objective_fn is None:
                objective_fn = _make_boozer_penalty_objective_closure(
                    coil_arrays=coil_arrays,
                    coil_set_spec=resolved_coil_set_spec,
                    quadpoints_phi=_hostify_tree(self.quadpoints_phi),
                    quadpoints_theta=_hostify_tree(self.quadpoints_theta),
                    mpol=self.mpol,
                    ntor=self.ntor,
                    nfp=self.nfp,
                    stellsym=self.stellsym,
                    scatter_indices=_hostify_tree(self.scatter_indices),
                    surface_kind=self._surface_geometry_kind,
                    label_quadpoints_phi=_hostify_tree(self.label_quadpoints_phi),
                    label_quadpoints_theta=_hostify_tree(self.label_quadpoints_theta),
                    label_mpol=self.label_mpol,
                    label_ntor=self.label_ntor,
                    label_nfp=self.label_nfp,
                    label_stellsym=self.label_stellsym,
                    label_scatter_indices=_hostify_tree(self.label_scatter_indices),
                    label_surface_kind=self._label_surface_geometry_kind,
                    targetlabel=self.targetlabel,
                    constraint_weight=resolved_constraint_weight,
                    label_type=self.label_type,
                    phi_idx=self.phi_idx,
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                    boozer_reduction_mode=boozer_reduction_mode,
                )
                objective_fn = _optimizer_jax._mark_cacheable_jit_value_and_grad(
                    objective_fn
                )
                self._reference_penalty_objective_cache[key] = objective_fn
            return objective_fn
        return _make_boozer_penalty_objective_closure(
            coil_arrays=coil_arrays,
            coil_set_spec=resolved_coil_set_spec,
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            scatter_indices=self.scatter_indices,
            surface_kind=self._surface_geometry_kind,
            label_quadpoints_phi=self.label_quadpoints_phi,
            label_quadpoints_theta=self.label_quadpoints_theta,
            label_mpol=self.label_mpol,
            label_ntor=self.label_ntor,
            label_nfp=self.label_nfp,
            label_stellsym=self.label_stellsym,
            label_scatter_indices=self.label_scatter_indices,
            label_surface_kind=self._label_surface_geometry_kind,
            targetlabel=self.targetlabel,
            constraint_weight=resolved_constraint_weight,
            label_type=self.label_type,
            phi_idx=self.phi_idx,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
            boozer_reduction_mode=boozer_reduction_mode,
        )

    def _make_penalty_value_and_grad_cpu_ordered_with(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
        coil_set_spec=None,
        coil_arrays=None,
    ):
        """Build the host-SciPy Boozer LS value/gradient parity closure.

        Selects the surface and Biot-Savart parity twins automatically when
        :func:`simsopt.backend.is_parity_mode` returns ``True``
        (``SIMSOPT_BACKEND_MODE`` set to ``jax_cpu_parity`` or
        ``jax_gpu_parity``). Production modes keep the matmul/jacfwd hot
        path. The cache key includes the resolved policy so production and
        parity closures do not collide.
        """
        resolved_coil_set_spec = _hostify_tree(
            _resolved_coil_set_spec(
                self.coil_set_spec,
                coil_arrays=coil_arrays,
                coil_set_spec=coil_set_spec,
            )
        )
        resolved_constraint_weight = self._resolve_constraint_weight(constraint_weight)
        parity_policy = "cpu_ordered" if is_parity_mode() else "production"
        boozer_reduction_mode = (
            "cpu_ordered_value_and_grad_parity_twins"
            if parity_policy == "cpu_ordered"
            else "cpu_ordered_value_and_grad"
        )
        key = self._reference_penalty_cache_key(
            optimize_G,
            weight_inv_modB,
            resolved_constraint_weight,
            resolved_coil_set_spec,
            boozer_reduction_mode=boozer_reduction_mode,
        )
        objective_fn = self._reference_penalty_value_and_grad_cache.get(key)
        if objective_fn is None:
            objective_fn = _make_boozer_penalty_closure(
                _boozer_penalty_value_and_grad_cpu_ordered,
                coil_arrays=coil_arrays,
                coil_set_spec=resolved_coil_set_spec,
                quadpoints_phi=_hostify_tree(self.quadpoints_phi),
                quadpoints_theta=_hostify_tree(self.quadpoints_theta),
                mpol=self.mpol,
                ntor=self.ntor,
                nfp=self.nfp,
                stellsym=self.stellsym,
                scatter_indices=_hostify_tree(self.scatter_indices),
                surface_kind=self._surface_geometry_kind,
                label_quadpoints_phi=_hostify_tree(self.label_quadpoints_phi),
                label_quadpoints_theta=_hostify_tree(self.label_quadpoints_theta),
                label_mpol=self.label_mpol,
                label_ntor=self.label_ntor,
                label_nfp=self.label_nfp,
                label_stellsym=self.label_stellsym,
                label_scatter_indices=_hostify_tree(self.label_scatter_indices),
                label_surface_kind=self._label_surface_geometry_kind,
                targetlabel=self.targetlabel,
                constraint_weight=resolved_constraint_weight,
                label_type=self.label_type,
                phi_idx=self.phi_idx,
                optimize_G=optimize_G,
                weight_inv_modB=weight_inv_modB,
                parity_policy=parity_policy,
            )
            objective_fn = jax.jit(objective_fn)
            self._reference_penalty_value_and_grad_cache[key] = objective_fn
        return objective_fn

    def _make_penalty_residual_with(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
        coil_set_spec=None,
        coil_arrays=None,
        *,
        hostify_inputs=True,
    ):
        """Build the LS residual-vector closure with explicit grouped-field inputs."""
        resolved_coil_set_spec = _resolved_coil_set_spec(
            self.coil_set_spec,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        resolved_constraint_weight = self._resolve_constraint_weight(constraint_weight)
        if hostify_inputs:
            resolved_coil_set_spec = _hostify_tree(resolved_coil_set_spec)
            key = self._reference_penalty_cache_key(
                optimize_G,
                weight_inv_modB,
                resolved_constraint_weight,
                resolved_coil_set_spec,
            )
            residual_fn = self._reference_penalty_residual_cache.get(key)
            if residual_fn is None:
                residual_fn = _make_boozer_penalty_residual_closure(
                    coil_arrays=coil_arrays,
                    coil_set_spec=resolved_coil_set_spec,
                    constraint_weight=resolved_constraint_weight,
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                    **self._traceable_surface_runtime_args(),
                )
                self._reference_penalty_residual_cache[key] = residual_fn
            return residual_fn
        return _make_boozer_penalty_residual_closure(
            coil_arrays=coil_arrays,
            coil_set_spec=resolved_coil_set_spec,
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            scatter_indices=self.scatter_indices,
            surface_kind=self._surface_geometry_kind,
            label_quadpoints_phi=self.label_quadpoints_phi,
            label_quadpoints_theta=self.label_quadpoints_theta,
            label_mpol=self.label_mpol,
            label_ntor=self.label_ntor,
            label_nfp=self.label_nfp,
            label_stellsym=self.label_stellsym,
            label_scatter_indices=self.label_scatter_indices,
            label_surface_kind=self._label_surface_geometry_kind,
            targetlabel=self.targetlabel,
            constraint_weight=resolved_constraint_weight,
            label_type=self.label_type,
            phi_idx=self.phi_idx,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )

    def _make_exact_objective_with(
        self,
        optimize_G,
        weight_inv_modB,
        coil_set_spec=None,
        coil_arrays=None,
        *,
        hostify_inputs=True,
    ):
        """Build the constrained-objective scalar used by exact-constraints Newton."""
        resolved_coil_set_spec = _resolved_coil_set_spec(
            self.coil_set_spec,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        quadpoints_phi = self.quadpoints_phi
        quadpoints_theta = self.quadpoints_theta
        scatter_indices = self.scatter_indices
        if hostify_inputs:
            resolved_coil_set_spec = _hostify_tree(resolved_coil_set_spec)
            quadpoints_phi = _hostify_tree(quadpoints_phi)
            quadpoints_theta = _hostify_tree(quadpoints_theta)
            scatter_indices = _hostify_tree(scatter_indices)

        def objective_fn(x):
            sdofs, iota, G = _split_decision_vector_jax(x, optimize_G=optimize_G)
            if not optimize_G:
                G = compute_G_from_currents(
                    _grouped_coil_currents(
                        coil_arrays=coil_arrays,
                        coil_set_spec=resolved_coil_set_spec,
                    )
                )
            gamma, xphi, xtheta = _surface_geometry_from_dofs(
                sdofs,
                quadpoints_phi,
                quadpoints_theta,
                self.mpol,
                self.ntor,
                self.nfp,
                self.stellsym,
                scatter_indices,
                surface_kind=self._surface_geometry_kind,
            )
            nphi, ntheta = gamma.shape[:2]
            B = _grouped_biot_savart_B_points(
                gamma.reshape(-1, 3),
                coil_arrays=coil_arrays,
                coil_set_spec=resolved_coil_set_spec,
            ).reshape(nphi, ntheta, 3)
            residual = boozer_residual_vector(
                G,
                iota,
                B,
                xphi,
                xtheta,
                weight_inv_modB=weight_inv_modB,
            )
            return _as_jax_float64(0.5) * jnp.sum(jnp.square(residual))

        return objective_fn

    def _make_exact_constraint_vector_with(
        self,
        optimize_G,
        coil_set_spec=None,
        coil_arrays=None,
        *,
        hostify_inputs=True,
    ):
        """Build the exact-constraints vector ``[label-target, z_axis]``."""
        resolved_coil_set_spec = _resolved_coil_set_spec(
            self.coil_set_spec,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        quadpoints_phi = self.quadpoints_phi
        quadpoints_theta = self.quadpoints_theta
        scatter_indices = self.scatter_indices
        label_quadpoints_phi = self.label_quadpoints_phi
        label_quadpoints_theta = self.label_quadpoints_theta
        label_scatter_indices = self.label_scatter_indices
        if hostify_inputs:
            resolved_coil_set_spec = _hostify_tree(resolved_coil_set_spec)
            quadpoints_phi = _hostify_tree(quadpoints_phi)
            quadpoints_theta = _hostify_tree(quadpoints_theta)
            scatter_indices = _hostify_tree(scatter_indices)
            label_quadpoints_phi = _hostify_tree(label_quadpoints_phi)
            label_quadpoints_theta = _hostify_tree(label_quadpoints_theta)
            label_scatter_indices = _hostify_tree(label_scatter_indices)

        def constraint_fn(x):
            sdofs, _iota, _G = _split_decision_vector_jax(x, optimize_G=optimize_G)
            geometry = _geometry_from_surface_dofs(
                sdofs,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
                mpol=self.mpol,
                ntor=self.ntor,
                nfp=self.nfp,
                stellsym=self.stellsym,
                scatter_indices=scatter_indices,
                surface_kind=self._surface_geometry_kind,
            )
            label_geometry = _geometry_from_surface_dofs(
                sdofs,
                quadpoints_phi=label_quadpoints_phi,
                quadpoints_theta=label_quadpoints_theta,
                mpol=self.label_mpol,
                ntor=self.label_ntor,
                nfp=self.label_nfp,
                stellsym=self.label_stellsym,
                scatter_indices=label_scatter_indices,
                surface_kind=self._label_surface_geometry_kind,
            )
            label_value, gamma_axis_z = _compute_label_and_axis_z(
                geometry=geometry,
                label_geometry=label_geometry,
                label_points=_field_points_from_geometry(label_geometry),
                label_type=self.label_type,
                phi_idx=self.phi_idx,
                coil_arrays=coil_arrays,
                coil_set_spec=resolved_coil_set_spec,
            )
            return _concat_jax_float64(
                [label_value - _as_jax_float64(self.targetlabel), gamma_axis_z]
            )

        return constraint_fn

    def _make_exact_constraints_residual_with(
        self,
        optimize_G,
        weight_inv_modB,
        coil_set_spec=None,
        coil_arrays=None,
        *,
        hostify_inputs=True,
    ):
        """Build the KKT residual for the exact-constraints Newton solve."""
        objective_fn = self._make_exact_objective_with(
            optimize_G,
            weight_inv_modB,
            coil_set_spec=coil_set_spec,
            coil_arrays=coil_arrays,
            hostify_inputs=hostify_inputs,
        )
        constraint_fn = self._make_exact_constraint_vector_with(
            optimize_G,
            coil_set_spec=coil_set_spec,
            coil_arrays=coil_arrays,
            hostify_inputs=hostify_inputs,
        )

        def constraint_value_with_aux(current_x):
            value = constraint_fn(current_x)
            return value, value

        def residual_fn(xl):
            x = xl[:-2]
            lm = _as_jax_float64(xl[-2:])
            constraint_jacobian, constraint_value = jax.jacfwd(
                constraint_value_with_aux,
                has_aux=True,
            )(x)
            stationarity = jax.grad(objective_fn)(x) - (constraint_jacobian.T @ lm)
            return _concat_jax_float64(stationarity, constraint_value)

        return residual_fn

    def _run_manual_penalty_least_squares(
        self,
        residual_fn,
        x0,
        *,
        tol,
        maxiter,
    ):
        """Compatibility damped Gauss-Newton loop for ``method='manual'``."""
        residual_and_jacobian = jax.jit(
            lambda x: (residual_fn(x), jax.jacobian(residual_fn)(x))
        )

        x_initial = _as_jax_float64(x0)
        always_true = jnp.all(jnp.equal(x_initial, x_initial))
        scalar_one = always_true.astype(x_initial.dtype)
        half = scalar_one / (scalar_one + scalar_one)
        damping_factor = scalar_one + scalar_one + scalar_one
        lam_initial = scalar_one
        residual, jacobian = residual_and_jacobian(x_initial)
        gradient = jacobian.T @ residual
        normal_matrix = jacobian.T @ jacobian
        norm = jnp.linalg.norm(gradient)
        cost = half * jnp.sum(jnp.square(residual))
        int_one = always_true.astype(jnp.int32)
        nit = int_one - int_one
        all_finite = (
            jnp.all(jnp.isfinite(x_initial))
            & jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(normal_matrix))
        )
        tol_value = scalar_one * tol
        maxiter_value = nit + maxiter

        def cond_fn(state):
            (
                _x,
                _residual,
                _jacobian,
                _gradient,
                _normal_matrix,
                state_norm,
                _cost,
                _lam,
                state_nit,
                _all_finite,
            ) = state
            return (state_nit < maxiter_value) & (state_norm > tol_value)

        def body_fn(state):
            (
                x,
                residual,
                jacobian,
                gradient,
                normal_matrix,
                norm,
                cost,
                lam,
                nit,
                all_finite,
            ) = state
            damping = lam * jnp.diag(jnp.diag(normal_matrix))
            dx = jnp.linalg.solve(normal_matrix + damping, gradient)
            candidate_x = x - dx
            candidate_residual, candidate_jacobian = residual_and_jacobian(candidate_x)
            candidate_gradient = candidate_jacobian.T @ candidate_residual
            candidate_normal_matrix = candidate_jacobian.T @ candidate_jacobian
            candidate_norm = jnp.linalg.norm(candidate_gradient)
            candidate_cost = half * jnp.sum(jnp.square(candidate_residual))
            candidate_is_finite = (
                jnp.all(jnp.isfinite(candidate_x))
                & jnp.all(jnp.isfinite(candidate_residual))
                & jnp.all(jnp.isfinite(candidate_gradient))
                & jnp.all(jnp.isfinite(candidate_normal_matrix))
            )
            accepted = candidate_is_finite & (candidate_cost < cost)
            return (
                jnp.where(accepted, candidate_x, x),
                jnp.where(accepted, candidate_residual, residual),
                jnp.where(accepted, candidate_jacobian, jacobian),
                jnp.where(accepted, candidate_gradient, gradient),
                jnp.where(accepted, candidate_normal_matrix, normal_matrix),
                jnp.where(accepted, candidate_norm, norm),
                jnp.where(accepted, candidate_cost, cost),
                jnp.where(accepted, lam / damping_factor, lam * damping_factor),
                nit + int_one,
                all_finite & candidate_is_finite,
            )

        (
            x,
            residual,
            jacobian,
            gradient,
            normal_matrix,
            norm,
            _cost,
            _lam,
            nit,
            all_finite,
        ) = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (
                x_initial,
                residual,
                jacobian,
                gradient,
                normal_matrix,
                norm,
                cost,
                lam_initial,
                nit,
                all_finite,
            ),
        )

        return {
            "x": x,
            "residual": residual,
            "gradient": gradient,
            "jacobian": normal_matrix,
            "nit": int(_host_scalar(nit, dtype=np.int64)),
            "success": bool(_host_scalar((norm <= tol_value) & all_finite)),
        }

    def _traceable_surface_signature(self):
        """Signature for metadata that becomes a traced constant in JAX closures."""
        return (
            int(self.mpol),
            int(self.ntor),
            int(self.nfp),
            bool(self.stellsym),
            str(self._surface_geometry_kind),
            int(self.label_mpol),
            int(self.label_ntor),
            int(self.label_nfp),
            bool(self.label_stellsym),
            str(self._label_surface_geometry_kind),
            float(self.targetlabel),
            str(self.label_type),
            int(self.phi_idx),
            _traceable_array_signature(self.quadpoints_phi),
            _traceable_array_signature(self.quadpoints_theta),
            _traceable_array_signature(self.scatter_indices),
            _traceable_array_signature(self.label_quadpoints_phi),
            _traceable_array_signature(self.label_quadpoints_theta),
            _traceable_array_signature(self.label_scatter_indices),
        )

    def _traceable_surface_runtime_args(self):
        return {
            "quadpoints_phi": _hostify_tree(self.quadpoints_phi),
            "quadpoints_theta": _hostify_tree(self.quadpoints_theta),
            "mpol": self.mpol,
            "ntor": self.ntor,
            "nfp": self.nfp,
            "stellsym": self.stellsym,
            "scatter_indices": _hostify_tree(self.scatter_indices),
            "surface_kind": self._surface_geometry_kind,
            "label_quadpoints_phi": _hostify_tree(self.label_quadpoints_phi),
            "label_quadpoints_theta": _hostify_tree(self.label_quadpoints_theta),
            "label_mpol": self.label_mpol,
            "label_ntor": self.label_ntor,
            "label_nfp": self.label_nfp,
            "label_stellsym": self.label_stellsym,
            "label_scatter_indices": _hostify_tree(self.label_scatter_indices),
            "label_surface_kind": self._label_surface_geometry_kind,
            "targetlabel": self.targetlabel,
            "label_type": self.label_type,
            "phi_idx": self.phi_idx,
        }

    def _resolve_constraint_weight(self, constraint_weight):
        return (
            self.constraint_weight if constraint_weight is None else constraint_weight
        )

    def _reference_penalty_cache_key(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight,
        coil_set_spec,
        *,
        boozer_reduction_mode="default",
    ):
        return (
            bool(optimize_G),
            bool(weight_inv_modB),
            float(constraint_weight),
            boozer_reduction_mode,
            self._traceable_surface_signature(),
            _runtime_cache_tree_signature(coil_set_spec),
        )

    def _traceable_penalty_cache_key(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
    ):
        return (
            bool(optimize_G),
            bool(weight_inv_modB),
            float(self._resolve_constraint_weight(constraint_weight)),
            self._traceable_surface_signature(),
        )

    def _traceable_exact_cache_key(self, weight_inv_modB, mask_indices):
        return (
            bool(weight_inv_modB),
            self._traceable_surface_signature(),
            _traceable_array_signature(mask_indices),
        )

    def _get_traceable_penalty_objective(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
    ):
        resolved_constraint_weight = self._resolve_constraint_weight(constraint_weight)
        key = self._traceable_penalty_cache_key(
            optimize_G,
            weight_inv_modB,
            resolved_constraint_weight,
        )
        objective_fn = self._traceable_penalty_objective_cache.get(key)
        if objective_fn is None:
            surface_args = self._traceable_surface_runtime_args()

            def objective_fn(x, coil_set_spec):
                return _boozer_penalty_objective(
                    x,
                    coil_set_spec=coil_set_spec,
                    constraint_weight=resolved_constraint_weight,
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                    **surface_args,
                )

            self._traceable_penalty_objective_cache[key] = objective_fn
        return self._traceable_penalty_objective_cache[key]

    def _get_traceable_penalty_residual(
        self,
        optimize_G,
        weight_inv_modB,
        constraint_weight=None,
    ):
        resolved_constraint_weight = self._resolve_constraint_weight(constraint_weight)
        key = self._traceable_penalty_cache_key(
            optimize_G,
            weight_inv_modB,
            resolved_constraint_weight,
        )
        residual_fn = self._traceable_penalty_residual_cache.get(key)
        if residual_fn is None:
            surface_args = self._traceable_surface_runtime_args()

            def residual_fn(x, coil_set_spec):
                return _boozer_penalty_residual_vector(
                    x,
                    coil_set_spec=coil_set_spec,
                    constraint_weight=resolved_constraint_weight,
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                    **surface_args,
                )

            self._traceable_penalty_residual_cache[key] = residual_fn
        return self._traceable_penalty_residual_cache[key]

    def _get_traceable_exact_residual(self, weight_inv_modB):
        mask_indices = self._compute_stellsym_mask_indices()
        key = self._traceable_exact_cache_key(weight_inv_modB, mask_indices)
        residual_fn = self._traceable_exact_residual_cache.get(key)
        if residual_fn is None:
            exact_residual = _select_exact_residual_fn(self.stellsym)
            surface_args = self._traceable_surface_runtime_args()
            host_mask_indices = _hostify_tree(mask_indices)

            def residual_fn(x, coil_set_spec):
                return exact_residual(
                    x,
                    coil_set_spec=coil_set_spec,
                    mask_indices=host_mask_indices,
                    weight_inv_modB=weight_inv_modB,
                    **surface_args,
                )

            self._traceable_exact_residual_cache[key] = residual_fn
        return self._traceable_exact_residual_cache[key]

    def run_code_traceable(
        self,
        coil_source,
        sdofs,
        iota,
        G,
        *,
        materialize_dense_linearization=True,
    ):
        """Trace-safe pure-array inner solve for the ondevice target lane.

        Accepts a preferred immutable ``GroupedCoilSetSpec`` or the legacy
        grouped-array payload plus warm-start state, returns only JAX arrays /
        scalars, and never reads or writes ``self.res``, ``self.surface``, or
        ``self.need_to_run_code``.

        Supported modes:
        - LS Boozer solve on the on-device optimizer lane.
        - Exact Boozer Newton solve (backend-independent).
        """
        weight_inv_modB = self.options["weight_inv_modB"]
        coil_set_spec = grouped_coil_set_spec_from_source(coil_source)

        if self.boozer_type == "exact":
            G_exact = (
                G
                if G is not None
                else compute_G_from_currents(
                    grouped_coil_currents_from_spec(coil_set_spec)
                )
            )
            x0 = _concat_jax_float64(sdofs, [iota, G_exact])
            res_fn = self._get_traceable_exact_residual(weight_inv_modB)
            result = newton_exact_traceable(
                res_fn,
                x0,
                maxiter=self.options["newton_maxiter"],
                tol=self.options["newton_tol"],
                args=(coil_set_spec,),
            )
            jacobian = result["jacobian"]
            jacobian_available = jacobian is not None
            finite = jnp.all(jnp.isfinite(result["x"])) & jnp.all(
                jnp.isfinite(result["residual"])
            )
            if jacobian_available:
                finite = finite & jnp.all(jnp.isfinite(jacobian))
            sdofs_exact, iota_exact, G_exact = self._unpack_decision_vector_jax(
                result["x"],
                True,
            )
            half = _as_runtime_float64(0.5, reference=result["residual"])
            primal_success = result["success"] & finite
            adjoint_linear_solve_available = primal_success
            exact_condition_estimate = _dense_condition_estimate_or_none(jacobian)
            return {
                "x": result["x"],
                "sdofs": sdofs_exact,
                "iota": iota_exact,
                "G": G_exact,
                "fun": half * jnp.mean(jnp.square(result["residual"])),
                "residual": result["residual"],
                "jacobian": jacobian,
                "plu": None,
                "lu_piv": None,
                "nit": result["nit"],
                "success": primal_success,
                "primal_success": primal_success,
                "adjoint_linear_solve_available": adjoint_linear_solve_available,
                "linearization_kind": "exact_jacobian",
                "linear_solve_backend": "operator",
                "dense_linear_solve_factors_available": False,
                "type": "exact",
                "weight_inv_modB": weight_inv_modB,
                **_exact_newton_reporting_fields(result),
                **_none_solve_quality_fields(SOLVE_QUALITY_EXACT_FIELDS),
                "exact_factorization_backend": EXACT_FACTORIZATION_BACKEND,
                "exact_condition_estimate": exact_condition_estimate,
                "exact_newton_linear_residual_rel": result.get(
                    "exact_newton_linear_residual_rel"
                ),
                "exact_refinement_correction_rel": result.get(
                    "exact_refinement_correction_rel"
                ),
            }

        optimize_G = G is not None
        method = self._resolve_optimizer_method(optimize_G=optimize_G)
        if method not in _ONDEVICE_OPTIMIZER_METHODS:
            raise RuntimeError(
                "run_code_traceable() requires optimizer_backend='ondevice' for LS solves."
            )

        x0 = self._pack_decision_vector(iota, G, sdofs=_as_jax_float64(sdofs))
        if method in _ONDEVICE_LEAST_SQUARES_METHODS:
            residual_fn = self._get_traceable_penalty_residual(
                optimize_G,
                weight_inv_modB,
            )
            least_squares_options = self._collect_least_squares_options()
            if method == "lm-minpack-ondevice":
                solver = levenberg_marquardt_minpack_traceable
            elif method == "optimistix-lm-ondevice":
                solver = jax_least_squares_optimistix
            else:
                solver = levenberg_marquardt_traceable
            gtol = least_squares_options.get("gtol")
            if method == "lm-minpack-ondevice" and gtol is None:
                gtol = 1e-8
            ls_state = solver(
                residual_fn,
                x0,
                maxiter=self.options["bfgs_maxiter"],
                tol=self.options["bfgs_tol"],
                ftol=least_squares_options.get("ftol", 1e-8),
                xtol=least_squares_options.get("xtol", 1e-8),
                gtol=gtol,
                materialize_dense_linearization=bool(
                    least_squares_options["materialize_dense_linearization"]
                    and materialize_dense_linearization
                ),
                max_dense_linearization_bytes=least_squares_options[
                    "max_dense_linearization_bytes"
                ],
                args=(coil_set_spec,),
            )
            x_ls = ls_state["x"]
        else:
            ls_obj_fn = self._make_penalty_objective_with(
                optimize_G,
                weight_inv_modB,
                coil_set_spec=coil_set_spec,
                hostify_inputs=False,
            )
            optimizer_options = self._collect_optimizer_options(method=method)

            if method == "bfgs-ondevice":
                ls_state = _optimizer_jax._minimize_bfgs_private(
                    ls_obj_fn,
                    x0,
                    maxiter=self.options["bfgs_maxiter"],
                    gtol=self.options["bfgs_tol"],
                    line_search_maxiter=int(
                        optimizer_options.get("line_search_maxiter", 10)
                    ),
                )
                x_ls = ls_state.x_k
            else:
                ls_state = _optimizer_jax._minimize_lbfgs_private(
                    ls_obj_fn,
                    x0,
                    maxiter=self.options["bfgs_maxiter"],
                    gtol=self.options["bfgs_tol"],
                    maxcor=int(optimizer_options.get("maxcor", 200)),
                    ftol=float(optimizer_options.get("ftol", 0.0)),
                    maxfun=optimizer_options.get("maxfun"),
                    maxls=int(optimizer_options.get("maxls", 20)),
                )
                x_ls = ls_state.x_k

        obj_fn = self._get_traceable_penalty_objective(
            optimize_G,
            weight_inv_modB,
        )

        materialize_traceable_hessian = bool(
            self.options["materialize_dense_linearization"]
            and materialize_dense_linearization
        )
        newton_result = self._run_newton_polish_for_method(
            method,
            obj_fn,
            x_ls,
            maxiter=self.options["newton_maxiter"],
            tol=self.options["newton_tol"],
            stab=self.options["newton_stab"],
            materialize_hessian=materialize_traceable_hessian,
            max_dense_hessian_bytes=self.options["max_dense_linearization_bytes"],
            objective_args=(coil_set_spec,),
        )
        sdofs_out, iota_out, G_out = self._unpack_decision_vector_jax(
            newton_result["x"],
            optimize_G,
            coil_set_spec=coil_set_spec,
        )
        finite = jnp.all(jnp.isfinite(newton_result["x"])) & jnp.all(
            jnp.isfinite(newton_result["grad"])
        )
        hessian = newton_result["hessian"]
        if hessian is not None:
            finite = finite & jnp.all(jnp.isfinite(hessian))
            # Phase 2 (docs/parity_scientific_equivalence_contract_2026-05-09.md
            # §5.3): factor once via lu_factor on the traceable lane so
            # the public PLU triple is derived from the same packed
            # factors that the IFT adjoint consumes. Failed-solve
            # propagation keeps the all-NaN ``(P, L, U)`` contract from
            # ``_traceable_plu_or_dummy`` so silent zero-gradient
            # propagation cannot occur.
            lu, piv = _traceable_lu_piv_or_dummy(
                hessian,
                finite=finite,
            )
            lu_piv = (lu, piv)
            P_shared, L_shared, U_shared = _optimizer_jax._plu_from_lu_piv(lu_piv)
            P_dummy, L_dummy, U_dummy = _traceable_plu_or_dummy(
                hessian,
                finite=finite,
            )
            # Use the shared-factor ``(P, L, U)`` on success; on failure
            # substitute the all-NaN dummy triple so failed-solve
            # consumers cannot treat a partially-finite ``P = I`` as a
            # valid factorization.
            P = jnp.where(finite, P_shared, P_dummy)
            L = jnp.where(finite, L_shared, L_dummy)
            U = jnp.where(finite, U_shared, U_dummy)
            plu = (P, L, U)
        else:
            lu_piv = None
            plu = None
        primal_success = newton_result["success"] & finite
        ls_condition_estimate = _dense_condition_estimate_or_none(
            hessian,
            lu_piv=lu_piv,
        )
        return {
            "x": newton_result["x"],
            "sdofs": sdofs_out,
            "iota": iota_out,
            "G": G_out,
            "fun": newton_result["fun"],
            "grad": newton_result["grad"],
            "hessian": hessian,
            "plu": plu,
            "lu_piv": lu_piv,
            "nit": newton_result["nit"],
            "success": primal_success,
            "primal_success": primal_success,
            "adjoint_linear_solve_available": primal_success,
            "linearization_kind": "hessian",
            "linear_solve_backend": "operator",
            "dense_linear_solve_factors_available": plu is not None,
            "optimizer_method": method,
            "type": "ls",
            "weight_inv_modB": weight_inv_modB,
            **_none_solve_quality_fields(SOLVE_QUALITY_LS_FIELDS),
            "ls_condition_estimate": ls_condition_estimate,
            "hessian_materialized": newton_result.get("hessian_materialized"),
            "dense_hessian_shape": newton_result.get("dense_hessian_shape"),
            "dense_hessian_bytes": newton_result.get("dense_hessian_bytes"),
            "max_dense_hessian_bytes": newton_result.get("max_dense_hessian_bytes"),
            "dense_newton_steps_materialized": newton_result.get(
                "dense_newton_steps_materialized"
            ),
            "dense_newton_steps_message": newton_result.get(
                "dense_newton_steps_message"
            ),
            "newton_iter": newton_result.get("newton_iter"),
            "final_gradient_norm": newton_result.get("final_gradient_norm"),
            "final_gradient_inf_norm": newton_result.get("final_gradient_inf_norm"),
            "iterative_refinement_ran": newton_result.get("iterative_refinement_ran"),
            "final_step_iterative_refinement_ran": newton_result.get(
                "final_step_iterative_refinement_ran"
            ),
            "dense_refinement_ran": newton_result.get("dense_refinement_ran"),
            "final_step_dense_refinement_ran": newton_result.get(
                "final_step_dense_refinement_ran"
            ),
            "failure_category": newton_result.get("failure_category"),
            "failure_stage": newton_result.get("failure_stage"),
            "message": newton_result.get("message"),
        }

    def _compute_residual_vector(
        self,
        sdofs,
        iota,
        G,
        weight_inv_modB,
        constraint_weight=None,
        coil_set_spec=None,
        coil_arrays=None,
    ):
        """Compute unscalarized penalty residual vector at given state.

        Reuses M3's ``boozer_residual_vector`` for the Boozer part,
        appends label and z-constraint residuals.

        Returns a JAX array matching CPU
        ``boozer_penalty_constraints(..., scalarize=False)``.
        """
        coil_set_spec = _resolved_coil_set_spec(
            self.coil_set_spec,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        geometry = _geometry_from_surface_dofs(
            sdofs,
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            scatter_indices=self.scatter_indices,
            surface_kind=self._surface_geometry_kind,
        )
        gamma, xphi, xtheta = geometry.gamma, geometry.xphi, geometry.xtheta
        label_geometry = _geometry_from_surface_dofs(
            sdofs,
            quadpoints_phi=self.label_quadpoints_phi,
            quadpoints_theta=self.label_quadpoints_theta,
            mpol=self.label_mpol,
            ntor=self.label_ntor,
            nfp=self.label_nfp,
            stellsym=self.label_stellsym,
            scatter_indices=self.label_scatter_indices,
            surface_kind=self._label_surface_geometry_kind,
        )
        nphi, ntheta = int(gamma.shape[0]), int(gamma.shape[1])
        points = gamma.reshape(-1, 3)
        B = _grouped_biot_savart_B_points(
            points,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        ).reshape(nphi, ntheta, 3)

        r_boozer_raw = boozer_residual_vector(
            G,
            iota,
            B,
            xphi,
            xtheta,
            weight_inv_modB=weight_inv_modB,
        )
        num_res = _as_jax_float64(3 * nphi * ntheta)
        r_boozer = r_boozer_raw / jnp.sqrt(num_res)

        constraint_weight = (
            self.constraint_weight if constraint_weight is None else constraint_weight
        )
        constraint_weight = constraint_weight if constraint_weight is not None else 1.0
        constraint_weight = _as_jax_float64(constraint_weight)
        label_value, gamma_axis_z = _compute_label_and_axis_z(
            geometry=geometry,
            label_geometry=label_geometry,
            label_points=_field_points_from_geometry(label_geometry),
            label_type=self.label_type,
            phi_idx=self.phi_idx,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        weight_sqrt = jnp.sqrt(constraint_weight)
        rl = weight_sqrt * (label_value - _as_jax_float64(self.targetlabel))
        rz = weight_sqrt * gamma_axis_z

        return _concat_jax_float64(r_boozer, [rl, rz])

    def _resolve_optimizer_method(self, limited_memory=None, *, optimize_G=True):
        """Resolve optimizer method string from options."""
        optimizer_backend = _optimizer_jax.resolve_optimizer_backend(
            self.options["optimizer_backend"]
        )
        if optimizer_backend not in VALID_OPTIMIZER_BACKENDS:
            raise ValueError("optimizer_backend must be one of: auto, scipy, ondevice.")
        require_target_backend_x64(optimizer_backend)
        if optimizer_backend != "ondevice":
            backend_config = get_backend_config()
            policy_optimizer_backend = get_backend_policy().default_optimizer_backend
            if (
                backend_config.backend == "jax"
                and optimizer_backend != policy_optimizer_backend
            ):
                raise RuntimeError(
                    "BoozerSurfaceJAX cannot use "
                    f"optimizer_backend={optimizer_backend!r} on the LS "
                    "reference solver lane while simsopt backend mode "
                    f"{backend_config.mode!r} requires "
                    f"optimizer_backend={policy_optimizer_backend!r}. Select "
                    f"optimizer_backend={policy_optimizer_backend!r} "
                    "or switch to the native_cpu reference backend."
                )
            raise_if_strict_jax_fallback(
                component="BoozerSurfaceJAX",
                detail=(
                    f"optimizer_backend={optimizer_backend!r} on the LS "
                    "reference solver lane"
                ),
            )
            warn_if_jax_fallback(
                component="BoozerSurfaceJAX",
                detail=(
                    f"optimizer_backend={optimizer_backend!r} on the LS "
                    "reference solver lane"
                ),
            )
        if limited_memory is None:
            limited_memory = self.options["limited_memory"]
        effective_limited_memory = bool(limited_memory)
        if optimizer_backend == "ondevice" and self.options.get(
            "force_ondevice_limited_memory", False
        ):
            effective_limited_memory = True
        least_squares_algorithm = self.options["least_squares_algorithm"]
        if (
            optimizer_backend == "ondevice"
            and least_squares_algorithm == "lm"
            and not optimize_G
        ):
            # The explicit-G full-state path is the on-device LM target lane.
            # The reduced fixed-G compatibility path remains more reliable on
            # the historical quasi-Newton formulation.
            least_squares_algorithm = "quasi-newton"
        if optimizer_backend == "ondevice":
            return resolve_target_least_squares_optimizer_method(
                limited_memory=effective_limited_memory,
                least_squares_algorithm=least_squares_algorithm,
            )
        return resolve_reference_least_squares_optimizer_method(
            limited_memory=effective_limited_memory,
            least_squares_algorithm=least_squares_algorithm,
        )

    def _collect_optimizer_options(self, *, method):
        """Gather optimizer-specific options from self.options."""
        optimizer_options = {
            k: self.options[k]
            for k in (
                "line_search_maxiter",
                "maxcor",
                "ftol",
                "maxfun",
                "maxls",
            )
            if k in self.options
        }
        if method in {"lbfgs", "lbfgs-ondevice"} and "maxcor" not in optimizer_options:
            optimizer_options["maxcor"] = 200
        if method not in _ONDEVICE_OPTIMIZER_METHODS:
            optimizer_options.update(
                {k: self.options[k] for k in _SCIPY_TRACE_OPTIONS if k in self.options}
            )
        return optimizer_options

    def _collect_least_squares_options(self):
        """Gather LM options from self.options."""
        options = {
            "materialize_dense_linearization": bool(
                self.options["materialize_dense_linearization"]
            ),
            "max_dense_linearization_bytes": self.options[
                "max_dense_linearization_bytes"
            ],
        }
        options.update(
            {k: self.options[k] for k in _LM_TUNING_OPTION_KEYS if k in self.options}
        )
        return options

    def _run_newton_polish_for_method(
        self,
        method,
        obj_fn,
        x0,
        *,
        maxiter,
        tol,
        stab,
        materialize_hessian,
        max_dense_hessian_bytes,
        progress_callback=None,
        objective_args=(),
    ):
        """Run the Newton polish implementation for a resolved optimizer method."""
        if method in _ONDEVICE_OPTIMIZER_METHODS:
            return newton_polish_traceable(
                obj_fn,
                x0,
                maxiter=maxiter,
                tol=tol,
                stab=stab,
                materialize_hessian=materialize_hessian,
                max_dense_hessian_bytes=max_dense_hessian_bytes,
                progress_callback=progress_callback,
                args=objective_args,
            )
        if objective_args:
            raise ValueError(
                "Newton objective args are only supported on the ondevice traceable path."
            )
        return newton_polish(
            obj_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            stab=stab,
            materialize_hessian=materialize_hessian,
            max_dense_hessian_bytes=max_dense_hessian_bytes,
            dense_newton_steps=materialize_hessian,
            progress_callback=progress_callback,
        )

    def minimize_boozer_penalty_constraints_LBFGS(
        self,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        tol=None,
        maxiter=None,
        verbose=None,
        limited_memory=None,
        weight_inv_modB=None,
    ):
        """Least-squares first stage of the LS solve.

        Accepts the CPU public argument shape, but when ``limited_memory`` is
        omitted it preserves the configured BoozerSurfaceJAX options default.
        """
        if not self.need_to_run_code:
            return self.res
        tol = tol if tol is not None else self.options["bfgs_tol"]
        maxiter = maxiter if maxiter is not None else self.options["bfgs_maxiter"]
        verbose = verbose if verbose is not None else self.options["verbose"]
        weight_inv_modB = (
            weight_inv_modB
            if weight_inv_modB is not None
            else self.options["weight_inv_modB"]
        )

        optimize_G = G is not None
        s = self.surface
        x0 = self._pack_decision_vector(iota, G)
        method = self._resolve_optimizer_method(
            limited_memory=limited_memory,
            optimize_G=optimize_G,
        )
        progress_callback = self._make_solver_progress_callback(method)
        if method in _LEAST_SQUARES_METHODS:
            residual_fn = self._make_penalty_residual_with(
                optimize_G,
                weight_inv_modB,
                constraint_weight,
            )
            least_squares_runner = (
                target_least_squares
                if method.endswith("-ondevice")
                else reference_least_squares
            )
            result = least_squares_runner(
                residual_fn,
                x0,
                method=method,
                tol=tol,
                maxiter=maxiter,
                options=self._collect_least_squares_options(),
                progress_callback=progress_callback,
            )
        else:
            optimizer_options = self._collect_optimizer_options(method=method)
            if method in {"bfgs", "lbfgs"}:
                obj_fn = self._make_penalty_value_and_grad_cpu_ordered_with(
                    optimize_G,
                    weight_inv_modB,
                    constraint_weight,
                )
                result = reference_minimize(
                    obj_fn,
                    x0,
                    method=method,
                    tol=tol,
                    maxiter=maxiter,
                    options=optimizer_options,
                    value_and_grad=True,
                    progress_callback=progress_callback,
                )
            else:
                obj_fn = self._make_penalty_objective_with(
                    optimize_G,
                    weight_inv_modB,
                    constraint_weight,
                )
                minimize_runner = (
                    target_minimize
                    if method.endswith("-ondevice")
                    else reference_minimize
                )
                result = minimize_runner(
                    obj_fn,
                    x0,
                    method=method,
                    tol=tol,
                    maxiter=maxiter,
                    options=optimizer_options,
                    progress_callback=progress_callback,
                )

        sdofs_final, iota_out, G_out = self._unpack_penalty_optimizer_state(
            result.x, optimize_G
        )
        self._set_surface_dofs(sdofs_final)

        gradient = _host_numpy(
            _boozer_penalty_optimizer_state_to_vector(
                result.jac,
                optimize_G=optimize_G,
            )
        )

        resdict = {
            "fun": float(_host_scalar(result.fun)),
            "gradient": gradient,
            "iter": int(_host_scalar(result.nit)),
            "info": result,
            "success": bool(_host_scalar(result.success)),
            "primal_success": bool(_host_scalar(result.success)),
            "adjoint_linear_solve_available": False,
            "sdofs": _as_jax_float64(sdofs_final),
            "G": G_out,
            "s": s,
            "iota": iota_out,
            "optimizer_method": method,
            "scipy_call_contract": getattr(result, "scipy_call_contract", None),
            "scipy_initial_call": getattr(result, "scipy_initial_call", None),
            "scipy_callback_trace": getattr(result, "scipy_callback_trace", None),
            "weight_inv_modB": weight_inv_modB,
            "type": "ls",
            **_none_solve_quality_fields(SOLVE_QUALITY_LS_FIELDS),
        }
        self.res = resdict
        self.need_to_run_code = False

        if verbose:
            print(
                f"{method} solve - "
                f"success={resdict['success']}  iter={resdict['iter']}, "
                f"iota={iota_out:.16f}, ||grad||_inf="
                f"{_host_inf_norm(resdict['gradient']):.3e}",
                flush=True,
            )
        return resdict

    def minimize_boozer_penalty_constraints_newton(
        self,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        tol=None,
        maxiter=None,
        stab=0.0,
        verbose=None,
        weight_inv_modB=None,
    ):
        """Newton polish stage of the LS solve."""
        if not self.need_to_run_code:
            return self.res
        tol = tol if tol is not None else self.options["newton_tol"]
        maxiter = maxiter if maxiter is not None else self.options["newton_maxiter"]
        verbose = verbose if verbose is not None else self.options["verbose"]
        weight_inv_modB = (
            weight_inv_modB
            if weight_inv_modB is not None
            else self.options["weight_inv_modB"]
        )

        optimize_G = G is not None
        G_provided = optimize_G
        s = self.surface
        x0 = self._pack_decision_vector(iota, G)
        obj_fn = self._make_penalty_objective_with(
            optimize_G, weight_inv_modB, constraint_weight
        )

        method = self._resolve_optimizer_method(optimize_G=optimize_G)
        result = self._run_newton_polish_for_method(
            method,
            obj_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            stab=stab,
            materialize_hessian=bool(self.options["materialize_dense_linearization"]),
            max_dense_hessian_bytes=self.options["max_dense_linearization_bytes"],
            progress_callback=self._resolve_newton_progress_callback(method),
        )

        sdofs_final, iota_out, G_out = self._unpack_decision_vector(
            result["x"], optimize_G
        )

        if (
            not _host_all_finite(result["x"])
            or not _host_all_finite(result["grad"])
            or (
                result["hessian"] is not None
                and not _host_all_finite(result["hessian"])
            )
        ):
            solve_generation = _advance_solver_generation(self)
            res = {
                "residual": None,
                "jacobian": None,
                "hessian": None,
                "iter": int(_host_scalar(result["nit"], dtype=np.int64)),
                "success": False,
                "sdofs": _as_jax_float64(sdofs_final),
                "G": G_out,
                "s": s,
                "iota": iota_out,
                "PLU": None,
                "LU_PIV": None,
                "vjp": None,
                "vjp_groups": None,
                "type": "ls",
                "optimizer_method": method,
                "solve_generation": solve_generation,
                "weight_inv_modB": weight_inv_modB,
                "fun": float(_host_scalar(result["fun"])),
                "primal_success": False,
                "adjoint_linear_solve_available": False,
                "linearization_kind": "hessian",
                "linear_solve_backend": "operator",
                "dense_linear_solve_factors_available": False,
                "hessian_materialized": result.get("hessian_materialized"),
                "dense_hessian_shape": result.get("dense_hessian_shape"),
                "dense_hessian_bytes": result.get("dense_hessian_bytes"),
                "max_dense_hessian_bytes": result.get("max_dense_hessian_bytes"),
                "dense_newton_steps_materialized": result.get(
                    "dense_newton_steps_materialized"
                ),
                "dense_newton_steps_message": result.get("dense_newton_steps_message"),
                "newton_iter": result.get("newton_iter"),
                "final_gradient_norm": result.get("final_gradient_norm"),
                "final_gradient_inf_norm": result.get("final_gradient_inf_norm"),
                "iterative_refinement_ran": result.get("iterative_refinement_ran"),
                "final_step_iterative_refinement_ran": result.get(
                    "final_step_iterative_refinement_ran"
                ),
                "dense_refinement_ran": result.get("dense_refinement_ran"),
                "final_step_dense_refinement_ran": result.get(
                    "final_step_dense_refinement_ran"
                ),
                "failure_category": result.get("failure_category"),
                "failure_stage": result.get("failure_stage"),
                "message": result.get("message"),
                **_none_solve_quality_fields(SOLVE_QUALITY_LS_FIELDS),
            }
            self.res = res
            self.need_to_run_code = False
            return res

        self._set_surface_dofs(sdofs_final)
        H = result["hessian"]
        # Phase 2 (docs/parity_scientific_equivalence_contract_2026-05-09.md
        # §5.3): factor once via lu_factor when the dense factor fits in
        # the byte budget so the LS forward and adjoint solves share the
        # same packed (lu, piv) bytes by construction. The public PLU
        # triple is derived from the same factorization.
        shared_dispatch_eligible = _ls_factor_once_dispatch_eligible(
            H,
            max_dense_jacobian_bytes=self.options["max_dense_jacobian_bytes"],
        )
        if H is not None and shared_dispatch_eligible:
            lu_piv = _optimizer_jax._factor_dense_hessian(
                H,
                optimizer_backend=self.options["optimizer_backend"],
            )
            P, L, U = _optimizer_jax._plu_from_lu_piv(lu_piv)
            plu = (P, L, U)
        elif H is not None:
            lu_piv = None
            if self.options["optimizer_backend"] == "scipy":
                P, L, U = scipy.linalg.lu(np.asarray(H, dtype=np.float64))
                plu = tuple(jnp.asarray(factor, dtype=H.dtype) for factor in (P, L, U))
            else:
                P, L, U = jax.scipy.linalg.lu(H)
                plu = (P, L, U)
        else:
            lu_piv = None
            plu = None
        linearization_residency = self.options["linearization_residency"]
        lu_piv = _place_linearization_factors_for_residency(
            lu_piv,
            linearization_residency,
        )
        plu = _place_linearization_factors_for_residency(
            plu,
            linearization_residency,
        )
        shared_lu_piv_dispatch = _ls_shared_lu_piv_dispatch(
            self.options["optimizer_backend"],
            lu_piv,
        )
        ls_hessian_symmetry_rel = _ls_hessian_symmetry_rel(H)
        ls_condition_estimate = _dense_condition_estimate_or_none(
            H,
            lu_piv=lu_piv,
        )
        ls_factorization_backend = _ls_factorization_backend(
            H if plu is not None else None,
            optimizer_backend=self.options["optimizer_backend"],
            shared_dispatch=shared_lu_piv_dispatch,
        )

        G_for_res = (
            G_out
            if G_out is not None
            else float(compute_G_from_currents(self.coil_currents))
        )
        residual_vec = self._compute_residual_vector(
            sdofs_final,
            iota_out,
            G_for_res,
            weight_inv_modB=weight_inv_modB,
        )
        solve_generation = _advance_solver_generation(self)
        vjp_callback = _prepare_result_callback(
            partial(_boozer_ls_coil_vjp, weight_inv_modB=weight_inv_modB),
            booz_surf=self,
            solve_generation=solve_generation,
            callback_name="vjp",
            G_provided=G_provided,
            freshness_guard=True,
        )
        vjp_groups_callback = _prepare_result_callback(
            _build_ls_group_vjp_callback(
                self,
                iota_out,
                G_out,
                solve_generation=solve_generation,
                weight_inv_modB=weight_inv_modB,
            ),
            booz_surf=self,
            solve_generation=solve_generation,
            callback_name="vjp_groups",
            G_provided=G_provided,
            freshness_guard=True,
        )

        res = {
            "residual": residual_vec,
            "jacobian": result["grad"],
            "hessian": H,
            "iter": int(_host_scalar(result["nit"], dtype=np.int64)),
            "success": bool(_host_scalar(result["success"])),
            "primal_success": bool(_host_scalar(result["success"])),
            "adjoint_linear_solve_available": bool(_host_scalar(result["success"])),
            "sdofs": _as_jax_float64(sdofs_final),
            "G": G_out,
            "s": s,
            "iota": iota_out,
            "PLU": plu,
            "LU_PIV": lu_piv,
            "vjp": vjp_callback,
            "vjp_groups": vjp_groups_callback,
            "type": "ls",
            "optimizer_method": method,
            "linearization_kind": "hessian",
            "linear_solve_backend": _ls_linear_solve_backend(
                optimizer_backend=self.options["optimizer_backend"],
                plu_available=plu is not None,
                shared_lu_piv_dispatch=shared_lu_piv_dispatch,
            ),
            "dense_linear_solve_factors_available": plu is not None,
            "linearization_residency": linearization_residency,
            "solve_generation": solve_generation,
            "weight_inv_modB": weight_inv_modB,
            "fun": float(_host_scalar(result["fun"])),
            "hessian_materialized": result.get("hessian_materialized"),
            "dense_hessian_shape": result.get("dense_hessian_shape"),
            "dense_hessian_bytes": result.get("dense_hessian_bytes"),
            "max_dense_hessian_bytes": result.get("max_dense_hessian_bytes"),
            "dense_newton_steps_materialized": result.get(
                "dense_newton_steps_materialized"
            ),
            "dense_newton_steps_message": result.get("dense_newton_steps_message"),
            "newton_iter": result.get("newton_iter"),
            "final_gradient_norm": result.get("final_gradient_norm"),
            "final_gradient_inf_norm": result.get("final_gradient_inf_norm"),
            "iterative_refinement_ran": result.get("iterative_refinement_ran"),
            "final_step_iterative_refinement_ran": result.get(
                "final_step_iterative_refinement_ran"
            ),
            "dense_refinement_ran": result.get("dense_refinement_ran"),
            "final_step_dense_refinement_ran": result.get(
                "final_step_dense_refinement_ran"
            ),
            "failure_category": result.get("failure_category"),
            "failure_stage": result.get("failure_stage"),
            "message": result.get("message"),
            # Scientific-equivalence ladder reporting fields per
            # docs/parity_scientific_equivalence_contract_2026-05-09.md §3.1.
            # action_max / step_abs_diff are populated by the parity
            # arbiter; condition_estimate is populated when dense H exists.
            **_none_solve_quality_fields(SOLVE_QUALITY_LS_FIELDS),
            "ls_hessian_symmetry_rel": ls_hessian_symmetry_rel,
            "ls_factorization_backend": ls_factorization_backend,
            "ls_condition_estimate": ls_condition_estimate,
        }
        self.res = res
        self.need_to_run_code = False

        if verbose:
            grad_norm = float(np.linalg.norm(_host_numpy(res["jacobian"])))
            print(
                f"NEWTON solve - success={res['success']}  "
                f"iter={res['iter']}, iota={iota_out:.16f}, "
                f"||grad||={grad_norm:.3e}",
                flush=True,
            )
        return res

    def minimize_boozer_penalty_constraints_ls(
        self,
        tol=1e-12,
        maxiter=10,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        method="lm",
        weight_inv_modB=True,
    ):
        """Public LS solver matching the baseline BoozerSurface API."""
        if not self.need_to_run_code:
            return self.res

        optimize_G = G is not None
        s = self.surface
        x0 = self._pack_decision_vector(iota, G)
        # Reuse the centralized backend/algorithm validation seam even though
        # this public method forces an LM/manual route rather than the
        # constructor-wide least_squares_algorithm policy.
        self._resolve_optimizer_method(
            limited_memory=False,
            optimize_G=optimize_G,
        )

        if method == "manual":
            residual_fn = self._make_penalty_residual_with(
                optimize_G,
                weight_inv_modB,
                constraint_weight,
                hostify_inputs=self.options["optimizer_backend"] != "ondevice",
            )
            result = self._run_manual_penalty_least_squares(
                residual_fn,
                x0,
                tol=tol,
                maxiter=maxiter,
            )
            sdofs_final, iota_out, G_out = self._unpack_decision_vector(
                result["x"],
                optimize_G,
            )
            self._set_surface_dofs(sdofs_final)
            resdict = {
                "residual": result["residual"],
                "gradient": result["gradient"],
                "jacobian": result["jacobian"],
                "success": result["success"],
                "primal_success": result["success"],
                "adjoint_linear_solve_available": False,
                "sdofs": _as_jax_float64(sdofs_final),
                "G": G_out,
                "s": s,
                "iota": iota_out,
                "type": "ls",
                "weight_inv_modB": weight_inv_modB,
                "optimizer_method": "manual",
                **_none_solve_quality_fields(SOLVE_QUALITY_LS_FIELDS),
            }
            self.res = resdict
            self.need_to_run_code = False
            return resdict

        if method != "lm":
            raise ValueError(
                "BoozerSurfaceJAX minimize_boozer_penalty_constraints_ls() "
                "supports method='lm' or method='manual'."
            )

        residual_fn = self._make_penalty_residual_with(
            optimize_G,
            weight_inv_modB,
            constraint_weight,
            hostify_inputs=self.options["optimizer_backend"] != "ondevice",
        )
        if self.options["optimizer_backend"] == "ondevice":
            resolved_method = self._resolve_optimizer_method(
                limited_memory=False,
                optimize_G=optimize_G,
            )
            optimizer_method = (
                resolved_method
                if resolved_method in _ONDEVICE_LEAST_SQUARES_METHODS
                else "lm-ondevice"
            )
            result = target_least_squares(
                residual_fn,
                x0,
                method=optimizer_method,
                tol=tol,
                maxiter=maxiter,
                options=self._collect_least_squares_options(),
            )
        else:
            result = reference_least_squares(
                residual_fn,
                x0,
                method="lm",
                tol=tol,
                maxiter=maxiter,
                options=self._collect_least_squares_options(),
            )
            optimizer_method = "lm"

        sdofs_final, iota_out, G_out = self._unpack_decision_vector(
            result.x, optimize_G
        )
        self._set_surface_dofs(sdofs_final)
        resdict = {
            "info": result,
            "residual": result.residual,
            "gradient": result.jac,
            "jacobian": result.residual_jacobian,
            "success": bool(_host_scalar(result.success)),
            "primal_success": bool(_host_scalar(result.success)),
            "adjoint_linear_solve_available": False,
            "sdofs": _as_jax_float64(sdofs_final),
            "G": G_out,
            "s": s,
            "iota": iota_out,
            "type": "ls",
            "weight_inv_modB": weight_inv_modB,
            "optimizer_method": optimizer_method,
            **_none_solve_quality_fields(SOLVE_QUALITY_LS_FIELDS),
        }
        self.res = resdict
        self.need_to_run_code = False
        return resdict

    def _make_exact_residual_with(
        self,
        mask_indices,
        coil_arrays=None,
        coil_set_spec=None,
        *,
        hostify_inputs=True,
    ):
        """Build the exact residual function with explicit grouped-field inputs."""
        residual_fn = _select_exact_residual_fn(self.stellsym)
        resolved_coil_set_spec = _resolved_coil_set_spec(
            self.coil_set_spec,
            coil_arrays=coil_arrays,
            coil_set_spec=coil_set_spec,
        )
        if hostify_inputs:
            resolved_coil_set_spec = _hostify_tree(resolved_coil_set_spec)
        return partial(
            residual_fn,
            coil_arrays=coil_arrays,
            coil_set_spec=resolved_coil_set_spec,
            quadpoints_phi=(
                _hostify_tree(self.quadpoints_phi)
                if hostify_inputs
                else self.quadpoints_phi
            ),
            quadpoints_theta=(
                _hostify_tree(self.quadpoints_theta)
                if hostify_inputs
                else self.quadpoints_theta
            ),
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            scatter_indices=(
                _hostify_tree(self.scatter_indices)
                if hostify_inputs
                else self.scatter_indices
            ),
            surface_kind=self._surface_geometry_kind,
            label_quadpoints_phi=(
                _hostify_tree(self.label_quadpoints_phi)
                if hostify_inputs
                else self.label_quadpoints_phi
            ),
            label_quadpoints_theta=(
                _hostify_tree(self.label_quadpoints_theta)
                if hostify_inputs
                else self.label_quadpoints_theta
            ),
            label_mpol=self.label_mpol,
            label_ntor=self.label_ntor,
            label_nfp=self.label_nfp,
            label_stellsym=self.label_stellsym,
            label_scatter_indices=(
                _hostify_tree(self.label_scatter_indices)
                if hostify_inputs
                else self.label_scatter_indices
            ),
            label_surface_kind=self._label_surface_geometry_kind,
            targetlabel=self.targetlabel,
            label_type=self.label_type,
            phi_idx=self.phi_idx,
            mask_indices=mask_indices,
            weight_inv_modB=self.options["weight_inv_modB"],
        )

    def _make_exact_residual(self, mask_indices):
        """Build the JIT-compiled exact residual function."""
        return self._make_exact_residual_with(mask_indices)

    def _compute_stellsym_mask_indices(self):
        """Compute and cache the integer exact-residual mask indices."""
        if self._exact_mask_indices is None:
            self._exact_mask_indices = compute_stellsym_mask_indices_for_grid(
                mpol=self.mpol,
                ntor=self.ntor,
                nfp=self.nfp,
                stellsym=self.stellsym,
                quadpoints_phi=self.quadpoints_phi,
                quadpoints_theta=self.quadpoints_theta,
            )
        return self._exact_mask_indices

    def solve_residual_equation_exactly_newton(
        self,
        tol=None,
        maxiter=None,
        iota=0.0,
        G=None,
        verbose=None,
    ):
        """Solve the Boozer residual system exactly via Newton's method.

        Public API matching CPU ``BoozerSurface.solve_residual_equation_exactly_newton()``.

        Args:
            tol: residual norm tolerance. Defaults to options['newton_tol'].
            maxiter: maximum Newton iterations. Defaults to options['newton_maxiter'].
            iota: initial guess for rotational transform.
            G: initial guess for G (None → compute from coil currents).
            verbose: print convergence info.

        Returns:
            dict with 'residual', 'fun', 'jacobian', 'iter', 'success', 'G',
            's', 'iota', 'PLU', 'mask', 'type', 'vjp', 'weight_inv_modB',
            'message', 'failure_category', 'failure_stage',
            'jacobian_materialized',
            'dense_jacobian_shape', 'dense_jacobian_bytes',
            'max_dense_jacobian_bytes', and Exact solve-quality reporting
            fields from ``SOLVE_QUALITY_EXACT_FIELDS``.
            Exact mode enforces options['max_dense_jacobian_bytes'] before the
            final dense Jacobian/PLU materialization step.
        """
        if not self.need_to_run_code:
            return self.res

        s = self.surface
        G_provided = G is not None
        if not _is_exact_surface_xyz_tensor_fourier(self._surface_geometry_kind):
            raise RuntimeError(
                "Exact solution of Boozer Surfaces only supported for "
                "SurfaceXYZTensorFourier"
            )

        tol = tol if tol is not None else self.options["newton_tol"]
        maxiter = maxiter if maxiter is not None else self.options["newton_maxiter"]
        verbose = verbose if verbose is not None else self.options["verbose"]

        if G is None:
            G = float(compute_G_from_currents(self.coil_currents))

        sdofs = self._get_surface_dofs()
        x0 = _concat_jax_float64(sdofs, [iota, G])

        mask_indices = self._compute_stellsym_mask_indices()
        res_fn = self._make_exact_residual(mask_indices)

        result = newton_exact(
            res_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            max_dense_jacobian_bytes=self.options["max_dense_jacobian_bytes"],
        )

        x_final = result["x"]
        exact_residual = res_fn(x_final)
        sdofs_final, iota_final_jax, G_final_jax = _split_decision_vector_jax(
            x_final,
            optimize_G=True,
        )
        iota_final = float(_host_scalar(iota_final_jax))
        G_final = float(_host_scalar(G_final_jax))
        jacobian = result["jacobian"]
        jacobian_available = jacobian is not None
        exact_reporting = _exact_newton_reporting_fields(result)
        materialization_message = exact_reporting["message"]
        primal_success = (
            bool(_host_scalar(result["success"]))
            and exact_reporting["failure_category"] is None
        )

        if (
            not primal_success
            or not _host_all_finite(x_final)
            or not _host_all_finite(exact_residual)
            or (jacobian_available and not _host_all_finite(jacobian))
        ):
            solve_generation = _advance_solver_generation(self)
            res = {
                "residual": None,
                "fun": float(0.5 * np.mean(np.square(_host_numpy(exact_residual)))),
                "jacobian": None,
                "iter": int(_host_scalar(result["nit"], dtype=np.int64)),
                "success": False,
                "G": G_final,
                "sdofs": _as_jax_float64(sdofs_final),
                "s": s,
                "iota": iota_final,
                "PLU": None,
                "mask": None,
                "type": "exact",
                "vjp": None,
                "vjp_groups": None,
                "solve_generation": solve_generation,
                "weight_inv_modB": self.options["weight_inv_modB"],
                "primal_success": False,
                "adjoint_linear_solve_available": False,
                "linearization_kind": "exact_jacobian",
                "linear_solve_backend": "operator",
                "dense_linear_solve_factors_available": False,
                "linearization_residency": self.options["linearization_residency"],
                **exact_reporting,
                **_none_solve_quality_fields(SOLVE_QUALITY_EXACT_FIELDS),
                "exact_factorization_backend": EXACT_FACTORIZATION_BACKEND,
            }
            self.res = res
            self.need_to_run_code = False
            if verbose and materialization_message is not None:
                print(materialization_message, flush=True)
            return res

        self._set_surface_dofs(sdofs_final)
        J = jacobian
        if jacobian_available:
            P, L, U = jax.scipy.linalg.lu(J)
            plu = _place_linearization_factors_for_residency(
                (P, L, U),
                self.options["linearization_residency"],
            )
        else:
            plu = None
        exact_condition_estimate = (
            _dense_condition_estimate_or_none(J) if verbose else None
        )

        nphi = len(self.quadpoints_phi)
        ntheta = len(self.quadpoints_theta)

        # Reconstruct raw (unmasked) Boozer residual for CPU-contract parity.
        gamma_final, xphi_final, xtheta_final = _surface_geometry_from_dofs(
            sdofs_final,
            self.quadpoints_phi,
            self.quadpoints_theta,
            self.mpol,
            self.ntor,
            self.nfp,
            self.stellsym,
            self.scatter_indices,
            surface_kind=self._surface_geometry_kind,
        )
        B_final = grouped_biot_savart_B_from_spec(
            gamma_final.reshape(-1, 3),
            self.coil_set_spec,
        ).reshape(nphi, ntheta, 3)
        r_raw = boozer_residual_vector(
            G_final,
            iota_final,
            B_final,
            xphi_final,
            xtheta_final,
            weight_inv_modB=self.options["weight_inv_modB"],
        )

        bool_mask = np.zeros(3 * nphi * ntheta, dtype=bool)
        bool_mask[np.asarray(mask_indices)] = True
        solve_generation = _advance_solver_generation(self)
        vjp_callback = _prepare_result_callback(
            _boozer_exact_coil_vjp,
            booz_surf=self,
            solve_generation=solve_generation,
            callback_name="vjp",
            G_provided=G_provided,
            freshness_guard=True,
        )
        vjp_groups_callback = _prepare_result_callback(
            _build_exact_group_vjp_callback(
                self,
                iota_final,
                G_final,
            ),
            booz_surf=self,
            solve_generation=solve_generation,
            callback_name="vjp_groups",
            G_provided=G_provided,
            freshness_guard=True,
        )

        res = {
            "residual": r_raw,
            "fun": float(0.5 * np.mean(np.square(_host_numpy(exact_residual)))),
            "jacobian": J,
            "iter": int(_host_scalar(result["nit"], dtype=np.int64)),
            "success": primal_success,
            "primal_success": primal_success,
            "adjoint_linear_solve_available": primal_success,
            "sdofs": _as_jax_float64(sdofs_final),
            "G": G_final,
            "s": s,
            "iota": iota_final,
            "PLU": plu,
            "mask": bool_mask,
            "type": "exact",
            "vjp": vjp_callback,
            "vjp_groups": vjp_groups_callback,
            "linearization_kind": "exact_jacobian",
            "linear_solve_backend": "operator",
            "dense_linear_solve_factors_available": plu is not None,
            "linearization_residency": self.options["linearization_residency"],
            "solve_generation": solve_generation,
            "weight_inv_modB": self.options["weight_inv_modB"],
            **exact_reporting,
            # Scientific-equivalence ladder reporting fields per
            # docs/parity_scientific_equivalence_contract_2026-05-09.md §3.2.
            # action_max / linear_residual / refinement_correction /
            # adjoint_solve_residual are populated by the parity arbiter or
            # downstream optimizer plumbing; condition_estimate is populated
            # when dense J exists.
            **_none_solve_quality_fields(SOLVE_QUALITY_EXACT_FIELDS),
            "exact_factorization_backend": EXACT_FACTORIZATION_BACKEND,
            "exact_condition_estimate": exact_condition_estimate,
            "exact_newton_linear_residual_rel": result.get(
                "exact_newton_linear_residual_rel"
            ),
            "exact_refinement_correction_rel": result.get(
                "exact_refinement_correction_rel"
            ),
        }
        self.res = res
        self.need_to_run_code = False

        if verbose:
            if materialization_message is not None:
                print(materialization_message, flush=True)
            res_norm = _host_inf_norm(res["residual"])
            print(
                f"NEWTON solve - success={res['success']}  "
                f"iter={res['iter']}, iota={iota_final:.16f}, "
                f"||residual||_inf={res_norm:.3e}",
                flush=True,
            )
        return res

    def run_code_functional(self, coil_arrays, sdofs, iota, G):
        """Compatibility shim returning the runtime-native traceable schema.

        This entrypoint survives only as a migration alias for callers that
        still expect a pure-function named solve. The historical
        ``run_code()``-shaped packaging has been retired; downstream users
        should migrate to ``run_code_traceable()`` directly.
        """
        return self.run_code_traceable(
            coil_arrays,
            _as_jax_float64(sdofs),
            iota,
            G,
        )

    def minimize_boozer_exact_constraints_newton(
        self,
        tol=1e-12,
        maxiter=10,
        iota=0.0,
        G=None,
        lm=(0.0, 0.0),
    ):
        """CPU-parity exact-constraints Newton solver matching the CPU API.

        Non-production path: this compatibility solver materializes dense
        Jacobians and uses ``jnp.linalg.solve`` inside the loop. The production
        exact runtime and adjoint path stays operator-backed and matrix-free.
        """
        if not self.need_to_run_code:
            return self.res

        optimize_G = G is not None
        s = self.surface
        lm_init = _as_jax_float64(lm)
        if lm_init.shape != (2,):
            raise ValueError("lm must contain exactly two Lagrange multipliers.")

        x0 = self._pack_decision_vector(iota, G)
        xl = _concat_jax_float64(x0, lm_init)
        residual_fn = self._make_exact_constraints_residual_with(
            optimize_G,
            self.options["weight_inv_modB"],
            hostify_inputs=False,
        )
        residual_and_jacobian = jax.jit(
            lambda x: (residual_fn(x), jax.jacobian(residual_fn)(x))
        )
        residual, jacobian = residual_and_jacobian(xl)
        norm = jnp.linalg.norm(residual)
        norm_value = float(_host_scalar(norm))
        nit = 0
        while nit < maxiter and norm_value > tol:
            if self.stellsym:
                solve_matrix = jacobian[:-1, :-1]
                solve_rhs = residual[:-1]
            else:
                solve_matrix = jacobian
                solve_rhs = residual

            dx = jnp.linalg.solve(solve_matrix, solve_rhs)
            if norm_value < 1e-9:
                dx = dx + jnp.linalg.solve(
                    solve_matrix,
                    solve_rhs - solve_matrix @ dx,
                )

            if self.stellsym:
                xl = _concat_jax_float64(xl[:-1] - dx, [xl[-1]])
            else:
                xl = xl - dx
            residual, jacobian = residual_and_jacobian(xl)
            norm = jnp.linalg.norm(residual)
            norm_value = float(_host_scalar(norm))
            nit += 1

        if optimize_G:
            sdofs_final = xl[:-4]
            iota_out = float(_host_scalar(xl[-4]))
            G_out = float(_host_scalar(xl[-3]))
        else:
            sdofs_final = xl[:-3]
            iota_out = float(_host_scalar(xl[-3]))
            G_out = None

        self._set_surface_dofs(sdofs_final)
        lm_out = float(_host_scalar(xl[-2])) if self.stellsym else _host_numpy(xl[-2:])
        success = bool(float(_host_scalar(norm)) <= tol)
        res = {
            "residual": residual,
            "jacobian": jacobian,
            "iter": nit,
            "success": success,
            "primal_success": success,
            "adjoint_linear_solve_available": False,
            "sdofs": _as_jax_float64(sdofs_final),
            "lm": lm_out,
            "G": G_out,
            "s": s,
            "iota": iota_out,
            "weight_inv_modB": self.options["weight_inv_modB"],
            "type": "exact_constraints",
            **_none_solve_quality_fields(SOLVE_QUALITY_EXACT_FIELDS),
        }
        self.res = res
        self.need_to_run_code = False
        return res

    def run_code(self, iota, G=None, *, sdofs=None):
        """Run the Boozer surface solver (LS or exact depending on config).

        Mirrors ``BoozerSurface.run_code()`` API.

        Args:
            iota: initial guess for rotational transform.
            G: initial guess for G (None → compute from coil currents,
               and coil currents must be fixed).
            sdofs: explicit surface DOFs for the initial guess. If None,
                reads from ``self.surface``.  When provided, syncs
                ``self.surface`` to ``sdofs`` before the solve so that
                failure paths leave the surface in a consistent state.

        Returns:
            dict with solver results, or None if solver was not dirty.
        """
        if not self.need_to_run_code:
            return

        # Sync surface DOFs when caller provides explicit warm-start.
        # This ensures failure paths (which skip _set_surface_dofs) leave
        # self.surface in a state consistent with the warm-start DOFs,
        # matching the old pre-solve ``surface.x = sdofs`` behavior.
        if sdofs is not None:
            self._set_surface_dofs(sdofs)

        # When G=None the gradient treats currents as constants,
        # so coil currents must be fixed to avoid silent gradient errors.
        self._validate_none_G_precondition(G)

        # Refresh coil data in case coils changed
        self._refresh_coil_data()

        if self.boozer_type == "exact":
            res = self.solve_residual_equation_exactly_newton(
                iota=iota,
                G=G,
                tol=self.options["newton_tol"],
                maxiter=self.options["newton_maxiter"],
                verbose=self.options["verbose"],
            )
            return res

        # BoozerLS: BFGS + Newton polish
        assert self.constraint_weight is not None
        first_stage_method = self._resolve_optimizer_method(optimize_G=G is not None)
        self._emit_stage_callback(
            "before_boozer_lbfgs",
            method=str(first_stage_method),
        )
        ls_res = self.minimize_boozer_penalty_constraints_LBFGS(
            constraint_weight=self.constraint_weight,
            iota=iota,
            G=G,
            tol=self.options["bfgs_tol"],
            maxiter=self.options["bfgs_maxiter"],
            verbose=self.options["verbose"],
            limited_memory=self.options["limited_memory"],
            weight_inv_modB=self.options["weight_inv_modB"],
        )
        self._emit_stage_callback(
            "after_boozer_lbfgs",
            solve_success=("true" if bool(ls_res["success"]) else "false"),
            iterations=float(ls_res["iter"]),
            method=str(ls_res["optimizer_method"]),
            **self._solver_diagnostics_payload(
                ls_res,
                gradient_key="gradient",
            ),
        )
        iota_out, G_out = ls_res["iota"], ls_res["G"]
        pre_newton_surface_dofs = _host_numpy(ls_res["sdofs"]).copy()
        pre_newton_decision_pieces = [
            pre_newton_surface_dofs,
            np.asarray([float(_host_scalar(iota_out))], dtype=float),
        ]
        if G_out is not None:
            pre_newton_decision_pieces.append(
                np.asarray([float(_host_scalar(G_out))], dtype=float)
            )
        pre_newton = {
            "optimizer_method": ls_res["optimizer_method"],
            "success": bool(ls_res["success"]),
            "iter": int(ls_res["iter"]),
            "fun": float(ls_res["fun"]),
            "iota": float(_host_scalar(iota_out)),
            "G": None if G_out is None else float(_host_scalar(G_out)),
            "surface_dofs": pre_newton_surface_dofs,
            "decision_vector": np.concatenate(pre_newton_decision_pieces),
            "gradient": _host_numpy(ls_res["gradient"]).copy(),
            "scipy_call_contract": ls_res.get("scipy_call_contract"),
            "scipy_initial_call": ls_res.get("scipy_initial_call"),
            "scipy_callback_trace": ls_res.get("scipy_callback_trace"),
        }

        # Polish with Newton
        self.need_to_run_code = True
        self._emit_stage_callback(
            "before_boozer_newton",
            method="newton-polish",
            ls_method=str(ls_res["optimizer_method"]),
        )
        res = self.minimize_boozer_penalty_constraints_newton(
            constraint_weight=self.constraint_weight,
            iota=iota_out,
            G=G_out,
            verbose=self.options["verbose"],
            tol=self.options["newton_tol"],
            maxiter=self.options["newton_maxiter"],
            stab=self.options["newton_stab"],
            weight_inv_modB=self.options["weight_inv_modB"],
        )
        res["optimizer_method"] = ls_res["optimizer_method"]
        res["pre_newton"] = pre_newton
        self._emit_stage_callback(
            "after_boozer_newton",
            solve_success=("true" if bool(res["success"]) else "false"),
            iterations=float(res["iter"]),
            **self._solver_diagnostics_payload(
                res,
                gradient_key="jacobian",
                residual_key="residual",
            ),
        )
        return res
