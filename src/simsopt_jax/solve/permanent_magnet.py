"""Solve-level JAX wrappers for fixed-state permanent-magnet optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.core._math_utils import has_tracer_leaf as _has_tracer_leaf
from simsopt_jax.core._math_utils import (
    as_runtime_array as _as_runtime_array,
    as_runtime_value as _as_runtime_value,
)
from simsopt_jax.runtime.host_boundary import host_array as _host_array
from simsopt_jax.runtime.host_boundary import host_float as _host_float
from simsopt_jax.core.pm_optimization import (
    GPMOArbVecBacktrackingResult as _CoreGPMOArbVecBacktrackingResult,
    GPMOArbVecBacktrackingSpec,
    GPMOArbVecResult as _CoreGPMOArbVecResult,
    GPMOArbVecSpec,
    GPMOBacktrackingResult as _CoreGPMOBacktrackingResult,
    GPMOBacktrackingSpec,
    GPMOBaselineResult as _CoreGPMOBaselineResult,
    GPMOBaselineSpec,
    GPMOMultiResult as _CoreGPMOMultiResult,
    GPMOMultiSpec,
    PMOptimizationSpec,
    gpmo_arbvec_backtracking_solve,
    gpmo_arbvec_solve,
    gpmo_backtracking_solve,
    gpmo_baseline_solve,
    gpmo_multi_solve,
    mwpgp_solve,
    projection_l2_balls,
)

__all__ = [
    "GPMOArbVecBacktrackingResult",
    "GPMOArbVecResult",
    "GPMOBacktrackingResult",
    "GPMOBaselineResult",
    "GPMOMultiResult",
    "GPMOPublicResult",
    "GPMO_ArbVec_backtracking_jax",
    "GPMO_ArbVec_jax",
    "GPMO_backtracking_jax",
    "GPMO_baseline_jax",
    "GPMO_multi_jax",
    "PMRelaxAndSplitResult",
    "projection_L2_balls_jax",
    "prox_l0_jax",
    "prox_l1_jax",
    "relax_and_split_jax",
    "setup_initial_condition_jax",
]


def _moments_as_matrix(name: str, value: object, ndipoles: int) -> jax.Array:
    moments = _as_runtime_array(value)
    if moments.shape == (ndipoles * 3,):
        return jnp.reshape(moments, (ndipoles, 3))
    if moments.shape != (ndipoles, 3):
        raise ValueError(f"{name} must have shape ({ndipoles}, 3).")
    return moments


def _flatten_like(reference: jax.Array, moments: jax.Array) -> jax.Array:
    return jnp.reshape(moments, reference.shape)


def _scalar_like(value: float, reference: jax.Array) -> jax.Array:
    return _as_runtime_value(value, reference=reference, dtype=reference.dtype)


def _zeros_like_shape(reference: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return _as_runtime_array(
        np.zeros(shape, dtype=np.dtype(reference.dtype)),
        dtype=reference.dtype,
        reference=reference,
    )


def _normalized_moment_magnitudes(matrix: jax.Array, m_maxima: jax.Array) -> jax.Array:
    return jnp.abs(matrix) / m_maxima[:, None]


def _is_tracing(value: object) -> bool:
    return isinstance(value, jax.core.Tracer)


def _raise_if_infeasible_initial_condition(
    moments: jax.Array, projected: jax.Array
) -> None:
    moments_host = _host_array(moments)
    projected_host = _host_array(projected)
    if not np.allclose(moments_host, projected_host):
        raise ValueError(
            "Initial dipole guess must contain values that satisfy the "
            "maximum bound constraints."
        )


def _host_scalar(name: str, value: jax.Array) -> float:
    array = _host_array(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return _host_float(array)


@dataclass(frozen=True)
class PMRelaxAndSplitResult:
    """Immutable result from ``relax_and_split_jax``."""

    errors: jax.Array
    m_history: jax.Array
    m_proxy_history: jax.Array
    m: jax.Array
    m_proxy: jax.Array
    residual_history: jax.Array


jax.tree_util.register_dataclass(
    PMRelaxAndSplitResult,
    data_fields=[
        "errors",
        "m_history",
        "m_proxy_history",
        "m",
        "m_proxy",
        "residual_history",
    ],
    meta_fields=[],
)


GPMOCoreResult: TypeAlias = (
    _CoreGPMOBaselineResult
    | _CoreGPMOMultiResult
    | _CoreGPMOBacktrackingResult
    | _CoreGPMOArbVecResult
    | _CoreGPMOArbVecBacktrackingResult
)


GPMOCoreResultType: TypeAlias = (
    type[_CoreGPMOBaselineResult]
    | type[_CoreGPMOMultiResult]
    | type[_CoreGPMOBacktrackingResult]
    | type[_CoreGPMOArbVecResult]
    | type[_CoreGPMOArbVecBacktrackingResult]
)


def _legacy_state_field(state: dict[str, object], name: str) -> jax.Array:
    value = state[name]
    if isinstance(value, jax.Array):
        return value
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    raise TypeError(f"legacy result field {name} must be an array")


class GPMOPublicResult:
    """Physical-moment public wrapper around a normalized core GPMO result."""

    _core_result_type: ClassVar[GPMOCoreResultType | None] = None
    _legacy_core_fields: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        m: jax.Array,
        m_history: jax.Array,
        *legacy_values: jax.Array,
        core_result: GPMOCoreResult | None = None,
        **legacy_fields: jax.Array,
    ) -> None:
        if core_result is not None:
            if legacy_values or legacy_fields:
                raise TypeError(
                    "core_result cannot be combined with legacy result fields"
                )
            core_result, core_fields = self._core_result_and_fields_from_core(
                core_result
            )
        else:
            core_result, core_fields = self._core_result_and_fields_from_legacy(
                legacy_values,
                legacy_fields,
            )
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "m_history", m_history)
        for name, value in core_fields.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_core_result", core_result)

    @property
    def core_result(self) -> GPMOCoreResult:
        return object.__getattribute__(self, "_core_result")

    def __setstate__(self, state: dict[str, object]) -> None:
        if "_core_result" in state:
            legacy_fields = {
                name: _legacy_state_field(state, name)
                for name in self._legacy_core_fields
            }
            self.__init__(
                _legacy_state_field(state, "m"),
                _legacy_state_field(state, "m_history"),
                core_result=state["_core_result"],
            )
            for name, value in legacy_fields.items():
                object.__setattr__(self, name, value)
            return
        if "core_result" in state:
            self.__init__(
                _legacy_state_field(state, "m"),
                _legacy_state_field(state, "m_history"),
                core_result=state["core_result"],
            )
            return
        legacy_fields = {
            name: _legacy_state_field(state, name) for name in self._legacy_core_fields
        }
        self.__init__(
            _legacy_state_field(state, "m"),
            _legacy_state_field(state, "m_history"),
            **legacy_fields,
        )

    def _core_result_and_fields_from_core(
        self,
        core_result: GPMOCoreResult,
    ) -> tuple[GPMOCoreResult, dict[str, jax.Array]]:
        core_result_type = self._core_result_type
        if core_result_type is None:
            raise TypeError("GPMOPublicResult requires core_result")
        if not isinstance(core_result, core_result_type):
            raise TypeError(
                f"core_result must be {core_result_type.__name__}, "
                f"got {type(core_result).__name__}"
            )
        core_fields = {
            name: getattr(core_result, name) for name in self._legacy_core_fields
        }
        return core_result, core_fields

    def _core_result_and_fields_from_legacy(
        self,
        legacy_values: tuple[jax.Array, ...],
        legacy_fields: dict[str, jax.Array],
    ) -> tuple[GPMOCoreResult, dict[str, jax.Array]]:
        core_result = self._core_result_from_legacy(legacy_values, legacy_fields)
        core_fields = {
            name: getattr(core_result, name) for name in self._legacy_core_fields
        }
        return core_result, core_fields

    def _core_result_from_legacy(
        self,
        legacy_values: tuple[jax.Array, ...],
        legacy_fields: dict[str, jax.Array],
    ) -> GPMOCoreResult:
        core_result_type = self._core_result_type
        if core_result_type is None:
            raise TypeError("GPMOPublicResult requires core_result")
        expected_fields = self._legacy_core_fields
        if len(legacy_values) > len(expected_fields):
            raise TypeError("too many positional legacy result fields")
        values = dict(legacy_fields)
        for name, value in zip(expected_fields, legacy_values, strict=False):
            if name in values:
                raise TypeError(f"multiple values for legacy result field: {name}")
            values[name] = value
        missing = tuple(name for name in expected_fields if name not in values)
        if missing:
            raise TypeError(
                "missing required legacy result fields: " + ", ".join(missing)
            )
        extra = tuple(sorted(set(values) - set(expected_fields)))
        if extra:
            raise TypeError("unexpected legacy result fields: " + ", ".join(extra))
        core_fields = {name: values[name] for name in expected_fields}
        return core_result_type(**core_fields)

    def __getattr__(self, name: str) -> object:
        return getattr(object.__getattribute__(self, "_core_result"), name)


@dataclass(frozen=True, init=False)
class GPMOBaselineResult(GPMOPublicResult):
    _core_result_type: ClassVar[GPMOCoreResultType] = _CoreGPMOBaselineResult
    _legacy_core_fields: ClassVar[tuple[str, ...]] = (
        "x",
        "x_history",
        "residual",
        "residual_history",
        "selected_dipoles",
        "selected_components",
        "selected_signs",
    )

    m: jax.Array
    m_history: jax.Array
    x: jax.Array
    x_history: jax.Array
    residual: jax.Array
    residual_history: jax.Array
    selected_dipoles: jax.Array
    selected_components: jax.Array
    selected_signs: jax.Array


@dataclass(frozen=True, init=False)
class GPMOMultiResult(GPMOPublicResult):
    _core_result_type: ClassVar[GPMOCoreResultType] = _CoreGPMOMultiResult
    _legacy_core_fields: ClassVar[tuple[str, ...]] = (
        "x",
        "x_history",
        "residual",
        "residual_history",
        "selected_seed_dipoles",
        "selected_components",
        "selected_signs",
        "selected_groups",
    )

    m: jax.Array
    m_history: jax.Array
    x: jax.Array
    x_history: jax.Array
    residual: jax.Array
    residual_history: jax.Array
    selected_seed_dipoles: jax.Array
    selected_components: jax.Array
    selected_signs: jax.Array
    selected_groups: jax.Array


@dataclass(frozen=True, init=False)
class GPMOBacktrackingResult(GPMOPublicResult):
    _core_result_type: ClassVar[GPMOCoreResultType] = _CoreGPMOBacktrackingResult
    _legacy_core_fields: ClassVar[tuple[str, ...]] = (
        "x",
        "x_history",
        "residual",
        "residual_history",
        "selected_dipoles",
        "selected_components",
        "selected_signs",
        "num_nonzeros_history",
        "removed_pair_count_history",
        "done_history",
    )

    m: jax.Array
    m_history: jax.Array
    x: jax.Array
    x_history: jax.Array
    residual: jax.Array
    residual_history: jax.Array
    selected_dipoles: jax.Array
    selected_components: jax.Array
    selected_signs: jax.Array
    num_nonzeros_history: jax.Array
    removed_pair_count_history: jax.Array
    done_history: jax.Array


@dataclass(frozen=True, init=False)
class GPMOArbVecResult(GPMOPublicResult):
    _core_result_type: ClassVar[GPMOCoreResultType] = _CoreGPMOArbVecResult
    _legacy_core_fields: ClassVar[tuple[str, ...]] = (
        "x",
        "x_history",
        "residual",
        "residual_history",
        "selected_dipoles",
        "selected_vector_indices",
        "selected_signs",
    )

    m: jax.Array
    m_history: jax.Array
    x: jax.Array
    x_history: jax.Array
    residual: jax.Array
    residual_history: jax.Array
    selected_dipoles: jax.Array
    selected_vector_indices: jax.Array
    selected_signs: jax.Array


@dataclass(frozen=True, init=False)
class GPMOArbVecBacktrackingResult(GPMOPublicResult):
    _core_result_type: ClassVar[GPMOCoreResultType] = _CoreGPMOArbVecBacktrackingResult
    _legacy_core_fields: ClassVar[tuple[str, ...]] = (
        "x",
        "x_history",
        "residual",
        "residual_history",
        "selected_dipoles",
        "selected_vector_indices",
        "selected_signs",
        "num_nonzeros_history",
        "removed_pair_count_history",
        "done_history",
        "initial_x",
        "initial_residual",
        "initial_num_nonzero",
    )

    m: jax.Array
    m_history: jax.Array
    x: jax.Array
    x_history: jax.Array
    residual: jax.Array
    residual_history: jax.Array
    selected_dipoles: jax.Array
    selected_vector_indices: jax.Array
    selected_signs: jax.Array
    num_nonzeros_history: jax.Array
    removed_pair_count_history: jax.Array
    done_history: jax.Array
    initial_x: jax.Array
    initial_residual: jax.Array
    initial_num_nonzero: jax.Array


for _gpmo_result_type in (
    GPMOBaselineResult,
    GPMOMultiResult,
    GPMOBacktrackingResult,
    GPMOArbVecResult,
    GPMOArbVecBacktrackingResult,
):
    jax.tree_util.register_dataclass(
        _gpmo_result_type,
        data_fields=["m", "m_history", *_gpmo_result_type._legacy_core_fields],
        meta_fields=[],
    )


_GPMOResultT = TypeVar("_GPMOResultT", bound=GPMOPublicResult)


def _gpmo_public_result(
    core_result: GPMOCoreResult,
    grid: PermanentMagnetGridJAX,
    result_type: type[_GPMOResultT],
) -> _GPMOResultT:
    return result_type(
        m=core_result.x * grid.m_maxima[:, None],
        m_history=core_result.x_history * grid.m_maxima[None, :, None],
        core_result=core_result,
    )


def projection_L2_balls_jax(x: object, mmax: object) -> jax.Array:
    """Project dipole moments onto per-dipole L2 balls, matching CPU shape."""

    moments = _as_runtime_array(x)
    m_maxima = _as_runtime_array(mmax)
    reshaped_moments = jnp.reshape(moments, (m_maxima.shape[0], 3))
    projected = projection_l2_balls(reshaped_moments, m_maxima)
    return _flatten_like(moments, projected)


def prox_l0_jax(m: object, mmax: object, reg_l0: float, nu: float) -> jax.Array:
    """JAX port of the CPU ``prox_l0`` thresholding rule."""

    moments = _as_runtime_array(m)
    m_maxima = _as_runtime_array(mmax)
    matrix = jnp.reshape(moments, (m_maxima.shape[0], 3))
    normalized = _normalized_moment_magnitudes(matrix, m_maxima)
    threshold = _scalar_like(2.0 * reg_l0 * nu, normalized)
    thresholded = matrix * (normalized > threshold)
    return _flatten_like(moments, thresholded)


def prox_l1_jax(m: object, mmax: object, reg_l1: float, nu: float) -> jax.Array:
    """JAX port of the CPU ``prox_l1`` soft-thresholding rule."""

    moments = _as_runtime_array(m)
    m_maxima = _as_runtime_array(mmax)
    matrix = jnp.reshape(moments, (m_maxima.shape[0], 3))
    normalized = _normalized_moment_magnitudes(matrix, m_maxima)
    threshold = _scalar_like(reg_l1 * nu, normalized)
    zero = _scalar_like(0.0, normalized)
    thresholded = (
        jnp.sign(matrix) * jnp.maximum(normalized - threshold, zero) * m_maxima[:, None]
    )
    return _flatten_like(moments, thresholded)


def setup_initial_condition_jax(
    grid: PermanentMagnetGridJAX, m0: object | None = None
) -> jax.Array:
    """Return the initial moment matrix for a fixed JAX PM grid.

    Explicit ``m0`` is an eager host-boundary input so infeasible values are
    rejected before a fixed-state solve starts. JIT callers should stage the
    validated initial condition in ``PermanentMagnetGridJAX.m0`` and pass
    ``m0=None``.
    """

    if m0 is None:
        return grid.m0
    moments = _moments_as_matrix("m0", m0, grid.ndipoles)
    if _is_tracing(moments):
        raise ValueError(
            "Explicit m0 validation requires an eager host-boundary call; "
            "stage the validated initial condition in PermanentMagnetGridJAX.m0 "
            "before JIT compilation."
        )
    projected = projection_l2_balls(moments, grid.m_maxima)
    _raise_if_infeasible_initial_condition(moments, projected)
    return moments


def _component_mmax(grid: PermanentMagnetGridJAX) -> jax.Array:
    return jnp.repeat(grid.m_maxima, 3)


def GPMO_baseline_jax(
    grid: PermanentMagnetGridJAX,
    *,
    K: int,
    reg_l2: float = 0.0,
    single_direction: int = -1,
    record_every: int | None = None,
    retain_history: bool = True,
) -> GPMOBaselineResult:
    """Run the baseline greedy GPMO algorithm on a fixed JAX PM grid.

    This remains an explicit baseline-only wrapper. Use ``GPMO_multi_jax`` for
    the multi-neighbour variant, ``GPMO_ArbVec_jax`` for arbitrary-vector
    placement, ``GPMO_backtracking_jax`` for baseline backtracking, and
    ``GPMO_ArbVec_backtracking_jax`` for the arbitrary-vector backtracking
    variant. Set ``retain_history=False`` for final-state-only workflows to
    avoid materializing the ``K x ndipoles x 3`` moment history.
    """

    mmax_vec = _component_mmax(grid)
    A_scaled = grid.A_obj * mmax_vec[None, :]
    core = gpmo_baseline_solve(
        GPMOBaselineSpec(
            m_maxima=grid.m_maxima,
            reg_l2=_scalar_like(reg_l2, grid.A_obj),
            single_direction=single_direction,
        ),
        A_scaled,
        grid.b_obj,
        K=K,
        record_every=record_every,
        retain_history=retain_history,
    )
    return _gpmo_public_result(core, grid, GPMOBaselineResult)


def GPMO_multi_jax(
    grid: PermanentMagnetGridJAX,
    *,
    K: int,
    reg_l2: float = 0.0,
    single_direction: int = -1,
    Nadjacent: int = 7,
    record_every: int | None = None,
) -> GPMOMultiResult:
    """Run the multi-neighbour greedy GPMO algorithm on a fixed JAX PM grid.

    This remains an explicit multi-only wrapper. Use ``GPMO_ArbVec_jax`` for
    arbitrary-vector placement, ``GPMO_backtracking_jax`` for baseline
    backtracking, and ``GPMO_ArbVec_backtracking_jax`` for the
    arbitrary-vector backtracking variant.
    """

    mmax_vec = _component_mmax(grid)
    A_scaled = grid.A_obj * mmax_vec[None, :]
    core = gpmo_multi_solve(
        GPMOMultiSpec(
            m_maxima=grid.m_maxima,
            reg_l2=_scalar_like(reg_l2, grid.A_obj),
            dipole_grid_xyz=grid.dipole_grid_xyz,
            single_direction=single_direction,
            Nadjacent=Nadjacent,
        ),
        A_scaled,
        grid.b_obj,
        K=K,
        record_every=record_every,
    )
    return _gpmo_public_result(core, grid, GPMOMultiResult)


def GPMO_backtracking_jax(
    grid: PermanentMagnetGridJAX,
    *,
    K: int,
    reg_l2: float = 0.0,
    single_direction: int = -1,
    Nadjacent: int = 7,
    backtracking: int = 100,
    max_nMagnets: int = 1000,
    record_every: int | None = None,
) -> GPMOBacktrackingResult:
    """Run the baseline backtracking GPMO algorithm on a fixed JAX PM grid.

    This remains an explicit backtracking-only wrapper. The arbitrary-vector
    backtracking variant lives in ``GPMO_ArbVec_backtracking_jax``. Neither
    wrapper mutates a CPU grid.
    """

    mmax_vec = _component_mmax(grid)
    A_scaled = grid.A_obj * mmax_vec[None, :]
    core = gpmo_backtracking_solve(
        GPMOBacktrackingSpec(
            m_maxima=grid.m_maxima,
            reg_l2=_scalar_like(reg_l2, grid.A_obj),
            dipole_grid_xyz=grid.dipole_grid_xyz,
            single_direction=single_direction,
            Nadjacent=Nadjacent,
            backtracking=backtracking,
            max_nMagnets=max_nMagnets,
        ),
        A_scaled,
        grid.b_obj,
        K=K,
        record_every=record_every,
    )
    return _gpmo_public_result(core, grid, GPMOBacktrackingResult)


def _gpmo_pol_vectors(
    grid: PermanentMagnetGridJAX, pol_vectors: object | None
) -> jax.Array:
    if pol_vectors is not None:
        return _as_runtime_array(pol_vectors)
    if grid.pol_vectors is None:
        raise ValueError("GPMO_ArbVec_jax requires pol_vectors.")
    return grid.pol_vectors


def GPMO_ArbVec_jax(
    grid: PermanentMagnetGridJAX,
    *,
    K: int,
    reg_l2: float = 0.0,
    pol_vectors: object | None = None,
    record_every: int | None = None,
) -> GPMOArbVecResult:
    """Run the arbitrary-vector greedy GPMO algorithm on a fixed JAX PM grid.

    This remains an explicit arbitrary-vector-only wrapper. Use
    ``GPMO_ArbVec_backtracking_jax`` for the angle-thresholded backtracking
    variant.
    """

    pol_vectors_arr = _gpmo_pol_vectors(grid, pol_vectors)
    mmax_vec = _component_mmax(grid)
    A_scaled = grid.A_obj * mmax_vec[None, :]
    core = gpmo_arbvec_solve(
        GPMOArbVecSpec(
            m_maxima=grid.m_maxima,
            reg_l2=_scalar_like(reg_l2, grid.A_obj),
            pol_vectors=pol_vectors_arr,
        ),
        A_scaled,
        grid.b_obj,
        K=K,
        record_every=record_every,
    )
    return _gpmo_public_result(core, grid, GPMOArbVecResult)


def GPMO_ArbVec_backtracking_jax(
    grid: PermanentMagnetGridJAX,
    *,
    K: int,
    reg_l2: float = 0.0,
    Nadjacent: int = 7,
    backtracking: int = 100,
    thresh_angle: float = 3.141592653589793,
    max_nMagnets: int = 1000,
    pol_vectors: object | None = None,
    m_init: object | None = None,
    record_every: int | None = None,
) -> GPMOArbVecBacktrackingResult:
    """Run the arbitrary-vector backtracking GPMO algorithm on a fixed JAX PM grid.

    Mirrors the ``algorithm='ArbVec_backtracking'`` branch of the CPU
    ``GPMO`` wrapper (see ``simsopt.solve.permanent_magnet_optimization``).
    The optional ``m_init`` input is converted to normalized ``x_init``
    coordinates via ``m_init / repeat(m_maxima, 3).reshape(N, 3)``,
    matching the upstream ``contig(kwargs['m_init'] / mmax_vec.reshape(N, 3))``
    transform.
    """

    pol_vectors_arr = _gpmo_pol_vectors(grid, pol_vectors)
    mmax_vec = _component_mmax(grid)
    A_scaled = grid.A_obj * mmax_vec[None, :]
    if m_init is None:
        x_init_arr = grid.m0 - grid.m0
    else:
        m_init_arr = _moments_as_matrix("m_init", m_init, grid.ndipoles)
        x_init_arr = m_init_arr / grid.m_maxima[:, None]
    core = gpmo_arbvec_backtracking_solve(
        GPMOArbVecBacktrackingSpec(
            m_maxima=grid.m_maxima,
            reg_l2=_scalar_like(reg_l2, grid.A_obj),
            dipole_grid_xyz=grid.dipole_grid_xyz,
            pol_vectors=pol_vectors_arr,
            Nadjacent=Nadjacent,
            backtracking=backtracking,
            thresh_angle=thresh_angle,
            max_nMagnets=max_nMagnets,
        ),
        A_scaled,
        grid.b_obj,
        K=K,
        x_init=x_init_arr,
        record_every=record_every,
    )
    return _gpmo_public_result(core, grid, GPMOArbVecBacktrackingResult)


def _mwpgp_spec(
    grid: PermanentMagnetGridJAX,
    m_proxy: jax.Array,
    *,
    alpha: object | None,
    nu: float,
    reg_l2: float,
) -> PMOptimizationSpec:
    hessian_scale = grid.ATA_scale + _scalar_like(2.0 * reg_l2 + 1.0 / nu, grid.A_obj)
    if alpha is None:
        alpha_value = _scalar_like(2.0 * (1.0 - 1.0e-5), grid.A_obj) / hessian_scale
    else:
        alpha_value = _as_runtime_array(alpha)
        if _has_tracer_leaf((alpha_value, hessian_scale)):
            raise ValueError(
                "Explicit alpha validation requires an eager host-boundary call."
            )
        alpha_host = _host_scalar("alpha", alpha_value)
        bound_host = _host_scalar(
            "2/lambda_max(H)", _scalar_like(2.0, hessian_scale) / hessian_scale
        )
        if alpha_host > bound_host:
            raise ValueError(
                "alpha must be <= 2/lambda_max(H) for MwPGP fixed-step "
                f"contraction; got {alpha_host} > {bound_host}."
            )
    return PMOptimizationSpec(
        m_maxima=grid.m_maxima,
        m_proxy=m_proxy,
        nu=_scalar_like(nu, grid.A_obj),
        reg_l2=_scalar_like(reg_l2, grid.A_obj),
        alpha=alpha_value,
    )


def _run_mwpgp(
    grid: PermanentMagnetGridJAX,
    m0: jax.Array,
    m_proxy: jax.Array,
    *,
    alpha: object | None,
    nu: float,
    reg_l2: float,
    max_iter: int,
) -> tuple[jax.Array, jax.Array]:
    spec = _mwpgp_spec(grid, m_proxy, alpha=alpha, nu=nu, reg_l2=reg_l2)
    return mwpgp_solve(spec, grid.A_obj, grid.ATb, m0, n_steps=max_iter)


def _run_mwpgp_with_alpha(
    grid: PermanentMagnetGridJAX,
    m0: jax.Array,
    m_proxy: jax.Array,
    *,
    alpha: jax.Array,
    nu: float,
    reg_l2: float,
    max_iter: int,
    record_residual: bool = True,
) -> tuple[jax.Array, jax.Array]:
    spec = PMOptimizationSpec(
        m_maxima=grid.m_maxima,
        m_proxy=m_proxy,
        nu=_scalar_like(nu, grid.A_obj),
        reg_l2=_scalar_like(reg_l2, grid.A_obj),
        alpha=alpha,
    )
    return mwpgp_solve(
        spec,
        grid.A_obj,
        grid.ATb,
        m0,
        n_steps=max_iter,
        record_residual=record_residual,
    )


@jax.jit
def _last_residual_error(residual_history: jax.Array) -> jax.Array:
    return residual_history[-1]


def _last_error(residual_history: jax.Array, max_iter: int) -> jax.Array:
    if max_iter == 0:
        return _scalar_like(0.0, residual_history)
    return _last_residual_error(residual_history)


def _relax_and_split_cost(
    grid: PermanentMagnetGridJAX,
    m: jax.Array,
    m_proxy: jax.Array,
    *,
    nu: float,
    reg_l2: float,
) -> jax.Array:
    m_flat = jnp.reshape(m, (-1,))
    residual = grid.A_obj @ m_flat - grid.b_obj
    r2 = 0.5 * jnp.sum(residual * residual)
    n2 = 0.5 * jnp.sum((m - m_proxy) * (m - m_proxy)) / _scalar_like(nu, m)
    l2 = _scalar_like(reg_l2, m) * jnp.sum(m * m)
    return r2 + n2 + l2


def _relax_and_split_scan(
    grid: PermanentMagnetGridJAX,
    m: jax.Array,
    m_proxy: jax.Array,
    *,
    alpha: jax.Array,
    nu: float,
    reg_l0: float,
    reg_l1: float,
    reg_l2: float,
    max_iter: int,
    max_iter_RS: int,
    epsilon_RS: float,
) -> PMRelaxAndSplitResult:
    prox = prox_l1_jax if np.isclose(reg_l0, 0.0, atol=1.0e-16) else prox_l0_jax
    reg_rs = reg_l1 if np.isclose(reg_l0, 0.0, atol=1.0e-16) else reg_l0
    epsilon = _scalar_like(epsilon_RS, m)
    zero = jnp.sum(m - m)
    done = zero != zero

    def _inactive_step(carry):
        m_done, m_proxy_done, done, error_done, residual_history_done = carry
        return carry, (error_done, m_done, m_proxy_done, residual_history_done)

    def _active_step(carry):
        m_current, m_proxy_current, _done, _error, _residual_history = carry
        m_next, residual_history = _run_mwpgp_with_alpha(
            grid,
            m_current,
            m_proxy_current,
            alpha=alpha,
            nu=nu,
            reg_l2=reg_l2,
            max_iter=max_iter,
            record_residual=False,
        )
        error = _relax_and_split_cost(
            grid,
            m_next,
            m_proxy_current,
            nu=nu,
            reg_l2=reg_l2,
        )
        m_proxy_next = _moments_as_matrix(
            "m_proxy", prox(m_next, grid.m_maxima, reg_rs, nu), grid.ndipoles
        )
        done_next = jnp.linalg.norm(m_next - m_proxy_next) < epsilon
        return (
            m_next,
            m_proxy_next,
            done_next,
            error,
            residual_history,
        ), (
            error,
            m_next,
            m_proxy_next,
            residual_history,
        )

    def _scan_body(carry, _iteration):
        return jax.lax.cond(carry[2], _inactive_step, _active_step, carry)

    final_carry, trace = jax.lax.scan(
        _scan_body,
        (
            m,
            m_proxy,
            done,
            zero,
            jnp.broadcast_to(zero, (max_iter,)),
        ),
        jax.lax.iota(jnp.int32, max_iter_RS),
    )
    errors, m_history, m_proxy_history, residual_histories = trace
    return PMRelaxAndSplitResult(
        errors=errors,
        m_history=m_history,
        m_proxy_history=m_proxy_history,
        m=final_carry[0],
        m_proxy=final_carry[1],
        residual_history=residual_histories,
    )


def relax_and_split_jax(
    grid: PermanentMagnetGridJAX,
    m0: object | None = None,
    *,
    alpha: object | None = None,
    max_iter: int = 100,
    max_iter_RS: int = 1,
    nu: float = 1.0e100,
    reg_l0: float = 0.0,
    reg_l1: float = 0.0,
    reg_l2: float = 0.0,
    epsilon_RS: float = 1.0e-3,
) -> PMRelaxAndSplitResult:
    """Run the fixed-step JAX relax-and-split PM solve wrapper.

    This is the solve-level adapter for the already-ported MwPGP kernel. It
    consumes an immutable ``PermanentMagnetGridJAX`` fixed-state payload and
    returns immutable arrays; it does not mutate a CPU ``PermanentMagnetGrid``.
    ``GPMO_baseline_jax``, ``GPMO_multi_jax``, ``GPMO_ArbVec_jax``,
    ``GPMO_backtracking_jax``, and ``GPMO_ArbVec_backtracking_jax`` cover the
    baseline, multi-neighbour, arbitrary-vector, baseline backtracking, and
    arbitrary-vector backtracking greedy variants respectively.
    """

    if max_iter_RS < 1:
        raise ValueError(f"max_iter_RS must be positive; got {max_iter_RS}")
    if (not np.isclose(reg_l0, 0.0, atol=1.0e-16)) and (
        not np.isclose(reg_l1, 0.0, atol=1.0e-16)
    ):
        raise ValueError("L0 and L1 loss terms cannot be used concurrently.")

    initial = setup_initial_condition_jax(grid, m0)
    no_l0 = np.isclose(reg_l0, 0.0, atol=1.0e-16)
    no_l1 = np.isclose(reg_l1, 0.0, atol=1.0e-16)
    if no_l0 and no_l1:
        m, residual_history = _run_mwpgp(
            grid,
            initial,
            initial,
            alpha=alpha,
            nu=nu,
            reg_l2=reg_l2,
            max_iter=max_iter,
        )
        return PMRelaxAndSplitResult(
            errors=jnp.reshape(_last_error(residual_history, max_iter), (1,)),
            m_history=m[None, :, :],
            m_proxy_history=_zeros_like_shape(m, (0, grid.ndipoles, 3)),
            m=m,
            m_proxy=m,
            residual_history=residual_history[None, :],
        )

    if no_l0:
        prox = prox_l1_jax
        reg_rs = reg_l1
    else:
        prox = prox_l0_jax
        reg_rs = reg_l0

    m = initial
    m_proxy = _moments_as_matrix(
        "m_proxy", prox(initial, grid.m_maxima, reg_rs, nu), grid.ndipoles
    )
    scan_spec = _mwpgp_spec(grid, m_proxy, alpha=alpha, nu=nu, reg_l2=reg_l2)
    return _relax_and_split_scan(
        grid,
        m=m,
        m_proxy=m_proxy,
        alpha=scan_spec.alpha,
        nu=nu,
        reg_l0=reg_l0,
        reg_l1=reg_l1,
        reg_l2=reg_l2,
        max_iter=max_iter,
        max_iter_RS=max_iter_RS,
        epsilon_RS=epsilon_RS,
    )
