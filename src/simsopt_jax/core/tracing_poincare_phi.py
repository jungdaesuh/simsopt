"""Differentiable phi-parametrized Poincare return tracing.

This module is additive to :mod:`simsopt_jax.core.tracing`: the arc-length
fieldline tracer keeps its step-grid/crossing contract, while this path uses the
toroidal angle as the independent variable. It is valid in the B_phi-dominated
nested-surface regime; B_phi = 0 is a singular model boundary and is reported as
non-finite output rather than hidden behind an epsilon.

The returned static phi grid is meant as a differentiable integration substrate
for future return-map, island-width, Greene-residue, converse-KAM, WBA, or
multisurface-QS objectives. This module only supplies the trace primitive and
gradient-capability tests; promoting a confinement objective onto it is a
separate optimizer-policy decision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from .sharding import (
    maybe_shard_trajectory_batch_inputs,
    trajectory_batch_sharding_config,
)
from .tracing import _as_device_array

__all__ = [
    "PoincareReturnResult",
    "PoincareTracingSpec",
    "_cyl_B_from_cartesian_field",
    "trace_poincare_phi",
    "trace_poincare_phi_batched",
]


@dataclass(frozen=True)
class PoincareTracingSpec:
    """Solver contract for phi-parametrized Poincare return tracing.

    ``max_steps`` is static because diffrax uses it to shape the solve; all
    other fields are device data so tolerance and bounds can participate in
    JAX transforms. Bounds are cylindrical ``(R_min, R_max)`` and
    ``(Z_min, Z_max)``.
    """

    rtol: float
    atol: float
    min_step_size: float = 1.0e-4
    max_steps: int = 1000
    bounds_R: tuple[float, float] = (0.0, np.inf)
    bounds_Z: tuple[float, float] = (-np.inf, np.inf)


jax.tree_util.register_dataclass(
    PoincareTracingSpec,
    data_fields=["rtol", "atol", "min_step_size", "bounds_R", "bounds_Z"],
    meta_fields=["max_steps"],
)


@dataclass(frozen=True)
class PoincareReturnResult:
    """Phi-grid return data for one fieldline.

    ``punctures`` has shape ``(n_phi, 2)`` with columns ``[R, Z]`` at the
    requested unwrapped phi grid. ``escaped`` marks non-finite save points,
    and ``status`` is ``0`` for success, ``-1`` for box exit, ``1`` otherwise.
    """

    punctures: jax.Array
    escaped: jax.Array
    status: jax.Array


jax.tree_util.register_dataclass(
    PoincareReturnResult,
    data_fields=["punctures", "escaped", "status"],
    meta_fields=[],
)


def _stage_spec(spec: PoincareTracingSpec) -> PoincareTracingSpec:
    return replace(
        spec,
        rtol=_as_device_array(spec.rtol, jnp.float64),
        atol=_as_device_array(spec.atol, jnp.float64),
        min_step_size=_as_device_array(spec.min_step_size, jnp.float64),
        bounds_R=(
            _as_device_array(spec.bounds_R[0], jnp.float64),
            _as_device_array(spec.bounds_R[1], jnp.float64),
        ),
        bounds_Z=(
            _as_device_array(spec.bounds_Z[0], jnp.float64),
            _as_device_array(spec.bounds_Z[1], jnp.float64),
        ),
    )


def _cyl_B_from_cartesian_field(
    field_fn: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Return cylindrical ``[B_R, B_phi, B_Z]`` from a Cartesian JAX field."""

    def rpz_B(R: jax.Array, phi: jax.Array, Z: jax.Array) -> jax.Array:
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)
        xyz = jnp.array([R * cos_phi, R * sin_phi, Z], dtype=jnp.float64)
        Bx, By, Bz = jnp.asarray(field_fn(xyz), dtype=jnp.float64).reshape((3,))
        return jnp.array(
            [
                Bx * cos_phi + By * sin_phi,
                -Bx * sin_phi + By * cos_phi,
                Bz,
            ],
            dtype=jnp.float64,
        )

    return rpz_B


def _phi_rhs(field_fn: Callable[[jax.Array], jax.Array]):
    rpz_B = _cyl_B_from_cartesian_field(field_fn)

    def rhs(phi: jax.Array, rz: jax.Array, _args) -> jax.Array:
        # Phi is the independent variable: dR/dphi = R*B_R/B_phi and
        # dZ/dphi = R*B_Z/B_phi. B_phi=0 is singular and must remain loud.
        R = rz[0]
        Z = rz[1]
        B_R, B_phi, B_Z = rpz_B(R, phi, Z)
        return jnp.array([R * B_R / B_phi, R * B_Z / B_phi], dtype=jnp.float64)

    return rhs


def _box_event(spec: PoincareTracingSpec):
    R_min, R_max = spec.bounds_R
    Z_min, Z_max = spec.bounds_Z

    def cond_fn(_phi, rz, _args, **_kwargs):
        R = rz[0]
        Z = rz[1]
        return (R < R_min) | (R > R_max) | (Z < Z_min) | (Z > Z_max)

    return diffrax.Event(cond_fn, root_finder=None)


def _normalize_status(sol_result, punctures: jax.Array, escaped: jax.Array) -> jax.Array:
    event_occurred = sol_result == diffrax.RESULTS.event_occurred
    successful = sol_result == diffrax.RESULTS.successful
    any_bad_puncture = jnp.any(escaped) | jnp.any(~jnp.isfinite(punctures))
    non_success = jnp.logical_or(jnp.logical_not(successful), any_bad_puncture)
    return jnp.where(
        event_occurred,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.where(
            non_success,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )


def _validate_concrete_phi_grid(phis) -> None:
    if isinstance(phis, jax.core.Tracer):
        return
    phis_np = np.asarray(phis, dtype=np.float64).reshape((-1,))
    if phis_np.shape[0] < 2:
        return
    diffs = np.diff(phis_np)
    increasing = np.all(diffs > 0.0)
    decreasing = np.all(diffs < 0.0)
    if not (increasing or decreasing):
        raise ValueError("phis must be strictly monotonic and unwrapped")


def _escaped_save_points(
    spec: PoincareTracingSpec, punctures: jax.Array
) -> jax.Array:
    R_min, R_max = spec.bounds_R
    Z_min, Z_max = spec.bounds_Z
    R = punctures[:, 0]
    Z = punctures[:, 1]
    nonfinite = jnp.any(~jnp.isfinite(punctures), axis=1)
    out_of_bounds = (R < R_min) | (R > R_max) | (Z < Z_min) | (Z > Z_max)
    escaped = nonfinite | out_of_bounds
    return jnp.cumsum(escaped.astype(jnp.int32)) > 0


def trace_poincare_phi(
    spec: PoincareTracingSpec,
    r0: jax.Array,
    z0: jax.Array,
    phis: jax.Array,
    field_fn: Callable[[jax.Array], jax.Array],
) -> PoincareReturnResult:
    """Integrate one line in phi and return ``[R, Z]`` at every requested phi."""

    dtype = jnp.float64
    _validate_concrete_phi_grid(phis)
    phis_arr = _as_device_array(phis, dtype).reshape((-1,))
    if phis_arr.shape[0] < 2:
        raise ValueError("phis must contain at least two unwrapped planes")
    max_steps = int(spec.max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")

    staged_spec = _stage_spec(spec)
    phi_span = phis_arr[-1] - phis_arr[0]
    step_sign = jnp.where(phi_span < 0.0, -1.0, 1.0)
    signed_min_step = step_sign * jnp.abs(staged_spec.min_step_size)
    y0 = jnp.array([r0, z0], dtype=dtype)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_phi_rhs(field_fn)),
        diffrax.Dopri5(),
        t0=phis_arr[0],
        t1=phis_arr[-1],
        dt0=signed_min_step,
        y0=y0,
        saveat=diffrax.SaveAt(ts=phis_arr),
        stepsize_controller=diffrax.PIDController(
            rtol=staged_spec.rtol,
            atol=staged_spec.atol,
            dtmin=signed_min_step,
        ),
        event=_box_event(staged_spec),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=max_steps,
        throw=False,
    )

    punctures = jnp.asarray(sol.ys, dtype=dtype)
    punctures = jnp.where(jnp.isinf(punctures), jnp.nan, punctures)
    escaped = _escaped_save_points(staged_spec, punctures)
    punctures = jnp.where(escaped[:, None], jnp.nan, punctures)
    return PoincareReturnResult(
        punctures=punctures,
        escaped=escaped,
        status=_normalize_status(sol.result, punctures, escaped),
    )


def _make_phi_trace_one(spec, field_fn, phis):
    def trace_one(
        r0: jax.Array, z0: jax.Array, field_state: object | None = None
    ) -> PoincareReturnResult:
        if field_state is None:
            active_field_fn = field_fn
        else:

            def active_field_fn(point):
                return field_fn(field_state, point)

        return trace_poincare_phi(spec, r0, z0, phis, active_field_fn)

    return trace_one


@partial(jax.jit, static_argnames=("field_fn",))
def _trace_poincare_phi_batched_unsharded(
    spec: PoincareTracingSpec,
    r0s: jax.Array,
    z0s: jax.Array,
    phis: jax.Array,
    field_fn: Callable[[jax.Array], jax.Array],
    magnetic_field_state: object | None = None,
) -> PoincareReturnResult:
    trace_one = _make_phi_trace_one(spec, field_fn, phis)
    if magnetic_field_state is None:
        return jax.vmap(trace_one)(r0s, z0s)
    return jax.vmap(trace_one, in_axes=(0, 0, None))(r0s, z0s, magnetic_field_state)


def trace_poincare_phi_batched(
    spec: PoincareTracingSpec,
    r0s: jax.Array,
    z0s: jax.Array,
    phis: jax.Array,
    field_fn: Callable[[jax.Array], jax.Array],
    magnetic_field_state: object | None = None,
) -> PoincareReturnResult:
    """Trace many phi-parametrized fieldlines with a lane-leading result shape."""

    staged_spec = _stage_spec(spec)
    r0s_arr = _as_device_array(r0s, jnp.float64).reshape((-1,))
    z0s_arr = _as_device_array(z0s, jnp.float64).reshape((-1,))
    phis_arr = _as_device_array(phis, jnp.float64).reshape((-1,))

    trace_one = _make_phi_trace_one(staged_spec, field_fn, phis_arr)
    config = trajectory_batch_sharding_config(r0s_arr)
    if config is not None:
        r0s_arr, z0s_arr = maybe_shard_trajectory_batch_inputs(
            r0s_arr,
            z0s_arr,
            config=config,
        )
        out_specs = PoincareReturnResult(
            punctures=P(config.axis_name, None, None),
            escaped=P(config.axis_name, None),
            status=P(config.axis_name),
        )
        if magnetic_field_state is None:

            @partial(
                jax.shard_map,
                mesh=config.mesh,
                in_specs=(P(config.axis_name), P(config.axis_name)),
                out_specs=out_specs,
                check_vma=True,
            )
            def trace_shard(r0s_block, z0s_block):
                return jax.lax.map(
                    lambda inputs: trace_one(*inputs),
                    (r0s_block, z0s_block),
                )

            return trace_shard(r0s_arr, z0s_arr)

        field_state_specs = jax.tree.map(lambda _leaf: P(), magnetic_field_state)

        @partial(
            jax.shard_map,
            mesh=config.mesh,
            in_specs=(P(config.axis_name), P(config.axis_name), field_state_specs),
            out_specs=out_specs,
            check_vma=True,
        )
        def trace_shard(r0s_block, z0s_block, field_state_block):
            return jax.lax.map(
                lambda inputs: trace_one(inputs[0], inputs[1], field_state_block),
                (r0s_block, z0s_block),
            )

        return trace_shard(r0s_arr, z0s_arr, magnetic_field_state)

    return _trace_poincare_phi_batched_unsharded(
        staged_spec,
        r0s_arr,
        z0s_arr,
        phis_arr,
        field_fn,
        magnetic_field_state,
    )
