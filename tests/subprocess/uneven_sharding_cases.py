"""Sharded JAX paths at extents that do not divide the device count.

Run by ``tests/jax/core/test_uneven_sharding.py`` in a child process whose
``XLA_FLAGS`` force several host devices (``argv[1]``, checked here). Every
case compares a sharded strategy (``SIMSOPT_JAX_SHARDING``) against the same
call with sharding off and, for fields, against native ``BiotSavart``.

``field <strategy>``: ``BiotSavartJAX`` ``B``, ``dB_by_dX``, ``A``,
``dA_by_dX``, ``d2B_by_dXdX`` and the ``B``/``dB_by_dX``/``A`` VJPs at point
counts 1, 4095, 4096 and 4097 (the first point at the origin) and coil counts
1, 3, 4 and 5. Each strategy meets every point count and every coil count, and
each (point, coil) pair appears under one strategy (a Latin square).

``consumers``: the other users of the sharding helpers at uneven extents:
pairwise reductions, the surface-quadrature flux integral, batched seed
scoring, batched trajectory tracing and the chunked tracing adapter.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field.tracing import MaxRStoppingCriterion
from simsopt.geo import create_equally_spaced_curves
from simsopt_jax.backend import get_sharding_tuning, set_backend
from simsopt_jax.core._pairwise_reductions import (
    pairwise_min_distance_pure,
    pairwise_thresholded_mean_square_distance_pure,
)
from simsopt_jax.core.integral_bdotn import (
    integral_BdotN,
    integral_BdotN_surface_sharded,
)
from simsopt_jax.core.tracing import (
    FieldlineTracingSpec,
    GuidingCenterTracingSpec,
    trace_fieldlines_batched,
    trace_guiding_centers_batched,
)
from simsopt_jax_adapters.field import tracing as tracing_adapter
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.field.toroidal_field import ToroidalFieldJAX
from simsopt_jax_adapters.geo.surface_objectives_traceable import (
    _make_traceable_batched_value_and_grad_pipeline,
)

_STRATEGIES = ("points", "coil_groups", "points_coils", "hybrid")
_POINT_COUNTS = (1, 4095, 4096, 4097)
_COIL_COUNTS = (1, 3, 4, 5)
# (rtol, atol) of tests/field/test_biotsavart_jax_correctness.py.
_FIELD_TOLERANCE = (1e-12, 1e-14)
_VJP_TOLERANCE = (1e-11, 1e-13)


def _select_strategy(strategy: str) -> None:
    os.environ["SIMSOPT_JAX_SHARDING"] = strategy
    set_backend("jax_cpu_parity")
    tuning = get_sharding_tuning()
    assert tuning.strategy == strategy, tuning
    assert tuning.active is (strategy != "none"), tuning


def _host(value) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _assert_matches(label: str, value, reference, tolerance) -> None:
    actual = _host(value)
    expected = np.asarray(reference)
    rtol, atol = tolerance
    assert np.all(np.isfinite(actual)), f"{label}: non-finite entries"
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=label)


def _coils(coil_count: int):
    curves = create_equally_spaced_curves(
        coil_count, 1, stellsym=False, R0=1.0, R1=0.5, order=3
    )
    currents = [Current(1e5 * (1.0 + 0.1 * index)) for index in range(coil_count)]
    return coils_via_symmetries(curves, currents, 1, False)


def _points(point_count: int) -> np.ndarray:
    points = np.random.default_rng(point_count).uniform(-0.3, 0.3, (point_count, 3))
    points[0] = 0.0
    return points


def _field_quantities(field, point_count: int) -> dict[str, Callable[[], object]]:
    rng = np.random.default_rng(7)
    v = rng.standard_normal((point_count, 3))
    vgrad = rng.standard_normal((point_count, 3, 3))
    return {
        "B": field.B,
        "dB_by_dX": field.dB_by_dX,
        "A": field.A,
        "dA_by_dX": field.dA_by_dX,
        "d2B_by_dXdX": field.d2B_by_dXdX,
        "B_vjp": lambda: field.B_vjp(v)(field),
        "dB_by_dX_vjp": lambda: field.B_and_dB_vjp(v, vgrad)[1](field),
        "A_vjp": lambda: field.A_vjp(v)(field),
    }


def _evaluate(field_type, coils, points) -> dict[str, np.ndarray]:
    field = field_type(coils)
    field.set_points(points)
    return {
        name: _host(quantity())
        for name, quantity in _field_quantities(field, points.shape[0]).items()
    }


def _run_field_case(strategy: str) -> None:
    shift = _STRATEGIES.index(strategy)
    combinations = [
        (_POINT_COUNTS[index], _COIL_COUNTS[(index + shift) % 4]) for index in range(4)
    ]
    for point_count, coil_count in combinations:
        coils = _coils(coil_count)
        points = _points(point_count)
        native = _evaluate(BiotSavart, coils, points)
        _select_strategy("none")
        single_device = _evaluate(BiotSavartJAX, coils, points)
        _select_strategy(strategy)
        sharded = _evaluate(BiotSavartJAX, coils, points)
        for name, value in sharded.items():
            tolerance = _VJP_TOLERANCE if name.endswith("_vjp") else _FIELD_TOLERANCE
            label = f"{strategy} points={point_count} coils={coil_count} {name}"
            _assert_matches(f"{label} vs native", value, native[name], tolerance)
            _assert_matches(
                f"{label} vs single device", value, single_device[name], tolerance
            )
        print(f"field {strategy} points={point_count} coils={coil_count} ok")


def _sharded_and_reference(strategy: str, evaluate: Callable[[], object]):
    """``evaluate()`` under ``strategy`` and with sharding off.

    JAX's trace caches are cleared before each call: these entry points are
    jitted and decide their sharding while tracing.
    """
    _select_strategy(strategy)
    jax.clear_caches()
    sharded = jax.tree.map(_host, evaluate())
    _select_strategy("none")
    jax.clear_caches()
    reference = jax.tree.map(_host, evaluate())
    return sharded, reference


def _assert_tree_matches(label: str, sharded, reference) -> None:
    rtol, atol = _FIELD_TOLERANCE
    for value, expected in zip(
        jax.tree.leaves(sharded), jax.tree.leaves(reference), strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(value), np.asarray(expected), rtol=rtol, atol=atol, err_msg=label
        )


def _run_pairwise_rows(device_count: int) -> None:
    rng = np.random.default_rng(3)
    right = rng.uniform(-1.0, 1.0, (5, 3))
    for row_count in (2 * device_count, 2 * device_count + 1):
        left = rng.uniform(-1.0, 1.0, (row_count, 3))
        for chunk_size in (0, 2):
            sharded, reference = _sharded_and_reference(
                "hybrid",
                lambda: (
                    pairwise_min_distance_pure(
                        jnp.asarray(left), jnp.asarray(right), chunk_size=chunk_size
                    ),
                    pairwise_thresholded_mean_square_distance_pure(
                        jnp.asarray(left),
                        jnp.asarray(right),
                        1.0,
                        chunk_size=chunk_size,
                    ),
                ),
            )
            _assert_tree_matches(f"pairwise rows={row_count}", sharded, reference)
            distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
            np.testing.assert_allclose(sharded[0], distances.min(), rtol=1e-14)
            np.testing.assert_allclose(
                sharded[1],
                np.mean(np.maximum(1.0 - distances, 0.0) ** 2),
                rtol=1e-12,
            )
    print("pairwise ok")


def _run_surface_quadrature() -> None:
    nphi, ntheta = 7, 5
    rng = np.random.default_rng(5)
    bcoil = rng.uniform(0.2, 1.4, (nphi, ntheta, 3))
    target = rng.uniform(-0.01, 0.02, (nphi, ntheta))
    normal = rng.uniform(0.3, 1.7, (nphi, ntheta, 3))
    for definition in ("quadratic flux", "normalized", "local"):
        sharded, _ = _sharded_and_reference(
            "points",
            lambda: integral_BdotN_surface_sharded(
                jnp.asarray(bcoil), jnp.asarray(target), jnp.asarray(normal), definition
            ),
        )
        reference = integral_BdotN(
            jnp.asarray(bcoil), jnp.asarray(target), jnp.asarray(normal), definition
        )
        _assert_tree_matches(f"surface quadrature {definition}", sharded, reference)
    print("surface quadrature ok")


def _run_seed_batch() -> None:
    def score_seeds():
        pipeline = _make_traceable_batched_value_and_grad_pipeline(
            jax.jit(jax.value_and_grad(lambda dofs: jnp.sum(jnp.sin(dofs) * dofs)))
        )
        return pipeline(jnp.asarray(seeds))

    seeds = np.linspace(-1.0, 1.0, 7 * 3).reshape(7, 3)
    sharded, reference = _sharded_and_reference("points", score_seeds)
    assert sharded[0].shape == (7,) and sharded[1].shape == (7, 3)
    _assert_tree_matches("seed batch", sharded, reference)
    print("seed batch ok")


def _toroidal_field(point):
    radius_squared = point[0] ** 2 + point[1] ** 2
    return 0.7 * jnp.stack((-point[1], point[0], 0.0 * point[2])) / radius_squared


def _uniform_field_and_gradient(point):
    del point
    return jnp.asarray([0.0, 0.0, 1.0]), jnp.zeros((3, 3))


def _run_trajectory_batches() -> None:
    lane_count = 3
    dtmaxs = jnp.full((lane_count,), 0.01)
    radii = jnp.asarray([1.0, 1.1, 1.2])
    zeros = jnp.zeros((lane_count,))
    fieldline_y0s = jnp.stack((radii, zeros, zeros), axis=1)
    particle_y0s = jnp.stack((radii, zeros, zeros, jnp.ones(lane_count)), axis=1)
    phis = jnp.asarray([0.0])
    traces = {
        "fieldlines": lambda: trace_fieldlines_batched(
            FieldlineTracingSpec(
                tmax=0.05,
                rtol=1e-8,
                atol=1e-10,
                dtmax=0.01,
                max_steps=16,
                max_phi_hits=2,
            ),
            fieldline_y0s,
            dtmaxs,
            _toroidal_field,
            phis=phis,
        ),
        "guiding centers": lambda: trace_guiding_centers_batched(
            GuidingCenterTracingSpec(
                tmax=0.05,
                rtol=1e-8,
                atol=1e-10,
                dtmax=0.01,
                max_steps=16,
                max_phi_hits=2,
            ),
            particle_y0s,
            dtmaxs,
            zeros,
            _uniform_field_and_gradient,
            m=1.0,
            q=1.0,
            phis=phis,
        ),
    }
    for name, trace in traces.items():
        sharded, reference = _sharded_and_reference("points", trace)
        assert sharded.trajectory.shape[0] == lane_count, name
        _assert_tree_matches(f"trajectories {name}", sharded, reference)

    def adapter_fieldlines():
        paths, hits, statuses = tracing_adapter.compute_fieldlines_with_status(
            ToroidalFieldJAX(1.3, 0.8),
            [1.2, 1.4, 1.2],
            [0.0] * lane_count,
            tmax=1,
            tol=1e-8,
            phis=(0.3,),
            stopping_criteria=[MaxRStoppingCriterion(1.3)],
        )
        return list(paths), list(hits), statuses

    sharded, reference = _sharded_and_reference("points", adapter_fieldlines)
    assert len(sharded[0]) == len(sharded[1]) == sharded[2].shape[0] == lane_count
    _assert_tree_matches("chunked tracing adapter", sharded, reference)
    print("trajectory batches ok")


def main() -> None:
    device_count, case = int(sys.argv[1]), sys.argv[2]
    assert len(jax.devices()) == device_count, jax.devices()
    for name in (
        "SIMSOPT_JAX_MIN_POINTS_TO_SHARD",
        "SIMSOPT_JAX_MIN_COILS_TO_SHARD",
        "SIMSOPT_JAX_MIN_PAIRWISE_ROWS_TO_SHARD",
    ):
        os.environ[name] = "1"
    if case == "field":
        _run_field_case(sys.argv[3])
        return
    assert case == "consumers", case
    _run_pairwise_rows(device_count)
    _run_surface_quadrature()
    _run_seed_batch()
    _run_trajectory_batches()


if __name__ == "__main__":
    main()
