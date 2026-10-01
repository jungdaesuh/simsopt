"""Short-horizon trajectory check for the planar-coils lanes at the bounded scale.

The two lanes' end objectives differ by several percent, and so do each lane's
own runs from starts one unit in the last place apart: the bounded two-stage
L-BFGS-B amplifies a rounding difference by about 0.2 decades per objective
evaluation.  An end point therefore cannot say whether the lanes follow the
same trajectory.  This file asks it where the question has an answer, early:

    at every objective evaluation ``k`` before the horizon, the distance between
    the native and the JAX lane's iterates is at most the largest distance
    between a lane and its own one-ulp twin at the same evaluation.

The rule, fixed before the numbers it judges were computed:

* twins are each lane restarted from upstream's one-ulp protocol
  (the sensitivity protocol recorded in the official-reference fixture:
  ``x0' = nextafter(x0, s inf)``, ``s`` from ``RandomState(20260920 + k)``,
  ``k = 1..8``), so a twin starts at exactly the state upstream's own draw ``k``
  recorded; the envelope is the largest of the sixteen twin distances;
* the distance between two runs at evaluation ``k`` is the larger, over the
  current and the geometry dofs, of ``max |dx| / max |x|`` within that block
  (the currents are ``1e5``, the geometry ``O(1)``);
* evaluations are compared only while the two runs are in the same minimize
  call, and the horizon is the first evaluation at which EVERY twin distance has
  reached ``1e-6``; beyond it the lanes' own scatter has left the linear regime.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import scipy.optimize
from examples.jax.parity.cases import (
    native_stage_two_optimization_planar_coils as planar_case,
)
from examples.jax.parity.input_bundle import load_input_bundle
from examples.jax.parity.official_reference import load_upstream_scatter
from simsopt_jax.solve import dispatch

pytestmark = pytest.mark.slow

CASE_ID = "native-stage-two-optimization-planar-coils"
TWINS = tuple(range(1, 9))
HORIZON_DISTANCE = 1.0e-6
CURRENT_DOFS = 3


def one_ulp_start(parameters: np.ndarray, k: int) -> np.ndarray:
    """Upstream's one-ulp protocol draw ``k`` of a start vector."""
    signs = np.random.RandomState(20260920 + k).choice(
        [-1.0, 1.0], size=parameters.size
    )
    return np.nextafter(parameters, signs * np.inf)


@dataclass(frozen=True)
class Trace:
    """Every objective evaluation of one run: minimize-call index and iterate."""

    call: np.ndarray
    x: np.ndarray


def _trace(
    run: Callable[[], object], monkeypatch: pytest.MonkeyPatch, module, name: str
) -> Trace:
    calls: list[int] = []
    iterates: list[np.ndarray] = []
    minimize_calls = [0]
    real_minimize = scipy.optimize.minimize

    def traced_minimize(fun, x0, *args, **kwargs):
        index = minimize_calls[0]
        minimize_calls[0] += 1

        def recorded(x, *fun_args):
            calls.append(index)
            iterates.append(np.array(x, dtype=np.float64, copy=True))
            return fun(x, *fun_args)

        return real_minimize(recorded, x0, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(module, name, traced_minimize)
        run()
    return Trace(np.asarray(calls), np.stack(iterates))


def distances(first: Trace, second: Trace) -> np.ndarray:
    """Per-evaluation block-normalised iterate distance while both runs share a minimize call."""
    count = min(first.call.size, second.call.size)
    same = first.call[:count] == second.call[:count]
    count = count if bool(np.all(same)) else int(np.argmin(same))
    a, b = first.x[:count], second.x[:count]
    blocks = (slice(0, CURRENT_DOFS), slice(CURRENT_DOFS, None))
    return np.max(
        [
            np.max(np.abs(a[:, s] - b[:, s]), axis=1) / np.max(np.abs(a[:, s]), axis=1)
            for s in blocks
        ],
        axis=0,
    )


def test_cross_lane_iterates_stay_inside_the_lanes_own_one_ulp_scatter(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "inputs"
    bundle = planar_case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)
    start = np.asarray(arrays["initial_parameters"], dtype=np.float64)
    scatter = load_upstream_scatter(CASE_ID, "bounded")
    for k in TWINS:
        np.testing.assert_array_equal(
            one_ulp_start(start, k), scatter.run(k).value("initial:parameters")
        )

    def lane_runs(lane: str, module, name: str) -> tuple[Trace, list[Trace]]:
        def run_from(parameters):
            return lambda: planar_case.execute(
                lane, bundle, dict(arrays, initial_parameters=parameters)
            )

        base = _trace(run_from(start), monkeypatch, module, name)
        twins = [
            _trace(run_from(one_ulp_start(start, k)), monkeypatch, module, name)
            for k in TWINS
        ]
        return base, twins

    native, native_twins = lane_runs("native-cpu", planar_case, "minimize")
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax_lane, jax_twins = lane_runs("jax-cpu", dispatch, "scipy_minimize")

    cross = distances(native, jax_lane)
    twin_distances = [distances(native, twin) for twin in native_twins] + [
        distances(jax_lane, twin) for twin in jax_twins
    ]
    count = min([cross.size] + [d.size for d in twin_distances])
    twin_matrix = np.stack([d[:count] for d in twin_distances])
    passed = np.all(twin_matrix >= HORIZON_DISTANCE, axis=0)
    assert bool(np.any(passed)), (
        "the one-ulp twins never all reached the horizon distance"
    )
    horizon = int(np.argmax(passed))
    envelope = np.max(twin_matrix[:, :horizon], axis=0)
    ratio = np.divide(
        cross[:horizon], envelope, out=np.zeros(horizon), where=envelope > 0
    )
    worst = int(np.argmax(ratio))
    assert np.all(cross[:horizon] <= envelope), (
        f"evaluation {worst}: lanes {cross[worst]:.3e} apart, over their own one-ulp "
        f"envelope {envelope[worst]:.3e} (horizon {horizon})"
    )
    print(
        f"horizon {horizon} evaluations; worst cross/envelope {ratio[worst]:.3f} at evaluation {worst}"
    )
