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
  (``x0' = nextafter(x0, s inf)``, ``s`` from ``RandomState(20260920 + k)``,
  ``k = 1..8``); the envelope is the largest of the sixteen twin distances;
* the distance between two runs at evaluation ``k`` is the larger, over the
  current and the geometry dofs, of ``max |dx| / max |x|`` within that block
  (the currents are ``1e5``, the geometry ``O(1)``);
* evaluations are compared only while the two runs are in the same minimize
  call, and the horizon is the first evaluation at which EVERY twin distance has
  reached ``1e-6``; beyond it the lanes' own scatter has left the linear regime.

Both lanes are built live: the native lane is upstream's two-stage L-BFGS-B
over the native planar objective, the JAX lane is
:func:`simsopt_jax.examples.solve_standard_stage_two`, whose SciPy calls are
traced through :data:`simsopt_jax.solve.dispatch.scipy_minimize`.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import jax
import numpy as np
import pytest
import scipy.optimize
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    LinkingNumber,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    create_equally_spaced_planar_curves,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import solve_standard_stage_two
from simsopt_jax.objectives import StageTwoObjectiveConfig
from simsopt_jax.solve import dispatch
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

pytestmark = pytest.mark.slow

TWINS = tuple(range(1, 9))
HORIZON_DISTANCE = 1.0e-6
CURRENT_DOFS = 3

SURFACE_INPUT = (
    Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
)

#: The reduced scale of ``stage_two_optimization_planar_coils.py``.
SURFACE_RESOLUTION = 4
CURVE_ORDER = 2
CURVE_QUADRATURE = 32
NUM_BASE_CURVES = 4
MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.5
INITIAL_CURRENT = 1.0e5
LENGTH_TARGET = 10.4
FIRST_LENGTH_WEIGHT = 10.0
SECOND_LENGTH_WEIGHT = 1.0
CURVE_CURVE_THRESHOLD = 0.08
CURVE_CURVE_WEIGHT = 1000.0
CURVE_SURFACE_THRESHOLD = 0.12
CURVE_SURFACE_WEIGHT = 10.0
CURVATURE_THRESHOLD = 10.0
CURVATURE_WEIGHT = 1.0e-6
MEAN_SQUARED_CURVATURE_THRESHOLD = 10.0
MEAN_SQUARED_CURVATURE_WEIGHT = 1.0e-6
LINKING_NUMBER_WEIGHT = 1.0
MAX_STEPS = 50
RTOL = 1.0e-15
ATOL = 1.0e-15
LBFGS_HISTORY = 300


def one_ulp_start(parameters: np.ndarray, k: int) -> np.ndarray:
    """Upstream's one-ulp protocol draw ``k`` of a start vector."""
    signs = np.random.RandomState(20260920 + k).choice(
        [-1.0, 1.0], size=parameters.size
    )
    return np.nextafter(parameters, signs * np.inf)


def _geometry():
    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        range="half period",
        nphi=SURFACE_RESOLUTION,
        ntheta=SURFACE_RESOLUTION,
    )
    surface.fix_all()
    base_curves = create_equally_spaced_planar_curves(
        NUM_BASE_CURVES,
        surface.nfp,
        stellsym=True,
        R0=MAJOR_RADIUS,
        R1=MINOR_RADIUS,
        order=CURVE_ORDER,
        numquadpoints=CURVE_QUADRATURE,
    )
    base_currents = [Current(INITIAL_CURRENT) for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    return surface, base_curves, coils


def _run_native(initial: np.ndarray, _direction: np.ndarray) -> None:
    """The native two-stage workflow; its SciPy calls go through ``scipy.optimize``."""
    surface, base_curves, coils = _geometry()
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    curves = [coil.curve for coil in coils]
    flux = SquaredFlux(surface, field)
    length_penalty = QuadraticPenalty(
        sum(CurveLength(curve) for curve in base_curves), LENGTH_TARGET
    )
    curve_curve = CurveCurveDistance(
        curves, CURVE_CURVE_THRESHOLD, num_basecurves=NUM_BASE_CURVES
    )
    curve_surface = CurveSurfaceDistance(curves, surface, CURVE_SURFACE_THRESHOLD)
    curvature = sum(
        LpCurveCurvature(curve, 2, CURVATURE_THRESHOLD) for curve in base_curves
    )
    mean_squared_curvature = sum(
        QuadraticPenalty(MeanSquaredCurvature(curve), MEAN_SQUARED_CURVATURE_THRESHOLD)
        for curve in base_curves
    )
    linking_number = LinkingNumber(curves)

    def weighted(length_weight: float) -> Optimizable:
        return (
            flux
            + length_weight * length_penalty
            + CURVE_CURVE_WEIGHT * curve_curve
            + CURVE_SURFACE_WEIGHT * curve_surface
            + CURVATURE_WEIGHT * curvature
            + MEAN_SQUARED_CURVATURE_WEIGHT * mean_squared_curvature
            + LINKING_NUMBER_WEIGHT * linking_number
        )

    def solve(current: Optimizable, start: np.ndarray) -> np.ndarray:
        def value_and_gradient(parameters: np.ndarray):
            current.x = parameters
            return float(current.J()), np.asarray(current.dJ(), dtype=np.float64)

        result = scipy.optimize.minimize(
            value_and_gradient,
            start,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": MAX_STEPS,
                "maxcor": LBFGS_HISTORY,
                "ftol": RTOL,
                "gtol": ATOL,
            },
        )
        return np.asarray(result.x, dtype=np.float64)

    # The Taylor test evaluates outside ``minimize`` and cannot move a trace.
    first_parameters = solve(weighted(FIRST_LENGTH_WEIGHT), initial)
    solve(weighted(SECOND_LENGTH_WEIGHT), first_parameters)


def _run_jax(initial: np.ndarray, direction: np.ndarray) -> None:
    """The JAX workflow; its SciPy calls go through ``dispatch.scipy_minimize``."""
    surface, _base_curves, coils = _geometry()
    field = BiotSavartJAX(coils)
    device = get_runtime_jax_device()

    def put(value: object) -> jax.Array:
        return jax.device_put(np.asarray(value, dtype=np.float64), device)

    solve_standard_stage_two(
        field=field,
        flux_spec=SquaredFluxJAX(surface, field).fixed_surface_flux_spec(),
        surface_gamma=put(surface.gamma().reshape((-1, 3))),
        surface_normal=put(surface.normal().reshape((-1, 3))),
        initial_parameters=put(initial),
        taylor_direction=put(direction),
        regularization_config=StageTwoObjectiveConfig(
            num_base_curves=NUM_BASE_CURVES,
            length_target=LENGTH_TARGET,
            length_target_mode="identity",
            curve_curve_minimum_distance=CURVE_CURVE_THRESHOLD,
            curve_curve_weight=CURVE_CURVE_WEIGHT,
            curve_surface_minimum_distance=CURVE_SURFACE_THRESHOLD,
            curve_surface_weight=CURVE_SURFACE_WEIGHT,
            curvature_threshold=CURVATURE_THRESHOLD,
            curvature_weight=CURVATURE_WEIGHT,
            mean_squared_curvature_threshold=MEAN_SQUARED_CURVATURE_THRESHOLD,
            mean_squared_curvature_target_mode="identity",
            mean_squared_curvature_weight=MEAN_SQUARED_CURVATURE_WEIGHT,
            linking_number_weight=LINKING_NUMBER_WEIGHT,
        ),
        first_length_weight=put(FIRST_LENGTH_WEIGHT),
        second_length_weight=put(SECOND_LENGTH_WEIGHT),
        max_steps=MAX_STEPS,
        rtol=RTOL,
        atol=ATOL,
        driver=Driver.SCIPY_LBFGSB,
    )


@dataclass(frozen=True)
class Trace:
    """Every objective evaluation of one run: minimize-call index and iterate."""

    call: np.ndarray
    x: np.ndarray


def _trace(
    run: Callable[[], object],
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    name: str,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = np.asarray(BiotSavart(_geometry()[2]).x, dtype=np.float64)
    direction = np.random.RandomState(1).uniform(size=start.shape)

    def lane_runs(
        lane: Callable[[np.ndarray, np.ndarray], None],
        module: ModuleType,
        name: str,
    ) -> tuple[Trace, list[Trace]]:
        def run_from(parameters: np.ndarray) -> Callable[[], None]:
            return lambda: lane(parameters, direction)

        base = _trace(run_from(start), monkeypatch, module, name)
        twins = [
            _trace(run_from(one_ulp_start(start, k)), monkeypatch, module, name)
            for k in TWINS
        ]
        return base, twins

    native, native_twins = lane_runs(_run_native, scipy.optimize, "minimize")
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax_lane, jax_twins = lane_runs(_run_jax, dispatch, "scipy_minimize")

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
