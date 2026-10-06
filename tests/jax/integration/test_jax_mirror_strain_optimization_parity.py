"""Exact parity for the ``2_Intermediate/strain_optimization.py`` mirror.

Both lanes are built live at the reduced (bounded) iteration budget: the native
lane is upstream's L-BFGS-B over the native strain penalties of a centroid
frame on the scaled HSX coil, the JAX lane is
:func:`simsopt_jax.examples.solve_strain_rotation` over the same fixed curve.
No reference data is stored.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import dataclass

import jax
import numpy as np
import pytest
from scipy.optimize import minimize
from simsopt.configs import get_data
from simsopt.geo import (
    CoilStrain,
    CurveXYZFourier,
    FramedCurveCentroid,
    FrameRotation,
    LPBinormalCurvatureStrainPenalty,
    LPTorsionalStrainPenalty,
)
from simsopt_contracts.optimization_endpoint import (
    StatusConvention,
    TerminalStatus,
    certify_optimization_endpoint,
    normalized_terminal_status,
)
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import solve_strain_rotation

COIL_INDEX = 1
COIL_ORDER = 10
POINTS_PER_PERIOD = 10
SCALE_FACTOR = 0.1
ROTATION_ORDER = 10
OBJECTIVE_WIDTH = 1.0e-3
REPORTING_WIDTH = 3.0e-3
TORSIONAL_THRESHOLD = 2.0e-3
CURVATURE_THRESHOLD = 2.0e-3
#: The reduced iteration budget; the rest is the official call's policy.
MAXITER = 50
MAXFUN = 15000
MAXCOR = 10
MAXLS = 20
GTOL = 1.0e-20
FTOL = 1.0e-20
SCIENTIFIC_GRADIENT_TOLERANCE = 1.0e-8

_STATE_OBSERVABLES = (
    "parameters",
    "objective",
    "gradient",
    "torsional_strain",
    "binormal_curvature_strain",
    "maximum_torsional_strain",
    "maximum_binormal_curvature_strain",
)


def _fixed_curve() -> CurveXYZFourier:
    """Coil 1 of HSX, scaled by 0.1 and fixed, as the official script builds it."""
    source = get_data(
        "hsx", coil_order=COIL_ORDER, points_per_period=POINTS_PER_PERIOD
    )[0][COIL_INDEX]
    curve = CurveXYZFourier(source.quadpoints, source.order)
    curve.x = np.asarray(source.x, dtype=np.float64) * SCALE_FACTOR
    curve.fix_all()
    return curve


def _construction_values(curve: CurveXYZFourier) -> dict[str, np.ndarray]:
    return {
        "construction:quadpoints": np.array(curve.quadpoints, dtype=np.float64),
        "construction:gamma": np.array(curve.gamma(), dtype=np.float64),
        "construction:gammadash": np.array(curve.gammadash(), dtype=np.float64),
        "construction:gammadashdash": np.array(curve.gammadashdash(), dtype=np.float64),
    }


@dataclass(frozen=True)
class _LaneRun:
    values: dict[str, np.ndarray]
    status_convention: StatusConvention
    success: bool
    status: int
    iterations: int


def _terminal_status(run: _LaneRun) -> TerminalStatus:
    """The library's fold of the stop and the case's scientific predicate."""
    values = run.values
    final_gradient = values["final:gradient"]
    reason = certify_optimization_endpoint(
        status_convention=run.status_convention,
        provider_success=run.success,
        provider_status=run.status,
        iterations=run.iterations,
        max_iterations=MAXITER,
        initial_gradient_inf_norm=float(np.max(np.abs(values["initial:gradient"]))),
        final_gradient_inf_norm=float(np.max(np.abs(final_gradient))),
        parameters_finite=bool(np.all(np.isfinite(values["final:parameters"]))),
        observables_finite=bool(np.isfinite(values["final:objective"])),
        inner_success=True,
    ).stopping_reason
    scientific_predicate = bool(
        np.isfinite(values["final:objective"])
        and values["final:objective"] < values["initial:objective"]
        and np.all(np.isfinite(final_gradient))
        and np.linalg.norm(final_gradient, ord=np.inf) <= SCIENTIFIC_GRADIENT_TOLERANCE
    )
    return normalized_terminal_status(
        scientific_predicate=scientific_predicate,
        stage_stopping_reasons=(reason,),
    )


def _run_native(initial: np.ndarray) -> _LaneRun:
    curve = _fixed_curve()
    framed_curve = FramedCurveCentroid(
        curve, FrameRotation(curve.quadpoints, ROTATION_ORDER)
    )
    strain = CoilStrain(framed_curve, REPORTING_WIDTH)
    objective = LPTorsionalStrainPenalty(
        framed_curve, width=OBJECTIVE_WIDTH, p=2, threshold=TORSIONAL_THRESHOLD
    ) + LPBinormalCurvatureStrainPenalty(
        framed_curve, width=OBJECTIVE_WIDTH, p=2, threshold=CURVATURE_THRESHOLD
    )

    def state(prefix: str, parameters: np.ndarray) -> dict[str, np.ndarray]:
        objective.x = parameters
        torsional = np.asarray(strain.torsional_strain(), dtype=np.float64)
        binormal = np.asarray(strain.binormal_curvature_strain(), dtype=np.float64)
        return {
            f"{prefix}:parameters": np.asarray(parameters, dtype=np.float64),
            f"{prefix}:objective": np.asarray(objective.J(), dtype=np.float64),
            f"{prefix}:gradient": np.asarray(objective.dJ(), dtype=np.float64),
            f"{prefix}:torsional_strain": torsional,
            f"{prefix}:binormal_curvature_strain": binormal,
            f"{prefix}:maximum_torsional_strain": np.asarray(np.max(torsional)),
            f"{prefix}:maximum_binormal_curvature_strain": np.asarray(np.max(binormal)),
        }

    initial_values = state("initial", initial)

    def value_and_gradient(parameters: np.ndarray):
        objective.x = parameters
        return float(objective.J()), np.asarray(objective.dJ(), dtype=np.float64)

    result = minimize(
        value_and_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        tol=GTOL,
        options={
            "maxiter": MAXITER,
            "maxfun": MAXFUN,
            "maxcor": MAXCOR,
            "maxls": MAXLS,
            "gtol": GTOL,
            "ftol": FTOL,
        },
    )
    return _LaneRun(
        values={
            **_construction_values(curve),
            **initial_values,
            **state("final", np.asarray(result.x, dtype=np.float64)),
        },
        status_convention="scipy-lbfgsb",
        success=bool(result.success),
        status=int(result.status),
        iterations=int(result.nit),
    )


def _run_jax(initial: np.ndarray) -> _LaneRun:
    curve = _fixed_curve()
    construction = _construction_values(curve)
    device = get_runtime_jax_device()

    def put(value: np.ndarray) -> jax.Array:
        return jax.device_put(value, device)

    result = jax.device_get(
        solve_strain_rotation(
            quadpoints=put(construction["construction:quadpoints"]),
            gamma=put(construction["construction:gamma"]),
            gammadash=put(construction["construction:gammadash"]),
            gammadashdash=put(construction["construction:gammadashdash"]),
            initial_parameters=put(initial),
            rotation_order=ROTATION_ORDER,
            objective_width=OBJECTIVE_WIDTH,
            reporting_width=REPORTING_WIDTH,
            torsional_threshold=TORSIONAL_THRESHOLD,
            curvature_threshold=CURVATURE_THRESHOLD,
            maxiter=MAXITER,
            maxfun=MAXFUN,
            gtol=GTOL,
            ftol=FTOL,
            maxcor=MAXCOR,
            maxls=MAXLS,
        )
    )
    values = dict(construction)
    for prefix, state in (("initial", result.initial), ("final", result.final)):
        values.update(
            {
                f"{prefix}:{name}": np.asarray(getattr(state, name), dtype=np.float64)
                for name in _STATE_OBSERVABLES
            }
        )
    return _LaneRun(
        values=values,
        # The on-device L-BFGS-B publishes the private L-BFGS-B status table.
        status_convention="private-lbfgsb",
        success=bool(result.success),
        status=int(result.status),
        iterations=int(result.iterations),
    )


def test_exact_strain_optimization_matches_native_and_jax_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = np.zeros(2 * ROTATION_ORDER + 1, dtype=np.float64)

    native = _run_native(initial)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax_lane = _run_jax(initial)

    # The bounded budget ends every stage on its iteration cap. The label says
    # so; it is neither convergence nor failure, and it still implies the case's
    # scientific predicate, because a false predicate is labelled ``failed``.
    for run in (native, jax_lane):
        terminal = _terminal_status(run)
        assert terminal.normalized_status == "budget_exhausted"
        assert terminal.success is False
    assert set(native.values) == set(jax_lane.values)

    for observable in (
        "construction:quadpoints",
        "construction:gamma",
        "construction:gammadash",
        "construction:gammadashdash",
        "initial:parameters",
    ):
        np.testing.assert_allclose(
            jax_lane.values[observable],
            native.values[observable],
            rtol=1.0e-13,
            atol=1.0e-14,
        )

    np.testing.assert_allclose(
        jax_lane.values["initial:objective"],
        native.values["initial:objective"],
        rtol=2.0e-7,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        jax_lane.values["initial:gradient"],
        native.values["initial:gradient"],
        rtol=2.0e-7,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        jax_lane.values["final:parameters"],
        native.values["final:parameters"],
        rtol=5.0e-3,
        atol=2.0e-4,
    )
    np.testing.assert_allclose(
        jax_lane.values["final:objective"],
        native.values["final:objective"],
        rtol=2.0e-6,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        jax_lane.values["final:gradient"],
        native.values["final:gradient"],
        rtol=1.0,
        atol=3.0e-9,
    )
    for observable in (
        "final:torsional_strain",
        "final:binormal_curvature_strain",
        "final:maximum_torsional_strain",
        "final:maximum_binormal_curvature_strain",
    ):
        np.testing.assert_allclose(
            jax_lane.values[observable],
            native.values[observable],
            rtol=2.0e-5,
            atol=5.0e-8,
        )

    assert float(native.values["final:objective"]) < float(
        native.values["initial:objective"]
    )
    assert float(jax_lane.values["final:objective"]) < float(
        jax_lane.values["initial:objective"]
    )
