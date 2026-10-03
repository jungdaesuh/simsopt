"""JAX mirror of ``examples/1_Simple/qfm.py``.

Host construction loads the canonical NCSX coils and fitted surface. Host SciPy
controls the reference L-BFGS-B/SLSQP sequence while compiled JAX kernels
evaluate every QFM value and derivative on the selected JAX device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from simsopt.configs.zoo import get_data
from simsopt.geo import SurfaceRZFourier
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_contracts.optimization_endpoint import (
    StoppingReason,
    scipy_minimize_stopping_reason,
)
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    run_example,
)
from simsopt_jax.examples.qfm_host_scipy import (
    QFM_EXACT_METHOD,
    QFM_HOST_SCIPY_DRIVER,
    QFM_SURFACE_RESOLUTION,
    QfmHostState,
    QfmHostStageResult,
    build_qfm_host_kernels,
    solve_qfm_host_scipy_sequence,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

EXAMPLE_ID = "native-qfm"
#: The penalty stage runs SciPy L-BFGS-B (``QFM_PENALTY_DRIVER =
#: Driver.SCIPY_LBFGSB``, dispatched at
#: ``src/simsopt_jax/solve/dispatch.py:362``); the exact stage's method is the
#: solver module's own ``QFM_EXACT_METHOD``.
PENALTY_METHOD = "L-BFGS-B"
#: Upstream's own budget (``maxiter=1000`` in ``examples/1_Simple/qfm.py``) and
#: the branch's bounded budget.
NATIVE_DEFAULT_STEPS = 1000
BOUNDED_STEPS = 80


@dataclass(frozen=True)
class StageResult:
    label: Literal["volume", "toroidal_flux", "area"]
    penalty_success: bool
    penalty_status: int
    penalty_message: str
    penalty_nit: int
    penalty_nfev: int
    penalty_njev: int
    exact_success: bool
    exact_status: int
    exact_message: str
    exact_nit: int
    exact_nfev: int
    exact_njev: int
    qfm_value: float
    constraint_objective: float
    #: What each of the two SciPy calls stopped for, in the shared contract's
    #: vocabulary.
    penalty_stopping_reason: StoppingReason
    exact_stopping_reason: StoppingReason


def _build_surface(scale: ExecutionScale) -> tuple[SurfaceRZFourier, BiotSavartJAX]:
    _curves, _currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    resolution = QFM_SURFACE_RESOLUTION[scale]
    quadrature_size = resolution.quadrature_size
    phis = np.linspace(0.0, 1.0 / nfp, quadrature_size, endpoint=False)
    thetas = np.linspace(0.0, 1.0, quadrature_size, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=resolution.order,
        ntor=resolution.order,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    return surface, BiotSavartJAX(native_field.coils)


def _endpoint_finite(state: QfmHostState) -> bool:
    """Whether the endpoint one provider call left is finite.

    The parameters the call returned and the objective value at them.
    """
    return bool(np.all(np.isfinite(state.parameters)) and np.isfinite(state.qfm_value))


def _stage_result(
    stage: QfmHostStageResult,
    label: Literal["volume", "toroidal_flux", "area"],
    *,
    max_steps: int,
) -> StageResult:
    label_residual = stage.exact.label_residual_abs
    return StageResult(
        label=label,
        penalty_success=bool(stage.penalty_optimizer.success),
        penalty_status=int(stage.penalty_optimizer.status),
        penalty_message=str(stage.penalty_optimizer.message),
        penalty_nit=int(stage.penalty_optimizer.nit),
        penalty_nfev=int(stage.penalty_optimizer.nfev),
        penalty_njev=int(stage.penalty_optimizer.njev),
        exact_success=bool(stage.exact_optimizer.success),
        exact_status=int(stage.exact_optimizer.status),
        exact_message=str(stage.exact_optimizer.message),
        exact_nit=int(stage.exact_optimizer.nit),
        exact_nfev=int(stage.exact_optimizer.nfev),
        exact_njev=int(stage.exact_optimizer.njev),
        qfm_value=stage.exact.qfm_value,
        constraint_objective=0.5 * label_residual * label_residual,
        # One owner for "what did this provider call stop for": the contract
        # routes each ``scipy.optimize.minimize`` method to its own status
        # vocabulary.
        penalty_stopping_reason=scipy_minimize_stopping_reason(
            method=PENALTY_METHOD,
            provider_success=bool(stage.penalty_optimizer.success),
            provider_status=int(stage.penalty_optimizer.status),
            iterations=int(stage.penalty_optimizer.nit),
            max_iterations=max_steps,
            endpoint_finite=_endpoint_finite(stage.penalty),
        ),
        exact_stopping_reason=scipy_minimize_stopping_reason(
            method=QFM_EXACT_METHOD,
            provider_success=bool(stage.exact_optimizer.success),
            provider_status=int(stage.exact_optimizer.status),
            iterations=int(stage.exact_optimizer.nit),
            max_iterations=max_steps,
            endpoint_finite=_endpoint_finite(stage.exact),
        ),
    )


def _stage_states_finite(stage: QfmHostStageResult) -> bool:
    """Whether every quantity this stage publishes is finite, on full arrays."""
    return all(
        bool(np.all(np.isfinite(value)))
        for state in (stage.initial, stage.penalty, stage.exact)
        for value in (
            state.parameters,
            state.qfm_value,
            state.qfm_gradient,
            state.label_value,
            state.label_gradient,
            state.label_residual_abs,
        )
    )


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    surface, field = _build_surface(scale)
    device = get_runtime_jax_device()
    coil_set_spec = field.coil_set_spec_from_dofs(
        explicit_device_array(field.x, dtype=np.float64, device=device)
    )
    initial_parameters = np.asarray(surface.get_dofs(), dtype=np.float64)
    kernels = build_qfm_host_kernels(
        initial_parameters=initial_parameters,
        quadpoints_phi=np.asarray(surface.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(surface.quadpoints_theta, dtype=np.float64),
        coil_set_spec=coil_set_spec,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )
    result = solve_qfm_host_scipy_sequence(
        initial_parameters,
        kernels=kernels,
        max_steps=max_steps,
        tolerance=1.0e-12,
        constraint_weight=1.0,
    )
    initial = result.volume.initial
    final = result.area.exact
    initial_label_residual = float(initial.label_value - result.volume.target)
    final_label_residual = float(final.label_value - result.area.target)
    initial_constraint = 0.5 * initial_label_residual * initial_label_residual
    final_constraint = 0.5 * final_label_residual * final_label_residual
    initial_residuals = np.asarray(
        (initial.qfm_value, initial_constraint),
        dtype=np.float64,
    )
    initial_jacobian = np.stack(
        (
            initial.qfm_gradient,
            initial_label_residual * initial.label_gradient,
        )
    )
    final_residuals = np.asarray(
        (final.qfm_value, final_constraint),
        dtype=np.float64,
    )
    final_jacobian = np.stack(
        (
            final.qfm_gradient,
            final_label_residual * final.label_gradient,
        )
    )
    stages = (
        _stage_result(result.volume, "volume", max_steps=max_steps),
        _stage_result(result.toroidal_flux, "toroidal_flux", max_steps=max_steps),
        _stage_result(result.area, "area", max_steps=max_steps),
    )
    # Three conditions, none of them a threshold of our own. Every one of the
    # six SciPy calls must have stopped for the reason ``converged`` in the
    # shared contract's vocabulary -- which is stricter than the raw
    # ``success`` flags this used to read, because a flag that contradicts the
    # reported status fails closed there; every quantity every stage publishes
    # must be finite (checked on the full arrays, before anything selects from
    # them); and the QFM value must have decreased, which is what the official
    # run does, by four orders of magnitude from the initial QFM value to the
    # exact area stage's.
    # The branch-added constraint gate (``final_constraint <= 1e-10``) stays
    # demoted to the published ``constraint_residual``: the official run itself
    # ends with exact-stage label residuals that would fail it.
    solver_success = bool(
        all(
            stage.penalty_stopping_reason == "converged"
            and stage.exact_stopping_reason == "converged"
            for stage in stages
        )
        and all(
            _stage_states_finite(stage)
            for stage in (result.volume, result.toroidal_flux, result.area)
        )
        and final.qfm_value < initial.qfm_value
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_parameters": tuple(float(value) for value in initial.parameters),
            "initial_residuals": tuple(float(value) for value in initial_residuals),
            "initial_jacobian": tuple(
                tuple(float(value) for value in row) for row in initial_jacobian
            ),
            "solution": tuple(float(value) for value in final.parameters),
            "final_residuals": tuple(float(value) for value in final_residuals),
            "final_jacobian": tuple(
                tuple(float(value) for value in row) for row in final_jacobian
            ),
            "constraint_residual": final_constraint,
            "solver_success": solver_success,
            "driver": QFM_HOST_SCIPY_DRIVER,
            "optimizer_execution": "host_scipy",
            "physics_execution": "jax_device",
            "physics_platform": device.platform,
            "stages": tuple(
                {
                    "label": stage.label,
                    "penalty_success": stage.penalty_success,
                    "penalty_status": stage.penalty_status,
                    "penalty_message": stage.penalty_message,
                    "penalty_nit": stage.penalty_nit,
                    "penalty_nfev": stage.penalty_nfev,
                    "penalty_njev": stage.penalty_njev,
                    "exact_success": stage.exact_success,
                    "exact_status": stage.exact_status,
                    "exact_message": stage.exact_message,
                    "exact_nit": stage.exact_nit,
                    "exact_nfev": stage.exact_nfev,
                    "exact_njev": stage.exact_njev,
                    "qfm_value": stage.qfm_value,
                    "constraint_objective": stage.constraint_objective,
                    "penalty_stopping_reason": stage.penalty_stopping_reason,
                    "exact_stopping_reason": stage.exact_stopping_reason,
                }
                for stage in stages
            ),
        },
        status="ok" if solver_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-qfm-",
        bounded_steps=BOUNDED_STEPS,
        native_default_steps=NATIVE_DEFAULT_STEPS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
