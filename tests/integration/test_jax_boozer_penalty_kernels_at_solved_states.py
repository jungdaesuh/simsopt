"""The ``2_Intermediate/boozer.py`` mirror's penalty kernels at SOLVED Boozer surfaces.

The JAX mirror's Levenberg-Marquardt stages drive its penalty residual and that
residual's Jacobian to zero.  ``tests/jax/test_mirror_same_state_parity.py``
compares both with the native library at the fitted start; this file compares
them at the area and flux end states of a live native run of upstream's workflow
(a first L-BFGS-B stage, then the area-labelled and the flux-labelled manual
least-squares solves, all through the native ``BoozerSurface``), at the reduced
``mpol = ntor = 2`` and the official ``mpol = ntor = 5`` resolutions.  The first
stage runs :data:`FIRST_STAGE_MAXITER` of upstream's 300 iterations to keep the
test small; the two least-squares stages run upstream's own budget and must
converge, so each state is a solved surface.  The tolerance is the one the
mirror's workflow parity applied at the initial state, not widened.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import Area, BoozerSurface, SurfaceXYZTensorFourier, ToroidalFlux
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_CONSTRAINT_WEIGHT,
    OFFICIAL_FLUX_MULTIPLIER,
    OFFICIAL_INITIAL_IOTA,
    OFFICIAL_LBFGS_MAXITER,
    OFFICIAL_LS_MAXITER,
    OFFICIAL_SOLVER_TOLERANCE,
    OFFICIAL_SURFACE_DISTANCE,
    OFFICIAL_SURFACE_RESOLUTION,
    BoozerStageOutcome,
    BoozerStageState,
    boozer_official_options,
    run_boozer_lbfgs_stage,
    run_boozer_manual_stage,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

SCALES: tuple[ExecutionScale, ...] = ("bounded", "native_default")
#: Surface resolution ``mpol = ntor`` per scale: the reduced one and upstream's.
SURFACE_RESOLUTION: dict[ExecutionScale, int] = {
    "bounded": 2,
    "native_default": OFFICIAL_SURFACE_RESOLUTION,
}
#: Iterations of the live native first stage: a tenth of upstream's budget.
FIRST_STAGE_MAXITER = OFFICIAL_LBFGS_MAXITER // 10


@dataclass(frozen=True)
class _Problem:
    """NCSX coils and the fitted tensor-Fourier start surface, as ``boozer.py`` builds them."""

    native_field: BiotSavart
    field: BiotSavart
    surface: SurfaceXYZTensorFourier
    G0: float


def _problem(scale: ExecutionScale) -> _Problem:
    _, base_currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    resolution = SURFACE_RESOLUTION[scale]
    surface = SurfaceXYZTensorFourier(
        mpol=resolution,
        ntor=resolution,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 2 * resolution + 1, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 2 * resolution + 1, endpoint=False),
    )
    surface.fit_to_curve(magnetic_axis, OFFICIAL_SURFACE_DISTANCE, flip_theta=True)
    return _Problem(
        native_field=native_field,
        field=BiotSavart(native_field.coils),
        surface=surface,
        G0=float(2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))),
    )


@dataclass(frozen=True)
class _SolvedStage:
    """One solved stage: its end state ``(surface dofs, iota, G)`` and its label target."""

    state: tuple[float, ...]
    target: float


def _stage_vector(outcome: BoozerStageOutcome) -> tuple[float, ...]:
    return tuple(
        np.concatenate(
            (outcome.state.surface_dofs, [outcome.state.iota, outcome.state.G])
        ).tolist()
    )


@lru_cache(maxsize=None)
def _native_workflow(scale: ExecutionScale) -> dict[str, _SolvedStage]:
    """Upstream's ``boozer.py`` stages with the native library, on fresh objects."""
    problem = _problem(scale)
    surface = problem.surface
    area = Area(surface)
    area_target = float(area.J())
    solver = BoozerSurface(problem.native_field, surface, area, area_target)
    rough = run_boozer_lbfgs_stage(
        solver,
        BoozerStageState(
            surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
            iota=OFFICIAL_INITIAL_IOTA,
            G=problem.G0,
        ),
        tol=OFFICIAL_SOLVER_TOLERANCE,
        maxiter=FIRST_STAGE_MAXITER,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
    )
    polished = run_boozer_manual_stage(
        solver,
        rough.state,
        tol=OFFICIAL_SOLVER_TOLERANCE,
        maxiter=OFFICIAL_LS_MAXITER,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
    )
    toroidal_flux = ToroidalFlux(surface, problem.field)
    flux_target = OFFICIAL_FLUX_MULTIPLIER * float(toroidal_flux.J())
    expanded = run_boozer_manual_stage(
        BoozerSurface(problem.native_field, surface, toroidal_flux, flux_target),
        polished.state,
        tol=OFFICIAL_SOLVER_TOLERANCE,
        maxiter=OFFICIAL_LS_MAXITER,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
    )
    assert polished.success and expanded.success
    return {
        "area": _SolvedStage(_stage_vector(polished), area_target),
        "flux": _SolvedStage(_stage_vector(expanded), flux_target),
    }


@pytest.mark.parametrize("stage", ("area", "flux"))
@pytest.mark.parametrize("scale", SCALES)
def test_jax_penalty_residual_and_jacobian_equal_native_at_a_solved_end_state(
    scale: ExecutionScale,
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    solved = _native_workflow(scale)[stage]
    problem = _problem(scale)
    surface = problem.surface
    label = Area(surface) if stage == "area" else ToroidalFlux(surface, problem.field)
    state = np.asarray(solved.state, dtype=np.float64)

    native_residual, native_jacobian = BoozerSurface(
        problem.native_field, surface, label, solved.target
    )._get_residual_vector_and_jacobian(state, OFFICIAL_CONSTRAINT_WEIGHT, True, True)

    jax_field = BiotSavartJAX(problem.native_field.coils)
    solver = BoozerSurfaceJAX(
        jax_field,
        surface,
        label,
        solved.target,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
        options=boozer_official_options(
            rough_maxiter=OFFICIAL_LBFGS_MAXITER,
            ls_maxiter=OFFICIAL_LS_MAXITER,
            tolerance=OFFICIAL_SOLVER_TOLERANCE,
        ),
    )
    kernels = solver._get_penalty_kernel_bundle(
        optimize_G=True,
        weight_inv_modB=True,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
    )
    coil_spec = jax_field.coil_set_spec()
    device_state = jnp.asarray(jax.device_put(state))
    jax_residual, jax_jacobian = jax.device_get(
        (
            kernels.residual(device_state, coil_spec),
            kernels.jacobian(device_state, coil_spec),
        )
    )

    np.testing.assert_allclose(
        jax_residual, native_residual, rtol=1.0e-11, atol=1.0e-13
    )
    np.testing.assert_allclose(
        jax_jacobian, native_jacobian, rtol=1.0e-11, atol=1.0e-13
    )
