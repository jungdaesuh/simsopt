"""The Boozer example selects the official rough optimizer contract."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import numpy as np
import pytest
from simsopt.configs import get_data
from simsopt.geo import Area, BoozerSurface, SurfaceXYZTensorFourier
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_CONSTRAINT_WEIGHT,
    OFFICIAL_LBFGS_MAXITER,
    OFFICIAL_LS_MAXITER,
    OFFICIAL_SOLVER_TOLERANCE,
    BoozerStageState,
    boozer_first_stage_budget,
    boozer_official_options,
    run_boozer_manual_stage,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX


@pytest.mark.parametrize("rough_maxiter", [60, 300])
def test_official_boozer_options_dispatch_limited_memory_driver(
    rough_maxiter: int,
) -> None:
    _, _, axis, nfp, native_field = get_data("ncsx")
    surface = SurfaceXYZTensorFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 3, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 3, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.10, flip_theta=True)
    area = Area(surface)
    solver = BoozerSurfaceJAX(
        BiotSavartJAX(native_field.coils),
        surface,
        area,
        float(area.J()),
        constraint_weight=100.0,
        options=boozer_official_options(
            rough_maxiter=rough_maxiter,
            ls_maxiter=100,
            tolerance=1.0e-10,
        ),
    )

    method = solver._resolve_optimizer_method(optimize_G=True)
    assert method == "lbfgs-ondevice"
    assert solver.options["limited_memory"] is True
    assert solver.options["bfgs_maxiter"] == rough_maxiter
    assert solver.options["newton_maxiter"] == 100
    assert solver._collect_optimizer_options(method=method) == {
        "maxcor": 200,
        "ftol": 1.0e-10,
        "maxfun": 15000,
        "maxls": 20,
    }


class _OneDofSurface:
    """Surface stand-in with the real ``get_dofs``/``set_dofs`` arity."""

    def __init__(self, dofs: np.ndarray) -> None:
        self._dofs = np.asarray(dofs, dtype=np.float64).copy()

    def get_dofs(self) -> np.ndarray:
        return self._dofs.copy()

    def set_dofs(self, dofs: np.ndarray) -> None:
        self._dofs = np.asarray(dofs, dtype=np.float64).copy()


def _diverging_solver(start_dof: float) -> BoozerSurface:
    """Real native solver whose manual stage raises the gradient norm.

    ``r(x) = [x0**2 - 1, x1 - 0.3, x2 - 0.05]`` starting at ``x0 = 0.1``: the
    first damped normal-equation step (lam = 1) moves ``x0`` to 2.575 and the
    gradient norm from 0.198 to 29, which is exactly the case the repository's
    ``_boozer_iterate_is_persistable`` rejects.
    """
    solver = BoozerSurface.__new__(BoozerSurface)
    solver.need_to_run_code = True
    solver.surface = _OneDofSurface(np.asarray([start_dof], dtype=np.float64))

    def residual_and_jacobian(x, constraint_weight, optimize_G, weight_inv_modB):
        del constraint_weight, weight_inv_modB
        assert optimize_G is True
        values = np.asarray(x, dtype=np.float64)
        residual = np.asarray(
            [values[0] ** 2 - 1.0, values[1] - 0.3, values[2] - 0.05],
            dtype=np.float64,
        )
        jacobian = np.diag(np.asarray([2.0 * values[0], 1.0, 1.0], dtype=np.float64))
        return residual, jacobian

    solver._get_residual_vector_and_jacobian = residual_and_jacobian
    return solver


def test_failed_stage_reports_that_the_provider_did_not_persist_the_iterate() -> None:
    """A reverted stage is reported as a revert and chains the START state.

    Upstream always writes the last iterate (boozersurface.py:603-618 at
    9e027eac3); this repository's native library reverts a failed stage that did
    not reduce the norm, so the stage entry has to say which happened instead of
    letting the next stage silently start somewhere else.
    """
    start = BoozerStageState(
        surface_dofs=np.asarray([0.1], dtype=np.float64),
        iota=0.3,
        G=0.05,
    )
    solver = _diverging_solver(0.1)

    outcome = run_boozer_manual_stage(
        solver,
        start,
        tol=OFFICIAL_SOLVER_TOLERANCE,
        maxiter=1,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
    )

    assert outcome.success is False
    assert outcome.provider_persisted_iterate is False
    np.testing.assert_array_equal(outcome.state.surface_dofs, start.surface_dofs)
    assert outcome.state.iota == start.iota
    assert outcome.state.G == start.G


def test_converged_stage_reports_that_the_provider_persisted_the_iterate() -> None:
    start = BoozerStageState(
        surface_dofs=np.asarray([1.05], dtype=np.float64),
        iota=0.3,
        G=0.05,
    )
    solver = _diverging_solver(1.05)

    outcome = run_boozer_manual_stage(
        solver,
        start,
        tol=OFFICIAL_SOLVER_TOLERANCE,
        maxiter=OFFICIAL_LS_MAXITER,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
    )

    assert outcome.success is True
    assert outcome.provider_persisted_iterate is True
    np.testing.assert_allclose(outcome.state.surface_dofs, [1.0], rtol=0.0, atol=1e-10)


def test_native_default_first_stage_runs_the_official_budget() -> None:
    """The official budget is read from its owner, not rebuilt from max_steps.

    ``rough_maxiter = 3 * max_steps`` reached upstream's 300 only because
    ``native_default_steps`` happens to be ``OFFICIAL_LS_MAXITER``; any other
    step count (``--max-steps``, or a change to either default) silently moved
    the shipped script's official first stage away from
    ``OFFICIAL_LBFGS_MAXITER``, upstream's own first-stage budget.
    """
    assert boozer_first_stage_budget(least_squares_steps=7, native_default=True) == (
        OFFICIAL_LBFGS_MAXITER
    )
    assert (
        boozer_first_stage_budget(
            least_squares_steps=OFFICIAL_LS_MAXITER,
            native_default=True,
        )
        == OFFICIAL_LBFGS_MAXITER
    )


def test_reduced_scale_first_stage_keeps_the_official_budget_ratio() -> None:
    """A quarter of the official least-squares budget buys a quarter of 300."""
    quarter = boozer_first_stage_budget(
        least_squares_steps=OFFICIAL_LS_MAXITER // 4,
        native_default=False,
    )
    assert 4 * quarter == OFFICIAL_LBFGS_MAXITER
    assert (
        boozer_first_stage_budget(
            least_squares_steps=OFFICIAL_LS_MAXITER,
            native_default=False,
        )
        == OFFICIAL_LBFGS_MAXITER
    )
