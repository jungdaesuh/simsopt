"""Official manual Boozer damping policy at analytic and shipped scales."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from simsopt.configs import get_data
from simsopt.geo import Area, SurfaceXYZTensorFourier
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX


def test_manual_step_takes_finite_cost_increase() -> None:
    solver = object.__new__(BoozerSurfaceJAX)

    def residual(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([x[0] ** 2 - 1.0], dtype=x.dtype)

    result = solver._run_manual_penalty_least_squares(
        residual,
        jnp.asarray([0.1], dtype=jnp.float64),
        tol=1.0e-10,
        maxiter=1,
    )

    # Upstream's lam=1 normal-equation step moves 0.1 to 2.575 even though
    # this first candidate raises the residual cost.
    np.testing.assert_allclose(np.asarray(result["x"]), [2.575], rtol=1.0e-14)
    assert result["nit"] == 1
    assert result["success"] is False


def test_manual_nonfinite_candidate_ends_the_loop_before_its_budget() -> None:
    """A NaN state, not the iteration budget, is what stops the loop.

    Upstream's ``while i < maxiter and norm > tol`` (boozersurface.py:594) stops
    on a NaN norm because ``nan > tol`` is False, and the port's ``cond_fn``
    keeps that rule.  The budget here is 5, so ``nit == 1`` proves the
    non-finite state ended it.
    """
    solver = object.__new__(BoozerSurfaceJAX)
    budget = 5

    def residual(x: jnp.ndarray) -> jnp.ndarray:
        value = jnp.where(x[0] > 0.5, jnp.nan, x[0] ** 2 - 1.0)
        return jnp.asarray([value], dtype=x.dtype)

    result = solver._run_manual_penalty_least_squares(
        residual,
        jnp.asarray([0.1], dtype=jnp.float64),
        tol=1.0e-10,
        maxiter=budget,
    )

    assert result["nit"] == 1
    assert result["nit"] < budget
    assert result["success"] is False
    assert np.isnan(np.asarray(result["residual"])).all()


def test_manual_nonfinite_iterate_at_zero_gradient_is_not_success() -> None:
    """The finiteness accumulator, not ``norm <= tol``, is what fails here.

    The rank-deficient normal matrix makes the first damped step non-finite,
    and at a non-finite iterate this residual and its Jacobian are identically
    zero, so ``norm == 0.0 <= tol`` holds.  Only ``all_finite`` keeps the loop
    from reporting success on a NaN iterate: delete it from the success
    expression (boozer_surface.py) and this test fails.
    """
    solver = object.__new__(BoozerSurfaceJAX)
    budget = 5

    def residual(x: jnp.ndarray) -> jnp.ndarray:
        finite = jnp.isfinite(x[0]) & jnp.isfinite(x[1])
        return jnp.asarray(
            [jnp.where(finite, x[0], 0.0), jnp.where(finite, 1.0, 0.0)],
            dtype=x.dtype,
        )

    result = solver._run_manual_penalty_least_squares(
        residual,
        jnp.asarray([2.0, 1.0], dtype=jnp.float64),
        tol=1.0e-10,
        maxiter=budget,
    )

    assert result["nit"] == 1
    assert result["nit"] < budget
    assert not np.all(np.isfinite(np.asarray(result["x"])))
    assert float(np.linalg.norm(np.asarray(result["gradient"]))) == 0.0
    assert result["success"] is False


def test_manual_damping_sequence_is_upstreams_multiplication_by_one_third() -> None:
    """``lam`` after k steps equals Python's ``lam *= 1/3`` repeated k times.

    Upstream is ``lam *= 1/3`` (boozersurface.py:603): multiplication by the
    rounded constant one third.  ``lam / 3`` is a different double-precision
    sequence -- 95 of these first 100 values differ from it -- so the emitted
    sequence is compared bitwise, not within a tolerance.  ``tol=-1.0`` makes
    ``norm > tol`` always true, so the loop runs exactly ``maxiter`` steps for
    every budget.
    """
    solver = object.__new__(BoozerSurfaceJAX)

    def residual(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([x[0] - 1.0], dtype=x.dtype)

    expected = 1.0
    for steps in range(1, 101):
        expected *= 1 / 3
        result = solver._run_manual_penalty_least_squares(
            residual,
            jnp.asarray([0.5], dtype=jnp.float64),
            tol=-1.0,
            maxiter=steps,
        )
        assert result["nit"] == steps
        assert result["damping"] == expected, f"damping differs at step {steps}"


def test_shipped_ncsx_area_manual_converges_from_initial_surface() -> None:
    _, base_currents, axis, nfp, native_field = get_data("ncsx")
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    initial_g = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    surface = SurfaceXYZTensorFourier(
        mpol=5,
        ntor=5,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 11, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 11, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.10, flip_theta=True)
    area = Area(surface)
    solver = BoozerSurfaceJAX(
        BiotSavartJAX(native_field.coils),
        surface,
        area,
        float(area.J()),
        constraint_weight=100.0,
        options={"newton_maxiter": 100, "newton_tol": 1.0e-10, "verbose": False},
    )

    result = solver.minimize_boozer_penalty_constraints_ls(
        tol=1.0e-10,
        maxiter=100,
        constraint_weight=100.0,
        iota=-0.4,
        G=initial_g,
        method="manual",
    )

    assert result["success"] is True
    assert np.linalg.norm(np.asarray(result["gradient"])) <= 1.0e-10
    assert np.linalg.norm(np.asarray(result["residual"])) < 1.0e-9
