"""The opt-in smooth self-contact penalty of one closed curve.

Kernel: ``simsopt_jax.core.curve_self_contact.curve_self_contact_penalty_pure``.
These tests check the kernel's contract on its domain (every node speed
positive): finite, finite-difference-consistent first and second derivatives
at an exact coincidence (bitwise-identical node coordinates) and near one,
continuity through the separation ramp and the activation (a hard mask, the
control, jumps), rigid-motion invariance, the straight-strand crossing energy
in the full-weight regime, and a jitted program that runs under the strict
transfer guard.  Outside the domain (a zero-speed node) the value is NaN, and
the SciPy L-BFGS-B route reports a solve that ends there as an explicit
non-finite failure.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize
from simsopt_jax.core.curve_self_contact import (
    curve_self_contact_penalty_pure,
    self_contact_energy,
    self_contact_separation_weight,
)
from simsopt_jax.solve import NONFINITE_RESULT_STATUS, Driver, ScipyLBFGSBOptions
from simsopt_jax.solve.dispatch import minimize
from simsopt_jax.solve.termination import (
    Emitter,
    EmitterStop,
    TerminationLabel,
    termination_report,
)

D = 0.01
W0 = 0.02
W1 = 0.03
FD_STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
# The sweep's activation distance: larger than W1, so ramp-band pairs are active.
SWEEP_D = 0.05


def _penalty(gamma, gammadash, d=D, w0=W0, w1=W1):
    return curve_self_contact_penalty_pure(gamma, gammadash, d, w0, w1)


_penalty_program = jax.jit(_penalty, static_argnums=(2, 3, 4))
_penalty_gradient = jax.jit(
    jax.grad(_penalty, argnums=(0, 1)), static_argnums=(2, 3, 4)
)


def _figure_eight(count: int, lift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Lemniscate of Gerono (~0.3 m); nodes 0 and count/2 sit on the crossing.

    Node ``count / 2`` is assigned node 0's coordinates (bitwise identical),
    then lifted by ``lift`` along z; the strands cross at a right angle.
    """
    theta = 2.0 * np.pi * np.arange(count) / count
    a = 0.15
    gamma = np.stack(
        [a * np.sin(theta), a * np.sin(theta) * np.cos(theta), np.zeros(count)],
        axis=1,
    )
    gammadash = np.stack(
        [
            2.0 * np.pi * a * np.cos(theta),
            2.0 * np.pi * a * np.cos(2.0 * theta),
            np.zeros(count),
        ],
        axis=1,
    )
    gamma[count // 2] = gamma[0]
    gamma[count // 2, 2] += lift
    return gamma, gammadash


def _circle(radius: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    theta = 2.0 * np.pi * np.arange(count) / count
    gamma = radius * np.stack([np.cos(theta), np.sin(theta), np.zeros(count)], axis=1)
    gammadash = (
        2.0
        * np.pi
        * radius
        * np.stack([-np.sin(theta), np.cos(theta), np.zeros(count)], axis=1)
    )
    return gamma, gammadash


def _direction(shape: tuple[int, ...], seed: int) -> np.ndarray:
    vector = np.random.default_rng(seed).standard_normal(shape)
    return vector / np.linalg.norm(vector)


def _kernel_directional_error(gamma: np.ndarray, gammadash: np.ndarray) -> float:
    """Best relative error of AD vs central differences along one direction in (gamma, gamma')."""
    direction_gamma = _direction(gamma.shape, seed=7)
    direction_dash = _direction(gammadash.shape, seed=8)
    gradient_gamma, gradient_dash = _penalty_gradient(gamma, gammadash)
    directional = float(
        np.sum(np.asarray(gradient_gamma) * direction_gamma)
        + np.sum(np.asarray(gradient_dash) * direction_dash)
    )
    errors = []
    for step in FD_STEPS:
        plus = float(
            _penalty_program(
                gamma + step * direction_gamma, gammadash + step * direction_dash
            )
        )
        minus = float(
            _penalty_program(
                gamma - step * direction_gamma, gammadash - step * direction_dash
            )
        )
        errors.append(abs((plus - minus) / (2.0 * step) - directional))
    return min(errors) / abs(directional)


def _hvp(gamma, gammadash, direction_gamma, direction_dash):
    return jax.jvp(
        lambda g, gd: jax.grad(_penalty, argnums=(0, 1))(g, gd),
        (gamma, gammadash),
        (direction_gamma, direction_dash),
    )[1]


_hvp_program = jax.jit(_hvp)


# ---------------------------------------------------------------------------
# Kernel contract.


def test_exact_coincidence_is_an_ordinary_smooth_point() -> None:
    gamma, gammadash = _figure_eight(256)
    assert gamma[0].tobytes() == gamma[128].tobytes()

    value = float(_penalty_program(gamma, gammadash))
    gradient_gamma, gradient_dash = _penalty_gradient(gamma, gammadash)

    assert np.isfinite(value) and value > 0.0
    assert np.all(np.isfinite(np.asarray(gradient_gamma)))
    assert np.all(np.isfinite(np.asarray(gradient_dash)))
    assert _kernel_directional_error(gamma, gammadash) <= 1e-7


def test_near_coincidence_gradient_is_continuous_with_the_exact_one() -> None:
    exact = _penalty_gradient(*_figure_eight(256))
    near = _penalty_gradient(*_figure_eight(256, lift=1e-9))

    for exact_part, near_part in zip(exact, near, strict=True):
        exact_part, near_part = np.asarray(exact_part), np.asarray(near_part)
        change = np.linalg.norm(near_part - exact_part) / np.linalg.norm(exact_part)
        assert change <= 1e-6, change
    assert _kernel_directional_error(*_figure_eight(256, lift=1e-9)) <= 1e-7


@pytest.mark.parametrize("lift", [0.0, 2e-3])
def test_hessian_vector_product_matches_central_differences(lift: float) -> None:
    """Second order: the HVP against central differences of the AD gradient.

    At an exact coincidence and with the crossing strands 2 mm apart (inside
    the 1 cm activation distance), no pair sits on the activation, where
    ``h''`` jumps.
    """
    gamma, gammadash = _figure_eight(256, lift=lift)
    direction_gamma = _direction(gamma.shape, seed=11)
    direction_dash = _direction(gammadash.shape, seed=12)
    hvp = np.concatenate(
        [
            np.asarray(part).ravel()
            for part in _hvp_program(gamma, gammadash, direction_gamma, direction_dash)
        ]
    )
    errors = []
    for step in FD_STEPS:
        plus = _penalty_gradient(
            gamma + step * direction_gamma, gammadash + step * direction_dash
        )
        minus = _penalty_gradient(
            gamma - step * direction_gamma, gammadash - step * direction_dash
        )
        central = np.concatenate(
            [
                (np.asarray(p) - np.asarray(m)).ravel() / (2.0 * step)
                for p, m in zip(plus, minus, strict=True)
            ]
        )
        errors.append(np.linalg.norm(central - hvp) / np.linalg.norm(hvp))

    assert np.all(np.isfinite(hvp)) and np.linalg.norm(hvp) > 0.0
    assert min(errors) <= 1e-6, errors


def test_a_zero_speed_node_is_outside_the_domain_and_the_value_is_nan() -> None:
    gamma, gammadash = _figure_eight(256)
    stalled = gammadash.copy()
    stalled[64] = 0.0
    slow = gammadash.copy()
    slow[64] *= 1e-12

    value = float(_penalty_program(gamma, stalled))
    _gradient_gamma, gradient_dash = _penalty_gradient(gamma, stalled)

    assert np.isnan(value)
    assert not np.all(np.isfinite(np.asarray(gradient_dash)))
    assert np.isfinite(float(_penalty_program(gamma, slow)))
    assert np.all(np.isfinite(np.asarray(_penalty_gradient(gamma, slow)[1])))


def test_value_is_invariant_under_rigid_motion() -> None:
    gamma, gammadash = _figure_eight(256)
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    base = float(_penalty_program(gamma, gammadash))
    moved = float(_penalty_program(gamma @ rotation.T + 0.3, gammadash @ rotation.T))

    assert abs(moved - base) <= 1e-12 * base


def test_right_angle_crossing_energy_is_the_straight_strand_value() -> None:
    """Full-weight regime (the strands are half the length apart): pi d^4 / 12."""
    value = float(_penalty_program(*_figure_eight(4096)))

    assert value == pytest.approx(np.pi * D**4 / 12.0, rel=2e-2)


def test_separation_weight_is_zero_local_one_nonlocal_and_symmetric() -> None:
    _gamma, gammadash = _circle(0.1, 512)
    speed = jnp.linalg.norm(jnp.asarray(gammadash), axis=1)
    weight = np.asarray(self_contact_separation_weight(speed, W0, W1))
    spacing = 2.0 * np.pi * 0.1 / 512
    offset = np.abs(np.arange(512)[:, None] - np.arange(512)[None, :])
    separation = np.minimum(offset, 512 - offset) * spacing

    np.testing.assert_array_equal(weight, weight.T)
    assert np.all(weight[separation < 0.99 * W0] == 0.0)
    assert np.all(weight[separation > 1.01 * W1] == 1.0)
    ramp = weight[(separation > 1.01 * W0) & (separation < 0.99 * W1)]
    assert ramp.size > 0 and np.all((ramp > 0.0) & (ramp < 1.0))
    assert np.asarray(
        self_contact_energy(jnp.asarray([0.0, D**2, 2 * D**2]), D)
    ) == pytest.approx([D**2 / 4.0, 0.0, 0.0], rel=1e-15, abs=0.0)


def _max_step_change(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.diff(values))))


def _hard_mask_penalty(gamma, gammadash):
    """The control: the same energy with a step at ``W0`` instead of the ramp."""
    speed = jnp.linalg.norm(gammadash, axis=1)
    count = gamma.shape[0]
    increments = 0.5 * (speed + jnp.roll(speed, -1)) / count
    arclength = jnp.concatenate([jnp.zeros(1), jnp.cumsum(increments)[:-1]])
    length = jnp.sum(increments)
    separation = jnp.abs(arclength[:, None] - arclength[None, :])
    separation = jnp.minimum(separation, length - separation)
    squared = jnp.sum((gamma[:, None, :] - gamma[None, :, :]) ** 2, axis=2)
    terms = speed[:, None] * speed[None, :] * self_contact_energy(squared, SWEEP_D)
    return 0.5 * jnp.sum(jnp.where(separation >= W0, terms, 0.0)) / (count * count)


def test_sweep_through_the_ramp_and_activation_is_continuous() -> None:
    """Scaling a circle moves node pairs through the ramp band and the activation.

    With d = SWEEP_D > W1 the pairs in the ramp band are active.  A continuous
    function's largest sample-to-sample change halves when the sample step
    halves; the hard-mask control's does not (it jumps at the mask edge).
    """
    ratios = {}
    for name, function in (
        ("smooth", lambda g, gd: _penalty(g, gd, SWEEP_D)),
        ("hard_mask", _hard_mask_penalty),
    ):
        program = jax.jit(jax.vmap(function))
        changes = []
        for samples in (2001, 4001):
            radii = np.linspace(0.095, 0.105, samples)
            curves = [_circle(radius, 256) for radius in radii]
            values = np.asarray(
                program(
                    np.stack([curve[0] for curve in curves]),
                    np.stack([curve[1] for curve in curves]),
                )
            )
            assert np.all(np.isfinite(values))
            changes.append(_max_step_change(values))
        ratios[name] = changes[1] / changes[0]

    assert 0.45 <= ratios["smooth"] <= 0.55, ratios
    assert ratios["hard_mask"] >= 0.9, ratios


def test_kernel_program_runs_under_the_strict_transfer_guard() -> None:
    """Value and gradient stay on the device: no host transfer inside the program."""
    gamma, gammadash = (
        jax.device_put(jnp.asarray(part)) for part in _figure_eight(256, lift=2e-3)
    )
    value_and_gradient = jax.jit(
        jax.value_and_grad(_penalty, argnums=(0, 1)), static_argnums=(2, 3, 4)
    )

    with jax.transfer_guard("disallow"):
        value, gradient = value_and_gradient(gamma, gammadash)
        jax.block_until_ready((value, gradient))

    assert np.isfinite(float(value)) and float(value) > 0.0
    assert all(np.all(np.isfinite(np.asarray(part))) for part in gradient)


# ---------------------------------------------------------------------------
# Solver-level domain-boundary safety: NaN emission alone does not show that a
# line-search trial landing on a zero-speed node is rejected by the real solver
# route.


def _zero_speed_node_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
    """A 1-DOF value/gradient built from the real self-contact kernel.

    Four fixed nodes, close enough that the self-contact energy is active;
    node 0's dash vector is ``(relu(x), 0, 0)``, so node 0's speed is exactly
    zero -- and the kernel returns NaN, by its domain policy -- for every
    ``x <= 0``, not only at one measure-zero point.  A quadratic pull with its
    unconstrained minimum at ``x = -5.0`` (inside that invalid region) drives a
    real L-BFGS-B line search toward it, so a trial point with a zero-speed
    node is actually evaluated on the real solver route.
    """
    gamma = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.004, 0.0, 0.0],
            [0.0, 0.004, 0.0],
            [0.004, 0.004, 0.0],
        ]
    )

    def f(x_dof: jax.Array) -> jax.Array:
        speed0 = jax.nn.relu(x_dof[0])
        gammadash = jnp.stack(
            [
                jnp.array([speed0, 0.0, 0.0]),
                jnp.array([1.0, 0.0, 0.0]),
                jnp.array([1.0, 0.0, 0.0]),
                jnp.array([1.0, 0.0, 0.0]),
            ]
        )
        contact = curve_self_contact_penalty_pure(gamma, gammadash, D, W0, W1)
        return 0.01 * (x_dof[0] - (-5.0)) ** 2 + contact

    value, gradient = jax.value_and_grad(f)(jnp.asarray(x))
    return float(value), np.asarray(gradient, dtype=float)


def test_a_zero_speed_line_search_endpoint_is_an_explicit_nonfinite_failure() -> None:
    """Drive the real ``minimize()``/SciPy L-BFGS-B route toward the domain
    boundary.  SciPy itself ends ON the zero-speed region and reports
    CONVERGENCE there (status 0, success) with a NaN value.  The contract: a
    successful return needs a finite value, parameters and gradient; an
    invalid endpoint may be returned as diagnostic evidence only as an
    explicit unsuccessful NONFINITE result that keeps SciPy's raw termination,
    with SciPy's own returned point (never x0 or another trial substituted),
    and the termination label rejects it.
    """
    options = ScipyLBFGSBOptions(maxiter=60, maxcor=10, ftol=1e-15, gtol=1e-13)
    x0 = np.array([1.0])
    result = minimize(
        _zero_speed_node_objective,
        x0,
        driver=Driver.SCIPY_LBFGSB,
        options=options,
    )
    direct = scipy_minimize(
        _zero_speed_node_objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": options.maxiter,
            "maxfun": options.maxfun,
            "gtol": options.gtol,
            "ftol": options.ftol,
            "maxcor": options.maxcor,
            "maxls": options.maxls,
        },
    )

    # The backend's false success this contract closes.
    assert (int(direct.status), bool(direct.success)) == (0, True)
    assert not np.isfinite(direct.fun) and direct.x[0] <= 0.0
    # The public result: explicit NONFINITE failure, raw termination kept.
    assert result.success is False
    assert result.status == NONFINITE_RESULT_STATUS
    assert "fun" in result.nonfinite_fields
    assert (result.raw_status, result.raw_success, result.raw_message) == (
        0,
        True,
        str(direct.message),
    )
    # SciPy's own returned state, not a substitute: the invalid point itself.
    assert result.x.tobytes() == np.asarray(direct.x, dtype=float).tobytes()
    assert result.x[0] <= 0.0
    assert result.jac is not None
    assert result.jac.tobytes() == np.asarray(direct.jac, dtype=float).tobytes()
    assert (result.nit, result.nfev, result.njev) == (
        direct.nit,
        direct.nfev,
        direct.njev,
    )
    # Downstream, the endpoint is rejected.
    report = termination_report(
        EmitterStop(
            Emitter.SCIPY_LBFGSB, result.status, result.message, result.success
        ),
        x=result.x,
        fun=result.fun,
        gradient=result.jac,
        bounds=None,
        accepted_fun=(),
    )
    assert report.label is TerminationLabel.NONFINITE
