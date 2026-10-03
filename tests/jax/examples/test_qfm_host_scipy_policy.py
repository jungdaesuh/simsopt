"""The QFM mirror preserves the reference host-SciPy stage policy.

Every assertion here comes from a real execution of the public sequence over
small analytic kernels: the SciPy providers really run, and the spies only
record what the mirror asked them for before delegating.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from simsopt_jax.examples import qfm_host_scipy
from simsopt_jax.examples.qfm_host_scipy import (
    QFM_EXACT_METHOD,
    QFM_LABELS,
    QfmHostKernels,
    solve_qfm_host_scipy_sequence,
    solve_qfm_host_scipy_stage,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions

TOLERANCE = 1.0e-12
MAX_STEPS = 1000


def _analytic_kernels() -> QfmHostKernels:
    """A two-dof stand-in with the same call signature as the real physics."""

    def compiled(value):
        return jax.jit(jax.value_and_grad(value))

    return QfmHostKernels(
        qfm=compiled(lambda parameters: jnp.vdot(parameters, parameters)),
        labels={
            "volume": compiled(lambda parameters: parameters[0]),
            "toroidal_flux": compiled(lambda parameters: parameters[1]),
            "area": compiled(lambda parameters: parameters[0] + parameters[1]),
        },
    )


def _host_labels():
    return {
        "volume": lambda parameters: (float(parameters[0]), np.asarray([1.0, 0.0])),
        "toroidal_flux": lambda parameters: (
            float(parameters[1]),
            np.asarray([0.0, 1.0]),
        ),
        "area": lambda parameters: (
            float(parameters[0] + parameters[1]),
            np.asarray([1.0, 1.0]),
        ),
    }


class _Spies:
    """Records each provider call and delegates to the real provider."""

    def __init__(self) -> None:
        self.methods: list[str] = []
        self.starts: list[np.ndarray] = []
        self.penalty_drivers: list[Driver] = []
        self.penalty_options: list[ScipyLBFGSBOptions] = []
        self.exact_options: list[dict[str, float]] = []
        self.constraints: list[list[dict[str, object]]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_dispatch = qfm_host_scipy.dispatch_minimize
        real_minimize = qfm_host_scipy.minimize

        def spy_dispatch(value_and_grad_fn, x0, *, driver, options, callback=None):
            self.methods.append(str(driver))
            self.starts.append(np.asarray(x0, dtype=np.float64))
            self.penalty_drivers.append(driver)
            self.penalty_options.append(options)
            return real_dispatch(
                value_and_grad_fn,
                x0,
                driver=driver,
                options=options,
                callback=callback,
            )

        def spy_minimize(objective, x0, **kwargs) -> OptimizeResult:
            self.methods.append(str(kwargs["method"]))
            self.starts.append(np.asarray(x0, dtype=np.float64))
            self.exact_options.append(dict(kwargs["options"]))
            self.constraints.append(list(kwargs["constraints"]))
            return real_minimize(objective, x0, **kwargs)

        monkeypatch.setattr(qfm_host_scipy, "dispatch_minimize", spy_dispatch)
        monkeypatch.setattr(qfm_host_scipy, "minimize", spy_minimize)


def _sequence(monkeypatch: pytest.MonkeyPatch, initial: np.ndarray):
    spies = _Spies()
    spies.install(monkeypatch)
    result = solve_qfm_host_scipy_sequence(
        initial,
        kernels=_analytic_kernels(),
        max_steps=MAX_STEPS,
        tolerance=TOLERANCE,
        constraint_weight=1.0,
    )
    return result, spies


def test_sequence_runs_six_official_calls_with_the_official_option_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Penalty then exact, three times, with upstream's two option sets."""
    initial = np.asarray([1.0, 2.0])
    _, spies = _sequence(monkeypatch, initial)

    assert spies.methods == [
        str(Driver.SCIPY_LBFGSB),
        QFM_EXACT_METHOD,
        str(Driver.SCIPY_LBFGSB),
        QFM_EXACT_METHOD,
        str(Driver.SCIPY_LBFGSB),
        QFM_EXACT_METHOD,
    ]
    assert spies.penalty_drivers == [Driver.SCIPY_LBFGSB] * 3
    # ``qfmsurface.py`` names maxiter, ftol, gtol and maxcor and leaves maxfun
    # and maxls at SciPy's defaults.
    assert (
        spies.penalty_options
        == [
            ScipyLBFGSBOptions(
                maxiter=MAX_STEPS,
                maxfun=15000,
                gtol=TOLERANCE,
                ftol=TOLERANCE,
                maxcor=200,
                maxls=20,
            )
        ]
        * 3
    )
    assert spies.exact_options == [{"maxiter": MAX_STEPS, "ftol": TOLERANCE}] * 3
    for constraints in spies.constraints:
        assert len(constraints) == 1
        assert constraints[0]["type"] == "eq"


def test_sequence_chains_three_labels_and_starts_slsqp_at_the_lbfgsb_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each stage starts where the previous one ended; SLSQP starts at ``x``."""
    initial = np.asarray([1.0, 2.0])
    result, spies = _sequence(monkeypatch, initial)

    stages = [getattr(result, label) for label in QFM_LABELS]
    assert [stage.label for stage in stages] == list(QFM_LABELS)

    expected_starts = [
        initial,
        stages[0].penalty.parameters,
        stages[0].exact.parameters,
        stages[1].penalty.parameters,
        stages[1].exact.parameters,
        stages[2].penalty.parameters,
    ]
    for recorded, expected in zip(spies.starts, expected_starts, strict=True):
        np.testing.assert_allclose(recorded, expected, rtol=0.0, atol=0.0)


def test_each_stage_target_is_the_label_at_the_state_reached_so_far(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The official rule, and not the labels of the first state."""
    initial = np.asarray([1.0, 2.0])
    result, _ = _sequence(monkeypatch, initial)
    labels = _host_labels()

    starts = (
        initial,
        result.volume.exact.parameters,
        result.toroidal_flux.exact.parameters,
    )
    for label, start in zip(QFM_LABELS, starts, strict=True):
        expected_target, _ = labels[label](start)
        assert getattr(result, label).target == pytest.approx(expected_target, abs=0.0)

    naive_targets = [labels[label](initial)[0] for label in QFM_LABELS]
    actual_targets = [getattr(result, label).target for label in QFM_LABELS]
    assert actual_targets[1] != naive_targets[1]
    assert actual_targets[2] != naive_targets[2]

    assert result.initial_volume == pytest.approx(labels["volume"](initial)[0])


def test_every_provider_really_runs_and_slsqp_restores_the_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real providers, real outcomes: the exact call pulls the label back."""
    initial = np.asarray([1.0, 2.0])
    result, _ = _sequence(monkeypatch, initial)

    for label in QFM_LABELS:
        stage = getattr(result, label)
        assert stage.penalty_optimizer.success is True
        assert stage.penalty_optimizer.status == 0
        assert stage.penalty_optimizer.nfev >= 1
        assert stage.exact_optimizer.success is True
        assert stage.exact_optimizer.status == 0
        assert stage.exact_optimizer.nfev >= 1
        assert stage.exact.label_residual_abs < 1.0e-8

    # The penalty call trades the label away -- ``min p0^2 + p1^2 +
    # 0.5 (p0 - 1)^2`` lands on p0 = 1/3 -- and the equality-constrained call
    # is what brings it back, which is why the mirror must keep both.
    assert result.volume.penalty.parameters[0] == pytest.approx(1.0 / 3.0)
    assert result.volume.penalty.label_residual_abs == pytest.approx(2.0 / 3.0)
    assert result.volume.exact.parameters[0] == pytest.approx(1.0, abs=1.0e-9)


def test_single_stage_entry_accepts_an_external_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staged replay starts from a given state with a given label target."""
    spies = _Spies()
    spies.install(monkeypatch)
    start = np.asarray([1.0, 2.0])
    stage = solve_qfm_host_scipy_stage(
        start,
        kernels=_analytic_kernels(),
        label="toroidal_flux",
        initial_volume=1.0,
        tolerance=TOLERANCE,
        max_steps=MAX_STEPS,
        target=0.25,
    )

    assert stage.target == 0.25
    assert stage.initial.label_value == pytest.approx(2.0)
    assert stage.initial.label_residual_abs == pytest.approx(1.75)
    assert stage.exact.label_value == pytest.approx(0.25, abs=1.0e-8)
    assert spies.methods == [str(Driver.SCIPY_LBFGSB), QFM_EXACT_METHOD]
    np.testing.assert_allclose(spies.starts[0], start, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        spies.starts[1],
        stage.penalty.parameters,
        rtol=0.0,
        atol=0.0,
    )
