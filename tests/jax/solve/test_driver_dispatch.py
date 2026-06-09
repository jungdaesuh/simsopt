import numpy as np
from scipy.optimize import OptimizeResult

import simsopt_jax.geo.optimizers.optimizer as legacy_optimizer
import simsopt_jax.solve.dispatch as dispatch
from simsopt_jax.solve.dispatch import least_squares, minimize
from simsopt_jax.solve import (
    Driver,
    SimsoptBFGSCallbackEvent,
    SimsoptBFGSOptions,
    SimsoptLMGMRESCallbackEvent,
    SimsoptLMGMRESOptions,
)


def _fake_result():
    return OptimizeResult(
        x=np.zeros(2),
        fun=0.0,
        jac=np.zeros(2),
        nit=0,
        nfev=1,
        njev=1,
        status=0,
        success=True,
        message="ok",
    )


def _value_and_grad(x):
    return 0.0, np.zeros_like(np.asarray(x, dtype=float))


def _residual(x):
    return np.asarray(x, dtype=float)


def test_every_minimize_driver_reaches_documented_dispatch_path(monkeypatch):
    calls = []

    def scipy_minimize(_fn, _x0, *, driver, options, callback):
        calls.append(("scipy", driver.value, type(options).__name__, callback))
        return _fake_result()

    def optax_minimize(_fn, _x0, *, driver, options, callback):
        calls.append(("optax", driver.value, type(options).__name__, callback))
        return _fake_result()

    def optimistix_minimize(_fn, _x0, *, options, callback):
        calls.append(("optimistix", type(options).__name__, callback))
        return _fake_result()

    def reference_minimize(*_args, **kwargs):
        calls.append(("legacy_reference", kwargs["method"]))
        return _fake_result()

    def target_minimize(*_args, **kwargs):
        calls.append(("legacy_target", kwargs["method"]))
        return _fake_result()

    monkeypatch.setattr(dispatch, "_run_scipy_minimize", scipy_minimize)
    monkeypatch.setattr(dispatch, "_run_optax_minimize", optax_minimize)
    monkeypatch.setattr(dispatch, "_run_optimistix_minimize", optimistix_minimize)
    monkeypatch.setattr(legacy_optimizer, "reference_minimize", reference_minimize)
    monkeypatch.setattr(legacy_optimizer, "target_minimize", target_minimize)

    for driver in [
        Driver.SCIPY_LBFGSB,
        Driver.SCIPY_BFGS,
        Driver.OPTAX_LBFGS,
        Driver.OPTAX_ADAM,
        Driver.OPTIMISTIX_LBFGS,
        Driver.SIMSOPT_LBFGSB,
        Driver.SIMSOPT_BFGS,
        Driver.SIMSOPT_TRACE_LBFGS,
        Driver.SIMSOPT_ADAM_HOST,
        Driver.SIMSOPT_ADAM,
    ]:
        result = minimize(_value_and_grad, np.zeros(2), driver=driver)
        assert result.driver is driver

    assert calls == [
        ("scipy", "scipy_lbfgsb", "ScipyLBFGSBOptions", None),
        ("scipy", "scipy_bfgs", "ScipyBFGSOptions", None),
        ("optax", "optax_lbfgs", "OptaxLBFGSOptions", None),
        ("optax", "optax_adam", "OptaxAdamOptions", None),
        ("optimistix", "OptimistixLBFGSOptions", None),
        ("legacy_target", "lbfgs-ondevice"),
        ("legacy_target", "bfgs-ondevice"),
        ("legacy_reference", "lbfgs-trace"),
        ("legacy_reference", "adam"),
        ("legacy_target", "adam-ondevice"),
    ]


def test_every_least_squares_driver_reaches_documented_dispatch_path(monkeypatch):
    calls = []

    def scipy_lm(_fn, _x0, *, options):
        calls.append(("scipy_lm", type(options).__name__))
        return _fake_result()

    def optimistix_lm(_fn, _x0, *, options):
        calls.append(("optimistix_lm", type(options).__name__))
        return _fake_result()

    def reference_least_squares(*_args, **kwargs):
        calls.append(("legacy_reference", kwargs["method"]))
        return _fake_result()

    def target_least_squares(*_args, **kwargs):
        calls.append(("legacy_target", kwargs["method"]))
        return _fake_result()

    monkeypatch.setattr(dispatch, "_scipy_lm_result", scipy_lm)
    monkeypatch.setattr(dispatch, "_run_optimistix_lm", optimistix_lm)
    monkeypatch.setattr(
        legacy_optimizer, "reference_least_squares", reference_least_squares
    )
    monkeypatch.setattr(legacy_optimizer, "target_least_squares", target_least_squares)

    for driver in [
        Driver.SCIPY_LM,
        Driver.OPTIMISTIX_LM,
        Driver.SIMSOPT_LM_GMRES_HOST,
        Driver.SIMSOPT_LM_GMRES,
        Driver.SIMSOPT_LM_QR,
    ]:
        result = least_squares(_residual, np.zeros(2), driver=driver)
        assert result.driver is driver

    assert calls == [
        ("scipy_lm", "ScipyLMOptions"),
        ("optimistix_lm", "OptimistixLMOptions"),
        ("legacy_reference", "lm"),
        ("legacy_target", "lm-ondevice"),
        ("legacy_target", "lm-minpack-ondevice"),
    ]


def test_legacy_dispatch_uses_driver_method_ssot(monkeypatch):
    calls = []

    def reference_minimize(*_args, **kwargs):
        calls.append(("reference_minimize", kwargs["method"]))
        return _fake_result()

    def target_minimize(*_args, **kwargs):
        calls.append(("target_minimize", kwargs["method"]))
        return _fake_result()

    def reference_least_squares(*_args, **kwargs):
        calls.append(("reference_least_squares", kwargs["method"]))
        return _fake_result()

    def target_least_squares(*_args, **kwargs):
        calls.append(("target_least_squares", kwargs["method"]))
        return _fake_result()

    monkeypatch.setattr(legacy_optimizer, "reference_minimize", reference_minimize)
    monkeypatch.setattr(legacy_optimizer, "target_minimize", target_minimize)
    monkeypatch.setattr(
        legacy_optimizer, "reference_least_squares", reference_least_squares
    )
    monkeypatch.setattr(legacy_optimizer, "target_least_squares", target_least_squares)
    monkeypatch.setattr(
        dispatch,
        "legacy_reference_minimize_method",
        lambda driver: f"ssot-reference-minimize:{driver.value}",
    )
    monkeypatch.setattr(
        dispatch,
        "legacy_target_minimize_method",
        lambda driver: f"ssot-target-minimize:{driver.value}",
    )
    monkeypatch.setattr(
        dispatch,
        "legacy_reference_least_squares_method",
        lambda driver: f"ssot-reference-ls:{driver.value}",
    )
    monkeypatch.setattr(
        dispatch,
        "legacy_target_least_squares_method",
        lambda driver: f"ssot-target-ls:{driver.value}",
    )

    minimize(_value_and_grad, np.zeros(2), driver=Driver.SIMSOPT_TRACE_LBFGS)
    minimize(_value_and_grad, np.zeros(2), driver=Driver.SIMSOPT_LBFGSB)
    least_squares(_residual, np.zeros(2), driver=Driver.SIMSOPT_LM_GMRES_HOST)
    least_squares(_residual, np.zeros(2), driver=Driver.SIMSOPT_LM_QR)

    assert calls == [
        ("reference_minimize", "ssot-reference-minimize:simsopt_trace_lbfgs"),
        ("target_minimize", "ssot-target-minimize:simsopt_lbfgsb"),
        ("reference_least_squares", "ssot-reference-ls:simsopt_lm_gmres_host"),
        ("target_least_squares", "ssot-target-ls:simsopt_lm_qr"),
    ]


def test_simsopt_minimize_callback_adapter_emits_typed_event(monkeypatch):
    events = []

    def target_minimize(*_args, **kwargs):
        kwargs["callback"](np.array([0.25, -0.5], dtype=np.float64))
        kwargs["progress_callback"](1, 0.3125, 1.0)
        return _fake_result()

    def value_and_grad(x):
        return float(np.dot(x, x)), 2.0 * np.asarray(x, dtype=float)

    monkeypatch.setattr(legacy_optimizer, "target_minimize", target_minimize)

    minimize(
        value_and_grad,
        np.array([1.0, -2.0]),
        driver=Driver.SIMSOPT_BFGS,
        options=SimsoptBFGSOptions(maxiter=1),
        callback=events.append,
    )

    assert len(events) == 1
    assert isinstance(events[0], SimsoptBFGSCallbackEvent)
    assert events[0].driver is Driver.SIMSOPT_BFGS
    np.testing.assert_allclose(events[0].x, np.array([0.25, -0.5]))
    assert events[0].fun == 0.3125


def test_simsopt_least_squares_callback_adapter_emits_typed_event(monkeypatch):
    events = []

    def target_least_squares(*_args, **kwargs):
        kwargs["callback"](np.array([0.25, -0.5], dtype=np.float64))
        kwargs["progress_callback"](1, 1.40625, 1.5)
        return _fake_result()

    def residual(x):
        return np.asarray(x, dtype=float) - np.array([1.0, -2.0])

    monkeypatch.setattr(legacy_optimizer, "target_least_squares", target_least_squares)

    least_squares(
        residual,
        np.array([1.0, -2.0]),
        driver=Driver.SIMSOPT_LM_GMRES,
        options=SimsoptLMGMRESOptions(maxiter=1),
        callback=events.append,
    )

    assert len(events) == 1
    assert isinstance(events[0], SimsoptLMGMRESCallbackEvent)
    assert events[0].driver is Driver.SIMSOPT_LM_GMRES
    np.testing.assert_allclose(events[0].x, np.array([0.25, -0.5]))
    assert np.isclose(events[0].residual_norm, np.linalg.norm([-0.75, 1.5]))


def test_simsopt_least_squares_callback_adapter_accepts_progress_before_state():
    events = []
    legacy_callback, legacy_progress = dispatch._legacy_least_squares_callbacks(
        events.append,
        driver=Driver.SIMSOPT_LM_GMRES,
        options=SimsoptLMGMRESOptions(maxiter=1),
    )

    legacy_progress(1, 1.40625, 1.5)
    assert events == []

    legacy_callback(np.array([0.25, -0.5], dtype=np.float64))

    assert len(events) == 1
    assert isinstance(events[0], SimsoptLMGMRESCallbackEvent)
    assert events[0].driver is Driver.SIMSOPT_LM_GMRES
    np.testing.assert_allclose(events[0].x, np.array([0.25, -0.5]))
    assert events[0].fun == 1.40625
