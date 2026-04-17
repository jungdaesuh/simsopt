from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from simsopt.geo.optimizer_jax_private import _result_converters as converters


def _make_bfgs_state():
    return SimpleNamespace(
        line_search_status=jnp.asarray(0, dtype=jnp.int32),
        status=jnp.asarray(0, dtype=jnp.int32),
        k=jnp.asarray(4, dtype=jnp.int32),
        x_k=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        f_k=jnp.asarray(3.5, dtype=jnp.float64),
        g_k=jnp.asarray([0.1, -0.2], dtype=jnp.float64),
        nfev=jnp.asarray(10, dtype=jnp.int32),
        ngev=jnp.asarray(11, dtype=jnp.int32),
        nhev=jnp.asarray(12, dtype=jnp.int32),
        converged=jnp.asarray(True),
        H_k=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
    )


def _make_lbfgs_state(
    *,
    status=0,
    k=5,
    f_k=1.25,
    g_k=(0.3, -0.4),
    nfev=13,
    ngev=14,
    converged=True,
):
    return SimpleNamespace(
        status=jnp.asarray(status, dtype=jnp.int32),
        k=jnp.asarray(k, dtype=jnp.int32),
        x_k=jnp.asarray([2.0, -1.0], dtype=jnp.float64),
        f_k=jnp.asarray(f_k, dtype=jnp.float64),
        g_k=jnp.asarray(g_k, dtype=jnp.float64),
        nfev=jnp.asarray(nfev, dtype=jnp.int32),
        ngev=jnp.asarray(ngev, dtype=jnp.int32),
        converged=jnp.asarray(converged),
        ls_status=jnp.asarray(0, dtype=jnp.int32),
    )


def _patch_host_metadata_helpers(monkeypatch):
    calls: list[tuple[str, object]] = []

    def record_int(value, *, dtype=np.int64):
        calls.append(("int", np.dtype(dtype).name))
        return int(np.asarray(value))

    def record_float(value, *, dtype=np.float64):
        calls.append(("float", np.dtype(dtype).name))
        return float(np.asarray(value))

    def record_bool(value):
        calls.append(("bool", None))
        return bool(np.asarray(value))

    monkeypatch.setattr(converters, "_host_int", record_int)
    monkeypatch.setattr(converters, "_host_float", record_float)
    monkeypatch.setattr(converters, "_host_bool", record_bool)
    return calls


def _assert_host_numeric_metadata_helper_calls(calls):
    assert ("int", "int64") in calls
    assert ("float", "float64") in calls


def test_private_bfgs_result_uses_explicit_host_metadata_helpers(monkeypatch):
    calls = _patch_host_metadata_helpers(monkeypatch)

    result = converters._private_bfgs_result_to_optimize_result(_make_bfgs_state())

    assert result.nit == 4
    assert result.status == 0
    assert result.success is True
    assert result.fun == 3.5
    assert result.line_search_status == 0
    _assert_host_numeric_metadata_helper_calls(calls)
    assert ("bool", None) in calls


def test_private_lbfgs_result_uses_explicit_host_metadata_helpers(monkeypatch):
    calls = _patch_host_metadata_helpers(monkeypatch)

    result = converters._private_lbfgs_result_to_optimize_result(_make_lbfgs_state())

    assert result.nit == 5
    assert result.status == 0
    assert result.success is True
    assert result.fun == 1.25
    assert result.ls_status == 0
    _assert_host_numeric_metadata_helper_calls(calls)


def test_private_lbfgs_result_treats_ftol_status_as_success():
    result = converters._private_lbfgs_result_to_optimize_result(
        _make_lbfgs_state(
            status=4,
            k=9,
            f_k=1.0,
            g_k=(1.0e-3, -2.0e-3),
            nfev=21,
            ngev=22,
            converged=False,
        )
    )

    assert result.status == 4
    assert result.success is True
    assert result.message == "Optimization terminated successfully (ftol)."


def test_private_lbfgs_result_reports_status_six_for_nonfinite_objective():
    result = converters._private_lbfgs_result_to_optimize_result(
        _make_lbfgs_state(
            status=6,
            k=3,
            f_k=float("nan"),
            g_k=(0.1, -0.2),
            nfev=7,
            ngev=8,
            converged=False,
        )
    )

    assert result.status == 6
    assert result.success is False
    assert "non-finite" in result.message.lower()


def test_private_lbfgs_result_reports_status_six_for_nonfinite_gradient():
    result = converters._private_lbfgs_result_to_optimize_result(
        _make_lbfgs_state(
            status=6,
            k=4,
            f_k=0.5,
            g_k=(float("inf"), 0.3),
            nfev=9,
            ngev=10,
            converged=False,
        )
    )

    assert result.status == 6
    assert result.success is False
    assert "non-finite" in result.message.lower()


def test_private_lbfgs_result_status_six_dict_entry_surfaces_when_state_is_finite():
    """status=6 with numerically finite f_k/g_k surfaces the dict entry.

    The runtime invalid_state post-check in the converter overrides the
    status-to-message mapping when f_k/g_k contain NaN/inf, but the
    status=6 dict entry still serves when the solver flagged non-finite
    mid-run but the final accepted iterate happens to be finite.
    """
    result = converters._private_lbfgs_result_to_optimize_result(
        _make_lbfgs_state(
            status=6,
            k=1,
            f_k=0.5,
            g_k=(0.1, -0.2),
            converged=False,
        )
    )

    assert result.status == 6
    assert result.success is False
    assert (
        result.message
        == "Non-finite objective or gradient encountered during iteration."
    )
