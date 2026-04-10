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
