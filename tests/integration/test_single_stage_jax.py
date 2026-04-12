"""JAX-only helper-path coverage for the single-stage objective stack.

These tests deliberately avoid ``simsoptpp`` so they still collect in a
JAX-only environment while the heavier CPU-reference integration suite in
``test_single_stage_jax_cpu_reference.py`` stays gated on the compiled
extension.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_REPO_ROOT_STR = str(_REPO_ROOT)
_RTOL = 1e-12
_ATOL = 1e-12
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(_SRC_ROOT)

import simsopt.geo.surfaceobjectives_jax as soj


class _FakeDependentOpt:
    def __init__(self) -> None:
        self.local_full_dof_size = 3
        self.local_dofs_free_status = np.array([True, True, False])


class _FakeDofs:
    def __init__(self, dep_opts: tuple[_FakeDependentOpt, ...]) -> None:
        self._dep_opts = dep_opts

    def dep_opts(self) -> tuple[_FakeDependentOpt, ...]:
        return self._dep_opts


class _FakeLineageOpt:
    def __init__(self, dep_opts: tuple[_FakeDependentOpt, ...]) -> None:
        self.local_dof_size = 2
        self.local_full_dof_size = 3
        self.dofs_free_status = np.array([True, True, False])
        self.local_dofs_free_status = np.array([True, True, False])
        self.dofs = _FakeDofs(dep_opts)


class _FakeBiotSavart:
    def __init__(self, lineage: _FakeLineageOpt) -> None:
        self.unique_dof_lineage = [lineage]


def _assert_allclose(
    actual: object,
    expected: object,
    *,
    dtype: np.typing.DTypeLike | None = None,
) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=dtype),
        np.asarray(expected, dtype=dtype),
        rtol=_RTOL,
        atol=_ATOL,
    )


def _build_shared_lineage_biotsavart() -> tuple[
    _FakeBiotSavart, _FakeDependentOpt, _FakeDependentOpt
]:
    dep_opt_a = _FakeDependentOpt()
    dep_opt_b = _FakeDependentOpt()
    lineage = _FakeLineageOpt((dep_opt_a, dep_opt_b))
    return _FakeBiotSavart(lineage), dep_opt_a, dep_opt_b


def test_coil_dofs_gradient_to_derivative_preserves_shared_dof_round_trip() -> None:
    """Shared DOF lineages must not amplify gradients when converted to Derivative."""
    biotsavart, dep_opt_a, dep_opt_b = _build_shared_lineage_biotsavart()

    derivative = soj._coil_dofs_gradient_to_derivative(
        biotsavart,
        np.array([2.0, -3.0]),
    )

    _assert_allclose(derivative(biotsavart), [2.0, -3.0], dtype=float)
    _assert_allclose(derivative.data[dep_opt_a], [1.0, -1.5, 0.0], dtype=float)
    _assert_allclose(derivative.data[dep_opt_b], [1.0, -1.5, 0.0], dtype=float)


def _patch_runtime_scalar_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    calls = {"count": 0}
    original_runtime_scalar = soj._runtime_float64_scalar

    def _counting_runtime_scalar(value: object, *, reference: object) -> object:
        calls["count"] += 1
        return original_runtime_scalar(value, reference=reference)

    monkeypatch.setattr(soj, "_runtime_float64_scalar", _counting_runtime_scalar)
    return calls


def test_coil_dofs_gradient_to_derivative_uses_explicit_host_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JAX coil gradients must be hostified via explicit device_get."""
    biotsavart, _, _ = _build_shared_lineage_biotsavart()
    calls = {"count": 0}
    original_device_get = soj.jax.device_get

    def _counting_device_get(value: object) -> object:
        calls["count"] += 1
        return original_device_get(value)

    monkeypatch.setattr(soj.jax, "device_get", _counting_device_get)

    derivative = soj._coil_dofs_gradient_to_derivative(
        biotsavart,
        jnp.array([2.0, -3.0], dtype=jnp.float64),
    )

    assert calls["count"] == 1
    _assert_allclose(derivative(biotsavart), [2.0, -3.0], dtype=float)


def test_split_x_inner_runtime_preserves_surface_iota_and_G(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit runtime splitting should preserve inner-state semantics."""
    calls = {"count": 0}
    original_device_put = soj.jax.device_put

    def _counting_device_put(value: object) -> object:
        calls["count"] += 1
        return original_device_put(value)

    monkeypatch.setattr(soj.jax, "device_put", _counting_device_put)

    x_with_g = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    sdofs, iota, g_value = soj._split_x_inner_runtime(x_with_g, True)
    _assert_allclose(sdofs, [1.0, 2.0])
    _assert_allclose(iota, 3.0)
    _assert_allclose(g_value, 4.0)

    x_without_g = jnp.array([5.0, 6.0, 7.0], dtype=jnp.float64)
    sdofs, iota, g_value = soj._split_x_inner_runtime(x_without_g, False)
    _assert_allclose(sdofs, [5.0, 6.0])
    _assert_allclose(iota, 7.0)
    assert g_value is None
    assert calls["count"] >= 4


def test_boozer_residual_inner_objective_uses_runtime_scalar_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict-safe inner-objective math should runtimeify host-backed scalars."""
    calls = _patch_runtime_scalar_counter(monkeypatch)
    monkeypatch.setattr(
        soj,
        "_surface_geometry_from_dofs",
        lambda *args, **kwargs: (
            jnp.zeros((1, 1, 3), dtype=jnp.float64),
            jnp.ones((1, 1, 3), dtype=jnp.float64),
            jnp.ones((1, 1, 3), dtype=jnp.float64),
        ),
    )
    monkeypatch.setattr(
        soj,
        "grouped_biot_savart_B_from_spec",
        lambda points, coil_set_spec: jnp.ones((points.shape[0], 3), dtype=jnp.float64),
    )
    monkeypatch.setattr(
        soj,
        "boozer_residual_scalar",
        lambda g_value, iota, B, xphi, xtheta, weight_inv_modB: (
            soj._runtime_float64_scalar(5.0 / 6.0, reference=B)
        ),
    )
    monkeypatch.setattr(
        soj,
        "_compute_label",
        lambda *args, **kwargs: jnp.asarray(1.5, dtype=jnp.float64),
    )

    value = soj._boozer_residual_J_of_x_inner(
        jnp.array([4.0, 0.25, 0.75], dtype=jnp.float64),
        coil_set_spec=object(),
        quadpoints_phi=jnp.asarray([0.0], dtype=jnp.float64),
        quadpoints_theta=jnp.asarray([0.0], dtype=jnp.float64),
        mpol=1,
        ntor=1,
        nfp=1,
        stellsym=True,
        scatter_indices=jnp.asarray([0], dtype=jnp.int32),
        surface_kind="tensor",
        optimize_G=True,
        weight_inv_modB=True,
        constraint_weight=3.0,
        targetlabel=1.0,
        label_type="axis",
        phi_idx=0,
    )

    assert calls["count"] >= 4
    _assert_allclose(value, 5.0 / 6.0 + 0.375)


def test_strict_scalar_value_and_grad_uses_explicit_pullback_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict scalar gradients should avoid JAX's implicit host seed creation."""
    calls = {"count": 0}
    original_seed = soj._explicit_scalar_pullback_seed

    def _counting_seed(value: object) -> object:
        calls["count"] += 1
        return original_seed(value)

    monkeypatch.setattr(soj, "_explicit_scalar_pullback_seed", _counting_seed)

    value, grad = soj._strict_scalar_value_and_grad(
        lambda x, scale: jnp.sum(scale * (x * x)),
        jnp.array([2.0, -3.0], dtype=jnp.float64),
        0.5,
    )

    assert calls["count"] == 1
    np.testing.assert_allclose(np.asarray(value), 6.5, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(grad),
        np.array([2.0, -3.0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_traceable_iota_target_penalty_uses_runtime_scalar_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The traceable iota penalty should avoid implicit host scalar promotion."""
    calls = _patch_runtime_scalar_counter(monkeypatch)

    penalty = soj._traceable_iota_target_penalty(
        jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64),
        optimize_G=False,
        iota_target=2.5,
    )

    assert calls["count"] >= 2
    _assert_allclose(penalty, 0.125)


def test_value_and_direct_coil_derivative_hostifies_objective_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Objective scalars should cross back to Python only via explicit device_get."""
    biotsavart, _, _ = _build_shared_lineage_biotsavart()
    calls = {"count": 0}
    original_host_scalar = soj._host_scalar

    def _counting_host_scalar(value: object) -> object:
        calls["count"] += 1
        return original_host_scalar(value)

    monkeypatch.setattr(soj, "_host_scalar", _counting_host_scalar)
    monkeypatch.setattr(
        soj,
        "_strict_scalar_value_and_grad",
        lambda objective, coil_dofs, *args: (
            jnp.asarray(3.5, dtype=jnp.float64),
            jnp.asarray([2.0, -3.0], dtype=jnp.float64),
        ),
    )

    value, derivative = soj._value_and_direct_coil_derivative(
        biotsavart,
        lambda coil_dofs: jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
    )

    assert calls["count"] == 1
    assert value == pytest.approx(3.5)
    _assert_allclose(derivative(biotsavart), [2.0, -3.0], dtype=float)
