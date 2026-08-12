"""JAX-only helper-path coverage for the single-stage objective stack.

These tests deliberately avoid ``simsoptpp`` so they still collect in a
JAX-only environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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

from simsopt._core.optimizable import DOFs, Optimizable
import simsopt_jax_adapters.geo.surface_objectives as soj
import simsopt_jax_adapters.geo.surface_objectives_traceable as sotj


class _FakeDependentOpt(Optimizable):
    return_fn_map = {}

    def __init__(self, dofs: DOFs) -> None:
        super().__init__(dofs=dofs)


class _FakeBiotSavart(Optimizable):
    return_fn_map = {}

    def __init__(self, lineage: tuple[_FakeDependentOpt, ...]) -> None:
        super().__init__(depends_on=lineage)


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
    shared_dofs = DOFs(
        x=np.zeros(3, dtype=float),
        free=np.array([True, True, False]),
    )
    dep_opt_a = _FakeDependentOpt(shared_dofs)
    dep_opt_b = _FakeDependentOpt(shared_dofs)
    return _FakeBiotSavart((dep_opt_a, dep_opt_b)), dep_opt_a, dep_opt_b


def test_coil_dofs_gradient_to_derivative_preserves_shared_dof_round_trip() -> None:
    """Shared DOF lineages must not amplify gradients when converted to Derivative."""
    biotsavart, dep_opt_a, dep_opt_b = _build_shared_lineage_biotsavart()

    derivative = soj.coil_dofs_gradient_to_derivative(
        biotsavart,
        np.array([2.0, -3.0]),
    )

    _assert_allclose(derivative(biotsavart), [2.0, -3.0], dtype=float)
    _assert_allclose(derivative.data[dep_opt_a], [1.0, -1.5, 0.0], dtype=float)
    _assert_allclose(derivative.data[dep_opt_b], [1.0, -1.5, 0.0], dtype=float)


def test_coil_dofs_gradient_to_derivative_canonical_jax_export() -> None:
    biotsavart, dep_opt_a, dep_opt_b = _build_shared_lineage_biotsavart()
    derivative = soj.coil_dofs_gradient_to_derivative(
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
    monkeypatch.setattr(sotj, "_runtime_float64_scalar", _counting_runtime_scalar)
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

    derivative = soj.coil_dofs_gradient_to_derivative(
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

    def _counting_device_put(value: object, *args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return original_device_put(value, *args, **kwargs)

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
        surface_kind="xyztensorfourier",
        label_quadpoints_phi=jnp.asarray([0.0], dtype=jnp.float64),
        label_quadpoints_theta=jnp.asarray([0.0], dtype=jnp.float64),
        label_mpol=1,
        label_ntor=1,
        label_nfp=1,
        label_stellsym=True,
        label_scatter_indices=jnp.asarray([0], dtype=jnp.int32),
        label_surface_kind="xyztensorfourier",
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


def test_surface_major_radius_from_geometry_recovers_circular_torus_radius() -> None:
    """The BoozerQA radius penalty must use SIMSOPT's geometric definition."""
    major_radius = 1.7
    minor_radius = 0.23
    phi = jnp.linspace(0.0, 1.0, 24, endpoint=False, dtype=jnp.float64)
    theta = jnp.linspace(0.0, 1.0, 25, endpoint=False, dtype=jnp.float64)
    phi_angle = 2.0 * jnp.pi * phi[:, None]
    theta_angle = 2.0 * jnp.pi * theta[None, :]
    cylindrical_radius = major_radius + minor_radius * jnp.cos(theta_angle)
    gamma = jnp.stack(
        jnp.broadcast_arrays(
            cylindrical_radius * jnp.cos(phi_angle),
            cylindrical_radius * jnp.sin(phi_angle),
            minor_radius * jnp.sin(theta_angle),
        ),
        axis=-1,
    )
    xphi = jnp.stack(
        (
            -2.0 * jnp.pi * cylindrical_radius * jnp.sin(phi_angle),
            2.0 * jnp.pi * cylindrical_radius * jnp.cos(phi_angle),
            jnp.zeros_like(gamma[..., 2]),
        ),
        axis=-1,
    )
    xtheta = jnp.stack(
        jnp.broadcast_arrays(
            -2.0 * jnp.pi * minor_radius * jnp.sin(theta_angle) * jnp.cos(phi_angle),
            -2.0 * jnp.pi * minor_radius * jnp.sin(theta_angle) * jnp.sin(phi_angle),
            2.0 * jnp.pi * minor_radius * jnp.cos(theta_angle),
        ),
        axis=-1,
    )

    with jax.transfer_guard("disallow"):
        actual = soj._surface_major_radius_from_geometry(gamma, xphi, xtheta)

    _assert_allclose(actual, major_radius)


def test_selected_coil_length_penalty_uses_total_selected_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BoozerQA regularizes the total base-coil length, not one expanded coil."""
    curve_speeds = {
        "base-0": 2.0,
        "base-1": 3.0,
        "expanded-copy": 7.0,
    }
    monkeypatch.setattr(
        soj,
        "coil_specs_from_dof_extraction_spec",
        lambda _extraction_spec, _coil_dofs: tuple(
            SimpleNamespace(curve=curve_name) for curve_name in curve_speeds
        ),
    )

    curve_lengths = {
        name: jnp.asarray(speed, dtype=jnp.float64)
        for name, speed in curve_speeds.items()
    }
    monkeypatch.setattr(
        soj,
        "curve_length_from_spec",
        lambda curve_name: curve_lengths[curve_name],
    )
    coil_dofs = jnp.zeros(1, dtype=jnp.float64)

    with jax.transfer_guard("disallow"):
        penalty = soj._selected_coil_length_penalty_from_coil_dofs(
            coil_dofs,
            object(),
            coil_indices=(0, 1),
            length_target=4.0,
        )

    _assert_allclose(penalty, 0.5)


def test_value_and_direct_coil_gradient_keeps_objective_value_on_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct-coil objective scalars should stay as JAX values at this helper."""
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

    value, gradient = soj._value_and_direct_coil_gradient(
        lambda coil_dofs: jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
    )

    assert calls["count"] == 0
    assert isinstance(value, jax.Array)
    _assert_allclose(value, 3.5)
    _assert_allclose(gradient, [2.0, -3.0], dtype=float)
