"""JAX-specific ToroidalFlux Taylor and tolerance-parity coverage.

These tests exercise the pure JAX label/objective ingredients directly:

1. Surface-DOF Hessian Taylor convergence for toroidal flux.
2. Coil-family DOF gradient Taylor convergence for toroidal flux.
3. Upstream-shaped ToroidalFlux CPU/JAX parity under tolerance-based checks.
"""

import os
from pathlib import Path
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    enable_strict_parity_backend,
    host_array,
    host_scalar,
    parity_default_device,
    parity_rng,
)
from benchmarks.validation_ladder_contract import parity_ladder_tolerances

# Add the src root so pure-JAX simsopt modules resolve from this repo
# without reloading the entire simsopt package during test collection.
_REPO_SRC_ROOT = str(Path(__file__).resolve().parents[2] / "src")
if _REPO_SRC_ROOT not in sys.path:
    sys.path.insert(0, _REPO_SRC_ROOT)

from simsopt.field.biotsavart_jax import biot_savart_A
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.configs.zoo import get_data
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo import optimizer_jax as optimizer_jax_module
from simsopt.geo import surfaceobjectives as surfaceobjectives_module
from simsopt.geo import surfaceobjectives_jax as surfaceobjectives_jax_module
from simsopt.backend import invalidate_backend_cache
from simsopt.geo._pairwise_reductions import pairwise_selected_smoothmin_distance_pure
from simsopt.geo.surfaceobjectives import ToroidalFlux
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.label_constraints_jax import toroidal_flux_jax
from simsopt.geo.surface_fourier_jax import (
    surface_gamma_from_dofs,
    surface_gammadash2_from_dofs,
    stellsym_scatter_indices,
)
from .surface_test_helpers import get_exact_surface, get_surface

_MPOL = 1
_NTOR = 1
_NFP = 1
_NPHI = 15
_NTHETA = 16
_QP_PHI = jnp.linspace(0, 1, _NPHI, endpoint=False)
_QP_THETA = jnp.linspace(0, 1, _NTHETA, endpoint=False)
_TF_COIL_DOFS = jnp.array(
    [
        0.02,
        -0.03,
        0.01,
        -0.02,
        0.03,
        -0.01,
        0.04,
        0.02,
        -0.03,
        0.01,
        -0.04,
        0.03,
        -0.02,
        0.01,
    ],
    dtype=jnp.float64,
)
_SURFACE_TYPES = (
    "SurfaceXYZFourier",
    "SurfaceRZFourier",
    "SurfaceXYZTensorFourier",
)


def _assert_nonfinite_gradient(grad):
    assert not np.any(np.isfinite(np.asarray(grad)))


def _assert_primal_value_with_nonfinite_gradient(value, grad, expected_value):
    np.testing.assert_allclose(np.asarray(value), np.asarray(expected_value))
    _assert_nonfinite_gradient(grad)


def _reject_coil_dofs_gradient_to_derivative(*_args):
    raise AssertionError("native gradient should not project to Derivative")


def _patch_reject_coil_dofs_gradient_to_derivative(monkeypatch):
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_coil_dofs_gradient_to_derivative",
        _reject_coil_dofs_gradient_to_derivative,
    )


_STELLSYM_OPTIONS = (True, False)
_TOROIDAL_FLUX_DERIVATIVE_HEAVY_TOLERANCES = parity_ladder_tolerances(
    "derivative_heavy"
)
_TOROIDAL_FLUX_VALUE_RTOL = _TOROIDAL_FLUX_DERIVATIVE_HEAVY_TOLERANCES[
    "scalar_value_rtol"
]
_TOROIDAL_FLUX_VALUE_ATOL = _TOROIDAL_FLUX_DERIVATIVE_HEAVY_TOLERANCES[
    "scalar_value_atol"
]
_TOROIDAL_FLUX_SURFACE_GRAD_RTOL = 1e-9
_TOROIDAL_FLUX_SURFACE_GRAD_ATOL = 1e-11
_TOROIDAL_FLUX_SURFACE_HESS_RTOL = 1e-8
_TOROIDAL_FLUX_SURFACE_HESS_ATOL = 1e-10
_TOROIDAL_FLUX_COIL_GRAD_RTOL = 1e-9
_TOROIDAL_FLUX_COIL_GRAD_ATOL = 1e-7
_ASPECT_RATIO_VALUE_RTOL = 1e-12
_ASPECT_RATIO_VALUE_ATOL = 1e-12
_ASPECT_RATIO_GRAD_RTOL = 1e-10
_ASPECT_RATIO_GRAD_ATOL = 1e-11
_ASPECT_RATIO_HESS_RTOL = 1e-9
_ASPECT_RATIO_HESS_ATOL = 1e-10
_PRINCIPAL_CURVATURE_VALUE_RTOL = 1e-12
_PRINCIPAL_CURVATURE_VALUE_ATOL = 1e-10
_PRINCIPAL_CURVATURE_GRAD_RTOL = 1e-10
_PRINCIPAL_CURVATURE_GRAD_ATOL = 1e-9
_QFM_VALUE_RTOL = 1e-12
_QFM_VALUE_ATOL = 1e-12
_QFM_GRAD_RTOL = 1e-10
_QFM_GRAD_ATOL = 1e-10
_PRINCIPAL_CURVATURE_KWARGS = {
    "kappamax1": 1.0,
    "kappamax2": 2.2,
    "weight1": 1.0,
    "weight2": 2.0,
}
_SCALAR_METRIC_CASES = (
    (
        "mean_cross_sectional_area",
        "surface_dmean_cross_sectional_area_jax_from_dofs",
        "surface_d2mean_cross_sectional_area_jax_from_dofs",
        "dmean_cross_sectional_area_by_dcoeff",
        "d2mean_cross_sectional_area_by_dcoeff_dcoeff",
    ),
    (
        "minor_radius",
        "surface_dminor_radius_jax_from_dofs",
        "surface_d2minor_radius_jax_from_dofs",
        "dminor_radius_by_dcoeff",
        "d2minor_radius_by_dcoeff_dcoeff",
    ),
    (
        "major_radius",
        "surface_dmajor_radius_jax_from_dofs",
        "surface_d2major_radius_jax_from_dofs",
        "dmajor_radius_by_dcoeff",
        "d2major_radius_by_dcoeff_dcoeff",
    ),
    (
        "aspect_ratio",
        "surface_daspect_ratio_jax_from_dofs",
        "surface_d2aspect_ratio_jax_from_dofs",
        "daspect_ratio_by_dcoeff",
        "d2aspect_ratio_by_dcoeff_dcoeff",
    ),
)


def _make_test_hessian_booz():
    return types.SimpleNamespace()


def _patch_traceable_hessian_solve(monkeypatch, solve_hessian_system_with_status):
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_boozer_penalty_objective_closure",
        lambda **_kwargs: "objective_fn",
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module._optimizer_jax,
        "_solve_hessian_least_squares_system_with_status",
        solve_hessian_system_with_status,
    )


def _patch_traceable_exact_warmstart_failure(monkeypatch, failed_dx):
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_exact_residual_kwargs",
        lambda _objective_kwargs: {},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_boozer_exact_residual",
        lambda x_inner, coil_set_spec, **_kwargs: x_inner + coil_set_spec,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_solve_linearization",
        lambda *_args, **_kwargs: (failed_dx, jnp.asarray(False, dtype=bool)),
    )


def _make_test_exact_failure_profile_suite(
    monkeypatch,
    baseline_x,
    failed_dx,
    *,
    objective_value=None,
):
    _patch_traceable_exact_warmstart_failure(monkeypatch, failed_dx)
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_batched_value_and_grad_pipeline",
        lambda compiled_value_and_grad_for: compiled_value_and_grad_for,
    )
    if objective_value is not None:
        monkeypatch.setattr(
            surfaceobjectives_jax_module,
            "_evaluate_traceable_total_objective",
            lambda *_args, **_kwargs: jnp.asarray(objective_value, dtype=jnp.float64),
        )

    compiled_bundle = {
        "compiled_forward_result_for": object(),
        "compiled_value_and_grad_for": object(),
        "state": {
            "objective_kwargs": {},
            "baseline_coil_dofs": jnp.asarray([0.0, 0.0], dtype=jnp.float64),
            "baseline_x": baseline_x,
            "baseline_linear_solve_factors": None,
            "optimize_G": False,
            "predictor_kind": "exact",
            "linearization_kind": "exact_jacobian",
            "linear_solve_tol": 1.0e-10,
            "linear_solve_stab": 0.0,
            "coil_set_spec_from_dofs": lambda coil_dofs: coil_dofs,
        },
    }
    exact_failure_booz = types.SimpleNamespace(
        _unpack_decision_vector_jax=lambda x, optimize_G, coil_set_spec: (
            x[:-1],
            x[-1],
            None,
        ),
        run_code_traceable=lambda *_args, **_kwargs: {
            "x": baseline_x,
            "plu": None,
            "fun": jnp.asarray(-999.0, dtype=jnp.float64),
            "success": jnp.asarray(True, dtype=bool),
            "nit": jnp.asarray(7, dtype=jnp.int64),
        },
    )
    return surfaceobjectives_jax_module._make_traceable_objective_profile_suite_from_compiled_bundle(
        compiled_bundle,
        exact_failure_booz,
        object(),
    )


def test_surface_to_surface_pairwise_distances_uses_square_primitive():
    gamma1 = jnp.asarray(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        dtype=jnp.float64,
    )
    gamma2 = jnp.asarray(
        [[0.5, -1.0, 1.5], [2.5, 0.25, -0.75], [1.0, 1.0, 1.0]],
        dtype=jnp.float64,
    )

    jaxpr = jax.make_jaxpr(
        surfaceobjectives_module.surface_to_surface_pairwise_distances
    )(gamma1, gamma2).jaxpr
    primitive_names = [eqn.primitive.name for eqn in jaxpr.eqns]

    assert "square" in primitive_names
    assert "integer_pow" not in primitive_names


def _surface_pairwise_reduction_inputs():
    gamma1 = jnp.asarray(
        [
            [0.00, 0.10, 0.03],
            [0.13, 0.17, 0.08],
            [0.24, 0.25, 0.11],
            [0.39, 0.31, 0.16],
            [0.48, 0.42, 0.21],
        ],
        dtype=jnp.float64,
    )
    gamma2 = jnp.asarray(
        [
            [0.04, 0.01, 0.19],
            [0.18, 0.07, 0.23],
            [0.29, 0.16, 0.27],
            [0.44, 0.22, 0.34],
            [0.57, 0.35, 0.38],
            [0.71, 0.41, 0.43],
            [0.83, 0.53, 0.49],
        ],
        dtype=jnp.float64,
    )
    return gamma1.reshape((5, 1, 3)), gamma2.reshape((7, 1, 3))


def _surface_distance_value_and_grad(gamma1, gamma2, minimum_distance):
    return jax.value_and_grad(
        lambda current_gamma1, current_gamma2: (
            surfaceobjectives_module.surface_to_surface_distance_pure(
                current_gamma1,
                current_gamma2,
                minimum_distance,
            )
        ),
        argnums=(0, 1),
    )(gamma1, gamma2)


def _dense_selected_smoothmin(values, temperature):
    values = values.reshape((-1,))
    bounded_temperature = jnp.maximum(
        temperature,
        jnp.asarray(np.finfo(np.float64).eps, dtype=values.dtype),
    )
    hard_min = jnp.min(values)
    logits = -(values - hard_min) / bounded_temperature
    selected = (
        values <= hard_min + jnp.asarray(4.0, dtype=values.dtype) * bounded_temperature
    )
    return hard_min - bounded_temperature * jax.nn.logsumexp(
        jnp.where(selected, logits, -jnp.inf)
    )


def _with_pairwise_chunk_size(monkeypatch, chunk_size):
    monkeypatch.setenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", str(chunk_size))
    invalidate_backend_cache()


def test_surface_to_surface_distance_chunking_matches_dense_value_and_grad(
    monkeypatch,
):
    gamma1, gamma2 = _surface_pairwise_reduction_inputs()
    minimum_distance = jnp.asarray(0.43, dtype=jnp.float64)

    _with_pairwise_chunk_size(monkeypatch, 0)
    try:
        dense_value, dense_grad = _surface_distance_value_and_grad(
            gamma1,
            gamma2,
            minimum_distance,
        )

        _with_pairwise_chunk_size(monkeypatch, 2)
        chunked_value, chunked_grad = _surface_distance_value_and_grad(
            gamma1,
            gamma2,
            minimum_distance,
        )
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    np.testing.assert_allclose(
        np.asarray(chunked_value),
        np.asarray(dense_value),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(chunked_grad[0]),
        np.asarray(dense_grad[0]),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(chunked_grad[1]),
        np.asarray(dense_grad[1]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_surface_to_surface_chunked_path_does_not_call_dense_matrix_api(
    monkeypatch,
):
    gamma1, gamma2 = _surface_pairwise_reduction_inputs()

    def fail_pairwise_distances(current_gamma1, current_gamma2):
        raise AssertionError("chunked scalar reducer must not materialize dense matrix")

    _with_pairwise_chunk_size(monkeypatch, 2)
    monkeypatch.setattr(
        surfaceobjectives_module,
        "surface_to_surface_pairwise_distances",
        fail_pairwise_distances,
    )
    try:
        value = surfaceobjectives_module.surface_to_surface_distance_pure(
            gamma1,
            gamma2,
            0.43,
        )
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    assert np.isfinite(float(value))


def test_surface_to_surface_shortest_distance_chunking_matches_dense(
    monkeypatch,
):
    gamma1, gamma2 = _surface_pairwise_reduction_inputs()

    _with_pairwise_chunk_size(monkeypatch, 2)
    try:
        chunked_min = (
            surfaceobjectives_module.surface_to_surface_shortest_distance_pure(
                gamma1,
                gamma2,
            )
        )
    finally:
        monkeypatch.setenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", "0")
        invalidate_backend_cache()

    dense_min = surfaceobjectives_module.surface_to_surface_shortest_distance_pure(
        gamma1,
        gamma2,
    )
    direct_min = np.min(
        np.linalg.norm(
            np.asarray(gamma1).reshape((-1, 3))[:, None, :]
            - np.asarray(gamma2).reshape((-1, 3))[None, :, :],
            axis=2,
        )
    )

    monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
    invalidate_backend_cache()

    assert float(chunked_min) == pytest.approx(float(dense_min), rel=1e-12, abs=1e-12)
    assert float(chunked_min) == pytest.approx(float(direct_min), rel=1e-12, abs=1e-12)


def test_surface_to_surface_chunked_gradient_respects_strict_transfer_guard(
    monkeypatch,
):
    gamma1, gamma2 = tuple(
        jax.device_put(array) for array in _surface_pairwise_reduction_inputs()
    )
    minimum_distance = jax.device_put(np.asarray(0.43, dtype=np.float64))
    seed = jax.device_put(np.asarray(1.0, dtype=np.float64))

    _with_pairwise_chunk_size(monkeypatch, 2)
    try:
        with jax.transfer_guard("disallow"):
            value, pullback = jax.vjp(
                lambda current_gamma1: (
                    surfaceobjectives_module.surface_to_surface_distance_pure(
                        current_gamma1,
                        gamma2,
                        minimum_distance,
                    )
                ),
                gamma1,
            )
            grad_gamma1 = pullback(seed)[0]
        assert np.isfinite(float(jax.device_get(value)))
        assert np.all(np.isfinite(np.asarray(jax.device_get(grad_gamma1))))
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()


def test_surface_to_surface_zero_penalty_has_finite_zero_gradient(monkeypatch):
    gamma1 = jnp.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=jnp.float64,
    ).reshape((3, 1, 3))
    gamma2 = jnp.asarray(
        [[4.0, 0.0, 0.0], [4.1, 0.0, 0.0], [4.2, 0.0, 0.0]],
        dtype=jnp.float64,
    ).reshape((3, 1, 3))

    _with_pairwise_chunk_size(monkeypatch, 2)
    try:
        value, grad_gamma1 = jax.value_and_grad(
            lambda current_gamma1: (
                surfaceobjectives_module.surface_to_surface_distance_pure(
                    current_gamma1,
                    gamma2,
                    0.2,
                )
            )
        )(gamma1)
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    assert float(value) == pytest.approx(0.0, abs=0.0)
    np.testing.assert_allclose(np.asarray(grad_gamma1), 0.0, atol=0.0)
    assert np.all(np.isfinite(np.asarray(grad_gamma1)))


def test_surface_to_surface_chunking_preserves_nan_semantics(monkeypatch):
    gamma1 = jnp.asarray(
        [[0.0, 0.0, 0.0], [jnp.nan, 0.0, 0.0]],
        dtype=jnp.float64,
    ).reshape((2, 1, 3))
    gamma2 = jnp.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=jnp.float64,
    ).reshape((2, 1, 3))

    _with_pairwise_chunk_size(monkeypatch, 0)
    try:
        dense_value = surfaceobjectives_module.surface_to_surface_distance_pure(
            gamma1,
            gamma2,
            2.0,
        )

        _with_pairwise_chunk_size(monkeypatch, 1)
        chunked_value = surfaceobjectives_module.surface_to_surface_distance_pure(
            gamma1,
            gamma2,
            2.0,
        )
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    assert np.isnan(float(dense_value))
    assert np.isnan(float(chunked_value))


def test_curve_curve_signed_constraint_chunking_matches_dense_value_and_grad(
    monkeypatch,
):
    coil_gamma = jnp.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.20, 0.05, 0.04],
            [0.41, 0.12, 0.09],
        ],
        dtype=jnp.float64,
    )
    other_coil_gamma = jnp.asarray(
        [
            [0.06, 0.18, 0.03],
            [0.24, 0.26, 0.08],
            [0.47, 0.35, 0.14],
        ],
        dtype=jnp.float64,
    )

    def value_and_grad(current_coil_gamma):
        return jax.value_and_grad(
            lambda current_gamma: (
                surfaceobjectives_jax_module._traceable_single_stage_curve_curve_signed_constraint(
                    (current_gamma, other_coil_gamma),
                    minimum_distance=0.32,
                    distance_smoothing=0.05,
                )
            )
        )(current_coil_gamma)

    _with_pairwise_chunk_size(monkeypatch, 0)
    try:
        dense_value, dense_grad = value_and_grad(coil_gamma)

        _with_pairwise_chunk_size(monkeypatch, 2)
        chunked_value, chunked_grad = value_and_grad(coil_gamma)
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    np.testing.assert_allclose(
        np.asarray(chunked_value),
        np.asarray(dense_value),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(chunked_grad),
        np.asarray(dense_grad),
        rtol=1e-10,
        atol=1e-12,
    )


def test_curve_surface_signed_constraint_chunking_matches_dense_value_and_grad(
    monkeypatch,
):
    coil_gamma = jnp.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.20, 0.05, 0.04],
            [0.41, 0.12, 0.09],
        ],
        dtype=jnp.float64,
    )
    surface_gamma, _vessel_gamma = _surface_pairwise_reduction_inputs()

    def value_and_grad(current_surface_gamma):
        return jax.value_and_grad(
            lambda current_gamma: (
                surfaceobjectives_jax_module._traceable_single_stage_curve_surface_signed_constraint(
                    (coil_gamma,),
                    current_gamma,
                    minimum_distance=0.32,
                    distance_smoothing=0.05,
                )
            )
        )(current_surface_gamma)

    _with_pairwise_chunk_size(monkeypatch, 0)
    try:
        dense_value, dense_grad = value_and_grad(surface_gamma)

        _with_pairwise_chunk_size(monkeypatch, 2)
        chunked_value, chunked_grad = value_and_grad(surface_gamma)
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    np.testing.assert_allclose(
        np.asarray(chunked_value),
        np.asarray(dense_value),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(chunked_grad),
        np.asarray(dense_grad),
        rtol=1e-10,
        atol=1e-12,
    )


def test_surface_surface_signed_constraint_chunking_matches_dense_value_and_grad(
    monkeypatch,
):
    surface_gamma, vessel_gamma = _surface_pairwise_reduction_inputs()

    def value_and_grad(current_surface_gamma):
        return jax.value_and_grad(
            lambda current_gamma: (
                surfaceobjectives_jax_module._traceable_single_stage_surface_surface_signed_constraint(
                    current_gamma,
                    vessel_gamma,
                    minimum_distance=0.32,
                    distance_smoothing=0.05,
                )
            )
        )(current_surface_gamma)

    _with_pairwise_chunk_size(monkeypatch, 0)
    try:
        dense_value, dense_grad = value_and_grad(surface_gamma)

        _with_pairwise_chunk_size(monkeypatch, 2)
        chunked_value, chunked_grad = value_and_grad(surface_gamma)
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    np.testing.assert_allclose(
        np.asarray(chunked_value),
        np.asarray(dense_value),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(chunked_grad),
        np.asarray(dense_grad),
        rtol=1e-10,
        atol=1e-12,
    )


def test_alm_smoothmin_chunked_vjps_respect_strict_transfer_guard(monkeypatch):
    coil_gamma = jax.device_put(
        jnp.asarray(
            [
                [0.00, 0.00, 0.00],
                [0.20, 0.05, 0.04],
                [0.41, 0.12, 0.09],
            ],
            dtype=jnp.float64,
        )
    )
    other_coil_gamma = jax.device_put(
        jnp.asarray(
            [
                [0.06, 0.18, 0.03],
                [0.24, 0.26, 0.08],
                [0.47, 0.35, 0.14],
            ],
            dtype=jnp.float64,
        )
    )
    surface_gamma, vessel_gamma = tuple(
        jax.device_put(array) for array in _surface_pairwise_reduction_inputs()
    )
    minimum_distance = jax.device_put(np.asarray(0.32, dtype=np.float64))
    distance_smoothing = jax.device_put(np.asarray(0.05, dtype=np.float64))
    seed = jax.device_put(np.asarray(1.0, dtype=np.float64))

    _with_pairwise_chunk_size(monkeypatch, 2)
    try:
        with jax.transfer_guard("disallow"):
            curve_curve_value, curve_curve_pullback = jax.vjp(
                lambda current_gamma: (
                    surfaceobjectives_jax_module._traceable_single_stage_curve_curve_signed_constraint(
                        (current_gamma, other_coil_gamma),
                        minimum_distance=minimum_distance,
                        distance_smoothing=distance_smoothing,
                    )
                ),
                coil_gamma,
            )
            curve_curve_grad = curve_curve_pullback(seed)[0]

            curve_surface_value, curve_surface_pullback = jax.vjp(
                lambda current_surface_gamma: (
                    surfaceobjectives_jax_module._traceable_single_stage_curve_surface_signed_constraint(
                        (coil_gamma,),
                        current_surface_gamma,
                        minimum_distance=minimum_distance,
                        distance_smoothing=distance_smoothing,
                    )
                ),
                surface_gamma,
            )
            curve_surface_grad = curve_surface_pullback(seed)[0]

            surface_surface_value, surface_surface_pullback = jax.vjp(
                lambda current_surface_gamma: (
                    surfaceobjectives_jax_module._traceable_single_stage_surface_surface_signed_constraint(
                        current_surface_gamma,
                        vessel_gamma,
                        minimum_distance=minimum_distance,
                        distance_smoothing=distance_smoothing,
                    )
                ),
                surface_gamma,
            )
            surface_surface_grad = surface_surface_pullback(seed)[0]
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    assert np.isfinite(float(jax.device_get(curve_curve_value)))
    assert np.isfinite(float(jax.device_get(curve_surface_value)))
    assert np.isfinite(float(jax.device_get(surface_surface_value)))
    assert np.all(np.isfinite(np.asarray(jax.device_get(curve_curve_grad))))
    assert np.all(np.isfinite(np.asarray(jax.device_get(curve_surface_grad))))
    assert np.all(np.isfinite(np.asarray(jax.device_get(surface_surface_grad))))


def test_pairwise_selected_smoothmin_temperature_floor_matches_dense_contract():
    points_a = jnp.asarray(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    points_b = jnp.asarray(
        [[0.11, 0.0, 0.0], [0.41, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    dists = jnp.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)
    temperature = jnp.asarray(1e-14, dtype=jnp.float64)

    expected = _dense_selected_smoothmin(dists, temperature)
    actual = pairwise_selected_smoothmin_distance_pure(
        ((points_a, points_b),),
        temperature=temperature,
        chunk_size=0,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.skipif(
    os.environ.get("SIMSOPT_RUN_PRODUCTION_PAIRWISE_PARITY") != "1",
    reason="production-shape dense pairwise parity is an opt-in memory gate",
)
def test_surface_to_surface_distance_production_shape_dense_parity(monkeypatch):
    phi = jnp.linspace(0.0, 1.0, 255, endpoint=False, dtype=jnp.float64)
    theta = jnp.linspace(0.0, 1.0, 64, endpoint=False, dtype=jnp.float64)
    grid_phi, grid_theta = jnp.meshgrid(phi, theta, indexing="ij")
    surface_gamma = jnp.stack(
        (
            0.7 * jnp.cos(2.0 * jnp.pi * grid_phi),
            0.7 * jnp.sin(2.0 * jnp.pi * grid_phi),
            0.1 * jnp.sin(2.0 * jnp.pi * grid_theta),
        ),
        axis=2,
    )
    vessel_gamma = 1.4 * surface_gamma + jnp.asarray([0.05, -0.03, 0.02])
    minimum_distance = jnp.asarray(0.25, dtype=jnp.float64)

    _with_pairwise_chunk_size(monkeypatch, 0)
    try:
        dense_value = surfaceobjectives_module.surface_to_surface_distance_pure(
            surface_gamma,
            vessel_gamma,
            minimum_distance,
        )

        _with_pairwise_chunk_size(monkeypatch, 256)
        chunked_value = surfaceobjectives_module.surface_to_surface_distance_pure(
            surface_gamma,
            vessel_gamma,
            minimum_distance,
        )
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    np.testing.assert_allclose(
        np.asarray(chunked_value),
        np.asarray(dense_value),
        rtol=1e-12,
        atol=1e-12,
    )


def test_traceable_objective_bundle_marks_value_and_grad_cacheable(monkeypatch):
    marked: dict[str, object] = {}
    original_mark = optimizer_jax_module._mark_cacheable_jit_value_and_grad

    def counting_mark(fun):
        marked["calls"] = int(marked.get("calls", 0)) + 1
        marked["fun"] = fun
        return original_mark(fun)

    monkeypatch.setattr(
        optimizer_jax_module,
        "_mark_cacheable_jit_value_and_grad",
        counting_mark,
    )

    state = {
        "objective_kwargs": {},
        "baseline_x": jnp.asarray([0.0], dtype=jnp.float64),
        "baseline_value": jnp.asarray(0.0, dtype=jnp.float64),
        "baseline_linear_solve_factors": None,
        "baseline_coil_dofs": jnp.asarray([0.0], dtype=jnp.float64),
        "coil_set_spec_from_dofs": lambda coil_dofs: coil_dofs,
        "optimize_G": False,
        "predictor_kind": "none",
        "linearization_kind": "hessian",
        "linear_solve_tol": 1.0e-10,
        "linear_solve_stab": 0.0,
    }

    bundle = surfaceobjectives_jax_module._build_traceable_objective_compiled_bundle_from_state(
        object(),
        state,
    )

    assert marked["calls"] == 1
    assert bundle["compiled_value_and_grad_for"] is marked["fun"]
    assert (
        getattr(
            bundle["compiled_value_and_grad_for"],
            optimizer_jax_module._CACHEABLE_VALUE_AND_GRAD_ATTR,
            False,
        )
        is True
    )


@pytest.mark.parametrize("linearization_kind", ["exact_jacobian", "hessian"])
def test_operator_adjoint_signoff_gate_failed_solve_returns_nan_gradient(
    monkeypatch,
    linearization_kind,
):
    baseline_coil_dofs = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    failed_gradient = jnp.asarray([0.25, -0.5], dtype=jnp.float64)
    state = {
        "objective_kwargs": {},
        "baseline_x": jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        "baseline_value": jnp.asarray(10.0, dtype=jnp.float64),
        "baseline_linear_solve_factors": None,
        "baseline_coil_dofs": baseline_coil_dofs,
        "coil_set_spec_from_dofs": lambda coil_dofs: coil_dofs,
        "optimize_G": False,
        "predictor_kind": "none",
        "linearization_kind": linearization_kind,
        "linear_solve_tol": 1.0e-10,
        "linear_solve_stab": 0.0,
    }

    def fake_forward_result(_booz_jax, _coil_set_spec_from_dofs, **_kwargs):
        return {
            "value": jnp.asarray(10.0, dtype=jnp.float64),
            "x": jnp.asarray([1.0, -1.0], dtype=jnp.float64),
            "sdofs": jnp.asarray([0.0], dtype=jnp.float64),
            "iota": jnp.asarray(0.0, dtype=jnp.float64),
            "G": jnp.asarray(0.0, dtype=jnp.float64),
            "linear_solve_factors": None,
            "success": jnp.asarray(True, dtype=bool),
            "primal_success": jnp.asarray(True, dtype=bool),
            "adjoint_linear_solve_available": jnp.asarray(False, dtype=bool),
        }

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_forward_result",
        fake_forward_result,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_total_gradient_with_status",
        lambda *_args, **_kwargs: (
            failed_gradient,
            jnp.asarray(False, dtype=bool),
        ),
    )

    bundle = surfaceobjectives_jax_module._build_traceable_objective_compiled_bundle_from_state(
        object(),
        state,
    )

    value, grad = bundle["compiled_value_and_grad_for"](baseline_coil_dofs)
    _assert_primal_value_with_nonfinite_gradient(value, grad, 10.0)


def test_checked_boozer_linear_solve_uses_explicit_host_bool_boundary(monkeypatch):
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    adjoint_state = types.SimpleNamespace(
        linearization_kind="hessian",
        solve_forward_with_status=lambda vector: (2.0 * vector, jnp.asarray(True)),
        solve_transpose_with_status=lambda vector: (3.0 * vector, jnp.asarray(True)),
    )
    original_asarray = surfaceobjectives_jax_module.np.asarray

    def reject_jax_array_asarray(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            raise AssertionError("unexpected implicit device bool materialization")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(
        surfaceobjectives_jax_module.np,
        "asarray",
        reject_jax_array_asarray,
    )

    solved = surfaceobjectives_jax_module._checked_boozer_linear_solve(
        adjoint_state,
        rhs,
        transpose=True,
    )

    np.testing.assert_allclose(
        original_asarray(solved),
        original_asarray(3.0 * rhs),
    )


def test_checked_boozer_linear_solve_rejects_factor_only_state():
    adjoint_state = types.SimpleNamespace(
        linearization_kind="exact_jacobian",
        linear_solve_factors=(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
        ),
    )

    with pytest.raises(RuntimeError, match="solve_transpose"):
        surfaceobjectives_jax_module._checked_boozer_linear_solve(
            adjoint_state,
            jnp.asarray([1.0, -2.0], dtype=jnp.float64),
            transpose=True,
        )


def test_exact_batched_adjoint_solves_each_rhs_column_via_operator():
    calls = []

    def solve_transpose_with_status(rhs):
        calls.append(np.asarray(rhs, dtype=np.float64))
        return 2.0 * rhs, jnp.asarray(True)

    adjoint_state = types.SimpleNamespace(
        linearization_kind="exact_jacobian",
        solve_transpose_with_status=solve_transpose_with_status,
        linear_solve_factors=(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
        ),
    )
    rhs_batch = jnp.asarray(
        [[1.0, -2.0], [0.5, 3.0], [-4.0, 1.25]],
        dtype=jnp.float64,
    )

    solved = surfaceobjectives_jax_module._solve_boozer_adjoint_batch(
        adjoint_state,
        rhs_batch,
    )

    assert len(calls) == rhs_batch.shape[0]
    for actual, expected in zip(calls, np.asarray(rhs_batch)):
        np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(np.asarray(solved), 2.0 * np.asarray(rhs_batch))


def test_traceable_solve_exact_linearization_uses_operator_with_factors_present(
    monkeypatch,
):
    solved_x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    rhs = jnp.asarray([0.25, 0.5], dtype=jnp.float64)
    coil_set_spec = jnp.asarray([3.0, -1.0], dtype=jnp.float64)
    calls = {}

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_exact_residual_kwargs",
        lambda _objective_kwargs: {},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_boozer_exact_residual",
        lambda x_inner, coil_set_spec, **_kwargs: x_inner + coil_set_spec,
    )

    def fake_solve_jacobian_system_with_status(
        residual_fn,
        x,
        solve_rhs,
        *,
        transpose,
        tol,
    ):
        calls["transpose"] = transpose
        calls["tol"] = tol
        np.testing.assert_allclose(np.asarray(x), np.asarray(solved_x))
        np.testing.assert_allclose(np.asarray(solve_rhs), np.asarray(rhs))
        np.testing.assert_allclose(
            np.asarray(residual_fn(x)),
            np.asarray(solved_x + coil_set_spec),
        )
        return solve_rhs + 1.0, jnp.asarray(True)

    monkeypatch.setattr(
        surfaceobjectives_jax_module._optimizer_jax,
        "_solve_jacobian_system_with_status",
        fake_solve_jacobian_system_with_status,
    )

    solved, success = surfaceobjectives_jax_module._traceable_solve_exact_linearization(
        solved_x,
        rhs,
        coil_set_spec,
        {},
        linear_solve_tol=1.0e-8,
        transpose=True,
    )

    assert calls == {"transpose": True, "tol": 1.0e-8}
    assert bool(np.asarray(success))
    np.testing.assert_allclose(np.asarray(solved), np.asarray(rhs + 1.0))


def test_traceable_exact_operator_and_dense_reference_share_residual_contract(
    monkeypatch,
):
    solved_x = jnp.asarray([0.2, -0.3], dtype=jnp.float64)
    rhs = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    matrix = jnp.asarray([[2.0, 0.25], [-0.1, 1.5]], dtype=jnp.float64)

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_exact_residual_kwargs",
        lambda _objective_kwargs: {},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_boozer_exact_residual",
        lambda x_inner, coil_set_spec, **_kwargs: matrix @ x_inner + coil_set_spec,
    )

    operator_solved, operator_success = (
        surfaceobjectives_jax_module._traceable_solve_exact_linearization(
            solved_x,
            rhs,
            jnp.zeros_like(rhs),
            {},
            linear_solve_tol=1.0e-10,
            transpose=True,
        )
    )
    dense_matrix_t = np.asarray(matrix.T)
    rhs_np = np.asarray(rhs)
    dense_solved = np.linalg.solve(dense_matrix_t, rhs_np)
    dense_residual = rhs_np - dense_matrix_t @ dense_solved
    dense_residual_norm = np.linalg.norm(dense_residual)
    dense_residual_tol = max(
        1.0e-12,
        10.0 * 1.0e-10 * max(np.linalg.norm(rhs_np), 1.0),
    )
    dense_success = (
        np.all(np.isfinite(dense_solved))
        and np.all(np.isfinite(dense_residual))
        and np.isfinite(dense_residual_norm)
        and dense_residual_norm <= dense_residual_tol
    )

    assert bool(np.asarray(operator_success)) is True
    assert dense_success
    np.testing.assert_allclose(
        np.asarray(matrix.T @ operator_solved),
        np.asarray(rhs),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        dense_matrix_t @ dense_solved,
        rhs_np,
        rtol=1e-9,
        atol=1e-9,
    )


def test_traceable_exact_warmstart_prediction_uses_operator_solve(monkeypatch):
    baseline_x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    baseline_coil_dofs = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    coil_dofs = jnp.asarray([0.75, 0.25], dtype=jnp.float64)
    delta = coil_dofs - baseline_coil_dofs
    calls = {}

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_exact_residual_kwargs",
        lambda _objective_kwargs: {},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_boozer_exact_residual",
        lambda x_inner, coil_set_spec, **_kwargs: x_inner + coil_set_spec,
    )

    def fake_solve_jacobian_system_with_status(
        residual_fn,
        x,
        rhs,
        *,
        transpose,
        tol,
    ):
        calls["transpose"] = transpose
        calls["tol"] = tol
        np.testing.assert_allclose(np.asarray(x), np.asarray(baseline_x))
        np.testing.assert_allclose(np.asarray(rhs), np.asarray(-delta))
        np.testing.assert_allclose(
            np.asarray(residual_fn(x)),
            np.asarray(baseline_x + baseline_coil_dofs),
        )
        return rhs, jnp.asarray(True)

    monkeypatch.setattr(
        surfaceobjectives_jax_module._optimizer_jax,
        "_solve_jacobian_system_with_status",
        fake_solve_jacobian_system_with_status,
    )

    predicted, success = surfaceobjectives_jax_module._traceable_predict_warmstart_x(
        object(),
        lambda current_coil_dofs: current_coil_dofs,
        coil_dofs=coil_dofs,
        baseline_coil_dofs=baseline_coil_dofs,
        baseline_x=baseline_x,
        baseline_linear_solve_factors=(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
        ),
        linearization_kind="exact_jacobian",
        linear_solve_tol=1.0e-7,
        linear_solve_stab=0.0,
        predictor_kind="exact",
        objective_kwargs={},
    )

    assert calls == {"transpose": False, "tol": 1.0e-7}
    assert bool(np.asarray(success)) is True
    np.testing.assert_allclose(
        np.asarray(predicted),
        np.asarray(baseline_x - delta),
    )


def test_traceable_exact_warmstart_success_matches_reference_operator_linearization(
    monkeypatch,
):
    baseline_x = jnp.asarray([0.25, -0.4], dtype=jnp.float64)
    baseline_coil_dofs = jnp.asarray([0.5, -0.1], dtype=jnp.float64)
    coil_dofs = jnp.asarray([0.7, 0.3], dtype=jnp.float64)
    delta_np = np.asarray(coil_dofs - baseline_coil_dofs)
    baseline_x_np = np.asarray(baseline_x)
    baseline_coil_dofs_np = np.asarray(baseline_coil_dofs)
    A_np = np.asarray(
        [[2.0, 0.1], [0.05, 1.8]],
        dtype=float,
    )
    B_np = np.asarray(
        [[1.0, -0.25], [0.4, 0.75]],
        dtype=float,
    )
    A = jnp.asarray(A_np, dtype=jnp.float64)
    B = jnp.asarray(B_np, dtype=jnp.float64)
    forcing_np = B_np @ delta_np
    dx_ref_np = np.linalg.solve(A_np, -forcing_np)
    calls = {}

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_exact_residual_kwargs",
        lambda _objective_kwargs: {},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_boozer_exact_residual",
        lambda x_inner, coil_set_spec, **_kwargs: A @ x_inner + B @ coil_set_spec,
    )

    def fake_solve_jacobian_system_with_status(
        residual_fn,
        x,
        rhs,
        *,
        transpose,
        tol,
    ):
        calls["transpose"] = transpose
        calls["tol"] = tol
        np.testing.assert_allclose(np.asarray(x), baseline_x_np)
        np.testing.assert_allclose(np.asarray(rhs), -forcing_np)
        np.testing.assert_allclose(
            np.asarray(residual_fn(x)),
            A_np @ baseline_x_np + B_np @ baseline_coil_dofs_np,
        )
        linear_operator = A.T if transpose else A
        return jnp.linalg.solve(linear_operator, rhs), jnp.asarray(True, dtype=bool)

    monkeypatch.setattr(
        surfaceobjectives_jax_module._optimizer_jax,
        "_solve_jacobian_system_with_status",
        fake_solve_jacobian_system_with_status,
    )

    predicted, success = surfaceobjectives_jax_module._traceable_predict_warmstart_x(
        object(),
        lambda current_coil_dofs: current_coil_dofs,
        coil_dofs=coil_dofs,
        baseline_coil_dofs=baseline_coil_dofs,
        baseline_x=baseline_x,
        baseline_linear_solve_factors=None,
        linearization_kind="exact_jacobian",
        linear_solve_tol=1.0e-7,
        linear_solve_stab=0.0,
        predictor_kind="exact",
        objective_kwargs={},
    )

    assert calls == {"transpose": False, "tol": 1.0e-7}
    assert bool(np.asarray(success)) is True
    np.testing.assert_allclose(
        np.asarray(predicted),
        baseline_x_np + dx_ref_np,
    )
    assert not np.allclose(np.asarray(predicted), baseline_x_np)


def test_traceable_hessian_solve_uses_configured_stabilization_once(monkeypatch):
    solved_x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    rhs = jnp.asarray([0.25, -0.5], dtype=jnp.float64)
    calls = []

    def fake_solve_hessian_system_with_status(
        objective_fn,
        current_x,
        current_rhs,
        *,
        stab,
        tol,
    ):
        del objective_fn, tol
        calls.append(stab)
        np.testing.assert_allclose(np.asarray(current_x), np.asarray(solved_x))
        np.testing.assert_allclose(np.asarray(current_rhs), np.asarray(rhs))
        assert stab == pytest.approx(1.0e-4)
        return 2.0 * current_rhs, jnp.asarray(True, dtype=bool)

    _patch_traceable_hessian_solve(
        monkeypatch,
        fake_solve_hessian_system_with_status,
    )

    solution, success = surfaceobjectives_jax_module._traceable_solve_linearization(
        _make_test_hessian_booz(),
        solved_x,
        rhs,
        coil_set_spec=object(),
        objective_kwargs={},
        linear_solve_factors=None,
        linearization_kind="hessian",
        linear_solve_tol=1.0e-10,
        linear_solve_stab=1.0e-4,
        transpose=True,
    )

    assert calls == [1.0e-4]
    assert bool(np.asarray(success)) is True
    np.testing.assert_allclose(np.asarray(solution), np.asarray(2.0 * rhs))


def test_traceable_hessian_solve_uses_configured_stabilization_under_jit(
    monkeypatch,
):
    solved_x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    rhs = jnp.asarray([0.25, -0.5], dtype=jnp.float64)
    recorded_stabs = []

    def _record_stab(stab):
        recorded_stabs.append(float(np.asarray(stab)))

    def fake_solve_hessian_system_with_status(
        objective_fn,
        current_x,
        current_rhs,
        *,
        stab,
        tol,
    ):
        del objective_fn, current_x, tol
        stab_value = jnp.asarray(stab, dtype=current_rhs.dtype)
        jax.debug.callback(_record_stab, stab_value, ordered=True)
        success = stab_value == jnp.asarray(1.0e-4, dtype=current_rhs.dtype)
        return 2.0 * current_rhs, success

    _patch_traceable_hessian_solve(
        monkeypatch,
        fake_solve_hessian_system_with_status,
    )

    compiled_solve = jax.jit(
        lambda current_rhs: surfaceobjectives_jax_module._traceable_solve_linearization(
            _make_test_hessian_booz(),
            solved_x,
            current_rhs,
            coil_set_spec=None,
            objective_kwargs={},
            linear_solve_factors=None,
            linearization_kind="hessian",
            linear_solve_tol=1.0e-10,
            linear_solve_stab=1.0e-4,
            transpose=True,
        )
    )

    solution, success = compiled_solve(rhs)
    np.testing.assert_allclose(np.asarray(solution), np.asarray(2.0 * rhs))
    assert bool(np.asarray(success)) is True
    np.testing.assert_allclose(
        np.asarray(recorded_stabs),
        np.asarray([1.0e-4]),
    )


def test_traceable_hessian_solve_uses_dense_plu_forward_and_transpose():
    matrix = jnp.asarray(
        [
            [3.0, -0.25, 0.5],
            [0.75, 2.5, -0.2],
            [-0.1, 0.4, 1.75],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([0.2, -0.6, 0.9], dtype=jnp.float64)
    linear_solve_factors = jax.scipy.linalg.lu(matrix)

    forward_solution, forward_success = (
        surfaceobjectives_jax_module._traceable_solve_linearization(
            object(),
            jnp.zeros_like(rhs),
            rhs,
            coil_set_spec=None,
            objective_kwargs={},
            linear_solve_factors=linear_solve_factors,
            linearization_kind="hessian",
            linear_solve_tol=1.0e-10,
            linear_solve_stab=0.0,
            transpose=False,
        )
    )
    transpose_solution, transpose_success = (
        surfaceobjectives_jax_module._traceable_solve_linearization(
            object(),
            jnp.zeros_like(rhs),
            rhs,
            coil_set_spec=None,
            objective_kwargs={},
            linear_solve_factors=linear_solve_factors,
            linearization_kind="hessian",
            linear_solve_tol=1.0e-10,
            linear_solve_stab=0.0,
            transpose=True,
        )
    )

    assert bool(np.asarray(forward_success)) is True
    assert bool(np.asarray(transpose_success)) is True
    np.testing.assert_allclose(
        np.asarray(forward_solution),
        np.linalg.solve(np.asarray(matrix), np.asarray(rhs)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(transpose_solution),
        np.linalg.solve(np.asarray(matrix.T), np.asarray(rhs)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_traceable_hessian_plu_solve_requires_forward_error_gate(monkeypatch):
    matrix = jnp.asarray(
        [
            [3.0, 0.25],
            [-0.5, 2.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([0.2, -0.6], dtype=jnp.float64)
    linear_solve_factors = jax.scipy.linalg.lu(matrix)
    residual_norm_calls = {"count": 0}
    original_relative_residual_1_norm = optimizer_jax_module._relative_residual_1_norm

    def relative_residual_1_norm(residual, current_rhs):
        residual_norm_calls["count"] += 1
        return original_relative_residual_1_norm(residual, current_rhs)

    monkeypatch.setattr(
        optimizer_jax_module,
        "_forward_error_success",
        lambda *_args, **_kwargs: jnp.asarray(False),
    )
    monkeypatch.setattr(
        optimizer_jax_module,
        "_relative_residual_1_norm",
        relative_residual_1_norm,
    )

    _solution, success = surfaceobjectives_jax_module._traceable_solve_linearization(
        object(),
        jnp.zeros_like(rhs),
        rhs,
        coil_set_spec=None,
        objective_kwargs={},
        linear_solve_factors=linear_solve_factors,
        linearization_kind="hessian",
        linear_solve_tol=1.0e-10,
        linear_solve_stab=0.0,
        transpose=False,
    )

    assert bool(np.asarray(success)) is False
    assert residual_norm_calls["count"] == 1


def test_traceable_hessian_plu_solve_is_jittable():
    matrix = jnp.asarray(
        [
            [4.0, 0.1, -0.3],
            [0.2, 3.5, 0.6],
            [-0.1, 0.4, 2.25],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.2, -0.4, 0.7], dtype=jnp.float64)
    linear_solve_factors = jax.scipy.linalg.lu(matrix)

    compiled_solve = jax.jit(
        lambda current_rhs: surfaceobjectives_jax_module._traceable_solve_linearization(
            object(),
            jnp.zeros_like(current_rhs),
            current_rhs,
            coil_set_spec=None,
            objective_kwargs={},
            linear_solve_factors=linear_solve_factors,
            linearization_kind="hessian",
            linear_solve_tol=1.0e-10,
            linear_solve_stab=0.0,
            transpose=True,
        )
    )

    solution, success = compiled_solve(rhs)

    assert bool(np.asarray(success)) is True
    np.testing.assert_allclose(
        np.asarray(solution),
        np.linalg.solve(np.asarray(matrix.T), np.asarray(rhs)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_traceable_exact_warmstart_failure_keeps_failed_operator_step(monkeypatch):
    baseline_x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    baseline_coil_dofs = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    coil_dofs = jnp.asarray([0.75, 0.25], dtype=jnp.float64)
    failed_dx = jnp.asarray([0.125, -0.375], dtype=jnp.float64)

    _patch_traceable_exact_warmstart_failure(monkeypatch, failed_dx)

    predicted, success = surfaceobjectives_jax_module._traceable_predict_warmstart_x(
        object(),
        lambda current_coil_dofs: current_coil_dofs,
        coil_dofs=coil_dofs,
        baseline_coil_dofs=baseline_coil_dofs,
        baseline_x=baseline_x,
        baseline_linear_solve_factors=None,
        linearization_kind="exact_jacobian",
        linear_solve_tol=1.0e-7,
        linear_solve_stab=0.0,
        predictor_kind="exact",
        objective_kwargs={},
    )

    assert bool(np.asarray(success)) is False
    np.testing.assert_allclose(
        np.asarray(predicted),
        np.asarray(baseline_x + failed_dx),
    )


def test_traceable_ls_warmstart_failure_preserves_baseline_state(monkeypatch):
    baseline_x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    baseline_coil_dofs = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    coil_dofs = jnp.asarray([0.75, 0.25], dtype=jnp.float64)
    failed_dx = jnp.asarray([0.125, -0.375], dtype=jnp.float64)

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_inner_stationarity_coil_jvp",
        lambda *_args, **_kwargs: jnp.asarray([0.25, -0.5], dtype=jnp.float64),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_solve_linearization",
        lambda *_args, **_kwargs: (failed_dx, jnp.asarray(False, dtype=bool)),
    )

    predicted, success = surfaceobjectives_jax_module._traceable_predict_warmstart_x(
        object(),
        lambda current_coil_dofs: current_coil_dofs,
        coil_dofs=coil_dofs,
        baseline_coil_dofs=baseline_coil_dofs,
        baseline_x=baseline_x,
        baseline_linear_solve_factors=(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
        ),
        linearization_kind="hessian",
        linear_solve_tol=1.0e-7,
        linear_solve_stab=0.0,
        predictor_kind="ls",
        objective_kwargs={},
    )

    assert bool(np.asarray(success)) is False
    np.testing.assert_allclose(np.asarray(predicted), np.asarray(baseline_x))


def test_traceable_exact_warmstart_failure_surfaces_unsuccessful_forward_result(
    monkeypatch,
):
    baseline_x = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    failed_dx = jnp.asarray([0.125, -0.375], dtype=jnp.float64)

    _patch_traceable_exact_warmstart_failure(monkeypatch, failed_dx)
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_evaluate_traceable_total_objective",
        lambda *_args, **_kwargs: jnp.asarray(1.0, dtype=jnp.float64),
    )

    booz = types.SimpleNamespace(
        _unpack_decision_vector_jax=lambda x, optimize_G, coil_set_spec: (
            x[:-1],
            x[-1],
            None,
        ),
        run_code_traceable=lambda *_args, **_kwargs: {
            "x": baseline_x,
            "plu": None,
            "success": jnp.asarray(True, dtype=bool),
            "primal_success": jnp.asarray(True, dtype=bool),
            "adjoint_linear_solve_available": jnp.asarray(True, dtype=bool),
        },
    )

    result = surfaceobjectives_jax_module._traceable_general_forward_result(
        booz,
        lambda coil_dofs: coil_dofs,
        coil_dofs=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        baseline_x=baseline_x,
        baseline_linear_solve_factors=None,
        linearization_kind="exact_jacobian",
        linear_solve_tol=1.0e-10,
        linear_solve_stab=0.0,
        optimize_G=False,
        baseline_coil_dofs=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        predictor_kind="exact",
        objective_kwargs={},
        success_filter=None,
    )

    assert bool(result["success"]) is False
    assert bool(result["primal_success"]) is False
    assert bool(result["adjoint_linear_solve_available"]) is False
    np.testing.assert_allclose(np.asarray(result["value"]), np.asarray(2.0))
    np.testing.assert_allclose(
        np.asarray(result["x"]),
        np.asarray(baseline_x + failed_dx),
    )


def test_traceable_profile_suite_warmstart_predict_surfaces_exact_failure(
    monkeypatch,
):
    baseline_x = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    failed_dx = jnp.asarray([0.125, -0.375], dtype=jnp.float64)
    profile_suite = _make_test_exact_failure_profile_suite(
        monkeypatch,
        baseline_x,
        failed_dx,
    )

    warmstart = profile_suite["warmstart_predict"](
        jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    )

    assert bool(np.asarray(warmstart["success"])) is False
    np.testing.assert_allclose(
        np.asarray(warmstart["x"]),
        np.asarray(baseline_x + failed_dx),
    )


def test_traceable_profile_suite_inner_solve_surfaces_exact_failure_state(
    monkeypatch,
):
    baseline_x = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    failed_dx = jnp.asarray([0.125, -0.375], dtype=jnp.float64)
    profile_suite = _make_test_exact_failure_profile_suite(
        monkeypatch,
        baseline_x,
        failed_dx,
        objective_value=1.0,
    )

    solve_result = profile_suite["inner_solve"](
        jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    )

    assert bool(np.asarray(solve_result["success"])) is False
    np.testing.assert_allclose(
        np.asarray(solve_result["x"]),
        np.asarray(baseline_x + failed_dx),
    )
    np.testing.assert_allclose(np.asarray(solve_result["fun"]), np.asarray(1.0))


def test_traceable_runtime_cache_key_avoids_value_hashing_runtime_state(monkeypatch):
    seen_trees = []
    optimizer_option_methods = []
    original_tree_signature = (
        surfaceobjectives_jax_module._traceable_cache_tree_signature
    )

    def recording_tree_signature(tree):
        seen_trees.append(tree)
        return original_tree_signature(tree)

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_cache_tree_signature",
        recording_tree_signature,
    )

    def collect_optimizer_options(*, method):
        optimizer_option_methods.append(method)
        return {}

    booz = types.SimpleNamespace(
        _solver_generation=7,
        options={},
        _collect_optimizer_options=collect_optimizer_options,
    )
    state = {
        "objective_kwargs": {
            "iota_target": 0.23,
            "outer_objective_config": {
                "curve_curve_weight": 1.0,
                "vessel_gamma": jnp.ones((8, 3), dtype=jnp.float64),
            },
        },
        "optimize_G": False,
        "predictor_kind": "ls",
        "objective_method": "bfgs-ondevice",
        "coil_dof_extraction_spec": {"unused": True},
        "baseline_x": jnp.arange(5, dtype=jnp.float64),
        "baseline_value": jnp.asarray(1.0, dtype=jnp.float64),
        "baseline_linear_solve_factors": (
            jnp.eye(3, dtype=jnp.float64),
            jnp.eye(3, dtype=jnp.float64),
            jnp.arange(3, dtype=jnp.int32),
        ),
        "baseline_coil_dofs": jnp.arange(4, dtype=jnp.float64),
        "linearization_kind": "hessian",
        "linear_solve_tol": 1.0e-10,
        "linear_solve_stab": 0.0,
    }

    surfaceobjectives_jax_module._traceable_runtime_cache_key(
        booz,
        object(),
        state,
        success_filter=None,
    )

    assert not any(tree is state["objective_kwargs"] for tree in seen_trees)
    assert not any(tree is state["baseline_x"] for tree in seen_trees)
    assert not any(tree is state["baseline_value"] for tree in seen_trees)
    assert not any(
        tree is state["baseline_linear_solve_factors"] for tree in seen_trees
    )
    assert not any(tree is state["baseline_coil_dofs"] for tree in seen_trees)
    assert optimizer_option_methods == ["bfgs-ondevice"]


def test_traceable_forward_result_keeps_primal_success_separate_from_adjoint_status(
    monkeypatch,
):
    objective_value = jnp.asarray(-123.0, dtype=jnp.float64)
    baseline_x = jnp.asarray([0.5, -0.25, 0.31], dtype=jnp.float64)

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_predict_warmstart_x",
        lambda *_args, **_kwargs: (baseline_x, jnp.asarray(True, dtype=bool)),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_evaluate_traceable_total_objective",
        lambda *_args, **_kwargs: objective_value,
    )

    booz = types.SimpleNamespace(
        _unpack_decision_vector_jax=lambda x, optimize_G, coil_set_spec: (
            x[:-1],
            x[-1],
            None,
        ),
        run_code_traceable=lambda coil_set_spec, warmstart_sdofs, warmstart_iota, warmstart_G: {
            "x": jnp.concatenate(
                (
                    warmstart_sdofs + 1.0,
                    jnp.asarray([warmstart_iota], dtype=jnp.float64),
                )
            ),
            "plu": None,
            "success": jnp.asarray(True, dtype=bool),
            "primal_success": jnp.asarray(True, dtype=bool),
            "adjoint_linear_solve_available": jnp.asarray(False, dtype=bool),
        },
    )

    coil_dofs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    baseline_coil_dofs = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    result = surfaceobjectives_jax_module._traceable_forward_result(
        booz,
        lambda dofs: {"coil_dofs": dofs},
        coil_dofs=coil_dofs,
        baseline_x=baseline_x,
        baseline_value=jnp.asarray(3.0, dtype=jnp.float64),
        baseline_linear_solve_factors=None,
        linearization_kind="hessian",
        linear_solve_tol=1.0e-10,
        linear_solve_stab=0.0,
        optimize_G=False,
        baseline_coil_dofs=baseline_coil_dofs,
        predictor_kind="ls",
        objective_kwargs={},
        success_filter=lambda _coil_dofs, _solved_x: jnp.asarray(True, dtype=bool),
    )

    assert bool(result["primal_success"]) is True
    assert bool(result["adjoint_linear_solve_available"]) is False
    assert bool(result["success"]) is True
    np.testing.assert_allclose(np.asarray(result["value"]), -123.0)
    np.testing.assert_allclose(np.asarray(objective_value), -123.0)


def test_traceable_runtime_cache_key_uses_structural_success_filter_signature():
    booz = types.SimpleNamespace(
        _solver_generation=7,
        options={},
        _collect_optimizer_options=lambda *, method: {},
    )
    state = {
        "objective_kwargs": {
            "iota_target": 0.23,
            "outer_objective_config": None,
        },
        "optimize_G": False,
        "predictor_kind": "ls",
        "objective_method": "bfgs-ondevice",
    }

    def success_filter_a(_coil_dofs, _solved_x):
        return jnp.asarray(True, dtype=bool)

    def success_filter_b(_coil_dofs, _solved_x):
        return jnp.asarray(True, dtype=bool)

    signature = ("single-stage-target-lane-hardware-success-filter", "sig-123")
    success_filter_a._traceable_runtime_cache_signature = signature
    success_filter_b._traceable_runtime_cache_signature = signature

    key_a = surfaceobjectives_jax_module._traceable_runtime_cache_key(
        booz,
        object(),
        state,
        success_filter=success_filter_a,
    )
    key_b = surfaceobjectives_jax_module._traceable_runtime_cache_key(
        booz,
        object(),
        state,
        success_filter=success_filter_b,
    )

    assert key_a == key_b


def test_traceable_runtime_cache_key_tracks_biotsavart_dof_generation():
    booz = types.SimpleNamespace(
        _solver_generation=7,
        options={},
        _collect_optimizer_options=lambda *, method: {},
    )
    bs_jax = types.SimpleNamespace(_coil_dofs_generation=11)
    state = {
        "objective_kwargs": {
            "iota_target": 0.23,
            "outer_objective_config": None,
        },
        "optimize_G": False,
        "predictor_kind": "ls",
        "objective_method": "bfgs-ondevice",
    }

    key_before = surfaceobjectives_jax_module._traceable_runtime_cache_key(
        booz,
        bs_jax,
        state,
        success_filter=None,
    )
    bs_jax._coil_dofs_generation = 12
    key_after = surfaceobjectives_jax_module._traceable_runtime_cache_key(
        booz,
        bs_jax,
        state,
        success_filter=None,
    )

    assert key_before != key_after


def test_traceable_cache_signoff_gate_covers_runtime_contract_inputs():
    def make_booz(*, solver_generation=7, bfgs_maxiter=5):
        return types.SimpleNamespace(
            _solver_generation=solver_generation,
            options={"bfgs_maxiter": bfgs_maxiter},
            _collect_optimizer_options=lambda *, method: {"maxls": 20},
        )

    def make_state(*, iota_target=0.23):
        return {
            "objective_kwargs": {
                "iota_target": iota_target,
                "outer_objective_config": None,
            },
            "optimize_G": False,
            "predictor_kind": "ls",
            "objective_method": "bfgs-ondevice",
        }

    def success_filter_a(_coil_dofs, _solved_x):
        return jnp.asarray(True, dtype=bool)

    def success_filter_b(_coil_dofs, _solved_x):
        return jnp.asarray(True, dtype=bool)

    success_filter_a._traceable_runtime_cache_signature = ("filter", "a")
    success_filter_b._traceable_runtime_cache_signature = ("filter", "b")

    booz = make_booz()
    bs_jax = types.SimpleNamespace(_coil_dofs_generation=11)
    state = make_state()
    base_key = surfaceobjectives_jax_module._traceable_runtime_cache_key(
        booz,
        bs_jax,
        state,
        success_filter=success_filter_a,
    )

    assert (
        surfaceobjectives_jax_module._traceable_runtime_cache_key(
            make_booz(solver_generation=8),
            bs_jax,
            state,
            success_filter=success_filter_a,
        )
        != base_key
    )
    assert (
        surfaceobjectives_jax_module._traceable_runtime_cache_key(
            booz,
            bs_jax,
            make_state(iota_target=0.24),
            success_filter=success_filter_a,
        )
        != base_key
    )
    assert (
        surfaceobjectives_jax_module._traceable_runtime_cache_key(
            booz,
            bs_jax,
            state,
            success_filter=success_filter_b,
        )
        != base_key
    )
    assert (
        surfaceobjectives_jax_module._traceable_runtime_cache_key(
            booz,
            types.SimpleNamespace(_coil_dofs_generation=12),
            state,
            success_filter=success_filter_a,
        )
        != base_key
    )
    assert (
        surfaceobjectives_jax_module._traceable_runtime_cache_key(
            make_booz(bfgs_maxiter=6),
            bs_jax,
            state,
            success_filter=success_filter_a,
        )
        != base_key
    )


def test_traceable_runtime_cache_key_does_not_hostify_jax_array_contract_leaves(
    monkeypatch,
):
    original_asarray = surfaceobjectives_jax_module.np.asarray

    def guarded_asarray(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            raise AssertionError("jax.Array contract leaves must stay on device")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(surfaceobjectives_jax_module.np, "asarray", guarded_asarray)

    booz = types.SimpleNamespace(
        _solver_generation=7,
        options={},
        _collect_optimizer_options=lambda *, method: {},
    )
    state = {
        "objective_kwargs": {
            "iota_target": 0.23,
            "outer_objective_config": {
                "coil_surface_weight": 1.0,
                "vessel_normal": jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
            },
        },
        "optimize_G": False,
        "predictor_kind": "ls",
        "objective_method": "bfgs-ondevice",
    }

    key = surfaceobjectives_jax_module._traceable_runtime_cache_key(
        booz,
        object(),
        state,
        success_filter=None,
    )

    assert key[6][0] == "tree"


def test_traceable_runtime_hostify_tree_explicitly_materializes_jax_array_leaves():
    hostified = surfaceobjectives_jax_module._traceable_runtime_hostify_tree(
        {
            "vector": jax.device_put(np.array([1.0, -2.0], dtype=np.float64)),
            "nested": (
                jnp.asarray([3, 4], dtype=jnp.int32),
                {"plain": 5.0},
            ),
        }
    )

    leaves = jax.tree_util.tree_leaves(hostified)

    assert not any(isinstance(leaf, jax.Array) for leaf in leaves)
    assert isinstance(hostified["vector"], np.ndarray)
    assert isinstance(hostified["nested"][0], np.ndarray)


def test_boozer_residual_inner_evaluates_label_on_label_geometry(monkeypatch):
    seen = {}

    def fake_geometry_from_dofs(
        _sdofs,
        quadpoints_phi,
        quadpoints_theta,
        _mpol,
        _ntor,
        _nfp,
        _stellsym,
        _scatter_indices,
        *,
        surface_kind,
    ):
        nphi = int(np.asarray(quadpoints_phi).shape[0])
        ntheta = int(np.asarray(quadpoints_theta).shape[0])
        marker = 10.0 if surface_kind == "label-kind" else 1.0
        gamma = jnp.full((nphi, ntheta, 3), marker, dtype=jnp.float64)
        xphi = jnp.ones_like(gamma)
        xtheta = jnp.full_like(gamma, 2.0)
        return gamma, xphi, xtheta

    def fake_compute_label(
        _label_type,
        geometry,
        _phi_idx,
        points,
        *,
        coil_set_spec,
    ):
        seen["gamma"] = geometry.gamma
        seen["points_shape"] = points.shape
        seen["coil_set_spec"] = coil_set_spec
        return jnp.asarray(2.0, dtype=jnp.float64)

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_surface_geometry_from_dofs",
        fake_geometry_from_dofs,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "grouped_biot_savart_B_from_spec",
        lambda points, _coil_set_spec: jnp.zeros(
            (points.shape[0], 3),
            dtype=jnp.float64,
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "boozer_residual_scalar",
        lambda *_args, **_kwargs: jnp.asarray(0.0, dtype=jnp.float64),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_compute_label",
        fake_compute_label,
    )

    coil_set_spec = object()
    value = surfaceobjectives_jax_module._boozer_residual_J_of_x_inner(
        jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64),
        coil_set_spec=coil_set_spec,
        quadpoints_phi=jnp.asarray([0.0, 0.5], dtype=jnp.float64),
        quadpoints_theta=jnp.asarray([0.0, 0.5], dtype=jnp.float64),
        mpol=1,
        ntor=1,
        nfp=1,
        stellsym=True,
        scatter_indices=jnp.asarray([0, 1], dtype=jnp.int32),
        surface_kind="solve-kind",
        label_quadpoints_phi=jnp.asarray([0.0, 0.25, 0.5], dtype=jnp.float64),
        label_quadpoints_theta=jnp.asarray([0.0, 0.5], dtype=jnp.float64),
        label_mpol=1,
        label_ntor=1,
        label_nfp=1,
        label_stellsym=True,
        label_scatter_indices=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        label_surface_kind="label-kind",
        optimize_G=True,
        weight_inv_modB=False,
        constraint_weight=4.0,
        targetlabel=1.0,
        label_type="volume",
        phi_idx=0,
    )

    assert seen["coil_set_spec"] is coil_set_spec
    assert seen["points_shape"] == (6, 3)
    np.testing.assert_allclose(np.asarray(seen["gamma"]), 10.0)
    np.testing.assert_allclose(np.asarray(value), 2.0)


def test_build_traceable_objective_state_hostifies_runtime_constants(monkeypatch):
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_ensure_solved_value_state",
        lambda _booz: None,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_canonicalize_traceable_exact_quadrature",
        lambda _booz: (
            jnp.asarray([0.0, 0.25], dtype=jnp.float64),
            jnp.asarray([0.0, 0.5], dtype=jnp.float64),
            jnp.asarray([0, 1], dtype=jnp.int32),
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_evaluate_traceable_total_objective",
        lambda *_args, **_kwargs: jnp.asarray(3.5, dtype=jnp.float64),
    )

    class _FakeSurface:
        quadpoints_phi = np.asarray([0.0, 0.5], dtype=np.float64)
        quadpoints_theta = np.asarray([0.0, 0.5], dtype=np.float64)

        def get_dofs(self):
            raise AssertionError(
                "traceable runtime build must use solved runtime state"
            )

    class _FakeBooz:
        boozer_type = "ls"
        surface = _FakeSurface()
        res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "weight_inv_modB": False,
            "linearization_kind": "hessian",
            "dense_linear_solve_factors_available": True,
            "iota": jnp.asarray(0.23, dtype=jnp.float64),
            "G": jnp.asarray(1.7, dtype=jnp.float64),
            "PLU": (
                jnp.eye(2, dtype=jnp.float64),
                jnp.eye(2, dtype=jnp.float64),
                jnp.asarray([0, 1], dtype=jnp.int32),
            ),
            "vjp": object(),
        }
        quadpoints_phi = np.asarray([0.0, 0.5], dtype=np.float64)
        quadpoints_theta = np.asarray([0.0, 0.5], dtype=np.float64)
        mpol = 1
        ntor = 1
        nfp = 1
        stellsym = True
        scatter_indices = np.asarray([0, 1], dtype=np.int32)
        _surface_geometry_kind = "surface-geometry-marker"
        label_quadpoints_phi = np.asarray([0.0, 0.25], dtype=np.float64)
        label_quadpoints_theta = np.asarray([0.0, 0.5], dtype=np.float64)
        label_mpol = 1
        label_ntor = 1
        label_nfp = 1
        label_stellsym = True
        label_scatter_indices = np.asarray([0, 1], dtype=np.int32)
        _label_surface_geometry_kind = "label-geometry-marker"
        options = {"weight_inv_modB": False}
        constraint_weight = 1.0
        targetlabel = 0.0
        label_type = "iota"
        phi_idx = 0
        need_to_run_code = False

        def _resolve_optimizer_method(self):
            return "lbfgs-ondevice"

        def _linear_solve_tolerance(self):
            return 1.0e-10

        def _pack_decision_vector(self, iota, G, *, sdofs):
            return jnp.concatenate(
                [
                    jnp.asarray(sdofs, dtype=jnp.float64),
                    jnp.asarray([iota], dtype=jnp.float64),
                    jnp.asarray([G], dtype=jnp.float64),
                ]
            )

        def get_solved_runtime_state(self):
            return types.SimpleNamespace(
                sdofs=jnp.asarray([1.0, 0.1], dtype=jnp.float64),
                iota=jnp.asarray(0.23, dtype=jnp.float64),
                G=jnp.asarray(1.7, dtype=jnp.float64),
                weight_inv_modB=False,
            )

    class _FakeBS:
        x = np.asarray([0.2, -0.1], dtype=np.float64)

        def coil_dof_extraction_spec(self):
            return {
                "gamma": jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float64),
            }

        def coil_set_spec_from_dofs(self, coil_dofs):
            return coil_dofs

    state = surfaceobjectives_jax_module._build_traceable_objective_state(
        _FakeBooz(),
        _FakeBS(),
        jnp.asarray(0.28, dtype=jnp.float64),
        outer_objective_config={
            "vessel_gamma": jnp.ones((2, 3), dtype=jnp.float64),
        },
    )

    runtime_constants = {
        "objective_kwargs": state["objective_kwargs"],
        "baseline_x": state["baseline_x"],
        "baseline_value": state["baseline_value"],
        "baseline_linear_solve_factors": state["baseline_linear_solve_factors"],
        "baseline_coil_dofs": state["baseline_coil_dofs"],
        "coil_dof_extraction_spec": state["coil_dof_extraction_spec"],
    }

    assert not any(
        isinstance(leaf, jax.Array)
        for leaf in jax.tree_util.tree_leaves(runtime_constants)
    )
    assert isinstance(state["objective_kwargs"]["iota_target"], np.ndarray)
    assert isinstance(state["objective_kwargs"]["label_quadpoints_phi"], np.ndarray)
    assert state["objective_kwargs"]["label_surface_kind"] == "label-geometry-marker"
    assert isinstance(
        state["objective_kwargs"]["outer_objective_config"]["vessel_gamma"],
        np.ndarray,
    )
    assert isinstance(state["baseline_x"], np.ndarray)
    assert isinstance(state["baseline_linear_solve_factors"][0], np.ndarray)
    assert isinstance(state["baseline_coil_dofs"], np.ndarray)


def test_build_traceable_objective_state_exact_carries_no_factors(monkeypatch):
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_canonicalize_traceable_exact_quadrature",
        lambda _booz: (
            jnp.asarray([0.0, 0.25], dtype=jnp.float64),
            jnp.asarray([0.0, 0.5], dtype=jnp.float64),
            jnp.asarray([0, 1], dtype=jnp.int32),
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_evaluate_traceable_total_objective",
        lambda *_args, **_kwargs: jnp.asarray(3.5, dtype=jnp.float64),
    )

    class _FakeSurface:
        quadpoints_phi = np.asarray([0.0, 0.5], dtype=np.float64)
        quadpoints_theta = np.asarray([0.0, 0.5], dtype=np.float64)

    class _FakeBooz:
        boozer_type = "exact"
        surface = _FakeSurface()
        res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "weight_inv_modB": False,
            "dense_linear_solve_factors_available": True,
            "iota": jnp.asarray(0.23, dtype=jnp.float64),
            "G": jnp.asarray(1.7, dtype=jnp.float64),
            "PLU": (
                jnp.eye(2, dtype=jnp.float64),
                jnp.eye(2, dtype=jnp.float64),
                jnp.eye(2, dtype=jnp.float64),
            ),
            "linearization_kind": "exact_jacobian",
        }
        quadpoints_phi = np.asarray([0.0, 0.5], dtype=np.float64)
        quadpoints_theta = np.asarray([0.0, 0.5], dtype=np.float64)
        mpol = 1
        ntor = 1
        nfp = 1
        stellsym = True
        scatter_indices = np.asarray([0, 1], dtype=np.int32)
        _surface_geometry_kind = "surface-geometry-marker"
        label_quadpoints_phi = np.asarray([0.0, 0.25], dtype=np.float64)
        label_quadpoints_theta = np.asarray([0.0, 0.5], dtype=np.float64)
        label_mpol = 1
        label_ntor = 1
        label_nfp = 1
        label_stellsym = True
        label_scatter_indices = np.asarray([0, 1], dtype=np.int32)
        _label_surface_geometry_kind = "label-geometry-marker"
        options = {"weight_inv_modB": False}
        constraint_weight = 1.0
        targetlabel = 0.0
        label_type = "iota"
        phi_idx = 0
        need_to_run_code = False

        def _linear_solve_tolerance(self):
            return 1.0e-10

        def _pack_decision_vector(self, iota, G, *, sdofs):
            return jnp.concatenate(
                [
                    jnp.asarray(sdofs, dtype=jnp.float64),
                    jnp.asarray([iota], dtype=jnp.float64),
                    jnp.asarray([G], dtype=jnp.float64),
                ]
            )

        def get_solved_runtime_state(self):
            return types.SimpleNamespace(
                sdofs=jnp.asarray([1.0, 0.1], dtype=jnp.float64),
                iota=jnp.asarray(0.23, dtype=jnp.float64),
                G=jnp.asarray(1.7, dtype=jnp.float64),
                weight_inv_modB=False,
            )

    class _FakeBS:
        x = np.asarray([0.2, -0.1], dtype=np.float64)

        def coil_dof_extraction_spec(self):
            return {
                "gamma": jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float64),
            }

        def coil_set_spec_from_dofs(self, coil_dofs):
            return coil_dofs

    state = surfaceobjectives_jax_module._build_traceable_objective_state(
        _FakeBooz(),
        _FakeBS(),
        jnp.asarray(0.28, dtype=jnp.float64),
    )

    assert state["linearization_kind"] == "exact_jacobian"
    assert state["baseline_linear_solve_factors"] is None


def test_iotas_jax_value_path_reads_solved_runtime_state(monkeypatch):
    fake_booz = types.SimpleNamespace(
        res={
            "success": True,
            "primal_success": True,
        },
        need_to_run_code=False,
        get_solved_runtime_state=lambda: types.SimpleNamespace(
            sdofs=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
            iota=jnp.asarray(0.37, dtype=jnp.float64),
            G=jnp.asarray(1.2, dtype=jnp.float64),
            weight_inv_modB=True,
        ),
    )

    obj = object.__new__(surfaceobjectives_jax_module.IotasJAX)
    obj.boozer_surface = fake_booz
    obj.biotsavart = None
    obj._J = None
    obj._dJ = None
    obj.compute(compute_gradient=False)

    np.testing.assert_allclose(np.asarray(obj._J), 0.37)


def test_iotas_jax_gradient_path_reads_adjoint_runtime_state(monkeypatch):
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_solve_boozer_adjoint",
        lambda adjoint_state, rhs: (
            np.testing.assert_allclose(np.asarray(rhs), np.asarray([0.0, 1.0])),
            np.testing.assert_equal(adjoint_state.decision_size, 2),
            np.testing.assert_equal(adjoint_state.dtype, jnp.float64),
            jnp.asarray([2.0, -3.0], dtype=jnp.float64),
        )[-1],
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_current_coil_dofs",
        lambda _biotsavart: jnp.asarray([0.0], dtype=jnp.float64),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_adjoint_coil_dofs_gradient",
        lambda stream_group_vjps, adjoint, biotsavart, coil_dofs: (
            np.testing.assert_allclose(np.asarray(adjoint), np.asarray([2.0, -3.0])),
            list(stream_group_vjps(adjoint)),
            np.testing.assert_allclose(np.asarray(coil_dofs), np.asarray([0.0])),
            jnp.asarray([0.0], dtype=jnp.float64),
        )[-1],
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_coil_dofs_gradient_to_derivative",
        lambda _biotsavart, gradient: (
            np.testing.assert_allclose(np.asarray(gradient), np.asarray([0.0])),
            surfaceobjectives_jax_module.Derivative({}),
        )[-1],
    )

    fake_booz = types.SimpleNamespace(
        res={"success": True, "primal_success": True},
        need_to_run_code=False,
        get_solved_runtime_state=lambda: types.SimpleNamespace(
            sdofs=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
            iota=jnp.asarray(0.37, dtype=jnp.float64),
            G=None,
            weight_inv_modB=True,
        ),
        get_adjoint_runtime_state=lambda: types.SimpleNamespace(
            solved_state=None,
            decision_size=2,
            dtype=jnp.float64,
            plu=None,
            solve_forward=lambda rhs: rhs,
            solve_transpose=lambda rhs: rhs,
            stream_group_vjps=lambda _adj: iter([("group-cotangent", (0,))]),
        ),
        biotsavart=object(),
    )

    obj = object.__new__(surfaceobjectives_jax_module.IotasJAX)
    obj.boozer_surface = fake_booz
    obj.biotsavart = fake_booz.biotsavart
    obj._J = None
    obj._dJ = None
    obj.compute(compute_gradient=True)

    np.testing.assert_allclose(np.asarray(obj._J), 0.37)


def test_boozer_residual_native_gradient_stays_flat_until_public_boundary(monkeypatch):
    obj = object.__new__(surfaceobjectives_jax_module.BoozerResidualJAX)
    obj.boozer_surface = types.SimpleNamespace(res={"success": True})
    obj.biotsavart = object()
    obj._J = None
    obj._dJ = None
    obj._dJ_by_dcoil_dofs = None
    obj._direct_objective_value_and_grad = object()
    obj._inner_objective_state = lambda _iota, _G, *, sdofs=None: (
        jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        True,
    )
    obj._compute_dJ_ds = lambda _coil_set_spec, _iota, _G, _weight_inv_modB: (
        jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    )

    solved_state = types.SimpleNamespace(
        sdofs=jnp.asarray([0.4], dtype=jnp.float64),
        iota=jnp.asarray(0.37, dtype=jnp.float64),
        G=jnp.asarray(1.2, dtype=jnp.float64),
        weight_inv_modB=True,
    )
    adjoint_state = types.SimpleNamespace(
        decision_size=2,
        dtype=jnp.float64,
        stream_group_vjps=lambda _adj: iter([("group-cotangent", (0,))]),
    )

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_resolved_boozer_solved_runtime_state",
        lambda _booz_surf: solved_state,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_resolved_boozer_adjoint_runtime_state",
        lambda _booz_surf: adjoint_state,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_current_coil_dofs_and_spec",
        lambda _biotsavart: (
            jnp.asarray([0.5, -0.25], dtype=jnp.float64),
            "coil-spec",
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_value_and_direct_coil_gradient",
        lambda *_args: (
            2.5,
            jnp.asarray([4.0, -1.0], dtype=jnp.float64),
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_solve_boozer_adjoint",
        lambda _adjoint_state, _rhs: jnp.asarray([2.0, -3.0], dtype=jnp.float64),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_adjoint_coil_dofs_gradient",
        lambda stream_group_vjps, adjoint, _biotsavart, coil_dofs: (
            list(stream_group_vjps(adjoint)),
            np.testing.assert_allclose(
                np.asarray(coil_dofs),
                np.asarray([0.5, -0.25]),
            ),
            jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        )[-1],
    )
    _patch_reject_coil_dofs_gradient_to_derivative(monkeypatch)

    gradient = obj.dJ_by_dcoil_dofs()

    np.testing.assert_allclose(np.asarray(gradient), np.asarray([3.0, -3.0]))
    assert obj._J == 2.5
    assert obj._dJ is None


def test_iotas_jax_native_gradient_stays_flat_until_public_boundary(monkeypatch):
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_solve_boozer_adjoint",
        lambda _adjoint_state, rhs: (
            np.testing.assert_allclose(np.asarray(rhs), np.asarray([0.0, 1.0])),
            jnp.asarray([2.0, -3.0], dtype=jnp.float64),
        )[-1],
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_current_coil_dofs",
        lambda _biotsavart: jnp.asarray([0.0], dtype=jnp.float64),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_adjoint_coil_dofs_gradient",
        lambda stream_group_vjps, adjoint, _biotsavart, _coil_dofs: (
            list(stream_group_vjps(adjoint)),
            jnp.asarray([7.0], dtype=jnp.float64),
        )[-1],
    )
    _patch_reject_coil_dofs_gradient_to_derivative(monkeypatch)

    fake_booz = types.SimpleNamespace(
        res={"success": True, "primal_success": True},
        need_to_run_code=False,
        get_solved_runtime_state=lambda: types.SimpleNamespace(
            sdofs=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
            iota=jnp.asarray(0.37, dtype=jnp.float64),
            G=None,
            weight_inv_modB=True,
        ),
        get_adjoint_runtime_state=lambda: types.SimpleNamespace(
            decision_size=2,
            dtype=jnp.float64,
            stream_group_vjps=lambda _adj: iter([("group-cotangent", (0,))]),
        ),
        biotsavart=object(),
    )
    obj = object.__new__(surfaceobjectives_jax_module.IotasJAX)
    obj.boozer_surface = fake_booz
    obj.biotsavart = fake_booz.biotsavart
    obj._J = None
    obj._dJ = None
    obj._dJ_by_dcoil_dofs = None

    gradient = obj.dJ_by_dcoil_dofs()

    np.testing.assert_allclose(np.asarray(gradient), np.asarray([-7.0]))
    np.testing.assert_allclose(np.asarray(obj._J), 0.37)
    assert obj._dJ is None


def test_non_qs_ratio_native_gradient_stays_flat_until_public_boundary(monkeypatch):
    obj = object.__new__(surfaceobjectives_jax_module.NonQuasiSymmetricRatioJAX)
    obj.boozer_surface = types.SimpleNamespace(res={"success": True})
    obj.biotsavart = object()
    obj._J = None
    obj._dJ = None
    obj._dJ_by_dcoil_dofs = None
    obj._compute_value = lambda _sdofs, _coil_set_spec: 1.75
    obj._direct_coil_gradient = lambda _coil_dofs, _sdofs: jnp.asarray(
        [4.0, -1.0], dtype=jnp.float64
    )
    obj._compute_dJ_ds = lambda _coil_set_spec, _sdofs, _decision_size: jnp.asarray(
        [0.0, 1.0], dtype=jnp.float64
    )

    solved_state = types.SimpleNamespace(
        sdofs=jnp.asarray([0.4], dtype=jnp.float64),
        iota=jnp.asarray(0.37, dtype=jnp.float64),
        G=None,
        weight_inv_modB=True,
    )
    adjoint_state = types.SimpleNamespace(
        decision_size=2,
        dtype=jnp.float64,
        stream_group_vjps=lambda _adj: iter([("group-cotangent", (0,))]),
    )

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_resolved_boozer_solved_runtime_state",
        lambda _booz_surf: solved_state,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_resolved_boozer_adjoint_runtime_state",
        lambda _booz_surf: adjoint_state,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_current_coil_dofs_and_spec",
        lambda _biotsavart: (
            jnp.asarray([0.5, -0.25], dtype=jnp.float64),
            "coil-spec",
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_solve_boozer_adjoint",
        lambda _adjoint_state, _rhs: jnp.asarray([2.0, -3.0], dtype=jnp.float64),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_adjoint_coil_dofs_gradient",
        lambda stream_group_vjps, adjoint, _biotsavart, coil_dofs: (
            list(stream_group_vjps(adjoint)),
            np.testing.assert_allclose(
                np.asarray(coil_dofs),
                np.asarray([0.5, -0.25]),
            ),
            jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        )[-1],
    )
    _patch_reject_coil_dofs_gradient_to_derivative(monkeypatch)

    gradient = obj.dJ_by_dcoil_dofs()

    np.testing.assert_allclose(np.asarray(gradient), np.asarray([3.0, -3.0]))
    assert obj._J == 1.75
    assert obj._dJ is None


@pytest.mark.parametrize(
    "wrapper_cls",
    [
        surfaceobjectives_jax_module.BoozerResidualJAX,
        surfaceobjectives_jax_module.IotasJAX,
        surfaceobjectives_jax_module.NonQuasiSymmetricRatioJAX,
    ],
)
def test_public_dJ_projects_cached_native_gradient_without_recomputing(
    monkeypatch,
    wrapper_cls,
):
    obj = object.__new__(wrapper_cls)
    obj.biotsavart = object()
    obj._dJ = None
    obj._dJ_by_dcoil_dofs = jnp.asarray([2.0, -3.0], dtype=jnp.float64)
    projected = surfaceobjectives_jax_module.Derivative({})

    def reject_compute(*_args, **_kwargs):
        raise AssertionError("dJ should project the cached native gradient")

    def project_native_gradient(biotsavart, gradient):
        assert biotsavart is obj.biotsavart
        np.testing.assert_allclose(np.asarray(gradient), np.asarray([2.0, -3.0]))
        return projected

    obj.compute = reject_compute
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_coil_dofs_gradient_to_derivative",
        project_native_gradient,
    )

    assert obj.dJ(partials=True) is projected


def test_iotas_jax_exact_wrapper_gradient_matches_dense_projection_unit(
    monkeypatch,
):
    """Wrapper-only exact-adjoint check; real operator parity lives in Boozer tests."""
    exact_lane = parity_ladder_tolerances("exact-well-conditioned-adjoint")
    A_np = np.asarray(
        [
            [2.0, 0.1, -0.05],
            [0.02, 2.2, 0.04],
            [-0.03, 0.05, 2.4],
        ],
        dtype=np.float64,
    )
    A = jnp.asarray(A_np, dtype=jnp.float64)
    rhs_np = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    projection = np.asarray(
        [
            [1.0, -0.5, 0.25],
            [0.25, 0.75, -1.0],
            [-0.4, 0.1, 0.6],
        ],
        dtype=np.float64,
    )

    def solve_transpose_with_status(rhs):
        np.testing.assert_allclose(np.asarray(rhs), rhs_np)
        return jnp.linalg.solve(A.T, rhs), jnp.asarray(True)

    adjoint_state = types.SimpleNamespace(
        linearization_kind="exact_jacobian",
        decision_size=3,
        dtype=jnp.float64,
        solve_transpose_with_status=solve_transpose_with_status,
        stream_group_vjps=lambda adjoint: iter([("projection-cotangent", adjoint)]),
    )
    dense_adjoint = np.linalg.solve(A_np.T, rhs_np)
    expected_gradient = -(projection @ dense_adjoint)

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_current_coil_dofs",
        lambda _biotsavart: jnp.zeros(3, dtype=jnp.float64),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_adjoint_coil_dofs_gradient",
        lambda stream_group_vjps, adjoint, _biotsavart, _coil_dofs: (
            list(stream_group_vjps(adjoint)),
            jnp.asarray(projection @ np.asarray(adjoint), dtype=jnp.float64),
        )[-1],
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_coil_dofs_gradient_to_derivative",
        lambda _biotsavart, gradient: np.asarray(gradient, dtype=float),
    )
    fake_booz = types.SimpleNamespace(
        res={"success": True, "primal_success": True},
        need_to_run_code=False,
        get_solved_runtime_state=lambda: types.SimpleNamespace(
            sdofs=jnp.asarray([0.0], dtype=jnp.float64),
            iota=jnp.asarray(0.37, dtype=jnp.float64),
            G=jnp.asarray(1.2, dtype=jnp.float64),
            weight_inv_modB=True,
        ),
        get_adjoint_runtime_state=lambda: adjoint_state,
        biotsavart=object(),
    )

    obj = object.__new__(surfaceobjectives_jax_module.IotasJAX)
    obj.boozer_surface = fake_booz
    obj.biotsavart = fake_booz.biotsavart
    obj._J = None
    obj._dJ = None
    obj.compute(compute_gradient=True)

    np.testing.assert_allclose(
        np.asarray(obj._dJ, dtype=float),
        expected_gradient,
        rtol=exact_lane["gradient_rtol"],
        atol=exact_lane["gradient_atol"],
    )
    residual_rel = np.linalg.norm(A_np.T @ dense_adjoint - rhs_np) / (
        1.0 + np.linalg.norm(rhs_np)
    )
    assert residual_rel <= exact_lane["residual_rel_tol"]


def test_boozersurface_get_adjoint_runtime_state_wraps_legacy_cpu_contract():
    captured = {}

    def legacy_vjp(adjoint, passed_booz, iota, G):
        captured["adjoint"] = np.asarray(adjoint)
        captured["booz"] = passed_booz
        captured["iota"] = iota
        captured["G"] = G
        return "legacy-derivative"

    fake_booz = types.SimpleNamespace(
        need_to_run_code=False,
        res={
            "PLU": (
                np.eye(2, dtype=np.float64),
                np.eye(2, dtype=np.float64),
                np.eye(2, dtype=np.float64),
            ),
            "vjp": legacy_vjp,
            "iota": 0.23,
            "G": 1.7,
            "type": "ls",
        },
    )

    adjoint_state = BoozerSurface.get_adjoint_runtime_state(fake_booz)

    assert adjoint_state.linearization_kind == "hessian"
    assert adjoint_state.decision_size == 2
    np.testing.assert_allclose(
        adjoint_state.solve_transpose(np.asarray([1.0, -2.0], dtype=np.float64)),
        np.asarray([1.0, -2.0], dtype=np.float64),
    )
    assert (
        adjoint_state.project_coil_adjoint_derivative(
            np.asarray([3.0, 4.0], dtype=np.float64)
        )
        == "legacy-derivative"
    )
    np.testing.assert_allclose(captured["adjoint"], np.asarray([3.0, 4.0]))
    assert captured["booz"] is fake_booz
    assert captured["iota"] == 0.23
    assert captured["G"] == 1.7


def test_solve_boozer_coil_adjoint_derivative_uses_runtime_projection_hook():
    adjoint_state = types.SimpleNamespace(
        linearization_kind="hessian",
        decision_size=2,
        solve_transpose=lambda rhs: 2.0 * np.asarray(rhs, dtype=np.float64),
        project_coil_adjoint_derivative=lambda adjoint: (
            "projected",
            tuple(np.asarray(adjoint, dtype=np.float64)),
        ),
    )
    fake_booz = types.SimpleNamespace(
        get_adjoint_runtime_state=lambda: adjoint_state,
    )

    derivative = surfaceobjectives_module._solve_boozer_coil_adjoint_derivative(
        fake_booz,
        np.asarray([1.0, -3.0], dtype=np.float64),
    )

    assert derivative == ("projected", (2.0, -6.0))


def test_major_radius_gradient_uses_boozer_surface_biotsavart_fallback():
    captured = {}

    class _FakeBiotSavart:
        def coil_cotangents_to_derivative(self, coil_arrays, coil_group_indices):
            captured["coil_arrays"] = coil_arrays
            captured["coil_group_indices"] = coil_group_indices
            return surfaceobjectives_module.Derivative({})

    fake_surface = types.SimpleNamespace(
        major_radius=lambda: 7.5,
        dmajor_radius_by_dcoeff=lambda: np.asarray([1.0, -2.0], dtype=np.float64),
    )
    adjoint_state = types.SimpleNamespace(
        linearization_kind="hessian",
        decision_size=2,
        solve_transpose=lambda rhs: np.asarray(rhs, dtype=np.float64),
        stream_group_vjps=lambda adjoint: iter(
            [
                (
                    np.asarray(adjoint, dtype=np.float64),
                    (0,),
                )
            ]
        ),
    )
    fake_booz = types.SimpleNamespace(
        need_to_run_code=False,
        surface=fake_surface,
        biotsavart=_FakeBiotSavart(),
        get_adjoint_runtime_state=lambda: adjoint_state,
    )

    obj = object.__new__(surfaceobjectives_module.MajorRadius)
    obj.boozer_surface = fake_booz
    obj.surface = fake_surface
    obj._J = None
    obj._dJ = None
    obj.compute(compute_gradient=True)

    assert obj._J == 7.5
    assert isinstance(obj._dJ, surfaceobjectives_module.Derivative)
    np.testing.assert_allclose(
        np.asarray(captured["coil_arrays"][0], dtype=np.float64),
        np.asarray([1.0, -2.0], dtype=np.float64),
    )
    assert captured["coil_group_indices"] == [(0,)]


def test_get_cached_traceable_runtime_entry_reuses_bundle_for_same_solver_generation(
    monkeypatch,
):
    build_state_calls = []
    build_bundle_calls = []

    booz = types.SimpleNamespace(
        _solver_generation=11,
        _traceable_runtime_entry_cache=None,
        options={},
        _collect_optimizer_options=lambda *, method: {},
    )
    bs = object()

    def build_state(_booz, _bs, iota_target, *, outer_objective_config=None):
        build_state_calls.append((iota_target, outer_objective_config))
        return {
            "objective_kwargs": {
                "iota_target": float(iota_target),
                "outer_objective_config": {
                    "curve_curve_weight": 1.0,
                    "vessel_gamma": np.ones((4, 3), dtype=np.float64),
                }
                if outer_objective_config is not None
                else None,
            },
            "optimize_G": False,
            "predictor_kind": "ls",
            "objective_method": "bfgs-ondevice",
            "coil_dof_extraction_spec": {"spec": "marker"},
            "baseline_x": jnp.arange(4, dtype=jnp.float64),
            "baseline_value": jnp.asarray(1.0, dtype=jnp.float64),
            "baseline_linear_solve_factors": (
                jnp.eye(2, dtype=jnp.float64),
                jnp.eye(2, dtype=jnp.float64),
                jnp.arange(2, dtype=jnp.int32),
            ),
            "baseline_coil_dofs": jnp.arange(3, dtype=jnp.float64),
            "linearization_kind": "hessian",
            "linear_solve_tol": 1.0e-10,
            "linear_solve_stab": 0.0,
        }

    def build_bundle(_booz, state, *, success_filter=None):
        build_bundle_calls.append(
            (state["objective_kwargs"]["iota_target"], success_filter)
        )
        return {
            "state": state,
            "compiled_forward_result_for": object(),
            "compiled_total_gradient_for": object(),
            "compiled_value_and_grad_for": object(),
        }

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_build_traceable_objective_state",
        build_state,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_build_traceable_objective_compiled_bundle_from_state",
        build_bundle,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_objective_from_compiled_bundle",
        lambda compiled_bundle: ("objective", id(compiled_bundle)),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_host_objective",
        lambda objective, **_kwargs: ("host_objective", objective),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_host_value_and_grad",
        lambda compiled_value_and_grad_for, **_kwargs: (
            "host_value_and_grad",
            id(compiled_value_and_grad_for),
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_batched_value_and_grad_pipeline",
        lambda compiled_value_and_grad_for: (
            "batched_value_and_grad",
            id(compiled_value_and_grad_for),
        ),
    )

    entry1 = surfaceobjectives_jax_module._get_cached_traceable_runtime_entry(
        booz,
        bs,
        0.23,
        outer_objective_config={"enabled": True},
        success_filter=None,
    )
    entry2 = surfaceobjectives_jax_module._get_cached_traceable_runtime_entry(
        booz,
        bs,
        0.23,
        outer_objective_config={"enabled": True},
        success_filter=None,
    )

    assert entry1 is entry2
    assert len(build_state_calls) == 2
    assert len(build_bundle_calls) == 1


def test_get_cached_traceable_runtime_entry_reuses_bundle_for_equivalent_success_filter_signatures(
    monkeypatch,
):
    build_bundle_calls = []

    booz = types.SimpleNamespace(
        _solver_generation=11,
        _traceable_runtime_entry_cache=None,
        options={},
        _collect_optimizer_options=lambda *, method: {},
    )
    bs = object()

    def build_state(_booz, _bs, iota_target, *, outer_objective_config=None):
        del outer_objective_config
        return {
            "objective_kwargs": {
                "iota_target": float(iota_target),
                "outer_objective_config": None,
            },
            "optimize_G": False,
            "predictor_kind": "ls",
            "objective_method": "bfgs-ondevice",
            "coil_dof_extraction_spec": {"spec": "marker"},
            "baseline_x": jnp.arange(4, dtype=jnp.float64),
            "baseline_value": jnp.asarray(1.0, dtype=jnp.float64),
            "baseline_linear_solve_factors": (
                jnp.eye(2, dtype=jnp.float64),
                jnp.eye(2, dtype=jnp.float64),
                jnp.arange(2, dtype=jnp.int32),
            ),
            "baseline_coil_dofs": jnp.arange(3, dtype=jnp.float64),
            "linearization_kind": "hessian",
            "linear_solve_tol": 1.0e-10,
            "linear_solve_stab": 0.0,
        }

    def build_bundle(_booz, state, *, success_filter=None):
        build_bundle_calls.append(
            (state["objective_kwargs"]["iota_target"], success_filter)
        )
        return {
            "state": state,
            "compiled_forward_result_for": object(),
            "compiled_total_gradient_for": object(),
            "compiled_value_and_grad_for": object(),
        }

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_build_traceable_objective_state",
        build_state,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_build_traceable_objective_compiled_bundle_from_state",
        build_bundle,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_objective_from_compiled_bundle",
        lambda compiled_bundle: ("objective", id(compiled_bundle)),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_host_objective",
        lambda objective, **_kwargs: ("host_objective", objective),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_host_value_and_grad",
        lambda compiled_value_and_grad_for, **_kwargs: (
            "host_value_and_grad",
            id(compiled_value_and_grad_for),
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_batched_value_and_grad_pipeline",
        lambda compiled_value_and_grad_for: (
            "batched_value_and_grad",
            id(compiled_value_and_grad_for),
        ),
    )

    def success_filter_a(_coil_dofs, _solved_x):
        return jnp.asarray(True, dtype=bool)

    def success_filter_b(_coil_dofs, _solved_x):
        return jnp.asarray(True, dtype=bool)

    signature = ("single-stage-target-lane-hardware-success-filter", "sig-123")
    success_filter_a._traceable_runtime_cache_signature = signature
    success_filter_b._traceable_runtime_cache_signature = signature

    entry1 = surfaceobjectives_jax_module._get_cached_traceable_runtime_entry(
        booz,
        bs,
        0.23,
        outer_objective_config=None,
        success_filter=success_filter_a,
    )
    entry2 = surfaceobjectives_jax_module._get_cached_traceable_runtime_entry(
        booz,
        bs,
        0.23,
        outer_objective_config=None,
        success_filter=success_filter_b,
    )

    assert entry1 is entry2
    assert len(build_bundle_calls) == 1


def test_get_cached_traceable_runtime_entry_invalidates_on_solver_generation_change(
    monkeypatch,
):
    build_bundle_calls = []

    booz = types.SimpleNamespace(
        _solver_generation=3,
        _traceable_runtime_entry_cache=None,
        options={},
        _collect_optimizer_options=lambda *, method: {},
    )
    bs = object()

    def build_state(_booz, _bs, iota_target, *, outer_objective_config=None):
        del outer_objective_config
        return {
            "objective_kwargs": {
                "iota_target": float(iota_target),
                "outer_objective_config": None,
            },
            "optimize_G": False,
            "predictor_kind": "ls",
            "objective_method": "bfgs-ondevice",
            "coil_dof_extraction_spec": {"spec": "marker"},
            "baseline_x": jnp.arange(2, dtype=jnp.float64),
            "baseline_value": jnp.asarray(1.0, dtype=jnp.float64),
            "baseline_linear_solve_factors": (
                jnp.eye(2, dtype=jnp.float64),
                jnp.eye(2, dtype=jnp.float64),
                jnp.arange(2, dtype=jnp.int32),
            ),
            "baseline_coil_dofs": jnp.arange(2, dtype=jnp.float64),
            "linearization_kind": "hessian",
            "linear_solve_tol": 1.0e-10,
            "linear_solve_stab": 0.0,
        }

    def build_bundle(_booz, state, *, success_filter=None):
        del success_filter
        build_bundle_calls.append(state["objective_kwargs"]["iota_target"])
        return {
            "state": state,
            "compiled_forward_result_for": object(),
            "compiled_total_gradient_for": object(),
            "compiled_value_and_grad_for": object(),
        }

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_build_traceable_objective_state",
        build_state,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_build_traceable_objective_compiled_bundle_from_state",
        build_bundle,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_objective_from_compiled_bundle",
        lambda compiled_bundle: ("objective", id(compiled_bundle)),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_host_objective",
        lambda objective, **_kwargs: ("host_objective", objective),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_host_value_and_grad",
        lambda compiled_value_and_grad_for, **_kwargs: (
            "host_value_and_grad",
            id(compiled_value_and_grad_for),
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_batched_value_and_grad_pipeline",
        lambda compiled_value_and_grad_for: (
            "batched_value_and_grad",
            id(compiled_value_and_grad_for),
        ),
    )

    surfaceobjectives_jax_module._get_cached_traceable_runtime_entry(
        booz,
        bs,
        0.23,
    )
    booz._solver_generation += 1
    surfaceobjectives_jax_module._get_cached_traceable_runtime_entry(
        booz,
        bs,
        0.23,
    )

    assert len(build_bundle_calls) == 2


def test_get_cached_traceable_runtime_entry_invalidates_on_target_change(
    monkeypatch,
):
    build_bundle_calls = []

    booz = types.SimpleNamespace(
        _solver_generation=5,
        _traceable_runtime_entry_cache=None,
        options={},
        _collect_optimizer_options=lambda *, method: {},
    )
    bs = object()

    def build_state(_booz, _bs, iota_target, *, outer_objective_config=None):
        del outer_objective_config
        return {
            "objective_kwargs": {
                "iota_target": jnp.asarray(iota_target, dtype=jnp.float64),
                "outer_objective_config": None,
            },
            "optimize_G": False,
            "predictor_kind": "ls",
            "objective_method": "bfgs-ondevice",
            "coil_dof_extraction_spec": {"spec": "marker"},
            "baseline_x": jnp.arange(2, dtype=jnp.float64),
            "baseline_value": jnp.asarray(1.0, dtype=jnp.float64),
            "baseline_linear_solve_factors": (
                jnp.eye(2, dtype=jnp.float64),
                jnp.eye(2, dtype=jnp.float64),
                jnp.arange(2, dtype=jnp.int32),
            ),
            "baseline_coil_dofs": jnp.arange(2, dtype=jnp.float64),
            "linearization_kind": "hessian",
            "linear_solve_tol": 1.0e-10,
            "linear_solve_stab": 0.0,
        }

    def build_bundle(_booz, state, *, success_filter=None):
        del success_filter
        build_bundle_calls.append(float(state["objective_kwargs"]["iota_target"]))
        return {
            "state": state,
            "compiled_forward_result_for": object(),
            "compiled_total_gradient_for": object(),
            "compiled_value_and_grad_for": object(),
        }

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_build_traceable_objective_state",
        build_state,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_build_traceable_objective_compiled_bundle_from_state",
        build_bundle,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_objective_from_compiled_bundle",
        lambda compiled_bundle: ("objective", id(compiled_bundle)),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_host_objective",
        lambda objective, **_kwargs: ("host_objective", objective),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_host_value_and_grad",
        lambda compiled_value_and_grad_for, **_kwargs: (
            "host_value_and_grad",
            id(compiled_value_and_grad_for),
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_batched_value_and_grad_pipeline",
        lambda compiled_value_and_grad_for: (
            "batched_value_and_grad",
            id(compiled_value_and_grad_for),
        ),
    )

    surfaceobjectives_jax_module._get_cached_traceable_runtime_entry(
        booz,
        bs,
        0.23,
    )
    surfaceobjectives_jax_module._get_cached_traceable_runtime_entry(
        booz,
        bs,
        0.28,
    )

    assert len(build_bundle_calls) == 2


def test_make_traceable_objective_runtime_bundle_omits_host_wrappers_by_default(
    monkeypatch,
):
    runtime_entry = {
        "compiled_bundle": {
            "compiled_value_and_grad_for": object(),
            "compiled_forward_result_for": object(),
            "state": {},
        },
        "objective": object(),
        "batched_value_and_grad": object(),
        "public_objective": None,
        "public_value_and_grad": None,
        "public_batched_value_and_grad": None,
        "public_forward_result": None,
        "public_reporting_metrics": None,
        "host_objective": None,
        "host_value_and_grad": None,
        "host_reporting_metrics": None,
        "profile_suite": None,
    }
    ensure_public_calls = []
    ensure_host_calls = []

    def ensure_public(entry):
        ensure_public_calls.append(entry)
        entry["public_objective"] = ("public_objective", entry["objective"])
        entry["public_value_and_grad"] = (
            "public_value_and_grad",
            entry["compiled_bundle"]["compiled_value_and_grad_for"],
        )
        entry["public_batched_value_and_grad"] = (
            "public_batched_value_and_grad",
            entry["batched_value_and_grad"],
        )
        entry["public_forward_result"] = (
            "public_forward_result",
            entry["compiled_bundle"]["compiled_forward_result_for"],
        )
        entry["public_reporting_metrics"] = (
            "public_reporting_metrics",
            entry["compiled_bundle"],
        )

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_get_cached_traceable_runtime_entry",
        lambda *_args, **_kwargs: runtime_entry,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_ensure_traceable_runtime_public_boundaries",
        ensure_public,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_ensure_traceable_runtime_host_wrappers",
        lambda entry, _booz: ensure_host_calls.append(entry),
    )

    bundle = surfaceobjectives_jax_module.make_traceable_objective_runtime_bundle(
        object(),
        object(),
        0.23,
        include_profile_suite=False,
    )

    assert bundle == {
        "objective": ("public_objective", runtime_entry["objective"]),
        "value_and_grad": (
            "public_value_and_grad",
            runtime_entry["compiled_bundle"]["compiled_value_and_grad_for"],
        ),
        "batched_value_and_grad": (
            "public_batched_value_and_grad",
            runtime_entry["batched_value_and_grad"],
        ),
        "forward_result": (
            "public_forward_result",
            runtime_entry["compiled_bundle"]["compiled_forward_result_for"],
        ),
        "reporting_metrics": (
            "public_reporting_metrics",
            runtime_entry["compiled_bundle"],
        ),
    }
    assert ensure_public_calls == [runtime_entry]
    assert ensure_host_calls == []


def test_make_traceable_objective_runtime_bundle_materializes_host_wrappers_on_demand(
    monkeypatch,
):
    runtime_entry = {
        "compiled_bundle": {
            "compiled_value_and_grad_for": object(),
            "compiled_forward_result_for": object(),
            "state": {},
        },
        "objective": object(),
        "batched_value_and_grad": object(),
        "public_objective": None,
        "public_value_and_grad": None,
        "public_batched_value_and_grad": None,
        "public_forward_result": None,
        "public_reporting_metrics": None,
        "host_objective": None,
        "host_value_and_grad": None,
        "host_reporting_metrics": None,
        "profile_suite": None,
    }
    ensure_public_calls = []
    ensure_host_calls = []

    def ensure_public(entry):
        ensure_public_calls.append(entry)
        entry["public_objective"] = ("public_objective", entry["objective"])
        entry["public_value_and_grad"] = (
            "public_value_and_grad",
            entry["compiled_bundle"]["compiled_value_and_grad_for"],
        )
        entry["public_batched_value_and_grad"] = (
            "public_batched_value_and_grad",
            entry["batched_value_and_grad"],
        )
        entry["public_forward_result"] = (
            "public_forward_result",
            entry["compiled_bundle"]["compiled_forward_result_for"],
        )
        entry["public_reporting_metrics"] = (
            "public_reporting_metrics",
            entry["compiled_bundle"],
        )

    def ensure_wrappers(entry, _booz):
        ensure_host_calls.append(entry)
        entry["host_objective"] = ("host_objective", entry["objective"])
        entry["host_value_and_grad"] = (
            "host_value_and_grad",
            entry["compiled_bundle"]["compiled_value_and_grad_for"],
        )
        entry["host_reporting_metrics"] = (
            "host_reporting_metrics",
            entry["public_reporting_metrics"],
        )

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_get_cached_traceable_runtime_entry",
        lambda *_args, **_kwargs: runtime_entry,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_ensure_traceable_runtime_public_boundaries",
        ensure_public,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_ensure_traceable_runtime_host_wrappers",
        ensure_wrappers,
    )

    bundle = surfaceobjectives_jax_module.make_traceable_objective_runtime_bundle(
        object(),
        object(),
        0.23,
        include_profile_suite=False,
        include_host_wrappers=True,
    )

    assert bundle == {
        "objective": ("public_objective", runtime_entry["objective"]),
        "value_and_grad": (
            "public_value_and_grad",
            runtime_entry["compiled_bundle"]["compiled_value_and_grad_for"],
        ),
        "batched_value_and_grad": (
            "public_batched_value_and_grad",
            runtime_entry["batched_value_and_grad"],
        ),
        "forward_result": (
            "public_forward_result",
            runtime_entry["compiled_bundle"]["compiled_forward_result_for"],
        ),
        "reporting_metrics": (
            "public_reporting_metrics",
            runtime_entry["compiled_bundle"],
        ),
        "host_objective": ("host_objective", runtime_entry["objective"]),
        "host_value_and_grad": (
            "host_value_and_grad",
            runtime_entry["compiled_bundle"]["compiled_value_and_grad_for"],
        ),
        "host_reporting_metrics": (
            "host_reporting_metrics",
            ("public_reporting_metrics", runtime_entry["compiled_bundle"]),
        ),
    }
    assert ensure_public_calls == [runtime_entry]
    assert ensure_host_calls == [runtime_entry]


def test_make_traceable_objective_runtime_bundle_reuses_stable_public_boundaries(
    monkeypatch,
):
    runtime_entry = {
        "compiled_bundle": {
            "compiled_value_and_grad_for": object(),
            "compiled_forward_result_for": object(),
            "state": {},
        },
        "objective": object(),
        "batched_value_and_grad": object(),
        "reporting_metrics": None,
        "public_objective": None,
        "public_value_and_grad": None,
        "public_batched_value_and_grad": None,
        "public_forward_result": None,
        "public_reporting_metrics": None,
    }
    expected_public_boundaries = {
        "objective": object(),
        "value_and_grad": object(),
        "batched_value_and_grad": object(),
        "forward_result": object(),
        "reporting_metrics": object(),
    }
    build_counts = {name: 0 for name in expected_public_boundaries}

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_get_cached_traceable_runtime_entry",
        lambda *_args, **_kwargs: runtime_entry,
    )

    def build_boundary(name, boundary):
        def _build(*_args):
            build_counts[name] += 1
            return boundary

        return _build

    for attr_name, boundary_name in (
        ("_make_traceable_objective_boundary", "objective"),
        ("_make_traceable_value_and_grad_boundary", "value_and_grad"),
        (
            "_make_traceable_batched_value_and_grad_boundary",
            "batched_value_and_grad",
        ),
        (
            "_make_traceable_forward_result_boundary",
            "forward_result",
        ),
        (
            "_make_traceable_lazy_reporting_metrics_boundary",
            "reporting_metrics",
        ),
    ):
        monkeypatch.setattr(
            surfaceobjectives_jax_module,
            attr_name,
            build_boundary(boundary_name, expected_public_boundaries[boundary_name]),
        )

    def build_runtime_bundle():
        return surfaceobjectives_jax_module.make_traceable_objective_runtime_bundle(
            object(),
            object(),
            0.23,
            include_profile_suite=False,
        )

    for bundle in (build_runtime_bundle(), build_runtime_bundle()):
        for boundary_name, expected_boundary in expected_public_boundaries.items():
            assert bundle[boundary_name] is expected_boundary

    assert build_counts == {name: 1 for name in expected_public_boundaries}


def test_ensure_traceable_runtime_public_boundaries_defers_reporting_metrics_until_used(
    monkeypatch,
):
    runtime_entry = {
        "compiled_bundle": {
            "compiled_value_and_grad_for": object(),
            "compiled_forward_result_for": object(),
            "state": {},
        },
        "objective": object(),
        "batched_value_and_grad": object(),
        "reporting_metrics": None,
        "public_objective": None,
        "public_value_and_grad": None,
        "public_batched_value_and_grad": None,
        "public_forward_result": None,
        "public_reporting_metrics": None,
    }
    reporting_calls = []

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_objective_boundary",
        lambda objective: ("public_objective", objective),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_value_and_grad_boundary",
        lambda compiled_value_and_grad_for: (
            "public_value_and_grad",
            compiled_value_and_grad_for,
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_batched_value_and_grad_boundary",
        lambda batched_value_and_grad: (
            "public_batched_value_and_grad",
            batched_value_and_grad,
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_forward_result_boundary",
        lambda compiled_forward_result_for: (
            "public_forward_result",
            compiled_forward_result_for,
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_as_jax_float64",
        lambda value: ("as_jax_float64", value),
    )

    def ensure_reporting(entry):
        reporting_calls.append(entry)
        entry["reporting_metrics"] = (
            lambda coil_dofs, *, include_distance_metrics=True: (
                "reporting_metrics",
                coil_dofs,
                include_distance_metrics,
            )
        )
        return entry

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_ensure_traceable_runtime_reporting_metrics",
        ensure_reporting,
    )

    surfaceobjectives_jax_module._ensure_traceable_runtime_public_boundaries(
        runtime_entry
    )

    assert reporting_calls == []
    assert runtime_entry["public_objective"] == (
        "public_objective",
        runtime_entry["objective"],
    )
    assert runtime_entry["public_value_and_grad"] == (
        "public_value_and_grad",
        runtime_entry["compiled_bundle"]["compiled_value_and_grad_for"],
    )
    assert runtime_entry["public_batched_value_and_grad"] == (
        "public_batched_value_and_grad",
        runtime_entry["batched_value_and_grad"],
    )
    assert runtime_entry["public_forward_result"] == (
        "public_forward_result",
        runtime_entry["compiled_bundle"]["compiled_forward_result_for"],
    )

    assert runtime_entry["public_reporting_metrics"](
        "coil_dofs",
        include_distance_metrics=False,
    ) == (
        "reporting_metrics",
        ("as_jax_float64", "coil_dofs"),
        False,
    )
    assert reporting_calls == [runtime_entry]


def test_ensure_traceable_runtime_host_wrappers_defers_reporting_metrics_until_used(
    monkeypatch,
):
    runtime_entry = {
        "compiled_bundle": {
            "compiled_value_and_grad_for": object(),
            "state": {
                "baseline_coil_dofs": np.asarray([0.0], dtype=np.float64),
                "baseline_value": np.asarray(1.0, dtype=np.float64),
                "baseline_x": np.asarray([0.0], dtype=np.float64),
                "baseline_linear_solve_factors": (
                    np.eye(1, dtype=np.float64),
                    np.eye(1, dtype=np.float64),
                    np.asarray([0], dtype=np.int32),
                ),
                "coil_set_spec_from_dofs": lambda coil_dofs: coil_dofs,
                "objective_kwargs": {"outer_objective_config": None},
                "optimize_G": False,
                "linearization_kind": "hessian",
                "linear_solve_tol": 1.0e-10,
                "linear_solve_stab": 0.0,
            },
        },
        "objective": object(),
        "reporting_metrics": None,
        "host_objective": None,
        "host_value_and_grad": None,
        "host_reporting_metrics": None,
    }
    reporting_calls = []

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_host_objective",
        lambda objective, **_kwargs: ("host_objective", objective),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_host_value_and_grad",
        lambda compiled_value_and_grad_for, **_kwargs: (
            "host_value_and_grad",
            compiled_value_and_grad_for,
        ),
    )

    def ensure_reporting(entry):
        reporting_calls.append(entry)
        entry["reporting_metrics"] = (
            lambda coil_dofs, *, include_distance_metrics=True: (
                "reporting_metrics",
                coil_dofs,
                include_distance_metrics,
            )
        )
        return entry

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_ensure_traceable_runtime_reporting_metrics",
        ensure_reporting,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_traceable_host_reporting_metrics",
        lambda reporting_metrics: (
            lambda coil_dofs, *, include_distance_metrics=True: (
                "host_reporting_metrics",
                reporting_metrics(
                    coil_dofs,
                    include_distance_metrics=include_distance_metrics,
                ),
            )
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_total_gradient_with_status",
        lambda *_args, **_kwargs: (
            jnp.asarray([0.25], dtype=jnp.float64),
            jnp.asarray(True),
        ),
    )

    surfaceobjectives_jax_module._ensure_traceable_runtime_host_wrappers(
        runtime_entry,
        object(),
    )

    assert reporting_calls == []
    assert runtime_entry["host_objective"] == (
        "host_objective",
        runtime_entry["objective"],
    )
    assert runtime_entry["host_value_and_grad"] == (
        "host_value_and_grad",
        runtime_entry["compiled_bundle"]["compiled_value_and_grad_for"],
    )

    assert runtime_entry["host_reporting_metrics"](
        "coil_dofs",
        include_distance_metrics=False,
    ) == (
        "host_reporting_metrics",
        ("reporting_metrics", "coil_dofs", False),
    )
    assert reporting_calls == [runtime_entry]


def test_traceable_seeded_initial_value_surfaces_failed_solve_gradient(monkeypatch):
    baseline_coil_dofs = np.asarray([0.5, -0.25], dtype=np.float64)
    failed_gradient = jnp.asarray([0.5, -0.75], dtype=jnp.float64)
    state = {
        "baseline_coil_dofs": baseline_coil_dofs,
        "baseline_value": np.asarray(1.25, dtype=np.float64),
        "baseline_x": np.asarray([0.0, 1.0], dtype=np.float64),
        "baseline_linear_solve_factors": None,
    }
    seeded_compiled_bundle = {
        "compiled_total_gradient_for": lambda *_args: (
            failed_gradient,
            jnp.asarray(False, dtype=bool),
        ),
        "compiled_value_and_grad_for": lambda coil_dofs: (
            jnp.asarray(1.25, dtype=jnp.float64),
            jnp.zeros_like(coil_dofs),
        ),
    }
    runtime_entry = {
        "compiled_bundle": {"state": state},
        "success_filter": None,
    }

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_build_traceable_objective_compiled_bundle_from_state",
        lambda *_args, **_kwargs: seeded_compiled_bundle,
    )

    seeded = (
        surfaceobjectives_jax_module._ensure_traceable_runtime_seeded_value_and_grad(
            runtime_entry,
            object(),
        )
    )

    value, grad = seeded.optimizer_initial_value_and_grad
    _assert_primal_value_with_nonfinite_gradient(value, grad, 1.25)


def test_traceable_seeded_value_and_grad_builds_general_only_bundle(monkeypatch):
    baseline_coil_dofs = np.asarray([0.5, -0.25], dtype=np.float64)
    baseline_gradient = jnp.asarray([0.125, -0.5], dtype=jnp.float64)
    state = {
        "baseline_coil_dofs": baseline_coil_dofs,
        "baseline_value": np.asarray(1.25, dtype=np.float64),
        "baseline_x": np.asarray([0.0, 1.0], dtype=np.float64),
        "baseline_linear_solve_factors": None,
    }
    seeded_compiled_bundle = {
        "compiled_total_gradient_for": lambda *_args: (
            baseline_gradient,
            jnp.asarray(True, dtype=bool),
        ),
        "compiled_value_and_grad_for": lambda coil_dofs: (
            jnp.asarray(1.25, dtype=jnp.float64),
            jnp.zeros_like(coil_dofs),
        ),
    }
    runtime_entry = {
        "compiled_bundle": {"state": state},
        "success_filter": "success-filter",
    }
    build_calls = []

    def build_compiled_bundle(_booz_jax, passed_state, **kwargs):
        build_calls.append((passed_state, kwargs))
        return seeded_compiled_bundle

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_build_traceable_objective_compiled_bundle_from_state",
        build_compiled_bundle,
    )

    seeded = (
        surfaceobjectives_jax_module._ensure_traceable_runtime_seeded_value_and_grad(
            runtime_entry,
            object(),
        )
    )
    cached_seeded = (
        surfaceobjectives_jax_module._ensure_traceable_runtime_seeded_value_and_grad(
            runtime_entry,
            object(),
        )
    )

    assert seeded is cached_seeded
    assert runtime_entry["seeded_compiled_bundle"] is seeded_compiled_bundle
    assert build_calls == [
        (
            state,
            {
                "success_filter": "success-filter",
                "general_only_forward": True,
            },
        )
    ]


def test_traceable_compiled_bundle_general_only_forward_avoids_public_same_coils_path(
    monkeypatch,
):
    state = {
        "objective_kwargs": {},
        "baseline_x": jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        "baseline_value": jnp.asarray(2.0, dtype=jnp.float64),
        "baseline_linear_solve_factors": None,
        "baseline_coil_dofs": jnp.asarray([0.5, -0.25], dtype=jnp.float64),
        "coil_set_spec_from_dofs": lambda coil_dofs: coil_dofs,
        "optimize_G": False,
        "predictor_kind": "none",
        "linearization_kind": "hessian",
        "linear_solve_tol": 1.0e-10,
        "linear_solve_stab": 0.0,
    }
    calls = {"general_forward": 0}

    def fake_general_forward_result(_booz_jax, _coil_set_spec_from_dofs, **kwargs):
        calls["general_forward"] += 1
        coil_dofs = kwargs["coil_dofs"]
        return {
            "value": jnp.sum(coil_dofs),
            "x": jnp.asarray([1.0, -1.0], dtype=jnp.float64),
            "sdofs": jnp.asarray([1.0], dtype=jnp.float64),
            "iota": jnp.asarray(-1.0, dtype=jnp.float64),
            "G": None,
            "linear_solve_factors": None,
            "success": jnp.asarray(True, dtype=bool),
            "primal_success": jnp.asarray(True, dtype=bool),
            "adjoint_linear_solve_available": jnp.asarray(True, dtype=bool),
        }

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_general_forward_result",
        fake_general_forward_result,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_forward_result",
        lambda *_args, **_kwargs: pytest.fail(
            "seeded optimizer bundle must not trace the public same_coils path"
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_total_gradient_with_status",
        lambda _booz_jax, _coil_set_spec_from_dofs, **kwargs: (
            jnp.ones_like(kwargs["coil_dofs"]),
            jnp.asarray(True, dtype=bool),
        ),
    )

    bundle = surfaceobjectives_jax_module._build_traceable_objective_compiled_bundle_from_state(
        object(),
        state,
        general_only_forward=True,
    )
    value, grad = bundle["compiled_value_and_grad_for"](state["baseline_coil_dofs"])

    assert calls["general_forward"] == 1
    np.testing.assert_allclose(np.asarray(value), 0.25)
    np.testing.assert_allclose(np.asarray(grad), np.ones(2, dtype=np.float64))


def test_host_boundary_with_baseline_peel_falls_through_for_traced_inputs():
    baseline = np.asarray([1.0, 2.0], dtype=np.float64)
    wrapped = surfaceobjectives_jax_module._host_boundary_with_baseline_peel(
        lambda coil_dofs: coil_dofs,
        baseline,
        "baseline",
    )

    traced_shape = jax.eval_shape(
        lambda coil_dofs: wrapped(coil_dofs),
        jnp.asarray([1.0, 2.0], dtype=jnp.float64),
    )

    assert traced_shape.shape == (2,)


def test_traceable_runtime_host_wrappers_peel_baseline_without_touching_jitted_boundaries(
    monkeypatch,
):
    baseline_coil_dofs = np.asarray([0.5, -0.25], dtype=np.float64)
    cacheable_public_value_and_grad = (
        optimizer_jax_module._mark_cacheable_jit_value_and_grad(
            lambda coil_dofs: coil_dofs
        )
    )
    runtime_entry = {
        "compiled_bundle": {
            "compiled_value_and_grad_for": lambda _coil_dofs: (_ for _ in ()).throw(
                AssertionError("baseline peel should skip compiled value_and_grad")
            ),
            "state": {
                "baseline_coil_dofs": baseline_coil_dofs,
                "baseline_value": np.asarray(1.25, dtype=np.float64),
                "baseline_x": np.asarray([0.0, 1.0], dtype=np.float64),
                "baseline_linear_solve_factors": (
                    np.eye(2, dtype=np.float64),
                    np.eye(2, dtype=np.float64),
                    np.asarray([0, 1], dtype=np.int32),
                ),
                "coil_set_spec_from_dofs": lambda coil_dofs: coil_dofs,
                "objective_kwargs": {"outer_objective_config": {"enabled": True}},
                "optimize_G": False,
                "linearization_kind": "hessian",
                "linear_solve_tol": 1.0e-10,
                "linear_solve_stab": 0.0,
            },
        },
        "objective": lambda _coil_dofs: (_ for _ in ()).throw(
            AssertionError("baseline peel should skip pure objective")
        ),
        "reporting_metrics": None,
        "public_value_and_grad": cacheable_public_value_and_grad,
        "host_objective": None,
        "host_value_and_grad": None,
        "host_reporting_metrics": None,
    }

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_total_gradient_with_status",
        lambda *_args, **_kwargs: (
            jnp.asarray([0.5, -0.75], dtype=jnp.float64),
            jnp.asarray(True, dtype=bool),
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_reporting_metrics_from_solution",
        lambda *_args, include_distance_metrics, **_kwargs: {
            "solver_success": jnp.asarray(True, dtype=bool),
            "has_G": jnp.asarray(False, dtype=bool),
            "final_G": jnp.asarray(0.0, dtype=jnp.float64),
            "final_non_qs": jnp.asarray(1.0, dtype=jnp.float64),
            "final_boozer_residual": jnp.asarray(2.0, dtype=jnp.float64),
            "final_iota_penalty": jnp.asarray(3.0, dtype=jnp.float64),
            "final_length_penalty": jnp.asarray(4.0, dtype=jnp.float64),
            "final_curve_curve_penalty": jnp.asarray(5.0, dtype=jnp.float64),
            "final_curve_surface_penalty": jnp.asarray(6.0, dtype=jnp.float64),
            "final_surface_vessel_penalty": jnp.asarray(7.0, dtype=jnp.float64),
            "final_curvature_penalty": jnp.asarray(8.0, dtype=jnp.float64),
            "coil_length": jnp.asarray(9.0, dtype=jnp.float64),
            "max_curvature": jnp.asarray(10.0, dtype=jnp.float64),
            "banana_current_A": jnp.asarray(10.25, dtype=jnp.float64),
            "field_error": jnp.asarray(10.5, dtype=jnp.float64),
            "curve_curve_min_dist": jnp.asarray(11.0, dtype=jnp.float64),
            "curve_surface_min_dist": jnp.asarray(12.0, dtype=jnp.float64),
            "surface_vessel_min_dist": jnp.asarray(13.0, dtype=jnp.float64),
            "final_volume": jnp.asarray(14.0, dtype=jnp.float64),
            "final_iota": jnp.asarray(15.0, dtype=jnp.float64),
        },
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_ensure_traceable_runtime_reporting_metrics",
        lambda _entry: (_ for _ in ()).throw(
            AssertionError("baseline reporting peel should stay on the host layer")
        ),
    )

    surfaceobjectives_jax_module._ensure_traceable_runtime_host_wrappers(
        runtime_entry,
        object(),
    )

    assert runtime_entry["host_objective"](baseline_coil_dofs.copy()) == pytest.approx(
        1.25
    )
    value, grad = runtime_entry["host_value_and_grad"](baseline_coil_dofs.tolist())
    assert value == pytest.approx(1.25)
    np.testing.assert_allclose(grad, np.asarray([0.5, -0.75], dtype=np.float64))
    grad[0] = 99.0
    _, second_grad = runtime_entry["host_value_and_grad"](baseline_coil_dofs.copy())
    np.testing.assert_allclose(second_grad, np.asarray([0.5, -0.75], dtype=np.float64))
    assert runtime_entry["host_reporting_metrics"](
        baseline_coil_dofs.copy(),
        include_distance_metrics=False,
    ) == {
        "solver_success": True,
        "final_G": None,
        "final_non_qs": 1.0,
        "final_boozer_residual": 2.0,
        "final_iota_penalty": 3.0,
        "final_length_penalty": 4.0,
        "final_curve_curve_penalty": 5.0,
        "final_curve_surface_penalty": 6.0,
        "final_surface_vessel_penalty": 7.0,
        "final_curvature_penalty": 8.0,
        "coil_length": 9.0,
        "max_curvature": 10.0,
        "banana_current_A": 10.25,
        "field_error": 10.5,
        "curve_curve_min_dist": None,
        "curve_surface_min_dist": None,
        "surface_vessel_min_dist": None,
        "final_volume": 14.0,
        "final_iota": 15.0,
    }
    assert runtime_entry["public_value_and_grad"] is cacheable_public_value_and_grad
    assert (
        getattr(
            runtime_entry["public_value_and_grad"],
            optimizer_jax_module._CACHEABLE_VALUE_AND_GRAD_ATTR,
            False,
        )
        is True
    )


def test_traceable_runtime_host_wrappers_surface_failed_solve_baseline_gradient(
    monkeypatch,
):
    baseline_coil_dofs = np.asarray([0.5, -0.25], dtype=np.float64)
    failed_gradient = np.asarray([0.5, -0.75], dtype=np.float64)
    runtime_entry = {
        "compiled_bundle": {
            "compiled_value_and_grad_for": lambda _coil_dofs: (_ for _ in ()).throw(
                AssertionError("baseline peel should skip compiled value_and_grad")
            ),
            "state": {
                "baseline_coil_dofs": baseline_coil_dofs,
                "baseline_value": np.asarray(1.25, dtype=np.float64),
                "baseline_x": np.asarray([0.0, 1.0], dtype=np.float64),
                "baseline_linear_solve_factors": None,
                "coil_set_spec_from_dofs": lambda coil_dofs: coil_dofs,
                "objective_kwargs": {"outer_objective_config": {"enabled": True}},
                "optimize_G": False,
                "linearization_kind": "exact_jacobian",
                "linear_solve_tol": 1.0e-10,
                "linear_solve_stab": 0.0,
            },
        },
        "objective": lambda _coil_dofs: (_ for _ in ()).throw(
            AssertionError("baseline peel should skip pure objective")
        ),
        "reporting_metrics": None,
        "host_objective": None,
        "host_value_and_grad": None,
        "host_reporting_metrics": None,
    }

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_total_gradient_with_status",
        lambda *_args, **_kwargs: (
            jnp.asarray(failed_gradient, dtype=jnp.float64),
            jnp.asarray(False, dtype=bool),
        ),
    )

    surfaceobjectives_jax_module._ensure_traceable_runtime_host_wrappers(
        runtime_entry,
        object(),
    )

    value, grad = runtime_entry["host_value_and_grad"](baseline_coil_dofs.copy())
    _assert_primal_value_with_nonfinite_gradient(value, grad, 1.25)


def test_traceable_custom_vjp_surfaces_adjoint_solve_failure_as_nan_gradient():
    failed_gradient = jnp.asarray([0.5, -0.75], dtype=jnp.float64)

    def compiled_forward_result_for(coil_dofs):
        return {
            "value": jnp.asarray(1.25, dtype=jnp.float64),
            "x": jnp.asarray([0.0, 1.0], dtype=jnp.float64),
            "linear_solve_factors": None,
            "success": jnp.asarray(True, dtype=bool),
            "primal_success": jnp.asarray(True, dtype=bool),
        }

    compiled_bundle = {
        "compiled_forward_result_for": compiled_forward_result_for,
        "compiled_total_gradient_for": lambda *_args: (
            failed_gradient,
            jnp.asarray(False, dtype=bool),
        ),
    }
    objective = (
        surfaceobjectives_jax_module._make_traceable_objective_from_compiled_bundle(
            compiled_bundle
        )
    )

    grad = jax.grad(objective)(jnp.asarray([0.5, -0.25], dtype=jnp.float64))
    _assert_nonfinite_gradient(grad)


def test_traceable_inner_stationarity_coil_jvp_matches_full_stationarity_jvp(
    monkeypatch,
):
    half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def _strict_quadratic_inner_objective_closure(*, coil_set_spec, **_kwargs):
        def inner_objective(x_inner):
            return half * jnp.dot(x_inner, x_inner) + jnp.dot(coil_set_spec, x_inner)

        return inner_objective

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_boozer_penalty_objective_closure",
        _strict_quadratic_inner_objective_closure,
    )

    x_inner = jnp.asarray([1.5, -0.25], dtype=jnp.float64)
    coil_dofs = jnp.asarray([0.5, -1.25], dtype=jnp.float64)
    coil_dofs_tangent = jnp.asarray([0.75, 2.0], dtype=jnp.float64)

    with jax.transfer_guard("disallow"):
        forcing = surfaceobjectives_jax_module._traceable_inner_stationarity_coil_jvp(
            x_inner,
            coil_dofs,
            coil_dofs_tangent,
            lambda current_coil_dofs: current_coil_dofs,
        )

    np.testing.assert_allclose(
        forcing,
        np.asarray(coil_dofs_tangent, dtype=np.float64),
    )


def test_traceable_objective_gradient_parts_use_strict_vjp_helpers(monkeypatch):
    half = jax.device_put(np.asarray(0.5, dtype=np.float64))
    true_value = jax.device_put(np.asarray(True, dtype=bool))

    def _strict_quadratic_inner_objective_closure(*, coil_set_spec, **_kwargs):
        def inner_objective(x_inner):
            return half * jnp.dot(x_inner, x_inner) + jnp.dot(coil_set_spec, x_inner)

        return inner_objective

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {"kind": "inner"},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_evaluate_traceable_total_objective",
        lambda x_inner, coil_dofs, coil_set_spec, _objective_kwargs: (
            jnp.dot(x_inner, coil_set_spec) + half * jnp.dot(coil_dofs, coil_dofs)
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_boozer_penalty_objective_closure",
        _strict_quadratic_inner_objective_closure,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_solve_linearization",
        lambda _booz_jax, solved_x, rhs, coil_set_spec, objective_kwargs, **_kwargs: (
            rhs,
            true_value,
        ),
    )

    original_vjp = surfaceobjectives_jax_module.jax.vjp
    vjp_calls = {"count": 0}

    def counting_vjp(fun, *primals, **kwargs):
        vjp_calls["count"] += 1
        return original_vjp(fun, *primals, **kwargs)

    monkeypatch.setattr(surfaceobjectives_jax_module.jax, "vjp", counting_vjp)
    monkeypatch.setattr(
        surfaceobjectives_jax_module.jax,
        "grad",
        lambda *_args, **_kwargs: pytest.fail(
            "_traceable_objective_gradient_parts should use strict scalar VJP "
            "helpers instead of jax.grad under transfer guard."
        ),
    )

    coil_dofs = jax.device_put(np.asarray([3.0, 4.0], dtype=np.float64))
    solved_x = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64))

    with jax.transfer_guard("disallow"):
        direct_grad, implicit_grad, total_grad, linear_solve_success = (
            surfaceobjectives_jax_module._traceable_objective_gradient_parts(
                object(),
                lambda coil_dofs: coil_dofs,
                coil_dofs=coil_dofs,
                solved_x=solved_x,
                solved_linear_solve_factors=(object(), object(), object()),
                linearization_kind="hessian",
                linear_solve_tol=1.0e-10,
                linear_solve_stab=0.0,
                objective_kwargs={},
            )
        )

    np.testing.assert_allclose(direct_grad, np.asarray([4.0, 6.0], dtype=np.float64))
    np.testing.assert_allclose(
        implicit_grad,
        np.asarray([3.0, 4.0], dtype=np.float64),
    )
    np.testing.assert_allclose(total_grad, np.asarray([1.0, 2.0], dtype=np.float64))
    assert bool(np.asarray(linear_solve_success))
    assert vjp_calls["count"] == 3


def test_traceable_objective_gradient_parts_skips_direct_vjp_for_iota_term(
    monkeypatch,
):
    true_value = jax.device_put(np.asarray(True, dtype=bool))

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {"kind": "inner"},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_directional_inner_stationarity",
        lambda _solved_x, tangent, current_coil_set_spec, **_kwargs: jnp.dot(
            tangent,
            current_coil_set_spec,
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_evaluate_traceable_weighted_single_stage_outer_term",
        lambda term_name, x_inner, coil_dofs, coil_set_spec, objective_kwargs: (
            surfaceobjectives_jax_module._take_runtime_scalar(x_inner, 0)
            if term_name == "iota"
            else pytest.fail("unexpected term")
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_solve_linearization",
        lambda _booz_jax, solved_x, rhs, coil_set_spec, objective_kwargs, **_kwargs: (
            rhs,
            true_value,
        ),
    )

    original_vjp = surfaceobjectives_jax_module.jax.vjp
    vjp_calls = {"count": 0}

    def counting_vjp(fun, *primals, **kwargs):
        vjp_calls["count"] += 1
        return original_vjp(fun, *primals, **kwargs)

    monkeypatch.setattr(surfaceobjectives_jax_module.jax, "vjp", counting_vjp)

    coil_dofs = jax.device_put(np.asarray([3.0, 4.0], dtype=np.float64))
    solved_x = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64))

    with jax.transfer_guard("disallow"):
        direct_grad, implicit_grad, total_grad, linear_solve_success = (
            surfaceobjectives_jax_module._traceable_objective_gradient_parts(
                object(),
                lambda coil_dofs: coil_dofs,
                coil_dofs=coil_dofs,
                solved_x=solved_x,
                solved_linear_solve_factors=(object(), object(), object()),
                linearization_kind="hessian",
                linear_solve_tol=1.0e-10,
                linear_solve_stab=0.0,
                objective_kwargs={},
                term_name="iota",
            )
        )

    np.testing.assert_allclose(direct_grad, np.zeros(2, dtype=np.float64))
    np.testing.assert_allclose(implicit_grad, np.asarray([1.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(total_grad, np.asarray([-1.0, 0.0], dtype=np.float64))
    assert bool(np.asarray(linear_solve_success))
    assert vjp_calls["count"] == 2


@pytest.mark.parametrize(
    ("term_name", "depends_on_x_inner", "depends_on_coil_dofs"),
    [
        ("non_qs", True, True),
        ("residual", True, True),
        ("iota", True, False),
        ("length", False, True),
        ("curvature", False, True),
        ("curve_curve", False, True),
        ("curve_surface", True, True),
        ("surface_vessel", True, False),
    ],
)
def test_traceable_single_stage_outer_term_dependency_flags(
    term_name,
    depends_on_x_inner,
    depends_on_coil_dofs,
):
    assert (
        surfaceobjectives_jax_module._traceable_single_stage_outer_term_dependency_flags(
            term_name
        )
        == (depends_on_x_inner, depends_on_coil_dofs)
    )


@pytest.mark.parametrize(
    ("term_name", "outer_objective_config", "expected_flags"),
    [
        ("non_qs", {"non_qs_weight": 0.0}, (False, False)),
        ("surface_vessel", {"surface_vessel_weight": 1.0}, (True, False)),
        (None, {"surface_vessel_weight": 1.0}, (True, False)),
        (None, {"length_weight": 1.0}, (False, True)),
        (None, {"non_qs_weight": 1.0}, (True, True)),
        (
            None,
            {
                weight_key: 0.0
                for _, weight_key in (
                    surfaceobjectives_jax_module._TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS
                )
            },
            (False, False),
        ),
    ],
)
def test_traceable_single_stage_effective_dependency_flags_respect_active_weights(
    term_name,
    outer_objective_config,
    expected_flags,
):
    assert (
        surfaceobjectives_jax_module._traceable_single_stage_effective_dependency_flags(
            term_name,
            objective_kwargs={"outer_objective_config": outer_objective_config},
        )
        == expected_flags
    )


def test_traceable_objective_gradient_parts_term_diagnostics_use_strict_vjp_direct_grad(
    monkeypatch,
):
    half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {"kind": "inner"},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_evaluate_traceable_weighted_single_stage_outer_term",
        lambda term_name, x_inner, coil_dofs, coil_set_spec, objective_kwargs: (
            half * jnp.dot(coil_dofs, coil_dofs)
            if term_name == "length"
            else pytest.fail("unexpected term")
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_solve_linearization",
        lambda *_args, **_kwargs: pytest.fail(
            "coil-only term diagnostics should skip the inner linear solve."
        ),
    )

    original_vjp = surfaceobjectives_jax_module.jax.vjp
    vjp_calls = {"count": 0}

    def counting_vjp(fun, *primals, **kwargs):
        vjp_calls["count"] += 1
        return original_vjp(fun, *primals, **kwargs)

    monkeypatch.setattr(surfaceobjectives_jax_module.jax, "vjp", counting_vjp)
    monkeypatch.setattr(
        surfaceobjectives_jax_module.jax,
        "jvp",
        lambda *_args, **_kwargs: pytest.fail(
            "coil-only term diagnostics should use strict scalar VJP instead "
            "of forward-mode coil JVP."
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module.jax,
        "grad",
        lambda *_args, **_kwargs: pytest.fail(
            "coil-only term diagnostics should use strict scalar VJP helpers "
            "instead of jax.grad under transfer guard."
        ),
    )

    coil_dofs = jax.device_put(np.asarray([3.0, 4.0], dtype=np.float64))
    solved_x = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64))

    with jax.transfer_guard("disallow"):
        direct_grad, implicit_grad, total_grad, linear_solve_success = (
            surfaceobjectives_jax_module._traceable_objective_gradient_parts(
                object(),
                lambda coil_dofs: coil_dofs,
                coil_dofs=coil_dofs,
                solved_x=solved_x,
                solved_linear_solve_factors=(object(), object(), object()),
                linearization_kind="hessian",
                linear_solve_tol=1.0e-10,
                linear_solve_stab=0.0,
                objective_kwargs={},
                term_name="length",
            )
        )

    np.testing.assert_allclose(direct_grad, np.asarray([3.0, 4.0], dtype=np.float64))
    np.testing.assert_allclose(implicit_grad, np.zeros(2, dtype=np.float64))
    np.testing.assert_allclose(total_grad, np.asarray([3.0, 4.0], dtype=np.float64))
    assert bool(np.asarray(linear_solve_success))
    assert vjp_calls["count"] == 1


def test_traceable_objective_gradient_parts_skip_all_autodiff_for_zero_weight_term(
    monkeypatch,
):
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_evaluate_traceable_weighted_single_stage_outer_term",
        lambda *_args, **_kwargs: pytest.fail(
            "zero-weight term diagnostics should not evaluate the weighted term."
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_solve_linearization",
        lambda *_args, **_kwargs: pytest.fail(
            "zero-weight term diagnostics should skip the inner linear solve."
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module.jax,
        "jvp",
        lambda *_args, **_kwargs: pytest.fail(
            "zero-weight term diagnostics should skip forward-mode coil JVP."
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module.jax,
        "vjp",
        lambda *_args, **_kwargs: pytest.fail(
            "zero-weight term diagnostics should skip reverse-mode VJP."
        ),
    )

    coil_dofs = jax.device_put(np.asarray([3.0, 4.0], dtype=np.float64))
    solved_x = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64))
    objective_kwargs = {"outer_objective_config": {"non_qs_weight": 0.0}}

    with jax.transfer_guard("disallow"):
        direct_grad, implicit_grad, total_grad, linear_solve_success = (
            surfaceobjectives_jax_module._traceable_objective_gradient_parts(
                object(),
                lambda current_coil_dofs: current_coil_dofs,
                coil_dofs=coil_dofs,
                solved_x=solved_x,
                solved_linear_solve_factors=(object(), object(), object()),
                linearization_kind="hessian",
                linear_solve_tol=1.0e-10,
                linear_solve_stab=0.0,
                objective_kwargs=objective_kwargs,
                term_name="non_qs",
            )
        )

    np.testing.assert_allclose(direct_grad, np.zeros(2, dtype=np.float64))
    np.testing.assert_allclose(implicit_grad, np.zeros(2, dtype=np.float64))
    np.testing.assert_allclose(total_grad, np.zeros(2, dtype=np.float64))
    assert bool(np.asarray(linear_solve_success))


def test_traceable_total_gradient_skips_direct_vjp_when_active_weights_are_inner_only(
    monkeypatch,
):
    true_value = jax.device_put(np.asarray(True, dtype=bool))

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {"kind": "inner"},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_directional_inner_stationarity",
        lambda _solved_x, tangent, current_coil_set_spec, **_kwargs: jnp.dot(
            tangent,
            current_coil_set_spec,
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_evaluate_traceable_total_objective",
        lambda x_inner, coil_dofs, coil_set_spec, objective_kwargs: (
            surfaceobjectives_jax_module._take_runtime_scalar(x_inner, 0)
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_solve_linearization",
        lambda _booz_jax, solved_x, rhs, coil_set_spec, objective_kwargs, **_kwargs: (
            rhs,
            true_value,
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module.jax,
        "jvp",
        lambda *_args, **_kwargs: pytest.fail(
            "inner-only active weights should skip direct coil JVP on the total objective."
        ),
    )

    original_vjp = surfaceobjectives_jax_module.jax.vjp
    vjp_calls = {"count": 0}

    def counting_vjp(fun, *primals, **kwargs):
        vjp_calls["count"] += 1
        return original_vjp(fun, *primals, **kwargs)

    monkeypatch.setattr(surfaceobjectives_jax_module.jax, "vjp", counting_vjp)

    coil_dofs = jax.device_put(np.asarray([3.0, 4.0], dtype=np.float64))
    solved_x = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64))
    objective_kwargs = {
        "outer_objective_config": {
            "surface_vessel_weight": 1.0,
            "non_qs_weight": 0.0,
            "residual_weight": 0.0,
            "iota_weight": 0.0,
            "length_weight": 0.0,
            "curvature_weight": 0.0,
            "curve_curve_weight": 0.0,
            "curve_surface_weight": 0.0,
        }
    }

    with jax.transfer_guard("disallow"):
        direct_grad, implicit_grad, total_grad, linear_solve_success = (
            surfaceobjectives_jax_module._traceable_objective_gradient_parts(
                object(),
                lambda current_coil_dofs: current_coil_dofs,
                coil_dofs=coil_dofs,
                solved_x=solved_x,
                solved_linear_solve_factors=(object(), object(), object()),
                linearization_kind="hessian",
                linear_solve_tol=1.0e-10,
                linear_solve_stab=0.0,
                objective_kwargs=objective_kwargs,
            )
        )

    np.testing.assert_allclose(direct_grad, np.zeros(2, dtype=np.float64))
    np.testing.assert_allclose(implicit_grad, np.asarray([1.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(total_grad, np.asarray([-1.0, 0.0], dtype=np.float64))
    assert bool(np.asarray(linear_solve_success))
    assert vjp_calls["count"] == 2


def test_traceable_objective_gradient_parts_skips_direct_jvp_for_surface_vessel_term(
    monkeypatch,
):
    true_value = jax.device_put(np.asarray(True, dtype=bool))

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {"kind": "inner"},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_directional_inner_stationarity",
        lambda _solved_x, tangent, current_coil_set_spec, **_kwargs: jnp.dot(
            tangent,
            current_coil_set_spec,
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_evaluate_traceable_weighted_single_stage_outer_term",
        lambda term_name, x_inner, coil_dofs, coil_set_spec, objective_kwargs: (
            surfaceobjectives_jax_module._take_runtime_scalar(x_inner, 0)
            if term_name == "surface_vessel"
            else pytest.fail("unexpected term")
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_solve_linearization",
        lambda _booz_jax, solved_x, rhs, coil_set_spec, objective_kwargs, **_kwargs: (
            rhs,
            true_value,
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module.jax,
        "jvp",
        lambda *_args, **_kwargs: pytest.fail(
            "surface_vessel diagnostics should skip coil JVP for inner-only terms."
        ),
    )

    original_vjp = surfaceobjectives_jax_module.jax.vjp
    vjp_calls = {"count": 0}

    def counting_vjp(fun, *primals, **kwargs):
        vjp_calls["count"] += 1
        return original_vjp(fun, *primals, **kwargs)

    monkeypatch.setattr(surfaceobjectives_jax_module.jax, "vjp", counting_vjp)

    coil_dofs = jax.device_put(np.asarray([3.0, 4.0], dtype=np.float64))
    solved_x = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64))

    with jax.transfer_guard("disallow"):
        direct_grad, implicit_grad, total_grad, linear_solve_success = (
            surfaceobjectives_jax_module._traceable_objective_gradient_parts(
                object(),
                lambda coil_dofs: coil_dofs,
                coil_dofs=coil_dofs,
                solved_x=solved_x,
                solved_linear_solve_factors=(object(), object(), object()),
                linearization_kind="hessian",
                linear_solve_tol=1.0e-10,
                linear_solve_stab=0.0,
                objective_kwargs={},
                term_name="surface_vessel",
            )
        )

    np.testing.assert_allclose(direct_grad, np.zeros(2, dtype=np.float64))
    np.testing.assert_allclose(implicit_grad, np.asarray([1.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(total_grad, np.asarray([-1.0, 0.0], dtype=np.float64))
    assert bool(np.asarray(linear_solve_success))
    assert vjp_calls["count"] == 2


def test_diagnose_traceable_objective_runtime_redevices_cached_baseline_arrays(
    monkeypatch,
):
    objective_config = {
        weight_key: 1.0
        for _, weight_key in (
            surfaceobjectives_jax_module._TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS
        )
    }
    call_checks: dict[str, bool] = {}

    def _record_array(name, value):
        call_checks[name] = isinstance(value, jax.Array)
        return value

    def fake_total_gradient_with_status(
        _booz_jax,
        _coil_set_spec_from_dofs,
        *,
        coil_dofs,
        solved_x,
        solved_linear_solve_factors,
        linearization_kind,
        linear_solve_tol,
        linear_solve_stab,
        objective_kwargs,
    ):
        del linearization_kind, linear_solve_tol, linear_solve_stab
        _record_array("total_gradient_coil_dofs", coil_dofs)
        _record_array("total_gradient_solved_x", solved_x)
        plu_leaves = jax.tree_util.tree_leaves(solved_linear_solve_factors)
        call_checks["total_gradient_solved_linear_solve_factors"] = all(
            isinstance(leaf, jax.Array) for leaf in plu_leaves
        )
        assert objective_kwargs["outer_objective_config"] is objective_config
        return (
            jnp.asarray([0.5, -0.75], dtype=jnp.float64),
            jnp.asarray(True, dtype=bool),
        )

    def fake_term_values(solved_x, coil_dofs, _coil_set_spec, **_objective_kwargs):
        _record_array("raw_terms_solved_x", solved_x)
        _record_array("raw_terms_coil_dofs", coil_dofs)
        return {
            term_name: jnp.asarray(float(index + 1), dtype=jnp.float64)
            for index, (term_name, _weight_key) in enumerate(
                surfaceobjectives_jax_module._TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS
            )
        }

    def fake_weighted_term_values(raw_terms, *, outer_objective_config):
        assert outer_objective_config is objective_config
        return dict(raw_terms)

    def fake_gradient_parts(
        _booz_jax,
        _coil_set_spec_from_dofs,
        *,
        coil_dofs,
        solved_x,
        solved_linear_solve_factors,
        linearization_kind,
        linear_solve_tol,
        linear_solve_stab,
        objective_kwargs,
        term_name=None,
    ):
        del linearization_kind, linear_solve_tol, linear_solve_stab
        _record_array("gradient_parts_coil_dofs", coil_dofs)
        _record_array("gradient_parts_solved_x", solved_x)
        plu_leaves = jax.tree_util.tree_leaves(solved_linear_solve_factors)
        call_checks["gradient_parts_solved_linear_solve_factors"] = all(
            isinstance(leaf, jax.Array) for leaf in plu_leaves
        )
        assert objective_kwargs["outer_objective_config"] is objective_config
        assert term_name is not None
        grad = jnp.asarray([0.5, -0.75], dtype=jnp.float64)
        return grad, grad, grad, jnp.asarray(True, dtype=bool)

    runtime_entry = {
        "compiled_bundle": {
            "state": {
                "objective_kwargs": {"outer_objective_config": objective_config},
                "optimize_G": False,
                "baseline_x": np.asarray([1.0, 2.0], dtype=np.float64),
                "baseline_value": np.asarray(1.25, dtype=np.float64),
                "baseline_linear_solve_factors": (
                    np.eye(2, dtype=np.float64),
                    np.asarray([0, 1], dtype=np.int32),
                    np.asarray([0, 1], dtype=np.int32),
                ),
                "baseline_coil_dofs": np.asarray([3.0, 4.0], dtype=np.float64),
                "coil_set_spec_from_dofs": lambda coil_dofs: ("coil-set", coil_dofs),
                "linearization_kind": "hessian",
                "linear_solve_tol": 1.0e-10,
                "linear_solve_stab": 0.0,
            },
        }
    }

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_get_cached_traceable_runtime_entry",
        lambda *_args, **_kwargs: runtime_entry,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_single_stage_outer_term_values",
        fake_term_values,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_total_objective_kwargs",
        lambda objective_kwargs: {
            "outer_objective_config": objective_kwargs["outer_objective_config"]
        },
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_weighted_single_stage_outer_term_values",
        fake_weighted_term_values,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_total_gradient_with_status",
        fake_total_gradient_with_status,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_objective_gradient_parts",
        fake_gradient_parts,
    )

    report = surfaceobjectives_jax_module.diagnose_traceable_objective_runtime(
        object(),
        object(),
        0.23,
    )

    assert report["all_finite"] is True
    assert report["baseline_success"] is True
    assert report["first_nonfinite_term"] is None
    assert call_checks == {
        "total_gradient_coil_dofs": True,
        "total_gradient_solved_x": True,
        "total_gradient_solved_linear_solve_factors": True,
        "raw_terms_solved_x": True,
        "raw_terms_coil_dofs": True,
        "gradient_parts_coil_dofs": True,
        "gradient_parts_solved_x": True,
        "gradient_parts_solved_linear_solve_factors": True,
    }


def test_traceable_batched_value_and_grad_pipeline_matches_scalar_calls():
    compiled_value_and_grad_for = jax.jit(
        lambda coil_dofs: (
            jnp.sum(coil_dofs**2),
            2.0 * coil_dofs,
        )
    )
    batched_value_and_grad = (
        surfaceobjectives_jax_module._make_traceable_batched_value_and_grad_pipeline(
            compiled_value_and_grad_for
        )
    )
    coil_dofs_batch = jnp.asarray(
        [[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]],
        dtype=jnp.float64,
    )

    batched_values, batched_grads = batched_value_and_grad(coil_dofs_batch)
    reference_values, reference_grads = jax.vmap(compiled_value_and_grad_for)(
        coil_dofs_batch
    )

    np.testing.assert_allclose(
        np.asarray(batched_values),
        np.asarray(reference_values),
    )
    np.testing.assert_allclose(
        np.asarray(batched_grads),
        np.asarray(reference_grads),
    )


def _make_torus_dofs(R=1.0, r=0.1, mpol=1, ntor=1, nfp=1, stellsym=False):
    ncols = 2 * ntor + 1
    xc = np.zeros((2 * mpol + 1, ncols))
    yc = np.zeros((2 * mpol + 1, ncols))
    zc = np.zeros((2 * mpol + 1, ncols))
    xc[0, 0] = R
    xc[1, 0] = r
    zc[mpol + 1, 0] = r
    full = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])

    if stellsym:
        scatter_idx = stellsym_scatter_indices(mpol, ntor)
        return full[scatter_idx], scatter_idx
    return full.copy(), None


def _surface_slice_from_dofs(surface_dofs, stellsym, scatter_idx):
    gamma = surface_gamma_from_dofs(
        surface_dofs,
        _QP_PHI,
        _QP_THETA,
        _MPOL,
        _NTOR,
        _NFP,
        stellsym,
        scatter_idx,
    )
    gammadash2 = surface_gammadash2_from_dofs(
        surface_dofs,
        _QP_PHI,
        _QP_THETA,
        _MPOL,
        _NTOR,
        _NFP,
        stellsym,
        scatter_idx,
    )
    return gamma[0], gammadash2[0]


def _make_surface_dofs(stellsym):
    surface_dofs_np, scatter_idx = _make_torus_dofs(
        R=1.0,
        r=0.1,
        mpol=_MPOL,
        ntor=_NTOR,
        nfp=_NFP,
        stellsym=stellsym,
    )
    return jnp.array(surface_dofs_np), scatter_idx


def _make_tf_coils_from_dofs(
    dofs,
    *,
    n_coils=6,
    nquad=48,
):
    twopi = 2 * np.pi
    t = jnp.linspace(0.0, 1.0, nquad, endpoint=False)
    angle = twopi * t

    R_center = 1.0 + 0.04 * dofs[0]
    r_coil = 0.28 + 0.02 * dofs[1]
    phase_offsets = (
        twopi * (jnp.arange(n_coils) / n_coils) + 0.12 * dofs[2 : 2 + n_coils]
    )
    currents = 1e5 * (1.0 + 0.05 * dofs[2 + n_coils : 2 + 2 * n_coils])

    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    coil_R = R_center + r_coil * cos_angle
    dcoil_R = -r_coil * twopi * sin_angle
    coil_z = r_coil * sin_angle
    dcoil_z = r_coil * twopi * cos_angle

    cos_phi = jnp.cos(phase_offsets)[:, None]
    sin_phi = jnp.sin(phase_offsets)[:, None]

    gammas = jnp.stack(
        [
            coil_R[None, :] * cos_phi,
            coil_R[None, :] * sin_phi,
            jnp.broadcast_to(coil_z, (n_coils, nquad)),
        ],
        axis=-1,
    )
    gammadashs = jnp.stack(
        [
            dcoil_R[None, :] * cos_phi,
            dcoil_R[None, :] * sin_phi,
            jnp.broadcast_to(dcoil_z, (n_coils, nquad)),
        ],
        axis=-1,
    )
    return gammas, gammadashs, currents


def _make_object_level_toroidal_flux_case():
    ncoils = 2
    nfp = 1
    stellsym = False

    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=stellsym,
        R0=1.0,
        R1=0.5,
        order=3,
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0, 19, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 21, endpoint=False),
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.2)
    surface.set_zs(1, 0, 0.2)
    return coils, surface


def _make_reference_object_toroidal_flux_pair():
    coils, surface = _make_object_level_toroidal_flux_case()
    return ToroidalFlux(surface, BiotSavart(coils)), ToroidalFlux(
        surface, BiotSavartJAX(coils)
    )


def _make_ncsx_biotsavart_pair():
    _, _, _, _, bs = get_data("ncsx")
    return BiotSavart(bs.coils), BiotSavartJAX(bs.coils)


def _make_toroidal_flux_pair(surfacetype, stellsym, *, idx=0):
    surface = get_surface(surfacetype, stellsym)
    bs_cpu, bs_jax = _make_ncsx_biotsavart_pair()
    return (
        ToroidalFlux(surface, bs_cpu, idx=idx),
        ToroidalFlux(surface, bs_jax, idx=idx),
        bs_cpu,
        bs_jax,
    )


def _make_aspect_ratio_surface(surfacetype, stellsym):
    return get_surface(
        surfacetype,
        stellsym,
        mpol=2,
        ntor=1,
        nfp=2,
        nphi=7,
        ntheta=8,
        full=True,
    )


def _make_surface_objective_pair(surfacetype, stellsym, cpu_cls, jax_cls, **kwargs):
    surface = _make_aspect_ratio_surface(surfacetype, stellsym)
    return cpu_cls(surface, **kwargs), jax_cls(surface, **kwargs)


def _make_aspect_ratio_pair(surfacetype, stellsym, **kwargs):
    return _make_surface_objective_pair(
        surfacetype,
        stellsym,
        surfaceobjectives_module.AspectRatio,
        surfaceobjectives_jax_module.AspectRatioJAX,
        **kwargs,
    )


def _make_area_pair(surfacetype, stellsym, **kwargs):
    return _make_surface_objective_pair(
        surfacetype,
        stellsym,
        surfaceobjectives_module.Area,
        surfaceobjectives_jax_module.AreaJAX,
        **kwargs,
    )


def _make_volume_pair(surfacetype, stellsym, **kwargs):
    return _make_surface_objective_pair(
        surfacetype,
        stellsym,
        surfaceobjectives_module.Volume,
        surfaceobjectives_jax_module.VolumeJAX,
        **kwargs,
    )


def _make_principal_curvature_pair(surfacetype, stellsym):
    surface = _make_aspect_ratio_surface(surfacetype, stellsym)
    return (
        surfaceobjectives_module.PrincipalCurvature(
            surface,
            **_PRINCIPAL_CURVATURE_KWARGS,
        ),
        surfaceobjectives_jax_module.PrincipalCurvatureJAX(
            surface,
            **_PRINCIPAL_CURVATURE_KWARGS,
        ),
        surface,
    )


def _make_qfm_biotsavart_pair():
    base_curves = create_equally_spaced_curves(
        2,
        1,
        stellsym=False,
        R0=1.0,
        R1=0.5,
        order=3,
    )
    base_currents = [Current(1e5) for _ in range(2)]
    coils = coils_via_symmetries(base_curves, base_currents, 1, False)
    return BiotSavart(coils), BiotSavartJAX(coils)


def _make_qfm_pair(surfacetype, stellsym):
    surface = get_surface(
        surfacetype,
        stellsym,
        mpol=1,
        ntor=1,
        nfp=1,
        nphi=7,
        ntheta=8,
        full=True,
    )
    bs_cpu, bs_jax = _make_qfm_biotsavart_pair()
    return (
        surfaceobjectives_module.QfmResidual(surface, bs_cpu),
        surfaceobjectives_jax_module.QfmResidualJAX(surface, bs_jax),
        surface,
    )


def _surface_gradient_value(tf, _):
    return tf.dJ_by_dsurfacecoefficients()


def _surface_hessian_value(tf, _):
    return tf.d2J_by_dsurfacecoefficientsdsurfacecoefficients()


def _coil_gradient_value(tf, bs):
    return tf.dJ_by_dcoils()(bs)


def _toroidal_flux_value(tf, _):
    return tf.J()


def _assert_toroidal_flux_value_parity(actual, reference):
    np.testing.assert_allclose(
        host_scalar(actual),
        reference,
        rtol=_TOROIDAL_FLUX_VALUE_RTOL,
        atol=_TOROIDAL_FLUX_VALUE_ATOL,
    )


def _assert_toroidal_flux_array_parity(actual, reference, *, rtol, atol):
    np.testing.assert_allclose(
        host_array(actual, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        rtol=rtol,
        atol=atol,
    )


def _assert_toroidal_flux_pair_parity(
    surfacetype,
    stellsym,
    *,
    value_getter,
    rtol,
    atol,
):
    tf_cpu, tf_jax, bs_cpu, bs_jax = _make_toroidal_flux_pair(surfacetype, stellsym)
    _assert_toroidal_flux_array_parity(
        value_getter(tf_jax, bs_jax),
        value_getter(tf_cpu, bs_cpu),
        rtol=rtol,
        atol=atol,
    )


def _taylor_test_first_order(
    f, grad_fn, x, *, epsilons=None, direction=None, atol=1e-9
):
    rng = parity_rng(3)
    if direction is None:
        direction = jnp.array(rng.rand(*x.shape) - 0.5)
    if epsilons is None:
        epsilons = np.power(2.0, -np.arange(10, 20, dtype=float))

    df0 = float(jnp.dot(grad_fn(x), direction))
    err_old = 1e9
    for eps in epsilons:
        f_plus = float(f(x + eps * direction))
        f_minus = float(f(x - eps * direction))
        fd_est = (f_plus - f_minus) / (2 * eps)
        err = abs(fd_est - df0)
        assert err < max(atol, 0.35 * err_old), (
            f"Taylor convergence stalled: err={err:.2e}, "
            f"prev={err_old:.2e}, ratio={err / err_old:.3f}"
        )
        err_old = err


def _taylor_test_second_order(f, grad_fn, hess_fn, x, *, epsilons=None):
    rng = parity_rng(5)
    direction1 = jnp.array(rng.rand(*x.shape) - 0.5)
    direction2 = jnp.array(rng.rand(*x.shape) - 0.5)
    if epsilons is None:
        epsilons = np.power(2.0, -np.arange(7, 20, dtype=float))

    df0 = float(jnp.dot(grad_fn(x), direction1))
    hess = hess_fn(x)
    d2f0 = float(direction2 @ (hess @ direction1))

    err_old = 1e9
    for eps in epsilons:
        df_eps = float(jnp.dot(grad_fn(x + eps * direction2), direction1))
        err = abs((df_eps - df0) / eps - d2f0)
        assert err <= 0.56 * err_old, (
            f"Second-order Taylor convergence stalled: err={err:.2e}, "
            f"prev={err_old:.2e}, ratio={err / err_old:.3f}"
        )
        err_old = err


class TestToroidalFluxJAXTaylor:
    @pytest.mark.parametrize("stellsym", [False, True])
    def test_toroidal_flux_surface_hessian_taylor(self, stellsym):
        """Pure-JAX ToroidalFlux Hessian gate for surface DOFs.

        Oracle: type 4 — second-order Taylor remainder
        ``|J(x + eps*d) - J(x) - eps*grad·d - 0.5*eps^2*d·H·d| = O(eps^3)``
        on a closed-form pure-JAX flux integrand; independent of any
        code-path comparison.
        Lane: smoke.
        """
        surface_dofs, scatter_idx = _make_surface_dofs(stellsym)
        coil_gammas, coil_gammadashs, coil_currents = _make_tf_coils_from_dofs(
            _TF_COIL_DOFS
        )

        def flux(surface_dofs_inner):
            points, gammadash2 = _surface_slice_from_dofs(
                surface_dofs_inner,
                stellsym,
                scatter_idx,
            )
            A = biot_savart_A(points, coil_gammas, coil_gammadashs, coil_currents)
            return toroidal_flux_jax(A, gammadash2, _NTHETA)

        _taylor_test_second_order(
            flux,
            jax.grad(flux),
            jax.hessian(flux),
            surface_dofs,
        )

    @pytest.mark.parametrize("stellsym", [False, True])
    def test_toroidal_flux_coil_dofs_taylor(self, stellsym):
        """Pure-JAX ToroidalFlux gradient gate for a traceable TF coil family.

        Oracle: type 4 — first-order Taylor remainder
        ``|J(x + eps*d) - J(x) - eps*grad·d| = O(eps^2)`` on the same
        pure-JAX flux integrand, varied over TF coil DOFs.
        Lane: smoke.
        """
        surface_dofs, scatter_idx = _make_surface_dofs(stellsym)
        points, gammadash2 = _surface_slice_from_dofs(
            surface_dofs, stellsym, scatter_idx
        )

        def flux(coil_dofs_inner):
            coil_gammas, coil_gammadashs, coil_currents = _make_tf_coils_from_dofs(
                coil_dofs_inner
            )
            A = biot_savart_A(points, coil_gammas, coil_gammadashs, coil_currents)
            return toroidal_flux_jax(A, gammadash2, _NTHETA)

        _taylor_test_first_order(
            flux,
            jax.grad(flux),
            _TF_COIL_DOFS,
        )


class TestAspectRatioJAXObjectParity:
    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_aspect_ratio_value_parity_matrix(self, surfacetype, stellsym):
        """Match CPU ``AspectRatio`` value against the JAX wrapper.

        Oracle: type 1 — CPU ``AspectRatio.J()`` calls
        ``Surface.aspect_ratio()`` (Python composition over C++
        ``Surface::volume`` in ``src/simsoptpp/surface.cpp`` plus a Python
        ``Surface.mean_cross_sectional_area`` quadrature).
        Lane: parity, rtol=1e-12, atol=1e-12.
        """
        aspect_cpu, aspect_jax = _make_aspect_ratio_pair(surfacetype, stellsym)

        np.testing.assert_allclose(
            aspect_jax.J(),
            aspect_cpu.J(),
            rtol=_ASPECT_RATIO_VALUE_RTOL,
            atol=_ASPECT_RATIO_VALUE_ATOL,
        )

    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_aspect_ratio_first_derivative_parity_matrix(
        self,
        surfacetype,
        stellsym,
    ):
        """Match CPU ``AspectRatio`` first derivatives against the JAX wrapper.

        Oracle: type 1 — CPU ``AspectRatio.dJ_by_dsurfacecoefficients()`` calls
        ``Surface.daspect_ratio_by_dcoeff()`` (Python chain rule over C++
        ``Surface::dvolume_by_dcoeff`` and Python
        ``Surface.dmean_cross_sectional_area_by_dcoeff``).
        Lane: parity, rtol=1e-10, atol=1e-11.
        """
        aspect_cpu, aspect_jax = _make_aspect_ratio_pair(surfacetype, stellsym)

        np.testing.assert_allclose(
            aspect_jax.dJ_by_dsurfacecoefficients(),
            aspect_cpu.dJ_by_dsurfacecoefficients(),
            rtol=_ASPECT_RATIO_GRAD_RTOL,
            atol=_ASPECT_RATIO_GRAD_ATOL,
        )
        np.testing.assert_allclose(
            aspect_jax.dJ(partials=True)(aspect_jax.surface),
            aspect_cpu.dJ(partials=True)(aspect_cpu.surface),
            rtol=_ASPECT_RATIO_GRAD_RTOL,
            atol=_ASPECT_RATIO_GRAD_ATOL,
        )

    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_aspect_ratio_second_derivative_parity_matrix(
        self,
        surfacetype,
        stellsym,
    ):
        """Match CPU ``AspectRatio`` Hessian against the JAX wrapper.

        Oracle: type 1 — CPU
        ``AspectRatio.d2J_by_dsurfacecoefficientsdsurfacecoefficients()``
        calls ``Surface.d2aspect_ratio_by_dcoeff_dcoeff()`` (Python
        chain rule over C++ ``Surface::d2volume_by_dcoeffdcoeff`` and
        Python ``Surface.d2mean_cross_sectional_area_by_dcoeff_dcoeff``).
        Lane: parity, rtol=1e-9, atol=1e-10.
        """
        aspect_cpu, aspect_jax = _make_aspect_ratio_pair(surfacetype, stellsym)

        np.testing.assert_allclose(
            aspect_jax.d2J_by_dsurfacecoefficientsdsurfacecoefficients(),
            aspect_cpu.d2J_by_dsurfacecoefficientsdsurfacecoefficients(),
            rtol=_ASPECT_RATIO_HESS_RTOL,
            atol=_ASPECT_RATIO_HESS_ATOL,
        )

    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_aspect_ratio_regridded_wrapper_parity(self, surfacetype, stellsym):
        """Match CPU/JAX ``AspectRatio`` under a regridded surface view.

        Oracle: type 1 — same CPU ``AspectRatio.J()`` and
        ``AspectRatio.dJ_by_dsurfacecoefficients()`` oracle as above,
        evaluated on a copied surface with ``nphi=9, ntheta=10`` to
        exercise the wrapper's ``surface.copy(...)`` regrid path.
        Lane: parity, rtol=1e-12 (value) / 1e-10 (gradient).
        """
        aspect_cpu, aspect_jax = _make_aspect_ratio_pair(
            surfacetype,
            stellsym,
            nphi=9,
            ntheta=10,
        )

        np.testing.assert_allclose(
            aspect_jax.J(),
            aspect_cpu.J(),
            rtol=_ASPECT_RATIO_VALUE_RTOL,
            atol=_ASPECT_RATIO_VALUE_ATOL,
        )
        np.testing.assert_allclose(
            aspect_jax.dJ_by_dsurfacecoefficients(),
            aspect_cpu.dJ_by_dsurfacecoefficients(),
            rtol=_ASPECT_RATIO_GRAD_RTOL,
            atol=_ASPECT_RATIO_GRAD_ATOL,
        )

    def test_aspect_ratio_regridded_tensor_clamped_dims_remains_unsupported(self):
        """Pin the unsupported-contract: clamped ``SurfaceXYZTensorFourier`` raises.

        Lane: routing — guards that ``AspectRatioJAX.J()`` raises
        ``NotImplementedError`` rather than silently returning NaN for
        ``SurfaceXYZTensorFourier(clamped_dims=...)`` (see the
        ``surface_mean_cross_sectional_area_jax_from_dofs`` guard in
        ``src/simsopt/geo/surfaceobjectives_jax.py``); no parity assertion.
        """
        from simsopt.geo import SurfaceXYZTensorFourier

        surface = SurfaceXYZTensorFourier(
            mpol=1,
            ntor=1,
            nfp=1,
            stellsym=False,
            clamped_dims=[True, False, False],
        )
        aspect_jax = surfaceobjectives_jax_module.AspectRatioJAX(
            surface,
            nphi=9,
            ntheta=10,
        )

        assert aspect_jax.surface.clamped_dims == [True, False, False]
        with pytest.raises(NotImplementedError, match="clamped_dims"):
            aspect_jax.J()

    def test_aspect_ratio_lazy_package_export(self):
        """Verify lazy ``simsopt.geo.__getattr__`` returns the canonical object.

        Lane: routing — guards the lazy export table in
        ``simsopt/geo/__init__.py`` against typos that would silently shadow
        ``AspectRatioJAX``; no parity assertion.
        """
        from simsopt.geo import AspectRatioJAX

        assert AspectRatioJAX is surfaceobjectives_jax_module.AspectRatioJAX

    def test_aspect_ratio_first_order_taylor(self):
        """First-order Taylor remainder convergence for aspect-ratio gradient.

        Oracle: type 4 — finite-difference Taylor remainder
        ``|J(x + eps*d) - J(x) - eps*grad·d| = O(eps^2)`` over geometric eps
        sweep; independent of any code-path comparison.
        Lane: smoke, atol=1e-11.
        """
        surface = _make_aspect_ratio_surface("SurfaceRZFourier", stellsym=False)
        spec = surface.surface_spec()
        dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)

        def aspect_ratio(inner_dofs):
            return surfaceobjectives_jax_module.surface_aspect_ratio_jax_from_dofs(
                spec,
                inner_dofs,
            )

        _taylor_test_first_order(
            aspect_ratio,
            jax.grad(aspect_ratio),
            dofs,
            epsilons=np.power(2.0, -np.arange(8, 16, dtype=float)),
            atol=1e-11,
        )

    def test_aspect_ratio_second_order_taylor(self):
        """Second-order Taylor remainder convergence for aspect-ratio Hessian.

        Oracle: type 4 — finite-difference Taylor remainder
        ``|J(x + eps*d) - J(x) - eps*grad·d - 0.5*eps^2*d·H·d| = O(eps^3)``;
        independent of any code-path comparison.
        Lane: smoke.
        """
        surface = _make_aspect_ratio_surface("SurfaceRZFourier", stellsym=False)
        spec = surface.surface_spec()
        dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)

        def aspect_ratio(inner_dofs):
            return surfaceobjectives_jax_module.surface_aspect_ratio_jax_from_dofs(
                spec,
                inner_dofs,
            )

        _taylor_test_second_order(
            aspect_ratio,
            jax.grad(aspect_ratio),
            jax.hessian(aspect_ratio),
            dofs,
            epsilons=np.power(2.0, -np.arange(11, 19, dtype=float)),
        )


class TestAreaVolumeJAXObjectParity:
    @pytest.mark.parametrize("pair_factory", (_make_area_pair, _make_volume_pair))
    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_area_volume_value_parity_matrix(
        self,
        pair_factory,
        surfacetype,
        stellsym,
    ):
        """Match CPU ``Area``/``Volume`` value against the JAX wrapper.

        Oracle: type 1 — CPU ``simsopt.geo.surfaceobjectives.Area`` /
        ``Volume`` route through C++ ``Surface::area()`` /
        ``Surface::volume()`` in ``src/simsoptpp/surface.cpp``.
        Lane: parity, rtol=1e-12, atol=1e-12.
        """
        cpu_obj, jax_obj = pair_factory(surfacetype, stellsym)

        np.testing.assert_allclose(
            jax_obj.J(),
            cpu_obj.J(),
            rtol=_ASPECT_RATIO_VALUE_RTOL,
            atol=_ASPECT_RATIO_VALUE_ATOL,
        )

    @pytest.mark.parametrize("pair_factory", (_make_area_pair, _make_volume_pair))
    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_area_volume_first_derivative_parity_matrix(
        self,
        pair_factory,
        surfacetype,
        stellsym,
    ):
        """Match CPU ``Area``/``Volume`` first derivatives against the JAX wrapper.

        Oracle: type 1 — CPU ``Area.dJ_by_dsurfacecoefficients()`` /
        ``Volume.dJ_by_dsurfacecoefficients()`` route through C++
        ``Surface::darea_by_dcoeff()`` / ``Surface::dvolume_by_dcoeff()``
        in ``src/simsoptpp/surface.cpp``.
        Lane: parity, rtol=1e-10, atol=1e-11.
        """
        cpu_obj, jax_obj = pair_factory(surfacetype, stellsym)

        np.testing.assert_allclose(
            jax_obj.dJ_by_dsurfacecoefficients(),
            cpu_obj.dJ_by_dsurfacecoefficients(),
            rtol=_ASPECT_RATIO_GRAD_RTOL,
            atol=_ASPECT_RATIO_GRAD_ATOL,
        )
        np.testing.assert_allclose(
            jax_obj.dJ(partials=True)(jax_obj.surface),
            cpu_obj.dJ(partials=True)(cpu_obj.surface),
            rtol=_ASPECT_RATIO_GRAD_RTOL,
            atol=_ASPECT_RATIO_GRAD_ATOL,
        )

    @pytest.mark.parametrize("pair_factory", (_make_area_pair, _make_volume_pair))
    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_area_volume_second_derivative_parity_matrix(
        self,
        pair_factory,
        surfacetype,
        stellsym,
    ):
        """Match CPU ``Area``/``Volume`` Hessian against the JAX wrapper.

        Oracle: type 1 — CPU ``Area.d2J_by_dsurfacecoefficientsdsurfacecoefficients()``
        / ``Volume.d2J_by_dsurfacecoefficientsdsurfacecoefficients()`` route through
        C++ ``Surface::d2area_by_dcoeffdcoeff()`` /
        ``Surface::d2volume_by_dcoeffdcoeff()`` in ``src/simsoptpp/surface.cpp``.
        Lane: parity, rtol=1e-9, atol=1e-10.
        """
        cpu_obj, jax_obj = pair_factory(surfacetype, stellsym)

        np.testing.assert_allclose(
            jax_obj.d2J_by_dsurfacecoefficientsdsurfacecoefficients(),
            cpu_obj.d2J_by_dsurfacecoefficientsdsurfacecoefficients(),
            rtol=_ASPECT_RATIO_HESS_RTOL,
            atol=_ASPECT_RATIO_HESS_ATOL,
        )

    @pytest.mark.parametrize("pair_factory", (_make_area_pair, _make_volume_pair))
    def test_area_volume_regridded_wrapper_parity(self, pair_factory):
        """Match CPU/JAX ``Area``/``Volume`` under a regridded surface view.

        Oracle: type 1 — same C++ ``Surface::area()`` / ``Surface::volume()``
        and their gradient counterparts, evaluated on a copied surface with
        ``nphi=9, ntheta=10``.
        Lane: parity, rtol=1e-12 (value) / 1e-10 (gradient).
        """
        cpu_obj, jax_obj = pair_factory(
            "SurfaceRZFourier",
            stellsym=False,
            nphi=9,
            ntheta=10,
        )

        np.testing.assert_allclose(
            jax_obj.J(),
            cpu_obj.J(),
            rtol=_ASPECT_RATIO_VALUE_RTOL,
            atol=_ASPECT_RATIO_VALUE_ATOL,
        )
        np.testing.assert_allclose(
            jax_obj.dJ_by_dsurfacecoefficients(),
            cpu_obj.dJ_by_dsurfacecoefficients(),
            rtol=_ASPECT_RATIO_GRAD_RTOL,
            atol=_ASPECT_RATIO_GRAD_ATOL,
        )

    @pytest.mark.parametrize(
        "export_name",
        (
            "AreaJAX",
            "VolumeJAX",
            "surface_area_jax_from_dofs",
            "surface_volume_jax_from_dofs",
        ),
    )
    def test_area_volume_lazy_package_exports(self, export_name):
        """Verify lazy ``simsopt.geo.__getattr__`` returns the canonical module object.

        Lane: routing — guards ``simsopt/geo/__init__.py``'s lazy export
        table against typos in the registration of the four new
        ``AreaJAX`` / ``VolumeJAX`` / ``surface_area_jax_from_dofs`` /
        ``surface_volume_jax_from_dofs`` exports; no parity assertion.
        """
        import simsopt.geo as geo_module

        assert getattr(geo_module, export_name) is getattr(
            surfaceobjectives_jax_module,
            export_name,
        )


class TestSurfaceScalarMetricJAXHelpers:
    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize(
        (
            "metric_name",
            "jax_grad_name",
            "jax_hess_name",
            "cpu_grad_name",
            "cpu_hess_name",
        ),
        _SCALAR_METRIC_CASES,
    )
    def test_surface_scalar_metric_gradient_and_hessian_parity(
        self,
        surfacetype,
        metric_name,
        jax_grad_name,
        jax_hess_name,
        cpu_grad_name,
        cpu_hess_name,
    ):
        """Match CPU surface-scalar gradient/Hessian methods against JAX helpers.

        Oracle: type 1 — CPU
        ``Surface.{dmean_cross_sectional_area,dminor_radius,dmajor_radius,daspect_ratio}_by_dcoeff()``
        and their ``d2..._by_dcoeff_dcoeff()`` counterparts are Python chain
        rules over C++ ``Surface::dvolume_by_dcoeff`` /
        ``Surface::d2volume_by_dcoeffdcoeff`` plus Python
        ``Surface.dmean_cross_sectional_area_by_dcoeff`` /
        ``Surface.d2mean_cross_sectional_area_by_dcoeff_dcoeff`` primitives.
        Lane: parity, rtol=1e-10 (grad) / 1e-9 (hess), atol=1e-11 / 1e-10.
        """
        surface = _make_aspect_ratio_surface(surfacetype, stellsym=False)
        spec = surface.surface_spec()
        dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
        jax_grad_fn = getattr(surfaceobjectives_jax_module, jax_grad_name)
        jax_hess_fn = getattr(surfaceobjectives_jax_module, jax_hess_name)
        cpu_grad_fn = getattr(surface, cpu_grad_name)
        cpu_hess_fn = getattr(surface, cpu_hess_name)

        np.testing.assert_allclose(
            jax_grad_fn(spec, dofs),
            cpu_grad_fn(),
            rtol=_ASPECT_RATIO_GRAD_RTOL,
            atol=_ASPECT_RATIO_GRAD_ATOL,
            err_msg=metric_name,
        )
        np.testing.assert_allclose(
            jax_hess_fn(spec, dofs),
            cpu_hess_fn(),
            rtol=_ASPECT_RATIO_HESS_RTOL,
            atol=_ASPECT_RATIO_HESS_ATOL,
            err_msg=metric_name,
        )

    @pytest.mark.parametrize(
        "export_name",
        (
            "surface_dmean_cross_sectional_area_jax_from_dofs",
            "surface_dminor_radius_jax_from_dofs",
            "surface_dmajor_radius_jax_from_dofs",
            "surface_daspect_ratio_jax_from_dofs",
            "surface_d2mean_cross_sectional_area_jax_from_dofs",
            "surface_d2minor_radius_jax_from_dofs",
            "surface_d2major_radius_jax_from_dofs",
            "surface_d2aspect_ratio_jax_from_dofs",
        ),
    )
    def test_surface_scalar_metric_helper_lazy_exports(self, export_name):
        """Verify lazy ``simsopt.geo.__getattr__`` for scalar-metric helper exports.

        Lane: routing — guards the lazy export table in
        ``simsopt/geo/__init__.py`` against typos for the eight
        ``surface_d*_jax_from_dofs`` and ``surface_d2*_jax_from_dofs``
        helper names; no parity assertion.
        """
        import simsopt.geo as geo_module

        assert getattr(geo_module, export_name) is getattr(
            surfaceobjectives_jax_module,
            export_name,
        )


class TestPrincipalCurvatureJAXObjectParity:
    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_surface_curvature_helper_value_parity_matrix(self, surfacetype, stellsym):
        """Match CPU ``Surface.surface_curvatures()`` against the JAX helper.

        Oracle: type 1 — direct C++ kernel ``Surface::surface_curvatures_impl``
        in ``src/simsoptpp/surface.cpp``.
        Lane: parity, rtol=1e-12, atol=1e-10.
        """
        surface = _make_aspect_ratio_surface(surfacetype, stellsym)
        spec = surface.surface_spec()
        dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)

        np.testing.assert_allclose(
            surfaceobjectives_jax_module.surface_curvatures_jax_from_dofs(
                spec,
                dofs,
            ),
            surface.surface_curvatures(),
            rtol=_PRINCIPAL_CURVATURE_VALUE_RTOL,
            atol=_PRINCIPAL_CURVATURE_VALUE_ATOL,
        )

    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_surface_curvature_helper_gradient_parity_matrix(
        self,
        surfacetype,
        stellsym,
    ):
        """Match CPU ``Surface.dsurface_curvatures_by_dcoeff()`` against the JAX helper.

        Oracle: type 1 — direct C++ kernel
        ``Surface::dsurface_curvatures_by_dcoeff_impl`` in
        ``src/simsoptpp/surface.cpp``.
        Lane: parity, rtol=1e-10, atol=1e-9.
        """
        surface = _make_aspect_ratio_surface(surfacetype, stellsym)
        spec = surface.surface_spec()
        dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)

        np.testing.assert_allclose(
            surfaceobjectives_jax_module.surface_dsurface_curvatures_jax_from_dofs(
                spec,
                dofs,
            ),
            surface.dsurface_curvatures_by_dcoeff(),
            rtol=_PRINCIPAL_CURVATURE_GRAD_RTOL,
            atol=_PRINCIPAL_CURVATURE_GRAD_ATOL,
        )

    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_principal_curvature_value_parity_matrix(self, surfacetype, stellsym):
        """Match CPU ``PrincipalCurvature.J()`` against the JAX wrapper.

        Oracle: type 1 — CPU ``PrincipalCurvature.J()`` is a Python penalty
        composed over the direct C++ kernels ``Surface::surface_curvatures``
        and ``Surface::normal``.
        Lane: parity, rtol=1e-12, atol=1e-10.
        """
        pc_cpu, pc_jax, _surface = _make_principal_curvature_pair(
            surfacetype,
            stellsym,
        )

        np.testing.assert_allclose(
            pc_jax.J(),
            pc_cpu.J(),
            rtol=_PRINCIPAL_CURVATURE_VALUE_RTOL,
            atol=_PRINCIPAL_CURVATURE_VALUE_ATOL,
        )

    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_principal_curvature_gradient_parity_matrix(self, surfacetype, stellsym):
        """Match CPU ``PrincipalCurvature.dJ`` against the JAX wrapper gradient.

        Oracle: type 1 — CPU ``PrincipalCurvature.dJ(partials=True)`` composes
        the gradient over the direct C++ kernels
        ``Surface::dsurface_curvatures_by_dcoeff`` and
        ``Surface::dnormal_by_dcoeff``.
        Lane: parity, rtol=1e-10, atol=1e-9.
        """
        pc_cpu, pc_jax, surface = _make_principal_curvature_pair(
            surfacetype,
            stellsym,
        )

        np.testing.assert_allclose(
            pc_jax.dJ_by_dsurfacecoefficients(),
            pc_cpu.dJ(partials=True)(surface),
            rtol=_PRINCIPAL_CURVATURE_GRAD_RTOL,
            atol=_PRINCIPAL_CURVATURE_GRAD_ATOL,
        )

    def test_principal_curvature_lazy_package_export(self):
        """Verify lazy ``simsopt.geo.__getattr__`` returns the canonical object.

        Lane: routing — guards the lazy export table in
        ``simsopt/geo/__init__.py`` against typos that would silently shadow
        ``PrincipalCurvatureJAX``; no parity assertion.
        """
        from simsopt.geo import PrincipalCurvatureJAX

        assert (
            PrincipalCurvatureJAX is surfaceobjectives_jax_module.PrincipalCurvatureJAX
        )

    @pytest.mark.parametrize(
        "export_name",
        (
            "surface_curvatures_jax_from_dofs",
            "surface_dsurface_curvatures_jax_from_dofs",
        ),
    )
    def test_surface_curvature_helper_lazy_exports(self, export_name):
        """Verify lazy ``simsopt.geo.__getattr__`` for curvature-helper exports.

        Lane: routing — guards the lazy export table in
        ``simsopt/geo/__init__.py`` against typos for the
        ``surface_curvatures_jax_from_dofs`` /
        ``surface_dsurface_curvatures_jax_from_dofs`` helper names; no
        parity assertion.
        """
        import simsopt.geo as geo_module

        assert getattr(geo_module, export_name) is getattr(
            surfaceobjectives_jax_module,
            export_name,
        )

    def test_principal_curvature_first_order_taylor(self):
        """First-order Taylor remainder convergence for principal-curvature gradient.

        Oracle: type 4 — finite-difference Taylor remainder
        ``|J(x + eps*d) - J(x) - eps*grad·d| = O(eps^2)``; independent of any
        code-path comparison.
        Lane: smoke, atol=1e-9.
        """
        surface = _make_aspect_ratio_surface("SurfaceRZFourier", stellsym=False)
        spec = surface.surface_spec()
        dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)

        def principal_curvature(inner_dofs):
            return (
                surfaceobjectives_jax_module.surface_principal_curvature_jax_from_dofs(
                    spec,
                    inner_dofs,
                    **_PRINCIPAL_CURVATURE_KWARGS,
                )
            )

        _taylor_test_first_order(
            principal_curvature,
            jax.grad(principal_curvature),
            dofs,
            epsilons=np.power(2.0, -np.arange(8, 16, dtype=float)),
            atol=1e-9,
        )


class TestQfmResidualJAXObjectParity:
    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_qfm_residual_value_parity_matrix(self, surfacetype, stellsym):
        """Match CPU ``QfmResidual.J()`` against the JAX wrapper.

        Oracle: type 1 — CPU ``QfmResidual.J()`` evaluates the ratio
        ``sum((B·n̂)² * |N|) / sum(|B|² * |N|)`` (i.e.
        ``∫(B·n̂)² dS / ∫B² dS``) over surface quadpoints using C++
        ``BiotSavart::B`` and C++ ``Surface::normal`` / ``Surface::gamma``.
        Lane: parity, rtol=1e-12, atol=1e-12.
        """
        qfm_cpu, qfm_jax, _surface = _make_qfm_pair(surfacetype, stellsym)

        np.testing.assert_allclose(
            qfm_jax.J(),
            qfm_cpu.J(),
            rtol=_QFM_VALUE_RTOL,
            atol=_QFM_VALUE_ATOL,
        )

    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_qfm_residual_gradient_parity_matrix(self, surfacetype, stellsym):
        """Match CPU ``QfmResidual.dJ_by_dsurfacecoefficients`` vs JAX wrapper.

        Oracle: type 1 — CPU
        ``QfmResidual.dJ_by_dsurfacecoefficients()`` composes the gradient
        from the same C++ ``BiotSavart`` and ``Surface`` derivative primitives.
        Lane: parity, rtol=1e-10, atol=1e-10.
        """
        qfm_cpu, qfm_jax, _surface = _make_qfm_pair(surfacetype, stellsym)

        np.testing.assert_allclose(
            qfm_jax.dJ_by_dsurfacecoefficients(),
            qfm_cpu.dJ_by_dsurfacecoefficients(),
            rtol=_QFM_GRAD_RTOL,
            atol=_QFM_GRAD_ATOL,
        )

    def test_qfm_residual_lazy_package_export(self):
        """Verify lazy ``simsopt.geo.__getattr__`` returns the canonical object.

        Lane: routing — guards the lazy export table in
        ``simsopt/geo/__init__.py`` against typos that would silently shadow
        ``QfmResidualJAX``; no parity assertion.
        """
        from simsopt.geo import QfmResidualJAX

        assert QfmResidualJAX is surfaceobjectives_jax_module.QfmResidualJAX

    def test_qfm_residual_first_order_taylor(self):
        """First-order Taylor remainder convergence for QFM residual gradient.

        Oracle: type 4 — finite-difference Taylor remainder
        ``|J(x + eps*d) - J(x) - eps*grad·d| = O(eps^2)``; independent of any
        code-path comparison.
        Lane: smoke, atol=1e-10.
        """
        _qfm_cpu, qfm_jax, surface = _make_qfm_pair(
            "SurfaceRZFourier",
            stellsym=False,
        )
        spec = surface.surface_spec()
        dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
        coil_set_spec = qfm_jax.biotsavart.coil_set_spec_from_dofs(
            jnp.asarray(qfm_jax.biotsavart.x, dtype=jnp.float64)
        )

        def qfm_residual(inner_dofs):
            return surfaceobjectives_jax_module.surface_qfm_residual_jax_from_dofs(
                spec,
                inner_dofs,
                coil_set_spec,
            )

        _taylor_test_first_order(
            qfm_residual,
            jax.grad(qfm_residual),
            dofs,
            epsilons=np.power(2.0, -np.arange(8, 16, dtype=float)),
            atol=1e-10,
        )

    def test_qfm_residual_surface_update_cache_parity(self):
        """Match CPU/JAX after a surface DOF perturbation + cache invalidation.

        Oracle: type 1 — same CPU ``QfmResidual.J`` /
        ``dJ_by_dsurfacecoefficients`` oracle as above, evaluated after
        ``surface.set_dofs(...)`` and ``invalidate_cache()`` on both wrappers,
        with a witness check that the value actually moved by more than 1e-14.
        Lane: parity, rtol=1e-12 (value) / 1e-10 (gradient).
        """
        qfm_cpu, qfm_jax, surface = _make_qfm_pair(
            "SurfaceRZFourier",
            stellsym=False,
        )
        initial_value = qfm_jax.J()
        dofs = surface.get_dofs()
        direction = np.zeros_like(dofs)
        direction[3] = 0.02
        direction[-5] = -0.01
        surface.set_dofs(dofs + direction)

        qfm_cpu.invalidate_cache()
        qfm_jax.invalidate_cache()

        np.testing.assert_allclose(
            qfm_jax.J(),
            qfm_cpu.J(),
            rtol=_QFM_VALUE_RTOL,
            atol=_QFM_VALUE_ATOL,
        )
        np.testing.assert_allclose(
            qfm_jax.dJ_by_dsurfacecoefficients(),
            qfm_cpu.dJ_by_dsurfacecoefficients(),
            rtol=_QFM_GRAD_RTOL,
            atol=_QFM_GRAD_ATOL,
        )
        assert abs(qfm_jax.J() - initial_value) > 1e-14


def test_major_radius_jax_value_and_native_adjoint_gradient():
    """Match CPU ``MajorRadius`` value and adjoint gradient via fake BoozerSurface.

    Oracle: type 1 — CPU ``MajorRadius.J()`` calls
    ``Surface.major_radius()`` (Python composition over C++
    ``Surface::volume`` and Python ``Surface.mean_cross_sectional_area``);
    the ``dJ(partials=True).data`` projection is the adjoint contract that
    the JAX wrapper's ``dJ_by_dcoil_dofs`` must satisfy.
    Lane: parity, rtol=1e-12 (value) / 1e-10 (gradient).
    """
    surface = _make_aspect_ratio_surface("SurfaceRZFourier", stellsym=False)
    sdofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    expected_surface_gradient = (
        surfaceobjectives_jax_module._surface_dmajor_radius_jax_from_dofs(
            surface.surface_spec(),
            sdofs,
        )
    )
    captured = {}

    class _FakeDofOptimizable:
        local_full_dof_size = 2
        local_dofs_free_status = np.asarray([True, True])

    dep_opt = _FakeDofOptimizable()
    lineage_opt = types.SimpleNamespace(
        local_dof_size=2,
        local_full_dof_size=2,
        local_dofs_free_status=np.asarray([True, True]),
        dofs=types.SimpleNamespace(dep_opts=lambda: (dep_opt,)),
    )

    class _FakeBiotSavart:
        x = np.asarray([1.0, -2.0], dtype=np.float64)
        unique_dof_lineage = (lineage_opt,)

        def coil_cotangents_to_dofs_gradient(
            self,
            coil_arrays,
            coil_group_indices,
            *,
            coil_dofs,
        ):
            captured["coil_arrays"] = coil_arrays
            captured["coil_group_indices"] = coil_group_indices
            captured["coil_dofs"] = coil_dofs
            total = jnp.sum(coil_arrays[0])
            return jnp.asarray([total, -total], dtype=jnp.float64)

    solved_state = types.SimpleNamespace(
        sdofs=sdofs,
        iota=jnp.asarray(0.3, dtype=jnp.float64),
        G=jnp.asarray(1.0, dtype=jnp.float64),
        weight_inv_modB=False,
    )
    expected_rhs = jnp.concatenate(
        (
            expected_surface_gradient,
            jnp.zeros(2, dtype=jnp.float64),
        )
    )
    expected_adjoint = 2.0 * expected_rhs
    adjoint_state = types.SimpleNamespace(
        decision_size=expected_rhs.size,
        dtype=jnp.float64,
        linearization_kind="hessian",
        solve_transpose_with_status=lambda rhs: (
            2.0 * rhs,
            jnp.asarray(True),
        ),
        project_coil_adjoint_derivative=lambda _adjoint: (
            surfaceobjectives_module.Derivative(
                {dep_opt: np.asarray([expected_total, -expected_total])}
            )
        ),
        stream_group_vjps=lambda adjoint: iter(((adjoint, (0,)),)),
    )
    fake_booz = types.SimpleNamespace(
        need_to_run_code=False,
        res={"primal_success": True, "success": True},
        surface=surface,
        biotsavart=_FakeBiotSavart(),
        get_solved_runtime_state=lambda: solved_state,
        get_adjoint_runtime_state=lambda: adjoint_state,
    )
    obj = object.__new__(surfaceobjectives_jax_module.MajorRadiusJAX)
    obj.boozer_surface = fake_booz
    obj.surface = surface
    obj.biotsavart = fake_booz.biotsavart
    obj._J = None
    obj._dJ = None
    obj._dJ_by_dcoil_dofs = None

    np.testing.assert_allclose(
        obj.J(),
        surface.major_radius(),
        rtol=_ASPECT_RATIO_VALUE_RTOL,
        atol=_ASPECT_RATIO_VALUE_ATOL,
    )
    expected_total = float(jnp.sum(expected_adjoint))
    cpu_obj = object.__new__(surfaceobjectives_module.MajorRadius)
    cpu_obj.boozer_surface = fake_booz
    cpu_obj.surface = surface
    cpu_obj._J = None
    cpu_obj._dJ = None
    cpu_obj.compute(compute_gradient=True)

    np.testing.assert_allclose(
        obj.J(),
        cpu_obj.J(),
        rtol=_ASPECT_RATIO_VALUE_RTOL,
        atol=_ASPECT_RATIO_VALUE_ATOL,
    )
    np.testing.assert_allclose(
        cpu_obj.dJ(partials=True).data[dep_opt],
        np.asarray([-expected_total, expected_total], dtype=np.float64),
        rtol=_ASPECT_RATIO_GRAD_RTOL,
        atol=_ASPECT_RATIO_GRAD_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(obj.dJ_by_dcoil_dofs(), dtype=np.float64),
        np.asarray([-expected_total, expected_total], dtype=np.float64),
        rtol=_ASPECT_RATIO_GRAD_RTOL,
        atol=_ASPECT_RATIO_GRAD_ATOL,
    )
    projected_derivative = obj.dJ(partials=True)
    np.testing.assert_allclose(
        projected_derivative.data[dep_opt],
        np.asarray([-expected_total, expected_total], dtype=np.float64),
        rtol=_ASPECT_RATIO_GRAD_RTOL,
        atol=_ASPECT_RATIO_GRAD_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(captured["coil_arrays"][0], dtype=np.float64),
        np.asarray(expected_adjoint, dtype=np.float64),
    )
    assert captured["coil_group_indices"] == [(0,)]


@pytest.mark.parametrize(
    ("surfacetype", "linearization_kind"),
    [
        pytest.param("SurfaceRZFourier", "hessian", id="ls-rzfourier"),
        pytest.param(
            "SurfaceXYZTensorFourier",
            "exact_jacobian",
            id="exact-xyztensorfourier",
        ),
    ],
)
def test_major_radius_jax_re_solve_directional_finite_difference(
    surfacetype,
    linearization_kind,
):
    """Validate the JAX ``MajorRadiusJAX`` adjoint vs a central FD probe.

    Oracle: type 4 — central finite-difference directional derivative
    ``(J(c0 + eps*d) - J(c0 - eps*d)) / (2*eps)`` for an eps consistent with
    the FD step / problem condition (eps=1e-5, rtol=1e-7); independent of any
    code-path comparison and exercises the run-code re-solve hook on both
    ``hessian`` (RZFourier) and ``exact_jacobian`` (XYZTensorFourier) lanes.
    Lane: smoke, rtol=1e-7, atol=1e-10.
    """
    surface = _make_aspect_ratio_surface(surfacetype, stellsym=False)
    base_sdofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    surface_dof_map = np.zeros((base_sdofs.size, 2), dtype=np.float64)
    surface_dof_map[3, 0] = 0.015
    surface_dof_map[-5, 1] = -0.02
    surface_dof_map_jax = jnp.asarray(surface_dof_map, dtype=jnp.float64)

    class _FakeBiotSavart:
        x = np.asarray([0.4, -0.2], dtype=np.float64)

        def coil_cotangents_to_dofs_gradient(
            self,
            coil_arrays,
            coil_group_indices,
            *,
            coil_dofs,
        ):
            adjoint = jnp.asarray(coil_arrays[0], dtype=jnp.float64)
            return -(surface_dof_map_jax.T @ adjoint[: base_sdofs.size])

    class _FakeBoozerSurface:
        def __init__(self, biotsavart):
            self.surface = surface
            self.biotsavart = biotsavart
            self.res = {
                "success": True,
                "primal_success": True,
                "iota": jnp.asarray(0.3, dtype=jnp.float64),
                "G": jnp.asarray(1.0, dtype=jnp.float64),
            }
            self.solved_state = None

        @property
        def need_to_run_code(self):
            return True

        def run_code(self, iota, G=None):
            coil_dofs = jnp.asarray(self.biotsavart.x, dtype=jnp.float64)
            sdofs = base_sdofs + surface_dof_map_jax @ coil_dofs
            self.solved_state = types.SimpleNamespace(
                sdofs=sdofs,
                iota=jnp.asarray(iota, dtype=jnp.float64),
                G=jnp.asarray(G, dtype=jnp.float64),
                weight_inv_modB=False,
            )
            return self.res

        def get_solved_runtime_state(self):
            return self.solved_state

        def get_adjoint_runtime_state(self):
            return types.SimpleNamespace(
                decision_size=base_sdofs.size + 2,
                dtype=jnp.float64,
                linearization_kind=linearization_kind,
                solve_transpose_with_status=lambda rhs: (
                    rhs,
                    jnp.asarray(True),
                ),
                stream_group_vjps=lambda adjoint: iter(((adjoint, (0,)),)),
            )

    fake_biotsavart = _FakeBiotSavart()
    fake_booz = _FakeBoozerSurface(fake_biotsavart)
    obj = object.__new__(surfaceobjectives_jax_module.MajorRadiusJAX)
    obj.boozer_surface = fake_booz
    obj.surface = surface
    obj.biotsavart = fake_biotsavart
    obj._J = None
    obj._dJ = None
    obj._dJ_by_dcoil_dofs = None

    base_coil_dofs = fake_biotsavart.x.copy()
    gradient = np.asarray(obj.dJ_by_dcoil_dofs(), dtype=np.float64)
    direction = np.asarray([0.6, -0.4], dtype=np.float64)

    def value_at(coil_dofs):
        fake_biotsavart.x = np.asarray(coil_dofs, dtype=np.float64)
        obj._J = None
        return obj.J()

    eps = 1.0e-5
    fd_directional = (
        value_at(base_coil_dofs + eps * direction)
        - value_at(base_coil_dofs - eps * direction)
    ) / (2.0 * eps)
    np.testing.assert_allclose(
        np.dot(gradient, direction),
        fd_directional,
        rtol=1e-7,
        atol=1e-10,
    )


def test_major_radius_jax_lazy_package_export():
    """Verify lazy ``simsopt.geo.__getattr__`` returns the canonical object.

    Lane: routing — guards the lazy export table in
    ``simsopt/geo/__init__.py`` against typos that would silently shadow
    ``MajorRadiusJAX``; no parity assertion.
    """
    from simsopt.geo import MajorRadiusJAX

    assert MajorRadiusJAX is surfaceobjectives_jax_module.MajorRadiusJAX


class TestToroidalFluxObjectParity:
    @pytest.fixture(autouse=True)
    def _strict_parity_lane(self, monkeypatch, request, parity_lane):
        enable_strict_parity_backend(monkeypatch, request, parity_lane)
        with parity_default_device(parity_lane):
            yield

    def test_reference_object_case_value_parity(self):
        """Match CPU ``ToroidalFlux.J()`` against the JAX BiotSavart path.

        Oracle: type 1 — CPU ``ToroidalFlux.J()`` integrates ``A · gammadash2``
        over the ϕ=const boundary using C++ ``BiotSavart::A`` and C++
        ``Surface::gammadash2``; the JAX variant routes through
        ``BiotSavartJAX``.
        Lane: parity, rtol/atol from ``parity_ladder_tolerances("derivative_heavy")``.
        """
        tf_cpu, tf_jax = _make_reference_object_toroidal_flux_pair()
        _assert_toroidal_flux_value_parity(tf_jax.J(), tf_cpu.J())

    def test_toroidal_flux_is_constant(self):
        """ToroidalFlux is invariant along ϕ on an equilibrium-like surface.

        Oracle: type 1 — CPU ``ToroidalFlux.J`` (same oracle as above) for
        each ϕ-slice, with a closed-form physical witness that the spread
        across slices is bounded by a 1e-2 relative-deviation budget for
        the NCSX equilibrium fixture (a physics modelling allowance, not a
        numerical tolerance).
        Lane: parity, rtol/atol from ``parity_ladder_tolerances("derivative_heavy")``.
        """
        surface = get_exact_surface()
        bs_cpu, bs_jax = _make_ncsx_biotsavart_pair()
        num_phi = surface.gamma().shape[0]
        tf_cpu_values = np.empty(num_phi, dtype=np.float64)
        tf_jax_values = np.empty(num_phi, dtype=np.float64)

        for idx in range(num_phi):
            tf_cpu = ToroidalFlux(surface, bs_cpu, idx=idx)
            tf_jax = ToroidalFlux(surface, bs_jax, idx=idx)
            tf_cpu_values[idx] = tf_cpu.J()
            tf_jax_values[idx] = host_scalar(tf_jax.J())

        np.testing.assert_allclose(
            tf_jax_values,
            tf_cpu_values,
            rtol=_TOROIDAL_FLUX_VALUE_RTOL,
            atol=_TOROIDAL_FLUX_VALUE_ATOL,
        )
        mean_tf = np.mean(tf_jax_values)
        max_err = np.max(np.abs(mean_tf - tf_jax_values)) / abs(mean_tf)
        assert max_err < 1e-2

    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_toroidal_flux_value_parity_matrix(self, surfacetype, stellsym):
        """Match CPU ``ToroidalFlux.J()`` against the JAX wrapper across (surfacetype, stellsym).

        Oracle: type 1 — same CPU ``ToroidalFlux.J()`` oracle as above,
        parametrized over surface kind and stellarator symmetry.
        Lane: parity, rtol/atol from ``parity_ladder_tolerances("derivative_heavy")``.
        """
        _assert_toroidal_flux_pair_parity(
            surfacetype,
            stellsym,
            value_getter=_toroidal_flux_value,
            rtol=_TOROIDAL_FLUX_VALUE_RTOL,
            atol=_TOROIDAL_FLUX_VALUE_ATOL,
        )

    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_toroidal_flux_first_derivative(self, surfacetype, stellsym):
        """Match CPU ``ToroidalFlux.dJ_by_dsurfacecoefficients`` vs JAX wrapper.

        Oracle: type 1 — CPU
        ``ToroidalFlux.dJ_by_dsurfacecoefficients()`` composes the gradient
        from the same C++ ``BiotSavart`` and ``Surface`` derivative primitives.
        Lane: parity, rtol=1e-9, atol=1e-11.
        """
        _assert_toroidal_flux_pair_parity(
            surfacetype,
            stellsym,
            value_getter=_surface_gradient_value,
            rtol=_TOROIDAL_FLUX_SURFACE_GRAD_RTOL,
            atol=_TOROIDAL_FLUX_SURFACE_GRAD_ATOL,
        )

    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_toroidal_flux_second_derivative(self, surfacetype, stellsym):
        """Match CPU ``ToroidalFlux.d2J_..._dsurfacecoefficients`` vs JAX wrapper.

        Oracle: type 1 — CPU
        ``ToroidalFlux.d2J_by_dsurfacecoefficientsdsurfacecoefficients()``
        composes the Hessian from the same C++ ``BiotSavart`` /
        ``Surface`` second-derivative primitives.
        Lane: parity, rtol=1e-8, atol=1e-10.
        """
        _assert_toroidal_flux_pair_parity(
            surfacetype,
            stellsym,
            value_getter=_surface_hessian_value,
            rtol=_TOROIDAL_FLUX_SURFACE_HESS_RTOL,
            atol=_TOROIDAL_FLUX_SURFACE_HESS_ATOL,
        )

    @pytest.mark.parametrize("surfacetype", _SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", _STELLSYM_OPTIONS)
    def test_toroidal_flux_partial_derivatives_wrt_coils(self, surfacetype, stellsym):
        """Match CPU ``ToroidalFlux.dJ_by_dcoils`` vs the JAX coil-derivative path.

        Oracle: type 1 — CPU ``ToroidalFlux.dJ_by_dcoils()`` composes the
        coil-derivative tensor over C++ ``BiotSavart`` coil-derivative
        primitives.
        Lane: parity, rtol=1e-9, atol=1e-7.
        """
        _assert_toroidal_flux_pair_parity(
            surfacetype,
            stellsym,
            value_getter=_coil_gradient_value,
            rtol=_TOROIDAL_FLUX_COIL_GRAD_RTOL,
            atol=_TOROIDAL_FLUX_COIL_GRAD_ATOL,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
