"""JAX-specific ToroidalFlux Taylor and tolerance-parity coverage.

These tests exercise the pure JAX label/objective ingredients directly:

1. Surface-DOF Hessian Taylor convergence for toroidal flux.
2. Coil-family DOF gradient Taylor convergence for toroidal flux.
3. Upstream-shaped ToroidalFlux CPU/JAX parity under tolerance-based checks.
"""

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
from simsopt.geo import optimizer_jax as optimizer_jax_module
from simsopt.geo import surfaceobjectives_jax as surfaceobjectives_jax_module
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
_STELLSYM_OPTIONS = (True, False)
_TOROIDAL_FLUX_VALUE_RTOL = 1e-10
_TOROIDAL_FLUX_VALUE_ATOL = 1e-12
_TOROIDAL_FLUX_SURFACE_GRAD_RTOL = 1e-9
_TOROIDAL_FLUX_SURFACE_GRAD_ATOL = 1e-11
_TOROIDAL_FLUX_SURFACE_HESS_RTOL = 1e-8
_TOROIDAL_FLUX_SURFACE_HESS_ATOL = 1e-10
_TOROIDAL_FLUX_COIL_GRAD_RTOL = 1e-9
_TOROIDAL_FLUX_COIL_GRAD_ATOL = 1e-7


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
        "baseline_dense_plu": None,
        "baseline_coil_dofs": jnp.asarray([0.0], dtype=jnp.float64),
        "coil_set_spec_from_dofs": lambda coil_dofs: coil_dofs,
        "optimize_G": False,
        "predictor_kind": "none",
        "linearization_kind": "hessian",
        "linear_solve_tol": 1.0e-10,
        "linear_solve_stab": 0.0,
        "failure_value": jnp.asarray(1.0, dtype=jnp.float64),
        "failure_scale": jnp.asarray(1.0, dtype=jnp.float64),
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


def test_traceable_runtime_cache_key_avoids_value_hashing_runtime_state(monkeypatch):
    seen_trees = []
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

    booz = types.SimpleNamespace(
        _solver_generation=7,
        options={},
        _collect_optimizer_options=lambda: {},
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
        "coil_dof_extraction_spec": {"unused": True},
        "baseline_x": jnp.arange(5, dtype=jnp.float64),
        "baseline_value": jnp.asarray(1.0, dtype=jnp.float64),
        "baseline_dense_plu": (
            jnp.eye(3, dtype=jnp.float64),
            jnp.eye(3, dtype=jnp.float64),
            jnp.arange(3, dtype=jnp.int32),
        ),
        "baseline_coil_dofs": jnp.arange(4, dtype=jnp.float64),
        "linearization_kind": "hessian",
        "linear_solve_tol": 1.0e-10,
        "linear_solve_stab": 0.0,
        "failure_value": jnp.asarray(2.0, dtype=jnp.float64),
        "failure_scale": jnp.asarray(1.0, dtype=jnp.float64),
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
    assert not any(tree is state["baseline_dense_plu"] for tree in seen_trees)
    assert not any(tree is state["baseline_coil_dofs"] for tree in seen_trees)
    assert not any(tree is state["failure_value"] for tree in seen_trees)
    assert not any(tree is state["failure_scale"] for tree in seen_trees)


def test_traceable_forward_result_requires_adjoint_runtime_even_with_success_filter(
    monkeypatch,
):
    objective_value = jnp.asarray(-123.0, dtype=jnp.float64)
    baseline_x = jnp.asarray([0.5, -0.25, 0.31], dtype=jnp.float64)

    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_predict_warmstart_x",
        lambda *_args, **_kwargs: baseline_x,
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
    failure_value = jnp.asarray(10.0, dtype=jnp.float64)
    failure_scale = jnp.asarray(4.0, dtype=jnp.float64)
    result = surfaceobjectives_jax_module._traceable_forward_result(
        booz,
        lambda dofs: {"coil_dofs": dofs},
        coil_dofs=coil_dofs,
        baseline_x=baseline_x,
        baseline_value=jnp.asarray(3.0, dtype=jnp.float64),
        baseline_dense_plu=None,
        linearization_kind="hessian",
        linear_solve_tol=1.0e-10,
        linear_solve_stab=0.0,
        optimize_G=False,
        baseline_coil_dofs=baseline_coil_dofs,
        failure_value=failure_value,
        failure_scale=failure_scale,
        predictor_kind="ls",
        objective_kwargs={},
        success_filter=lambda _coil_dofs, _solved_x: jnp.asarray(True, dtype=bool),
    )

    expected_penalty = 10.0 + 0.5 * 4.0 * 5.0

    assert bool(result["primal_success"]) is True
    assert bool(result["adjoint_linear_solve_available"]) is False
    assert bool(result["success"]) is False
    np.testing.assert_allclose(np.asarray(result["value"]), expected_penalty)
    np.testing.assert_allclose(np.asarray(objective_value), -123.0)


def test_traceable_runtime_cache_key_uses_structural_success_filter_signature():
    booz = types.SimpleNamespace(
        _solver_generation=7,
        options={},
        _collect_optimizer_options=lambda: {},
    )
    state = {
        "objective_kwargs": {
            "iota_target": 0.23,
            "outer_objective_config": None,
        },
        "optimize_G": False,
        "predictor_kind": "ls",
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
        _collect_optimizer_options=lambda: {},
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
    }

    key = surfaceobjectives_jax_module._traceable_runtime_cache_key(
        booz,
        object(),
        state,
        success_filter=None,
    )

    assert key[5][0] == "tree"


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
        "baseline_dense_plu": state["baseline_dense_plu"],
        "baseline_coil_dofs": state["baseline_coil_dofs"],
        "failure_value": state["failure_value"],
        "failure_scale": state["failure_scale"],
        "coil_dof_extraction_spec": state["coil_dof_extraction_spec"],
    }

    assert not any(
        isinstance(leaf, jax.Array)
        for leaf in jax.tree_util.tree_leaves(runtime_constants)
    )
    assert isinstance(state["objective_kwargs"]["iota_target"], np.ndarray)
    assert isinstance(
        state["objective_kwargs"]["outer_objective_config"]["vessel_gamma"],
        np.ndarray,
    )
    assert isinstance(state["baseline_x"], np.ndarray)
    assert isinstance(state["baseline_dense_plu"][0], np.ndarray)
    assert isinstance(state["baseline_coil_dofs"], np.ndarray)


def test_iotas_jax_value_path_reads_solved_runtime_state(monkeypatch):
    fake_booz = types.SimpleNamespace(
        res={
            "success": True,
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
        "_adjoint_coil_derivative",
        lambda stream_group_vjps, adjoint, biotsavart: (
            np.testing.assert_allclose(np.asarray(adjoint), np.asarray([2.0, -3.0])),
            list(stream_group_vjps(adjoint)),
            surfaceobjectives_jax_module.Derivative({}),
        )[-1],
    )

    fake_booz = types.SimpleNamespace(
        res={"success": True},
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
        biotsavart=None,
    )

    obj = object.__new__(surfaceobjectives_jax_module.IotasJAX)
    obj.boozer_surface = fake_booz
    obj.biotsavart = None
    obj._J = None
    obj._dJ = None
    obj.compute(compute_gradient=True)

    np.testing.assert_allclose(np.asarray(obj._J), 0.37)


def test_get_cached_traceable_runtime_entry_reuses_bundle_for_same_solver_generation(
    monkeypatch,
):
    build_state_calls = []
    build_bundle_calls = []

    booz = types.SimpleNamespace(
        _solver_generation=11,
        _traceable_runtime_entry_cache=None,
        options={},
        _collect_optimizer_options=lambda: {},
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
            "coil_dof_extraction_spec": {"spec": "marker"},
            "baseline_x": jnp.arange(4, dtype=jnp.float64),
            "baseline_value": jnp.asarray(1.0, dtype=jnp.float64),
            "baseline_dense_plu": (
                jnp.eye(2, dtype=jnp.float64),
                jnp.eye(2, dtype=jnp.float64),
                jnp.arange(2, dtype=jnp.int32),
            ),
            "baseline_coil_dofs": jnp.arange(3, dtype=jnp.float64),
            "linearization_kind": "hessian",
            "linear_solve_tol": 1.0e-10,
            "linear_solve_stab": 0.0,
            "failure_value": jnp.asarray(2.0, dtype=jnp.float64),
            "failure_scale": jnp.asarray(1.0, dtype=jnp.float64),
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
            "failure_gradient_for": object(),
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
        _collect_optimizer_options=lambda: {},
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
            "coil_dof_extraction_spec": {"spec": "marker"},
            "baseline_x": jnp.arange(4, dtype=jnp.float64),
            "baseline_value": jnp.asarray(1.0, dtype=jnp.float64),
            "baseline_dense_plu": (
                jnp.eye(2, dtype=jnp.float64),
                jnp.eye(2, dtype=jnp.float64),
                jnp.arange(2, dtype=jnp.int32),
            ),
            "baseline_coil_dofs": jnp.arange(3, dtype=jnp.float64),
            "linearization_kind": "hessian",
            "linear_solve_tol": 1.0e-10,
            "linear_solve_stab": 0.0,
            "failure_value": jnp.asarray(2.0, dtype=jnp.float64),
            "failure_scale": jnp.asarray(1.0, dtype=jnp.float64),
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
            "failure_gradient_for": object(),
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
        _collect_optimizer_options=lambda: {},
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
            "coil_dof_extraction_spec": {"spec": "marker"},
            "baseline_x": jnp.arange(2, dtype=jnp.float64),
            "baseline_value": jnp.asarray(1.0, dtype=jnp.float64),
            "baseline_dense_plu": (
                jnp.eye(2, dtype=jnp.float64),
                jnp.eye(2, dtype=jnp.float64),
                jnp.arange(2, dtype=jnp.int32),
            ),
            "baseline_coil_dofs": jnp.arange(2, dtype=jnp.float64),
            "linearization_kind": "hessian",
            "linear_solve_tol": 1.0e-10,
            "linear_solve_stab": 0.0,
            "failure_value": jnp.asarray(2.0, dtype=jnp.float64),
            "failure_scale": jnp.asarray(1.0, dtype=jnp.float64),
        }

    def build_bundle(_booz, state, *, success_filter=None):
        del success_filter
        build_bundle_calls.append(state["objective_kwargs"]["iota_target"])
        return {
            "state": state,
            "compiled_forward_result_for": object(),
            "compiled_total_gradient_for": object(),
            "compiled_value_and_grad_for": object(),
            "failure_gradient_for": object(),
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
        _collect_optimizer_options=lambda: {},
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
            "coil_dof_extraction_spec": {"spec": "marker"},
            "baseline_x": jnp.arange(2, dtype=jnp.float64),
            "baseline_value": jnp.asarray(1.0, dtype=jnp.float64),
            "baseline_dense_plu": (
                jnp.eye(2, dtype=jnp.float64),
                jnp.eye(2, dtype=jnp.float64),
                jnp.arange(2, dtype=jnp.int32),
            ),
            "baseline_coil_dofs": jnp.arange(2, dtype=jnp.float64),
            "linearization_kind": "hessian",
            "linear_solve_tol": 1.0e-10,
            "linear_solve_stab": 0.0,
            "failure_value": jnp.asarray(2.0, dtype=jnp.float64),
            "failure_scale": jnp.asarray(1.0, dtype=jnp.float64),
        }

    def build_bundle(_booz, state, *, success_filter=None):
        del success_filter
        build_bundle_calls.append(float(state["objective_kwargs"]["iota_target"]))
        return {
            "state": state,
            "compiled_forward_result_for": object(),
            "compiled_total_gradient_for": object(),
            "compiled_value_and_grad_for": object(),
            "failure_gradient_for": object(),
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
        def _build(_):
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
                "baseline_dense_plu": (
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
        "_traceable_total_gradient",
        lambda *_args, **_kwargs: jnp.asarray([0.25], dtype=jnp.float64),
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
                "baseline_dense_plu": (
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
        "_traceable_total_gradient",
        lambda *_args, **_kwargs: jnp.asarray([0.5, -0.75], dtype=jnp.float64),
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


def _quadratic_inner_objective_closure(*, coil_set_spec, **_kwargs):
    def inner_objective(x_inner):
        return 0.5 * jnp.dot(x_inner, x_inner) + jnp.dot(coil_set_spec, x_inner)

    return inner_objective


def test_traceable_inner_stationarity_grad_matches_directional_inner_objective(
    monkeypatch,
):
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_boozer_penalty_objective_closure",
        _quadratic_inner_objective_closure,
    )

    x_inner = jnp.asarray([1.5, -0.25], dtype=jnp.float64)
    tangent = jnp.asarray([0.75, 2.0], dtype=jnp.float64)
    coil_set_spec = jnp.asarray([0.5, -1.25], dtype=jnp.float64)

    stationarity_grad = surfaceobjectives_jax_module._traceable_inner_stationarity_grad(
        x_inner,
        coil_set_spec,
    )
    directional = surfaceobjectives_jax_module._traceable_directional_inner_objective(
        x_inner,
        tangent,
        coil_set_spec,
    )

    np.testing.assert_allclose(
        stationarity_grad,
        np.asarray(x_inner + coil_set_spec, dtype=np.float64),
    )
    assert directional == pytest.approx(
        float(jnp.dot(tangent, stationarity_grad)),
    )


def test_traceable_objective_gradient_parts_use_combined_vjp(monkeypatch):
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_traceable_inner_objective_kwargs",
        lambda _objective_kwargs: {"kind": "inner"},
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_evaluate_traceable_total_objective",
        lambda x_inner, coil_dofs, coil_set_spec, _objective_kwargs: (
            jnp.dot(x_inner, coil_set_spec) + 0.5 * jnp.dot(coil_dofs, coil_dofs)
        ),
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_make_boozer_penalty_objective_closure",
        _quadratic_inner_objective_closure,
    )
    monkeypatch.setattr(
        surfaceobjectives_jax_module,
        "_solve_plu_transpose_with_refinement",
        lambda *_args: _args[-1],
    )

    original_vjp = surfaceobjectives_jax_module.jax.vjp
    vjp_calls = {"count": 0}

    def counting_vjp(fun, *primals, **kwargs):
        vjp_calls["count"] += 1
        return original_vjp(fun, *primals, **kwargs)

    monkeypatch.setattr(surfaceobjectives_jax_module.jax, "vjp", counting_vjp)

    direct_grad, implicit_grad, total_grad = (
        surfaceobjectives_jax_module._traceable_objective_gradient_parts(
            object(),
            lambda coil_dofs: coil_dofs,
            coil_dofs=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
            solved_x=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
            solved_dense_plu=(object(), object(), object()),
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
    assert vjp_calls["count"] == 1


def test_diagnose_traceable_objective_runtime_redevices_cached_baseline_arrays(
    monkeypatch,
):
    objective_config = {
        weight_key: 1.0
        for _, weight_key in surfaceobjectives_jax_module._TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS
    }
    call_checks: dict[str, bool] = {}

    def _record_array(name, value):
        call_checks[name] = isinstance(value, jax.Array)
        return value

    def compiled_forward_result_for(coil_dofs):
        _record_array("compiled_forward_result_for", coil_dofs)
        return {"success": jnp.asarray(True, dtype=bool)}

    def compiled_value_and_grad_for(coil_dofs):
        _record_array("compiled_value_and_grad_for", coil_dofs)
        return (
            jnp.asarray(1.25, dtype=jnp.float64),
            jnp.asarray([0.5, -0.75], dtype=jnp.float64),
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
        solved_dense_plu,
        linearization_kind,
        linear_solve_tol,
        linear_solve_stab,
        objective_kwargs,
        term_name=None,
    ):
        del linearization_kind, linear_solve_tol, linear_solve_stab
        _record_array("gradient_parts_coil_dofs", coil_dofs)
        _record_array("gradient_parts_solved_x", solved_x)
        plu_leaves = jax.tree_util.tree_leaves(solved_dense_plu)
        call_checks["gradient_parts_solved_dense_plu"] = all(
            isinstance(leaf, jax.Array) for leaf in plu_leaves
        )
        assert objective_kwargs["outer_objective_config"] is objective_config
        assert term_name is not None
        grad = jnp.asarray([0.5, -0.75], dtype=jnp.float64)
        return grad, grad, grad

    runtime_entry = {
        "compiled_bundle": {
            "state": {
                "objective_kwargs": {"outer_objective_config": objective_config},
                "baseline_x": np.asarray([1.0, 2.0], dtype=np.float64),
                "baseline_dense_plu": (
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
            "compiled_forward_result_for": compiled_forward_result_for,
            "compiled_value_and_grad_for": compiled_value_and_grad_for,
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
        "compiled_forward_result_for": True,
        "compiled_value_and_grad_for": True,
        "raw_terms_solved_x": True,
        "raw_terms_coil_dofs": True,
        "gradient_parts_coil_dofs": True,
        "gradient_parts_solved_x": True,
        "gradient_parts_solved_dense_plu": True,
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


def _surface_gradient_value(tf, _):
    return tf.dJ_by_dsurfacecoefficients()


def _surface_hessian_value(tf, _):
    return tf.d2J_by_dsurfacecoefficientsdsurfacecoefficients()


def _coil_gradient_value(tf, bs):
    return tf.dJ_by_dcoils()(bs)


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
        """Pure-JAX ToroidalFlux Hessian gate for surface DOFs."""
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
        """Pure-JAX ToroidalFlux gradient gate for a traceable TF coil family."""
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


class TestToroidalFluxObjectParity:
    @pytest.fixture(autouse=True)
    def _strict_parity_lane(self, monkeypatch, request, parity_lane):
        enable_strict_parity_backend(monkeypatch, request, parity_lane)
        with parity_default_device(parity_lane):
            yield

    def test_reference_object_case_value_parity(self):
        tf_cpu, tf_jax = _make_reference_object_toroidal_flux_pair()
        _assert_toroidal_flux_value_parity(tf_jax.J(), tf_cpu.J())

    def test_toroidal_flux_is_constant(self):
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
    def test_toroidal_flux_first_derivative(self, surfacetype, stellsym):
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
        _assert_toroidal_flux_pair_parity(
            surfacetype,
            stellsym,
            value_getter=_coil_gradient_value,
            rtol=_TOROIDAL_FLUX_COIL_GRAD_RTOL,
            atol=_TOROIDAL_FLUX_COIL_GRAD_ATOL,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
