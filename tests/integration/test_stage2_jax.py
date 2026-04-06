"""
Stage 2 JAX backend parity tests.

Validates:
1. SquaredFluxJAX.J() matches SquaredFlux.J() within 1e-12 relative error.
2. SquaredFluxJAX.dJ() gradient matches CPU within 1e-11 relative error.
3. Short L-BFGS-B run produces comparable field error and objective.

All tests require ``simsoptpp`` for the CPU reference.
"""

import json
import inspect
import importlib
import importlib.util
import warnings
from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import jax
import jax.numpy as jnp
import pytest
from conftest import (
    enable_non_strict_jax_backend,
    enable_strict_jax_backend,
    relative_error,
)
import numpy as np

sopp = pytest.importorskip(
    "simsoptpp",
    reason="Stage 2 integration tests require simsoptpp (use candidate-fixed env)",
)

from simsopt.field import (  # noqa: E402
    BiotSavart,
    Current,
    Coil,
    coils_via_symmetries,
)
from simsopt.field.coil import ScaledCurrent  # noqa: E402
from simsopt.geo import (  # noqa: E402
    SurfaceRZFourier,
    CurveCWSFourier,
    CurveCWSFourierCPP,
    CurveXYZFourier,
    create_equally_spaced_curves,
    CurveCurveDistance,
    CurveLength,
    LpCurveCurvature,
)
from simsopt.objectives import SquaredFlux, QuadraticPenalty  # noqa: E402

from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
import simsopt.field.biotsavart_jax_backend as biotsavart_jax_backend_module
from simsopt.field.biotsavart_jax import grouped_biot_savart_A
from simsopt.jax_core import (
    apply_coil_symmetry,
    closed_curve_self_intersection_summary,
    curve_spec_from_curve,
    make_coil_symmetry_spec,
    grouped_biot_savart_B_from_inputs,
    grouped_biot_savart_B_from_spec,
)
from simsopt.geo.optimizer_jax import (
    PRIVATE_OPTIMIZER_JAX_VERSION,
    jax_minimize,
    private_optimizer_runtime_is_supported,
)
from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX
import simsopt.objectives.stage2_target_objective_jax as stage2_target_objective_module
from simsopt.objectives.stage2_target_objective_jax import (
    Stage2PenaltyConfig,
    Stage2TargetOptimizerState,
    _split_stage2_dofs,
    build_stage2_target_objective,
    stage2_target_optimizer_state_from_dofs,
    stage2_target_optimizer_state_to_dofs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "single_stage_optimization"
    / "STAGE_2"
    / "banana_coil_solver.py"
)
WARM_TIMING_REFERENCE_LANE_ERROR = (
    "--record-warm-timings is only supported on the JAX Stage 2 ondevice lane."
)
WARM_TIMING_NO_OPTIMIZATION_ERROR = (
    "--record-warm-timings requires an actual Stage 2 optimization run and "
    "cannot be combined with --probe-only or --init-only."
)
PROFILE_STEP_REFERENCE_LANE_ERROR = (
    "--profile-step-json is only supported on explicit Stage 2 reference lanes."
)
EXPECTED_SQUARED_FLUX_INTERNAL_TIMING_KEYS = {
    "field_B_for_J_s",
    "integral_only_s",
    "field_B_for_dJ_s",
    "integral_value_grad_s",
    "field_B_vjp_s",
}
EXPECTED_B_VJP_COMPONENT_TIMING_KEYS = {
    "curve_geometry_s",
    "current_value_s",
    "single_coil_pullback_s",
    "coil_vjp_s",
}
REDUCED_STAGE2_ARGS = (
    "--backend",
    "jax",
    "--nphi",
    "31",
    "--ntheta",
    "16",
)
_SHORT_RUN_PARITY_RTOL = 1e-3
_STAGE2_VALUE_PARITY_RTOL = 1e-12
_STAGE2_GRADIENT_PARITY_RTOL = 1e-11
_STAGE2_GRADIENT_PARITY_ATOL = 1e-15
_SQUARED_FLUX_DEFINITIONS = (
    "quadratic flux",
    "normalized",
    "local",
)


def test_split_stage2_dofs_matches_legacy_slice_layout():
    dofs = jnp.asarray([7.5, -1.2, 0.3, 4.4, -8.1], dtype=jnp.float64)

    current, curve_dofs = _split_stage2_dofs(dofs, curve_dof_count=4)

    np.testing.assert_allclose(np.asarray(current), np.asarray(dofs[0]), atol=0.0)
    np.testing.assert_allclose(np.asarray(curve_dofs), np.asarray(dofs[1:]), atol=0.0)


def test_stage2_target_optimizer_state_round_trips_flat_dofs():
    dofs = np.asarray([7.5, -1.2, 0.3, 4.4, -8.1], dtype=np.float64)

    state = stage2_target_optimizer_state_from_dofs(dofs, curve_dof_count=4)

    assert isinstance(state, Stage2TargetOptimizerState)
    np.testing.assert_allclose(
        np.asarray(state.current_dof), np.asarray(dofs[0]), atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(state.curve_dofs), np.asarray(dofs[1:]), atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(stage2_target_optimizer_state_to_dofs(state)),
        dofs,
        atol=0.0,
    )


_TARGET_OBJECTIVE_GRAD_ATOL = 5e-12
_TARGET_OBJECTIVE_COMPOSITE_GRAD_ATOL = 1.5e-11
_TARGET_OBJECTIVE_FD_EPS = 1e-6
_TARGET_OBJECTIVE_FD_ATOL = 2e-5
_BACKEND_RUNTIME_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_JAX_DEBUG_NANS",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "SIMSOPT_JAX_COMPILATION_CACHE_DIR",
    "SIMSOPT_JAX_COIL_CHUNK_SIZE",
    "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE",
    "SIMSOPT_BACKEND",
    "STAGE2_BACKEND",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
    "JAX_PLATFORMS",
)


_enable_strict_jax_backend = partial(enable_strict_jax_backend, mode="jax_cpu_parity")
_enable_non_strict_jax_backend = partial(
    enable_non_strict_jax_backend,
    mode="jax_cpu_parity",
)


def _assert_stage2_value_parity(actual, reference, *, definition=None):
    rel_err = relative_error(actual, reference)
    if definition is not None:
        print(
            f"[{definition}] J_cpu={reference:.12e}  J_jax={actual:.12e}  "
            f"rel_err={rel_err:.2e}"
        )
    assert rel_err < _STAGE2_VALUE_PARITY_RTOL, (
        f"Relative error {rel_err:.2e} exceeds {_STAGE2_VALUE_PARITY_RTOL:.0e}"
    )


def _assert_stage2_gradient_parity(actual, reference, *, err_msg):
    np.testing.assert_allclose(
        actual,
        reference,
        rtol=_STAGE2_GRADIENT_PARITY_RTOL,
        atol=_STAGE2_GRADIENT_PARITY_ATOL,
        err_msg=err_msg,
    )


def _assert_jax_objective_fallback_active(squared_flux_jax):
    assert not squared_flux_jax._use_jax_native
    assert squared_flux_jax._uses_jax_objective_fallback


def _stage2_context_kwargs():
    return {
        "squared_flux_weight": 1.0,
        "length_weight": 1.0,
        "length_target": 1.75,
        "cc_weight": 1.0,
        "cc_threshold": 0.25,
        "curvature_weight": 1.0,
    }


def _load_stage2_script_module():
    saved_meta_path = list(sys.meta_path)
    saved_simsopt_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "simsopt" or name.startswith("simsopt.")
    }
    spec = importlib.util.spec_from_file_location(
        "stage2_banana_coil_solver",
        STAGE2_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Stage 2 script module from {STAGE2_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.meta_path[:] = saved_meta_path
        for name in list(sys.modules):
            if name == "simsopt" or name.startswith("simsopt."):
                del sys.modules[name]
        sys.modules.update(saved_simsopt_modules)


def _fresh_import(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _run_stage2_script(*args):
    return subprocess.run(
        [
            sys.executable,
            str(STAGE2_SCRIPT),
            *args,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_closed_curve_self_intersection_summary_detects_crossing_curve():
    crossing_gamma = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        dtype=jnp.float64,
    )
    square_gamma = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=jnp.float64,
    )

    crossing_summary = closed_curve_self_intersection_summary(
        crossing_gamma,
        neighbor_skip=1,
    )
    square_summary = closed_curve_self_intersection_summary(
        square_gamma,
        neighbor_skip=1,
    )

    assert bool(np.asarray(crossing_summary[3]))
    assert float(np.asarray(crossing_summary[2])) > 0.0
    assert not bool(np.asarray(square_summary[3]))
    assert float(np.asarray(square_summary[2])) == pytest.approx(0.0)


def _assert_stage2_script_failure(result, expected_error):
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert expected_error in output


def _assert_target_backend_runtime_gate(result):
    if private_optimizer_runtime_is_supported(jax.__version__):
        return True
    _assert_stage2_script_failure(
        result,
        f"On-device optimizer requires JAX >= {PRIVATE_OPTIMIZER_JAX_VERSION}",
    )
    return False


def _assert_target_backend_success(result):
    output = f"{result.stdout}\n{result.stderr}"
    if not _assert_target_backend_runtime_gate(result):
        return None
    assert result.returncode == 0, output
    return output


def _load_stage2_results_json(output_root):
    return json.loads(
        next(output_root.glob("**/results.json")).read_text(encoding="utf-8")
    )


def _run_stage2_probe_and_load_payload(*args):
    with tempfile.TemporaryDirectory(prefix="stage2-probe-") as temp_dir:
        export_json = Path(temp_dir) / "probe.json"
        result = _run_stage2_script(
            *args,
            "--probe-only",
            "--export-objective-json",
            str(export_json),
        )
        output = f"{result.stdout}\n{result.stderr}"
        assert result.returncode == 0, output
        payload = json.loads(export_json.read_text(encoding="utf-8"))
    return result, payload


@contextmanager
def _isolated_backend_runtime(mode: str):
    from simsopt.backend import invalidate_backend_cache, set_backend
    from simsopt.jax_core import invalidate_kernel_cache

    previous_env = {name: os.environ.get(name) for name in _BACKEND_RUNTIME_ENV_VARS}
    try:
        if mode == "native_cpu":
            set_backend(mode, strict=False, configure_runtime=False)
        else:
            set_backend(mode, strict=False)
        invalidate_kernel_cache()
        yield
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        invalidate_backend_cache()
        invalidate_kernel_cache()


def _build_fake_b_vjp_profile():
    return {
        "wall_time_s": 0.125,
        "component_timings_s": {
            "curve_geometry_s": 0.01,
            "current_value_s": 0.03,
            "single_coil_pullback_s": 0.04,
            "coil_vjp_s": 0.05,
        },
        "dominant_components": [
            {
                "name": "coil_vjp_s",
                "elapsed_s": 0.05,
                "share": 0.3333333333333333,
            }
        ],
        "per_coil_timings_s": [],
        "dominant_coils": [
            {
                "coil_index": 0,
                "elapsed_s": 0.125,
                "share": 1.0,
            }
        ],
    }


def _assert_b_vjp_profile_payload(payload):
    assert (
        set(payload["squared_flux_field_b_vjp_component_timings_s"])
        == EXPECTED_B_VJP_COMPONENT_TIMING_KEYS
    )
    assert payload["dominant_squared_flux_field_b_vjp_components"]
    assert payload["dominant_squared_flux_field_b_vjp_coils"]


def _build_stage2_target_objective_contract_case(
    definition: str = "quadratic flux",
    *,
    return_context: bool = False,
):
    eval_surf = SurfaceRZFourier(
        nfp=5,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.arange(31) / 31,
        quadpoints_theta=np.arange(16) / 16,
    )
    eval_surf.set_rc(0, 0, 0.915)
    eval_surf.set_rc(1, 0, 0.15)
    eval_surf.set_zs(1, 0, 0.15)
    eval_surf.fix_all()

    coil_surf = SurfaceRZFourier(
        nfp=5,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.arange(64) / 64,
        quadpoints_theta=np.arange(64) / 64,
    )
    coil_surf.set_rc(0, 0, 0.976)
    coil_surf.set_rc(1, 0, 0.22)
    coil_surf.set_zs(1, 0, 0.22)

    tf_curves = create_equally_spaced_curves(
        20,
        1,
        stellsym=False,
        R0=0.976,
        R1=0.4,
        order=1,
    )
    tf_currents = [Current(1.0) * 1e5 for _ in range(20)]
    for tf_curve in tf_curves:
        tf_curve.fix_all()
    for tf_current in tf_currents:
        tf_current.fix_all()
    tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]

    banana_curve = CurveCWSFourier(
        np.linspace(0, 1, 128, endpoint=False),
        order=2,
        surf=coil_surf,
    )
    banana_curve.set("phic(0)", 0.06)
    banana_curve.set("thetac(0)", 0.5)
    banana_curve.set("phic(1)", 0.03)
    banana_curve.set("thetas(1)", 0.1)
    banana_current = Current(1.0)
    banana_coils = coils_via_symmetries(
        [banana_curve],
        [ScaledCurrent(banana_current, 1e4)],
        coil_surf.nfp,
        coil_surf.stellsym,
    )

    all_coils = tf_coils + banana_coils
    bs_jax = BiotSavartJAX(all_coils)
    jf = SquaredFluxJAX(eval_surf, bs_jax, definition=definition)
    jls = CurveLength(banana_curve)
    jccdist = CurveCurveDistance([coil.curve for coil in all_coils], 0.05)
    jc = LpCurveCurvature(banana_curve, 4, 40)
    objective = (
        jf + 0.0005 * QuadraticPenalty(jls, 1.75, "max") + 100.0 * jccdist + 0.0001 * jc
    )

    target_bundle = build_stage2_target_objective(
        surface=eval_surf,
        tf_coils=tf_coils,
        banana_coils=banana_coils,
        banana_curve=banana_curve,
        penalty_config=Stage2PenaltyConfig(
            squared_flux_weight=1.0,
            length_weight=0.0005,
            length_target=1.75,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            curvature_p_norm=4,
            squared_flux_definition=definition,
        ),
    )

    if return_context:
        return (
            objective,
            target_bundle,
            {
                "eval_surf": eval_surf,
                "coil_surf": coil_surf,
                "tf_coils": tf_coils,
                "banana_coils": banana_coils,
                "banana_curve": banana_curve,
                "banana_current": banana_current,
            },
        )
    return objective, target_bundle


@contextmanager
def _forbid_stage2_host_materialization(monkeypatch, message: str):
    def _reject_device_get(*_args, **_kwargs):
        raise AssertionError(message)

    with monkeypatch.context() as patch:
        patch.setattr(
            stage2_target_objective_module.jax,
            "device_get",
            _reject_device_get,
        )
        yield


def _closure_has_jax_array_leaf(fn) -> bool:
    inspect_target = inspect.unwrap(fn)
    if not inspect.isfunction(inspect_target):
        return False
    for value in inspect.getclosurevars(inspect_target).nonlocals.values():
        if callable(value):
            continue
        if any(
            isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(value)
        ):
            return True
    return False


def _stage2_contract_case_base_curve_geometry(banana_curve):
    banana_curve_spec = curve_spec_from_curve(banana_curve)
    base_gamma, base_gammadash, _base_gammadashdash = (
        stage2_target_objective_module.curve_geometry_from_dofs(
            banana_curve_spec,
            jnp.asarray(banana_curve_spec.dofs, dtype=jnp.float64),
        )
    )
    return base_gamma, base_gammadash


def _stage2_contract_case_banana_symmetry_inputs(banana_coils):
    return stage2_target_objective_module._banana_symmetry_runtime_inputs_from_coils(
        banana_coils
    )


def _centered_fd_gradient(fun, x, *, eps):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        grad[i] = (float(fun(x + step)) - float(fun(x - step))) / (2.0 * eps)
    return grad


def _assert_first_order_taylor_contract(fun, x, grad, *, seed):
    rng = np.random.RandomState(seed)
    direction = rng.randn(*x.shape)
    direction /= np.linalg.norm(direction)
    epsilons = (1.0e-5, 5.0e-6)
    errors = []
    baseline = float(fun(x))
    directional_derivative = float(np.dot(grad, direction))
    for eps in epsilons:
        trial = x + eps * direction
        residual = float(fun(trial)) - baseline - eps * directional_derivative
        errors.append(abs(residual))

    assert errors[1] < 0.55 * errors[0], (
        f"Taylor convergence stalled: err0={errors[0]:.3e}, err1={errors[1]:.3e}"
    )


def test_stage2_curvature_threshold_policy_caps_at_40():
    stage2_script = _load_stage2_script_module()

    assert stage2_script.resolve_curvature_threshold(10.0) == pytest.approx(20.0)
    assert stage2_script.resolve_curvature_threshold(30.0) == pytest.approx(30.0)
    assert stage2_script.resolve_curvature_threshold(80.0) == pytest.approx(40.0)


class _FakeStage2SquaredFluxTerm:
    def __init__(self, value, grad):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)

    def J(self):
        return self._value

    def dJ(self):
        return self._grad


class _DummyDerivative:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def __call__(self, _optim):
        return self._values


def _build_fake_lbfgs_result(x0, value, grad):
    from simsopt.geo.optimizer_jax_private import _LBFGSResults

    x0_array = np.asarray(x0, dtype=np.float64)
    grad_array = np.asarray(grad, dtype=np.float64)
    d = len(x0_array)
    maxcor = 3
    return _LBFGSResults(
        converged=False,
        failed=False,
        k=0,
        nfev=1,
        ngev=1,
        x_k=x0_array,
        f_k=np.float64(value),
        g_k=grad_array,
        s_history=np.zeros((maxcor, d), dtype=np.float64),
        y_history=np.zeros((maxcor, d), dtype=np.float64),
        rho_history=np.zeros((maxcor,), dtype=np.float64),
        gamma=np.float64(1.0),
        status=0,
        ls_status=0,
    )


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def coil_surf_setup():
    """Create a lightweight but non-trivial coil + surface configuration.

    Uses 4 equally-spaced circular-ish coils with nfp=2, stellsym,
    and a simple torus surface.  The configuration is cheap to evaluate
    but exercises the full Optimizable chain.
    """
    # Use nfp=1 with 2 coils — deliberately poor coverage to produce
    # a non-trivial B·n (objective ~ 1e-3 .. 1e-1), avoiding the
    # near-zero regime where relative error is meaningless.
    ncoils = 2
    nfp = 1
    stellsym = False
    R0 = 1.0
    R1 = 0.5
    order = 3

    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=stellsym,
        R0=R0,
        R1=R1,
        order=order,
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    nphi = 32
    ntheta = 32
    surf = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0, 1, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False),
    )
    surf.set_rc(0, 0, R0)
    surf.set_rc(1, 0, 0.2)
    surf.set_zs(1, 0, 0.2)
    surf.fix_all()

    return coils, surf, base_curves, base_currents


# -----------------------------------------------------------------------
# Test 1: Objective value parity
# -----------------------------------------------------------------------


class TestObjectiveValueParity:
    """SquaredFluxJAX.J() must match SquaredFlux.J()."""

    @pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
    def test_j_parity(self, coil_surf_setup, definition):
        coils, surf, _, _ = coil_surf_setup

        # CPU reference
        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
        jf_cpu = SquaredFlux(surf, bs_cpu, definition=definition)
        j_cpu = jf_cpu.J()

        # JAX
        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax, definition=definition)
        j_jax = jf_jax.J()

        _assert_stage2_value_parity(j_jax, j_cpu, definition=definition)

    def test_j_with_target(self, coil_surf_setup):
        """Parity with a non-zero target field."""
        coils, surf, _, _ = coil_surf_setup

        rng = np.random.RandomState(42)
        target = rng.randn(*surf.normal().shape[:2]) * 0.01

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
        jf_cpu = SquaredFlux(surf, bs_cpu, target=target)
        j_cpu = jf_cpu.J()

        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax, target=target)
        j_jax = jf_jax.J()

        _assert_stage2_value_parity(j_jax, j_cpu)


# -----------------------------------------------------------------------
# Test 2: Gradient parity
# -----------------------------------------------------------------------


class TestGradientParity:
    """SquaredFluxJAX.dJ() must match SquaredFlux.dJ()."""

    @pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
    def test_gradient_parity(self, coil_surf_setup, definition):
        coils, surf, _, _ = coil_surf_setup

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
        jf_cpu = SquaredFlux(surf, bs_cpu, definition=definition)
        grad_cpu = jf_cpu.dJ()

        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax, definition=definition)
        grad_jax = jf_jax.dJ()

        _assert_stage2_gradient_parity(
            grad_jax,
            grad_cpu,
            err_msg=f"Gradient mismatch between JAX and CPU for {definition!r}",
        )

    def test_j_only_uses_forward_path_until_gradient_is_requested(
        self, coil_surf_setup, monkeypatch
    ):
        coils, surf, _, _ = coil_surf_setup
        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax)
        assert jf_jax._use_jax_native

        calls = {"forward": 0, "value_grad": 0}
        original_forward = jf_jax._jit_forward_dofs
        original_value_grad = jf_jax._jit_val_grad_dofs

        def counted_forward(flat_dofs):
            calls["forward"] += 1
            return original_forward(flat_dofs)

        def counted_value_grad(flat_dofs):
            calls["value_grad"] += 1
            return original_value_grad(flat_dofs)

        monkeypatch.setattr(jf_jax, "_jit_forward_dofs", counted_forward)
        monkeypatch.setattr(jf_jax, "_jit_val_grad_dofs", counted_value_grad)

        first_value = jf_jax.J()
        second_value = jf_jax.J()

        assert first_value == pytest.approx(second_value)
        assert calls == {"forward": 1, "value_grad": 0}

        jf_jax.dJ()
        assert calls == {"forward": 1, "value_grad": 1}

    def test_gradient_then_value_reuses_cached_squared_flux_value(
        self, coil_surf_setup, monkeypatch
    ):
        coils, surf, _, _ = coil_surf_setup
        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax)
        assert jf_jax._use_jax_native

        calls = {"forward": 0, "value_grad": 0}
        original_forward = jf_jax._jit_forward_dofs
        original_value_grad = jf_jax._jit_val_grad_dofs

        def counted_forward(flat_dofs):
            calls["forward"] += 1
            return original_forward(flat_dofs)

        def counted_value_grad(flat_dofs):
            calls["value_grad"] += 1
            return original_value_grad(flat_dofs)

        monkeypatch.setattr(jf_jax, "_jit_forward_dofs", counted_forward)
        monkeypatch.setattr(jf_jax, "_jit_val_grad_dofs", counted_value_grad)

        grad = jf_jax.dJ()
        value = jf_jax.J()

        assert np.asarray(grad).shape[0] > 0
        assert np.isfinite(value)
        assert calls == {"forward": 0, "value_grad": 1}


# -----------------------------------------------------------------------
# Test 3: Composite objective gradient (SquaredFlux + curve penalties)
# -----------------------------------------------------------------------


class TestCompositeGradient:
    """Gradient through JF = Jf + penalty works correctly with JAX backend."""

    def test_composite_gradient_parity(self, coil_surf_setup):
        coils, surf, base_curves, _ = coil_surf_setup

        LENGTH_WEIGHT = 1e-4
        LENGTH_TARGET = 5.0

        # CPU composite
        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
        jf_cpu = SquaredFlux(surf, bs_cpu)
        jls_cpu = CurveLength(base_curves[0])
        JF_cpu = jf_cpu + LENGTH_WEIGHT * QuadraticPenalty(
            jls_cpu, LENGTH_TARGET, "max"
        )
        j_cpu = JF_cpu.J()
        grad_cpu = JF_cpu.dJ()

        # JAX composite (only SquaredFlux is JAX; CurveLength stays CPU)
        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax)
        jls_jax = CurveLength(base_curves[0])
        JF_jax = jf_jax + LENGTH_WEIGHT * QuadraticPenalty(
            jls_jax, LENGTH_TARGET, "max"
        )
        j_jax = JF_jax.J()
        grad_jax = JF_jax.dJ()

        rel_err_j = relative_error(j_jax, j_cpu)
        assert rel_err_j < 1e-10

        np.testing.assert_allclose(
            grad_jax,
            grad_cpu,
            rtol=1e-9,
            atol=1e-15,
            err_msg="Composite objective gradient mismatch",
        )


# -----------------------------------------------------------------------
# Test 4: Short optimization run parity
# -----------------------------------------------------------------------


class TestShortOptimizationRun:
    """A short L-BFGS-B run must produce comparable results."""

    def test_short_run_parity(self, coil_surf_setup):
        coils, surf, base_curves, base_currents = coil_surf_setup

        MAXITER = 20
        LENGTH_WEIGHT = 1e-4
        LENGTH_TARGET = 5.0

        def _build_and_run(use_jax):
            # Rebuild fresh coils (optimization mutates DOFs)
            nfp, stellsym = 1, False
            curves = create_equally_spaced_curves(
                2,
                nfp,
                stellsym=stellsym,
                R0=1.0,
                R1=0.5,
                order=3,
            )
            currents = [Current(1e5) for _ in range(2)]
            all_coils = coils_via_symmetries(curves, currents, nfp, stellsym)

            s = SurfaceRZFourier(
                nfp=nfp,
                stellsym=stellsym,
                mpol=1,
                ntor=1,
                quadpoints_phi=np.linspace(0, 1, 32, endpoint=False),
                quadpoints_theta=np.linspace(0, 1, 32, endpoint=False),
            )
            s.set_rc(0, 0, 1.0)
            s.set_rc(1, 0, 0.2)
            s.set_zs(1, 0, 0.2)
            s.fix_all()

            if use_jax:
                bs = BiotSavartJAX(all_coils)
                Jf = SquaredFluxJAX(s, bs)
            else:
                bs_obj = BiotSavart(all_coils)
                bs_obj.set_points(s.gamma().reshape((-1, 3)))
                Jf = SquaredFlux(s, bs_obj)

            Jls = CurveLength(curves[0])
            JF = Jf + LENGTH_WEIGHT * QuadraticPenalty(Jls, LENGTH_TARGET, "max")

            dofs = JF.x.copy()

            def fun(x):
                JF.x = x
                return JF.J(), JF.dJ()

            res = jax_minimize(
                fun,
                dofs,
                method="lbfgs",
                tol=1e-10,
                maxiter=MAXITER,
                options={"ftol": 0.0},
                value_and_grad=True,
            )
            return res.fun, res.nit

        j_cpu, nit_cpu = _build_and_run(use_jax=False)
        j_jax, nit_jax = _build_and_run(use_jax=True)

        print(f"CPU: J={j_cpu:.8e}, nit={nit_cpu}")
        print(f"JAX: J={j_jax:.8e}, nit={nit_jax}")

        rel_diff = relative_error(j_jax, j_cpu)
        assert rel_diff < _SHORT_RUN_PARITY_RTOL, (
            f"Short-run final objectives differ by {rel_diff:.2%}: "
            f"CPU={j_cpu:.6e}, JAX={j_jax:.6e}"
        )


# -----------------------------------------------------------------------
# Test 4b: Optimizer trajectory parity (P28)
# -----------------------------------------------------------------------

# P28/P29/P30 tolerances.
#
# Calibration rationale (see jax_port_code_review_2026-04-01.md §6):
#
# - JAX does NOT guarantee bitwise reproducibility across JIT states
#   (jax/docs/api_compatibility.md: "exact numerics are not necessarily
#   stable ... within or without jax.jit").
# - XLA reduction order is unspecified; floating-point non-associativity
#   means accumulated sums can differ by O(eps_mach * condition_number).
# - JAX's own optimizer tests use rtol=2e-4 for converged solutions and
#   explicitly note "cannot compare step for step with scipy BFGS."
# - Upstream SIMSOPT uses np.allclose (rtol=1e-5) for algorithm parity
#   and 1e-3 to 1e-4 for physics quantities at convergence.
#
# P28 (identical start): trajectories match to ~1e-13, set gate at 1e-6.
# P29 (physics at convergence): upstream range 1e-3 to 1e-4, set 1e-3.
# P30 (perturbed start): gradient differences compound through L-BFGS-B
#   Hessian approximation over 50 steps.  Upstream does not test basin
#   stability.  Gate set at 1e-2 (10x achieved ~3e-3), matching upstream
#   cross-stage tolerance range (test_qfm.py: 1e-3).
_TRAJECTORY_MAXITER = 50
_TRAJECTORY_OBJ_PARITY_RTOL = 1e-6
_TRAJECTORY_DOF_L2_RTOL = 1e-6
_PHYSICS_PARITY_RTOL = 1e-3
_BASIN_OBJ_PARITY_RTOL = 1e-2
_BASIN_DOF_L2_RTOL = 1e-2
_BASIN_PERTURBATION_FRACTION = 5e-3
_LENGTH_WEIGHT = 1e-3
_LENGTH_TARGET = 5.0


def _stage2_relative_l2_error(lhs, rhs) -> float:
    lhs_array = np.asarray(lhs, dtype=float)
    rhs_array = np.asarray(rhs, dtype=float)
    return float(np.linalg.norm(lhs_array - rhs_array) / np.linalg.norm(lhs_array))


def _stage2_basin_perturbation(dofs, rng) -> np.ndarray:
    dof_array = np.asarray(dofs, dtype=float)
    dof_scale = np.maximum(np.abs(dof_array), 1.0)
    return _BASIN_PERTURBATION_FRACTION * dof_scale * rng.randn(*dof_array.shape)


def _stage2_max_bdotn_over_b(all_coils, surface) -> float:
    bs_eval = BiotSavart(all_coils)
    bs_eval.set_points(surface.gamma().reshape((-1, 3)))
    b_surface = np.asarray(bs_eval.B())
    normal = surface.unitnormal().reshape((-1, 3))
    bdotn = np.sum(b_surface * normal, axis=1)
    bmag = np.linalg.norm(b_surface, axis=1)
    return float(np.max(np.abs(bdotn) / bmag))


def _stage2_coil_lengths(curves) -> list[float]:
    return [float(CurveLength(curve).J()) for curve in curves]


def _build_and_run_stage2(use_jax, maxiter=_TRAJECTORY_MAXITER, dof_perturbation=None):
    """Build a Stage 2 composite problem, run L-BFGS-B, return full results.

    Shared helper for P28/P29/P30.  Creates all objects from scratch — no
    shared state with other tests. Pins the backend mode per lane so earlier
    suite activity cannot bleed a different runtime contract into this helper.
    """
    from scipy.optimize import minimize as scipy_minimize

    with _isolated_backend_runtime("jax_cpu_parity" if use_jax else "native_cpu"):
        nfp, stellsym = 1, False
        curves = create_equally_spaced_curves(
            2,
            nfp,
            stellsym=stellsym,
            R0=1.0,
            R1=0.5,
            order=3,
        )
        currents_list = [Current(1e5) for _ in range(2)]
        all_coils = coils_via_symmetries(curves, currents_list, nfp, stellsym)

        s = SurfaceRZFourier(
            nfp=nfp,
            stellsym=stellsym,
            mpol=1,
            ntor=1,
            quadpoints_phi=np.linspace(0, 1, 32, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 32, endpoint=False),
        )
        s.set_rc(0, 0, 1.0)
        s.set_rc(1, 0, 0.2)
        s.set_zs(1, 0, 0.2)
        s.fix_all()

        if use_jax:
            bs = BiotSavartJAX(all_coils)
            Jf = SquaredFluxJAX(s, bs)
        else:
            bs_obj = BiotSavart(all_coils)
            bs_obj.set_points(s.gamma().reshape((-1, 3)))
            Jf = SquaredFlux(s, bs_obj)

        Jls = CurveLength(curves[0])
        JF = Jf + _LENGTH_WEIGHT * QuadraticPenalty(Jls, _LENGTH_TARGET, "max")

        dofs = JF.x.copy()
        if dof_perturbation is not None:
            dofs = dofs + dof_perturbation

        iteration_objs = []

        def callback(x):
            JF.x = x
            iteration_objs.append(float(JF.J()))

        def fun(x):
            JF.x = x
            return JF.J(), JF.dJ()

        res = scipy_minimize(
            fun,
            dofs,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 0.0},
            callback=callback,
        )

        JF.x = res.x
        return {
            "trajectory": np.array(iteration_objs),
            "initial_dofs": np.array(dofs).copy(),
            "final_dofs": np.array(res.x).copy(),
            "final_obj": float(res.fun),
            "nit": int(res.nit),
            # Measure final physics with a fresh CPU BiotSavart instance so
            # backend-specific caches cannot affect the parity check.
            "max_BdotN_over_B": _stage2_max_bdotn_over_b(all_coils, s),
            "coil_lengths": _stage2_coil_lengths(curves),
        }


class TestOptimizerTrajectoryParity:
    """P28/P29/P30: optimizer trajectory, physics, and basin stability."""

    @pytest.fixture(scope="class")
    def trajectory_results(self):
        """Run CPU and JAX optimizations once, share across P28/P29."""
        return _build_and_run_stage2(False), _build_and_run_stage2(True)

    def test_lbfgs_trajectory_parity(self, trajectory_results):
        """P28: 50 L-BFGS-B steps — trajectory and final DOFs must match."""
        cpu, jax_r = trajectory_results

        min_len = min(len(cpu["trajectory"]), len(jax_r["trajectory"]))
        assert min_len >= 5, (
            f"Too few iterations: CPU={len(cpu['trajectory'])}, "
            f"JAX={len(jax_r['trajectory'])}"
        )

        traj_cpu = cpu["trajectory"][:min_len]
        traj_jax = jax_r["trajectory"][:min_len]
        traj_rel_err = np.max(
            np.abs(traj_cpu - traj_jax) / np.maximum(np.abs(traj_cpu), 1e-30)
        )
        print(f"Trajectory: {min_len} iters, max rel err = {traj_rel_err:.2e}")
        assert traj_rel_err < _TRAJECTORY_OBJ_PARITY_RTOL, (
            f"Trajectory diverged: max rel err = {traj_rel_err:.2e} "
            f"(threshold {_TRAJECTORY_OBJ_PARITY_RTOL:.0e})"
        )

        obj_rel_err = relative_error(jax_r["final_obj"], cpu["final_obj"])
        print(
            f"Final objective: CPU={cpu['final_obj']:.8e}, "
            f"JAX={jax_r['final_obj']:.8e}, rel_err={obj_rel_err:.2e}"
        )
        assert obj_rel_err < _TRAJECTORY_OBJ_PARITY_RTOL

        dof_l2 = _stage2_relative_l2_error(cpu["final_dofs"], jax_r["final_dofs"])
        print(f"Final DOFs L2 relative error = {dof_l2:.2e}")
        assert dof_l2 < _TRAJECTORY_DOF_L2_RTOL, (
            f"Final DOFs diverged: L2 rel = {dof_l2:.2e}"
        )

    def test_physics_quantities_at_convergence(self, trajectory_results):
        """P29: max|B.n|/|B| and coil lengths agree to 3+ digits."""
        cpu, jax_r = trajectory_results

        bn_rel = relative_error(jax_r["max_BdotN_over_B"], cpu["max_BdotN_over_B"])
        print(
            f"|B.n|/|B|: CPU={cpu['max_BdotN_over_B']:.6e}, "
            f"JAX={jax_r['max_BdotN_over_B']:.6e}, rel_err={bn_rel:.2e}"
        )
        assert bn_rel < _PHYSICS_PARITY_RTOL, (
            f"|B.n|/|B| diverged: rel_err={bn_rel:.2e}"
        )

        assert len(cpu["coil_lengths"]) == len(jax_r["coil_lengths"]), (
            "Coil-count mismatch at convergence: "
            f"CPU={len(cpu['coil_lengths'])}, JAX={len(jax_r['coil_lengths'])}"
        )
        for i, (lc, lj) in enumerate(zip(cpu["coil_lengths"], jax_r["coil_lengths"])):
            rel = relative_error(lj, lc)
            print(f"Coil {i} length: CPU={lc:.6f}, JAX={lj:.6f}, rel={rel:.2e}")
            assert rel < _PHYSICS_PARITY_RTOL, (
                f"Coil {i} length diverged: rel={rel:.2e}"
            )

    def test_basin_stability(self):
        """P30: Perturbed initial coils converge to same local minimum."""
        rng = np.random.RandomState(42)

        # Get DOF count from a throwaway build
        probe = _build_and_run_stage2(use_jax=False, maxiter=0)
        perturbation = _stage2_basin_perturbation(probe["final_dofs"], rng)

        cpu = _build_and_run_stage2(False, dof_perturbation=perturbation)
        jax_r = _build_and_run_stage2(True, dof_perturbation=perturbation)
        np.testing.assert_allclose(cpu["initial_dofs"], jax_r["initial_dofs"])

        obj_rel = relative_error(jax_r["final_obj"], cpu["final_obj"])
        print(
            f"Basin stability: CPU={cpu['final_obj']:.8e}, "
            f"JAX={jax_r['final_obj']:.8e}, rel_err={obj_rel:.2e}"
        )
        assert obj_rel < _BASIN_OBJ_PARITY_RTOL, (
            f"Perturbed runs diverged: rel_err={obj_rel:.2e}"
        )

        dof_l2 = _stage2_relative_l2_error(cpu["final_dofs"], jax_r["final_dofs"])
        print(f"Basin stability DOFs L2 relative error = {dof_l2:.2e}")
        assert dof_l2 < _BASIN_DOF_L2_RTOL, (
            f"Perturbed final DOFs diverged: L2 rel={dof_l2:.2e}"
        )

    def test_stage2_helper_restores_backend_runtime_contract(self, monkeypatch):
        from simsopt.backend import get_backend_config, invalidate_backend_cache

        monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_parity")
        monkeypatch.setenv("SIMSOPT_BACKEND_STRICT", "1")
        invalidate_backend_cache()

        try:
            _build_and_run_stage2(True, maxiter=0)

            restored = get_backend_config()
            assert restored.mode == "jax_gpu_parity"
            assert restored.strict is True
        finally:
            invalidate_backend_cache()


# -----------------------------------------------------------------------
# Test 5: BiotSavartJAX.B() parity
# -----------------------------------------------------------------------


class TestBiotSavartJAXParity:
    """BiotSavartJAX.B() must match BiotSavart.B()."""

    def test_b_parity(self, coil_surf_setup):
        coils, surf, _, _ = coil_surf_setup
        points = surf.gamma().reshape((-1, 3))

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(points)
        B_cpu = bs_cpu.B()

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        B_jax = np.asarray(bs_jax.B())

        np.testing.assert_allclose(
            B_jax,
            B_cpu,
            rtol=1e-10,
            atol=1e-15,
            err_msg="BiotSavartJAX.B() does not match CPU",
        )

    def test_b_vjp_parity(self, coil_surf_setup):
        """B_vjp returns the same Derivative as CPU B_vjp."""
        coils, surf, _, _ = coil_surf_setup
        points = surf.gamma().reshape((-1, 3))

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(points)
        B_cpu = bs_cpu.B()

        # Use B as the cotangent vector (arbitrary but non-zero)
        v = B_cpu.copy()

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        deriv_cpu = bs_cpu.B_vjp(v)
        deriv_jax = bs_jax.B_vjp(v)
        b_vjp_rel_tol = 1e-10
        b_vjp_abs_tol = 1e-14

        for coil in coils:
            np.testing.assert_allclose(
                deriv_jax(coil),
                deriv_cpu(coil),
                rtol=b_vjp_rel_tol,
                atol=b_vjp_abs_tol,
                err_msg="BiotSavartJAX.B_vjp() does not match CPU",
            )

    def test_profile_b_vjp_reports_component_breakdown(self, coil_surf_setup):
        coils, surf, _, _ = coil_surf_setup
        points = surf.gamma().reshape((-1, 3))

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        profile = bs_jax.profile_B_vjp(np.asarray(bs_jax.B()))

        assert profile["wall_time_s"] >= 0.0
        assert (
            set(profile["component_timings_s"]) == EXPECTED_B_VJP_COMPONENT_TIMING_KEYS
        )
        assert profile["dominant_components"]
        assert profile["dominant_coils"]
        assert profile["dominant_pullback_groups"]
        assert len(profile["per_coil_timings_s"]) == len(coils)
        component_total = sum(profile["component_timings_s"].values())
        assert component_total == pytest.approx(
            sum(entry["total_s"] for entry in profile["per_coil_timings_s"])
            + sum(entry["elapsed_s"] for entry in profile["pullback_group_timings_s"])
        )
        assert component_total >= 0.9 * profile["wall_time_s"]
        for entry in profile["per_coil_timings_s"]:
            assert (
                set(entry["component_timings_s"])
                == EXPECTED_B_VJP_COMPONENT_TIMING_KEYS
            )
            assert entry["total_s"] >= 0.0
        for entry in profile["pullback_group_timings_s"]:
            assert entry["kind"] in {"prep", "group_pullback"}
            assert entry["elapsed_s"] >= 0.0
            assert entry["coil_indices"]

    def test_dB_by_dX_parity(self, coil_surf_setup):
        """dB/dX spatial Jacobian must match CPU at the same evaluation points.

        Validates the forward Jacobian tensor convention:
        ``dB_by_dX[p, j, l] = ∂_j B_l(x_p)`` (axis 1 = derivative direction,
        axis 2 = B component).

        Requires simsoptpp for the CPU reference.
        """
        coils, surf, _, _ = coil_surf_setup
        points = surf.gamma().reshape((-1, 3))

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(points)
        dB_cpu = bs_cpu.dB_by_dX()

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        dB_jax = np.asarray(bs_jax.dB_by_dX())

        assert dB_jax.shape == dB_cpu.shape, (
            f"Shape mismatch: JAX {dB_jax.shape} vs CPU {dB_cpu.shape}"
        )

        np.testing.assert_allclose(
            dB_jax,
            dB_cpu,
            rtol=1e-10,
            atol=1e-15,
            err_msg="BiotSavartJAX.dB_by_dX() does not match CPU",
        )

    def test_A_parity(self, coil_surf_setup):
        """Vector potential A must match CPU at the same evaluation points.

        BiotSavartJAX does not expose an A() method, so this test uses the
        pure function ``grouped_biot_savart_A`` with coil data extracted from
        the adapter.

        Requires simsoptpp for the CPU reference.
        """
        coils, surf, _, _ = coil_surf_setup
        points = surf.gamma().reshape((-1, 3))

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(points)
        A_cpu = bs_cpu.A()

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        groups = bs_jax._extract_coil_data_grouped()
        coil_arrays = [(g, gd, c) for g, gd, c, _ in groups]
        A_jax = np.asarray(grouped_biot_savart_A(bs_jax._points_jax, coil_arrays))

        assert A_jax.shape == A_cpu.shape, (
            f"Shape mismatch: JAX {A_jax.shape} vs CPU {A_cpu.shape}"
        )

        np.testing.assert_allclose(
            A_jax,
            A_cpu,
            rtol=1e-10,
            atol=1e-15,
            err_msg="biot_savart_A (via grouped_biot_savart_A) does not match CPU",
        )


# -----------------------------------------------------------------------
# Test 6: DOF mutation round-trip
# -----------------------------------------------------------------------


class TestDOFMutation:
    """Verify JAX re-evaluates correctly after DOF mutation."""

    def test_j_changes_after_dof_mutation(self, coil_surf_setup):
        coils, surf, _, _ = coil_surf_setup

        bs_jax = BiotSavartJAX(coils)
        jf = SquaredFluxJAX(surf, bs_jax)

        j_before = jf.J()

        # Perturb DOFs
        old_dofs = jf.x.copy()
        delta = np.zeros_like(old_dofs)
        delta[0] = 1e-3
        jf.x = old_dofs + delta
        j_after = jf.J()

        assert j_before != j_after, "J did not change after DOF mutation"

        # Restore and verify round-trip
        jf.x = old_dofs
        j_restored = jf.J()
        np.testing.assert_allclose(j_restored, j_before, rtol=1e-12)


# -----------------------------------------------------------------------
# Test 7: Mixed-quadrature parity (TF + banana-like coils)
# -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed_quad_setup():
    """Coils with mixed quadrature point counts.

    Mimics the real Columbia workflow where TF coils use default
    quadrature (15*order) and banana coils use explicit higher
    quadrature (128 points).
    """
    nfp = 1
    stellsym = False
    R0 = 1.0
    R1 = 0.5
    order = 3

    # Two TF coils with default 45 quadrature points (15 * order=3)
    tf_curves = create_equally_spaced_curves(
        2,
        nfp,
        stellsym=stellsym,
        R0=R0,
        R1=R1,
        order=order,
    )
    tf_currents = [Current(1e5), Current(1e5)]
    tf_coils = [Coil(c, cur) for c, cur in zip(tf_curves, tf_currents)]

    # One "banana" coil with 128 quadrature points (same Fourier order)
    banana = CurveXYZFourier(
        np.linspace(0, 1, 128, endpoint=False),
        order=order,
    )
    banana.x = tf_curves[0].x.copy()
    banana_coil = Coil(banana, Current(1e5))

    all_coils = tf_coils + [banana_coil]

    nphi = 32
    ntheta = 32
    surf = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0, 1, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False),
    )
    surf.set_rc(0, 0, R0)
    surf.set_rc(1, 0, 0.2)
    surf.set_zs(1, 0, 0.2)
    surf.fix_all()

    return all_coils, surf


class TestMixedQuadratureParity:
    """Parity when coils have different quadrature point counts."""

    def test_b_parity(self, mixed_quad_setup):
        """BiotSavartJAX.B() matches CPU with mixed quadrature."""
        coils, surf = mixed_quad_setup
        points = surf.gamma().reshape((-1, 3))

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(points)
        B_cpu = bs_cpu.B()

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        B_jax = np.asarray(bs_jax.B())

        np.testing.assert_allclose(
            B_jax,
            B_cpu,
            rtol=1e-10,
            atol=1e-15,
            err_msg="BiotSavartJAX.B() mixed-quad parity failure",
        )

    def test_chunked_grouped_paths_match_cpu_on_large_point_cloud(
        self, mixed_quad_setup
    ):
        """Spec and array-compat grouped paths agree with CPU on multi-chunk inputs."""
        coils, surf = mixed_quad_setup
        base_points = surf.gamma().reshape((-1, 3))
        point_offsets = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.01, -0.015, 0.005],
                [-0.02, 0.01, -0.01],
            ]
        )
        points = np.concatenate(
            [base_points + offset for offset in point_offsets], axis=0
        )
        assert points.shape[0] > 2 * 256

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(points)
        B_cpu = bs_cpu.B()

        bs_jax = BiotSavartJAX(coils)
        coil_arrays = bs_jax.grouped_coil_arrays_from_dofs(jnp.asarray(bs_jax.x))
        coil_spec = bs_jax.coil_set_spec()

        B_from_inputs = np.asarray(
            grouped_biot_savart_B_from_inputs(points, coil_arrays)
        )
        B_from_spec = np.asarray(grouped_biot_savart_B_from_spec(points, coil_spec))

        np.testing.assert_allclose(
            B_from_inputs,
            B_from_spec,
            rtol=1e-12,
            atol=1e-15,
            err_msg="Chunked grouped compatibility path diverged from spec path",
        )
        np.testing.assert_allclose(
            B_from_spec,
            B_cpu,
            rtol=1e-10,
            atol=1e-15,
            err_msg="Chunked grouped path lost CPU parity on large point cloud",
        )

    @pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
    def test_j_parity(self, mixed_quad_setup, definition):
        """SquaredFluxJAX.J() matches CPU with mixed quadrature."""
        coils, surf = mixed_quad_setup

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
        jf_cpu = SquaredFlux(surf, bs_cpu, definition=definition)
        j_cpu = jf_cpu.J()

        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax, definition=definition)
        j_jax = jf_jax.J()

        _assert_stage2_value_parity(j_jax, j_cpu)

    @pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
    def test_gradient_parity(self, mixed_quad_setup, definition):
        """SquaredFluxJAX.dJ() matches CPU with mixed quadrature."""
        coils, surf = mixed_quad_setup

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
        jf_cpu = SquaredFlux(surf, bs_cpu, definition=definition)
        grad_cpu = jf_cpu.dJ()

        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax, definition=definition)
        grad_jax = jf_jax.dJ()

        _assert_stage2_gradient_parity(
            grad_jax,
            grad_cpu,
            err_msg=f"Gradient mismatch with mixed quadrature for {definition!r}",
        )

    def test_j_recomputes_after_field_points_change(self, mixed_quad_setup):
        """Fallback J() must not reuse stale values after field.set_points(...)."""
        coils, surf = mixed_quad_setup

        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax)
        assert not jf_jax._use_jax_native

        initial_value = jf_jax.J()
        shifted_points = surf.gamma().reshape((-1, 3)) + np.array([0.05, 0.0, 0.0])
        bs_jax.set_points(shifted_points)

        updated_value = jf_jax.J()
        jf_jax.recompute_bell()
        recomputed_value = jf_jax.J()

        assert recomputed_value != pytest.approx(initial_value)
        assert updated_value == pytest.approx(recomputed_value)

    def test_gradient_then_value_reuses_cached_fallback_squared_flux_value(
        self,
        mixed_quad_setup,
        monkeypatch,
    ):
        coils, surf = mixed_quad_setup

        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax)
        assert not jf_jax._use_jax_native

        calls = {"B": 0}
        original_B = bs_jax.B

        def counted_B():
            calls["B"] += 1
            return original_B()

        monkeypatch.setattr(bs_jax, "B", counted_B)

        grad = jf_jax.dJ()
        value = jf_jax.J()

        assert np.asarray(grad).shape[0] > 0
        assert value >= 0.0
        assert calls["B"] == 1

    def test_value_then_gradient_recomputes_fallback_squared_flux_gradient(
        self,
        mixed_quad_setup,
        monkeypatch,
    ):
        coils, surf = mixed_quad_setup

        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax)
        assert not jf_jax._use_jax_native

        calls = {"B": 0, "B_vjp": 0}
        original_B = bs_jax.B
        original_B_vjp = bs_jax.B_vjp

        def counted_B():
            calls["B"] += 1
            return original_B()

        def counted_B_vjp(v):
            calls["B_vjp"] += 1
            return original_B_vjp(v)

        monkeypatch.setattr(bs_jax, "B", counted_B)
        monkeypatch.setattr(bs_jax, "B_vjp", counted_B_vjp)

        value = jf_jax.J()
        grad = jf_jax.dJ()

        assert value >= 0.0
        assert np.asarray(grad).shape[0] > 0
        assert calls == {"B": 2, "B_vjp": 1}

    def test_strict_mode_allows_squared_flux_jax_objective_fallback(
        self,
        mixed_quad_setup,
        monkeypatch,
        request,
    ):
        coils, surf = mixed_quad_setup
        _enable_strict_jax_backend(monkeypatch, request)

        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax)

        _assert_jax_objective_fallback_active(jf_jax)
        assert jf_jax.J() >= 0.0
        grad = jf_jax.dJ()
        assert np.asarray(grad).shape[0] > 0

    def test_non_strict_squared_flux_jax_objective_fallback_is_silent(
        self,
        mixed_quad_setup,
        monkeypatch,
        request,
    ):
        coils, surf = mixed_quad_setup
        _enable_non_strict_jax_backend(monkeypatch, request)

        bs_jax = BiotSavartJAX(coils)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jf_jax = SquaredFluxJAX(surf, bs_jax)

        _assert_jax_objective_fallback_active(jf_jax)
        assert not caught


class TestStrictFieldFallbacks:
    def test_biotsavart_rejects_removed_cpu_geometry_fallback(self):
        class _UnsupportedCurve:
            pass

        bs_jax = object.__new__(BiotSavartJAX)
        with pytest.raises(
            TypeError,
            match="BiotSavartJAX.*JAX geometry hooks.*CPU curve-geometry fallback was removed",
        ):
            bs_jax._base_curve_geometry_with_timings(_UnsupportedCurve())

    def test_strict_mode_uses_spec_native_forward_path_before_cpu_geometry_fallback(
        self,
        coil_surf_setup,
        monkeypatch,
        request,
    ):
        coils, surf, _, _ = coil_surf_setup
        expected_points = surf.gamma().reshape((-1, 3))
        _enable_strict_jax_backend(monkeypatch, request)

        bs_jax = BiotSavartJAX(coils)
        bs_jax._jax_native = False
        bs_jax.set_points(expected_points)
        monkeypatch.setattr(
            bs_jax,
            "coil_specs",
            lambda: (_ for _ in ()).throw(NotImplementedError),
        )
        monkeypatch.setattr(
            biotsavart_jax_backend_module,
            "_supports_native_curve_geometry",
            lambda curve: False,
        )

        field_value = bs_jax.B()
        assert np.asarray(field_value).shape == np.asarray(expected_points).shape

    @pytest.mark.parametrize("strict_mode", [False, True])
    def test_biotsavart_rejects_removed_cpu_pullback_fallback(
        self,
        monkeypatch,
        request,
        strict_mode,
    ):
        if strict_mode:
            _enable_strict_jax_backend(monkeypatch, request)
        else:
            _enable_non_strict_jax_backend(monkeypatch, request)

        class _UnsupportedCurve:
            pass

        class _FrozenCurrent:
            dof_size = 0

        class _Coil:
            def __init__(self):
                self.curve = _UnsupportedCurve()
                self.current = _FrozenCurrent()

        with pytest.raises(
            TypeError,
            match="BiotSavartJAX.*JAX pullback hooks.*CPU coil-pullback fallback was removed",
        ):
            biotsavart_jax_backend_module._project_single_coil_cotangent_data(
                _Coil(),
                jnp.ones((1, 3), dtype=jnp.float64),
                jnp.ones((1, 3), dtype=jnp.float64),
                jnp.ones((1,), dtype=jnp.float64),
            )


# -----------------------------------------------------------------------
# Test 8: CurveCWSFourierCPP banana coil (real production curve type)
# -----------------------------------------------------------------------


def _build_banana_coil_cpp_setup():
    """TF coils + a real CurveCWSFourierCPP banana coil with production-like DOFs.

    This exercises the exact curve type and DOF pattern used in the Columbia
    Stage 2 workflow (banana_coil_solver.py), not just a CurveXYZFourier
    proxy with different nquad.  Non-zero DOFs (phic, thetac, thetas) ensure
    the CWS gradient components are at meaningful magnitudes for parity
    testing — with all-zero DOFs the curve degenerates to a plain circle
    and all CWS DOF gradients fall to the noise floor (~1e-18).

    The CurveCWSFourierCPP curve is defined on a *coil winding surface*
    (r=0.5) that encloses the smaller *evaluation surface* (r=0.2),
    keeping the coil well-separated from evaluation points.
    """
    from simsopt.geo import CurveCWSFourierCPP

    nfp = 1
    stellsym = False
    R0 = 1.0

    # Coil winding surface (large minor radius)
    coil_surf = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0, 1, 32, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 32, endpoint=False),
    )
    coil_surf.set_rc(0, 0, R0)
    coil_surf.set_rc(1, 0, 0.5)
    coil_surf.set_zs(1, 0, 0.5)

    # Evaluation surface (smaller, inside coil winding surface)
    eval_surf = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0, 1, 16, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 16, endpoint=False),
    )
    eval_surf.set_rc(0, 0, R0)
    eval_surf.set_rc(1, 0, 0.2)
    eval_surf.set_zs(1, 0, 0.2)
    eval_surf.fix_all()

    # TF coils with 45 quadpoints (order=3)
    tf_curves = create_equally_spaced_curves(
        2,
        nfp,
        stellsym=stellsym,
        R0=R0,
        R1=0.5,
        order=3,
    )
    tf_coils = [Coil(c, Current(1e5)) for c in tf_curves]

    # CurveCWSFourierCPP banana coil with 128 quadpoints and production-like DOFs.
    # Matches banana_coil_solver.py initialization pattern (H=0 default, localized curve).
    banana = CurveCWSFourierCPP(
        np.linspace(0, 1, 128, endpoint=False),
        order=1,
        surf=coil_surf,
    )
    banana.set("phic(0)", 0.06)
    banana.set("thetac(0)", 0.5)
    banana.set("phic(1)", 0.03)
    banana.set("thetas(1)", 0.1)
    banana_coil = Coil(banana, Current(1e5))

    all_coils = tf_coils + [banana_coil]
    return all_coils, eval_surf, banana_coil


def _build_jax_field_on_surface(coils, surf):
    points = surf.gamma().reshape((-1, 3))
    bs_jax = BiotSavartJAX(coils)
    bs_jax.set_points(points)
    return points, bs_jax


@pytest.fixture(scope="module")
def banana_coil_setup():
    coils, eval_surf, _banana_coil = _build_banana_coil_cpp_setup()
    return coils, eval_surf


@pytest.fixture(scope="module")
def banana_coil_cpp_setup():
    return _build_banana_coil_cpp_setup()


@pytest.fixture
def banana_coil_jax_setup():
    nfp = 1
    stellsym = False
    R0 = 1.0

    coil_surf = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0, 1, 32, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 32, endpoint=False),
    )
    coil_surf.set_rc(0, 0, R0)
    coil_surf.set_rc(1, 0, 0.5)
    coil_surf.set_zs(1, 0, 0.5)

    eval_surf = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0, 1, 16, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 16, endpoint=False),
    )
    eval_surf.set_rc(0, 0, R0)
    eval_surf.set_rc(1, 0, 0.2)
    eval_surf.set_zs(1, 0, 0.2)
    eval_surf.fix_all()

    tf_curves = create_equally_spaced_curves(
        2,
        nfp,
        stellsym=stellsym,
        R0=R0,
        R1=0.5,
        order=3,
    )
    tf_coils = [Coil(c, Current(1e5)) for c in tf_curves]
    for coil in tf_coils:
        coil.curve.fix_all()
        coil.current.fix_all()

    banana = CurveCWSFourier(
        np.linspace(0, 1, 128, endpoint=False),
        order=1,
        surf=coil_surf,
    )
    banana.set("phic(0)", 0.06)
    banana.set("thetac(0)", 0.5)
    banana.set("phic(1)", 0.03)
    banana.set("thetas(1)", 0.1)
    banana_coil = Coil(banana, Current(1e5))

    return tf_coils + [banana_coil], eval_surf, banana_coil


class TestCurveCWSFourierCPPParity:
    """Parity checks for the shared CurveCWSFourierCPP CPU/JAX contract."""

    def test_direct_cpu_third_derivative_is_unsupported(self, banana_coil_cpp_setup):
        """Raw CPU third-derivative parity is intentionally out of contract."""
        _, _, banana_coil = banana_coil_cpp_setup

        with pytest.raises(
            RuntimeError,
            match="gammadashdashdash_impl was not implemented",
        ):
            banana_coil.curve.gammadashdashdash()

    def test_b_parity(self, banana_coil_setup):
        """BiotSavartJAX.B() matches CPU with CurveCWSFourierCPP coils."""
        coils, surf = banana_coil_setup
        points = surf.gamma().reshape((-1, 3))

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(points)
        B_cpu = bs_cpu.B()

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        B_jax = np.asarray(bs_jax.B())

        np.testing.assert_allclose(
            B_jax,
            B_cpu,
            rtol=1e-10,
            atol=1e-15,
            err_msg="B parity failure with CurveCWSFourierCPP banana coil",
        )

    def test_j_parity(self, banana_coil_setup):
        """SquaredFluxJAX.J() matches CPU with CurveCWSFourierCPP coils."""
        coils, surf = banana_coil_setup

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
        j_cpu = SquaredFlux(surf, bs_cpu).J()

        bs_jax = BiotSavartJAX(coils)
        j_jax = SquaredFluxJAX(surf, bs_jax).J()

        _assert_stage2_value_parity(j_jax, j_cpu)

    def test_gradient_parity(self, banana_coil_setup):
        """SquaredFluxJAX.dJ() matches CPU with CurveCWSFourierCPP coils."""
        coils, surf = banana_coil_setup

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
        grad_cpu = SquaredFlux(surf, bs_cpu).dJ()

        bs_jax = BiotSavartJAX(coils)
        grad_jax = SquaredFluxJAX(surf, bs_jax).dJ()

        _assert_stage2_gradient_parity(
            grad_jax,
            grad_cpu,
            err_msg="Gradient mismatch with CurveCWSFourierCPP banana coil",
        )


class TestCurveCWSFourierCPPJaxFieldPath:
    def test_strict_mode_allows_squared_flux_for_curvecwsfouriercpp(
        self,
        banana_coil_cpp_setup,
        monkeypatch,
        request,
    ):
        coils, surf, _banana_coil = banana_coil_cpp_setup
        _enable_strict_jax_backend(monkeypatch, request)

        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax)

        _assert_jax_objective_fallback_active(jf_jax)
        assert jf_jax.J() >= 0.0
        grad = jf_jax.dJ()
        assert np.asarray(grad).shape[0] > 0

    def test_b_uses_jax_curvecwsfouriercpp_geometry(
        self, banana_coil_cpp_setup, monkeypatch
    ):
        coils, surf, banana_coil = banana_coil_cpp_setup
        points, bs_jax = _build_jax_field_on_surface(coils, surf)

        monkeypatch.setattr(
            banana_coil.curve,
            "gamma",
            lambda: (_ for _ in ()).throw(
                AssertionError(
                    "BiotSavartJAX.B() should use CurveCWSFourierCPP.gamma_jax"
                )
            ),
        )
        monkeypatch.setattr(
            banana_coil.curve,
            "gammadash",
            lambda: (_ for _ in ()).throw(
                AssertionError(
                    "BiotSavartJAX.B() should use CurveCWSFourierCPP.gammadash_jax"
                )
            ),
        )

        B_jax = np.asarray(bs_jax.B())

        assert B_jax.shape == points.shape

    def test_b_vjp_bypasses_python_coil_vjp_for_curvecwsfouriercpp(
        self,
        banana_coil_cpp_setup,
        monkeypatch,
    ):
        coils, surf, banana_coil = banana_coil_cpp_setup
        points, bs_jax = _build_jax_field_on_surface(coils, surf)

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(points)
        v = np.asarray(bs_jax.B())
        deriv_cpu = bs_cpu.B_vjp(v)

        monkeypatch.setattr(
            banana_coil,
            "vjp",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "BiotSavartJAX.B_vjp() should bypass Coil.vjp() for CurveCWSFourierCPP"
                )
            ),
        )

        deriv = bs_jax.B_vjp(v)

        assert (
            deriv(banana_coil.curve).shape[0] == banana_coil.curve.local_full_dof_size
        )
        np.testing.assert_allclose(
            deriv(banana_coil.curve),
            deriv_cpu(banana_coil.curve),
            rtol=1e-9,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            deriv(banana_coil.current),
            deriv_cpu(banana_coil.current),
            rtol=1e-9,
            atol=1e-15,
        )

    def test_b_vjp_includes_curvecwsfouriercpp_surface_derivative(
        self,
        banana_coil_cpp_setup,
    ):
        coils, surf, banana_coil = banana_coil_cpp_setup
        _points, bs_jax = _build_jax_field_on_surface(coils, surf)
        deriv = bs_jax.B_vjp(np.asarray(bs_jax.B()))

        surface_grad = deriv(banana_coil.curve.surf)
        assert surface_grad.shape[0] == banana_coil.curve.surf.local_dof_size
        assert np.all(np.isfinite(surface_grad))
        assert np.linalg.norm(surface_grad) > 0.0

    def test_b_vjp_uses_spec_pullback_for_curvecwsfouriercpp(
        self,
        banana_coil_cpp_setup,
        monkeypatch,
    ):
        coils, surf, banana_coil = banana_coil_cpp_setup
        points, bs_jax = _build_jax_field_on_surface(coils, surf)

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(points)
        v = np.asarray(bs_jax.B())
        deriv_cpu = bs_cpu.B_vjp(v)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError(
                "BiotSavartJAX.B_vjp() should use the immutable-spec pullback for "
                "CurveCWSFourierCPP"
            )

        monkeypatch.setattr(
            banana_coil.curve, "dgamma_by_dcoeff_vjp_jax", fail_if_called
        )
        monkeypatch.setattr(
            banana_coil.curve,
            "dgammadash_by_dcoeff_vjp_jax",
            fail_if_called,
        )
        monkeypatch.setattr(
            banana_coil.curve, "dgamma_by_dsurf_vjp_jax", fail_if_called
        )
        monkeypatch.setattr(
            banana_coil.curve,
            "dgammadash_by_dsurf_vjp_jax",
            fail_if_called,
        )

        deriv = bs_jax.B_vjp(v)

        np.testing.assert_allclose(
            deriv(banana_coil.curve),
            deriv_cpu(banana_coil.curve),
            rtol=1e-9,
            atol=1e-15,
        )
        surface_grad = deriv(banana_coil.curve.surf)
        assert surface_grad.shape[0] == banana_coil.curve.surf.local_dof_size
        assert np.all(np.isfinite(surface_grad))
        assert np.linalg.norm(surface_grad) > 0.0


class TestCurveCWSFourierNativeFieldPath:
    def test_b_uses_native_curvecwsfourier_geometry(
        self, banana_coil_jax_setup, monkeypatch
    ):
        coils, surf, banana_coil = banana_coil_jax_setup
        points = surf.gamma().reshape((-1, 3))

        monkeypatch.setattr(
            banana_coil.curve,
            "gamma",
            lambda: (_ for _ in ()).throw(
                AssertionError("BiotSavartJAX.B() should use CurveCWSFourier.gamma_jax")
            ),
        )
        monkeypatch.setattr(
            banana_coil.curve,
            "gammadash",
            lambda: (_ for _ in ()).throw(
                AssertionError(
                    "BiotSavartJAX.B() should use CurveCWSFourier.gammadash_jax"
                )
            ),
        )

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        B_jax = np.asarray(bs_jax.B())
        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(points)
        B_cpu = bs_cpu.B()

        assert B_jax.shape == points.shape
        np.testing.assert_allclose(B_jax, B_cpu, rtol=1e-10, atol=1e-15)

    def test_b_vjp_bypasses_python_coil_vjp_for_free_curvecwsfourier(
        self,
        banana_coil_jax_setup,
        monkeypatch,
    ):
        coils, surf, banana_coil = banana_coil_jax_setup
        points = surf.gamma().reshape((-1, 3))

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(points)
        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        v = np.asarray(bs_jax.B())
        deriv_cpu = bs_cpu.B_vjp(v)

        monkeypatch.setattr(
            banana_coil,
            "vjp",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "BiotSavartJAX.B_vjp() should bypass Coil.vjp() for CurveCWSFourier"
                )
            ),
        )

        deriv = bs_jax.B_vjp(v)

        assert (
            deriv(banana_coil.curve).shape[0] == banana_coil.curve.local_full_dof_size
        )
        np.testing.assert_allclose(
            deriv(banana_coil.curve),
            deriv_cpu(banana_coil.curve),
            rtol=1e-9,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            deriv(banana_coil.current),
            deriv_cpu(banana_coil.current),
            rtol=1e-9,
            atol=1e-15,
        )

    def test_b_vjp_includes_curvecwsfourier_surface_derivative(
        self,
        banana_coil_jax_setup,
    ):
        coils, surf, banana_coil = banana_coil_jax_setup
        points = surf.gamma().reshape((-1, 3))

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        deriv = bs_jax.B_vjp(np.asarray(bs_jax.B()))

        surface_grad = deriv(banana_coil.curve.surf)
        assert surface_grad.shape[0] == banana_coil.curve.surf.local_dof_size
        assert np.all(np.isfinite(surface_grad))
        assert np.linalg.norm(surface_grad) > 0.0

    def test_b_vjp_keeps_native_pullback_on_device_for_curvecwsfourier(
        self,
        banana_coil_jax_setup,
        monkeypatch,
    ):
        def fail_if_called(*_args, **_kwargs):
            raise AssertionError(
                "Native CurveCWSFourier B_vjp path should not fall back to coil.vjp()"
            )

        coils, surf, _banana_coil = banana_coil_jax_setup
        points = surf.gamma().reshape((-1, 3))

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)

        monkeypatch.setattr(_banana_coil, "vjp", fail_if_called)

        deriv = bs_jax.B_vjp(np.asarray(bs_jax.B()))
        assert np.linalg.norm(deriv(coils[-1].curve)) > 0.0


class TestStage2BananaBoundary:
    def test_stage2_curve_classes_match_on_stage2_surface(self):
        quadpoints = np.linspace(0, 1, 128, endpoint=False)
        surf = SurfaceRZFourier(
            nfp=5,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(64) / 64,
            quadpoints_theta=np.arange(64) / 64,
        )
        surf.set_rc(0, 0, 0.976)
        surf.set_rc(1, 0, 0.22)
        surf.set_zs(1, 0, 0.22)

        curve_jax = CurveCWSFourier(quadpoints, 2, surf)
        curve_cpp = CurveCWSFourierCPP(quadpoints, 2, surf)
        for curve in (curve_jax, curve_cpp):
            curve.set("phic(0)", 0.06)
            curve.set("thetac(0)", 0.5)
            curve.set("phic(1)", 0.03)
            curve.set("thetas(1)", 0.1)

        np.testing.assert_allclose(curve_jax.gamma(), curve_cpp.gamma(), atol=1e-14)
        np.testing.assert_allclose(
            curve_jax.gammadash(),
            curve_cpp.gammadash(),
            atol=1e-13,
        )
        np.testing.assert_allclose(curve_jax.kappa(), curve_cpp.kappa(), atol=1e-12)

    def test_stage2_composite_objective_matches_across_curve_classes(self):
        eval_surf = SurfaceRZFourier(
            nfp=5,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(31) / 31,
            quadpoints_theta=np.arange(16) / 16,
        )
        eval_surf.set_rc(0, 0, 0.915)
        eval_surf.set_rc(1, 0, 0.15)
        eval_surf.set_zs(1, 0, 0.15)
        eval_surf.fix_all()

        coil_surf = SurfaceRZFourier(
            nfp=5,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(64) / 64,
            quadpoints_theta=np.arange(64) / 64,
        )
        coil_surf.set_rc(0, 0, 0.976)
        coil_surf.set_rc(1, 0, 0.22)
        coil_surf.set_zs(1, 0, 0.22)

        tf_curves = create_equally_spaced_curves(
            20,
            1,
            stellsym=False,
            R0=0.976,
            R1=0.4,
            order=1,
        )
        tf_currents = [Current(1.0) * 1e5 for _ in range(20)]
        for tf_curve in tf_curves:
            tf_curve.fix_all()
        for tf_current in tf_currents:
            tf_current.fix_all()

        def build_objective(curve_cls):
            curve = curve_cls(
                np.linspace(0, 1, 128, endpoint=False), order=2, surf=coil_surf
            )
            curve.set("phic(0)", 0.06)
            curve.set("thetac(0)", 0.5)
            curve.set("phic(1)", 0.03)
            curve.set("thetas(1)", 0.1)
            coils = [
                Coil(curve_obj, current)
                for curve_obj, current in zip(tf_curves, tf_currents)
            ]
            banana_coil = Coil(curve, Current(1.0) * 1e4)
            all_coils = coils + [banana_coil]
            bs_jax = BiotSavartJAX(all_coils)
            jf = SquaredFluxJAX(eval_surf, bs_jax)
            jls = CurveLength(curve)
            jccdist = CurveCurveDistance([coil.curve for coil in all_coils], 0.05)
            jc = LpCurveCurvature(curve, 4, 40)
            objective = (
                jf
                + 0.0005 * QuadraticPenalty(jls, 1.75, "max")
                + 100.0 * jccdist
                + 0.0001 * jc
            )
            return float(objective.J()), np.asarray(objective.dJ(), dtype=float)

        objective_cpp, grad_cpp = build_objective(CurveCWSFourierCPP)
        objective_jax, grad_jax = build_objective(CurveCWSFourier)

        np.testing.assert_allclose(objective_jax, objective_cpp, rtol=1e-12, atol=1e-18)
        # CurveCurveDistance pulls back through different implementations
        # for the wrapper and native curve classes. The resulting drift is
        # confined to nominally flat ~1e-13 components, while the resolved
        # objective value and large gradient entries remain parity-clean.
        composite_grad_abs_tol = 1e-12
        np.testing.assert_allclose(
            grad_jax,
            grad_cpp,
            rtol=1e-9,
            atol=composite_grad_abs_tol,
        )

    @pytest.mark.parametrize("backend", ["cpu", "jax"])
    def test_stage2_probe_reports_shared_production_banana_curve(self, backend):
        result, payload = _run_stage2_probe_and_load_payload(
            "--backend",
            backend,
            "--nphi",
            "31",
            "--ntheta",
            "16",
        )

        assert result.returncode == 0
        assert payload["banana_curve_class"] == "CurveCWSFourierCPP"

    def test_stage2_probe_only_cpu_backend_ondevice_exports_target_objective_source(
        self,
    ):
        result, payload = _run_stage2_probe_and_load_payload(
            *REDUCED_STAGE2_ARGS,
            "--backend",
            "cpu",
            "--optimizer-backend",
            "ondevice",
            "--skip-postprocess",
        )

        probe_output = f"{result.stdout}\n{result.stderr}"
        assert result.returncode == 0, probe_output
        assert payload["backend"] == "cpu"
        assert payload["optimizer_backend"] == "ondevice"
        assert payload["composite"]["objective_source"] == "target-objective"

    def test_stage2_probe_override_dofs_evaluates_requested_state(self):
        with tempfile.TemporaryDirectory(prefix="stage2-override-dofs-") as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            result = _run_stage2_script(
                *REDUCED_STAGE2_ARGS,
                "--optimizer-backend",
                "scipy",
                "--skip-postprocess",
                "--maxiter",
                "0",
                "--output-root",
                str(output_root),
            )

            output = f"{result.stdout}\n{result.stderr}"
            assert result.returncode == 0, output
            results_payload = _load_stage2_results_json(output_root)

            export_json = Path(temp_dir) / "probe.json"
            override_json = Path(temp_dir) / "override_dofs.json"
            override_json.write_text(
                json.dumps(results_payload["FINAL_DOFS"]),
                encoding="utf-8",
            )
            probe_result = _run_stage2_script(
                *REDUCED_STAGE2_ARGS,
                "--optimizer-backend",
                "scipy",
                "--probe-only",
                "--skip-postprocess",
                "--override-dofs-json",
                str(override_json),
                "--export-objective-json",
                str(export_json),
                "--output-root",
                str(output_root),
            )

            probe_output = f"{probe_result.stdout}\n{probe_result.stderr}"
            assert probe_result.returncode == 0, probe_output
            probe_payload = json.loads(export_json.read_text(encoding="utf-8"))

        np.testing.assert_allclose(
            probe_payload["composite"]["J"],
            results_payload["FINAL_OBJECTIVE"],
            rtol=1e-12,
            atol=1e-18,
        )
        np.testing.assert_allclose(
            probe_payload["composite"]["mean_abs_relBfinal_norm"],
            results_payload["FINAL_MEAN_ABS_RELBN"],
            rtol=1e-12,
            atol=1e-18,
        )


class TestStage2OptimizerContract:
    def test_parse_args_defaults_jax_backend_to_ondevice_optimizer_lane(
        self, monkeypatch
    ):
        stage2_script = _load_stage2_script_module()
        monkeypatch.delenv("SIMSOPT_BACKEND", raising=False)
        monkeypatch.delenv("STAGE2_BACKEND", raising=False)
        monkeypatch.delenv("STAGE2_OPTIMIZER_BACKEND", raising=False)
        monkeypatch.delenv("OPTIMIZER_BACKEND", raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["banana_coil_solver.py", "--backend", "jax"],
        )

        args = stage2_script.parse_args()

        assert args.backend == "jax"
        assert args.optimizer_backend == "ondevice"
        assert args.least_squares_algorithm == "lm"

    def test_parse_args_preserves_cpu_default_reference_lane(self, monkeypatch):
        stage2_script = _load_stage2_script_module()
        monkeypatch.delenv("SIMSOPT_BACKEND", raising=False)
        monkeypatch.delenv("STAGE2_BACKEND", raising=False)
        monkeypatch.delenv("STAGE2_OPTIMIZER_BACKEND", raising=False)
        monkeypatch.delenv("OPTIMIZER_BACKEND", raising=False)
        monkeypatch.setattr(sys, "argv", ["banana_coil_solver.py"])

        args = stage2_script.parse_args()

        assert args.backend == "cpu"
        assert args.optimizer_backend == "scipy"
        assert args.least_squares_algorithm == "quasi-newton"

    def test_parse_args_accepts_least_squares_algorithm_override(self, monkeypatch):
        stage2_script = _load_stage2_script_module()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--backend",
                "jax",
                "--least-squares-algorithm",
                "quasi-newton",
            ],
        )

        args = stage2_script.parse_args()

        assert args.backend == "jax"
        assert args.optimizer_backend == "ondevice"
        assert args.least_squares_algorithm == "quasi-newton"

    def test_stage2_hardware_constraints_fail_on_self_intersection(self):
        stage2_script = _load_stage2_script_module()

        status = stage2_script.evaluate_stage2_hardware_constraints(
            1.7,
            1.75,
            0.06,
            0.05,
            39.0,
            40.0,
            self_intersecting=True,
        )

        assert status["success"] is False
        assert status["self_intersecting"] is True
        assert "banana_curve is self-intersecting" in status["violations"]

    @pytest.mark.parametrize(
        (
            "field_backend",
            "optimizer_backend",
            "least_squares_algorithm",
            "expected_method",
        ),
        [
            ("cpu", "scipy", "quasi-newton", "lbfgs"),
            ("jax", "scipy", "quasi-newton", "lbfgs"),
            ("jax", "ondevice", "quasi-newton", "lbfgs-ondevice"),
            ("jax", "ondevice", "lm", "lm-ondevice"),
        ],
    )
    def test_resolve_stage2_optimizer_method_contract(
        self,
        field_backend,
        optimizer_backend,
        least_squares_algorithm,
        expected_method,
    ):
        stage2_script = _load_stage2_script_module()
        assert (
            stage2_script.resolve_stage2_optimizer_method(
                field_backend,
                optimizer_backend,
                least_squares_algorithm=least_squares_algorithm,
            )
            == expected_method
        )

    def test_resolve_stage2_optimizer_method_rejects_hybrid(self):
        stage2_script = _load_stage2_script_module()
        with pytest.raises(
            ValueError,
            match="optimizer_backend='hybrid'.*not supported for the Stage 2 outer loop",
        ):
            stage2_script.resolve_stage2_optimizer_method("jax", "hybrid")

    def test_resolve_stage2_optimizer_method_rejects_reference_lm(self):
        stage2_script = _load_stage2_script_module()
        with pytest.raises(
            ValueError,
            match="Stage 2 least_squares_algorithm='lm' currently requires "
            "backend='jax' and optimizer_backend='ondevice'",
        ):
            stage2_script.resolve_stage2_optimizer_method(
                "jax",
                "scipy",
                least_squares_algorithm="lm",
            )

    @pytest.mark.parametrize(
        ("field_backend", "optimizer_backend", "least_squares_algorithm", "expected"),
        [
            ("cpu", "scipy", "quasi-newton", False),
            ("jax", "scipy", "quasi-newton", False),
            ("jax", "ondevice", "quasi-newton", True),
            ("jax", "ondevice", "lm", True),
        ],
    )
    def test_target_objective_bundle_is_built_only_for_target_lane(
        self,
        field_backend,
        optimizer_backend,
        least_squares_algorithm,
        expected,
    ):
        stage2_script = _load_stage2_script_module()
        assert (
            stage2_script.should_build_stage2_target_objective(
                field_backend,
                optimizer_backend,
                least_squares_algorithm=least_squares_algorithm,
            )
            is expected
        )

    @pytest.mark.parametrize(
        (
            "field_backend",
            "optimizer_backend",
            "least_squares_algorithm",
            "probe_only",
            "export_objective_json",
            "expects_contract",
            "expects_target_objective_lane",
            "expects_target_probe_payload",
            "expects_probe_only_target_payload",
        ),
        [
            (
                "cpu",
                "scipy",
                "quasi-newton",
                False,
                None,
                True,
                False,
                False,
                False,
            ),
            (
                "jax",
                "scipy",
                "quasi-newton",
                False,
                "probe.json",
                True,
                False,
                False,
                False,
            ),
            (
                "jax",
                "ondevice",
                "quasi-newton",
                False,
                None,
                True,
                True,
                False,
                False,
            ),
            (
                "jax",
                "ondevice",
                "lm",
                False,
                None,
                True,
                True,
                False,
                False,
            ),
            (
                "jax",
                "ondevice",
                "lm",
                False,
                "probe.json",
                True,
                True,
                True,
                False,
            ),
            (
                "cpu",
                "ondevice",
                "quasi-newton",
                True,
                "probe.json",
                False,
                False,
                True,
                True,
            ),
        ],
    )
    def test_resolve_stage2_target_lane_requirements(
        self,
        field_backend,
        optimizer_backend,
        least_squares_algorithm,
        probe_only,
        export_objective_json,
        expects_contract,
        expects_target_objective_lane,
        expects_target_probe_payload,
        expects_probe_only_target_payload,
    ):
        stage2_script = _load_stage2_script_module()

        (
            outer_contract,
            use_target_objective_lane,
            needs_target_probe_payload,
            probe_only_target_payload,
        ) = stage2_script.resolve_stage2_target_lane_requirements(
            field_backend,
            optimizer_backend,
            least_squares_algorithm=least_squares_algorithm,
            probe_only=probe_only,
            export_objective_json=export_objective_json,
        )

        assert (outer_contract is not None) is expects_contract
        assert use_target_objective_lane is expects_target_objective_lane
        assert needs_target_probe_payload is expects_target_probe_payload
        assert probe_only_target_payload is expects_probe_only_target_payload

    def test_run_stage2_optimizer_uses_shared_adapter(
        self,
        monkeypatch,
    ):
        optimizer_jax = _fresh_import("simsopt.geo.optimizer_jax")

        stage2_script = _load_stage2_script_module()

        cpu_scipy_captured = {}

        def fake_scipy_value_and_grad_adapter(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
        ):
            cpu_scipy_captured["method"] = method
            cpu_scipy_captured["tol"] = tol
            cpu_scipy_captured["maxiter"] = maxiter
            cpu_scipy_captured["options"] = dict(options)
            value, grad = fun(np.asarray(x0, dtype=float))
            return types.SimpleNamespace(
                x=np.asarray(x0, dtype=float),
                fun=float(value),
                jac=np.asarray(grad, dtype=float),
                nit=0,
                message="ok",
            )

        monkeypatch.setattr(
            optimizer_jax,
            "_scipy_minimize_value_and_grad",
            fake_scipy_value_and_grad_adapter,
        )
        cpu_contract = stage2_script.resolve_stage2_optimizer_contract("cpu", "scipy")
        cpu_result = stage2_script.run_stage2_optimizer(
            lambda x: (float(np.dot(x, x)), np.asarray(2.0 * x, dtype=float)),
            np.asarray([1.0, -2.0], dtype=float),
            contract=cpu_contract,
            maxiter=25,
            ftol=1e-15,
            gtol=1e-12,
        )

        assert cpu_scipy_captured["method"] == "lbfgs"
        assert cpu_scipy_captured["tol"] == pytest.approx(1e-12)
        assert cpu_scipy_captured["maxiter"] == 25
        assert cpu_scipy_captured["options"] == {"maxcor": 300, "ftol": 1e-15}
        assert cpu_result.message == "ok"

        ondevice_captured = {}

        def fake_lbfgs_private_value_and_grad(
            fun,
            x0,
            *,
            maxiter,
            gtol,
            maxcor,
            ftol,
            maxfun=None,
            maxgrad=None,
            maxls,
            callback=None,
            progress_callback=None,
        ):
            ondevice_captured["x0"] = np.asarray(x0, dtype=float)
            ondevice_captured["maxiter"] = maxiter
            ondevice_captured["gtol"] = gtol
            ondevice_captured["maxcor"] = maxcor
            ondevice_captured["ftol"] = ftol
            ondevice_captured["maxfun"] = maxfun
            ondevice_captured["maxgrad"] = maxgrad
            ondevice_captured["maxls"] = maxls
            ondevice_captured["callback"] = callback
            ondevice_captured["progress_callback"] = progress_callback
            value, grad = fun(np.asarray(x0, dtype=np.float64))
            return _build_fake_lbfgs_result(x0, value, grad)

        def _reject_scalar_private(*_args, **_kwargs):
            raise AssertionError(
                "Stage 2 ondevice lane should not route through scalar lbfgs_private "
                "once the fused value_and_grad contract is available."
            )

        monkeypatch.setattr(
            optimizer_jax,
            "_minimize_lbfgs_private_value_and_grad",
            fake_lbfgs_private_value_and_grad,
        )
        monkeypatch.setattr(
            optimizer_jax,
            "_minimize_lbfgs_private",
            _reject_scalar_private,
        )

        target_contract = stage2_script.resolve_stage2_optimizer_contract(
            "jax", "ondevice"
        )
        target_result = stage2_script.run_stage2_optimizer(
            lambda x: (
                jax.numpy.dot(x, x),
                jax.numpy.asarray(2.0 * x, dtype=jax.numpy.float64),
            ),
            np.asarray([1.0, -2.0], dtype=float),
            contract=target_contract,
            maxiter=12,
            ftol=1e-15,
            gtol=1e-11,
            scalar_fun=lambda x: jax.numpy.dot(x, x),
        )

        np.testing.assert_allclose(
            ondevice_captured["x0"],
            np.asarray([1.0, -2.0], dtype=float),
        )
        assert ondevice_captured["maxiter"] == 12
        assert ondevice_captured["gtol"] == pytest.approx(1e-11)
        assert ondevice_captured["maxcor"] == 300
        assert ondevice_captured["ftol"] == pytest.approx(1e-15)
        assert ondevice_captured["maxfun"] is None
        assert ondevice_captured["maxgrad"] is None
        assert ondevice_captured["maxls"] == 20
        assert ondevice_captured["callback"] is None
        assert ondevice_captured["progress_callback"] is None
        assert target_result.message == "Optimization terminated successfully."

    def test_make_fun_caches_field_diagnostics_between_stride_refreshes(
        self,
        monkeypatch,
    ):
        stage2_script = _load_stage2_script_module()
        calls = {"jf": 0, "relbn": 0, "length": 0, "distance": 0, "curvature": 0}

        class DummyJF:
            def __init__(self):
                self.x = np.zeros(2, dtype=float)

        class DummyScalar:
            def __init__(self, value, counter_key=None):
                self._value = float(value)
                self._counter_key = counter_key

            def J(self):
                if self._counter_key is not None:
                    calls[self._counter_key] += 1
                return self._value

            def dJ(self, partials=False):
                grad = np.asarray([self._value, -self._value], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        class DummyCurvature(DummyScalar):
            def __init__(self, value, counter_key=None):
                super().__init__(value, counter_key)
                self.curve = self

            def kappa(self):
                return np.asarray([self._value], dtype=float)

        class DummyDistance:
            def __init__(self, distances):
                self._distances = [float(value) for value in distances]
                self._index = 0

            def J(self):
                return 0.25

            def dJ(self, partials=False):
                grad = np.asarray([0.25, -0.25], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

            def shortest_distance(self):
                calls["distance"] += 1
                value = self._distances[min(self._index, len(self._distances) - 1)]
                self._index += 1
                return value

        class DummyFlux:
            def J(self):
                calls["jf"] += 1
                return 0.125

            def dJ(self, partials=False):
                grad = np.asarray([0.5, -0.5], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        def fake_relbn(_surf, _bs):
            calls["relbn"] += 1
            return 0.25

        monkeypatch.setattr(stage2_script, "compute_mean_abs_relbn", fake_relbn)

        expected_distances = (0.5, 0.1)
        trajectory = []
        fun = stage2_script.make_fun(
            DummyJF(),
            object(),
            object(),
            DummyFlux(),
            DummyScalar(1.25, "length"),
            DummyDistance(expected_distances),
            DummyCurvature(0.75, "curvature"),
            1.0,
            1.0,
            1.75,
            1.0,
            0.2,
            1.0,
            trajectory_sink=trajectory,
            field_diagnostic_stride=10,
        )

        first_value, _ = fun(np.asarray([1.0, -2.0], dtype=float))
        second_value, _ = fun(np.asarray([0.5, -1.0], dtype=float))

        assert first_value == pytest.approx(1.125)
        assert second_value == pytest.approx(1.125)
        assert calls["jf"] == 2
        assert calls["relbn"] == 1
        assert calls["length"] == 2
        assert calls["distance"] == 2
        assert calls["curvature"] == 2
        assert len(trajectory) == 2
        assert trajectory[0]["Jf"] == pytest.approx(0.125)
        assert trajectory[1]["Jf"] == pytest.approx(0.125)
        assert trajectory[0]["curve_length"] == pytest.approx(1.25)
        assert trajectory[1]["curve_length"] == pytest.approx(1.25)
        assert trajectory[0]["mean_abs_relBfinal_norm"] == pytest.approx(0.25)
        assert trajectory[1]["mean_abs_relBfinal_norm"] == pytest.approx(0.25)
        assert trajectory[0]["coil_coil_distance"] == pytest.approx(
            expected_distances[0]
        )
        assert trajectory[1]["coil_coil_distance"] == pytest.approx(
            expected_distances[1]
        )
        assert trajectory[0]["distance_constraint_violated"] is False
        assert trajectory[1]["distance_constraint_violated"] is True

    def test_evaluate_stage2_objective_requests_squared_flux_gradient_before_value(
        self,
        monkeypatch,
    ):
        stage2_script = _load_stage2_script_module()
        call_order = []

        class DummyFlux:
            def J(self):
                call_order.append("J")
                return 0.75

            def dJ(self, partials=False):
                call_order.append("dJ")
                grad = np.asarray([2.0, -1.0], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        class DummyScalar:
            def __init__(self, value):
                self._value = float(value)

            def J(self):
                return self._value

            def dJ(self, partials=False):
                grad = np.asarray([self._value, -self._value], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        class DummyDistance:
            def J(self):
                return 0.125

            def dJ(self, partials=False):
                grad = np.asarray([0.125, -0.125], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

            def shortest_distance(self):
                return 0.25

        monkeypatch.setattr(
            stage2_script,
            "compute_stage2_field_diagnostics",
            lambda *_args, **_kwargs: {
                "mean_abs_relBfinal_norm": 0.5,
            },
        )

        context = stage2_script.Stage2ObjectiveContext(
            object(),
            object(),
            object(),
            DummyFlux(),
            DummyScalar(2.25),
            DummyDistance(),
            DummyScalar(0.5),
            **_stage2_context_kwargs(),
        )
        snapshot, grad, diagnostics = stage2_script.evaluate_stage2_objective(
            context,
        )

        assert call_order[:2] == ["dJ", "J"]
        np.testing.assert_allclose(grad, np.asarray([3.75, -2.75], dtype=float))
        assert snapshot["J"] == pytest.approx(1.5)
        assert snapshot["Jf"] == pytest.approx(0.75)
        assert diagnostics["mean_abs_relBfinal_norm"] == pytest.approx(0.5)

    def test_evaluate_stage2_objective_marks_distance_violation_when_barrier_is_active(
        self,
        monkeypatch,
    ):
        stage2_script = _load_stage2_script_module()
        reported_distance = 0.25

        class DummyFlux:
            def J(self):
                return 0.5

            def dJ(self, partials=False):
                grad = np.asarray([0.5, -0.25], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        class DummyScalar:
            def __init__(self, value):
                self._value = float(value)

            def J(self):
                return self._value

            def dJ(self, partials=False):
                grad = np.asarray([self._value, -self._value], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        class DummyDistance:
            def J(self):
                return np.inf

            def dJ(self, partials=False):
                grad = np.asarray([np.nan, np.nan], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

            def shortest_distance(self):
                return reported_distance

        monkeypatch.setattr(
            stage2_script,
            "compute_stage2_field_diagnostics",
            lambda *_args, **_kwargs: {
                "mean_abs_relBfinal_norm": 0.125,
            },
        )

        context = stage2_script.Stage2ObjectiveContext(
            object(),
            object(),
            object(),
            DummyFlux(),
            DummyScalar(1.0),
            DummyDistance(),
            DummyScalar(0.5),
            **_stage2_context_kwargs(),
        )

        snapshot, _grad, diagnostics = stage2_script.evaluate_stage2_objective(context)

        assert diagnostics["coil_coil_distance"] == pytest.approx(reported_distance)
        assert snapshot["coil_coil_distance"] == pytest.approx(reported_distance)
        assert snapshot["distance_constraint_violated"] is True
        assert np.isposinf(snapshot["J"])
        assert np.isnan(snapshot["grad_norm"])

    def test_evaluate_stage2_objective_recomputes_distance_state_without_diagnostics(
        self,
        monkeypatch,
    ):
        stage2_script = _load_stage2_script_module()
        stale_diagnostics = {
            "mean_abs_relBfinal_norm": 0.125,
            "coil_coil_distance": 999.0,
        }

        class DummyFlux:
            def J(self):
                return 0.5

            def dJ(self, partials=False):
                grad = np.asarray([0.5, -0.25], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        class DummyScalar:
            def __init__(self, value):
                self._value = float(value)

            def J(self):
                return self._value

            def dJ(self, partials=False):
                grad = np.asarray([self._value, -self._value], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        class DummyDistance:
            def __init__(self):
                self.calls = 0

            def J(self):
                return 0.0

            def dJ(self, partials=False):
                grad = np.asarray([0.0, 0.0], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

            def shortest_distance(self):
                self.calls += 1
                return 0.2

        distance = DummyDistance()
        context = stage2_script.Stage2ObjectiveContext(
            object(),
            object(),
            object(),
            DummyFlux(),
            DummyScalar(1.0),
            distance,
            DummyScalar(0.5),
            **_stage2_context_kwargs(),
        )

        snapshot, _grad, diagnostics = stage2_script.evaluate_stage2_objective(
            context,
            diagnostics=stale_diagnostics.copy(),
            recompute_diagnostics=False,
        )

        assert distance.calls == 1
        assert diagnostics["mean_abs_relBfinal_norm"] == pytest.approx(0.125)
        assert diagnostics["coil_coil_distance"] == pytest.approx(0.2)
        assert snapshot["coil_coil_distance"] == pytest.approx(0.2)
        assert snapshot["distance_constraint_violated"] is True

    def test_profile_stage2_explicit_step_reports_component_breakdown(
        self, monkeypatch
    ):
        stage2_script = _load_stage2_script_module()
        expected_objective_term_names = {
            "squared_flux",
            "length_penalty",
            "coil_distance",
            "curvature",
        }

        class DummyJF:
            def __init__(self):
                self.x = np.zeros(2, dtype=float)

            def J(self):
                return 1.5

            def dJ(self, partials=False):
                grad = np.asarray([2.0, -1.0], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        class DummyScalar:
            def __init__(self, value):
                self._value = float(value)

            def J(self):
                return self._value

            def dJ(self, partials=False):
                grad = np.asarray([self._value, -self._value], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        class DummyDistance:
            def J(self):
                return 0.125

            def dJ(self, partials=False):
                grad = np.asarray([0.125, -0.125], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

            def shortest_distance(self):
                return 0.25

        monkeypatch.setattr(
            stage2_script, "compute_mean_abs_relbn", lambda _surf, _bs: 0.5
        )

        context = stage2_script.Stage2ObjectiveContext(
            DummyJF(),
            object(),
            object(),
            DummyScalar(0.75),
            DummyScalar(1.25),
            DummyDistance(),
            DummyScalar(0.5),
            **{
                **_stage2_context_kwargs(),
                "length_target": 1.25,
            },
        )
        payload = stage2_script.profile_stage2_explicit_step(context)

        assert payload["observed_step_total_s"] >= 0.0
        assert set(payload["objective_path_timings_s"]) == {"JF_J_s", "JF_dJ_s"}
        assert set(payload["extra_diagnostic_timings_s"]) == {
            "Jf_J_s",
            "mean_abs_relBfinal_norm_s",
            "curve_length_s",
            "coil_coil_distance_s",
            "curvature_s",
        }
        assert (
            set(payload["objective_term_value_timings_s"])
            == expected_objective_term_names
        )
        assert (
            set(payload["objective_term_gradient_timings_s"])
            == expected_objective_term_names
        )
        assert payload["extra_diagnostic_total_s"] >= 0.0
        assert payload["objective_term_value_total_s"] >= 0.0
        assert payload["objective_term_gradient_total_s"] >= 0.0
        assert payload["squared_flux_internal_timings_s"] == {}
        assert payload["squared_flux_internal_total_s"] == pytest.approx(0.0)
        assert payload["dominant_squared_flux_internal_components"] == []
        assert payload["squared_flux_field_b_vjp_component_timings_s"] == {}
        assert payload["dominant_squared_flux_field_b_vjp_components"] == []
        assert payload["dominant_squared_flux_field_b_vjp_coils"] == []
        assert payload["dominant_extra_diagnostics"]
        assert payload["dominant_objective_value_terms"]
        assert payload["dominant_objective_gradient_terms"]
        assert payload["snapshot"]["J"] == pytest.approx(1.375)
        assert payload["snapshot"]["Jf"] == pytest.approx(0.75)
        assert payload["snapshot"]["curve_length"] == pytest.approx(1.25)
        assert payload["snapshot"]["coil_coil_distance"] == pytest.approx(0.25)
        assert payload["snapshot"]["curvature"] == pytest.approx(0.5)
        assert payload["snapshot"]["grad_norm"] == pytest.approx(np.sqrt(2.0) * 1.375)

    def test_profile_stage2_explicit_step_reports_squared_flux_fallback_components(
        self,
        monkeypatch,
    ):
        stage2_script = _load_stage2_script_module()

        class DummyFallbackField:
            def B(self):
                return np.zeros((2, 3), dtype=float)

            def B_vjp(self, dJ_dB):
                raise AssertionError(
                    "profile_stage2_squared_flux_internal_components should use profile_B_vjp when available"
                )

            def profile_B_vjp(self, dJ_dB):
                assert dJ_dB.shape == (2, 3)
                return _build_fake_b_vjp_profile()

        class DummyFallbackSquaredFlux:
            def __init__(self):
                self._use_jax_native = False
                self.field = DummyFallbackField()

            def J(self):
                return 0.75

            def dJ(self, partials=False):
                grad = np.asarray([0.75, -0.75], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

            def _jit_integral(self, B):
                assert B.shape == (2, 3)
                return 0.75

            def _jit_integral_value_grad(self, B):
                assert B.shape == (2, 3)
                return 0.75, np.zeros_like(B)

        class DummyCompositeObjective:
            def __init__(self):
                self.x = np.zeros(2, dtype=float)

            def J(self):
                return 1.5

            def dJ(self, partials=False):
                grad = np.asarray([2.0, -1.0], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        class DummyScalar:
            def __init__(self, value):
                self._value = float(value)

            def J(self):
                return self._value

            def dJ(self, partials=False):
                grad = np.asarray([self._value, -self._value], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

        class DummyDistance:
            def J(self):
                return 0.125

            def dJ(self, partials=False):
                grad = np.asarray([0.125, -0.125], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

            def shortest_distance(self):
                return 0.25

        monkeypatch.setattr(
            stage2_script, "compute_mean_abs_relbn", lambda _surf, _bs: 0.5
        )

        context = stage2_script.Stage2ObjectiveContext(
            DummyCompositeObjective(),
            object(),
            object(),
            DummyFallbackSquaredFlux(),
            DummyScalar(1.25),
            DummyDistance(),
            DummyScalar(0.5),
            **{
                **_stage2_context_kwargs(),
                "length_target": 1.25,
            },
        )
        payload = stage2_script.profile_stage2_explicit_step(context)

        assert (
            set(payload["squared_flux_internal_timings_s"])
            == EXPECTED_SQUARED_FLUX_INTERNAL_TIMING_KEYS
        )
        assert payload["squared_flux_internal_total_s"] >= 0.0
        assert payload["dominant_squared_flux_internal_components"]
        assert (
            payload["dominant_squared_flux_internal_components"][0]["elapsed_s"] >= 0.0
        )
        _assert_b_vjp_profile_payload(payload)

    def test_stage2_script_probe_only_writes_step_profile(self):
        with tempfile.TemporaryDirectory(prefix="stage2-step-profile-") as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            profile_json = Path(temp_dir) / "step_profile.json"
            result = _run_stage2_script(
                *REDUCED_STAGE2_ARGS,
                "--optimizer-backend",
                "scipy",
                "--probe-only",
                "--skip-postprocess",
                "--profile-step-json",
                str(profile_json),
                "--output-root",
                str(output_root),
            )

            output = f"{result.stdout}\n{result.stderr}"
            assert result.returncode == 0, output
            payload = json.loads(profile_json.read_text(encoding="utf-8"))

        assert payload["observed_step_total_s"] >= 0.0
        assert payload["dominant_extra_diagnostics"]
        assert payload["dominant_extra_diagnostics"][0]["elapsed_s"] >= 0.0
        assert payload["dominant_objective_value_terms"]
        assert payload["dominant_objective_gradient_terms"]
        assert (
            set(payload["squared_flux_internal_timings_s"])
            == EXPECTED_SQUARED_FLUX_INTERNAL_TIMING_KEYS
        )
        assert payload["dominant_squared_flux_internal_components"]
        _assert_b_vjp_profile_payload(payload)

    def test_stage2_script_rejects_step_profile_on_target_lane(self):
        with tempfile.TemporaryDirectory(
            prefix="stage2-step-profile-invalid-"
        ) as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            profile_json = Path(temp_dir) / "step_profile.json"
            result = _run_stage2_script(
                *REDUCED_STAGE2_ARGS,
                "--optimizer-backend",
                "ondevice",
                "--skip-postprocess",
                "--profile-step-json",
                str(profile_json),
                "--output-root",
                str(output_root),
            )

        if not _assert_target_backend_runtime_gate(result):
            return
        _assert_stage2_script_failure(result, PROFILE_STEP_REFERENCE_LANE_ERROR)

    def test_stage2_script_ondevice_warm_timing_is_recorded(self):
        with tempfile.TemporaryDirectory(prefix="stage2-ondevice-timing-") as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            result = _run_stage2_script(
                *REDUCED_STAGE2_ARGS,
                "--optimizer-backend",
                "ondevice",
                "--record-warm-timings",
                "--skip-postprocess",
                "--maxiter",
                "0",
                "--output-root",
                str(output_root),
            )

            if _assert_target_backend_success(result) is None:
                return
            payload = _load_stage2_results_json(output_root)

        timings = payload["OPTIMIZER_TIMINGS"]
        assert timings["cold_run_s"] >= 0.0
        assert timings["warm_run_s"] >= 0.0
        assert timings["compile_overhead_s"] >= 0.0

    def test_stage2_script_skip_postprocess_preserves_field_error(self):
        with tempfile.TemporaryDirectory(
            prefix="stage2-ondevice-skip-"
        ) as skip_dir, tempfile.TemporaryDirectory(
            prefix="stage2-ondevice-noskip-"
        ) as full_dir:
            skip_output_root = Path(skip_dir) / "outputs"
            full_output_root = Path(full_dir) / "outputs"
            common_args = (
                *REDUCED_STAGE2_ARGS,
                "--optimizer-backend",
                "ondevice",
                "--maxiter",
                "0",
            )
            skip_result = _run_stage2_script(
                *common_args,
                "--skip-postprocess",
                "--output-root",
                str(skip_output_root),
            )
            full_result = _run_stage2_script(
                *common_args,
                "--output-root",
                str(full_output_root),
            )

            if _assert_target_backend_success(skip_result) is None:
                return
            if _assert_target_backend_success(full_result) is None:
                return
            skip_payload = _load_stage2_results_json(skip_output_root)
            full_payload = _load_stage2_results_json(full_output_root)

        assert skip_payload["FIELD_ERROR"] == pytest.approx(full_payload["FIELD_ERROR"])

    def test_stage2_script_rejects_warm_timing_on_reference_lane(self):
        with tempfile.TemporaryDirectory(
            prefix="stage2-warm-timing-invalid-"
        ) as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            result = _run_stage2_script(
                *REDUCED_STAGE2_ARGS,
                "--optimizer-backend",
                "scipy",
                "--record-warm-timings",
                "--skip-postprocess",
                "--maxiter",
                "0",
                "--output-root",
                str(output_root),
            )

        _assert_stage2_script_failure(result, WARM_TIMING_REFERENCE_LANE_ERROR)

    def test_stage2_script_rejects_warm_timing_without_optimization(self):
        with tempfile.TemporaryDirectory(
            prefix="stage2-warm-timing-init-only-"
        ) as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            result = _run_stage2_script(
                *REDUCED_STAGE2_ARGS,
                "--optimizer-backend",
                "ondevice",
                "--record-warm-timings",
                "--init-only",
                "--skip-postprocess",
                "--maxiter",
                "0",
                "--output-root",
                str(output_root),
            )

        if not _assert_target_backend_runtime_gate(result):
            return
        _assert_stage2_script_failure(result, WARM_TIMING_NO_OPTIMIZATION_ERROR)

    def test_stage2_script_rejects_warm_timing_in_probe_only_mode(self):
        with tempfile.TemporaryDirectory(
            prefix="stage2-warm-timing-probe-only-"
        ) as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            result = _run_stage2_script(
                *REDUCED_STAGE2_ARGS,
                "--optimizer-backend",
                "ondevice",
                "--record-warm-timings",
                "--probe-only",
                "--skip-postprocess",
                "--output-root",
                str(output_root),
            )

        if not _assert_target_backend_runtime_gate(result):
            return
        _assert_stage2_script_failure(result, WARM_TIMING_NO_OPTIMIZATION_ERROR)

    def test_strict_mode_allows_target_scalar_objective_evaluation(
        self,
        monkeypatch,
        request,
    ):
        _enable_strict_jax_backend(monkeypatch, request)
        objective, target_bundle = _build_stage2_target_objective_contract_case()
        dofs = jax.device_put(np.asarray(objective.x, dtype=np.float64))
        value = target_bundle.objective(dofs)
        assert target_bundle.value_and_grad is not None
        value_vg, grad_vg = target_bundle.value_and_grad(dofs)
        assert target_bundle.least_squares_residual is not None
        residual = target_bundle.least_squares_residual(dofs)
        assert target_bundle.raw_terms is not None
        raw_terms = target_bundle.raw_terms(dofs)
        grad = jax.grad(target_bundle.objective)(dofs)
        vg_jaxpr = jax.make_jaxpr(target_bundle.value_and_grad)(dofs)
        residual_jaxpr = jax.make_jaxpr(target_bundle.least_squares_residual)(dofs)

        assert np.isfinite(float(value))
        assert np.isfinite(float(value_vg))
        assert np.all(np.isfinite(np.asarray(residual, dtype=float)))
        assert np.all(np.isfinite(np.asarray(raw_terms, dtype=float)))
        assert np.all(np.isfinite(np.asarray(grad, dtype=float)))
        assert np.all(np.isfinite(np.asarray(grad_vg, dtype=float)))
        np.testing.assert_allclose(float(value_vg), float(value), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            0.5
            * float(
                np.vdot(
                    np.asarray(residual, dtype=float), np.asarray(residual, dtype=float)
                )
            ),
            float(value),
            rtol=1e-12,
            atol=1e-18,
        )
        np.testing.assert_allclose(
            np.asarray(grad_vg, dtype=float),
            np.asarray(grad, dtype=float),
            rtol=1e-15,
            atol=1e-15,
        )
        assert "pure_callback" not in str(vg_jaxpr)
        assert "pure_callback" not in str(residual_jaxpr)

    def test_target_scalar_objective_accepts_structured_optimizer_state(self):
        objective, target_bundle = _build_stage2_target_objective_contract_case()
        dofs = np.asarray(objective.x, dtype=np.float64)
        optimizer_state = stage2_target_optimizer_state_from_dofs(
            dofs,
            curve_dof_count=target_bundle.expected_dof_count - 1,
        )

        value = target_bundle.objective(optimizer_state)
        assert target_bundle.value_and_grad is not None
        value_vg, grad_vg = target_bundle.value_and_grad(optimizer_state)
        assert target_bundle.least_squares_residual is not None
        residual = target_bundle.least_squares_residual(optimizer_state)

        np.testing.assert_allclose(float(value_vg), float(value), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            0.5
            * float(
                np.vdot(
                    np.asarray(residual, dtype=float), np.asarray(residual, dtype=float)
                )
            ),
            float(value),
            rtol=1e-12,
            atol=1e-18,
        )
        np.testing.assert_allclose(
            np.asarray(stage2_target_optimizer_state_to_dofs(grad_vg), dtype=float),
            np.asarray(jax.grad(target_bundle.objective)(dofs), dtype=float),
            rtol=1e-15,
            atol=1e-15,
        )

    def test_target_scalar_objective_build_does_not_hostify_immutable_state(
        self,
        monkeypatch,
    ):
        with _forbid_stage2_host_materialization(
            monkeypatch,
            "Stage 2 target-objective build should keep immutable runtime "
            "state in JAX form rather than hostifying it.",
        ):
            objective, target_bundle = _build_stage2_target_objective_contract_case()
            dofs = jax.device_put(np.asarray(objective.x, dtype=np.float64))

            value = target_bundle.objective(dofs)
            assert target_bundle.value_and_grad is not None
            value_vg, grad_vg = target_bundle.value_and_grad(dofs)
            assert target_bundle.least_squares_residual is not None
            residual = target_bundle.least_squares_residual(dofs)

        assert np.isfinite(float(value))
        assert np.isfinite(float(value_vg))
        assert np.all(np.isfinite(np.asarray(grad_vg, dtype=float)))
        assert np.all(np.isfinite(np.asarray(residual, dtype=float)))

    def test_target_scalar_objective_does_not_reenter_host_snapshot_after_build(
        self,
        monkeypatch,
    ):
        objective, target_bundle = _build_stage2_target_objective_contract_case()
        dofs = jax.device_put(np.asarray(objective.x, dtype=np.float64))

        with _forbid_stage2_host_materialization(
            monkeypatch,
            "Stage 2 ondevice objective should not hostify immutable state "
            "inside the compiled hot path.",
        ):
            value = target_bundle.objective(dofs)
            assert target_bundle.value_and_grad is not None
            value_vg, grad_vg = target_bundle.value_and_grad(dofs)
            assert target_bundle.least_squares_residual is not None
            residual = target_bundle.least_squares_residual(dofs)
            assert target_bundle.raw_terms is not None
            raw_terms = target_bundle.raw_terms(dofs)
            grad = jax.grad(target_bundle.objective)(dofs)

        assert np.isfinite(float(value))
        assert np.isfinite(float(value_vg))
        assert np.all(np.isfinite(np.asarray(residual, dtype=float)))
        assert np.all(np.isfinite(np.asarray(raw_terms, dtype=float)))
        assert np.all(np.isfinite(np.asarray(grad, dtype=float)))
        assert np.all(np.isfinite(np.asarray(grad_vg, dtype=float)))

    def test_target_scalar_objective_matches_sharded_field_contract(
        self,
        monkeypatch,
    ):
        import simsopt.jax_core.sharding as sharding_core

        objective, target_bundle = _build_stage2_target_objective_contract_case()
        dofs = jax.device_put(np.asarray(objective.x, dtype=np.float64))
        dense_value = target_bundle.objective(dofs)
        dense_value_vg, dense_grad = target_bundle.value_and_grad(dofs)

        monkeypatch.setattr(
            sharding_core,
            "get_sharding_tuning",
            lambda mode=None: types.SimpleNamespace(
                active=True,
                strategy="hybrid",
                min_points_to_shard=1,
                min_pairwise_rows_to_shard=1,
                platform="cpu",
                mesh_axis_name="d",
            ),
        )

        sharded_value = target_bundle.objective(dofs)
        sharded_value_vg, sharded_grad = target_bundle.value_and_grad(dofs)
        summary = target_bundle.field_sharding_summary(dofs)

        np.testing.assert_allclose(
            float(sharded_value),
            float(dense_value),
            rtol=1e-12,
            atol=1e-18,
        )
        np.testing.assert_allclose(
            float(sharded_value_vg),
            float(dense_value_vg),
            rtol=1e-12,
            atol=1e-18,
        )
        np.testing.assert_allclose(
            np.asarray(sharded_grad, dtype=float),
            np.asarray(dense_grad, dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
        assert summary["kind"] in {"NamedSharding", "SingleDeviceSharding"}
        assert summary["device_count"] >= 1

    def test_target_scalar_objective_exposes_pairwise_penalty_sharding_summary(
        self,
        monkeypatch,
    ):
        import simsopt.jax_core.sharding as sharding_core

        objective, target_bundle = _build_stage2_target_objective_contract_case()
        dofs = jax.device_put(np.asarray(objective.x, dtype=np.float64))

        monkeypatch.setattr(
            sharding_core,
            "get_sharding_tuning",
            lambda mode=None: types.SimpleNamespace(
                active=True,
                strategy="hybrid",
                min_points_to_shard=1,
                min_pairwise_rows_to_shard=1,
                platform="cpu",
                mesh_axis_name="d",
            ),
        )

        assert target_bundle.pairwise_penalty_sharding_summary is not None
        summary = target_bundle.pairwise_penalty_sharding_summary(dofs)

        assert summary["dynamic_row_count"] >= 1
        assert len(summary["dynamic_vs_tf_groups"]) >= 1
        for pair_summary in [
            *summary["dynamic_vs_tf_groups"],
            summary["dynamic_self"],
        ]:
            assert pair_summary["left"]["gammas"]["kind"] in {
                "NamedSharding",
                "SingleDeviceSharding",
            }
            assert pair_summary["left"]["gammas"]["device_count"] >= 1
            assert pair_summary["right"]["gammas"]["kind"] in {
                "NamedSharding",
                "SingleDeviceSharding",
            }
            assert pair_summary["right"]["gammas"]["device_count"] >= 1

    def test_target_dynamic_curve_builder_matches_apply_coil_symmetry(self):
        _objective, _target_bundle, context = (
            _build_stage2_target_objective_contract_case(return_context=True)
        )
        banana_curve = context["banana_curve"]
        banana_coils = context["banana_coils"]
        current_dof = jnp.asarray(1.25e4, dtype=jnp.float64)
        base_gamma, base_gammadash = _stage2_contract_case_base_curve_geometry(
            banana_curve
        )
        banana_rotmats, banana_current_scales = (
            _stage2_contract_case_banana_symmetry_inputs(banana_coils)
        )

        dynamic_gammas, dynamic_gammadashs, dynamic_currents = (
            stage2_target_objective_module._build_dynamic_curve_data(
                base_gamma,
                base_gammadash,
                jnp.asarray(banana_rotmats, dtype=jnp.float64),
                jnp.asarray(banana_current_scales, dtype=jnp.float64),
                current_dof,
            )
        )

        expected_gammas = []
        expected_gammadashs = []
        expected_currents = []
        for rotmat, scale in zip(banana_rotmats, banana_current_scales):
            gamma, gammadash, current = apply_coil_symmetry(
                base_gamma,
                base_gammadash,
                current_dof,
                make_coil_symmetry_spec(rotmat=rotmat, scale=scale),
            )
            expected_gammas.append(gamma)
            expected_gammadashs.append(gammadash)
            expected_currents.append(current)

        np.testing.assert_allclose(
            np.asarray(dynamic_gammas, dtype=float),
            np.asarray(jnp.stack(expected_gammas), dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(dynamic_gammadashs, dtype=float),
            np.asarray(jnp.stack(expected_gammadashs), dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(dynamic_currents, dtype=float),
            np.asarray(jnp.stack(expected_currents), dtype=float),
            rtol=0.0,
            atol=0.0,
        )

    def test_target_curve_distance_scan_matches_legacy_nested_loops(self):
        _objective, _target_bundle, context = (
            _build_stage2_target_objective_contract_case(return_context=True)
        )
        banana_curve = context["banana_curve"]
        banana_coils = context["banana_coils"]
        tf_coils = context["tf_coils"]
        current_dof = jnp.asarray(1.25e4, dtype=jnp.float64)
        minimum_distance = jnp.asarray(0.05, dtype=jnp.float64)
        base_gamma, base_gammadash = _stage2_contract_case_base_curve_geometry(
            banana_curve
        )
        banana_rotmats, banana_current_scales = (
            _stage2_contract_case_banana_symmetry_inputs(banana_coils)
        )

        dynamic_gammas, dynamic_gammadashs, _dynamic_currents = (
            stage2_target_objective_module._build_dynamic_curve_data(
                base_gamma,
                base_gammadash,
                jnp.asarray(banana_rotmats, dtype=jnp.float64),
                jnp.asarray(banana_current_scales, dtype=jnp.float64),
                current_dof,
            )
        )

        tf_coil_spec = (
            stage2_target_objective_module.grouped_coil_set_spec_from_coil_specs(
                tuple(coil.to_spec() for coil in tf_coils)
            )
        )
        tf_curve_data = (
            stage2_target_objective_module._curve_pairs_from_grouped_coil_set_spec(
                tf_coil_spec
            )
        )
        tf_curve_groups = (
            stage2_target_objective_module._curve_groups_from_grouped_coil_set_spec(
                tf_coil_spec
            )
        )
        fixed_curve_penalty = stage2_target_objective_module._fixed_curve_penalty(
            tf_curve_data,
            float(minimum_distance),
        )

        legacy_total = stage2_target_objective_module._runtime_float64_scalar(
            fixed_curve_penalty,
            reference=minimum_distance,
        )
        dynamic_pairs = tuple(zip(dynamic_gammas, dynamic_gammadashs))
        for gamma, gammadash in dynamic_pairs:
            for tf_gamma, tf_gammadash in tf_curve_data:
                legacy_total = (
                    legacy_total
                    + stage2_target_objective_module.cc_distance_pure(
                        gamma,
                        gammadash,
                        stage2_target_objective_module._runtime_float64_array(
                            tf_gamma,
                            reference=gamma,
                        ),
                        stage2_target_objective_module._runtime_float64_array(
                            tf_gammadash,
                            reference=gammadash,
                        ),
                        minimum_distance,
                    )
                )
        for i, (gamma_i, gammadash_i) in enumerate(dynamic_pairs):
            for gamma_j, gammadash_j in dynamic_pairs[:i]:
                legacy_total = (
                    legacy_total
                    + stage2_target_objective_module.cc_distance_pure(
                        gamma_i,
                        gammadash_i,
                        gamma_j,
                        gammadash_j,
                        minimum_distance,
                    )
                )

        scan_total = stage2_target_objective_module._dynamic_curve_distance_penalty(
            dynamic_gammas,
            dynamic_gammadashs,
            stage2_target_objective_module._runtimeify_tree(tf_curve_groups),
            minimum_distance,
            float(fixed_curve_penalty),
        )

        np.testing.assert_allclose(
            float(scan_total),
            float(legacy_total),
            rtol=1e-12,
            atol=1e-18,
        )
        scan_jaxpr = jax.make_jaxpr(
            stage2_target_objective_module._dynamic_curve_distance_penalty
        )(
            dynamic_gammas,
            dynamic_gammadashs,
            stage2_target_objective_module._runtimeify_tree(tf_curve_groups),
            minimum_distance,
            float(fixed_curve_penalty),
        )
        assert "scan[" in str(scan_jaxpr)

    def test_target_curve_distance_scan_routes_pairwise_row_sharding(
        self,
        monkeypatch,
    ):
        import simsopt.jax_core.sharding as sharding_core

        _objective, _target_bundle, context = (
            _build_stage2_target_objective_contract_case(return_context=True)
        )
        banana_curve = context["banana_curve"]
        banana_coils = context["banana_coils"]
        tf_coils = context["tf_coils"]
        current_dof = jnp.asarray(1.25e4, dtype=jnp.float64)
        minimum_distance = jnp.asarray(0.05, dtype=jnp.float64)
        base_gamma, base_gammadash = _stage2_contract_case_base_curve_geometry(
            banana_curve
        )
        banana_rotmats, banana_current_scales = (
            _stage2_contract_case_banana_symmetry_inputs(banana_coils)
        )
        dynamic_gammas, dynamic_gammadashs, _dynamic_currents = (
            stage2_target_objective_module._build_dynamic_curve_data(
                base_gamma,
                base_gammadash,
                jnp.asarray(banana_rotmats, dtype=jnp.float64),
                jnp.asarray(banana_current_scales, dtype=jnp.float64),
                current_dof,
            )
        )
        tf_coil_spec = (
            stage2_target_objective_module.grouped_coil_set_spec_from_coil_specs(
                tuple(coil.to_spec() for coil in tf_coils)
            )
        )
        tf_curve_groups = (
            stage2_target_objective_module._curve_groups_from_grouped_coil_set_spec(
                tf_coil_spec
            )
        )
        fixed_curve_penalty = stage2_target_objective_module._fixed_curve_penalty(
            stage2_target_objective_module._curve_pairs_from_grouped_coil_set_spec(
                tf_coil_spec
            ),
            float(minimum_distance),
        )

        monkeypatch.setattr(
            sharding_core,
            "get_sharding_tuning",
            lambda mode=None: types.SimpleNamespace(
                active=True,
                strategy="hybrid",
                min_points_to_shard=1,
                min_pairwise_rows_to_shard=1,
                platform="cpu",
                mesh_axis_name="d",
            ),
        )

        observed_summaries: list[tuple[dict[str, object], dict[str, object]]] = []
        real_maybe_shard_pairwise_row_trees = (
            stage2_target_objective_module.maybe_shard_pairwise_row_trees
        )

        def _record_pairwise_row_sharding(left_tree, right_tree, *, mode=None):
            sharded_left, sharded_right = real_maybe_shard_pairwise_row_trees(
                left_tree,
                right_tree,
                mode=mode,
            )
            observed_summaries.append(
                (
                    sharding_core.summarize_array_sharding(sharded_left[1]),
                    sharding_core.summarize_array_sharding(sharded_right[1]),
                )
            )
            return sharded_left, sharded_right

        monkeypatch.setattr(
            stage2_target_objective_module,
            "maybe_shard_pairwise_row_trees",
            _record_pairwise_row_sharding,
        )

        scan_total = stage2_target_objective_module._dynamic_curve_distance_penalty(
            dynamic_gammas,
            dynamic_gammadashs,
            stage2_target_objective_module._runtimeify_tree(tf_curve_groups),
            minimum_distance,
            float(fixed_curve_penalty),
        )

        assert jnp.isfinite(scan_total)
        assert len(observed_summaries) == len(tf_curve_groups) + 1
        for left_summary, right_summary in observed_summaries:
            assert left_summary["kind"] in {"NamedSharding", "SingleDeviceSharding"}
            assert left_summary["device_count"] >= 1
            assert right_summary["kind"] in {"NamedSharding", "SingleDeviceSharding"}
            assert right_summary["device_count"] >= 1

    @pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
    def test_target_scalar_objective_matches_stage2_composite_contract(
        self,
        definition,
    ):
        objective, target_bundle = _build_stage2_target_objective_contract_case(
            definition
        )
        dofs = np.asarray(objective.x, dtype=float)
        value_ref = float(objective.J())
        grad_ref = np.asarray(objective.dJ(), dtype=float)
        assert target_bundle.value_and_grad is not None
        value_target, grad_target = target_bundle.value_and_grad(
            np.asarray(dofs, dtype=np.float64)
        )
        assert target_bundle.least_squares_residual is not None
        residual = np.asarray(
            target_bundle.least_squares_residual(np.asarray(dofs, dtype=np.float64)),
            dtype=float,
        )

        assert target_bundle.expected_dof_count == dofs.size
        np.testing.assert_allclose(
            float(value_target),
            value_ref,
            rtol=1e-12,
            atol=1e-18,
        )
        np.testing.assert_allclose(
            np.asarray(grad_target, dtype=float),
            grad_ref,
            rtol=1e-9,
            atol=_TARGET_OBJECTIVE_COMPOSITE_GRAD_ATOL,
        )
        np.testing.assert_allclose(
            0.5 * float(np.vdot(residual, residual)),
            float(value_target),
            rtol=1e-12,
            atol=1e-18,
        )
        assert target_bundle.raw_terms is not None
        raw_terms = np.asarray(target_bundle.raw_terms(dofs), dtype=float)
        raw_term_grads = np.asarray(
            jax.jacrev(target_bundle.raw_terms)(dofs), dtype=float
        )
        weighted_value = sum(
            term.weight * raw_terms[index]
            for index, term in enumerate(target_bundle.terms)
        )
        weighted_grad = sum(
            term.weight * raw_term_grads[index]
            for index, term in enumerate(target_bundle.terms)
        )
        np.testing.assert_allclose(
            float(weighted_value),
            float(value_target),
            rtol=1e-12,
            atol=1e-18,
        )
        np.testing.assert_allclose(
            np.asarray(weighted_grad, dtype=float),
            np.asarray(grad_target, dtype=float),
            rtol=1e-9,
            atol=_TARGET_OBJECTIVE_COMPOSITE_GRAD_ATOL,
        )

    @pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
    def test_target_scalar_objective_gradient_matches_centered_fd(
        self,
        definition,
    ):
        objective, target_bundle = _build_stage2_target_objective_contract_case(
            definition
        )
        dofs = np.asarray(objective.x, dtype=float)
        assert target_bundle.value_and_grad is not None
        _, grad_target = target_bundle.value_and_grad(dofs)
        grad_target = np.asarray(grad_target, dtype=float)
        grad_fd = _centered_fd_gradient(
            target_bundle.objective,
            dofs,
            eps=_TARGET_OBJECTIVE_FD_EPS,
        )

        np.testing.assert_allclose(
            grad_target,
            grad_fd,
            rtol=2e-5,
            atol=_TARGET_OBJECTIVE_FD_ATOL,
        )

    @pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
    def test_target_scalar_objective_gradient_satisfies_first_order_taylor_test(
        self,
        definition,
    ):
        objective, target_bundle = _build_stage2_target_objective_contract_case(
            definition
        )
        dofs = np.asarray(objective.x, dtype=float)
        assert target_bundle.value_and_grad is not None
        _, grad_target = target_bundle.value_and_grad(dofs)
        grad_target = np.asarray(grad_target, dtype=float)
        _assert_first_order_taylor_contract(
            target_bundle.objective,
            dofs,
            grad_target,
            seed=17,
        )

    def test_target_scalar_objective_bundle_no_longer_depends_on_live_banana_curve(
        self,
        monkeypatch,
    ):
        objective, target_bundle, context = (
            _build_stage2_target_objective_contract_case(return_context=True)
        )
        dofs = np.asarray(objective.x, dtype=np.float64)
        assert target_bundle.value_and_grad is not None
        baseline_value, baseline_grad = target_bundle.value_and_grad(dofs)
        baseline_value = float(baseline_value)
        baseline_grad = np.asarray(baseline_grad, dtype=float)

        def _fail(*_args, **_kwargs):
            assert False, "target objective should not touch the live banana curve"

        banana_curve = context["banana_curve"]
        monkeypatch.setattr(banana_curve, "gamma_jax", _fail)
        monkeypatch.setattr(banana_curve, "gammadash_jax", _fail)
        monkeypatch.setattr(banana_curve, "gammadashdash_jax", _fail)
        monkeypatch.setattr(banana_curve.surf, "get_dofs", _fail)

        value, grad = target_bundle.value_and_grad(dofs)
        value = float(value)
        grad = np.asarray(grad, dtype=float)

        np.testing.assert_allclose(value, baseline_value, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(grad, baseline_grad, rtol=1e-12, atol=1e-18)

    def test_target_scalar_objective_closures_do_not_capture_device_arrays(self):
        _objective, target_bundle = _build_stage2_target_objective_contract_case()

        assert not _closure_has_jax_array_leaf(target_bundle.objective)
        assert target_bundle.value_and_grad is not None
        assert not _closure_has_jax_array_leaf(target_bundle.value_and_grad)
        assert target_bundle.raw_terms is not None
        assert not _closure_has_jax_array_leaf(target_bundle.raw_terms)
        assert target_bundle.least_squares_residual is not None
        assert not _closure_has_jax_array_leaf(target_bundle.least_squares_residual)

    @pytest.mark.parametrize(
        ("field_backend", "optimizer_backend", "export_objective_json"),
        [
            ("jax", "ondevice", None),
            ("cpu", "ondevice", "dummy.json"),
        ],
    )
    def test_stage2_target_objective_dof_layout_guard_covers_active_target_lanes(
        self,
        field_backend,
        optimizer_backend,
        export_objective_json,
    ):
        stage2_script = _load_stage2_script_module()
        target_objective_bundle = types.SimpleNamespace(expected_dof_count=3)
        dofs = np.zeros(2, dtype=float)

        if field_backend == "jax":
            assert stage2_script.should_build_stage2_target_objective(
                field_backend,
                optimizer_backend,
            )
        else:
            assert export_objective_json is not None
        with pytest.raises(
            RuntimeError,
            match=stage2_script.STAGE2_TARGET_OBJECTIVE_DOF_LAYOUT_ERROR,
        ):
            stage2_script.validate_stage2_target_objective_dof_layout(
                target_objective_bundle,
                dofs,
            )

    def test_stage2_run_optimizer_prefers_fused_target_value_and_grad_on_ondevice_lane(
        self,
        monkeypatch,
    ):
        stage2_script = _load_stage2_script_module()
        contract = stage2_script.resolve_stage2_optimizer_contract("jax", "ondevice")
        dofs = np.asarray([0.25, -0.5], dtype=np.float64)
        optimizer_state = stage2_script.build_stage2_target_optimizer_state(
            types.SimpleNamespace(expected_dof_count=2),
            dofs,
        )

        calls = {}

        def fake_jax_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad,
        ):
            calls["fun"] = fun
            calls["x0"] = x0
            calls["method"] = method
            calls["tol"] = tol
            calls["maxiter"] = maxiter
            calls["options"] = dict(options)
            calls["value_and_grad"] = value_and_grad
            return types.SimpleNamespace(
                x=x0,
                nit=0,
                success=True,
                message="ok",
            )

        optimizer_jax_module = _fresh_import("simsopt.geo.optimizer_jax")
        monkeypatch.setattr(optimizer_jax_module, "jax_minimize", fake_jax_minimize)

        def target_value_and_grad(x):
            x = jax.numpy.asarray(
                stage2_script.flatten_stage2_target_optimizer_state(x),
                dtype=jax.numpy.float64,
            )
            value = jax.numpy.sum(jax.numpy.square(x))
            return value, 2.0 * x

        stage2_script.run_stage2_optimizer(
            value_and_grad_fun=target_value_and_grad,
            dofs=optimizer_state,
            contract=contract,
            maxiter=20,
            ftol=0.0,
            gtol=1e-12,
            maxcor=7,
            scalar_fun=lambda x: jax.numpy.sum(
                jax.numpy.square(
                    jax.numpy.asarray(
                        stage2_script.flatten_stage2_target_optimizer_state(x),
                        dtype=jax.numpy.float64,
                    )
                )
            ),
        )

        assert calls["fun"] is target_value_and_grad
        assert calls["method"] == "lbfgs-ondevice"
        assert calls["value_and_grad"] is True
        assert hasattr(calls["x0"], "current_dof")
        np.testing.assert_allclose(
            stage2_script.flatten_stage2_target_optimizer_state(calls["x0"]),
            dofs,
        )
        assert calls["maxiter"] == 20
        assert calls["options"]["maxcor"] == 7

    def test_stage2_run_optimizer_routes_lm_target_lane_through_jax_least_squares(
        self,
        monkeypatch,
    ):
        stage2_script = _load_stage2_script_module()
        contract = stage2_script.resolve_stage2_optimizer_contract(
            "jax",
            "ondevice",
            least_squares_algorithm="lm",
        )
        dofs = np.asarray([0.25, -0.5], dtype=np.float64)
        optimizer_state = stage2_script.build_stage2_target_optimizer_state(
            types.SimpleNamespace(expected_dof_count=2),
            dofs,
        )

        calls = {}

        def fake_jax_least_squares(
            residual_fn,
            x0,
            *,
            method,
            tol,
            maxiter,
            options=None,
            callback=None,
            progress_callback=None,
        ):
            calls["residual_fn"] = residual_fn
            calls["x0"] = x0
            calls["method"] = method
            calls["tol"] = tol
            calls["maxiter"] = maxiter
            calls["options"] = options
            calls["callback"] = callback
            calls["progress_callback"] = progress_callback
            residual = residual_fn(x0)
            value = 0.5 * jax.numpy.vdot(residual, residual).real
            grad = jax.grad(
                lambda state: 0.5
                * jax.numpy.vdot(residual_fn(state), residual_fn(state)).real
            )(x0)
            return types.SimpleNamespace(
                x=x0,
                fun=value,
                jac=grad,
                nit=0,
                success=True,
                message="ok",
            )

        optimizer_jax_module = _fresh_import("simsopt.geo.optimizer_jax")
        monkeypatch.setattr(
            optimizer_jax_module,
            "jax_least_squares",
            fake_jax_least_squares,
        )

        def target_residual(x):
            flat_x = jax.numpy.asarray(
                stage2_target_optimizer_state_to_dofs(x),
                dtype=jax.numpy.float64,
            )
            return flat_x + 1.0

        stage2_script.run_stage2_optimizer(
            dofs=optimizer_state,
            contract=contract,
            maxiter=9,
            ftol=0.0,
            gtol=1e-9,
            residual_fun=target_residual,
        )

        assert calls["residual_fn"] is target_residual
        assert calls["method"] == "lm-ondevice"
        assert calls["tol"] == pytest.approx(1e-9)
        assert calls["maxiter"] == 9
        assert calls["options"] is None
        assert calls["callback"] is None
        assert calls["progress_callback"] is None
        assert hasattr(calls["x0"], "current_dof")
        np.testing.assert_allclose(
            stage2_script.flatten_stage2_target_optimizer_state(calls["x0"]),
            dofs,
        )

    @pytest.mark.parametrize(
        ("optimizer_backend", "expected_source"),
        [
            ("scipy", "explicit-composite"),
            ("ondevice", "target-objective"),
        ],
    )
    def test_stage2_probe_payload_uses_lane_ssot_objective_source(
        self,
        monkeypatch,
        optimizer_backend,
        expected_source,
    ):
        stage2_script = _load_stage2_script_module()
        dofs = np.asarray([0.25, -0.5], dtype=float)
        explicit_snapshot = {
            "J": 11.0,
            "Jf": 0.125,
            "mean_abs_relBfinal_norm": 0.03125,
            "curve_length": 1.8,
            "coil_coil_distance": 0.12,
            "curvature": 18.0,
            "grad_norm": 9.0,
            "distance_constraint_violated": False,
        }
        explicit_grad = np.asarray([3.0, -4.0], dtype=float)

        def fake_evaluate_stage2_objective(
            _context, *, diagnostics=None, recompute_diagnostics=True
        ):
            assert diagnostics is None
            assert recompute_diagnostics is True
            return (
                dict(explicit_snapshot),
                explicit_grad.copy(),
                {
                    "mean_abs_relBfinal_norm": explicit_snapshot[
                        "mean_abs_relBfinal_norm"
                    ],
                    "coil_coil_distance": explicit_snapshot["coil_coil_distance"],
                },
            )

        monkeypatch.setattr(
            stage2_script,
            "evaluate_stage2_objective",
            fake_evaluate_stage2_objective,
        )

        fake_root = types.SimpleNamespace(x=dofs.copy())
        fake_flux = _FakeStage2SquaredFluxTerm(0.5, [0.75, -0.25])
        fake_curvature_term = types.SimpleNamespace(threshold=40.0)
        value_and_grad_calls = {"count": 0}

        def target_value_and_grad(x):
            value_and_grad_calls["count"] += 1
            return (
                jax.numpy.sum(jax.numpy.square(x + 1.0)) + 0.5 * (x[0] + 2.0 * x[1]),
                jax.numpy.asarray(
                    (2.0 * (x[0] + 1.0) + 0.5, 2.0 * (x[1] + 1.0) + 1.0),
                    dtype=jax.numpy.float64,
                ),
            )

        target_objective_bundle = types.SimpleNamespace(
            objective=lambda x: (
                jax.numpy.sum(jax.numpy.square(x + 1.0)) + 0.5 * (x[0] + 2.0 * x[1])
            ),
            value_and_grad=target_value_and_grad,
            raw_terms=lambda x: jax.numpy.asarray(
                (
                    jax.numpy.sum(jax.numpy.square(x + 1.0)),
                    x[0] + 2.0 * x[1],
                ),
                dtype=jax.numpy.float64,
            ),
            terms=(
                types.SimpleNamespace(name="squared_flux", weight=1.0),
                types.SimpleNamespace(name="curvature_penalty", weight=0.5),
            ),
            field_sharding_summary=lambda x: {
                "kind": "SingleDeviceSharding",
                "dof_count": int(np.asarray(x).size),
            },
            pairwise_penalty_sharding_summary=lambda x: {
                "dynamic_row_count": int(np.asarray(x).size),
                "dynamic_vs_tf_groups": [],
                "dynamic_self": {
                    "left": {"gammas": {"kind": "SingleDeviceSharding"}},
                    "right": {"gammas": {"kind": "SingleDeviceSharding"}},
                },
            },
        )
        expected_self_intersection = {
            "min_distance": 0.25,
            "tolerance": 0.05,
            "penalty": 0.0,
            "intersecting": False,
            "npts": 2000,
            "tolerance_factor": 0.1,
            "neighbor_skip": 3,
        }

        payload = stage2_script.build_stage2_probe_payload(
            fake_root,
            object(),
            object(),
            object(),
            fake_flux,
            object(),
            object(),
            fake_curvature_term,
            backend="jax",
            optimizer_backend=optimizer_backend,
            equilibrium_path="dummy.nc",
            nphi=31,
            ntheta=16,
            squared_flux_weight=1.0,
            length_weight=0.1,
            length_target=1.75,
            cc_weight=10.0,
            cc_threshold=0.05,
            curvature_weight=1e-4,
            self_intersection_summary=expected_self_intersection,
            target_objective_bundle=(
                target_objective_bundle if optimizer_backend == "ondevice" else None
            ),
        )

        expected_value = np.sum(np.square(dofs + 1.0)) + 0.5 * (dofs[0] + 2.0 * dofs[1])
        expected_grad = np.asarray(
            (2.0 * (dofs[0] + 1.0) + 0.5, 2.0 * (dofs[1] + 1.0) + 1.0),
            dtype=float,
        )
        expected_terms = np.asarray(
            target_objective_bundle.raw_terms(dofs), dtype=float
        )
        expected_term_grads = np.asarray(
            jax.jacrev(target_objective_bundle.raw_terms)(dofs.astype(np.float64)),
            dtype=float,
        )

        assert payload["composite"]["objective_source"] == expected_source
        assert payload["composite"]["mean_abs_relBfinal_norm"] == pytest.approx(
            explicit_snapshot["mean_abs_relBfinal_norm"]
        )
        assert payload["curvature_threshold"] == pytest.approx(40.0)
        assert payload["curvature_within_threshold"] is True
        assert payload["curvature_margin"] == pytest.approx(22.0)
        assert payload["self_intersection"] == expected_self_intersection
        if optimizer_backend == "ondevice":
            assert payload["composite"]["J"] == pytest.approx(float(expected_value))
            np.testing.assert_allclose(
                np.asarray(payload["composite"]["dJ"], dtype=float),
                np.asarray(expected_grad, dtype=float),
                rtol=0.0,
                atol=0.0,
            )
            assert value_and_grad_calls["count"] == 1
            assert payload["sharding_summaries"]["field"] == {
                "kind": "SingleDeviceSharding",
                "dof_count": 2,
            }
            assert payload["sharding_summaries"]["pairwise_penalty"] == {
                "dynamic_row_count": 2,
                "dynamic_vs_tf_groups": [],
                "dynamic_self": {
                    "left": {"gammas": {"kind": "SingleDeviceSharding"}},
                    "right": {"gammas": {"kind": "SingleDeviceSharding"}},
                },
            }
            assert payload["composite"]["terms"] == {
                "squared_flux": {
                    "weight": pytest.approx(1.0),
                    "raw_J": pytest.approx(expected_terms[0]),
                    "J": pytest.approx(expected_terms[0]),
                    "dJ": pytest.approx(expected_term_grads[0].tolist()),
                    "grad_norm": pytest.approx(
                        float(np.linalg.norm(expected_term_grads[0]))
                    ),
                },
                "curvature_penalty": {
                    "weight": pytest.approx(0.5),
                    "raw_J": pytest.approx(expected_terms[1]),
                    "J": pytest.approx(0.5 * expected_terms[1]),
                    "dJ": pytest.approx((0.5 * expected_term_grads[1]).tolist()),
                    "grad_norm": pytest.approx(
                        float(np.linalg.norm(0.5 * expected_term_grads[1]))
                    ),
                },
            }
        else:
            assert payload["composite"]["J"] == pytest.approx(explicit_snapshot["J"])
            np.testing.assert_allclose(
                np.asarray(payload["composite"]["dJ"], dtype=float),
                explicit_grad,
                rtol=0.0,
                atol=0.0,
            )
            assert "terms" not in payload["composite"]
            assert "sharding_summaries" not in payload

    def test_compute_mean_abs_relbn_explicitly_materializes_jax_field_output(
        self, monkeypatch
    ):
        stage2_script = _load_stage2_script_module()
        host_calls = {"float": 0, "array": 0}
        original_host_float = stage2_script.host_float
        original_host_array = stage2_script.host_array

        def counted_host_float(value):
            host_calls["float"] += 1
            return original_host_float(value)

        def counted_host_array(value, *, dtype=np.float64):
            host_calls["array"] += 1
            return original_host_array(value, dtype=dtype)

        monkeypatch.setattr(stage2_script, "host_float", counted_host_float)
        monkeypatch.setattr(stage2_script, "host_array", counted_host_array)

        class _Surface:
            quadpoints_theta = np.linspace(0.0, 1.0, 4, endpoint=False)
            quadpoints_phi = np.linspace(0.0, 1.0, 3, endpoint=False)

            def normal(self):
                normal = np.zeros((3, 4, 3))
                normal[..., 2] = 2.0
                return normal

            def gamma(self):
                return np.ones((3, 4, 3))

        class _Field:
            def set_points(self, points):
                self._points = np.asarray(points)

            def B(self):
                return jnp.ones((self._points.shape[0], 3), dtype=jnp.float64)

        result = stage2_script.compute_mean_abs_relbn(_Surface(), _Field())

        assert np.isfinite(result)
        assert host_calls["array"] >= 1
        assert host_calls["float"] >= 1

    @pytest.mark.parametrize(
        ("backend", "optimizer_backend", "expected_error"),
        [
            (
                "cpu",
                "ondevice",
                "CPU/reference lane only supports optimizer_backend='scipy'",
            ),
        ],
    )
    def test_stage2_script_rejects_unsupported_optimizer_backend_pairs(
        self,
        backend,
        optimizer_backend,
        expected_error,
    ):
        result = _run_stage2_script(
            "--backend",
            backend,
            "--optimizer-backend",
            optimizer_backend,
            "--skip-postprocess",
            "--nphi",
            "31",
            "--ntheta",
            "16",
            "--maxiter",
            "0",
        )

        assert result.returncode != 0
        error_text = f"{result.stdout}\n{result.stderr}"
        assert expected_error in error_text

    def test_stage2_script_target_backend_matches_runtime_contract(self):
        result = _run_stage2_script(
            "--backend",
            "jax",
            "--optimizer-backend",
            "ondevice",
            "--skip-postprocess",
            "--nphi",
            "31",
            "--ntheta",
            "16",
            "--maxiter",
            "0",
        )

        output = f"{result.stdout}\n{result.stderr}"
        if private_optimizer_runtime_is_supported(jax.__version__):
            assert result.returncode == 0, output
        else:
            assert result.returncode != 0
            assert (
                f"On-device optimizer requires JAX >= "
                f"{PRIVATE_OPTIMIZER_JAX_VERSION}" in output
            )

    def test_stage2_script_target_backend_writes_nonempty_trajectory(self):
        with tempfile.TemporaryDirectory(
            prefix="stage2-ondevice-trajectory-"
        ) as temp_dir:
            trajectory_json = Path(temp_dir) / "trajectory.json"
            result = _run_stage2_script(
                "--backend",
                "jax",
                "--optimizer-backend",
                "ondevice",
                "--skip-postprocess",
                "--nphi",
                "31",
                "--ntheta",
                "16",
                "--maxiter",
                "0",
                "--trajectory-json",
                str(trajectory_json),
            )

            if _assert_target_backend_success(result) is None:
                return
            payload = json.loads(trajectory_json.read_text(encoding="utf-8"))

        evaluations = payload["evaluations"]
        assert payload["backend"] == "jax"
        assert len(evaluations) >= 2
        assert evaluations[0]["J"] >= evaluations[-1]["J"]
