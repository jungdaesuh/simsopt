"""
Stage 2 JAX backend parity tests.

Validates:
1. SquaredFluxJAX.J() matches SquaredFlux.J() within 1e-10 relative error.
2. SquaredFluxJAX.dJ() gradient matches CPU within 1e-9 relative error.
3. Short L-BFGS-B run produces comparable field error and objective.

All tests require ``simsoptpp`` for the CPU reference.
"""

import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import jax
import pytest
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
from simsopt.geo.optimizer_jax import (
    PRIVATE_OPTIMIZER_JAX_VERSION,
    jax_minimize,
)
from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX
from simsopt.objectives.stage2_target_objective_jax import build_stage2_target_objective


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
    "curve_gamma_s",
    "curve_gammadash_s",
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


def _stage2_context_kwargs():
    return {
        "squared_flux_weight": 1.0,
        "length_weight": 1.0,
        "length_target": 1.75,
        "cc_weight": 1.0,
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


def _assert_stage2_script_failure(result, expected_error):
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert expected_error in output


def _assert_target_backend_runtime_gate(result):
    if jax.__version__ == PRIVATE_OPTIMIZER_JAX_VERSION:
        return True
    _assert_stage2_script_failure(
        result,
        f"On-device optimizer is validated on JAX {PRIVATE_OPTIMIZER_JAX_VERSION}",
    )
    return False


def _assert_target_backend_success(result):
    output = f"{result.stdout}\n{result.stderr}"
    if not _assert_target_backend_runtime_gate(result):
        return None
    assert result.returncode == 0, output
    return output


def _load_stage2_results_json(output_root):
    return json.loads(next(output_root.glob("**/results.json")).read_text(encoding="utf-8"))


def _build_fake_b_vjp_profile():
    return {
        "wall_time_s": 0.125,
        "component_timings_s": {
            "curve_gamma_s": 0.01,
            "curve_gammadash_s": 0.02,
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
    assert set(payload["squared_flux_field_b_vjp_component_timings_s"]) == EXPECTED_B_VJP_COMPONENT_TIMING_KEYS
    assert payload["dominant_squared_flux_field_b_vjp_components"]
    assert payload["dominant_squared_flux_field_b_vjp_coils"]


def _build_stage2_target_objective_contract_case():
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
    jf = SquaredFluxJAX(eval_surf, bs_jax)
    jls = CurveLength(banana_curve)
    jccdist = CurveCurveDistance([coil.curve for coil in all_coils], 0.05)
    jc = LpCurveCurvature(banana_curve, 4, 40)
    objective = (
        jf
        + 0.0005 * QuadraticPenalty(jls, 1.75, "max")
        + 100.0 * jccdist
        + 0.0001 * jc
    )

    target_bundle = build_stage2_target_objective(
        surface=eval_surf,
        tf_coils=tf_coils,
        banana_coils=banana_coils,
        banana_curve=banana_curve,
        squared_flux_weight=1.0,
        length_weight=0.0005,
        length_target=1.75,
        cc_weight=100.0,
        cc_threshold=0.05,
        curvature_weight=0.0001,
        curvature_threshold=40.0,
        curvature_p_norm=4,
    )

    return objective, target_bundle


def _centered_fd_gradient(fun, x, *, eps):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        grad[i] = (float(fun(x + step)) - float(fun(x - step))) / (2.0 * eps)
    return grad


class _DummyDerivative:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def __call__(self, _optim):
        return self._values


def _build_fake_bfgs_result(optimizer_jax, x0, value, grad):
    x0_array = np.asarray(x0, dtype=np.float64)
    grad_array = np.asarray(grad, dtype=np.float64)
    return optimizer_jax._BFGSResults(
        converged=False,
        failed=False,
        k=0,
        nfev=1,
        ngev=1,
        nhev=0,
        x_k=x0_array,
        f_k=np.float64(value),
        g_k=grad_array,
        H_k=np.eye(len(x0_array), dtype=np.float64),
        old_old_fval=np.float64(value),
        status=0,
        line_search_status=0,
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

    @pytest.mark.parametrize(
        "definition",
        [
            "quadratic flux",
            "normalized",
            "local",
        ],
    )
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

        rel_err = abs(j_jax - j_cpu) / (abs(j_cpu) + 1e-30)
        print(
            f"[{definition}] J_cpu={j_cpu:.12e}  J_jax={j_jax:.12e}  rel_err={rel_err:.2e}"
        )
        assert rel_err < 1e-10, f"Relative error {rel_err:.2e} exceeds 1e-10"

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

        rel_err = abs(j_jax - j_cpu) / (abs(j_cpu) + 1e-30)
        assert rel_err < 1e-10


# -----------------------------------------------------------------------
# Test 2: Gradient parity
# -----------------------------------------------------------------------


class TestGradientParity:
    """SquaredFluxJAX.dJ() must match SquaredFlux.dJ()."""

    @pytest.mark.parametrize(
        "definition",
        [
            "quadratic flux",
            "normalized",
            "local",
        ],
    )
    def test_gradient_parity(self, coil_surf_setup, definition):
        coils, surf, _, _ = coil_surf_setup

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
        jf_cpu = SquaredFlux(surf, bs_cpu, definition=definition)
        grad_cpu = jf_cpu.dJ()

        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax, definition=definition)
        grad_jax = jf_jax.dJ()

        np.testing.assert_allclose(
            grad_jax,
            grad_cpu,
            rtol=1e-9,
            atol=1e-15,
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

        rel_err_j = abs(j_jax - j_cpu) / (abs(j_cpu) + 1e-30)
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

        # Final objectives should be in the same regime.
        # Allow wider tolerance since optimizer trajectories may diverge slightly.
        rel_diff = abs(j_jax - j_cpu) / (abs(j_cpu) + 1e-30)
        assert rel_diff < 0.01, (
            f"Short-run final objectives differ by {rel_diff:.2%}: "
            f"CPU={j_cpu:.6e}, JAX={j_jax:.6e}"
        )


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

        for coil in coils:
            np.testing.assert_allclose(
                deriv_jax(coil),
                deriv_cpu(coil),
                rtol=1e-8,
                atol=1e-14,
                err_msg="BiotSavartJAX.B_vjp() does not match CPU",
            )

    def test_profile_b_vjp_reports_component_breakdown(self, coil_surf_setup):
        coils, surf, _, _ = coil_surf_setup
        points = surf.gamma().reshape((-1, 3))

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        profile = bs_jax.profile_B_vjp(np.asarray(bs_jax.B()))

        assert profile["wall_time_s"] >= 0.0
        assert set(profile["component_timings_s"]) == EXPECTED_B_VJP_COMPONENT_TIMING_KEYS
        assert profile["dominant_components"]
        assert profile["dominant_coils"]
        assert len(profile["per_coil_timings_s"]) == len(coils)
        component_total = sum(profile["component_timings_s"].values())
        assert component_total == pytest.approx(
            sum(entry["total_s"] for entry in profile["per_coil_timings_s"])
        )
        assert component_total >= 0.9 * profile["wall_time_s"]
        for entry in profile["per_coil_timings_s"]:
            assert set(entry["component_timings_s"]) == EXPECTED_B_VJP_COMPONENT_TIMING_KEYS
            assert entry["total_s"] >= 0.0


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

    @pytest.mark.parametrize(
        "definition",
        [
            "quadratic flux",
            "normalized",
            "local",
        ],
    )
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

        rel_err = abs(j_jax - j_cpu) / (abs(j_cpu) + 1e-30)
        assert rel_err < 1e-10, f"Relative error {rel_err:.2e} exceeds 1e-10"

    def test_gradient_parity(self, mixed_quad_setup):
        """SquaredFluxJAX.dJ() matches CPU with mixed quadrature."""
        coils, surf = mixed_quad_setup

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
        jf_cpu = SquaredFlux(surf, bs_cpu)
        grad_cpu = jf_cpu.dJ()

        bs_jax = BiotSavartJAX(coils)
        jf_jax = SquaredFluxJAX(surf, bs_jax)
        grad_jax = jf_jax.dJ()

        np.testing.assert_allclose(
            grad_jax,
            grad_cpu,
            rtol=1e-9,
            atol=1e-15,
            err_msg="Gradient mismatch with mixed quadrature",
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


# -----------------------------------------------------------------------
# Test 8: CurveCWSFourierCPP banana coil (real production curve type)
# -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def banana_coil_setup():
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
    return all_coils, eval_surf


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
    """Parity with real CurveCWSFourierCPP banana coils."""

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

        rel_err = abs(j_jax - j_cpu) / (abs(j_cpu) + 1e-30)
        assert rel_err < 1e-10, f"Relative error {rel_err:.2e} exceeds 1e-10"

    def test_gradient_parity(self, banana_coil_setup):
        """SquaredFluxJAX.dJ() matches CPU with CurveCWSFourierCPP coils."""
        coils, surf = banana_coil_setup

        bs_cpu = BiotSavart(coils)
        bs_cpu.set_points(surf.gamma().reshape((-1, 3)))
        grad_cpu = SquaredFlux(surf, bs_cpu).dJ()

        bs_jax = BiotSavartJAX(coils)
        grad_jax = SquaredFluxJAX(surf, bs_jax).dJ()

        np.testing.assert_allclose(
            grad_jax,
            grad_cpu,
            rtol=1e-9,
            atol=1e-15,
            err_msg="Gradient mismatch with CurveCWSFourierCPP banana coil",
        )


class TestCurveCWSFourierNativeFieldPath:
    def test_b_uses_native_curvecwsfourier_geometry(self, banana_coil_jax_setup, monkeypatch):
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
                AssertionError("BiotSavartJAX.B() should use CurveCWSFourier.gammadash_jax")
            ),
        )

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        B_jax = np.asarray(bs_jax.B())

        assert B_jax.shape == points.shape

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
                AssertionError("BiotSavartJAX.B_vjp() should bypass Coil.vjp() for CurveCWSFourier")
            ),
        )

        deriv = bs_jax.B_vjp(v)

        assert deriv(banana_coil.curve).shape[0] == banana_coil.curve.local_full_dof_size
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
            curve = curve_cls(np.linspace(0, 1, 128, endpoint=False), order=2, surf=coil_surf)
            curve.set("phic(0)", 0.06)
            curve.set("thetac(0)", 0.5)
            curve.set("phic(1)", 0.03)
            curve.set("thetas(1)", 0.1)
            coils = [Coil(curve_obj, current) for curve_obj, current in zip(tf_curves, tf_currents)]
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
        np.testing.assert_allclose(grad_jax, grad_cpp, rtol=1e-9, atol=1e-15)

    @pytest.mark.parametrize(
        ("backend", "expected_curve_class"),
        [
            ("cpu", "CurveCWSFourierCPP"),
            ("jax", "CurveCWSFourier"),
        ],
    )
    def test_stage2_probe_reports_backend_specific_banana_curve(
        self, backend, expected_curve_class
    ):
        stage2_script = (
            REPO_ROOT
            / "examples"
            / "single_stage_optimization"
            / "STAGE_2"
            / "banana_coil_solver.py"
        )
        with tempfile.TemporaryDirectory(prefix=f"stage2-boundary-{backend}-") as temp_dir:
            export_json = Path(temp_dir) / f"{backend}_probe.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(stage2_script),
                    "--backend",
                    backend,
                    "--probe-only",
                    "--nphi",
                    "31",
                    "--ntheta",
                    "16",
                    "--export-objective-json",
                    str(export_json),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(export_json.read_text(encoding="utf-8"))

        assert result.returncode == 0
        assert payload["banana_curve_class"] == expected_curve_class


class TestStage2OptimizerContract:
    @pytest.mark.parametrize(
        ("field_backend", "optimizer_backend", "expected_method"),
        [
            ("cpu", "scipy", "lbfgs"),
            ("jax", "scipy", "lbfgs"),
            ("jax", "ondevice", "bfgs-ondevice"),
        ],
    )
    def test_resolve_stage2_optimizer_method_contract(
        self,
        field_backend,
        optimizer_backend,
        expected_method,
    ):
        stage2_script = _load_stage2_script_module()
        assert stage2_script.resolve_stage2_optimizer_method(
            field_backend,
            optimizer_backend,
        ) == expected_method

    def test_resolve_stage2_optimizer_method_rejects_hybrid(self):
        stage2_script = _load_stage2_script_module()
        with pytest.raises(
            ValueError,
            match="optimizer_backend='hybrid'.*not supported for the Stage 2 outer loop",
        ):
            stage2_script.resolve_stage2_optimizer_method("jax", "hybrid")

    @pytest.mark.parametrize(
        ("field_backend", "optimizer_backend", "expected"),
        [
            ("cpu", "scipy", False),
            ("jax", "scipy", False),
            ("jax", "ondevice", True),
        ],
    )
    def test_target_objective_bundle_is_built_only_for_target_lane(
        self,
        field_backend,
        optimizer_backend,
        expected,
    ):
        stage2_script = _load_stage2_script_module()
        assert (
            stage2_script.should_build_stage2_target_objective(
                field_backend,
                optimizer_backend,
            )
            is expected
        )

    def test_run_stage2_optimizer_uses_shared_adapter(
        self,
        monkeypatch,
    ):
        import simsopt.geo.optimizer_jax as optimizer_jax

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
        cpu_result = stage2_script.run_stage2_optimizer(
            lambda x: (float(np.dot(x, x)), np.asarray(2.0 * x, dtype=float)),
            np.asarray([1.0, -2.0], dtype=float),
            field_backend="cpu",
            optimizer_backend="scipy",
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
        def fake_bfgs_private(
            fun,
            x0,
            *,
            maxiter,
            gtol,
            line_search_maxiter,
            initial_state=None,
        ):
            ondevice_captured["x0"] = np.asarray(x0, dtype=float)
            ondevice_captured["maxiter"] = maxiter
            ondevice_captured["gtol"] = gtol
            ondevice_captured["line_search_maxiter"] = line_search_maxiter
            ondevice_captured["initial_state"] = initial_state
            value = float(fun(np.asarray(x0, dtype=float)))
            grad = np.asarray(jax.grad(fun)(np.asarray(x0, dtype=np.float64)), dtype=float)
            return _build_fake_bfgs_result(optimizer_jax, x0, value, grad)

        monkeypatch.setattr(
            optimizer_jax,
            "_minimize_bfgs_private",
            fake_bfgs_private,
        )

        target_result = stage2_script.run_stage2_optimizer(
            lambda x: (float(np.dot(x, x)), np.asarray(2.0 * x, dtype=float)),
            np.asarray([1.0, -2.0], dtype=float),
            field_backend="jax",
            optimizer_backend="ondevice",
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
        assert ondevice_captured["line_search_maxiter"] == 10
        assert ondevice_captured["initial_state"] is None
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

        class DummyDistance:
            def J(self):
                return 0.25

            def dJ(self, partials=False):
                grad = np.asarray([0.25, -0.25], dtype=float)
                if partials:
                    return _DummyDerivative(grad)
                return grad

            def shortest_distance(self):
                calls["distance"] += 1
                return 0.5

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

        trajectory = []
        fun = stage2_script.make_fun(
            DummyJF(),
            object(),
            object(),
            DummyFlux(),
            DummyScalar(1.25, "length"),
            DummyDistance(),
            DummyScalar(0.75, "curvature"),
            1.0,
            1.0,
            1.75,
            1.0,
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
        assert calls["distance"] == 1
        assert calls["curvature"] == 2
        assert len(trajectory) == 2
        assert trajectory[0]["Jf"] == pytest.approx(0.125)
        assert trajectory[1]["Jf"] == pytest.approx(0.125)
        assert trajectory[0]["curve_length"] == pytest.approx(1.25)
        assert trajectory[1]["curve_length"] == pytest.approx(1.25)

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

    def test_profile_stage2_explicit_step_reports_component_breakdown(self, monkeypatch):
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

        monkeypatch.setattr(stage2_script, "compute_mean_abs_relbn", lambda _surf, _bs: 0.5)

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
        assert set(payload["objective_term_value_timings_s"]) == expected_objective_term_names
        assert set(payload["objective_term_gradient_timings_s"]) == expected_objective_term_names
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
                raise AssertionError("profile_stage2_squared_flux_internal_components should use profile_B_vjp when available")

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

        monkeypatch.setattr(stage2_script, "compute_mean_abs_relbn", lambda _surf, _bs: 0.5)

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

        assert set(payload["squared_flux_internal_timings_s"]) == EXPECTED_SQUARED_FLUX_INTERNAL_TIMING_KEYS
        assert payload["squared_flux_internal_total_s"] >= 0.0
        assert payload["dominant_squared_flux_internal_components"]
        assert payload["dominant_squared_flux_internal_components"][0]["elapsed_s"] >= 0.0
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
        assert set(payload["squared_flux_internal_timings_s"]) == EXPECTED_SQUARED_FLUX_INTERNAL_TIMING_KEYS
        assert payload["dominant_squared_flux_internal_components"]
        _assert_b_vjp_profile_payload(payload)

    def test_stage2_script_rejects_step_profile_on_target_lane(self):
        with tempfile.TemporaryDirectory(prefix="stage2-step-profile-invalid-") as temp_dir:
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
        with tempfile.TemporaryDirectory(prefix="stage2-ondevice-skip-") as skip_dir, tempfile.TemporaryDirectory(
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
        with tempfile.TemporaryDirectory(prefix="stage2-warm-timing-invalid-") as temp_dir:
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
        with tempfile.TemporaryDirectory(prefix="stage2-warm-timing-init-only-") as temp_dir:
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
        with tempfile.TemporaryDirectory(prefix="stage2-warm-timing-probe-only-") as temp_dir:
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

    def test_target_scalar_objective_matches_stage2_composite_contract(self):
        objective, target_bundle = _build_stage2_target_objective_contract_case()
        dofs = np.asarray(objective.x, dtype=float)
        value_ref = float(objective.J())
        grad_ref = np.asarray(objective.dJ(), dtype=float)
        value_target, grad_target = jax.value_and_grad(target_bundle.objective)(
            np.asarray(dofs, dtype=np.float64)
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
            atol=1e-15,
        )

    def test_target_scalar_objective_gradient_matches_centered_fd(self):
        objective, target_bundle = _build_stage2_target_objective_contract_case()
        dofs = np.asarray(objective.x, dtype=float)
        grad_target = np.asarray(jax.grad(target_bundle.objective)(dofs), dtype=float)
        grad_fd = _centered_fd_gradient(target_bundle.objective, dofs, eps=1e-7)

        np.testing.assert_allclose(
            grad_target,
            grad_fd,
            rtol=2e-5,
            atol=5e-7,
        )

    @pytest.mark.parametrize(
        ("backend", "optimizer_backend", "expected_error"),
        [
            ("cpu", "ondevice", "CPU/reference lane only supports optimizer_backend='scipy'"),
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
        if jax.__version__ == PRIVATE_OPTIMIZER_JAX_VERSION:
            assert result.returncode == 0, output
        else:
            assert result.returncode != 0
            assert (
                f"On-device optimizer is validated on JAX "
                f"{PRIVATE_OPTIMIZER_JAX_VERSION}" in output
            )

    def test_stage2_script_target_backend_writes_nonempty_trajectory(self):
        with tempfile.TemporaryDirectory(prefix="stage2-ondevice-trajectory-") as temp_dir:
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
