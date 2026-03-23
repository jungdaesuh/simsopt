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
from simsopt.geo.optimizer_jax import jax_minimize, resolve_optimizer_backend_method
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
            ("jax", "ondevice", "lbfgs-ondevice"),
        ],
    )
    def test_resolve_stage2_optimizer_method_contract(
        self,
        field_backend,
        optimizer_backend,
        expected_method,
    ):
        if field_backend != "jax" and optimizer_backend != "scipy":
            pytest.skip("Shared optimizer backend mapping is only meaningful on the JAX lane.")
        assert resolve_optimizer_backend_method(
            optimizer_backend,
            limited_memory=True,
        ) == expected_method

    def test_resolve_stage2_optimizer_method_rejects_hybrid(self):
        with pytest.raises(
            ValueError,
            match="optimizer_backend='hybrid'.*limited_memory=True",
        ):
            resolve_optimizer_backend_method("hybrid", limited_memory=True)

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

        captured = {}

        def fake_scipy_adapter(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
        ):
            captured["method"] = method
            captured["tol"] = tol
            captured["maxiter"] = maxiter
            captured["options"] = dict(options)
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
            fake_scipy_adapter,
        )
        def toy_fun(x):
            return float(np.dot(x, x)), np.asarray(2.0 * x, dtype=float)

        result = optimizer_jax.jax_minimize(
            toy_fun,
            np.asarray([1.0, -2.0], dtype=float),
            method="lbfgs",
            tol=1e-12,
            maxiter=25,
            options={"maxcor": 123, "ftol": 1e-15},
            value_and_grad=True,
        )

        assert captured["method"] == "lbfgs"
        assert captured["tol"] == pytest.approx(1e-12)
        assert captured["maxiter"] == 25
        assert captured["options"] == {"maxcor": 123, "ftol": 1e-15}
        assert result.message == "ok"

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
        if jax.__version__ == "0.6.2":
            assert result.returncode == 0, output
        else:
            assert result.returncode != 0
            assert "On-device optimizer is pinned to JAX 0.6.2" in output
