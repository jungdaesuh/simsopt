"""
Stage 2 JAX backend parity tests.

Validates:
1. SquaredFluxJAX.J() matches SquaredFlux.J() within 1e-10 relative error.
2. SquaredFluxJAX.dJ() gradient matches CPU within 1e-9 relative error.
3. Short L-BFGS-B run produces comparable field error and objective.

All tests require ``simsoptpp`` for the CPU reference.
"""

import pytest
import numpy as np

sopp = pytest.importorskip(
    "simsoptpp",
    reason="Stage 2 integration tests require simsoptpp (use candidate-fixed env)",
)

from scipy.optimize import minimize  # noqa: E402

from simsopt.field import (  # noqa: E402
    BiotSavart,
    Current,
    Coil,
    coils_via_symmetries,
)
from simsopt.geo import (  # noqa: E402
    SurfaceRZFourier,
    CurveXYZFourier,
    create_equally_spaced_curves,
    CurveLength,
)
from simsopt.objectives import SquaredFlux, QuadraticPenalty  # noqa: E402

from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX


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

            res = minimize(
                fun, dofs, jac=True, method="L-BFGS-B", options={"maxiter": MAXITER}
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

        deriv_cpu = bs_cpu.B_vjp(v)
        grad_cpu = deriv_cpu(coils[0])

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points)
        deriv_jax = bs_jax.B_vjp(v)
        grad_jax = deriv_jax(coils[0])

        np.testing.assert_allclose(
            grad_jax,
            grad_cpu,
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
