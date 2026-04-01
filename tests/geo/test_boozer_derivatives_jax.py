"""
FD validation tests for M3 JAX Boozer derivative path.

Tests:
1. Surface coefficient Jacobians (dgamma_by_dcoeff, etc.) via FD.
2. Composed penalty gradient via FD.
3. Composed residual Jacobian via FD.
4. Outer coil VJP consistency.
5. Hessian symmetry and FD validation.
"""

import pytest
import numpy as np

import jax
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from simsopt.geo.surface_fourier_jax import (
    surface_gamma_from_dofs,
    surface_gammadash1_from_dofs,
    surface_gammadash2_from_dofs,
    dgamma_by_dcoeff,
    dgammadash1_by_dcoeff,
    dgammadash2_by_dcoeff,
    stellsym_scatter_indices,
)
from simsopt.field.biotsavart_jax import biot_savart_B
from simsopt.geo.boozer_residual_jax import (
    _boozer_residual_vector_composed,
    boozer_penalty_composed,
    boozer_penalty_grad_composed,
    boozer_residual_coil_vjp,
    boozer_residual_jacobian_composed,
    boozer_residual_vector,
)


_FIRST_ORDER_EPSILONS = np.power(2.0, -np.arange(2, 15))
_SECOND_ORDER_EPSILONS = np.power(2.0, -np.arange(2, 12))


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_torus_dofs(mpol=1, ntor=0, nfp=1, R=1.0, r=0.1, stellsym=False):
    """Create DOF vector for a simple circular torus (non-stellsym)."""
    n_per = (2 * mpol + 1) * (2 * ntor + 1)
    dofs = np.zeros(3 * n_per)
    ncol = 2 * ntor + 1
    # xc[0,0] = R (constant term)
    dofs[0 * ncol + 0] = R
    # xc[1,0] = r (cos theta term)
    dofs[1 * ncol + 0] = r
    # zc[mpol+1, 0] = r (sin theta term)
    dofs[2 * n_per + (mpol + 1) * ncol + 0] = r
    return jnp.array(dofs)


def _make_torus_dofs_stellsym(mpol=1, ntor=0, nfp=1, R=1.0, r=0.1):
    """Create DOF vector for a simple circular torus (stellsym)."""
    scatter_idx = stellsym_scatter_indices(mpol, ntor)
    n_per = (2 * mpol + 1) * (2 * ntor + 1)

    # Build full coefficient arrays
    full = np.zeros(3 * n_per)
    ncol = 2 * ntor + 1
    full[0 * ncol + 0] = R  # xc[0,0]
    full[1 * ncol + 0] = r  # xc[1,0]
    full[2 * n_per + (mpol + 1) * ncol + 0] = r  # zc[mpol+1,0]

    # Extract only the free DOFs
    sdofs = full[scatter_idx]
    return jnp.array(sdofs), jnp.array(scatter_idx)


def _make_coil_data(ncoils=3, nquad=32):
    """Create synthetic coil data for a simple coilset."""
    R_coil = 1.5
    gammas = np.zeros((ncoils, nquad, 3))
    gammadashs = np.zeros((ncoils, nquad, 3))

    for i in range(ncoils):
        phi_offset = 2 * np.pi * i / ncoils
        t = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
        gammas[i, :, 0] = R_coil * np.cos(t + phi_offset)
        gammas[i, :, 1] = R_coil * np.sin(t + phi_offset)
        gammas[i, :, 2] = 0.0
        gammadashs[i, :, 0] = -R_coil * np.sin(t + phi_offset) * 2 * np.pi
        gammadashs[i, :, 1] = R_coil * np.cos(t + phi_offset) * 2 * np.pi
        gammadashs[i, :, 2] = 0.0

    currents = np.array([1e5, 1e5, 1e5])
    return (
        jnp.array(gammas),
        jnp.array(gammadashs),
        jnp.array(currents),
    )


def _make_unit_direction(seed, shape, dtype):
    """Return a deterministic unit-norm tangent direction."""
    tangent = jax.random.normal(jax.random.key(seed), shape, dtype=dtype)
    return tangent / jnp.linalg.norm(tangent)


def _make_adjoint(seed, size, dtype):
    """Return a deterministic adjoint vector."""
    return jax.random.normal(jax.random.key(seed), (size,), dtype=dtype)


def _paired_seeds(seed):
    """Return two deterministic seeds for matrix-valued directional contracts."""
    return (seed, seed + 1000)


def _l2_norm(value):
    """Return the Euclidean norm of an array-like value as a Python float."""
    return float(jnp.linalg.norm(jnp.ravel(jnp.asarray(value))))


def _directional_jacobian_action(jacobian, direction):
    """Apply an output[..., ndofs] Jacobian to a parameter-space direction."""
    return jnp.tensordot(jnp.asarray(jacobian), direction, axes=[-1, 0])


def _assert_first_order_linearization(
    f,
    x,
    direction,
    linear_term,
    *,
    epsilons=None,
    best_defect_tol=1e-6,
):
    """Assert that f(x + eps h) matches the first-order linearization on a ladder."""
    if epsilons is None:
        epsilons = _FIRST_ORDER_EPSILONS

    base = f(x)
    linear_scale = max(_l2_norm(linear_term), 1.0)
    defects = []
    for eps in epsilons:
        actual = f(x + eps * direction)
        approx = base + eps * linear_term
        defect = _l2_norm(actual - approx) / (float(eps) * linear_scale)
        defects.append(defect)

    best_defect = min(defects)
    assert best_defect <= best_defect_tol, (
        "First-order linearization never reached the required defect floor: "
        f"best={best_defect:.2e}, tol={best_defect_tol:.2e}"
    )


def _assert_directional_jacobian_contract(
    f,
    x,
    jacobian,
    *,
    seeds=None,
    epsilons=None,
    linearization_tol=1e-6,
    fd_tol=1e-6,
):
    """Assert a Jacobian via first-order linearization and best central FD."""
    if epsilons is None:
        epsilons = _FIRST_ORDER_EPSILONS

    if seeds is None:
        raise ValueError("Provide seeds for directional Jacobian validation.")

    for direction_seed in seeds:
        direction = _make_unit_direction(direction_seed, x.shape, x.dtype)
        linear_term = _directional_jacobian_action(jacobian, direction)

        _assert_first_order_linearization(
            f,
            x,
            direction,
            linear_term,
            epsilons=epsilons,
            best_defect_tol=linearization_tol,
        )

        fd_scale = max(_l2_norm(linear_term), 1.0)
        fd_errors = []
        for eps in epsilons:
            fd_estimate = (f(x + eps * direction) - f(x - eps * direction)) / (2 * eps)
            fd_errors.append(_l2_norm(fd_estimate - linear_term) / fd_scale)

        best_fd_error = min(fd_errors)
        assert best_fd_error <= fd_tol, (
            "Directional central FD never matched the Jacobian action tightly enough: "
            f"seed={direction_seed}, best={best_fd_error:.2e}, tol={fd_tol:.2e}"
        )


def _assert_scalar_directional_consistency(
    f,
    x,
    grad,
    *,
    seed,
    rtol=1e-10,
    atol=1e-12,
):
    """Check reverse-mode gradient against a forward-mode directional derivative."""
    tangent = _make_unit_direction(seed, x.shape, x.dtype)
    _, directional_jvp = jax.jvp(f, (x,), (tangent,))
    directional_grad = jnp.vdot(grad, tangent)
    np.testing.assert_allclose(
        np.asarray(directional_grad),
        np.asarray(directional_jvp),
        rtol=rtol,
        atol=atol,
    )


def _assert_scalar_grad_matches(
    f,
    x,
    grad,
    *,
    seed,
    rtol=1e-10,
    atol=1e-12,
):
    """Check a provided reverse-mode gradient against the scalar objective."""
    expected_grad = jax.grad(f)(x)
    np.testing.assert_allclose(
        np.asarray(grad),
        np.asarray(expected_grad),
        rtol=rtol,
        atol=atol,
    )
    _assert_scalar_directional_consistency(
        f,
        x,
        grad,
        seed=seed,
        rtol=rtol,
        atol=atol,
    )


def _assert_scalar_grad_contract(f, x, *, atol, rtol, eps=None):
    """Run JAX's built-in gradient checker for a scalar objective."""
    check_grads(
        f,
        (x,),
        order=1,
        modes=("fwd", "rev"),
        atol=atol,
        rtol=rtol,
        eps=eps,
    )


def _assert_composed_penalty_gradient_contract(
    x,
    kwargs,
    *,
    seed,
    atol=1e-12,
    rtol=1e-10,
    check_grads_atol=1e-6,
    check_grads_rtol=1e-5,
    check_grads_eps=1e-6,
):
    """Check the composed scalar objective gradient with JAX-native contracts."""

    def objective(arg):
        return boozer_penalty_composed(arg, **kwargs)

    _, grad = boozer_penalty_grad_composed(x, **kwargs)
    _assert_scalar_grad_matches(
        objective,
        x,
        grad,
        seed=seed,
        rtol=rtol,
        atol=atol,
    )
    _assert_scalar_grad_contract(
        objective,
        x,
        atol=check_grads_atol,
        rtol=check_grads_rtol,
        eps=check_grads_eps,
    )


def _assert_composed_residual_jacobian_contract(
    x,
    kwargs,
    *,
    seed,
    linearization_tol=2e-4,
    fd_tol=1e-4,
):
    """Check the composed residual Jacobian with the shared directional contract."""

    def residual(arg):
        return _boozer_residual_vector_composed(arg, **kwargs)

    _, jacobian = boozer_residual_jacobian_composed(x, **kwargs)
    _assert_directional_jacobian_contract(
        residual,
        x,
        jacobian,
        seeds=_paired_seeds(seed),
        linearization_tol=linearization_tol,
        fd_tol=fd_tol,
    )


def _assert_second_order_taylor_contract(
    f,
    x,
    grad,
    hessian,
    *,
    seed,
    epsilons=None,
    best_defect_tol=1e-8,
    min_observed_order=2.0,
):
    """Assert that a scalar objective exhibits a stable second-order Taylor regime."""
    if epsilons is None:
        epsilons = _SECOND_ORDER_EPSILONS

    direction = _make_unit_direction(seed, x.shape, x.dtype)
    value0 = float(f(x))
    linear_term = float(jnp.vdot(grad, direction))
    quadratic_term = float(direction @ hessian @ direction)

    errors = []
    defects = []
    for eps in epsilons:
        actual = float(f(x + eps * direction))
        approx = value0 + float(eps) * linear_term + 0.5 * float(eps) ** 2 * quadratic_term
        error = abs(actual - approx)
        scale = max(
            abs(float(eps) * linear_term),
            abs(0.5 * float(eps) ** 2 * quadratic_term),
            1.0,
        )
        errors.append(error)
        defects.append(error / scale)

    orders = []
    for prev_eps, next_eps, prev_err, next_err in zip(
        epsilons[:-1],
        epsilons[1:],
        errors[:-1],
        errors[1:],
    ):
        if prev_err > 0.0 and next_err > 0.0 and next_err < prev_err:
            orders.append(np.log(prev_err / next_err) / np.log(float(prev_eps / next_eps)))

    best_defect = min(defects)
    assert best_defect <= best_defect_tol, (
        "Second-order Taylor remainder never reached the required defect floor: "
        f"best={best_defect:.2e}, tol={best_defect_tol:.2e}"
    )
    assert orders and max(orders) >= min_observed_order, (
        "Second-order Taylor regime was not observed on the epsilon ladder: "
        f"best_order={max(orders) if orders else float('nan'):.2f}, "
        f"required>={min_observed_order:.2f}"
    )


def _coil_residual_scalar_objective(
    *,
    gamma,
    xphi,
    xtheta,
    coil_inputs,
    iota,
    G,
    adjoint,
    grad_idx,
):
    """Build adjoint @ residual as a scalar objective of one coil input block."""
    nphi, ntheta = gamma.shape[:2]

    def objective(x):
        points = gamma.reshape(-1, 3)
        args = list(coil_inputs)
        args[grad_idx] = x
        B = biot_savart_B(points, *args).reshape(nphi, ntheta, 3)
        residual = boozer_residual_vector(
            G,
            iota,
            B,
            xphi,
            xtheta,
            weight_inv_modB=False,
        )
        return jnp.vdot(adjoint, residual)

    return objective


def _assert_surface_jacobian_contract(
    surface_fn,
    dofs,
    jacobian,
    *,
    seed,
    linearization_tol=1e-8,
    fd_tol=1e-7,
):
    """Check one surface map Jacobian against the shared directional contract."""
    _assert_directional_jacobian_contract(
        surface_fn,
        dofs,
        jacobian,
        seeds=_paired_seeds(seed),
        linearization_tol=linearization_tol,
        fd_tol=fd_tol,
    )


# ---------------------------------------------------------------------------
# Test: Surface coefficient Jacobians
# ---------------------------------------------------------------------------


class TestDgammaByDcoeff:
    """Validate dgamma_by_dcoeff via directional finite-difference contracts."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 8, 8

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)

    def test_dgamma_shape(self):
        J = dgamma_by_dcoeff(
            self.dofs,
            self.phis,
            self.thetas,
            self.mpol,
            self.ntor,
            self.nfp,
            stellsym=False,
        )
        ndofs = len(self.dofs)
        assert J.shape == (self.nphi, self.ntheta, 3, ndofs)

    def test_dgamma_fd(self):
        """dgamma_by_dcoeff satisfies a directional Jacobian contract."""
        args = (self.phis, self.thetas, self.mpol, self.ntor, self.nfp, False)
        J = dgamma_by_dcoeff(self.dofs, *args)

        def gamma_fn(dofs):
            return surface_gamma_from_dofs(dofs, *args)

        _assert_surface_jacobian_contract(
            gamma_fn,
            self.dofs,
            J,
            seed=11,
        )

    def test_dgammadash1_fd(self):
        """dgammadash1_by_dcoeff satisfies a directional Jacobian contract."""
        args = (self.phis, self.thetas, self.mpol, self.ntor, self.nfp, False)
        J = dgammadash1_by_dcoeff(self.dofs, *args)

        def gammadash1_fn(dofs):
            return surface_gammadash1_from_dofs(dofs, *args)

        _assert_surface_jacobian_contract(
            gammadash1_fn,
            self.dofs,
            J,
            seed=12,
        )

    def test_dgammadash2_fd(self):
        """dgammadash2_by_dcoeff satisfies a directional Jacobian contract."""
        args = (self.phis, self.thetas, self.mpol, self.ntor, self.nfp, False)
        J = dgammadash2_by_dcoeff(self.dofs, *args)

        def gammadash2_fn(dofs):
            return surface_gammadash2_from_dofs(dofs, *args)

        _assert_surface_jacobian_contract(
            gammadash2_fn,
            self.dofs,
            J,
            seed=13,
        )


class TestDgammaByDcoeffStellsym:
    """Validate surface coefficient Jacobians with stellarator symmetry."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 8, 8

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs, self.scatter_idx = _make_torus_dofs_stellsym(
            self.mpol,
            self.ntor,
            self.nfp,
        )

    def test_dgamma_stellsym_fd(self):
        """dgamma_by_dcoeff with stellsym satisfies a directional Jacobian contract."""
        args = (self.phis, self.thetas, self.mpol, self.ntor, self.nfp, True)

        def gamma_fn(dofs):
            return surface_gamma_from_dofs(
                dofs,
                *args,
                scatter_indices=self.scatter_idx,
            )

        _assert_surface_jacobian_contract(
            gamma_fn,
            self.dofs,
            dgamma_by_dcoeff(
                self.dofs,
                *args,
                scatter_indices=self.scatter_idx,
            ),
            seed=14,
        )

    def test_dgamma_stellsym_fewer_dofs(self):
        """Stellsym Jacobian has fewer DOF columns than non-stellsym."""
        args = (self.phis, self.thetas, self.mpol, self.ntor, self.nfp)
        J_stellsym = dgamma_by_dcoeff(
            self.dofs,
            *args,
            stellsym=True,
            scatter_indices=self.scatter_idx,
        )
        ndofs_stellsym = J_stellsym.shape[-1]

        n_per = (2 * self.mpol + 1) * (2 * self.ntor + 1)
        ndofs_full = 3 * n_per
        assert ndofs_stellsym < ndofs_full


# ---------------------------------------------------------------------------
# Test: Composed penalty gradient
# ---------------------------------------------------------------------------


class TestBoozerPenaltyGradComposed:
    """Validate the full-pipeline VJP gradient via finite differences."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 6, 6

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0
        # Decision vector: [sdofs, iota, G]
        self.x = jnp.concatenate([self.dofs, jnp.array([self.iota, self.G])])
        self.kwargs = dict(
            coil_arrays=self.coil_arrays,
            quadpoints_phi=self.phis,
            quadpoints_theta=self.thetas,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=False,
            scatter_indices=None,
            optimize_G=True,
            weight_inv_modB=False,
        )

    def test_gradient_fd(self):
        """Composed gradient satisfies JAX-native scalar derivative checks."""
        _assert_composed_penalty_gradient_contract(
            self.x,
            self.kwargs,
            seed=100,
        )

    def test_surface_dof_gradient_nonzero(self):
        """Unlike M1, composed gradient has nonzero surface DOF entries."""
        _, grad = boozer_penalty_grad_composed(self.x, **self.kwargs)
        grad = jnp.asarray(grad)
        # Surface DOFs are x[:-2]; at least some should be nonzero
        sdof_grad = grad[:-2]
        assert float(jnp.max(jnp.abs(sdof_grad))) > 1e-12

    def test_gradient_optimize_G_false(self):
        """Gradient works with optimize_G=False (G from currents)."""
        x_no_G = jnp.concatenate([self.dofs, jnp.array([self.iota])])
        kwargs = {**self.kwargs, "optimize_G": False}
        _assert_composed_penalty_gradient_contract(
            x_no_G,
            kwargs,
            seed=101,
        )


# ---------------------------------------------------------------------------
# Test: Composed residual Jacobian
# ---------------------------------------------------------------------------


class TestBoozerResidualJacobianComposed:
    """Validate the BoozerExact Jacobian via finite differences."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 4, 4

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0
        self.x = jnp.concatenate([self.dofs, jnp.array([self.iota, self.G])])
        self.kwargs = dict(
            coil_arrays=self.coil_arrays,
            quadpoints_phi=self.phis,
            quadpoints_theta=self.thetas,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=False,
            scatter_indices=None,
            weight_inv_modB=False,
        )

    def test_jacobian_shape(self):
        """Jacobian has correct shape (n_res, n_dofs)."""
        r, J = boozer_residual_jacobian_composed(self.x, **self.kwargs)
        n_res = 3 * self.nphi * self.ntheta
        n_dofs = len(self.x)
        assert J.shape == (n_res, n_dofs)
        assert r.shape == (n_res,)

    def test_jacobian_fd(self):
        """Jacobian satisfies a directional finite-difference contract."""
        _assert_composed_residual_jacobian_contract(
            self.x,
            self.kwargs,
            seed=21,
        )


# ---------------------------------------------------------------------------
# Test: Composed Hessian
# ---------------------------------------------------------------------------


class TestBoozerHessianComposed:
    """Validate Hessian of the composed penalty objective."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 4, 4

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0
        self.x = jnp.concatenate([self.dofs, jnp.array([self.iota, self.G])])
        self.kwargs = dict(
            coil_arrays=self.coil_arrays,
            quadpoints_phi=self.phis,
            quadpoints_theta=self.thetas,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=False,
            scatter_indices=None,
            optimize_G=True,
            weight_inv_modB=False,
        )

    def test_hessian_symmetry(self):
        """Hessian of composed objective is symmetric."""
        H = jax.hessian(boozer_penalty_composed)(self.x, **self.kwargs)
        np.testing.assert_allclose(np.array(H), np.array(H.T), atol=1e-12)

    def test_hessian_fd(self):
        """Hessian satisfies a directional Jacobian-of-gradient contract."""
        H = jax.hessian(boozer_penalty_composed)(self.x, **self.kwargs)

        def gradient_fn(x):
            return boozer_penalty_grad_composed(x, **self.kwargs)[1]

        _assert_directional_jacobian_contract(
            gradient_fn,
            self.x,
            H,
            seeds=_paired_seeds(31),
            linearization_tol=2e-4,
            fd_tol=1e-4,
        )

    def test_hessian_taylor_convergence(self):
        """Composed Hessian reaches a stable second-order Taylor regime."""
        def objective(arg):
            return boozer_penalty_composed(arg, **self.kwargs)

        _, grad0 = boozer_penalty_grad_composed(self.x, **self.kwargs)
        H = jax.hessian(boozer_penalty_composed)(self.x, **self.kwargs)
        _assert_second_order_taylor_contract(
            objective,
            self.x,
            grad0,
            H,
            seed=32,
            best_defect_tol=1e-8,
            min_observed_order=2.0,
        )


# ---------------------------------------------------------------------------
# Test: Outer coil VJP
# ---------------------------------------------------------------------------


class TestBoozerResidualCoilVJP:
    """Validate the outer residual VJP w.r.t. coil parameters."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 4, 4

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0

        # Evaluate fixed surface geometry
        self.gamma = surface_gamma_from_dofs(
            self.dofs,
            self.phis,
            self.thetas,
            self.mpol,
            self.ntor,
            self.nfp,
            stellsym=False,
        )
        self.xphi = surface_gammadash1_from_dofs(
            self.dofs,
            self.phis,
            self.thetas,
            self.mpol,
            self.ntor,
            self.nfp,
            stellsym=False,
        )
        self.xtheta = surface_gammadash2_from_dofs(
            self.dofs,
            self.phis,
            self.thetas,
            self.mpol,
            self.ntor,
            self.nfp,
            stellsym=False,
        )

    def _assert_coil_vjp_scalar_contract(self, grad_idx, *, adjoint_seed, check_seed):
        """Check one coil-input VJP block against a scalarized JAX objective."""
        n_res = 3 * self.nphi * self.ntheta
        adjoint = _make_adjoint(adjoint_seed, n_res, self.gamma.dtype)

        (d_coil_arrays,) = boozer_residual_coil_vjp(
            adjoint,
            gamma=self.gamma,
            xphi=self.xphi,
            xtheta=self.xtheta,
            coil_arrays=self.coil_arrays,
            iota=self.iota,
            G=self.G,
            weight_inv_modB=False,
        )
        coil_inputs = [self.coil_gammas, self.coil_gammadashs, self.coil_currents]
        target = coil_inputs[grad_idx]
        grad = d_coil_arrays[0][grad_idx]

        objective = _coil_residual_scalar_objective(
            gamma=self.gamma,
            xphi=self.xphi,
            xtheta=self.xtheta,
            coil_inputs=coil_inputs,
            iota=self.iota,
            G=self.G,
            adjoint=adjoint,
            grad_idx=grad_idx,
        )

        _assert_scalar_grad_matches(
            objective,
            target,
            grad,
            seed=check_seed,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_coil_vjp_currents_fd(self):
        """VJP w.r.t. coil currents matches JAX-native scalarization."""
        self._assert_coil_vjp_scalar_contract(2, adjoint_seed=99, check_seed=1099)

    def test_coil_vjp_shapes(self):
        """VJP outputs have correct shapes."""
        n_res = 3 * self.nphi * self.ntheta
        adjoint = jnp.ones(n_res)

        (d_coil_arrays,) = boozer_residual_coil_vjp(
            adjoint,
            gamma=self.gamma,
            xphi=self.xphi,
            xtheta=self.xtheta,
            coil_arrays=self.coil_arrays,
            iota=self.iota,
            G=self.G,
            weight_inv_modB=False,
        )
        # Single group → shapes match the input group
        dcg, dcgd, dci = d_coil_arrays[0]
        assert dcg.shape == self.coil_gammas.shape
        assert dcgd.shape == self.coil_gammadashs.shape
        assert dci.shape == self.coil_currents.shape

    @pytest.mark.parametrize(
        "grad_idx,seed",
        [(0, 42), (1, 43)],
        ids=["gammas", "gammadashs"],
    )
    def test_coil_vjp_geometry_fd(self, grad_idx, seed):
        """VJP w.r.t. coil gammas/gammadashs matches JAX-native scalarization."""
        self._assert_coil_vjp_scalar_contract(
            grad_idx,
            adjoint_seed=seed,
            check_seed=seed + 1000,
        )


# ---------------------------------------------------------------------------
# Test: weight_inv_modB=True (reviewer finding #1)
# ---------------------------------------------------------------------------


class TestComposedWeightInvModB:
    """Validate composed derivatives with weight_inv_modB=True."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 4, 4

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0
        self.x = jnp.concatenate([self.dofs, jnp.array([self.iota, self.G])])
        self.kwargs = dict(
            coil_arrays=self.coil_arrays,
            quadpoints_phi=self.phis,
            quadpoints_theta=self.thetas,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=False,
            scatter_indices=None,
            optimize_G=True,
            weight_inv_modB=True,
        )

    def test_gradient_weighted_fd(self):
        """Composed gradient with 1/|B| weighting satisfies JAX-native checks."""
        _assert_composed_penalty_gradient_contract(
            self.x,
            self.kwargs,
            seed=2026,
        )

    def test_jacobian_weighted_fd(self):
        kwargs_res = {k: v for k, v in self.kwargs.items() if k not in ("optimize_G",)}
        _assert_composed_residual_jacobian_contract(
            self.x,
            kwargs_res,
            seed=22,
        )


# ---------------------------------------------------------------------------
# Test: stellsym=True in composed path (reviewer finding #2)
# ---------------------------------------------------------------------------


class TestComposedStellsym:
    """Validate composed gradient with stellarator symmetry."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 4, 4

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs, self.scatter_idx = _make_torus_dofs_stellsym(
            self.mpol,
            self.ntor,
            self.nfp,
        )
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0
        self.x = jnp.concatenate([self.dofs, jnp.array([self.iota, self.G])])
        self.kwargs = dict(
            coil_arrays=self.coil_arrays,
            quadpoints_phi=self.phis,
            quadpoints_theta=self.thetas,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=True,
            scatter_indices=self.scatter_idx,
            optimize_G=True,
            weight_inv_modB=False,
        )

    def test_gradient_stellsym_fd(self):
        """Composed gradient with stellsym satisfies JAX-native checks."""
        _assert_composed_penalty_gradient_contract(
            self.x,
            self.kwargs,
            seed=2027,
        )

    def test_decision_vector_shorter(self):
        """Stellsym decision vector is shorter than non-stellsym."""
        n_per = (2 * self.mpol + 1) * (2 * self.ntor + 1)
        ndofs_full = 3 * n_per + 2  # + iota, G
        assert len(self.x) < ndofs_full


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
