"""Item 25 parity tests for ``simsopt.jax_core.pm_optimization``.

Validates the MwPGP solver port against:

1. Internal consistency: the per-dipole helper kernels
   (``projection_l2_balls``, ``phi_mwpgp``, ``g_reduced_gradient``,
   ``g_reduced_projected_gradient``, ``find_max_alphaf``) match the
   scalar C++ formulas.
2. Algorithm monotonicity: the cost ``1/2 ||A m - b||^2 + reg_l2 ||m||^2
   + (1/(2 nu)) ||m - m_proxy||^2`` is non-increasing under
   ``mwpgp_solve``.
3. Optimality recovery: for a synthetic problem whose unconstrained
   minimiser lives inside the L2 balls, ``mwpgp_solve`` recovers the
   minimiser to within an O(geometric-decay) tolerance.
4. C++ oracle parity: ``mwpgp_solve`` matches ``simsoptpp.MwPGP_algorithm``
   iterate-by-iterate up to floating-point rounding for a 5-iteration
   trace (``epsilon=0`` is not exposed on the C++ side so we use a
   medium-sized problem and a step count that fits before the C++ kernel
   triggers its history snapshot path).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import simsoptpp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.jax_core.pm_optimization import (
    GPMOArbVecBacktrackingSpec,
    GPMOArbVecSpec,
    GPMOBacktrackingSpec,
    GPMOBaselineSpec,
    GPMOMultiSpec,
    PMOptimizationSpec,
    find_max_alphaf,
    gpmo_arbvec_backtracking_solve,
    gpmo_connectivity_matrix,
    gpmo_arbvec_solve,
    gpmo_backtracking_solve,
    gpmo_baseline_candidate_costs,
    gpmo_baseline_solve,
    gpmo_baseline_step,
    gpmo_multi_solve,
    g_reduced_gradient,
    g_reduced_projected_gradient,
    initialize_gpmo_arbvec,
    mwpgp_initial_state,
    mwpgp_solve,
    mwpgp_step,
    phi_mwpgp,
    projection_l2_balls,
)
from .jaxpr_utils import count_jaxpr_primitives


# ---------------------------------------------------------------------
# Tolerance contracts:
#
# - PER-KERNEL parity exercises closed-form scalar algebra and uses the
#   shared ``direct_kernel`` ladder lane.
# - MwPGP iterate/state checks use the shared ``pm_mwpgp_fixed_step``
#   lane, which records the solver-specific trace, optimality, and
#   monotonicity tolerances.
# ---------------------------------------------------------------------

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_PM_FIXED_STEP = parity_ladder_tolerances("pm_mwpgp_fixed_step")

_PER_KERNEL_RTOL = _DIRECT_KERNEL["rtol"]
_PER_KERNEL_ATOL = _DIRECT_KERNEL["atol"]

_STATE_TRACE_RTOL = _PM_FIXED_STEP["state_trace_rtol"]
_STATE_TRACE_ATOL = _PM_FIXED_STEP["state_trace_atol"]

_OPTIMALITY_ATOL = _PM_FIXED_STEP["optimality_atol"]

_MONOTONICITY_RTOL = _PM_FIXED_STEP["monotonicity_rtol"]
_SINGLE_STEP_RTOL = _PM_FIXED_STEP["single_step_rtol"]
_SINGLE_STEP_ATOL = _PM_FIXED_STEP["single_step_atol"]


# ---------------------------------------------------------------------
# Fixtures and random generators.
# ---------------------------------------------------------------------


def _random_problem(seed: int, M: int, N: int, m_maxima_scale: float = 1.0):
    """Construct ``(A, b, m_maxima, m_proxy, m0)`` for a small problem.

    ``A`` and ``b`` are drawn from N(0, 1) with no special structure
    other than that ``A`` is well-conditioned (we add ``+ 0.5 * I`` to
    the implicit Gram matrix by drawing more rows than columns).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(M, 3 * N))
    b = rng.standard_normal(size=(M,))
    m_maxima = np.full(N, m_maxima_scale)
    m_proxy = np.zeros((N, 3))
    m0 = np.zeros((N, 3))
    return A, b, m_maxima, m_proxy, m0


def _make_spec(
    m_maxima: np.ndarray,
    m_proxy: np.ndarray,
    *,
    alpha: float,
    reg_l2: float = 0.0,
    nu: float = 1.0e100,
):
    """Wrap the test inputs into the public ``PMOptimizationSpec``."""
    return PMOptimizationSpec(
        m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
        m_proxy=jnp.asarray(m_proxy, dtype=jnp.float64),
        nu=jnp.asarray(nu, dtype=jnp.float64),
        reg_l2=jnp.asarray(reg_l2, dtype=jnp.float64),
        alpha=jnp.asarray(alpha, dtype=jnp.float64),
    )


def _cost(A, b, m_proxy, m_maxima, nu, reg_l2, m):
    """Evaluate the convex MwPGP objective at ``m``.

    ``J(m) = 1/2 ||A m - b||^2 + reg_l2 ||m||^2 + (1/(2 nu)) ||m - m_proxy||^2``

    Per the C++ ``print_MwPGP`` (line 132-140), the constant terms ``L1``
    and ``L0`` are NOT part of the optimisation (they are recorded only).
    The objective is the sum ``R2 + N2 + L2``.
    """
    m_flat = m.reshape(3 * m.shape[0])
    R2 = 0.5 * np.sum((A @ m_flat - b) ** 2)
    L2 = reg_l2 * np.sum(m**2)
    N2 = 0.5 * np.sum((m - m_proxy) ** 2) / nu
    return R2 + L2 + N2


def _gpmo_spec(m_maxima: np.ndarray, reg_l2: float, single_direction: int = -1):
    return GPMOBaselineSpec(
        m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
        reg_l2=jnp.asarray(reg_l2, dtype=jnp.float64),
        single_direction=single_direction,
    )


def _gpmo_problem(seed: int, M: int = 9, N: int = 4):
    rng = np.random.default_rng(seed)
    A_scaled = np.ascontiguousarray(rng.standard_normal(size=(M, 3 * N)))
    b = np.ascontiguousarray(rng.standard_normal(size=(M,)))
    m_maxima = np.ascontiguousarray(0.3 + rng.random(size=N))
    normal_norms = np.ones(M, dtype=np.float64)
    return A_scaled, b, m_maxima, normal_norms


def _gpmo_spatial_problem(seed: int, M: int = 9, N: int = 6):
    rng = np.random.default_rng(seed)
    A_scaled = np.ascontiguousarray(rng.standard_normal(size=(M, 3 * N)))
    b = np.ascontiguousarray(rng.standard_normal(size=(M,)))
    m_maxima = np.ascontiguousarray(0.3 + rng.random(size=N))
    normal_norms = np.ones(M, dtype=np.float64)
    dipoles = np.ascontiguousarray(rng.standard_normal(size=(N, 3)))
    return A_scaled, b, m_maxima, normal_norms, dipoles


def _gpmo_pol_vectors(seed: int, N: int, P: int = 4):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(size=(N, P, 3))
    norms = np.linalg.norm(raw, axis=2)
    return np.ascontiguousarray(raw / norms[:, :, None])


# ---------------------------------------------------------------------
# Per-kernel parity (closed-form scalar checks vs C++ formula).
# ---------------------------------------------------------------------


class TestPMKernelHelpers:
    def test_projection_inside_ball_passes_through(self):
        m = jnp.asarray(np.array([[0.1, 0.2, -0.3], [0.0, 0.0, 0.0]]))
        m_maxima = jnp.asarray(np.array([1.0, 1.0]))
        out = np.asarray(projection_l2_balls(m, m_maxima))
        np.testing.assert_allclose(
            out, np.asarray(m), rtol=_PER_KERNEL_RTOL, atol=_PER_KERNEL_ATOL
        )

    def test_projection_outside_ball_renormalises(self):
        m = jnp.asarray(np.array([[3.0, 4.0, 0.0], [10.0, 0.0, 0.0]]))
        m_maxima = jnp.asarray(np.array([1.0, 2.0]))
        out = np.asarray(projection_l2_balls(m, m_maxima))
        # Row 0: ||m|| = 5 -> divide by 5 -> radius 1.
        np.testing.assert_allclose(
            out[0],
            np.array([3.0 / 5.0, 4.0 / 5.0, 0.0]),
            rtol=_PER_KERNEL_RTOL,
            atol=_PER_KERNEL_ATOL,
        )
        # Row 1: ||m|| = 10, radius 2 -> divide by 5.
        np.testing.assert_allclose(
            out[1],
            np.array([2.0, 0.0, 0.0]),
            rtol=_PER_KERNEL_RTOL,
            atol=_PER_KERNEL_ATOL,
        )

    def test_projection_zero_mmax_zero_row_is_finite(self):
        m = jnp.asarray(
            np.array(
                [
                    [0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.4, -0.3, 0.0],
                ]
            )
        )
        m_maxima = jnp.asarray(np.array([1.0, 0.0, 1.0, np.nan]))

        out = np.asarray(projection_l2_balls(m, m_maxima))

        np.testing.assert_allclose(
            out,
            np.array(
                [
                    [0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.4, -0.3, 0.0],
                ]
            ),
            rtol=_PER_KERNEL_RTOL,
            atol=_PER_KERNEL_ATOL,
        )
        assert np.isfinite(out).all()

    def test_phi_off_ball_returns_g(self):
        m = jnp.asarray(np.array([[0.0, 0.0, 0.0]]))
        g = jnp.asarray(np.array([[1.0, 2.0, 3.0]]))
        m_maxima = jnp.asarray(np.array([1.0]))
        out = np.asarray(phi_mwpgp(m, g, m_maxima))
        np.testing.assert_allclose(
            out, np.asarray(g), rtol=_PER_KERNEL_RTOL, atol=_PER_KERNEL_ATOL
        )

    def test_phi_on_ball_returns_zero(self):
        # On-ball dipole (||m||=1 with m_maxima=1).
        m = jnp.asarray(np.array([[1.0, 0.0, 0.0]]))
        g = jnp.asarray(np.array([[1.0, 2.0, 3.0]]))
        m_maxima = jnp.asarray(np.array([1.0]))
        out = np.asarray(phi_mwpgp(m, g, m_maxima))
        np.testing.assert_allclose(
            out,
            np.zeros_like(np.asarray(g)),
            rtol=_PER_KERNEL_RTOL,
            atol=_PER_KERNEL_ATOL,
        )

    def test_g_reduced_gradient_inside_ball_returns_g(self):
        # If m - alpha g stays inside the ball, projection is identity
        # and (m - proj(m-alpha g))/alpha = g.
        m = jnp.asarray(np.array([[0.0, 0.0, 0.0]]))
        g = jnp.asarray(np.array([[0.5, 0.1, -0.2]]))
        alpha = jnp.asarray(0.1)
        m_maxima = jnp.asarray(np.array([5.0]))
        out = np.asarray(g_reduced_gradient(m, g, alpha, m_maxima))
        np.testing.assert_allclose(
            out, np.asarray(g), rtol=_PER_KERNEL_RTOL, atol=_PER_KERNEL_ATOL
        )

    def test_find_max_alphaf_no_p_returns_sentinel(self):
        m = jnp.asarray(np.array([[0.5, 0.0, 0.0]]))
        p = jnp.asarray(np.array([[0.0, 0.0, 0.0]]))
        m_maxima = jnp.asarray(np.array([1.0]))
        out = np.asarray(find_max_alphaf(m, p, m_maxima))
        assert out[0] == 1.0e100

    def test_find_max_alphaf_solves_boundary_quadratic(self):
        # m=(0.5,0,0), p=(1,0,0), m_maxima=1.0 -> ball at radius 1
        # ``m - alpha p = (0.5 - alpha, 0, 0)``. Boundary hit at
        # ``|0.5 - alpha| = 1`` -> alpha = -0.5 (back into ball) or
        # alpha = 1.5 (forward beyond ball). The function returns the
        # positive root via the (-b + sqrt(...)) / (2a) branch.
        # a = 1, b = -2 * 0.5 = -1, c = 0.25 - 1 = -0.75
        # alphaf_plus = (1 + sqrt(1 + 3)) / 2 = (1 + 2) / 2 = 1.5
        m = jnp.asarray(np.array([[0.5, 0.0, 0.0]]))
        p = jnp.asarray(np.array([[1.0, 0.0, 0.0]]))
        m_maxima = jnp.asarray(np.array([1.0]))
        out = np.asarray(find_max_alphaf(m, p, m_maxima))
        np.testing.assert_allclose(
            out[0], 1.5, rtol=_PER_KERNEL_RTOL, atol=_PER_KERNEL_ATOL
        )

    def test_g_reduced_projected_gradient_is_phi_plus_beta(self):
        # Smoke: assemble random inputs and check that the public helper
        # equals the explicit ``phi + beta_tilde`` sum.
        rng = np.random.default_rng(42)
        m = jnp.asarray(rng.standard_normal(size=(7, 3)))
        g = jnp.asarray(rng.standard_normal(size=(7, 3)))
        m_maxima = jnp.asarray(np.full(7, 0.5))
        alpha = jnp.asarray(0.1)
        rg = np.asarray(g_reduced_projected_gradient(m, g, alpha, m_maxima))
        phi_arr = np.asarray(phi_mwpgp(m, g, m_maxima))
        # All rows are off-ball given random N(0,1) m vs m_maxima=0.5
        # (||m||^2 is generally far from 0.25).
        np.testing.assert_allclose(
            rg, phi_arr, rtol=_PER_KERNEL_RTOL, atol=_PER_KERNEL_ATOL
        )


class TestGPMOBaseline:
    def test_candidate_costs_match_explicit_residual_scores(self):
        A_scaled, b, m_maxima, _ = _gpmo_problem(seed=2501, M=5, N=2)
        residual = -b
        available = np.ones((2, 3), dtype=bool)
        reg_l2 = 0.125
        costs = np.asarray(
            gpmo_baseline_candidate_costs(
                _gpmo_spec(m_maxima, reg_l2),
                jnp.asarray(A_scaled),
                jnp.asarray(residual),
                jnp.asarray(available),
            )
        )
        mmax_vec = np.repeat(m_maxima, 3)
        expected_plus = np.sum((residual[:, None] + A_scaled) ** 2, axis=0)
        expected_minus = np.sum((residual[:, None] - A_scaled) ** 2, axis=0)
        expected = np.concatenate(
            [
                expected_plus + reg_l2 * mmax_vec**2,
                expected_minus + reg_l2 * mmax_vec**2,
            ]
        )
        np.testing.assert_allclose(
            costs, expected, rtol=_PER_KERNEL_RTOL, atol=_PER_KERNEL_ATOL
        )

    def test_step_uses_cpp_min_element_tie_order(self):
        A_scaled = jnp.zeros((3, 6), dtype=jnp.float64)
        b = jnp.zeros((3,), dtype=jnp.float64)
        m_maxima = np.ones(2, dtype=np.float64)
        state = (
            jnp.zeros((2, 3), dtype=jnp.float64),
            -b,
            jnp.ones((2, 3), dtype=bool),
        )
        new_state, trace = gpmo_baseline_step(
            _gpmo_spec(m_maxima, reg_l2=0.0), state, A_scaled
        )
        dipole, component, sign, residual_sq = trace

        assert int(dipole) == 0
        assert int(component) == 0
        assert float(sign) == 1.0
        assert float(residual_sq) == 0.0
        np.testing.assert_array_equal(
            np.asarray(new_state[0]),
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )

    def test_solver_matches_cpp_baseline_for_all_single_direction_modes(self):
        for single_direction in (-1, 0, 1, 2):
            A_scaled, b, m_maxima, normal_norms = _gpmo_problem(
                seed=2600 + single_direction, M=11, N=5
            )
            K = 4
            reg_l2 = 0.0
            _, _, _, x_cpp = simsoptpp.GPMO_baseline(
                np.ascontiguousarray(A_scaled.T),
                b,
                np.sqrt(reg_l2) * np.repeat(m_maxima, 3),
                normal_norms,
                K=K,
                verbose=False,
                nhistory=K,
                single_direction=single_direction,
            )
            result = gpmo_baseline_solve(
                _gpmo_spec(m_maxima, reg_l2, single_direction),
                jnp.asarray(A_scaled),
                jnp.asarray(b),
                K=K,
            )
            np.testing.assert_allclose(
                np.asarray(result.x),
                x_cpp,
                rtol=_STATE_TRACE_RTOL,
                atol=_STATE_TRACE_ATOL,
            )

    def test_solver_matches_cpp_baseline_with_l2_regularization(self):
        A_scaled, b, m_maxima, normal_norms = _gpmo_problem(seed=2503, M=13, N=5)
        K = 3
        reg_l2 = 0.2
        _, _, _, x_cpp = simsoptpp.GPMO_baseline(
            np.ascontiguousarray(A_scaled.T),
            b,
            np.sqrt(reg_l2) * np.repeat(m_maxima, 3),
            normal_norms,
            K=K,
            verbose=False,
            nhistory=K,
            single_direction=-1,
        )
        result = gpmo_baseline_solve(
            _gpmo_spec(m_maxima, reg_l2),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=K,
        )
        np.testing.assert_allclose(
            np.asarray(result.x),
            x_cpp,
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )

    def test_residual_history_matches_running_residual_invariant(self):
        A_scaled, b, m_maxima, _ = _gpmo_problem(seed=2504, M=8, N=4)
        result = gpmo_baseline_solve(
            _gpmo_spec(m_maxima, reg_l2=0.0),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=3,
        )
        residual = A_scaled @ np.asarray(result.x).reshape(-1) - b
        np.testing.assert_allclose(
            np.asarray(result.residual),
            residual,
            rtol=_PER_KERNEL_RTOL,
            atol=_PER_KERNEL_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(result.residual_history[-1]),
            np.sum(residual**2),
            rtol=_PER_KERNEL_RTOL,
            atol=_PER_KERNEL_ATOL,
        )

    def test_solver_jits_under_strict_transfer_guard(self):
        A_scaled, b, m_maxima, _ = _gpmo_problem(seed=2505, M=7, N=3)
        spec = _gpmo_spec(m_maxima, reg_l2=0.0)
        A_device = jax.device_put(jnp.asarray(A_scaled))
        b_device = jax.device_put(jnp.asarray(b))

        @jax.jit
        def _run(spec_data: GPMOBaselineSpec, A_data: jax.Array, b_data: jax.Array):
            return gpmo_baseline_solve(spec_data, A_data, b_data, K=2).x

        _run(spec, A_device, b_device).block_until_ready()
        with jax.transfer_guard("disallow"):
            out = _run(spec, A_device, b_device)
            out.block_until_ready()

        assert out.shape == (3, 3)
        assert np.all(np.isfinite(np.asarray(out)))


class TestGPMOMulti:
    def test_connectivity_matrix_matches_numpy_nearest_order(self):
        _, _, _, _, dipoles = _gpmo_spatial_problem(seed=2550, M=5, N=6)
        distances = np.linalg.norm(dipoles[:, None, :] - dipoles[None, :, :], axis=2)
        expected = np.argsort(distances, axis=1)

        np.testing.assert_array_equal(
            np.asarray(gpmo_connectivity_matrix(jnp.asarray(dipoles))),
            expected,
        )

    def test_solver_matches_cpp_multi_for_single_direction_modes(self):
        for single_direction in (-1, 0, 1, 2):
            A_scaled, b, m_maxima, normal_norms, dipoles = _gpmo_spatial_problem(
                seed=2650 + single_direction, M=11, N=7
            )
            K = 2
            Nadjacent = 2
            reg_l2 = 0.0
            _, _, _, x_cpp = simsoptpp.GPMO_multi(
                np.ascontiguousarray(A_scaled.T),
                b,
                np.sqrt(reg_l2) * np.repeat(m_maxima, 3),
                normal_norms,
                K=K,
                verbose=False,
                nhistory=K,
                dipole_grid_xyz=dipoles,
                single_direction=single_direction,
                Nadjacent=Nadjacent,
            )
            result = gpmo_multi_solve(
                GPMOMultiSpec(
                    m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                    reg_l2=jnp.asarray(reg_l2, dtype=jnp.float64),
                    dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
                    single_direction=single_direction,
                    Nadjacent=Nadjacent,
                ),
                jnp.asarray(A_scaled),
                jnp.asarray(b),
                K=K,
            )
            np.testing.assert_allclose(
                np.asarray(result.x),
                x_cpp,
                rtol=_STATE_TRACE_RTOL,
                atol=_STATE_TRACE_ATOL,
            )

    def test_solver_matches_cpp_multi_with_l2_regularization(self):
        A_scaled, b, m_maxima, normal_norms, dipoles = _gpmo_spatial_problem(
            seed=2551, M=12, N=7
        )
        K = 2
        Nadjacent = 2
        reg_l2 = 0.17
        _, _, _, x_cpp = simsoptpp.GPMO_multi(
            np.ascontiguousarray(A_scaled.T),
            b,
            np.sqrt(reg_l2) * np.repeat(m_maxima, 3),
            normal_norms,
            K=K,
            verbose=False,
            nhistory=K,
            dipole_grid_xyz=dipoles,
            single_direction=-1,
            Nadjacent=Nadjacent,
        )
        result = gpmo_multi_solve(
            GPMOMultiSpec(
                m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                reg_l2=jnp.asarray(reg_l2, dtype=jnp.float64),
                dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
                Nadjacent=Nadjacent,
            ),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=K,
        )
        np.testing.assert_allclose(
            np.asarray(result.x),
            x_cpp,
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )

    def test_multi_residual_history_matches_running_residual_invariant(self):
        A_scaled, b, m_maxima, _, dipoles = _gpmo_spatial_problem(seed=2552, M=8, N=6)
        result = gpmo_multi_solve(
            GPMOMultiSpec(
                m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
                dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
                Nadjacent=2,
            ),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=2,
        )
        residual = A_scaled @ np.asarray(result.x).reshape(-1) - b
        np.testing.assert_allclose(
            np.asarray(result.residual),
            residual,
            rtol=_PER_KERNEL_RTOL,
            atol=_PER_KERNEL_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(result.residual_history[-1]),
            np.sum(residual**2),
            rtol=_PER_KERNEL_RTOL,
            atol=_PER_KERNEL_ATOL,
        )

    def test_multi_rejects_oversubscribed_fixed_step_contract(self):
        A_scaled, b, m_maxima, _, dipoles = _gpmo_spatial_problem(seed=2554, M=8, N=5)
        with np.testing.assert_raises(ValueError):
            gpmo_multi_solve(
                GPMOMultiSpec(
                    m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                    reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
                    dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
                    Nadjacent=2,
                ),
                jnp.asarray(A_scaled),
                jnp.asarray(b),
                K=3,
            )

    def test_multi_solver_jits_under_strict_transfer_guard(self):
        A_scaled, b, m_maxima, _, dipoles = _gpmo_spatial_problem(seed=2553, M=7, N=5)
        spec = GPMOMultiSpec(
            m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
            reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
            dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
            Nadjacent=2,
        )
        A_device = jax.device_put(jnp.asarray(A_scaled))
        b_device = jax.device_put(jnp.asarray(b))

        @jax.jit
        def _run(spec_data: GPMOMultiSpec, A_data: jax.Array, b_data: jax.Array):
            return gpmo_multi_solve(spec_data, A_data, b_data, K=2).x

        _run(spec, A_device, b_device).block_until_ready()
        with jax.transfer_guard("disallow"):
            out = _run(spec, A_device, b_device)
            out.block_until_ready()

        assert out.shape == (5, 3)
        assert np.all(np.isfinite(np.asarray(out)))


class TestGPMOBacktracking:
    def test_solver_matches_cpp_backtracking_and_allows_reopened_sites(self):
        A_scaled, b, m_maxima, normal_norms, dipoles = _gpmo_spatial_problem(
            seed=100, M=10, N=6
        )
        K = 8
        Nadjacent = 3
        backtracking = 2
        reg_l2 = 0.0
        _, _, _, _, x_cpp = simsoptpp.GPMO_backtracking(
            np.ascontiguousarray(A_scaled.T),
            b,
            np.sqrt(reg_l2) * np.repeat(m_maxima, 3),
            normal_norms,
            K=K,
            verbose=False,
            nhistory=K,
            backtracking=backtracking,
            dipole_grid_xyz=dipoles,
            single_direction=-1,
            Nadjacent=Nadjacent,
            max_nMagnets=m_maxima.size,
        )
        result = gpmo_backtracking_solve(
            GPMOBacktrackingSpec(
                m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                reg_l2=jnp.asarray(reg_l2, dtype=jnp.float64),
                dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
                Nadjacent=Nadjacent,
                backtracking=backtracking,
                max_nMagnets=m_maxima.size,
            ),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=K,
        )

        np.testing.assert_allclose(
            np.asarray(result.x),
            x_cpp,
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )
        assert np.max(np.asarray(result.removed_pair_count_history)) > 0
        assert K > m_maxima.size

    def test_solver_matches_cpp_backtracking_stop_at_max_nmagnets(self):
        A_scaled, b, m_maxima, normal_norms, dipoles = _gpmo_spatial_problem(
            seed=101, M=9, N=5
        )
        K = 7
        Nadjacent = 2
        backtracking = 2
        max_nMagnets = 3
        reg_l2 = 0.0
        _, _, _, _, x_cpp = simsoptpp.GPMO_backtracking(
            np.ascontiguousarray(A_scaled.T),
            b,
            np.sqrt(reg_l2) * np.repeat(m_maxima, 3),
            normal_norms,
            K=K,
            verbose=False,
            nhistory=K,
            backtracking=backtracking,
            dipole_grid_xyz=dipoles,
            single_direction=-1,
            Nadjacent=Nadjacent,
            max_nMagnets=max_nMagnets,
        )
        result = gpmo_backtracking_solve(
            GPMOBacktrackingSpec(
                m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                reg_l2=jnp.asarray(reg_l2, dtype=jnp.float64),
                dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
                Nadjacent=Nadjacent,
                backtracking=backtracking,
                max_nMagnets=max_nMagnets,
            ),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=K,
        )

        np.testing.assert_allclose(
            np.asarray(result.x),
            x_cpp,
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )
        assert np.any(np.asarray(result.done_history))
        assert np.asarray(result.num_nonzeros_history).max() == max_nMagnets

    def test_backtracking_solver_jits_under_strict_transfer_guard(self):
        A_scaled, b, m_maxima, _, dipoles = _gpmo_spatial_problem(seed=2556, M=8, N=5)
        spec = GPMOBacktrackingSpec(
            m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
            reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
            dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
            Nadjacent=2,
            backtracking=2,
            max_nMagnets=4,
        )
        A_device = jax.device_put(jnp.asarray(A_scaled))
        b_device = jax.device_put(jnp.asarray(b))

        @jax.jit
        def _run(spec_data: GPMOBacktrackingSpec, A_data: jax.Array, b_data: jax.Array):
            return gpmo_backtracking_solve(spec_data, A_data, b_data, K=6).x

        _run(spec, A_device, b_device).block_until_ready()
        with jax.transfer_guard("disallow"):
            out = _run(spec, A_device, b_device)
            out.block_until_ready()

        assert out.shape == (5, 3)
        assert np.all(np.isfinite(np.asarray(out)))


class TestGPMOArbVec:
    def test_solver_matches_cpp_arbvec_with_l2_regularization(self):
        A_scaled, b, m_maxima, normal_norms = _gpmo_problem(seed=2570, M=12, N=6)
        pol_vectors = _gpmo_pol_vectors(seed=2571, N=m_maxima.size, P=4)
        K = 3
        reg_l2 = 0.19
        _, _, _, x_cpp = simsoptpp.GPMO_ArbVec(
            np.ascontiguousarray(A_scaled.T),
            b,
            np.sqrt(reg_l2) * np.repeat(m_maxima, 3),
            normal_norms,
            pol_vectors,
            K=K,
            verbose=False,
            nhistory=K,
        )
        result = gpmo_arbvec_solve(
            GPMOArbVecSpec(
                m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                reg_l2=jnp.asarray(reg_l2, dtype=jnp.float64),
                pol_vectors=jnp.asarray(pol_vectors, dtype=jnp.float64),
            ),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=K,
        )
        np.testing.assert_allclose(
            np.asarray(result.x),
            x_cpp,
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )

    def test_identity_arbvec_matches_baseline(self):
        A_scaled, b, m_maxima, _ = _gpmo_problem(seed=2572, M=10, N=5)
        identity = np.broadcast_to(np.eye(3), (m_maxima.size, 3, 3)).copy()
        K = 4
        reg_l2 = 0.0
        baseline = gpmo_baseline_solve(
            _gpmo_spec(m_maxima, reg_l2),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=K,
        )
        arbvec = gpmo_arbvec_solve(
            GPMOArbVecSpec(
                m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                reg_l2=jnp.asarray(reg_l2, dtype=jnp.float64),
                pol_vectors=jnp.asarray(identity, dtype=jnp.float64),
            ),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=K,
        )
        np.testing.assert_allclose(
            np.asarray(arbvec.x),
            np.asarray(baseline.x),
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )

    def test_arbvec_residual_history_matches_running_residual_invariant(self):
        A_scaled, b, m_maxima, _ = _gpmo_problem(seed=2573, M=8, N=5)
        pol_vectors = _gpmo_pol_vectors(seed=2574, N=m_maxima.size, P=3)
        result = gpmo_arbvec_solve(
            GPMOArbVecSpec(
                m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
                pol_vectors=jnp.asarray(pol_vectors, dtype=jnp.float64),
            ),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=3,
        )
        residual = A_scaled @ np.asarray(result.x).reshape(-1) - b
        np.testing.assert_allclose(
            np.asarray(result.residual),
            residual,
            rtol=_PER_KERNEL_RTOL,
            atol=_PER_KERNEL_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(result.residual_history[-1]),
            np.sum(residual**2),
            rtol=_PER_KERNEL_RTOL,
            atol=_PER_KERNEL_ATOL,
        )

    def test_arbvec_solver_jits_under_strict_transfer_guard(self):
        A_scaled, b, m_maxima, _ = _gpmo_problem(seed=2575, M=7, N=4)
        pol_vectors = _gpmo_pol_vectors(seed=2576, N=m_maxima.size, P=3)
        spec = GPMOArbVecSpec(
            m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
            reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
            pol_vectors=jnp.asarray(pol_vectors, dtype=jnp.float64),
        )
        A_device = jax.device_put(jnp.asarray(A_scaled))
        b_device = jax.device_put(jnp.asarray(b))

        @jax.jit
        def _run(spec_data: GPMOArbVecSpec, A_data: jax.Array, b_data: jax.Array):
            return gpmo_arbvec_solve(spec_data, A_data, b_data, K=2).x

        _run(spec, A_device, b_device).block_until_ready()
        with jax.transfer_guard("disallow"):
            out = _run(spec, A_device, b_device)
            out.block_until_ready()

        assert out.shape == (4, 3)
        assert np.all(np.isfinite(np.asarray(out)))


class TestGPMOArbVecBacktracking:
    def test_initialize_gpmo_arbvec_matches_cpp_oracle(self):
        """``initialize_gpmo_arbvec`` selects the nearest signed pol vector.

        The C++ ``initialize_GPMO_ArbVec`` (lines 994-1117) is purely a
        Python-side initialization routine; we exercise it indirectly via
        ``GPMO_ArbVec_backtracking`` with ``K=0`` so the C++ main loop never
        runs and the returned ``x`` is exactly the initialization assignment.
        """

        A_scaled, b, m_maxima, normal_norms = _gpmo_problem(seed=2600, M=10, N=6)
        pol_vectors = _gpmo_pol_vectors(seed=2601, N=m_maxima.size, P=4)
        rng = np.random.default_rng(2602)
        # Build an x_init that mixes exact, near, and ambiguous candidates.
        x_init = np.zeros((m_maxima.size, 3))
        x_init[0] = pol_vectors[0, 2]  # exact match -> sign=+1, m=2
        x_init[1] = -0.98 * pol_vectors[1, 1]  # near minus -> sign=-1, m=1
        x_init[2] = 0.05 * rng.standard_normal(size=3)  # tiny -> sign=0
        x_init[4] = 0.7 * pol_vectors[4, 0]  # ambiguous, sign=+1, m=0
        dipoles = np.ascontiguousarray(rng.standard_normal(size=(m_maxima.size, 3)))
        _, _, _, _, x_cpp = simsoptpp.GPMO_ArbVec_backtracking(
            np.ascontiguousarray(A_scaled.T),
            b,
            np.repeat(m_maxima, 3) * 0.0,
            normal_norms,
            pol_vectors,
            K=0,
            verbose=False,
            nhistory=1,
            backtracking=1,
            dipole_grid_xyz=dipoles,
            Nadjacent=1,
            thresh_angle=float(np.pi),
            max_nMagnets=m_maxima.size,
            x_init=np.ascontiguousarray(x_init),
        )
        x_jax, residual_jax, available_jax, _, _, num_nonzero = initialize_gpmo_arbvec(
            jnp.asarray(x_init),
            jnp.asarray(pol_vectors),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
        )
        np.testing.assert_allclose(
            np.asarray(x_jax),
            x_cpp,
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )
        # Running residual is ``A x - b`` for the initialization state.
        expected_residual = A_scaled @ np.asarray(x_jax).reshape(-1) - b
        np.testing.assert_allclose(
            np.asarray(residual_jax),
            expected_residual,
            rtol=_PER_KERNEL_RTOL,
            atol=_PER_KERNEL_ATOL,
        )
        # ``available`` is False exactly at placed dipoles.
        placed_jax = np.any(np.asarray(x_jax) != 0.0, axis=1)
        np.testing.assert_array_equal(
            np.asarray(available_jax),
            ~placed_jax,
        )
        assert int(num_nonzero) == int(np.sum(placed_jax))

    def test_gpmo_arbvec_backtracking_jax_matches_cpp_oracle(self):
        """JAX state-trace parity vs ``simsoptpp.GPMO_ArbVec_backtracking``.

        The fixture places exactly N dipoles and never triggers any
        dewyrming pass (thresh_angle = π → cos_thresh = −1, no pair removed).
        """

        rng = np.random.default_rng(2610)
        M, N, P = 8, 4, 3
        K = 3
        Nadjacent = 2
        backtracking = 2
        max_nMagnets = N
        thresh_angle = float(np.pi)
        reg_l2 = 0.0

        A_scaled = np.ascontiguousarray(rng.standard_normal(size=(M, 3 * N)))
        b = np.ascontiguousarray(rng.standard_normal(size=(M,)))
        m_maxima = np.ascontiguousarray(0.3 + rng.random(size=N))
        dipoles = np.ascontiguousarray(rng.standard_normal(size=(N, 3)))
        normal_norms = np.ones(M, dtype=np.float64)
        raw = rng.standard_normal(size=(N, P, 3))
        pol_vectors = np.ascontiguousarray(
            raw / np.linalg.norm(raw, axis=2)[:, :, None]
        )

        _, _, _, _, x_cpp = simsoptpp.GPMO_ArbVec_backtracking(
            np.ascontiguousarray(A_scaled.T),
            b,
            np.sqrt(reg_l2) * np.repeat(m_maxima, 3),
            normal_norms,
            pol_vectors,
            K=K,
            verbose=False,
            nhistory=K,
            backtracking=backtracking,
            dipole_grid_xyz=dipoles,
            Nadjacent=Nadjacent,
            thresh_angle=thresh_angle,
            max_nMagnets=max_nMagnets,
            x_init=np.zeros((N, 3)),
        )
        result = gpmo_arbvec_backtracking_solve(
            GPMOArbVecBacktrackingSpec(
                m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                reg_l2=jnp.asarray(reg_l2, dtype=jnp.float64),
                dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
                pol_vectors=jnp.asarray(pol_vectors, dtype=jnp.float64),
                Nadjacent=Nadjacent,
                backtracking=backtracking,
                thresh_angle=thresh_angle,
                max_nMagnets=max_nMagnets,
            ),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=K,
        )
        np.testing.assert_allclose(
            np.asarray(result.x),
            x_cpp,
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )

    def test_gpmo_arbvec_backtracking_jax_with_l2_regularization(self):
        """Parity in the angle-threshold-active regime with reg_l2 > 0."""

        rng = np.random.default_rng(2620)
        M, N, P = 14, 8, 3
        K = 12
        Nadjacent = 3
        backtracking = 2
        max_nMagnets = N + 10
        thresh_angle = float(np.pi / 4.0)
        reg_l2 = 0.2

        A_scaled = np.ascontiguousarray(rng.standard_normal(size=(M, 3 * N)))
        b = np.ascontiguousarray(rng.standard_normal(size=(M,)))
        m_maxima = np.ascontiguousarray(0.3 + rng.random(size=N))
        dipoles = np.ascontiguousarray(rng.standard_normal(size=(N, 3)))
        normal_norms = np.ones(M, dtype=np.float64)
        raw = rng.standard_normal(size=(N, P, 3))
        pol_vectors = np.ascontiguousarray(
            raw / np.linalg.norm(raw, axis=2)[:, :, None]
        )

        _, _, _, _, x_cpp = simsoptpp.GPMO_ArbVec_backtracking(
            np.ascontiguousarray(A_scaled.T),
            b,
            np.sqrt(reg_l2) * np.repeat(m_maxima, 3),
            normal_norms,
            pol_vectors,
            K=K,
            verbose=False,
            nhistory=K,
            backtracking=backtracking,
            dipole_grid_xyz=dipoles,
            Nadjacent=Nadjacent,
            thresh_angle=thresh_angle,
            max_nMagnets=max_nMagnets,
            x_init=np.zeros((N, 3)),
        )
        result = gpmo_arbvec_backtracking_solve(
            GPMOArbVecBacktrackingSpec(
                m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                reg_l2=jnp.asarray(reg_l2, dtype=jnp.float64),
                dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
                pol_vectors=jnp.asarray(pol_vectors, dtype=jnp.float64),
                Nadjacent=Nadjacent,
                backtracking=backtracking,
                thresh_angle=thresh_angle,
                max_nMagnets=max_nMagnets,
            ),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=K,
        )
        np.testing.assert_allclose(
            np.asarray(result.x),
            x_cpp,
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )
        # Aggressive thresh_angle must trigger at least one dewyrming pass.
        assert np.max(np.asarray(result.removed_pair_count_history)) > 0

    def test_gpmo_arbvec_backtracking_jax_with_nonzero_x_init(self):
        """Parity when ``x_init`` seeds the solver via the init routine."""

        rng = np.random.default_rng(2630)
        M, N, P = 12, 6, 4
        K = 6
        Nadjacent = 2
        backtracking = 2
        thresh_angle = float(np.pi)
        reg_l2 = 0.0

        A_scaled = np.ascontiguousarray(rng.standard_normal(size=(M, 3 * N)))
        b = np.ascontiguousarray(rng.standard_normal(size=(M,)))
        m_maxima = np.ascontiguousarray(0.3 + rng.random(size=N))
        dipoles = np.ascontiguousarray(rng.standard_normal(size=(N, 3)))
        normal_norms = np.ones(M, dtype=np.float64)
        raw = rng.standard_normal(size=(N, P, 3))
        pol_vectors = np.ascontiguousarray(
            raw / np.linalg.norm(raw, axis=2)[:, :, None]
        )

        x_init = np.zeros((N, 3))
        x_init[1] = pol_vectors[1, 0]  # exact match
        x_init[3] = -0.97 * pol_vectors[3, 2]  # near minus
        x_init = np.ascontiguousarray(x_init)
        max_nMagnets = N

        _, _, _, _, x_cpp = simsoptpp.GPMO_ArbVec_backtracking(
            np.ascontiguousarray(A_scaled.T),
            b,
            np.sqrt(reg_l2) * np.repeat(m_maxima, 3),
            normal_norms,
            pol_vectors,
            K=K,
            verbose=False,
            nhistory=K,
            backtracking=backtracking,
            dipole_grid_xyz=dipoles,
            Nadjacent=Nadjacent,
            thresh_angle=thresh_angle,
            max_nMagnets=max_nMagnets,
            x_init=x_init,
        )
        result = gpmo_arbvec_backtracking_solve(
            GPMOArbVecBacktrackingSpec(
                m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
                reg_l2=jnp.asarray(reg_l2, dtype=jnp.float64),
                dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
                pol_vectors=jnp.asarray(pol_vectors, dtype=jnp.float64),
                Nadjacent=Nadjacent,
                backtracking=backtracking,
                thresh_angle=thresh_angle,
                max_nMagnets=max_nMagnets,
            ),
            jnp.asarray(A_scaled),
            jnp.asarray(b),
            K=K,
            x_init=jnp.asarray(x_init),
        )
        np.testing.assert_allclose(
            np.asarray(result.x),
            x_cpp,
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )
        # Initial state should reflect the two seeded dipoles.
        assert int(result.initial_num_nonzero) == 2

    def test_gpmo_arbvec_backtracking_jits_under_strict_transfer_guard(self):
        """Solver compiles and executes under strict device-to-host guard."""

        rng = np.random.default_rng(2640)
        M, N, P = 9, 5, 3
        A_scaled, b, m_maxima, _, dipoles = _gpmo_spatial_problem(seed=2641, M=M, N=N)
        raw = rng.standard_normal(size=(N, P, 3))
        pol_vectors = np.ascontiguousarray(
            raw / np.linalg.norm(raw, axis=2)[:, :, None]
        )
        spec = GPMOArbVecBacktrackingSpec(
            m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
            reg_l2=jnp.asarray(0.0, dtype=jnp.float64),
            dipole_grid_xyz=jnp.asarray(dipoles, dtype=jnp.float64),
            pol_vectors=jnp.asarray(pol_vectors, dtype=jnp.float64),
            Nadjacent=2,
            backtracking=2,
            thresh_angle=float(np.pi / 3.0),
            max_nMagnets=4,
        )
        A_device = jax.device_put(jnp.asarray(A_scaled))
        b_device = jax.device_put(jnp.asarray(b))

        @jax.jit
        def _run(
            spec_data: GPMOArbVecBacktrackingSpec,
            A_data: jax.Array,
            b_data: jax.Array,
        ):
            return gpmo_arbvec_backtracking_solve(spec_data, A_data, b_data, K=6).x

        _run(spec, A_device, b_device).block_until_ready()
        with jax.transfer_guard("disallow"):
            out = _run(spec, A_device, b_device)
            out.block_until_ready()

        assert out.shape == (N, 3)
        assert np.all(np.isfinite(np.asarray(out)))


# ---------------------------------------------------------------------
# Algorithm-level checks: monotonicity, optimality, C++ parity.
# ---------------------------------------------------------------------


class TestMwPGPSolver:
    def test_cost_monotone_decreasing(self):
        """Cost is non-increasing iterate-to-iterate (modulo FP noise)."""
        A, b, m_maxima, m_proxy, m0 = _random_problem(
            seed=7, M=40, N=8, m_maxima_scale=0.5
        )
        # Step size: 1 / largest eigenvalue of A^T A (conservative).
        s = np.linalg.svd(A, compute_uv=False)
        alpha = 1.0 / (s[0] ** 2)
        spec = _make_spec(m_maxima, m_proxy, alpha=alpha, reg_l2=0.0, nu=1.0e100)
        ATb = (A.T @ b).reshape(m_maxima.size, 3)

        n_steps = 50
        _, _ = mwpgp_solve(
            spec,
            jnp.asarray(A),
            jnp.asarray(ATb),
            jnp.asarray(m0),
            n_steps=n_steps,
        )
        # Reconstruct per-iteration ``x`` via the single-step API and
        # verify the full convex objective is non-increasing iterate to
        # iterate. ``residual_history`` from the scan is only the
        # ``m^T A^T A m - 2 m^T ATb`` proxy; the full cost includes
        # ``||b||^2`` and the relax-and-split / L2 contributions. Tests
        # check the full cost.
        full_costs = []
        full_costs.append(_cost(A, b, m_proxy, m_maxima, 1.0e100, 0.0, m0))

        state = mwpgp_initial_state(
            spec,
            jnp.asarray(A),
            jnp.asarray(ATb),
            jnp.asarray(m0),
        )
        for _ in range(n_steps):
            state = mwpgp_step(
                spec,
                state,
                jnp.asarray(A),
                jnp.asarray(ATb),
            )
            full_costs.append(
                _cost(A, b, m_proxy, m_maxima, 1.0e100, 0.0, np.asarray(state[0]))
            )

        full_costs = np.asarray(full_costs)
        diffs = np.diff(full_costs)
        # Allow a tiny positive tolerance because the optimization may
        # also take an "expand" step that uses ``alpha_f`` (not
        # guaranteed to decrease at the line-search level if ``alpha``
        # exceeds the safe step). We use ``rtol`` against the initial
        # cost which is ``0.5 ||b||^2`` here.
        scale = full_costs[0]
        tol = _MONOTONICITY_RTOL * max(scale, 1.0)
        assert np.all(diffs <= tol), (
            f"Cost increased somewhere: max diff = {diffs.max()}, tol={tol}"
        )

    def test_optimality_recovers_unconstrained_minimiser_in_interior(self):
        """When the LS minimiser lives inside the L2 balls, mwpgp_solve
        recovers it.
        """
        # Solve ``A m* = b`` with M >= 3N, then build the problem so
        # ``m*`` lives inside the balls.
        rng = np.random.default_rng(123)
        N = 6
        M = 3 * N + 10  # well-conditioned (rectangular)
        A = rng.standard_normal(size=(M, 3 * N))
        # Pick a small ground-truth ``m_true`` clearly inside unit balls.
        m_true = 0.3 * rng.standard_normal(size=(N, 3))
        # Ensure each |m_i| <= 0.6, so the unit balls are non-binding.
        m_true = np.clip(m_true, -0.4, 0.4)
        b = A @ m_true.reshape(-1)

        m_maxima = np.full(N, 1.0)
        m_proxy = np.zeros((N, 3))
        m0 = np.zeros((N, 3))

        s = np.linalg.svd(A, compute_uv=False)
        # Slightly conservative step size for the projected-gradient
        # contraction: alpha = 1 / sigma_max(A)^2.
        alpha = 1.0 / (s[0] ** 2)

        spec = _make_spec(m_maxima, m_proxy, alpha=alpha, reg_l2=0.0, nu=1.0e100)
        ATb = (A.T @ b).reshape(N, 3)
        n_steps = 2000

        m_final, _ = mwpgp_solve(
            spec,
            jnp.asarray(A),
            jnp.asarray(ATb),
            jnp.asarray(m0),
            n_steps=n_steps,
        )
        m_final_np = np.asarray(m_final)

        # Residual should be tiny (problem is rectangular consistent).
        resid = np.linalg.norm(A @ m_final_np.reshape(-1) - b)
        # The unconstrained LS minimum residual here is zero (we built
        # ``b = A m_true``), so the projected-gradient iterate should
        # close in to within a tight tolerance.
        assert resid < _OPTIMALITY_ATOL, (
            f"residual {resid} exceeded _OPTIMALITY_ATOL {_OPTIMALITY_ATOL}"
        )

    def test_cpp_oracle_parity_state_trace(self):
        """5-iteration state trace matches simsoptpp.MwPGP_algorithm.

        Use a small, well-conditioned problem. The C++ kernel snapshots
        a history every ``max_iter / 5`` iterations and may break out on
        ``min_fb``; we side-step both by setting ``min_fb=0`` and
        running ``max_iter=5`` so the early-exit at line 318 (``x_sum <
        epsilon``) requires drift in the per-iteration ``x_k1`` that we
        avoid by setting ``epsilon=0``.
        """
        rng = np.random.default_rng(2026)
        N = 4
        M = 12
        A = rng.standard_normal(size=(M, 3 * N))
        b = rng.standard_normal(size=(M,))
        m_maxima = np.full(N, 0.4)
        m_proxy = np.zeros((N, 3))
        m0 = np.zeros((N, 3))
        ATb = (A.T @ b).reshape(N, 3)
        s = np.linalg.svd(A, compute_uv=False)
        alpha = 1.0 / (s[0] ** 2)
        n_steps = 5

        # C++ run with verbose=False, epsilon=0, min_fb=0.
        # NOTE: ``epsilon=0`` would still trip the convergence check at
        # the very first iteration where ``x_k1 == x_k_prev``? The C++
        # convergence test is ``x_sum < epsilon`` (strict-less-than). With
        # ``epsilon = 0`` and ``x_sum = 0``, ``0 < 0`` is false so the
        # loop continues. Good.
        _, _, _, x_cpp = simsoptpp.MwPGP_algorithm(
            A,
            b,
            ATb,
            m_proxy,
            m0,
            m_maxima,
            alpha,  # alpha
            1.0e100,  # nu
            0.0,  # epsilon (no early break)
            0.0,  # reg_l0
            0.0,  # reg_l1
            0.0,  # reg_l2
            n_steps,  # max_iter
            0.0,  # min_fb
            False,  # verbose
        )

        spec = _make_spec(m_maxima, m_proxy, alpha=alpha, reg_l2=0.0, nu=1.0e100)
        m_final, _ = mwpgp_solve(
            spec,
            jnp.asarray(A),
            jnp.asarray(ATb),
            jnp.asarray(m0),
            n_steps=n_steps,
        )
        np.testing.assert_allclose(
            np.asarray(m_final),
            np.asarray(x_cpp),
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
            err_msg="Final iterate diverged from C++ oracle.",
        )

    def test_cpp_oracle_parity_with_l2_regularization(self):
        """C++ parity with a nonzero reg_l2 weight."""
        rng = np.random.default_rng(11)
        N = 5
        M = 18
        A = rng.standard_normal(size=(M, 3 * N))
        b = rng.standard_normal(size=(M,))
        m_maxima = np.full(N, 0.6)
        m_proxy = np.zeros((N, 3))
        m0 = np.zeros((N, 3))
        ATb = (A.T @ b).reshape(N, 3)
        reg_l2 = 0.1
        # Step size includes the L2 regularizer contribution to the
        # effective Hessian. ``H = A^T A + 2 reg_l2 I`` so
        # ``sigma_max(H) = sigma_max(A)^2 + 2 reg_l2``.
        s = np.linalg.svd(A, compute_uv=False)
        alpha = 1.0 / (s[0] ** 2 + 2.0 * reg_l2)
        n_steps = 10

        _, _, _, x_cpp = simsoptpp.MwPGP_algorithm(
            A,
            b,
            ATb,
            m_proxy,
            m0,
            m_maxima,
            alpha,
            1.0e100,
            0.0,
            0.0,
            0.0,
            reg_l2,
            n_steps,
            0.0,
            False,
        )

        spec = _make_spec(m_maxima, m_proxy, alpha=alpha, reg_l2=reg_l2, nu=1.0e100)
        m_final, _ = mwpgp_solve(
            spec,
            jnp.asarray(A),
            jnp.asarray(ATb),
            jnp.asarray(m0),
            n_steps=n_steps,
        )
        np.testing.assert_allclose(
            np.asarray(m_final),
            np.asarray(x_cpp),
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )

    def test_cpp_oracle_parity_with_relax_and_split(self):
        """C++ parity with finite ``nu`` (relax-and-split active)."""
        rng = np.random.default_rng(99)
        N = 4
        M = 16
        A = rng.standard_normal(size=(M, 3 * N))
        b = rng.standard_normal(size=(M,))
        m_maxima = np.full(N, 0.5)
        m_proxy = 0.1 * rng.standard_normal(size=(N, 3))
        m0 = np.zeros((N, 3))
        ATb = (A.T @ b).reshape(N, 3)
        nu = 5.0
        s = np.linalg.svd(A, compute_uv=False)
        alpha = 1.0 / (s[0] ** 2 + 1.0 / nu)
        n_steps = 8

        _, _, _, x_cpp = simsoptpp.MwPGP_algorithm(
            A,
            b,
            ATb,
            m_proxy,
            m0,
            m_maxima,
            alpha,
            nu,
            0.0,
            0.0,
            0.0,
            0.0,
            n_steps,
            0.0,
            False,
        )

        spec = _make_spec(m_maxima, m_proxy, alpha=alpha, reg_l2=0.0, nu=nu)
        m_final, _ = mwpgp_solve(
            spec,
            jnp.asarray(A),
            jnp.asarray(ATb),
            jnp.asarray(m0),
            n_steps=n_steps,
        )
        np.testing.assert_allclose(
            np.asarray(m_final),
            np.asarray(x_cpp),
            rtol=_STATE_TRACE_RTOL,
            atol=_STATE_TRACE_ATOL,
        )

    def test_solver_jits(self):
        """``mwpgp_solve`` survives ``jax.jit`` with ``n_steps`` as static."""
        rng = np.random.default_rng(2)
        N = 3
        M = 10
        A = rng.standard_normal(size=(M, 3 * N))
        b = rng.standard_normal(size=(M,))
        m_maxima = np.full(N, 0.5)
        m_proxy = np.zeros((N, 3))
        m0 = np.zeros((N, 3))
        ATb = (A.T @ b).reshape(N, 3)
        s = np.linalg.svd(A, compute_uv=False)
        alpha = 1.0 / (s[0] ** 2)

        spec = _make_spec(m_maxima, m_proxy, alpha=alpha, reg_l2=0.0, nu=1.0e100)

        @jax.jit
        def _wrapped(spec, A, ATb, m0):
            m, _ = mwpgp_solve(spec, A, ATb, m0, n_steps=4)
            return m

        m_final = _wrapped(spec, jnp.asarray(A), jnp.asarray(ATb), jnp.asarray(m0))
        assert m_final.shape == (N, 3)
        assert np.all(np.isfinite(np.asarray(m_final)))

    def test_zero_steps_returns_m0(self):
        """``n_steps=0`` returns ``m0`` unchanged."""
        N = 3
        m0 = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [-0.1, 0.1, -0.1]])
        m_maxima = np.full(N, 1.0)
        m_proxy = np.zeros((N, 3))
        rng = np.random.default_rng(0)
        A = rng.standard_normal(size=(9, 3 * N))
        b = rng.standard_normal(size=(9,))
        ATb = (A.T @ b).reshape(N, 3)

        spec = _make_spec(m_maxima, m_proxy, alpha=0.01)
        m_final, history = mwpgp_solve(
            spec,
            jnp.asarray(A),
            jnp.asarray(ATb),
            jnp.asarray(m0),
            n_steps=0,
        )
        np.testing.assert_array_equal(np.asarray(m_final), m0)
        assert history.shape == (0,)


# ---------------------------------------------------------------------
# Single-iteration parity (the building block).
# ---------------------------------------------------------------------


class TestMwPGPSingleStep:
    def test_step_matches_solver_with_one_iter(self):
        """``mwpgp_step`` repeated == ``mwpgp_solve(n_steps=k)``."""
        rng = np.random.default_rng(8)
        N = 3
        M = 9
        A = rng.standard_normal(size=(M, 3 * N))
        b = rng.standard_normal(size=(M,))
        m_maxima = np.full(N, 0.5)
        m_proxy = np.zeros((N, 3))
        m0 = np.zeros((N, 3))
        ATb = (A.T @ b).reshape(N, 3)
        s = np.linalg.svd(A, compute_uv=False)
        alpha = 1.0 / (s[0] ** 2)

        spec = _make_spec(m_maxima, m_proxy, alpha=alpha, reg_l2=0.0, nu=1.0e100)
        k = 6
        m_solve, _ = mwpgp_solve(
            spec,
            jnp.asarray(A),
            jnp.asarray(ATb),
            jnp.asarray(m0),
            n_steps=k,
        )

        state = mwpgp_initial_state(
            spec,
            jnp.asarray(A),
            jnp.asarray(ATb),
            jnp.asarray(m0),
        )
        for _ in range(k):
            state = mwpgp_step(spec, state, jnp.asarray(A), jnp.asarray(ATb))
        m_step = state[0]
        np.testing.assert_allclose(
            np.asarray(m_step),
            np.asarray(m_solve),
            rtol=_SINGLE_STEP_RTOL,
            atol=_SINGLE_STEP_ATOL,
        )

    def test_step_body_uses_dynamic_branch_conditionals(self):
        """MwPGP branches through ``lax.cond`` instead of eager ``select``."""

        A, b, m_maxima, m_proxy, m0 = _random_problem(seed=25025, M=7, N=2)
        ATb = (A.T @ b).reshape(2, 3)
        spec = _make_spec(m_maxima, m_proxy, alpha=0.05, reg_l2=0.0, nu=1.0e100)
        A_jax = jnp.asarray(A, dtype=jnp.float64)
        ATb_jax = jnp.asarray(ATb, dtype=jnp.float64)
        state = mwpgp_initial_state(
            spec,
            A_jax,
            ATb_jax,
            jnp.asarray(m0, dtype=jnp.float64),
        )

        jaxpr = jax.make_jaxpr(lambda st: mwpgp_step(spec, st, A_jax, ATb_jax))(state)
        assert count_jaxpr_primitives(jaxpr, "cond") == 2
