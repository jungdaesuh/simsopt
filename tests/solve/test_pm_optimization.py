import dataclasses
import math
from pathlib import Path
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from monty.tempfile import ScratchDir

import simsoptpp as sopp
from simsopt.solve.permanent_magnet_optimization import prox_l0, prox_l1
from simsopt.solve.permanent_magnet_optimization import setup_initial_condition
from simsopt.solve import relax_and_split, GPMO
from simsopt.util import *
from simsopt.geo import SurfaceRZFourier, PermanentMagnetGrid
from simsopt.field import BiotSavart
from simsopt_jax.core import pm_optimization
from simsopt_jax.core.pm_optimization import (
    GPMOArbVecBacktrackingSpec,
    gpmo_arbvec_backtracking_solve,
    gpmo_arbvec_backtracking_step,
    gpmo_connectivity_matrix,
    initialize_gpmo_arbvec,
)

# Iteration budget for the ArbVec-backtracking post-freeze-skip fixture.
_ARBVEC_BACKTRACKING_K = 24


def _arbvec_backtracking_fixture():
    """Small ArbVec-backtracking problem that dewyrms and then freezes early.

    ``max_nMagnets=6`` is reached at iteration 11 of ``_ARBVEC_BACKTRACKING_K``,
    so the last 12 iterations run against an already-terminated state, and
    ``thresh_angle=pi/2`` makes the dewyrming pass remove pairs before then.
    """

    rng = np.random.default_rng(7)
    M, N, P = 14, 8, 3
    A_scaled = np.ascontiguousarray(rng.standard_normal(size=(M, 3 * N)))
    b = np.ascontiguousarray(rng.standard_normal(size=(M,)))
    m_maxima = np.ascontiguousarray(0.3 + rng.random(size=N))
    dipole_grid_xyz = np.ascontiguousarray(rng.standard_normal(size=(N, 3)))
    raw = rng.standard_normal(size=(N, P, 3))
    pol_vectors = np.ascontiguousarray(raw / np.linalg.norm(raw, axis=2)[:, :, None])
    spec = GPMOArbVecBacktrackingSpec(
        m_maxima=jnp.asarray(m_maxima, dtype=jnp.float64),
        reg_l2=jnp.asarray(0.2, dtype=jnp.float64),
        dipole_grid_xyz=jnp.asarray(dipole_grid_xyz, dtype=jnp.float64),
        pol_vectors=jnp.asarray(pol_vectors, dtype=jnp.float64),
        Nadjacent=3,
        backtracking=3,
        thresh_angle=float(np.pi / 2.0),
        max_nMagnets=6,
    )
    return spec, jnp.asarray(A_scaled), jnp.asarray(b)


def _solve_arbvec_backtracking(spec, A_scaled, b, *, record_every, post_freeze_skip):
    """Solve with, or without, the post-freeze skip inside the step.

    ``post_freeze_skip=False`` restores the pre-change semantics by routing the
    solver straight at the always-scan branch. ``gpmo_arbvec_backtracking_solve``
    is jitted, so the caches have to be dropped for the patch to be traced.
    """

    jax.clear_caches()
    try:
        if post_freeze_skip:
            return jax.block_until_ready(
                gpmo_arbvec_backtracking_solve(
                    spec,
                    A_scaled,
                    b,
                    K=_ARBVEC_BACKTRACKING_K,
                    record_every=record_every,
                )
            )
        with mock.patch.object(
            pm_optimization,
            "gpmo_arbvec_backtracking_step",
            pm_optimization._gpmo_arbvec_backtracking_active_step,
        ):
            return jax.block_until_ready(
                gpmo_arbvec_backtracking_solve(
                    spec,
                    A_scaled,
                    b,
                    K=_ARBVEC_BACKTRACKING_K,
                    record_every=record_every,
                )
            )
    finally:
        jax.clear_caches()


def _raw_bytes(array):
    host = np.ascontiguousarray(np.atleast_1d(np.asarray(jax.device_get(array))))
    return host.view(np.uint8)


def _primitive_names(jaxpr):
    """Every primitive reachable from ``jaxpr``, including nested sub-jaxprs."""

    names = set()
    pending = [jaxpr]
    while pending:
        current = pending.pop()
        for equation in current.eqns:
            names.add(equation.primitive.name)
            for parameter in equation.params.values():
                values = (
                    parameter if isinstance(parameter, (tuple, list)) else (parameter,)
                )
                for value in values:
                    inner = getattr(value, "jaxpr", value)
                    if hasattr(inner, "eqns"):
                        pending.append(inner)
    return names


class Testing(unittest.TestCase):

    def test_prox(self):
        m = np.random.rand(3000)
        mmax = np.ones(1000)
        reg_l0 = 0.5
        nu = 0.5
        m_thresholded = prox_l0(m, mmax, reg_l0, nu)
        m_thresholded = m_thresholded[~np.isclose(m_thresholded, 0.0)]
        assert np.all(m_thresholded >= 0.5)
        nu = 1
        m_thresholded = prox_l1(m, mmax, reg_l0, nu)
        assert np.linalg.norm(m_thresholded) < np.linalg.norm(m)

    def test_MwPGP(self):
        """ 
            Test the MwPGP algorithm for solving the convex
            part of the permanent magnet problem. 
        """
        ndipoles = 100
        nquad = 512
        max_iter = 100
        m_maxima = np.random.rand(ndipoles) * 10
        m0 = np.zeros((ndipoles, 3))
        b = np.random.rand(nquad)
        A = np.random.rand(nquad, ndipoles, 3)
        ATA = np.tensordot(A, A, axes=([1, 1]))
        alpha = 2.0 / np.linalg.norm(ATA.reshape(nquad * 3, nquad * 3), ord=2)
        ATb = np.tensordot(A, b, axes=([0, 0]))
        with ScratchDir("."):
            MwPGP_hist, RS_hist, m_hist, dipoles = sopp.MwPGP_algorithm(
                A_obj=A, b_obj=b, ATb=ATb, m_proxy=m0, m0=m0, m_maxima=m_maxima,
                alpha=alpha, nu=1e100, epsilon=1e-4, max_iter=max_iter,  # verbose=True,
                reg_l0=0.0, reg_l1=0.0, reg_l2=0.0)
            m_hist = np.array(m_hist)
            assert dipoles.shape == (ndipoles, 3)
            assert m_hist.shape == (ndipoles, 3, 21)

    def test_algorithms(self):
        """ 
            Test the relax and split algorithm for solving
            the permanent magnet problem. Test the GPMO
            algorithm variants in the limit that they should
            all produce the same solution.
        """
        nphi = 8  # nphi = ntheta >= 64 needed for accurate full-resolution runs
        ntheta = 8
        dr = 0.04  # cylindrical bricks with radial extent 4 cm
        coff = 0.1  # PM grid starts offset ~ 10 cm from the plasma surface
        poff = 0.05  # PM grid end offset ~ 15 cm from the plasma surface
        input_name = 'input.LandremanPaul2021_QA_lowres'

        # Read in the plasma equilibrium file
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

        # Make the inner and outer surfaces by extending the plasma surface
        s_inner.extend_via_projected_normal(poff)
        s_outer.extend_via_projected_normal(poff + coff)

        # optimize the currents in the TF coils
        with ScratchDir("."):
            base_curves, curves, coils = initialize_coils_for_pm_optimization('qa', TEST_DIR, s)
            bs = BiotSavart(coils)
            bs = coil_optimization(s, bs, base_curves, curves)
            bs.set_points(s.gamma().reshape((-1, 3)))
            Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

            kwargs_geo = {"dr": dr}
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                s, Bnormal, s_inner, s_outer, **kwargs_geo
            )
            setup_initial_condition(pm_opt, np.zeros(pm_opt.ndipoles * 3))

            reg_l0 = 0.05  # Threshold off magnets with 5% or less strength
            nu = 1e10  # how strongly to make proxy variable w close to values in m

            # Rescale the hyperparameters and then add contributions to ATA and ATb
            reg_l0, _, _, nu = pm_opt.rescale_for_opt(reg_l0, 0.0, 0.0, nu)

            # Set some hyperparameters for the optimization
            kwargs = initialize_default_kwargs()
            kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
            kwargs['max_iter'] = 40  # Number of iterations to take in a convex step
            kwargs['max_iter_RS'] = 20  # Number of total iterations of the relax-and-split algorithm
            kwargs['reg_l0'] = reg_l0
            relax_and_split(pm_opt, **kwargs)
            w = pm_opt.m_proxy[~np.isclose(pm_opt.m_proxy, 0.0)]
            assert np.all(np.abs(w) >= reg_l0 * pm_opt.m_maxima[0])

            # Try again with more aggressive thresholding
            reg_l0 = 0.5  # Threshold off magnets with 50% or less strength
            nu = 1e10  # how strongly to make proxy variable w close to values in m

            # Rescale the hyperparameters and then add contributions to ATA and ATb
            reg_l0, _, _, nu = pm_opt.rescale_for_opt(reg_l0, 0.0, 0.0, nu)

            # Set some hyperparameters for the optimization
            kwargs = initialize_default_kwargs()
            kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
            kwargs['reg_l0'] = reg_l0
            relax_and_split(pm_opt, **kwargs)
            w = pm_opt.m_proxy[~np.isclose(pm_opt.m_proxy, 0.0)]
            assert np.all(np.abs(w) >= reg_l0 * pm_opt.m_maxima[0])
            kwargs['reg_l1'] = reg_l0
            with self.assertRaises(ValueError):
                relax_and_split(pm_opt, **kwargs)
            kwargs['reg_l0'] = 0.0
            kwargs['epsilon_RS'] = 1e5
            relax_and_split(pm_opt, **kwargs)

            # Test that all the GPMO variants return the same solutions
            # in various limits.
            kwargs = initialize_default_kwargs('GPMO')
            with self.assertRaises(ValueError):
                GPMO(pm_opt, algorithm='baseline', **kwargs)
            kwargs['nhistory'] = 10
            kwargs['K'] = 10
            errors1, Bn_errors1, m_history1 = GPMO(pm_opt, algorithm='baseline', **kwargs)
            m1 = pm_opt.m
            ndipoles = pm_opt.ndipoles
            pol_vector_x = np.zeros((ndipoles, 3))
            pol_vector_x[:, 0] = 1.0
            pol_vector_y = np.zeros((ndipoles, 3))
            pol_vector_y[:, 1] = 1.0
            pol_vector_z = np.zeros((ndipoles, 3))
            pol_vector_z[:, 2] = 1.0
            pol_vectors = np.transpose(np.array([pol_vector_x, pol_vector_y, pol_vector_z]), [1, 0, 2])
            pm_opt.pol_vectors = pol_vectors
            errors2, Bn_errors2, m_history2 = GPMO(pm_opt, algorithm='ArbVec', **kwargs)
            m2 = pm_opt.m
            assert np.allclose(m1, m2)
            assert np.allclose(errors1, errors2)
            assert np.allclose(Bn_errors1, Bn_errors2)
            assert np.allclose(m_history1, m_history2)
            kwargs['Nadjacent'] = 1
            kwargs['dipole_grid_xyz'] = pm_opt.dipole_grid_xyz
            errors3, Bn_errors3, m_history3 = GPMO(pm_opt, algorithm='multi', **kwargs)
            m3 = pm_opt.m
            assert np.allclose(m1, m3)
            assert np.allclose(errors1, errors3)
            assert np.allclose(Bn_errors1, Bn_errors3)
            assert np.allclose(m_history1, m_history3)
            kwargs['backtracking'] = 500
            kwargs['max_nMagnets'] = 1000
            errors4, Bn_errors4, m_history4 = GPMO(pm_opt, algorithm='backtracking', **kwargs)
            m4 = pm_opt.m
            assert np.allclose(m1, m4)
            assert np.allclose(errors1, errors4)
            assert np.allclose(Bn_errors1, Bn_errors4)
            assert np.allclose(m_history1, m_history4)

            # Note: ArbVec_backtracking history arrays contain one additional
            # entry at the beginning for the initialized solution

            errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)
            m5 = pm_opt.m
            assert np.allclose(m1, m5)
            assert np.allclose(errors1, errors5[1:])
            assert np.allclose(Bn_errors1, Bn_errors5[1:])
            assert np.allclose(m_history1, m_history5[:, :, 1:])
            with self.assertRaises(ValueError):
                pm_opt.coordinate_flag = 'cylindrical'
                errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)
            with self.assertRaises(NotImplementedError):
                errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='random_name', **kwargs)

            kwargs['m_init'] = pm_opt.m.reshape([-1, 3])
            pm_opt.coordinate_flag = 'cartesian'
            errors6, Bn_errors6, m_history6 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)
            # Note: when K = n_history, m_history[:,:,-1] will be zeros
            assert np.allclose(m_history5[:, :, -2], m_history6[:, :, 0])
            with self.assertRaises(ValueError):
                kwargs['m_init'] = m_history6[:-1, :, -1]
                errors6, Bn_errors6, m_history6 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)


class TestGPMOArbVecBacktrackingPostFreezeSkip(unittest.TestCase):
    """Post-freeze skip in ``gpmo_arbvec_backtracking_step``.

    Once the run reports ``done`` the greedy loop is a fixed point, so the step
    takes a cheap branch instead of re-scanning every candidate. Two properties
    make that safe, and each gets a test: the results stay bitwise identical to
    always scanning, and the expensive work really does sit behind the branch.
    """

    def test_outputs_are_bitwise_identical_to_always_scanning(self):
        spec, A_scaled, b = _arbvec_backtracking_fixture()
        for record_every in (None, 4):
            with self.subTest(record_every=record_every):
                reference = _solve_arbvec_backtracking(
                    spec, A_scaled, b, record_every=record_every, post_freeze_skip=False
                )
                skipped = _solve_arbvec_backtracking(
                    spec, A_scaled, b, record_every=record_every, post_freeze_skip=True
                )
                for field in dataclasses.fields(skipped):
                    skipped_value = getattr(skipped, field.name)
                    reference_value = getattr(reference, field.name)
                    self.assertEqual(
                        np.asarray(jax.device_get(skipped_value)).dtype,
                        np.asarray(jax.device_get(reference_value)).dtype,
                        f"{field.name} changed dtype under the post-freeze skip",
                    )
                    self.assertTrue(
                        np.array_equal(
                            _raw_bytes(skipped_value), _raw_bytes(reference_value)
                        ),
                        f"{field.name} is not bitwise identical to always scanning",
                    )

        # Guard against a vacuous comparison: the fixture must actually reach
        # the frozen regime, and must exercise the dewyrming pass before it.
        full_trace = _solve_arbvec_backtracking(
            spec, A_scaled, b, record_every=None, post_freeze_skip=True
        )
        done_history = np.asarray(jax.device_get(full_trace.done_history))
        removed = np.asarray(jax.device_get(full_trace.removed_pair_count_history))
        self.assertTrue(
            bool(done_history[:-1].any()),
            "fixture never freezes, so the skip is never taken",
        )
        self.assertGreater(int(removed.sum()), 0, "fixture never runs a dewyrming pass")

    def test_frozen_branch_omits_the_candidate_scan(self):
        spec, A_scaled, b = _arbvec_backtracking_fixture()
        ndipoles = int(spec.m_maxima.shape[0])
        x0, residual0, available0, vector_indices0, signs0, _ = initialize_gpmo_arbvec(
            jnp.zeros((ndipoles, 3), dtype=A_scaled.dtype),
            spec.pol_vectors,
            A_scaled,
            b,
        )
        selected0 = jnp.full((_ARBVEC_BACKTRACKING_K,), -1, dtype=vector_indices0.dtype)
        state = (
            x0,
            residual0,
            available0,
            vector_indices0,
            signs0,
            selected0,
            selected0,
            jnp.zeros((_ARBVEC_BACKTRACKING_K,), dtype=x0.dtype),
            jnp.asarray(False),
        )
        connectivity = gpmo_connectivity_matrix(spec.dipole_grid_xyz)
        cos_thresh_angle = jnp.asarray(math.cos(spec.thresh_angle), dtype=x0.dtype)

        jaxpr = jax.make_jaxpr(
            lambda step_state: gpmo_arbvec_backtracking_step(
                spec,
                step_state,
                A_scaled,
                connectivity,
                cos_thresh_angle,
                jnp.asarray(0, dtype=jnp.int32),
            )
        )(state).jaxpr
        # The candidate scan is an einsum (``dot_general``) and the dewyrming
        # pass is a ``lax.scan``. Neither may appear outside the branch, or a
        # frozen iteration would still pay for it.
        unconditional = {eqn.primitive.name for eqn in jaxpr.eqns}
        self.assertFalse(
            unconditional & {"dot_general", "scan"},
            "step does work outside the done branch that a frozen iteration pays for",
        )
        conditionals = [eqn for eqn in jaxpr.eqns if eqn.primitive.name == "cond"]
        self.assertEqual(
            len(conditionals), 1, "step no longer branches on the done flag"
        )
        branches = [
            _primitive_names(branch.jaxpr)
            for branch in conditionals[0].params["branches"]
        ]
        # Exactly one branch carries both, and the other -- the one taken once
        # ``done`` holds -- neither.
        scanning = [names for names in branches if {"dot_general", "scan"} <= names]
        cheap = [names for names in branches if not names & {"dot_general", "scan"}]
        self.assertEqual(
            len(scanning), 1, "no branch runs the candidate scan and dewyrming pass"
        )
        self.assertEqual(
            len(cheap), 1, "no branch skips the candidate scan and dewyrming pass"
        )


if __name__ == "__main__":
    unittest.main()
