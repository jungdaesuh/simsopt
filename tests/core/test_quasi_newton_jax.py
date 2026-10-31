"""``simsopt_jax.core.quasi_newton`` against SciPy 1.17's BFGS and L-BFGS-B.

Each solver runs beside ``scipy.optimize.minimize(jac=True)`` on a quadratic
(eigenvalues 1 to 1e3), the Rosenbrock function and an ill-conditioned
quartic (scales 1 to 1e6), SciPy evaluating the same JAX objective on the CPU.
Capped after k = 1, ..., 10 iterations, each solve stops at ``maxiter``
(status 1) with SciPy's message, iteration and evaluation counts, and its
iterate and ``hess_inv`` (BFGS's matrix; L-BFGS-B's correction pairs in the
order of SciPy's ring buffer) agree with SciPy's, relative to the largest
entry, to 1e-12, ``fun`` and ``jac`` to 1e-11 (up to k = 9 for L-BFGS-B on
the quadratic, see ``_EARLY_NUMERIC_ITERATIONS``). Capped, L-BFGS-B holds
three corrections, so its memory wraps. Run to convergence (ten corrections),
status, message, counts and ``x`` (to 1e-12) are SciPy's, ``fun`` and ``jac``
agree absolutely to 1e-14 and 1e-12, and ``hess_inv`` agrees
to 1e-7 after forty or more updates; L-BFGS-B's 100 to 200 iterations on the
quadratic and the quartic follow round-off instead (their counts change with
how XLA compiles the loop), and there the solutions agree to 1e-5, what
``gtol = 1e-6`` leaves two points with unit smallest curvature (measured worst
3.1e-7).

A gradient that contradicts the value fails every line search: BFGS ends with
SciPy's status 2 ("precision loss", after DCSRCH and then
``line_search_wolfe2`` fail), L-BFGS-B with "ABNORMAL", and with ``maxls=1``
L-BFGS-B also restarts from an empty memory before that, with SciPy's counts.
BFGS's fallbacks take SciPy's paths: past an infinite wall DCSRCH's step turns
NaN and ``scalar_search_wolfe2`` zooms; on a linear objective its ten
doublings end its loop; a first step below ``amin`` (gradients of 1e110)
stops DCSRCH at its start. A NaN wall ends BFGS with status 3, and
``maxiter=0`` ends BFGS at once and L-BFGS-B after one iteration, as in SciPy.
Corrections near 1e-100 underflow formt's products and reset L-BFGS-B's memory
as in SciPy (compared op by op, see that test). Invalid inputs end or raise as
in SciPy. Each case runs on the CPU and, when present, the GPU lane. The
zoom's interpolants and trial steps, which these objectives reach only partly,
are checked against SciPy's ``_cubicmin``, ``_quadmin`` and ``_zoom``
directly.
"""
from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.core.quasi_newton import (
        BFGS_MESSAGES,
        LBFGSB_MESSAGES,
        _cubicmin,
        _quadmin,
        _zoom_trial,
        minimize_bfgs,
        minimize_lbfgsb,
    )
from unittest_jax_support import (
    assert_matches_native,
    host_array,
    parity_default_device,
    parity_rng,
)

from functools import partial
import unittest
import warnings

import numpy as np
import scipy
from scipy.optimize import LbfgsInvHessProduct, OptimizeResult, minimize
from scipy.optimize import _linesearch as scipy_linesearch

# Early iterations: status, message and counts are compared up to k = 10;
# iterates, fun, jac and hess_inv up to k = 10 unless listed here (see
# test_early_iterates_match_scipy for why L-BFGS-B on the quadratic stops at 9).
_EARLY_ITERATIONS = 10
_EARLY_NUMERIC_ITERATIONS = {("L-BFGS-B", "quadratic"): 9}
_EARLY_RTOL = 1e-12
_FINAL_RTOL = 1e-5
_FAILURE_RTOL = 1e-12
_TINY_RTOL = 1e-14
# hess_inv after a converged walk: forty or more updates, each rounded
# differently (XLA's sums and fused multiply-adds); measured worst 1.1e-9 of the
# largest entry, BFGS on the Rosenbrock function.
_CONVERGED_HESS_INV_RTOL = 1e-7
# Measured worst early gaps (JAX 0.10.0 CPU and GPU, 0.10.2 CPU): iterates
# 3.9e-13, BFGS on the quartic on the GPU; fun and jac, which magnify an
# iterate's difference, 1.5e-12 of their size, L-BFGS-B on the quadratic at
# k = 9 on JAX 0.10.2.
_EARLY_FUN_JAC_RTOL = 10 * _EARLY_RTOL
# At a converged solution fun and jac are near zero, so they are compared
# absolutely (measured worst 3.3e-24 and 4.5e-14 where the walks match).
_CONVERGED_FUN_ATOL = 1e-14
_CONVERGED_JAC_ATOL = 1e-12
_ROUND_OFF_WALKS = frozenset(
    {
        # 180 to 200 iterations: early-iterate gaps grow from 1e-16 to 1e-1 by k = 110.
        ("L-BFGS-B", "quadratic"),
        # Its count changes with how XLA compiles the same loop (107 or 147; SciPy 109).
        ("L-BFGS-B", "ill-conditioned"),
    }
)
_GTOL = 1e-6
_LANES = ("cpu", "gpu")


def _quadratic(x, params):
    matrix, center = params
    r = x - center
    return 0.5 * r @ (matrix @ r)


def _rosenbrock(x, params):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def _ill_conditioned(x, params):
    return 0.5 * jnp.sum(params * (x - 1.0) ** 2) + 0.25 * jnp.sum(x**4)


def _contradicting(x, params):
    return 0.5 * jnp.sum(x**2)


def _nan_wall(x, params):
    return jnp.where(x[0] > 1.5, jnp.nan, jnp.sum((x - 2.0) ** 2))


def _inf_wall(x, params):
    valley = jnp.sum((x - 0.5) ** 2) + 3.0 * jnp.sum((x - 0.5) ** 4)
    return jnp.where(x[0] > 0.6, jnp.inf, valley)


def _linear(x, params):
    return jnp.sum(x)


def _steep(x, params):
    return 1e110 * (0.5 * jnp.sum(jnp.arange(1.0, x.shape[0] + 1) * x**2) + jnp.sum(x**4))


def _unavailable(x, params):
    raise RuntimeError("objective unavailable")


def _tiny_scale(x, params):
    u = x * 1e100
    return jnp.sum(1e-200 * (u**2 / 2 + 0.1 * jnp.sin(u)))


def _badly_scaled(x, params):
    return 5e17 * jnp.sum(x**2)


_OBJECTIVES = {
    "quadratic": _quadratic,
    "rosenbrock": _rosenbrock,
    "ill-conditioned": _ill_conditioned,
    "contradicting": _contradicting,
    "nan-wall": _nan_wall,
    "inf-wall": _inf_wall,
    "linear": _linear,
    "steep": _steep,
    "tiny-scale": _tiny_scale,
    "badly-scaled": _badly_scaled,
    "unavailable": _unavailable,
}


def _value_and_grad(name, x, params):
    """The objective's value and gradient; ``contradicting`` returns the
    gradient of its negative."""
    value, grad = jax.value_and_grad(_OBJECTIVES[name])(x, params)
    return (value, -grad) if name == "contradicting" else (value, grad)


def _problems():
    """``name -> (x0, params)`` for the comparison problems."""
    rng = parity_rng(11)
    q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    matrix = (q * np.logspace(0, 3, 20)) @ q.T
    return {
        "quadratic": (rng.uniform(-1, 1, 20), (0.5 * (matrix + matrix.T), rng.uniform(-1, 1, 20))),
        "rosenbrock": (rng.uniform(-2, 2, 6), np.zeros(0)),
        "ill-conditioned": (rng.uniform(-3, 3, 8), np.logspace(0, 6, 8)),
    }


if JAX_IMPORT_ERROR is None:
    # The compiled solvers exist only where the JAX runtime does, so discovery
    # can import this module and skip it without JAX.

    @partial(jit, static_argnames=("name",))
    def _device_bfgs(x0, params, gtol, maxiter, name):
        return minimize_bfgs(partial(_value_and_grad, name, params=params), x0, gtol, maxiter)

    @partial(jit, static_argnames=("name", "maxcor", "maxls"))
    def _device_lbfgsb(x0, params, ftol, gtol, maxiter, maxls, name, maxcor):
        return minimize_lbfgsb(
            partial(_value_and_grad, name, params=params), x0, ftol, gtol, maxiter, maxcor=maxcor, maxls=maxls
        )


def _scipy(name, x0, params, method, options) -> OptimizeResult:
    """SciPy on the same JAX objective, evaluated on the CPU."""
    with parity_default_device("cpu"):
        params = jax.tree.map(jnp.asarray, params)

        def fun(x):
            value, grad = _value_and_grad(name, jnp.asarray(x), params)
            return host_array(value), host_array(grad)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return minimize(fun, x0, jac=True, method=method, options=options)


def _device(name, x0, params, method, options, lane):
    """The device solver on ``lane`` with SciPy's options; returns the host
    result and SciPy's message for it."""
    with parity_default_device(lane):
        x0, params = jnp.asarray(x0), jax.tree.map(jnp.asarray, params)
        if method == "BFGS":
            result = _device_bfgs(x0, params, options["gtol"], float(options["maxiter"]), name)
            messages = BFGS_MESSAGES
        else:
            result = _device_lbfgsb(
                x0,
                params,
                options["ftol"],
                options["gtol"],
                float(options["maxiter"]),
                options.get("maxls", 20),
                name,
                options["maxcor"],
            )
            messages = LBFGSB_MESSAGES
        result = jax.tree.map(host_array, result)
    return result, messages[int(result.message)]


def _options(method, maxiter, **extra):
    if method == "BFGS":
        return {"gtol": _GTOL, "maxiter": maxiter, **extra}
    return {"gtol": _GTOL, "ftol": 1e-14, "maxiter": maxiter, "maxcor": 3, **extra}


# The port's claims, and so these comparisons (records, messages and the
# private _linesearch helpers), are SciPy 1.17's; another SciPy is not their reference.
_SCIPY_REFERENCE = (1, 17)


@unittest.skipUnless(
    tuple(int(part) for part in scipy.__version__.split(".")[:2]) == _SCIPY_REFERENCE,
    f"compares with SciPy {'.'.join(map(str, _SCIPY_REFERENCE))}, not {scipy.__version__}",
)
class TestQuasiNewtonJax(JaxTestCase):
    def _assert_scipy_record(self, expected: OptimizeResult, result, message: str, name: str):
        """Status, message, nit, nfev and njev are SciPy's."""
        self.assertEqual(
            (int(result.status), message, int(result.iterations), int(result.nfev), int(result.njev)),
            (expected.status, expected.message, expected.nit, expected.nfev, expected.njev),
            f"{name}: (status, message, nit, nfev, njev) differ from SciPy's",
        )

    def _assert_scipy_hess_inv(self, expected: OptimizeResult, result, name: str, rtol: float):
        """``hess_inv`` is SciPy's: BFGS's matrix, or the correction pairs of
        L-BFGS-B's ``LbfgsInvHessProduct`` in SciPy's (ring-buffer) order."""
        if isinstance(expected.hess_inv, LbfgsInvHessProduct):
            corrections = result.hess_inv
            count = int(corrections.count)
            self.assertEqual(count, expected.hess_inv.n_corrs, f"{name}: number of corrections")
            assert_matches_native(corrections.s[:count], expected.hess_inv.sk, f"{name} hess_inv sk", rtol)
            assert_matches_native(corrections.y[:count], expected.hess_inv.yk, f"{name} hess_inv yk", rtol)
        else:
            assert_matches_native(result.hess_inv, expected.hess_inv, f"{name} hess_inv", rtol)

    def test_early_iterates_match_scipy(self):
        """Capped at k iterations, both solvers stop as SciPy does with its
        iterate to round-off, at every k where round-off determines SciPy's
        own iterate to these bounds, and with SciPy's status, message and
        counts at every k.

        SciPy's runs from ``x0`` moved by one ulp per entry (nine sign
        patterns) differ from its run from ``x0`` by at most 8.6e-14 (``yk``)
        and 2.6e-12 (``jac``) of the largest entry after nine L-BFGS-B
        iterations on the quadratic, but by 2.0e-12 and 1.1e-11 after ten,
        beyond the bounds: ten iterations there magnify round-off about
        1e4-fold, so a run that rounds differently from SciPy, as compiled XLA
        does, can miss the bounds with no error in its algorithm (on JAX
        0.10.2's CPU lane ``yk`` differs by 1.2e-12). The numerical comparison
        stops at nine there (``_EARLY_NUMERIC_ITERATIONS``); the record is still
        compared at ten, and the converged comparison follows that walk to its
        end."""
        for lane in _LANES:
            for method in ("BFGS", "L-BFGS-B"):
                for name in _problems():
                    with self.subTest(lane=lane, method=method, name=name), self.case():
                        self._case_early_iterates_match_scipy(lane, method, name)

    def _case_early_iterates_match_scipy(self, lane, method, name):
        x0, params = _problems()[name]
        numeric_iterations = _EARLY_NUMERIC_ITERATIONS.get((method, name), _EARLY_ITERATIONS)
        for k in range(1, _EARLY_ITERATIONS + 1):
            options = _options(method, k)
            expected = _scipy(name, x0, params, method, options)
            self.assertEqual(expected.status, 1, f"{name}: SciPy converged before k={k}")
            result, message = _device(name, x0, params, method, options, lane)
            self._assert_scipy_record(expected, result, message, f"{method} {name} k={k}")
            if k > numeric_iterations:
                continue
            assert_matches_native(result.x, expected.x, f"{method} {name} x at k={k}", _EARLY_RTOL)
            rtol = _EARLY_FUN_JAC_RTOL
            assert_matches_native(result.fun, expected.fun, f"{method} {name} fun at k={k}", rtol, abs(expected.fun))
            assert_matches_native(result.jac, expected.jac, f"{method} {name} jac at k={k}", rtol)
            self._assert_scipy_hess_inv(expected, result, f"{method} {name} k={k}", _EARLY_RTOL)

    def test_converged_solves_match_scipy(self):
        """Run to convergence, both solvers end as SciPy does: status,
        message, counts, x, fun, jac (to round-off of their scale at x0) and
        hess_inv, except the counts of the round-off walks listed in
        ``_ROUND_OFF_WALKS``, which end at SciPy's solution to what ``gtol``
        allows."""
        for lane in _LANES:
            for method in ("BFGS", "L-BFGS-B"):
                for name in _problems():
                    with self.subTest(lane=lane, method=method, name=name), self.case():
                        self._case_converged_solves_match_scipy(lane, method, name)

    def _case_converged_solves_match_scipy(self, lane, method, name):
        x0, params = _problems()[name]
        options = _options(method, 5000) if method == "BFGS" else _options(method, 5000, maxcor=10)
        expected = _scipy(name, x0, params, method, options)
        self.assertEqual(expected.status, 0, f"{name}: SciPy did not converge")
        result, message = _device(name, x0, params, method, options, lane)
        if (method, name) in _ROUND_OFF_WALKS:  # Round-off decides these counts; see _ROUND_OFF_WALKS.
            self.assertEqual(
                (int(result.status), message),
                (expected.status, expected.message),
                f"{method} {name}: status and message differ from SciPy's",
            )
            assert_matches_native(result.x, expected.x, f"{method} {name} solution", _FINAL_RTOL)
            return
        self._assert_scipy_record(expected, result, message, f"{method} {name}")
        assert_matches_native(result.x, expected.x, f"{method} {name} solution", _EARLY_RTOL)
        assert_matches_native(result.fun, expected.fun, f"{method} {name} fun", _CONVERGED_FUN_ATOL, 1.0)
        assert_matches_native(result.jac, expected.jac, f"{method} {name} jac", _CONVERGED_JAC_ATOL, 1.0)
        self._assert_scipy_hess_inv(expected, result, f"{method} {name}", _CONVERGED_HESS_INV_RTOL)

    def test_failed_line_searches_end_as_in_scipy(self):
        """A contradicting gradient: BFGS loses precision (status 2), L-BFGS-B is ABNORMAL."""
        cases = {
            "BFGS": ("contradicting", np.ones(4), _options("BFGS", 100)),
            "L-BFGS-B": ("contradicting", np.ones(4), _options("L-BFGS-B", 100)),
            # One trial per line search: a search that needs a second restarts
            # from an empty memory, and fails without corrections.
            "L-BFGS-B restart": ("rosenbrock", _problems()["rosenbrock"][0], _options("L-BFGS-B", 100, maxls=1)),
        }
        for lane in _LANES:
            for case, (name, x0, options) in cases.items():
                with self.subTest(lane=lane, case=case), self.case():
                    method = "BFGS" if case == "BFGS" else "L-BFGS-B"
                    expected = _scipy(name, x0, np.zeros(0), method, options)
                    self.assertEqual(expected.status, 2, f"{case}: SciPy did not fail its line search")
                    result, message = _device(name, x0, np.zeros(0), method, options, lane)
                    self._assert_scipy_record(expected, result, message, case)
                    assert_matches_native(result.x, expected.x, f"{case} x", _FAILURE_RTOL)

    def test_bfgs_line_search_fallbacks_take_scipys_paths(self):
        """DCSRCH failing (a NaN step past an infinite wall, a start below
        ``amin``) or exhausting ``wolfe2``'s doublings leads where SciPy's does."""
        cases = {
            # Converges after zooming in scalar_search_wolfe2.
            "infinite wall": ("inf-wall", np.array([0.0, 0.2]), _options("BFGS", 100), 0),
            # Each iteration ends with the tenth doubling, its gradient requested by BFGS.
            "unbounded": ("linear", np.array([0.5, -1.0]), _options("BFGS", 5), 1),
            # One iteration: the inverse Hessian is still the identity.
            "steep": ("steep", np.array([0.3, -0.2, 0.25]), _options("BFGS", 1), 1),
        }
        for lane in _LANES:
            for case, (name, x0, options, status) in cases.items():
                with self.subTest(lane=lane, case=case), self.case():
                    expected = _scipy(name, x0, np.zeros(0), "BFGS", options)
                    self.assertEqual(expected.status, status, case)
                    result, message = _device(name, x0, np.zeros(0), "BFGS", options, lane)
                    self._assert_scipy_record(expected, result, message, case)
                    assert_matches_native(result.x, expected.x, f"{case} x", _FAILURE_RTOL)

    def test_bfgs_update_keeps_badly_scaled_inverse_hessian(self):
        """On f = 5e17 x^2 from x = 1 the first update's rho s s^T (1e-18) lies
        below the round-off of terms the size of H = 1. SciPy's factored update
        keeps H = 1 / (2 * 5e17); so does the port, after that first update
        (with SciPy's record) and to convergence (with SciPy's status). An
        expanded rank-two update leaves H wrong after the first update (on
        the CPU negative, ending with status 2; on the GPU 200 times too
        large, which the next update repairs, so there only the first update
        shows it). The iteration count to convergence is not compared: the
        second step lands within round-off of x = 0, exactly on it on the GPU,
        which then stops one update before SciPy. H is still compared: in one
        variable each update sets H to s / y = 1 / (2 * 5e17), to round-off,
        whatever H was."""
        x0 = np.ones(1)
        for lane in _LANES:
            for maxiter in (1, 100):
                with self.subTest(lane=lane, maxiter=maxiter), self.case():
                    options = _options("BFGS", maxiter)
                    expected = _scipy("badly-scaled", x0, np.zeros(0), "BFGS", options)
                    self.assertEqual(expected.status, 1 if maxiter == 1 else 0, "SciPy's status changed")
                    result, message = _device("badly-scaled", x0, np.zeros(0), "BFGS", options, lane)
                    if maxiter == 1:
                        self._assert_scipy_record(expected, result, message, "badly scaled, one update")
                    else:
                        self.assertEqual((int(result.status), message), (expected.status, expected.message))
                    self._assert_scipy_hess_inv(expected, result, f"badly scaled, maxiter={maxiter}", _FAILURE_RTOL)

    def test_zoom_interpolants_match_scipy(self):
        """The cubic and quadratic minimizers are SciPy's, ``None`` where SciPy's
        raise (coincident points, a negative radical, overflow, infinities), and
        ``_zoom``'s first two trials (bisection or quadratic, then cubic) are SciPy's."""
        rng = parity_rng(12)
        draws = [tuple(rng.uniform(-2.0, 2.0, 7)) for _ in range(200)]
        a, fa, fpa, b, fb, c, fc = draws[0]
        edges = [
            (a, fa, fpa, a, fb, c, fc),
            (a, fa, fpa, b, fb, a, fc),
            (a, fa, fpa, b, fb, b, fc),
            (0.0, 0.0, -1.0, 1.0, 10.0, 2.0, -50.0),
            (a, fa, fpa, b * 1e200, fb, c * 1e200, fc),
            (a, fa, fpa, b, np.inf, c, fc),
            (a, fa, fpa, b, fb, c, np.nan),
        ]
        for case, (a, fa, fpa, b, fb, c, fc) in enumerate(draws + edges):
            for name, scipy_min, jax_min, args in (
                ("cubic", scipy_linesearch._cubicmin, _cubicmin, (a, fa, fpa, b, fb, c, fc)),
                ("quadratic", scipy_linesearch._quadmin, _quadmin, (a, fa, fpa, b, fb)),
            ):
                expected = scipy_min(*(np.float64(value) for value in args))
                value, valid = (host_array(out) for out in jax_min(*(jnp.float64(value) for value in args)))
                self.assertEqual(bool(valid), expected is not None, f"{name} case {case}: SciPy gives {expected}")
                if expected is not None:
                    np.testing.assert_allclose(value, expected, rtol=1e-12, err_msg=f"{name} case {case}")

        class Captured(Exception):
            pass

        brackets = [(rng.uniform(0.0, 1.0), rng.uniform(1.0, 3.0)) for _ in range(100)]
        for case, (a_lo, a_hi, phi_lo, phi_hi, derphi_lo) in enumerate(
            # Half the brackets are reversed (a_lo > a_hi), as _zoom's callers pass them.
            (*(bracket if case % 2 else bracket[::-1]), -rng.uniform(), rng.uniform(), -rng.uniform())
            for case, bracket in enumerate(brackets)
        ):
            phi0, derphi0, high = 0.0, -1.0, phi_lo + 1.0
            trials = []

            def phi(alpha):
                # A high value: the trial replaces a_hi, and a_hi becomes a_rec.
                trials.append(alpha)
                if len(trials) == 2:
                    raise Captured
                return high

            with self.assertRaises(Captured):
                scipy_linesearch._zoom(
                    a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, None, phi0, derphi0, 1e-4, 0.9, None
                )
            first = _zoom_trial(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, 0.0, phi0, 0)
            second = _zoom_trial(a_lo, first, phi_lo, high, derphi_lo, a_hi, phi_hi, 1)
            np.testing.assert_allclose(
                [host_array(first), host_array(second)], trials, rtol=1e-12, err_msg=f"zoom trials, case {case}"
            )

    def test_lbfgsb_tiny_corrections_match_scipy_op_by_op(self):
        """Corrections near 1e-100 underflow the products of formt's ``L D^-1
        L^T`` (``sy_ik sy_jk / sy_kk``, multiplied before dividing as in SciPy's
        C), so ``T`` loses its last pivot and the memory empties, as in SciPy:
        for k = 1, ..., 8 iterates (to 1e-14), counts, statuses, messages and
        the pairs held (whose drops are the resets) are SciPy's. Op by op on
        the CPU: compiled, XLA fuses multiply-adds into one rounding, an ulp
        from SciPy, and DCSTEP's cubic, whose terms cancel ten orders of
        magnitude here, magnifies that to 1e-5; the reset turns on a pivot of
        rounding residue, which only the same LAPACK Cholesky rounds alike."""
        x0 = np.array([2e-100])
        pairs, scipy_pairs = [], []
        for k in range(1, 9):
            options = {"gtol": 0.0, "ftol": 0.0, "maxiter": k, "maxcor": 3}
            with self.subTest(k=k), self.case(), jax.disable_jit():
                expected = _scipy("tiny-scale", x0, np.zeros(0), "L-BFGS-B", options)
                result, message = _device("tiny-scale", x0, np.zeros(0), "L-BFGS-B", options, "cpu")
                self._assert_scipy_record(expected, result, message, f"tiny-scale k={k}")
                assert_matches_native(result.x, expected.x, f"tiny-scale x at k={k}", _TINY_RTOL)
                pairs.append(int(result.hess_inv.count))
                scipy_pairs.append(expected.hess_inv.n_corrs)
        self.assertEqual(pairs, scipy_pairs, "pairs held after k = 1, ..., 8 differ from SciPy's")
        resets = sum(after < before for before, after in zip(scipy_pairs, scipy_pairs[1:]))
        self.assertEqual(resets, 2, f"SciPy's pairs held {scipy_pairs} no longer show its two resets")

    def test_invalid_inputs_end_as_in_scipy(self):
        """SciPy's checks that apply without bounds: L-BFGS-B stops before
        iterating on a negative ``ftol`` (``factr``), no corrections or no
        variables, with SciPy's status, message, counters, zero ``fun`` and
        ``jac`` and empty ``hess_inv``; ``maxls <= 0`` raises SciPy's
        ``ValueError``, and so does BFGS without variables. Each exit first
        evaluates the objective at ``x0``, as SciPy does, so an objective that
        raises raises there."""
        x0, params = _problems()["rosenbrock"]
        stopped = {
            "ftol < 0": (x0, _options("L-BFGS-B", 100, ftol=-1e-8)),
            "maxcor = 0": (x0, _options("L-BFGS-B", 100, maxcor=0)),
            "no variables": (np.zeros(0), _options("L-BFGS-B", 100)),
        }
        for lane in _LANES:
            for case, (start, options) in stopped.items():
                with self.subTest(lane=lane, case=case), self.case():
                    expected = _scipy("rosenbrock", start, params, "L-BFGS-B", options)
                    self.assertEqual(expected.status, 2, f"{case}: SciPy did not stop")
                    result, message = _device("rosenbrock", start, params, "L-BFGS-B", options, lane)
                    self._assert_scipy_record(expected, result, message, case)
                    for field in ("x", "fun", "jac"):
                        np.testing.assert_array_equal(
                            getattr(result, field), expected[field], err_msg=f"{case}: {field} differs from SciPy's"
                        )
                    self._assert_scipy_hess_inv(expected, result, case, 0.0)
        raised = {
            "L-BFGS-B maxls = 0": ("L-BFGS-B", x0, _options("L-BFGS-B", 100, maxls=0)),
            "BFGS without variables": ("BFGS", np.zeros(0), _options("BFGS", 100)),
        }
        for case, (method, start, options) in raised.items():
            with self.subTest(case=case), self.case():
                with self.assertRaises(ValueError) as scipy_error:
                    _scipy("rosenbrock", start, params, method, options)
                with self.assertRaises(ValueError) as device_error:
                    _device("rosenbrock", start, params, method, options, "cpu")
                self.assertEqual(str(device_error.exception), str(scipy_error.exception), case)
        # Every exit evaluates the objective at x0 first, so its errors come first.
        exits = {case: ("L-BFGS-B", start, options) for case, (start, options) in stopped.items()} | raised
        for case, (method, start, options) in exits.items():
            with self.subTest(case=f"{case}, objective raising"), self.case():
                with self.assertRaises(RuntimeError) as scipy_error:
                    _scipy("unavailable", start, params, method, options)
                with self.assertRaises(RuntimeError) as device_error:
                    _device("unavailable", start, params, method, options, "cpu")
                self.assertEqual(str(device_error.exception), str(scipy_error.exception), case)

    def test_nan_and_maxiter_zero_end_as_in_scipy(self):
        """A NaN wall gives BFGS status 3; maxiter=0 stops BFGS at x0 and L-BFGS-B after one step."""
        x0, params = _problems()["rosenbrock"]
        cases = {
            "BFGS nan": ("nan-wall", np.zeros(3), "BFGS", _options("BFGS", 100)),
            "BFGS maxiter=0": ("rosenbrock", x0, "BFGS", _options("BFGS", 0)),
            "L-BFGS-B maxiter=0": ("rosenbrock", x0, "L-BFGS-B", _options("L-BFGS-B", 0)),
        }
        expected_status = {"BFGS nan": 3, "BFGS maxiter=0": 1, "L-BFGS-B maxiter=0": 1}
        for lane in _LANES:
            for case, (name, start, method, options) in cases.items():
                with self.subTest(lane=lane, case=case), self.case():
                    expected = _scipy(name, start, params, method, options)
                    self.assertEqual(expected.status, expected_status[case], case)
                    result, message = _device(name, start, params, method, options, lane)
                    self._assert_scipy_record(expected, result, message, case)
                    assert_matches_native(result.x, expected.x, f"{case} x", _FAILURE_RTOL)
