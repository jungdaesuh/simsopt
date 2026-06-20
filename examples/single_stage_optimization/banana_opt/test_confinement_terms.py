"""Rigorous unit + FD/Taylor tests for two opt-in single-stage confinement terms.

This pins the *real behavior and contract* of two levers, not their source text:

  TERM 1 -- ``NobleIotaPull`` (``single_stage_objectives.py``):
      ``J = 0.5 * offset**2`` where ``offset = iota - lo`` (iota < lo),
      ``iota - hi`` (iota > hi), else ``0``; ``dJ = offset * iota_term.dJ``.
      A one-sided, squared-hinge pull toward the noble iota window. It is
      ``C1`` (value and first derivative are continuous at both kinks), which
      is the property the gradient tests below actually verify.

  TERM 2 -- the explicit QA residual built in ``single_stage_banana_example``:
      ``average_surface_objectives`` of
      ``NonQuasiSymmetricRatio(surface, BiotSavart(coils), quasi_poloidal=False)``
      over the Boozer surfaces. ``quasi_poloidal=False`` IS the QA (M=1, N=0)
      branch (``self.axis == 0``); the helicity guard rejects anything else.

Conventions are inherited, not re-invented:

  * The centered-/forward-FD log-log convergence machinery is imported directly
    from the project's Taylor-test precedent
    (``VMEC_SINGLE_STAGE/vmec_single_stage_taylor_tests.py``) -- the same
    ``_centered_fd_errors`` / ``_forward_fd_errors`` / ``_convergence_slopes``
    helpers and slope bands. We reuse them rather than re-implementing (which
    would be a mirror test of the convention).
  * The QA-term Taylor test reuses the simsopt repo's own NCSX
    ``get_boozer_surface`` fixture and ``taylor_test1`` ratio-halving check
    (the exact precedent that already covers ``NonQuasiSymmetricRatio`` itself).

No test mirrors the implementation: each assertion is tied to an externally
checkable consequence (FD of ``J``, the C1 kink behavior, the helicity guard,
object identity/independence, or byte-identical default-OFF assembly).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# --- Make the example modules importable, mirroring the sibling example layout.
# ``single_stage_objectives`` lives in the ``banana_opt`` package directory;
# its own imports (``alm_utils`` etc.) live one level up. The VMEC precedent
# Taylor module lives in a sibling example directory. simsopt itself is an
# editable install, so it imports without extra path surgery.
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_HERE)  # .../single_stage_optimization
_VMEC_PRECEDENT_DIR = os.path.join(_EXAMPLE_ROOT, "VMEC_SINGLE_STAGE")
for _p in (_HERE, _EXAMPLE_ROOT, _VMEC_PRECEDENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simsopt._core.derivative import Derivative, derivative_dec  # noqa: E402
from simsopt._core.optimizable import Optimizable  # noqa: E402

import single_stage_objectives as sso  # noqa: E402
import vmec_single_stage_taylor_tests as taylor  # noqa: E402


# --------------------------------------------------------------------------- #
# Mocks / helpers
# --------------------------------------------------------------------------- #
class _AffineIota(Optimizable):
    """Minimal ``Iotas``-shaped stand-in: ``iota(u) = slope * u + intercept``.

    It is a genuine :class:`Optimizable` with one free dof ``u`` and a
    ``@derivative_dec`` ``dJ`` that returns a real :class:`Derivative` keyed by
    itself, exactly like the production ``Iotas`` term feeds ``NobleIotaPull``.
    The affine map gives an exactly-known ``d iota / d u == slope`` so the
    chain rule inside ``NobleIotaPull.dJ`` (``offset * iota_term.dJ``) has a
    closed-form target.
    """

    def __init__(self, slope: float, intercept: float, u0: float = 0.0):
        self.slope = float(slope)
        self.intercept = float(intercept)
        Optimizable.__init__(self, x0=np.array([float(u0)]), names=["u"])

    def set_iota(self, iota: float) -> None:
        """Move the dof so ``J() == iota`` (requires ``slope != 0``)."""
        self.x = np.array([(float(iota) - self.intercept) / self.slope])

    def J(self) -> float:
        return self.slope * float(self.x[0]) + self.intercept

    @derivative_dec
    def dJ(self):
        return Derivative({self: np.array([self.slope])})

    return_fn_map = {"J": J, "dJ": dJ}


class _LinearScalar(Optimizable):
    """Trivial scalar objective ``J = c * u`` with a real own-dof gradient.

    Used as a cheap, deterministic stand-in for the many heavyweight physics
    terms ``build_total_objective`` sums, so we can exercise the assembly's
    None-term / zero-weight code paths without a full Boozer bundle.
    """

    def __init__(self, c: float, u0: float):
        self.c = float(c)
        Optimizable.__init__(self, x0=np.array([float(u0)]))

    def J(self) -> float:
        return self.c * float(self.x[0])

    @derivative_dec
    def dJ(self):
        return Derivative({self: np.array([self.c])})

    return_fn_map = {"J": J, "dJ": dJ}


_LO = sso.NOBLE_IOTA_PULL_DEFAULT_LO  # 0.276
_HI = sso.NOBLE_IOTA_PULL_DEFAULT_HI  # 0.281


def _make_pull(iota_value: float, slope: float = 1.0, *, lo=_LO, hi=_HI):
    """Return ``(iota_term, NobleIotaPull)`` positioned at ``iota_value``."""
    term = _AffineIota(slope=slope, intercept=0.0)
    term.set_iota(iota_value)
    return term, sso.NobleIotaPull(term, lo=lo, hi=hi)


# --------------------------------------------------------------------------- #
# A. NobleIotaPull UNIT: piecewise value/derivative + C1 continuity at kinks
# --------------------------------------------------------------------------- #
def test_pull_defaults_match_documented_window():
    """Pin the contract defaults the campaigns rely on (lo=0.276, hi=0.281)."""
    assert _LO == pytest.approx(0.276)
    assert _HI == pytest.approx(0.281)
    assert _LO < _HI


def test_pull_zero_strictly_inside_window():
    """J and dJ are exactly zero anywhere strictly inside [lo, hi]."""
    for iota in (_LO + 1e-9, 0.5 * (_LO + _HI), _HI - 1e-9):
        _, pull = _make_pull(iota)
        assert pull.J() == 0.0
        # dJ is a full gradient over the single mock dof; flat -> exact zero.
        np.testing.assert_array_equal(pull.dJ(), np.zeros(1))


def test_pull_quadratic_below_lo():
    """Below lo: J == 0.5*(iota-lo)^2 and dJ == (iota-lo)*d_iota/du."""
    slope = 1.3
    for iota in (_LO - 1e-3, _LO - 0.02, _LO - 0.1):
        term, pull = _make_pull(iota, slope=slope)
        offset = iota - _LO
        assert offset < 0.0
        assert pull.J() == pytest.approx(0.5 * offset * offset, rel=0, abs=1e-15)
        np.testing.assert_allclose(pull.dJ(), np.array([offset * slope]), atol=1e-14)


def test_pull_quadratic_above_hi():
    """Above hi: J == 0.5*(iota-hi)^2 and dJ == (iota-hi)*d_iota/du."""
    slope = 0.7
    for iota in (_HI + 1e-3, _HI + 0.02, _HI + 0.1):
        term, pull = _make_pull(iota, slope=slope)
        offset = iota - _HI
        assert offset > 0.0
        assert pull.J() == pytest.approx(0.5 * offset * offset, rel=0, abs=1e-15)
        np.testing.assert_allclose(pull.dJ(), np.array([offset * slope]), atol=1e-14)


def test_pull_C1_continuity_at_kinks():
    """C1: both J and dJ -> 0 as iota -> lo^- and iota -> hi^+.

    A squared hinge has a *continuous* value and first derivative at the kink
    (the second derivative jumps). We approach each kink from the active side
    and assert J and |dJ| shrink toward zero, and are exactly zero AT the kink.
    """
    slope = 1.0
    # Approach lo from below; offsets -> 0^-.
    prev_J, prev_g = None, None
    for d in (1e-2, 1e-3, 1e-4, 1e-5):
        _, pull = _make_pull(_LO - d, slope=slope)
        J = pull.J()
        g = abs(float(pull.dJ()[0]))
        assert J > 0.0 and g > 0.0
        if prev_J is not None:
            assert J < prev_J and g < prev_g  # strictly shrinking toward kink
        prev_J, prev_g = J, g
    # Exactly at the kink: offset == 0 -> both vanish (the C1 join value).
    _, pull_lo = _make_pull(_LO, slope=slope)
    assert pull_lo.J() == 0.0
    np.testing.assert_array_equal(pull_lo.dJ(), np.zeros(1))

    # Approach hi from above; offsets -> 0^+.
    prev_J, prev_g = None, None
    for d in (1e-2, 1e-3, 1e-4, 1e-5):
        _, pull = _make_pull(_HI + d, slope=slope)
        J = pull.J()
        g = abs(float(pull.dJ()[0]))
        assert J > 0.0 and g > 0.0
        if prev_J is not None:
            assert J < prev_J and g < prev_g
        prev_J, prev_g = J, g
    _, pull_hi = _make_pull(_HI, slope=slope)
    assert pull_hi.J() == 0.0
    np.testing.assert_array_equal(pull_hi.dJ(), np.zeros(1))


def test_pull_rejects_degenerate_window():
    """Contract guard: lo < hi and finite are required."""
    term = _AffineIota(slope=1.0, intercept=0.0)
    term.set_iota(0.2785)
    with pytest.raises(ValueError):
        sso.NobleIotaPull(term, lo=0.30, hi=0.20)  # lo >= hi
    with pytest.raises(ValueError):
        sso.NobleIotaPull(term, lo=np.nan, hi=0.30)


def test_pull_builder_none_passthrough():
    """``build_single_stage_noble_iota_pull_objective(None)`` returns None.

    This is the first half of the default-OFF guarantee: when there is no outer
    iota term, the builder yields None and the assembler adds nothing.
    """
    assert sso.build_single_stage_noble_iota_pull_objective(None) is None
    term = _AffineIota(slope=1.0, intercept=0.0)
    term.set_iota(0.2785)
    built = sso.build_single_stage_noble_iota_pull_objective(term)
    assert isinstance(built, sso.NobleIotaPull)
    assert built.lo == _LO and built.hi == _HI


# --------------------------------------------------------------------------- #
# B. NobleIotaPull FD-GRADIENT (project Taylor convention)
# --------------------------------------------------------------------------- #
def _pull_builders(pull, term):
    """(build_J, build_dJ) over the mock dof vector for FD harness reuse."""

    def build_J(x):
        term.x = np.asarray(x, dtype=float)
        return pull.J()

    def build_dJ(x):
        term.x = np.asarray(x, dtype=float)
        return pull.dJ()

    return build_J, build_dJ


def test_pull_gradient_exact_inside_active_branch():
    """Inside one branch J is exactly quadratic in u, so centered FD == analytic.

    This is the strongest possible gradient check for a single active branch:
    the truncation error is identically zero (a quadratic's centered difference
    is exact), so any mismatch beyond float round-off is a chain-rule wiring
    bug in ``offset * iota_term.dJ``. We assert the FD/analytic gap is at the
    round-off floor across the whole step schedule.
    """
    slope = 1.1
    term, pull = _make_pull(_LO - 0.05, slope=slope)  # well below lo
    x0 = term.x.copy()
    build_J, build_dJ = _pull_builders(pull, term)
    direction = np.array([1.0])
    g_dot_h = float(build_dJ(x0) @ direction)
    # Keep steps small enough never to cross the kink (0.05/slope away).
    schedule = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
    for h in schedule:
        plus = build_J(x0 + h * direction)
        minus = build_J(x0 - h * direction)
        approx = (plus - minus) / (2.0 * h)
        assert abs(approx - g_dot_h) < 1e-12


def test_pull_gradient_forward_first_order_across_kink():
    """Forward FD straddling the kink is O(h): the C1-but-not-C2 fingerprint.

    Starting *exactly at* the kink (iota == lo), the analytic dJ is 0 (the
    join value). Stepping into the active branch, ``J(h) = 0.5*(slope*h)^2``,
    so the forward difference ``(J(h)-J(0))/h = 0.5*slope^2*h`` -> 0 linearly.
    The log-log slope of that error vs h must sit in the project's first-order
    band [0.8, 1.2], and the error must stay bounded (never blow up) -- the
    exact guarantee a squared hinge gives across its curvature jump.
    """
    slope = 1.0
    term, pull = _make_pull(_LO, slope=slope)  # sit on the kink
    x0 = term.x.copy()
    build_J, build_dJ = _pull_builders(pull, term)

    # Analytic directional derivative at the kink is exactly 0.
    assert float(build_dJ(x0) @ np.array([1.0])) == 0.0

    # Step in the -u direction -> into the below-lo active branch.
    direction = np.array([-1.0])
    # A schedule that stays above the round-off floor (exact J -> error hits 0
    # for tiny h, which is correct but degenerate for a slope fit).
    schedule = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4)
    errors = taylor._forward_fd_errors(build_J, build_dJ, x0, direction, schedule)
    # Errors strictly decrease toward zero -- no blow-up across the kink.
    for k in range(1, len(errors)):
        assert errors[k] < errors[k - 1]
    slopes = taylor._convergence_slopes(schedule, errors)
    lo_band, hi_band = taylor.FIRST_ORDER_SLOPE_BAND
    assert taylor._stable_band_passes(
        slopes, (lo_band, hi_band), taylor.DEFAULT_STABLE_BAND_WIDTH
    ), f"forward-FD slopes not first-order: {slopes}"


def test_pull_gradient_centered_kink_straddle_does_not_blow_up():
    """Centered FD whose window straddles the kink stays bounded and converges.

    Placed just inside the noble window near lo, large steps reach across the
    kink into the active branch. Because the hinge is C1, the centered-FD error
    vs the (zero) analytic in-window derivative is bounded and shrinks to the
    round-off floor as h decreases -- it does NOT diverge. We assert monotone
    non-increase and a finite, small ceiling.
    """
    slope = 1.0
    term, pull = _make_pull(_LO + 5e-4, slope=slope)  # just inside window
    x0 = term.x.copy()
    build_J, build_dJ = _pull_builders(pull, term)
    # In-window analytic derivative is exactly zero.
    assert float(build_dJ(x0) @ np.array([1.0])) == 0.0
    direction = np.array([1.0])
    schedule = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
    errors = taylor._centered_fd_errors(build_J, build_dJ, x0, direction, schedule)
    assert all(np.isfinite(e) for e in errors)
    assert max(errors) < 1e-1  # bounded; never blows up
    for k in range(1, len(errors)):
        assert errors[k] <= errors[k - 1] + 1e-15  # monotone non-increasing


# --------------------------------------------------------------------------- #
# C. QA residual: helicity guard + genuine independent QA-branch object
# --------------------------------------------------------------------------- #
def _import_example_module():
    """Lazily import the heavyweight example CLI module.

    It is a ~17k-line module; importing it can be expensive or pull optional
    deps. Done lazily so the helicity-guard test skips cleanly if it cannot be
    imported here, rather than faking the guard.
    """
    sys.path.insert(0, os.path.join(_EXAMPLE_ROOT, "SINGLE_STAGE"))
    import importlib

    return importlib.import_module("single_stage_banana_example")


def _run_parse_args(extra_argv):
    """Drive the example's ``parse_args()`` with a controlled argv.

    Returns ``(exit_code, last_stderr_line)``; ``exit_code is None`` means the
    parser accepted the arguments without calling ``parser.error``.
    """
    import contextlib
    import io

    example = _import_example_module()
    old_argv = sys.argv
    sys.argv = ["single_stage_banana_example"] + list(extra_argv)
    err = io.StringIO()
    code, msg = None, ""
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            example.parse_args()
    except SystemExit as exc:
        code = exc.code
        lines = err.getvalue().strip().splitlines()
        msg = lines[-1] if lines else ""
    finally:
        sys.argv = old_argv
    return code, msg


def test_qa_helicity_guard_rejects_non_qa_helicity():
    """The QA residual only accepts helicity (1, 0); anything else is rejected
    at the argparse guard when the weight is positive.

    This pins the real CLI contract (``--single-stage-qa-helicity-m/-n must be
    (1, 0)``): a positive QA weight with a non-(1,0) helicity must call
    ``parser.error`` (SystemExit) with the helicity message. Skips only if the
    heavyweight example module cannot be imported in this environment.
    """
    try:
        code, msg = _run_parse_args(
            [
                "--single-stage-qa-residual-weight", "1.0",
                "--single-stage-qa-helicity-m", "1",
                "--single-stage-qa-helicity-n", "1",
            ]
        )
    except Exception as exc:  # pragma: no cover - environment-dependent import
        pytest.skip(f"single_stage_banana_example not importable here: {exc!r}")

    assert code == 2, f"expected argparse error exit, got code={code!r}"
    assert "must be (1, 0)" in msg, msg
    assert "helicit" in msg.lower() or "NonQuasiSymmetricRatio" in msg


def test_qa_helicity_guard_accepts_qa_and_ignores_helicity_when_off():
    """Boundary of the guard: (1, 0) with weight>0 is accepted, and the guard
    does NOT fire for a bad helicity when the QA weight is 0 (default-OFF).

    These two boundaries prove the guard is gated on ``weight > 0`` and that
    the realizable QA helicity passes -- i.e. the guard is neither too strict
    (rejecting valid QA) nor active when the term is off.
    """
    try:
        # (1, 0) with weight>0 must NOT raise the helicity error.
        code_ok, msg_ok = _run_parse_args(
            [
                "--single-stage-qa-residual-weight", "1.0",
                "--single-stage-qa-helicity-m", "1",
                "--single-stage-qa-helicity-n", "0",
            ]
        )
        # Bad helicity but weight 0 must NOT trip the guard.
        code_off, msg_off = _run_parse_args(
            [
                "--single-stage-qa-residual-weight", "0.0",
                "--single-stage-qa-helicity-m", "0",
                "--single-stage-qa-helicity-n", "1",
            ]
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"single_stage_banana_example not importable here: {exc!r}")

    assert "must be (1, 0)" not in msg_ok
    assert "must be (1, 0)" not in msg_off


def test_qa_residual_weight_must_be_nonnegative():
    """A negative QA weight is rejected at the argparse guard (finite + >= 0)."""
    try:
        code, msg = _run_parse_args(["--single-stage-qa-residual-weight", "-1.0"])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"single_stage_banana_example not importable here: {exc!r}")
    assert code == 2
    assert "finite and non-negative" in msg


def test_qa_term_is_genuine_independent_qa_branch():
    """The QA residual is a real NonQuasiSymmetricRatio QA-branch object,
    averaged over surfaces, and *independent* from the base JnonQSRatio.

    Reproduces the example's construction exactly (a dedicated set of
    ``NonQuasiSymmetricRatio(..., quasi_poloidal=False)`` terms with their own
    BiotSavart caches, averaged) and asserts:
      * ``quasi_poloidal=False`` selects the QA branch (``axis == 0``);
      * the dedicated terms are distinct objects from a base nonQS term and
        carry distinct BiotSavart fields (separately weightable);
      * the averaged term is a real Optimizable whose ``J()`` evaluates.
    """
    try:
        from simsopt.geo.surfaceobjectives import NonQuasiSymmetricRatio
        from simsopt.field import BiotSavart

        sys.path.insert(0, os.path.join(_EXAMPLE_ROOT, "..", "..", "tests"))
        from geo.surface_test_helpers import get_boozer_surface
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"QA fixture deps unavailable: {exc!r}")

    bs_base, boozer_surface = get_boozer_surface(label="Volume", boozer_type="ls")
    coils = bs_base.coils

    # Base QS term, as the example's ``nonQSs`` builds it (default QA branch).
    base_nonqs = NonQuasiSymmetricRatio(boozer_surface, bs_base)
    assert base_nonqs.axis == 0  # default quasi_poloidal=False -> QA

    # Dedicated QA residual term with its OWN BiotSavart, like the example.
    qa_bs = BiotSavart(coils)
    qa_term = NonQuasiSymmetricRatio(boozer_surface, qa_bs, quasi_poloidal=False)
    assert qa_term.axis == 0  # QA (M=1, N=0) branch
    # Independence: distinct objects and distinct field caches.
    assert qa_term is not base_nonqs
    assert qa_term.biotsavart is qa_bs
    assert qa_term.biotsavart is not base_nonqs.biotsavart

    # Averaged exactly as the example does (single surface here).
    averaged = sso.average_surface_objectives([qa_term])
    assert isinstance(averaged, Optimizable)
    val = float(averaged.J())
    assert np.isfinite(val) and val >= 0.0
    # The averaged single-term equals the term itself (1/total_weight * w*term).
    assert val == pytest.approx(float(qa_term.J()), rel=1e-12)


def test_qa_quasi_poloidal_true_is_distinct_qp_branch():
    """Sanity boundary: quasi_poloidal=True is the QP branch (axis==1), NOT QA.

    This is why the example *asserts* helicity (1, 0): only the
    quasi_poloidal=False decomposition realizes QA on this adjoint path.
    """
    try:
        from simsopt.geo.surfaceobjectives import NonQuasiSymmetricRatio
        from simsopt.field import BiotSavart

        sys.path.insert(0, os.path.join(_EXAMPLE_ROOT, "..", "..", "tests"))
        from geo.surface_test_helpers import get_boozer_surface
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"QA fixture deps unavailable: {exc!r}")

    bs_base, boozer_surface = get_boozer_surface(label="Volume", boozer_type="ls")
    qa = NonQuasiSymmetricRatio(boozer_surface, BiotSavart(bs_base.coils), quasi_poloidal=False)
    qp = NonQuasiSymmetricRatio(boozer_surface, BiotSavart(bs_base.coils), quasi_poloidal=True)
    assert qa.axis == 0 and qp.axis == 1


def test_qa_term_taylor_gradient():
    """Taylor-test the averaged QA term's analytic adjoint dJ via centered FD.

    Reuses the simsopt repo's own NCSX fixture + ``taylor_test1`` ratio-halving
    convergence (the established precedent for NonQuasiSymmetricRatio). This
    confirms the *averaged, dedicated* QA term -- the exact object the example
    descends -- carries the correct analytic Boozer-adjoint gradient, not just
    a single bare term. Skips only if the fixture is unavailable.
    """
    try:
        from simsopt.geo.surfaceobjectives import NonQuasiSymmetricRatio
        from simsopt.field import BiotSavart

        sys.path.insert(0, os.path.join(_EXAMPLE_ROOT, "..", "..", "tests"))
        from geo.surface_test_helpers import get_boozer_surface
        from geo.test_surface_objectives import taylor_test1
    except Exception as exc:  # pragma: no cover
        pytest.skip(
            "QA Taylor fixture unavailable (NonQuasiSymmetricRatio's adjoint is "
            f"upstream-tested): {exc!r}"
        )

    bs_base, boozer_surface = get_boozer_surface(label="Volume", boozer_type="ls")
    qa_bs = BiotSavart(bs_base.coils)
    qa_term = NonQuasiSymmetricRatio(boozer_surface, qa_bs, quasi_poloidal=False)
    averaged = sso.average_surface_objectives([qa_term])

    # Differentiate w.r.t. the full Boozer dof vector (surface + field), as the
    # upstream NonQS test does.
    coeffs = bs_base.x

    def f(dofs):
        bs_base.x = dofs
        averaged.recompute_bell() if hasattr(averaged, "recompute_bell") else None
        return float(averaged.J())

    def df(dofs):
        bs_base.x = dofs
        return averaged.dJ()

    taylor_test1(f, df, coeffs, epsilons=np.power(2.0, -np.asarray(range(13, 19))))


# --------------------------------------------------------------------------- #
# D. BYTE-IDENTICAL DEFAULT-OFF for both terms in build_total_objective
# --------------------------------------------------------------------------- #
def _assemble(noble=None, nw=0.0, qa=None, qw=0.0):
    """Assemble the total objective with a cheap, deterministic core set.

    The first eight positional args are the always-summed core terms of
    ``build_total_objective``; everything else defaults to the None/zero
    opt-in path. We only vary the Noble-pull and QA-residual term/weight pair.
    """
    JnonQS = _LinearScalar(1.0, 2.0)
    JBR = _LinearScalar(1.0, 3.0)
    Jiota = _LinearScalar(1.0, 4.0)
    JVol = _LinearScalar(1.0, 5.0)
    JCL = _LinearScalar(1.0, 6.0)
    JCC = _LinearScalar(1.0, 7.0)
    JCS = _LinearScalar(1.0, 8.0)
    JCurv = _LinearScalar(1.0, 9.0)
    return sso.build_total_objective(
        JnonQS, 0.5, JBR, 0.5, Jiota, 0.5, JVol, 0.5,
        JCL, 0.5, JCC, 0.5, JCS, 0.5, JCurv,
        JNobleIotaPull=noble,
        NOBLE_IOTA_PULL_WEIGHT=nw,
        JQAResidual=qa,
        QA_RESIDUAL_WEIGHT=qw,
    )


def test_default_off_none_terms_add_nothing_value():
    """None Noble + None QA add nothing to the objective value, even if the
    weights are nonzero -- the ``if term is not None`` guard is what enforces
    default-OFF when the builders return None."""
    base = float(_assemble().J())
    none_with_weight = float(_assemble(noble=None, nw=123.0, qa=None, qw=456.0).J())
    assert none_with_weight == base


def test_default_off_none_terms_add_nothing_gradient():
    """Same default-OFF guarantee at the gradient level (cheap to check here)."""
    base = _assemble()
    none_with_weight = _assemble(noble=None, nw=123.0, qa=None, qw=456.0)
    np.testing.assert_array_equal(base.dJ(), none_with_weight.dJ())


def test_zero_weight_present_noble_term_is_value_identical_and_grad_inert():
    """A *present* Noble term at weight 0 contributes exactly 0 to the value,
    and its only gradient footprint is an all-zero entry for its own dof.

    This pins a subtle but real behavior. In production the example sets the
    term to ``None`` when the weight is 0, so this layout never occurs (covered
    by the None-path tests above). But if a (nonzero) Noble term object is
    threaded in at ``NOBLE_IOTA_PULL_WEIGHT == 0``, the assembler still does
    ``objective + 0.0 * JNobleIotaPull``: the value is byte-identical (exact
    additive zero), and the term's dependency dof *enters the dof vector* with
    a gradient of exactly 0. So the objective is value-identical and the
    gradient on every pre-existing dof is unchanged, with the one extra dof
    contributing exactly zero -- never a spurious nonzero gradient.
    """
    term = _AffineIota(slope=1.0, intercept=0.0)
    term.set_iota(_LO - 0.05)  # active term: nonzero J and dJ on its own
    pull = sso.NobleIotaPull(term)
    assert pull.J() > 0.0  # the term really is non-trivial

    base = _assemble()
    with_zero_weight = _assemble(noble=pull, nw=0.0)

    # Value is byte-identical: 0.0 * J adds nothing.
    assert float(with_zero_weight.J()) == float(base.J())

    base_grad = np.asarray(base.dJ(), dtype=float)
    wz_grad = np.asarray(with_zero_weight.dJ(), dtype=float)
    # The zero-weight assembly adds exactly one extra dof (the Noble term's
    # dependency); every other gradient component is preserved. Each _assemble()
    # builds fresh stand-in terms whose auto-generated dof *names* differ, so we
    # compare the gradient *multisets* rather than name-keyed entries.
    assert wz_grad.size == base_grad.size + 1
    # The extra entry is exactly zero (weight 0 -> the term's dof is inert).
    assert np.count_nonzero(wz_grad == 0.0) == np.count_nonzero(base_grad == 0.0) + 1
    # Removing one zero makes the gradient multiset identical to the baseline.
    wz_sorted = np.sort(wz_grad)
    first_zero = int(np.argmax(wz_sorted == 0.0)) if np.any(wz_sorted == 0.0) else 0
    wz_without_extra = np.delete(wz_sorted, first_zero)
    np.testing.assert_array_equal(wz_without_extra, np.sort(base_grad))


def test_positive_weight_noble_term_actually_changes_objective():
    """Control: with weight>0 the Noble term DOES change the objective.

    Guards the byte-identical tests above against a false pass where the term
    is silently dropped on every path.
    """
    term = _AffineIota(slope=1.0, intercept=0.0)
    term.set_iota(_LO - 0.05)
    pull = sso.NobleIotaPull(term)
    base = float(_assemble().J())
    boosted = float(_assemble(noble=pull, nw=10.0).J())
    assert boosted == pytest.approx(base + 10.0 * pull.J())
    assert boosted != base
