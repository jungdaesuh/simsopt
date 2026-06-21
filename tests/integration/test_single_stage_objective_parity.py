"""Cross-boundary (native C++ vs JAX) parity for the single-stage banana objective.

This is a *genuine* cross-implementation parity test: it builds the full 8-term
single-stage banana objective ``JF`` twice through the example's SSOT factory
``build_single_stage_objective`` -- once with the native C++ backend
(``Iotas`` / ``NonQuasiSymmetricRatio`` / ``BoozerResidual`` over ``BiotSavart``)
and once with the JAX backend (the ``*JAX`` wrappers over ``BiotSavartJAX``) --
and compares the scalar value ``JF.J()`` and the gradient ``JF.dJ()`` at the
*same* coil-DOF point.

It is deliberately NOT a JAX-vs-JAX tautology: the two lanes route every
objective term through different implementations via the factory's ``use_jax``
switch (see ``select_single_stage_objective_backend``), which the test body
asserts *structurally* (native ``simsopt.*`` classes vs
``simsopt_jax_adapters.*`` ``*JAX`` classes).  Both lanes evaluate at the *same
resolved Boozer state* -- the JAX fixture is seeded with the CPU fixture's
converged surface DOFs / iota / G -- so the only differences are the
floating-point reduction order and the Boozer adjoint/VJP implementation.

Both lanes run float64 on CPU, so agreement is very tight.  Measured here, the
scalar total ``JF.J()`` is in fact *bit-identical* (the only active terms differ
~1e-20 absolute, well below the total's ULP), so the numerical cross-boundary
signal is carried by the gradient ``JF.dJ()`` (max abs diff ~1.2e-13), which the
test asserts is non-zero -- a zero gradient diff between distinct implementation
classes would mean the JAX VJP collapsed onto the native one and is worth
investigating, not silently passing.

Mirrors the bootstrap-import / skip conventions and tolerance bookkeeping of
``tests/integration/test_single_stage_jax_cpu_reference.py`` (the canonical
reduced-real CPU-vs-JAX wrapper parity file), and replicates its matched-fixture
pattern (``_build_real_fixture_ondevice_m5_same_state_pair``) so both objectives
are built on the identical converged Boozer state.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

# Boozer/quasi-symmetry CPU reference terms are C++-backed; the whole point of
# this file is to exercise the native lane, so it requires simsoptpp exactly like
# the canonical CPU-reference integration file.
sopp = pytest.importorskip(
    "simsoptpp",
    reason="Single-stage cross-boundary parity requires simsoptpp (use the in-tree .conda/jax env)",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.single_stage_smoke_fixture import (  # noqa: E402
    DEFAULT_NUM_TF_COILS,
    build_real_single_stage_init_fixture,
)
from benchmarks.validation_ladder_contract import (  # noqa: E402
    parity_ladder_tolerances,
)
from examples.single_stage_optimization.SINGLE_STAGE import (  # noqa: E402
    single_stage_banana_example as single_stage_example,
)


def _build_matched_same_state_fixture_pair():
    """Build CPU + JAX fixtures on the *identical* resolved Boozer state.

    Replicates ``_build_real_fixture_ondevice_m5_same_state_pair`` from
    ``test_single_stage_jax_cpu_reference.py``: solve the Boozer surface once on
    the native CPU lane, then seed the JAX lane's Boozer solve with the CPU
    lane's converged surface DOFs / iota / G so both objectives evaluate at the
    same converged state (same-state parity, not two independent solves).
    """
    cpu_fixture = build_real_single_stage_init_fixture(
        backend="cpu",
        optimizer_backend="scipy",
    )
    cpu_boozer = cpu_fixture["boozer_surface"]
    cpu_result = cpu_boozer.res
    assert cpu_result is not None and cpu_result.get("success", False), (
        "CPU reduced-real fixture did not converge"
    )
    jax_fixture = build_real_single_stage_init_fixture(
        backend="jax",
        optimizer_backend="ondevice",
        boozer_surface_dofs_override=np.asarray(
            cpu_boozer.surface.get_dofs(), dtype=float
        ),
        boozer_iota_override=float(cpu_result["iota"]),
        boozer_G_override=float(cpu_result["G"]),
    )
    return cpu_fixture, jax_fixture


# ``main()`` selects the residual kind via ``boozer_type[stage]`` where
# ``boozer_type = {"initial": "least_squares", "final": "exact"}`` and the
# default ``--boozer-stage`` is "initial".  "least_squares" is the
# constraint_weight=1.0 LS Boozer surface the smoke fixture builds.
# ``select_boozer_residual_class`` only special-cases the literal "exact"; every
# non-"exact" kind routes to ``BoozerResidual`` (native) / ``BoozerResidualJAX``
# (jax), so this LS string is the correct ``boozer_kind`` for this fixture.
_BOOZER_KIND = "least_squares"

# First ``num_tf_coils`` coils are the fixed TF set; the remainder are the banana
# coils whose curves drive the length / curvature penalties.
_NUM_TF_COILS = DEFAULT_NUM_TF_COILS


def _objective_weights_from_example_defaults():
    """Build ``SingleStageObjectiveWeights`` from the example's argparse defaults.

    Mirrors ``main()``'s weight assembly (including the ``max``/``min`` clamps
    against the module's hardware-minimum constants) so the objective under test
    is the realistic production objective rather than an ad-hoc one. The parity
    assertion only requires both lanes to use *identical* weights; deriving them
    from the shared defaults keeps this drift-proof if the example's defaults
    change.
    """
    argv_backup = sys.argv
    try:
        sys.argv = ["single_stage_banana_example.py"]
        args = single_stage_example.parse_args()
    finally:
        sys.argv = argv_backup

    cc_dist = max(args.cc_dist, single_stage_example.COIL_COIL_MIN_DIST_M)
    cs_dist = max(args.cs_dist, single_stage_example.COIL_PLASMA_MIN_DIST_M)
    ss_dist = max(args.ss_dist, single_stage_example.PLASMA_VESSEL_MIN_DIST_M)
    length_target = min(
        float(args.length_target), single_stage_example.COIL_LENGTH_TARGET_M
    )
    return single_stage_example.SingleStageObjectiveWeights(
        non_qs=args.non_qs_weight,
        res=args.res_weight,
        iotas=args.iotas_weight,
        length=args.length_weight,
        cc=args.cc_weight,
        cs=args.cs_weight,
        surf_dist=args.surf_dist_weight,
        curvature=args.curvature_weight,
        cc_dist=cc_dist,
        cs_dist=cs_dist,
        ss_dist=ss_dist,
        curvature_threshold=args.curvature_threshold,
        length_target=length_target,
    )


def _build_lane_objective(fixture, *, use_jax):
    """Assemble the SSOT single-stage objective for one backend lane.

    Uses exactly the same factory entry points ``main()`` uses so the native and
    JAX lanes minimize byte-identical objective *definitions*; only the term
    implementations differ via ``use_jax``.
    """
    bs = fixture["bs"]
    coils = bs.coils
    curves = [c.curve for c in coils]
    banana_curves = [c.curve for c in coils[_NUM_TF_COILS:]]

    backend = single_stage_example.select_single_stage_objective_backend(
        use_jax=use_jax,
        boozer_kind=_BOOZER_KIND,
        # ``jax_bs`` is only consumed on the JAX lane; ``coils`` only on the
        # native lane (it builds its own ``BiotSavart(coils)``).
        jax_bs=bs if use_jax else None,
        coils=coils,
    )
    objective = single_stage_example.build_single_stage_objective(
        boozer_surface=fixture["boozer_surface"],
        backend=backend,
        banana_curves=banana_curves,
        curves=curves,
        vessel_surface=single_stage_example.build_single_stage_vessel_surface(),
        iota_target=fixture["iota_target"],
        weights=_objective_weights_from_example_defaults(),
    )
    return objective


# --- Tolerances -------------------------------------------------------------
#
# Both lanes are float64 on CPU evaluated at the *same* resolved Boozer state, so
# the only sources of disagreement are (a) floating-point reduction order across
# the native-C++ vs JAX/XLA implementations of each term and (b) the Boozer
# adjoint used inside the ``*Ratio``/``Iotas``/``BoozerResidual`` gradients.
#
# MEASURED on this clean repo (51 coil DOFs; native simsoptpp ABI from the
# in-tree .conda/jax env, all Python from the clean repo), at the matched
# converged Boozer state:
#   * VALUE  ``JF.J()``  : abs diff == 0.0 (the weighted total is bit-identical).
#     This is a genuine forward-pass coincidence, NOT a tautology: the only
#     active terms (JnonQSRatio ~1.7e-4, JBoozerResidual ~7.6e-6) differ between
#     native and JAX at ~5e-20 / ~7e-20 absolute -- about 4 orders below the
#     total's ULP (total ~1.13, ULP ~2e-16) -- so the sum rounds identically.
#     The cross-boundary signal is carried by the gradient (below) and by the
#     per-term values; non-tautology is *also* asserted structurally
#     (native ``simsopt.*`` classes vs ``simsopt_jax_adapters.*`` ``*JAX``
#     classes) in the test body, so a bit-identical total cannot mask a
#     jax-vs-jax comparison.
#   * GRAD   ``JF.dJ()`` : max abs diff == 1.19e-13, inf-norm rel == 2.83e-14,
#     max per-element rel == 1.09e-12 (||cpu_dJ||_inf == 4.22).  This is the
#     native-vs-JAX adjoint/VJP reduction-order drift.
#
# VALUE band: SOURCED from the parity-ladder SSOT -- the ``ls_wrapper_gradient``
# tier is the repo's objective-subtotal/gradient same-state contract
# (``objective_native_subtotal``/``gradient`` map to it in
# ``validation_ladder_contract``).  The total is bit-identical here so this band
# never binds, but sourcing it keeps the test in lockstep with the ladder rather
# than hardcoding a copy that can drift.  GRAD band: kept as MEASURED literals --
# rtol 1e-11 (~10x over the 1.09e-12 per-element rel) and atol 1e-12 (~8x over
# the 1.19e-13 abs floor).  These are deliberately TIGHTER than the ladder's
# 1e-10 (never loosened to it); they bracket the measured cross-implementation
# adjoint drift with single-digit headroom.
#
# If collected without the equilibrium / Stage-2 fixtures or a usable simsoptpp,
# the test is skipped above (importorskip) rather than run uncalibrated.
_VALUE_TOLS = parity_ladder_tolerances("ls-wrapper-gradient")
_VALUE_RTOL = _VALUE_TOLS["rtol"]
_VALUE_ATOL = _VALUE_TOLS["atol"]
_GRAD_RTOL = 1e-11
_GRAD_ATOL = 1e-12


@pytest.mark.integration
@pytest.mark.slow
class TestSingleStageObjectiveCrossBoundaryParity:
    """Native C++ objective vs JAX objective at one matched coil-DOF point."""

    def test_value_and_gradient_match_native_vs_jax(self):
        cpu_fixture, jax_fixture = _build_matched_same_state_fixture_pair()

        # Sanity: matched-state fixture really did converge both Boozer solves and
        # the JAX lane ran the on-device LS solver (not a CPU re-solve).
        cpu_booz = cpu_fixture["boozer_surface"]
        jax_booz = jax_fixture["boozer_surface"]
        assert cpu_booz.res is not None and cpu_booz.res.get("success", False)
        assert jax_booz.res is not None and jax_booz.res.get("success", False)
        assert jax_fixture["boozer_optimizer_backend"] == "ondevice"

        cpu_obj = _build_lane_objective(cpu_fixture, use_jax=False)
        jax_obj = _build_lane_objective(jax_fixture, use_jax=True)

        # STRUCTURAL non-tautology guarantee: the native lane must route every
        # Boozer/quasi-symmetry term through the C++-backed ``simsopt.*`` classes
        # over a native ``BiotSavart`` field, and the JAX lane through the
        # ``simsopt_jax_adapters.*`` ``*JAX`` wrappers over ``BiotSavartJAX``.
        # This makes the cross-boundary claim hold even when the scalar totals
        # happen to coincide bit-for-bit (they do here; see the tolerance note),
        # so the test can never silently degrade into a jax-vs-jax comparison.
        for term in (cpu_obj.nonQSs[0], cpu_obj.brs[0], cpu_obj.iota):
            module = type(term).__module__
            assert module.startswith("simsopt.") and "_jax" not in module, (
                f"Native lane term {type(term).__name__} came from {module!r}; "
                "expected a C++-backed simsopt.* class"
            )
        for term in (jax_obj.nonQSs[0], jax_obj.brs[0], jax_obj.iota):
            module = type(term).__module__
            assert module.startswith("simsopt_jax"), (
                f"JAX lane term {type(term).__name__} came from {module!r}; "
                "expected a simsopt_jax_adapters.* *JAX class"
            )

        cpu_JF = cpu_obj.JF
        jax_JF = jax_obj.JF

        # The two objectives are built from the same Stage-2 seed coils, so the
        # Optimizable free-DOF vectors must share a layout.  Compare in a matched
        # layout (NOT raw if sizes differ): assert identical size, then drive both
        # graphs to the *same* coil-DOF point ``x0`` (the native lane's current
        # DOFs).  Setting ``JF.x`` to its current value keeps both lanes on the
        # already-resolved Boozer state.
        cpu_x = np.asarray(cpu_JF.x, dtype=float)
        jax_x = np.asarray(jax_JF.x, dtype=float)
        assert cpu_x.shape == jax_x.shape, (
            "Native and JAX JF.x layouts differ "
            f"(cpu={cpu_x.shape}, jax={jax_x.shape}); the parity comparison "
            "requires a matched DOF layout."
        )
        # Same seed -> same initial coil DOFs; this also documents the shared
        # coordinate system the gradients are compared in.
        np.testing.assert_allclose(
            jax_x,
            cpu_x,
            rtol=0.0,
            atol=1e-12,
            err_msg="Native and JAX JF.x seeds disagree before evaluation",
        )

        x0 = cpu_x.copy()
        cpu_JF.x = x0
        jax_JF.x = x0

        # COVERAGE LOCK.  At this feasible Stage-2 seed the five geometric penalty
        # terms are hard-zero (each is threshold-gated: a soft-min distance with no
        # pair under the threshold returns 0, and the "max"/Lp penalties clamp to
        # exactly 0.0 below their length/curvature thresholds), so the cross-boundary
        # signal asserted below is carried by the field/surface-coupled terms
        # (NonQS, BoozerResidual, iota) -- the ONLY terms the factory switches by
        # backend (native ``simsopt.*`` vs ``simsopt_jax_adapters.*JAX``).  The five
        # geometric penalties are NOT separately exercised across the boundary here:
        # the coil-curve terms (length, curve-curve, curvature) use the same curve
        # class in both lanes, and only the surface-distance terms see a different
        # surface object -- and all five are dormant at this seed regardless.
        # Assert they are inactive in BOTH lanes so the documented active-term scope
        # is locked rather than trusted: if a future change activates any of them,
        # this assertion fails and the test must be revisited.
        for _label, _cpu_term, _jax_term in (
            ("JCurveLength", cpu_obj.JCurveLength, jax_obj.JCurveLength),
            ("JCurveCurve", cpu_obj.JCurveCurve, jax_obj.JCurveCurve),
            ("JCurveSurface", cpu_obj.JCurveSurface, jax_obj.JCurveSurface),
            ("JSurfSurf", cpu_obj.JSurfSurf, jax_obj.JSurfSurf),
            ("JCurvature", cpu_obj.JCurvature, jax_obj.JCurvature),
        ):
            _cpu_term_value = float(np.asarray(_cpu_term.J(), dtype=float))
            _jax_term_value = float(np.asarray(_jax_term.J(), dtype=float))
            assert abs(_cpu_term_value) <= 1e-12 and abs(_jax_term_value) <= 1e-12, (
                f"{_label} expected inactive (~0) at the feasible seed but got "
                f"native={_cpu_term_value:.3e} jax={_jax_term_value:.3e}; the "
                "geometric penalties are now active -- extend this test to compare "
                "them across the native/JAX boundary."
            )

        cpu_J = float(cpu_JF.J())
        jax_J = float(np.asarray(jax_JF.J(), dtype=float))

        cpu_dJ = np.asarray(cpu_JF.dJ(), dtype=float)
        jax_dJ = np.asarray(jax_JF.dJ(), dtype=float)

        assert cpu_dJ.shape == jax_dJ.shape, (
            f"Gradient layouts differ (cpu={cpu_dJ.shape}, jax={jax_dJ.shape})"
        )
        assert np.all(np.isfinite(cpu_dJ)) and np.linalg.norm(cpu_dJ) > 0
        assert np.all(np.isfinite(jax_dJ)) and np.linalg.norm(jax_dJ) > 0

        value_abs_diff = abs(jax_J - cpu_J)
        value_rel_diff = value_abs_diff / (abs(cpu_J) + 1e-30)
        grad_abs_diff = float(np.max(np.abs(jax_dJ - cpu_dJ)))
        grad_rel_diff = grad_abs_diff / (float(np.max(np.abs(cpu_dJ))) + 1e-30)

        # Diagnostics surface the measured cross-boundary diffs on failure so the
        # tolerance can be (re)calibrated against real numbers, not guesses.
        diagnostics = (
            f"native J={cpu_J:.15e} jax J={jax_J:.15e} "
            f"|Δ|={value_abs_diff:.3e} rel={value_rel_diff:.3e}\n"
            f"grad |Δ|_inf={grad_abs_diff:.3e} rel={grad_rel_diff:.3e} "
            f"(ndof={cpu_dJ.size})"
        )

        # Numeric non-tautology guard (complements the structural class check
        # above).  The scalar total is legitimately bit-identical here (the
        # active terms differ ~1e-20, below the total's ULP), so the *gradient*
        # carries the numerical cross-boundary signal: native and JAX use
        # different adjoint/VJP reductions and must NOT agree bit-for-bit.  A
        # zero gradient diff with distinct implementation classes would mean the
        # JAX VJP collapsed onto the native one -- worth investigating, not
        # silently passing.
        assert grad_abs_diff != 0.0, (
            "Native and JAX gradients agreed bit-for-bit despite distinct "
            "implementation classes; the cross-implementation adjoint drift "
            f"vanished unexpectedly -- investigate.\n{diagnostics}"
        )

        np.testing.assert_allclose(
            jax_J,
            cpu_J,
            rtol=_VALUE_RTOL,
            atol=_VALUE_ATOL,
            err_msg="Single-stage objective VALUE native-vs-JAX parity failed.\n"
            + diagnostics,
        )
        np.testing.assert_allclose(
            jax_dJ,
            cpu_dJ,
            rtol=_GRAD_RTOL,
            atol=_GRAD_ATOL,
            err_msg="Single-stage objective GRADIENT native-vs-JAX parity failed.\n"
            + diagnostics,
        )
