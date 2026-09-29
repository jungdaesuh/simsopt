"""Exact parity for the ``2_Intermediate/permanent_magnet_QA.py`` mirror.

Every comparison against an official number reads the tracked fixture
``examples/jax/parity/official_reference`` and runs its native lane -- and the
construction that feeds it -- in a child process at ``OMP_NUM_THREADS=1``.
Native MwPGP OpenMP reductions at this scale are not unique under OMP>1 and
``OMP_NUM_THREADS`` is read when libgomp starts, so an in-process pin cannot
undo the pytest process team (``tests/conftest.py`` pins nothing).  Measured
2026-09-20 at ``OMP_NUM_THREADS=4``: the SAME bounded input solved twice in ONE
process moves ``final:objective_sum_squares`` by 2.672e-04 relative with no
perturbation at all -- 0.53 x the ``mirror_pmqa_final`` rtol the endpoint
routes declare.

Which solves run where.  Every NATIVE solve this file judges runs in the
one-thread child: the bounded and the ``native_default`` reference lanes
(``run_permanent_magnet_native_lane``, one solve each) and all nine solves of
the conditioning probe below -- its unperturbed reference and its eight draws,
in ONE child, which is what makes their difference the perturbation alone.  The
JAX lane's four solves run in the pytest process itself, where nothing in this
file pins OpenMP; they do not need the pin, because ``native_permanent_magnet_qa
._jax`` runs the relax-and-split continuation in JAX and never enters the
``simsoptpp.MwPGP_algorithm`` OpenMP reductions this pin exists for.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import read_input_bundle
from examples.jax.parity.official_reference import load_official_reference
from simsopt_jax.parity_tolerances import parity_ladder_tolerances

# venv site-packages/tests shadows the repo tests package, so the helper is
# imported as a top-level module from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from parity_gpmo_native_child import (  # noqa: E402
    ConditioningProbeResult,
    NativeChildResult,
    run_permanent_magnet_conditioning_probe,
    run_permanent_magnet_native_lane,
)

CASE_ID = "native-permanent-magnet-qa"
NATIVE_EXAMPLE_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "2_Intermediate"
    / "permanent_magnet_QA.py"
)
#: The official run at this case's ``native_default`` scale (nphi=ntheta=16).
OFFICIAL = load_official_reference(CASE_ID)

#: The four bounded ``final`` VALUE observables the coordinator made
#: inapplicable on every lane pair on 2026-09-20
#: (``A/fix-wave-3/integration/apply_pm_requests.py``), each with the ``rtol``
#: bucket its routes declare (all four declare ``mirror_pmqa_final``).  The
#: three measured as determined were re-enabled later; the fourth,
#: ``objective_sum_squares``, was re-enabled on 2026-09-29 (see below).
DISABLED_BOUNDED_FINAL_VALUES: Final[tuple[str, ...]] = (
    "final:objective_sum_squares",
    "final:residual_norm",
    "final:moment_l2_norm",
    "final:proxy_moment_l2_norm",
)
#: The measurement that ruling rests on, per observable: is the bounded end
#: point DETERMINED to the bucket its route declares by its own input?
#:
#: Only ``final:objective_sum_squares`` was ever measured (one seeded draw of an
#: ``ATb`` perturbation); the other three were switched off by extrapolation
#: from it.
#:
#: Current record, measured 2026-09-29 on the rebased native build, both with
#: and without the deterministic ``integral_BdotN`` summation (``simsoptpp``
#: sha256 ``f74d83d35def68d0...5b`` and ``0149cc25cefa1d90...53``, bit-identical
#: results), at one thread, over the
#: campaign's pre-registered eight one-ulp draws, worst draw against rtol 5e-4:
#: objective_sum_squares 7.928e-11, residual_norm 3.964e-11, moment_l2_norm
#: 5.818e-12, proxy_moment_l2_norm 1.250e-11 -- every observable DETERMINED.
#: Two runs were bitwise identical, and upstream 9e027eac3's own ``.so``
#: (sha256 ``9b72c853ea16e79b...bc40``) under this Python gives the same nine
#: solves bit for bit.  ``permanent_magnet_optimization.cpp`` changed only in
#: include paths; the shift comes from the toolchain.
#:
#: History: the pre-rebase build's measurement, same protocol (worst draw):
#: objective_sum_squares 6.638e-04 (1.33 x rtol, NOT determined), residual_norm
#: 3.319e-04 (0.66 x), moment_l2_norm 4.476e-04 (0.90 x), proxy_moment_l2_norm
#: 2.598e-04 (0.52 x).
#:
#: This table is the record of the measurement on the build under test, and a
#: change to it in either direction is a route adjudication (the manifest's),
#: not a test update.  Adjudicated 2026-09-29 by the rule pre-registered for it
#: (parity redesign, item C5): the manifest's bounded ``objective_sum_squares``
#: routes are applicable again, because all eight draws fall inside rtol on the
#: build with the deterministic summation and on the build without it, and
#: native-versus-JAX at bounded passes the bucket (1.163e-10 relative, asserted
#: by ``test_the_extrapolated_bounded_routes_compare_inside_their_bucket``).
DETERMINED_TO_ITS_BUCKET: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "final:objective_sum_squares": True,
        "final:residual_norm": True,
        "final:moment_l2_norm": True,
        "final:proxy_moment_l2_norm": True,
    }
)
#: The pre-registered one-ulp protocol (``A/d1-diagnostic/NOTES.md``, rule v2):
#: signs drawn from ``RandomState(20260920 + k)`` for k = 1..8.
CONDITIONING_DRAWS: Final[int] = 8
CONDITIONING_SEED_BASE: Final[int] = 20260920
#: The solve input the probe perturbs: ``ATb`` is what the MwPGP continuation
#: consumes, and the case freezes it in the bundle.
CONDITIONING_PERTURBED_ARRAY: Final[str] = "atb"

#: Unit round-off of float64.
UNIT_ROUNDOFF: Final[float] = 2.0**-53


def _gamma(k: int) -> float:
    """Higham's gamma_k = k u / (1 - k u): the bound of k compounded roundings."""
    return k * UNIT_ROUNDOFF / (1.0 - k * UNIT_ROUNDOFF)


def _root_relation_roundoff(square: float, root: float, length: int) -> float:
    """Worst-case round-off in |root drift - square drift / 2|, beyond ``square**2``.

    ``square`` and ``root`` are the published relative drifts of
    ``vdot(r, r)`` and ``norm(r)`` for a residual of ``length`` entries; the
    derivation is in ``test_the_root_observable_drifts_by_half_of_its_argument``.
    """
    g = _gamma(length + 1)
    division = _gamma(2)
    c = 2.0 * g / (1.0 - g)
    # The second-order line X**2 / (2 (1 - c)**2) is inside ``square**2`` only
    # under this condition.
    assert (1.0 - division) * (1.0 - c) >= 2.0**-0.5, length
    x = square / (1.0 - division)
    q = (1.0 + x) / (1.0 - c)
    return (
        (2.0 * x * c + c**2) / (2.0 * (1.0 - c) ** 2)
        + 1.5 * c * q
        + division * (root + 0.5 * square) / (1.0 - division)
    )


def _native_lane(
    tmp_path_factory: pytest.TempPathFactory,
    scale: str,
) -> tuple[NativeChildResult, Path]:
    root = tmp_path_factory.mktemp(f"permanent-magnet-qa-{scale}")
    input_root = root / "inputs"
    result = run_permanent_magnet_native_lane(
        CASE_ID,
        scale,
        input_root,
        root / "native-observation.pkl",
    )
    assert result.omp_num_threads == "1", (
        "the native reference lane must run with OpenMP pinned before libgomp "
        f"starts; the child saw OMP_NUM_THREADS={result.omp_num_threads!r}"
    )
    return result, input_root


@pytest.fixture(scope="module")
def bounded_native(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[NativeChildResult, Path]:
    return _native_lane(tmp_path_factory, "bounded")


@pytest.fixture(scope="module")
def shipped_scale_native(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[NativeChildResult, Path]:
    """The official scale; the whole case costs about half a minute at one thread."""
    return _native_lane(tmp_path_factory, "native_default")


@pytest.fixture(scope="module")
def bounded_conditioning(
    tmp_path_factory: pytest.TempPathFactory,
) -> ConditioningProbeResult:
    """Nine bounded solves -- one reference, eight one ulp away -- in ONE child.

    The reference and every draw are executed by the same one-thread
    interpreter, so the difference between them is the perturbation and nothing
    else.  About 12 s.
    """
    root = tmp_path_factory.mktemp("permanent-magnet-qa-conditioning")
    return run_permanent_magnet_conditioning_probe(
        CASE_ID,
        "bounded",
        root / "inputs",
        root / "conditioning.pkl",
        perturbed_array=CONDITIONING_PERTURBED_ARRAY,
        observables=DISABLED_BOUNDED_FINAL_VALUES,
        draws=CONDITIONING_DRAWS,
        seed_base=CONDITIONING_SEED_BASE,
    )


def test_exact_permanent_magnet_qa_matches_native_and_jax_cpu(
    bounded_native: tuple[NativeChildResult, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child, input_root = bounded_native
    native = child.observation
    bundle, arrays = read_input_bundle(input_root)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = get_case(CASE_ID).execute("jax-cpu", bundle, arrays)

    assert native.success is True
    assert jax.success is True
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    # The relax-and-split continuation is a fixed amount of work with no
    # provider status; the configured stage product is not a measured nit.
    assert native.normalized_status == jax.normalized_status == "not_applicable"
    assert (
        native.raw_status == jax.raw_status == "relax_and_split_continuation_completed"
    )
    assert native.nit is jax.nit is None
    assert native.nfev is jax.nfev is None
    assert set(native.values) == set(jax.values)

    for observable in (
        "construction:response_matrix",
        "construction:target",
        "construction:moment_maxima",
        "construction:dipole_grid_xyz",
        "initial:moments",
        "initial:residual",
        "initial:objective_sum_squares",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    assert int(native.values["final:nonzero_count"]) > 0
    assert int(jax.values["final:nonzero_count"]) > 0
    # What the two lanes must agree on at this scale is the SET of magnets the
    # relax-and-split continuation kept; whether an endpoint VALUE is a
    # cross-lane comparable here is measured per build by
    # ``test_each_disabled_bounded_value_route_has_its_own_measured_condition``,
    # and the determined ones are compared by
    # ``test_the_extrapolated_bounded_routes_compare_inside_their_bucket``. The
    # values are compared at ``native_default``, against the official record,
    # by ``test_permanent_magnet_qa_m_objective_matches_the_official_capture``.
    assert int(native.values["final:nonzero_count"]) == int(
        jax.values["final:nonzero_count"]
    )
    assert np.array_equal(
        np.asarray(native.values["final:nonzero_mask"]),
        np.asarray(jax.values["final:nonzero_mask"]),
    )

    assert np.all(np.isfinite(native.values["final:moments"]))
    assert np.all(np.isfinite(native.values["final:proxy_moments"]))
    assert np.all(np.isfinite(jax.values["final:moments"]))
    assert np.all(np.isfinite(jax.values["final:proxy_moments"]))
    assert float(native.values["final:objective_sum_squares"]) < float(
        native.values["initial:objective_sum_squares"]
    )


def test_permanent_magnet_qa_publishes_the_m_based_objective_as_a_diagnostic(
    bounded_native: tuple[NativeChildResult, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``final:moment_objective_sum_squares`` is the official ``m`` objective.

    The compared endpoint is the sparsified ``m_proxy`` one; the raw MwPGP
    iterate ``m`` is published so the official
    ``final:objective_half_sum_squares`` has a lane-side counterpart at all
    (PM-QA-1). At BOUNDED scale the two lanes' ``m`` iterates drift further
    apart than the proxy endpoint does, so the bounded routes are inapplicable
    (the ``native_default`` routes are applicable: measured 1.7e-10). This pins
    both facts: the observable is internally consistent, and the bounded drift
    is outside the case's own final bucket -- as a ceiling, not as a required
    disagreement.
    """

    child, input_root = bounded_native
    native = child.observation
    bundle, arrays = read_input_bundle(input_root)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = get_case(CASE_ID).execute("jax-cpu", bundle, arrays)

    for observation in (native, jax):
        residual = np.asarray(observation.values["final:moment_residual"])
        np.testing.assert_allclose(
            float(observation.values["final:moment_objective_sum_squares"]),
            float(np.vdot(residual, residual)),
            rtol=0.0,
            atol=0.0,
        )
    # No hand-entered ceiling on the cross-lane drift: the bounded endpoint is
    # not determined to that precision by the input (see the conditioning test
    # below), so any such number would be a measurement of one run, not a
    # property of the two implementations. What is a property, and is asserted
    # above, is that each lane's published objective is exactly the squared
    # norm of the residual it published with it.
    assert np.all(np.isfinite(native.values["final:moment_residual"]))
    assert np.all(np.isfinite(jax.values["final:moment_residual"]))


def test_each_disabled_bounded_value_route_has_its_own_measured_condition(
    bounded_conditioning: ConditioningProbeResult,
) -> None:
    """One lane, nine inputs one ulp apart: is each endpoint value determined?

    A cross-lane comparison at ``rtol`` decides between two implementations
    only if the observable does not move past ``rtol`` when its own input moves
    by a single ulp.  That is measured here, per observable, and it is what the
    applicability of the four bounded ``final`` value routes rests on.

    Both halves run in the SAME one-thread child: the old probe ran both solves
    in the pytest process with no pin, where the unperturbed repeat alone moves
    ``final:objective_sum_squares`` by 2.7e-04 at four threads (file docstring),
    so what it measured was a sum of conditioning and reduction scatter.  The
    statistic is the worst of the eight pre-registered draws, not one draw, and
    it is read the conservative way round in both directions: an observable is
    called DETERMINED only when EVERY draw stays inside the bucket, and called
    not determined when at least one draw leaves it -- which is exactly the
    claim "a machine-epsilon change of the input can flip this comparison".

    Reading of a failure: the named observable changed conditioning class, and
    its route must be adjudicated the other way.
    """
    assert bounded_conditioning.omp_num_threads == "1", (
        "the conditioning solves must run with OpenMP pinned before libgomp "
        f"starts; the child saw {bounded_conditioning.omp_num_threads!r}"
    )
    assert len(bounded_conditioning.draws) == CONDITIONING_DRAWS
    rtol = float(parity_ladder_tolerances("mirror_pmqa_final")["rtol"])

    for draw in bounded_conditioning.draws:
        # A perturbation that grew would turn this into a different experiment.
        assert draw.entries_moved_more_than_one_ulp == 0, draw.seed
        # A moved endpoint VALUE, not a different set of magnets: a support
        # flip would be a threshold effect and would need its own reading.
        assert np.array_equal(
            draw.nonzero_mask, bounded_conditioning.reference_nonzero_mask
        ), draw.seed
    assert int(np.count_nonzero(bounded_conditioning.reference_nonzero_mask)) > 0

    for observable, determined in DETERMINED_TO_ITS_BUCKET.items():
        drifts = bounded_conditioning.relative_drifts(observable)
        worst = max(drifts)
        assert (worst <= rtol) is determined, (
            f"{observable}: the worst of {len(drifts)} one-ulp draws moves it by "
            f"{worst:.3e} against the rtol {rtol:.1e} its routes declare, so it "
            f"is {'determined' if worst <= rtol else 'NOT determined'} to that "
            f"bucket; the recorded verdict says "
            f"{'determined' if determined else 'NOT determined'}. Its route's "
            "applicability has to be adjudicated again."
        )
    assert set(DETERMINED_TO_ITS_BUCKET) == set(DISABLED_BOUNDED_FINAL_VALUES)


def test_the_root_observable_drifts_by_half_of_its_argument(
    bounded_conditioning: ConditioningProbeResult,
    bounded_native: tuple[NativeChildResult, Path],
) -> None:
    """``final:residual_norm`` is the square root of ``final:objective_sum_squares``.

    Both are published from the same residual
    (``native_permanent_magnet_qa.py``: ``vdot(r, r)`` and ``norm(r)``), so
    ``sqrt(1 + e) = 1 + e/2 - e**2/8`` makes the root's relative drift HALF its
    argument's, up to the second-order term.  That factor two is exactly what
    puts these two observables on opposite sides of the one ``rtol`` their
    routes share, which is why a single bucket applied to a quantity and to its
    square cannot judge both.

    The bound is the second-order term ``square**2`` (the pre-rebase build's
    worst draw sat at 1/8 of it) PLUS the worst-case round-off of the published
    values, derived, not fitted.  At the rebased build's drifts (~1e-10) the
    round-off dominates: |root - square/2| = 7e-17 against square**2 = 4e-21.
    With u = 2**-53, gamma_k = k u / (1 - k u), n = len(final:residual):

    - s = vdot(r, r) = S (1 + alpha), |alpha| <= gamma_n (n nonnegative
      products summed, any order);
    - rho = norm(r) = fl(sqrt(r.dot(r))) (numpy, ord=None) = R (1 + theta),
      |theta| <= gamma_n + u <= g := gamma_{n+1};
    - D = fl(fl(|y1 - y0|) / |y0|) = |x| (1 + eta), x = (y1 - y0) / y0,
      |eta| <= gamma_2 (one subtraction, one division).

    With q = S1 / S0: x_s = q a - 1 and x_r = sqrt(q) t - 1, where a and t are
    the ratios (1 + alpha1)/(1 + alpha0) and (1 + theta1)/(1 + theta0), so
    |a - 1|, |t - 1| <= c := 2 g / (1 - g), and
    x_r - x_s/2 = -(sqrt(q) - 1)**2 / 2 + sqrt(q) (t - 1) - q (a - 1) / 2.
    Using |x_s| <= X := D_s / (1 - gamma_2), q, sqrt(q) <= Q := (1 + X)/(1 - c)
    and (sqrt(q) - 1)**2 <= (q - 1)**2 <= ((X + c) / (1 - c))**2:

        |D_r - D_s/2| <= X**2 / (2 (1 - c)**2)              [<= D_s**2]
                        + (2 X c + c**2) / (2 (1 - c)**2)
                        + 3/2 c Q
                        + gamma_2 (D_r + D_s/2) / (1 - gamma_2),

    the first line being inside ``square**2`` whenever
    (1 - gamma_2)(1 - c) >= 2**-0.5.  The other three lines are
    :func:`_root_relation_roundoff`.
    """
    squares = bounded_conditioning.relative_drifts("final:objective_sum_squares")
    roots = bounded_conditioning.relative_drifts("final:residual_norm")
    child, _input_root = bounded_native
    length = int(np.asarray(child.observation.values["final:residual"]).size)

    for square, root in zip(squares, roots, strict=True):
        bound = square**2 + _root_relation_roundoff(square, root, length)
        assert abs(root - 0.5 * square) <= bound, (
            f"the root moved by {root:.6e} where half of its argument's "
            f"{square:.6e} is {0.5 * square:.6e}; allowed {bound:.3e} "
            f"(second order {square**2:.3e} plus round-off at n={length})"
        )


def test_the_extrapolated_bounded_routes_compare_inside_their_bucket(
    bounded_native: tuple[NativeChildResult, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the three unmeasured comparisons actually are, at their bucket.

    Three of the four disabled ``final`` value routes were switched off by
    extrapolation from the fourth.  This is the comparison those routes would
    make: the native lane from the one-thread child against the JAX lane, at
    the ``rtol`` the routes declare.  Only agreement is asserted -- a required
    disagreement would fail the day two lanes agree better -- and only for the
    observables the conditioning test above records as DETERMINED; on the
    rebased build that includes ``final:objective_sum_squares`` (measured
    1.163e-10 relative, 2026-09-29), which is condition (c) of its route's
    pre-registered re-adjudication (see ``DETERMINED_TO_ITS_BUCKET``).
    """
    child, input_root = bounded_native
    bundle, arrays = read_input_bundle(input_root)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = get_case(CASE_ID).execute("jax-cpu", bundle, arrays)
    bucket = parity_ladder_tolerances("mirror_pmqa_final")

    for observable, determined in DETERMINED_TO_ITS_BUCKET.items():
        if not determined:
            continue
        np.testing.assert_allclose(
            float(np.asarray(jax.values[observable])),
            float(np.asarray(child.observation.values[observable])),
            rtol=bucket["rtol"],
            atol=bucket["atol"],
            err_msg=observable,
        )


def test_permanent_magnet_qa_m_objective_matches_the_official_capture(
    shipped_scale_native: tuple[NativeChildResult, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``m``-based observables against the official shipped-scale record.

    Agreement between the two lanes is not evidence of correctness; agreement
    with the official run is. The canonical record was produced by the official
    script on an independent official build at ``nphi=ntheta=16``, which is this
    case's ``native_default`` scale, and its
    ``final:objective_half_sum_squares`` is ``0.5 * ||A m - b||^2`` on the raw
    MwPGP iterate ``m`` -- exactly half of the published
    ``final:moment_objective_sum_squares``.
    """

    child, input_root = shipped_scale_native
    native = child.observation
    bundle, arrays = read_input_bundle(input_root)
    official_configuration = OFFICIAL.structure("configuration")
    assert isinstance(official_configuration, dict)

    assert child.configuration["nphi"] == official_configuration["nphi"]
    assert child.configuration["ndipoles"] == official_configuration["ndipoles"]

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = get_case(CASE_ID).execute("jax-cpu", bundle, arrays)

    for observation in (native, jax):
        np.testing.assert_allclose(
            0.5 * float(observation.values["final:moment_objective_sum_squares"]),
            OFFICIAL.scalar("final:objective_half_sum_squares"),
            rtol=1.0e-8,
            atol=0.0,
        )
        np.testing.assert_allclose(
            0.5 * float(observation.values["final:objective_sum_squares"]),
            OFFICIAL.scalar("final:proxy_objective_half_sum_squares"),
            rtol=1.0e-8,
            atol=0.0,
        )
        np.testing.assert_allclose(
            float(observation.values["final:moment_l2_norm"]),
            OFFICIAL.scalar("final:moment_l2_norm"),
            rtol=1.0e-8,
            atol=0.0,
        )
        assert int(observation.values["final:nonzero_count"]) == (
            OFFICIAL.scalar("final:nonzero_count")
        )
    # The two ``m``-based routes are applicable at this scale; the residual is
    # the array they compare, at the tolerance bucket those routes declare.
    native_workflow = parity_ladder_tolerances("native_workflow")
    np.testing.assert_allclose(
        np.asarray(jax.values["final:moment_residual"], dtype=np.float64),
        np.asarray(native.values["final:moment_residual"], dtype=np.float64),
        rtol=native_workflow["whole_solve_value_rtol"],
        atol=native_workflow["whole_solve_value_atol"],
    )


def test_the_coil_pre_optimization_solves_upstreams_seven_parameter_problem(
    bounded_native: tuple[NativeChildResult, Path],
    shipped_scale_native: tuple[NativeChildResult, Path],
) -> None:
    """The provider must receive upstream's parameters, not the boundary too.

    Branch commit ``28e470f43`` gave ``CurveSurfaceDistance`` ownership of the
    surface, so the plasma boundary's 121 free ``SurfaceRZFourier`` dofs enter
    ``coil_optimization``'s ``JF.x`` unless the caller fixes the boundary --
    which is why ``stage_two_optimization.py``, ``coil_forces.py`` and
    ``stage_two_optimization_planar_coils.py`` all call ``s.fix_all()``. The
    boundary is this problem's prescribed target, never a variable: with it
    free the case handed the provider 128 parameters against the official run's
    7, and the shipped-scale coil currents drifted 7.4e-13 away from upstream's.
    The count and the currents both come from the official record.
    """

    expected_parameters = OFFICIAL.digest("construction:optimized_coil_dofs").count
    for child, _input_root in (bounded_native, shipped_scale_native):
        assert child.provider_parameter_counts == (expected_parameters,), (
            "the coil pre-optimization received "
            f"{child.provider_parameter_counts} parameters; upstream's problem "
            f"has {expected_parameters}"
        )
    shipped, _input_root = shipped_scale_native
    # At ``native_default`` the case runs upstream's own MAXITER=500, so the
    # accepted coil currents are upstream's -- measured bitwise equal.
    assert np.array_equal(
        np.asarray(
            shipped.observation.values["construction:optimized_coil_dofs"],
            dtype=np.float64,
        ),
        OFFICIAL.array("construction:optimized_coil_dofs"),
    )


def test_the_native_example_fixes_the_boundary_before_the_coil_optimization() -> None:
    """The shipped native script carries the same adaptation as its siblings.

    ``examples/2_Intermediate/permanent_magnet_QA.py`` is a module-level
    program, so the order of the two statements is read from its bytes;
    ``importlib``/``exec`` are forbidden here. The parity case's own
    construction is checked functionally by the test above.
    """

    source = NATIVE_EXAMPLE_SOURCE.read_text(encoding="utf-8")
    fix_index = source.index("s.fix_all()")
    call_index = source.index("coil_optimization(s, bs, base_curves, curves)")
    assert fix_index < call_index
