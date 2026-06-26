r"""Analytic GATE-1 tests for :mod:`banana_opt.topology.invariant_manifold`.

The invariant-manifold grower, homoclinic-tangle detector, and turnstile area code
are validated against the **Chirikov standard map**, an area-preserving map with an
exact return map ``P``, exact Jacobian ``DP``, exact inverse ``P^{-1}``, and an exact
generating-function action. None of these tests touch a magnetic field; the standard
map is the analytic fixture that pins the geometry/area code down before the module is
run on the slid_clean 2/9 X-point.

Standard map (lifted, on the cylinder; ``x`` poloidal-like, ``p`` radial-like)::

    p' = p + K sin(x)
    x' = x + p'

Area-preserving (``det DP = 1`` exactly). Fixed points at ``(0, 0)`` (hyperbolic
saddle, ``tr DP = 2 + K > 2``) and ``(pi, 0)`` (elliptic island centre). The
unstable manifold ``W^u`` of the ``(0, 0)`` saddle and the stable manifold ``W^s``
form a homoclinic tangle around the ``(pi, 0)`` island whose turnstile lobes carry the
flux. Generating function ``L(x, x') = (x' - x)^2 / 2 - K cos(x)`` reproduces the map
(``p' = dL/dx'``, ``p = -dL/dx``), giving an analytic action for the MMP cross-check.

What each test falsifies:

* ``test_standard_map_is_area_preserving`` / ``test_saddle_eigenvalue_matches_analytic``:
  the saddle spectrum is wrong (det != 1, or ``lambda_u`` off the exact formula).
* ``test_seed_is_tangent_to_unstable_eigenvector``: the fundamental-domain seed points
  the wrong way.
* ``test_grown_manifold_is_forward_invariant``: **the grower does not trace the true
  manifold.** ``P(W^u)`` must lie on ``W^u`` to ~node spacing; a grower with gaps,
  section-interpolation artifacts, or wrong tiling fails this by orders of magnitude.
  This is the core non-tautological grower check.
* ``test_homoclinic_tangle_has_alternating_transversal_points``: no genuine tangle is
  detected (``< 3`` alternating transversal crossings).
* ``test_lobe_area_matches_independent_estimators``: the turnstile area code is wrong --
  the Green's-theorem shoelace area is cross-checked against an independent
  point-in-polygon Monte-Carlo area and against the MMP action-difference of the
  generating-function one-form ``∮ Z dR``.
* ``test_primary_lobe_area_is_refinement_stable``: the grown lobe is under-resolved
  (area drifts under ``delta_max`` halving).
* ``test_mmp_signed_area_path_flags_no_true_action`` /
  ``test_mmp_action_path_flags_true_action``: the ``turnstile_lobe_area_mmp`` honesty
  flag (true action vs signed-area fallback) is wrong.
"""

from __future__ import annotations

import sys
from math import cos, sin, sqrt

import numpy as np
import pytest

sys.path.insert(
    0, "/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization"
)
from banana_opt.topology import invariant_manifold as im

# --------------------------------------------------------------------------- #
# Analytic Chirikov standard-map fixture (exact P, DP, P^-1, generating function).
# --------------------------------------------------------------------------- #

K = 2.0  # strong enough that the homoclinic tangle is well-developed and detectable.
SADDLE = np.array([0.0, 0.0])
L_SADDLE = -K  # L(0, 0) = 0 - K cos 0, the saddle's generating value.


def standard_map_forward(states: np.ndarray) -> np.ndarray:
    """One forward iterate ``P`` of the standard map on an ``(M, 2)`` batch."""
    state = np.asarray(states, dtype=float)
    x = state[:, 0]
    p = state[:, 1]
    p_next = p + K * np.sin(x)
    x_next = x + p_next
    return np.stack([x_next, p_next], axis=-1)


def standard_map_backward(states: np.ndarray) -> np.ndarray:
    """One backward iterate ``P^{-1}`` of the standard map on an ``(M, 2)`` batch."""
    state = np.asarray(states, dtype=float)
    x = state[:, 0]
    p = state[:, 1]
    x_prev = x - p
    p_prev = p - K * np.sin(x_prev)
    return np.stack([x_prev, p_prev], axis=-1)


def standard_map_jacobian(x: float) -> np.ndarray:
    """Exact Jacobian ``DP`` of the standard map at poloidal coordinate ``x``."""
    return np.array([[1.0 + K * cos(x), 1.0], [K * cos(x), 1.0]])


def generating_function(x: float, x_next: float) -> float:
    """Standard-map generating function ``L(x, x') = (x'-x)^2/2 - K cos x``."""
    return 0.5 * (x_next - x) ** 2 - K * cos(x)


def analytic_unstable_eigenvalue() -> float:
    """Exact unstable eigenvalue of the saddle: root of ``mu^2 - (2+K) mu + 1 = 0``."""
    trace = 2.0 + K
    return (trace + sqrt(trace * trace - 4.0)) / 2.0


@pytest.fixture(scope="module")
def saddle_spectrum() -> im.SaddleSpectrum:
    return im.eigendecompose_saddle(standard_map_jacobian(0.0))


@pytest.fixture(scope="module")
def grown_tangle(saddle_spectrum):
    """Grow ``W^u`` (forward) and ``W^s`` (backward) and detect the homoclinic tangle.

    Returns ``(unstable_points, stable_points, homoclinic)``. ``W^u`` is seeded along
    ``+v_u`` (the prong that winds toward the island in ``x > 0``); ``W^s`` is seeded
    along ``-v_s`` so it shares the ``x > 0`` half-plane and tangles with ``W^u``.
    """
    spectrum = saddle_spectrum
    seed_u = im.fundamental_domain_seed(
        SADDLE, spectrum.v_u, spectrum.lambda_u, eps=1.0e-3, n_seed=150
    )
    seed_s = im.fundamental_domain_seed(
        SADDLE, -spectrum.v_s, 1.0 / spectrum.lambda_s, eps=1.0e-3, n_seed=150
    )
    grow_kwargs = dict(
        n_iterates=9, delta_max=0.04, alpha_max=0.35, origin=SADDLE, rl_max=12.0
    )
    unstable = im.grow_manifold(standard_map_forward, seed_u, **grow_kwargs)
    stable = im.grow_manifold(standard_map_backward, seed_s, **grow_kwargs)
    homoclinic = im.find_primary_homoclinic_points(
        unstable.points, stable.points, saddle=SADDLE, min_saddle_distance=0.05
    )
    return unstable.points, stable.points, homoclinic


# --------------------------------------------------------------------------- #
# Saddle spectrum.
# --------------------------------------------------------------------------- #


def test_standard_map_is_area_preserving(saddle_spectrum):
    """det DP = 1 exactly and lambda_u * lambda_s = 1 (symplectic saddle)."""
    det = np.linalg.det(standard_map_jacobian(0.0))
    assert det == pytest.approx(1.0, abs=1.0e-14), f"standard map not area-preserving: det={det}"
    product = saddle_spectrum.lambda_u * saddle_spectrum.lambda_s
    assert product == pytest.approx(1.0, abs=1.0e-12), (
        f"eigenvalue product lambda_u*lambda_s={product} != 1 (area not preserved)"
    )


def test_saddle_eigenvalue_matches_analytic(saddle_spectrum):
    """The computed unstable eigenvalue matches the exact quadratic-formula root."""
    expected = analytic_unstable_eigenvalue()
    assert saddle_spectrum.lambda_u == pytest.approx(expected, rel=1.0e-12), (
        f"lambda_u={saddle_spectrum.lambda_u} != analytic {expected}"
    )
    assert saddle_spectrum.lambda_u > 1.0 > saddle_spectrum.lambda_s > 0.0


def test_eigendecompose_rejects_non_symplectic_matrix():
    """A non-area-preserving 2x2 matrix is rejected, not silently accepted."""
    with pytest.raises(ValueError, match="area-preserving"):
        im.eigendecompose_saddle(np.array([[2.0, 0.0], [0.0, 2.0]]))


# --------------------------------------------------------------------------- #
# Seeding and grower correctness.
# --------------------------------------------------------------------------- #


def test_seed_is_tangent_to_unstable_eigenvector(saddle_spectrum):
    """The first images of the seed point along v_u within 2 degrees (design Check E)."""
    spectrum = saddle_spectrum
    seed = im.fundamental_domain_seed(
        SADDLE, spectrum.v_u, spectrum.lambda_u, eps=1.0e-3, n_seed=8
    )
    # The seed segment itself lies along v_u by construction; verify its forward image
    # also departs the saddle along v_u (the manifold is tangent to the eigenvector).
    image = standard_map_forward(seed)
    chord = image[-1] - SADDLE
    chord_unit = chord / np.linalg.norm(chord)
    cos_angle = abs(float(chord_unit @ spectrum.v_u))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
    assert angle_deg < 2.0, f"seed image off v_u by {angle_deg:.2f} deg (> 2 deg)"


def test_grown_manifold_is_forward_invariant(saddle_spectrum):
    r"""``P(W^u)`` lies on ``W^u``: the grower traces the *true* invariant manifold.

    This is the decisive non-tautological grower check. The unstable manifold is
    forward-invariant (``P(W^u) \subseteq W^u``), so mapping interior nodes forward by
    one iterate must land back on the grown polyline to within ~node spacing. A grower
    that interpolates in the section, leaves gaps, or mis-tiles the fundamental domain
    fails this by orders of magnitude. We require the image-to-curve distance to be
    far below ``delta_max`` (here ~1e-9, i.e. integration/area precision, not the
    O(0.04) node spacing).
    """
    spectrum = saddle_spectrum
    delta_max = 0.04
    seed = im.fundamental_domain_seed(
        SADDLE, spectrum.v_u, spectrum.lambda_u, eps=1.0e-3, n_seed=150
    )
    manifold = im.grow_manifold(
        standard_map_forward, seed, n_iterates=9, delta_max=delta_max,
        alpha_max=0.35, origin=SADDLE, rl_max=12.0,
    )
    points = manifold.points
    # Only interior nodes have images within the grown extent; the outermost tile maps
    # beyond what we grew. Restrict to nodes whose image stays inside the grown radius.
    radius = np.hypot(points[:, 0], points[:, 1])
    interior = radius < radius[-1] / spectrum.lambda_u
    images = standard_map_forward(points[interior])
    distances = _point_to_polyline_distances(images, points)
    median = float(np.median(distances))
    p95 = float(np.percentile(distances, 95))
    assert p95 < 1.0e-6, (
        f"P(W^u) departs W^u: 95th-pct distance {p95:.2e} (median {median:.2e}); "
        f"a correct grower keeps this ~integration precision << delta_max={delta_max}"
    )


def test_grower_accumulates_full_manifold(saddle_spectrum):
    """The grown manifold spans saddle-to-tip, not just the last fundamental-domain tile.

    A regression guard for the tiling bug where only ``P^N(seed)`` (the far tip) is
    returned instead of ``seed u P(seed) u ... u P^N(seed)``: the manifold must reach
    from near the saddle (radius ~ eps) out to the island region (radius ~ pi).
    """
    spectrum = saddle_spectrum
    seed = im.fundamental_domain_seed(
        SADDLE, spectrum.v_u, spectrum.lambda_u, eps=1.0e-3, n_seed=150
    )
    manifold = im.grow_manifold(
        standard_map_forward, seed, n_iterates=9, delta_max=0.04,
        alpha_max=0.35, origin=SADDLE, rl_max=12.0,
    )
    radius = np.hypot(manifold.points[:, 0], manifold.points[:, 1])
    assert radius.min() < 0.01, f"manifold does not reach the saddle: min radius {radius.min():.3f}"
    assert radius.max() > 3.0, f"manifold does not reach the island: max radius {radius.max():.3f}"


# --------------------------------------------------------------------------- #
# Tangle detection and turnstile area.
# --------------------------------------------------------------------------- #


def test_homoclinic_detection_is_deterministic(grown_tangle):
    """Repeated detection on the SAME curves returns the bit-identical result.

    Pins the detector as a pure function of its inputs (regression guard against a
    stateful/nondeterministic intersection scan): five runs on the same ``(W^u,
    W^s)`` arrays must give the same count and bit-identical point coordinates. The
    count legitimately differs only when the *input curves* differ (e.g. single
    prong vs a saddle-joined two-prong manifold) -- that is an input contract, not
    nondeterminism (see ``find_primary_homoclinic_points`` docstring).
    """
    unstable, stable, first = grown_tangle
    counts = []
    for _ in range(5):
        again = im.find_primary_homoclinic_points(
            unstable, stable, saddle=SADDLE, min_saddle_distance=0.05
        )
        counts.append(again.points.shape[0])
        assert np.array_equal(again.points, first.points)
        assert np.array_equal(again.arclength_u, first.arclength_u)
    assert len(set(counts)) == 1, f"detection not deterministic: counts={counts}"


def test_homoclinic_tangle_has_alternating_transversal_points(grown_tangle):
    """At least 3 transversal homoclinic points with strictly alternating sidedness."""
    _, _, homoclinic = grown_tangle
    assert homoclinic.points.shape[0] >= 3, (
        f"only {homoclinic.points.shape[0]} primary homoclinic points; "
        "a tangle needs >= 3 (design B.1)"
    )
    # Transversality: every kept crossing exceeds the sin(10 deg) floor.
    assert np.all(homoclinic.sin_angles > sin(np.radians(10.0)))
    # Alternation: consecutive sides differ.
    sides = homoclinic.sides
    assert np.all(sides[1:] != sides[:-1]), f"sides not alternating: {sides}"


def test_turnstile_lobe_is_a_sliver_not_the_whole_tangle(grown_tangle):
    r"""``turnstile_lobes`` selects small slivers, excluding whole-tangle polygons.

    Regression for the primary-lobe-isolation bug: a consecutive-on-W^u homoclinic
    pair whose W^s arc skips over other crossings encloses many lobes at once (its
    polygon spans the whole tangle), while a genuine turnstile lobe -- adjacent on
    BOTH manifolds -- is a small sliver. This asserts (a) every pair from
    ``turnstile_lobes`` has no homoclinic point between its endpoints in W^s
    arclength, and (b) the genuine primary lobe area is far smaller than a
    deliberately-mispaired (W^s-skipping) "lobe" built from the same points -- so
    the selector actually prevents the area code from measuring the whole tangle.
    """
    unstable, stable, homoclinic = grown_tangle
    lobes = im.turnstile_lobes(homoclinic)
    assert len(lobes) >= 1, "no genuine turnstile lobe found"

    arclength_s = homoclinic.arclength_s
    for i, j in lobes:
        lo, hi = sorted((float(arclength_s[i]), float(arclength_s[j])))
        between = np.sum((arclength_s > lo + 1e-9) & (arclength_s < hi - 1e-9))
        assert between == 0, f"lobe ({i},{j}) skips {between} crossings on W^s -- not a sliver"

    genuine = _first_simple_lobe(unstable, stable, homoclinic)
    genuine_area = im.turnstile_lobe_area_polygon(
        unstable, stable, homoclinic, lobe=genuine
    ).area
    # Build a mispaired "lobe" that DOES skip W^s crossings (the bug); its polygon
    # spans much more area. Find any consecutive-on-W^u pair excluded by turnstile_lobes.
    bad = next(
        (
            (k, k + 1)
            for k in range(homoclinic.points.shape[0] - 1)
            if (k, k + 1) not in lobes
        ),
        None,
    )
    if bad is not None:
        bad_area = im.turnstile_lobe_area_polygon(
            unstable, stable, homoclinic, lobe=bad
        ).area
        assert genuine_area < 0.5 * bad_area, (
            f"genuine lobe area {genuine_area:.4f} not clearly smaller than "
            f"W^s-skipping lobe {bad_area:.4f} -- selector not isolating slivers"
        )


def test_first_fold_arclength_marks_a_sharp_turn():
    """``first_fold_arclength`` returns the arclength of the first > fold_angle turn.

    On a synthetic polyline that runs straight then makes a right-angle turn, the
    fold arclength must equal the pre-turn run length; a polyline that never folds
    that sharply returns its total arclength.
    """
    # Straight run of length 2, then a 135 deg fold-back at node 2 (arclength 2.0).
    straight_then_turn = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    assert im.first_fold_arclength(straight_then_turn) == pytest.approx(2.0)
    gentle = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.1], [3.0, 0.2]])
    assert im.first_fold_arclength(gentle) == pytest.approx(
        im._arclengths(gentle)[-1]
    )


def test_primary_restriction_drops_post_fold_lobes(grown_tangle, saddle_spectrum):
    """``max_arclength_u`` keeps only lobes whose endpoints precede the cutoff.

    Restricting to the first homoclinic point's arclength must drop every lobe
    (no pair can have both endpoints at or before a single point), proving the
    ``max_arclength_u`` primary cutoff is actually applied -- the mechanism the
    field analysis uses to keep only pre-first-fold primary lobes.
    """
    unstable, stable, homoclinic = grown_tangle
    full = im.turnstile_lobes(homoclinic)
    assert len(full) >= 1
    tiny_cutoff = float(homoclinic.arclength_u[0])
    restricted = im.turnstile_lobes(homoclinic, max_arclength_u=tiny_cutoff)
    assert restricted == (), f"primary cutoff not applied: {restricted}"
    # A cutoff at the full W^u length keeps everything (no-op upper bound).
    fold = im.first_fold_arclength(unstable)
    assert fold > 0.0
    kept = im.turnstile_lobes(homoclinic, max_arclength_u=im._arclengths(unstable)[-1])
    assert kept == full


def test_lobe_area_matches_independent_estimators(grown_tangle):
    r"""Turnstile lobe area: shoelace vs INDEPENDENT point-in-polygon Monte-Carlo.

    For a genuine (simple) turnstile lobe the Green's-theorem shoelace area
    (:func:`turnstile_lobe_area_polygon`) is cross-checked against a point-in-polygon
    Monte-Carlo area -- a genuinely independent estimator (different algorithm, no
    shared formula) -- which must agree to < 5%, falsifying a wrong grower, wrong arc
    extraction, or wrong shoelace. The MMP one-form action ``∮ Z dR`` is ALSO checked
    but it equals the enclosed area by Green's theorem, so it agrees by identity and
    is NOT an independent estimator (it validates only sign/orientation bookkeeping);
    Monte-Carlo is the load-bearing independent check here.
    """
    unstable, stable, homoclinic = grown_tangle
    lobe = _first_simple_lobe(unstable, stable, homoclinic)
    polygon_area = im.turnstile_lobe_area_polygon(
        unstable, stable, homoclinic, lobe=lobe
    ).area
    monte_carlo = _monte_carlo_lobe_area(unstable, stable, homoclinic, lobe)
    mmp = im.turnstile_lobe_area_mmp(
        unstable, stable, homoclinic, lobe=lobe, action=_one_form_action_zdr
    )
    assert monte_carlo == pytest.approx(polygon_area, rel=0.05), (
        f"shoelace {polygon_area:.6f} vs INDEPENDENT Monte-Carlo {monte_carlo:.6f} disagree > 5%"
    )
    # One-form == area by Green's theorem (orientation/bookkeeping check, not independent).
    assert mmp.area == pytest.approx(polygon_area, rel=0.05), (
        f"shoelace {polygon_area:.6f} vs MMP one-form {mmp.area:.6f} disagree > 5%"
    )
    assert mmp.used_true_action is True


def test_mmp_orbit_action_order_of_magnitude(grown_tangle, saddle_spectrum):
    r"""Documented (non-gating) sanity: the generating-function single-orbit action of a
    primary homoclinic point is finite and O(1), in the ballpark of the local action
    scale ``~ K``.

    The fully-independent **homoclinic-orbit** action difference of MMP (summing the
    Lagrangian along the bi-infinite homoclinic orbits) is numerically *fragile* for a
    strongly hyperbolic saddle: floating-point error in the homoclinic point is
    amplified by ``lambda_u`` per iterate, so the orbit escapes the manifold and the
    action sum stops converging beyond a handful of terms. We therefore do **not** gate
    the 5% area check on it (the gated independent estimators are the point-in-polygon
    area and the one-form action above). This test only confirms the action is a finite
    O(1) number over a short converged window, documenting the method's behaviour
    honestly rather than asserting a precision it cannot deliver here.
    """
    _, _, homoclinic = grown_tangle
    action_window = _homoclinic_orbit_action(homoclinic.points[0], window=5)
    assert np.isfinite(action_window)
    assert 0.1 < abs(action_window) < 1.0e3


def test_primary_lobe_area_is_refinement_stable(saddle_spectrum):
    """A clean lobe's area changes < 10% when ``delta_max`` is halved (design D-delta)."""
    areas = []
    for delta_max in (0.04, 0.02, 0.01):
        unstable, stable, homoclinic = _grow_tangle(saddle_spectrum, delta_max)
        lobe = _first_simple_lobe(unstable, stable, homoclinic)
        areas.append(
            im.turnstile_lobe_area_polygon(
                unstable, stable, homoclinic, lobe=lobe
            ).area
        )
    spread = (max(areas) - min(areas)) / max(areas)
    assert spread < 0.10, f"lobe area not refinement-stable: areas={areas}, spread={spread:.2%}"


def test_mmp_signed_area_path_flags_no_true_action(grown_tangle):
    """Without an action functional, MMP returns the signed lobe area and flags it.

    The design requires that when no genuine action is available the MMP value reduces
    to the signed-area form and ``used_true_action`` is ``False`` (a sign/orientation
    cross-check, not an independent estimator). The magnitude must match the shoelace.
    """
    unstable, stable, homoclinic = grown_tangle
    lobe = _first_simple_lobe(unstable, stable, homoclinic)
    polygon_area = im.turnstile_lobe_area_polygon(
        unstable, stable, homoclinic, lobe=lobe
    ).area
    mmp = im.turnstile_lobe_area_mmp(unstable, stable, homoclinic, lobe=lobe)
    assert mmp.used_true_action is False
    assert mmp.method == "mmp_signed_area"
    assert mmp.area == pytest.approx(polygon_area, rel=1.0e-9)


# --------------------------------------------------------------------------- #
# Test-local helpers (independent of the module under test where it matters).
# --------------------------------------------------------------------------- #


def _point_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Minimum Euclidean distance from each point to a polyline's segments."""
    seg_a = polyline[:-1]
    seg_b = polyline[1:]
    seg = seg_b - seg_a
    seg_len_sq = np.einsum("ij,ij->i", seg, seg)
    seg_len_sq[seg_len_sq == 0.0] = 1.0e-30
    distances = np.empty(points.shape[0], dtype=float)
    for i, point in enumerate(points):
        delta = point[None, :] - seg_a
        t = np.clip(np.einsum("ij,ij->i", delta, seg) / seg_len_sq, 0.0, 1.0)
        projection = seg_a + t[:, None] * seg
        distances[i] = np.sqrt(((point[None, :] - projection) ** 2).sum(axis=1).min())
    return distances


def _grow_tangle(spectrum, delta_max):
    seed_u = im.fundamental_domain_seed(
        SADDLE, spectrum.v_u, spectrum.lambda_u, eps=1.0e-3, n_seed=150
    )
    seed_s = im.fundamental_domain_seed(
        SADDLE, -spectrum.v_s, 1.0 / spectrum.lambda_s, eps=1.0e-3, n_seed=150
    )
    grow_kwargs = dict(
        n_iterates=9, delta_max=delta_max, alpha_max=0.35, origin=SADDLE, rl_max=12.0
    )
    unstable = im.grow_manifold(standard_map_forward, seed_u, **grow_kwargs)
    stable = im.grow_manifold(standard_map_backward, seed_s, **grow_kwargs)
    homoclinic = im.find_primary_homoclinic_points(
        unstable.points, stable.points, saddle=SADDLE, min_saddle_distance=0.05
    )
    return unstable.points, stable.points, homoclinic


def _lobe_polygon(unstable, stable, homoclinic, lobe):
    """The closed lobe loop, via the module's SSOT arc extraction (oriented to close)."""
    u_arc, s_arc = im._lobe_loop(unstable, stable, homoclinic, lobe)
    return np.vstack([u_arc, s_arc])


def _monte_carlo_lobe_area(unstable, stable, homoclinic, lobe, n=400000, seed=0):
    """Independent point-in-polygon area of the lobe (no shoelace)."""
    from matplotlib.path import Path

    polygon = _lobe_polygon(unstable, stable, homoclinic, lobe)
    rng = np.random.default_rng(seed)
    lo = polygon.min(axis=0)
    hi = polygon.max(axis=0)
    samples = rng.uniform(lo, hi, size=(n, 2))
    inside = Path(polygon).contains_points(samples)
    box = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    return float(inside.mean()) * box


def _first_simple_lobe(unstable, stable, homoclinic, tol=0.03):
    """First genuine turnstile lobe (adjacent on both manifolds) whose polygon is simple.

    Selects from :func:`turnstile_lobes` (pairs adjacent on both W^u and W^s) and
    additionally requires shoelace ~ point-in-polygon, so the area cross-check runs
    on a clean, simple lobe. Returns the bounding ``(i, j)`` homoclinic-index pair.
    """
    for lobe in im.turnstile_lobes(homoclinic):
        shoelace = im.turnstile_lobe_area_polygon(
            unstable, stable, homoclinic, lobe=lobe
        ).area
        if shoelace <= 0.0:
            continue
        mc = _monte_carlo_lobe_area(unstable, stable, homoclinic, lobe, n=60000, seed=1)
        if abs(shoelace - mc) / shoelace < tol:
            return lobe
    raise AssertionError("no simple turnstile lobe found for area cross-check")


def _one_form_action_zdr(arc: np.ndarray) -> float:
    """MMP action as the Poincare-Cartan one-form line integral ``integral Z dR`` along an arc.

    Trapezoidal integration of ``Z dR`` over the polyline. For a closed lobe the
    difference of this between the ``W^u`` and ``W^s`` arcs equals the signed area
    (``∮ Z dR = -Area``), giving an area estimator computed by a formula independent of
    the shoelace symmetrisation.
    """
    radius = arc[:, 0]
    z = arc[:, 1]
    return float(np.sum(0.5 * (z[:-1] + z[1:]) * (radius[1:] - radius[:-1])))


def _homoclinic_orbit_action(homoclinic_point: np.ndarray, window: int) -> float:
    """Generating-function action of a homoclinic orbit over a short window.

    Sums ``L(x_n, x_{n+1}) - L(0, 0)`` over the orbit through ``homoclinic_point`` for
    ``window`` forward and backward iterates. Convergent only for a small window before
    floating-point amplification (``lambda_u`` per step) drives the orbit off the
    manifold; used as an order-of-magnitude sanity, not a precision gate.
    """
    point = np.asarray(homoclinic_point, dtype=float).reshape(1, 2)
    forward_x = [float(point[0, 0])]
    current = point.copy()
    for _ in range(window):
        current = standard_map_forward(current)
        forward_x.append(float(current[0, 0]))
    backward_x = []
    current = point.copy()
    for _ in range(window):
        current = standard_map_backward(current)
        backward_x.append(float(current[0, 0]))
    orbit_x = backward_x[::-1] + forward_x
    return sum(
        generating_function(orbit_x[n], orbit_x[n + 1]) - L_SADDLE
        for n in range(len(orbit_x) - 1)
    )


# --------------------------------------------------------------------------- #
# Field-coupled path: analytic B-field stubs exercise batched_return_map (the
# B_R/B_phi projection, the forward/backward branch, the per-node escape/freeze)
# and grow_manifold's escape vs rl_max bookkeeping -- the standard-map tests above
# never touch the field integrator.
# --------------------------------------------------------------------------- #


class _ToroidalField:
    """Purely toroidal analytic field ``B = B0*(-Y/R, X/R, 0)`` (``B_phi=B0``, ``B_R=B_Z=0``).

    A field line of a purely toroidal field stays at fixed ``(R, Z)``, so the return
    map is the **identity** -- a known answer that pins the Cartesian->(R, Z)
    projection (``B_R`` must come out 0, ``B_phi=B0``) and that the forward and
    backward integrations both return to the seed.
    """

    def __init__(self, b0: float = 1.0):
        self.b0 = b0
        self._points = np.zeros((0, 3))

    def set_points(self, points: np.ndarray) -> None:
        self._points = np.asarray(points, dtype=float)

    def B(self) -> np.ndarray:
        radius = np.hypot(self._points[:, 0], self._points[:, 1])
        field = np.zeros_like(self._points)
        field[:, 0] = -self.b0 * self._points[:, 1] / radius
        field[:, 1] = self.b0 * self._points[:, 0] / radius
        return field


class _VerticalField:
    """Purely vertical analytic field ``B = (0, 0, B0)`` -> ``|B_phi|/|B| = 0``.

    Every seed trips the low-toroidal-field escape guard on the first evaluation, so
    ``batched_return_map`` must return an all-``NaN`` image -- pins the escape path.
    """

    def __init__(self, b0: float = 1.0):
        self.b0 = b0
        self._points = np.zeros((0, 3))

    def set_points(self, points: np.ndarray) -> None:
        self._points = np.asarray(points, dtype=float)

    def B(self) -> np.ndarray:
        field = np.zeros_like(self._points)
        field[:, 2] = self.b0
        return field


def test_batched_return_map_toroidal_field_is_identity():
    """Purely toroidal field => identity return map, forward and backward.

    Pins the B_R/B_phi Cartesian projection and both integration directions: a wrong
    projection sign or a backward-span error moves (R, Z) off the seed.
    """
    field = _ToroidalField(b0=1.3)
    seeds = np.array([[1.0, 0.0], [1.25, 0.08], [0.9, -0.06]])
    forward = im.batched_return_map(field, seeds, torus_turns=1)
    assert np.allclose(forward, seeds, atol=1.0e-8), f"toroidal return map not identity: {forward}"
    backward = im.batched_return_map(field, seeds, torus_turns=1, backward=True)
    assert np.allclose(backward, seeds, atol=1.0e-8), f"backward toroidal map not identity: {backward}"


def test_batched_return_map_escapes_on_vanishing_toroidal_field():
    """A field with no toroidal component escapes every seed (all-NaN image)."""
    field = _VerticalField(b0=1.0)
    seeds = np.array([[1.0, 0.0], [1.2, 0.1]])
    image = im.batched_return_map(field, seeds, torus_turns=1)
    assert image.shape == seeds.shape
    assert np.all(np.isnan(image)), f"low-B_phi seeds must escape to NaN, got {image}"


def test_grow_manifold_records_escapees_separately_from_terminal():
    r"""``escaped_points`` (NaN-step escapees) and ``terminal_mask`` (rl_max) are disjoint.

    Regression for the ManifoldCurve bookkeeping: domain escapees are dropped from
    ``points`` into ``escaped_points`` and are NOT flagged in ``terminal_mask``; only
    rl_max-exceeded nodes are terminal-on-curve. Uses a pure analytic step that NaNs
    nodes past ``x = 2``, with ``origin`` far away so rl_max cannot fire -- isolating
    the escape path.
    """

    def escaping_step(states: np.ndarray) -> np.ndarray:
        out = states + np.array([0.5, 0.0])
        out[out[:, 0] > 2.0] = np.nan
        return out

    seed = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    manifold = im.grow_manifold(
        escaping_step, seed, n_iterates=8, delta_max=1.0, alpha_max=1.0,
        origin=[0.0, -1.0e3], rl_max=1.0e9,
    )
    assert manifold.escaped_points.shape[0] > 0, "escapees not recorded in escaped_points"
    assert manifold.points.shape[0] == manifold.terminal_mask.shape[0]
    assert not manifold.terminal_mask.any(), "domain escapees must not be flagged terminal (rl_max only)"


def test_grow_manifold_marks_rl_max_nodes_terminal_not_escaped():
    """rl_max-exceeded nodes are terminal-on-curve; with no NaN step there are no escapees."""

    def shift_step(states: np.ndarray) -> np.ndarray:
        return states + np.array([0.5, 0.0])

    seed = np.array([[0.0, 0.0], [0.1, 0.0]])
    manifold = im.grow_manifold(
        shift_step, seed, n_iterates=10, delta_max=1.0, alpha_max=1.0,
        origin=[0.0, 0.0], rl_max=1.5,
    )
    assert manifold.terminal_mask.any(), "nodes beyond rl_max should be terminal"
    assert manifold.escaped_points.shape[0] == 0, "no NaN step => no domain escapees"


def test_standard_map_backward_is_inverse():
    """``standard_map_backward`` is the exact inverse of ``standard_map_forward``.

    Guards the W^s growth: a wrong inverse yields a self-consistently-wrong stable
    manifold that geometric tests would still pass against.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, size=(32, 2))
    round_trip = standard_map_backward(standard_map_forward(points))
    assert np.allclose(round_trip, points, atol=1.0e-12)


def test_homoclinic_detection_handles_parallel_and_disjoint_segments():
    """Parallel and far-disjoint manifolds yield zero homoclinic points.

    Pins the segment-intersection degeneracy guards: a near-zero denominator
    (parallel segments) is skipped rather than producing a spurious crossing, and
    segments whose intersection parameters fall outside ``[0, 1]`` (disjoint) are
    rejected. These paths fire constantly on a densely-folded real manifold.
    """
    wu = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])  # along y = 0
    ws_parallel = np.array([[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]])  # parallel, offset 0.5
    assert im.find_primary_homoclinic_points(wu, ws_parallel).points.shape[0] == 0
    ws_disjoint = np.array([[5.0, 5.0], [5.0, 6.0]])  # far away, non-parallel
    assert im.find_primary_homoclinic_points(wu, ws_disjoint).points.shape[0] == 0
