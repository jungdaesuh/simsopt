r"""Invariant-manifold growth, homoclinic-tangle detection, and turnstile flux.

This module builds the **unstable / stable manifolds** :math:`W^u, W^s` of a
hyperbolic saddle of an area-preserving section map, detects the **primary
homoclinic points** where they cross, and measures the **turnstile lobe area**
(the phase-space flux carried across a broken separatrix per map iterate). It is
written for the field-line return map of a stellarator section (the 2/9 X-point
of ``slid_clean``), but every routine is **callable-driven**: the grower and the
turnstile detector consume a generic ``step`` callable ``(M, 2) -> (M, 2)`` and
know nothing about magnetic fields, so they are unit-tested against an analytic
area-preserving map (the Chirikov standard map) with an exact generating
function before they ever touch a field.

Numerical single-source-of-truth: one batched field tracer
(:func:`batched_return_map`) is the *only* place a field line is integrated;
one polyline grower (:func:`grow_manifold`) is the *only* place a manifold is
resampled. All result containers are immutable.

Algorithm references (in the experiment design
``GPD/analysis/EXPERIMENT-DESIGN-invariant-manifold-turnstile.md``):

* You-Kostelich-Yorke / Hobson fundamental-domain seeding and pre-image
  refinement (Sections A.2, A.3): refine an exponentially-stretched manifold by
  subdividing **pre-images** and mapping them forward, never by interpolating in
  the section.
* MacKay-Meiss-Percival (MMP) turnstile lobe area (Section B.3): the flux is the
  signed area between the :math:`W^u` and :math:`W^s` arcs joining two
  consecutive primary homoclinic points.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import hypot, isfinite, pi

import numpy as np

__all__ = [
    "StepFunction",
    "SaddleSpectrum",
    "ManifoldCurve",
    "HomoclinicPoints",
    "LobeArea",
    "batched_return_map",
    "eigendecompose_saddle",
    "fundamental_domain_seed",
    "grow_manifold",
    "find_primary_homoclinic_points",
    "turnstile_lobes",
    "first_fold_arclength",
    "turnstile_lobe_area_polygon",
    "turnstile_lobe_area_mmp",
]


# A section map applied to a batch of points: takes an ``(M, 2)`` array of
# ``(R, Z)`` states and returns the ``(M, 2)`` array of their images under one
# iterate of the (forward or backward) map. Rows whose image is invalid (the
# field line escaped the valid domain) are returned as ``NaN`` so the grower can
# drop them; the callable itself never raises on a single bad row.
StepFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class SaddleSpectrum:
    """Eigendecomposition of a hyperbolic saddle's monodromy ``M`` (2x2).

    ``lambda_u > 1 > lambda_s > 0`` are the unstable/stable eigenvalues;
    ``v_u``/``v_s`` are the corresponding **unit** eigenvectors in section
    ``(R, Z)`` coordinates. For an area-preserving map ``lambda_u*lambda_s = 1``.
    """

    lambda_u: float
    lambda_s: float
    v_u: np.ndarray
    v_s: np.ndarray


@dataclass(frozen=True, slots=True)
class ManifoldCurve:
    """One grown invariant manifold as an ordered polyline.

    ``points`` is an ``(n, 2)`` array of ``(R, Z)`` section nodes ordered by
    arclength from the saddle outward (so ``points[0]`` is the seed end nearest
    the saddle). ``terminal_mask[i]`` is ``True`` where node ``i`` stopped being
    iterated because its radial label from the growth ``origin`` exceeded
    ``rl_max`` -- such nodes stay on the curve (in ``points``) but are not
    propagated to the next iterate. Domain escapees are tracked **separately**: a
    node whose field line leaves the valid domain (low toroidal field
    ``|B_phi|/|B|``, non-positive ``R``, or non-finite field) is dropped from
    ``points`` and collected, in iterate order, in ``escaped_points`` -- these loci
    are a primary physical output (the observed tail). So ``terminal_mask`` flags
    rl_max truncation only, ``escaped_points`` holds the domain escapees, and the
    two are disjoint.
    """

    points: np.ndarray
    terminal_mask: np.ndarray
    escaped_points: np.ndarray
    n_iterates: int


@dataclass(frozen=True, slots=True)
class HomoclinicPoints:
    """Transversal primary homoclinic points where ``W^u`` crosses ``W^s``.

    ``points`` is an ``(k, 2)`` array of section intersection coordinates ordered
    by arclength along ``W^u`` from the saddle. ``sin_angles[i]`` is the sine of
    the crossing angle (``|t_u x t_s|`` of the unit tangents) at ``points[i]``;
    ``sides[i] in {+1, -1}`` is the signed side of ``W^u`` relative to ``W^s``.
    Only transversal (``sin_angle > min_sin``) crossings with **alternating**
    sides survive; ``>= 3`` are required to claim a tangle.
    ``arclength_u``/``arclength_s`` give the arclength of each crossing measured
    along ``W^u`` and ``W^s`` respectively (used to slice turnstile lobe arcs).
    """

    points: np.ndarray
    sin_angles: np.ndarray
    sides: np.ndarray
    arclength_u: np.ndarray
    arclength_s: np.ndarray


@dataclass(frozen=True, slots=True)
class LobeArea:
    """A turnstile lobe area with the method that produced it.

    ``area`` is the lobe area (m^2 of section). ``method`` is ``"polygon"`` for
    the Green's-theorem shoelace area or ``"mmp_action"`` / ``"mmp_signed_area"``
    for the MacKay-Meiss-Percival action difference. ``used_true_action`` records
    honestly whether a genuine action functional was available (``True``) or the
    MMP value reduced to the signed-area form (``False``) -- the cross-check
    between polygon and MMP is only two *independent* estimators when this is
    ``True``.
    """

    area: float
    method: str
    used_true_action: bool


# --------------------------------------------------------------------------- #
# Batched field tracer (the perf core; the only field-line integrator here).
# --------------------------------------------------------------------------- #


def batched_return_map(
    field,
    states: Sequence[Sequence[float]],
    *,
    torus_turns: int,
    backward: bool = False,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    max_step: float = 0.05,
    samples_per_full_torus: int = 256,
    min_bphi_over_b: float = 1.0e-8,
) -> np.ndarray:
    r"""Map ``N`` section points by ``torus_turns`` toroidal turns in one solve.

    Integrates the field-line ODE ``dR/dphi = R B_R/B_phi``,
    ``dZ/dphi = R B_Z/B_phi`` for all ``N`` seeds **simultaneously** in a single
    ``solve_ivp`` over a flat ``2 N``-component state sharing one adaptive ``phi``
    march: the right-hand side calls ``field.set_points`` with an ``(N, 3)``
    Cartesian array and ``field.B()`` *once* per integrator substep, so the cost is
    one batched field evaluation per substep rather than ``N`` scalar ones. The step
    is capped so the field is sampled at least ``samples_per_full_torus`` times per
    turn. This is the lever that makes manifold growth and the convergence sweep
    affordable.

    With ``backward=False`` the map is the forward return map ``P^{torus_turns}``
    (integrate ``phi: 0 -> +2*pi*torus_turns``); with ``backward=True`` it is the
    inverse map ``P^{-torus_turns}`` (integrate ``phi: 0 -> -2*pi*torus_turns``),
    used to grow the stable manifold.

    A seed whose field line leaves the valid domain (``|B_phi|/|B|`` drops to the
    Greene threshold, ``R`` goes non-positive, or the field returns non-finite
    values) is **frozen** at that ``phi`` -- its derivative is set to zero for the
    rest of the integration -- and its returned row is ``NaN``. This is genuine
    per-node escape detection: the shared solver advances the survivors while the
    escapee is marked, instead of one bad line aborting the whole batch.

    Parameters
    ----------
    field : object with ``set_points((N,3))`` and ``B() -> (N,3)``
        The magnetic field (e.g. a simsopt ``BiotSavart``).
    states : (N, 2) array-like
        Section seeds ``(R, Z)`` in metres.
    torus_turns : int
        Number of full ``2*pi`` toroidal turns (the saddle is a fixed point of
        ``P^q`` with ``torus_turns = q``).

    Returns
    -------
    (N, 2) ndarray
        Image ``(R, Z)`` of each seed; escaped seeds are ``NaN`` rows.
    """
    seeds = np.asarray(states, dtype=float)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ValueError(f"states must have shape (N, 2), got {seeds.shape}")
    turns = int(torus_turns)
    if turns <= 0:
        raise ValueError("torus_turns must be positive")
    n_points = seeds.shape[0]
    if n_points == 0:
        return seeds.copy()

    direction = -1.0 if backward else 1.0
    phi_stop = direction * 2.0 * pi * float(turns)
    # Cap the integrator step so the field is sampled at least ``samples_per_full_torus``
    # times per toroidal turn (the field-evaluation density that controls the
    # accuracy-vs-cost trade-off), but never coarser than the caller's ``max_step``.
    step_cap = min(float(max_step), 2.0 * pi / float(samples_per_full_torus))

    # ``alive[j]`` flips to False the first time seed j's field line is invalid;
    # from then on its RHS is zero so the shared adaptive step is not poisoned by
    # NaN derivatives. ``escaped`` records which seeds ever died.
    alive = np.ones(n_points, dtype=bool)
    escaped = np.zeros(n_points, dtype=bool)

    def rhs(phi: float, flat: np.ndarray) -> np.ndarray:
        rz = flat.reshape(n_points, 2)
        radius = rz[:, 0]
        z = rz[:, 1]
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cartesian = np.empty((n_points, 3), dtype=float)
        cartesian[:, 0] = radius * cos_phi
        cartesian[:, 1] = radius * sin_phi
        cartesian[:, 2] = z
        field.set_points(cartesian)
        b_cartesian = np.asarray(field.B(), dtype=float)
        b_r = b_cartesian[:, 0] * cos_phi + b_cartesian[:, 1] * sin_phi
        b_phi = -b_cartesian[:, 0] * sin_phi + b_cartesian[:, 1] * cos_phi
        b_z = b_cartesian[:, 2]
        b_norm = np.sqrt(
            b_cartesian[:, 0] ** 2 + b_cartesian[:, 1] ** 2 + b_cartesian[:, 2] ** 2
        )
        # Mark newly-invalid seeds: low toroidal field, degenerate radius, or
        # non-finite field. Freeze them (zero derivative) for the remaining span.
        with np.errstate(invalid="ignore", divide="ignore"):
            bphi_ratio = np.abs(b_phi) / b_norm
        invalid = (
            (bphi_ratio <= float(min_bphi_over_b))
            | (radius <= 0.0)
            | ~np.isfinite(b_r)
            | ~np.isfinite(b_phi)
            | ~np.isfinite(b_z)
            | ~np.isfinite(b_norm)
        )
        newly_dead = invalid & alive
        if newly_dead.any():
            alive[newly_dead] = False
            escaped[newly_dead] = True
        derivative = np.zeros((n_points, 2), dtype=float)
        live = alive & ~invalid
        if live.any():
            safe_bphi = b_phi[live]
            derivative[live, 0] = radius[live] * b_r[live] / safe_bphi
            derivative[live, 1] = radius[live] * b_z[live] / safe_bphi
        return derivative.reshape(-1)

    from scipy.integrate import solve_ivp  # local: scipy is heavy, only field path needs it

    solution = solve_ivp(
        rhs,
        (0.0, phi_stop),
        seeds.reshape(-1),
        method="DOP853",
        rtol=float(rtol),
        atol=float(atol),
        max_step=step_cap,
        t_eval=np.asarray([0.0, phi_stop]),
    )
    if not solution.success:
        raise RuntimeError(f"Batched field-line integration failed: {solution.message}")
    final = solution.y[:, -1].reshape(n_points, 2)
    final[escaped] = np.nan
    return final


# --------------------------------------------------------------------------- #
# Saddle spectrum and fundamental-domain seeding.
# --------------------------------------------------------------------------- #


def eigendecompose_saddle(
    monodromy: np.ndarray,
    *,
    det_tol: float = 1.0e-6,
    product_tol: float = 1.0e-6,
) -> SaddleSpectrum:
    r"""Eigendecompose a 2x2 hyperbolic, area-preserving monodromy ``M``.

    Asserts the map is symplectic (``|det M - 1| < det_tol``) and hyperbolic with
    real eigenvalues satisfying ``|lambda_u*lambda_s - 1| < product_tol``. Returns
    the unstable eigenvalue ``lambda_u > 1``, the stable ``lambda_s < 1``, and the
    corresponding **unit** eigenvectors. The unstable eigenvector is sign-fixed so
    its dominant component is non-negative; both prongs ``+/- v_u`` are grown by
    the caller, so this only pins a deterministic representative.
    """
    matrix = np.asarray(monodromy, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(f"monodromy must be 2x2, got {matrix.shape}")
    det = float(np.linalg.det(matrix))
    if abs(det - 1.0) > float(det_tol):
        raise ValueError(f"monodromy is not area-preserving: det M = {det}")
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    if np.any(np.abs(eigenvalues.imag) > 1.0e-9):
        raise ValueError(
            f"saddle requires real eigenvalues (hyperbolic); got {eigenvalues}"
        )
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    order = np.argsort(np.abs(eigenvalues))
    lambda_s = float(eigenvalues[order[0]])
    lambda_u = float(eigenvalues[order[-1]])
    if not (abs(lambda_u) > 1.0 > abs(lambda_s)):
        raise ValueError(
            f"saddle requires |lambda_u| > 1 > |lambda_s|; got {lambda_u}, {lambda_s}"
        )
    if abs(lambda_u * lambda_s - 1.0) > float(product_tol):
        raise ValueError(
            f"area preservation requires lambda_u*lambda_s = 1; got {lambda_u * lambda_s}"
        )
    v_u = eigenvectors[:, order[-1]]
    v_s = eigenvectors[:, order[0]]
    v_u = v_u / np.linalg.norm(v_u)
    v_s = v_s / np.linalg.norm(v_s)
    # Deterministic sign: dominant component non-negative.
    if v_u[int(np.argmax(np.abs(v_u)))] < 0.0:
        v_u = -v_u
    if v_s[int(np.argmax(np.abs(v_s)))] < 0.0:
        v_s = -v_s
    return SaddleSpectrum(lambda_u=lambda_u, lambda_s=lambda_s, v_u=v_u, v_s=v_s)


def fundamental_domain_seed(
    saddle: Sequence[float],
    direction: Sequence[float],
    lambda_factor: float,
    *,
    eps: float,
    n_seed: int = 64,
) -> np.ndarray:
    r"""Log-spaced fundamental-domain segment along an eigendirection.

    Returns ``n_seed`` section points ``x0 + s*v`` with ``s`` log-spaced over
    ``[eps, lambda_factor*eps]`` -- i.e. from the ``eps``-point out to its own
    first image under the saddle's ``q``-iterate (``s -> lambda_u*s`` to leading
    order, so ``lambda_factor = lambda_u`` for ``W^u``). Iterating this half-open
    segment under the map tiles the manifold with no gaps and no double coverage.
    Log spacing puts more nodes near the saddle where curvature is highest once
    the first images return. ``v`` is taken as given (unit length expected); pass
    ``-v_u`` to seed the opposite prong.
    """
    x0 = np.asarray(saddle, dtype=float)
    v = np.asarray(direction, dtype=float)
    if x0.shape != (2,) or v.shape != (2,):
        raise ValueError("saddle and direction must be length-2")
    if eps <= 0.0 or not isfinite(eps):
        raise ValueError("eps must be a positive finite offset")
    if lambda_factor <= 1.0:
        raise ValueError("lambda_factor must exceed 1 (unstable stretch)")
    count = int(n_seed)
    if count < 2:
        raise ValueError("n_seed must be at least 2")
    s_values = np.geomspace(eps, float(lambda_factor) * eps, count)
    return x0[None, :] + s_values[:, None] * v[None, :]


# --------------------------------------------------------------------------- #
# Adaptive ordered-polyline grower (callable-driven; no field knowledge).
# --------------------------------------------------------------------------- #


def _radial_labels(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    delta = points - origin[None, :]
    return np.hypot(delta[:, 0], delta[:, 1])


def _turning_angles(points: np.ndarray) -> np.ndarray:
    r"""Interior turning angle (rad) at each node of a polyline.

    Angle between the incoming chord ``points[i]-points[i-1]`` and the outgoing
    chord ``points[i+1]-points[i]``. Endpoints get ``0``. Degenerate (zero-length)
    chords also yield ``0`` -- a repeated node has no turn.
    """
    n = points.shape[0]
    angles = np.zeros(n, dtype=float)
    if n < 3:
        return angles
    incoming = points[1:-1] - points[:-2]
    outgoing = points[2:] - points[1:-1]
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = incoming[:, 0] * outgoing[:, 0] + incoming[:, 1] * outgoing[:, 1]
    angles[1:-1] = np.abs(np.arctan2(cross, dot))
    return angles


def _refine_segment_preimage(
    step: StepFunction,
    pre_a: np.ndarray,
    pre_b: np.ndarray,
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    delta_max: float,
    depth: int,
    max_depth: int,
    force_once: bool = False,
) -> list[np.ndarray]:
    r"""Recursively fill the image segment ``img_a -> img_b`` via pre-images.

    The midpoint of the *pre-image* segment ``pre_a -> pre_b`` is mapped forward by
    one iterate and inserted -- never a section-space interpolation, which would cut
    across real folds of an exponentially-stretched curve. Subdivision continues
    while the mapped chord exceeds ``delta_max`` (and recursion budget remains).
    ``force_once`` subdivides this segment one level even when its chord is already
    short: the polyline-level turning-angle pass uses it to add a probe node near a
    sharp corner, where the mapped midpoint reveals curvature the straight chord
    hides. Returns the interior image points to insert between ``img_a`` and
    ``img_b`` (exclusive), in order.
    """
    chord = hypot(float(img_b[0] - img_a[0]), float(img_b[1] - img_a[1]))
    if depth >= max_depth or (chord <= delta_max and not force_once):
        return []
    pre_mid = 0.5 * (pre_a + pre_b)
    img_mid = step(pre_mid[None, :])[0]
    if not np.all(np.isfinite(img_mid)):
        # Midpoint field line escaped: cannot subdivide here; leave the chord.
        return []
    # Forced subdivision applies only to this level; deeper recursion reverts to the
    # plain chord criterion.
    left = _refine_segment_preimage(
        step, pre_a, pre_mid, img_a, img_mid,
        delta_max=delta_max, depth=depth + 1, max_depth=max_depth,
    )
    right = _refine_segment_preimage(
        step, pre_mid, pre_b, img_mid, img_b,
        delta_max=delta_max, depth=depth + 1, max_depth=max_depth,
    )
    return [*left, img_mid, *right]


def grow_manifold(
    step: StepFunction,
    seed: np.ndarray,
    *,
    n_iterates: int,
    delta_max: float,
    alpha_max: float,
    origin: Sequence[float],
    rl_max: float,
    max_depth: int = 6,
) -> ManifoldCurve:
    r"""Grow an invariant manifold by iterating a seed polyline under ``step``.

    Starting from the fundamental-domain ``seed`` (an ``(n_seed, 2)`` polyline
    along the eigendirection), apply the section map ``step`` ``n_iterates`` times.
    After each iterate the polyline is **resampled** to stay resolved as it
    stretches: between consecutive mapped nodes a new node is inserted (by mapping
    the *pre-image* midpoint forward, recursively, up to ``max_depth``) whenever
    the mapped chord exceeds ``delta_max``; a separate turning-angle pass inserts
    pre-image midpoints where the polyline turns by more than ``alpha_max``; and
    nearly-straight, finely-spaced runs are decimated to bound the node count.

    A node stops being iterated (marked terminal) when its radial label from
    ``origin`` exceeds ``rl_max`` or its field line escapes the valid domain (the
    ``step`` returns ``NaN`` for it). Escaped nodes are recorded in
    ``escaped_points`` -- they are a primary physical output (the manifold's tail).

    ``step`` must map an ``(M, 2)`` array to an ``(M, 2)`` array, returning ``NaN``
    rows for points that leave the valid domain; it is the *only* coupling to the
    field, so this grower is exercised on analytic maps in the unit tests.
    """
    seed_tile = np.asarray(seed, dtype=float)
    if seed_tile.ndim != 2 or seed_tile.shape[1] != 2:
        raise ValueError(f"seed must have shape (n_seed, 2), got {seed_tile.shape}")
    iterations = int(n_iterates)
    if iterations < 1:
        raise ValueError("n_iterates must be at least 1")
    origin_point = np.asarray(origin, dtype=float)
    if origin_point.shape != (2,):
        raise ValueError("origin must be length-2")

    # The manifold is the ordered concatenation of the fundamental-domain seed and
    # its successive map images (tiles): W = seed u P(seed) u P^2(seed) u ...,
    # marching outward from the saddle. Each iterate maps the previous tile forward,
    # refines it against that tile (pre-image midpoints), and appends it. Only the
    # survivors within rl_max propagate to the next tile; nodes beyond rl_max are
    # appended as terminal but not iterated further; escaped (NaN) nodes are recorded
    # as the tail and not placed on the curve.
    tiles: list[np.ndarray] = [seed_tile]
    terminal_flags: list[np.ndarray] = [np.zeros(seed_tile.shape[0], dtype=bool)]
    escaped_accumulator: list[np.ndarray] = []
    pre_tile = seed_tile

    for _ in range(iterations):
        if pre_tile.shape[0] < 2:
            break
        mapped = step(pre_tile)
        finite_rows = np.all(np.isfinite(mapped), axis=1)
        if not finite_rows.all():
            escaped_accumulator.append(pre_tile[~finite_rows])
        survived_pre = pre_tile[finite_rows]
        mapped = mapped[finite_rows]
        if mapped.shape[0] < 2:
            break
        refined = _resample_polyline(
            step, survived_pre, mapped,
            delta_max=delta_max, alpha_max=alpha_max, max_depth=max_depth,
        )
        labels = _radial_labels(refined, origin_point)
        within = labels <= float(rl_max)
        tile_terminal = ~within
        tiles.append(refined)
        terminal_flags.append(tile_terminal)
        # Propagate only the still-active (within rl_max) sub-curve to the next tile.
        pre_tile = refined[within]

    points = np.vstack(tiles)
    terminal_mask = np.concatenate(terminal_flags)
    escaped_points = (
        np.vstack(escaped_accumulator) if escaped_accumulator else np.empty((0, 2), dtype=float)
    )
    return ManifoldCurve(
        points=points,
        terminal_mask=terminal_mask,
        escaped_points=escaped_points,
        n_iterates=iterations,
    )


def _resample_polyline(
    step: StepFunction,
    previous: np.ndarray,
    mapped: np.ndarray,
    *,
    delta_max: float,
    alpha_max: float,
    max_depth: int,
) -> np.ndarray:
    r"""Insert pre-image midpoints then decimate straight runs.

    ``previous`` and ``mapped`` are the polyline before and after one map iterate
    (same length, row-aligned). Inserts refinement nodes between consecutive
    ``mapped`` rows whenever their chord exceeds ``delta_max`` or the polyline
    turns by more than ``alpha_max`` there, by mapping the corresponding
    ``previous`` midpoint forward; then drops middle nodes of nearly-collinear,
    finely-spaced triples to keep the count bounded. Returns the refined,
    decimated ``(m, 2)`` polyline.
    """
    n = mapped.shape[0]
    turning = _turning_angles(mapped)
    out_points: list[np.ndarray] = [mapped[0]]
    for i in range(n - 1):
        chord = hypot(
            float(mapped[i + 1, 0] - mapped[i, 0]),
            float(mapped[i + 1, 1] - mapped[i, 1]),
        )
        sharp = max(turning[i], turning[i + 1]) > alpha_max
        if chord > delta_max or sharp:
            interior = _refine_segment_preimage(
                step,
                previous[i], previous[i + 1],
                mapped[i], mapped[i + 1],
                delta_max=delta_max, depth=0, max_depth=max_depth,
                force_once=sharp,
            )
            out_points.extend(interior)
        out_points.append(mapped[i + 1])
    refined = np.asarray(out_points, dtype=float)
    return _decimate_polyline(refined, delta_max=delta_max, alpha_max=alpha_max)


def _decimate_polyline(
    points: np.ndarray, *, delta_max: float, alpha_max: float
) -> np.ndarray:
    r"""Drop middle nodes of nearly-collinear, finely-spaced triples.

    A node ``i`` is removed when both its adjacent chords are shorter than
    ``delta_max/4`` and the turn there is below ``alpha_max/4`` -- keeping the
    point count bounded on straight runs without coarsening corners or long
    segments. One left-to-right pass; endpoints are always kept.
    """
    n = points.shape[0]
    if n < 3:
        return points
    fine_chord = delta_max / 4.0
    flat_turn = alpha_max / 4.0
    keep = np.ones(n, dtype=bool)
    last = 0
    for i in range(1, n - 1):
        prev_pt = points[last]
        here = points[i]
        nxt = points[i + 1]
        c0 = hypot(float(here[0] - prev_pt[0]), float(here[1] - prev_pt[1]))
        c1 = hypot(float(nxt[0] - here[0]), float(nxt[1] - here[1]))
        incoming = here - prev_pt
        outgoing = nxt - here
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
        turn = abs(float(np.arctan2(cross, dot)))
        if c0 < fine_chord and c1 < fine_chord and turn < flat_turn:
            keep[i] = False
        else:
            last = i
    return points[keep]


# --------------------------------------------------------------------------- #
# Homoclinic intersection detection.
# --------------------------------------------------------------------------- #


def _arclengths(points: np.ndarray) -> np.ndarray:
    r"""Cumulative arclength along a polyline, starting at 0 for ``points[0]``."""
    if points.shape[0] == 0:
        return np.empty(0, dtype=float)
    diffs = np.diff(points, axis=0)
    seg = np.hypot(diffs[:, 0], diffs[:, 1])
    return np.concatenate([[0.0], np.cumsum(seg)])


_SIN_TEN_DEGREES = 0.17364817766693041  # sin(10 deg): transversality floor (design B.1)


def find_primary_homoclinic_points(
    unstable: np.ndarray,
    stable: np.ndarray,
    *,
    min_sin: float = _SIN_TEN_DEGREES,
    saddle: Sequence[float] | None = None,
    min_saddle_distance: float = 0.0,
) -> HomoclinicPoints:
    r"""Transversal, alternating homoclinic crossings of ``W^u`` and ``W^s``.

    Intersects the two polylines segment-by-segment, keeps only **transversal**
    crossings (sine of the angle between the unit tangents ``> min_sin``, default
    ``sin(10 deg)``), orders them by arclength along ``W^u`` from the saddle, and
    enforces **alternating sidedness** -- the signed side of ``W^u`` relative to
    ``W^s`` must flip between consecutive kept crossings, the defining
    lobe-alternation of a tangle. Tangential crossings are discarded. ``>= 3``
    surviving points are required to claim a tangle (the caller checks the count).

    When ``saddle`` is given, crossings within ``min_saddle_distance`` of it are
    **dropped**: ``W^u`` and ``W^s`` both emanate from the saddle, so they cross
    there degenerately, and that crossing is not a turnstile boundary. Always pass
    the saddle (with ``min_saddle_distance`` a few times the seed offset ``eps``)
    when the manifolds are seeded at the saddle, or the first "homoclinic point"
    is the saddle itself and any lobe built from it spans the whole tangle.

    Both polylines are taken ordered from the saddle outward so the arclength
    ordering is the physical lobe order. Use :func:`turnstile_lobes` on the result
    to select the index pairs that bound genuine (small) turnstile lobes.

    **Input contract (determinism).** This is a pure, deterministic function of its
    inputs: identical ``(unstable, stable)`` arrays always give the bit-identical
    result. The count, however, depends on *which* curves are passed -- a SINGLE
    ordered prong of ``W^u`` against a single prong of ``W^s`` counts that
    prong-pair's tangle, whereas a TWO-prong manifold joined through the saddle
    (``vstack([neg[::-1], pos])``) counts every ``W^u``-prong x ``W^s``-prong
    crossing *plus* near-degenerate crossings at the saddle seam, so it yields a
    much larger, less well-posed count. Pass single ordered prongs (and ``saddle``
    + ``min_saddle_distance``) for a reproducible per-prong tangle; if you pass a
    joined manifold, always supply the saddle so the seam crossings are dropped,
    and record explicitly which inputs produced the count.
    """
    wu = np.asarray(unstable, dtype=float)
    ws = np.asarray(stable, dtype=float)
    if wu.ndim != 2 or wu.shape[1] != 2 or ws.ndim != 2 or ws.shape[1] != 2:
        raise ValueError("manifold curves must have shape (n, 2)")
    saddle_point = None if saddle is None else np.asarray(saddle, dtype=float)
    if saddle_point is not None and saddle_point.shape != (2,):
        raise ValueError("saddle must be length-2")
    arc_u = _arclengths(wu)
    arc_s = _arclengths(ws)

    # Vectorised over W^s for each W^u segment: a single segment-vs-all-segments
    # intersection test per W^u node (numpy over the W^s array) is markedly faster
    # than a Python double loop and, on a densely-folded manifold, faster than a
    # spatial-grid prune (whose per-cell candidate sets stay large). Exact: every
    # crossing the all-pairs scan would find is found.
    ws_p1 = ws[:-1]
    ws_seg = ws[1:] - ws[:-1]
    ws_norm = np.hypot(ws_seg[:, 0], ws_seg[:, 1])
    crossings: list[tuple[float, float, np.ndarray, float, int]] = []
    for i in range(wu.shape[0] - 1):
        p1 = wu[i]
        tangent_u = wu[i + 1] - p1
        norm_u = hypot(float(tangent_u[0]), float(tangent_u[1]))
        if norm_u == 0.0:
            continue
        denom = tangent_u[0] * ws_seg[:, 1] - tangent_u[1] * ws_seg[:, 0]
        diff = ws_p1 - p1
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (diff[:, 0] * ws_seg[:, 1] - diff[:, 1] * ws_seg[:, 0]) / denom
            u = (diff[:, 0] * tangent_u[1] - diff[:, 1] * tangent_u[0]) / denom
        hits = (
            (np.abs(denom) >= 1.0e-15)
            & (t >= -1.0e-12)
            & (t <= 1.0 + 1.0e-12)
            & (u >= -1.0e-12)
            & (u <= 1.0 + 1.0e-12)
        )
        for j in np.flatnonzero(hits):
            norm_s = float(ws_norm[j])
            if norm_s == 0.0:
                continue
            cross = (
                tangent_u[0] * ws_seg[j, 1] - tangent_u[1] * ws_seg[j, 0]
            ) / (norm_u * norm_s)
            sin_angle = abs(float(cross))
            if sin_angle <= float(min_sin):
                continue
            point = p1 + float(t[j]) * tangent_u
            if saddle_point is not None and (
                hypot(float(point[0] - saddle_point[0]), float(point[1] - saddle_point[1]))
                < float(min_saddle_distance)
            ):
                continue
            side = 1 if cross > 0.0 else -1
            arclength_u = float(arc_u[i] + float(t[j]) * norm_u)
            arclength_s = float(arc_s[j] + float(u[j]) * norm_s)
            crossings.append((arclength_u, arclength_s, point, sin_angle, side))

    crossings.sort(key=lambda c: c[0])
    # Enforce alternating sidedness greedily along arclength on W^u.
    kept_u: list[float] = []
    kept_s: list[float] = []
    kept_points: list[np.ndarray] = []
    kept_sin: list[float] = []
    kept_side: list[int] = []
    last_side = 0
    for arclength_u, arclength_s, point, sin_angle, side in crossings:
        if side == last_side:
            continue
        kept_u.append(arclength_u)
        kept_s.append(arclength_s)
        kept_points.append(point)
        kept_sin.append(sin_angle)
        kept_side.append(side)
        last_side = side
    if kept_points:
        points = np.asarray(kept_points, dtype=float)
    else:
        points = np.empty((0, 2), dtype=float)
    return HomoclinicPoints(
        points=points,
        sin_angles=np.asarray(kept_sin, dtype=float),
        sides=np.asarray(kept_side, dtype=int),
        arclength_u=np.asarray(kept_u, dtype=float),
        arclength_s=np.asarray(kept_s, dtype=float),
    )


def turnstile_lobes(
    homoclinic: HomoclinicPoints,
    *,
    min_lobe_extent: float = 0.0,
    max_arclength_u: float | None = None,
) -> tuple[tuple[int, int], ...]:
    r"""Index pairs of homoclinic points that bound genuine turnstile lobes.

    A turnstile lobe is bounded by the ``W^u`` arc and the ``W^s`` arc joining two
    homoclinic points that are **adjacent on both manifolds**: consecutive in
    ``W^u`` arclength *and* with no other homoclinic point between them in ``W^s``
    arclength. Only then is the bounded region a single simple lobe (a small
    sliver). Consecutive-on-``W^u`` pairs whose ``W^s`` arc skips over other
    crossings would enclose many lobes at once -- the polygon-double-counts failure
    mode -- and are excluded.

    ``min_lobe_extent`` (in section distance units, e.g. metres) additionally
    drops **near-tangential** pairs whose two endpoints are closer together than
    this -- the "last homoclinic points" where ``W^u`` grazes ``W^s`` and the
    enclosed area is a numerical-noise degeneracy, not a transport lobe (design
    B.1 records but excludes these from the primary turnstile). Set it to a small
    fraction of the local chain scale ``L_loc`` (e.g. ``0.1 * L_loc``).

    ``max_arclength_u`` restricts lobes to those whose both endpoints lie within
    that ``W^u`` arclength -- pass :func:`first_fold_arclength` of the ``W^u``
    polyline to keep only **primary** lobes reachable before the manifold first
    folds back (design B.1); crossings past the first fold are higher-order.

    Returns the qualifying ``(k, k+1)`` index pairs in ``W^u`` arclength order
    (the primary lobes nearest the saddle come first). Pass a pair as ``lobe`` to
    :func:`turnstile_lobe_area_polygon` / :func:`turnstile_lobe_area_mmp`.
    """
    arclength_s = homoclinic.arclength_s
    arclength_u = homoclinic.arclength_u
    points = homoclinic.points
    n = arclength_s.shape[0]
    lobes: list[tuple[int, int]] = []
    for k in range(n - 1):
        if max_arclength_u is not None and (
            arclength_u[k] > float(max_arclength_u)
            or arclength_u[k + 1] > float(max_arclength_u)
        ):
            continue
        lo, hi = sorted((float(arclength_s[k]), float(arclength_s[k + 1])))
        between = int(np.sum((arclength_s > lo + 1e-12) & (arclength_s < hi - 1e-12)))
        if between != 0:
            continue
        extent = hypot(
            float(points[k + 1, 0] - points[k, 0]),
            float(points[k + 1, 1] - points[k, 1]),
        )
        if extent < float(min_lobe_extent):
            continue
        lobes.append((k, k + 1))
    return tuple(lobes)


def first_fold_arclength(curve: np.ndarray, *, fold_angle: float = pi / 2) -> float:
    r"""Arclength of the first sharp fold-back along a manifold polyline.

    The first node whose interior turning angle exceeds ``fold_angle`` (default
    ``pi/2``) marks where the manifold folds back on itself; crossings beyond it are
    higher-order, not primary (design B.1). Returns that node's cumulative arclength,
    or the total arclength if the curve never folds that sharply. Pass the result as
    ``max_arclength_u`` to :func:`turnstile_lobes` to keep only **primary** lobes.
    """
    polyline = np.asarray(curve, dtype=float)
    if polyline.ndim != 2 or polyline.shape[1] != 2:
        raise ValueError(f"curve must have shape (n, 2), got {polyline.shape}")
    arclengths = _arclengths(polyline)
    angles = _turning_angles(polyline)
    folded = np.flatnonzero(angles > float(fold_angle))
    return float(arclengths[folded[0]]) if folded.size else float(arclengths[-1])


# --------------------------------------------------------------------------- #
# Turnstile lobe area (two methods).
# --------------------------------------------------------------------------- #


def _arc_between(points: np.ndarray, arclengths: np.ndarray, a: float, b: float) -> np.ndarray:
    r"""Sub-polyline of ``points`` traversed from arclength ``a`` to arclength ``b``.

    The endpoints ``a`` and ``b`` are appended as exact crossing coordinates
    (interpolated on the bracketing segment) so a lobe polygon closes precisely on
    the crossings rather than on the nearest node. The returned points are ordered
    **from ``a`` to ``b``** (reversed when ``a > b``), so a caller can build a
    correctly-closed lobe loop by going ``h_k -> h_{k+1}`` along one manifold and
    ``h_{k+1} -> h_k`` along the other, irrespective of how the two homoclinic
    points are ordered in each manifold's own arclength.
    """
    lo, hi = (a, b) if a <= b else (b, a)
    interior_mask = (arclengths > lo) & (arclengths < hi)
    end_lo = _interp_point_at_arclength(points, arclengths, lo)
    end_hi = _interp_point_at_arclength(points, arclengths, hi)
    interior = points[interior_mask]
    arc = np.vstack([end_lo[None, :], interior, end_hi[None, :]])
    return arc if a <= b else arc[::-1]


def _lobe_loop(
    unstable: np.ndarray,
    stable: np.ndarray,
    homoclinic: HomoclinicPoints,
    lobe: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    r"""The two oriented arcs bounding the turnstile lobe on homoclinic pair ``lobe``.

    ``lobe = (i, j)`` are the two homoclinic-point indices bounding the lobe (from
    :func:`turnstile_lobes`). Returns ``(u_arc, s_arc)`` where ``u_arc`` traverses
    ``W^u`` from ``h_i`` to ``h_j`` and ``s_arc`` traverses ``W^s`` from ``h_j``
    back to ``h_i``, so ``concatenate([u_arc, s_arc])`` is a closed simple loop
    around the lobe. SSOT for both area methods.
    """
    i, j = int(lobe[0]), int(lobe[1])
    n = homoclinic.points.shape[0]
    if not (0 <= i < n and 0 <= j < n) or i == j:
        raise ValueError(f"lobe indices {lobe} invalid for {n} homoclinic points")
    arc_u_lengths = _arclengths(unstable)
    arc_s_lengths = _arclengths(stable)
    u_arc = _arc_between(
        unstable, arc_u_lengths, homoclinic.arclength_u[i], homoclinic.arclength_u[j]
    )
    s_arc = _arc_between(
        stable, arc_s_lengths, homoclinic.arclength_s[j], homoclinic.arclength_s[i]
    )
    return u_arc, s_arc


def _interp_point_at_arclength(
    points: np.ndarray, arclengths: np.ndarray, target: float
) -> np.ndarray:
    r"""Linear-interpolate the section point at a given arclength along a polyline."""
    if target <= arclengths[0]:
        return points[0]
    if target >= arclengths[-1]:
        return points[-1]
    idx = int(np.searchsorted(arclengths, target))
    seg_lo = arclengths[idx - 1]
    seg_hi = arclengths[idx]
    span = seg_hi - seg_lo
    frac = 0.0 if span == 0.0 else (target - seg_lo) / span
    return points[idx - 1] + frac * (points[idx] - points[idx - 1])


def _shoelace_area(polygon: np.ndarray) -> float:
    r"""Absolute area of a closed polygon via the shoelace formula (Green's thm)."""
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def turnstile_lobe_area_polygon(
    unstable: np.ndarray,
    stable: np.ndarray,
    homoclinic: HomoclinicPoints,
    *,
    lobe: tuple[int, int],
) -> LobeArea:
    r"""Green's-theorem polygon area of one turnstile lobe (Method 1).

    ``lobe = (i, j)`` are the two homoclinic-point indices bounding the lobe, as
    returned by :func:`turnstile_lobes` (which guarantees they are adjacent on both
    manifolds, so the region is a single sliver). The closed polygon concatenates
    the ``W^u`` arc ``h_i -> h_j`` with the ``W^s`` arc ``h_j -> h_i``; the area is
    the shoelace sum over the fully-resolved nodes.
    """
    u_arc, s_arc = _lobe_loop(
        np.asarray(unstable, dtype=float),
        np.asarray(stable, dtype=float),
        homoclinic,
        lobe,
    )
    polygon = np.vstack([u_arc, s_arc])
    return LobeArea(area=_shoelace_area(polygon), method="polygon", used_true_action=False)


def turnstile_lobe_area_mmp(
    unstable: np.ndarray,
    stable: np.ndarray,
    homoclinic: HomoclinicPoints,
    *,
    lobe: tuple[int, int],
    action: Callable[[np.ndarray], float] | None = None,
) -> LobeArea:
    r"""MacKay-Meiss-Percival turnstile lobe area / flux (Method 2).

    ``lobe = (i, j)`` are the two bounding homoclinic-point indices (see
    :func:`turnstile_lobes`). The MMP flux through a broken separatrix is the
    **action difference** between the two homoclinic orbits bounding the lobe. When
    a genuine action functional ``action(arc_points) -> float`` is supplied (e.g.
    the standard map's generating-function action, or a field-line action), this
    returns the closed-loop action ``|S(arc_u) + S(arc_s)|`` over the two oriented
    arcs joining ``h_i`` and ``h_j`` and flags ``used_true_action=True``.

    **Independence caveat.** For an area-giving one-form (such as ``integral Z dR``)
    the closed-loop action equals the enclosed area by Green's theorem, so it agrees
    with :func:`turnstile_lobe_area_polygon` by identity -- it is *not* a second
    independent estimator. A genuinely independent MMP value requires a true action
    that is not an exact differential of the area (the field-line action), which is
    not available from the bare field here.

    When no action functional is available (``action is None``), the MMP area
    reduces to the **signed** area of the same lobe polygon; ``used_true_action=False``
    records that the cross-check against the polygon area is then a sign/orientation
    check, not two independent estimates -- as the design requires us to state.

    The two arcs are oriented as a closed loop (``h_i -> h_j`` along ``W^u``,
    ``h_j -> h_i`` along ``W^s``).
    """
    u_arc, s_arc = _lobe_loop(
        np.asarray(unstable, dtype=float),
        np.asarray(stable, dtype=float),
        homoclinic,
        lobe,
    )
    if action is not None:
        # u_arc: h_i -> h_j; s_arc: h_j -> h_i. The closed-loop integral of a
        # one-form is action(u_arc) + action(s_arc); for an area-giving form this is
        # the signed lobe area (see the independence caveat in the docstring).
        loop_action = float(action(u_arc)) + float(action(s_arc))
        return LobeArea(
            area=abs(loop_action), method="mmp_action", used_true_action=True
        )
    polygon = np.vstack([u_arc, s_arc])
    x = polygon[:, 0]
    y = polygon[:, 1]
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    return LobeArea(
        area=abs(signed), method="mmp_signed_area", used_true_action=False
    )
