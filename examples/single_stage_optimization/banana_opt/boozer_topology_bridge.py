"""Non-Boozer topology bridge metrics (Phase 3b of the Stage 2 / single-stage
Boozer handoff plan).

Two diagnostics, both pure-function and Boozer-independent:

- :func:`compute_helical_field_content_S_HEL`: spectral fraction of the
  on-surface ``|B|`` field that lives in helical (``n != 0``) modes. Scale-
  invariant; ``0`` for an axisymmetric field, ``1`` when all power is in
  helical modes. Used as a ranking signal and, when explicitly enabled after
  gradient validation per plan line 308, as a ramped Stage 2 objective term.

- :func:`compute_fieldline_iota_proxy`: rotational-transform proxy derived
  from field-line tracing, with a convergence sub-study that fails closed
  when ``|iota_FL(2N) - iota_FL(N)| > 0.005`` (plan line 274). Designed for
  post-solve / on-checkpoint use only; plan line 270 ("evaluate post-solve
  or on checkpoint only, not every optimizer iteration") and line 283
  ("keep field-line iota as validation/truth signal, not the per-iteration
  objective") explicitly forbid use as a per-iteration objective.

Artifact-key ownership (Phase 3a vs Phase 3b):

The live in-loop ``FIELDLINE_IOTA_PROXY`` / ``FIELDLINE_IOTA_PROXY_VALID``
keys are produced exclusively by :mod:`banana_opt.topology_bridge` (Phase
3a). The convergence-validated proxy implemented in this module is the
sub-study tool driven by ``scripts/fieldline_iota_proxy_convergence.py``;
its results do not flow into the per-run artifact directly. Phase 3b's
artifact contribution is the helical-content keys
(``HELICAL_FIELD_CONTENT``, ``S_HEL_OBJECTIVE_WEIGHT``) and the composite
ranker (``PRE_BOOZER_TOPOLOGY_SCORE``). This split keeps the "valid"
semantics for ``FIELDLINE_IOTA_PROXY_VALID`` unambiguous: it means
"transit-threshold met for the live trace", produced by exactly one
caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simsopt.field.tracing import compute_fieldlines
from topology_scorer import (
    build_stopping_criteria,
    midplane_seed_radii,
)

# Phase 3b deliberately does NOT re-export ``DEFAULT_NFIELDLINES`` from
# the shared module: this module's two field-line callers use distinct
# hardcoded ``n_lines`` values intentionally (see the per-function
# docstrings on :func:`compute_fieldline_iota_proxy` and
# :func:`safe_compute_fieldline_iota_proxy`) — wiring the shared default
# through here would silently rewrite those values. The shared
# ``DEFAULT_NFIELDLINES`` is the canonical default for the *Phase 3a*
# pipeline (:mod:`banana_opt.topology_bridge`); consumers that want the
# shared default should import it from ``topology_bridge_shared``.
from banana_opt.topology_bridge_shared import (
    DEFAULT_TMAX,
    DEFAULT_TOL,
    SurfaceCentroidCoordinateAxis,
    build_surface_centroid_axis,
)


__all__ = [
    "DEFAULT_TMAX",
    "DEFAULT_TOL",
    "FIELDLINE_IOTA_CONVERGENCE_TOL",
    "FieldlineIotaProxyResult",
    "HelicalFieldContentObjective",
    "boozer_topology_bridge_artifact_fields",
    "compute_fieldline_iota_proxy",
    "compute_helical_field_content_S_HEL",
    "compute_pre_boozer_topology_score",
    "compute_S_HEL_gradient_relative_error",
    "helical_field_content_from_modB_grid",
    "safe_compute_fieldline_iota_proxy",
    "safe_compute_helical_field_content_S_HEL",
]


# Plan line 274: ``|iota_FL(N=200) - iota_FL(N=100)| < 0.005``.
FIELDLINE_IOTA_CONVERGENCE_TOL = 0.005


@dataclass(frozen=True)
class FieldlineIotaProxyResult:
    """Diagnostic-only field-line iota proxy with convergence-gated trust.

    ``valid=True`` iff every paired short/long iota (where both rungs
    survived for the *same* seed index) agreed within
    :data:`FIELDLINE_IOTA_CONVERGENCE_TOL`. Callers must NOT use this as a
    per-iteration objective — plan lines 270 and 283.

    ``iota_per_line`` is reported per-seed-index with ``np.nan`` for any
    seed that escaped before completing a usable arc (so the tuple index
    is a stable seed identifier across short/long rungs and pair-wise
    convergence checking is well-defined).
    """

    iota_proxy_mean: float | None
    iota_proxy_std: float | None
    iota_per_line: tuple[float, ...]
    n_lines_seeded: int
    n_lines_survived: int
    tmax: float
    valid: bool
    convergence_residual: float | None
    invalid_reason: str | None


def _s_hel_from_modB(
    modB: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """SSOT for the helical-content spectrum, total power, and ``S_HEL`` value.

    Computes the real-input FFT spectrum of ``|B|`` (``np.fft.rfft2`` —
    the imaginary half is redundant for real input so we never allocate
    it), the total spectral power (Parseval denominator), and the
    ``S_HEL`` ratio ``sum(|F[n!=0]|^2) / sum(|F|^2)``.

    Returns:
        s_hel: helical-power fraction in ``[0, 1]``.
        total_power: full Parseval-style power sum (denominator).
        spectrum: the unmasked half-spectrum ``np.fft.rfft2(modB)``;
            callers needing the gradient consume this directly.

    Half-spectrum power accounting (real-input FFT):

    The ``rfft2`` output covers axis 1 indices ``0 .. ntheta//2``; the
    columns at index 0 and (when ``ntheta`` is even) ``ntheta // 2``
    represent symmetric pairs of length 1 — their power is counted once.
    All interior columns represent a conjugate pair — their power is
    counted twice to match the full ``fft2`` Parseval sum.

    Raises:
        ValueError: when ``modB`` has zero total spectral power
            (``|B| ≡ 0``); the metric is undefined.
    """
    nphi, ntheta = modB.shape
    spectrum = np.fft.rfft2(modB)
    nrows = spectrum.shape[1]
    # Conjugate-pair multiplicity vector: 1 for the DC + Nyquist columns
    # and 2 for every interior column whose conjugate sits in the
    # discarded half-spectrum.
    multiplicity = np.full(nrows, 2.0)
    multiplicity[0] = 1.0
    if ntheta % 2 == 0:
        multiplicity[-1] = 1.0
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power * multiplicity))
    if total_power == 0.0:
        raise ValueError("|B| grid has zero spectral power; S_HEL is undefined")
    # Helical = total − axisymmetric (n = 0 row, axis 0 index 0). The
    # axisymmetric row carries the same conjugate-pair multiplicity as
    # any other row.
    axisymmetric_power = float(np.sum(power[0, :] * multiplicity))
    helical_power = total_power - axisymmetric_power
    s_hel = helical_power / total_power
    return s_hel, total_power, spectrum


def helical_field_content_from_modB_grid(modB: np.ndarray) -> float:
    """Compute ``S_HEL`` directly from a sampled ``|B|`` grid.

    ``modB`` has shape ``(nphi, ntheta)`` and must contain finite samples
    of the field magnitude on a uniform (phi, theta) grid covering one
    field period (or the full torus) at uniform spacing.

    Plan definition (Stage 2 → single-stage Boozer handoff plan, lines
    278-280)::

        S_HEL = sum(|B_hat[m,n]|^2 for n != 0) / sum(|B_hat[m,n]|^2)

    where the FFT lays out axis=0 = phi (toroidal, mode ``n``) and axis=1
    = theta (poloidal, mode ``m``). The denominator is the full Parseval
    sum so the result is the fraction of total spectral power sitting in
    helical (``n != 0``) modes. Result is in ``[0, 1]``: ``0`` for an
    axisymmetric field, ``1`` when all spectral power is helical. The
    metric is invariant to a uniform scaling of ``|B|`` because the
    numerator and denominator scale identically.

    Implementation note: uses ``np.fft.rfft2`` (real-input FFT) so the
    redundant complex-conjugate half-spectrum is never allocated. The
    conjugate-pair multiplicity is folded into the power sum so the
    result matches the full ``fft2`` Parseval accounting bit-for-bit.

    Raises:
        ValueError: when ``modB`` is not 2D, contains non-finite samples,
            or has zero total spectral power (the metric is undefined).
    """
    modB = np.asarray(modB, dtype=float)
    if modB.ndim != 2:
        raise ValueError(
            f"|B| grid must be 2D (nphi, ntheta); got shape {modB.shape}"
        )
    if not np.all(np.isfinite(modB)):
        raise ValueError("|B| grid contains non-finite samples")
    s_hel, _, _ = _s_hel_from_modB(modB)
    return s_hel


def compute_helical_field_content_S_HEL(
    field, surface
) -> float:
    """Compute ``S_HEL`` for ``field`` sampled on ``surface``.

    Wraps :func:`helical_field_content_from_modB_grid` with the standard
    simsopt sampling protocol: set the field points to ``surface.gamma()``,
    read ``B``, take the magnitude on the (nphi, ntheta) grid. We
    transform ``|B|`` — NOT ``B`` itself, NOT ``B·n`` — so the metric
    captures field-magnitude modulation, which is what governs particle
    confinement on the surface.

    Side effect (documented; no defensive restore per project guideline):
    this routine calls ``field.set_points(surface.gamma())`` and does NOT
    restore the field's previous evaluation points. The integration site
    in ``banana_opt.STAGE_2.banana_coil_solver`` evaluates S_HEL FIRST
    and then invokes the field-line tracer (which always re-sets the
    field's points on every step), so the side effect is invisible in
    production. Callers that share the field with another consumer that
    reads stale points after this call must reset the points themselves.
    """
    modB, _ = _modB_and_B_unit_on_surface(field, surface)
    return helical_field_content_from_modB_grid(modB)


def safe_compute_helical_field_content_S_HEL(
    field, surface
) -> float | None:
    """Best-effort wrapper that returns ``None`` on the documented failure modes.

    Per the Phase 3 plan ("Failed diagnostics fail closed as unavailable
    metrics, not as fake zero-quality success", line 305), the diagnostic
    path must not crash the parent solver when a degenerate ``|B|`` grid
    makes ``S_HEL`` mathematically undefined. The pure-function
    :func:`compute_helical_field_content_S_HEL` keeps strict semantics for
    unit tests; this wrapper is what the artifact pipeline calls.

    Returns ``None`` for the failure cases
    :func:`helical_field_content_from_modB_grid` raises ``ValueError`` for:
    a 2-D non-3-component gamma, non-finite ``|B|`` samples, or a grid
    with zero TOTAL spectral power (i.e. ``|B| ≡ 0`` everywhere — a
    truly degenerate Boozer surface). A field with finite DC and zero
    non-DC content reports ``S_HEL = 0.0`` cleanly and is NOT a failure
    case. All other exceptions propagate so they are not silently
    swallowed.
    """
    try:
        return compute_helical_field_content_S_HEL(field, surface)
    except ValueError:
        return None


def _continuous_angle_iota(
    histories,
    coordinate_axis: SurfaceCentroidCoordinateAxis,
) -> np.ndarray:
    """Compute per-line iota by continuous-angle accumulation.

    For each trajectory, accumulate the unwrapped toroidal angle ``phi``
    and the unwrapped poloidal angle ``theta`` along the entire arc and
    return ``iota = delta_theta / delta_phi``. This bypasses the
    integer-counting ``np.round`` that :func:`compute_toroidal_transits`
    / :func:`compute_poloidal_transits` use upstream — those quantize
    the iota to ``k/ntor`` resolution where ``ntor`` is the *integer*
    transit count, capping the achievable convergence tolerance at
    ``1/ntor``.

    Returns:
        iota: shape ``(nlines,)`` array of per-line iota values. Lines
            that did not accumulate at least one full toroidal revolution
            (``|delta_phi| < 2*pi``) are reported as ``np.nan`` — no
            convention attempt would yield a meaningful rotational
            transform from less than one revolution of toroidal arc.

    The poloidal angle is the same arctan-around-centroid the upstream
    counter uses (boozersurface.py companion ``compute_poloidal_transits``
    flux=False branch), so the average iota agrees with the legacy
    transit-counting estimate at high-tmax to within the truncation
    floor.
    """
    nlines = len(histories)
    iota = np.full(nlines, np.nan, dtype=float)
    gamma_buf = np.zeros((1, 3))
    for line_index in range(nlines):
        traj = np.asarray(histories[line_index], dtype=float)
        ntraj = traj.shape[0]
        if ntraj < 2:
            continue
        # Cumulative phi unwrap. ``np.arctan2`` lives in [-pi, pi]; the
        # native simsopt ``compute_toroidal_transits`` uses ``sopp.get_phi``
        # which does the same wrap. ``np.unwrap`` extends to continuous
        # phi by adding 2*pi multiples where consecutive jumps exceed pi
        # in magnitude.
        phi_wrapped = np.arctan2(traj[:, 2], traj[:, 1])
        phi_continuous = np.unwrap(phi_wrapped)
        delta_phi = phi_continuous[-1] - phi_continuous[0]
        if abs(delta_phi) < 2.0 * np.pi:
            continue
        # Poloidal angle: arctan(R - R_ma, Z - Z_ma) on each sample,
        # then unwrap. The coordinate axis is sampled at each
        # trajectory-side phi (in the [0, 1) fractional convention the
        # adapter expects).
        R_traj = np.sqrt(traj[:, 1] ** 2 + traj[:, 2] ** 2)
        Z_traj = traj[:, 3]
        # The trajectory-side phi for indexing the centroid axis is the
        # *unnormalized* arctan2 angle, divided by 2*pi and reduced into
        # [0, 1) by the adapter's modulo step.
        phi_frac = phi_wrapped / (2.0 * np.pi)
        theta_samples = np.empty(ntraj, dtype=float)
        for step in range(ntraj):
            coordinate_axis.gamma_impl(gamma_buf, float(phi_frac[step]))
            R_ma = float(np.sqrt(gamma_buf[0, 0] ** 2 + gamma_buf[0, 1] ** 2))
            Z_ma = float(gamma_buf[0, 2])
            # arctan2(R - R_ma, Z - Z_ma) matches the
            # ``compute_poloidal_transits(flux=False)`` convention
            # (boozersurface.py companion: ``sopp.get_phi(R-R_ma, Z-Z_ma, ...)``)
            theta_samples[step] = np.arctan2(
                R_traj[step] - R_ma, Z_traj[step] - Z_ma
            )
        theta_continuous = np.unwrap(theta_samples)
        delta_theta = theta_continuous[-1] - theta_continuous[0]
        iota[line_index] = float(delta_theta) / float(delta_phi)
    return iota


def _trace_fieldlines_and_compute_iota(
    field,
    surface,
    *,
    coordinate_axis: SurfaceCentroidCoordinateAxis,
    n_lines: int,
    tmax: float,
    tol: float,
) -> tuple[np.ndarray, int]:
    """Trace ``n_lines`` field lines and return per-seed-index iota + survival.

    Returns:
        iota_per_seed: shape ``(n_lines,)`` array; entry ``i`` is the
            iota for seed index ``i`` (``np.nan`` if that seed did not
            survive long enough to define iota). Seed indices align across
            short and long rungs because they index the same midplane-seed
            radii array, which is the key invariant the C2 fix preserves.
        survived: number of seeds with a finite iota entry.

    The C2 fix replaces the legacy ``list.append`` pattern (which dropped
    escapees so seed indices drifted between short and long rungs) with a
    fixed-shape ``np.nan``-padded array so the caller can do paired
    convergence checking by seed index without alignment hazards.
    """
    radii = midplane_seed_radii(surface, int(n_lines))
    R0 = np.asarray(radii, dtype=float)
    Z0 = np.zeros_like(R0)
    stopping_criteria, _ = build_stopping_criteria(surface)
    histories, _ = compute_fieldlines(
        field,
        R0,
        Z0,
        tmax=float(tmax),
        tol=float(tol),
        phis=[0.0],
        stopping_criteria=stopping_criteria,
    )
    iota_per_seed = _continuous_angle_iota(histories, coordinate_axis)
    survived = int(np.sum(np.isfinite(iota_per_seed)))
    return iota_per_seed, survived


def compute_fieldline_iota_proxy(
    field,
    surface,
    *,
    n_lines: int = 8,
    tmax: float = DEFAULT_TMAX,
    tol: float = 1.0e-8,
    convergence_tol: float = FIELDLINE_IOTA_CONVERGENCE_TOL,
) -> FieldlineIotaProxyResult:
    """Compute the field-line iota proxy with a convergence sub-study.

    The proxy is the mean per-surviving-line iota at trace length
    ``tmax`` computed by continuous-angle accumulation (see
    :func:`_continuous_angle_iota` — bypasses the integer-counting
    resolution floor that the legacy
    ``compute_toroidal_transits / compute_poloidal_transits`` divide
    inherited). The convergence sub-study traces the same seeds at
    ``2 * tmax`` and the proxy is only flagged ``valid=True`` when every
    seed index that produced an iota at *both* rungs agrees within
    ``convergence_tol`` (plan line 274).

    Pair-wise convergence (C2 fix): the short and long rungs return
    per-seed-index ``np.nan``-padded iota arrays so a seed that escaped
    short but survived long (or vice versa) is dropped from the
    comparison instead of misaligning the comparison by list index.

    Failures (no surviving lines, divergent per-line iota, non-finite
    transit counts) yield ``valid=False`` and ``iota_proxy_mean=None``.

    The ``n_lines=8`` default is intentional for the convergence sub-study
    call site (denser midplane seeding so the convergence verdict is
    robust to a few escaping seeds). The Phase 3a live-loop default in
    :data:`banana_opt.topology_bridge_shared.DEFAULT_NFIELDLINES` is 5 —
    those values are independent on purpose; do NOT wire them together.
    """
    if n_lines <= 0:
        raise ValueError("n_lines must be positive")
    if tmax <= 0:
        raise ValueError("tmax must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    coordinate_axis = build_surface_centroid_axis(surface)
    iota_short, survived_short = _trace_fieldlines_and_compute_iota(
        field,
        surface,
        coordinate_axis=coordinate_axis,
        n_lines=n_lines,
        tmax=tmax,
        tol=tol,
    )
    if survived_short == 0:
        return FieldlineIotaProxyResult(
            iota_proxy_mean=None,
            iota_proxy_std=None,
            iota_per_line=tuple(iota_short.tolist()),
            n_lines_seeded=int(n_lines),
            n_lines_survived=0,
            tmax=float(tmax),
            valid=False,
            convergence_residual=None,
            invalid_reason="no_lines_survived",
        )
    iota_long, _survived_long = _trace_fieldlines_and_compute_iota(
        field,
        surface,
        coordinate_axis=coordinate_axis,
        n_lines=n_lines,
        tmax=2.0 * tmax,
        tol=tol,
    )
    # Pair-wise convergence by seed index: a seed contributes to the
    # convergence residual only when it survived (finite iota) at BOTH
    # rungs. This is the C2 fix — the legacy code paired list-index
    # ``iota_short[i]`` with ``iota_long[i]`` after both had had escapees
    # dropped, so seeds that escaped in one rung but not the other got
    # silently misaligned.
    valid_pairs = np.isfinite(iota_short) & np.isfinite(iota_long)
    n_compare = int(np.sum(valid_pairs))
    if n_compare == 0:
        return FieldlineIotaProxyResult(
            iota_proxy_mean=None,
            iota_proxy_std=None,
            iota_per_line=tuple(iota_short.tolist()),
            n_lines_seeded=int(n_lines),
            n_lines_survived=int(survived_short),
            tmax=float(tmax),
            valid=False,
            convergence_residual=None,
            invalid_reason="long_trace_had_no_survivors",
        )
    residuals = np.abs(iota_long[valid_pairs] - iota_short[valid_pairs])
    convergence_residual = float(np.max(residuals))
    valid = convergence_residual < float(convergence_tol)
    invalid_reason = None if valid else "convergence_tol_exceeded"
    finite_short = iota_short[np.isfinite(iota_short)]
    return FieldlineIotaProxyResult(
        iota_proxy_mean=float(np.mean(finite_short)),
        iota_proxy_std=float(np.std(finite_short)),
        iota_per_line=tuple(iota_short.tolist()),
        n_lines_seeded=int(n_lines),
        n_lines_survived=int(survived_short),
        tmax=float(tmax),
        valid=valid,
        convergence_residual=convergence_residual,
        invalid_reason=invalid_reason,
    )


def safe_compute_fieldline_iota_proxy(
    field,
    surface,
    *,
    n_lines: int = 4,
    tmax: float = 100.0,
    tol: float = 1.0e-8,
    convergence_tol: float = FIELDLINE_IOTA_CONVERGENCE_TOL,
) -> "FieldlineIotaProxyResult | None":
    """Best-effort wrapper that returns ``None`` on tracing failures.

    The convergence sub-study and direct-iota Phase 3b path call this when
    the convergence-validated proxy is needed without crashing the caller
    when field-line tracing fails (degenerate field, simsoptpp internal
    error). Per plan line 305 ("Failed diagnostics fail closed as
    unavailable metrics, not as fake zero-quality success"), a tracing
    failure must surface as ``None`` so downstream consumers can detect
    "diagnostic unavailable" rather than read a fabricated value.

    Live-loop artifact callers should NOT use this — the Phase 3a module
    (:mod:`banana_opt.topology_bridge`) is the canonical producer of the
    ``FIELDLINE_IOTA_PROXY`` artifact key. This wrapper exists for the
    convergence sub-study and external tooling that want the stronger
    convergence-tolerance ``valid`` semantics.

    The ``n_lines=4`` default (vs. ``n_lines=8`` on the wrapped
    :func:`compute_fieldline_iota_proxy`) is intentional for this
    safe-wrapper call site: the external-tooling and sub-study contexts
    that consume the convergence-tolerance semantics trade seed density
    for runtime, so the wrapper defaults to the cheaper 4-seed sweep.
    The Phase 3a live-loop default in
    :data:`banana_opt.topology_bridge_shared.DEFAULT_NFIELDLINES` is 5
    and lives on a different code path; do NOT wire any of the three
    together.

    Only ``ValueError`` (the parameter-validation failure modes documented
    on :func:`compute_fieldline_iota_proxy`) and ``RuntimeError`` (raised by
    ``simsopt.field.tracing`` on numerical breakdown) are swallowed. Every
    other exception propagates so that genuine bugs are not masked.
    """
    try:
        return compute_fieldline_iota_proxy(
            field,
            surface,
            n_lines=n_lines,
            tmax=tmax,
            tol=tol,
            convergence_tol=convergence_tol,
        )
    except (ValueError, RuntimeError):
        return None


def compute_pre_boozer_topology_score(
    *,
    s_hel: float | None,
    fieldline_iota_proxy_mean: float | None,
    fieldline_iota_proxy_valid: bool,
    iota_target: float | None,
) -> float | None:
    """Composite ranking score combining helical content + iota proxy.

    Higher is better. ``None`` when neither diagnostic is available so
    downstream rankers can fail closed rather than reading a fabricated
    zero. The composite is intentionally simple — the goal is "good donor
    for Boozer activation", not a research-grade quality metric.

    Definition (one-sentence, recomputable from the artifact alone):

    .. code-block:: text

        score = S_HEL * (1 - |iota_proxy - iota_target|)   when both available
              = S_HEL                                       when only S_HEL
              = max(0, 1 - |iota_proxy - iota_target|)      when only iota_proxy
              = None                                        otherwise

    When the iota proxy is flagged invalid by the convergence gate, it is
    treated as unavailable.
    """
    proxy_available = (
        fieldline_iota_proxy_mean is not None
        and bool(fieldline_iota_proxy_valid)
    )
    if s_hel is None and not proxy_available:
        return None
    if s_hel is None:
        if iota_target is None:
            return None
        return max(
            0.0,
            1.0 - abs(float(fieldline_iota_proxy_mean) - float(iota_target)),
        )
    if not proxy_available or iota_target is None:
        return float(s_hel)
    proximity = 1.0 - abs(
        float(fieldline_iota_proxy_mean) - float(iota_target)
    )
    return float(s_hel) * max(0.0, proximity)


def boozer_topology_bridge_artifact_fields(
    *,
    s_hel: float | None,
    fieldline_iota_proxy_mean: float | None,
    fieldline_iota_proxy_valid: bool | None,
    s_hel_objective_weight: float | None,
    iota_target: float | None,
) -> dict[str, object]:
    """Build the Phase 3b artifact-key payload (helical content + composite).

    Phase 3a (:mod:`banana_opt.topology_bridge`) owns the
    ``FIELDLINE_IOTA_PROXY`` / ``FIELDLINE_IOTA_PROXY_VALID`` keys and
    emits them via :func:`fieldline_iota_proxy_artifact_fields` after
    this function runs. This function emits only the three Phase 3b keys
    so the canonical producer of the field-line proxy keys is
    unambiguous (no override race between Phase 3a and Phase 3b results
    with different ``valid`` semantics).

    The composite ranker (``PRE_BOOZER_TOPOLOGY_SCORE``) consumes the
    Phase 3a field-line result via the explicit
    ``fieldline_iota_proxy_mean`` / ``fieldline_iota_proxy_valid``
    arguments; pass ``None`` / ``False`` when the field-line proxy was
    not invoked or returned invalid.
    """
    composite = compute_pre_boozer_topology_score(
        s_hel=s_hel,
        fieldline_iota_proxy_mean=fieldline_iota_proxy_mean,
        fieldline_iota_proxy_valid=bool(fieldline_iota_proxy_valid),
        iota_target=iota_target,
    )
    return {
        "HELICAL_FIELD_CONTENT": (
            None if s_hel is None else float(s_hel)
        ),
        "S_HEL_OBJECTIVE_WEIGHT": (
            None
            if s_hel_objective_weight is None
            else float(s_hel_objective_weight)
        ),
        "PRE_BOOZER_TOPOLOGY_SCORE": (
            None if composite is None else float(composite)
        ),
    }


def _modB_and_B_unit_on_surface(
    field, surface
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sample ``|B|`` and (optionally) the unit ``B``-vector on the surface.

    Returns:
        modB: shape ``(nphi, ntheta)``.
        B_unit: shape ``(nphi, ntheta, 3)``, the per-point unit vector
            ``B / |B|`` used by the gradient chain rule. ``None`` is never
            returned in the current code path; the optional return type is
            reserved for future callers that only need ``|B|``.

    Side effect (documented; no defensive restore per project guideline):
    this routine calls ``field.set_points(surface.gamma())`` and does NOT
    restore the field's previous evaluation points. See the docstring on
    :func:`compute_helical_field_content_S_HEL` for the integration-site
    ordering that makes this safe in production.

    Raises:
        ValueError: when ``surface.gamma()`` is not ``(nphi, ntheta, 3)``,
            or when any sample has ``|B| == 0`` (degenerate Boozer surface).
            The zero-|B| raise is what makes the safe-wrapper pattern in
            :func:`safe_compute_helical_field_content_S_HEL` and the
            Phase 3 plan's "Failed diagnostics fail closed as unavailable
            metrics" rule work: ``ValueError -> valid=False / None`` in
            the artifact, no NaN propagation into the optimizer.

    The flat ``(nphi * ntheta, 3)`` ordering matches what
    :func:`BiotSavart.B_vjp` expects as its co-tangent argument.
    """
    gamma = surface.gamma()
    if gamma.ndim != 3 or gamma.shape[-1] != 3:
        raise ValueError(
            f"surface.gamma() must be (nphi, ntheta, 3); got {gamma.shape}"
        )
    nphi, ntheta, _ = gamma.shape
    flat_points = gamma.reshape((-1, 3))
    field.set_points(flat_points)
    B = np.asarray(field.B(), dtype=float).reshape((nphi, ntheta, 3))
    modB = np.linalg.norm(B, axis=-1)
    # C6 fix: a single zero-|B| sample turned B_unit into NaN, which then
    # propagated through the gradient chain rule and corrupted the
    # optimizer update silently. Raise explicitly so the safe wrapper
    # converts it to a ``valid=False`` artifact entry.
    if not np.all(modB > 0.0):
        raise ValueError(
            "zero magnetic field on working surface — degenerate Boozer solve"
        )
    B_unit = B / modB[..., None]
    return modB, B_unit


def _S_HEL_value_and_dS_HEL_by_dmodB(
    modB: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return ``S_HEL`` and ``dS_HEL/d(modB)`` from the |B| grid alone.

    Math (see plan derivation):

    - ``F = np.fft.fft2(modB)``.
    - Mask ``M`` zeroes the ``n = 0`` (axisymmetric) row of ``F``:
      ``M[0, :] = 0``, ``M[k, :] = 1`` otherwise. ``M`` is real-valued so
      ``M = M^2``.
    - Numerator ``N(modB) = || M ⊙ F ||^2``.
      ``dN/d(modB) = 2 * (nphi * ntheta) * Re( np.fft.ifft2(M^2 ⊙ F) )``
      (``np.fft.ifft2`` includes the ``1 / (nphi * ntheta)`` factor, so the
      explicit ``(nphi * ntheta)`` multiplier restores the Parseval-style
      adjoint).
    - Denominator ``D(modB) = || F ||^2 = (nphi * ntheta) * ||modB||^2``
      (Parseval).  ``dD/d(modB) = 2 * (nphi * ntheta) * modB``.
    - ``dS/d(modB) = (dN * D - N * dD) / D^2``.

    The value path delegates to :func:`_s_hel_from_modB` (the SSOT for
    ``S_HEL`` value and total power) so the value here and the value
    consumed by :func:`compute_helical_field_content_S_HEL` are guaranteed
    bit-identical. The gradient continues to use ``np.fft.fft2`` /
    ``np.fft.ifft2`` rather than ``rfft2`` / ``irfft2`` because the
    inverse-FFT adjoint expression is cleaner when the full
    complex-conjugate spectrum is explicit — the cost is one extra FFT
    pair, paid once per gradient call.
    """
    nphi, ntheta = modB.shape
    S_HEL, total_power, _ = _s_hel_from_modB(modB)
    D = total_power
    # Gradient path uses the full ``fft2`` so the inverse-FFT mask write
    # is on a regular (nphi, ntheta) complex array. The value above used
    # ``rfft2`` for memory efficiency; the two are numerically identical
    # for real input.
    F_full = np.fft.fft2(modB)
    masked_F = F_full.copy()
    masked_F[0, :] = 0.0
    # Recompute the masked-spectrum power here (instead of reusing the
    # ``total_power - axisymmetric_power`` value from the rfft2 path) so
    # the numerator and gradient share the same ``fft2``-domain spectrum
    # — symmetry-rounding mismatches between the rfft2 and fft2 paths
    # would otherwise inject a ~1e-16 inconsistency into the chain rule.
    N = float(np.sum(np.abs(masked_F) ** 2))
    # Sanity: ``N / D`` here must equal the SSOT ``S_HEL`` from
    # :func:`_s_hel_from_modB`; the recomputation only happens because
    # the gradient adjoint needs the masked full-spectrum array.
    S_HEL = N / D
    grid_size = float(nphi * ntheta)
    dN_dmodB = 2.0 * grid_size * np.real(np.fft.ifft2(masked_F))
    dD_dmodB = 2.0 * grid_size * modB
    dS_dmodB = (dN_dmodB * D - N * dD_dmodB) / (D * D)
    return S_HEL, dS_dmodB


class HelicalFieldContentObjective:
    """Stage 2 ramped helical-content objective with simsopt-style J / dJ API.

    Sampled on a fixed working surface (the optimizer's ``new_surf``); the
    field is a BiotSavart object whose coils are the optimization DOFs.
    The class caches the FFT and gradient and is the SSOT for ``S_HEL``
    contributions to ``make_stage2_fun``.

    Cache invalidation (H2 fix): the cache is fingerprinted on
    ``field.x`` so a hot-loop caller that mutates the field's DOFs (the
    optimizer does this implicitly through ``BiotSavart`` and the
    surrogate's ``set_dofs``) will re-compute J/dJ on the next call
    without needing an explicit ``recompute_bell``. The fingerprint is
    the byte representation of the current ``field.x`` array (cheaper
    than a hash; ``np.array_equal`` is O(n) either way and the array is
    a few hundred floats for a realistic coil set). The explicit
    :meth:`recompute_bell` remains available for callers that want
    deterministic invalidation without inspecting the cache state.
    """

    def __init__(self, field, surface):
        self.field = field
        self.surface = surface
        self._value: float | None = None
        self._grad: np.ndarray | None = None
        self._x_fingerprint: bytes | None = None

    def _current_fingerprint(self) -> bytes:
        return np.asarray(self.field.x, dtype=float).tobytes()

    def recompute_bell(self) -> None:
        """Invalidate the J / dJ cache; the next call recomputes both."""
        self._value = None
        self._grad = None
        self._x_fingerprint = None

    def _ensure_computed(self) -> None:
        current_fingerprint = self._current_fingerprint()
        cache_hit = (
            self._value is not None
            and self._grad is not None
            and self._x_fingerprint is not None
            and self._x_fingerprint == current_fingerprint
        )
        if cache_hit:
            return
        modB, B_unit = _modB_and_B_unit_on_surface(self.field, self.surface)
        S_HEL, dS_dmodB = _S_HEL_value_and_dS_HEL_by_dmodB(modB)
        # Chain rule from |B|_{ij} = ||B_{ij}||:
        #   d|B|/dB_k = B_k / |B|
        # so dS/dB_{ij,k} = dS/d|B|_{ij} * B_unit_{ij,k}.
        dS_dB = dS_dmodB[..., None] * B_unit
        nphi, ntheta, _ = dS_dB.shape
        flat_dS_dB = dS_dB.reshape((nphi * ntheta, 3))
        # BiotSavart.B_vjp returns a Derivative; calling it on ``field``
        # yields a flat numpy array of the field's free coil DOFs.
        derivative = self.field.B_vjp(flat_dS_dB)
        self._value = float(S_HEL)
        self._grad = np.asarray(derivative(self.field), dtype=float)
        self._x_fingerprint = current_fingerprint

    def J(self) -> float:
        """Return scalar ``S_HEL`` in ``[0, 1]``."""
        self._ensure_computed()
        return self._value

    def dJ_by_dcoils(self) -> np.ndarray:
        """Return ``dS_HEL / d(coil DOFs)`` matching ``field.x`` ordering."""
        self._ensure_computed()
        return self._grad


def compute_S_HEL_gradient_relative_error(
    field,
    surface,
    *,
    perturbation: float = 1.0e-5,
    seed: int = 0,
) -> float:
    """Validate the analytic S_HEL gradient against centered finite differences.

    Samples up to four free coil DOFs (deterministic given ``seed``) of
    ``field``, perturbs each by ``± perturbation``, recomputes ``S_HEL``
    end-to-end at both points, and compares the centered-difference
    estimate to the analytic gradient component returned by
    :class:`HelicalFieldContentObjective`. Returns the maximum
    relative-error across the sampled components; a centered difference
    at ``eps = 1e-5`` carries an inherent ~1e-5 truncation floor so
    ``max_rel_error < 1e-3`` is the operational pass bar.
    """
    if perturbation <= 0.0:
        raise ValueError("perturbation must be positive")
    objective = HelicalFieldContentObjective(field, surface)
    objective.recompute_bell()
    analytic_grad = objective.dJ_by_dcoils()
    n_dofs = analytic_grad.size
    if n_dofs == 0:
        raise ValueError("field has no free DOFs to validate against")
    rng = np.random.default_rng(int(seed))
    n_sample = int(min(4, n_dofs))
    indices = rng.choice(n_dofs, size=n_sample, replace=False)
    x0 = np.asarray(field.x, dtype=float).copy()
    max_rel_error = 0.0
    for idx in indices:
        x_plus = x0.copy()
        x_plus[idx] += perturbation
        field.x = x_plus
        plus = HelicalFieldContentObjective(field, surface)
        s_plus = plus.J()
        x_minus = x0.copy()
        x_minus[idx] -= perturbation
        field.x = x_minus
        minus = HelicalFieldContentObjective(field, surface)
        s_minus = minus.J()
        fd_grad = (s_plus - s_minus) / (2.0 * perturbation)
        analytic = float(analytic_grad[idx])
        denom = max(abs(fd_grad), abs(analytic), 1.0e-12)
        rel_error = abs(fd_grad - analytic) / denom
        if rel_error > max_rel_error:
            max_rel_error = rel_error
    field.x = x0
    return float(max_rel_error)


# Backwards-compatibility aliases for the legacy private names. The
# canonical centroid-axis lives in :mod:`topology_bridge_shared` (SSOT
# for both Phase 3a and Phase 3b); these aliases keep any internal
# references resolved without forcing a rename in this module.
_SurfaceCentroidCoordinateAxis = SurfaceCentroidCoordinateAxis
_surface_centroid_coordinate_axis = build_surface_centroid_axis
