"""Phase 3a non-Boozer field-line iota proxy (post-solve diagnostic only).

This module implements the read-only field-line rotational-transform proxy
described in the Stage 2 / single-stage Boozer handoff plan (Phase 3,
lines 257-319 of
``autoresearch/docs/stage2_single_stage_boozer_handoff_impl_plan_2026-05-15.md``).

The proxy is a *diagnostic*, not an objective. Plan line 270 mandates
"evaluate post-solve or on checkpoint only, not every optimizer iteration";
plan line 283 mandates "keep field-line iota as validation/truth signal,
not the per-iteration objective". The companion helical-content objective
``S_HEL`` lives in :mod:`banana_opt.boozer_topology_bridge` (Part B).

Failure semantics (plan line 306): "Failed diagnostics fail closed as
unavailable metrics, not as fake zero-quality success". A trace that
escapes the cage, returns too few transits, or otherwise cannot define a
rotational transform yields ``valid=False`` and ``iota_proxy=None``.
Callers persisting the proxy must serialize ``None`` rather than a
fabricated ``0.0``.

The convergence sub-study lives in
``autoresearch/scripts/fieldline_iota_proxy_convergence.py``; the plan
requires ``|iota_FL(N=200) - iota_FL(N=100)| < 0.005`` before the proxy
is trusted (plan line 274).

Phase 3a vs Phase 3b semantics: ``FIELDLINE_IOTA_PROXY_VALID`` emitted
from this module carries the *live diagnostic* semantics — at least one
seed cleared ``n_transits_target`` toroidal transits and the iota mean
was computable. The convergence-tolerance semantics (``|iota_FL(2N) -
iota_FL(N)| < 0.005``) live in the standalone convergence sub-study; the
artifact pipeline declares Phase 3a the sole producer of the field-line
proxy keys so the "valid" semantics is unambiguous downstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simsopt.field import (
    MaxRStoppingCriterion,
    MaxZStoppingCriterion,
    MinRStoppingCriterion,
    MinZStoppingCriterion,
)
from simsopt.field.tracing import (
    compute_fieldlines,
    compute_poloidal_transits,
    compute_toroidal_transits,
)
from topology_scorer import midplane_seed_radii

from banana_opt.topology_bridge_shared import (
    DEFAULT_NFIELDLINES,
    DEFAULT_TMAX,
    DEFAULT_TOL,
    SurfaceCentroidCoordinateAxis,
    build_surface_centroid_axis,
)


__all__ = [
    "DEFAULT_NFIELDLINES",
    "DEFAULT_TMAX",
    "DEFAULT_TOL",
    "FAILURE_ESCAPED",
    "FAILURE_INSUFFICIENT_TRANSITS",
    "FAILURE_NO_HISTORY",
    "FAILURE_TRACER_EXCEPTION",
    "Phase3aFieldlineIotaProxyResult",
    "compute_fieldline_iota_proxy",
    "fieldline_iota_proxy_artifact_fields",
    "safe_compute_fieldline_iota_proxy",
]


FAILURE_ESCAPED = "all_lines_escaped"
FAILURE_INSUFFICIENT_TRANSITS = "insufficient_transits"
FAILURE_NO_HISTORY = "tracer_returned_no_history"
FAILURE_TRACER_EXCEPTION = "tracer_exception"


@dataclass(frozen=True)
class Phase3aFieldlineIotaProxyResult:
    """Outcome of a single field-line iota-proxy computation.

    ``iota_proxy`` is the mean of ``poloidal_transits / toroidal_transits``
    across lines that achieved at least ``n_transits_target`` toroidal
    transits. The mean is reported as a *float* only when
    ``valid is True``; otherwise the field is ``None`` so downstream
    consumers see "diagnostic unavailable" instead of a fabricated 0.

    Plan line 306 ("Failed diagnostics fail closed as unavailable metrics,
    not as fake zero-quality success") is enforced by the dataclass:
    ``iota_proxy is None`` implies ``valid is False``, and the failure
    cause is reported in ``reason`` for downstream rankers.

    Fields:
        iota_proxy: Mean rotational-transform proxy (poloidal /
            toroidal transits) across trusted lines. ``None`` on failure.
        valid: ``True`` iff at least one line cleared the transit
            threshold and a usable iota mean was computed.
        n_surviving: Lines that reached the ``tmax`` cap without
            tripping the escape cage. Disjoint from ``escape_count``;
            ``n_surviving + escape_count == nfieldlines``.
        n_transits_used: Toroidal-transit threshold actually applied;
            ``0`` when the proxy failed before the threshold gate.
        escape_count: Lines that tripped the radius / Z box guard
            (plan line 272 escape-radius cage) OR returned no phi crossings.
            See :func:`_line_was_lost`.
        reason: One of the ``FAILURE_*`` tags when ``valid=False``;
            ``None`` on success.
    """

    iota_proxy: float | None
    valid: bool
    n_surviving: int
    n_transits_used: int
    escape_count: int
    reason: str | None


def _build_escape_cage(
    surface,
    *,
    escape_radius: float,
) -> list:
    """Construct the escape-radius stopping criteria.

    Plan line 272 calls for "an escape-radius stopping criterion for
    wandering non-bootable fields". SIMSOPT does not ship a single
    "escape sphere" criterion; the operationally equivalent guard is a
    four-sided cylindrical box at ``r = escape_radius`` and
    ``|z| = escape_radius``. We bound R below by the surface inner
    extent (with a small safety margin) so a line that diverges through
    the magnetic axis still trips a stop rather than going singular.

    The escape cage is *additive* to whatever the field's native
    surface-exit classifier reports — wandering field lines that never
    cross the original surface still get caught by the radius bound.
    """
    if escape_radius <= 0.0:
        raise ValueError(
            f"escape_radius must be positive (meters); got {escape_radius!r}"
        )
    gamma = np.asarray(surface.gamma(), dtype=float)
    rr = np.sqrt(gamma[..., 0] ** 2 + gamma[..., 1] ** 2)
    rmin = float(np.min(rr))
    # Sit the inner-R guard just below the actual surface inner radius
    # to avoid spurious early stops at the first integration step.
    inner_r_bound = max(0.0, rmin * 0.5)
    return [
        MaxRStoppingCriterion(float(escape_radius)),
        MinRStoppingCriterion(inner_r_bound),
        MaxZStoppingCriterion(float(escape_radius)),
        MinZStoppingCriterion(-float(escape_radius)),
    ]


def _line_was_lost(phi_hits: np.ndarray) -> bool:
    """Return ``True`` iff the trace did not survive cleanly to ``tmax``.

    SIMSOPT marks stopping-criterion hits with ``idx < 0`` in the
    ``phi_hits`` array. A line that completed ``tmax`` without tripping
    any stop ends with ``idx >= 0``. A line whose hit array is empty
    crossed no phi plane at all — its trace fate is unknown; treat it as
    lost so it does not silently inflate ``n_surviving`` (the proxy
    pipeline already drops it from the iota mean because no transit can
    be counted).
    """
    hits = np.asarray(phi_hits)
    if hits.size == 0 or hits.ndim < 2:
        # No phi crossings: trace fate unknown; conservatively treat as
        # lost so the survival bookkeeping is monotone with the iota mean.
        return True
    return bool(hits[-1, 1] < 0)


def compute_fieldline_iota_proxy(
    field,
    surface,
    *,
    nfieldlines: int = DEFAULT_NFIELDLINES,
    tmax: float,
    tol: float = DEFAULT_TOL,
    escape_radius: float,
    n_transits_target: int,
) -> Phase3aFieldlineIotaProxyResult:
    """Compute the field-line iota proxy from real Biot-Savart tracing.

    Seeds ``nfieldlines`` midplane radii via
    :func:`topology_scorer.midplane_seed_radii`, integrates the field
    lines with :func:`simsopt.field.tracing.compute_fieldlines` (capped
    at ``tmax`` with ODE tolerance ``tol``), and computes the
    rotational-transform proxy as the mean of
    ``compute_poloidal_transits / compute_toroidal_transits`` across the
    surviving lines that achieved at least ``n_transits_target``
    toroidal transits.

    Failure semantics (plan line 306):

    - Tracer returned zero history rows for every line:
      ``reason = FAILURE_NO_HISTORY``.
    - Every line tripped the escape cage / returned no crossings:
      ``reason = FAILURE_ESCAPED``, ``valid = False``,
      ``iota_proxy = None``.
    - Some lines survived but none accumulated ``n_transits_target``
      toroidal transits: ``reason = FAILURE_INSUFFICIENT_TRANSITS``,
      ``valid = False``, ``iota_proxy = None`` and the partial
      ``n_surviving`` count is still reported for diagnostics.

    The function does NOT swallow exceptions raised by the tracer; the
    caller in the Stage 2 artifact pipeline uses
    :func:`safe_compute_fieldline_iota_proxy` for that (per the
    no-defensive-try/except project guideline — that wrapper is the only
    swallow point). Parameter validation raises :class:`ValueError`.

    Args:
        field: A configured ``simsopt.field.MagneticField`` (typically a
            ``BiotSavart`` already wrapped in an ``InterpolatedField``).
        surface: Plasma boundary surface used both for midplane seeding
            and to scale the escape cage. Must expose ``.gamma()`` and
            ``.cross_section(phi, thetas=...)``.
        nfieldlines: Number of midplane seed radii (plan default 5).
        tmax: Hard cap on integration time (plan line 271). Must be > 0.
        tol: Adaptive ODE tolerance (plan line 273 default ``1e-8``).
        escape_radius: Maximum permitted cylindrical R (meters) before
            the line is declared lost (plan line 272). Required.
        n_transits_target: Minimum number of toroidal transits a line
            must complete before its iota contribution is trusted. Lines
            with fewer transits are dropped from the mean.
    """
    if nfieldlines <= 0:
        raise ValueError(f"nfieldlines must be positive; got {nfieldlines}")
    if tmax <= 0.0:
        raise ValueError(f"tmax must be positive; got {tmax}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive; got {tol}")
    if n_transits_target <= 0:
        raise ValueError(
            f"n_transits_target must be positive; got {n_transits_target}"
        )

    radii = np.asarray(midplane_seed_radii(surface, int(nfieldlines)), dtype=float)
    R0 = radii
    Z0 = np.zeros_like(radii)
    stopping_criteria = _build_escape_cage(
        surface, escape_radius=float(escape_radius)
    )
    histories, phi_hits = compute_fieldlines(
        field,
        R0,
        Z0,
        tmax=float(tmax),
        tol=float(tol),
        phis=[0.0],
        stopping_criteria=stopping_criteria,
    )

    if len(histories) == 0:
        return Phase3aFieldlineIotaProxyResult(
            iota_proxy=None,
            valid=False,
            n_surviving=0,
            n_transits_used=0,
            escape_count=0,
            reason=FAILURE_NO_HISTORY,
        )

    # ``n_surviving`` and ``escape_count`` are mutually exclusive: a line
    # either reached the tmax cap (survived) or tripped the escape cage
    # / box guard (escaped). They sum to ``len(histories)`` by construction
    # (which equals ``nfieldlines`` for a healthy tracer).
    escape_count = int(sum(_line_was_lost(hits) for hits in phi_hits))
    n_surviving = int(len(histories) - escape_count)

    coordinate_axis = build_surface_centroid_axis(surface)
    toroidal = compute_toroidal_transits(histories, flux=False)
    poloidal = compute_poloidal_transits(histories, coordinate_axis, flux=False)
    toroidal = np.asarray(toroidal, dtype=float)
    poloidal = np.asarray(poloidal, dtype=float)

    trusted_iotas: list[float] = []
    for ntor, npol in zip(toroidal, poloidal):
        if not np.isfinite(ntor) or not np.isfinite(npol):
            continue
        if abs(ntor) < float(n_transits_target):
            continue
        trusted_iotas.append(float(npol) / float(ntor))

    # Guard ordering matters for failure-tag reachability:
    #   1. FAILURE_NO_HISTORY  — already returned above (empty histories).
    #   2. FAILURE_ESCAPED     — every seed tripped the cage or returned
    #                            no crossings; no usable trusted iotas.
    #   3. FAILURE_INSUFFICIENT_TRANSITS — some seeds survived but none
    #                            cleared ``n_transits_target``.
    #   4. success             — iota mean over the trusted population.
    if n_surviving == 0 and not trusted_iotas:
        return Phase3aFieldlineIotaProxyResult(
            iota_proxy=None,
            valid=False,
            n_surviving=0,
            n_transits_used=0,
            escape_count=escape_count,
            reason=FAILURE_ESCAPED,
        )
    if not trusted_iotas:
        return Phase3aFieldlineIotaProxyResult(
            iota_proxy=None,
            valid=False,
            n_surviving=n_surviving,
            n_transits_used=0,
            escape_count=escape_count,
            reason=FAILURE_INSUFFICIENT_TRANSITS,
        )

    iota_array = np.asarray(trusted_iotas, dtype=float)
    return Phase3aFieldlineIotaProxyResult(
        iota_proxy=float(np.mean(iota_array)),
        valid=True,
        n_surviving=n_surviving,
        n_transits_used=int(n_transits_target),
        escape_count=escape_count,
        reason=None,
    )


def safe_compute_fieldline_iota_proxy(
    field,
    surface,
    *,
    nfieldlines: int = DEFAULT_NFIELDLINES,
    tmax: float,
    tol: float = DEFAULT_TOL,
    escape_radius: float,
    n_transits_target: int,
) -> Phase3aFieldlineIotaProxyResult:
    """Artifact-pipeline wrapper that fails closed on tracer exceptions.

    Per the no-defensive-try/except project guideline this is the *only*
    place in the Phase 3a code path that swallows tracer exceptions. It
    exists because the artifact pipeline must fail closed (plan line
    305 / 306) when ``simsopt.field.tracing`` raises on a numerically
    pathological field rather than crash the parent solver after the
    optimization has already completed.

    Narrow contract: ``ValueError`` (parameter validation in
    :func:`compute_fieldline_iota_proxy`) and ``RuntimeError`` (simsoptpp
    numerical-breakdown failures) are caught and converted to a
    :class:`Phase3aFieldlineIotaProxyResult` with ``valid=False``,
    ``iota_proxy=None``, and ``reason=FAILURE_TRACER_EXCEPTION``. Every
    other exception propagates so genuine bugs are not masked.

    The wrapper returns a fully-populated dataclass rather than ``None``
    so :func:`fieldline_iota_proxy_artifact_fields` can serialize the
    failure tag consistently; downstream donor-ranking tooling reads
    ``reason`` to triage why the proxy was unavailable.
    """
    try:
        return compute_fieldline_iota_proxy(
            field,
            surface,
            nfieldlines=nfieldlines,
            tmax=tmax,
            tol=tol,
            escape_radius=escape_radius,
            n_transits_target=n_transits_target,
        )
    except (ValueError, RuntimeError):
        return Phase3aFieldlineIotaProxyResult(
            iota_proxy=None,
            valid=False,
            n_surviving=0,
            n_transits_used=0,
            escape_count=0,
            reason=FAILURE_TRACER_EXCEPTION,
        )


def fieldline_iota_proxy_artifact_fields(
    result: Phase3aFieldlineIotaProxyResult | None,
) -> dict[str, object]:
    """Return the artifact-key payload for a single proxy invocation.

    Always returns the four contract keys so the Stage 2 artifact schema
    is stable across lanes (the field-line proxy is opt-in and may be
    skipped when the bridge diagnostic is disabled).

    Plan-mandated key shape (lines 287-288, extended with the convergence
    metadata required by plan line 274):

    - ``FIELDLINE_IOTA_PROXY``: float when the trace converged and was
      trusted; ``None`` otherwise. Persisting ``None`` (rather than 0.0)
      is required by plan line 306.
    - ``FIELDLINE_IOTA_PROXY_VALID``: explicit bool so consumers do not
      have to interpret ``None``.
    - ``FIELDLINE_IOTA_PROXY_N_TRANSITS``: integer toroidal-transit
      threshold used to admit a line, or ``None`` when the proxy was
      not invoked or no line cleared the threshold. Persisted to the
      registry via the ``fieldline_iota_proxy_n_transits`` column added
      in autoresearch/registry/migrations/031_fieldline_proxy_diagnostics.sql
      and surfaced through the canonical ``v_full_run`` view.
    - ``FIELDLINE_IOTA_PROXY_REASON``: one of the ``FAILURE_*`` tags
      (``FAILURE_ESCAPED``, ``FAILURE_INSUFFICIENT_TRANSITS``,
      ``FAILURE_NO_HISTORY``, ``FAILURE_TRACER_EXCEPTION``) or ``None``
      on success. Persisted to the registry via the
      ``fieldline_iota_proxy_reason`` column added in migration 031 so
      donor-ranking tooling (``autoresearch/scripts/build_donor_topology_ranking.py``
      and downstream report queries) can triage why the proxy was
      unavailable without re-tracing.
    """
    if result is None:
        return {
            "FIELDLINE_IOTA_PROXY": None,
            "FIELDLINE_IOTA_PROXY_VALID": None,
            "FIELDLINE_IOTA_PROXY_N_TRANSITS": None,
            "FIELDLINE_IOTA_PROXY_REASON": None,
        }
    iota_value = (
        None if result.iota_proxy is None else float(result.iota_proxy)
    )
    n_transits = (
        int(result.n_transits_used) if result.n_transits_used > 0 else None
    )
    return {
        "FIELDLINE_IOTA_PROXY": iota_value,
        "FIELDLINE_IOTA_PROXY_VALID": bool(result.valid),
        "FIELDLINE_IOTA_PROXY_N_TRANSITS": n_transits,
        "FIELDLINE_IOTA_PROXY_REASON": result.reason,
    }


# Backwards-compatibility alias so any internal callers of the private
# helper (tests, etc.) still resolve to the shared module without
# duplicating the centroid-axis class. The leading underscore signals
# "internal"; consumers should import from :mod:`topology_bridge_shared`.
# L12: the byte-identical ``_surface_centroid_axis`` factory alias was
# previously exposed alongside the class, but the entire codebase
# imports ``build_surface_centroid_axis`` from
# :mod:`topology_bridge_shared` directly; no consumer ever resolved the
# alias here. Removed to keep SSOT to one symbol.
_SurfaceCentroidCoordinateAxis = SurfaceCentroidCoordinateAxis
