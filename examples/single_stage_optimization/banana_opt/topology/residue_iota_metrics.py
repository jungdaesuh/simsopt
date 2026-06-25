"""Greene-residue + iota-profile topology-certificate metrics.

`residue_iota_metrics(bs, *, axis_r, axis_z, nfp, radial_lo, radial_hi)` does two
things for a vacuum BiotSavart field, given a *provided* magnetic axis (it does
NOT locate the axis -- another agent owns that):

  (a) Samples the realized rotational-transform profile iota(r) over
      [radial_lo, radial_hi] using ``sample_iota_profile`` (the same full-torus
      poloidal-winding convention the Greene residue probe reduces), and reports
      the iota(r) array + radial labels + edge/min/max iota.

  (b) Probes the Greene residue at the low-order rational crossings the field
      actually realizes inside a physical iota band, building a convention-locked
      ``RationalTarget`` per crossing (radial_label + radial_window bracket +
      fourier_m=q, fourier_n=p, nfp) and running ``run_residue_probe``. It
      collects residue + O/X/parabolic classification + per-branch convergence
      ``branch_status``, and surfaces the probe's own ``branch_status_counts``
      (it does NOT swallow non-convergence in a try/except -- the probe reports
      it).

The certification verdict is reduced from the per-resonance residues by
``confinement_boundary_from_residues``, which classifies each island chain by its
ELLIPTIC O-point residue (0<R<1 intact / KAM-confined; R<=0 or R>=1 torn -- the
elliptic point has gone hyperbolic/parabolic). The chain's hyperbolic X-point is a
saddle (R<0) whether or not the island survives, so only the O-branch is the
confinement discriminator. The KAM confinement edge is the smallest in-band radial
label where a converged O-branch crosses from intact to destroyed: inside it the
core island chain is residue-confined, outside it the chain is torn -- the
quantitative, load-bearing boundary the cert reports.

Selection of rationals (why not "every crossing"):
``IotaProfile.all_rational_crossings(p, q)`` returns every integer branch
``n + p/q`` realized in the traced band. Near the magnetic axis the full-torus
winding carries an integer geometric/section offset (here iota climbs above 1
for labels below ~0.02), so a raw scan over q<=16 yields >150 crossings, almost
all of them artifacts of the steep near-axis gradient sweeping through the
integer-offset region. This module keeps only crossings whose *realized* branch
value ``crossing.iota`` lands inside ``iota_band`` -- the physical operating
window -- and dedupes by reduced fraction, exactly the in-band low-order load
that ``banana_opt.topology.iota_profile.band_shear_load`` already counts. The
result is a bounded, physical target list (the confined band of this design
rotates near 0.76-0.80 and carries a clean 7/9 island chain).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd

from .fieldline_map import (
    DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    FieldlineIntegratorOptions,
)
from .greene_residue import classify_greene_residue
from .iota_profile import (
    DEFAULT_IOTA_PROFILE_TORUS_TURNS,
    RationalCrossing,
    linear_radial_labels,
    sample_iota_profile,
)
from .periodic_orbit import (
    DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    PeriodicOrbitSolverOptions,
)
from .poincare_chart import PoincareChart
from .rational_target import GREENE_BRANCH_O, RationalTarget
from .residue_diagnostics import (
    DEFAULT_BRANCH_PHASE_ANGLES,
    run_residue_probe,
)


# How many midplane radial seeds to trace for iota(r). 24 mirrors the campaign's
# tracing.nfieldlines and resolves the rational ladder of the band.
DEFAULT_RADIAL_LABEL_COUNT = 24

# Largest rational denominator kept distinct when scanning for in-band crossings.
# Matches the WBA island scan ceiling (kam_birkhoff DEFAULT_WBA_RATIONAL_MAX
# _DENOMINATOR=24); 16 is the task's requested scan ceiling for this low-shear
# nontwist design where the realized resonances are high-q.
DEFAULT_MAX_DENOMINATOR = 16

# Residue classes the certification treats as an intact (KAM-confined, elliptic)
# island chain vs. a destroyed (hyperbolic / overlapped) one. An intact chain has
# 0<R<1; a destroyed chain has R<0 or R>1. Parabolic/period-doubling boundary
# cases (R==0, R==1) are neither, and are reported as the literal boundary class
# rather than coerced into "intact" or "destroyed".
RESIDUE_CLASS_INTACT = "intact_elliptic"
RESIDUE_CLASS_DESTROYED = "destroyed_hyperbolic"
RESIDUE_CLASS_BOUNDARY = "marginal_boundary"


def _in_band_low_order_crossings(
    profile,
    *,
    iota_band: tuple[float, float],
    max_denominator: int,
) -> tuple[RationalCrossing, ...]:
    """Every realized reduced-p/q crossing whose branch iota lies in ``iota_band``.

    Enumerates reduced proper fractions p/q (2 <= q <= max_denominator, 1 <= p < q)
    -- the low-order ladder -- and, for each, takes every realized integer-branch
    crossing from ``all_rational_crossings`` keeping only those whose realized
    ``crossing.iota`` falls inside the band. This is the same in-band low-order
    filter ``band_shear_load`` applies, so near-axis ``n+p/q`` branches outside the
    physical window are dropped rather than probed. Sorted by radial label.
    """

    band_lo, band_hi = float(iota_band[0]), float(iota_band[1])
    crossings: list[RationalCrossing] = []
    for q in range(2, int(max_denominator) + 1):
        for p in range(1, q):
            if gcd(p, q) != 1:
                continue
            for crossing in profile.all_rational_crossings(p, q):
                if band_lo <= float(crossing.iota) <= band_hi:
                    crossings.append(crossing)
    return tuple(sorted(crossings, key=lambda c: c.radial_label))


def _target_from_crossing(
    crossing: RationalCrossing,
    *,
    nfp: int,
) -> RationalTarget:
    """Convention-locked RationalTarget on a realized crossing.

    Sets ``radial_label`` to the (interpolated) crossing label and
    ``radial_window`` to the enclosing traced bracket (a data-derived window
    guaranteed to contain the resonance), and locks the Fourier metadata
    ``fourier_m=q, fourier_n=p`` -- which RationalTarget validates against
    ``m*(p/q) - n = 0`` (rational_target.py:104) -- with the field-period count.
    """

    return RationalTarget(
        p=int(crossing.p),
        q=int(crossing.q),
        radial_label=float(crossing.radial_label),
        radial_window=(
            float(crossing.bracket_lower_label),
            float(crossing.bracket_upper_label),
        ),
        fourier_m=int(crossing.q),
        fourier_n=int(crossing.p),
        nfp=int(nfp),
    )


def _island_confinement_class(o_point_residue: float) -> str:
    """Classify an island chain's confinement from its elliptic O-point residue.

    The KAM confinement discriminator for a p/q island chain is the Greene residue
    of its *elliptic O-point*: 0<R<1 means the island is intact (a confining,
    KAM-like elliptic chain), R<=0 or R>=1 means the elliptic point has gone
    hyperbolic/parabolic, i.e. the chain is torn. This defers to the module-level
    ``classify_greene_residue`` (the SSOT for the 0<R<1 window) and maps its
    elliptic verdict -> intact, hyperbolic -> destroyed, and the marginal parabolic
    / period-doubling boundary cases -> ``RESIDUE_CLASS_BOUNDARY`` (neither confined
    nor torn -- surfaced as the literal marginal case rather than coerced).

    The hyperbolic X-point of an island is ALWAYS a saddle (R<0) whether the island
    is intact or destroyed, so the X-branch is never the confinement discriminator;
    only the O-branch residue enters here.
    """

    classification = classify_greene_residue(float(o_point_residue))
    if classification == "elliptic_o_point":
        return RESIDUE_CLASS_INTACT
    if classification == "hyperbolic_x_point":
        return RESIDUE_CLASS_DESTROYED
    return RESIDUE_CLASS_BOUNDARY


def confinement_boundary_from_residues(residues: Sequence[dict]) -> dict:
    """Locate the KAM confinement edge from the per-branch residue diagnostics.

    ``residues`` is the ``aggregate["residues"]`` list emitted by
    ``residue_iota_metrics``: each entry is ONE branch (O or X) of one resonance,
    carrying ``branch``, a realized ``residue`` (or None when the branch did not
    converge), a ``radial_label``, and p/q. An island chain's confinement is set by
    its ELLIPTIC O-point residue (0<R<1 intact, otherwise torn), so this reduces to
    the converged **O-branches** only: the X-branch is the island's saddle (always
    hyperbolic) and would otherwise spuriously read every intact island as
    "destroyed" at its own radius. The boundary is the smallest radial label where a
    converged O-branch transitions from intact to destroyed as the label increases
    -- the radius where the last KAM-confining island chain is torn.

    A resonance whose O-branch did not converge has no elliptic-point residue to
    classify, so it does not vote (fail-closed): it cannot certify confinement or
    place the edge.

    Returns
    -------
    dict
        ``confinement_boundary_radial_label``: label of the destroyed (outer) island
        at the first intact->destroyed O-point transition, or None when there is no
        such transition (all-intact, all-destroyed, or fewer than two converged
        O-branches).
        ``inner_resonance`` / ``outer_resonance``: the bracketing
        (p, q, residue, residue_class, radial_label) at that transition, or None.
        ``core_residue_class`` / ``edge_residue_class``: the class of the innermost /
        outermost converged O-branch (None when none converged).
        ``confined_core``: True iff at least one O-branch converged and the innermost
        one is intact -- i.e. the core island chain is residue-confined.
        ``n_converged_resonances``: how many converged O-branches voted.
    """

    o_points = [
        {
            "p": int(entry["p"]),
            "q": int(entry["q"]),
            "residue": float(entry["residue"]),
            "residue_class": _island_confinement_class(float(entry["residue"])),
            "radial_label": float(entry["radial_label"]),
        }
        for entry in residues
        if entry.get("branch") == GREENE_BRANCH_O
        and entry.get("converged") is True
        and entry.get("residue") is not None
        and entry.get("radial_label") is not None
    ]
    o_points.sort(key=lambda item: item["radial_label"])

    core_class = o_points[0]["residue_class"] if o_points else None
    edge_class = o_points[-1]["residue_class"] if o_points else None
    confined_core = bool(o_points) and core_class == RESIDUE_CLASS_INTACT

    boundary_label: float | None = None
    inner_resonance: dict | None = None
    outer_resonance: dict | None = None
    for inner, outer in zip(o_points, o_points[1:]):
        if (
            inner["residue_class"] == RESIDUE_CLASS_INTACT
            and outer["residue_class"] == RESIDUE_CLASS_DESTROYED
        ):
            boundary_label = outer["radial_label"]
            inner_resonance = inner
            outer_resonance = outer
            break

    return {
        "confinement_boundary_radial_label": boundary_label,
        "inner_resonance": inner_resonance,
        "outer_resonance": outer_resonance,
        "core_residue_class": core_class,
        "edge_residue_class": edge_class,
        "confined_core": confined_core,
        "n_converged_resonances": len(o_points),
    }


def residue_iota_metrics(
    bs,
    *,
    axis_r: float,
    axis_z: float,
    nfp: int,
    radial_lo: float,
    radial_hi: float,
    radial_label_count: int = DEFAULT_RADIAL_LABEL_COUNT,
    max_denominator: int = DEFAULT_MAX_DENOMINATOR,
    iota_band: tuple[float, float] | None = None,
    toroidal_turns: int = DEFAULT_IOTA_PROFILE_TORUS_TURNS,
    poloidal_orientation: int = 1,
    radial_label_scale: float = 1.0,
    iota_integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    phase_angles: Sequence[float] = DEFAULT_BRANCH_PHASE_ANGLES,
    max_targets: int | None = None,
) -> dict:
    """Sample iota(r) and probe Greene residue at realized low-order rationals.

    Parameters
    ----------
    bs
        A differentiable magnetic field (BiotSavart) to trace. The magnetic axis
        is taken as given via ``axis_r``/``axis_z``; this function does NOT locate
        it.
    axis_r, axis_z
        Provided magnetic-axis (R, Z) on the phi=0 section. The Poincare chart and
        every iota seed are centred here.
    nfp
        Field-period count, stamped onto every ``RationalTarget``.
    radial_lo, radial_hi
        Inclusive midplane radial-label band for the iota seeds (label = distance
        from the axis in chart units). ``0 < radial_lo < radial_hi``; pick a small
        inset to ~minor radius from the surface extent.
    radial_label_count
        Number of evenly spaced iota seeds over (radial_lo, radial_hi].
    max_denominator
        Largest rational denominator scanned for in-band crossings.
    iota_band
        Physical ``(iota_lo, iota_hi)`` window; only crossings whose realized
        branch iota lands here become probe targets (drops the near-axis
        integer-offset region). Default: the realized iota band restricted to
        ``iota <= 1`` (i.e. ``(min(iota), min(1, max(iota)))``), which strips the
        near-axis branch where the chart winding exceeds one full poloidal turn.
    toroidal_turns
        Full toroidal turns traced per iota seed.
    poloidal_orientation, radial_label_scale
        Poincare chart geometry, shared by the iota sampler and the residue probe.
    iota_integrator_options
        Field-line integrator options for the multi-turn iota(r) sampling. Kept
        separate from ``integrator_options`` because the iota trace spans tens of
        toroidal turns; a fail-fast RHS budget tuned for the few-turn residue probe
        would abort every iota seed. Defaults to the unbudgeted module default.
    integrator_options, solver_options, phase_angles
        Passed through to ``run_residue_probe`` (the per-resonance probe). Use a
        ``max_rhs_evaluations`` budget here to gate non-converging high-q probes as
        ``integration_failed`` rather than hanging.
    max_targets
        Optional cap on the number of (innermost-first) targets probed, so a
        dense band cannot blow up runtime. ``None`` = probe all in-band crossings.

    Returns
    -------
    dict
        ``iota_profile``: axis, labels, iota array, valid/failed counts, and
        edge/min/max iota.
        ``rational_crossings``: every selected in-band crossing (p, q, realized
        iota, label, bracket, local shear).
        ``residue_probe``: the raw ``run_residue_probe`` payload (chart, targets,
        diagnostics, ``branch_status_counts``) -- or ``None`` when no crossing was
        selected.
        ``aggregate``: resonances probed, branch diagnostics emitted, how many
        converged, and the residue value + classification + branch_status per
        emitted branch diagnostic.
        ``confinement_boundary``: the KAM confinement edge reduced from
        ``aggregate["residues"]`` by ``confinement_boundary_from_residues`` -- the
        smallest radial label where an intact island chain gives way to a
        destroyed one, plus the bracketing resonances and core/edge classes.
    """

    radial_labels = linear_radial_labels(
        lower=float(radial_lo),
        upper=float(radial_hi),
        count=int(radial_label_count),
    )
    # The iota sampler integrates ``toroidal_turns`` (tens of) full turns per
    # seed -- far more than a per-resonance probe -- so it gets its own
    # integrator options. A fail-fast ``max_rhs_evaluations`` sized for a few-turn
    # residue probe would abort every multi-turn iota trace (every seed failing),
    # which is why this is NOT shared with ``integrator_options``.
    profile = sample_iota_profile(
        bs,
        axis_r=float(axis_r),
        axis_z=float(axis_z),
        radial_labels=radial_labels,
        toroidal_turns=int(toroidal_turns),
        poloidal_orientation=int(poloidal_orientation),
        radial_label_scale=float(radial_label_scale),
        integrator_options=iota_integrator_options,
    )

    bounds = profile.iota_bounds()
    valid_samples = profile.valid_samples()
    iota_profile_summary: dict = {
        "axis_r": float(profile.axis_r),
        "axis_z": float(profile.axis_z),
        "radial_lo": float(radial_lo),
        "radial_hi": float(radial_hi),
        "toroidal_turns": int(toroidal_turns),
        "n_labels": len(profile.samples),
        "n_valid": len(valid_samples),
        "n_failed": len(profile.samples) - len(valid_samples),
        "radial_labels": [float(s.radial_label) for s in profile.samples],
        "iota": [None if s.iota is None else float(s.iota) for s in profile.samples],
        "iota_min": None if bounds is None else float(bounds[0]),
        "iota_max": None if bounds is None else float(bounds[1]),
        # Edge iota = realized iota at the largest valid radial label (the
        # outermost traced surface), which is what a confinement/edge metric
        # quotes. None if no seed wound successfully.
        "iota_edge": (
            None
            if len(valid_samples) == 0
            else float(
                max(valid_samples, key=lambda s: s.radial_label).iota
            )
        ),
    }

    if bounds is None:
        resolved_band = None
        selected: tuple[RationalCrossing, ...] = ()
    else:
        if iota_band is None:
            # Strip the near-axis branch where full-torus winding exceeds one
            # poloidal turn (the integer chart offset), keeping the physical band.
            resolved_band = (float(bounds[0]), float(min(1.0, bounds[1])))
        else:
            resolved_band = (float(iota_band[0]), float(iota_band[1]))
        selected = _in_band_low_order_crossings(
            profile,
            iota_band=resolved_band,
            max_denominator=int(max_denominator),
        )
        if max_targets is not None:
            selected = selected[: int(max_targets)]

    crossings_payload = [
        {
            "p": int(c.p),
            "q": int(c.q),
            "iota": float(c.iota),
            "radial_label": float(c.radial_label),
            "bracket_lower_label": float(c.bracket_lower_label),
            "bracket_upper_label": float(c.bracket_upper_label),
            "local_shear": float(c.local_shear),
        }
        for c in selected
    ]

    targets = [_target_from_crossing(c, nfp=int(nfp)) for c in selected]

    if len(targets) == 0:
        return {
            "iota_profile": iota_profile_summary,
            "iota_band_probed": resolved_band,
            "rational_crossings": crossings_payload,
            "residue_probe": None,
            "aggregate": {
                "n_resonances_probed": 0,
                "n_branch_diagnostics": 0,
                "n_converged": 0,
                "branch_status_counts": {},
                "residues": [],
            },
            "confinement_boundary": confinement_boundary_from_residues([]),
        }

    probe = run_residue_probe(
        bs,
        targets=targets,
        chart=PoincareChart(
            axis_r=float(axis_r),
            axis_z=float(axis_z),
            poloidal_orientation=int(poloidal_orientation),
            radial_label_scale=float(radial_label_scale),
        ),
        integrator_options=integrator_options,
        solver_options=solver_options,
        phase_angles=phase_angles,
    )

    diagnostics = probe["diagnostics"]
    branch_status_counts = dict(probe["branch_status_counts"])
    n_converged = sum(
        1 for d in diagnostics if d.get("converged") is True
    )
    residues = [
        {
            "p": int(d["target"]["p"]),
            "q": int(d["target"]["q"]),
            "branch": d["branch"],
            "branch_status": d["branch_status"],
            "converged": bool(d["converged"]),
            "residue": d.get("residue"),
            "residue_classification": d.get("residue_classification"),
            "traceM": d.get("traceM"),
            "detM": d.get("detM"),
            "winding": d.get("winding"),
            "radial_label": d.get("radial_label"),
        }
        for d in diagnostics
    ]

    return {
        "iota_profile": iota_profile_summary,
        "iota_band_probed": resolved_band,
        "rational_crossings": crossings_payload,
        "residue_probe": probe,
        "aggregate": {
            "n_resonances_probed": len(targets),
            "n_branch_diagnostics": len(diagnostics),
            "n_converged": int(n_converged),
            "branch_status_counts": branch_status_counts,
            "residues": residues,
        },
        "confinement_boundary": confinement_boundary_from_residues(residues),
    }
