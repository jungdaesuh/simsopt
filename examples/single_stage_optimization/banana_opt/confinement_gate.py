"""Decisive confinement acceptance gate for banana-coil candidates (Phase 5 / T5.1).

The optimizer ships several confinement *steering* proxies (NonQuasiSymmetricRatio,
IotaShearShortfall, MagneticWellVolumeShortfall, MinLGradBShortfall, the resonant-flux
penalty) plus a coarse in-loop field-line survival filter and an OFFLINE strict topology
ladder (``topology_fidelity_ladder``). None of them is a decisive accept/reject gate, so a
geometrically-buildable candidate can still ship with broken confinement (the 0/50 strict
Poincare wall).

This module promotes those proxies into ONE validated physics gate: a pure evaluator that
takes already-computed metrics for a converged candidate and returns a structured verdict.
It is intentionally pure (no simsopt/field dependency) so it is cheap to unit-test and can
be reused by both the single-stage solver and the campaign judge; the caller is responsible
for computing the metrics from the existing evaluators (NonQuasiSymmetricRatio.J(),
MagneticWellVolumeShortfall.magnetic_well_proxy(), the iota spread, the STRICT topology
status, the equilibrium beta / DMerc).

Design contract:
  * Topology is the DECISIVE criterion (the plan: "gate payoff on strict topology"). It must
    be present and pass, else the candidate is rejected -- a confinement gate fails CLOSED
    when there is no topology evidence (an integrity boundary, never defined away).
  * QA / magnetic-well / shear are computed, thresholded, and recorded. They are advisory by
    default (their banana-context thresholds are uncalibrated; see the audit) and become hard
    accept/reject only when the config marks them required.
  * Mercier is APPLICABLE only at finite beta. The campaign device is vacuum (beta=0), where
    the Mercier criterion is identically zero and physically meaningless, so it is reported
    not-applicable rather than passed; at finite beta the caller supplies ``mercier_dmerc_min``
    (>= 0 is stable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Criterion identifiers (stable strings recorded in results.json / the judge ledger).
CRITERION_TOPOLOGY = "topology"
CRITERION_ISLANDS = "island_chains"
CRITERION_CLASSIFIABILITY = "classifiability"
CRITERION_QA = "quasisymmetry"
CRITERION_MAGNETIC_WELL = "magnetic_well"
CRITERION_SHEAR = "iota_shear"
CRITERION_MERCIER = "mercier"


@dataclass(frozen=True)
class ConfinementMetrics:
    """Already-computed confinement metrics for ONE converged candidate.

    All physics fields are Optional: ``None`` means "not evaluated for this candidate"
    (e.g. magnetic well needs >=3 Boozer surfaces, shear needs >=2), which the gate treats
    as not-applicable rather than a failure. ``beta`` selects the Mercier branch.

    Fields:
      survived_lines / total_lines: strict field-line survival count (the topology ground
        truth, e.g. 47 of 50 at 50 lines / tmax 7000). ``total_lines <= 0`` or
        ``survived_lines is None`` => no topology evidence => fail-closed.
      island_chains: number of island chains found by the strict WBA tier (0 is ideal);
        None when the WBA island classifier was not run.
      classifiable_fraction: fraction of SURVIVING lines the WBA tier could reliably classify
        (>= floor is ideal; a low value means the topology verdict is unreliable); None when
        the classifier was not run.
      qa_nonqs_ratio: NonQuasiSymmetricRatio.J() value on the outer Boozer surface
        (lower = closer to quasi-symmetric); None when not evaluated.
      magnetic_well: the normalized Boozer-volume well proxy W (negative favorable);
        None when <3 surfaces.
      iota_shear: |iota_edge - iota_axis| (larger escapes the rational soup); None when
        <2 surfaces.
      beta: equilibrium beta. 0.0 => vacuum => Mercier not applicable.
      mercier_dmerc_min: min DMerc over the profile (>=0 stable); only meaningful at beta>0.
    """

    survived_lines: Optional[int] = None
    total_lines: int = 0
    island_chains: Optional[int] = None
    classifiable_fraction: Optional[float] = None
    qa_nonqs_ratio: Optional[float] = None
    magnetic_well: Optional[float] = None
    iota_shear: Optional[float] = None
    beta: float = 0.0
    mercier_dmerc_min: Optional[float] = None


@dataclass(frozen=True)
class ConfinementGateConfig:
    """Thresholds + required-flags for the confinement gate.

    Survival, island-chains, and classifiability are the three components of the STRICT
    topology standard (they are exactly the gates the reused ``topology_tier_verdict`` strict
    tier applies), so all three are required/decisive by default: a confinement gate that
    ignored islands or classifiability would silently accept candidates the strict tier
    rejects. QA / magnetic-well / shear are recorded and thresholded but advisory
    (``require_*`` default False) because their banana-context thresholds are not yet
    calibrated; a caller (or a future calibration) can flip them to decisive.

    ``min_survival_fraction`` is the survival accept bar. It defaults to 0.90 (45/50), which
    is INTENTIONALLY more permissive than the reused strict tier's own ``survival_threshold``
    of 1.0 (50/50): real validated champions sit at ~47-49/50, so a 50/50 bar would reject
    them. The island + classifiability gates still match the tier exactly.
    """

    min_survival_fraction: float = 0.90
    max_island_chains: int = 0
    require_islands: bool = True
    min_classifiable_fraction: float = 0.90
    require_classifiability: bool = True
    qa_nonqs_max: float = 1.0e-2
    require_qa: bool = False
    magnetic_well_max: float = 0.0
    require_magnetic_well: bool = False
    iota_shear_min: float = 0.0
    require_shear: bool = False
    mercier_dmerc_min: float = 0.0
    require_mercier: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_survival_fraction <= 1.0:
            raise ValueError("min_survival_fraction must be in [0, 1].")
        if not 0.0 <= self.min_classifiable_fraction <= 1.0:
            raise ValueError("min_classifiable_fraction must be in [0, 1].")
        if self.max_island_chains < 0:
            raise ValueError("max_island_chains must be >= 0.")


@dataclass(frozen=True)
class CriterionResult:
    """Outcome of one confinement criterion.

    ``applicable`` is False when the metric was not evaluated (or Mercier in vacuum); such a
    criterion never contributes to rejection. ``required`` mirrors the config; a candidate is
    accepted iff every applicable required criterion passed.
    """

    name: str
    applicable: bool
    passed: bool
    required: bool
    value: Optional[float]
    threshold: Optional[float]
    detail: str


@dataclass(frozen=True)
class ConfinementVerdict:
    accepted: bool
    decisive_reason: str
    criteria: tuple[CriterionResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """JSON-serializable verdict for results.json / the campaign ledger."""
        return {
            "accepted": self.accepted,
            "decisive_reason": self.decisive_reason,
            "criteria": [
                {
                    "name": c.name,
                    "applicable": c.applicable,
                    "passed": c.passed,
                    "required": c.required,
                    "value": c.value,
                    "threshold": c.threshold,
                    "detail": c.detail,
                }
                for c in self.criteria
            ],
        }


def evaluate_confinement_gate(
    metrics: ConfinementMetrics,
    config: ConfinementGateConfig | None = None,
) -> ConfinementVerdict:
    """Decisive confinement verdict for a converged candidate.

    Returns a ``ConfinementVerdict`` whose ``accepted`` is True iff the strict topology
    criterion is present and passes AND every other applicable *required* criterion passes.
    Topology fails CLOSED: a missing/empty topology metric is a rejection, not a skip.
    """
    cfg = config or ConfinementGateConfig()
    criteria: list[CriterionResult] = []

    # --- Topology (decisive, fail-closed) -------------------------------------------------
    has_topology = metrics.survived_lines is not None and metrics.total_lines > 0
    survival_fraction = (
        metrics.survived_lines / metrics.total_lines if has_topology else None
    )
    topology_passed = bool(
        has_topology and survival_fraction >= cfg.min_survival_fraction
    )
    criteria.append(
        CriterionResult(
            name=CRITERION_TOPOLOGY,
            applicable=True,  # always evaluated; absence => fail-closed below
            passed=topology_passed,
            required=True,
            value=survival_fraction,
            threshold=cfg.min_survival_fraction,
            detail=(
                f"{metrics.survived_lines}/{metrics.total_lines} strict field lines survived"
                if has_topology
                else "no strict topology evidence (fail-closed)"
            ),
        )
    )

    # --- Island chains (strict WBA tier) --------------------------------------------------
    islands_applicable = metrics.island_chains is not None
    criteria.append(
        CriterionResult(
            name=CRITERION_ISLANDS,
            applicable=islands_applicable,
            passed=bool(
                islands_applicable and metrics.island_chains <= cfg.max_island_chains
            ),
            required=cfg.require_islands,
            value=(
                float(metrics.island_chains) if islands_applicable else None
            ),
            threshold=float(cfg.max_island_chains),
            detail=(
                f"{metrics.island_chains} island chains (max {cfg.max_island_chains})"
                if islands_applicable
                else "WBA island classifier not run"
            ),
        )
    )

    # --- Classifiability (strict-tier: enough surviving lines were WBA-classifiable) -------
    classif_applicable = metrics.classifiable_fraction is not None
    criteria.append(
        CriterionResult(
            name=CRITERION_CLASSIFIABILITY,
            applicable=classif_applicable,
            passed=bool(
                classif_applicable
                and metrics.classifiable_fraction >= cfg.min_classifiable_fraction
            ),
            required=cfg.require_classifiability,
            value=metrics.classifiable_fraction,
            threshold=cfg.min_classifiable_fraction,
            detail=(
                f"classifiable fraction {metrics.classifiable_fraction:.3f} "
                f"(min {cfg.min_classifiable_fraction:.3f})"
                if classif_applicable
                else "WBA classifiability not evaluated"
            ),
        )
    )

    # --- Quasi-symmetry (lower NonQSRatio = better) ---------------------------------------
    qa_applicable = metrics.qa_nonqs_ratio is not None
    criteria.append(
        CriterionResult(
            name=CRITERION_QA,
            applicable=qa_applicable,
            passed=bool(qa_applicable and metrics.qa_nonqs_ratio <= cfg.qa_nonqs_max),
            required=cfg.require_qa,
            value=metrics.qa_nonqs_ratio,
            threshold=cfg.qa_nonqs_max,
            detail=(
                f"non-QS ratio {metrics.qa_nonqs_ratio:.3e} (max {cfg.qa_nonqs_max:.3e})"
                if qa_applicable
                else "QA ratio not evaluated"
            ),
        )
    )

    # --- Magnetic well (negative favorable; reject above max) -----------------------------
    well_applicable = metrics.magnetic_well is not None
    criteria.append(
        CriterionResult(
            name=CRITERION_MAGNETIC_WELL,
            applicable=well_applicable,
            passed=bool(well_applicable and metrics.magnetic_well <= cfg.magnetic_well_max),
            required=cfg.require_magnetic_well,
            value=metrics.magnetic_well,
            threshold=cfg.magnetic_well_max,
            detail=(
                f"well proxy W={metrics.magnetic_well:.3e} (max {cfg.magnetic_well_max:.3e})"
                if well_applicable
                else "magnetic-well proxy needs >=3 surfaces"
            ),
        )
    )

    # --- Iota shear (larger escapes the rational soup; reject below min) ------------------
    shear_applicable = metrics.iota_shear is not None
    criteria.append(
        CriterionResult(
            name=CRITERION_SHEAR,
            applicable=shear_applicable,
            passed=bool(shear_applicable and metrics.iota_shear >= cfg.iota_shear_min),
            required=cfg.require_shear,
            value=metrics.iota_shear,
            threshold=cfg.iota_shear_min,
            detail=(
                f"|iota spread| {metrics.iota_shear:.3e} (min {cfg.iota_shear_min:.3e})"
                if shear_applicable
                else "iota shear needs >=2 surfaces"
            ),
        )
    )

    # --- Mercier (finite-beta only; vacuum => not applicable, NOT a pass) -----------------
    mercier_applicable = metrics.beta > 0.0 and metrics.mercier_dmerc_min is not None
    criteria.append(
        CriterionResult(
            name=CRITERION_MERCIER,
            applicable=mercier_applicable,
            passed=bool(
                mercier_applicable
                and metrics.mercier_dmerc_min >= cfg.mercier_dmerc_min
            ),
            required=cfg.require_mercier,
            value=metrics.mercier_dmerc_min,
            threshold=cfg.mercier_dmerc_min,
            detail=(
                f"min DMerc {metrics.mercier_dmerc_min:.3e} (min {cfg.mercier_dmerc_min:.3e})"
                if mercier_applicable
                else f"vacuum/N/A (beta={metrics.beta:g}; DMerc identically 0)"
            ),
        )
    )

    # --- Overall verdict ------------------------------------------------------------------
    failing_required = [
        c for c in criteria if c.required and c.applicable and not c.passed
    ]
    accepted = topology_passed and not failing_required
    if not topology_passed:
        first = next(c for c in criteria if c.name == CRITERION_TOPOLOGY)
        decisive_reason = f"REJECT: topology — {first.detail}"
    elif failing_required:
        c = failing_required[0]
        decisive_reason = f"REJECT: {c.name} — {c.detail}"
    else:
        decisive_reason = "ACCEPT: strict topology passed; all required criteria met"

    return ConfinementVerdict(
        accepted=accepted,
        decisive_reason=decisive_reason,
        criteria=tuple(criteria),
    )


def metrics_from_topology_verdict(
    topology_verdict: dict,
    nfieldlines: int,
    *,
    qa_nonqs_ratio: Optional[float] = None,
    magnetic_well: Optional[float] = None,
    iota_shear: Optional[float] = None,
    beta: float = 0.0,
    mercier_dmerc_min: Optional[float] = None,
) -> ConfinementMetrics:
    """Adapt a ``topology_tier_verdict`` payload into ``ConfinementMetrics``.

    Maps the strict-tier verdict (``survival_fraction`` / ``surviving_seed_count`` /
    ``island_chain_count`` / ``classifiable_fraction``) onto the gate's topology + island +
    classifiability inputs and attaches the separately-computed physics values. A ``broken``
    topology result carries NO survival evidence (``survived_lines=None``) so the gate fails
    closed. ``surviving_seed_count`` is preferred when present (the WBA island gate populates
    it); otherwise the survived count is recovered from ``survival_fraction * nfieldlines``.
    """
    broken = bool(topology_verdict.get("broken", False))
    surviving = topology_verdict.get("surviving_seed_count")
    if broken:
        survived: Optional[int] = None
    elif surviving is not None:
        survived = int(surviving)
    else:
        survival_fraction = float(topology_verdict.get("survival_fraction", 0.0))
        survived = int(round(survival_fraction * int(nfieldlines)))
    islands = topology_verdict.get("island_chain_count")
    classifiable = topology_verdict.get("classifiable_fraction")
    return ConfinementMetrics(
        survived_lines=survived,
        total_lines=int(nfieldlines),
        island_chains=int(islands) if islands is not None else None,
        classifiable_fraction=(
            float(classifiable) if classifiable is not None else None
        ),
        qa_nonqs_ratio=qa_nonqs_ratio,
        magnetic_well=magnetic_well,
        iota_shear=iota_shear,
        beta=float(beta),
        mercier_dmerc_min=mercier_dmerc_min,
    )
