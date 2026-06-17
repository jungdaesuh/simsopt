from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .topology.kam_birkhoff import (
    DEFAULT_WBA_MIN_RETURNS,
    island_verdict_counts,
)


# Keep this aligned with topology_scorer.SEED_MODE_MIDPLANE without forcing
# the reporting helper to depend on a particular import layout.
SEED_MODE_MIDPLANE = "midplane_radial_sweep"
TOPOLOGY_FIDELITY_LADDER_SCHEMA_VERSION = "topology_fidelity_ladder_v2"
DEFAULT_INSET_FRACTION = 0.05

# Key carrying the per-line WBA classifications in a topology-score result.
WBA_SEED_CLASSIFICATIONS_KEY = "wba_seed_classifications"

# Certification-gate defaults (opt-in; default OFF preserves the legacy verdict
# byte-for-byte). The island gate certifies the *absence* of island_chain lines
# rather than the presence of invariant_torus -- at production trace tolerances
# genuine smooth tori read "chaotic" (a trace-accuracy artefact), so requiring
# invariant_torus would reject real tori (Agent F2 calibration, 2026-06-11).
DEFAULT_N_ISLAND_MAX = 0
# Fraction of surviving lines that must clear min_returns for the verdict to be
# certifiable. Production strict traces give ~256+ returns for every surviving
# seed (Agent F2 measured 583-596 returns across all 50 survivors), so 0.90
# tolerates a small minority of short-return edge seeds while still failing
# closed if a substantial fraction of lines is unclassifiable.
DEFAULT_CLASSIFIABLE_FRACTION_MIN = 0.90


@dataclass(frozen=True)
class TopologyTierSpec:
    name: str
    nfieldlines: int
    tmax: float
    nphis: int
    survival_threshold: float
    seed_mode: str
    inset_fraction: float
    field_policy: str

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def _build_tier_spec(
    name: str,
    *,
    nfieldlines: int,
    tmax: float,
    nphis: int,
    survival_threshold: float,
    field_policy: str,
) -> TopologyTierSpec:
    return TopologyTierSpec(
        name=name,
        nfieldlines=nfieldlines,
        tmax=tmax,
        nphis=nphis,
        survival_threshold=survival_threshold,
        seed_mode=SEED_MODE_MIDPLANE,
        inset_fraction=DEFAULT_INSET_FRACTION,
        field_policy=field_policy,
    )


DEFAULT_TOPOLOGY_TIER_SPECS = {
    "cheap": _build_tier_spec(
        "cheap",
        nfieldlines=4,
        tmax=2.0,
        nphis=1,
        survival_threshold=0.25,
        field_policy="never",
    ),
    "medium": _build_tier_spec(
        "medium",
        nfieldlines=12,
        tmax=50.0,
        nphis=4,
        survival_threshold=1.0,
        field_policy="auto",
    ),
    "strict": _build_tier_spec(
        "strict",
        nfieldlines=50,
        tmax=7000.0,
        nphis=4,
        survival_threshold=1.0,
        field_policy="auto",
    ),
}


def _survival_gate_passed(
    result: Mapping[str, object],
    *,
    survival_threshold: float,
) -> bool:
    if bool(result.get("broken", False)):
        return False
    return float(result.get("survival_fraction", 0.0)) >= float(survival_threshold)


def _require_seed_classifications(result: Mapping[str, object]) -> Sequence[object]:
    """Return the per-line WBA classifications, raising loudly when absent.

    A certification gate that asks for an island / classifiability verdict on a
    result that never carried classification data is a misconfiguration, not a
    pass -- so this fails closed with a clear error rather than silently
    certifying an island-blind result.
    """

    if (
        WBA_SEED_CLASSIFICATIONS_KEY not in result
        or result.get(WBA_SEED_CLASSIFICATIONS_KEY) is None
    ):
        raise ValueError(
            "island/classifiability certification gate was requested but the "
            f"topology result has no '{WBA_SEED_CLASSIFICATIONS_KEY}' payload; "
            "refusing to certify an island-blind result"
        )
    classifications = result[WBA_SEED_CLASSIFICATIONS_KEY]
    if not isinstance(classifications, Sequence):
        raise ValueError(
            "island/classifiability certification gate was requested but "
            f"'{WBA_SEED_CLASSIFICATIONS_KEY}' is not a sequence; refusing to "
            "certify an invalid topology result payload"
        )
    return classifications


def topology_tier_verdict(
    result: Mapping[str, object],
    *,
    survival_threshold: float,
    require_island_gate: bool = False,
    n_island_max: int = DEFAULT_N_ISLAND_MAX,
    require_classifiability_floor: bool = False,
    classifiable_fraction_min: float = DEFAULT_CLASSIFIABLE_FRACTION_MIN,
    min_returns: int = DEFAULT_WBA_MIN_RETURNS,
) -> dict[str, object]:
    """Evaluate the (optionally tightened) topology certification verdict.

    The verdict is the conjunction of up to three gates:

    * **survival gate** (always on): not ``broken`` and
      ``survival_fraction >= survival_threshold`` -- byte-identical to the
      legacy verdict.
    * **island gate** (opt-in via ``require_island_gate``): at most
      ``n_island_max`` lines (default 0) are classified ``island_chain``.
    * **classifiability floor** (opt-in via ``require_classifiability_floor``):
      at least ``classifiable_fraction_min`` of the *surviving* lines cleared
      ``min_returns`` Poincare returns (lines below the floor are unreliable and
      counted as not-classifiable; a result with no surviving lines fails
      closed).

    Both opt-in gates can only *tighten* the verdict -- a result rejected by the
    survival gate is never rescued here. Requesting either opt-in gate on a
    result that carries no per-line WBA classifications raises loudly.

    The returned payload surfaces the per-line tallies (e.g. survived /
    island_chain / classifiable counts) so reports can quote
    "43/50 survived, 0 island-chain, 47/50 classifiable".
    """

    survival_gate_passed = _survival_gate_passed(
        result,
        survival_threshold=survival_threshold,
    )

    verdict: dict[str, object] = {
        "passed": survival_gate_passed,
        "survival_gate_passed": survival_gate_passed,
        "broken": bool(result.get("broken", False)),
        "survival_fraction": float(result.get("survival_fraction", 0.0)),
        "survival_threshold": float(survival_threshold),
        "island_gate_enabled": bool(require_island_gate),
        "island_gate_passed": None,
        "n_island_max": int(n_island_max),
        "island_chain_count": None,
        "island_chain_seed_indices": None,
        "classifiability_floor_enabled": bool(require_classifiability_floor),
        "classifiability_floor_passed": None,
        "classifiable_fraction_min": float(classifiable_fraction_min),
        "min_returns": int(min_returns),
        "surviving_seed_count": None,
        "classifiable_seed_count": None,
        "classifiable_fraction": None,
    }

    if not (require_island_gate or require_classifiability_floor):
        return verdict

    counts = island_verdict_counts(
        _require_seed_classifications(result),
        min_returns=min_returns,
    )
    verdict["seed_count"] = int(counts["seed_count"])
    verdict["surviving_seed_count"] = int(counts["surviving_seed_count"])
    verdict["classifiable_seed_count"] = int(counts["classifiable_seed_count"])
    verdict["classifiable_fraction"] = counts["classifiable_fraction"]
    verdict["island_chain_count"] = int(counts["island_chain_count"])
    verdict["island_chain_seed_indices"] = list(counts["island_chain_seed_indices"])

    passed = survival_gate_passed

    if require_island_gate:
        island_gate_passed = (
            int(counts["island_chain_count"]) <= int(n_island_max)
        )
        verdict["island_gate_passed"] = island_gate_passed
        passed = passed and island_gate_passed

    if require_classifiability_floor:
        classifiable_fraction = counts["classifiable_fraction"]
        classifiability_floor_passed = (
            classifiable_fraction is not None
            and float(classifiable_fraction) >= float(classifiable_fraction_min)
        )
        verdict["classifiability_floor_passed"] = classifiability_floor_passed
        passed = passed and classifiability_floor_passed

    verdict["passed"] = passed
    return verdict


def topology_tier_passed(
    result: Mapping[str, object],
    *,
    survival_threshold: float,
    require_island_gate: bool = False,
    n_island_max: int = DEFAULT_N_ISLAND_MAX,
    require_classifiability_floor: bool = False,
    classifiable_fraction_min: float = DEFAULT_CLASSIFIABLE_FRACTION_MIN,
    min_returns: int = DEFAULT_WBA_MIN_RETURNS,
) -> bool:
    """Boolean topology certification verdict.

    Default arguments preserve the legacy survival-only verdict byte-for-byte;
    production certification lanes opt in to the island gate and classifiability
    floor explicitly. This is strictly a tightening of the verdict -- it can
    never loosen the survival gate.
    """

    return bool(
        topology_tier_verdict(
            result,
            survival_threshold=survival_threshold,
            require_island_gate=require_island_gate,
            n_island_max=n_island_max,
            require_classifiability_floor=require_classifiability_floor,
            classifiable_fraction_min=classifiable_fraction_min,
            min_returns=min_returns,
        )["passed"]
    )


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(values_array.size, dtype=float)
    index = 0
    while index < values_array.size:
        end = index + 1
        while (
            end < values_array.size
            and values_array[order[end]] == values_array[order[index]]
        ):
            end += 1
        average_rank = 0.5 * (index + end - 1) + 1.0
        ranks[order[index:end]] = average_rank
        index = end
    return ranks


def spearman_rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("Spearman inputs must have the same length")
    if len(xs) < 2:
        return None
    x_ranks = _rankdata(xs)
    y_ranks = _rankdata(ys)
    x_std = float(np.std(x_ranks))
    y_std = float(np.std(y_ranks))
    if x_std == 0.0 or y_std == 0.0:
        return None
    correlation = np.corrcoef(x_ranks, y_ranks)[0, 1]
    return float(correlation)


def summarize_tier_agreement(
    case_records: Sequence[Mapping[str, object]],
    *,
    tier_name: str,
    reference_tier_name: str = "strict",
) -> dict[str, object]:
    false_passes: list[str] = []
    false_rejects: list[str] = []
    tier_scores: list[float] = []
    reference_scores: list[float] = []

    for case_record in case_records:
        label = str(case_record["label"])
        tier_record = case_record[tier_name]
        reference_record = case_record[reference_tier_name]
        tier_pass = bool(tier_record["passed"])
        reference_pass = bool(reference_record["passed"])
        if tier_pass and not reference_pass:
            false_passes.append(label)
        if not tier_pass and reference_pass:
            false_rejects.append(label)
        tier_scores.append(float(tier_record["confinement_score"]))
        reference_scores.append(float(reference_record["confinement_score"]))

    return {
        "tier": tier_name,
        "reference_tier": reference_tier_name,
        "num_cases": int(len(case_records)),
        "false_pass_count": int(len(false_passes)),
        "false_reject_count": int(len(false_rejects)),
        "false_pass_labels": false_passes,
        "false_reject_labels": false_rejects,
        "spearman_rank_correlation": spearman_rank_correlation(
            tier_scores,
            reference_scores,
        ),
    }


def build_topology_fidelity_report(
    case_records: Sequence[Mapping[str, object]],
    *,
    tier_specs: Mapping[str, TopologyTierSpec] | None = None,
) -> dict[str, object]:
    resolved_tier_specs = (
        DEFAULT_TOPOLOGY_TIER_SPECS
        if tier_specs is None
        else dict(tier_specs)
    )
    return {
        "schema_version": TOPOLOGY_FIDELITY_LADDER_SCHEMA_VERSION,
        "tier_specs": {
            tier_name: tier_spec.to_json_dict()
            for tier_name, tier_spec in resolved_tier_specs.items()
        },
        "cases": list(case_records),
        "agreements": {
            "cheap_vs_strict": summarize_tier_agreement(
                case_records,
                tier_name="cheap",
                reference_tier_name="strict",
            ),
            "medium_vs_strict": summarize_tier_agreement(
                case_records,
                tier_name="medium",
                reference_tier_name="strict",
            ),
        },
    }
