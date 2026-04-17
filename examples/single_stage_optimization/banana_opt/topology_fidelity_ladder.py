from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


# Keep this aligned with topology_scorer.SEED_MODE_MIDPLANE without forcing
# the reporting helper to depend on a particular import layout.
SEED_MODE_MIDPLANE = "midplane_radial_sweep"
TOPOLOGY_FIDELITY_LADDER_SCHEMA_VERSION = "topology_fidelity_ladder_v2"
DEFAULT_INSET_FRACTION = 0.05


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


def topology_tier_passed(
    result: Mapping[str, object],
    *,
    survival_threshold: float,
) -> bool:
    if bool(result.get("broken", False)):
        return False
    return float(result.get("survival_fraction", 0.0)) >= float(survival_threshold)


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
