"""Produce the canonical unmeasured Phase-0 performance-gap policy.

The policy is intentionally maximally noncommittal: no measured canary exists at
Phase 0, so conservative reductions are zero, optimistic reductions are only the
mathematical elimination ceiling, and every plan-derived implementation lever is
unbounded.  The existing gap-input artifact builder adds the matched complete-path
timings and limits the resulting budget to diagnostic routing evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from examples.jax.parity.artifacts import canonical_json_bytes

from benchmarks.single_stage_compute_graph_attribution_control import (
    AttributionControlError,
    require_promoting_attribution_evidence,
)
from benchmarks.single_stage_compute_graph_complete_path import (
    CompletePathEvidenceError,
    FaithfulLever,
    GapBudgetPolicyInput,
    PhaseReductionAssumption,
    build_gap_budget_inputs_artifact,
    build_staged_gap_budget_timing_input,
    validate_gap_budget_inputs_artifact,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
PLAN_PATH: Final = (
    REPO_ROOT
    / "docs"
    / "single_stage_jax_gpu_compute_graph_optimization_implementation_plan.md"
)
_PHASE_HEADING = re.compile(
    r"^### Phase (?P<number>[0-9]+) — (?P<title>.+?)(?: \([^\n]*\))?$",
    re.MULTILINE,
)
_SLUG_COMPONENT = re.compile(r"[^a-z0-9]+")
_BASELINE_TITLE_PREFIX: Final = "Freeze a phase-complete baseline"
_COMPOSITION_TITLE_PREFIX: Final = "Compose, replay, and freeze a candidate"
_CONSERVATIVE_UNMEASURED_REDUCTION: Final = 0.0
_OPTIMISTIC_ELIMINATION_CEILING: Final = 1.0


class GapPolicyProducerError(RuntimeError):
    """The canonical gap policy cannot be derived without inventing evidence."""


@dataclass(frozen=True, slots=True)
class _PlanPhase:
    number: int
    title: str
    section: str


def _canonical_mapping(path: Path, context: str) -> Mapping[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                GapPolicyProducerError(
                    f"{context} contains non-finite constant {constant}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GapPolicyProducerError(f"{context} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise GapPolicyProducerError(f"{context} must be a JSON object")
    if raw != canonical_json_bytes(value):
        raise GapPolicyProducerError(f"{context} is not canonical JSON")
    return value


def _plan_phases(plan_path: Path) -> tuple[_PlanPhase, ...]:
    try:
        text = plan_path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise GapPolicyProducerError("optimization plan is not UTF-8") from error
    matches = tuple(_PHASE_HEADING.finditer(text))
    if not matches:
        raise GapPolicyProducerError("optimization plan has no Phase headings")
    phases = tuple(
        _PlanPhase(
            number=int(match.group("number")),
            title=match.group("title").strip(),
            section=text[
                match.start() : (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(text)
                )
            ],
        )
        for index, match in enumerate(matches)
    )
    if tuple(phase.number for phase in phases) != tuple(range(len(phases))):
        raise GapPolicyProducerError(
            "optimization plan Phase headings must be unique and contiguous from zero"
        )
    if not phases[0].title.startswith(_BASELINE_TITLE_PREFIX):
        raise GapPolicyProducerError("optimization plan Phase 0 is not the baseline")
    if not phases[-1].title.startswith(_COMPOSITION_TITLE_PREFIX):
        raise GapPolicyProducerError("optimization plan final Phase is not composition")
    implementation_phases = phases[1:-1]
    if not implementation_phases:
        raise GapPolicyProducerError("optimization plan has no implementation levers")
    if any("**Exit gate:**" not in phase.section for phase in implementation_phases):
        raise GapPolicyProducerError(
            "every optimization-plan implementation phase must define an exit gate"
        )
    return implementation_phases


def _lever_id(phase: _PlanPhase) -> str:
    slug = _SLUG_COMPONENT.sub("-", phase.title.lower()).strip("-")
    if not slug:
        raise GapPolicyProducerError("optimization-plan phase title has no slug")
    return f"phase-{phase.number}-{slug}"


def _plan_levers(plan_path: Path) -> tuple[FaithfulLever, ...]:
    return tuple(
        FaithfulLever(
            lever_id=_lever_id(phase),
            disposition="unbounded",
            evidence_sha256=hashlib.sha256(phase.section.encode("utf-8")).hexdigest(),
        )
        for phase in _plan_phases(plan_path)
    )


def _selected_phase_ids(
    attribution_document: Mapping[str, object],
) -> tuple[str, ...]:
    try:
        require_promoting_attribution_evidence(attribution_document)
    except AttributionControlError as error:
        raise GapPolicyProducerError(
            "attribution evidence is not promotion-safe"
        ) from error
    selected = attribution_document.get("selected_attribution")
    phase_rows = selected.get("phase_shares") if isinstance(selected, Mapping) else None
    if not isinstance(phase_rows, list) or not phase_rows:
        raise GapPolicyProducerError("selected attribution has no phase shares")
    phase_ids: list[str] = []
    for row in phase_rows:
        phase_id = row.get("phase_id") if isinstance(row, Mapping) else None
        if not isinstance(phase_id, str) or not phase_id:
            raise GapPolicyProducerError("selected attribution phase_id is invalid")
        phase_ids.append(phase_id)
    if len(phase_ids) != len(set(phase_ids)):
        raise GapPolicyProducerError("selected attribution phase IDs are duplicated")
    return tuple(phase_ids)


def _require_matching_binding(
    complete_path_document: Mapping[str, object],
    attribution_document: Mapping[str, object],
) -> None:
    complete_identity = complete_path_document.get("identity")
    attribution_binding = attribution_document.get("production_binding")
    if not isinstance(complete_identity, Mapping) or not isinstance(
        attribution_binding, Mapping
    ):
        raise GapPolicyProducerError("complete-path or attribution identity is missing")
    field_pairs = (
        ("candidate_sha256", "candidate_sha256"),
        ("specimen_sha256", "specimen_sha256"),
        ("input_bundle_sha256", "input_bundle_sha256"),
        ("source_sha256", "source_sha256"),
        ("runtime_identity_sha256", "production_runtime_identity_sha256"),
        ("lane_id", "lane_id"),
        ("gpu_uuid", "gpu_uuid"),
        ("gate_checkpoint_sha256", "gate_checkpoint_sha256"),
        ("warm_checkpoint_sha256", "warm_checkpoint_sha256"),
        ("warm_p50_ns", "warm_p50_ns"),
    )
    for complete_field, attribution_field in field_pairs:
        if complete_identity.get(complete_field) != attribution_binding.get(
            attribution_field
        ):
            raise GapPolicyProducerError(
                f"attribution binding differs from complete path for {complete_field}"
            )


def _policy_document(policy: GapBudgetPolicyInput) -> dict[str, object]:
    return {
        "phase_reduction_assumptions": {
            phase_id: {
                "conservative_reduction": assumption.conservative_reduction,
                "optimistic_reduction": assumption.optimistic_reduction,
                "overlap_disposition": assumption.overlap_disposition,
            }
            for phase_id, assumption in policy.phase_reduction_assumptions.items()
        },
        "unattributed_conservative_reduction": (
            policy.unattributed_conservative_reduction
        ),
        "unattributed_optimistic_reduction": (policy.unattributed_optimistic_reduction),
        "faithful_levers": [
            {
                "lever_id": lever.lever_id,
                "disposition": lever.disposition,
                "evidence_sha256": lever.evidence_sha256,
            }
            for lever in policy.faithful_levers
        ],
    }


def build_phase0_gap_policy(
    complete_path_document: Mapping[str, object],
    attribution_document: Mapping[str, object],
    *,
    plan_path: Path = PLAN_PATH,
) -> dict[str, object]:
    """Derive honest unknown bounds and validate them against current inputs."""

    try:
        build_staged_gap_budget_timing_input(complete_path_document)
    except CompletePathEvidenceError as error:
        raise GapPolicyProducerError(
            "complete-path formal inputs are invalid"
        ) from error
    phase_ids = _selected_phase_ids(attribution_document)
    _require_matching_binding(complete_path_document, attribution_document)
    policy = GapBudgetPolicyInput(
        phase_reduction_assumptions={
            phase_id: PhaseReductionAssumption(
                conservative_reduction=_CONSERVATIVE_UNMEASURED_REDUCTION,
                optimistic_reduction=_OPTIMISTIC_ELIMINATION_CEILING,
                overlap_disposition="disjoint",
            )
            for phase_id in phase_ids
        },
        unattributed_conservative_reduction=_CONSERVATIVE_UNMEASURED_REDUCTION,
        unattributed_optimistic_reduction=_OPTIMISTIC_ELIMINATION_CEILING,
        faithful_levers=_plan_levers(plan_path),
    )
    artifact = build_gap_budget_inputs_artifact(complete_path_document, policy)
    validate_gap_budget_inputs_artifact(artifact, complete_path_document)
    return _policy_document(policy)


def produce_phase0_gap_policy(
    *,
    complete_path_path: Path,
    attribution_evidence_path: Path,
    output_path: Path,
) -> Path:
    """Write one canonical policy exclusively after full upstream validation."""

    if output_path.exists():
        raise GapPolicyProducerError("output path must not already exist")
    complete = _canonical_mapping(complete_path_path, "complete-path evidence")
    attribution = _canonical_mapping(attribution_evidence_path, "attribution evidence")
    policy_document = build_phase0_gap_policy(complete, attribution)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(canonical_json_bytes(policy_document))
        stream.flush()
        os.fsync(stream.fileno())

    persisted = _canonical_mapping(output_path, "produced gap policy")
    if persisted != policy_document:
        raise GapPolicyProducerError("persisted gap policy differs from built policy")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-path", type=Path, required=True)
    parser.add_argument("--attribution-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    try:
        produce_phase0_gap_policy(
            complete_path_path=options.complete_path,
            attribution_evidence_path=options.attribution_evidence,
            output_path=options.output,
        )
    except (OSError, GapPolicyProducerError) as error:
        print(f"Gap-policy production failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
