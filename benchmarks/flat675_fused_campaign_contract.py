"""Frozen arithmetic and verdict rules of the flat-675 fused campaign (F3).

Charter: ``docs/jax_gpu_flat675_fused_campaign_plan.md`` on pr/jax-port-squashed
at commit b7ec63b6e (sha256 embedded below).  Everything this module computes —
policy identity, per-row contract binding, counter liveness, the three anchor
formulas, the quality gate, the dual verdict rule, cap accounting, and
``validate`` — is stated in that charter and is non-amendable post-evidence.

This module is deliberately free of every instrument import.  The orchestrator
beside it (``flat675_fused_campaign.py``) reuses the fair-bar harness, which
only resolves under the pinned instrument tree; the charter arithmetic must
stay checkable in the ordinary production environment, so it lives here and the
orchestrator imports it.  Nothing here launches a process, reads a device, or
consults the clock.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Frozen campaign identity
# --------------------------------------------------------------------------

F3_CHARTER_SHA256: Final[str] = (
    "0a61ed647afc08424a149a06a6e247535d4da931136bc5d2294874634b9564dc"
)
F3_CHARTER_COMMIT: Final[str] = "b7ec63b6e"
F3_CHARTER_PATH: Final[str] = (
    "docs/jax_gpu_flat675_fused_campaign_plan.md (pr/jax-port-squashed)"
)
# Append-only: the sha of the charter bytes at each amendment commit, seeded
# with the freeze.  Rows bind the sha current when they executed; validate
# accepts any lineage member and recomputes against the row's own sha.
F3_CHARTER_LINEAGE: Final[tuple[str, ...]] = (F3_CHARTER_SHA256,)

F3_ROW_SCHEMA: Final[str] = "flat675-fused-campaign-row.v1"
F3_RUN_MANIFEST_SCHEMA: Final[str] = "flat675-fused-campaign-manifest.v1"

L1_LANE: Final[str] = "fused_gpu"
L2_LANE: Final[str] = "native_cpp_cpu"
L3_LANE: Final[str] = "host_loop_gpu"

# --------------------------------------------------------------------------
# Shared optimizer policy (charter "Shared optimizer policy")
# --------------------------------------------------------------------------

POLICY_METHOD: Final[str] = "L-BFGS-B"
POLICY_MAXCOR: Final[int] = 300
POLICY_MAXLS: Final[int] = 8
POLICY_FTOL: Final[float] = 0.0
POLICY_GTOL: Final[float] = 0.001
# The fused lane caps maxfun at maxiter x 20.  With maxls=8 an L-BFGS-B
# iteration costs at most nine evaluations, so the cap cannot bind at any
# chartered budget (37 x 9 = 333 < 740).
MAXFUN_MULTIPLIER: Final[int] = 20
POLICY_FIELDS: Final[tuple[str, ...]] = (
    "method",
    "maxiter",
    "maxfun",
    "gtol",
    "ftol",
    "maxcor",
    "maxls",
)
# The archived native policy record defines no maxfun: the native lane imposes
# no function-evaluation cap.  The policy sha covers one payload per rung, so
# native rows carry the chartered value and disclose the substitution rather
# than hashing a different payload than the lane they describe.
NATIVE_MAXFUN_SOURCE: Final[str] = (
    "chartered_default_native_lane_imposes_no_evaluation_cap"
)

RUNG_BUDGETS: Final[Mapping[str, int]] = {"b3": 3, "b37": 37}

# --------------------------------------------------------------------------
# Anchors, thresholds, caps (charter "Verdict rule", "Caps and aborts")
# --------------------------------------------------------------------------

ARCHIVED_B3_PROCESS_WALL: Final[float] = 58.702
ARCHIVED_STEADY_PER_EVAL: Final[float] = 52.807 / 9

WIN_MEDIAN_THRESHOLD: Final[float] = 1.10
WIN_PAIR_THRESHOLD: Final[float] = 1.00
QUALITY_OBJECTIVE_RTOL: Final[float] = 1.0e-10
GRADIENT_FACTOR_K: Final[float] = 2.0

PAIR_COUNT: Final[int] = 5
NOT_PRODUCED_ABORT: Final[int] = 3
MAX_TIMED_LEGS: Final[int] = 51
MAX_SOLVE_CHILD_PROCESSES: Final[int] = 130
MAX_CAMPAIGN_WALL_SECONDS: Final[float] = 12 * 3600.0

BQ_MAX_PROBES_PER_LANE: Final[int] = 12
BQ_MAX_MAXITER: Final[int] = 1024
BQ_MAX_SEARCH_SECONDS: Final[float] = 2 * 3600.0
BQ_SEARCH_START: Final[int] = 37

NATIVE_SWEEP_OMP_MATRIX: Final[tuple[int, ...]] = (1, 2, 4, 8, 16)
NATIVE_SWEEP_REPS: Final[int] = 3


class Verdict(str, Enum):
    """The charter's complete terminal vocabulary; no other outcome exists."""

    WIN = "WIN"
    CLOSED_BOUNDED_NEGATIVE = "CLOSED_BOUNDED_NEGATIVE"
    NOT_PRODUCED = "NOT_PRODUCED"


class F3ContractError(RuntimeError):
    """Raised when campaign evidence violates the frozen charter."""


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F3ContractError(f"{where} must be a real scalar.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise F3ContractError(f"{where} must be finite.")
    return scalar


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise F3ContractError(f"{where} must be a positive integer.")
    return value


def _count(value: object, where: str, *, default: int = 0) -> int:
    """Read a recorded count without coercing a foreign type into one."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise F3ContractError(f"{where} must be an integer count.")
    return value


def _seconds(value: object, where: str, *, default: float = 0.0) -> float:
    if value is None:
        return default
    return _finite(value, where)


def _string_list(value: object, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise F3ContractError(f"{where} must be a JSON array.")
    return [str(entry) for entry in value]


# --------------------------------------------------------------------------
# Policy identity (charter: mismatch voids the leg)
# --------------------------------------------------------------------------


def policy_payload(budget: int) -> dict[str, object]:
    """The chartered L-BFGS-B policy at one rung budget, as hashed."""
    maxiter = _positive_int(budget, "budget")
    return {
        "method": POLICY_METHOD,
        "maxiter": maxiter,
        "maxfun": maxiter * MAXFUN_MULTIPLIER,
        "gtol": POLICY_GTOL,
        "ftol": POLICY_FTOL,
        "maxcor": POLICY_MAXCOR,
        "maxls": POLICY_MAXLS,
    }


def policy_identity_sha256(budget: int) -> str:
    """The frozen per-rung policy constant both lanes must reproduce."""
    return _sha256_bytes(_canonical_bytes(policy_payload(budget)))


def observed_policy_sha256(observed: Mapping[str, object]) -> str:
    """Hash a lane's own constructed policy in the chartered field order."""
    missing = sorted(set(POLICY_FIELDS) - set(observed))
    if missing:
        raise F3ContractError(f"observed policy is missing {missing!r}.")
    return _sha256_bytes(
        _canonical_bytes({name: observed[name] for name in POLICY_FIELDS})
    )


def policy_identity_failures(
    observed: Mapping[str, object],
    *,
    budget: int,
) -> list[str]:
    """Name every chartered policy field the lane did not reproduce."""
    expected = policy_payload(budget)
    failures = [
        f"policy_{name}_{observed.get(name)!r}_expected_{expected[name]!r}"
        for name in POLICY_FIELDS
        if name in observed and observed[name] != expected[name]
    ]
    failures.extend(
        f"policy_{name}_absent" for name in POLICY_FIELDS if name not in observed
    )
    return failures


# --------------------------------------------------------------------------
# Per-row contract binding (charter "Governance")
# --------------------------------------------------------------------------


def f3_contract_sha256(
    *,
    campaign_manifest_sha256: str,
    budget: int,
    production_commit: str,
    instrument_commit: str,
    fair_bar_charter_sha256: str,
    charter_sha256: str = F3_CHARTER_SHA256,
) -> str:
    """Bind one leg row to every frozen component the charter enumerates."""
    return _sha256_bytes(
        _canonical_bytes(
            {
                "f3_charter_sha256": charter_sha256,
                "rung_policy_sha256": policy_identity_sha256(budget),
                "campaign_input_manifest_sha256": campaign_manifest_sha256,
                "fair_bar_charter_sha256": fair_bar_charter_sha256,
                "production_commit": production_commit,
                "instrument_commit": instrument_commit,
            }
        )
    )


# --------------------------------------------------------------------------
# Counter liveness (charter "Work matching")
# --------------------------------------------------------------------------


def counter_liveness_failures(
    *,
    l1_nfev: object,
    l2_compact_evaluations: object,
) -> list[str]:
    """Void a leg whose evaluation counter silently arrived as zero.

    ``dispatch.py`` populates ``nfev`` through ``getattr(..., 0)``, so a
    missing counter reads as a legitimate zero; a zero must never reach an
    anchor formula.
    """
    failures: list[str] = []
    for name, value in (
        ("l1_nfev", l1_nfev),
        ("l2_compact_candidate_evaluations", l2_compact_evaluations),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            failures.append(f"{name}_not_an_integer")
        elif value <= 0:
            failures.append(f"{name}_nonpositive_{value}")
    return failures


# --------------------------------------------------------------------------
# Anchors (charter "Verdict rule", rule 2 — formulas are non-amendable)
# --------------------------------------------------------------------------


def b3_anchor() -> float:
    """B3 anchor: the archived B3 process wall, used directly."""
    return ARCHIVED_B3_PROCESS_WALL


def b37_anchor(
    *,
    l2_compact_evaluations: Sequence[int],
    l1_nfev: Sequence[int],
) -> float:
    """B37 anchor: archived per-evaluation mean x min(median L2, median L1).

    The smaller of the two lanes' medians is deliberate: the GPU is never
    credited with native work it did not do, and never charged extra
    line-search evaluations at native prices.
    """
    if not l2_compact_evaluations or not l1_nfev:
        raise F3ContractError("B37 anchor needs both lanes' counters.")
    matched = min(
        statistics.median(l2_compact_evaluations),
        statistics.median(l1_nfev),
    )
    return ARCHIVED_STEADY_PER_EVAL * matched


def bq_anchor(*, l2_compact_evaluations_at_nstar: Sequence[int]) -> float:
    """BQ anchor: archived per-evaluation mean x median L2 evaluations at n*.

    Native's minimal cost of producing ``Q*``, priced at the uncontended
    archived rate — the time-to-quality currency.
    """
    if not l2_compact_evaluations_at_nstar:
        raise F3ContractError("BQ anchor needs the native counters at n*.")
    return ARCHIVED_STEADY_PER_EVAL * statistics.median(l2_compact_evaluations_at_nstar)


# --------------------------------------------------------------------------
# Quality gate (charter "Quality gate", oracle-adjudicated, one-sided)
# --------------------------------------------------------------------------


def fixed_budget_quality_failures(
    *,
    l1_oracle_objective: float,
    l2_oracle_objective: float,
    l1_oracle_gradient_inf: float,
    l2_oracle_gradient_inf: float,
) -> list[str]:
    """B3/B37: fused endpoint no worse than its paired native endpoint."""
    failures: list[str] = []
    l1_objective = _finite(l1_oracle_objective, "l1_oracle_objective")
    l2_objective = _finite(l2_oracle_objective, "l2_oracle_objective")
    if l1_objective > l2_objective * (1.0 + QUALITY_OBJECTIVE_RTOL):
        failures.append("l1_objective_above_paired_native")
    l1_gradient = _finite(l1_oracle_gradient_inf, "l1_oracle_gradient_inf")
    l2_gradient = _finite(l2_oracle_gradient_inf, "l2_oracle_gradient_inf")
    if l1_gradient > GRADIENT_FACTOR_K * l2_gradient:
        failures.append("l1_gradient_above_k_times_native")
    return failures


def bq_quality_failures(
    *,
    l1_oracle_objective: float,
    l2_oracle_objective: float,
    l1_oracle_gradient_inf: float,
    l2_oracle_gradient_inf: float,
    quality_target: float,
) -> list[str]:
    """BQ: both endpoints at or below Q*, re-verified per timed leg."""
    failures: list[str] = []
    target = _finite(quality_target, "quality_target")
    for name, value in (
        ("l1", _finite(l1_oracle_objective, "l1_oracle_objective")),
        ("l2", _finite(l2_oracle_objective, "l2_oracle_objective")),
    ):
        if value > target * (1.0 + QUALITY_OBJECTIVE_RTOL):
            failures.append(f"{name}_objective_above_quality_target")
    l1_gradient = _finite(l1_oracle_gradient_inf, "l1_oracle_gradient_inf")
    l2_gradient = _finite(l2_oracle_gradient_inf, "l2_oracle_gradient_inf")
    if l1_gradient > GRADIENT_FACTOR_K * l2_gradient:
        failures.append("l1_gradient_above_k_times_native")
    return failures


# --------------------------------------------------------------------------
# Verdict (charter "Verdict rule" — dual, both rules must hold)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RungOutcome:
    """One rung's dual-rule adjudication, with every number it rests on."""

    verdict: Verdict
    pair_speedups: tuple[float, ...]
    median_speedup: float | None
    minimum_speedup: float | None
    anchor_seconds: float | None
    anchor_over_l1_median: float | None
    l1_median_wall: float | None
    l2_median_wall: float | None
    live_rule_holds: bool
    anchor_rule_holds: bool
    failures: tuple[str, ...]


def pair_speedups(
    *,
    l1_walls: Sequence[float],
    l2_walls: Sequence[float],
) -> tuple[float, ...]:
    """Per-pair native/fused process-wall ratios, in pair order."""
    if len(l1_walls) != len(l2_walls):
        raise F3ContractError("each timed pair needs one wall from each lane.")
    ratios: list[float] = []
    for index, (l1_wall, l2_wall) in enumerate(zip(l1_walls, l2_walls, strict=True)):
        fused = _finite(l1_wall, f"l1_walls[{index}]")
        native = _finite(l2_wall, f"l2_walls[{index}]")
        if fused <= 0.0:
            raise F3ContractError(f"l1_walls[{index}] must be positive.")
        ratios.append(native / fused)
    return tuple(ratios)


def adjudicate_rung(
    *,
    l1_walls: Sequence[float],
    l2_walls: Sequence[float],
    anchor_seconds: float | None,
    not_produced_pairs: int = 0,
    gate_failures: Sequence[str] = (),
) -> RungOutcome:
    """Apply the dual verdict rule to one rung's completed pairs."""
    failures = list(gate_failures)
    if not_produced_pairs >= NOT_PRODUCED_ABORT:
        failures.append(f"rung_aborted_on_{not_produced_pairs}_not_produced_pairs")
    if len(l1_walls) != PAIR_COUNT or len(l2_walls) != PAIR_COUNT:
        failures.append(
            f"pair_count_{min(len(l1_walls), len(l2_walls))}_expected_{PAIR_COUNT}"
        )
    if failures:
        return RungOutcome(
            verdict=Verdict.NOT_PRODUCED,
            pair_speedups=(),
            median_speedup=None,
            minimum_speedup=None,
            anchor_seconds=anchor_seconds,
            anchor_over_l1_median=None,
            l1_median_wall=None,
            l2_median_wall=None,
            live_rule_holds=False,
            anchor_rule_holds=False,
            failures=tuple(failures),
        )

    speedups = pair_speedups(l1_walls=l1_walls, l2_walls=l2_walls)
    median_speedup = statistics.median(speedups)
    minimum_speedup = min(speedups)
    l1_median = statistics.median(_finite(w, "l1_wall") for w in l1_walls)
    l2_median = statistics.median(_finite(w, "l2_wall") for w in l2_walls)
    live_rule = (
        median_speedup >= WIN_MEDIAN_THRESHOLD and minimum_speedup > WIN_PAIR_THRESHOLD
    )
    if anchor_seconds is None:
        anchor_ratio = None
        anchor_rule = False
        failures.append("anchor_unavailable")
    else:
        anchor_ratio = _finite(anchor_seconds, "anchor_seconds") / l1_median
        anchor_rule = anchor_ratio >= WIN_MEDIAN_THRESHOLD
    return RungOutcome(
        verdict=(
            Verdict.WIN
            if live_rule and anchor_rule
            else Verdict.CLOSED_BOUNDED_NEGATIVE
        ),
        pair_speedups=speedups,
        median_speedup=median_speedup,
        minimum_speedup=minimum_speedup,
        anchor_seconds=anchor_seconds,
        anchor_over_l1_median=anchor_ratio,
        l1_median_wall=l1_median,
        l2_median_wall=l2_median,
        live_rule_holds=live_rule,
        anchor_rule_holds=anchor_rule,
        failures=tuple(failures),
    )


# --------------------------------------------------------------------------
# Cap accounting (charter "Caps and aborts")
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapLedger:
    """Campaign-level resource use, checked at every rung boundary."""

    timed_legs: int = 0
    solve_child_processes: int = 0
    campaign_wall_seconds: float = 0.0

    def with_legs(
        self,
        *,
        timed: int = 0,
        solve_children: int = 0,
        wall_seconds: float = 0.0,
    ) -> CapLedger:
        return CapLedger(
            timed_legs=self.timed_legs + timed,
            solve_child_processes=self.solve_child_processes + solve_children,
            campaign_wall_seconds=self.campaign_wall_seconds + wall_seconds,
        )

    def breaches(self) -> list[str]:
        """Name every cap this ledger has exceeded."""
        failures: list[str] = []
        if self.timed_legs > MAX_TIMED_LEGS:
            failures.append(f"timed_legs_{self.timed_legs}_over_{MAX_TIMED_LEGS}")
        if self.solve_child_processes > MAX_SOLVE_CHILD_PROCESSES:
            failures.append(
                f"solve_children_{self.solve_child_processes}"
                f"_over_{MAX_SOLVE_CHILD_PROCESSES}"
            )
        if self.campaign_wall_seconds > MAX_CAMPAIGN_WALL_SECONDS:
            failures.append(
                f"campaign_wall_{self.campaign_wall_seconds:.0f}s"
                f"_over_{MAX_CAMPAIGN_WALL_SECONDS:.0f}s"
            )
        return failures


def budget_search_breaches(
    *,
    probe_count: int,
    largest_maxiter: int,
    search_wall_seconds: float,
) -> list[str]:
    """BQ step 2 caps; any breach makes BQ NOT_PRODUCED."""
    failures: list[str] = []
    if probe_count > BQ_MAX_PROBES_PER_LANE:
        failures.append(f"probes_{probe_count}_over_{BQ_MAX_PROBES_PER_LANE}")
    if largest_maxiter > BQ_MAX_MAXITER:
        failures.append(f"maxiter_{largest_maxiter}_over_{BQ_MAX_MAXITER}")
    if search_wall_seconds > BQ_MAX_SEARCH_SECONDS:
        failures.append(
            f"search_wall_{search_wall_seconds:.0f}s_over_{BQ_MAX_SEARCH_SECONDS:.0f}s"
        )
    return failures


# --------------------------------------------------------------------------
# validate (charter "Validate entrypoint": recompute from the run dir alone)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """What ``validate`` recomputed, beside what the run directory recorded.

    Both numbers travel together on purpose: a reviewer reads the comparison,
    not a bare pass/fail.  ``findings`` is empty exactly when the directory
    validates.
    """

    run_dir: str
    rung: str
    budget: int
    row_count: int
    timed_pair_count: int
    recomputed_verdict: Verdict
    recorded_verdict: str | None
    recomputed_anchor_seconds: float | None
    median_speedup: float | None
    minimum_speedup: float | None
    anchor_over_l1_median: float | None
    live_rule_holds: bool
    anchor_rule_holds: bool
    findings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.findings

    def as_payload(self) -> dict[str, object]:
        """The report as JSON-serializable data for the CLI and receipts."""
        return {
            "run_dir": self.run_dir,
            "rung": self.rung,
            "budget": self.budget,
            "row_count": self.row_count,
            "timed_pair_count": self.timed_pair_count,
            "recomputed_verdict": self.recomputed_verdict.value,
            "recorded_verdict": self.recorded_verdict,
            "recomputed_anchor_seconds": self.recomputed_anchor_seconds,
            "median_speedup": self.median_speedup,
            "minimum_speedup": self.minimum_speedup,
            "anchor_over_l1_median": self.anchor_over_l1_median,
            "live_rule_holds": self.live_rule_holds,
            "anchor_rule_holds": self.anchor_rule_holds,
            "findings": list(self.findings),
            "valid": self.valid,
        }


def _load_json(path: Path, where: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise F3ContractError(f"{where} is not readable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise F3ContractError(f"{where} must be a JSON object: {path}")
    return payload


def _row_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/row.json"))


def validate_run_dir(run_dir: Path | str) -> ValidationReport:
    """Recompute every gate and the rung verdict from a run directory alone.

    Reads only the run directory: no charter file, no instrument, no network.
    Every recomputed number is returned beside the recorded one so a caller
    sees the comparison rather than a bare pass/fail.
    """
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json", "run manifest")
    if manifest.get("schema") != F3_RUN_MANIFEST_SCHEMA:
        raise F3ContractError(
            f"run manifest schema is {manifest.get('schema')!r}, "
            f"expected {F3_RUN_MANIFEST_SCHEMA!r}."
        )
    charter_sha = manifest.get("f3_charter_sha256")
    if charter_sha not in F3_CHARTER_LINEAGE:
        raise F3ContractError(
            "run manifest binds a charter sha outside the append-only lineage."
        )
    rung = str(manifest.get("rung", ""))
    budget = manifest.get("budget")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise F3ContractError("run manifest carries no positive integer budget.")

    findings: list[str] = []
    rows = [_load_json(path, f"row {path.parent.name}") for path in _row_paths(root)]
    if not rows:
        raise F3ContractError(f"run directory holds no rows: {root}")

    expected_policy_sha = policy_identity_sha256(budget)
    l1_walls: list[float] = []
    l2_walls: list[float] = []
    l1_nfev: list[int] = []
    l2_compact: list[int] = []
    for row in rows:
        label = f"{row.get('lane')}/{row.get('role')}"
        if row.get("schema") != F3_ROW_SCHEMA:
            findings.append(f"{label}: foreign row schema {row.get('schema')!r}")
            continue
        policy = row.get("policy")
        if not isinstance(policy, dict):
            findings.append(f"{label}: row carries no policy block")
            continue
        recomputed = observed_policy_sha256(policy)
        if recomputed != row.get("policy_identity_sha256"):
            findings.append(f"{label}: recorded policy sha differs from its policy")
        if recomputed != expected_policy_sha:
            findings.append(f"{label}: policy differs from the rung constant")
        contract = f3_contract_sha256(
            campaign_manifest_sha256=str(row.get("campaign_input_manifest_sha256")),
            budget=budget,
            production_commit=str(row.get("production_commit")),
            instrument_commit=str(row.get("instrument_commit")),
            fair_bar_charter_sha256=str(row.get("fair_bar_charter_sha256")),
            charter_sha256=str(charter_sha),
        )
        if contract != row.get("campaign_contract_sha256"):
            findings.append(f"{label}: row is bound to a foreign contract sha")
        if not row.get("timed", False):
            continue
        wall = _finite(row.get("process_wall_seconds"), f"{label}.process_wall")
        if row.get("lane") == L1_LANE:
            l1_walls.append(wall)
            l1_nfev.append(_count(row.get("evaluation_count"), f"{label}.count"))
        elif row.get("lane") == L2_LANE:
            l2_walls.append(wall)
            l2_compact.append(_count(row.get("evaluation_count"), f"{label}.count"))

    for index, (nfev, compact) in enumerate(zip(l1_nfev, l2_compact, strict=False)):
        liveness = counter_liveness_failures(
            l1_nfev=nfev, l2_compact_evaluations=compact
        )
        findings.extend(f"pair{index}: {failure}" for failure in liveness)

    recorded_anchor = manifest.get("anchor_process_wall_seconds")
    anchor = None if recorded_anchor is None else _finite(recorded_anchor, "anchor")
    if rung == "b3" and anchor is not None and anchor != b3_anchor():
        findings.append("recorded B3 anchor is not the archived process wall")
    if rung == "b37" and l1_nfev and l2_compact:
        recomputed_anchor = b37_anchor(
            l2_compact_evaluations=l2_compact, l1_nfev=l1_nfev
        )
        if anchor is not None and not math.isclose(
            anchor, recomputed_anchor, rel_tol=1.0e-12
        ):
            findings.append("recorded B37 anchor differs from the charter formula")

    outcome = adjudicate_rung(
        l1_walls=l1_walls,
        l2_walls=l2_walls,
        anchor_seconds=anchor,
        not_produced_pairs=_count(
            manifest.get("not_produced_pairs"), "manifest.not_produced_pairs"
        ),
        gate_failures=_string_list(
            manifest.get("gate_failures"), "manifest.gate_failures"
        ),
    )
    recorded_verdict = manifest.get("verdict")
    if recorded_verdict != outcome.verdict.value:
        findings.append(
            f"recorded verdict {recorded_verdict!r} != recomputed "
            f"{outcome.verdict.value!r}"
        )
    ledger = CapLedger(
        timed_legs=_count(
            manifest.get("timed_legs"),
            "manifest.timed_legs",
            default=len(l1_walls) + len(l2_walls),
        ),
        solve_child_processes=_count(
            manifest.get("solve_child_processes"), "manifest.solve_child_processes"
        ),
        campaign_wall_seconds=_seconds(
            manifest.get("campaign_wall_seconds"), "manifest.campaign_wall_seconds"
        ),
    )
    findings.extend(ledger.breaches())
    return ValidationReport(
        run_dir=str(root),
        rung=rung,
        budget=budget,
        row_count=len(rows),
        timed_pair_count=min(len(l1_walls), len(l2_walls)),
        recomputed_verdict=outcome.verdict,
        recorded_verdict=(None if recorded_verdict is None else str(recorded_verdict)),
        recomputed_anchor_seconds=outcome.anchor_seconds,
        median_speedup=outcome.median_speedup,
        minimum_speedup=outcome.minimum_speedup,
        anchor_over_l1_median=outcome.anchor_over_l1_median,
        live_rule_holds=outcome.live_rule_holds,
        anchor_rule_holds=outcome.anchor_rule_holds,
        findings=tuple(findings),
    )


__all__ = [
    "ARCHIVED_B3_PROCESS_WALL",
    "ARCHIVED_STEADY_PER_EVAL",
    "BQ_MAX_MAXITER",
    "BQ_MAX_PROBES_PER_LANE",
    "BQ_MAX_SEARCH_SECONDS",
    "BQ_SEARCH_START",
    "F3_CHARTER_COMMIT",
    "F3_CHARTER_LINEAGE",
    "F3_CHARTER_PATH",
    "F3_CHARTER_SHA256",
    "F3_ROW_SCHEMA",
    "F3_RUN_MANIFEST_SCHEMA",
    "GRADIENT_FACTOR_K",
    "L1_LANE",
    "L2_LANE",
    "L3_LANE",
    "MAXFUN_MULTIPLIER",
    "MAX_CAMPAIGN_WALL_SECONDS",
    "MAX_SOLVE_CHILD_PROCESSES",
    "MAX_TIMED_LEGS",
    "NATIVE_MAXFUN_SOURCE",
    "NATIVE_SWEEP_OMP_MATRIX",
    "NATIVE_SWEEP_REPS",
    "NOT_PRODUCED_ABORT",
    "PAIR_COUNT",
    "POLICY_FIELDS",
    "POLICY_GTOL",
    "POLICY_MAXCOR",
    "POLICY_MAXLS",
    "QUALITY_OBJECTIVE_RTOL",
    "RUNG_BUDGETS",
    "WIN_MEDIAN_THRESHOLD",
    "WIN_PAIR_THRESHOLD",
    "CapLedger",
    "F3ContractError",
    "RungOutcome",
    "ValidationReport",
    "Verdict",
    "adjudicate_rung",
    "b3_anchor",
    "b37_anchor",
    "bq_anchor",
    "bq_quality_failures",
    "budget_search_breaches",
    "counter_liveness_failures",
    "f3_contract_sha256",
    "fixed_budget_quality_failures",
    "observed_policy_sha256",
    "pair_speedups",
    "policy_identity_failures",
    "policy_identity_sha256",
    "policy_payload",
    "validate_run_dir",
]
