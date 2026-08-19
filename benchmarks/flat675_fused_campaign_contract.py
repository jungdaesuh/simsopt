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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Frozen campaign identity
# --------------------------------------------------------------------------

F3_CHARTER_FREEZE_SHA256: Final[str] = (
    "0a61ed647afc08424a149a06a6e247535d4da931136bc5d2294874634b9564dc"
)
# Amendment 1 (2026-08-19, pre-evidence): the endpoint inner-state computation
# is charged inside the L1 primary timer.
F3_CHARTER_SHA256: Final[str] = (
    "b710ff423667b7fa3c2d9e194ee1e3ccca94ed4821df7c9081fb4deb76e298d2"
)
F3_CHARTER_COMMIT: Final[str] = "595b7da60"
F3_CHARTER_PATH: Final[str] = (
    "docs/jax_gpu_flat675_fused_campaign_plan.md (pr/jax-port-squashed)"
)
# Append-only: the sha of the charter bytes at each amendment commit, seeded
# with the freeze.  Rows bind the sha current when they executed; validate
# accepts any lineage member and recomputes against the row's own sha.
F3_CHARTER_LINEAGE: Final[tuple[str, ...]] = (
    F3_CHARTER_FREEZE_SHA256,
    F3_CHARTER_SHA256,
)

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
    # The charter's BQ clause is "both endpoints' oracle objectives <= Q*",
    # with no tolerance: the 1e-10 belongs to the fixed-budget gate alone.
    for name, value in (
        ("l1", _finite(l1_oracle_objective, "l1_oracle_objective")),
        ("l2", _finite(l2_oracle_objective, "l2_oracle_objective")),
    ):
        if value > target:
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


F3_ROW_DIRECTORY: Final[str] = "rows"


def row_paths(run_dir: Path) -> list[Path]:
    """Every F3 row in a run directory.

    F3 rows live in their own ``rows/`` subtree rather than beside a leg's
    output: the fair-bar ``run_leg`` writes its own ``row.json`` into each
    native leg directory, and a glob over leg directories would read that
    foreign row instead of this campaign's.
    """
    return sorted((Path(run_dir) / F3_ROW_DIRECTORY).glob("*.json"))


@dataclass(frozen=True, slots=True)
class RowEvidence:
    """One leg row, parsed into the fields every gate and anchor reads."""

    lane: str
    role: str
    rung: str
    pair_index: int
    budget: int
    timed: bool
    process_wall_seconds: float
    evaluation_count: int
    nit: int
    policy: Mapping[str, object]
    policy_identity_sha256: str
    oracle_objective: float | None
    oracle_gradient_inf_norm: float | None
    campaign_input_manifest_sha256: str
    production_commit: str
    instrument_commit: str
    fair_bar_charter_sha256: str
    campaign_contract_sha256: str
    host_transfer_ledger: Mapping[str, int]
    source: str


def parse_row(payload: Mapping[str, object], *, source: str) -> RowEvidence:
    """Read one row, refusing a foreign schema rather than partially trusting."""
    if payload.get("schema") != F3_ROW_SCHEMA:
        raise F3ContractError(
            f"{source}: row schema is {payload.get('schema')!r}, "
            f"expected {F3_ROW_SCHEMA!r}."
        )
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        raise F3ContractError(f"{source}: row carries no policy block.")
    oracle_objective = payload.get("oracle_objective")
    oracle_gradient = payload.get("oracle_gradient_inf_norm")
    ledger = payload.get("host_transfer_ledger")
    if ledger is not None and not isinstance(ledger, dict):
        raise F3ContractError(f"{source}: host_transfer_ledger must be an object.")
    return RowEvidence(
        lane=str(payload.get("lane")),
        role=str(payload.get("role")),
        rung=str(payload.get("rung")),
        pair_index=_count(payload.get("pair_index"), f"{source}.pair_index"),
        budget=_positive_int(payload.get("budget"), f"{source}.budget"),
        timed=bool(payload.get("timed", False)),
        process_wall_seconds=_finite(
            payload.get("process_wall_seconds"), f"{source}.process_wall_seconds"
        ),
        evaluation_count=_count(
            payload.get("evaluation_count"), f"{source}.evaluation_count"
        ),
        nit=_count(payload.get("nit"), f"{source}.nit"),
        policy=policy,
        policy_identity_sha256=str(payload.get("policy_identity_sha256")),
        oracle_objective=(
            None
            if oracle_objective is None
            else _finite(oracle_objective, f"{source}.oracle_objective")
        ),
        oracle_gradient_inf_norm=(
            None
            if oracle_gradient is None
            else _finite(oracle_gradient, f"{source}.oracle_gradient_inf_norm")
        ),
        campaign_input_manifest_sha256=str(
            payload.get("campaign_input_manifest_sha256")
        ),
        production_commit=str(payload.get("production_commit")),
        instrument_commit=str(payload.get("instrument_commit")),
        fair_bar_charter_sha256=str(payload.get("fair_bar_charter_sha256")),
        campaign_contract_sha256=str(payload.get("campaign_contract_sha256")),
        host_transfer_ledger={
            str(phase): _count(calls, f"{source}.host_transfer_ledger.{phase}")
            for phase, calls in (ledger or {}).items()
        },
        source=source,
    )


@dataclass(frozen=True, slots=True)
class PairEvidence:
    """One interleaved pair's two timed rows."""

    index: int
    l1: RowEvidence
    l2: RowEvidence


def pair_rows(rows: Sequence[RowEvidence]) -> tuple[PairEvidence, ...]:
    """Group timed rows into pairs; an unpaired timed row is a contract error."""
    by_index: dict[int, dict[str, RowEvidence]] = {}
    for row in rows:
        if not row.timed:
            continue
        if row.lane not in (L1_LANE, L2_LANE):
            continue
        slot = by_index.setdefault(row.pair_index, {})
        if row.lane in slot:
            raise F3ContractError(f"pair {row.pair_index} has two {row.lane} rows.")
        slot[row.lane] = row
    pairs: list[PairEvidence] = []
    for index in sorted(by_index):
        slot = by_index[index]
        if L1_LANE not in slot or L2_LANE not in slot:
            raise F3ContractError(
                f"pair {index} is missing its "
                f"{L2_LANE if L1_LANE in slot else L1_LANE} row."
            )
        pairs.append(PairEvidence(index=index, l1=slot[L1_LANE], l2=slot[L2_LANE]))
    return tuple(pairs)


def pair_gate_failures(
    pair: PairEvidence,
    *,
    rung: str,
    quality_target: float | None,
) -> list[str]:
    """Recompute every per-pair gate from the pair's own recorded evidence.

    This is the only place the gates are evaluated: the producer calls it
    before publishing and ``validate`` calls it again from the rows alone, so
    a manifest can never assert a gate result its rows do not support.
    """
    failures = [
        f"pair{pair.index}: {failure}"
        for failure in counter_liveness_failures(
            l1_nfev=pair.l1.evaluation_count,
            l2_compact_evaluations=pair.l2.evaluation_count,
        )
    ]
    # Work matching is fail-closed on BOTH lanes: each must have spent the
    # budget its own row declares.
    for row, label in ((pair.l1, "l1"), (pair.l2, "l2")):
        if row.nit != row.budget:
            failures.append(
                f"pair{pair.index}: {label}_nit_{row.nit}_expected_{row.budget}"
            )
    # The fused lane's claim is a device-resident solve; a host round trip per
    # step would make it a different program than the one under test.
    for phase in ("advance", "callback", "unclassified"):
        if pair.l1.host_transfer_ledger.get(phase, 0):
            failures.append(f"pair{pair.index}: l1_host_{phase}_transfers")
    if pair.l1.oracle_objective is None or pair.l2.oracle_objective is None:
        failures.append(f"pair{pair.index}: oracle_objective_absent")
        return failures
    if (
        pair.l1.oracle_gradient_inf_norm is None
        or pair.l2.oracle_gradient_inf_norm is None
    ):
        failures.append(f"pair{pair.index}: oracle_gradient_absent")
        return failures
    if rung == "bq":
        if quality_target is None:
            failures.append(f"pair{pair.index}: bq_quality_target_absent")
            return failures
        quality = bq_quality_failures(
            l1_oracle_objective=pair.l1.oracle_objective,
            l2_oracle_objective=pair.l2.oracle_objective,
            l1_oracle_gradient_inf=pair.l1.oracle_gradient_inf_norm,
            l2_oracle_gradient_inf=pair.l2.oracle_gradient_inf_norm,
            quality_target=quality_target,
        )
    else:
        quality = fixed_budget_quality_failures(
            l1_oracle_objective=pair.l1.oracle_objective,
            l2_oracle_objective=pair.l2.oracle_objective,
            l1_oracle_gradient_inf=pair.l1.oracle_gradient_inf_norm,
            l2_oracle_gradient_inf=pair.l2.oracle_gradient_inf_norm,
        )
    failures.extend(f"pair{pair.index}: {failure}" for failure in quality)
    return failures


def rung_anchor(pairs: Sequence[PairEvidence], *, rung: str) -> float | None:
    """The rung's chartered anchor, computed from its passing pairs."""
    if rung == "b3":
        return b3_anchor()
    if not pairs:
        return None
    if rung == "b37":
        return b37_anchor(
            l2_compact_evaluations=[pair.l2.evaluation_count for pair in pairs],
            l1_nfev=[pair.l1.evaluation_count for pair in pairs],
        )
    if rung == "bq":
        return bq_anchor(
            l2_compact_evaluations_at_nstar=[pair.l2.evaluation_count for pair in pairs]
        )
    return None


def adjudicate_rows(
    rows: Sequence[RowEvidence],
    *,
    rung: str,
    quality_target: float | None = None,
) -> tuple[RungOutcome, tuple[PairEvidence, ...]]:
    """The single adjudication rule, shared by the producer and ``validate``.

    Gates are recomputed from the rows; only pairs that pass every gate carry
    timings into the verdict, and the pairs they displace are counted as
    ``NOT_PRODUCED`` exactly as the charter's abort rule requires.
    """
    pairs = pair_rows(rows)
    failures: list[str] = []
    passing: list[PairEvidence] = []
    for pair in pairs:
        pair_failures = pair_gate_failures(
            pair, rung=rung, quality_target=quality_target
        )
        if pair_failures:
            failures.extend(pair_failures)
        else:
            passing.append(pair)
    outcome = adjudicate_rung(
        l1_walls=[pair.l1.process_wall_seconds for pair in passing],
        l2_walls=[pair.l2.process_wall_seconds for pair in passing],
        anchor_seconds=rung_anchor(passing, rung=rung),
        not_produced_pairs=len(pairs) - len(passing),
        gate_failures=failures,
    )
    return outcome, pairs


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
    quality_target = (
        None
        if manifest.get("quality_target") is None
        else _finite(manifest["quality_target"], "manifest.quality_target")
    )

    paths = row_paths(root)
    if not paths:
        raise F3ContractError(f"run directory holds no rows: {root}")
    rows = [
        parse_row(_load_json(path, f"row {path.stem}"), source=path.stem)
        for path in paths
    ]

    findings: list[str] = []
    for row in rows:
        # Each lane runs its own budget (BQ pairs fused m* against native n*),
        # so the policy and contract a row must satisfy come from that row.
        recomputed_policy = observed_policy_sha256(row.policy)
        if recomputed_policy != row.policy_identity_sha256:
            findings.append(
                f"{row.source}: recorded policy sha differs from its policy"
            )
        if recomputed_policy != policy_identity_sha256(row.budget):
            findings.append(
                f"{row.source}: policy differs from the constant for its budget"
            )
        contract = f3_contract_sha256(
            campaign_manifest_sha256=row.campaign_input_manifest_sha256,
            budget=row.budget,
            production_commit=row.production_commit,
            instrument_commit=row.instrument_commit,
            fair_bar_charter_sha256=row.fair_bar_charter_sha256,
            charter_sha256=str(charter_sha),
        )
        if contract != row.campaign_contract_sha256:
            findings.append(f"{row.source}: row is bound to a foreign contract sha")
        # Identity must agree between the manifest and every row it covers.
        for field, recorded in (
            ("production_commit", row.production_commit),
            ("instrument_commit", row.instrument_commit),
            ("campaign_input_manifest_sha256", row.campaign_input_manifest_sha256),
        ):
            expected = manifest.get(field)
            if expected is not None and str(expected) != recorded:
                findings.append(
                    f"{row.source}: {field} {recorded!r} differs from the "
                    f"manifest's {expected!r}"
                )

    outcome, pairs = adjudicate_rows(rows, rung=rung, quality_target=quality_target)
    findings.extend(outcome.failures)

    recorded_anchor = manifest.get("anchor_process_wall_seconds")
    if (
        recorded_anchor is not None
        and outcome.anchor_seconds is not None
        and not math.isclose(
            _finite(recorded_anchor, "manifest.anchor"),
            outcome.anchor_seconds,
            rel_tol=1.0e-12,
        )
    ):
        findings.append(
            "recorded anchor differs from the charter formula for this rung"
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
            default=2 * len(pairs),
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
        budget=_count(manifest.get("budget"), "manifest.budget"),
        row_count=len(rows),
        timed_pair_count=len(pairs),
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


# --------------------------------------------------------------------------
# BQ budget search and the native sweep, as pure algorithms
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BudgetSearch:
    """One lane's minimal-budget search: its probes, caps, and result."""

    probes: tuple[Mapping[str, object], ...]
    breaches: tuple[str, ...]
    star: int | None

    def as_payload(self) -> dict[str, object]:
        return {
            "probes": [dict(probe) for probe in self.probes],
            "breaches": list(self.breaches),
            "star": self.star,
        }


def search_minimal_budget(
    probe: Callable[[int], float],
    *,
    quality_target: float,
    elapsed_seconds: Callable[[], float] = lambda: 0.0,
    start: int = BQ_SEARCH_START,
) -> BudgetSearch:
    """Smallest budget whose probe reaches the target: double up, then bisect.

    ``probe`` returns the ORACLE-evaluated endpoint objective at a budget; a
    lane's self-reported objective is never admissible here.  The caps are
    checked before each probe so a breach costs nothing further.
    """
    probes: list[Mapping[str, object]] = []
    maxiter = start
    low: int | None = None
    high: int | None = None
    while True:
        breaches = budget_search_breaches(
            probe_count=len(probes) + 1,
            largest_maxiter=maxiter,
            search_wall_seconds=elapsed_seconds(),
        )
        if breaches:
            return BudgetSearch(
                probes=tuple(probes), breaches=tuple(breaches), star=None
            )
        objective = probe(maxiter)
        reached = objective <= quality_target
        probes.append(
            {
                "maxiter": maxiter,
                "oracle_objective": objective,
                "reached_target": reached,
            }
        )
        if reached:
            high = maxiter
        else:
            low = maxiter
        if high is None:
            maxiter *= 2
            continue
        # A start that already reaches must still be bisected DOWNWARD: the
        # charter's n* is <= 37 by construction, so returning the start would
        # report a budget larger than the smallest one that reaches.
        lower_bound = 0 if low is None else low
        if high - lower_bound <= 1:
            return BudgetSearch(probes=tuple(probes), breaches=(), star=high)
        maxiter = (lower_bound + high) // 2


def select_sweep_config(medians: Mapping[str, float]) -> str:
    """The fair-bar selection rule: the config with the smallest median wall."""
    if not medians:
        raise F3ContractError("a native sweep must measure at least one config.")
    return min(sorted(medians), key=lambda label: medians[label])


# --------------------------------------------------------------------------
# Campaign state (charter "Caps and aborts": the rung-boundary breach rule)
# --------------------------------------------------------------------------

CAMPAIGN_STATE_SCHEMA: Final[str] = "flat675-fused-campaign-state.v1"
# Per warm pair: two primers, two timed children, two oracle children.
SOLVE_CHILDREN_PER_WARM_PAIR: Final[int] = 6
# A cold pair runs no primers.
SOLVE_CHILDREN_PER_COLD_PAIR: Final[int] = 4


@dataclass(frozen=True, slots=True)
class CampaignState:
    """Cumulative campaign resource use and each rung's recorded fate."""

    ledger: CapLedger = CapLedger()
    completed_rungs: tuple[str, ...] = ()
    stopped: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CampaignState:
        if payload.get("schema") != CAMPAIGN_STATE_SCHEMA:
            raise F3ContractError(
                f"campaign state schema is {payload.get('schema')!r}."
            )
        return cls(
            ledger=CapLedger(
                timed_legs=_count(payload.get("timed_legs"), "state.timed_legs"),
                solve_child_processes=_count(
                    payload.get("solve_child_processes"), "state.solve_children"
                ),
                campaign_wall_seconds=_seconds(
                    payload.get("campaign_wall_seconds"), "state.campaign_wall"
                ),
            ),
            completed_rungs=tuple(
                _string_list(payload.get("completed_rungs"), "state.completed_rungs")
            ),
            stopped=bool(payload.get("stopped", False)),
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_STATE_SCHEMA,
            "timed_legs": self.ledger.timed_legs,
            "solve_child_processes": self.ledger.solve_child_processes,
            "campaign_wall_seconds": self.ledger.campaign_wall_seconds,
            "completed_rungs": list(self.completed_rungs),
            "stopped": self.stopped,
        }

    def admits(
        self, *, timed: int, solve_children: int, wall_seconds: float = 0.0
    ) -> list[str]:
        """Cap breaches a rung of this size would cause if it were started.

        The charter ends the campaign at a rung BOUNDARY, so this is asked
        before a rung begins and never mid-rung.
        """
        if self.stopped:
            return ["campaign_already_stopped_at_a_rung_boundary"]
        return self.ledger.with_legs(
            timed=timed, solve_children=solve_children, wall_seconds=wall_seconds
        ).breaches()

    def completing(
        self,
        rung: str,
        *,
        timed: int,
        solve_children: int,
        wall_seconds: float = 0.0,
    ) -> CampaignState:
        """Record one finished rung's cost, stopping if it exhausted a cap."""
        ledger = self.ledger.with_legs(
            timed=timed, solve_children=solve_children, wall_seconds=wall_seconds
        )
        return CampaignState(
            ledger=ledger,
            completed_rungs=(*self.completed_rungs, rung),
            stopped=bool(ledger.breaches()),
        )


__all__ = [
    "ARCHIVED_B3_PROCESS_WALL",
    "ARCHIVED_STEADY_PER_EVAL",
    "BQ_MAX_MAXITER",
    "BQ_MAX_PROBES_PER_LANE",
    "BQ_MAX_SEARCH_SECONDS",
    "BQ_SEARCH_START",
    "CAMPAIGN_STATE_SCHEMA",
    "F3_CHARTER_COMMIT",
    "F3_CHARTER_FREEZE_SHA256",
    "F3_CHARTER_LINEAGE",
    "F3_CHARTER_PATH",
    "F3_CHARTER_SHA256",
    "F3_ROW_DIRECTORY",
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
    "SOLVE_CHILDREN_PER_COLD_PAIR",
    "SOLVE_CHILDREN_PER_WARM_PAIR",
    "WIN_MEDIAN_THRESHOLD",
    "WIN_PAIR_THRESHOLD",
    "BudgetSearch",
    "CampaignState",
    "CapLedger",
    "F3ContractError",
    "PairEvidence",
    "RowEvidence",
    "RungOutcome",
    "ValidationReport",
    "Verdict",
    "adjudicate_rows",
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
    "pair_gate_failures",
    "pair_rows",
    "pair_speedups",
    "parse_row",
    "policy_identity_failures",
    "policy_identity_sha256",
    "policy_payload",
    "row_paths",
    "rung_anchor",
    "search_minimal_budget",
    "select_sweep_config",
    "validate_run_dir",
]
