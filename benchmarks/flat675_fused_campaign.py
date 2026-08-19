"""Flat-675 fused single-stage campaign harness (F3).

Charter (frozen): ``docs/jax_gpu_flat675_fused_campaign_plan.md`` on
pr/jax-port-squashed at commit b7ec63b6e; its sha and the charter arithmetic
live in :mod:`benchmarks.flat675_fused_campaign_contract`, which this module
imports rather than restates.

This orchestrator owns sequencing, environment construction, priming, gating,
and evidence for three lanes:

* **L1 (verdict)** — the fused GPU lane, one production-tree child per leg
  (``flat675_fused_lane_child.py``), pinned by that child's import-origin
  guard.
* **L2 (verdict)** — the native C++ CPU lane, run bit-identically to the
  fair-bar campaign by delegating to its ``run_leg``.
* **Oracle** — the fair-bar native cross-evaluator, which adjudicates every
  timed endpoint.

Everything lane-agnostic is reused from ``genuine_675_fair_bar`` by import:
the partition-integrity gate, the child-observed conformance gate, the
provenance shim, the frozen-bundle campaign-manifest validator, the native
environment and leg runner, and the oracle wiring.

That reuse fixes how this file must be invoked.  ``genuine_675_fair_bar``
resolves its ``simsopt_jax`` imports only from the pinned instrument tree,
while the ``benchmarks`` package containing both files must resolve from THIS
tree — so the production root comes first and the instrument's ``src`` supplies
``simsopt_jax``::

    PYTHONPATH=<production-root>:<instrument-root>:<instrument-root>/src \\
        <runtime-env>/bin/python \\
        <production-root>/benchmarks/flat675_fused_campaign.py <subcommand>

The L1 children are launched with a production-only ``PYTHONPATH`` of their
own, so the instrument tree never follows the harness into the lane under test.

Subcommands: ``pairs``, ``budget-search``, ``pairs-bq``, ``cold-pair``,
``native-sweep``, ``validate``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from benchmarks.flat675_fused_campaign_contract import (
    BQ_MAX_MAXITER,
    BQ_MAX_PROBES_PER_LANE,
    BQ_MAX_SEARCH_SECONDS,
    BQ_SEARCH_START,
    F3_CHARTER_COMMIT,
    F3_CHARTER_LINEAGE,
    F3_CHARTER_SHA256,
    F3_ROW_SCHEMA,
    F3_RUN_MANIFEST_SCHEMA,
    L1_LANE,
    L2_LANE,
    NATIVE_MAXFUN_SOURCE,
    NATIVE_SWEEP_OMP_MATRIX,
    NATIVE_SWEEP_REPS,
    PAIR_COUNT,
    RUNG_BUDGETS,
    CapLedger,
    F3ContractError,
    Verdict,
    adjudicate_rung,
    b3_anchor,
    b37_anchor,
    bq_anchor,
    bq_quality_failures,
    budget_search_breaches,
    counter_liveness_failures,
    f3_contract_sha256,
    fixed_budget_quality_failures,
    observed_policy_sha256,
    policy_identity_failures,
    policy_payload,
    validate_run_dir,
)

# The fair-bar harness is the lane-agnostic machinery this campaign reuses; it
# is imported, never copied and never modified.
from benchmarks.genuine_675_fair_bar import (
    CHARTER_SHA256 as FAIR_BAR_CHARTER_SHA256,
)
from benchmarks.genuine_675_fair_bar import (
    INSTRUMENT_COMMIT,
    NativeConfig,
    enforce_child_conformance,
    gpu_environment,
    load_campaign_manifest,
    native_environment,
    partition_integrity_gate,
    run_leg,
    run_oracle,
    write_provenance_shim,
)
from benchmarks.genuine_675_fair_bar import (
    SOURCE_ROOT as INSTRUMENT_ROOT,
)

PRODUCTION_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("/home/jungdaesuh/simsopt_mixed_artifacts/flat675_fused_campaign")
L1_CHILD = PRODUCTION_ROOT / "benchmarks" / "flat675_fused_lane_child.py"

# The charter's B3 native denominator is the fair-bar B3 matrix's own
# selection; B37's is that matrix's selection, resolved at execution.
B3_NATIVE_CONFIG = "omp16"


# --------------------------------------------------------------------------
# Tree identity (charter: clean-tree requirement, recorded per row)
# --------------------------------------------------------------------------


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *arguments),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def tree_identity(root: Path, *, label: str) -> dict[str, object]:
    """Commit and dirty-file count of one tree, recorded on every row."""
    dirty = [line for line in _git(root, "status", "--porcelain").splitlines() if line]
    return {
        "label": label,
        "commit": _git(root, "rev-parse", "HEAD"),
        "dirty_file_count": len(dirty),
    }


def require_clean_trees() -> dict[str, object]:
    """Both trees clean at their pinned commits before any timed leg."""
    production = tree_identity(PRODUCTION_ROOT, label="production")
    instrument = tree_identity(INSTRUMENT_ROOT, label="instrument")
    if instrument["commit"] != INSTRUMENT_COMMIT:
        raise F3ContractError(
            f"instrument tree is at {instrument['commit']}, expected "
            f"{INSTRUMENT_COMMIT}."
        )
    for identity in (production, instrument):
        if identity["dirty_file_count"]:
            raise F3ContractError(
                f"{identity['label']} tree has {identity['dirty_file_count']} "
                "dirty files; every timed row must bind a clean tree."
            )
    return {"production": production, "instrument": instrument}


# --------------------------------------------------------------------------
# L1 legs (fused GPU, production-tree children)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FusedLegResult:
    """One fused child's reported work, timing, and endpoint."""

    budget: int
    timed: bool
    role: str
    process_wall_seconds: float
    endpoint_inner_state_seconds: float
    nfev: int
    nit: int
    objective_value: float
    endpoint_candidate: Mapping[str, object]
    endpoint_inner_state: tuple[float, float]
    host_transfer_ledger: Mapping[str, int]
    policy: Mapping[str, object]
    provenance: Mapping[str, object]


def run_fused_leg(
    *,
    budget: int,
    leg_root: Path,
    role: str,
    timed: bool,
    cache_dir: Path,
    shim_dir: Path,
    source_manifest: Path,
    affinity: tuple[int, ...] | None,
) -> FusedLegResult:
    """Run one fused child and read its record; never adjudicates."""
    leg_root.mkdir(parents=True, exist_ok=False)
    output_path = leg_root / "lane.json"
    provenance_out = leg_root / "child_provenance.json"
    environment = gpu_environment(cache_dir=cache_dir, shim_dir=shim_dir)
    environment["FAIR_BAR_PROVENANCE_OUT"] = str(provenance_out)
    # The child imports the flat-675 program from THIS tree; the instrument
    # PYTHONPATH this harness runs under must not follow it into the child.
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(shim_dir), str(PRODUCTION_ROOT), str(PRODUCTION_ROOT / "src"))
    )
    command = (
        sys.executable,
        str(L1_CHILD),
        "--input-manifest",
        str(source_manifest),
        "--output-json",
        str(output_path),
        "--maxiter",
        str(budget),
        "--role",
        role,
        "--expected-charter-sha256",
        F3_CHARTER_SHA256,
    )
    if timed:
        partition_integrity_gate()

    def _preexec() -> None:
        if affinity is not None:
            os.sched_setaffinity(0, affinity)

    with (leg_root / "stdout.log").open("xb") as stdout, (leg_root / "stderr.log").open(
        "xb"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=PRODUCTION_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            preexec_fn=_preexec if affinity is not None else None,
            check=False,
        )
    if completed.returncode != 0:
        raise F3ContractError(
            f"fused {role} leg failed: {(leg_root / 'stderr.log').read_text()[-2000:]}"
        )
    payload = json.loads(output_path.read_text())
    provenance = (
        json.loads(provenance_out.read_text()) if provenance_out.is_file() else {}
    )
    enforce_child_conformance(
        lane=L1_LANE,
        config_label="fused",
        omp_threads=None,
        affinity=affinity,
        provenance=provenance,
    )
    result = payload["result"]
    return FusedLegResult(
        budget=budget,
        timed=timed,
        role=role,
        process_wall_seconds=float(payload["process_wall_seconds"]),
        endpoint_inner_state_seconds=float(payload["endpoint_inner_state_seconds"]),
        nfev=int(result["nfev"]),
        nit=int(result["nit"]),
        objective_value=float(result["objective_value"]),
        endpoint_candidate=result["endpoint_candidate"],
        endpoint_inner_state=(
            float(result["endpoint_inner_state"][0]),
            float(result["endpoint_inner_state"][1]),
        ),
        host_transfer_ledger=dict(payload.get("host_transfer_ledger", {})),
        policy=payload["policy"],
        provenance=provenance,
    )


# --------------------------------------------------------------------------
# Rows (charter "Governance": every row binds the full contract)
# --------------------------------------------------------------------------


def _write_row(
    path: Path,
    *,
    lane: str,
    role: str,
    rung: str,
    budget: int,
    timed: bool,
    process_wall_seconds: float,
    evaluation_count: int,
    counter_name: str,
    nit: int,
    policy: Mapping[str, object],
    oracle: Mapping[str, object] | None,
    endpoint_inner_state: Sequence[float],
    trees: Mapping[str, Mapping[str, object]],
    campaign_manifest_sha256: str,
    extra: Mapping[str, object],
) -> Mapping[str, object]:
    row = {
        "schema": F3_ROW_SCHEMA,
        "lane": lane,
        "role": role,
        "rung": rung,
        "budget": budget,
        "timed": timed,
        "process_wall_seconds": process_wall_seconds,
        "evaluation_count": evaluation_count,
        "evaluation_counter_name": counter_name,
        "nit": nit,
        "policy": dict(policy),
        "policy_identity_sha256": observed_policy_sha256(policy),
        "policy_identity_failures": policy_identity_failures(policy, budget=budget),
        "endpoint_inner_state": list(endpoint_inner_state),
        "oracle_objective": (None if oracle is None else _oracle_objective(oracle)),
        "oracle_gradient_inf_norm": (
            None if oracle is None else _oracle_gradient_inf(oracle)
        ),
        "f3_charter_sha256": F3_CHARTER_SHA256,
        "fair_bar_charter_sha256": FAIR_BAR_CHARTER_SHA256,
        "campaign_input_manifest_sha256": campaign_manifest_sha256,
        "production_commit": trees["production"]["commit"],
        "instrument_commit": trees["instrument"]["commit"],
        "campaign_contract_sha256": f3_contract_sha256(
            campaign_manifest_sha256=campaign_manifest_sha256,
            budget=budget,
            production_commit=str(trees["production"]["commit"]),
            instrument_commit=str(trees["instrument"]["commit"]),
            fair_bar_charter_sha256=FAIR_BAR_CHARTER_SHA256,
        ),
        "git": {name: dict(value) for name, value in trees.items()},
        **dict(extra),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    return row


# --------------------------------------------------------------------------
# Run directories
# --------------------------------------------------------------------------


def _new_run_root(phase: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = OUTPUT_ROOT / f"{stamp}-{phase}-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _prepare_run(phase: str, source_manifest: Path) -> tuple[Path, Path, str, dict]:
    trees = require_clean_trees()
    root = _new_run_root(phase)
    shim_dir = write_provenance_shim(root)
    campaign_manifest_sha = load_campaign_manifest(
        source_manifest, source_manifest.parent
    )
    return root, shim_dir, campaign_manifest_sha, trees


def _finish_run(root: Path, manifest: Mapping[str, object]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


# --------------------------------------------------------------------------
# Pair phases (charter "Verdict rule", "Priming and cold-start")
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairOutcome:
    """One interleaved pair's timings, counters, and gate result."""

    index: int
    l1_process_wall_seconds: float
    l2_process_wall_seconds: float
    l1_nfev: int
    l2_compact_candidate_evaluations: int
    l1_oracle_objective: float
    l2_oracle_objective: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_payload(self) -> dict[str, object]:
        return {
            "index": self.index,
            "l1_process_wall_seconds": self.l1_process_wall_seconds,
            "l2_process_wall_seconds": self.l2_process_wall_seconds,
            "l1_nfev": self.l1_nfev,
            "l2_compact_candidate_evaluations": (self.l2_compact_candidate_evaluations),
            "l1_oracle_objective": self.l1_oracle_objective,
            "l2_oracle_objective": self.l2_oracle_objective,
            "failures": list(self.failures),
            "passed": self.passed,
        }


def _oracle_objective(oracle: Mapping[str, object]) -> float:
    """The oracle's own endpoint objective; the only one a gate may read."""
    return float(str(oracle["objective_value"]))


def _oracle_gradient_inf(oracle: Mapping[str, object]) -> float:
    gradient = oracle["gradient"]
    if not isinstance(gradient, Mapping):
        raise F3ContractError("oracle payload carries no gradient block.")
    full = gradient["full_675"]
    if not isinstance(full, list):
        raise F3ContractError("oracle gradient.full_675 is not an array.")
    return max(abs(float(str(entry))) for entry in full)


def _oracle_for(
    *,
    candidate: Mapping[str, object],
    inner_state: Sequence[float],
    oracle_root: Path,
    source_manifest: Path,
) -> Mapping[str, object]:
    return run_oracle(
        candidate=candidate,
        anchor=(float(inner_state[0]), float(inner_state[1])),
        oracle_root=oracle_root,
        source_manifest=source_manifest,
    )


def _run_pair(
    *,
    index: int,
    rung: str,
    budget: int,
    root: Path,
    shim_dir: Path,
    source_manifest: Path,
    campaign_manifest_sha: str,
    trees: Mapping[str, Mapping[str, object]],
    native_config: NativeConfig,
    quality_target: float | None,
    native_budget: int | None,
    warm: bool,
) -> PairOutcome:
    """One interleaved (L1, L2) pair with symmetric discarded primers."""
    native_maxiter = budget if native_budget is None else native_budget
    cache_dir = root / f"gpu-cache-pair{index}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if warm:
        run_fused_leg(
            budget=budget,
            leg_root=root / f"pair{index}-l1-primer",
            role="primer",
            timed=False,
            cache_dir=cache_dir,
            shim_dir=shim_dir,
            source_manifest=source_manifest,
            affinity=native_config.affinity,
        )
    fused = run_fused_leg(
        budget=budget,
        leg_root=root / f"pair{index}-l1",
        role="timed",
        timed=True,
        cache_dir=cache_dir,
        shim_dir=shim_dir,
        source_manifest=source_manifest,
        affinity=native_config.affinity,
    )
    native_env = native_environment(native_config, shim_dir=shim_dir)
    if warm:
        run_leg(
            lane=L2_LANE,
            budget=native_maxiter,
            environment=native_env,
            omp_threads=native_config.omp_threads,
            affinity=native_config.affinity,
            leg_root=root / f"pair{index}-l2-primer",
            timed=False,
            role="primer",
            config_label=native_config.label,
            campaign_manifest_sha256=campaign_manifest_sha,
            source_manifest=source_manifest,
            git_identity=trees,
        )
    native = run_leg(
        lane=L2_LANE,
        budget=native_maxiter,
        environment=native_env,
        omp_threads=native_config.omp_threads,
        affinity=native_config.affinity,
        leg_root=root / f"pair{index}-l2",
        timed=True,
        role="timed",
        config_label=native_config.label,
        campaign_manifest_sha256=campaign_manifest_sha,
        source_manifest=source_manifest,
        git_identity=trees,
    )

    fused_oracle = _oracle_for(
        candidate=fused.endpoint_candidate,
        inner_state=fused.endpoint_inner_state,
        oracle_root=root / f"pair{index}-l1-oracle",
        source_manifest=source_manifest,
    )
    native_oracle = _oracle_for(
        candidate=native.endpoint_candidate,
        inner_state=native.endpoint_inner_state,
        oracle_root=root / f"pair{index}-l2-oracle",
        source_manifest=source_manifest,
    )

    failures = counter_liveness_failures(
        l1_nfev=fused.nfev,
        l2_compact_evaluations=native.compact_candidate_evaluations,
    )
    failures.extend(policy_identity_failures(fused.policy, budget=budget))
    if quality_target is None:
        failures.extend(
            fixed_budget_quality_failures(
                l1_oracle_objective=_oracle_objective(fused_oracle),
                l2_oracle_objective=_oracle_objective(native_oracle),
                l1_oracle_gradient_inf=_oracle_gradient_inf(fused_oracle),
                l2_oracle_gradient_inf=_oracle_gradient_inf(native_oracle),
            )
        )
        if fused.nit != budget:
            failures.append(f"l1_nit_{fused.nit}_expected_{budget}")
    else:
        failures.extend(
            bq_quality_failures(
                l1_oracle_objective=_oracle_objective(fused_oracle),
                l2_oracle_objective=_oracle_objective(native_oracle),
                l1_oracle_gradient_inf=_oracle_gradient_inf(fused_oracle),
                l2_oracle_gradient_inf=_oracle_gradient_inf(native_oracle),
                quality_target=quality_target,
            )
        )
    for phase in ("advance", "callback", "unclassified"):
        if fused.host_transfer_ledger.get(phase, 0):
            failures.append(f"l1_host_{phase}_transfers")

    _write_row(
        root / f"pair{index}-l1" / "row.json",
        lane=L1_LANE,
        role="timed",
        rung=rung,
        budget=budget,
        timed=True,
        process_wall_seconds=fused.process_wall_seconds,
        evaluation_count=fused.nfev,
        counter_name="nfev",
        nit=fused.nit,
        policy=fused.policy,
        oracle=fused_oracle,
        endpoint_inner_state=fused.endpoint_inner_state,
        trees=trees,
        campaign_manifest_sha256=campaign_manifest_sha,
        extra={
            "endpoint_inner_state_seconds": fused.endpoint_inner_state_seconds,
            "host_transfer_ledger": dict(fused.host_transfer_ledger),
            "self_reported_objective": fused.objective_value,
            "child_provenance": dict(fused.provenance),
        },
    )
    _write_row(
        root / f"pair{index}-l2" / "f3_row.json",
        lane=L2_LANE,
        role="timed",
        rung=rung,
        budget=native_maxiter,
        timed=True,
        process_wall_seconds=native.process_wall_seconds,
        evaluation_count=native.compact_candidate_evaluations,
        counter_name="compact_candidate_evaluations",
        nit=native.accepted_callback_count,
        # The archived native policy record defines no maxfun — the native
        # lane imposes no evaluation cap — so the row carries the chartered
        # seven-field payload and discloses the substitution in
        # native_maxfun_source rather than hashing a different payload.
        policy=policy_payload(native_maxiter),
        oracle=native_oracle,
        endpoint_inner_state=native.endpoint_inner_state,
        trees=trees,
        campaign_manifest_sha256=campaign_manifest_sha,
        extra={
            "native_maxfun_source": NATIVE_MAXFUN_SOURCE,
            "native_config": native_config.label,
            "self_reported_objective": native.endpoint_objective,
            "fair_bar_row_path": str(native.row_path),
        },
    )
    return PairOutcome(
        index=index,
        l1_process_wall_seconds=fused.process_wall_seconds,
        l2_process_wall_seconds=native.process_wall_seconds,
        l1_nfev=fused.nfev,
        l2_compact_candidate_evaluations=native.compact_candidate_evaluations,
        l1_oracle_objective=_oracle_objective(fused_oracle),
        l2_oracle_objective=_oracle_objective(native_oracle),
        failures=tuple(failures),
    )


def cmd_pairs(args: argparse.Namespace) -> None:
    rung = str(args.rung)
    budget = RUNG_BUDGETS[rung]
    root, shim_dir, campaign_manifest_sha, trees = _prepare_run(
        f"pairs-{rung}", args.input_manifest
    )
    native_config = NativeConfig.pinned(args.omp_threads)
    pairs: list[PairOutcome] = []
    for index in range(PAIR_COUNT):
        pairs.append(
            _run_pair(
                index=index,
                rung=rung,
                budget=budget,
                root=root,
                shim_dir=shim_dir,
                source_manifest=args.input_manifest,
                campaign_manifest_sha=campaign_manifest_sha,
                trees=trees,
                native_config=native_config,
                quality_target=None,
                native_budget=None,
                warm=True,
            )
        )
    _publish_rung(root, rung, budget, pairs, trees, campaign_manifest_sha)


def _publish_rung(
    root: Path,
    rung: str,
    budget: int,
    pairs: Sequence[PairOutcome],
    trees: Mapping[str, Mapping[str, object]],
    campaign_manifest_sha: str,
    *,
    quality_target: float | None = None,
    native_budget: int | None = None,
) -> None:
    passed = [pair for pair in pairs if pair.passed]
    not_produced = len(pairs) - len(passed)
    l1_walls = [pair.l1_process_wall_seconds for pair in passed]
    l2_walls = [pair.l2_process_wall_seconds for pair in passed]
    l1_nfev = [pair.l1_nfev for pair in passed]
    l2_compact = [pair.l2_compact_candidate_evaluations for pair in passed]
    if rung == "b3":
        anchor = b3_anchor()
    elif rung == "b37":
        anchor = (
            b37_anchor(l2_compact_evaluations=l2_compact, l1_nfev=l1_nfev)
            if l1_nfev and l2_compact
            else None
        )
    else:
        anchor = (
            bq_anchor(l2_compact_evaluations_at_nstar=l2_compact)
            if l2_compact
            else None
        )
    outcome = adjudicate_rung(
        l1_walls=l1_walls,
        l2_walls=l2_walls,
        anchor_seconds=anchor,
        not_produced_pairs=not_produced,
    )
    ledger = CapLedger(
        timed_legs=2 * len(pairs),
        solve_child_processes=4 * len(pairs),
    )
    _finish_run(
        root,
        {
            "schema": F3_RUN_MANIFEST_SCHEMA,
            "rung": rung,
            "budget": budget,
            "native_budget": native_budget,
            "quality_target": quality_target,
            "f3_charter_sha256": F3_CHARTER_SHA256,
            "f3_charter_commit": F3_CHARTER_COMMIT,
            "f3_charter_lineage": list(F3_CHARTER_LINEAGE),
            "fair_bar_charter_sha256": FAIR_BAR_CHARTER_SHA256,
            "campaign_input_manifest_sha256": campaign_manifest_sha,
            "rung_policy_sha256": observed_policy_sha256(policy_payload(budget)),
            "anchor_process_wall_seconds": anchor,
            "anchor_over_l1_median": outcome.anchor_over_l1_median,
            "median_speedup": outcome.median_speedup,
            "minimum_speedup": outcome.minimum_speedup,
            "pair_speedups": list(outcome.pair_speedups),
            "l1_median_wall_seconds": outcome.l1_median_wall,
            "l2_median_wall_seconds": outcome.l2_median_wall,
            "live_rule_holds": outcome.live_rule_holds,
            "anchor_rule_holds": outcome.anchor_rule_holds,
            "not_produced_pairs": not_produced,
            "gate_failures": [f for pair in pairs for f in pair.failures],
            "verdict": outcome.verdict.value,
            "timed_legs": ledger.timed_legs,
            "solve_child_processes": ledger.solve_child_processes,
            "cap_breaches": ledger.breaches(),
            "pairs": [pair.as_payload() for pair in pairs],
            "git": {name: dict(value) for name, value in trees.items()},
        },
    )
    print(json.dumps({"run_dir": str(root), "verdict": outcome.verdict.value}))


# --------------------------------------------------------------------------
# BQ budget search (charter "BQ protocol", step 2 — untimed, capped)
# --------------------------------------------------------------------------


def _probe_quality(objective: float, target: float) -> bool:
    return objective <= target


def _run_search_probe(
    *,
    lane: str,
    maxiter: int,
    label: str,
    root: Path,
    shim_dir: Path,
    cache_dir: Path,
    source_manifest: Path,
    campaign_manifest_sha: str,
    trees: Mapping[str, Mapping[str, object]],
    native_config: NativeConfig,
) -> float:
    """Run one untimed probe and return its ORACLE-evaluated objective.

    The lane's self-reported objective is never consulted: the charter makes
    the oracle the sole adjudicator of whether a budget reached the target.
    """
    if lane == L1_LANE:
        leg = run_fused_leg(
            budget=maxiter,
            leg_root=root / label,
            role="probe",
            timed=False,
            cache_dir=cache_dir,
            shim_dir=shim_dir,
            source_manifest=source_manifest,
            affinity=native_config.affinity,
        )
        candidate = leg.endpoint_candidate
        inner: Sequence[float] = leg.endpoint_inner_state
    else:
        environment = native_environment(native_config, shim_dir=shim_dir)
        # The native probe keeps its primer discipline even untimed, so a
        # probe measures the same program a timed native leg would.
        run_leg(
            lane=L2_LANE,
            budget=maxiter,
            environment=environment,
            omp_threads=native_config.omp_threads,
            affinity=native_config.affinity,
            leg_root=root / f"{label}-primer",
            timed=False,
            role="primer",
            config_label=native_config.label,
            campaign_manifest_sha256=campaign_manifest_sha,
            source_manifest=source_manifest,
            git_identity=trees,
        )
        native_leg = run_leg(
            lane=L2_LANE,
            budget=maxiter,
            environment=environment,
            omp_threads=native_config.omp_threads,
            affinity=native_config.affinity,
            leg_root=root / label,
            timed=False,
            role="probe",
            config_label=native_config.label,
            campaign_manifest_sha256=campaign_manifest_sha,
            source_manifest=source_manifest,
            git_identity=trees,
        )
        candidate = native_leg.endpoint_candidate
        inner = native_leg.endpoint_inner_state
    oracle = _oracle_for(
        candidate=candidate,
        inner_state=inner,
        oracle_root=root / f"{label}-oracle",
        source_manifest=source_manifest,
    )
    return _oracle_objective(oracle)


def _search_one_lane(
    *,
    lane: str,
    quality_target: float,
    root: Path,
    shim_dir: Path,
    cache_dir: Path,
    source_manifest: Path,
    campaign_manifest_sha: str,
    trees: Mapping[str, Mapping[str, object]],
    native_config: NativeConfig,
) -> dict[str, object]:
    """Smallest maxiter reaching the target: double upward, then bisect."""
    started = time.perf_counter()
    probes: list[dict[str, object]] = []
    maxiter = BQ_SEARCH_START
    low: int | None = None
    high: int | None = None
    while True:
        breaches = budget_search_breaches(
            probe_count=len(probes) + 1,
            largest_maxiter=maxiter,
            search_wall_seconds=time.perf_counter() - started,
        )
        if breaches:
            return {"probes": probes, "breaches": breaches, "star": None}
        objective = _run_search_probe(
            lane=lane,
            maxiter=maxiter,
            label=f"{lane}-probe{len(probes)}-m{maxiter}",
            root=root,
            shim_dir=shim_dir,
            cache_dir=cache_dir,
            source_manifest=source_manifest,
            campaign_manifest_sha=campaign_manifest_sha,
            trees=trees,
            native_config=native_config,
        )
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
        if low is None or high - low <= 1:
            return {"probes": probes, "breaches": [], "star": high}
        maxiter = (low + high) // 2


def cmd_budget_search(args: argparse.Namespace) -> None:
    """Find each lane's smallest maxiter whose oracle endpoint reaches Q*.

    Both lanes run the same procedure and the same caps; neither receives a
    budget the other's discipline would not grant it.  The fused probes share
    one warm compile cache; the native probes each keep their primer.
    """
    root, shim_dir, campaign_manifest_sha, trees = _prepare_run(
        "budget-search", args.input_manifest
    )
    target = float(args.quality_target)
    native_config = NativeConfig.pinned(args.omp_threads)
    cache_dir = root / "gpu-cache-search"
    cache_dir.mkdir(parents=True, exist_ok=True)
    searches = {
        lane: _search_one_lane(
            lane=lane,
            quality_target=target,
            root=root,
            shim_dir=shim_dir,
            cache_dir=cache_dir,
            source_manifest=args.input_manifest,
            campaign_manifest_sha=campaign_manifest_sha,
            trees=trees,
            native_config=native_config,
        )
        for lane in (L1_LANE, L2_LANE)
    }
    _finish_run(
        root,
        {
            "schema": F3_RUN_MANIFEST_SCHEMA,
            "rung": "bq-budget-search",
            "budget": BQ_SEARCH_START,
            "quality_target": target,
            "f3_charter_sha256": F3_CHARTER_SHA256,
            "f3_charter_lineage": list(F3_CHARTER_LINEAGE),
            "campaign_input_manifest_sha256": campaign_manifest_sha,
            "searches": searches,
            "verdict": (
                Verdict.NOT_PRODUCED.value
                if any(search["star"] is None for search in searches.values())
                else "SEARCH_COMPLETE"
            ),
            "caps": {
                "max_probes_per_lane": BQ_MAX_PROBES_PER_LANE,
                "max_maxiter": BQ_MAX_MAXITER,
                "max_search_seconds": BQ_MAX_SEARCH_SECONDS,
            },
            "git": {name: dict(value) for name, value in trees.items()},
        },
    )
    print(json.dumps({"run_dir": str(root), "searches": searches}, default=str))


def cmd_pairs_bq(args: argparse.Namespace) -> None:
    """Five timed pairs: fused at m*, native at n*, both re-gated on Q*."""
    root, shim_dir, campaign_manifest_sha, trees = _prepare_run(
        "pairs-bq", args.input_manifest
    )
    native_config = NativeConfig.pinned(args.omp_threads)
    pairs = [
        _run_pair(
            index=index,
            rung="bq",
            budget=int(args.fused_maxiter),
            root=root,
            shim_dir=shim_dir,
            source_manifest=args.input_manifest,
            campaign_manifest_sha=campaign_manifest_sha,
            trees=trees,
            native_config=native_config,
            quality_target=float(args.quality_target),
            native_budget=int(args.native_maxiter),
            warm=True,
        )
        for index in range(PAIR_COUNT)
    ]
    _publish_rung(
        root,
        "bq",
        int(args.fused_maxiter),
        pairs,
        trees,
        campaign_manifest_sha,
        quality_target=float(args.quality_target),
        native_budget=int(args.native_maxiter),
    )


def cmd_cold_pair(args: argparse.Namespace) -> None:
    """One fresh-cache disclosure pair per rung: report-only, never a verdict."""
    rung = str(args.rung)
    budget = RUNG_BUDGETS[rung]
    root, shim_dir, campaign_manifest_sha, trees = _prepare_run(
        f"cold-{rung}", args.input_manifest
    )
    cache_dir = root / "gpu-cache-cold"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    pair = _run_pair(
        index=0,
        rung=rung,
        budget=budget,
        root=root,
        shim_dir=shim_dir,
        source_manifest=args.input_manifest,
        campaign_manifest_sha=campaign_manifest_sha,
        trees=trees,
        native_config=NativeConfig.pinned(args.omp_threads),
        quality_target=None,
        native_budget=None,
        warm=False,
    )
    _finish_run(
        root,
        {
            "schema": F3_RUN_MANIFEST_SCHEMA,
            "rung": f"{rung}-cold-disclosure",
            "budget": budget,
            "f3_charter_sha256": F3_CHARTER_SHA256,
            "f3_charter_lineage": list(F3_CHARTER_LINEAGE),
            "campaign_input_manifest_sha256": campaign_manifest_sha,
            "disclosure_only": True,
            "verdict": Verdict.NOT_PRODUCED.value,
            "verdict_note": "cold legs are disclosed, never claimed",
            "pairs": [pair.as_payload()],
            "git": {name: dict(value) for name, value in trees.items()},
        },
    )
    print(json.dumps({"run_dir": str(root), "pair": pair.as_payload()}))


def cmd_native_sweep(args: argparse.Namespace) -> None:
    """B37 contingency: five OMP configs x three reps, fair-bar selection rule."""
    root, shim_dir, campaign_manifest_sha, trees = _prepare_run(
        "native-sweep", args.input_manifest
    )
    budget = RUNG_BUDGETS["b37"]
    medians: dict[str, float] = {}
    for threads in NATIVE_SWEEP_OMP_MATRIX:
        config = NativeConfig.pinned(threads)
        walls: list[float] = []
        for rep in range(NATIVE_SWEEP_REPS):
            environment = native_environment(config, shim_dir=shim_dir)
            run_leg(
                lane=L2_LANE,
                budget=budget,
                environment=environment,
                omp_threads=config.omp_threads,
                affinity=config.affinity,
                leg_root=root / f"{config.label}-rep{rep}-primer",
                timed=False,
                role="primer",
                config_label=config.label,
                campaign_manifest_sha256=campaign_manifest_sha,
                source_manifest=args.input_manifest,
                git_identity=trees,
            )
            leg = run_leg(
                lane=L2_LANE,
                budget=budget,
                environment=environment,
                omp_threads=config.omp_threads,
                affinity=config.affinity,
                leg_root=root / f"{config.label}-rep{rep}",
                timed=True,
                role="timed",
                config_label=config.label,
                campaign_manifest_sha256=campaign_manifest_sha,
                source_manifest=args.input_manifest,
                git_identity=trees,
            )
            walls.append(leg.process_wall_seconds)
        medians[config.label] = statistics.median(walls)
    selection = min(medians, key=lambda label: medians[label])
    _finish_run(
        root,
        {
            "schema": F3_RUN_MANIFEST_SCHEMA,
            "rung": "b37-native-sweep",
            "budget": budget,
            "f3_charter_sha256": F3_CHARTER_SHA256,
            "f3_charter_lineage": list(F3_CHARTER_LINEAGE),
            "campaign_input_manifest_sha256": campaign_manifest_sha,
            "median_process_wall_seconds": medians,
            "selection": selection,
            "verdict": "SWEEP_COMPLETE",
            "timed_legs": len(NATIVE_SWEEP_OMP_MATRIX) * NATIVE_SWEEP_REPS,
            "git": {name: dict(value) for name, value in trees.items()},
        },
    )
    print(json.dumps({"run_dir": str(root), "selection": selection}))


def cmd_validate(args: argparse.Namespace) -> None:
    report = validate_run_dir(args.run_dir)
    print(json.dumps(report.as_payload(), indent=2, sort_keys=True))
    if not report.valid:
        raise SystemExit(1)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _with_common(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sub.add_argument("--input-manifest", required=True, type=Path)
        sub.add_argument("--omp-threads", type=int, default=16)
        return sub

    pairs = _with_common(subparsers.add_parser("pairs"))
    pairs.add_argument("--rung", required=True, choices=sorted(RUNG_BUDGETS))
    pairs.set_defaults(handler=cmd_pairs)

    search = _with_common(subparsers.add_parser("budget-search"))
    search.add_argument("--quality-target", required=True, type=float)
    search.set_defaults(handler=cmd_budget_search)

    bq = _with_common(subparsers.add_parser("pairs-bq"))
    bq.add_argument("--fused-maxiter", required=True, type=int)
    bq.add_argument("--native-maxiter", required=True, type=int)
    bq.add_argument("--quality-target", required=True, type=float)
    bq.set_defaults(handler=cmd_pairs_bq)

    cold = _with_common(subparsers.add_parser("cold-pair"))
    cold.add_argument("--rung", required=True, choices=sorted(RUNG_BUDGETS))
    cold.set_defaults(handler=cmd_cold_pair)

    sweep = _with_common(subparsers.add_parser("native-sweep"))
    sweep.set_defaults(handler=cmd_native_sweep)

    validate = subparsers.add_parser("validate")
    validate.add_argument("run_dir", type=Path)
    validate.set_defaults(handler=cmd_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
