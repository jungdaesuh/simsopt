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

``--input-manifest`` is the sealed BUNDLE's own ``manifest.json`` (sha
``84febc05…``), exactly as the fair-bar CLI takes it; the campaign manifest
(the fair-bar reclassification wrapper, sha ``2a381125…``) is minted fresh
into every run root rather than supplied — the sealed bundle's exact member
census forbids it living beside the members.

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
    DISCLOSURE_RUNGS,
    F3_CHARTER_COMMIT,
    F3_CHARTER_LINEAGE,
    F3_CHARTER_SHA256,
    F3_ROW_DIRECTORY,
    F3_ROW_SCHEMA,
    F3_RUN_MANIFEST_SCHEMA,
    L1_LANE,
    L2_LANE,
    NATIVE_MAXFUN_SOURCE,
    NATIVE_SWEEP_OMP_MATRIX,
    NATIVE_SWEEP_REPS,
    PAIR_COUNT,
    RUNG_BUDGETS,
    SOLVE_CHILDREN_PER_COLD_PAIR,
    SOLVE_CHILDREN_PER_WARM_PAIR,
    BudgetSearch,
    CampaignState,
    F3ContractError,
    Verdict,
    adjudicate_rows,
    f3_contract_sha256,
    observed_policy_sha256,
    parse_row,
    policy_identity_failures,
    policy_payload,
    production_child_environment,
    resolve_disclosure_budgets,
    search_minimal_budget,
    select_sweep_config,
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
    mint_campaign_manifest,
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
    # The fair-bar builder is the instrument's, and its output is shaped for
    # the instrument tree; production_child_environment is the F3-owned
    # adapter that makes it safe for a production-tree child.
    environment = production_child_environment(
        gpu_environment(cache_dir=cache_dir, shim_dir=shim_dir)
    )
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
    run_root: Path,
    *,
    name: str,
    lane: str,
    role: str,
    rung: str,
    pair_index: int,
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
        "pair_index": pair_index,
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
    # F3 rows live in their own subtree: the fair-bar run_leg writes its own
    # row.json into each native leg directory, and a row discovered there
    # would be that foreign row rather than this campaign's.
    row_dir = run_root / F3_ROW_DIRECTORY
    row_dir.mkdir(parents=True, exist_ok=True)
    (row_dir / f"{name}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )
    return row


# --------------------------------------------------------------------------
# Run directories
# --------------------------------------------------------------------------


CAMPAIGN_STATE_PATH = OUTPUT_ROOT / "campaign_state.json"


def _campaign_state() -> CampaignState:
    """Cumulative caps across every phase this campaign has run."""
    if not CAMPAIGN_STATE_PATH.is_file():
        return CampaignState()
    return CampaignState.from_payload(json.loads(CAMPAIGN_STATE_PATH.read_text()))


def _write_campaign_state(state: CampaignState) -> None:
    CAMPAIGN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAMPAIGN_STATE_PATH.write_text(
        json.dumps(state.as_payload(), indent=2, sort_keys=True) + "\n"
    )


def _require_rung_admission(*, timed: int, solve_children: int) -> None:
    """Refuse to START a rung whose cost would breach a cap.

    The charter ends the campaign at a rung boundary: completed rungs keep
    their verdicts and unstarted rungs are NOT_PRODUCED, so the check belongs
    here rather than mid-rung.
    """
    breaches = _campaign_state().admits(timed=timed, solve_children=solve_children)
    if breaches:
        raise F3ContractError(
            "campaign caps end this campaign at the previous rung boundary; "
            f"this rung is NOT_PRODUCED: {breaches}"
        )


def _new_run_root(phase: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = OUTPUT_ROOT / f"{stamp}-{phase}-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _prepare_run(phase: str, source_manifest: Path) -> tuple[Path, Path, str, dict]:
    """Open a run root and bind the frozen input, fair-bar pattern.

    ``source_manifest`` is the sealed BUNDLE's own manifest.json (the file the
    lane children consume); the campaign manifest is minted fresh into the run
    root and validated against the bundle directory.  The two cannot share a
    directory: the instrument's envelope check enforces an exact member census
    on the sealed 0550 bundle, so any cohabiting extra file fails it.
    """
    trees = require_clean_trees()
    root = _new_run_root(phase)
    shim_dir = write_provenance_shim(root)
    campaign_manifest = root / "campaign_input_manifest.json"
    mint_campaign_manifest(source_manifest, campaign_manifest)
    campaign_manifest_sha = load_campaign_manifest(
        campaign_manifest, source_manifest.parent
    )
    return root, shim_dir, campaign_manifest_sha, trees


def _finish_run(root: Path, manifest: Mapping[str, object]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


# --------------------------------------------------------------------------
# Pair phases (charter "Verdict rule", "Priming and cold-start")
# --------------------------------------------------------------------------


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
) -> tuple[Mapping[str, object], Mapping[str, object]]:
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

    l1_row = _write_row(
        root,
        name=f"pair{index}-l1",
        lane=L1_LANE,
        role="timed",
        rung=rung,
        pair_index=index,
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
    l2_row = _write_row(
        root,
        name=f"pair{index}-l2",
        lane=L2_LANE,
        role="timed",
        rung=rung,
        pair_index=index,
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
    # No gate is evaluated here: the rows carry every number a gate reads and
    # adjudicate_rows applies the one rule the validator also applies.
    return (l1_row, l2_row)


def cmd_pairs(args: argparse.Namespace) -> None:
    rung = str(args.rung)
    budget = RUNG_BUDGETS[rung]
    root, shim_dir, campaign_manifest_sha, trees = _prepare_run(
        f"pairs-{rung}", args.input_manifest
    )
    _require_rung_admission(
        timed=2 * PAIR_COUNT,
        solve_children=SOLVE_CHILDREN_PER_WARM_PAIR * PAIR_COUNT,
    )
    native_config = NativeConfig.pinned(args.omp_threads)
    rows: list[Mapping[str, object]] = []
    for index in range(PAIR_COUNT):
        rows.extend(
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
    _publish_rung(root, rung, budget, rows, trees, campaign_manifest_sha)


def _publish_rung(
    root: Path,
    rung: str,
    budget: int,
    rows: Sequence[Mapping[str, object]],
    trees: Mapping[str, Mapping[str, object]],
    campaign_manifest_sha: str,
    *,
    quality_target: float | None = None,
    native_budget: int | None = None,
    cold_disclosure: bool = False,
) -> Verdict:
    """Adjudicate the rung with the validator's own rule and publish it.

    The producer does not carry a second opinion: it parses the rows it just
    wrote and calls ``adjudicate_rows``, which is the function ``validate``
    calls against the same bytes.  A published verdict is therefore one
    ``validate`` must reproduce.
    """
    parsed = [
        parse_row(row, source=f"{row.get('lane')}-pair{row.get('pair_index')}")
        for row in rows
    ]
    outcome, pairs = adjudicate_rows(
        parsed,
        rung=rung,
        quality_target=quality_target,
        disclosure=cold_disclosure,
    )
    timed_legs = 2 * len(pairs)
    solve_children = len(pairs) * (
        SOLVE_CHILDREN_PER_COLD_PAIR
        if cold_disclosure
        else SOLVE_CHILDREN_PER_WARM_PAIR
    )
    state = _campaign_state()
    updated = state.completing(rung, timed=timed_legs, solve_children=solve_children)
    _write_campaign_state(updated)
    # The disclosure token comes from the shared rule, not from a second
    # opinion here: a cold rung whose gates fail is voided like any evidence.
    verdict = outcome.verdict
    _finish_run(
        root,
        {
            "schema": F3_RUN_MANIFEST_SCHEMA,
            "rung": rung,
            "budget": budget,
            "native_budget": native_budget,
            "quality_target": quality_target,
            "disclosure_only": cold_disclosure,
            "f3_charter_sha256": F3_CHARTER_SHA256,
            "f3_charter_commit": F3_CHARTER_COMMIT,
            "f3_charter_lineage": list(F3_CHARTER_LINEAGE),
            "fair_bar_charter_sha256": FAIR_BAR_CHARTER_SHA256,
            "campaign_input_manifest_sha256": campaign_manifest_sha,
            "production_commit": trees["production"]["commit"],
            "instrument_commit": trees["instrument"]["commit"],
            "rung_policy_sha256": observed_policy_sha256(policy_payload(budget)),
            "anchor_process_wall_seconds": outcome.anchor_seconds,
            "anchor_over_l1_median": outcome.anchor_over_l1_median,
            "median_speedup": outcome.median_speedup,
            "minimum_speedup": outcome.minimum_speedup,
            "pair_speedups": list(outcome.pair_speedups),
            "l1_median_wall_seconds": outcome.l1_median_wall,
            "l2_median_wall_seconds": outcome.l2_median_wall,
            "live_rule_holds": outcome.live_rule_holds,
            "anchor_rule_holds": outcome.anchor_rule_holds,
            "not_produced_pairs": outcome.not_produced_pairs,
            "gate_failures": list(outcome.failures),
            "verdict": verdict.value,
            "timed_legs": updated.ledger.timed_legs,
            "solve_child_processes": updated.ledger.solve_child_processes,
            "campaign_wall_seconds": updated.ledger.campaign_wall_seconds,
            "cap_breaches": updated.ledger.breaches(),
            "campaign_stopped_at_rung_boundary": updated.stopped,
            "git": {name: dict(value) for name, value in trees.items()},
        },
    )
    print(json.dumps({"run_dir": str(root), "verdict": verdict.value}))
    return verdict


# --------------------------------------------------------------------------
# BQ budget search (charter "BQ protocol", step 2 — untimed, capped)
# --------------------------------------------------------------------------


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
) -> BudgetSearch:
    """Run the chartered search for one lane against real probe children.

    The search itself is ``search_minimal_budget``: this supplies the probe
    (a real untimed leg adjudicated by the oracle) and the clock, so the
    doubling, bisection, and cap rules have exactly one implementation.
    """
    started = time.perf_counter()
    probe_index = [0]

    def probe(maxiter: int) -> float:
        label = f"{lane}-probe{probe_index[0]}-m{maxiter}"
        probe_index[0] += 1
        return _run_search_probe(
            lane=lane,
            maxiter=maxiter,
            label=label,
            root=root,
            shim_dir=shim_dir,
            cache_dir=cache_dir,
            source_manifest=source_manifest,
            campaign_manifest_sha=campaign_manifest_sha,
            trees=trees,
            native_config=native_config,
        )

    return search_minimal_budget(
        probe,
        quality_target=quality_target,
        elapsed_seconds=lambda: time.perf_counter() - started,
    )


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
    searches: dict[str, BudgetSearch] = {
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
            "searches": {
                lane: search.as_payload() for lane, search in searches.items()
            },
            "verdict": (
                Verdict.NOT_PRODUCED.value
                if any(search.star is None for search in searches.values())
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
    print(
        json.dumps(
            {
                "run_dir": str(root),
                "searches": {
                    lane: search.as_payload() for lane, search in searches.items()
                },
            }
        )
    )


def cmd_pairs_bq(args: argparse.Namespace) -> None:
    """Five timed pairs: fused at m*, native at n*, both re-gated on Q*."""
    root, shim_dir, campaign_manifest_sha, trees = _prepare_run(
        "pairs-bq", args.input_manifest
    )
    _require_rung_admission(
        timed=2 * PAIR_COUNT,
        solve_children=SOLVE_CHILDREN_PER_WARM_PAIR * PAIR_COUNT,
    )
    native_config = NativeConfig.pinned(args.omp_threads)
    rows = [
        row
        for index in range(PAIR_COUNT)
        for row in _run_pair(
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
    ]
    _publish_rung(
        root,
        "bq",
        int(args.fused_maxiter),
        rows,
        trees,
        campaign_manifest_sha,
        quality_target=float(args.quality_target),
        native_budget=int(args.native_maxiter),
    )


def cmd_cold_pair(args: argparse.Namespace) -> None:
    """One fresh-cache disclosure pair per rung: report-only, never a verdict."""
    budgets = resolve_disclosure_budgets(
        str(args.rung),
        fused_maxiter=args.fused_maxiter,
        native_maxiter=args.native_maxiter,
        quality_target=args.quality_target,
    )
    root, shim_dir, campaign_manifest_sha, trees = _prepare_run(
        f"cold-{budgets.rung}", args.input_manifest
    )
    _require_rung_admission(timed=2, solve_children=SOLVE_CHILDREN_PER_COLD_PAIR)
    cache_dir = root / "gpu-cache-cold"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    rows = _run_pair(
        index=0,
        rung=budgets.rung_label,
        budget=budgets.fused_maxiter,
        root=root,
        shim_dir=shim_dir,
        source_manifest=args.input_manifest,
        campaign_manifest_sha=campaign_manifest_sha,
        trees=trees,
        native_config=NativeConfig.pinned(args.omp_threads),
        quality_target=budgets.quality_target,
        native_budget=budgets.native_maxiter,
        warm=False,
    )
    # Cold legs are disclosed, never claimed: the pair publishes through the
    # same rule, which labels it FRESH_REPORTED rather than entering a verdict.
    _publish_rung(
        root,
        budgets.rung_label,
        budgets.fused_maxiter,
        rows,
        trees,
        campaign_manifest_sha,
        quality_target=budgets.quality_target,
        native_budget=budgets.native_maxiter,
        cold_disclosure=True,
    )


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
    selection = select_sweep_config(medians)
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
    cold.add_argument("--rung", required=True, choices=sorted(DISCLOSURE_RUNGS))
    # Required for --rung bq, refused for the fixed-budget rungs; the check is
    # in _cold_pair_budgets so both halves of the rule live in one place.
    cold.add_argument("--fused-maxiter", type=int, default=None)
    cold.add_argument("--native-maxiter", type=int, default=None)
    cold.add_argument("--quality-target", type=float, default=None)
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
