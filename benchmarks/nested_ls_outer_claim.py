"""Nested-LS eight-term outer B3/B37 process-wall claim run.

Charter: ``docs/jax_nested_ls_outer_charter.md``. Moving-coil scipy
L-BFGS-B over the 11 coil DOFs on both lanes, both configured from the
same sealed F3 native-lane optimizer policy at this rung's budget and
``maxcor``. The JAX lane eliminates the 661 surface DOFs with the reduced
nested-LS inner solve; the native lane is the banana nested twin at
best-of-contract OMP.

Claim clock is the **full parent subprocess wait on both sides, with no
subtraction on either side** — this deliberately differs from Gate-6,
whose JAX clock subtracted reconstruct and rejudge. The physics gates
(endpoint C++ LS Newton rejudge no-op, reduced-gradient check, endpoint
eight-term ``J`` parity) run untimed after the timed pairs. Not F3 7.70x
and not a supersession of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Never

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(REPO / "src")

from simsopt_jax_adapters.geo.nested_ls_contract import (
    F3_B37_BANANA_OMP_CONTRACT_THREADS,
    NESTED_LS_GATE6_AGGREGATION,
    NESTED_LS_GATE6_NATIVE_OMP_THREADS,
    NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
    NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
    NESTED_LS_OUTER_OMP_SWEEP_REPEATS,
    NESTED_LS_OUTER_REJUDGE_SCHEMA,
)

from benchmarks.nested_ls_shamanskii_attribution import (
    PYTHON,
    REPEATS,
    git_implementation_dirty,
    write_strict_json,
)

EVIDENCE = REPO / "docs" / "receipts" / "evidence"
NATIVE_CHILD = REPO / "benchmarks" / "nested_ls_outer_native_child.py"
JAX_CHILD = REPO / "benchmarks" / "nested_ls_outer_jax_child.py"
CACHE_OUTER = REPO / ".artifacts" / "nested-ls-outer-xla"
EVIDENCE_DATE: Final[str] = "20260824"
CLAIM_BUDGETS: Final[tuple[int, ...]] = (3, 37)
CLAIM_SCHEMA: Final[str] = "nested-ls-outer-claim.v2"
SWEEP_SCHEMA: Final[str] = "nested-ls-outer-native-omp-sweep.v2"
# Charter: "B37 runs only after B3 lands physics-green." The interlock is
# the receipt, not a promise, so B37 must be handed the B3 artifact.
B3_BUDGET: Final[int] = 3
B37_BUDGET: Final[int] = 37
# scipy's own L-BFGS-B default. The driver is the single source of this
# value: both children receive it on the command line and feed it to the
# same sealed F3 lane policy loader, so the lanes cannot drift apart. The
# per-pair gate re-checks that the two published policies are identical.
DEFAULT_MAXCOR: Final[int] = 10
# The frozen 5090 contract sweep set and repeat count, restated here
# because the parent must not import the jax-side module that seals them
# (F3_B37_BANANA_OMP_CONTRACT_THREADS / F3_B37_BANANA_OMP_REPEATS in
# nested_ls_reduced_scale): importing it would initialize a device in the
# process that owns the claim clock. Other hosts pass --omp-set.
DEFAULT_OMP_SET: Final[str] = ",".join(
    str(threads) for threads in F3_B37_BANANA_OMP_CONTRACT_THREADS
)
A100_OMP_SET: Final[tuple[int, ...]] = (14, 16, 20, 24)
SWEEP_REPEATS: Final[int] = NESTED_LS_OUTER_OMP_SWEEP_REPEATS
# The charter sweeps the native denominator per rung. A number typed on
# the command line is not a swept bar, so claim runs must cite the sweep
# artifact and B37 must inherit that provenance through its B3 receipt.
OMP_PROVENANCE_SWEPT: Final[str] = "swept_artifact"


def _reject_nonfinite_json_constant(token: str) -> Never:
    raise ValueError(f"non-finite JSON constant {token!r}")


def _load_strict_evidence_json(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
        )
    except ValueError as error:
        raise SystemExit(f"{label} is not strict finite JSON: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object")
    return payload


def _require_finite_number(
    value: object,
    *,
    label: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label} must be a JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise SystemExit(f"{label} must be finite")
    if positive and number <= 0.0:
        raise SystemExit(f"{label} must be positive")
    if nonnegative and number < 0.0:
        raise SystemExit(f"{label} must be nonnegative")
    return number


def _parse_j_parity_rtol(value: object) -> float:
    return _require_finite_number(
        value,
        label="--j-parity-rtol",
        nonnegative=True,
    )


def _verified_embedded_child_payload(
    *, lane: str, row: dict[str, object]
) -> tuple[dict[str, object] | None, str | None]:
    raw = row.get("child_payload_raw")
    if not isinstance(raw, str):
        return None, f"{lane}_child_payload_raw_missing"
    if hashlib.sha256(raw.encode("utf-8")).hexdigest() != str(
        row.get("child_payload_sha256")
    ):
        return None, f"{lane}_child_payload_sha256_mismatch"
    try:
        payload = json.loads(
            raw,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except ValueError:
        return None, f"{lane}_child_payload_not_strict_json"
    if not isinstance(payload, dict):
        return None, f"{lane}_child_payload_not_object"
    if payload != row.get("child_payload"):
        return None, f"{lane}_embedded_child_payload_mismatch"
    return payload, None


def _child_row_mismatch(
    *, lane: str, row: dict[str, object], payload: dict[str, object]
) -> str | None:
    if lane == "native":
        endpoint = payload["endpoint"]
        start = payload["start"]
        threading = payload["threading"]
        expected = {
            "observed_omp_num_threads": int(payload["omp_num_threads"]),
            "omp_pinned": payload["omp_pinned"],
            "omp_proc_bind": threading["OMP_PROC_BIND"],
            "omp_places": threading["OMP_PLACES"],
            "success": payload["success"],
            "nit": int(payload["nit"]),
            "nfev": int(payload["nfev"]),
            "restart_count": int(payload["restart_count"]),
            "endpoint_is_optimizer_endpoint": payload["endpoint_is_optimizer_x"],
            "outer_policy": payload["outer_policy"],
            "endpoint_j": float(endpoint["objective"]),
            "endpoint_iota": float(endpoint["iota"]),
            "endpoint_g": float(endpoint["G"]),
            "endpoint_gradient_l2": float(endpoint["gradient_l2"]),
            "endpoint_coil_sha256": str(endpoint["coil_sha256"]),
            "endpoint_surface_sha256": str(endpoint["surface_sha256"]),
            "start_coil_dofs": [float(entry) for entry in start["coil_dofs"]],
            "endpoint_coil_dofs": [float(entry) for entry in endpoint["coil_dofs"]],
        }
    else:
        expected = {
            "success": payload["success"],
            "nit": int(payload["nit"]),
            "nfev": int(payload["nfev"]),
            "restart_count": int(payload["restart_count"]),
            "endpoint_is_optimizer_endpoint": payload["endpoint_is_optimizer_x"],
            "outer_policy": payload["outer_policy"],
            "start_policy": str(payload["start_policy"]),
            "iota_branch_guard": float(payload["iota_branch_guard"]),
            "feasible_evaluations": int(payload["feasible_evaluations"]),
            "rejected_evaluations": int(payload["rejected_evaluations"]),
            "endpoint_j": float(payload["endpoint_j"]),
            "endpoint_grad_l2": float(payload["endpoint_grad_l2"]),
            "endpoint_grad_inf": float(payload["endpoint_grad_inf"]),
            "endpoint_iota": float(payload["endpoint_iota"]),
            "endpoint_g": float(payload["endpoint_g"]),
            "endpoint_adjoint_live_eta": float(payload["endpoint_adjoint_live_eta"]),
            "endpoint_coil_sha256": str(payload["endpoint_coil_sha256"]),
            "endpoint_surface_sha256": str(payload["endpoint_surface_sha256"]),
            "start_coil_dofs": [float(entry) for entry in payload["start_coil_dofs"]],
            "endpoint_coil_dofs": [
                float(entry) for entry in payload["endpoint_coil_dofs"]
            ],
        }
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            return f"{lane}_row_{field}_mismatch"
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested-LS eight-term outer B3/B37 process-wall claim run."
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        choices=CLAIM_BUDGETS,
        help="Charter rung: outer scipy L-BFGS-B maxiter on both lanes.",
    )
    parser.add_argument(
        "--omp",
        type=int,
        default=NESTED_LS_GATE6_NATIVE_OMP_THREADS,
        help="Native lane OMP_NUM_THREADS (best-of-contract on this host).",
    )
    parser.add_argument(
        "--maxcor",
        type=int,
        default=DEFAULT_MAXCOR,
        help="scipy L-BFGS-B maxcor, identical on both lanes.",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=int(REPEATS),
        help=(
            "Interleaved measure pairs. The charter default is the contract "
            "REPEATS; Amendment 5 permits 1 for a fault-rerun whose verdict "
            "is the deterministic physics gate, walls informational."
        ),
    )
    parser.add_argument(
        "--skip-prime",
        action="store_true",
        help=(
            "Skip the untimed JAX cache-priming run (Amendment 5 fault-rerun "
            "only: requires a demonstrably warm persistent compile cache)."
        ),
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Receipt suffix, e.g. a100 → nested_ls_outer_b3_20260824.a100.json",
    )
    parser.add_argument(
        "--b3-receipt",
        default=None,
        help=(
            "Physics-green B3 claim receipt. Required for --budget 37 and "
            "forbidden for --budget 3: the charter runs B37 only after B3 "
            "lands green, and the B3 receipt is also where this run's --omp "
            "provenance and frozen J-parity band come from."
        ),
    )
    parser.add_argument(
        "--sweep-native-omp",
        action="store_true",
        help=(
            "Sweep the native denominator over --omp-set instead of running a "
            "claim: no JAX lane, no prime, no pairs. Produces the OMP evidence "
            "artifact a B3 claim run must cite."
        ),
    )
    parser.add_argument(
        "--omp-set",
        default=None,
        help=(
            "Comma-separated OMP_NUM_THREADS values for --sweep-native-omp "
            f"(default {DEFAULT_OMP_SET}, the frozen 5090 contract set). "
            "Sweep mode only."
        ),
    )
    parser.add_argument(
        "--omp-evidence",
        default=None,
        help=(
            "Native OMP sweep artifact whose best_omp_num_threads must equal "
            "--omp. Required for --budget 3 claim runs; forbidden for "
            f"--budget {B37_BUDGET}, which inherits the swept bar through its "
            "B3 receipt."
        ),
    )
    parser.add_argument(
        "--j-parity-rtol",
        type=float,
        default=None,
        help=(
            "Frozen one-sided endpoint-J band for B37. No default: charter "
            "Amendment 1 has B3 measure the achievable fork band and B37 "
            "freeze it, so this must be at least the B3 receipt's "
            "measured_j_rel_gap_max. Forbidden at B3, which only observes."
        ),
    )
    return parser.parse_args(argv)


def _require_clean_tree() -> str:
    dirty = git_implementation_dirty().strip()
    if dirty:
        raise SystemExit(
            "Outer claim requires a clean tree (implementation, not evidence):\n"
            f"{dirty}"
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()


def _make_logger(out_log: Path) -> Callable[[str], None]:
    """Tee the driver's own lines to the execution log the receipt names.

    The receipt cites ``execution_log``; a cited file that no run writes is
    a dangling citation, so the driver authors it as it goes and appends
    line by line rather than buffering, keeping the log truthful even if a
    child raises mid-run.
    """

    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_log.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with out_log.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    return log


def _require_omp_evidence(
    *,
    omp_evidence: Path,
    omp_num_threads: int,
    expected_git_head: str,
    expected_maxcor: int,
    expected_omp_set: tuple[int, ...],
) -> dict[str, object]:
    """Refuse a claim run whose native OMP is not the swept artifact's best.

    The charter sweeps the native denominator per rung and takes
    best-of-contract as the bar. Binding the artifact here is what makes
    ``--omp`` a measurement rather than an assertion.
    """

    if not omp_evidence.is_file():
        raise SystemExit(f"--omp-evidence does not exist: {omp_evidence}")
    raw = omp_evidence.read_bytes()
    payload = _load_strict_evidence_json(raw, label="--omp-evidence")
    schema = str(payload["schema"])
    if schema != SWEEP_SCHEMA:
        raise SystemExit(
            f"--omp-evidence schema is {schema!r}, expected {SWEEP_SCHEMA!r}"
        )
    artifact_git_head = str(payload["git_head"])
    if artifact_git_head != expected_git_head:
        raise SystemExit(
            f"--omp-evidence git_head is {artifact_git_head!r}, expected the "
            f"claim implementation {expected_git_head!r}"
        )
    if int(payload["budget"]) != B3_BUDGET:
        raise SystemExit(f"--omp-evidence is not a B{B3_BUDGET} sweep")
    if int(payload["maxcor"]) != expected_maxcor:
        raise SystemExit(
            f"--omp-evidence maxcor is {payload['maxcor']!r}, expected "
            f"{expected_maxcor}"
        )
    if str(payload["aggregation"]) != NESTED_LS_GATE6_AGGREGATION:
        raise SystemExit(
            f"--omp-evidence aggregation is {payload['aggregation']!r}, expected "
            f"{NESTED_LS_GATE6_AGGREGATION!r}"
        )
    repeats = int(payload["repeats"])
    if repeats != SWEEP_REPEATS:
        raise SystemExit(
            f"--omp-evidence repeats is {repeats}, expected {SWEEP_REPEATS}"
        )
    omp_set = tuple(int(value) for value in payload["omp_set"])
    if not omp_set or len(set(omp_set)) != len(omp_set):
        raise SystemExit("--omp-evidence omp_set is empty or contains duplicates")
    if omp_set != expected_omp_set:
        raise SystemExit(
            f"--omp-evidence omp_set is {omp_set!r}, expected the frozen host "
            f"set {expected_omp_set!r}"
        )
    rows = payload["rows"]
    expected_row_keys = {
        (repeat, value) for repeat in range(repeats) for value in omp_set
    }
    observed_row_keys: set[tuple[int, int]] = set()
    per_omp_walls: dict[int, list[float]] = {value: [] for value in omp_set}
    for row in rows:
        row_key = (int(row["repeat"]), int(row["omp_num_threads"]))
        if row_key in observed_row_keys:
            raise SystemExit(f"--omp-evidence repeats row {row_key}")
        observed_row_keys.add(row_key)
        if row.get("success") is not True:
            raise SystemExit(f"--omp-evidence row {row_key} did not succeed")
        if row.get("omp_pinned") is not True:
            raise SystemExit(f"--omp-evidence row {row_key} was not OMP-pinned")
        if int(row["observed_omp_num_threads"]) != row_key[1]:
            raise SystemExit(
                f"--omp-evidence row {row_key} observed a different OMP count"
            )
        if "child_schema" not in row:
            raise SystemExit(
                f"--omp-evidence row {row_key} declares no child schema; it "
                "predates the source-binding contract and cannot be cited"
            )
        if str(row["child_schema"]) != NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA:
            raise SystemExit(f"--omp-evidence row {row_key} has a stale child schema")
        if row_key[1] in per_omp_walls:
            wall = _require_finite_number(
                row["process_wall_seconds"],
                label=f"--omp-evidence row {row_key} process wall",
                positive=True,
            )
            per_omp_walls[row_key[1]].append(wall)
    if observed_row_keys != expected_row_keys:
        raise SystemExit(
            "--omp-evidence rows do not cover every declared "
            "(repeat, omp_num_threads) pair exactly once"
        )
    recomputed_best = min(
        omp_set,
        key=lambda value: (min(per_omp_walls[value]), value),
    )
    declared_minima = payload["per_omp_min_process_wall_seconds"]
    for value in omp_set:
        declared_minimum = _require_finite_number(
            declared_minima[str(value)],
            label=f"--omp-evidence minimum wall for OMP {value}",
            positive=True,
        )
        if declared_minimum != min(per_omp_walls[value]):
            raise SystemExit(f"--omp-evidence minimum wall for OMP {value} is stale")
    best = int(payload["best_omp_num_threads"])
    if best != recomputed_best:
        raise SystemExit(
            f"--omp-evidence declares best OMP {best}, but its rows recompute "
            f"to {recomputed_best}"
        )
    if best != int(omp_num_threads):
        raise SystemExit(
            f"--omp {omp_num_threads} is not the swept best-of-contract "
            f"{best}; the claim must run at the swept bar"
        )
    return {
        "path": str(omp_evidence),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "best_omp_num_threads": best,
        "rows": int(len(rows)),
        "omp_set": list(omp_set),
        "git_head": artifact_git_head,
    }


def _require_b3_green(
    *,
    b3_receipt: Path,
    omp_num_threads: int,
    expected_git_head: str,
    expected_maxcor: int,
    expected_omp_set: tuple[int, ...],
) -> dict[str, object]:
    """Refuse B37 unless the handed B3 receipt is a green B3 at this OMP."""

    if not b3_receipt.is_file():
        raise SystemExit(f"--b3-receipt does not exist: {b3_receipt}")
    raw = b3_receipt.read_bytes()
    payload = _load_strict_evidence_json(raw, label="--b3-receipt")
    schema = str(payload["schema"])
    if schema != CLAIM_SCHEMA:
        raise SystemExit(
            f"--b3-receipt schema is {schema!r}, expected {CLAIM_SCHEMA!r}"
        )
    receipt_git_head = str(payload["git_head"])
    if receipt_git_head != expected_git_head:
        raise SystemExit(
            f"--b3-receipt git_head is {receipt_git_head!r}, expected the "
            f"B37 implementation {expected_git_head!r}"
        )
    boundary = payload["claim_boundary"]
    receipt_budget = int(boundary["budget"])
    if receipt_budget != B3_BUDGET:
        raise SystemExit(f"--b3-receipt is a B{receipt_budget} run, not B{B3_BUDGET}")
    if int(boundary["maxcor"]) != expected_maxcor:
        raise SystemExit(
            f"--b3-receipt maxcor is {boundary['maxcor']!r}, expected {expected_maxcor}"
        )
    reason = payload["fail_closed_reason"]
    if reason is not None:
        raise SystemExit(f"--b3-receipt is not physics-green: {reason!r}")
    pairs = payload["pairs"]
    if len(pairs) != REPEATS or int(boundary["repeats"]) != REPEATS:
        raise SystemExit(f"--b3-receipt must carry exactly {REPEATS} chartered pairs")
    observed_repeats: set[int] = set()
    recomputed_gaps: list[float] = []
    for pair in pairs:
        repeat = int(pair["repeat"])
        if repeat in observed_repeats:
            raise SystemExit(f"--b3-receipt repeats pair {repeat}")
        observed_repeats.add(repeat)
        for lane, row in (("native", pair["native"]), ("jax", pair["jax"])):
            if int(row.get("repeat", -1)) != repeat:
                raise SystemExit(
                    f"--b3-receipt {lane} row repeat disagrees at pair {repeat}"
                )
            if row.get("role") != "measure" or row.get("timed") is not True:
                raise SystemExit(
                    f"--b3-receipt {lane} row is not a timed measure at repeat {repeat}"
                )
            process_wall = _require_finite_number(
                row.get("process_wall_seconds"),
                label=f"--b3-receipt {lane} process wall at repeat {repeat}",
                positive=True,
            )
            claim_wall = _require_finite_number(
                row.get("claim_wall_seconds"),
                label=f"--b3-receipt {lane} claim wall at repeat {repeat}",
                positive=True,
            )
            if process_wall != claim_wall:
                raise SystemExit(
                    f"--b3-receipt {lane} claim wall excludes work at repeat {repeat}"
                )
        for lane, envelope in (
            ("native", pair["native_rejudge"]),
            ("jax", pair["jax_rejudge"]),
        ):
            if int(envelope.get("repeat", -1)) != repeat:
                raise SystemExit(
                    f"--b3-receipt {lane} rejudge repeat disagrees at pair {repeat}"
                )
            if envelope.get("timed") is not False:
                raise SystemExit(
                    f"--b3-receipt {lane} rejudge must be untimed at repeat {repeat}"
                )
        recomputed_reason = _physics_ok(
            native=pair["native"],
            jax_row=pair["jax"],
            native_rejudge=pair["native_rejudge"],
            jax_rejudge=pair["jax_rejudge"],
            omp_num_threads=omp_num_threads,
            j_parity_rtol=None,
            budget=B3_BUDGET,
            maxcor=expected_maxcor,
        )
        if recomputed_reason is not None:
            raise SystemExit(
                "--b3-receipt has a failed pair at repeat "
                f"{repeat}: {recomputed_reason!r}"
            )
        if pair.get("physics_ok") is not True or pair["fail_closed_reason"] is not None:
            raise SystemExit(
                f"--b3-receipt pair {repeat} disagrees with recomputed physics"
            )
        native_j = _require_finite_number(
            pair["native"]["endpoint_j"],
            label=f"--b3-receipt native endpoint J at repeat {repeat}",
            positive=True,
        )
        jax_j = _require_finite_number(
            pair["jax"]["endpoint_j"],
            label=f"--b3-receipt JAX endpoint J at repeat {repeat}",
            nonnegative=True,
        )
        signed_gap = (jax_j - native_j) / abs(native_j)
        recomputed_gap = max(0.0, signed_gap)
        declared_native_j = _require_finite_number(
            pair.get("endpoint_j_native"),
            label=f"--b3-receipt endpoint_j_native at repeat {repeat}",
            positive=True,
        )
        declared_jax_j = _require_finite_number(
            pair.get("endpoint_j_jax"),
            label=f"--b3-receipt endpoint_j_jax at repeat {repeat}",
            nonnegative=True,
        )
        declared_signed_gap = _require_finite_number(
            pair.get("endpoint_j_rel_gap"),
            label=f"--b3-receipt signed endpoint-J gap at repeat {repeat}",
        )
        if (
            declared_native_j != native_j
            or declared_jax_j != jax_j
            or declared_signed_gap != signed_gap
        ):
            raise SystemExit(
                f"--b3-receipt pair {repeat} carries stale endpoint-J fields"
            )
        if pair.get("endpoint_j_within_frozen_band") is not None:
            raise SystemExit(
                f"--b3-receipt pair {repeat} must keep the B3 J band observational"
            )
        declared_gap = _require_finite_number(
            pair["endpoint_j_rel_gap_worse_direction"],
            label=f"--b3-receipt endpoint-J gap at repeat {repeat}",
            nonnegative=True,
        )
        if declared_gap != recomputed_gap:
            raise SystemExit(
                f"--b3-receipt pair {repeat} carries a stale endpoint-J gap"
            )
        recomputed_gaps.append(recomputed_gap)
    if observed_repeats != set(range(REPEATS)):
        raise SystemExit("--b3-receipt repeat indices are not the chartered set")
    receipt_omp = int(boundary["native_omp_num_threads"])
    if receipt_omp != int(omp_num_threads):
        raise SystemExit(
            f"--omp {omp_num_threads} does not match the B3 receipt's swept "
            f"native OMP {receipt_omp}; B37 must run at B3's bar"
        )
    # B37 inherits its swept bar transitively, so the inheritance has to be
    # of a swept bar: a B3 that only asserted its OMP cannot launder one.
    provenance = str(boundary["omp_provenance"])
    if provenance != OMP_PROVENANCE_SWEPT:
        raise SystemExit(
            f"--b3-receipt omp_provenance is {provenance!r}, expected "
            f"{OMP_PROVENANCE_SWEPT!r}; B37 cannot inherit an unswept bar"
        )
    # Charter Amendment 1: B3 measures the achievable fork band and B37
    # freezes it. A B3 receipt without the measurement cannot found a band.
    if "measured_j_rel_gap_max" not in boundary:
        raise SystemExit(
            "--b3-receipt has no claim_boundary.measured_j_rel_gap_max; it "
            "predates the Amendment-1 J-parity protocol and cannot feed B37"
        )
    measured = _require_finite_number(
        boundary["measured_j_rel_gap_max"],
        label="--b3-receipt measured_j_rel_gap_max",
        nonnegative=True,
    )
    if measured != max(recomputed_gaps):
        raise SystemExit(
            "--b3-receipt measured_j_rel_gap_max does not match its pair rows"
        )
    omp_evidence = boundary["omp_evidence"]
    if str(omp_evidence["git_head"]) != expected_git_head:
        raise SystemExit(
            "--b3-receipt inherited OMP evidence from a different implementation"
        )
    omp_path = Path(str(omp_evidence["path"]))
    if not omp_path.is_absolute():
        omp_path = REPO / omp_path
    validated_omp = _require_omp_evidence(
        omp_evidence=omp_path,
        omp_num_threads=omp_num_threads,
        expected_git_head=expected_git_head,
        expected_maxcor=expected_maxcor,
        expected_omp_set=expected_omp_set,
    )
    if str(validated_omp["sha256"]) != str(omp_evidence["sha256"]):
        raise SystemExit("--b3-receipt OMP evidence SHA does not match its artifact")
    native_walls = [
        _require_finite_number(
            pair["native"]["claim_wall_seconds"],
            label=f"--b3-receipt native wall at repeat {pair['repeat']}",
            positive=True,
        )
        for pair in pairs
    ]
    jax_walls = [
        _require_finite_number(
            pair["jax"]["claim_wall_seconds"],
            label=f"--b3-receipt JAX wall at repeat {pair['repeat']}",
            positive=True,
        )
        for pair in pairs
    ]
    native_min = min(native_walls)
    jax_min = min(jax_walls)
    expected_aggregates = {
        "native_min_process_wall_seconds": native_min,
        "native_median_process_wall_seconds": float(statistics.median(native_walls)),
        "native_max_process_wall_seconds": max(native_walls),
        "jax_min_process_wall_seconds": jax_min,
        "jax_median_process_wall_seconds": float(statistics.median(jax_walls)),
        "jax_max_process_wall_seconds": max(jax_walls),
        "speedup_min_over_min": native_min / jax_min,
    }
    for field, expected_value in expected_aggregates.items():
        declared_value = _require_finite_number(
            payload[field],
            label=f"--b3-receipt aggregate {field}",
            positive=True,
        )
        if declared_value != expected_value:
            raise SystemExit(f"--b3-receipt aggregate {field} is stale")
    expected_speed_claim = jax_min < native_min
    if boundary.get("nested_speed_claim") is not expected_speed_claim:
        raise SystemExit("--b3-receipt nested_speed_claim is stale")
    return {
        "path": str(b3_receipt),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_head": receipt_git_head,
        "native_omp_num_threads": receipt_omp,
        "omp_provenance": provenance,
        "measured_j_rel_gap_max": measured,
        "pairs": int(len(pairs)),
    }


def _jax_env() -> dict[str, str]:
    env = dict(os.environ)
    env["SIMSOPT_BACKEND_MODE"] = "jax_gpu_fast"
    env["JAX_PLATFORMS"] = "cuda,cpu"
    env["JAX_ENABLE_X64"] = "1"
    env.pop("SIMSOPT_DISABLE_JAX_COMPILATION_CACHE", None)
    CACHE_OUTER.mkdir(parents=True, exist_ok=True)
    env["JAX_COMPILATION_CACHE_DIR"] = str(CACHE_OUTER)
    env["SIMSOPT_JAX_COMPILATION_CACHE_DIR"] = str(CACHE_OUTER)
    return env


def _child_json_path(prefix: str) -> Path:
    # Child payloads live under the repo's ignored artifact tree, never the
    # shared /tmp: a usrquota-exhausted /tmp killed a live run's shell
    # environment on 2026-08-24 and would have destroyed the run itself at
    # the next payload write. The artifact filesystem is the one whose
    # health the campaign already depends on.
    tmp_dir = REPO / ".artifacts" / "nested-ls-outer-tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix=prefix, dir=tmp_dir, delete=False
    ) as handle:
        return Path(handle.name)


def _run_child(
    *,
    label: str,
    script: Path,
    extra_argv: list[str],
    env: dict[str, str],
    out_path: Path,
    expected_schema: str,
) -> tuple[dict[str, object], float]:
    """Full parent subprocess wait around one child. No subtraction."""

    started = time.perf_counter()
    completed = subprocess.run(
        [PYTHON, str(script), str(out_path), *extra_argv],
        cwd=str(REPO),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    process_wall_seconds = float(time.perf_counter() - started)
    if completed.returncode != 0 or not out_path.is_file():
        raise RuntimeError(
            f"outer {label} child failed rc={completed.returncode} "
            f"{_child_failure_evidence(out_path, expected_schema=expected_schema)}"
            f"stderr={completed.stderr[-2000:]}"
        )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _require_child_schema(
        payload,
        label=label,
        expected_schema=expected_schema,
    )
    return payload, process_wall_seconds


def _child_failure_evidence(out_path: Path, *, expected_schema: str) -> str:
    """Name the honest-failure receipt a nonzero child left behind, if any.

    A child that fails closed at the accepted-step callback writes its
    complete payload — ledger rows, committed incumbent, telemetry — and
    only then exits nonzero. Without this the parent would report a return
    code and a stderr tail and never mention the evidence the child went
    out of its way to preserve.

    This runs only on the failure path, and it is spliced into the message
    of the ``RuntimeError`` the caller is already raising. It therefore has
    one hard obligation: **never make the report worse than saying nothing**.
    A child that died mid-write is the expected cause here — the recorded
    incident that moved child payloads off a quota-exhausted ``/tmp`` was
    exactly a truncated write — so a half-written file must degrade to a
    note, not to a decode traceback that swallows the return code and the
    stderr tail. Hence the widened guard and the shape check: ``json.loads``
    accepts ``3`` and ``[]`` as valid JSON, and ``.get`` on those raises.
    """

    if not out_path.is_file():
        return ""
    if out_path.stat().st_size == 0:
        # `_child_json_path` creates the file before the child starts
        # (`NamedTemporaryFile(delete=False)`), so an empty file means the
        # child wrote nothing — a segfault, an OOM kill, an import error.
        # Calling that "unreadable" would accuse it of leaving a corrupt
        # receipt, which is the opposite of the truth.
        return ""
    try:
        payload = json.loads(out_path.read_bytes())
    except (OSError, ValueError):
        # OSError: unreadable. ValueError: covers both JSONDecodeError and
        # the UnicodeDecodeError a non-UTF-8 truncation raises.
        return f"child_payload={out_path} (unreadable) "
    if not isinstance(payload, dict):
        return f"child_payload={out_path} (not a JSON object) "
    if payload.get("schema") != expected_schema:
        return ""
    if "child_fault_reason" not in payload:
        return f"child_payload={out_path} (no fault reason recorded) "
    return (
        f"child_fault_reason={payload['child_fault_reason']!r} "
        f"child_payload={out_path} "
    )


def _require_child_schema(
    payload: dict[str, object], *, label: str, expected_schema: str
) -> None:
    """Refuse a child payload whose producer contract is not this campaign's."""

    observed_schema = str(payload.get("schema"))
    if observed_schema != expected_schema:
        raise RuntimeError(
            f"outer {label} child schema is {observed_schema!r}, "
            f"expected {expected_schema!r}"
        )


def _launch_native(
    *,
    omp_num_threads: int,
    budget: int,
    maxcor: int,
    log: Callable[[str], None],
) -> tuple[dict[str, object], Path]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(omp_num_threads)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env.pop("SIMSOPT_BACKEND_MODE", None)
    child_out = _child_json_path("nested_ls_outer_native_")
    payload, process_wall_seconds = _run_child(
        label="native",
        script=NATIVE_CHILD,
        extra_argv=["--budget", str(budget), "--maxcor", str(maxcor)],
        env=env,
        out_path=child_out,
        expected_schema=NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
    )
    threading = payload["threading"]
    start = payload["start"]
    endpoint = payload["endpoint"]
    child_payload_raw = child_out.read_text(encoding="utf-8")
    row = {
        "side": "native",
        "omp_num_threads": int(omp_num_threads),
        "observed_omp_num_threads": int(payload["omp_num_threads"]),
        "omp_pinned": bool(payload["omp_pinned"]),
        "omp_proc_bind": threading["OMP_PROC_BIND"],
        "omp_places": threading["OMP_PLACES"],
        "success": bool(payload["success"]),
        "nit": int(payload["nit"]),
        "nfev": int(payload["nfev"]),
        "restart_count": int(payload["restart_count"]),
        "endpoint_is_optimizer_endpoint": bool(payload["endpoint_is_optimizer_x"]),
        "outer_policy": payload["outer_policy"],
        "endpoint_j": float(endpoint["objective"]),
        "endpoint_iota": float(endpoint["iota"]),
        "endpoint_g": float(endpoint["G"]),
        "endpoint_gradient_l2": float(endpoint["gradient_l2"]),
        "endpoint_coil_sha256": str(endpoint["coil_sha256"]),
        "endpoint_surface_sha256": str(endpoint["surface_sha256"]),
        "start_coil_dofs": [float(entry) for entry in start["coil_dofs"]],
        "endpoint_coil_dofs": [float(entry) for entry in endpoint["coil_dofs"]],
        "process_wall_seconds": process_wall_seconds,
        "claim_wall_seconds": process_wall_seconds,
        "child_payload": payload,
        "child_payload_raw": child_payload_raw,
        "child_payload_sha256": hashlib.sha256(
            child_payload_raw.encode("utf-8")
        ).hexdigest(),
    }
    log(
        "outer native"
        f" success={row['success']!r} nit={row['nit']} nfev={row['nfev']}"
        f" J={row['endpoint_j']!r} iota={row['endpoint_iota']!r}"
        f" wall={process_wall_seconds!r} omp={omp_num_threads}"
    )
    return row, child_out


def _launch_jax(
    *, budget: int, maxcor: int, log: Callable[[str], None]
) -> tuple[dict[str, object], Path]:
    child_out = _child_json_path("nested_ls_outer_jax_")
    payload, process_wall_seconds = _run_child(
        label="jax",
        script=JAX_CHILD,
        extra_argv=["--budget", str(budget), "--maxcor", str(maxcor)],
        env=_jax_env(),
        out_path=child_out,
        expected_schema=NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
    )
    child_payload_raw = child_out.read_text(encoding="utf-8")
    row = {
        "side": "jax",
        "success": bool(payload["success"]),
        "nit": int(payload["nit"]),
        "nfev": int(payload["nfev"]),
        "restart_count": int(payload["restart_count"]),
        "endpoint_is_optimizer_endpoint": bool(payload["endpoint_is_optimizer_x"]),
        "outer_policy": payload["outer_policy"],
        "start_policy": str(payload["start_policy"]),
        "iota_branch_guard": float(payload["iota_branch_guard"]),
        "feasible_evaluations": int(payload["feasible_evaluations"]),
        "rejected_evaluations": int(payload["rejected_evaluations"]),
        "endpoint_j": float(payload["endpoint_j"]),
        "endpoint_grad_l2": float(payload["endpoint_grad_l2"]),
        "endpoint_grad_inf": float(payload["endpoint_grad_inf"]),
        "endpoint_iota": float(payload["endpoint_iota"]),
        "endpoint_g": float(payload["endpoint_g"]),
        "endpoint_adjoint_live_eta": float(payload["endpoint_adjoint_live_eta"]),
        "endpoint_coil_sha256": str(payload["endpoint_coil_sha256"]),
        "start_coil_dofs": [float(entry) for entry in payload["start_coil_dofs"]],
        "endpoint_coil_dofs": [float(entry) for entry in payload["endpoint_coil_dofs"]],
        "endpoint_surface_sha256": str(payload["endpoint_surface_sha256"]),
        "process_wall_seconds": process_wall_seconds,
        "claim_wall_seconds": process_wall_seconds,
        "child_payload": payload,
        "child_payload_raw": child_payload_raw,
        "child_payload_sha256": hashlib.sha256(
            child_payload_raw.encode("utf-8")
        ).hexdigest(),
    }
    log(
        "outer jax"
        f" success={row['success']!r} nit={row['nit']} nfev={row['nfev']}"
        f" feasible={row['feasible_evaluations']}"
        f" rejected={row['rejected_evaluations']}"
        f" J={row['endpoint_j']!r} grad_l2={row['endpoint_grad_l2']!r}"
        f" wall={process_wall_seconds!r}"
    )
    return row, child_out


def _launch_rejudge(
    *,
    lane: str,
    endpoint_path: Path,
    budget: int,
    maxcor: int,
    expected_child_payload_sha256: str,
    log: Callable[[str], None],
) -> dict[str, object]:
    """Untimed physics gate on one lane's endpoint, after every timed pair.

    The same gate judges both lanes: the charter's reconstruct no-op is a
    per-lane-endpoint physics gate, so the native endpoint earns it too.
    """

    observed_child_payload_sha256 = hashlib.sha256(
        endpoint_path.read_bytes()
    ).hexdigest()
    if observed_child_payload_sha256 != expected_child_payload_sha256:
        raise RuntimeError(
            f"rejudge {lane} endpoint payload changed after the timed child: "
            f"{observed_child_payload_sha256} != {expected_child_payload_sha256}"
        )
    child_out = _child_json_path("nested_ls_outer_rejudge_")
    payload, process_wall_seconds = _run_child(
        label="rejudge",
        script=JAX_CHILD,
        extra_argv=[
            "--budget",
            str(budget),
            "--maxcor",
            str(maxcor),
            "--rejudge-endpoint",
            str(endpoint_path),
        ],
        env=_jax_env(),
        out_path=child_out,
        expected_schema=NESTED_LS_OUTER_REJUDGE_SCHEMA,
    )
    if str(payload["judged_lane"]) != lane:
        raise RuntimeError(
            f"rejudge judged lane {payload['judged_lane']!r}, expected {lane!r}"
        )
    # Bind by the child's exact bytes, never by a re-encode. The timed-child
    # binding above already works this way (``child_payload_raw`` +
    # ``_verified_embedded_child_payload``); the rejudge binding used to
    # re-serialize the parsed payload to check its own digest, which cannot
    # round-trip: the child writes compact and insertion-ordered, the parent
    # re-encoded with ``indent=2``, and a reload re-encodes in sorted order
    # because the receipt writer sets ``sort_keys=True``. Every claim.v2 pair
    # would have failed closed on ``{lane}_rejudge_payload_sha256_mismatch``.
    payload_raw = child_out.read_text(encoding="utf-8")
    payload_sha256 = hashlib.sha256(payload_raw.encode("utf-8")).hexdigest()
    child_out.unlink(missing_ok=True)
    log(
        "outer rejudge"
        f" lane={lane} noop={payload['rejudge_noop']!r}"
        f" iter={payload['native_rejudge_iter']!r}"
        f" native_grad_l2={payload['native_rejudge_grad_l2']!r}"
        f" reduced_grad_l2={payload['reduced_grad_l2']!r}"
        f" reason={payload['fail_closed_reason']!r}"
        f" wall={process_wall_seconds!r}"
    )
    return {
        "payload": payload,
        "payload_raw": payload_raw,
        "payload_sha256": payload_sha256,
        "process_wall_seconds": process_wall_seconds,
        "timed": False,
    }


def _rejudge_endpoint_mismatch(
    *,
    lane: str,
    row: dict[str, object],
    rejudge_envelope: dict[str, object],
    budget: int,
    maxcor: int,
) -> str | None:
    rejudge = rejudge_envelope["payload"]
    payload_raw = rejudge_envelope.get("payload_raw")
    if not isinstance(payload_raw, str):
        return f"{lane}_rejudge_payload_raw_missing"
    if hashlib.sha256(payload_raw.encode("utf-8")).hexdigest() != str(
        rejudge_envelope["payload_sha256"]
    ):
        return f"{lane}_rejudge_payload_sha256_mismatch"
    if (
        json.loads(payload_raw, parse_constant=_reject_nonfinite_json_constant)
        != rejudge
    ):
        return f"{lane}_rejudge_embedded_payload_mismatch"
    if str(rejudge["schema"]) != NESTED_LS_OUTER_REJUDGE_SCHEMA:
        return f"{lane}_rejudge_schema_mismatch"
    if str(rejudge["judged_lane"]) != lane:
        return f"{lane}_rejudge_lane_mismatch"
    if int(rejudge["budget"]) != budget:
        return f"{lane}_rejudge_budget_mismatch"
    if int(rejudge["maxcor"]) != maxcor:
        return f"{lane}_rejudge_maxcor_mismatch"
    if str(rejudge["source_child_schema"]) != str(row["child_payload"]["schema"]):
        return f"{lane}_rejudge_source_child_schema_mismatch"
    if str(rejudge["source_child_payload_sha256"]) != str(row["child_payload_sha256"]):
        return f"{lane}_rejudge_source_child_payload_sha256_mismatch"
    for field in ("endpoint_coil_sha256", "endpoint_surface_sha256"):
        if str(rejudge[field]) != str(row[field]):
            return f"{lane}_rejudge_{field}_mismatch"
    for row_field, rejudge_field in (
        ("endpoint_j", "endpoint_j"),
        ("endpoint_iota", "endpoint_iota"),
        ("endpoint_g", "endpoint_g"),
    ):
        if float(rejudge[rejudge_field]) != float(row[row_field]):
            return f"{lane}_rejudge_{rejudge_field}_mismatch"
    return None


def _physics_ok(
    *,
    native: dict[str, object],
    jax_row: dict[str, object],
    native_rejudge: dict[str, object],
    jax_rejudge: dict[str, object],
    omp_num_threads: int,
    j_parity_rtol: float | None,
    budget: int,
    maxcor: int,
) -> str | None:
    for lane, row, expected_schema in (
        ("native", native, NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA),
        ("jax", jax_row, NESTED_LS_OUTER_JAX_CHILD_SCHEMA),
    ):
        child_payload, payload_mismatch = _verified_embedded_child_payload(
            lane=lane,
            row=row,
        )
        if payload_mismatch is not None or child_payload is None:
            return payload_mismatch
        if str(child_payload["schema"]) != expected_schema:
            return f"{lane}_child_schema_mismatch"
        if int(child_payload["budget"]) != budget:
            return f"{lane}_child_budget_mismatch"
        if int(child_payload["maxcor"]) != maxcor:
            return f"{lane}_child_maxcor_mismatch"
        boolean_fields = ["success", "endpoint_is_optimizer_x"]
        if lane == "native":
            boolean_fields.append("omp_pinned")
        for field in boolean_fields:
            if not isinstance(child_payload.get(field), bool):
                return f"{lane}_child_{field}_not_boolean"
        row_mismatch = _child_row_mismatch(
            lane=lane,
            row=row,
            payload=child_payload,
        )
        if row_mismatch is not None:
            return row_mismatch
    if native.get("success") is not True:
        return "native_failed"
    if native.get("omp_pinned") is not True:
        return "native_omp_unpinned"
    if int(native["observed_omp_num_threads"]) != int(omp_num_threads):
        return "native_omp_not_contract"
    if native.get("endpoint_is_optimizer_endpoint") is not True:
        return "native_endpoint_not_optimizer_endpoint"
    if jax_row.get("success") is not True:
        return "jax_failed"
    if jax_row.get("endpoint_is_optimizer_endpoint") is not True:
        return "jax_endpoint_not_optimizer_endpoint"
    if native["outer_policy"] != jax_row["outer_policy"]:
        return "outer_optimizer_policy_differs"
    if native["start_coil_dofs"] != jax_row["start_coil_dofs"]:
        return "start_coils_differ"
    if len(native["endpoint_coil_dofs"]) != len(jax_row["endpoint_coil_dofs"]):
        return "endpoint_coil_dim_mismatch"
    for lane, row, rejudge_envelope in (
        ("native", native, native_rejudge),
        ("jax", jax_row, jax_rejudge),
    ):
        mismatch = _rejudge_endpoint_mismatch(
            lane=lane,
            row=row,
            rejudge_envelope=rejudge_envelope,
            budget=budget,
            maxcor=maxcor,
        )
        if mismatch is not None:
            return mismatch
    native_rejudge_payload = native_rejudge["payload"]
    jax_rejudge_payload = jax_rejudge["payload"]
    native_rejudge_reason = native_rejudge_payload["fail_closed_reason"]
    if native_rejudge_reason is not None:
        return f"native_{native_rejudge_reason}"
    jax_rejudge_reason = jax_rejudge_payload["fail_closed_reason"]
    if jax_rejudge_reason is not None:
        return f"jax_{jax_rejudge_reason}"
    # Charter Amendment 1: at B3 the endpoint-J comparison is observational
    # — lane-vs-lane gradient agreement is ~1e-8, so budget-truncated
    # trajectories fork and a hard band here is a known false-reject class.
    # B3 measures the band; B37 gates on the value frozen from it.
    if j_parity_rtol is None:
        return None
    native_j = float(native["endpoint_j"])
    jax_j = float(jax_row["endpoint_j"])
    if not math.isfinite(native_j) or native_j <= 0.0:
        return "native_endpoint_j_invalid"
    if not math.isfinite(jax_j) or jax_j < 0.0:
        return "jax_endpoint_j_invalid"
    if not math.isfinite(j_parity_rtol) or j_parity_rtol < 0.0:
        return "j_parity_rtol_invalid"
    if jax_j > native_j * (1.0 + j_parity_rtol):
        return "jax_endpoint_j_above_frozen_band"
    return None


def _parse_omp_set(raw: str) -> tuple[int, ...]:
    values = tuple(int(entry.strip()) for entry in str(raw).split(",") if entry.strip())
    if not values:
        raise SystemExit(f"--omp-set parsed to nothing: {raw!r}")
    if len(set(values)) != len(values):
        raise SystemExit(f"--omp-set repeats a value: {raw!r}")
    if any(value < 1 for value in values):
        raise SystemExit(f"--omp-set has a non-positive value: {raw!r}")
    return values


def _host_omp_set(tag: str) -> tuple[int, ...]:
    if tag == "":
        return F3_B37_BANANA_OMP_CONTRACT_THREADS
    if tag == "a100":
        return A100_OMP_SET
    raise SystemExit(
        f"--tag {tag!r} has no frozen native OMP set; declare it in the contract "
        "before running a claim"
    )


def _run_native_omp_sweep(
    *,
    budget: int,
    omp_values: tuple[int, ...],
    maxcor: int,
    tag: str,
    sha: str,
) -> None:
    """Sweep the native denominator. No JAX lane, no prime, no pairs.

    This produces the artifact a claim run cites for ``--omp``. Repeats
    are interleaved across the whole set rather than run back to back, so
    a thermal or scheduling drift lands on every value instead of
    concentrating on whichever one happened to run last.
    """

    suffix = f".{tag}" if tag else ""
    stem = f"nested_ls_outer_native_omp_sweep_{EVIDENCE_DATE}{suffix}"
    out_json = EVIDENCE / f"{stem}.json"
    out_log = EVIDENCE / f"{stem}.log"
    log = _make_logger(out_log)
    log(
        f"outer native omp sweep budget {budget} maxcor {maxcor}"
        f" set {','.join(str(value) for value in omp_values)}"
        f" repeats {SWEEP_REPEATS} git_head {sha}"
    )
    rows: list[dict[str, object]] = []
    for repeat in range(SWEEP_REPEATS):
        for omp_num_threads in omp_values:
            row, child_out = _launch_native(
                omp_num_threads=omp_num_threads,
                budget=budget,
                maxcor=maxcor,
                log=log,
            )
            child_out.unlink(missing_ok=True)
            rows.append(
                {
                    "omp_num_threads": int(omp_num_threads),
                    "observed_omp_num_threads": int(row["observed_omp_num_threads"]),
                    "omp_pinned": bool(row["omp_pinned"]),
                    "repeat": int(repeat),
                    "success": bool(row["success"]),
                    "child_schema": str(row["child_payload"]["schema"]),
                    "nit": int(row["nit"]),
                    "nfev": int(row["nfev"]),
                    "endpoint_j": float(row["endpoint_j"]),
                    "endpoint_iota": float(row["endpoint_iota"]),
                    "process_wall_seconds": float(row["process_wall_seconds"]),
                }
            )
    for row in rows:
        if not bool(row["success"]):
            raise SystemExit(
                f"native sweep run failed at omp {row['omp_num_threads']} "
                f"repeat {row['repeat']}"
            )
        if not bool(row["omp_pinned"]):
            raise SystemExit(
                f"native sweep run at omp {row['omp_num_threads']} "
                "did not observe a pinned OMP_NUM_THREADS"
            )
        if int(row["observed_omp_num_threads"]) != int(row["omp_num_threads"]):
            raise SystemExit(
                f"native sweep run asked for omp {row['omp_num_threads']} but "
                f"observed {row['observed_omp_num_threads']}"
            )
    per_omp_min = {
        value: min(
            float(row["process_wall_seconds"])
            for row in rows
            if int(row["omp_num_threads"]) == value
        )
        for value in omp_values
    }
    # Ties resolve to the smaller thread count: same wall, less machine.
    best_omp = min(omp_values, key=lambda value: (per_omp_min[value], value))
    payload: dict[str, object] = {
        "aggregation": NESTED_LS_GATE6_AGGREGATION,
        "best_omp_num_threads": int(best_omp),
        "budget": int(budget),
        "claim": None,
        "command": (
            "JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python "
            "benchmarks/nested_ls_outer_claim.py --sweep-native-omp"
            f" --budget {budget} --maxcor {maxcor}"
            f" --omp-set {','.join(str(value) for value in omp_values)}"
            + (f" --tag {tag}" if tag else "")
        ),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "driver": "benchmarks.nested_ls_outer_claim",
        "execution_log": str(out_log.relative_to(REPO)),
        "git_head": sha,
        "interleaved_repeats": True,
        "jax_lane_run": False,
        "maxcor": int(maxcor),
        "omp_set": [int(value) for value in omp_values],
        "per_omp_min_process_wall_seconds": {
            str(value): per_omp_min[value] for value in omp_values
        },
        "publication": (
            f"Native nested-twin OMP sweep at B{budget}: full parent process "
            f"wall per run, {SWEEP_REPEATS} interleaved repeats per value, "
            "best-of-contract by min wall. Denominator evidence only; not a "
            "speed claim and not F3 7.70x."
        ),
        "repeats": int(SWEEP_REPEATS),
        "rows": rows,
        "schema": SWEEP_SCHEMA,
        "tag": tag or None,
        "written_by_pytest": False,
    }
    write_strict_json(out_json, payload)
    log(f"wrote {out_json}")
    log(
        f"best_omp_num_threads {best_omp}"
        f" min_wall {per_omp_min[best_omp]!r}"
        " per_omp_min "
        + " ".join(f"{value}:{per_omp_min[value]!r}" for value in omp_values)
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    budget = int(args.budget)
    omp_num_threads = int(args.omp)
    maxcor = int(args.maxcor)
    tag = str(args.tag).strip()
    host_omp_set = _host_omp_set(tag)
    if args.sweep_native_omp:
        if budget != B3_BUDGET:
            raise SystemExit(
                f"--sweep-native-omp runs at --budget {B3_BUDGET}: the sweep "
                "produces the B3 rung's denominator evidence"
            )
        for flag, value in (
            ("--b3-receipt", args.b3_receipt),
            ("--omp-evidence", args.omp_evidence),
            ("--j-parity-rtol", args.j_parity_rtol),
        ):
            if value is not None:
                raise SystemExit(f"{flag} is forbidden with --sweep-native-omp")
        omp_values = _parse_omp_set(
            DEFAULT_OMP_SET if args.omp_set is None else args.omp_set
        )
        if omp_values != host_omp_set:
            raise SystemExit(
                f"--omp-set is {omp_values!r}, expected the frozen host set "
                f"{host_omp_set!r} for tag {tag!r}"
            )
        _run_native_omp_sweep(
            budget=budget,
            omp_values=omp_values,
            maxcor=maxcor,
            tag=tag,
            sha=_require_clean_tree(),
        )
        return
    if args.omp_set is not None:
        raise SystemExit("--omp-set is only meaningful with --sweep-native-omp")
    if budget == B3_BUDGET:
        if args.b3_receipt is not None:
            raise SystemExit(
                f"--b3-receipt is forbidden for --budget {B3_BUDGET}: this run "
                "is the B3 rung, it cannot be gated on itself"
            )
        if args.j_parity_rtol is not None:
            raise SystemExit(
                f"--j-parity-rtol is forbidden for --budget {B3_BUDGET}: "
                "Amendment 1 has B3 measure the fork band, not gate on one"
            )
        if args.omp_evidence is None:
            raise SystemExit(
                f"--budget {B3_BUDGET} requires --omp-evidence: the charter "
                "sweeps the native denominator per rung, so the bar must come "
                "from a sweep artifact and not from the command line"
            )
    if budget == B37_BUDGET:
        if args.b3_receipt is None:
            raise SystemExit(
                f"--budget {B37_BUDGET} requires --b3-receipt: the charter "
                "runs B37 only after B3 lands physics-green"
            )
        if args.j_parity_rtol is None:
            raise SystemExit(
                f"--budget {B37_BUDGET} requires --j-parity-rtol: Amendment 1 "
                "freezes the band from B3's measurement before B37 runs"
            )
        if args.omp_evidence is not None:
            raise SystemExit(
                f"--omp-evidence is forbidden for --budget {B37_BUDGET}: the "
                "swept bar is inherited through --b3-receipt"
            )
    sha = _require_clean_tree()
    if args.omp_evidence is None:
        omp_evidence: dict[str, object] | None = None
    else:
        omp_evidence = _require_omp_evidence(
            omp_evidence=Path(args.omp_evidence),
            omp_num_threads=omp_num_threads,
            expected_git_head=sha,
            expected_maxcor=maxcor,
            expected_omp_set=host_omp_set,
        )
    if args.b3_receipt is None:
        b3_receipt: dict[str, object] | None = None
    else:
        b3_receipt = _require_b3_green(
            b3_receipt=Path(args.b3_receipt),
            omp_num_threads=omp_num_threads,
            expected_git_head=sha,
            expected_maxcor=maxcor,
            expected_omp_set=host_omp_set,
        )
    # Both rungs stand on a swept bar: B3 cites the sweep artifact, B37
    # inherits it through a B3 receipt that was itself required to cite one.
    omp_provenance = OMP_PROVENANCE_SWEPT
    if b3_receipt is None:
        j_parity_rtol: float | None = None
        j_parity_mode = "observational_b3"
        b3_measured_gap: float | None = None
    else:
        j_parity_rtol = _parse_j_parity_rtol(args.j_parity_rtol)
        j_parity_mode = "frozen_from_b3"
        b3_measured_gap = float(b3_receipt["measured_j_rel_gap_max"])
        if j_parity_rtol < b3_measured_gap:
            raise SystemExit(
                f"--j-parity-rtol {j_parity_rtol!r} is below the B3 receipt's "
                f"measured_j_rel_gap_max {b3_measured_gap!r}; the frozen band "
                "cannot be tighter than the fork B3 actually measured"
            )
    suffix = f".{tag}" if tag else ""
    stem = f"nested_ls_outer_b{budget}_{EVIDENCE_DATE}{suffix}"
    out_json = EVIDENCE / f"{stem}.json"
    out_log = EVIDENCE / f"{stem}.log"
    publication = (
        f"Nested-LS eight-term outer B{budget} claim run: moving-coil scipy "
        f"L-BFGS-B over the 11 coil DOFs at maxcor={maxcor}, JAX reduced "
        f"nested-LS inner versus the native nested twin at OMP="
        f"{omp_num_threads}. Full parent process wall on both sides, no "
        "subtraction; physics gates untimed. Not F3 7.70x."
    )
    log = _make_logger(out_log)
    log(
        f"outer prime jax cache budget {budget} maxcor {maxcor}"
        f" omp {omp_num_threads} omp_provenance {omp_provenance}"
        f" git_head {sha}"
    )
    if omp_evidence is not None:
        log(
            f"outer omp evidence path {omp_evidence['path']}"
            f" sha256 {omp_evidence['sha256']}"
            f" best_omp {omp_evidence['best_omp_num_threads']}"
            f" rows {omp_evidence['rows']}"
        )
    if b3_receipt is not None:
        log(
            f"outer b3 interlock path {b3_receipt['path']}"
            f" sha256 {b3_receipt['sha256']}"
            f" pairs {b3_receipt['pairs']}"
            f" native_omp {b3_receipt['native_omp_num_threads']}"
            f" measured_j_rel_gap_max {b3_receipt['measured_j_rel_gap_max']!r}"
            f" frozen j_parity_rtol {j_parity_rtol!r}"
        )
    if args.skip_prime:
        prime = {
            "role": "prime",
            "repeat": -1,
            "timed": False,
            "skipped": True,
            "reason": "amendment-4 fault-rerun: persistent compile cache warm",
        }
        log("outer prime skipped (amendment-4 fault-rerun, warm cache)")
    else:
        prime, prime_endpoint = _launch_jax(budget=budget, maxcor=maxcor, log=log)
        prime_endpoint.unlink(missing_ok=True)
        prime["role"] = "prime"
        prime["repeat"] = -1
        prime["timed"] = False

    measures: list[tuple[dict[str, object], dict[str, object], Path, Path]] = []
    native_walls: list[float] = []
    jax_walls: list[float] = []
    pairs_n = int(args.pairs)
    if pairs_n < 1:
        raise SystemExit(f"--pairs must be >= 1, got {pairs_n}")
    for repeat in range(pairs_n):
        native, native_endpoint = _launch_native(
            omp_num_threads=omp_num_threads,
            budget=budget,
            maxcor=maxcor,
            log=log,
        )
        native["role"] = "measure"
        native["repeat"] = int(repeat)
        native["timed"] = True
        jax_row, jax_endpoint = _launch_jax(budget=budget, maxcor=maxcor, log=log)
        jax_row["role"] = "measure"
        jax_row["repeat"] = int(repeat)
        jax_row["timed"] = True
        native_walls.append(float(native["claim_wall_seconds"]))
        jax_walls.append(float(jax_row["claim_wall_seconds"]))
        measures.append((native, jax_row, native_endpoint, jax_endpoint))

    # The lane's start policy is a property of the TIMED lane, so read it off
    # the timed rows and not off the prime. The prime is an untimed cache
    # warm-up that ``--skip-prime`` legitimately omits, and reading a published
    # claim field out of it meant every Amendment-5 fault-rerun died with a
    # KeyError at receipt time, after burning its whole wall.
    jax_start_policies = {str(row["start_policy"]) for _n, row, _ne, _je in measures}
    if len(jax_start_policies) != 1:
        raise SystemExit(
            "the timed JAX rows disagree on their start policy: "
            f"{sorted(jax_start_policies)}"
        )
    jax_start_policy = jax_start_policies.pop()

    pairs: list[dict[str, object]] = []
    fail_reason: str | None = None
    for repeat, (native, jax_row, native_endpoint, jax_endpoint) in enumerate(measures):
        native_rejudge = _launch_rejudge(
            lane="native",
            endpoint_path=native_endpoint,
            budget=budget,
            maxcor=maxcor,
            expected_child_payload_sha256=str(native["child_payload_sha256"]),
            log=log,
        )
        native_rejudge["repeat"] = int(repeat)
        jax_rejudge = _launch_rejudge(
            lane="jax",
            endpoint_path=jax_endpoint,
            budget=budget,
            maxcor=maxcor,
            expected_child_payload_sha256=str(jax_row["child_payload_sha256"]),
            log=log,
        )
        jax_rejudge["repeat"] = int(repeat)
        reason = _physics_ok(
            native=native,
            jax_row=jax_row,
            native_rejudge=native_rejudge,
            jax_rejudge=jax_rejudge,
            omp_num_threads=omp_num_threads,
            j_parity_rtol=j_parity_rtol,
            budget=budget,
            maxcor=maxcor,
        )
        native_j = float(native["endpoint_j"])
        jax_j = float(jax_row["endpoint_j"])
        # Signed, so the receipt shows which way the fork went; the band
        # measure below is one-sided because only "JAX worse" can fail.
        j_rel_gap = (jax_j - native_j) / abs(native_j)
        pairs.append(
            {
                "repeat": int(repeat),
                "native": native,
                "jax": jax_row,
                "native_rejudge": native_rejudge,
                "jax_rejudge": jax_rejudge,
                "endpoint_j_native": native_j,
                "endpoint_j_jax": jax_j,
                "endpoint_j_rel_gap": j_rel_gap,
                "endpoint_j_rel_gap_worse_direction": max(0.0, j_rel_gap),
                "endpoint_j_within_frozen_band": (
                    None
                    if j_parity_rtol is None
                    else bool(jax_j <= native_j * (1.0 + j_parity_rtol))
                ),
                "physics_ok": reason is None,
                "fail_closed_reason": reason,
            }
        )
        log(
            "outer pair"
            f" repeat={repeat} physics={reason is None}"
            f" native_wall={native['claim_wall_seconds']!r}"
            f" jax_wall={jax_row['claim_wall_seconds']!r}"
            f" native_J={native_j!r} jax_J={jax_j!r}"
            f" j_rel_gap={j_rel_gap!r}"
            f" reason={reason!r}"
        )
        if fail_reason is None and reason is not None:
            fail_reason = reason

    native_min = min(native_walls)
    jax_min = min(jax_walls)
    physics_ok = fail_reason is None
    nested_speed_claim = bool(physics_ok and jax_min < native_min)
    measured_j_rel_gap_max = max(
        float(pair["endpoint_j_rel_gap_worse_direction"]) for pair in pairs
    )
    payload: dict[str, object] = {
        "claim_boundary": {
            "aggregation": NESTED_LS_GATE6_AGGREGATION,
            "b3_measured_j_rel_gap_max": b3_measured_gap,
            "b3_receipt": b3_receipt,
            "budget": budget,
            "cap_2048_attempted": False,
            "comparable_operators": False,
            "endpoint_j_parity_one_sided": True,
            "explicit_inverse_m_production": False,
            "f3_sealed": True,
            "inherits_f3_7_70x": False,
            "interleaved_repeats": True,
            "jax_claim_clock": "parent_wait",
            "jax_persistent_cache": True,
            "jax_start_policy": jax_start_policy,
            # Amendment 1: B3 measures the achievable fork band and does not
            # gate on it; B37 gates at the value frozen from that receipt.
            "j_parity_mode": j_parity_mode,
            "j_parity_rtol": j_parity_rtol,
            # Charter Amendment 1: both lanes open on the raw un-nest
            # archived lane surface and pay their own start-point inner
            # convergence inside their own claim wall.
            "lane_start_work_symmetric": True,
            # The sealed rejection vocabulary both lanes' per-evaluation
            # ledgers use. Any evaluation neither lane can complete costs
            # both the same sentinel, so the reasons are shared, not
            # per-lane.
            "lane_rejection_reasons": ["iota_branch_guard", "inner_solve_failed"],
            "maxcor": maxcor,
            "measured_j_rel_gap_max": measured_j_rel_gap_max,
            "moving_coil": True,
            "native_claim_clock": "parent_wait",
            "native_start_policy": "start_point_inner_solve_inside_wall",
            "native_omp_num_threads": omp_num_threads,
            "nested_speed_claim": nested_speed_claim,
            "omp_evidence": omp_evidence,
            "omp_provenance": omp_provenance,
            "one_lane_per_process": True,
            "outer_optimizer_loop": True,
            "physics_gate_untimed": True,
            "rejudged_lanes": ["native", "jax"],
            "repeats": int(pairs_n),
            "subtractions": "none",
            "tag": tag or None,
            "trajectory_parity_claimed": False,
        },
        "command": (
            "SIMSOPT_BACKEND_MODE=jax_gpu_fast JAX_PLATFORMS=cuda,cpu "
            "JAX_ENABLE_X64=1 python benchmarks/nested_ls_outer_claim.py"
            f" --budget {budget} --omp {omp_num_threads} --maxcor {maxcor}"
            + (f" --tag {tag}" if tag else "")
            + (
                f" --omp-evidence {omp_evidence['path']}"
                if omp_evidence is not None
                else ""
            )
            + (f" --b3-receipt {b3_receipt['path']}" if b3_receipt is not None else "")
            + (
                f" --j-parity-rtol {j_parity_rtol!r}"
                if j_parity_rtol is not None
                else ""
            )
        ),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "driver": "benchmarks.nested_ls_outer_claim",
        "execution_log": str(out_log.relative_to(REPO)),
        "fail_closed_reason": fail_reason,
        "git_head": sha,
        "jax_compilation_cache_dir": str(CACHE_OUTER.relative_to(REPO)),
        "jax_max_process_wall_seconds": max(jax_walls),
        "jax_median_process_wall_seconds": float(statistics.median(jax_walls)),
        "jax_min_process_wall_seconds": jax_min,
        "native_max_process_wall_seconds": max(native_walls),
        "native_median_process_wall_seconds": float(statistics.median(native_walls)),
        "native_min_process_wall_seconds": native_min,
        "pairs": pairs,
        "prime": prime,
        "publication": publication,
        "schema": CLAIM_SCHEMA,
        "speedup_min_over_min": float(native_min / jax_min),
        "written_by_pytest": False,
    }
    write_strict_json(out_json, payload)
    log(f"wrote {out_json}")
    for _native, _jax_row, native_endpoint, jax_endpoint in measures:
        native_endpoint.unlink(missing_ok=True)
        jax_endpoint.unlink(missing_ok=True)
    log(
        f"ok {physics_ok} nested_speed_claim {nested_speed_claim}"
        f" jax_min {jax_min!r} native_min {native_min!r}"
        f" speedup {float(native_min / jax_min)!r}"
    )
    if not physics_ok:
        raise SystemExit(f"Outer B{budget} physics failed: {fail_reason}")


if __name__ == "__main__":
    main()
