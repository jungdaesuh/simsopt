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
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from simsopt_jax_adapters.geo.nested_ls_contract import (
    F3_B37_BANANA_OMP_CONTRACT_THREADS,
    NESTED_LS_GATE6_AGGREGATION,
    NESTED_LS_GATE6_NATIVE_OMP_THREADS,
    NESTED_LS_OUTER_OMP_SWEEP_REPEATS,
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
EVIDENCE_DATE: Final[str] = "20260823"
CLAIM_BUDGETS: Final[tuple[int, ...]] = (3, 37)
CLAIM_SCHEMA: Final[str] = "nested-ls-outer-claim.v1"
SWEEP_SCHEMA: Final[str] = "nested-ls-outer-native-omp-sweep.v1"
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
SWEEP_REPEATS: Final[int] = NESTED_LS_OUTER_OMP_SWEEP_REPEATS
# The charter sweeps the native denominator per rung. A number typed on
# the command line is not a swept bar, so claim runs must cite the sweep
# artifact and B37 must inherit that provenance through its B3 receipt.
OMP_PROVENANCE_SWEPT: Final[str] = "swept_artifact"


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
        "--tag",
        default="",
        help="Receipt suffix, e.g. a100 → nested_ls_outer_b3_20260823.a100.json",
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
    *, omp_evidence: Path, omp_num_threads: int
) -> dict[str, object]:
    """Refuse a claim run whose native OMP is not the swept artifact's best.

    The charter sweeps the native denominator per rung and takes
    best-of-contract as the bar. Binding the artifact here is what makes
    ``--omp`` a measurement rather than an assertion.
    """

    if not omp_evidence.is_file():
        raise SystemExit(f"--omp-evidence does not exist: {omp_evidence}")
    raw = omp_evidence.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    schema = str(payload["schema"])
    if schema != SWEEP_SCHEMA:
        raise SystemExit(
            f"--omp-evidence schema is {schema!r}, expected {SWEEP_SCHEMA!r}"
        )
    best = int(payload["best_omp_num_threads"])
    if best != int(omp_num_threads):
        raise SystemExit(
            f"--omp {omp_num_threads} is not the swept best-of-contract "
            f"{best}; the claim must run at the swept bar"
        )
    rows = payload["rows"]
    if not rows:
        raise SystemExit("--omp-evidence carries no rows")
    return {
        "path": str(omp_evidence),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "best_omp_num_threads": best,
        "rows": int(len(rows)),
        "omp_set": [int(value) for value in payload["omp_set"]],
        "git_head": str(payload["git_head"]),
    }


def _require_b3_green(*, b3_receipt: Path, omp_num_threads: int) -> dict[str, object]:
    """Refuse B37 unless the handed B3 receipt is a green B3 at this OMP."""

    if not b3_receipt.is_file():
        raise SystemExit(f"--b3-receipt does not exist: {b3_receipt}")
    raw = b3_receipt.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    schema = str(payload["schema"])
    if schema != CLAIM_SCHEMA:
        raise SystemExit(
            f"--b3-receipt schema is {schema!r}, expected {CLAIM_SCHEMA!r}"
        )
    boundary = payload["claim_boundary"]
    receipt_budget = int(boundary["budget"])
    if receipt_budget != B3_BUDGET:
        raise SystemExit(f"--b3-receipt is a B{receipt_budget} run, not B{B3_BUDGET}")
    reason = payload["fail_closed_reason"]
    if reason is not None:
        raise SystemExit(f"--b3-receipt is not physics-green: {reason!r}")
    pairs = payload["pairs"]
    if not pairs:
        raise SystemExit("--b3-receipt carries no pairs")
    for pair in pairs:
        if not bool(pair["physics_ok"]):
            raise SystemExit(
                "--b3-receipt has a failed pair at repeat "
                f"{pair['repeat']}: {pair['fail_closed_reason']!r}"
            )
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
    measured = float(boundary["measured_j_rel_gap_max"])
    return {
        "path": str(b3_receipt),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_head": str(payload["git_head"]),
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
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix=prefix, delete=False
    ) as handle:
        return Path(handle.name)


def _run_child(
    *,
    label: str,
    script: Path,
    extra_argv: list[str],
    env: dict[str, str],
    out_path: Path,
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
            f"stderr={completed.stderr[-2000:]}"
        )
    return (
        json.loads(out_path.read_text(encoding="utf-8")),
        process_wall_seconds,
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
    )
    threading = payload["threading"]
    start = payload["start"]
    endpoint = payload["endpoint"]
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
        "endpoint_is_optimizer_endpoint": bool(payload["endpoint_is_optimizer_x"]),
        "outer_policy": payload["outer_policy"],
        "endpoint_j": float(endpoint["objective"]),
        "endpoint_iota": float(endpoint["iota"]),
        "endpoint_g": float(endpoint["G"]),
        "endpoint_gradient_l2": float(endpoint["gradient_l2"]),
        "endpoint_surface_sha256": str(endpoint["surface_sha256"]),
        "start_coil_dofs": [float(entry) for entry in start["coil_dofs"]],
        "endpoint_coil_dofs": [float(entry) for entry in endpoint["coil_dofs"]],
        "process_wall_seconds": process_wall_seconds,
        "claim_wall_seconds": process_wall_seconds,
        "child_payload": payload,
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
    )
    row = {
        "side": "jax",
        "success": bool(payload["success"]),
        "nit": int(payload["nit"]),
        "nfev": int(payload["nfev"]),
        "endpoint_is_optimizer_endpoint": bool(payload["optimizer_endpoint_is_anchor"]),
        "outer_policy": payload["outer_policy"],
        "start_policy": str(payload["start_policy"]),
        "iota_branch_guard": float(payload["iota_branch_guard"]),
        "accepted_evaluations": int(payload["accepted_evaluations"]),
        "rejected_evaluations": int(payload["rejected_evaluations"]),
        "endpoint_j": float(payload["endpoint_j"]),
        "endpoint_grad_l2": float(payload["endpoint_grad_l2"]),
        "endpoint_grad_inf": float(payload["endpoint_grad_inf"]),
        "endpoint_iota": float(payload["endpoint_iota"]),
        "endpoint_g": float(payload["endpoint_g"]),
        "endpoint_adjoint_live_eta": float(payload["endpoint_adjoint_live_eta"]),
        "start_coil_dofs": [float(entry) for entry in payload["start_coil_dofs"]],
        "endpoint_coil_dofs": [float(entry) for entry in payload["endpoint_coil_dofs"]],
        "endpoint_surface_sha256": str(payload["endpoint_surface_sha256"]),
        "process_wall_seconds": process_wall_seconds,
        "claim_wall_seconds": process_wall_seconds,
        "child_payload": payload,
    }
    log(
        "outer jax"
        f" success={row['success']!r} nit={row['nit']} nfev={row['nfev']}"
        f" accepted={row['accepted_evaluations']}"
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
    log: Callable[[str], None],
) -> dict[str, object]:
    """Untimed physics gate on one lane's endpoint, after every timed pair.

    The same gate judges both lanes: the charter's reconstruct no-op is a
    per-lane-endpoint physics gate, so the native endpoint earns it too.
    """

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
    )
    child_out.unlink(missing_ok=True)
    if str(payload["judged_lane"]) != lane:
        raise RuntimeError(
            f"rejudge judged lane {payload['judged_lane']!r}, expected {lane!r}"
        )
    payload["process_wall_seconds"] = process_wall_seconds
    payload["timed"] = False
    log(
        "outer rejudge"
        f" lane={lane} noop={payload['rejudge_noop']!r}"
        f" iter={payload['native_rejudge_iter']!r}"
        f" native_grad_l2={payload['native_rejudge_grad_l2']!r}"
        f" reduced_grad_l2={payload['reduced_grad_l2']!r}"
        f" reason={payload['fail_closed_reason']!r}"
        f" wall={process_wall_seconds!r}"
    )
    return payload


def _physics_ok(
    *,
    native: dict[str, object],
    jax_row: dict[str, object],
    native_rejudge: dict[str, object],
    jax_rejudge: dict[str, object],
    omp_num_threads: int,
    j_parity_rtol: float | None,
) -> str | None:
    if not bool(native["success"]):
        return "native_failed"
    if not bool(native["omp_pinned"]):
        return "native_omp_unpinned"
    if int(native["observed_omp_num_threads"]) != int(omp_num_threads):
        return "native_omp_not_contract"
    if not bool(native["endpoint_is_optimizer_endpoint"]):
        return "native_endpoint_not_optimizer_endpoint"
    if not bool(jax_row["success"]):
        return "jax_failed"
    if not bool(jax_row["endpoint_is_optimizer_endpoint"]):
        return "jax_endpoint_not_optimizer_endpoint"
    if native["outer_policy"] != jax_row["outer_policy"]:
        return "outer_optimizer_policy_differs"
    if native["start_coil_dofs"] != jax_row["start_coil_dofs"]:
        return "start_coils_differ"
    if len(native["endpoint_coil_dofs"]) != len(jax_row["endpoint_coil_dofs"]):
        return "endpoint_coil_dim_mismatch"
    native_rejudge_reason = native_rejudge["fail_closed_reason"]
    if native_rejudge_reason is not None:
        return f"native_{native_rejudge_reason}"
    jax_rejudge_reason = jax_rejudge["fail_closed_reason"]
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
    if args.omp_evidence is None:
        omp_evidence: dict[str, object] | None = None
    else:
        omp_evidence = _require_omp_evidence(
            omp_evidence=Path(args.omp_evidence), omp_num_threads=omp_num_threads
        )
    if args.b3_receipt is None:
        b3_receipt: dict[str, object] | None = None
    else:
        b3_receipt = _require_b3_green(
            b3_receipt=Path(args.b3_receipt), omp_num_threads=omp_num_threads
        )
    # Both rungs stand on a swept bar: B3 cites the sweep artifact, B37
    # inherits it through a B3 receipt that was itself required to cite one.
    omp_provenance = OMP_PROVENANCE_SWEPT
    if b3_receipt is None:
        j_parity_rtol: float | None = None
        j_parity_mode = "observational_b3"
        b3_measured_gap: float | None = None
    else:
        j_parity_rtol = float(args.j_parity_rtol)
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
    sha = _require_clean_tree()
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
    prime, prime_endpoint = _launch_jax(budget=budget, maxcor=maxcor, log=log)
    prime_endpoint.unlink(missing_ok=True)
    prime["role"] = "prime"
    prime["repeat"] = -1
    prime["timed"] = False

    measures: list[tuple[dict[str, object], dict[str, object], Path, Path]] = []
    native_walls: list[float] = []
    jax_walls: list[float] = []
    for repeat in range(REPEATS):
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

    pairs: list[dict[str, object]] = []
    fail_reason: str | None = None
    for repeat, (native, jax_row, native_endpoint, jax_endpoint) in enumerate(measures):
        native_rejudge = _launch_rejudge(
            lane="native",
            endpoint_path=native_endpoint,
            budget=budget,
            maxcor=maxcor,
            log=log,
        )
        native_rejudge["repeat"] = int(repeat)
        jax_rejudge = _launch_rejudge(
            lane="jax",
            endpoint_path=jax_endpoint,
            budget=budget,
            maxcor=maxcor,
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
            "jax_start_policy": str(prime["start_policy"]),
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
            "repeats": int(REPEATS),
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
