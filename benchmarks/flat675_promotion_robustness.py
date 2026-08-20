"""Flat-675 promotion robustness harness (F4 work package C3).

Charter: ``docs/jax_flat675_promotion_plan.md`` requirement 4.  Every sealed
flat-675 result stands on ONE start candidate.  This harness produces the
evidence that bounds how far that generalizes, and it is deliberately built so
that the evidence cannot say more than it measured.

**The licensed claim is pre-committed and is the only claim this campaign may
support**, verbatim:

    robust to perturbations of the certified start at relative amplitudes
    <= 1e-1 (surface-block, coil-block, full-vector) and to one
    constructor-built start

Nothing stronger, whatever the outcomes.  **No timing claim is made.**  Any
seconds in the records are incidental and labelled so.

Layering, which is forced rather than chosen: the native oracle resolves its
imports from the pinned instrument tree, while the lane under test must
resolve only from the production tree, so the solves live in a child process
with a production-only path.  That is the F3 split, reused here.  The
invocation contract is therefore F3's::

    PYTHONPATH=<production-root>:<instrument-root>:<instrument-root>/src \\
        <runtime-env>/bin/python \\
        <production-root>/benchmarks/flat675_promotion_robustness.py run

Gates, exactly as chartered and applied per run:

* bundle-problem runs (the archived control and the nine perturbations) —
  finite endpoint, objective strictly decreased, and a native-oracle
  cross-evaluation that both accepts the endpoint and agrees with the fused
  objective within the fair bar's own ``ENDPOINT_OBJECTIVE_RTOL``.
* the constructor-built run — finite, monotone improvement, and endpoint
  gradient infinity norm strictly below the start's.  **The oracle does not
  apply**: it is wired to the archived bundle's native material, so it can
  only speak about candidates in that problem.  Its absence here is a scope
  statement, not a missing gate.

If any run fails, the harness records the failure precisely and stops.  It
does not add a guard: designing one is a reviewed decision, not a repair the
measurement is allowed to make on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final, Mapping, Sequence

from benchmarks.flat675_fused_campaign_contract import production_child_environment
from benchmarks.genuine_675_fair_bar import (
    ENDPOINT_OBJECTIVE_RTOL,
    INSTRUMENT_COMMIT,
    gpu_environment,
    run_oracle,
    write_provenance_shim,
)
from benchmarks.genuine_675_fair_bar import (
    SOURCE_ROOT as INSTRUMENT_ROOT,
)

PRODUCTION_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
CHILD: Final[Path] = (
    PRODUCTION_ROOT / "benchmarks" / "flat675_promotion_robustness_child.py"
)
OUTPUT_ROOT: Final[Path] = Path(
    "/home/jungdaesuh/simsopt_mixed_artifacts/flat675_promotion"
)

RECORD_SCHEMA: Final[str] = "flat675-promotion-robustness.v1"

# Chartered budget and seed.  Both are recorded in the note so a reader can
# reproduce any single row.
MAXITER: Final[int] = 37
SEED: Final[int] = 20260819

LICENSED_CLAIM: Final[str] = (
    "robust to perturbations of the certified start at relative amplitudes "
    "<= 1e-1 (surface-block, coil-block, full-vector) and to one "
    "constructor-built start"
)


class RobustnessError(RuntimeError):
    """A C3 contract was not satisfied."""


def _child_environment(*, cache_dir: Path, shim_dir: Path) -> dict[str, str]:
    """The F3 child environment, unchanged, with a production-only path.

    Both builders are the campaign's own: ``gpu_environment`` pins the CUDA
    lane and the persistent compilation cache (which is what makes the ten
    bundle solves share one compile), and ``production_child_environment`` is
    the F3-owned adapter that makes an instrument-shaped environment safe for
    a production-tree child.  Rebuilding either here would be a twin.

    The PYTHONPATH is then narrowed: the instrument tree this harness runs
    under must not follow the harness into the child, or the child would
    solve a different program than the promotion ships.
    """
    environment = production_child_environment(
        gpu_environment(cache_dir=cache_dir, shim_dir=shim_dir)
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(shim_dir), str(PRODUCTION_ROOT), str(PRODUCTION_ROOT / "src"))
    )
    return environment


def _run_child(
    *,
    mode: str,
    output_json: Path,
    input_manifest: Path | None,
    cache_dir: Path,
    shim_dir: Path,
) -> Mapping[str, object]:
    command = [
        sys.executable,
        str(CHILD),
        "--mode",
        mode,
        "--maxiter",
        str(MAXITER),
        "--seed",
        str(SEED),
        "--output-json",
        str(output_json),
    ]
    if input_manifest is not None:
        command.extend(("--input-manifest", str(input_manifest)))
    completed = subprocess.run(
        command,
        cwd=PRODUCTION_ROOT,
        env=_child_environment(cache_dir=cache_dir, shim_dir=shim_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RobustnessError(
            f"robustness child ({mode}) failed:\n{completed.stderr[-4000:]}"
        )
    return json.loads(output_json.read_text())


def _oracle_or_refusal(
    *,
    record: Mapping[str, object],
    oracle_root: Path,
    source_manifest: Path,
) -> tuple[Mapping[str, object] | None, str | None]:
    """Cross-evaluate one endpoint, or report why the oracle would not.

    A native refusal is a gate OUTCOME, not a harness crash: a perturbed
    endpoint the native evaluator rejects is exactly the kind of result this
    campaign exists to find, and it has to reach the record intact rather
    than take the process down before the record is written.  Catching it
    here changes no algorithm and repairs nothing; it only ensures the
    failure is reported with the run that produced it.
    """
    try:
        oracle = run_oracle(
            candidate=record["candidate"],  # type: ignore[arg-type]
            anchor=(
                float(record["endpoint_inner_state"][0]),  # type: ignore[index]
                float(record["endpoint_inner_state"][1]),  # type: ignore[index]
            ),
            oracle_root=oracle_root,
            source_manifest=source_manifest,
        )
    except (RuntimeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"[:2000]
    return oracle, None


def _gate_bundle_run(
    record: Mapping[str, object],
    oracle: Mapping[str, object] | None,
    oracle_refusal: str | None,
) -> dict[str, object]:
    """Finite endpoint, objective decreased, and the native oracle agrees."""
    failures: list[str] = []
    if not bool(record["endpoint_finite"]) or not bool(record["objective_finite"]):
        failures.append("nonfinite_endpoint")
    start_objective = float(str(record["start_objective"]))
    endpoint_objective = float(str(record["endpoint_objective"]))
    if not endpoint_objective < start_objective:
        failures.append("objective_not_decreased")
    if oracle is None:
        return {
            "passed": False,
            "failures": [*failures, "oracle_refused_endpoint"],
            "oracle_refusal": oracle_refusal,
            "oracle_objective": None,
            "oracle_gradient_inf_norm": None,
            "oracle_relative_gap": None,
        }
    oracle_objective = float(str(oracle["objective_value"]))
    gap = abs(oracle_objective - endpoint_objective) / abs(oracle_objective)
    if gap > ENDPOINT_OBJECTIVE_RTOL:
        failures.append("oracle_objective_gap")
    return {
        "passed": not failures,
        "failures": failures,
        "oracle_refusal": None,
        "oracle_objective": oracle_objective,
        "oracle_gradient_inf_norm": float(str(oracle["gradient_inf_norm"])),
        "oracle_relative_gap": gap,
    }


def _gate_constructor_run(record: Mapping[str, object]) -> dict[str, object]:
    """Finite, improved, and the gradient came down.

    The oracle is bundle-scoped and cannot speak about this problem, so the
    gradient clause carries what the oracle would otherwise have carried.
    """
    failures: list[str] = []
    if not bool(record["endpoint_finite"]) or not bool(record["objective_finite"]):
        failures.append("nonfinite_endpoint")
    if not float(str(record["endpoint_objective"])) < float(
        str(record["start_objective"])
    ):
        failures.append("objective_not_decreased")
    if not float(str(record["endpoint_gradient_inf_norm"])) < float(
        str(record["start_gradient_inf_norm"])
    ):
        failures.append("gradient_not_reduced")
    return {
        "passed": not failures,
        "failures": failures,
        "oracle_applicable": False,
        "oracle_scope_note": (
            "the fair-bar oracle is wired to the archived bundle's native "
            "material and can only evaluate candidates of that problem"
        ),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_root: Path = args.run_root
    run_root.mkdir(parents=True, exist_ok=False)

    # One cache and one shim for both children: the compilation cache is what
    # lets the ten bundle solves share a compile, and reusing it across the
    # constructor child costs nothing (its shapes differ, so it compiles once
    # more regardless).
    cache_dir = run_root / "jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    shim_dir = write_provenance_shim(run_root)

    bundle_payload = _run_child(
        mode="bundle",
        output_json=run_root / "bundle_child.json",
        input_manifest=args.input_manifest,
        cache_dir=cache_dir,
        shim_dir=shim_dir,
    )
    constructor_payload = _run_child(
        mode="constructor",
        output_json=run_root / "constructor_child.json",
        input_manifest=None,
        cache_dir=cache_dir,
        shim_dir=shim_dir,
    )

    rows: list[dict[str, object]] = []
    for record in bundle_payload["runs"]:  # type: ignore[index]
        run_id = str(record["run_id"])
        oracle, refusal = _oracle_or_refusal(
            record=record,
            oracle_root=run_root / f"oracle-{run_id}",
            source_manifest=args.input_manifest,
        )
        rows.append(
            {
                "configuration": "certified-frozen-bundle",
                **{k: v for k, v in record.items() if k != "candidate"},
                "gates": _gate_bundle_run(record, oracle, refusal),
            }
        )
    for record in constructor_payload["runs"]:  # type: ignore[index]
        rows.append(
            {
                "configuration": "repository-geometry",
                **{k: v for k, v in record.items() if k != "candidate"},
                "gates": _gate_constructor_run(record),
            }
        )

    failed = [row for row in rows if not row["gates"]["passed"]]  # type: ignore[index]
    summary = {
        "schema": RECORD_SCHEMA,
        "licensed_claim": LICENSED_CLAIM,
        "timing_disclosure": (
            "no timing claim; every seconds field is incidental and non-verdict"
        ),
        "instrument_commit": INSTRUMENT_COMMIT,
        "instrument_root": str(INSTRUMENT_ROOT),
        "input_manifest": str(args.input_manifest),
        "maxiter": MAXITER,
        "seed": SEED,
        "platform": str(bundle_payload["jax_platform"]),
        "device": str(bundle_payload["jax_device"]),
        "bundle_process_note": (
            "all ten bundle-problem starts ran in one process reusing one "
            "compiled program; the starts are shape-identical and no timing "
            "claim depends on process isolation"
        ),
        "runs": rows,
        "all_gates_passed": not failed,
        "failed_runs": [row["run_id"] for row in failed],
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"all_gates_passed": not failed, "runs": len(rows)}))
    if failed:
        raise RobustnessError(
            "C3 gates failed for: "
            + ", ".join(
                f"{row['run_id']}({','.join(row['gates']['failures'])})"  # type: ignore[index]
                for row in failed
            )
            + ". Recorded, not repaired: guard design is a reviewed decision."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
