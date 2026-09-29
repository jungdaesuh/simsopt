"""Generate one tracked upstream scatter record from the per-run artefacts of an upstream sampling campaign.

A record holds upstream's OWN end states at one of the parity harness's scales: the official script (verbatim at
``native_default``, or DERIVED at a reduced scale -- the official body with only its scale lines changed, the diff
stored in the record) run on the official build at one thread from the one-ulp start protocol. It holds raw numbers
only; the contracts derived from it live in ``examples/jax/parity/official_scatter_contracts.py``.

Every input is a command-line argument, because the campaign and the official clone live outside the repository.
Example (native-boozer, bounded)::

    python examples/jax/parity/official_reference/build_upstream_scatter.py \\
        --case-id native-boozer --scale bounded \\
        --runs-root <campaign>/upstream/runs \\
        --runner-receipt <campaign>/upstream/generated/examples/2_Intermediate/band_boozer.bounded.receipt.json \\
        --key area:iota=area:iota --key flux:surface_dofs=flux:surface_dofs ... \\
        --success-key area:solver_success --success-key flux:solver_success \\
        --first-k 0 --last-k 8 --pre-registered-in "<plan path> (C3)" \\
        --output examples/jax/parity/official_reference/9e027eac3
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from examples.jax.parity.official_reference import (
    UPSTREAM_COMMIT,
    UPSTREAM_SCATTER_ROOT,
    load_official_reference,
)
from examples.jax.parity.official_reference.build_official_reference import (
    CAPTURE_ARRAYS_NAME,
    CAPTURE_JSON_NAME,
    SENSITIVITY_PERTURBATION_NAME,
    SENSITIVITY_PERTURBATION_RULE,
    SENSITIVITY_PERTURBED_CALL_INDEX,
    SENSITIVITY_RUN_NAME,
    SENSITIVITY_SEED_RULE,
    SENSITIVITY_UNPERTURBED_K,
    array_entry,
    observable_entry,
    provider_outcome_entry,
    render_case_file,
    sha256_file,
)

PROTOCOL_TEXT = (
    "Runs of upstream's official script on the official build (9e027eac3 Python and its compiled extension), one "
    "thread, at the parity harness's scale named by this record: verbatim at native_default, and at a reduced scale "
    "the official body with ONLY its scale lines changed (runner.body_diff). k = 0 is the unperturbed run; for "
    "k >= 1 only the FIRST provider call's start vector is moved by exactly one unit in the last place per entry. "
    "These are upstream's own numbers: the record carries no band, set or other derived bound."
)


def _key_pairs(arguments: Sequence[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for argument in arguments:
        lane_key, separator, capture_key = argument.partition("=")
        if not separator or not lane_key or not capture_key or lane_key in pairs:
            raise SystemExit(
                f"--key must be a unique LANE_KEY=CAPTURE_KEY pair, got {argument!r}"
            )
        pairs[lane_key] = capture_key
    return pairs


def build_run(
    directory: Path,
    k: int,
    capture_keys: dict[str, str],
    success_keys: Sequence[str],
    expected_threads: dict[str, str] | None,
) -> tuple[dict[str, object], dict[str, str]]:
    """One run's record; refuses a run that failed, ran unpinned, or lacks a named observable."""
    run_record = json.loads(
        (directory / SENSITIVITY_RUN_NAME).read_text(encoding="utf-8")
    )
    if run_record["exit_code"] != 0 or int(run_record["k"]) != k:
        raise SystemExit(
            f"{directory}: exit {run_record['exit_code']}, k {run_record['k']}; not evidence"
        )
    capture_path = directory / CAPTURE_JSON_NAME
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    threads = {str(name): str(value) for name, value in capture["threads"].items()}
    if set(threads.values()) != {"1"} or (
        expected_threads is not None and threads != expected_threads
    ):
        raise SystemExit(
            f"{directory}: thread environment {threads} is not the one-thread protocol"
        )
    scalars = capture["observables"]
    arrays = {}
    if "array_file" in capture:
        with np.load(directory / CAPTURE_ARRAYS_NAME) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
    values: dict[str, object] = {}
    for lane_key, capture_key in capture_keys.items():
        if capture_key in scalars:
            values[lane_key] = observable_entry(scalars[capture_key])
        elif capture_key in arrays:
            values[lane_key] = array_entry(arrays[capture_key])
        else:
            raise SystemExit(f"{directory}: capture has no observable {capture_key!r}")
    return (
        {
            "k": k,
            "workflow_success": all(scalars[key] is True for key in success_keys),
            "provider_calls": [
                provider_outcome_entry(index, call)
                for index, call in enumerate(capture["optimizer_calls"])
            ],
            "capture_sha256": sha256_file(capture_path),
            "perturbation_sha256": sha256_file(
                directory / SENSITIVITY_PERTURBATION_NAME
            ),
            "values": values,
        },
        threads,
    )


def build_payload(
    case_id: str,
    scale: str,
    runs_root: Path,
    runner_receipt: Path,
    capture_keys: dict[str, str],
    success_keys: Sequence[str],
    ks: Sequence[int],
    pre_registered_in: str,
) -> dict[str, object]:
    """The whole record of one case at one scale."""
    receipt = json.loads(runner_receipt.read_text(encoding="utf-8"))
    official = load_official_reference(case_id)
    if receipt["case_id"] != case_id or receipt["scale"] != scale:
        raise SystemExit(
            f"{runner_receipt} is the runner of {receipt['case_id']} at {receipt['scale']}"
        )
    if receipt["official_script_sha256"] != official.official_script_sha256:
        raise SystemExit(
            f"{runner_receipt}: official script sha256 differs from the canonical fixture"
        )
    verbatim = bool(receipt["verbatim_body_verified"])
    if verbatim == bool(receipt["derived_body_equals_official_with_listed_edits"]):
        raise SystemExit(
            f"{runner_receipt}: a runner is either verbatim or derived with listed edits"
        )
    runs = []
    threads: dict[str, str] | None = None
    for k in ks:
        run, threads = build_run(
            runs_root / case_id / scale / f"k{k}",
            k,
            capture_keys,
            success_keys,
            threads,
        )
        runs.append(run)
    return {
        "schema_version": 1,
        "case_id": case_id,
        "scale": scale,
        "upstream_commit": UPSTREAM_COMMIT,
        "official_script": official.official_script,
        "official_script_sha256": official.official_script_sha256,
        "runner": {
            "kind": "verbatim" if verbatim else "derived",
            "body_sha256": receipt["body_sha256"],
            "body_diff": "" if verbatim else receipt["body_unified_diff_vs_official"],
        },
        "protocol": {
            "text": PROTOCOL_TEXT,
            "perturbation_rule": SENSITIVITY_PERTURBATION_RULE,
            "seed_rule": SENSITIVITY_SEED_RULE,
            "perturbed_call_index": SENSITIVITY_PERTURBED_CALL_INDEX,
            "unperturbed_k": SENSITIVITY_UNPERTURBED_K,
            "pre_registered_in": pre_registered_in,
            "threads": threads or {},
        },
        "capture_keys": capture_keys,
        "runs": runs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--scale", required=True, choices=("bounded", "native_default"))
    parser.add_argument(
        "--runs-root",
        required=True,
        type=Path,
        help="<campaign>/runs/<case_id>/<scale>/k<k>/",
    )
    parser.add_argument("--runner-receipt", required=True, type=Path)
    parser.add_argument(
        "--key", action="append", required=True, help="LANE_KEY=CAPTURE_KEY; repeat"
    )
    parser.add_argument(
        "--success-key",
        action="append",
        default=[],
        help="capture key that must be true",
    )
    parser.add_argument("--first-k", type=int, default=0)
    parser.add_argument("--last-k", type=int, default=8)
    parser.add_argument("--pre-registered-in", required=True)
    parser.add_argument(
        "--output", required=True, type=Path, help="the fixture directory (9e027eac3/)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    payload = build_payload(
        case_id=arguments.case_id,
        scale=arguments.scale,
        runs_root=arguments.runs_root,
        runner_receipt=arguments.runner_receipt,
        capture_keys=_key_pairs(arguments.key),
        success_keys=tuple(arguments.success_key),
        ks=tuple(range(arguments.first_k, arguments.last_k + 1)),
        pre_registered_in=arguments.pre_registered_in,
    )
    directory = arguments.output / UPSTREAM_SCATTER_ROOT.name
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{arguments.case_id}.{arguments.scale}.json"
    target.write_text(render_case_file(payload), encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
