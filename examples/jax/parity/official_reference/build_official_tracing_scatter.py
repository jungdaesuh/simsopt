"""Generate the tracked TRACING scatter records from the tracing-sampling investigation.

The three tracing mirrors have no single optimizer end value, so their record is not the ``sensitivity/`` one: it is
upstream's own trajectory scatter under the pre-registered one-ulp perturbation of the START DATA of the traced
objects (the ``R0``/``Z0`` arrays of ``compute_fieldlines``, the ``xyz_inits`` of ``trace_particles``).

Like every other input of this package, the sampling directory is a command-line argument: it lives outside the
repository tree that ships the fixture.  The generator refuses to write a record whose ``k = 0`` run is not the
canonical official run: every canonical observable the fixture holds element-exact is re-hashed from the ``k = 0``
capture and must equal the canonical digest.

Regenerate the shipped set with::

    PYTHONDONTWRITEBYTECODE=1 bash .artifacts/official-scope-cleanup-20260919/run-python.sh \\
        examples/jax/parity/official_reference/build_official_tracing_scatter.py \\
        --tracing-sampling .artifacts/official-mirror-closure-20260919/investigations/tracing-sampling \\
        --reference-root examples/jax/parity/official_reference/9e027eac3 \\
        --upstream-commit 9e027eac38028d57aa23777be52a781aa860e347
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from examples.jax.parity.official_reference import (
    SCHEMA_VERSION,
    TRACING_ROOT,
    TRACING_SCATTER_CASE_QUANTITIES,
    TRACING_SCATTER_QUANTITIES,
    ArrayDigest,
)
from examples.jax.parity.official_reference.build_official_reference import (
    CAPTURE_ARRAYS_NAME,
    CAPTURE_JSON_NAME,
    SENSITIVITY_PERTURBATION_NAME as PERTURBATION_NAME,
    SENSITIVITY_RUN_NAME as RUN_NAME,
    is_numeric_array,
    render_case_file,
    sha256_file,
)

#: Sub-directory of the reference root holding the tracing records.
TRACING_DIRECTORY_NAME = TRACING_ROOT.name

#: Machine-readable output of the tracing-sampling investigation.
TRACING_SUMMARY_NAME = "summary.json"

#: The protocol was pre-registered before any tracing sample existed; this text is copied into every record.
TRACING_PROTOCOL_TEXT = (
    "Nine runs of the official tracing script bytes on the official build "
    "(/data/code/columbia/simsopt-official-reference-9e027eac3/venv), one thread, the capture lane's environment. "
    "k = 0 is the unperturbed rerun and reproduces the canonical capture bitwise. For k = 1..8 only the START DATA "
    "of the traced objects is moved by exactly one unit in the last place per entry; everything else is verbatim. "
    "Recorded per run: terminal status, final time and final state of every line, the hit count per plane and all "
    "hits. These are upstream's own numbers: the record carries no ceiling, band or other derived bound."
)
TRACING_PERTURBATION_RULE = "start' = np.nextafter(start, s * np.inf)"
TRACING_SEED_RULE = (
    "s = np.random.RandomState(20260920 + k).choice([-1.0, 1.0], size=start.size)"
)
TRACING_UNPERTURBED_K = 0
TRACING_PRE_REGISTRATION = (
    ".artifacts/official-mirror-closure-20260919/d1-diagnostic/NOTES.md "
    "(tracing sampling protocol, pre-registered 2026-09-20 05:23 EDT)"
)


def run_directory(sampling_root: Path, case_id: str, k: int) -> Path:
    """Directory of one sampled run."""
    return sampling_root / "runs" / case_id / f"k{k}"


def capture_observable_sha256(capture_directory: Path) -> dict[str, str]:
    """sha256 of every numeric-array observable of one captured run, by the fixture's byte convention."""
    capture = json.loads(
        (capture_directory / CAPTURE_JSON_NAME).read_text(encoding="utf-8")
    )
    digests = {
        str(key): ArrayDigest.of(np.asarray(value)).sha256
        for key, value in capture["observables"].items()
        if is_numeric_array(value)
    }
    if "array_file" in capture:
        with np.load(capture_directory / CAPTURE_ARRAYS_NAME) as stored:
            for name in stored.files:
                digests[str(name)] = ArrayDigest.of(stored[name]).sha256
    return digests


def unperturbed_record(
    case_id: str,
    sampling_root: Path,
    case_summary: Mapping[str, object],
    canonical: Mapping[str, object],
) -> dict[str, object]:
    """The absolute k = 0 facts, cross-checked against the canonical fixture record element by element."""
    directory = run_directory(sampling_root, case_id, TRACING_UNPERTURBED_K)
    measured = capture_observable_sha256(directory)
    observables = dict(canonical["observables"])
    inline = {
        str(key): str(entry["sha256"])
        for key, entry in observables.items()
        if entry["kind"] == "array" and "values" in entry
    }
    for key, expected in sorted(inline.items()):
        if key not in measured:
            raise SystemExit(
                f"{case_id} k={TRACING_UNPERTURBED_K}: the canonical record holds {key!r} element-exact but the "
                f"unperturbed run did not record it"
            )
        if measured[key] != expected:
            raise SystemExit(
                f"{case_id} k={TRACING_UNPERTURBED_K}: {key!r} hashes to {measured[key]} but the canonical record "
                f"holds {expected}; the unperturbed run is not the canonical official run"
            )
    return {
        "final_status_histogram": {
            str(status): int(count)
            for status, count in dict(case_summary["k0_final_status_histogram"]).items()
        },
        "poincare_hits_total": int(case_summary["k0_poincare_hits_total"]),
        "observable_sha256": {key: measured[key] for key in sorted(inline)},
    }


def perturbation_entry(directory: Path) -> dict[str, object]:
    """What the sampling moved in one run, read from the run's own perturbation record."""
    payload = json.loads((directory / PERTURBATION_NAME).read_text(encoding="utf-8"))
    return {
        "seed": payload["seed"],
        "perturbed_arguments": payload["perturbed_arguments"],
        "entries": payload["entries"],
        "entries_changed": payload["entries_changed"],
        "max_abs_delta": payload["max_abs_delta"],
        "max_rel_delta": payload["max_rel_delta"],
    }


def case_quantities(per_k: Mapping[str, object]) -> tuple[str, ...]:
    """The quantity names one sampled run recorded: the universal ones, then the per-case ones it measured.

    A case records a member of :data:`TRACING_SCATTER_CASE_QUANTITIES` only when the object it traces has that
    quantity (a field line has no parallel speed), so the sampling itself decides, and no table here does.
    """
    return TRACING_SCATTER_QUANTITIES + tuple(
        name for name in TRACING_SCATTER_CASE_QUANTITIES if name in per_k
    )


def scatter_entry(per_k: Mapping[str, object]) -> dict[str, object]:
    """The report's quantities (a)-(e) of one perturbed run, plus (f) where the case measured it."""
    return {name: per_k[name] for name in case_quantities(per_k)}


def run_entry(
    case_id: str,
    sampling_root: Path,
    k: int,
    per_k: Mapping[str, object] | None,
) -> dict[str, object]:
    """One run of the record: provenance, what was perturbed, and the scatter (``None`` for the unperturbed run)."""
    directory = run_directory(sampling_root, case_id, k)
    run_record = json.loads((directory / RUN_NAME).read_text(encoding="utf-8"))
    if run_record["exit_code"] != 0:
        raise SystemExit(
            f"{case_id} k={k}: the official run exited {run_record['exit_code']}; a failed run is not evidence"
        )
    return {
        "k": k,
        "capture_sha256": sha256_file(directory / CAPTURE_JSON_NAME),
        "perturbation_sha256": sha256_file(directory / PERTURBATION_NAME),
        "perturbation": perturbation_entry(directory),
        "scatter": None if per_k is None else scatter_entry(per_k),
    }


def maxima_over_k(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Element-wise maximum of the stored scatter over the perturbed runs.

    The quantity list is the runs' own, and every perturbed run must carry the SAME one: a quantity measured in
    some runs of a case but not in others would give a maximum over a subset of the samples, which is not the
    maximum over ``k = 1..8`` the contract derives its ceiling from.
    """
    scatters = [dict(run["scatter"]) for run in runs if run["scatter"] is not None]
    if not scatters:
        raise SystemExit("a tracing record needs at least one perturbed run")
    names = tuple(scatters[0])
    for scatter in scatters:
        if tuple(scatter) != names:
            raise SystemExit(
                f"the perturbed runs recorded different scatter quantities: {names} and {tuple(scatter)}"
            )
    return {name: max(scatter[name] for scatter in scatters) for name in names}


def build_tracing_payload(
    case_id: str,
    sampling_root: Path,
    case_summary: Mapping[str, object],
    canonical: Mapping[str, object],
    upstream_commit: str,
) -> dict[str, object]:
    """Assemble one tracing scatter record from the per-run artefacts, cross-checked against the sampling summary."""
    per_k = dict(case_summary["per_k"])
    runs = [
        run_entry(case_id, sampling_root, TRACING_UNPERTURBED_K, None),
        *(
            run_entry(case_id, sampling_root, int(k), per_k[k])
            for k in sorted(per_k, key=int)
        ),
    ]
    measured = maxima_over_k(runs)
    recorded = {name: dict(case_summary["maxima_over_k"])[name] for name in measured}
    if measured != recorded:
        raise SystemExit(
            f"{case_id}: the maxima computed from the per-run scatter are {measured} but the sampling summary "
            f"records {recorded}"
        )
    directory = run_directory(sampling_root, case_id, TRACING_UNPERTURBED_K)
    capture = json.loads((directory / CAPTURE_JSON_NAME).read_text(encoding="utf-8"))
    perturbed_arguments = sorted(
        {
            str(name)
            for run in runs
            for name in dict(run["perturbation"])["perturbed_arguments"] or ()
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "upstream_commit": upstream_commit,
        "official_script": str(canonical["official_script"]),
        "official_script_sha256": str(canonical["official_script_sha256"]),
        "lines": int(case_summary["lines"]),
        "protocol": {
            "text": TRACING_PROTOCOL_TEXT,
            "perturbation_rule": TRACING_PERTURBATION_RULE,
            "seed_rule": TRACING_SEED_RULE,
            "perturbed_arguments": perturbed_arguments,
            "unperturbed_k": TRACING_UNPERTURBED_K,
            "pre_registered_in": TRACING_PRE_REGISTRATION,
            "threads": {
                str(name): str(value) for name, value in capture["threads"].items()
            },
        },
        "unperturbed": unperturbed_record(
            case_id, sampling_root, case_summary, canonical
        ),
        "runs": runs,
        "maxima_over_k": measured,
    }


def build_tracing_payloads(
    sampling_root: Path,
    reference_root: Path,
    upstream_commit: str,
) -> dict[str, dict[str, object]]:
    """One record per case the sampling directory covers; empty when that directory has no summary."""
    summary_path = sampling_root / TRACING_SUMMARY_NAME
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payloads = {}
    for case_summary in sorted(summary["cases"], key=lambda entry: entry["case_id"]):
        case_id = str(case_summary["case_id"])
        canonical_path = reference_root / f"{case_id}.json"
        if not canonical_path.is_file():
            raise SystemExit(
                f"{case_id}: sampled for tracing scatter but it has no canonical record at {canonical_path}"
            )
        payloads[case_id] = build_tracing_payload(
            case_id=case_id,
            sampling_root=sampling_root,
            case_summary=case_summary,
            canonical=json.loads(canonical_path.read_text(encoding="utf-8")),
            upstream_commit=upstream_commit,
        )
    return payloads


def write_official_tracing_scatter(
    *,
    tracing_sampling: Path,
    reference_root: Path,
    upstream_commit: str,
) -> tuple[Path, ...]:
    """Write one JSON file per sampled tracing case under ``<reference-root>/tracing``; return the paths, sorted."""
    payloads = build_tracing_payloads(tracing_sampling, reference_root, upstream_commit)
    if not payloads:
        return ()
    output = reference_root / TRACING_DIRECTORY_NAME
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for case_id, payload in payloads.items():
        path = output / f"{case_id}.json"
        path.write_text(render_case_file(payload), encoding="utf-8")
        written.append(path)
    return tuple(sorted(written))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tracing-sampling",
        type=Path,
        required=True,
        help=(
            "directory of the tracing-sampling investigation whose per-run artefacts become the records "
            f"(it must hold {TRACING_SUMMARY_NAME} and runs/<case_id>/k<k>/)"
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help=(
            "the shipped fixture directory holding the canonical <case_id>.json files; the records are written to "
            f"its {TRACING_DIRECTORY_NAME}/ sub-directory"
        ),
    )
    parser.add_argument(
        "--upstream-commit", required=True, help="full 40-hex upstream commit"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    written = write_official_tracing_scatter(
        tracing_sampling=args.tracing_sampling,
        reference_root=args.reference_root,
        upstream_commit=args.upstream_commit,
    )
    print(
        f"wrote {len(written)} files to {args.reference_root / TRACING_DIRECTORY_NAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
