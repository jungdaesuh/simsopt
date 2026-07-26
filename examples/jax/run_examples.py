"""Run manifest-selected JAX examples in isolated, fail-closed subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, TextIO

if __package__:
    from ._manifest import (
        JaxExampleRecord,
        JaxExamplesManifest,
        ManifestValidationError,
        load_manifest,
    )
else:
    from _manifest import (  # type: ignore[no-redef]
        JaxExampleRecord,
        JaxExamplesManifest,
        ManifestValidationError,
        load_manifest,
    )


Lane = Literal["cpu-smoke", "gpu-strict"]

_LANE_ENVIRONMENT: dict[Lane, dict[str, str]] = {
    "cpu-smoke": {
        "SIMSOPT_BACKEND_MODE": "jax_cpu_parity",
        "SIMSOPT_BACKEND_STRICT": "1",
        "SIMSOPT_JAX_TRANSFER_GUARD": "log",
        "SIMSOPT_PRECISION": "fp64",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        "CUDA_VISIBLE_DEVICES": "",
    },
    "gpu-strict": {
        "SIMSOPT_BACKEND_MODE": "jax_gpu_parity",
        "SIMSOPT_BACKEND_STRICT": "1",
        "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
        "SIMSOPT_PRECISION": "fp64",
        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true",
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_X64": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    },
}


class ChildResultValidationError(ValueError):
    """A child result cannot certify the runtime lane that selected it."""


@dataclass(frozen=True)
class ChildResult:
    example_id: str
    backend_mode: str
    platform: str
    precision: str
    status: str
    observables: dict[str, object]


def build_child_command(
    example: JaxExampleRecord, *, repo_root: Path
) -> tuple[str, ...]:
    """Return the only supported bounded child command for one example."""

    return (
        sys.executable,
        str(repo_root / "examples" / "jax" / example.path),
        "--smoke",
        "--json",
        *example.smoke_args,
    )


def build_lane_environment(
    lane: Lane,
    base_environment: Mapping[str, str],
    *,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Overlay one typed lane selection before the child can import JAX."""

    environment = dict(base_environment)
    environment.update(_LANE_ENVIRONMENT[lane])
    if repo_root is not None:
        source_root = str(repo_root / "src")
        inherited_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_root
            if not inherited_pythonpath
            else os.pathsep.join((source_root, inherited_pythonpath))
        )
    return environment


def _required_result_string(
    result: dict[str, object], field: str, example_id: str
) -> str:
    value = result.get(field)
    if not isinstance(value, str) or not value:
        raise ChildResultValidationError(
            f"{example_id}: result field {field} must be a non-empty string"
        )
    return value


def _parse_child_result(stdout: str, example_id: str) -> ChildResult:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ChildResultValidationError(f"{example_id}: child emitted no result")
    try:
        raw_result: object = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise ChildResultValidationError(
            f"{example_id}: final stdout line is not valid JSON"
        ) from error
    if not isinstance(raw_result, dict) or not all(
        isinstance(key, str) for key in raw_result
    ):
        raise ChildResultValidationError(
            f"{example_id}: final stdout line must be a JSON object"
        )
    result = {key: value for key, value in raw_result.items() if isinstance(key, str)}
    observables_value = result.get("observables")
    if not isinstance(observables_value, dict) or not all(
        isinstance(key, str) for key in observables_value
    ):
        raise ChildResultValidationError(
            f"{example_id}: result field observables must be a JSON object"
        )
    observables = {
        key: value for key, value in observables_value.items() if isinstance(key, str)
    }
    return ChildResult(
        example_id=_required_result_string(result, "example_id", example_id),
        backend_mode=_required_result_string(result, "backend_mode", example_id),
        platform=_required_result_string(result, "platform", example_id),
        precision=_required_result_string(result, "precision", example_id),
        status=_required_result_string(result, "status", example_id),
        observables=observables,
    )


def _validate_child_result(
    result: ChildResult, example: JaxExampleRecord, lane: Lane
) -> None:
    expected_runtime = {
        "cpu-smoke": ("jax_cpu_parity", "cpu", "fp64"),
        "gpu-strict": ("jax_gpu_parity", "gpu", "fp64"),
    }[lane]
    expected_backend, expected_platform, expected_precision = expected_runtime
    if result.example_id != example.id:
        raise ChildResultValidationError(
            f"{example.id}: example_id must be {example.id}, got {result.example_id}"
        )
    if result.status != "ok":
        raise ChildResultValidationError(
            f"{example.id}: status must be ok, got {result.status}"
        )
    if result.backend_mode != expected_backend:
        raise ChildResultValidationError(
            f"{example.id}: backend_mode must be {expected_backend}, "
            f"got {result.backend_mode}"
        )
    if result.platform != expected_platform:
        raise ChildResultValidationError(
            f"{example.id}: platform must be {expected_platform}, got {result.platform}"
        )
    if result.precision != expected_precision:
        raise ChildResultValidationError(
            f"{example.id}: precision must be {expected_precision}, "
            f"got {result.precision}"
        )


def _write_child_failure(
    *,
    example: JaxExampleRecord,
    command: tuple[str, ...],
    child_stdout: str,
    child_stderr: str,
    reason: str,
    stderr: TextIO,
) -> None:
    print(f"FAIL {example.id}: {reason}", file=stderr)
    print(f"inspired_by: {', '.join(example.inspired_by)}", file=stderr)
    print(f"command: {shlex.join(command)}", file=stderr)
    print("stdout:", file=stderr)
    print(child_stdout, file=stderr, end="" if child_stdout.endswith("\n") else "\n")
    print("stderr:", file=stderr)
    print(child_stderr, file=stderr, end="" if child_stderr.endswith("\n") else "\n")


def run_lane(
    manifest: JaxExamplesManifest,
    lane: Lane,
    *,
    repo_root: Path,
    base_environment: Mapping[str, str],
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Run every ready record in manifest order and aggregate lane failures."""

    selected = tuple(
        example
        for example in manifest.jax_examples
        if example.status == "ready" and lane in example.lanes
    )
    if not selected:
        print(f"FAIL {lane}: no ready examples selected", file=stderr)
        return 1

    environment = build_lane_environment(lane, base_environment, repo_root=repo_root)
    failed = False
    for example in selected:
        command = build_child_command(example, repo_root=repo_root)
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            _write_child_failure(
                example=example,
                command=command,
                child_stdout=completed.stdout,
                child_stderr=completed.stderr,
                reason=f"child exited {completed.returncode}",
                stderr=stderr,
            )
            failed = True
            continue
        try:
            child_result = _parse_child_result(completed.stdout, example.id)
            _validate_child_result(child_result, example, lane)
        except ChildResultValidationError as error:
            _write_child_failure(
                example=example,
                command=command,
                child_stdout=completed.stdout,
                child_stderr=completed.stderr,
                reason=str(error),
                stderr=stderr,
            )
            failed = True
            continue
        print(f"PASS {example.id}", file=stdout)
    return 1 if failed else 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=tuple(_LANE_ENVIRONMENT), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("manifest.json"),
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    parsed = _argument_parser().parse_args(arguments)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        manifest = load_manifest(parsed.manifest, repo_root=repo_root)
    except ManifestValidationError as error:
        print(f"manifest validation failed: {error}", file=sys.stderr)
        return 2
    lane: Lane = parsed.lane
    return run_lane(
        manifest,
        lane,
        repo_root=repo_root,
        base_environment=os.environ,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
