"""Run manifest-selected JAX examples in isolated, fail-closed subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TextIO

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPO_ROOT / "src"
for import_root in (str(_SOURCE_ROOT), str(_REPO_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from simsopt_jax.config import ExecutionIntent, JaxDevice, JaxExecutionProfile
from examples.jax._lane_environment import (
    LEGACY_JAX_LANES as _LEGACY_JAX_LANES,
)
from examples.jax._lane_environment import (
    JaxLane as Lane,
)
from examples.jax._lane_environment import (
    build_execution_environment,
)
from examples.jax._manifest import JaxExampleRecord, JaxExamplesManifest
from examples.jax.manifest_runtime import (
    RuntimeContractPair,
    RuntimeExample,
    load_runtime_contract_pair,
)


class ChildResultValidationError(ValueError):
    """A child result cannot certify the runtime lane that selected it."""


def manifest_observability_payload(
    manifest: JaxExamplesManifest | RuntimeContractPair,
) -> dict[str, int | bool]:
    """Return the schema/adapter fields emitted by every runner invocation."""
    if isinstance(manifest, RuntimeContractPair):
        return {
            "examples_manifest_schema_version": manifest.version_pair[0],
            "parity_manifest_schema_version": manifest.version_pair[1],
            "used_legacy_manifest_adapter": manifest.used_legacy_adapter,
        }
    return {
        "manifest_schema_version": manifest.schema_version,
        "used_legacy_manifest_adapter": manifest.used_legacy_manifest_adapter,
    }


@dataclass(frozen=True)
class ChildResult:
    example_id: str
    backend_mode: str
    platform: str
    precision: str
    status: str
    observables: dict[str, object]


def build_child_command(
    example: JaxExampleRecord | RuntimeExample, *, repo_root: Path
) -> tuple[str, ...]:
    """Return the only supported bounded child command for one example."""

    return (
        sys.executable,
        str(repo_root / "examples" / "jax" / example.path),
        "--smoke",
        "--json",
        *example.smoke_args,
    )


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
    result: ChildResult,
    example: JaxExampleRecord | RuntimeExample,
    profile: JaxExecutionProfile,
) -> None:
    expected_backend = profile.mode
    expected_platform = profile.device
    expected_precision = "fp64"
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
    example: JaxExampleRecord | RuntimeExample,
    command: tuple[str, ...],
    child_stdout: str,
    child_stderr: str,
    reason: str,
    stderr: TextIO,
) -> None:
    print(f"FAIL {example.id}: {reason}", file=stderr)
    lineage = (
        example.source or "non-covering tutorial"
        if isinstance(example, RuntimeExample)
        else ", ".join(example.inspired_by)
    )
    print(f"native_source: {lineage}", file=stderr)
    print(f"command: {shlex.join(command)}", file=stderr)
    print("stdout:", file=stderr)
    print(child_stdout, file=stderr, end="" if child_stdout.endswith("\n") else "\n")
    print("stderr:", file=stderr)
    print(child_stderr, file=stderr, end="" if child_stderr.endswith("\n") else "\n")


def run_lane(
    manifest: JaxExamplesManifest | RuntimeContractPair,
    lane: Lane,
    *,
    repo_root: Path,
    base_environment: Mapping[str, str],
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Run one retained legacy parity lane."""
    warnings.warn(
        "--lane is deprecated; use --device with --intent parity",
        DeprecationWarning,
        stacklevel=2,
    )
    device: JaxDevice = "cpu" if lane == "cpu-smoke" else "gpu"
    return run_profile(
        manifest,
        device,
        "parity",
        repo_root=repo_root,
        base_environment=base_environment,
        stdout=stdout,
        stderr=stderr,
    )


def run_profile(
    manifest: JaxExamplesManifest | RuntimeContractPair,
    device: JaxDevice,
    intent: ExecutionIntent,
    *,
    repo_root: Path,
    base_environment: Mapping[str, str],
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Run every ready record for one explicit device and intent."""
    capability_lane: Lane = "cpu-smoke" if device == "cpu" else "gpu-strict"

    examples = (
        manifest.examples
        if isinstance(manifest, RuntimeContractPair)
        else manifest.jax_examples
    )
    selected = tuple(
        example
        for example in examples
        if example.status == "ready" and capability_lane in example.lanes
    )
    if not selected:
        print(f"FAIL {device}/{intent}: no ready examples selected", file=stderr)
        return 1

    profile, environment = build_execution_environment(
        device,
        intent,
        base_environment,
        repo_root=repo_root,
    )
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
            _validate_child_result(child_result, example, profile)
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
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--lane", choices=_LEGACY_JAX_LANES)
    selector.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--intent", choices=("fast", "parity"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("manifest.json"),
    )
    parser.add_argument(
        "--parity-manifest",
        type=Path,
        default=Path(__file__).with_name("parity_manifest.json"),
    )
    return parser


def _parse_arguments(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = _argument_parser()
    parsed = parser.parse_args(arguments)
    if parsed.lane is not None and parsed.intent is not None:
        parser.error("--lane cannot be combined with --intent; use --device")
    if parsed.intent is None:
        parsed.intent = "fast"
    return parsed


def main(arguments: list[str] | None = None) -> int:
    parsed = _parse_arguments(arguments)
    repo_root = _REPO_ROOT
    try:
        manifest = load_runtime_contract_pair(
            parsed.manifest,
            parsed.parity_manifest,
            repo_root=repo_root,
        )
    except ValueError as error:
        print(f"manifest pair validation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            manifest_observability_payload(manifest),
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    if parsed.lane is not None:
        lane: Lane = parsed.lane
        print(
            "warning: --lane is deprecated; use --device with --intent parity",
            file=sys.stderr,
        )
        return run_lane(
            manifest,
            lane,
            repo_root=repo_root,
            base_environment=os.environ,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    device: JaxDevice = parsed.device
    intent: ExecutionIntent = parsed.intent
    return run_profile(
        manifest,
        device,
        intent,
        repo_root=repo_root,
        base_environment=os.environ,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
