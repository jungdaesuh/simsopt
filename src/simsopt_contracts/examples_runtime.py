"""Dependency-neutral CLI and JSON contract for executable examples."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, Literal

ExecutionScale = Literal["bounded", "native_default"]
EXECUTION_SCALES: Final[tuple[ExecutionScale, ExecutionScale]] = (
    "bounded",
    "native_default",
)


@dataclass(frozen=True)
class ExampleResult:
    """Immutable scientific result published by an executable example."""

    example_id: str
    observables: Mapping[str, object]
    status: Literal["ok", "failed"]


ExampleSolve = Callable[[Path, int, ExecutionScale], ExampleResult]
ExampleRuntimeMetadata = Callable[[ExecutionScale], Mapping[str, str]]


def serialize_example_result(
    result: ExampleResult,
    scale: ExecutionScale,
    runtime_metadata: Mapping[str, str],
) -> dict[str, object]:
    """Materialize the common output envelope with caller-owned runtime facts."""

    return {
        "example_id": result.example_id,
        **runtime_metadata,
        "scale": scale,
        "status": result.status,
        "observables": dict(result.observables),
    }


def run_example(
    arguments: list[str] | None,
    *,
    description: str | None,
    temporary_prefix: str,
    bounded_steps: int,
    native_default_steps: int,
    solve: ExampleSolve,
    runtime_metadata: ExampleRuntimeMetadata,
    result_filename: str | None = None,
) -> int:
    """Validate the CLI and runtime before solving, then publish the result."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args(arguments)
    if options.max_steps is not None and options.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    scale: ExecutionScale = "bounded" if options.smoke else "native_default"
    max_steps = (
        options.max_steps
        if options.max_steps is not None
        else (bounded_steps if options.smoke else native_default_steps)
    )
    runtime_facts = runtime_metadata(scale)
    if options.output_dir is not None:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        output_directory = options.output_dir
        result = solve(output_directory, max_steps, scale)
    elif options.smoke:
        with TemporaryDirectory(prefix=temporary_prefix) as temporary:
            output_directory = Path(temporary)
            result = solve(output_directory, max_steps, scale)
    else:
        output_directory = Path.cwd()
        result = solve(output_directory, max_steps, scale)
    serialized = json.dumps(
        serialize_example_result(result, scale, runtime_facts),
        sort_keys=True,
    )
    if result_filename is not None and options.output_dir is not None:
        (output_directory / result_filename).write_text(serialized + "\n")
    if options.json:
        print(serialized)
    else:
        print(f"example={result.example_id}")
        print(f"status={result.status}")
        for name, value in result.observables.items():
            print(f"{name}={value}")
    return 0 if result.status == "ok" else 1
