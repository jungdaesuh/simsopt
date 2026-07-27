"""Shared command and result envelope for exact-name JAX example mirrors."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Literal, Mapping

import jax

from simsopt_jax.backend.runtime import get_backend_mode, get_resolved_precision


@dataclass(frozen=True)
class MirrorResult:
    """Immutable result published by one exact-name example mirror."""

    example_id: str
    observables: Mapping[str, object]
    status: Literal["ok", "failed"]

    def json_object(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "backend_mode": get_backend_mode(),
            "platform": jax.devices()[0].platform,
            "precision": get_resolved_precision(),
            "status": self.status,
            "observables": dict(self.observables),
        }


MirrorSolve = Callable[[Path, int], MirrorResult]


def run_mirror(
    arguments: list[str] | None,
    *,
    description: str | None,
    temporary_prefix: str,
    bounded_steps: int,
    native_default_steps: int,
    solve: MirrorSolve,
) -> int:
    """Parse the common mirror CLI, execute one solve, and publish JSON."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args(arguments)
    max_steps = options.max_steps or (
        bounded_steps if options.smoke else native_default_steps
    )
    if options.output_dir is not None:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        result = solve(options.output_dir, max_steps)
    elif options.smoke:
        with TemporaryDirectory(prefix=temporary_prefix) as temporary:
            result = solve(Path(temporary), max_steps)
    else:
        result = solve(Path.cwd(), max_steps)
    if options.json:
        print(json.dumps(result.json_object(), sort_keys=True))
    else:
        print(f"example={result.example_id}")
        print(f"status={result.status}")
        for name, value in result.observables.items():
            print(f"{name}={value}")
    return 0 if result.status == "ok" else 1
