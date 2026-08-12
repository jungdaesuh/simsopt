"""Produce or validate the sealed NEQ-GNTR1 native reference on CPU."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import jax
import jaxlib
import numpy as np
import simsopt
import simsopt_jax
import simsopt_jax_adapters.geo.single_stage_native_endpoint as native_endpoint
import simsoptpp
from examples.jax.parity.input_bundle import read_input_bundle
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    build_single_stage_fullspace_bootstrap,
)
from simsopt_jax_adapters.geo.single_stage_native_endpoint import (
    NativeSingleStageEndpointRuntime,
    build_native_single_stage_endpoint_runtime,
)

from benchmarks.single_stage_native_equivalent_reference import (
    REQUIRED_SOURCE_LOGICAL_PATHS,
    HistoricalAuthorityPaths,
    NativeReferenceRuntime,
    ReferenceValidationResult,
    RuntimeProvenance,
    SourcePath,
    produce_native_equivalent_reference,
    validate_native_equivalent_reference,
)

DEFAULT_CAMPAIGN_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/"
    ".single-stage-speed-20260804.partial-20260805T052535Z-2add24ec"
)
DEFAULT_RECEIPT: Final = DEFAULT_CAMPAIGN_ROOT / (
    "receipts/000-cold-native_cpu-sample-none-position-0/lane_result.json"
)
DEFAULT_TRAJECTORY: Final = DEFAULT_CAMPAIGN_ROOT / (
    "receipts/000-cold-native_cpu-sample-none-position-0/trajectory.jsonl"
)
DEFAULT_INPUT_BUNDLE: Final = DEFAULT_CAMPAIGN_ROOT / "inputs/input_bundle.json"

_ENVIRONMENT_KEYS: Final = (
    "JAX_ENABLE_X64",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "LD_LIBRARY_PATH",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PATH",
    "PYTHONPATH",
    "XLA_FLAGS",
)


@dataclass(frozen=True, slots=True)
class ReferenceRunRequest:
    output_root: Path
    receipt: Path
    trajectory: Path
    input_bundle: Path
    repo_root: Path


class NativeEndpointAdapter(Protocol):
    """Stable construction seam used by production and typed test fakes."""

    def build_runtime(
        self,
        input_root: Path,
    ) -> NativeReferenceRuntime: ...


@dataclass(frozen=True, slots=True)
class ProductionNativeEndpointAdapter:
    """Production binding to the stable native endpoint adapter API."""

    def build_runtime(
        self,
        input_root: Path,
    ) -> NativeSingleStageEndpointRuntime:
        bundle, arrays = read_input_bundle(input_root)
        bootstrap = build_single_stage_fullspace_bootstrap()
        return build_native_single_stage_endpoint_runtime(bundle, arrays, bootstrap)


def _module_path(module: object, name: str) -> str:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name} module has no concrete source path")
    return str(Path(value).resolve(strict=True))


def _git_output(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout


def collect_runtime_provenance(repo_root: Path) -> RuntimeProvenance:
    """Capture current source-control, interpreter, and imported-runtime identity."""

    git_head = _git_output(repo_root, "rev-parse", "HEAD").decode().strip()
    tracked_diff = _git_output(repo_root, "diff", "--binary", "HEAD")
    status = _git_output(repo_root, "status", "--porcelain=v1", "-z")
    python_executable = Path(sys.executable).resolve(strict=True)
    native_extension = Path(_module_path(simsoptpp, "simsoptpp"))
    simsopt_path = Path(_module_path(simsopt, "simsopt"))
    simsopt_jax_path = Path(_module_path(simsopt_jax, "simsopt_jax"))
    adapter_path = Path(_module_path(native_endpoint, "native endpoint adapter"))
    environment = "\n".join(
        f"{key}={os.environ.get(key, '')}" for key in _ENVIRONMENT_KEYS
    ).encode()
    return RuntimeProvenance(
        argv=tuple(sys.argv),
        cwd=str(Path.cwd().resolve(strict=True)),
        python_executable=str(python_executable),
        python_version=platform.python_version(),
        platform=platform.platform(),
        numpy_version=np.__version__,
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        simsopt_path=str(simsopt_path),
        simsopt_jax_path=str(simsopt_jax_path),
        adapter_path=str(adapter_path),
        simsopt_sha256=hashlib.sha256(simsopt_path.read_bytes()).hexdigest(),
        simsopt_jax_sha256=hashlib.sha256(simsopt_jax_path.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        native_extension_path=str(native_extension),
        native_extension_sha256=hashlib.sha256(
            native_extension.read_bytes()
        ).hexdigest(),
        python_executable_sha256=hashlib.sha256(
            python_executable.read_bytes()
        ).hexdigest(),
        effective_environment_sha256=hashlib.sha256(environment).hexdigest(),
        git_head=git_head,
        tracked_diff_sha256=hashlib.sha256(tracked_diff).hexdigest(),
        repository_dirty=bool(status),
    )


def source_paths(repo_root: Path) -> tuple[SourcePath, ...]:
    """Return the complete owned/reference execution source set."""

    return tuple(
        SourcePath(logical_path=relative, path=repo_root / relative)
        for relative in REQUIRED_SOURCE_LOGICAL_PATHS
    )


def run_native_equivalent_reference(
    request: ReferenceRunRequest,
    *,
    adapter: NativeEndpointAdapter,
    runtime_provenance: RuntimeProvenance,
    execution_sources: tuple[SourcePath, ...],
) -> ReferenceValidationResult:
    """Construct the canonical bootstrap/runtime and produce one sealed artifact."""

    runtime = adapter.build_runtime(request.input_bundle.parent)
    return produce_native_equivalent_reference(
        output_root=request.output_root,
        historical_paths=HistoricalAuthorityPaths(
            receipt=request.receipt,
            trajectory=request.trajectory,
            input_bundle=request.input_bundle,
        ),
        runtime=runtime,
        runtime_provenance=runtime_provenance,
        source_paths=execution_sources,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce or validate the sealed native-equivalent reference",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--input-bundle", type=Path, default=DEFAULT_INPUT_BUNDLE)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve(strict=args.validate_only)
    if args.validate_only:
        validation = validate_native_equivalent_reference(output)
    else:
        repo_root = args.repo_root.resolve(strict=True)
        request = ReferenceRunRequest(
            output_root=output,
            receipt=args.receipt.resolve(strict=True),
            trajectory=args.trajectory.resolve(strict=True),
            input_bundle=args.input_bundle.resolve(strict=True),
            repo_root=repo_root,
        )
        validation = run_native_equivalent_reference(
            request,
            adapter=ProductionNativeEndpointAdapter(),
            runtime_provenance=collect_runtime_provenance(repo_root),
            execution_sources=source_paths(repo_root),
        )
    print(
        f"disposition={validation.disposition} "
        f"usable={str(validation.usable).lower()} "
        f"reference_sha256={validation.artifact_sha256}"
    )
    return 0 if validation.usable else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DEFAULT_INPUT_BUNDLE",
    "DEFAULT_RECEIPT",
    "DEFAULT_TRAJECTORY",
    "NativeEndpointAdapter",
    "ProductionNativeEndpointAdapter",
    "ReferenceRunRequest",
    "collect_runtime_provenance",
    "main",
    "run_native_equivalent_reference",
    "source_paths",
)
