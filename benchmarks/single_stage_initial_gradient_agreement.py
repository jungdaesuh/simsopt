"""Record the native-default direct-LU versus parity initial-gradient check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import read_input_bundle
from simsopt_jax.parity_tolerances import parity_ladder_tolerances

from benchmarks.run_jax_native_example_measurements import (
    build_measurement_environment,
)
from benchmarks.single_stage_speed_campaign_receipt import (
    DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE,
)

_CASE_ID = "native-single-stage-boozer-vacuum-optimization"
_EXACT_ADJOINT_SELECTOR = "SIMSOPT_EXACT_ADJOINT_DENSE_LU"
_PARITY_ADJOINT_ROUTE = "parity_mode_exact_jacobian_dense_fp64_lu"
_SCHEMA_VERSION = 1

ProbeRoute = Literal["direct", "parity"]


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _write_json_exclusive(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _initial_only_optimizer(
    optimizer: Callable[..., object],
) -> Callable[..., object]:
    """Return the real host optimizer constrained to its initial evaluation."""

    def run(
        evaluator: Callable[[np.ndarray], tuple[float, np.ndarray]],
        initial_parameters: np.ndarray,
        **options: object,
    ) -> object:
        initial_value_and_grad = options.get("initial_value_and_grad")
        return optimizer(
            evaluator,
            initial_parameters,
            maxiter=0,
            initial_value_and_grad=initial_value_and_grad,
        )

    return run


def _probe_document(input_root: Path, route: ProbeRoute) -> dict[str, object]:
    from unittest.mock import patch

    import jax
    from simsopt_jax.geo import optimizer_host_lbfgs

    bundle, arrays = read_input_bundle(input_root)
    if bundle.case_id != _CASE_ID or bundle.scale != "native_default":
        raise ValueError(
            "gradient agreement requires the native-default campaign bundle"
        )
    expected_mode = "jax_cpu_fast" if route == "direct" else "jax_cpu_parity"
    actual_mode = os.environ.get("SIMSOPT_BACKEND_MODE")
    if actual_mode != expected_mode:
        raise ValueError(f"{route} probe requires backend mode {expected_mode}")
    observed_selector = os.environ.get(_EXACT_ADJOINT_SELECTOR)
    if route == "direct" and observed_selector != "1":
        raise ValueError("direct probe requires the dense-LU exact-adjoint selector")
    if route == "parity" and observed_selector is not None:
        raise ValueError("parity probe must select dense LU through parity mode")

    case = get_case(_CASE_ID)
    with patch.object(
        optimizer_host_lbfgs,
        "minimize_bfgs_host_core",
        _initial_only_optimizer(optimizer_host_lbfgs.minimize_bfgs_host_core),
    ), patch.object(
        optimizer_host_lbfgs,
        "minimize_lbfgs_host_core",
        _initial_only_optimizer(optimizer_host_lbfgs.minimize_lbfgs_host_core),
    ):
        observation = case.execute("jax-cpu", bundle, arrays)
    jax.block_until_ready(tuple(observation.values.values()))

    gradient = np.asarray(observation.values["initial:gradient"])
    parameters = np.asarray(observation.values["initial:parameters"])
    objective = np.asarray(observation.values["initial:objective"])
    if (
        gradient.dtype != np.dtype(np.float64)
        or gradient.ndim != 1
        or not gradient.size
    ):
        raise ValueError("initial gradient must be a nonempty FP64 vector")
    if parameters.dtype != np.dtype(np.float64) or parameters.shape != gradient.shape:
        raise ValueError(
            "initial parameters must be an FP64 vector matching the gradient"
        )
    if objective.dtype != np.dtype(np.float64) or objective.size != 1:
        raise ValueError("initial objective must be an FP64 scalar")
    if not bool(np.all(np.isfinite(gradient))):
        raise ValueError(f"{route} initial gradient is not finite")

    return {
        "schema_version": _SCHEMA_VERSION,
        "case_id": _CASE_ID,
        "scale": "native_default",
        "route": route,
        "backend_mode": observation.backend_mode,
        "adjoint_route": (
            DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE
            if route == "direct"
            else _PARITY_ADJOINT_ROUTE
        ),
        "optimizer_iterations_suppressed_after_initial_evaluation": True,
        "input_fingerprint": observation.input_fingerprint,
        "configuration_fingerprint": observation.configuration_fingerprint,
        "effective_construction_fingerprint": (
            observation.effective_construction_fingerprint
        ),
        "initial_parameters_sha256": _array_sha256(parameters),
        "initial_objective": float(objective.item()),
        "gradient": gradient.tolist(),
        "gradient_sha256": _array_sha256(gradient),
        "gradient_dimension": int(gradient.size),
        "gradient_l2_norm": float(np.linalg.norm(gradient)),
        "gradient_inf_norm": float(np.linalg.norm(gradient, ord=np.inf)),
        "gradient_finite": True,
    }


def compare_probe_documents(
    direct: Mapping[str, object],
    parity: Mapping[str, object],
) -> dict[str, object]:
    """Validate two raw probe records and return the complete agreement record."""
    identity_keys = (
        "input_fingerprint",
        "configuration_fingerprint",
        "effective_construction_fingerprint",
        "initial_parameters_sha256",
    )
    for key in identity_keys:
        if direct.get(key) != parity.get(key):
            raise ValueError(f"gradient probe identity mismatch for {key}")
    if direct.get("route") != "direct" or parity.get("route") != "parity":
        raise ValueError("gradient probe route labels are invalid")

    direct_gradient = np.asarray(direct.get("gradient"))
    parity_gradient = np.asarray(parity.get("gradient"))
    if (
        direct_gradient.dtype != np.dtype(np.float64)
        or parity_gradient.dtype != np.dtype(np.float64)
        or direct_gradient.ndim != 1
        or parity_gradient.shape != direct_gradient.shape
        or not direct_gradient.size
    ):
        raise ValueError("gradient probes must contain matching nonempty FP64 vectors")
    if not bool(
        np.all(np.isfinite(direct_gradient)) and np.all(np.isfinite(parity_gradient))
    ):
        raise ValueError("gradient probes must be finite")

    contract = parity_ladder_tolerances("mirror_single_stage_initial_gradient")
    rtol = contract["rtol"]
    atol = contract["atol"]
    if not isinstance(rtol, float) or not isinstance(atol, float):
        raise TypeError("initial-gradient tolerance contract must be numeric")
    difference = direct_gradient - parity_gradient
    absolute_difference = np.abs(difference)
    allowed_difference = atol + rtol * np.abs(parity_gradient)
    symmetric_scale = np.maximum(
        np.maximum(np.abs(direct_gradient), np.abs(parity_gradient)),
        np.finfo(np.float64).tiny,
    )
    passed = bool(np.all(absolute_difference <= allowed_difference))
    max_tolerance_ratio = float(np.max(absolute_difference / allowed_difference))
    agreement = {
        "schema_version": _SCHEMA_VERSION,
        "case_id": _CASE_ID,
        "scale": "native_default",
        "passed": passed,
        "tolerance_contract": "mirror_single_stage_initial_gradient",
        "rtol": rtol,
        "atol": atol,
        "gradient_dimension": int(direct_gradient.size),
        "direct_gradient_sha256": _array_sha256(direct_gradient),
        "parity_gradient_sha256": _array_sha256(parity_gradient),
        "direct_gradient_l2_norm": float(np.linalg.norm(direct_gradient)),
        "parity_gradient_l2_norm": float(np.linalg.norm(parity_gradient)),
        "direct_gradient_inf_norm": float(np.linalg.norm(direct_gradient, ord=np.inf)),
        "parity_gradient_inf_norm": float(np.linalg.norm(parity_gradient, ord=np.inf)),
        "difference_l2_norm": float(np.linalg.norm(difference)),
        "difference_inf_norm": float(np.linalg.norm(difference, ord=np.inf)),
        "max_abs_difference": float(np.max(absolute_difference)),
        "max_relative_difference": float(np.max(absolute_difference / symmetric_scale)),
        "max_tolerance_ratio": max_tolerance_ratio,
        "direct_initial_objective": float(direct["initial_objective"]),
        "parity_initial_objective": float(parity["initial_objective"]),
        "direct_adjoint_route": str(direct["adjoint_route"]),
        "parity_adjoint_route": str(parity["adjoint_route"]),
        **{key: str(direct[key]) for key in identity_keys},
    }
    if not passed:
        raise ValueError(
            "direct and parity initial gradients disagree under the frozen tolerance "
            f"contract (max tolerance ratio {max_tolerance_ratio:.17g})"
        )
    return agreement


def run_initial_gradient_agreement(
    *,
    artifact_root: Path,
    python_executable: str,
    repo_root: Path,
    base_environment: Mapping[str, str] = os.environ,
) -> Path:
    """Create one durable native-default agreement artifact from isolated probes."""
    artifact_root = artifact_root.resolve()
    if artifact_root.exists():
        raise ValueError("gradient agreement artifact root must not already exist")
    if artifact_root.is_relative_to(Path("/tmp").resolve()):
        raise ValueError("gradient agreement artifact must not be written under /tmp")
    artifact_root.mkdir(parents=True)
    inputs = artifact_root / "inputs"
    bundle = get_case(_CASE_ID).create_input(inputs, "native_default")
    probe_paths = {
        "direct": artifact_root / "direct.json",
        "parity": artifact_root / "parity.json",
    }
    for route, profile_id in (("direct", "jax_cpu_fast"), ("parity", "jax_cpu_parity")):
        environment = build_measurement_environment(
            profile_id,
            allocation_sensitive=False,
            base_environment=base_environment,
            repo_root=repo_root,
        )
        if route == "direct":
            environment[_EXACT_ADJOINT_SELECTOR] = "1"
        command = (
            python_executable,
            "-m",
            "benchmarks.single_stage_initial_gradient_agreement",
            "--probe-route",
            route,
            "--probe-input-root",
            str(inputs),
            "--probe-output",
            str(probe_paths[route]),
        )
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        _write_json_exclusive(
            artifact_root / f"{route}-process.json",
            {
                "command": list(command),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{route} gradient probe failed; see process record")

    direct = json.loads(probe_paths["direct"].read_text(encoding="utf-8"))
    parity = json.loads(probe_paths["parity"].read_text(encoding="utf-8"))
    agreement = compare_probe_documents(direct, parity)
    agreement.update(
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "input_fingerprint": bundle.input_fingerprint,
        }
    )
    _write_json_exclusive(artifact_root / "agreement.json", agreement)
    return artifact_root / "agreement.json"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--probe-route", choices=("direct", "parity"))
    parser.add_argument("--probe-input-root", type=Path)
    parser.add_argument("--probe-output", type=Path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _argument_parser().parse_args(arguments)
    if options.probe_route is not None:
        if options.probe_input_root is None or options.probe_output is None:
            _argument_parser().error(
                "probe mode requires --probe-input-root and --probe-output"
            )
        document = _probe_document(options.probe_input_root, options.probe_route)
        _write_json_exclusive(options.probe_output, document)
        return 0
    if options.artifact_root is None:
        _argument_parser().error("--artifact-root is required")
    agreement = run_initial_gradient_agreement(
        artifact_root=options.artifact_root,
        python_executable=options.python_executable,
        repo_root=options.repo_root.resolve(),
    )
    print(agreement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
