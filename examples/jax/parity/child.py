"""Internal child entrypoint for one statically registered parity case."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import read_input_bundle
from examples.jax.parity.measurement import MeasurementExecution
from examples.jax.parity.provenance import (
    collect_lane_provenance,
    collect_snapshot_lane_provenance,
    load_snapshot_lane_identity,
)
from examples.jax.parity.receipts import write_lane_observation
from simsopt_jax.examples import EXECUTION_SCALES, ExecutionScale

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NATIVE_MEASUREMENT_SYNCHRONIZATION = "native synchronous execution"
_JAX_MEASUREMENT_SYNCHRONIZATION = (
    "jax.block_until_ready over published observation values"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--lane", choices=("native-cpu", "jax-cpu", "jax-gpu"), required=True
    )
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--trajectory-path", type=Path)
    parser.add_argument("--optimization-timing-path", type=Path)
    parser.add_argument("--optimizer-backend", choices=("optax-lbfgs",))
    parser.add_argument("--immutable-snapshot-provenance", type=Path)
    parser.add_argument("--scale", choices=EXECUTION_SCALES, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load one immutable bundle, execute one lane, and publish its receipt."""
    args = _parser().parse_args(argv)
    snapshot_identity = (
        None
        if args.immutable_snapshot_provenance is None
        else load_snapshot_lane_identity(args.immutable_snapshot_provenance)
    )
    if snapshot_identity is not None:
        if snapshot_identity.snapshot_root.resolve() != _REPO_ROOT.resolve():
            raise ValueError("snapshot lane identity root does not match executed root")
        expected_profile = (
            "native_cpu"
            if args.lane == "native-cpu"
            else "jax_gpu_optax"
            if args.lane == "jax-gpu" and args.optimizer_backend == "optax-lbfgs"
            else "jax_gpu_fast"
            if args.lane == "jax-gpu"
            else None
        )
        if snapshot_identity.profile_id != expected_profile:
            raise ValueError(
                "snapshot lane identity profile does not match child route"
            )
    case = get_case(args.case)
    bundle, arrays = read_input_bundle(args.input_bundle.parent)
    scale: ExecutionScale = args.scale
    if bundle.case_id != case.case_id:
        raise ValueError(
            f"input bundle case {bundle.case_id} does not match {case.case_id}"
        )
    if bundle.scale != scale:
        raise ValueError(
            f"input bundle scale {bundle.scale} does not match requested {scale}"
        )
    if (
        args.trajectory_path is None
        and args.optimization_timing_path is None
        and args.optimizer_backend is None
    ):
        observation = case.execute(args.lane, bundle, arrays)
    else:
        if case.measurement_execute is None:
            raise ValueError(f"case {case.case_id} does not support measurements")
        observation = case.measurement_execute(
            args.lane,
            bundle,
            arrays,
            MeasurementExecution(
                trajectory_path=args.trajectory_path,
                optimization_timing_path=args.optimization_timing_path,
                optimizer_backend=args.optimizer_backend,
            ),
        )
    measurement_synchronization = _NATIVE_MEASUREMENT_SYNCHRONIZATION
    if args.lane.startswith("jax-"):
        import jax

        jax.block_until_ready(  # pyright: ignore[reportAttributeAccessIssue]
            tuple(observation.values.values())
        )
        measurement_synchronization = _JAX_MEASUREMENT_SYNCHRONIZATION
    if args.immutable_snapshot_provenance is None:
        provenance = collect_lane_provenance(
            _REPO_ROOT,
            measurement_synchronization=measurement_synchronization,
        )
    else:
        assert snapshot_identity is not None
        if (
            observation.lane != snapshot_identity.lane
            or observation.backend_mode != snapshot_identity.backend_mode
            or observation.driver != snapshot_identity.driver
            or observation.platform != snapshot_identity.execution_platform
        ):
            raise ValueError("snapshot lane identity does not match observation route")
        provenance = collect_snapshot_lane_provenance(
            snapshot_identity,
            measurement_synchronization=measurement_synchronization,
        )
    observation = dataclasses.replace(observation, provenance=provenance)
    write_lane_observation(args.result_directory, observation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
