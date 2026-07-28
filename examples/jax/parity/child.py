"""Internal child entrypoint for one statically registered parity case."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from simsopt_jax.examples import EXECUTION_SCALES, ExecutionScale

from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import read_input_bundle
from examples.jax.parity.provenance import collect_lane_provenance
from examples.jax.parity.receipts import write_lane_observation

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--lane", choices=("native-cpu", "jax-cpu", "jax-gpu"), required=True
    )
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--scale", choices=EXECUTION_SCALES, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load one immutable bundle, execute one lane, and publish its receipt."""
    args = _parser().parse_args(argv)
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
    observation = case.execute(args.lane, bundle, arrays)
    if args.lane.startswith("jax-"):
        import jax

        jax.block_until_ready(tuple(observation.values.values()))
    observation = dataclasses.replace(
        observation,
        provenance=collect_lane_provenance(_REPO_ROOT),
    )
    write_lane_observation(args.result_directory, observation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
