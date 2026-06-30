#!/usr/bin/env python3
"""Materialize the fail-closed outer-loop decision for a DESC joint candidate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(EXAMPLES_ROOT))

from banana_opt.desc_joint_outer_loop import (  # noqa: E402
    materialize_desc_joint_outer_loop_decision,
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result_path = Path(args.result).expanduser().resolve()
    validation_manifest_path = Path(args.validation_manifest).expanduser().resolve()
    artifacts = materialize_desc_joint_outer_loop_decision(
        result_payload=_read_json_object(result_path, field_name="--result"),
        validation_manifest=_read_json_object(
            validation_manifest_path,
            field_name="--validation-manifest",
        ),
        output_root=Path(args.output_root).expanduser().resolve(),
        validation_manifest_path=validation_manifest_path,
    )
    print(os.fspath(artifacts.decision_path))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reject or accept a DESC joint candidate for a constrained outer "
            "production search loop using the checksum-bound validation manifest."
        ),
    )
    parser.add_argument("--result", required=True, help="DESC joint desc_result.json.")
    parser.add_argument(
        "--validation-manifest",
        required=True,
        help="DESC joint validation manifest with physics and hardware evidence.",
    )
    parser.add_argument("--output-root", required=True)
    return parser


def _read_json_object(path: Path, *, field_name: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must contain a JSON object: {path}.")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
