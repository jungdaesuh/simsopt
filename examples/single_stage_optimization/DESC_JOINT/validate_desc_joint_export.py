#!/usr/bin/env python3
"""Materialize DESC joint SIMSOPT physics validation from sidecar artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(EXAMPLES_ROOT))

from banana_opt.desc_joint_simsopt_validation import (  # noqa: E402
    materialize_desc_joint_simsopt_validation,
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result_payload = _read_json_object(
        Path(args.result).expanduser().resolve(),
        field_name="--result",
    )
    artifacts = materialize_desc_joint_simsopt_validation(
        result_payload=result_payload,
        exported_artifact_paths=tuple(args.exported_artifact),
        poincare_metrics_paths=tuple(args.poincare_metrics),
        boozer_state_paths=tuple(args.boozer_state),
        require_boozer_state=args.require_boozer_state,
        validated_surface_path=args.validated_surface,
        output_root=Path(args.output_root).expanduser().resolve(),
    )
    print(os.fspath(artifacts.validation_manifest_path))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build DESC joint SIMSOPT physics-validation evidence from existing "
            "Poincare/Boozer sidecar artifacts."
        ),
    )
    parser.add_argument("--result", required=True, help="DESC joint desc_result.json.")
    parser.add_argument(
        "--exported-artifact",
        action="append",
        required=True,
        help=(
            "Exported SIMSOPT artifact validated by the sidecars. Repeat for "
            "multiple exported artifacts."
        ),
    )
    parser.add_argument(
        "--poincare-metrics",
        action="append",
        required=True,
        help=(
            "Existing PoincareMetrics*.json sidecar to consume. Repeat for "
            "multiple metrics artifacts."
        ),
    )
    parser.add_argument(
        "--boozer-state",
        action="append",
        default=[],
        help="Existing *_boozer_state.json sidecar. Repeat for multiple states.",
    )
    parser.add_argument(
        "--require-boozer-state",
        action="store_true",
        help="Fail physics validation if no Boozer state sidecar is supplied.",
    )
    parser.add_argument(
        "--validated-surface",
        help=(
            "Surface artifact validated by the supplied sidecars. Required for "
            "joint-mode result payloads unless a Boozer state sidecar records "
            "surface_path."
        ),
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
