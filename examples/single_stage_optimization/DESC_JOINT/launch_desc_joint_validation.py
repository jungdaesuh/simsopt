#!/usr/bin/env python3
"""Launch SIMSOPT Poincare/Boozer validation for a DESC joint export."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(EXAMPLES_ROOT))

from banana_opt.desc_joint_validation_launcher import (  # noqa: E402
    infer_desc_joint_exported_artifact_paths,
    launch_desc_joint_simsopt_validation,
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result_path = Path(args.result).expanduser().resolve()
    result_payload = _read_json_object(result_path, field_name="--result")
    exported_artifacts = tuple(args.exported_artifact)
    if not exported_artifacts:
        exported_artifacts = infer_desc_joint_exported_artifact_paths(result_payload)
    if not exported_artifacts:
        raise ValueError(
            "No exported artifacts supplied and desc_result.json does not record "
            "an exported SIMSOPT artifact."
        )
    requested_poincare_render_modes = (
        None
        if not args.poincare_render_mode
        else tuple(args.poincare_render_mode)
    )
    artifacts = launch_desc_joint_simsopt_validation(
        result_payload=result_payload,
        exported_artifact_paths=exported_artifacts,
        output_root=Path(args.output_root).expanduser().resolve(),
        surface_path=args.surface,
        python_executable=args.python_executable,
        poincare_render_modes=requested_poincare_render_modes,
        poincare_timeout_seconds=args.poincare_timeout_seconds,
        run_poincare=not args.skip_poincare,
        run_boozer=not args.skip_boozer,
        require_boozer_state=args.require_boozer_state,
        dry_run=args.dry_run,
        iota_guess=args.iota,
        G_guess=args.G,
    )
    manifest_path = artifacts.validation_manifest_path()
    print(
        os.fspath(
            artifacts.launch_report_path
            if manifest_path is None
            else manifest_path
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and launch high-cost SIMSOPT Poincare/Boozer validation "
            "for a DESC-exported SIMSOPT BiotSavart artifact."
        ),
    )
    parser.add_argument("--result", required=True, help="DESC joint desc_result.json.")
    parser.add_argument(
        "--exported-artifact",
        action="append",
        default=[],
        help=(
            "DESC-exported SIMSOPT artifact to validate. If omitted, the path "
            "is inferred from desc_result.json."
        ),
    )
    parser.add_argument(
        "--surface",
        help=(
            "Surface JSON to validate against. Fixed-polish runs default to "
            "input_contract.selected_seed.surface from desc_result.json; joint "
            "runs require this argument because the plasma boundary moved."
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--poincare-render-mode",
        action="append",
        default=[],
        choices=("validation", "diagnostic", "default"),
    )
    parser.add_argument("--poincare-timeout-seconds", type=float)
    parser.add_argument("--skip-poincare", action="store_true")
    parser.add_argument("--skip-boozer", action="store_true")
    parser.add_argument(
        "--require-boozer-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require a passing Boozer sidecar in the physics evidence.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--iota", type=float)
    parser.add_argument("--G", type=float)
    return parser


def _read_json_object(path: Path, *, field_name: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must contain a JSON object: {path}.")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
