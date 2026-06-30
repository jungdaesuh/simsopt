#!/usr/bin/env python3
"""Launch direct hardware/contact oracle for a DESC joint export."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(EXAMPLES_ROOT))

from banana_opt.desc_joint_hardware_oracle_launcher import (  # noqa: E402
    infer_hardware_oracle_exported_artifact_paths,
    launch_desc_joint_hardware_oracle,
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result_payload = _read_json_object(
        Path(args.result).expanduser().resolve(),
        field_name="--result",
    )
    exported_artifacts = tuple(args.exported_artifact)
    if not exported_artifacts:
        exported_artifacts = infer_hardware_oracle_exported_artifact_paths(
            result_payload
        )
    if not exported_artifacts:
        raise ValueError(
            "No exported artifacts supplied and desc_result.json does not record "
            "an exported SIMSOPT artifact."
        )
    artifacts = launch_desc_joint_hardware_oracle(
        result_payload=result_payload,
        exported_artifact_paths=exported_artifacts,
        oracle_source_artifact_path=args.oracle_source_artifact,
        output_root=Path(args.output_root).expanduser().resolve(),
        physics_report_path=args.physics_report,
        audit_script_path=args.audit_script,
        python_executable=args.python_executable,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
    )
    print(
        os.fspath(
            artifacts.launch_report_path
            if artifacts.validation_manifest_path is None
            else artifacts.validation_manifest_path
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the direct hardware/contact oracle for a DESC-exported artifact "
            "and write checksum-bound DESC final-oracle evidence."
        ),
    )
    parser.add_argument("--result", required=True, help="DESC joint desc_result.json.")
    parser.add_argument(
        "--exported-artifact",
        action="append",
        default=[],
        help=(
            "DESC-exported SIMSOPT artifact covered by the oracle evidence. If "
            "omitted, infer from desc_result.json."
        ),
    )
    parser.add_argument(
        "--oracle-source-artifact",
        required=True,
        help=(
            "Boozer/BiotSavart source artifact consumed by the existing "
            "hardware-contact auditor. For fixed-polish DESC exports this is "
            "normally simsopt_validation_run/surf_desc_export_boozer_surface.json."
        ),
    )
    parser.add_argument("--physics-report")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--audit-script",
        default="/Users/suhjungdae/code/columbia/autoresearch/scripts/audit_hardware_contacts.py",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_json_object(path: Path, *, field_name: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must contain a JSON object: {path}.")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
