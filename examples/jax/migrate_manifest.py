"""Render a validated JAX examples manifest-v2 candidate without writing it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPO_ROOT / "src"
for import_root in (str(_SOURCE_ROOT), str(_REPO_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from examples.jax._manifest import convert_v1_document_to_v2, parse_manifest_document

_COMPATIBILITY_DURATION = "one release"
_ROLLBACK_COMMAND = "git checkout -- examples/jax/manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Print deterministic candidate bytes and migration evidence only."""
    parsed = _parser().parse_args(arguments)
    document = json.loads(parsed.input.read_text(encoding="utf-8"))
    observed = parse_manifest_document(
        document,
        repo_root=_REPO_ROOT,
        warn_legacy=True,
        allow_historical_catalog=True,
    )
    candidate_bytes, semantic_diff = convert_v1_document_to_v2(
        document,
        repo_root=_REPO_ROOT,
        allow_historical_catalog=True,
    )
    candidate_text = candidate_bytes.decode("utf-8")
    output = "\n".join(
        (
            f"manifest_schema_version={observed.schema_version}",
            "used_legacy_manifest_adapter="
            f"{str(observed.used_legacy_manifest_adapter).lower()}",
            f"compatibility_duration={_COMPATIBILITY_DURATION}",
            "semantic_diff="
            + json.dumps(semantic_diff, separators=(",", ":"), sort_keys=True),
            f"candidate_sha256={hashlib.sha256(candidate_bytes).hexdigest()}",
            f"rollback_command={_ROLLBACK_COMMAND}",
            "candidate_v2:",
        )
    )
    sys.stdout.write(output + "\n" + candidate_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
