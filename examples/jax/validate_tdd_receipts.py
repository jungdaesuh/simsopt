"""Validate and optionally replay one-to-one JAX example TDD receipts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.jax.tdd_receipts import (
    render_receipt_markdown,
    validate_receipt_document,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt_path", type=Path)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--write-markdown", type=Path)
    return parser


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(arguments: Sequence[str] | None = None) -> int:
    """Validate one trusted receipt document and print its audit summary."""
    args = _parser().parse_args(arguments)
    document = json.loads(args.receipt_path.read_text(encoding="utf-8"))
    audit = validate_receipt_document(
        document,
        repo_root=args.repo_root.resolve(),
        replay=bool(args.replay),
    )
    if args.write_markdown is not None:
        _write_text_atomic(args.write_markdown, render_receipt_markdown(document))
    print(json.dumps(dataclasses.asdict(audit), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
