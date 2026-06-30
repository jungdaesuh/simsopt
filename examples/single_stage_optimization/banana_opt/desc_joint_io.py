"""Shared DESC-joint JSON and checksum helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path


def read_json_mapping(
    path: Path,
    *,
    error_message: str | Callable[[object], str],
) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        message = error_message(payload) if callable(error_message) else error_message
        raise ValueError(message)
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
