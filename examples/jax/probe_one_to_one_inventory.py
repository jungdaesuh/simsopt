"""Revalidate frozen source evidence in the one-to-one example inventory."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
NATIVE_TIERS: Final = frozenset(
    {"1_Simple", "2_Intermediate", "3_Advanced", "stellarator_benchmarks"}
)


class InventoryProbeError(ValueError):
    """Frozen inventory evidence disagrees with the checked-out native source."""


@dataclass(frozen=True)
class ProbeResult:
    """Immutable evidence emitted for one native source."""

    source: str
    source_sha256: str
    python_import_roots: tuple[str, ...]
    external_runtimes: tuple[str, ...]
    validated: bool = True


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise InventoryProbeError(f"{context} must be a JSON object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise InventoryProbeError(f"{context} must be a JSON array")
    return list(value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise InventoryProbeError(f"{context} must be a non-empty string")
    return value


def _strings(value: object, context: str) -> tuple[str, ...]:
    entries = tuple(
        _string(entry, f"{context} item") for entry in _sequence(value, context)
    )
    if entries != tuple(sorted(set(entries))):
        raise InventoryProbeError(f"{context} must be sorted and unique")
    return entries


def _source_path(source: str) -> Path:
    relative = PurePosixPath(source)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] not in NATIVE_TIERS
        or relative.suffix != ".py"
    ):
        raise InventoryProbeError(f"invalid native source path: {source}")
    path = REPO_ROOT / "examples" / Path(*relative.parts)
    if not path.is_file():
        raise InventoryProbeError(f"native source does not exist: {source}")
    return path


def _import_roots(tree: ast.AST) -> tuple[str, ...]:
    roots = {
        alias.name.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots.update(
        node.module.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    return tuple(sorted(roots))


def _inventory_rows(path: Path) -> tuple[dict[str, object], ...]:
    root = _mapping(json.loads(path.read_text(encoding="utf-8")), "inventory")
    if root.get("schema_version") != 1:
        raise InventoryProbeError("unsupported inventory schema_version")
    rows = tuple(
        _mapping(value, f"native_sources[{index}]")
        for index, value in enumerate(
            _sequence(root.get("native_sources"), "native_sources")
        )
    )
    sources = tuple(_string(row.get("source"), "row.source") for row in rows)
    if sources != tuple(sorted(set(sources))):
        raise InventoryProbeError("native_sources must be sorted and unique")
    return rows


def probe_row(row: dict[str, object]) -> ProbeResult:
    """Compile and compare one source against its frozen dependency evidence."""
    source = _string(row.get("source"), "row.source")
    source_path = _source_path(source)
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != _string(row.get("source_sha256"), "row.source_sha256"):
        raise InventoryProbeError(f"source hash changed: {source}")

    tree = ast.parse(source_bytes, filename=str(source_path))
    dependencies = _mapping(row.get("runtime_dependencies"), "runtime_dependencies")
    if set(dependencies) != {"python_import_roots", "external_runtimes"}:
        raise InventoryProbeError(f"invalid runtime dependency fields: {source}")
    expected_roots = _strings(
        dependencies["python_import_roots"], "python_import_roots"
    )
    actual_roots = _import_roots(tree)
    if actual_roots != expected_roots:
        raise InventoryProbeError(
            f"Python import roots changed for {source}: "
            f"expected={expected_roots}, actual={actual_roots}"
        )
    external_runtimes = _strings(dependencies["external_runtimes"], "external_runtimes")
    return ProbeResult(
        source=source,
        source_sha256=source_sha256,
        python_import_roots=actual_roots,
        external_runtimes=external_runtimes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--source")
    selection.add_argument("--all", action="store_true")
    return parser


def main() -> int:
    """Run a selected or complete inventory probe and emit canonical JSON."""
    arguments = _parser().parse_args()
    inventory_path: Path = arguments.inventory
    selected_source: str | None = arguments.source
    rows = _inventory_rows(inventory_path)
    selected = (
        rows
        if arguments.all
        else tuple(row for row in rows if row.get("source") == selected_source)
    )
    if not selected:
        raise InventoryProbeError(f"source is not inventoried: {selected_source}")
    results = tuple(probe_row(row) for row in selected)
    payload = {
        "source_count": len(results),
        "validated": True,
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
