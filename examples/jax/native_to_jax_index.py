"""Generate the source-owned native-to-JAX example index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.jax.manifest_contracts_v3 import load_manifest_contract_pair_documents
from examples.jax.official_source_catalog import (
    OFFICIAL_UPSTREAM_COMMIT,
    OFFICIAL_UPSTREAM_DEFAULT_BRANCH,
    OFFICIAL_UPSTREAM_REPOSITORY,
)

JAX_EXAMPLES_DIRECTORY = Path(__file__).resolve().parent
REPO_ROOT = JAX_EXAMPLES_DIRECTORY.parents[1]
INDEX_PATH = JAX_EXAMPLES_DIRECTORY / "NATIVE_TO_JAX_INDEX.md"
MANIFEST_PATH = JAX_EXAMPLES_DIRECTORY / "manifest.json"
PARITY_MANIFEST_PATH = JAX_EXAMPLES_DIRECTORY / "parity_manifest.json"


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def render_native_to_jax_index(*, repo_root: Path = REPO_ROOT) -> str:
    """Render the complete index from validated manifest contracts."""
    jax_examples_directory = repo_root / "examples" / "jax"
    pair = load_manifest_contract_pair_documents(
        _load_json(jax_examples_directory / MANIFEST_PATH.name),
        _load_json(jax_examples_directory / PARITY_MANIFEST_PATH.name),
        repo_root=repo_root,
    )
    examples_by_id = {example.id: example for example in pair.examples.jax_examples}
    parity_by_source = {
        relationship.native_source: relationship
        for relationship in pair.parity.all_relationships
    }
    table_header = (
        "| Native example | JAX mirror | Classification | "
        "Runtime dependencies | Device scope | Scale |",
        "| --- | --- | --- | --- | --- | --- |",
    )
    lines = [
        "# Native-to-JAX example index",
        "",
        (
            "Generated from `manifest.json` and `parity_manifest.json`. "
            "Do not edit this table by hand."
        ),
        (
            f"Official upstream scope: `{OFFICIAL_UPSTREAM_REPOSITORY}` "
            f"`{OFFICIAL_UPSTREAM_DEFAULT_BRANCH}` at "
            f"`{OFFICIAL_UPSTREAM_COMMIT}` "
            f"({len(pair.examples.source_catalog)} source files)."
        ),
        (
            f"Local experimental registrations: "
            f"{len(pair.examples.experimental_sources)} source files."
        ),
        "",
        "## Official upstream catalog",
        "",
        *table_header,
    ]
    for source_index, source in enumerate(pair.examples.all_sources):
        if source_index == len(pair.examples.source_catalog):
            lines.extend(("", "## Experimental local registrations", "", *table_header))
        example = (
            examples_by_id[source.mirror_example_id]
            if source.mirror_example_id is not None
            else None
        )
        relationship = parity_by_source.get(source.source)
        mirror_path = "—" if example is None else f"`examples/jax/{example.path}`"
        classification = source.disposition
        if example is not None:
            classification = f"{source.disposition} / {example.classification}"
        dependencies = (
            ", ".join(source.dependencies.external_runtimes)
            if source.dependencies.external_runtimes
            else "none"
        )
        device_scope = "—"
        if example is not None:
            device_scope = ", ".join(
                f"{device}: {scope}"
                for device, scope in example.supported_device_scopes
            )
            if example.outer_optimizer_policy is not None:
                device_scope = "outer: CPU SciPy; " + device_scope
        scale = (
            "—" if relationship is None else ", ".join(relationship.supported_scales)
        )
        cells = (
            f"`examples/{source.source}`",
            mirror_path,
            classification,
            dependencies,
            device_scope,
            scale,
        )
        lines.append("| " + " | ".join(_markdown_cell(cell) for cell in cells) + " |")
    lines.extend(
        (
            "",
            "Regenerate with:",
            "",
            "```bash",
            "python -m examples.jax.native_to_jax_index --write",
            "```",
            "",
            "`--check` verifies index consistency.",
            "",
        )
    )
    return "\n".join(lines)


def main(arguments: list[str] | None = None) -> int:
    """Write or verify the generated index."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    rendered = render_native_to_jax_index()
    if options.write:
        INDEX_PATH.write_text(rendered, encoding="utf-8")
        return 0
    if not INDEX_PATH.is_file() or INDEX_PATH.read_text(encoding="utf-8") != rendered:
        raise SystemExit("native-to-JAX index is stale; regenerate it with --write")
    print("Index consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
