from pathlib import Path
import re

from benchmarks.validation_ladder_contract import PARITY_LADDER_TOLERANCES


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "jax_parity_manifest.md"
COMPLETE_STATUSES = {
    "complete",
    "contract-complete",
    "cpu-contract-complete",
    "reduced-strict-complete",
    "exact",
}
PATH_COLUMNS = (
    "Upstream Python test file",
    "Upstream C++ implementation file",
    "JAX implementation file",
    "JAX parity test file",
)


def _extract_table(markdown: str, heading: str) -> list[dict[str, str]]:
    marker = f"## {heading}"
    start = markdown.index(marker)
    lines = markdown[start:].splitlines()
    table_start = next(
        index for index, line in enumerate(lines) if line.startswith("| ")
    )
    header = [cell.strip() for cell in lines[table_start].strip("|").split("|")]
    rows = []
    for line in lines[table_start + 2 :]:
        if not line.startswith("| "):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == len(header), line
        rows.append(dict(zip(header, cells)))
    return rows


def _paths_from_cell(cell: str) -> list[Path]:
    if cell.startswith("N/A:"):
        return []
    paths = [Path(match) for match in re.findall(r"`([^`]+)`", cell)]
    assert paths, f"Expected at least one backtick path in manifest cell: {cell}"
    return paths


def _banana_inventory_rows() -> list[dict[str, str]]:
    return _extract_table(
        MANIFEST.read_text(encoding="utf-8"),
        "Banana Coverage Inventory",
    )


def _row_has_cpp_implementation(row: dict[str, str]) -> bool:
    return not row["Upstream C++ implementation file"].startswith("N/A:")


def test_banana_coverage_inventory_references_existing_paths_and_lanes():
    rows = _banana_inventory_rows()

    assert rows, "Banana coverage inventory must contain at least one row."
    for row in rows:
        for column in PATH_COLUMNS:
            for relative_path in _paths_from_cell(row[column]):
                assert (REPO_ROOT / relative_path).exists(), (
                    f"{row['Coverage row']} references missing {column}: "
                    f"{relative_path}"
                )

        tolerance_lane = row["Tolerance lane"].strip("`")
        assert tolerance_lane in PARITY_LADDER_TOLERANCES, (
            f"{row['Coverage row']} references unknown tolerance lane "
            f"{tolerance_lane!r}."
        )

        status = row["CPU/JAX status"]
        carve_out = row["Known carve-out"].lower()
        if status in COMPLETE_STATUSES:
            assert carve_out == "none", (
                f"{row['Coverage row']} claims {status} but has carve-out: "
                f"{row['Known carve-out']}"
            )


def test_banana_required_non_cuda_cpp_lanes_are_cpu_jax_complete():
    rows = _banana_inventory_rows()
    cpp_rows = [row for row in rows if _row_has_cpp_implementation(row)]

    assert cpp_rows, "Banana inventory must include required C++ oracle lanes."
    for row in cpp_rows:
        assert row["CPU/JAX status"] in COMPLETE_STATUSES, (
            f"{row['Coverage row']} names a C++ oracle but is not CPU/JAX "
            f"complete: {row['CPU/JAX status']}"
        )
        assert row["Known carve-out"].lower() == "none", (
            f"{row['Coverage row']} names a C++ oracle but still carries a "
            f"carve-out: {row['Known carve-out']}"
        )
