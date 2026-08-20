"""Source-owned contract for the shipped flat-675 single-stage example.

The example is a thin driver over the production module, so what needs holding
here is the seam between the two: that it builds the certified 11+3+661 layout
from repository geometry rather than from the host-local campaign bundle, that
its ``--bundle`` mode refuses by name instead of failing obscurely when that
bundle is absent, and that its manifest entry declares the fused lane's
transfer discipline rather than merely asserting it in prose.

The end-to-end run under a strict global transfer guard belongs to
``tests/integration/test_jax_examples.py``, which executes every ready example.
This file deliberately stops short of the solve: the fused program's compile
dominates that run, and paying it twice buys no coverage the integration lane
does not already have.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from simsopt_jax_adapters.geo.flat675 import (
    FLAT675_COIL_DOF_COUNT,
    FLAT675_OBJECTIVE_TERM_KEYS,
    FLAT675_OUTER_DOF_COUNT,
    FLAT675_SURFACE_DOF_COUNT,
    FLAT675_VESSEL_DOF_COUNT,
    Flat675ContractError,
    flat675_weighted_terms,
)

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples" / "jax" / "3_Advanced" / "single_stage_flat675.py"
MANIFEST = ROOT / "examples" / "jax" / "manifest.json"

EXAMPLE_ID = "flat675-single-stage-coupled-optimization"
RECEIPT_PATH = "docs/receipts/flat675_fused_campaign.md"


def _example_module() -> ModuleType:
    """Import the shipped example the way a reader would run it.

    The tier directories are not packages, so a file-location import is the
    only way to reach the script; this is the convention the sibling example
    contracts already use.
    """
    specification = importlib.util.spec_from_file_location(
        "flat675_single_stage_example", EXAMPLE
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example() -> ModuleType:
    return _example_module()


@pytest.fixture(scope="module")
def manifest_entry() -> dict[str, object]:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    matches = [entry for entry in document["jax_examples"] if entry["id"] == EXAMPLE_ID]
    assert len(matches) == 1, f"expected exactly one {EXAMPLE_ID} entry"
    return matches[0]


@pytest.fixture(scope="module")
def bounded_problem(example: ModuleType) -> object:
    """One repository-geometry build, shared by the tests that read it."""
    return example._repository_problem("bounded")


# --- registration -----------------------------------------------------------


def test_manifest_entry_points_at_this_script(
    manifest_entry: dict[str, object],
) -> None:
    assert manifest_entry["path"] == "3_Advanced/single_stage_flat675.py"
    assert (ROOT / "examples" / "jax" / str(manifest_entry["path"])) == EXAMPLE
    assert manifest_entry["status"] == "ready"
    assert manifest_entry["tier"] == "3_Advanced"


def test_manifest_entry_names_its_host_construction_seam(
    manifest_entry: dict[str, object],
) -> None:
    """``host_boundaries`` describes the workflow's seams, not the solve.

    The two live on different sides of a line this example draws explicitly.
    ``host_boundaries`` names the native construction the workflow still does
    on the host — the boundary, the coils and the winding surface are built
    from simsopt objects before anything is traced.  The claim that the SOLVE
    crosses no host boundary is a different claim, and it is carried by the
    transfer ledger this example publishes rather than by this field; a
    non-empty declaration here does not weaken it.
    """
    assert manifest_entry["host_boundaries"] == [
        "native boundary, coil, and winding-surface construction"
    ]
    assert manifest_entry["classification"] == "tutorial"
    assert manifest_entry["teaching_kind"] == "combined"
    assert manifest_entry["compatibility"] is None


def test_manifest_entry_declares_both_device_lanes(
    manifest_entry: dict[str, object],
) -> None:
    """A ready example must offer cpu-smoke and gpu-strict."""
    assert manifest_entry["supported_device_scopes"] == {
        "cpu": "full_workflow",
        "gpu": "full_workflow",
    }


def test_example_id_matches_the_manifest(example: ModuleType) -> None:
    assert example.EXAMPLE_ID == EXAMPLE_ID


def test_script_imports_no_host_scipy_optimizer() -> None:
    """A gpu-strict example may not hide a host optimizer behind JAX metrics."""
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert not [name for name in imported if name.split(".")[0] == "scipy"]


# --- the repository-geometry configuration ---------------------------------


def test_repository_geometry_builds_the_certified_layout(
    bounded_problem: object,
) -> None:
    """Clone-runnable geometry, the same 11+3+661 layout as the campaign."""
    candidate = bounded_problem.start_candidate  # type: ignore[attr-defined]
    assert candidate.outer_vector().shape == (FLAT675_OUTER_DOF_COUNT,)
    assert len(candidate.coil_coordinates) == FLAT675_COIL_DOF_COUNT
    assert len(candidate.vessel_coordinates) == FLAT675_VESSEL_DOF_COUNT
    assert len(candidate.surface_coordinates) == FLAT675_SURFACE_DOF_COUNT


def test_repository_geometry_objective_is_finite(bounded_problem: object) -> None:
    """Every certified term evaluates: the shipped start is not degenerate."""
    problem = bounded_problem
    terms = np.asarray(
        flat675_weighted_terms(
            problem.start_candidate.outer_vector(),  # type: ignore[attr-defined]
            material=problem.material,  # type: ignore[attr-defined]
            objective_policy=problem.objective_policy,  # type: ignore[attr-defined]
            boozer_policy=problem.boozer_policy,  # type: ignore[attr-defined]
        ),
        dtype=np.float64,
    )

    assert terms.shape == (len(FLAT675_OBJECTIVE_TERM_KEYS),)
    assert np.all(np.isfinite(terms))


def test_repository_geometry_optimizes_a_free_winding_surface_coil(
    bounded_problem: object,
    example: ModuleType,
) -> None:
    """The shape penalties point at a free coil, never at a fixed TF coil."""
    index = bounded_problem.objective_policy.optimized_coil_index  # type: ignore[attr-defined]
    coils = bounded_problem.material.boozer.coil_dof_extraction.coils  # type: ignore[attr-defined]
    assert index >= 0
    assert tuple(coils[index].current_map.owner_segments) == ((0, 1, 0, 1),)
    # The fixed TF coils come first and claim nothing.
    assert not tuple(coils[0].curve_map.owner_segments)
    assert example.TF_BASE_COIL_COUNT > 0


# --- the certified frozen-bundle configuration ------------------------------


def test_bundle_mode_refuses_by_name_when_the_bundle_is_absent(
    example: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An input-boundary refusal that says where it looked and what to do."""
    monkeypatch.setattr(example, "BUNDLE_ROOT", Path("/nonexistent/flat675-bundle"))

    with pytest.raises(Flat675ContractError) as excinfo:
        example._bundle_problem()

    message = str(excinfo.value)
    assert "--bundle" in message
    assert "/nonexistent/flat675-bundle" in message


def test_bundle_flag_selects_the_certified_configuration(
    example: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--bundle`` routes to the bundle solve and is not forwarded onward.

    The shared example CLI does not know this flag, so the script must consume
    it; forwarding it would make every ``--bundle`` run die in argument
    parsing instead of running the certified configuration.
    """
    captured: dict[str, object] = {}

    def _capture(arguments, **keywords):  # type: ignore[no-untyped-def]
        captured["arguments"] = list(arguments)
        captured["solve"] = keywords["solve"]
        return 0

    monkeypatch.setattr(example, "run_example", _capture)

    assert example.main(["--bundle", "--json"]) == 0
    assert captured["arguments"] == ["--json"]
    assert captured["solve"] is example.solve_bundle

    assert example.main(["--json"]) == 0
    assert captured["arguments"] == ["--json"]
    assert captured["solve"] is example.solve


# --- the disclosure the receipt requires ------------------------------------


def test_docstring_scopes_the_receipt_and_discloses_cold_start() -> None:
    """The example may not inherit the receipt's number past its own scope."""
    docstring = ast.get_docstring(ast.parse(EXAMPLE.read_text(encoding="utf-8")))
    assert docstring is not None
    # The prose is hard-wrapped, so compare against a single-spaced rendering.
    prose = " ".join(docstring.split())

    assert RECEIPT_PATH in prose
    # The certification is the bundle configuration's, not the default's.
    assert "--bundle" in prose
    assert "NO timing claim of its own" in prose
    # Cold start is disclosed rather than claimed.
    assert "XLA compile" in prose
