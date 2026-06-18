"""#1: the in-process finalization wiring of the decisive confinement gate.

These tests exercise the example-level promotion helpers that turn one confinement-gate run
into results.json fields and a FAIL-CLOSED promotion verdict:

  * run_final_confinement_gate -- the single automated caller; skips (init/off) with a loud
    not-certified marker, or runs the SSOT strict-tier composition and returns the verdict.
  * confinement_certified_promotion -- folds the gate into the upstream feasibility verdict;
    a rejected OR un-certified candidate is never promoted.
  * confinement_gate_results_payload -- the flat results.json record of the outcome.

The strict tier (50 lines / tmax 7000) is too expensive for a unit test, so the strict-tier
runner is stubbed before the example module is imported; the tests assert observable behavior
(skip vs run, verdict file written, promotion blocked on reject/skip), never a mock-only
tautology.
"""

import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
EXAMPLE_MODULE_PATH = EXAMPLES_ROOT / "SINGLE_STAGE" / "single_stage_banana_example.py"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

_STRONG_STRICT_VERDICT = {
    "broken": False,
    "survival_fraction": 0.96,
    "surviving_seed_count": 48,
    "island_chain_count": 0,
    "classifiable_fraction": 0.95,
    "passed": True,
}
_BROKEN_STRICT_VERDICT = {"broken": True, "survival_fraction": 0.0}


def _install_fake_strict_tier(topology_verdict, *, passed):
    fake = types.ModuleType("run_topology_fidelity_ladder")

    def evaluate_case(_run_dir):
        return {
            "field_label": "opt",
            "strict": {"passed": passed, "topology_verdict": topology_verdict},
        }

    fake.evaluate_case = evaluate_case
    sys.modules["run_topology_fidelity_ladder"] = fake
    return fake


@pytest.fixture
def example_module():
    # Stub the heavy strict-tier runner BEFORE the example imports it lazily, so the import
    # and run_final_confinement_gate(strict) stay cheap and deterministic. Loaded fresh under
    # a unique name (importlib pattern) for isolation.
    _install_fake_strict_tier(_STRONG_STRICT_VERDICT, passed=True)
    spec = importlib.util.spec_from_file_location(
        f"single_stage_banana_example_{uuid.uuid4().hex}", EXAMPLE_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gate_off_skips_and_marks_not_certified(example_module):
    status = example_module.run_final_confinement_gate(
        "off",
        "/nonexistent",
        init_only=False,
        nonqs_ratio=3e-3,
        beta=0.0,
        magnetic_well=None,
        iota_shear=0.05,
    )
    assert status["ran"] is False
    assert status["accepted"] is None  # skipped is NOT a pass
    assert "NOT confinement-certified" in status["decisive_reason"]


def test_init_only_skips_even_when_strict_requested(example_module):
    status = example_module.run_final_confinement_gate(
        "strict",
        "/nonexistent",
        init_only=True,
        nonqs_ratio=None,
        beta=0.0,
        magnetic_well=None,
        iota_shear=None,
    )
    assert status["ran"] is False
    assert status["accepted"] is None


def test_strict_accept_runs_writes_verdict_and_allows_promotion(example_module, tmp_path):
    status = example_module.run_final_confinement_gate(
        "strict",
        tmp_path,
        init_only=False,
        nonqs_ratio=3e-3,
        beta=0.0,
        magnetic_well=None,
        iota_shear=0.05,
    )
    assert status["ran"] is True
    assert status["accepted"] is True
    verdict_file = tmp_path / "confinement_verdict.json"
    assert verdict_file.exists()
    assert json.loads(verdict_file.read_text())["confinement_verdict"]["accepted"] is True
    # Promotion is the conjunction of base feasibility AND gate acceptance.
    assert example_module.confinement_certified_promotion(status, True) is True
    assert example_module.confinement_certified_promotion(status, False) is False


def test_strict_reject_blocks_promotion_even_with_base_ok(example_module, tmp_path):
    # Re-stub for a broken-topology run (fresh module already imported; only evaluate_case
    # is consulted at call time, so swapping it here is sufficient and observable).
    _install_fake_strict_tier(_BROKEN_STRICT_VERDICT, passed=False)
    status = example_module.run_final_confinement_gate(
        "strict",
        tmp_path,
        init_only=False,
        nonqs_ratio=3e-3,
        beta=0.0,
        magnetic_well=None,
        iota_shear=0.05,
    )
    assert status["ran"] is True
    assert status["accepted"] is False
    assert status["decisive_reason"].startswith("REJECT: topology")
    # FAIL-CLOSED: a gate-rejected candidate is never promoted, even if hardware/frontier OK.
    assert example_module.confinement_certified_promotion(status, True) is False


def test_skipped_gate_is_never_promoted(example_module):
    skipped = example_module.run_final_confinement_gate(
        "off",
        "/nonexistent",
        init_only=False,
        nonqs_ratio=None,
        beta=0.0,
        magnetic_well=None,
        iota_shear=None,
    )
    # Absence of certification must read as not-promoted, not as a pass.
    assert example_module.confinement_certified_promotion(skipped, True) is False


def test_results_payload_records_outcome(example_module, tmp_path):
    status = example_module.run_final_confinement_gate(
        "strict",
        tmp_path,
        init_only=False,
        nonqs_ratio=3e-3,
        beta=0.0,
        magnetic_well=None,
        iota_shear=0.05,
    )
    payload = example_module.confinement_gate_results_payload(status)
    assert payload["CONFINEMENT_GATE_MODE"] == "strict"
    assert payload["CONFINEMENT_GATE_RAN"] is True
    assert payload["CONFINEMENT_GATE_ACCEPTED"] is True
    assert payload["CONFINEMENT_GATE_VERDICT_FILENAME"] == "confinement_verdict.json"
    assert "ACCEPT" in payload["CONFINEMENT_GATE_DECISIVE_REASON"]
    json.dumps(payload)  # results.json must serialize the block


def test_results_payload_for_skipped_gate_has_no_verdict_file(example_module):
    status = example_module.run_final_confinement_gate(
        "off",
        "/nonexistent",
        init_only=False,
        nonqs_ratio=None,
        beta=0.0,
        magnetic_well=None,
        iota_shear=None,
    )
    payload = example_module.confinement_gate_results_payload(status)
    assert payload["CONFINEMENT_GATE_RAN"] is False
    assert payload["CONFINEMENT_GATE_ACCEPTED"] is None
    assert payload["CONFINEMENT_GATE_VERDICT_FILENAME"] is None
