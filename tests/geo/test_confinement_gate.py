import json
import sys
import types
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import run_confinement_gate as driver  # noqa: E402 (cheap: heavy single-stage import is lazy)

from banana_opt.confinement_gate import (  # noqa: E402
    CRITERION_CLASSIFIABILITY,
    CRITERION_MERCIER,
    CRITERION_TOPOLOGY,
    ConfinementGateConfig,
    ConfinementMetrics,
    certify_confinement,
    evaluate_confinement_gate,
    metrics_from_topology_verdict,
)
from banana_opt.single_stage_search_contracts import (  # noqa: E402
    CLAMPED_VACUOUS_FRONTIER_INVARIANT_TORUS_WARNING,
    DEFAULT_FRONTIER_INVARIANT_TORUS_MIN,
    resolve_frontier_invariant_torus_min,
)


def _criterion(verdict, name):
    return next(c for c in verdict.criteria if c.name == name)


def test_vacuum_candidate_with_strong_topology_is_accepted():
    # 47/50 strict lines, vacuum (beta=0): Mercier N/A, QA/well/shear advisory.
    metrics = ConfinementMetrics(
        survived_lines=47, total_lines=50, beta=0.0,
        qa_nonqs_ratio=5.0e-3, magnetic_well=-0.1, iota_shear=0.05, island_chains=0,
    )
    verdict = evaluate_confinement_gate(metrics)
    assert verdict.accepted
    assert "ACCEPT" in verdict.decisive_reason


def test_zero_survival_is_rejected_on_topology():
    metrics = ConfinementMetrics(survived_lines=0, total_lines=50, beta=0.0)
    verdict = evaluate_confinement_gate(metrics)
    assert not verdict.accepted
    assert _criterion(verdict, CRITERION_TOPOLOGY).passed is False
    assert verdict.decisive_reason.startswith("REJECT: topology")


def test_missing_topology_fails_closed():
    # No topology evidence at all -> reject (integrity boundary, never defined away).
    metrics = ConfinementMetrics(survived_lines=None, total_lines=0, beta=0.0)
    verdict = evaluate_confinement_gate(metrics)
    assert not verdict.accepted
    topo = _criterion(verdict, CRITERION_TOPOLOGY)
    assert topo.applicable is True and topo.passed is False
    assert "fail-closed" in topo.detail


def test_survival_fraction_threshold_boundary():
    # 45/50 == 0.90 default bar passes; 44/50 fails.
    assert evaluate_confinement_gate(
        ConfinementMetrics(survived_lines=45, total_lines=50)
    ).accepted
    assert not evaluate_confinement_gate(
        ConfinementMetrics(survived_lines=44, total_lines=50)
    ).accepted


def test_mercier_not_applicable_in_vacuum_even_when_required():
    # require_mercier=True (default) but beta=0 -> Mercier N/A, must NOT block.
    metrics = ConfinementMetrics(survived_lines=50, total_lines=50, beta=0.0)
    verdict = evaluate_confinement_gate(metrics)
    merc = _criterion(verdict, CRITERION_MERCIER)
    assert merc.applicable is False
    assert verdict.accepted
    assert "vacuum/N/A" in merc.detail


def test_mercier_finite_beta_unstable_is_rejected():
    # beta>0 with negative min DMerc (unstable) and require_mercier=True -> reject.
    metrics = ConfinementMetrics(
        survived_lines=50, total_lines=50, beta=0.02, mercier_dmerc_min=-1.0e-3,
    )
    verdict = evaluate_confinement_gate(metrics)
    merc = _criterion(verdict, CRITERION_MERCIER)
    assert merc.applicable is True and merc.passed is False
    assert not verdict.accepted
    assert verdict.decisive_reason.startswith("REJECT: mercier")


def test_mercier_finite_beta_stable_is_accepted():
    metrics = ConfinementMetrics(
        survived_lines=50, total_lines=50, beta=0.02, mercier_dmerc_min=1.0e-4,
    )
    assert evaluate_confinement_gate(metrics).accepted


def test_qa_is_advisory_by_default_but_decisive_when_required():
    # Bad QA (above max) but topology fine: advisory default accepts; require_qa rejects.
    metrics = ConfinementMetrics(
        survived_lines=50, total_lines=50, beta=0.0, qa_nonqs_ratio=0.5,
    )
    assert evaluate_confinement_gate(metrics).accepted  # advisory
    rejected = evaluate_confinement_gate(
        metrics, ConfinementGateConfig(require_qa=True)
    )
    assert not rejected.accepted
    assert rejected.decisive_reason.startswith("REJECT: quasisymmetry")


def test_island_chains_block_by_default_and_can_be_made_advisory():
    metrics = ConfinementMetrics(
        survived_lines=50, total_lines=50, beta=0.0, island_chains=2,
    )
    # Islands are part of the strict-topology standard => required by default => reject.
    assert not evaluate_confinement_gate(metrics).accepted
    # Explicitly relaxing islands to advisory accepts (survival still decisive).
    assert evaluate_confinement_gate(
        metrics, ConfinementGateConfig(require_islands=False)
    ).accepted


def test_classifiability_below_floor_blocks_by_default():
    metrics = ConfinementMetrics(
        survived_lines=50, total_lines=50, beta=0.0, classifiable_fraction=0.5,
    )
    # Required by default (strict-tier standard) => unreliable verdict is rejected.
    assert not evaluate_confinement_gate(metrics).accepted
    assert evaluate_confinement_gate(
        metrics, ConfinementGateConfig(require_classifiability=False)
    ).accepted


def test_classifiability_above_floor_accepts():
    metrics = ConfinementMetrics(
        survived_lines=50, total_lines=50, beta=0.0, classifiable_fraction=0.95,
    )
    assert evaluate_confinement_gate(metrics).accepted


def test_classifiability_not_evaluated_is_not_applicable():
    metrics = ConfinementMetrics(survived_lines=50, total_lines=50, beta=0.0)
    verdict = evaluate_confinement_gate(metrics)  # classifiable_fraction None
    c = _criterion(verdict, CRITERION_CLASSIFIABILITY)
    assert c.applicable is False
    assert verdict.accepted  # required-but-not-applicable does not block


def test_required_but_not_applicable_criterion_does_not_block():
    # require magnetic well, but <3 surfaces (well=None) -> N/A, must not reject.
    metrics = ConfinementMetrics(
        survived_lines=50, total_lines=50, beta=0.0, magnetic_well=None,
    )
    verdict = evaluate_confinement_gate(
        metrics, ConfinementGateConfig(require_magnetic_well=True)
    )
    assert verdict.accepted


def test_required_shear_below_min_is_rejected():
    metrics = ConfinementMetrics(
        survived_lines=50, total_lines=50, beta=0.0, iota_shear=0.001,
    )
    verdict = evaluate_confinement_gate(
        metrics, ConfinementGateConfig(iota_shear_min=0.01, require_shear=True)
    )
    assert not verdict.accepted
    assert verdict.decisive_reason.startswith("REJECT: iota_shear")


def test_magnetic_well_sign_convention_negative_is_favorable():
    base = dict(survived_lines=50, total_lines=50, beta=0.0)
    favorable = ConfinementMetrics(**base, magnetic_well=-0.2)
    unfavorable = ConfinementMetrics(**base, magnetic_well=0.2)
    cfg = ConfinementGateConfig(require_magnetic_well=True)
    assert evaluate_confinement_gate(favorable, cfg).accepted
    assert not evaluate_confinement_gate(unfavorable, cfg).accepted


def test_verdict_to_dict_is_json_serializable_and_complete():
    import json

    metrics = ConfinementMetrics(
        survived_lines=47, total_lines=50, beta=0.0, qa_nonqs_ratio=5e-3,
    )
    payload = evaluate_confinement_gate(metrics).to_dict()
    json.dumps(payload)  # must not raise
    assert payload["accepted"] is True
    names = {c["name"] for c in payload["criteria"]}
    assert CRITERION_TOPOLOGY in names and CRITERION_MERCIER in names


def test_config_rejects_out_of_range_survival_fraction():
    with pytest.raises(ValueError):
        ConfinementGateConfig(min_survival_fraction=1.5)


# --- run_confinement_gate.resolve_beta (driver helper) -----------------------------------

def test_resolve_beta_override_wins(tmp_path):
    assert driver.resolve_beta(tmp_path, 0.03) == 0.03


def test_resolve_beta_missing_results_is_vacuum(tmp_path):
    assert driver.resolve_beta(tmp_path, None) == 0.0


def test_resolve_beta_zero_boozer_i_is_vacuum_without_note(tmp_path, capsys):
    (tmp_path / "results.json").write_text(json.dumps({"BOOZER_I": 0.0}))
    assert driver.resolve_beta(tmp_path, None) == 0.0
    assert "NOTE" not in capsys.readouterr().out


def test_resolve_beta_nonzero_boozer_i_defaults_zero_with_loud_note(tmp_path, capsys):
    # Finite current present, but the driver cannot reconstruct beta -> default 0 + warn,
    # never silently certify Mercier-N/A on a finite-current candidate.
    (tmp_path / "results.json").write_text(json.dumps({"BOOZER_I": 1.5}))
    assert driver.resolve_beta(tmp_path, None) == 0.0
    out = capsys.readouterr().out
    assert "NOTE" in out and "BOOZER_I" in out and "--beta" in out


# --- metrics_from_topology_verdict (the strict-tier adapter) -----------------------------

def test_adapter_prefers_surviving_seed_count_and_maps_islands():
    verdict = {
        "broken": False,
        "survival_fraction": 0.94,
        "surviving_seed_count": 47,
        "island_chain_count": 0,
    }
    m = metrics_from_topology_verdict(verdict, 50, qa_nonqs_ratio=5e-3, beta=0.0)
    assert m.survived_lines == 47
    assert m.total_lines == 50
    assert m.island_chains == 0
    assert m.qa_nonqs_ratio == 5e-3


def test_adapter_recovers_survived_from_fraction_when_count_absent():
    verdict = {"broken": False, "survival_fraction": 0.9}  # 0.9 * 50 = 45
    m = metrics_from_topology_verdict(verdict, 50)
    assert m.survived_lines == 45
    assert m.island_chains is None
    assert m.classifiable_fraction is None


def test_adapter_maps_classifiable_fraction_from_verdict():
    verdict = {
        "broken": False,
        "survival_fraction": 0.96,
        "surviving_seed_count": 48,
        "island_chain_count": 0,
        "classifiable_fraction": 0.92,
    }
    m = metrics_from_topology_verdict(verdict, 50)
    assert m.classifiable_fraction == 0.92
    assert m.island_chains == 0


def test_adapter_broken_topology_yields_failclosed_metrics():
    verdict = {"broken": True, "survival_fraction": 0.0}
    m = metrics_from_topology_verdict(verdict, 50)
    assert m.survived_lines is None
    # And the gate rejects it on topology.
    assert not evaluate_confinement_gate(m).accepted


def test_adapter_to_gate_endtoend_accept_in_vacuum():
    verdict = {
        "broken": False,
        "survival_fraction": 0.96,
        "surviving_seed_count": 48,
        "island_chain_count": 0,
    }
    m = metrics_from_topology_verdict(verdict, 50, qa_nonqs_ratio=3e-3, beta=0.0)
    result = evaluate_confinement_gate(m)
    assert result.accepted
    assert "ACCEPT" in result.decisive_reason


# --- #17: configured 0.0 frontier WBA floor is hard-clamped, not silently re-vacuated -----
# Fail-on-old: before the fix, resolve_frontier_invariant_torus_min(0.0, None) returned 0.0,
# which re-vacuates the binding invariant-torus certification gate (deficit = max(min-frac,0)
# == 0 for every config). The resolver must now clamp <=0 / non-finite up to the default with
# a loud warning, consistent with the calibrator's recommended-floor clamp.


@pytest.mark.parametrize("configured", [0.0, -0.1, -1.0])
def test_configured_nonpositive_floor_is_clamped_to_default_with_warning(configured):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_frontier_invariant_torus_min(configured, None)
    assert resolved == DEFAULT_FRONTIER_INVARIANT_TORUS_MIN
    assert resolved > 0.0  # the binding gate stays non-vacuous
    messages = [str(w.message) for w in caught]
    assert CLAMPED_VACUOUS_FRONTIER_INVARIANT_TORUS_WARNING in messages


def test_configured_nonfinite_floor_is_clamped_to_default_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_frontier_invariant_torus_min(float("nan"), None)
    assert resolved == DEFAULT_FRONTIER_INVARIANT_TORUS_MIN
    assert any(
        str(w.message) == CLAMPED_VACUOUS_FRONTIER_INVARIANT_TORUS_WARNING
        for w in caught
    )


def test_positive_configured_floor_passes_through_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_frontier_invariant_torus_min(0.42, None)
    assert resolved == 0.42
    assert caught == []


def test_unconfigured_floor_returns_default_without_clamp_warning():
    # The clean default path must stay silent: only an explicitly-vacuated config warns.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_frontier_invariant_torus_min(None, None)
    assert resolved == DEFAULT_FRONTIER_INVARIANT_TORUS_MIN
    assert caught == []


def test_zero_kam_min_alias_and_env_are_also_clamped():
    # The legacy frontier_kam_min positional and the env-var aliases route through the same
    # clamp, so none of them can silently re-vacuate the gate.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        assert (
            resolve_frontier_invariant_torus_min(None, 0.0)
            == DEFAULT_FRONTIER_INVARIANT_TORUS_MIN
        )
        assert (
            resolve_frontier_invariant_torus_min(
                None, None, environment={"FRONTIER_INVARIANT_TORUS_MIN": "0.0"}
            )
            == DEFAULT_FRONTIER_INVARIANT_TORUS_MIN
        )
        assert (
            resolve_frontier_invariant_torus_min(
                None, None, environment={"FRONTIER_KAM_MIN": "-0.2"}
            )
            == DEFAULT_FRONTIER_INVARIANT_TORUS_MIN
        )


# --- #1: SSOT strict-tier + gate composition (certify_confinement) ------------------------
# certify_confinement is the ONE composition shared by the offline CLI and the in-process
# solver finalization. The strict tier (50 lines / tmax 7000) is too expensive for a unit
# test, so the strict-tier runner is stubbed; the test asserts the OBSERVABLE composition
# behavior: the gate runs on the stubbed verdict, writes confinement_verdict.json, and the
# accept/reject verdict propagates fail-closed.


def _install_fake_strict_tier(monkeypatch, topology_verdict, *, passed):
    fake = types.ModuleType("run_topology_fidelity_ladder")

    def evaluate_case(_run_dir):
        return {
            "field_label": "opt",
            "strict": {
                "passed": passed,
                "topology_verdict": topology_verdict,
            },
        }

    fake.evaluate_case = evaluate_case
    monkeypatch.setitem(sys.modules, "run_topology_fidelity_ladder", fake)
    return fake


_STRONG_STRICT_VERDICT = {
    "broken": False,
    "survival_fraction": 0.96,
    "surviving_seed_count": 48,
    "island_chain_count": 0,
    "classifiable_fraction": 0.95,
    "passed": True,
}
_BROKEN_STRICT_VERDICT = {"broken": True, "survival_fraction": 0.0}


def test_certify_confinement_composes_strict_tier_and_writes_verdict(monkeypatch, tmp_path):
    _install_fake_strict_tier(monkeypatch, _STRONG_STRICT_VERDICT, passed=True)
    payload = certify_confinement(
        tmp_path,
        config=ConfinementGateConfig(),
        beta=0.0,
        qa_nonqs_ratio=3e-3,
        iota_shear=0.05,
    )
    assert payload["confinement_verdict"]["accepted"] is True
    assert payload["strict_topology_passed"] is True
    out = tmp_path / "confinement_verdict.json"
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk["confinement_verdict"]["accepted"] is True


def test_certify_confinement_fails_closed_on_broken_topology(monkeypatch, tmp_path):
    _install_fake_strict_tier(monkeypatch, _BROKEN_STRICT_VERDICT, passed=False)
    payload = certify_confinement(tmp_path, config=ConfinementGateConfig())
    verdict = payload["confinement_verdict"]
    assert verdict["accepted"] is False
    assert verdict["decisive_reason"].startswith("REJECT: topology")


def test_cli_certify_shares_certify_confinement_composition(monkeypatch, tmp_path):
    # The CLI certify() must route through the SAME composition (no duplicate wiring): stub
    # evaluate_case once and assert the CLI path produces the identical accept verdict + file.
    _install_fake_strict_tier(monkeypatch, _STRONG_STRICT_VERDICT, passed=True)
    namespace = types.SimpleNamespace(
        beta=0.0,
        qa_nonqs=None,
        magnetic_well=None,
        iota_shear=None,
        mercier_dmerc=None,
        min_survival=0.90,
        min_classifiable=0.90,
        advisory_islands=False,
        advisory_classifiability=False,
        require_qa=False,
        require_magnetic_well=False,
        require_shear=False,
        qa_nonqs_max=1.0e-2,
        iota_shear_min=0.0,
        # Mirror the real parser default (--converse-kam-gate-interval 0 = the
        # advisory converse-KAM diagnostic OFF); converse_kam_config_from_args
        # reads this attribute, so the hand-built namespace must carry it.
        converse_kam_gate_interval=0,
    )
    payload = driver.certify(str(tmp_path), namespace)
    assert payload["confinement_verdict"]["accepted"] is True
    assert (tmp_path / "confinement_verdict.json").exists()
