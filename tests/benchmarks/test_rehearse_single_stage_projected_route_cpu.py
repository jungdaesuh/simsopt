"""Gates on the projected route's bounded CPU rehearsal.

The expensive test in this file launches the rehearsal's real entry path in a
subprocess, with no module constant monkeypatched, and every cheap test that
follows reads the artifact it published.  That shape is deliberate: the
predecessor route spent a one-shot campaign root on a ``NameError`` in its
launcher's first phase while a fifty-nine-test suite was green, because the
suite imported the module and monkeypatched the very constant that was
missing.  A rehearsal whose own entry path is never executed as launched
inherits that failure mode.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import ModuleType, SimpleNamespace

import benchmarks.rehearse_single_stage_projected_route_cpu as rehearsal
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks.single_stage_fullspace_snapshot import canonical_json_bytes
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
    DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    DIAG5_EXECUTION_SOURCE_ENTRY_COUNT,
)
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256

REPOSITORY = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- entry path


@pytest.fixture(scope="module")
def published_rehearsal(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real rehearsal, launched the way the certification lane launches it.

    Nothing here is stubbed: the subprocess builds the campaign problem, binds
    it by its observables, lowers the route, compiles the engine, solves three
    attempts, publishes a sealed artifact and re-validates it.  Every later
    assertion in this file reads that artifact rather than re-running the
    chain.
    """

    output_root = tmp_path_factory.mktemp("rehearsal") / "projected-route-rehearsal"
    environment = {
        **os.environ,
        **rehearsal.REQUIRED_ENVIRONMENT,
        "PYTHONPATH": os.pathsep.join((str(REPOSITORY / "src"), str(REPOSITORY))),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(REPOSITORY / "benchmarks/rehearse_single_stage_projected_route_cpu.py"),
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        check=False,
        cwd=REPOSITORY,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, (
        f"rehearsal entry path failed\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert output_root.is_dir()
    return output_root


def _evidence(root: Path) -> dict:
    return json.loads((root / rehearsal.EVIDENCE_FILENAME).read_bytes())


def test_published_rehearsal_revalidates_from_its_sealed_bytes(
    published_rehearsal: Path,
) -> None:
    """A third party can re-derive every claim the artifact makes."""

    evidence = rehearsal.validate_rehearsal_artifact(published_rehearsal)
    assert evidence["schema_version"] == rehearsal.REHEARSAL_SCHEMA_VERSION
    assert evidence["route"] == rehearsal.REHEARSAL_ROUTE


def test_rehearsal_exercises_the_whole_chain(published_rehearsal: Path) -> None:
    """Every phase a certified run performs left evidence in the artifact."""

    evidence = _evidence(published_rehearsal)
    assert evidence["environment"] == dict(rehearsal.REQUIRED_ENVIRONMENT)
    assert evidence["execution_sources"]["manifest"]["entry_count"] == (
        DIAG5_EXECUTION_SOURCE_ENTRY_COUNT
    )
    assert evidence["problem_identity"]["bound"] is True
    assert evidence["lowering_pre_gate"]["budget_independent"] is True
    assert evidence["solve"]["iterations_run"] == rehearsal.REHEARSAL_MAXIMUM_ITERATIONS
    assert evidence["endpoint_agreement"]["terminal_state_sha256"]
    assert set(evidence["timing_seconds"]) == {
        "bootstrap",
        "chain_wall",
        "engine_compile",
        "engine_solve",
        "engine_wall",
        "lowering_pre_gate",
        "problem_identity",
    }


def test_rehearsal_binds_the_route_module_it_certifies(
    published_rehearsal: Path,
) -> None:
    """The engine under certification is named, hashed and manifest-bound.

    An artifact that cannot say which bytes of the optimizer ran is not
    evidence about the route; it is evidence about a process.
    """

    evidence = _evidence(published_rehearsal)
    bound = {
        entry["relative_path"]: entry
        for entry in evidence["execution_sources"]["bound_modules"]
    }
    engine = "src/simsopt_jax/geo/optimizers/projected_lbfgs.py"
    assert engine in bound
    assert bound[engine]["sha256"] == hashlib.sha256(
        (REPOSITORY / engine).read_bytes()
    ).hexdigest()
    # The virtualenv lives inside this checkout, so thousands of its modules
    # resolve under the repository root.  They are counted, not listed: an
    # evidence section where the engine hides among two thousand site-packages
    # files is not evidence a reviewer can read.
    installation = evidence["execution_sources"]["interpreter_installation_modules"]
    assert installation["count"] > len(bound)
    assert all(root.startswith(".") for root in installation["roots"])


def test_endpoint_ledger_reports_physics_terms_beside_native(
    published_rehearsal: Path,
) -> None:
    """Quality parity is defined per term, and G is reported, never gated.

    A total objective cannot distinguish "reached the same physics" from
    "reached the same scalar".  The non-QS term is a field-scale-invariant
    ratio and nothing in the shared objective pins the net poloidal current, so
    G is a flat valley direction along which distinct equal-quality minima
    exist; gating it would manufacture a false reject.
    """

    ledger = _evidence(published_rehearsal)["endpoint_ledger"]
    assert ledger["pinned_quality_terms"] == list(
        rehearsal.PINNED_ENDPOINT_QUALITY_TERMS
    )
    assert ledger["informational_observables"] == ["observable.G", "state.G"]
    assert not set(ledger["informational_observables"]) & set(
        ledger["pinned_quality_terms"]
    )
    for term in rehearsal.PINNED_ENDPOINT_QUALITY_TERMS:
        assert term in ledger["terminal"]
        assert term in ledger["native"]
    # The native side is the sealed C++ endpoint re-evaluated through THIS
    # repository's objective, so its own physics must come out on spec.
    assert abs(ledger["native"]["constraint.volume"]) < 1.0e-10
    assert ledger["native"]["observable.boozer_residual_rms"] < 1.0e-10
    assert ledger["gated_at_this_budget"] is False


def test_endpoint_ledger_scope_cannot_be_restated_by_an_artifact(
    tmp_path: Path,
) -> None:
    """An artifact may not narrow the terms it is then judged on."""

    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    evidence = _minimal_evidence(coordinates)
    evidence["endpoint_ledger"]["pinned_quality_terms"] = ["weighted_total"]
    published = _publish_minimal(tmp_path, evidence)
    with pytest.raises(rehearsal.RehearsalError, match="ledger scope differs"):
        rehearsal.validate_rehearsal_artifact(published)


def test_rehearsal_claims_no_science(published_rehearsal: Path) -> None:
    """Three attempts cannot reach the target and the receipt must say so."""

    evidence = _evidence(published_rehearsal)
    assert evidence["quality_claim"] == "NOT_CLAIMED_AT_REHEARSAL_BUDGET"
    assert evidence["solve"]["terminal_objective"] > evidence["native_reference"][
        "target_objective"
    ]
    assert evidence["timing_boundary"] == "engine_compile_plus_solve"


def test_published_margin_is_the_smallest_recorded_scale_over_the_floor(
    published_rehearsal: Path,
) -> None:
    """The collapse-proximity margin is derived from the rows it is published with.

    It is telemetry, not a control input: a reader must be able to recompute it
    from the same artifact rather than trust a number the run asserted.
    """

    evidence = _evidence(published_rehearsal)
    solve = evidence["solve"]
    smallest = min(row["step_scale"] for row in solve["rows"])
    assert solve["collapse_proximity_margin"] == (
        smallest / rehearsal.CERTIFIED_ROUTE_OPTIONS.minimum_step_scale
    )
    # The rehearsal budget places every step it tries, so it is nowhere near
    # the floor; the metric earns its keep on runs that are.
    assert solve["collapse_proximity_margin"] > 1.0


def test_collapse_proximity_margin_names_the_collapse_it_measures() -> None:
    """Below one is the collapse itself, and an empty run has no margin.

    The sub-one case is the A100 arm that terminated in ``LINE_SEARCH_COLLAPSE``
    at a step scale of 9.456e-7 against the route's 1e-6 floor: the margin has to
    report that arm below one while every arm that latched stays above it.
    """

    options = rehearsal.CERTIFIED_ROUTE_OPTIONS
    latched = SimpleNamespace(
        iterations=tuple(
            SimpleNamespace(step_scale=scale) for scale in (1.0, 0.125, 0.5)
        )
    )
    assert rehearsal.collapse_proximity_margin(latched, options) == 125000.0

    collapsed = SimpleNamespace(
        iterations=(
            SimpleNamespace(step_scale=1.0),
            SimpleNamespace(step_scale=9.456e-7),
        )
    )
    assert rehearsal.collapse_proximity_margin(collapsed, options) < 1.0

    assert np.isnan(
        rehearsal.collapse_proximity_margin(SimpleNamespace(iterations=()), options)
    )


def test_rehearsal_never_binds_identity_to_the_problem_sha(
    published_rehearsal: Path,
) -> None:
    """The sha is provenance, not identity: it differs run to run on one GPU."""

    identity = _evidence(published_rehearsal)["problem_identity"]
    assert identity["sha_is_binding"] is False
    assert len(identity["recorded_problem_sha256"]) == 64
    assert identity["relative_difference"]["objective"] <= 1.0e-10


def test_rehearsal_differs_from_the_certified_run_only_in_its_budget(
    published_rehearsal: Path,
) -> None:
    """The published delta is the whole difference between the two runs."""

    evidence = _evidence(published_rehearsal)
    assert evidence["certified_options_delta"] == {
        "maximum_iterations": rehearsal.REHEARSAL_MAXIMUM_ITERATIONS
    }


# ------------------------------------------------------------- configuration


def test_rehearsal_options_replace_only_the_attempt_budget() -> None:
    """A rehearsal cannot rehearse a route other than the certified one."""

    options = rehearsal.rehearsal_options(3)
    differing = [
        field.name
        for field in fields(rehearsal.CERTIFIED_ROUTE_OPTIONS)
        if getattr(options, field.name)
        != getattr(rehearsal.CERTIFIED_ROUTE_OPTIONS, field.name)
    ]
    assert differing == ["maximum_iterations"]
    assert options.maximum_iterations == 3


def test_certified_options_are_the_configuration_the_latches_used() -> None:
    """The frozen configuration is the one the two 5090 latches were taken at.

    EVERY field is pinned, not only the ones the constructor names.  The
    unnamed fields take optimizer defaults, so a default changed elsewhere in
    the repository would silently redefine the certified route while the
    rehearsal-versus-certified delta gate stayed green.  The field this matters
    most for is ``projector_tangency_tolerance``: at zero the carried projector
    is never refreshed on tangency, which is the known, accepted mechanism
    behind the A100 no-latch arm (true tangency 12.5 at ages 2-3).  Certifying
    this configuration means certifying that value.
    """

    options = rehearsal.CERTIFIED_ROUTE_OPTIONS

    assert {field.name: getattr(options, field.name) for field in fields(options)} == {
        "maximum_iterations": rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
        "memory": 10,
        "feasibility_tolerance": 1.0e-10,
        "maximum_retraction_corrections": 8,
        "maximum_step_norm": 0.25,
        "armijo_coefficient": 1.0e-4,
        "backtracking_factor": 0.5,
        "maximum_line_search_trials": 24,
        "minimum_step_scale": 1.0e-6,
        "objective_target": rehearsal.NATIVE_TARGET_OBJECTIVE,
        "gauss_newton": False,
        "gauss_newton_levenberg_relative": 1.0e-12,
        "model_step_rescue_scale": 1.0e-2,
        "lagrangian_newton": True,
        "newton_cg_maximum_iterations": 40,
        "newton_cg_forcing_maximum": 0.5,
        "hybrid_plateau_window": 0,
        "hybrid_plateau_factor": 0.995,
        "dense_curvature": False,
        "vector_transport": False,
        "frozen_projector_line_search": True,
        "projector_refresh_period": 4,
        "projector_tangency_tolerance": 0.0,
        "newton_tangent_fraction_threshold": 0.25,
    }


def test_environment_gate_names_the_variable_that_differs() -> None:
    """A silent backend swap is the failure this gate exists to prevent."""

    complete = dict(rehearsal.REQUIRED_ENVIRONMENT)
    assert rehearsal.validate_environment(complete) == complete
    with pytest.raises(rehearsal.RehearsalError, match="JAX_ENABLE_X64"):
        rehearsal.validate_environment({**complete, "JAX_ENABLE_X64": "false"})
    with pytest.raises(rehearsal.RehearsalError, match="JAX_PLATFORMS"):
        rehearsal.validate_environment(
            {name: value for name, value in complete.items() if name != "JAX_PLATFORMS"}
        )


def test_nonfinite_evidence_scalars_become_null() -> None:
    """Canonical JSON refuses NaN, and the protocol writes null instead."""

    assert rehearsal.json_scalar(float("nan")) is None
    assert rehearsal.json_scalar(float("inf")) is None
    assert rehearsal.json_scalar(True) is True
    assert rehearsal.json_scalar(3) == 3
    assert rehearsal.json_scalar((1.0, float("nan"))) == [1.0, None]


# ------------------------------------------------------- execution-source gate


def _synthetic_repository(root: Path, probe_source: bytes) -> Path:
    """A repository whose manifest admits exactly one real file.

    The remaining entries are fabricated paths that no module resolves to, so
    the frozen entry count is satisfied without the test asserting a different
    count than the authority freezes.
    """

    (root / "src").mkdir(parents=True)
    (root / "benchmarks").mkdir(parents=True)
    (root / "src/probe_module.py").write_bytes(probe_source)
    entries = {
        "src/probe_module.py": {
            "sha256": hashlib.sha256(probe_source).hexdigest(),
            "size_bytes": len(probe_source),
        }
    }
    for index in range(DIAG5_EXECUTION_SOURCE_ENTRY_COUNT - 1):
        entries[f"src/unreached_{index:05d}.py"] = {"sha256": "0" * 64, "size_bytes": 0}
    document = {
        "entries": entries,
        "entries_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
        "schema_version": "single-stage-neq-gntr3-execution-source-authority-v1",
    }
    manifest = root / "benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json"
    manifest.write_bytes(canonical_json_bytes(document))
    return root


def _register_probe_module(
    monkeypatch: pytest.MonkeyPatch, path: Path, name: str = "rehearsal_probe_module"
) -> None:
    module = ModuleType(name)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, module)


def test_execution_source_binding_accepts_the_live_repository() -> None:
    """The repository as committed satisfies its own execution-source gate."""

    binding = rehearsal.bind_execution_sources(REPOSITORY)
    assert binding["manifest"]["entry_count"] == DIAG5_EXECUTION_SOURCE_ENTRY_COUNT
    assert any(
        entry["relative_path"]
        == "benchmarks/rehearse_single_stage_projected_route_cpu.py"
        for entry in binding["bound_modules"]
    )


def test_execution_source_binding_rejects_drifted_module_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An import that resolves to bytes the manifest never described fails.

    This is the shape of the editable-install escape that spent a one-shot
    root: the manifest was intact and the module executing was not the one it
    described.
    """

    root = _synthetic_repository(tmp_path / "repo", b"probe = 1\n")
    _register_probe_module(monkeypatch, root / "src/probe_module.py")
    assert rehearsal.bind_execution_sources(root)["bound_modules"]

    (root / "src/probe_module.py").write_bytes(b"probe = 2\n")
    with pytest.raises(rehearsal.RehearsalError, match="bytes the manifest"):
        rehearsal.bind_execution_sources(root)


def test_execution_source_binding_rejects_an_unmanifested_broad_root_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A new execution source that never entered the manifest fails closed."""

    root = _synthetic_repository(tmp_path / "repo", b"probe = 1\n")
    _register_probe_module(monkeypatch, root / "src/probe_module.py")
    _register_probe_module(
        monkeypatch, root / "benchmarks/newcomer.py", name="rehearsal_probe_newcomer"
    )
    with pytest.raises(rehearsal.RehearsalError, match="which the manifest omits"):
        rehearsal.bind_execution_sources(root)


def test_execution_source_binding_refuses_a_distribution_module_outside_the_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The gate hashes what it can see; what it cannot see must not pass silently.

    ``_imported_repository_modules`` returned only modules under the checkout,
    so a ``simsopt*`` module resolved elsewhere was neither hashed nor recorded
    -- invisible -- and the sole liveness check is satisfied by ``benchmarks/*``
    alone.  Today that is closed by where the editable install happens to point;
    launching from a moved or copied checkout would put the baked-in redirect
    outside the tree and silence the gate on exactly the class it exists for.
    """

    rehearsal.refuse_redirected_distribution_modules()

    elsewhere = tmp_path / "elsewhere/simsopt_jax/redirected.py"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_bytes(b"redirected = 1\n")
    _register_probe_module(monkeypatch, elsewhere, name="simsopt_jax.redirected")
    with pytest.raises(rehearsal.RehearsalError, match="which is outside"):
        rehearsal.refuse_redirected_distribution_modules()
    with pytest.raises(rehearsal.RehearsalError, match="which is outside"):
        rehearsal.bind_execution_sources(REPOSITORY)


def test_execution_source_manifest_gate_rejects_a_stale_digest(
    tmp_path: Path,
) -> None:
    """The manifest's own digest is recomputed, never trusted."""

    root = _synthetic_repository(tmp_path / "repo", b"probe = 1\n")
    manifest = (
        root
        / "benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json"
    )
    document = json.loads(manifest.read_bytes())
    document["entries_sha256"] = "1" * 64
    manifest.write_bytes(canonical_json_bytes(document))
    with pytest.raises(rehearsal.RehearsalError, match="entries digest differs"):
        rehearsal.load_execution_source_manifest(root)


def test_execution_source_manifest_gate_rejects_a_count_the_authority_denies(
    tmp_path: Path,
) -> None:
    """A manifest and an authority that disagree cannot both be believed."""

    root = _synthetic_repository(tmp_path / "repo", b"probe = 1\n")
    manifest = (
        root
        / "benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json"
    )
    document = json.loads(manifest.read_bytes())
    document["entries"].pop("src/unreached_00000.py")
    document["entries_sha256"] = hashlib.sha256(
        canonical_json_bytes(document["entries"])
    ).hexdigest()
    manifest.write_bytes(canonical_json_bytes(document))
    with pytest.raises(rehearsal.RehearsalError, match="authority freezes"):
        rehearsal.load_execution_source_manifest(root)


# ------------------------------------------------- pinned bands vs the evidence

# The per-term endpoint ledger of the two banked RTX 5090 latches and of the
# sealed native endpoint, all three evaluated through THIS repository's
# objective on CPU at commit a3ea4983d (bootstrap 7.3 s, ledger 1.8 s).  Pinned
# as literals so the bands are checked against the campaign's own measured
# evidence without the suite reaching into a sealed campaign root: the numbers
# below ARE the derivation recorded in plan section 12.8.
BANKED_ENDPOINT_LEDGER: dict[str, dict[str, float]] = {
    "native": {
        "constraint.boozer|inf": 3.209172479736042e-14,
        "constraint.volume": -2.332159450539864e-18,
        "observable.G": 13.88747208750538,
        "observable.iota": -0.4062027259574152,
        "observable.major_radius": 1.467443804809453,
        "observable.non_qs_ratio": 4.480897876285335e-08,
        "observable.total_length": 20.98916289206094,
        "observable.volume": -0.2904457582995848,
        "raw.non_qs": 4.480897876285335e-08,
        "raw.residual": 3.739666640426295e-29,
        "state.G": 13.88747208750538,
        "weighted_total": 4.482224653311689e-08,
    },
    "Q1": {
        "constraint.boozer|inf": 5.815039065069709e-13,
        "constraint.volume": -2.265665533787766e-16,
        "observable.G": 13.77850219496056,
        "observable.iota": -0.4062026361994761,
        "observable.major_radius": 1.46743558397157,
        "observable.non_qs_ratio": 4.441225770763212e-08,
        "observable.total_length": 20.98911498799002,
        "observable.volume": -0.290445758299585,
        "raw.non_qs": 4.441225770763212e-08,
        "raw.residual": 1.775213079010563e-26,
        "state.G": 13.77850219496056,
        "weighted_total": 4.444127944835292e-08,
    },
    "Q2": {
        "constraint.boozer|inf": 3.785997957504678e-12,
        "constraint.volume": -1.459855719984079e-15,
        "observable.G": 13.7578227650719,
        "observable.iota": -0.4062043201689589,
        "observable.major_radius": 1.467445317012639,
        "observable.non_qs_ratio": 4.477005226987877e-08,
        "observable.total_length": 20.98916226534961,
        "observable.volume": -0.2904457582995862,
        "raw.non_qs": 4.477005226987877e-08,
        "raw.residual": 3.910766621870086e-25,
        "state.G": 13.7578227650719,
        "weighted_total": 4.47825530664393e-08,
    },
}


def _banked_ledger(arm: str) -> dict[str, dict[str, float]]:
    return {
        "terminal": BANKED_ENDPOINT_LEDGER[arm],
        "native": BANKED_ENDPOINT_LEDGER["native"],
    }


@pytest.mark.parametrize("arm", ["Q1", "Q2"])
def test_the_pinned_bands_admit_both_banked_5090_latches(arm: str) -> None:
    """The gate must not refuse the very evidence the campaign banked.

    This is the V260 shell gate and the SQP rho floor a third time: a band
    drawn tighter than the spread the shared objective leaves free refuses
    legitimate minima.  The root is the pinned-term gate's FIRST execution
    against real endpoint numbers, and section 12.1 bars an automatic successor,
    so it is checked here against the sealed arms instead of there.
    """

    verdict = rehearsal.gate_endpoint_ledger(_banked_ledger(arm))
    assert verdict["failed_terms"] == []
    assert verdict["passed"] is True


@pytest.mark.parametrize(
    ("arm", "term", "measured"),
    [
        ("Q1", "observable.major_radius", 5.602e-06),
        ("Q2", "observable.iota", 3.925e-06),
        ("Q2", "observable.major_radius", 1.031e-06),
    ],
)
def test_the_predecessor_1e_6_bands_refused_the_banked_latches(
    arm: str, term: str, measured: float
) -> None:
    """The regression this remediation exists for, stated as a number.

    Every geometry observable was gated at 1e-6 relative on the strength of a
    prose claim that the banked endpoints agreed "to 1e-6 relative or better".
    Measured, they do not: the gate the root would have run refuses Q2 on iota
    and BOTH arms on major radius.
    """

    ledger = _banked_ledger(arm)
    relative = abs(
        ledger["terminal"][term] - ledger["native"][term]
    ) / abs(ledger["native"][term])
    assert relative == pytest.approx(measured, rel=1.0e-3)
    assert relative > 1.0e-6
    comparison, band = rehearsal.PINNED_ENDPOINT_QUALITY_GATES[term]
    assert comparison == "relative"
    # At least a decade of margin over the worst measurement, and still inside
    # the freedom two equal-quality minima of the shared objective may take.
    assert band >= 10.0 * relative
    assert band < rehearsal.EQUAL_MINIMA_PENALTY_SPREAD / abs(ledger["native"][term])


def test_the_weighted_total_band_cannot_refuse_an_attempt_that_latched() -> None:
    """A latch passes ``weighted_total`` by construction, not by luck.

    A latching attempt's objective is at or below ``NATIVE_TARGET_OBJECTIVE``,
    and the native endpoint re-evaluated through this repository's objective
    lands 2.1e-08 relative BELOW that literal -- cross-executable ULP, the band
    class plan section 5 exists for.  The one-sided 1e-6 band exceeds that gap
    by nearly two decades, so no latch can be refused on the total.
    """

    native_total = BANKED_ENDPOINT_LEDGER["native"]["weighted_total"]
    gap = (rehearsal.NATIVE_TARGET_OBJECTIVE - native_total) / abs(native_total)
    assert 0.0 < gap < 1.0e-7
    comparison, band = rehearsal.PINNED_ENDPOINT_QUALITY_GATES["weighted_total"]
    assert comparison == "not_worse"
    assert band > 10.0 * gap
    at_the_target = {
        "terminal": {
            **BANKED_ENDPOINT_LEDGER["native"],
            "weighted_total": rehearsal.NATIVE_TARGET_OBJECTIVE,
        },
        "native": BANKED_ENDPOINT_LEDGER["native"],
    }
    assert rehearsal.gate_endpoint_ledger(at_the_target)["passed"] is True


def test_a_non_latching_attempt_would_fail_the_gate_that_no_longer_binds_it() -> None:
    """Why the ledger gate binds the latch and nothing else.

    The A100 no-latch arm terminated at Phi 8.96e-8, twice the target, so its
    ``weighted_total`` excess is ~1.0 against a 1e-6 band: gated on every
    certified-budget attempt, the first stochastic miss publishes
    ``GATE_REFUSED:endpoint_ledger`` and ends the protocol after one of three
    attempts.
    """

    missed = {
        "terminal": {**BANKED_ENDPOINT_LEDGER["native"], "weighted_total": 8.96e-08},
        "native": BANKED_ENDPOINT_LEDGER["native"],
    }
    verdict = rehearsal.gate_endpoint_ledger(missed)
    assert verdict["failed_terms"] == ["weighted_total"]
    assert verdict["terms"]["weighted_total"]["measured"] == pytest.approx(1.0, rel=0.1)
    assert not rehearsal.endpoint_ledger_is_gated(
        iterations=rehearsal.CERTIFIED_MAXIMUM_ITERATIONS, latched=False
    )


def test_the_gated_ledger_branch_executes_on_real_physics() -> None:
    """The certified-budget branch, run once against the real objective.

    No lane could reach it: the rehearsal budget switches it off, the GPU smoke
    runs bounded, and every other test drives ``gate_endpoint_ledger`` as a pure
    function on a synthetic fixture -- so the root was to be its first
    execution.  Comparing the native endpoint against itself exercises the whole
    path (bootstrap, two standalone evaluations, the verdict build) with a
    verdict whose expected value cannot be argued about.
    """

    case = rehearsal.BoundCase()
    ledger = rehearsal.build_endpoint_ledger(
        case, SimpleNamespace(coordinates=case.native_endpoint), gated=True
    )
    assert ledger["gated_at_this_budget"] is True
    assert ledger["pinned_term_gate"]["failed_terms"] == []
    assert ledger["pinned_term_gate"]["passed"] is True
    assert json.loads(canonical_json_bytes(ledger)) == ledger
    # The rows the bands were derived from, reproduced by the live objective.
    for term, expected in BANKED_ENDPOINT_LEDGER["native"].items():
        assert ledger["native"][term] == pytest.approx(expected, rel=1.0e-9, abs=1.0e-30)


# ------------------------------------------------- native reference custody


def test_the_native_endpoint_reference_is_pinned_by_both_of_its_digests() -> None:
    """The reference side of the quality gate is the one unsealed input.

    It lives outside the repository, outside the execution-source manifest and
    outside the GPU root's source snapshot, in a ``.partial-`` staging directory
    of a publication that never completed -- so replacing it silently redefines
    what quality parity means.  Nothing compared its digest to anything.  The
    campaign's convention names the file by the CONTENT digest
    (``arrays/{content_sha256}.npy``), which is not the file digest, so both are
    pinned and both are checked.
    """

    state = rehearsal.load_native_endpoint_state()
    assert state.values.shape == (716,)
    assert state.values.dtype == np.float64
    assert state.file_sha256 == rehearsal.NATIVE_ENDPOINT_STATE_FILE_SHA256
    assert state.content_sha256 == rehearsal.NATIVE_ENDPOINT_STATE_CONTENT_SHA256
    assert rehearsal.NATIVE_ENDPOINT_STATE_PATH.stem == state.content_sha256
    assert state.file_sha256 != state.content_sha256
    assert (
        hashlib.sha256(
            np.ascontiguousarray(state.values, dtype=np.float64).tobytes(order="C")
        ).hexdigest()
        == state.content_sha256
    )
    assert (
        hashlib.sha256(rehearsal.NATIVE_ENDPOINT_STATE_PATH.read_bytes()).hexdigest()
        == state.file_sha256
    )


def test_a_substituted_native_endpoint_reference_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Altered bytes at the pinned path stop the chain instead of redefining it."""

    substitute = tmp_path / f"{rehearsal.NATIVE_ENDPOINT_STATE_CONTENT_SHA256}.npy"
    with substitute.open("wb") as stream:
        np.save(stream, np.zeros((716,), dtype=np.float64), allow_pickle=False)
    monkeypatch.setattr(rehearsal, "NATIVE_ENDPOINT_STATE_PATH", substitute)
    with pytest.raises(rehearsal.RehearsalError, match="not the pinned"):
        rehearsal.load_native_endpoint_state()


def test_the_published_rehearsal_records_both_native_reference_digests(
    published_rehearsal: Path,
) -> None:
    """A reader of the sealed bytes can re-identify the reference it used."""

    ledger = _evidence(published_rehearsal)["endpoint_ledger"]
    assert ledger["native_state_sha256"] == (
        rehearsal.NATIVE_ENDPOINT_STATE_FILE_SHA256
    )
    assert ledger["native_state_content_sha256"] == (
        rehearsal.NATIVE_ENDPOINT_STATE_CONTENT_SHA256
    )
    assert ledger["native_state_relative_path"] == (
        f"{rehearsal.NATIVE_ENDPOINT_STATE_CONTENT_SHA256}.npy"
    )


# ------------------------------------------------------------ identity gate


def _stub_case(**observables: float) -> SimpleNamespace:
    point = SimpleNamespace(**observables)
    return SimpleNamespace(
        evaluate_point=lambda coordinates: point,
        start=jnp.zeros((1,), dtype=jnp.float64),
        problem_sha256="a" * 64,
        bootstrap_sha256="b" * 64,
    )


def test_problem_identity_admits_the_measured_backend_spread() -> None:
    """Observables that differ by fp64 rounding still name the same problem.

    A CPU run of this problem reproduces the frozen gradient norm to one ULP
    and the GPU runs to ~1e-14 relative; a gate tighter than that would reject
    the very evidence the campaign banked.
    """

    reference = dict(rehearsal.CPU_BOOTSTRAP_OBSERVABLES)
    evidence = rehearsal.bind_problem_identity(
        _stub_case(**{**reference, "gradient_norm": reference["gradient_norm"] * (1 + 1e-15)})
    )
    assert evidence["bound"] is True
    assert evidence["sha_is_binding"] is False


def test_problem_identity_refuses_a_different_problem() -> None:
    """A per-mille objective shift is a different problem, not rounding."""

    reference = dict(rehearsal.CPU_BOOTSTRAP_OBSERVABLES)
    with pytest.raises(rehearsal.RehearsalError, match="do not reproduce"):
        rehearsal.bind_problem_identity(
            _stub_case(**{**reference, "objective": reference["objective"] * 1.001})
        )


def test_problem_identity_refuses_an_infeasible_bootstrap() -> None:
    """A start point off the manifold cannot bind to an on-manifold campaign."""

    reference = dict(rehearsal.CPU_BOOTSTRAP_OBSERVABLES)
    with pytest.raises(rehearsal.RehearsalError, match="do not reproduce"):
        rehearsal.bind_problem_identity(
            _stub_case(**{**reference, "feasibility_inf": 1.0e-6})
        )


# ------------------------------------------------------ publication and gate


def _minimal_evidence(coordinates: jax.Array) -> dict:
    """A structurally complete receipt with no solve behind it.

    The publication and validation gates are about bytes, modes and recorded
    agreements, so they are exercised here without paying for a solve.
    """

    return {
        "schema_version": rehearsal.REHEARSAL_SCHEMA_VERSION,
        "route": rehearsal.REHEARSAL_ROUTE,
        "problem_identity": {"bound": True, "sha_is_binding": False},
        "lowering_pre_gate": {"budget_independent": True},
        "endpoint_ledger": {
            "pinned_quality_terms": list(rehearsal.PINNED_ENDPOINT_QUALITY_TERMS),
            "informational_observables": list(
                rehearsal.INFORMATIONAL_ENDPOINT_OBSERVABLES
            ),
            "gated_at_this_budget": False,
        },
        "endpoint_agreement": {
            "loop_terminal_objective": 1.0,
            "standalone_terminal_objective": 1.0 + 5.0e-16,
            "relative_tolerance": DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE,
            "absolute_floor": DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
            "terminal_state_sha256": exact_numeric_tree_sha256(coordinates),
        },
    }


def _publish_minimal(root: Path, evidence: dict) -> Path:
    staging = root / "staging"
    staging.mkdir(parents=True)
    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    return rehearsal.publish_rehearsal(
        staging, root / "final", evidence, np.asarray(coordinates, dtype=np.float64)
    )


def test_published_artifact_is_sealed_and_self_describing(tmp_path: Path) -> None:
    """Seal, manifest and re-validation agree on the published bytes."""

    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    published = _publish_minimal(tmp_path, _minimal_evidence(coordinates))

    assert stat.S_IMODE(published.stat().st_mode) == 0o555
    for path in published.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert rehearsal.validate_rehearsal_artifact(published)


def test_publication_refuses_to_replace_an_existing_final_root(
    tmp_path: Path,
) -> None:
    """Two rehearsals cannot silently overwrite one another's evidence."""

    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    _publish_minimal(tmp_path, _minimal_evidence(coordinates))
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(FileExistsError):
        rehearsal.publish_rehearsal(
            second,
            tmp_path / "final",
            _minimal_evidence(coordinates),
            np.asarray(coordinates, dtype=np.float64),
        )


def test_validation_rejects_an_endpoint_tolerance_the_campaign_never_froze(
    tmp_path: Path,
) -> None:
    """An artifact may not widen the band it is then judged against."""

    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    evidence = _minimal_evidence(coordinates)
    evidence["endpoint_agreement"]["relative_tolerance"] = 1.0e-3
    evidence["endpoint_agreement"]["standalone_terminal_objective"] = 1.0 + 1.0e-6
    published = _publish_minimal(tmp_path, evidence)
    with pytest.raises(rehearsal.RehearsalError, match="tolerances differ"):
        rehearsal.validate_rehearsal_artifact(published)


def test_validation_rejects_a_cross_executable_divergence_that_matters(
    tmp_path: Path,
) -> None:
    """ULP-scale disagreement passes; state-scale disagreement fails closed."""

    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    evidence = _minimal_evidence(coordinates)
    evidence["endpoint_agreement"]["standalone_terminal_objective"] = 1.0 + 1.0e-6
    published = _publish_minimal(tmp_path, evidence)
    with pytest.raises(ValueError, match="terminal objective differs"):
        rehearsal.validate_rehearsal_artifact(published)


def test_validation_rejects_a_terminal_state_that_was_swapped(
    tmp_path: Path,
) -> None:
    """The published state is bound to its hash EXACTLY: it is one copy."""

    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    evidence = _minimal_evidence(coordinates)
    evidence["endpoint_agreement"]["terminal_state_sha256"] = exact_numeric_tree_sha256(
        jnp.asarray([0.25, -0.5, 1.0 + 1.0e-15], dtype=jnp.float64)
    )
    published = _publish_minimal(tmp_path, evidence)
    with pytest.raises(rehearsal.RehearsalError, match="terminal state differs"):
        rehearsal.validate_rehearsal_artifact(published)


def test_validation_rejects_a_tampered_artifact_tree(tmp_path: Path) -> None:
    """The artifact manifest is recomputed from the tree, never trusted."""

    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    published = _publish_minimal(tmp_path, _minimal_evidence(coordinates))
    tampered = published / rehearsal.EVIDENCE_FILENAME
    tampered.chmod(0o644)
    tampered.write_bytes(tampered.read_bytes() + b" ")
    with pytest.raises(rehearsal.RehearsalError, match="manifest differs"):
        rehearsal.validate_rehearsal_artifact(published)


def test_validation_rejects_an_artifact_that_binds_to_the_problem_sha(
    tmp_path: Path,
) -> None:
    """Binding identity to an unstable sha is refused at validation too."""

    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    evidence = _minimal_evidence(coordinates)
    evidence["problem_identity"]["sha_is_binding"] = True
    published = _publish_minimal(tmp_path, evidence)
    with pytest.raises(rehearsal.RehearsalError, match="unstable sha"):
        rehearsal.validate_rehearsal_artifact(published)


def test_rehearsal_refuses_to_publish_over_an_existing_root(tmp_path: Path) -> None:
    """The chain claims its output root before it spends a second of compute."""

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(rehearsal.RehearsalError, match="already exists"):
        rehearsal.run_bounded_rehearsal(
            occupied, environment=dict(rehearsal.REQUIRED_ENVIRONMENT)
        )


def test_rehearsal_refuses_a_nonpositive_budget(tmp_path: Path) -> None:
    """A zero-attempt rehearsal would exercise no solve loop at all."""

    with pytest.raises(rehearsal.RehearsalError, match="budget must be positive"):
        rehearsal.run_bounded_rehearsal(
            tmp_path / "unused",
            iterations=0,
            environment=dict(rehearsal.REQUIRED_ENVIRONMENT),
        )


def test_cleanup_of_sealed_trees_needs_no_recursive_force(tmp_path: Path) -> None:
    """Sealed artifacts are 0555/0444, so removal must reopen them first.

    Recorded as a test because two sessions lost time to it: a sealed tree
    cannot be deleted until its modes are restored, and the campaign's cleanup
    path is a Python walk rather than a recursive-force shell glob.
    """

    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    published = _publish_minimal(tmp_path, _minimal_evidence(coordinates))

    # The seal is real: an ordinary removal cannot unlink through a 0555
    # directory, which is what turns a routine cleanup into a wedged session.
    with pytest.raises(PermissionError):
        shutil.rmtree(published)
    assert published.is_dir()

    for path in (published, *published.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    shutil.rmtree(published)
    assert not published.exists()
