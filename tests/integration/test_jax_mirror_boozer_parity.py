"""Exact parity for the ``2_Intermediate/boozer.py`` mirror."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.parity.arbiter import (
    LaneObservation,
    upstream_branch_representatives,
    upstream_end_state_matches,
)
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases.native_boozer import (
    FIRST_STAGE_STATUS_CONVENTION_BY_DRIVER,
    JAX_DRIVER,
    NATIVE_DRIVER,
    STOPPING_REASON_CODES,
    _observation,
    _scale_configuration,
    first_stage_stopping_reason,
)
from examples.jax.parity.input_bundle import create_input_bundle, load_input_bundle
from examples.jax.parity.official_reference import load_official_reference
from examples.jax.parity.runtime import ParityLane
from simsopt_jax.examples.boozer_official import (
    BoozerStageOutcome,
    BoozerStageState,
    boozer_official_options,
)

# venv site-packages/tests shadows the repo tests package, so the helper
# is imported as a top-level module from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from parity_native_cpu import run_native_cpu_child

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Child source is a string so OpenMP is in the environment before the
# extension is imported. ``run_native_cpu_child`` applies
# ``build_parity_lane_environment``, the SSOT that sets
# ``OMP_NUM_THREADS=1`` for native-cpu. An in-process env pin cannot undo
# the pytest process team; on kernel B that team-dependent L-BFGS warm
# start walks the area-constrained residual onto a second Boozer root
# (iota collapsing toward 0) while JAX and the OMP=1 native lane stay on
# the NCSX-like root near the initial iota -0.4.
_NATIVE_CHILD = """\
import pickle
import sys
from dataclasses import fields
from pathlib import Path

from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import read_input_bundle

bundle_root = Path(sys.argv[1])
out_path = Path(sys.argv[2])
observation = get_case("native-boozer").execute(
    "native-cpu", *read_input_bundle(bundle_root)
)
payload = {field.name: getattr(observation, field.name) for field in fields(observation)}
payload["values"] = dict(observation.values)
payload["applicability"] = dict(observation.applicability)
out_path.write_bytes(pickle.dumps(payload))
"""


def _native_observation(input_root: Path, out_path: Path) -> LaneObservation:
    completed = run_native_cpu_child(
        _NATIVE_CHILD,
        str(input_root),
        str(out_path),
        repo_root=_REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    return LaneObservation(**pickle.loads(out_path.read_bytes()))


@pytest.mark.parametrize("lane", ["native-cpu", "jax-cpu"])
@pytest.mark.parametrize(
    ("area_success", "flux_success", "expected_success"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_boozer_observation_requires_both_solver_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: ParityLane,
    area_success: bool,
    flux_success: bool,
    expected_success: bool,
) -> None:
    bundle = create_input_bundle(
        tmp_path / "inputs",
        case_id="native-boozer",
        random_seed=0,
        arrays={"surface_dofs": np.asarray([1.0])},
        configuration={"mpol": 2},
    )
    values = {
        "construction:axis_dofs": np.asarray([0.0]),
        "construction:field_dofs": np.asarray([1.0]),
        "initial:surface_dofs": np.asarray([1.0]),
        "initial:residual_norm": np.asarray(10.0),
        "area:solver_success": np.asarray(area_success),
        "flux:solver_success": np.asarray(flux_success),
        "flux:surface_dofs": np.asarray([1.0]),
        "flux:residual_norm": np.asarray(1.0),
        "flux:iota": np.asarray(-0.4),
        "flux:G": np.asarray(1.0),
    }
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")

    observation = _observation(
        lane,
        bundle,
        values,
        platform="cpu",
        precision="fp64",
        driver="test-provider",
    )

    assert observation.success is expected_success
    assert observation.normalized_status == (
        "converged" if expected_success else "failed"
    )
    assert observation.raw_status == f"area={area_success};flux={flux_success}"


def test_exact_boozer_surface_workflow_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native lane is a one-thread child; kernel-B L-BFGS is not unique under OMP>1.

    ``OMP_NUM_THREADS`` is read when libgomp starts, so an in-process env
    pin cannot undo the pytest process team. The native lane therefore
    runs in a subprocess whose environment comes from
    ``run_native_cpu_child`` / ``build_parity_lane_environment`` (the
    SSOT that sets ``OMP_NUM_THREADS=1`` before the child imports the
    extension).
    """
    case = get_case("native-boozer")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native = _native_observation(input_root, tmp_path / "native-observation.pkl")

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    assert native.success is True
    assert jax.success is True
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)

    for observable in (
        "construction:axis_dofs",
        "construction:field_dofs",
        "initial:surface_dofs",
        "initial:residual",
        "initial:jacobian",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-11,
            atol=1.0e-13,
        )

    # Which Boozer surface the workflow lands on is not a function of its input:
    # the first stage stops at its iteration cap, unconverged, and upstream's
    # own script reaches five different surfaces from nine one-ulp starts at
    # this scale (tracked upstream scatter record). Each lane's end state is
    # therefore judged as the arbiter judges it: it must match one of upstream's
    # branch representatives (the lowest-k draw of each branch), under the
    # case's route comparator for every judged key.
    end_states = get_case("native-boozer").end_states("bounded")
    assert end_states is not None
    routes = _scale_routes("bounded")
    for observation in (native, jax):
        assert upstream_end_state_matches(end_states, routes, observation.values), (
            observation.lane
        )
    assert float(native.values["flux:residual_norm"]) < float(
        native.values["initial:residual_norm"]
    )
    assert float(jax.values["flux:residual_norm"]) < float(
        jax.values["initial:residual_norm"]
    )

    # The official first stage is published on both lanes, with the provider's
    # own outcome, so a stage-by-stage comparison against the official capture
    # is possible (A/claude/runs/native-boozer/captured-omp1/capture.json
    # records that stage's iota, G, residual_norm, iter 300 of 300 and
    # solver_success False under its own lbfgs_area: prefix).
    for observation in (native, jax):
        maxiter = _rough_maxiter(observation)
        assert 0 < int(observation.values["first:nit"]) <= maxiter
        assert int(observation.values["first:nfev"]) >= int(
            observation.values["first:nit"]
        )
        assert int(observation.values["first:njev"]) >= 1
        assert observation.values["first:status"].dtype == np.int64
        assert observation.values["first:stopping_reason_code"].dtype == np.int64
        # The first stage is a budget exit on both lanes, as it is on upstream,
        # so the published classification is pinned by the lane's own published
        # facts and the contract's vocabulary -- not by re-running the
        # classifier here.
        assert int(observation.values["first:nit"]) == maxiter
        assert bool(observation.values["first:solver_success"]) is False
        assert (
            int(observation.values["first:stopping_reason_code"])
            == STOPPING_REASON_CODES["iteration-limit"]
        )
        assert observation.values["first:solver_success"].dtype == np.bool_
        assert np.isfinite(float(observation.values["first:objective"]))
        assert float(observation.values["first:residual_norm"]) > 0.0
        assert observation.values["first:surface_dofs"].shape == (
            observation.values["initial:surface_dofs"].shape
        )
        assert float(observation.values["first:label"]) > 0.0
    # This scale is the one where the two lanes do NOT share the first stage's
    # budget, so status, nit and solver_success are published diagnostics here
    # and are compared at native_default instead, where both lanes run
    # OFFICIAL_LBFGS_MAXITER.  Equalising the budgets here is not free: it
    # moves the JAX lane onto a different Boozer branch in stage two (see
    # _scale_configuration), so this assertion is a guard in both directions.
    assert not _shares_first_stage_budget(bundle.scale)
    # The persistence flags are compared at every scale: lanes that disagree
    # chained stage two from different end states, which would surface only as
    # an unexplained area:/flux: mismatch.
    for observable in _PERSISTED_KEYS:
        assert bool(native.values[observable]) is bool(jax.values[observable])
        assert bool(native.values[observable]) is True
    _assert_route_matrix(set(jax.values))


def _scale_routes(scale: str):
    """The case's comparison routes at ``scale``, as the runner resolves them."""
    pair = load_runtime_contract_pair(
        _REPO_ROOT / "examples/jax/manifest.json",
        _REPO_ROOT / "examples/jax/parity_manifest.json",
        repo_root=_REPO_ROOT,
    )
    relationship = next(
        item
        for item in pair.parity.all_relationships
        if item.case_id == "native-boozer"
    )
    return relationship.resolve_scale(scale).comparison_routes


@pytest.mark.parametrize("scale", ("bounded", "native_default"))
def test_upstream_end_states_span_several_boozer_surfaces(scale: str) -> None:
    """The end-state contract exists only because upstream's nine draws disagree.

    The pre-registered rule: an end-state set is declared at a scale only when
    upstream's own nine draws land on at least two end states that the case's
    own route comparators tell apart; with one, the end state would be a
    function of the input and the lanes would be compared with each other.
    Each draw then belongs to exactly one branch, represented by its
    lowest-k draw, and matches that representative alone.
    """
    end_states = get_case("native-boozer").end_states(scale)
    assert end_states is not None
    routes = _scale_routes(scale)
    all_draws = tuple(state.k for state in end_states.states)
    assert all_draws == tuple(range(9))
    representatives = tuple(
        state.k for state in upstream_branch_representatives(end_states, routes)
    )
    assert len(representatives) >= 2
    for state in end_states.states:
        (branch,) = upstream_end_state_matches(end_states, routes, state.values)
        assert branch <= state.k
        assert (branch == state.k) == (state.k in representatives)


#: The first stage's exactly comparable facts. They are equal across the lanes
#: whenever both lanes run the same budget and that budget is exhausted, which
#: is upstream's own outcome (iteration-limit, nit 300, success false).
#: ``first:status`` is NOT among them: the two lanes' first stages are two
#: different emitters, so only the normalized stopping reason is comparable and
#: the raw integer stays a published diagnostic.
_SHARED_BUDGET_FIRST_KEYS = (
    "first:stopping_reason_code",
    "first:nit",
    "first:solver_success",
)
#: Whether each stage left the provider's iterate behind or restored the stage
#: start; a lane pair that disagrees chained the next stage from two different
#: states.
_PERSISTED_KEYS = (
    "first:provider_persisted_iterate",
    "area:provider_persisted_iterate",
    "flux:provider_persisted_iterate",
)


def _shares_first_stage_budget(scale: str) -> bool:
    """True when both lanes get the same first-stage budget at ``scale``."""
    configuration = _scale_configuration(scale)
    return int(configuration["native_bfgs_maxiter"]) == int(
        configuration["jax_bfgs_maxiter"]
    )


def _rough_maxiter(observation: LaneObservation) -> int:
    configuration = _scale_configuration(observation.scale)
    key = (
        "native_bfgs_maxiter"
        if observation.lane == "native-cpu"
        else "jax_bfgs_maxiter"
    )
    return int(configuration[key])


def _declared_routes() -> tuple[list[dict[str, object]], ...]:
    document = json.loads(
        (_REPO_ROOT / "examples/jax/parity_manifest.json").read_text(encoding="utf-8")
    )
    relationship = next(
        item for item in document["relationships"] if item["case_id"] == "native-boozer"
    )
    return (
        relationship["comparison_routes"],
        relationship["scale_contracts"]["native_default"]["comparison_routes"],
    )


def _assert_route_matrix(published: set[str]) -> None:
    """Every published key has a complete lane-pair matrix at every scale.

    This is the invariant ``examples/jax/parity/arbiter.py`` enforces in
    ``_validate_route_matrix``: a published observable with no declared route
    makes the whole case fail closed with an ArbitrationError, which is exactly
    what the fourteen keys added in fix wave 1 did.

    The first stage's applicability is not a free choice either.
    ``stopping_reason_code``, ``nit`` and ``solver_success`` are comparable at
    exactly the scales where ``_scale_configuration`` gives the two lanes the
    same first-stage budget, so the expectation is DERIVED from that function
    instead of restated here: declaring the whole stage uncompared while both
    lanes run ``OFFICIAL_LBFGS_MAXITER`` is the round-2 defect this pins.
    ``first:status`` is a published DIAGNOSTIC at every scale -- it is the raw
    integer of two different emitters -- so it falls in the uncompared group
    below, which is the round-3 defect this pins.  The same stage's evaluation
    counts and its floats stay uncomparable (line-search trials and a
    path-dependent budget-exit point), the two plain-Boozer initial keys are
    same-state quantities, and the three persistence flags say which end state
    each stage chained from, so they are compared.
    """
    lane_pairs = {"native-cpu:jax-cpu", "native-cpu:jax-gpu", "jax-cpu:jax-gpu"}
    for scale, routes in zip(
        ("bounded", "native_default"), _declared_routes(), strict=True
    ):
        by_key: dict[str, set[str]] = {}
        applicability: dict[str, set[bool]] = {}
        for route in routes:
            key = f"{route['phase']}:{route['observable']}"
            by_key.setdefault(key, set()).add(route["lane_pair"])
            applicability.setdefault(key, set()).add(route["applicable"])
        assert set(by_key) == published
        assert all(pairs == lane_pairs for pairs in by_key.values())
        for key, flags in applicability.items():
            assert len(flags) == 1, (scale, key)
        shared_budget = _shares_first_stage_budget(scale)
        for key in _SHARED_BUDGET_FIRST_KEYS:
            assert applicability[key] == {shared_budget}, (scale, key)
        for key in published:
            if (
                key.startswith("first:")
                and key not in _SHARED_BUDGET_FIRST_KEYS
                and key not in _PERSISTED_KEYS
            ):
                assert applicability[key] == {False}, (scale, key)
        for key in _PERSISTED_KEYS:
            assert applicability[key] == {True}, (scale, key)
        assert applicability["initial:boozer_residual"] == {True}
        assert applicability["initial:boozer_jacobian"] == {True}


def _first_stage_outcome(
    status: int,
    *,
    iterations: int,
    endpoint_finite: bool,
) -> BoozerStageOutcome:
    """One L-BFGS first-stage outcome with a chosen provider report."""
    objective = 0.25 if endpoint_finite else float("nan")
    return BoozerStageOutcome(
        state=BoozerStageState(
            surface_dofs=np.asarray([1.0, 2.0]),
            iota=-0.4,
            G=1.0,
        ),
        provider_persisted_iterate=True,
        success=False,
        status=status,
        message="",
        nit=iterations,
        nfev=iterations,
        njev=iterations,
        objective=objective,
        gradient_norm=1.0,
        penalty_residual_norm=None,
    )


def _published_code(
    driver: str,
    outcome: BoozerStageOutcome,
    *,
    max_iterations: int,
) -> int:
    """The integer this case publishes as ``first:stopping_reason_code``."""
    return STOPPING_REASON_CODES[
        first_stage_stopping_reason(
            outcome,
            max_iterations=max_iterations,
            status_convention=FIRST_STAGE_STATUS_CONVENTION_BY_DRIVER[driver],
        )
    ]


def test_first_stage_stopping_reason_normalizes_the_two_solver_vocabularies() -> None:
    """The compared fact is the classification, not the raw provider integer.

    The native lane's first stage is ``scipy.optimize.minimize(method=
    'L-BFGS-B')`` and the JAX lane's is the private on-device L-BFGS-B port, and
    ``simsopt_contracts.optimization_endpoint`` keeps their status tables apart
    because the same integer means different things in the two. The two cases
    below are the two ways a raw comparison gets the answer wrong, and each one
    also asserts what the RAW integers do, so this test fails if the published
    value were the raw status again.
    """
    maxiter = int(_scale_configuration("native_default")["native_bfgs_maxiter"])

    # Same stop, different integers: L-BFGS-B reports its budget code 1 with a
    # non-finite endpoint, the port reports its own non-finite code 6. Both are
    # the same endpoint, so the classification is equal and the raw route would
    # have failed.
    native_nonfinite = _first_stage_outcome(
        1, iterations=maxiter, endpoint_finite=False
    )
    jax_nonfinite = _first_stage_outcome(6, iterations=maxiter, endpoint_finite=False)
    assert native_nonfinite.status != jax_nonfinite.status
    assert _published_code(
        NATIVE_DRIVER, native_nonfinite, max_iterations=maxiter
    ) == _published_code(JAX_DRIVER, jax_nonfinite, max_iterations=maxiter)
    assert (
        _published_code(JAX_DRIVER, jax_nonfinite, max_iterations=maxiter)
        == STOPPING_REASON_CODES["nonfinite"]
    )

    # Different stops, the same integer: 99 is the port's callback stop and is
    # outside SciPy's vocabulary, where the endpoint is simply the exhausted
    # iteration budget. The raw route would have PASSED this pair.
    native_99 = _first_stage_outcome(99, iterations=maxiter, endpoint_finite=True)
    jax_99 = _first_stage_outcome(99, iterations=maxiter, endpoint_finite=True)
    assert native_99.status == jax_99.status
    assert _published_code(
        NATIVE_DRIVER, native_99, max_iterations=maxiter
    ) != _published_code(JAX_DRIVER, jax_99, max_iterations=maxiter)
    assert (
        _published_code(NATIVE_DRIVER, native_99, max_iterations=maxiter)
        == STOPPING_REASON_CODES["iteration-limit"]
    )
    assert (
        _published_code(JAX_DRIVER, jax_99, max_iterations=maxiter)
        == STOPPING_REASON_CODES["callback-stopped"]
    )

    # Upstream's own first-stage outcome, which both lanes reach at
    # native_default: status 1 at the exhausted budget classifies identically
    # under the two conventions, which is why today's receipts agree.
    budget_exit = _first_stage_outcome(1, iterations=maxiter, endpoint_finite=True)
    assert _published_code(
        NATIVE_DRIVER, budget_exit, max_iterations=maxiter
    ) == _published_code(JAX_DRIVER, budget_exit, max_iterations=maxiter)
    assert (
        _published_code(NATIVE_DRIVER, budget_exit, max_iterations=maxiter)
        == STOPPING_REASON_CODES["iteration-limit"]
    )


def test_first_stage_budget_and_outcome_match_the_official_record() -> None:
    """The declared first-stage policy is upstream's own, read from the record.

    ``examples/jax/parity/official_reference`` is the tracked copy of the
    official run at 9e027eac3.  Its first provider call IS the official first
    stage, so the case's declared budget and tolerance are compared against it
    instead of against a literal, and its result is the evidence that the stage
    is a budget exit -- which is what makes ``first:status`` / ``first:nit`` /
    ``first:solver_success`` comparable across lanes at all, and what makes the
    evaluation counts incomparable.
    """
    reference = load_official_reference("native-boozer")
    call = reference.provider_calls[0]
    assert call.function == "scipy.optimize.minimize"
    assert call.method == "L-BFGS-B"

    options = dict(call.options)
    configuration = _scale_configuration("native_default")
    assert options["maxiter"] == int(configuration["native_bfgs_maxiter"])
    assert options["maxiter"] == int(configuration["jax_bfgs_maxiter"])
    assert options["ftol"] == float(configuration["solver_tolerance"])
    assert options["gtol"] == float(configuration["solver_tolerance"])
    assert (
        options["maxcor"]
        == boozer_official_options(
            rough_maxiter=int(configuration["jax_bfgs_maxiter"]),
            ls_maxiter=int(configuration["jax_ls_maxiter"]),
            tolerance=float(configuration["solver_tolerance"]),
        )["maxcor"]
    )

    result = dict(call.result)
    assert result["status"] == 1
    assert result["success"] is False
    assert result["nit"] == options["maxiter"]
    assert result["nfev"] > result["nit"]
    assert result["njev"] == result["nfev"]
    assert reference.scalar("lbfgs_area:solver_success") is False


def test_every_published_observable_has_a_declared_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wave-1 regression: fourteen published keys with no manifest route."""
    case = get_case("native-boozer")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    observation = case.execute("jax-cpu", bundle, arrays)

    _assert_route_matrix(set(observation.values))


def test_boozer_initial_plain_residual_is_the_official_definition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``initial:boozer_residual`` is the capture's residual, not the penalty one.

    The official capture stores ``boozer_surface_residual(s, iota, G, bs,
    derivatives=1)`` at the start point: ``3*nphi*ntheta`` rows, unscaled,
    ``weight_inv_modB=False`` and no constraint rows. The solver's own penalty
    residual carries two extra rows (the label and the ``z(0,0)`` constraint),
    so the two published pairs must differ in exactly that way.
    """
    case = get_case("native-boozer")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    observation = case.execute("jax-cpu", bundle, arrays)

    mpol = int(bundle.configuration["mpol"])
    ntor = int(bundle.configuration["ntor"])
    quadrature_points = (2 * ntor + 1) * (2 * mpol + 1)
    surface_dofs = observation.values["initial:surface_dofs"].shape[0]

    boozer_residual = observation.values["initial:boozer_residual"]
    boozer_jacobian = observation.values["initial:boozer_jacobian"]
    assert boozer_residual.shape == (3 * quadrature_points,)
    assert boozer_jacobian.shape == (3 * quadrature_points, surface_dofs + 2)
    assert observation.values["initial:residual"].shape == (3 * quadrature_points + 2,)
    assert observation.values["initial:jacobian"].shape == (
        3 * quadrature_points + 2,
        surface_dofs + 2,
    )
    np.testing.assert_allclose(
        float(np.linalg.norm(boozer_residual)),
        float(observation.values["initial:residual_norm"]),
        rtol=0.0,
        atol=0.0,
    )
