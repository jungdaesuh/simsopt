"""Contract tests for the flat-675 nested-endpoint comparison harness.

The harness stitches together three things it does not own: the unit-A bridge
(``nested_view_from_flat675``), the unit-B correction lanes
(``correct_with_nested_ls_jax`` / ``correct_with_nested_ls_native`` /
``flat675_boozer_term``) and the two shipped example scripts.  So the tests
here come in three layers:

1. **Arity proofs.**  ``inspect.signature`` and ``dataclasses.fields`` assert
   that the real collaborators still have exactly the shape the stand-ins
   below mirror.  Without this layer the stand-ins would be free to drift away
   from production and still pass.
2. **Stand-in runs.**  The payload schema, the gate arithmetic and the
   Markdown writer are exercised through stand-ins that carry those proven
   signatures, so a gate can be driven to both of its outcomes without a
   minutes-long physics run.
3. **One tiny end-to-end.**  ``--configuration repository-geometry
   --max-steps 1`` on CPU, through ``main``, proving the underresolved fixture
   fails closed and writes a strict-JSON payload and a Markdown table.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmarks.flat675_nested_endpoint as endpoint_module
import pytest
from benchmarks.flat675_nested_endpoint import (
    EXIT_STATUS_FROM_FULL_DECISION_GRADIENT,
    EXIT_STATUS_FROM_REDUCED_GRADIENT,
    NestedCorrection,
    correct_with_nested_ls_jax,
    correct_with_nested_ls_native,
    flat675_boozer_term,
    stays_on_incoming_branch,
)
from benchmarks.flat675_nested_endpoint_comparison import (
    EVALUATION_CORRECTION,
    EVALUATION_FAILED_SOLVE,
    EVALUATION_REJECTED_BRANCH_CHANGE,
    OBJECTIVE_RTOL,
    SCHEMA,
    NestedLanes,
    ObjectiveLane,
    SolveOutcome,
    build_problem,
    compare,
    corrected_vector,
    correction_evaluation_status,
    main,
    nested_correction_is_noop,
    parse_args,
    render_markdown,
    require_platform,
)
from simsopt_jax_adapters.geo.flat675 import (
    FLAT675_COIL_DOF_COUNT,
    FLAT675_COIL_SLICE,
    FLAT675_OBJECTIVE_TERM_KEYS,
    FLAT675_OUTER_DOF_COUNT,
    FLAT675_SURFACE_DOF_COUNT,
    FLAT675_SURFACE_SLICE,
    FLAT675_VESSEL_DOF_COUNT,
    FLAT675_VESSEL_SLICE,
    Flat675Problem,
)
from simsopt_jax_adapters.geo.flat675.nested_bridge import (
    NestedView,
    nested_view_from_flat675,
)
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_BANANA_NEWTON_TOL,
    NESTED_LS_JAX_INNER_POLICY_NAME,
    NESTED_LS_NATIVE_INNER_POLICY_NAME,
    NESTED_LS_NEWTON_EXIT_CONVERGED,
    NESTED_LS_NEWTON_EXIT_FAILED,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
)
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import dump_strict_json

NATIVE_TWIN_TEST: Final[Path] = (
    REPO_ROOT
    / "tests"
    / "jax"
    / "examples"
    / "test_single_stage_flat675_native_twin.py"
)

# Field names the harness relies on.  Layer 1 proves the real dataclasses carry
# exactly these, which is what makes the stand-ins below legitimate.
NESTED_VIEW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "coil_dofs",
        "vessel_dofs",
        "surface_dofs",
        "iota",
        "G",
        "coils_native",
        "biotsavart_native",
        "surface_native",
        "jax_inputs",
        # Added by unit A's INTERFACE addendum (section 3); the harness does
        # not read them, but pinning the full field set is what makes the
        # stand-in below provably a subset rather than a guess.
        "label_native",
        "layout",
        "vector",
    }
)
NESTED_CORRECTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "lane",
        # Per-lane inner provenance: the contract forbids a lane-agnostic
        # inner record, so these are part of the shape, not decoration.
        "inner_policy",
        "bfgs_iterations",
        "newton_iterations",
        "exit_status",
        "exit_status_quantity",
        "persisted",
        "failure_reason",
        "reduced_gradient_l2",
        "coil_delta_inf",
        "omp_num_threads",
        "tolerance",
        "residual_norm_before",
        "residual_norm_after",
        "converged",
        "surface_dofs_after",
        "iota_before",
        "iota_after",
        "same_branch_as_incoming",
        "iota_branch_guard",
        "G_after",
        "dof_displacement_l2",
        "dof_displacement_max",
        "point_displacement_max_m",
        "point_displacement_rms_m",
        "minor_radius_m",
        "wall_s",
    }
)

POINT_NAME_ORDER: Final[tuple[str, str]] = ("start", "endpoint")


def _table_rows(markdown: str, heading: str) -> list[dict[str, str]]:
    """One Markdown table's data rows, keyed by that table's own header cells.

    Column-aware assertions are the only ones worth making about a rendered
    table: a substring test passes on any cell anywhere, including the one it
    was not written for.
    """
    lines = markdown.splitlines()
    index = lines.index(f"#### {heading}")
    while not lines[index].startswith("|"):
        index += 1
    table: list[list[str]] = []
    while index < len(lines) and lines[index].startswith("|"):
        table.append([cell.strip() for cell in lines[index].strip("|").split("|")])
        index += 1
    header = table[0]
    return [dict(zip(header, row)) for row in table[2:]]


# The two bars, restated as the tests read them.
PHYSICS_TOL: Final[float] = float(NESTED_LS_NEWTON_TOL)
TIMING_TOL: Final[float] = float(NESTED_LS_BANANA_NEWTON_TOL)


# --------------------------------------------------------------------------
# Layer 1 — the real collaborators still have the shape the stand-ins mirror
# --------------------------------------------------------------------------


def test_bridge_and_lane_signatures_match_the_frozen_contract() -> None:
    view_signature = inspect.signature(nested_view_from_flat675)
    assert [name for name in view_signature.parameters] == ["problem", "vector"]
    for lane in (correct_with_nested_ls_jax, correct_with_nested_ls_native):
        lane_signature = inspect.signature(lane)
        assert [name for name in lane_signature.parameters] == ["view"], lane.__name__
    term_signature = inspect.signature(flat675_boozer_term)
    assert [name for name in term_signature.parameters] == ["problem", "vector"]


def test_nested_dataclasses_carry_exactly_the_fields_the_harness_reads() -> None:
    assert dataclasses.is_dataclass(NestedView)
    assert dataclasses.is_dataclass(NestedCorrection)
    assert {field.name for field in dataclasses.fields(NestedView)} == (
        NESTED_VIEW_FIELDS
    )
    assert {field.name for field in dataclasses.fields(NestedCorrection)} == (
        NESTED_CORRECTION_FIELDS
    )
    # The view stand-in is a strict subset of the real view, and the correction
    # stand-in builds the real dataclass, so neither can drift from production.
    assert {field.name for field in dataclasses.fields(_ViewStandIn)} <= (
        NESTED_VIEW_FIELDS
    )


def test_objective_rtol_does_not_drift_from_the_native_twin_test() -> None:
    """The twin test owns the bar; this harness transcribes it."""
    module = ast.parse(NATIVE_TWIN_TEST.read_text())
    found = [
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "OBJECTIVE_RTOL"
            for target in node.targets
        )
    ]
    assert len(found) == 1, "OBJECTIVE_RTOL is not a single module-level assignment"
    assert ast.literal_eval(found[0]) == OBJECTIVE_RTOL


# --------------------------------------------------------------------------
# Stand-ins carrying those signatures
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ViewStandIn:
    """Mirrors the fields of ``NestedView`` the harness actually reads."""

    coil_dofs: NDArray[np.float64]
    vessel_dofs: NDArray[np.float64]
    surface_dofs: NDArray[np.float64]
    iota: float
    G: float


def _vector(seed: int) -> NDArray[np.float64]:
    generator = np.random.default_rng(seed)
    return generator.normal(size=FLAT675_OUTER_DOF_COUNT) * 1.0e-3


def _correction(
    *,
    lane: Literal["jax", "native"],
    surface_after: NDArray[np.float64],
    residual_before: float,
    residual_after: float,
    bfgs_iterations: int | None,
    newton_iterations: int,
    converged: bool,
    iota_before: float,
    iota_after: float,
    G_after: float,
    displacement_max: float,
    exit_status: str = NESTED_LS_NEWTON_EXIT_CONVERGED,
    persisted: bool = True,
    reduced_gradient_l2: float | None = None,
    coil_delta_inf: float = 0.0,
    omp_num_threads: str = "1",
) -> NestedCorrection:
    """Build a real ``NestedCorrection``, so the dataclass enforces the arity.

    The branch flag is computed by the production predicate rather than passed
    in: a stand-in that could declare itself on-branch while its own
    ``iota_after`` says otherwise would let the gate under test pass on a
    contradiction.
    """
    return NestedCorrection(
        lane=lane,
        inner_policy=(
            NESTED_LS_JAX_INNER_POLICY_NAME
            if lane == "jax"
            else NESTED_LS_NATIVE_INNER_POLICY_NAME
        ),
        tolerance=TIMING_TOL,
        residual_norm_before=residual_before,
        residual_norm_after=residual_after,
        bfgs_iterations=bfgs_iterations,
        newton_iterations=newton_iterations,
        converged=converged,
        exit_status=exit_status,
        # The lane determines the quantity, exactly as production does: a
        # stand-in free to name the other lane's norm would let a per-lane
        # assertion pass on a record production cannot emit.
        exit_status_quantity=(
            EXIT_STATUS_FROM_REDUCED_GRADIENT
            if lane == "jax"
            else EXIT_STATUS_FROM_FULL_DECISION_GRADIENT
        ),
        persisted=persisted,
        failure_reason=None,
        reduced_gradient_l2=reduced_gradient_l2,
        coil_delta_inf=coil_delta_inf,
        surface_dofs_after=surface_after,
        iota_before=iota_before,
        iota_after=iota_after,
        G_after=G_after,
        same_branch_as_incoming=stays_on_incoming_branch(
            iota_before=iota_before, iota_after=iota_after
        ),
        iota_branch_guard=float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
        dof_displacement_l2=displacement_max,
        dof_displacement_max=displacement_max,
        point_displacement_max_m=displacement_max * 2.0,
        point_displacement_rms_m=displacement_max,
        minor_radius_m=0.25,
        omp_num_threads=omp_num_threads,
        wall_s=0.5,
    )


def _lanes(
    *,
    problem: Flat675Problem,
    jax_correction: NestedCorrection,
    native_correction: NestedCorrection,
) -> NestedLanes:
    """Stand-ins with the production signatures, asserting their own arity."""

    def view_fn(
        received_problem: Flat675Problem, vector: NDArray[np.float64]
    ) -> NestedView:
        assert received_problem is problem
        vector = np.asarray(vector, dtype=np.float64)
        assert vector.shape == (FLAT675_OUTER_DOF_COUNT,)
        surface = vector[FLAT675_SURFACE_SLICE]
        return _ViewStandIn(  # type: ignore[return-value]
            coil_dofs=vector[FLAT675_COIL_SLICE],
            vessel_dofs=vector[FLAT675_VESSEL_SLICE],
            surface_dofs=surface,
            iota=0.15 + float(surface[0]),
            G=1.0 + float(surface[1]),
        )

    def jax_fn(view: NestedView) -> NestedCorrection:
        assert isinstance(view, _ViewStandIn)
        return jax_correction

    def native_fn(view: NestedView) -> NestedCorrection:
        assert isinstance(view, _ViewStandIn)
        return native_correction

    def boozer_term_fn(
        received_problem: Flat675Problem, vector: NDArray[np.float64]
    ) -> float:
        assert received_problem is problem
        return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))

    return NestedLanes(
        view_fn=view_fn,
        jax_fn=jax_fn,
        native_fn=native_fn,
        boozer_term_fn=boozer_term_fn,
    )


def _objective_lane() -> ObjectiveLane:
    def terms(vector: NDArray[np.float64]) -> dict[str, float]:
        array = np.asarray(vector, dtype=np.float64)
        return {
            str(key): float(np.sum(array[index :: len(FLAT675_OBJECTIVE_TERM_KEYS)]))
            for index, key in enumerate(FLAT675_OBJECTIVE_TERM_KEYS)
        }

    def value(vector: NDArray[np.float64]) -> float:
        return float(sum(terms(vector).values()))

    return ObjectiveLane(value=value, weighted_terms=terms)


def _solve_outcome(
    start: NDArray[np.float64], endpoint: NDArray[np.float64]
) -> SolveOutcome:
    return SolveOutcome(
        start_vector=start,
        endpoint_vector=endpoint,
        iterations=3,
        objective_evaluations=7,
        final_objective=1.25,
        success=True,
        endpoint_is_optimizer_x=True,
        transfer_ledger={"advance": 0, "callback": 0, "final_result": 1},
        wall_s=12.5,
    )


def _payload(
    *,
    jax_correction: NestedCorrection,
    native_correction: NestedCorrection,
    twin_offset: float = 0.0,
) -> dict[str, object]:
    # The harness treats the problem as opaque -- it only forwards it to the
    # bridge and the two lanes -- so an uninitialised instance of the real
    # frozen dataclass is the right stand-in: every stand-in below asserts it
    # received THIS object, and touching any field would raise.
    problem = object.__new__(Flat675Problem)
    start = _vector(1)
    endpoint = _vector(2)
    objective = _objective_lane()

    def solve_fn(received_problem: Flat675Problem, *, max_steps: int) -> SolveOutcome:
        assert received_problem is problem
        assert max_steps == 3
        return _solve_outcome(start, endpoint)

    def twin_value(vector: NDArray[np.float64]) -> float:
        return objective.value(vector) + twin_offset

    return compare(
        problem=problem,
        objective=objective,
        twin_value=twin_value,
        configuration="repository-geometry",
        jax_platform="cpu",
        max_steps=3,
        lanes=_lanes(
            problem=problem,
            jax_correction=jax_correction,
            native_correction=native_correction,
        ),
        solve_fn=solve_fn,
    )


def _noop_kwargs(lane: Literal["jax", "native"]) -> dict[str, object]:
    """A lane-shaped no-op at the PHYSICS bar, as a mutable base.

    ``residual_before`` is half the PHYSICS tolerance, not half the timing
    one: the gate under test is the certified rejudge gate's bar, and a value
    that only clears the timing bar must NOT read as a no-op.
    """
    return dict(
        lane=lane,
        surface_after=_vector(2)[FLAT675_SURFACE_SLICE],
        residual_before=PHYSICS_TOL * 0.5,
        residual_after=PHYSICS_TOL * 0.5,
        bfgs_iterations=None if lane == "jax" else 0,
        newton_iterations=0,
        converged=True,
        iota_before=0.15,
        iota_after=0.15,
        G_after=1.0,
        displacement_max=0.0,
        reduced_gradient_l2=PHYSICS_TOL * 0.5 if lane == "jax" else None,
    )


def _noop_correction(lane: Literal["jax", "native"]) -> NestedCorrection:
    return _correction(**_noop_kwargs(lane))


def _moving_correction(lane: Literal["jax", "native"]) -> NestedCorrection:
    surface = np.array(_vector(2)[FLAT675_SURFACE_SLICE], copy=True)
    surface[0] += 1.0e-4
    return _correction(
        lane=lane,
        surface_after=surface,
        residual_before=1.0e-6,
        residual_after=TIMING_TOL * 0.1,
        bfgs_iterations=None if lane == "jax" else 3,
        newton_iterations=4,
        converged=True,
        iota_before=0.15,
        iota_after=0.1503,
        G_after=1.0007,
        displacement_max=1.0e-4,
        reduced_gradient_l2=TIMING_TOL * 0.1 if lane == "jax" else None,
    )


def _branch_changing_correction(lane: Literal["jax", "native"]) -> NestedCorrection:
    """A converged inner solve that landed on a DIFFERENT Boozer branch.

    The numbers are the contract's own motivating case
    (``iota 0.1409 -> -0.0024`` with every inner solve converging).
    """
    surface = np.array(_vector(2)[FLAT675_SURFACE_SLICE], copy=True)
    surface[0] += 2.0e-3
    return _correction(
        lane=lane,
        surface_after=surface,
        residual_before=1.0e-6,
        residual_after=TIMING_TOL * 0.1,
        bfgs_iterations=None if lane == "jax" else 5,
        newton_iterations=6,
        converged=True,
        iota_before=0.1409,
        iota_after=-0.0024,
        G_after=1.0007,
        displacement_max=2.0e-3,
        reduced_gradient_l2=TIMING_TOL * 0.1 if lane == "jax" else None,
    )


def _diverged_correction(lane: Literal["jax", "native"]) -> NestedCorrection:
    """A failed, divergent walk whose ``iota`` happens to stay put.

    These are the repository-geometry B3 CPU row's own numbers: the solve did
    not converge and did not persist, it moved the surface block by 2.95e14,
    and its ``iota`` drifted by 4.15e-3 -- well inside the branch guard.  A
    status read off the branch guard alone calls this a "correction".
    """
    surface = np.array(_vector(2)[FLAT675_SURFACE_SLICE], copy=True)
    surface[0] += 2.95e14
    return _correction(
        lane=lane,
        surface_after=surface,
        residual_before=7.1513,
        residual_after=7.1513,
        bfgs_iterations=None if lane == "jax" else 264,
        newton_iterations=1,
        converged=False,
        exit_status=NESTED_LS_NEWTON_EXIT_FAILED,
        persisted=False,
        iota_before=0.1516,
        iota_after=0.1516 + 4.15e-3,
        G_after=1.0007,
        displacement_max=2.95e14,
    )


# --------------------------------------------------------------------------
# Layer 2 — schema, gates, Markdown
# --------------------------------------------------------------------------


def test_payload_carries_the_schema_id_and_every_documented_block() -> None:
    payload = _payload(
        jax_correction=_moving_correction("jax"),
        native_correction=_moving_correction("native"),
    )
    assert payload["schema"] == "flat675-nested-endpoint-comparison-v4"
    assert frozenset(payload) >= frozenset(
        {
            "schema",
            "question",
            "configuration",
            "execution_scale",
            "max_steps",
            "jax_platform",
            "outer_dof_count",
            "objective_term_keys",
            "nested_contract",
            "objective_rtol",
            "shared_residual_rtol",
            "shared_residual_atol",
            "solve",
            "points",
            "verdict",
            "walls",
            "runtime",
        }
    )
    assert payload["schema"] == "flat675-nested-endpoint-comparison-v4", (
        "the schema id must move with the fields"
    )
    nested = payload["nested_contract"]
    # Both bars, both named, so no ratio in this payload is ambiguous.
    assert nested["tolerance"] == TIMING_TOL
    assert nested["physics_tolerance"] == PHYSICS_TOL
    assert nested["tolerance_bar"] and nested["physics_tolerance_bar"]
    assert nested["tolerance_bar"] != nested["physics_tolerance_bar"]
    assert nested["iota_branch_guard"] == float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD)
    # H8: the native lane's pin here is not the certified lane's thread count,
    # and the payload has to carry both numbers for that to be readable.
    assert nested["certified_native_omp_threads"] > 1
    assert frozenset(payload["points"]) == frozenset({"start", "endpoint"})
    for point in payload["points"].values():
        assert frozenset(point["corrections"]) == frozenset({"jax", "native"})
        assert frozenset(point["weighted_terms"]) == frozenset(
            FLAT675_OBJECTIVE_TERM_KEYS
        )
        for lane, record in point["corrections"].items():
            # Every NestedCorrection field reaches the payload except the 661
            # corrected DOFs themselves, which are carried by sha256.
            assert frozenset(record) >= (
                NESTED_CORRECTION_FIELDS - {"surface_dofs_after"}
            ) | {"surface_dofs_after_sha256"}
            assert len(record["surface_dofs_after_sha256"]) == 64
            assert frozenset(record["weighted_term_deltas_vs_base"]) == frozenset(
                FLAT675_OBJECTIVE_TERM_KEYS
            )
            # Provenance is per lane and says which solver ran.
            assert record["inner_policy"] == (
                NESTED_LS_JAX_INNER_POLICY_NAME
                if lane == "jax"
                else NESTED_LS_NATIVE_INNER_POLICY_NAME
            )
            assert (record["bfgs_iterations"] is None) is (lane == "jax")
            assert record["reduced_gradient_clause_available"] is (lane == "jax")
            # Each ratio is taken against the bar its NAME claims.  A
            # "physics > timing" assertion here would follow from
            # ``PHYSICS_TOL < TIMING_TOL`` alone and would still pass if both
            # ratios had been divided by the same bar.
            assert record["residual_before_over_timing_bar"] == pytest.approx(
                record["residual_norm_before"] / nested["tolerance"]
            )
            assert record["residual_before_over_physics_bar"] == pytest.approx(
                record["residual_norm_before"] / nested["physics_tolerance"]
            )
            assert record["residual_after_over_timing_bar"] == pytest.approx(
                record["residual_norm_after"] / nested["tolerance"]
            )
            assert record["residual_after_over_physics_bar"] == pytest.approx(
                record["residual_norm_after"] / nested["physics_tolerance"]
            )
            # The exit status names the norm it was classified from, and the
            # two lanes do not classify the same one.
            assert record["exit_status_quantity"] == (
                EXIT_STATUS_FROM_REDUCED_GRADIENT
                if lane == "jax"
                else EXIT_STATUS_FROM_FULL_DECISION_GRADIENT
            )
            assert (
                record["nested_correction_noop_bar"]
                == (nested["physics_tolerance_bar"])
            )
            assert record["evaluation_status"] == EVALUATION_CORRECTION
    assert frozenset(payload["verdict"]) == frozenset(
        {
            "noop_bar",
            "endpoint_jax_correction_is_noop_at_physics_bar",
            "endpoint_native_correction_is_noop_at_physics_bar",
            "endpoint_residual_over_timing_bar_jax",
            "endpoint_residual_over_timing_bar_native",
            "endpoint_residual_over_physics_bar_jax",
            "endpoint_residual_over_physics_bar_native",
            "iota_branch_guard",
            "endpoint_jax_correction_stayed_on_branch",
            "endpoint_native_correction_stayed_on_branch",
            "start_jax_correction_stayed_on_branch",
            "start_native_correction_stayed_on_branch",
            "every_correction_stayed_on_branch",
            "shared_residual_definition_ok",
            "native_twin_control_ok",
            "native_twin_objective_valid_for_band",
            "endpoint_j_parity_within_band",
        }
    )
    assert frozenset(payload["walls"]) == frozenset(
        {"solve_s", "start_point_s", "endpoint_point_s", "total_s"}
    )
    # The payload must survive strict JSON (no NaN, no non-serializable value).
    assert json.loads(dump_strict_json(payload))["schema"] == SCHEMA


def test_noop_gate_has_the_shape_of_the_nested_ls_rejudge_gate() -> None:
    payload = _payload(
        jax_correction=_noop_correction("jax"),
        native_correction=_noop_correction("native"),
    )
    verdict = payload["verdict"]
    assert verdict["endpoint_jax_correction_is_noop_at_physics_bar"] is True
    assert verdict["endpoint_native_correction_is_noop_at_physics_bar"] is True
    # The residual is half the PHYSICS bar, i.e. a hundredth of the timing one.
    assert verdict["endpoint_residual_over_physics_bar_jax"] == pytest.approx(0.5)
    assert verdict["endpoint_residual_over_timing_bar_jax"] == pytest.approx(0.005)
    assert verdict["noop_bar"] == payload["nested_contract"]["physics_tolerance_bar"]


def test_the_noop_gate_is_the_physics_bar_not_the_timing_bar() -> None:
    """A residual between the two bars is a no-op under the old gate only.

    This is the H3 defect as an executable discriminator: 5e-12 clears the
    timing bar (1e-11) and misses the physics bar (1e-13) by 50x, and the
    certified rejudge gate this harness claims comparability with judges at
    the physics bar.
    """
    between = dict(_noop_kwargs("jax"))
    between["residual_before"] = TIMING_TOL * 0.5
    between["residual_after"] = TIMING_TOL * 0.5
    between["reduced_gradient_l2"] = TIMING_TOL * 0.5
    correction = _correction(**between)
    assert correction.converged is True
    assert correction.residual_norm_before <= TIMING_TOL
    assert nested_correction_is_noop(correction) is False


def _assert_only_this_lane_loses_its_noop(
    lane: Literal["jax", "native"], overrides: dict[str, object]
) -> dict[str, object]:
    """Break one clause on one lane; the other lane's no-op must survive."""
    base = dict(_noop_kwargs(lane))
    base.update(overrides)
    corrections = {
        "jax": _noop_correction("jax"),
        "native": _noop_correction("native"),
    }
    corrections[lane] = _correction(**base)
    payload = _payload(
        jax_correction=corrections["jax"],
        native_correction=corrections["native"],
    )
    other = "native" if lane == "jax" else "jax"
    verdict = payload["verdict"]
    assert verdict[f"endpoint_{lane}_correction_is_noop_at_physics_bar"] is False
    assert verdict[f"endpoint_{other}_correction_is_noop_at_physics_bar"] is True
    return payload


@pytest.mark.parametrize("lane", ["jax", "native"])
@pytest.mark.parametrize(
    "overrides",
    [
        {"converged": False},
        {"newton_iterations": 1},
        {"displacement_max": 1.0e-12},
        {"coil_delta_inf": 1.0e-18},
        {"residual_before": PHYSICS_TOL * 2.0},
        # The branch guard: converged, unmoved in every other column, and on a
        # different Boozer branch.
        {"iota_after": 0.15 + 2.0 * float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD)},
    ],
)
def test_each_noop_condition_can_break_the_gate(
    lane: Literal["jax", "native"], overrides: dict[str, object]
) -> None:
    """Clauses both lanes can actually reach, driven on both lanes.

    ``persisted`` and ``bfgs_iterations`` are deliberately NOT here: each is a
    state only one lane can emit, and driving it on the other would prove the
    gate against a record production never produces.  They get one test each
    below, on their own lane.
    """
    _assert_only_this_lane_loses_its_noop(lane, overrides)


def test_a_non_persisted_jax_walk_is_not_a_noop() -> None:
    """``persisted=False`` is reachable on the JAX lane only.

    The native lane's C++ solver writes its iterate into the surface
    unconditionally, so ``persisted`` is the constant ``True`` there; the
    reduced Schur Newton is the walk that can decline to commit, and a walk
    that returned the caller's own input has corrected nothing.
    """
    _assert_only_this_lane_loses_its_noop("jax", {"persisted": False})


def test_a_native_walk_that_took_a_bfgs_step_is_not_a_noop() -> None:
    """``bfgs_iterations >= 1`` is reachable on the native lane only.

    The certified rejudge gate's ``rejudge_iter == 0`` clause covers EVERY
    inner stage, and the native lane has two.  A BFGS pre-stage that moved is
    work the Newton-only count would not show, so a record with zero Newton
    iterations and one BFGS iteration must still fail the gate.  On the JAX
    lane there is no such stage: ``bfgs_iterations`` is ``None`` there, which
    is the statement "no such stage" rather than "zero steps".
    """
    _assert_only_this_lane_loses_its_noop("native", {"bfgs_iterations": 1})


def test_the_jax_reduced_gradient_clause_can_break_the_gate() -> None:
    """The clause the native lane cannot supply, on the lane that can.

    Everything else is a no-op; only the reduced gradient at the projected
    ``y*`` is over the bar, which is exactly the case
    ``nested_ls_outer_jax_child.py`` fails with ``reason="reduced_grad_tol"``.
    """
    over = dict(_noop_kwargs("jax"))
    over["reduced_gradient_l2"] = PHYSICS_TOL * 10.0
    assert nested_correction_is_noop(_correction(**over)) is False
    assert nested_correction_is_noop(_noop_correction("jax")) is True
    # The native lane has no such value, so the clause is vacuous there and
    # the payload says so rather than implying the stronger statement.
    native = _noop_correction("native")
    assert native.reduced_gradient_l2 is None
    assert nested_correction_is_noop(native) is True


def test_a_branch_changing_correction_is_reported_as_a_rejected_evaluation() -> None:
    """Convergence alone rejects nothing; the branch guard does.

    A lane that converged onto a different Boozer branch did not correct THIS
    point, so its row must not read as a correction of it.
    """
    payload = _payload(
        jax_correction=_branch_changing_correction("jax"),
        native_correction=_moving_correction("native"),
    )
    endpoint = payload["points"]["endpoint"]["corrections"]
    assert endpoint["jax"]["converged"] is True
    assert endpoint["jax"]["same_branch_as_incoming"] is False
    assert endpoint["jax"]["evaluation_status"] == EVALUATION_REJECTED_BRANCH_CHANGE
    assert endpoint["native"]["evaluation_status"] == EVALUATION_CORRECTION
    verdict = payload["verdict"]
    assert verdict["endpoint_jax_correction_stayed_on_branch"] is False
    assert verdict["endpoint_native_correction_stayed_on_branch"] is True
    assert verdict["every_correction_stayed_on_branch"] is False
    assert verdict["iota_branch_guard"] == float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD)
    assert render_markdown(payload).count(EVALUATION_REJECTED_BRANCH_CHANGE) >= 1


def test_a_diverged_solve_is_not_reported_as_a_correction() -> None:
    """The D01 defect, as an executable discriminator.

    The lane did not converge, did not persist and displaced the surface by
    2.95e14 -- but its ``iota`` stayed inside the branch guard, so a status
    read off the guard alone published it as a ``correction``.  The clause
    order is what fixes it: there has to be a solve before there is a branch.
    """
    diverged = _diverged_correction("jax")
    assert diverged.same_branch_as_incoming is True
    payload = _payload(
        jax_correction=diverged,
        native_correction=_moving_correction("native"),
    )
    endpoint = payload["points"]["endpoint"]["corrections"]
    assert endpoint["jax"]["evaluation_status"] == EVALUATION_FAILED_SOLVE
    assert endpoint["jax"]["evaluation_status"] != EVALUATION_CORRECTION
    assert endpoint["native"]["evaluation_status"] == EVALUATION_CORRECTION
    # A failed correction has no branch claim, even if its raw iota happens
    # to sit inside the guard.
    assert payload["verdict"]["endpoint_jax_correction_stayed_on_branch"] is None
    assert render_markdown(payload).count(EVALUATION_FAILED_SOLVE) >= 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"converged": False},
        {"persisted": False},
        {"displacement_max": float("inf")},
    ],
)
def test_each_failed_solve_clause_is_enough_on_its_own(
    overrides: dict[str, object],
) -> None:
    """Each of the three clauses demotes an otherwise on-branch row alone."""
    base = dict(_noop_kwargs("jax"))
    base.update(overrides)
    correction = _correction(**base)
    assert correction.same_branch_as_incoming is True
    payload = _payload(
        jax_correction=correction, native_correction=_noop_correction("native")
    )
    endpoint = payload["points"]["endpoint"]["corrections"]
    assert endpoint["jax"]["evaluation_status"] == EVALUATION_FAILED_SOLVE
    assert endpoint["native"]["evaluation_status"] == EVALUATION_CORRECTION


def test_mocked_native_linalg_error_writes_a_typed_failed_solve_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mocked child ``LinAlgError`` writes the native failure protocol.

    This is branch coverage for the record and payload protocol only.  It
    deliberately replaces the native solver with an exception, so it is not
    numerical evidence or native-solver recertification.
    """

    class ProtocolSurface:
        def __init__(self, dofs: NDArray[np.float64]) -> None:
            self._dofs = np.array(dofs, dtype=np.float64, copy=True)

        def set_dofs(self, dofs: NDArray[np.float64]) -> None:
            self._dofs = np.array(dofs, dtype=np.float64, copy=True)

        def gamma(self) -> NDArray[np.float64]:
            return self._dofs.reshape(-1, 1)

        def minor_radius(self) -> float:
            return 1.0

    surface_before = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    native_surface = ProtocolSurface(surface_before)
    geometry_surface = ProtocolSurface(surface_before)
    scalars = {
        "surface_dofs": surface_before,
        "iota": 0.17,
        "G": 2.3,
        "label_target": 1.0,
        "constraint_weight": 1.0,
    }
    (tmp_path / endpoint_module._SCALARS_FILE).write_bytes(pickle.dumps(scalars))

    def raise_native_linalg_error(*_args: object, **_kwargs: object) -> None:
        raise np.linalg.LinAlgError("forced native failure protocol")

    monkeypatch.setattr(
        endpoint_module.simsopt,
        "load",
        lambda _path: (object(), native_surface),
    )
    monkeypatch.setattr(
        endpoint_module,
        "native_boozer_at",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        endpoint_module,
        "native_penalty_evaluation",
        lambda *_args, **_kwargs: SimpleNamespace(
            gradient=np.array([3.0, 4.0], dtype=np.float64)
        ),
    )
    monkeypatch.setattr(
        endpoint_module,
        "clone_surface_xyz_tensor_fourier",
        lambda _surface: geometry_surface,
    )
    monkeypatch.setattr(
        endpoint_module,
        "_run_native_banana_bfgs_then_newton",
        raise_native_linalg_error,
    )
    monkeypatch.setattr(
        endpoint_module,
        "nested_ls_threading_env",
        lambda: {"OMP_NUM_THREADS": "1"},
    )

    endpoint_module.run_native_correction_child(str(tmp_path))

    record = pickle.loads((tmp_path / endpoint_module._CORRECTION_FILE).read_bytes())
    assert isinstance(record, NestedCorrection)
    assert record.lane == "native"
    assert record.converged is False
    assert record.persisted is False
    assert record.exit_status == NESTED_LS_NEWTON_EXIT_FAILED
    assert record.failure_reason == "numpy.linalg.LinAlgError"
    assert record.bfgs_iterations is None
    assert record.newton_iterations is None
    assert record.reduced_gradient_l2 is None
    assert record.coil_delta_inf is None
    assert record.surface_dofs_after is None
    assert record.residual_norm_after is None
    assert record.iota_after is None
    assert record.G_after is None
    assert record.same_branch_as_incoming is None
    assert record.dof_displacement_l2 is None
    assert record.dof_displacement_max is None
    assert record.point_displacement_max_m is None
    assert record.point_displacement_rms_m is None
    assert record.minor_radius_m is None
    assert record.omp_num_threads == "1"
    assert record.residual_norm_before == pytest.approx(5.0)
    assert correction_evaluation_status(record) == EVALUATION_FAILED_SOLVE


def test_failed_correction_nulls_post_state_and_gated_claims() -> None:
    """A contained child failure cannot mint a finite corrected-point claim."""
    failed = dataclasses.replace(
        _diverged_correction("native"),
        residual_norm_after=None,
        coil_delta_inf=None,
        surface_dofs_after=None,
        iota_after=None,
        G_after=None,
        same_branch_as_incoming=None,
        dof_displacement_l2=None,
        dof_displacement_max=None,
        point_displacement_max_m=None,
        point_displacement_rms_m=None,
        minor_radius_m=None,
        failure_reason="numpy.linalg.LinAlgError",
    )
    payload = _payload(
        jax_correction=_moving_correction("jax"), native_correction=failed
    )
    record = payload["points"]["endpoint"]["corrections"]["native"]
    assert record["evaluation_status"] == EVALUATION_FAILED_SOLVE
    assert record["failure_reason"] == "numpy.linalg.LinAlgError"
    for key in (
        "coil_delta_inf",
        "surface_dofs_after_sha256",
        "residual_norm_after",
        "residual_after_over_timing_bar",
        "residual_after_over_physics_bar",
        "iota_after",
        "iota_delta",
        "same_branch_as_incoming",
        "G_after",
        "dof_displacement_l2",
        "dof_displacement_max",
        "point_displacement_max_m",
        "point_displacement_rms_m",
        "minor_radius_m",
        "point_displacement_max_over_minor_radius",
        "point_displacement_rms_over_minor_radius",
        "y_solve_iota_at_corrected",
        "y_solve_G_at_corrected",
        "y_solve_minus_lane_iota",
        "y_solve_minus_lane_G",
        "flat675_boozer_term_at_corrected",
        "objective",
        "objective_minus_base",
        "weighted_terms",
        "weighted_term_deltas_vs_base",
        "native_twin_objective",
        "native_twin_relative_gap",
        "native_twin_within_objective_rtol",
    ):
        assert record[key] is None
    assert all(
        value is None
        for value in payload["points"]["endpoint"]["lane_agreement"].values()
    )
    verdict = payload["verdict"]
    assert verdict["endpoint_native_correction_stayed_on_branch"] is None
    assert verdict["every_correction_stayed_on_branch"] is None
    assert verdict["shared_residual_definition_ok"] is None
    assert "n/a" in render_markdown(payload)
    assert json.loads(dump_strict_json(payload))["schema"] == SCHEMA


def test_native_unsuccessful_return_is_failed_even_with_finite_post_state() -> None:
    """A returned iterate is not convergence, as the old tiny fixture showed."""
    unsuccessful = dataclasses.replace(
        _moving_correction("native"),
        converged=False,
        persisted=True,
        exit_status=NESTED_LS_NEWTON_EXIT_FAILED,
        residual_norm_after=2.0 * NESTED_LS_BANANA_NEWTON_TOL,
        failure_reason=None,
    )
    payload = _payload(
        jax_correction=_moving_correction("jax"), native_correction=unsuccessful
    )
    record = payload["points"]["endpoint"]["corrections"]["native"]
    assert record["evaluation_status"] == EVALUATION_FAILED_SOLVE
    assert record["persisted"] is True
    assert record["converged"] is False
    assert record["failure_reason"] is None
    assert record["residual_norm_after"] > record["tolerance"]
    assert record["objective"] is None
    assert payload["verdict"]["endpoint_native_correction_stayed_on_branch"] is None


def test_a_branch_change_outranks_nothing_but_still_outranks_a_correction() -> None:
    """A converged, persisted, finite walk off the branch is still rejected."""
    payload = _payload(
        jax_correction=_branch_changing_correction("jax"),
        native_correction=_noop_correction("native"),
    )
    endpoint = payload["points"]["endpoint"]["corrections"]
    assert endpoint["jax"]["converged"] is True
    assert endpoint["jax"]["persisted"] is True
    assert endpoint["jax"]["evaluation_status"] == EVALUATION_REJECTED_BRANCH_CHANGE


def test_a_non_finite_residual_publishes_null_ratios_instead_of_aborting() -> None:
    """D15: ``dump_strict_json`` refuses infinity, so the ratios are nullable.

    An unguarded ``residual / bar`` would raise at write time and take the
    whole run's payload with it -- on exactly the diverged solve the page
    exists to report.  The residuals here are finite and serialize fine; it is
    the division by the physics bar (a factor 1e13) that overflows, which is
    why the guard is on the quotient rather than on the input.
    """
    base = dict(_noop_kwargs("jax"))
    base["residual_before"] = 1.0e300
    base["residual_after"] = 1.0e300
    payload = _payload(
        jax_correction=_correction(**base),
        native_correction=_noop_correction("native"),
    )
    record = payload["points"]["endpoint"]["corrections"]["jax"]
    for key in (
        "residual_before_over_timing_bar",
        "residual_before_over_physics_bar",
        "residual_after_over_timing_bar",
        "residual_after_over_physics_bar",
    ):
        assert record[key] is None
    assert record["residual_before_under_timing_bar"] is False
    assert record["residual_before_under_physics_bar"] is False
    assert payload["verdict"]["endpoint_residual_over_timing_bar_jax"] is None
    assert payload["verdict"]["endpoint_residual_over_physics_bar_jax"] is None
    # The native lane still has real ratios, so "null" is per lane.
    assert math.isfinite(
        float(payload["verdict"]["endpoint_residual_over_timing_bar_native"])
    )
    assert (
        json.loads(dump_strict_json(payload))["points"]["endpoint"]["corrections"][
            "jax"
        ]["residual_before_over_timing_bar"]
        is None
    )
    assert "n/a" in render_markdown(payload)


def test_every_correction_on_branch_is_true_when_no_lane_left_the_branch() -> None:
    payload = _payload(
        jax_correction=_moving_correction("jax"),
        native_correction=_moving_correction("native"),
    )
    assert payload["verdict"]["every_correction_stayed_on_branch"] is True


def test_shared_residual_definition_gate_catches_two_different_residuals() -> None:
    disagreeing = _correction(
        lane="native",
        surface_after=_vector(2)[FLAT675_SURFACE_SLICE],
        residual_before=2.0e-6,
        residual_after=1.0e-12,
        bfgs_iterations=3,
        newton_iterations=4,
        converged=True,
        iota_before=0.15,
        iota_after=0.15,
        G_after=1.0,
        displacement_max=1.0e-4,
    )
    payload = _payload(
        jax_correction=_moving_correction("jax"),
        native_correction=disagreeing,
    )
    assert payload["verdict"]["shared_residual_definition_ok"] is False
    agreeing = _payload(
        jax_correction=_moving_correction("jax"),
        native_correction=_moving_correction("native"),
    )
    assert agreeing["verdict"]["shared_residual_definition_ok"] is True


def test_corrected_vector_replaces_only_the_surface_block() -> None:
    vector = _vector(5)
    correction = _moving_correction("jax")
    updated = corrected_vector(vector, correction)
    assert updated.shape == (FLAT675_OUTER_DOF_COUNT,)
    assert np.array_equal(
        updated[: FLAT675_COIL_DOF_COUNT + FLAT675_VESSEL_DOF_COUNT],
        vector[: FLAT675_COIL_DOF_COUNT + FLAT675_VESSEL_DOF_COUNT],
    )
    assert np.array_equal(updated[FLAT675_SURFACE_SLICE], correction.surface_dofs_after)
    assert correction.surface_dofs_after.shape == (FLAT675_SURFACE_DOF_COUNT,)
    # The input must not be mutated: the harness reuses it for the other lane.
    assert not np.array_equal(updated, vector)


def test_corrected_vector_refuses_a_wrong_width_surface_block() -> None:
    correction = dataclasses.replace(
        _moving_correction("jax"), surface_dofs_after=np.zeros(3, dtype=np.float64)
    )
    with pytest.raises(SystemExit):
        corrected_vector(_vector(6), correction)


def test_markdown_states_the_four_points_and_every_term() -> None:
    payload = _payload(
        jax_correction=_moving_correction("jax"),
        native_correction=_moving_correction("native"),
    )
    markdown = render_markdown(payload)
    for heading in (
        "Nested correction per lane, per point",
        "Inner-solver provenance, exit and the branch guard",
        "Eight-term objective at the four points",
        "iota, G and lane agreement",
        "Solve, gates and wall clocks",
    ):
        assert heading in markdown
    for column in ("endpoint + jax", "endpoint + native", "native twin J"):
        assert column in markdown
    for key in FLAT675_OBJECTIVE_TERM_KEYS:
        assert str(key) in markdown
    for label in (
        "max ‖Δx‖ (m)",
        "max ‖Δx‖ / a",
        "Δiota",
        "ΔG",
        # H3: no ratio on the page without its bar; H2/H5/H8: provenance.
        "before / timing bar",
        "before / physics bar",
        "inner policy",
        "bfgs it",
        "newton it",
        "exit status",
        "exit classified on",
        "persisted",
        "reduced ‖g‖₂",
        "same branch",
        "evaluation",
        "no-op (physics bar)",
        "OMP_NUM_THREADS",
    ):
        assert label in markdown
    # Both bar names, and the guard, appear in the prose line above the tables.
    assert payload["nested_contract"]["tolerance_bar"] in markdown
    assert payload["nested_contract"]["physics_tolerance_bar"] in markdown
    assert "iota branch guard" in markdown
    # ``n/a`` has to land in the RIGHT cells: a bare ``"n/a" in markdown`` is
    # satisfied by the JAX lane's bfgs column and says nothing about the
    # reduced-gradient column it was written for.
    rows = _table_rows(markdown, "Inner-solver provenance, exit and the branch guard")
    assert len(rows) == len(POINT_NAME_ORDER) * 2
    for row in rows:
        if row["lane"] == "native":
            assert row["reduced ‖g‖₂"] == "n/a"
            assert row["bfgs it"] != "n/a"
            assert row["exit classified on"] == EXIT_STATUS_FROM_FULL_DECISION_GRADIENT
        else:
            assert row["bfgs it"] == "n/a"
            assert row["reduced ‖g‖₂"] != "n/a"
            assert row["exit classified on"] == EXIT_STATUS_FROM_REDUCED_GRADIENT
    assert NESTED_LS_JAX_INNER_POLICY_NAME in markdown
    assert NESTED_LS_NATIVE_INNER_POLICY_NAME in markdown
    # Every table must be rectangular, or the page renders as prose.
    tables: list[list[str]] = []
    for line in markdown.splitlines():
        if line.startswith("|"):
            if not tables or not tables[-1]:
                tables.append([])
            tables[-1].append(line)
        elif tables and tables[-1]:
            tables.append([])
    filled = [table for table in tables if table]
    assert len(filled) == 5
    for table in filled:
        widths = {line.count("|") for line in table}
        # A pipe inside a cell would silently shear the table in a renderer,
        # so a ragged table here is a real defect, not a cosmetic one.
        assert len(widths) == 1, table[0]
        assert widths.pop() >= 3


# --------------------------------------------------------------------------
# CLI guards
# --------------------------------------------------------------------------


def test_cli_parses_exactly_the_frozen_flags() -> None:
    args = parse_args(
        [
            "--configuration",
            "bundle",
            "--max-steps",
            "37",
            "--out-json",
            "/tmp/x.json",
            "--jax-platform",
            "gpu",
        ]
    )
    assert args.configuration == "bundle"
    assert args.max_steps == 37
    assert args.jax_platform == "gpu"
    assert Path(args.out_json) == Path("/tmp/x.json")


def test_a_gpu_row_is_refused_on_a_cpu_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmarks.flat675_nested_endpoint_comparison.jax",
        SimpleNamespace(default_backend=lambda: "cpu"),
    )
    with pytest.raises(SystemExit):
        require_platform("gpu")


def test_a_cpu_row_is_accepted_on_a_cpu_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmarks.flat675_nested_endpoint_comparison.jax",
        SimpleNamespace(default_backend=lambda: "cpu"),
    )
    require_platform("cpu")


def test_non_positive_budget_is_refused() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--configuration",
                "repository-geometry",
                "--max-steps",
                "0",
                "--out-json",
                "/tmp/never-written.json",
                "--jax-platform",
                "cpu",
            ]
        )


# --------------------------------------------------------------------------
# Layer 3 — one tiny CPU end-to-end through the real wiring
# --------------------------------------------------------------------------


def test_repository_geometry_has_unobservable_surface_directions() -> None:
    """The tiny sampling cannot determine the 661 surface coefficients.

    Boozer's sampled residual and Volume depend on position and two tangents.
    A common null direction of those three maps is therefore a structural
    degeneracy, independent of the QR seed or Newton's stopping tolerance.
    """
    problem = build_problem("repository-geometry")
    view = nested_view_from_flat675(problem, problem.start_candidate.outer_vector())
    surface = view.surface_native
    geometry_map = np.concatenate(
        [
            derivative.reshape(-1, FLAT675_SURFACE_DOF_COUNT)
            for derivative in (
                surface.dgamma_by_dcoeff(),
                surface.dgammadash1_by_dcoeff(),
                surface.dgammadash2_by_dcoeff(),
            )
        ]
    )
    assert geometry_map.shape == (324, FLAT675_SURFACE_DOF_COUNT)
    assert geometry_map.shape[0] < geometry_map.shape[1]
    _, singular_values, right_vectors = np.linalg.svd(geometry_map, full_matrices=True)
    null_direction = right_vectors[-1]
    roundoff = np.finfo(np.float64).eps * max(geometry_map.shape) * singular_values[0]
    assert np.linalg.norm(geometry_map @ null_direction) <= roundoff


def test_underresolved_cpu_end_to_end_records_native_linalg_failure_and_fails_closed(
    tmp_path: Path,
) -> None:
    """The underresolved repository native child hits ``LinAlgError``.

    The geometry-map test above proves a structural nullspace. Controlled
    old/new source runs also show the old seed returned success=False; an
    older green serialization test never established native convergence.
    The real CLI must publish failure with null post-state and return nonzero.
    """
    out_json = tmp_path / "repository_geometry_b1_cpu.json"
    assert (
        main(
            [
                "--configuration",
                "repository-geometry",
                "--max-steps",
                "1",
                "--out-json",
                str(out_json),
                "--jax-platform",
                "cpu",
            ]
        )
        == 1
    )
    payload = json.loads(out_json.read_text())
    assert payload["schema"] == "flat675-nested-endpoint-comparison-v4"
    assert payload["configuration"] == "repository-geometry"
    assert payload["max_steps"] == 1
    assert payload["outer_dof_count"] == FLAT675_OUTER_DOF_COUNT
    assert payload["runtime"]["simsoptpp_sha256"]
    endpoint = payload["points"]["endpoint"]
    assert np.isfinite(endpoint["objective"])
    for lane in ("jax", "native"):
        record = endpoint["corrections"][lane]
        assert record["lane"] == lane
        assert record["tolerance"] == float(NESTED_LS_BANANA_NEWTON_TOL)
        assert np.isfinite(record["residual_norm_before"])
    native = endpoint["corrections"]["native"]
    assert native["evaluation_status"] == EVALUATION_FAILED_SOLVE
    assert native["exit_status"] == NESTED_LS_NEWTON_EXIT_FAILED
    assert native["persisted"] is False
    assert native["converged"] is False
    assert native["failure_reason"] == "numpy.linalg.LinAlgError"
    for key in (
        "surface_dofs_after_sha256",
        "residual_norm_after",
        "iota_after",
        "G_after",
        "same_branch_as_incoming",
        "objective",
        "weighted_terms",
    ):
        assert native[key] is None
    assert endpoint["lane_agreement"]["shared_residual_definition_ok"] is None
    assert payload["verdict"]["endpoint_native_correction_stayed_on_branch"] is None
    markdown = out_json.with_suffix(".md").read_text()
    assert EVALUATION_FAILED_SOLVE in markdown
    assert "n/a" in markdown


def test_shared_residual_gate_has_the_solver_tolerance_as_its_floor() -> None:
    """Two norms below the nested tolerance agree by construction (round-off);
    two norms that differ above it are a definitional disagreement."""
    jax_start = dataclasses.replace(
        _moving_correction("jax"), residual_norm_before=1.443954e-14
    )
    native_start = dataclasses.replace(
        _moving_correction("native"), residual_norm_before=1.436954e-14
    )
    at_round_off = _payload(jax_correction=jax_start, native_correction=native_start)
    assert at_round_off["verdict"]["shared_residual_definition_ok"] is True
    native_off = dataclasses.replace(
        _moving_correction("native"), residual_norm_before=0.2
    )
    jax_off = dataclasses.replace(_moving_correction("jax"), residual_norm_before=0.1)
    disagreeing = _payload(jax_correction=jax_off, native_correction=native_off)
    assert disagreeing["verdict"]["shared_residual_definition_ok"] is False
