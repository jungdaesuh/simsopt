"""Exact parity for ``2_Intermediate/stage_two_optimization_planar_coils.py``."""

from __future__ import annotations

import ast
import pickle
import sys
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle
from examples.jax.parity.runtime import ParityLane

# venv site-packages/tests shadows the repo tests package, so the helper
# is imported as a top-level module from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from parity_native_cpu import run_parity_lane_child

CASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/jax/parity/cases/native_stage_two_optimization_planar_coils.py"
)


def test_planar_topology_jit_receives_device_arrays_as_operands() -> None:
    module = ast.parse(CASE_PATH.read_text(encoding="utf-8"))
    topology_assignments = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "topology_states"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
    ]

    assert len(topology_assignments) == 1
    assert [
        argument.id
        for argument in topology_assignments[0].value.args
        if isinstance(argument, ast.Name)
    ] == ["extraction", "parameter_states"]


_REPO_ROOT = Path(__file__).resolve().parents[2]
# Each lane runs in a child under its parity environment, as the harness runs
# it: the native SquaredFlux reduction depends on the OpenMP team, which an
# in-process pin cannot undo once the pytest process has started it.
_LANE_CHILD = """\
import pickle
import sys
from dataclasses import fields
from pathlib import Path

from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import read_input_bundle

lane = sys.argv[1]
bundle_root = Path(sys.argv[2])
out_path = Path(sys.argv[3])
observation = get_case("native-stage-two-optimization-planar-coils").execute(
    lane, *read_input_bundle(bundle_root)
)
payload = {field.name: getattr(observation, field.name) for field in fields(observation)}
payload["values"] = dict(observation.values)
payload["applicability"] = dict(observation.applicability)
out_path.write_bytes(pickle.dumps(payload))
"""


def _lane_observation(
    lane: ParityLane, input_root: Path, work: Path
) -> LaneObservation:
    work.mkdir()
    out_path = work / "observation.pkl"
    completed = run_parity_lane_child(
        lane,
        _LANE_CHILD,
        lane,
        str(input_root),
        str(out_path),
        repo_root=_REPO_ROOT,
        cwd=work,
    )
    assert completed.returncode == 0, completed.stderr
    return LaneObservation(**pickle.loads(out_path.read_bytes()))


def test_exact_planar_stage_two_matches_native_and_jax_cpu(tmp_path: Path) -> None:
    case = get_case("native-stage-two-optimization-planar-coils")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    load_input_bundle(input_root, bundle)

    native = _lane_observation("native-cpu", input_root, tmp_path / "native")
    jax = _lane_observation("jax-cpu", input_root, tmp_path / "jax")

    # The bounded budget ends every stage on its iteration cap. The label says
    # so; it is neither convergence nor failure, and it still implies the case's
    # scientific predicate, because a false predicate is labelled ``failed``.
    # It is also not upstream's terminal status: upstream 9e027eac3 stops by
    # line-search stagnation at 135 and 69 iterations because its four
    # CurvePlanarFourier Jacobians sit in the persistent cache, so its end point
    # is not reachable by a gradient that describes the objective. The same-state
    # comparison with upstream is value-at-upstream's-states and gradient-at-the-
    # start-state, in tests/integration/test_jax_mirror_planar_coils_official_states.py.
    # The end point forks between runs at round-off, so it is informational at
    # this scale (PLAN.md amendment 5, P1); the stages are judged by
    # test_jax_mirror_planar_coils_bounded_upstream_states.py and
    # test_jax_mirror_planar_coils_bounded_trajectory_twins.py.
    for observation in (native, jax):
        assert observation.normalized_status == "budget_exhausted"
        assert observation.success is False
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)

    for observable in (
        "parameters",
        "objective",
        "objective_gradient",
        "squared_flux",
        "geometric_penalty",
        "planarity_penalty",
        "linking_number",
        "canonical_geometry",
    ):
        np.testing.assert_allclose(
            jax.values[f"initial:{observable}"],
            native.values[f"initial:{observable}"],
            rtol=2.0e-8,
            atol=2.0e-10,
        )

    for stage in ("first", "final"):
        for observation in (native, jax):
            assert (
                observation.values[f"{stage}:objective"]
                < (observation.values["initial:objective"])
            )
            assert np.all(
                np.isfinite(observation.values[f"{stage}:objective_gradient"])
            )
            assert float(observation.values[f"{stage}:planarity_penalty"]) <= 1.0e-24
            assert float(observation.values[f"{stage}:linking_number"]) == 0.0
            assert float(observation.values[f"{stage}:squared_flux"]) <= 1.0e-2
            assert float(observation.values[f"{stage}:geometric_penalty"]) <= 1.0e-3
            canonical_geometry = observation.values[
                f"{stage}:canonical_geometry"
            ].reshape((-1, 13))
            assert np.all(np.isfinite(canonical_geometry))
            assert abs(float(np.sum(canonical_geometry[:, 0])) - 10.4) <= 3.0e-2

    # The first stage's end objective still decides lane against lane, as the
    # case's not_worse route does (mirror_optimization_3e2); the final one is
    # informational.
    assert float(jax.values["first:objective"]) <= (
        1.03 * float(native.values["first:objective"]) + 1.0e-9
    )

    for observation in (native, jax):
        taylor_errors = np.abs(observation.values["taylor:errors"][:3])
        assert taylor_errors[1] <= 2.0e-2 * taylor_errors[0]
        assert taylor_errors[2] <= 2.0e-2 * taylor_errors[1]
        assert taylor_errors[2] <= 1.0e-4
