"""Exact parity for ``2_Intermediate/stage_two_optimization_planar_coils.py``."""

from __future__ import annotations

import ast
from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle

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


def test_exact_planar_stage_two_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-stage-two-optimization-planar-coils")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native_directory = tmp_path / "native"
    native_directory.mkdir()
    with chdir(native_directory):
        native = case.execute("native-cpu", bundle, arrays)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    # The bounded budget ends every stage on its iteration cap. The label says
    # so; it is neither convergence nor failure, and it still implies the case's
    # scientific predicate, because a false predicate is labelled ``failed``.
    # It is also not upstream's terminal status: upstream 9e027eac3 stops by
    # line-search stagnation at 135 and 69 iterations because its four
    # CurvePlanarFourier Jacobians sit in the persistent cache, so its end point
    # is not reachable by a gradient that describes the objective. The same-state
    # comparison with upstream is value-at-upstream's-states and gradient-at-the-
    # start-state, in tests/integration/test_jax_mirror_planar_coils_official_states.py;
    # the end point, which forks between runs at round-off, is judged below by
    # the band from upstream's own scatter at this scale.
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

    # Both lanes stop at the iteration cap on a path that forks at round-off,
    # so the end point is judged as the arbiter judges it: each lane against
    # the band from upstream's own nine one-ulp draws at this scale.
    band = case.quality_band("bounded")
    assert band is not None
    assert band.observable == "final:objective"
    for observation in (native, jax):
        assert float(observation.values["final:objective"]) <= band.max_value

    for observation in (native, jax):
        taylor_errors = np.abs(observation.values["taylor:errors"][:3])
        assert taylor_errors[1] <= 2.0e-2 * taylor_errors[0]
        assert taylor_errors[2] <= 2.0e-2 * taylor_errors[1]
        assert taylor_errors[2] <= 1.0e-4
