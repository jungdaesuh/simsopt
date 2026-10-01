"""Exact parity for the ``3_Advanced/coil_forces.py`` mirror."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
from pathlib import Path

import numpy as np
import pytest
import scipy.optimize
from examples.jax.parity.cases import get_case, native_coil_forces
from examples.jax.parity.input_bundle import load_input_bundle
from simsopt_jax.solve.driver import Driver


def test_exact_coil_forces_uses_traceable_runtime_boundaries() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/jax/parity/cases/native_coil_forces.py"
    ).read_text(encoding="utf-8")

    assert "TraceableArrayFunction" in source
    assert "jax.value_and_grad(current_objective)" not in source


def test_exact_coil_forces_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-coil-forces")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native = case.execute("native-cpu", bundle, arrays)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    # The bounded budget ends every stage on its iteration cap. The label says
    # so; it is neither convergence nor failure, and it still implies the case's
    # scientific predicate, because a false predicate is labelled ``failed``.
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

    for observable in native.values:
        if observable.startswith("initial:"):
            np.testing.assert_allclose(
                jax.values[observable],
                native.values[observable],
                rtol=2.0e-8,
                atol=2.0e-10,
            )

    for stage in ("first", "final"):
        for observation in (native, jax):
            assert float(observation.values[f"{stage}:objective"]) < float(
                observation.values["initial:objective"]
            )
            assert np.all(
                np.isfinite(observation.values[f"{stage}:objective_gradient"])
            )

    np.testing.assert_allclose(
        jax.values["final:objective"],
        native.values["final:objective"],
        rtol=5.0e-2,
        atol=1.0e-9,
    )
    assert np.max(np.abs(native.values["taylor:errors"][:3])) <= 1.0e-4
    assert np.max(np.abs(jax.values["taylor:errors"][:3])) <= 1.0e-4


def _official_start(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """``examples/3_Advanced/coil_forces.py``: ``dofs = JF.x`` after the Taylor loop.

    The loop's last evaluation is ``f(dofs - 1e-7 * h)``, which is what ``JF.x``
    then holds, so that is where the official optimization starts.
    """
    return arrays["initial_parameters"] - 1.0e-7 * arrays["taylor_direction"]


def test_native_lane_minimizes_from_the_official_post_taylor_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-coil-forces")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)
    real_minimize = scipy.optimize.minimize
    starts: list[np.ndarray] = []
    endpoints: list[np.ndarray] = []

    def recording_minimize(fun, x0, *args, **kwargs):
        starts.append(np.array(x0, dtype=np.float64))
        result = real_minimize(fun, x0, *args, **kwargs)
        endpoints.append(np.array(result.x, dtype=np.float64))
        return result

    monkeypatch.setattr(scipy.optimize, "minimize", recording_minimize)
    native = case.execute("native-cpu", bundle, arrays)

    assert len(starts) == 2
    np.testing.assert_array_equal(starts[0], _official_start(arrays))
    assert not np.array_equal(starts[0], arrays["initial_parameters"])
    np.testing.assert_array_equal(starts[1], endpoints[0])
    # The published initial state and the Taylor test stay at the unperturbed set.
    np.testing.assert_array_equal(
        native.values["initial:parameters"], arrays["initial_parameters"]
    )


def test_jax_lane_minimizes_from_the_same_bits_as_the_native_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-coil-forces")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)
    real_solve = native_coil_forces.solve_scalar_stage
    starts: list[np.ndarray] = []
    policies: list[dict[str, object]] = []

    def recording_solve(problem, **kwargs):
        starts.append(np.asarray(problem.x, dtype=np.float64))
        policies.append(dict(kwargs))
        return real_solve(problem, **kwargs)

    monkeypatch.setattr(native_coil_forces, "solve_scalar_stage", recording_solve)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    assert len(starts) == 2
    np.testing.assert_array_equal(starts[0], _official_start(arrays))
    np.testing.assert_array_equal(
        jax.values["initial:parameters"], arrays["initial_parameters"]
    )
    # Both stages run the delivered route under the official call's own names:
    # one ``tol``, the configured budget, and no ``maxls`` (SciPy's default).
    budget = int(bundle.configuration["max_steps"])
    assert (
        policies
        == [
            {
                "driver": Driver.SCIPY_LBFGSB,
                "max_steps": budget,
                "maxcor": min(budget, 300),
                "tol": 1.0e-15,
            }
        ]
        * 2
    )
    assert jax.driver == Driver.SCIPY_LBFGSB.value


def _module_constant(path: Path, name: str) -> str:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.unparse(node.value)
    raise AssertionError(f"{path.name} defines no module constant {name}")


def test_parity_case_runs_the_driver_the_delivered_mirror_ships() -> None:
    """The evidence must describe what ships: one L-BFGS-B route, named once each."""
    root = Path(__file__).resolve().parents[2]
    delivered = _module_constant(
        root / "examples" / "jax" / "3_Advanced" / "coil_forces.py", "STAGE_DRIVER"
    )
    case_source = (
        root / "examples" / "jax" / "parity" / "cases" / "native_coil_forces.py"
    ).read_text(encoding="utf-8")
    routed = [
        ast.unparse(keyword.value)
        for node in ast.walk(ast.parse(case_source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "solve_scalar_stage"
        for keyword in node.keywords
        if keyword.arg == "driver"
    ]

    assert delivered == "Driver.SCIPY_LBFGSB"
    assert routed == [delivered]
