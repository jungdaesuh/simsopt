"""Exact parity for the ``2_Intermediate/wireframe_rcls_basic.py`` mirror."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases import native_wireframe_rcls_basic as rcls_basic
from examples.jax.parity.cases.native_wireframe_rcls_basic import (
    _constraint_feasibility_limit,
)
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_wireframe_rcls_basic_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-wireframe-rcls-basic")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native = case.execute("native-cpu", bundle, arrays)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    assert native.success is True
    assert jax.success is True
    assert native.scale == jax.scale == "bounded"
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)

    for observable in (
        "construction:response_matrix",
        "construction:target",
        "construction:constraint_matrix",
        "construction:constraint_target",
        "construction:free_segments",
        "construction:plasma_points",
        "construction:wireframe_nodes",
        "construction:wireframe_segments",
        "construction:wireframe_segment_signs",
        "construction:plasma_unit_normal",
        "construction:plasma_area_weights",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-13,
            atol=1.0e-20,
        )

    for phase in ("initial", "final"):
        for observable in (
            "currents",
            "normal_field_residual",
            "normal_objective",
            "regularization_objective",
            "total_objective",
            "constraint_residual",
        ):
            np.testing.assert_allclose(
                jax.values[f"{phase}:{observable}"],
                native.values[f"{phase}:{observable}"],
                rtol=1.0e-11,
                atol=1.0e-7,
            )

    for observable in (
        "final:magnetic_field",
        "final:normal_field",
        "final:mean_relative_normal_field",
        "final:maximum_current",
        "final:degrees_of_freedom",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-11,
            atol=1.0e-12,
        )

    assert native.values["construction:response_matrix"].shape == (256, 48)
    assert native.values["final:magnetic_field"].shape == (256, 3)
    assert float(native.values["final:mean_relative_normal_field"]) < 0.04
    assert float(native.values["final:normal_objective"]) < float(
        native.values["initial:normal_objective"]
    )


def _both_lanes(
    case,
    bundle,
    arrays: dict[str, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
):
    native = case.execute("native-cpu", bundle, arrays)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    return native, case.execute("jax-cpu", bundle, arrays)


def test_common_initial_infeasibility_fails_both_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two lanes that agree on an infeasible initial state must both fail.

    ``initial_currents`` is the one operand both lanes really read for the
    initial constraint residual, so perturbing it injects the identical
    physical violation into both. The lane-versus-lane roundoff route still
    agrees perfectly; only the per-lane physical gate can reject this.
    """
    case = get_case("native-wireframe-rcls-basic")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    limit = _constraint_feasibility_limit(arrays["constraint_target"])
    column = int(np.argmax(np.abs(arrays["constraint_matrix"]).sum(axis=0)))
    segment = int(arrays["free_segments"][column])
    arrays["initial_currents"] = arrays["initial_currents"].copy()
    arrays["initial_currents"][segment, 0] += 1.0e3 * limit

    native, jax = _both_lanes(case, bundle, arrays, monkeypatch)

    for observation in (native, jax):
        residual = observation.values["initial:constraint_residual"]
        assert np.linalg.norm(residual, ord=np.inf) > limit
        assert not bool(observation.values["initial:constraint_satisfied"])
        assert observation.success is False
        assert observation.normalized_status == "failed"

    # The roundoff pair route, on its own, would have accepted this.
    assert np.allclose(
        native.values["initial:constraint_roundoff_units"],
        jax.values["initial:constraint_roundoff_units"],
        rtol=0.0,
        atol=1.0,
    )


class _FailFinalGate:
    """``_constraint_feasible`` stand-in: passes initial, fails final."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, residual: np.ndarray, target: np.ndarray) -> bool:
        self.calls += 1
        assert residual.shape[0] == target.shape[0]
        return self.calls == 1


def test_final_feasibility_gate_is_a_live_conjunct_in_both_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-wireframe-rcls-basic")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native_gate = _FailFinalGate()
    monkeypatch.setattr(rcls_basic, "_constraint_feasible", native_gate)
    native = case.execute("native-cpu", bundle, arrays)

    jax_gate = _FailFinalGate()
    monkeypatch.setattr(rcls_basic, "_constraint_feasible", jax_gate)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    for observation, gate in ((native, native_gate), (jax, jax_gate)):
        assert gate.calls == 2
        assert bool(observation.values["initial:constraint_satisfied"])
        assert observation.success is False
        assert observation.normalized_status == "failed"


def test_jax_lane_operands_match_the_frozen_constraint_arrays_bitwise(
    tmp_path: Path,
) -> None:
    """Precondition of the roundoff oracle: both lanes reduce the same C, b, x.

    The native lane reduces the frozen arrays directly, but the device path
    re-derives the constraint system from the ``wireframe`` object
    (``simsopt_jax/examples/wireframe_rcls.py``), so this equality is what
    makes the two lanes' residuals comparable at all.
    """
    case = get_case("native-wireframe-rcls-basic")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    _, wireframe = rcls_basic._build_geometry(bundle.configuration)
    constraint, target = wireframe.constraint_matrices(
        assume_no_crossings=bool(bundle.configuration["assume_no_crossings"]),
        remove_constrained_segments=True,
    )
    np.testing.assert_array_equal(
        np.asarray(constraint, dtype=np.float64),
        arrays["constraint_matrix"],
    )
    np.testing.assert_array_equal(
        np.asarray(target, dtype=np.float64).reshape((-1, 1)),
        arrays["constraint_target"],
    )
    np.testing.assert_array_equal(
        np.asarray(wireframe.unconstrained_segments(), dtype=np.int64),
        arrays["free_segments"],
    )
