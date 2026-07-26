"""R04 host routing: sealed factor handoffs across evaluations.

Lightweight synthetic factors only — no full GPU / trajectory campaigns.
Reuse stays default-off (None factors pass; raw trees fail closed at host gate).
"""

from __future__ import annotations

import numpy as np
import pytest

import simsopt_jax_adapters.geo.surface_objectives_traceable as sot
from simsopt_jax_adapters.geo.factor_handoff_identity import (
    ExactFactorHandoff,
    build_exact_factor_handoff,
)


def _routing_state():
    objective_kwargs = {
        "quadpoints_phi": np.asarray([0.0, 0.5], dtype=np.float64),
        "quadpoints_theta": np.asarray([0.0, 0.25], dtype=np.float64),
        "mpol": 2,
        "ntor": 1,
        "nfp": 2,
        "stellsym": True,
    }
    return {
        "objective_kwargs": objective_kwargs,
        # Production compiled-bundle key; identity falls back when certificate
        # mapper is absent.
        "coil_set_spec_from_dofs": (
            lambda coil_dofs: {"currents": np.asarray(coil_dofs)}
        ),
        "linear_solve_stab": 1.0e-4,
        "linearization_kind": "hessian",
        "linear_solve_tol": 1.0e-11,
    }


def _synthetic_factors():
    return (
        np.eye(2, dtype=np.float64),
        np.asarray([0, 1], dtype=np.int32),
    )


def test_host_producer_seals_and_consumer_releases_same_state() -> None:
    """Eval 1 seals factors; eval 2 requires them successfully at same state."""
    state = _routing_state()
    coil_dofs = np.asarray([1.0, -2.0], dtype=np.float64)
    solved_x = np.asarray([0.5, 0.25], dtype=np.float64)
    factors = _synthetic_factors()
    graph = "producer-graph-routing-v1"

    # Evaluation 1 — producer boundary.
    handoff = sot._host_seal_linear_solve_factors_for_reuse(
        factors,
        state=state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        producer_graph_sha256=graph,
    )
    assert isinstance(handoff, ExactFactorHandoff)
    # Re-sealing an already-sealed handoff is a no-op identity.
    assert (
        sot._host_seal_linear_solve_factors_for_reuse(
            handoff,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256=graph,
        )
        is handoff
    )
    # Reuse default-off: missing factors stay None.
    assert (
        sot._host_seal_linear_solve_factors_for_reuse(
            None,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256=graph,
        )
        is None
    )

    # Evaluation 2 — consumer boundary, same state.
    released = sot._host_unwrap_linear_solve_factors_for_consumption(
        handoff,
        state=state,
        coil_dofs=coil_dofs.copy(),
        solved_x=solved_x.copy(),
        producer_graph_sha256=graph,
    )
    assert released is factors
    np.testing.assert_array_equal(released[0], factors[0])


def test_host_gate_rejects_raw_factors_and_mismatched_state() -> None:
    """When the host gate is on, raw trees and mismatched seals fail closed."""
    state = _routing_state()
    coil_dofs = np.asarray([1.0, -2.0], dtype=np.float64)
    solved_x = np.asarray([0.5, 0.25], dtype=np.float64)
    factors = _synthetic_factors()
    graph = "producer-graph-routing-v1"

    handoff = sot._host_seal_linear_solve_factors_for_reuse(
        factors,
        state=state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        producer_graph_sha256=graph,
    )

    with pytest.raises(TypeError, match="unaffiliated raw factors"):
        sot._host_unwrap_linear_solve_factors_for_consumption(
            factors,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256=graph,
        )

    with pytest.raises(RuntimeError, match="coil_state_sha256"):
        sot._host_unwrap_linear_solve_factors_for_consumption(
            handoff,
            state=state,
            coil_dofs=coil_dofs + 1.0e-9,
            solved_x=solved_x,
            producer_graph_sha256=graph,
        )

    # None remains the reuse-off path (no receipt required).
    assert (
        sot._host_unwrap_linear_solve_factors_for_consumption(
            None,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256=graph,
        )
        is None
    )


def test_seal_from_res_and_solved_pair_routing_helpers(monkeypatch) -> None:
    """Pair producer helpers seal res/raw factors; gated consumer rejects raw."""
    state = _routing_state()
    factors = _synthetic_factors()
    coil_dofs = np.asarray([1.0, -2.0], dtype=np.float64)
    solved_x = np.asarray([0.5, 0.25], dtype=np.float64)
    released_to_device: list[object] = []

    def fake_total_objective(
        solved_x_arg, coil_dofs_arg, coil_set_spec, objective_kwargs
    ):
        del coil_set_spec, objective_kwargs
        return float(
            np.sum(np.asarray(solved_x_arg)) + np.sum(np.asarray(coil_dofs_arg))
        )

    def compiled_total_gradient_for(coil_dofs_arg, solved_x_arg, solved_factors):
        released_to_device.append(solved_factors)
        return 2.0 * np.asarray(coil_dofs_arg), True

    def passthrough_jit(fun=None, **_kwargs):
        def wrapped(inner):
            return inner

        if fun is None:
            return wrapped
        return wrapped(fun)

    monkeypatch.setattr(sot.jax, "jit", passthrough_jit)
    monkeypatch.setattr(
        sot, "_evaluate_traceable_total_objective", fake_total_objective
    )
    monkeypatch.setattr(
        sot,
        "_traceable_runtime_deviceify_tree",
        lambda tree: tree,
    )
    monkeypatch.setattr(
        sot,
        "_as_jax_float64",
        lambda value: np.asarray(value, dtype=np.float64),
    )
    monkeypatch.setattr(
        sot,
        "_traceable_adjoint_gradient_or_nan",
        lambda grad, _success: grad,
    )

    compiled_bundle = {
        "state": {
            **state,
            "coil_set_spec_from_dofs": lambda dofs: ("coil-set", dofs),
            "optimize_G": False,
            "predictor_kind": "none",
        },
        "compiled_forward_result_for": lambda dofs: {
            "x": solved_x,
            "linear_solve_factors": factors,
            "success": True,
        },
        "compiled_total_gradient_for": compiled_total_gradient_for,
    }
    pair = sot._build_traceable_optimizer_solved_pair(compiled_bundle)

    # --- Producer: seal raw factors from eval 1 ---
    handoff = pair.build_exact_factor_handoff_for(coil_dofs, solved_x, factors)
    assert isinstance(handoff, ExactFactorHandoff)
    assert pair.build_exact_factor_handoff_for(coil_dofs, solved_x, None) is None

    res = {
        "PLU": (factors[0], factors[0], factors[0]),
        "LU_PIV": (factors[0], factors[1]),
    }
    res_handoff = pair.seal_res_factors_for_reuse(
        res,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
    )
    assert isinstance(res_handoff, ExactFactorHandoff)
    assert (
        pair.seal_res_factors_for_reuse({}, coil_dofs=coil_dofs, solved_x=solved_x)
        is None
    )

    # --- Consumer eval 2: sealed handoff releases factors into device path ---
    value, grad = pair.value_grad_from_solved(coil_dofs, solved_x, handoff)
    assert released_to_device[-1] is factors
    assert float(value) == pytest.approx(float(np.sum(solved_x) + np.sum(coil_dofs)))
    np.testing.assert_allclose(np.asarray(grad), 2.0 * coil_dofs)

    # Explicit require API also succeeds at same state.
    assert (
        pair.require_exact_factor_handoff_for(
            handoff,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
        )
        is factors
    )

    # Host gate rejects raw unaffiliated factors when gate is on.
    with pytest.raises(TypeError, match="unaffiliated raw factors"):
        pair.value_grad_from_solved(coil_dofs, solved_x, factors)

    with pytest.raises(RuntimeError, match="coil_state_sha256"):
        pair.value_grad_from_solved(coil_dofs + 1.0e-9, solved_x, handoff)

    # Direct atomic builder still rejects bare None at the identity layer.
    with pytest.raises(ValueError, match="without factors"):
        build_exact_factor_handoff(
            state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            factors=None,
            producer_graph_sha256="graph",
        )
