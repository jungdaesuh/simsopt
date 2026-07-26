"""Tests for production exact factor-handoff identities and integrity receipts."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from simsopt_jax_adapters.geo.factor_handoff_identity import (
    ExactFactorHandoff,
    ExactHandoffProducerSeal,
    build_exact_factor_handoff,
    build_exact_handoff_identity,
    require_exact_factor_handoff,
    seal_exact_handoff_identity,
    validate_same_state_identity,
)


def _sample_state():
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
        "certificate_coil_set_spec_from_dofs": (
            lambda coil_dofs: {"currents": np.asarray(coil_dofs)}
        ),
        "linear_solve_stab": 1.0e-4,
        "linearization_kind": "hessian",
        "linear_solve_tol": 1.0e-11,
    }


def _sample_factors():
    return (
        np.eye(2, dtype=np.float64),
        np.asarray([0, 1], dtype=np.int32),
    )


def test_same_state_identity_fails_closed_on_every_bound_class() -> None:
    identity_fields = {
        "coil_state_sha256": "coil",
        "solved_state_sha256": "state",
        "factor_tree_sha256": "factor",
        "objective_and_weights_sha256": "objective",
        "stabilization": {"value": 0.0},
        "grid_modes": {"mpol": 10},
        "coil_current_configuration_sha256": "coils",
        "dtype_reduction_policy_sha256": "dtype",
        "producer_graph_sha256": "graph",
    }
    identity = seal_exact_handoff_identity(identity_fields)

    validate_same_state_identity(
        identity,
        seal_exact_handoff_identity(identity_fields),
    )
    for field in identity_fields:
        mismatched_fields = copy.deepcopy(identity_fields)
        mismatched_fields[field] = "changed"
        with pytest.raises(RuntimeError, match=field):
            validate_same_state_identity(
                identity,
                seal_exact_handoff_identity(mismatched_fields),
            )


def test_producer_seal_is_immutable_and_rejects_digest_replacement() -> None:
    producer = seal_exact_handoff_identity({"coil_state_sha256": "anchor"})

    with pytest.raises(FrozenInstanceError):
        producer.__setattr__("same_state_key_sha256", "replaced")
    with pytest.raises(ValueError, match="protected"):
        seal_exact_handoff_identity(
            {
                "coil_state_sha256": "anchor",
                "same_state_key_sha256": "attacker-selected",
            }
        )

    forged = ExactHandoffProducerSeal(
        canonical_payload_json=producer.canonical_payload_json,
        same_state_key_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="integrity"):
        validate_same_state_identity(producer, forged)


def test_producer_seal_rejects_noncanonical_payload() -> None:
    canonical = seal_exact_handoff_identity({"a": 1, "b": 2})
    forged = ExactHandoffProducerSeal(
        canonical_payload_json='{"b":2, "a":1}',
        same_state_key_sha256=canonical.same_state_key_sha256,
    )

    with pytest.raises(RuntimeError, match="not canonical"):
        validate_same_state_identity(canonical, forged)


def test_built_identity_binds_numeric_state_and_runtime_policy() -> None:
    state = _sample_state()
    coil_dofs = np.asarray([1.0, -2.0], dtype=np.float64)
    solved_x = np.asarray([0.5, 0.25], dtype=np.float64)
    factors = _sample_factors()

    first = build_exact_handoff_identity(
        state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        factors=factors,
        producer_graph_sha256="graph",
    )
    duplicate = build_exact_handoff_identity(
        state,
        coil_dofs=coil_dofs.copy(),
        solved_x=solved_x.copy(),
        factors=tuple(np.array(factor, copy=True) for factor in factors),
        producer_graph_sha256="graph",
    )
    changed = build_exact_handoff_identity(
        state,
        coil_dofs=coil_dofs,
        solved_x=solved_x + np.asarray([0.0, 1.0e-12]),
        factors=factors,
        producer_graph_sha256="graph",
    )

    validate_same_state_identity(first, duplicate)
    with pytest.raises(RuntimeError, match="solved_state_sha256"):
        validate_same_state_identity(first, changed)


def test_exact_factor_handoff_happy_path_releases_factors() -> None:
    state = _sample_state()
    coil_dofs = np.asarray([1.0, -2.0], dtype=np.float64)
    solved_x = np.asarray([0.5, 0.25], dtype=np.float64)
    factors = _sample_factors()
    graph = "producer-graph-v1"

    handoff = build_exact_factor_handoff(
        state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        factors=factors,
        producer_graph_sha256=graph,
    )

    assert isinstance(handoff, ExactFactorHandoff)
    assert isinstance(handoff.integrity_receipt, ExactHandoffProducerSeal)
    released = require_exact_factor_handoff(
        handoff,
        state=state,
        coil_dofs=coil_dofs.copy(),
        solved_x=solved_x.copy(),
        producer_graph_sha256=graph,
    )
    assert released is factors
    np.testing.assert_array_equal(released[0], factors[0])


def test_exact_factor_handoff_rejects_direct_construction() -> None:
    seal = seal_exact_handoff_identity({"coil_state_sha256": "anchor"})
    with pytest.raises(RuntimeError, match="build_exact_factor_handoff"):
        ExactFactorHandoff(
            _factors=_sample_factors(),
            _integrity_receipt=seal,
            _construction_token=object(),
        )


def test_require_rejects_raw_factors_without_receipt() -> None:
    state = _sample_state()
    coil_dofs = np.asarray([1.0, -2.0], dtype=np.float64)
    solved_x = np.asarray([0.5, 0.25], dtype=np.float64)
    raw_factors = _sample_factors()

    with pytest.raises(TypeError, match="integrity receipt"):
        require_exact_factor_handoff(
            raw_factors,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256="graph",
        )
    with pytest.raises(TypeError, match="integrity receipt"):
        require_exact_factor_handoff(
            None,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256="graph",
        )


def test_require_fails_closed_on_forged_integrity_receipt() -> None:
    state = _sample_state()
    coil_dofs = np.asarray([1.0, -2.0], dtype=np.float64)
    solved_x = np.asarray([0.5, 0.25], dtype=np.float64)
    factors = _sample_factors()
    handoff = build_exact_factor_handoff(
        state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        factors=factors,
        producer_graph_sha256="graph",
    )
    forged_receipt = ExactHandoffProducerSeal(
        canonical_payload_json=handoff.integrity_receipt.canonical_payload_json,
        same_state_key_sha256="0" * 64,
    )
    # Bypass construction guard to simulate a substituted receipt on a stolen
    # handoff object (object.__setattr__ on frozen dataclass).
    object.__setattr__(handoff, "_integrity_receipt", forged_receipt)

    with pytest.raises(RuntimeError, match="integrity"):
        require_exact_factor_handoff(
            handoff,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256="graph",
        )


def test_require_fails_closed_on_factor_substitution_under_stolen_receipt() -> None:
    state = _sample_state()
    coil_dofs = np.asarray([1.0, -2.0], dtype=np.float64)
    solved_x = np.asarray([0.5, 0.25], dtype=np.float64)
    honest = build_exact_factor_handoff(
        state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        factors=_sample_factors(),
        producer_graph_sha256="graph",
    )
    substituted = (
        2.0 * np.eye(2, dtype=np.float64),
        np.asarray([1, 0], dtype=np.int32),
    )
    object.__setattr__(honest, "_factors", substituted)

    with pytest.raises(RuntimeError, match="factor_tree_sha256"):
        require_exact_factor_handoff(
            honest,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256="graph",
        )


def test_require_fails_closed_on_mismatched_consumer_state() -> None:
    state = _sample_state()
    coil_dofs = np.asarray([1.0, -2.0], dtype=np.float64)
    solved_x = np.asarray([0.5, 0.25], dtype=np.float64)
    handoff = build_exact_factor_handoff(
        state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        factors=_sample_factors(),
        producer_graph_sha256="graph",
    )

    with pytest.raises(RuntimeError, match="coil_state_sha256"):
        require_exact_factor_handoff(
            handoff,
            state=state,
            coil_dofs=coil_dofs + 1.0e-9,
            solved_x=solved_x,
            producer_graph_sha256="graph",
        )
    with pytest.raises(RuntimeError, match="solved_state_sha256"):
        require_exact_factor_handoff(
            handoff,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x + np.asarray([1.0e-12, 0.0]),
            producer_graph_sha256="graph",
        )
    with pytest.raises(RuntimeError, match="producer_graph_sha256"):
        require_exact_factor_handoff(
            handoff,
            state=state,
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            producer_graph_sha256="different-graph",
        )


def test_build_exact_factor_handoff_rejects_missing_factors() -> None:
    with pytest.raises(ValueError, match="without factors"):
        build_exact_factor_handoff(
            _sample_state(),
            coil_dofs=np.asarray([1.0], dtype=np.float64),
            solved_x=np.asarray([0.5], dtype=np.float64),
            factors=None,
            producer_graph_sha256="graph",
        )


def test_identity_accepts_production_coil_set_spec_from_dofs_fallback() -> None:
    """Production compiled-bundle state stores coil_set_spec_from_dofs only."""
    state = {
        "objective_kwargs": {
            "quadpoints_phi": np.asarray([0.0, 0.5], dtype=np.float64),
            "quadpoints_theta": np.asarray([0.0, 0.25], dtype=np.float64),
            "mpol": 2,
            "ntor": 1,
            "nfp": 2,
            "stellsym": True,
        },
        "coil_set_spec_from_dofs": lambda coil_dofs: {
            "currents": np.asarray(coil_dofs)
        },
        "linear_solve_stab": 1.0e-4,
        "linearization_kind": "hessian",
        "linear_solve_tol": 1.0e-11,
    }
    coil_dofs = np.asarray([1.0, -2.0], dtype=np.float64)
    solved_x = np.asarray([0.5, 0.25], dtype=np.float64)
    factors = _sample_factors()

    handoff = build_exact_factor_handoff(
        state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        factors=factors,
        producer_graph_sha256="graph",
    )
    released = require_exact_factor_handoff(
        handoff,
        state=state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        producer_graph_sha256="graph",
    )
    assert released is factors
