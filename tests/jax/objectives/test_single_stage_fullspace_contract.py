from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from simsopt_jax.objectives.single_stage_fullspace import (
    AUTHORITATIVE_DTYPE,
    EXAMPLE_ID,
    FROZEN_LAYOUT,
    FROZEN_PROBLEM_CONTRACT,
    SCHEMA_VERSION,
    TERM_LEDGER,
    FullSpaceState,
    TermClassification,
    frozen_problem_contract_payload,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_frozen_problem_contract_pins_joint_layout_and_physics() -> None:
    contract = FROZEN_PROBLEM_CONTRACT

    assert contract.schema_version == SCHEMA_VERSION
    assert contract.example_id == EXAMPLE_ID
    assert contract.dtype == AUTHORITATIVE_DTYPE == "float64"
    assert contract.nfp == 3
    assert contract.exact_grid_shape == (13, 13)
    assert contract.non_qs_grid_shape == (40, 40)
    assert contract.stellsym is True
    assert contract.optimize_G is True
    assert contract.weight_inv_modB is False
    assert contract.layout.ordering == (
        "coil_dofs",
        "surface_dofs",
        "iota",
        "G",
    )
    assert contract.layout.coil_dof_count == 461
    assert contract.layout.surface_dof_count == 253
    assert contract.layout.scalar_dof_count == 2
    assert contract.layout.total_dof_count == 716
    assert contract.layout.equality_count == 255
    assert contract.layout.fixed_first_base_current is True


def test_ledger_has_unique_rows_and_exact_term_classification() -> None:
    ids = tuple(row.term_id for row in TERM_LEDGER)
    fullspace_names = tuple(row.fullspace_name for row in TERM_LEDGER)

    assert len(ids) == len(set(ids))
    assert len(fullspace_names) == len(set(fullspace_names))
    assert {row.term_id: row.classification for row in TERM_LEDGER} == {
        "non_qs": TermClassification.OBJECTIVE,
        "boozer_residual_objective": TermClassification.OBJECTIVE,
        "iota_penalty": TermClassification.OBJECTIVE_PENALTY,
        "major_radius_penalty": TermClassification.OBJECTIVE_PENALTY,
        "length_penalty": TermClassification.OBJECTIVE_PENALTY,
        "boozer_equality": TermClassification.EQUALITY,
        "volume_equality": TermClassification.EQUALITY,
        "first_base_current": TermClassification.FIXED_STATE,
        "inactive_hardware_terms": TermClassification.INACTIVE,
    }
    assert all(row.dtype == "float64" for row in TERM_LEDGER)
    assert all(row.tolerance.rtol >= 0.0 for row in TERM_LEDGER)
    assert all(row.tolerance.atol >= 0.0 for row in TERM_LEDGER)


def test_ledger_source_hashes_match_current_authoritative_bytes() -> None:
    source_by_path = {row.source.path: row.source for row in TERM_LEDGER}

    for relative_path, source in source_by_path.items():
        source_path = REPOSITORY_ROOT / relative_path
        assert source_path.is_file(), f"missing ledger source: {relative_path}"
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source.sha256
        assert source.symbol


def test_contract_and_nested_ledger_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        FROZEN_PROBLEM_CONTRACT.nfp = 5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        TERM_LEDGER[0].weight = 2.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        TERM_LEDGER[0] = TERM_LEDGER[1]  # type: ignore[index]


def test_payload_is_a_detached_json_compatible_copy() -> None:
    first = frozen_problem_contract_payload()
    second = frozen_problem_contract_payload()

    assert first == second
    assert first is not second
    assert isinstance(first["terms"], tuple)
    first["nfp"] = 99
    assert frozen_problem_contract_payload()["nfp"] == 3


def test_joint_layout_round_trip_preserves_exact_block_order_under_jit() -> None:
    state = FullSpaceState(
        coil_dofs=jnp.arange(461, dtype=jnp.float64),
        surface_dofs=1000.0 + jnp.arange(253, dtype=jnp.float64),
        iota=jnp.asarray(-0.406, dtype=jnp.float64),
        G=jnp.asarray(1234.5, dtype=jnp.float64),
    )

    packed = FROZEN_LAYOUT.pack(state)
    restored = FROZEN_LAYOUT.unpack(packed)
    round_trip = jax.jit(lambda value: FROZEN_LAYOUT.pack(FROZEN_LAYOUT.unpack(value)))(
        packed
    )

    assert packed.shape == (716,)
    assert packed.dtype == jnp.float64
    assert packed[460] == 460.0
    assert packed[461] == 1000.0
    assert packed[713] == 1252.0
    assert packed[714] == -0.406
    assert packed[715] == 1234.5
    assert jnp.array_equal(restored.coil_dofs, state.coil_dofs)
    assert jnp.array_equal(restored.surface_dofs, state.surface_dofs)
    assert jnp.array_equal(round_trip, packed)


def test_joint_layout_rejects_wrong_shape_and_dtype() -> None:
    with pytest.raises(ValueError, match="shape"):
        FROZEN_LAYOUT.unpack(jnp.zeros(715, dtype=jnp.float64))
    with pytest.raises(TypeError, match="float64"):
        FROZEN_LAYOUT.unpack(jnp.zeros(716, dtype=jnp.float32))
