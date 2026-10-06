"""Public endpoint records retain their identities and share one host schema."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import pickle
from dataclasses import FrozenInstanceError, asdict, fields, replace

import jax
import numpy as np
import pytest
from simsopt_jax.pytree import registered_pytree_classes
from simsopt_jax_adapters.geo.boozer_endpoint import BoozerEndpoint
from simsopt_jax_adapters.geo.boozer_qa_problem import BoozerQAEndpoint
from simsopt_jax_adapters.geo.single_stage_boozer_vacuum_problem import (
    SingleStageVacuumEndpoint,
)


@pytest.mark.parametrize("endpoint_type", (BoozerQAEndpoint, SingleStageVacuumEndpoint))
def test_endpoint_contract_and_serialization(endpoint_type):
    endpoint = endpoint_type(
        value=1.25,
        gradient=np.asarray([2.0, -3.0], dtype=np.float64),
        inner_success=True,
        iota=-0.4,
        volume=0.5,
        non_qs_ratio=0.125,
        boozer_residual=0.03125,
        major_radius_penalty=0.0625,
        length_penalty=0.25,
    )

    assert isinstance(endpoint, BoozerEndpoint)
    assert fields(endpoint_type) == fields(BoozerEndpoint)
    assert tuple(field.name for field in fields(endpoint)) == (
        "value",
        "gradient",
        "inner_success",
        "iota",
        "volume",
        "non_qs_ratio",
        "boozer_residual",
        "major_radius_penalty",
        "length_penalty",
    )
    assert endpoint.boozer_residual_rms == 0.25
    assert endpoint_type not in registered_pytree_classes()
    assert jax.tree.leaves(endpoint)[0] is endpoint
    assert not hasattr(endpoint, "__dict__")
    with pytest.raises(FrozenInstanceError):
        endpoint.value = 2.0

    updated = replace(endpoint, value=2.0)
    assert type(updated) is endpoint_type
    assert updated.value == 2.0
    assert endpoint.value == 1.25
    restored = pickle.loads(pickle.dumps(endpoint))
    assert type(restored) is endpoint_type
    assert restored.__class__.__module__ == endpoint_type.__module__
    original_fields = asdict(endpoint)
    restored_fields = asdict(restored)
    np.testing.assert_array_equal(
        restored_fields.pop("gradient"), original_fields.pop("gradient")
    )
    assert restored_fields == original_fields
