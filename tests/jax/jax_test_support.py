"""Pytest fixtures for descendants awaiting conversion; runtime support is shared."""

import pytest
from core.test_buffer_ownership import make_execution_gate
from unittest_jax_support import (
    host_array as host_array,
    host_materialize as host_materialize,
    jax_runtime_isolation,
    parity_default_device as parity_default_device,
    parity_rng as parity_rng,
    parity_seed as parity_seed,
)


@pytest.fixture(autouse=True, name="jax_runtime_guard")
def fixture_jax_runtime_guard():
    """Restore each descendant pytest case through the unittest runtime owner."""
    with jax_runtime_isolation():
        yield


@pytest.fixture(params=("cpu", "gpu"), ids=("cpu_parity", "gpu_parity"), name="parity_lane")
def fixture_parity_lane(request):
    """Retain the original native-parity lane product."""
    return request.param


@pytest.fixture(name="execution_gate")
def fixture_execution_gate():
    """Build the original queued-work fixture through its moved PR3 owner."""
    return make_execution_gate()
