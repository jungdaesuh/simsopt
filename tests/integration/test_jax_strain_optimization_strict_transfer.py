"""Device-boundary contracts for the exact strain-optimization workflow."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples/jax/2_Intermediate/strain_optimization.py"
WORKFLOW_PATH = (
    REPOSITORY_ROOT / "src/simsopt_jax/examples/strain_optimization.py"
)


def test_strain_example_uses_public_workflow_and_one_batched_host_boundary() -> None:
    source = EXAMPLE_PATH.read_text()

    assert "from simsopt_jax.examples import" in source
    assert "solve_strain_rotation" in source
    assert "simsopt_jax.core.framedcurve" not in source
    assert "simsopt_jax.solve.dispatch" not in source
    assert source.count("jax.device_get(") == 1


def test_strain_workflow_explicitly_publishes_host_optimizer_state_to_device() -> None:
    source = WORKFLOW_PATH.read_text()

    assert "jax.device_put(optimizer_result.x, initial_array.sharding)" in source
