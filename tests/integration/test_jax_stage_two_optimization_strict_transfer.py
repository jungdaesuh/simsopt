"""Device-boundary contracts for the standard Stage-II mirror."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples/jax/2_Intermediate/stage_two_optimization.py"


def test_standard_stage_two_uses_public_workflow_and_one_host_publication() -> None:
    source = EXAMPLE_PATH.read_text()

    assert "solve_standard_stage_two" in source
    assert "make_stage_two_objective" not in source
    assert "serial_solve_jax" not in source
    assert source.count("jax.device_get(") == 1
    assert "first_length_weight_device = jax.device_put(" in source
    assert "second_length_weight_device = jax.device_put(" in source
    assert "first_length_weight=first_length_weight_device" in source
    assert "second_length_weight=second_length_weight_device" in source
