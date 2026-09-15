"""Strict-transfer coverage for the exact stochastic Stage-II workflow."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_stochastic_stage_two_has_explicit_device_transfer_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-stage-two-optimization-stochastic")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    # execute() reads os.environ["SIMSOPT_BACKEND_MODE"]; sister parity tests pin it.
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")

    with jax.transfer_guard("disallow"):
        observation = case.execute("jax-cpu", bundle, arrays)

    assert observation.success is True
