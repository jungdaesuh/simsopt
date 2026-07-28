"""Level-set termination invariants for the QA field-line mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_qa_stopped_fieldline_remains_in_levelset_localization_band(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-tracing-fieldlines-qa")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native = case.execute("native-cpu", bundle, arrays)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    for observation in (native, jax):
        stopped = observation.values["final:status"] < 0
        distances = observation.values["final:levelset_distance"]
        assert np.count_nonzero(stopped) == 1
        assert np.max(np.abs(distances[stopped])) <= 3.0e-3
