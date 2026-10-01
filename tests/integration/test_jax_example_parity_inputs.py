from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from pathlib import Path

import numpy as np
from examples.jax.parity.input_bundle import (
    create_input_bundle,
    load_input_bundle,
)


def test_input_bundle_round_trips_persisted_samples(tmp_path: Path) -> None:
    bundle = create_input_bundle(
        tmp_path,
        case_id="quadratic",
        random_seed=9,
        arrays={
            "initial_parameters": np.array([1.0, -2.0]),
            "quadrature": np.linspace(0.0, 1.0, 5),
        },
        configuration={"max_steps": 8, "weights": [1.0, 2.0]},
    )

    loaded, arrays = load_input_bundle(tmp_path, bundle)

    assert loaded == bundle
    np.testing.assert_array_equal(arrays["initial_parameters"], [1.0, -2.0])
    np.testing.assert_array_equal(arrays["quadrature"], np.linspace(0.0, 1.0, 5))


def test_input_fingerprint_changes_for_value_dtype_seed_and_configuration(
    tmp_path: Path,
) -> None:
    def fingerprint(
        name: str,
        *,
        values: np.ndarray,
        seed: int = 4,
        max_steps: int = 8,
    ) -> str:
        return create_input_bundle(
            tmp_path / name,
            case_id="quadratic",
            random_seed=seed,
            arrays={"initial_parameters": values},
            configuration={"max_steps": max_steps},
        ).input_fingerprint

    baseline = fingerprint("baseline", values=np.array([1.0], dtype=np.float64))

    assert fingerprint("value", values=np.array([2.0], dtype=np.float64)) != baseline
    assert fingerprint("dtype", values=np.array([1.0], dtype=np.float32)) != baseline
    assert fingerprint("seed", values=np.array([1.0]), seed=5) != baseline
    assert fingerprint("config", values=np.array([1.0]), max_steps=9) != baseline


def test_stochastic_samples_are_generated_once_then_loaded(tmp_path: Path) -> None:
    samples = np.random.default_rng(31).normal(size=6)
    bundle = create_input_bundle(
        tmp_path,
        case_id="stochastic",
        random_seed=31,
        arrays={"samples": samples},
        configuration={"sample_count": 6},
    )

    _, first = load_input_bundle(tmp_path, bundle)
    _, second = load_input_bundle(tmp_path, bundle)

    np.testing.assert_array_equal(first["samples"], samples)
    np.testing.assert_array_equal(second["samples"], samples)
