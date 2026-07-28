from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from examples.jax.parity.cases.surface_geometry import (
    _build_surface,
)
from examples.jax.parity.cases.surface_geometry import (
    _effective_fingerprint as surface_effective_fingerprint,
)
from examples.jax.parity.cases.surface_geometry import (
    create_input as create_surface_input,
)
from examples.jax.parity.cases.traceable_least_squares import (
    _effective_fingerprint as least_squares_effective_fingerprint,
)
from examples.jax.parity.cases.traceable_least_squares import (
    create_input as create_least_squares_input,
)
from examples.jax.parity.input_bundle import (
    create_input_bundle,
    load_input_bundle,
    read_input_bundle,
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


def test_real_case_construction_receipts_change_for_every_input_class(
    tmp_path: Path,
) -> None:
    least_squares_root = tmp_path / "least-squares"
    create_least_squares_input(least_squares_root, "bounded")
    bundle, arrays = read_input_bundle(least_squares_root)
    baseline = least_squares_effective_fingerprint(bundle, arrays)

    for field, mutated_bundle, mutated_arrays in (
        (
            "parameter",
            bundle,
            {**arrays, "initial_parameters": arrays["initial_parameters"] + 1.0},
        ),
        ("weight", bundle, {**arrays, "weights": arrays["weights"] + 1.0}),
        (
            "dtype",
            bundle,
            {**arrays, "targets": arrays["targets"].astype(np.float32)},
        ),
        ("seed", dataclasses.replace(bundle, random_seed=1), arrays),
        (
            "stopping option",
            dataclasses.replace(
                bundle,
                configuration={**bundle.configuration, "max_steps": 21},
            ),
            arrays,
        ),
    ):
        changed = least_squares_effective_fingerprint(mutated_bundle, mutated_arrays)
        assert changed != baseline, field

    surface_root = tmp_path / "surface"
    create_surface_input(surface_root, "bounded")
    surface_bundle, surface_arrays = read_input_bundle(surface_root)
    surface = _build_surface(surface_bundle, surface_arrays)
    surface_baseline = surface_effective_fingerprint(
        surface_bundle, surface_arrays, surface
    )
    changed_quadrature_arrays = {
        **surface_arrays,
        "quadrature": surface_arrays["quadrature"] + 1.0e-3,
    }
    changed_quadrature_surface = _build_surface(
        surface_bundle, changed_quadrature_arrays
    )
    assert (
        surface_effective_fingerprint(
            surface_bundle, changed_quadrature_arrays, changed_quadrature_surface
        )
        != surface_baseline
    )
    changed_constraint_arrays = {
        **surface_arrays,
        "targets": surface_arrays["targets"] + 1.0e-3,
    }
    assert (
        surface_effective_fingerprint(
            surface_bundle, changed_constraint_arrays, surface
        )
        != surface_baseline
    )
