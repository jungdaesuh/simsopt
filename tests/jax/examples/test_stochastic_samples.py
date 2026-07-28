"""Canonical native-compatible stochastic sample materialization contracts."""

from __future__ import annotations

from numpy.random import PCG64DXSM, Generator
import numpy as np
import pytest
from simsopt.geo import GaussianSampler, PerturbationSample

from simsopt_jax.examples import materialize_stochastic_coil_perturbations


def _rotation(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    ).T


def test_materialized_samples_match_native_draw_order_and_symmetry() -> None:
    sampler = GaussianSampler(
        np.linspace(0.0, 1.0, 8, endpoint=False),
        sigma=1.0e-3,
        length_scale=0.5,
        n_derivs=1,
    )
    source_indices = (0, 1, 0, 1)
    rotations = np.stack(
        (np.eye(3), np.eye(3), _rotation(np.pi), _rotation(np.pi))
    )
    bundle = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=source_indices,
        rotations=rotations,
        base_curve_count=2,
        sample_count=3,
        seed=7,
    )

    randomgen = Generator(PCG64DXSM(7))
    expected_gamma = []
    expected_gammadash = []
    for _ in range(3):
        systematic = tuple(
            PerturbationSample(sampler, randomgen=randomgen) for _ in range(2)
        )
        statistical = tuple(
            PerturbationSample(sampler, randomgen=randomgen) for _ in source_indices
        )
        expected_gamma.append(
            np.stack(
                tuple(
                    systematic[source][0] @ rotation + independent[0]
                    for source, rotation, independent in zip(
                        source_indices,
                        rotations,
                        statistical,
                        strict=True,
                    )
                )
            )
        )
        expected_gammadash.append(
            np.stack(
                tuple(
                    systematic[source][1] @ rotation + independent[1]
                    for source, rotation, independent in zip(
                        source_indices,
                        rotations,
                        statistical,
                        strict=True,
                    )
                )
            )
        )

    np.testing.assert_array_equal(bundle.gamma, np.stack(expected_gamma))
    np.testing.assert_array_equal(bundle.gammadash, np.stack(expected_gammadash))
    assert bundle.generator == "PCG64DXSM"
    assert bundle.seed == 7
    assert bundle.dtype == "<f8"
    assert bundle.byte_order == "little"
    assert len(bundle.sha256) == 64
    assert not bundle.gamma.flags.writeable
    assert not bundle.gammadash.flags.writeable


def test_materialized_sample_hash_binds_values_and_metadata() -> None:
    sampler = GaussianSampler(
        np.linspace(0.0, 1.0, 4, endpoint=False),
        sigma=1.0e-3,
        length_scale=0.5,
        n_derivs=1,
    )
    arguments = {
        "source_indices": (0,),
        "rotations": np.eye(3)[None, :, :],
        "base_curve_count": 1,
        "sample_count": 1,
    }

    first = materialize_stochastic_coil_perturbations(sampler, seed=0, **arguments)
    repeated = materialize_stochastic_coil_perturbations(
        sampler, seed=0, **arguments
    )
    changed = materialize_stochastic_coil_perturbations(sampler, seed=1, **arguments)

    assert first.sha256 == repeated.sha256
    assert changed.sha256 != first.sha256
    with pytest.raises(ValueError, match="read-only"):
        first.gamma[0, 0, 0, 0] = 1.0
