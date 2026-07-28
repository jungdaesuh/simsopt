"""Native-compatible stochastic coil perturbation materialization."""

from __future__ import annotations

from numpy.random import Generator, PCG64DXSM
import numpy as np
from simsopt.field import Coil, Current, coils_via_symmetries
from simsopt.geo import (
    CurvePerturbed,
    GaussianSampler,
    PerturbationSample,
    create_equally_spaced_curves,
)

from simsopt_jax_adapters.geo import materialize_stochastic_coil_perturbations


def _native_sample(
    base_curves: list[object],
    base_currents: list[Current],
    sampler: GaussianSampler,
    *,
    seed: int,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray]:
    random_generator = Generator(PCG64DXSM(seed))
    base_perturbed = [
        CurvePerturbed(
            curve,
            PerturbationSample(sampler, randomgen=random_generator),
        )
        for curve in base_curves
    ]
    systematic_coils = coils_via_symmetries(
        base_perturbed,
        base_currents,
        nfp,
        True,
    )
    perturbed_coils = [
        Coil(
            CurvePerturbed(
                coil.curve,
                PerturbationSample(sampler, randomgen=random_generator),
            ),
            coil.current,
        )
        for coil in systematic_coils
    ]
    nominal_coils = coils_via_symmetries(
        base_curves,
        base_currents,
        nfp,
        True,
    )
    gamma = np.stack(
        [
            perturbed.curve.gamma() - nominal.curve.gamma()
            for perturbed, nominal in zip(
                perturbed_coils,
                nominal_coils,
                strict=True,
            )
        ]
    )
    gammadash = np.stack(
        [
            perturbed.curve.gammadash() - nominal.curve.gammadash()
            for perturbed, nominal in zip(
                perturbed_coils,
                nominal_coils,
                strict=True,
            )
        ]
    )
    return gamma, gammadash


def test_materialized_bundle_replays_native_draw_order_and_symmetries() -> None:
    nfp = 2
    base_curves = create_equally_spaced_curves(
        2,
        nfp,
        stellsym=True,
        R0=1.0,
        R1=0.25,
        order=2,
        numquadpoints=12,
    )
    base_currents = [Current(1.0e5) for _ in base_curves]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    sampler = GaussianSampler(
        base_curves[0].quadpoints,
        sigma=1.0e-3,
        length_scale=0.5,
        n_derivs=1,
    )

    bundle = materialize_stochastic_coil_perturbations(
        base_curves,
        coils,
        sampler,
        sample_count=2,
        seed=7,
    )
    expected_gamma, expected_gammadash = _native_sample(
        base_curves,
        base_currents,
        sampler,
        seed=7,
        nfp=nfp,
    )
    repeated = materialize_stochastic_coil_perturbations(
        base_curves,
        coils,
        sampler,
        sample_count=2,
        seed=7,
    )

    np.testing.assert_array_equal(bundle.gamma[0], expected_gamma)
    np.testing.assert_array_equal(bundle.gammadash[0], expected_gammadash)
    np.testing.assert_array_equal(bundle.gamma, repeated.gamma)
    np.testing.assert_array_equal(bundle.gammadash, repeated.gammadash)
    assert bundle.sha256 == repeated.sha256
    assert bundle.gamma.shape == (2, 8, 12, 3)
    assert bundle.gamma.dtype.str == "<f8"
    assert bundle.byte_order == "little"
    assert bundle.generator == "numpy.random.Generator(PCG64DXSM)"
    assert bundle.seed == 7
    assert not bundle.gamma.flags.writeable
    assert not bundle.gammadash.flags.writeable
