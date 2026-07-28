"""Canonical host materialization for stochastic coil perturbations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.random import Generator, PCG64DXSM
from simsopt.field import Coil
from simsopt.geo import Curve, GaussianSampler
from simsopt_jax.examples import StochasticPerturbationBundle

from simsopt_jax_adapters.field._coil_graph import (
    _unwrap_coil_curve_and_current_objects,
)

_CANONICAL_DTYPE = np.dtype("<f8")


def materialize_stochastic_coil_perturbations(
    base_curves: Sequence[Curve],
    coils: Sequence[Coil],
    sampler: GaussianSampler,
    *,
    sample_count: int,
    seed: int,
) -> StochasticPerturbationBundle:
    """Replay native PCG64DXSM draw order into canonical combined samples."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sampler.n_derivs < 1:
        raise ValueError("sampler must include the first derivative")

    source_by_identity = {id(curve): index for index, curve in enumerate(base_curves)}
    if len(source_by_identity) != len(base_curves):
        raise ValueError("base_curves must contain distinct curve objects")
    unwrapped: list[tuple[int, np.ndarray]] = []
    for coil in coils:
        curve, rotation, _current, _scale = _unwrap_coil_curve_and_current_objects(
            coil.curve,
            coil.current,
        )
        source_index = source_by_identity.get(id(curve))
        if source_index is None:
            raise ValueError("every final coil must descend from one base curve")
        unwrapped.append(
            (
                source_index,
                np.eye(3, dtype=np.float64)
                if rotation is None
                else np.asarray(rotation, dtype=np.float64),
            )
        )

    point_count = len(sampler.points)
    expected_quadpoints = np.asarray(sampler.points, dtype=np.float64)
    for curve in base_curves:
        if not np.array_equal(
            np.asarray(curve.quadpoints, dtype=np.float64), expected_quadpoints
        ):
            raise ValueError("all base curves must use the sampler quadrature")

    source_gamma = tuple(
        np.asarray(curve.gamma(), dtype=np.float64) for curve in base_curves
    )
    source_gammadash = tuple(
        np.asarray(curve.gammadash(), dtype=np.float64) for curve in base_curves
    )
    shape = (sample_count, len(coils), point_count, 3)
    gamma = np.empty(shape, dtype=_CANONICAL_DTYPE)
    gammadash = np.empty(shape, dtype=_CANONICAL_DTYPE)
    random_generator = Generator(PCG64DXSM(seed))

    for sample_index in range(sample_count):
        systematic = tuple(
            sampler.draw_sample(random_generator) for _curve in base_curves
        )
        for coil_index, (source_index, rotation) in enumerate(unwrapped):
            statistical = sampler.draw_sample(random_generator)
            nominal_gamma = source_gamma[source_index]
            nominal_gammadash = source_gammadash[source_index]
            perturbed_gamma = nominal_gamma + systematic[source_index][0]
            perturbed_gammadash = (
                nominal_gammadash + systematic[source_index][1]
            )
            perturbed_gamma = perturbed_gamma @ rotation
            perturbed_gammadash = perturbed_gammadash @ rotation
            nominal_gamma = nominal_gamma @ rotation
            nominal_gammadash = nominal_gammadash @ rotation
            gamma[sample_index, coil_index] = (
                perturbed_gamma + statistical[0] - nominal_gamma
            )
            gammadash[sample_index, coil_index] = (
                perturbed_gammadash + statistical[1] - nominal_gammadash
            )

    return StochasticPerturbationBundle(
        gamma=gamma,
        gammadash=gammadash,
        seed=int(seed),
        base_curve_count=len(base_curves),
        source_indices=tuple(source_index for source_index, _ in unwrapped),
        rotations=np.stack(tuple(rotation for _, rotation in unwrapped)),
        sampler_points=np.asarray(sampler.points, dtype="<f8"),
        sigma=float(sampler.sigma),
        length_scale=float(sampler.length_scale),
        n_derivs=int(sampler.n_derivs),
    )


__all__ = (
    "materialize_stochastic_coil_perturbations",
)
