"""Canonical host materialization for stochastic coil perturbations."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.random import Generator, PCG64DXSM
from simsopt.field import Coil
from simsopt.geo import Curve, GaussianSampler

from simsopt_jax_adapters.field._coil_graph import (
    _unwrap_coil_curve_and_current_objects,
)

_CANONICAL_DTYPE = np.dtype("<f8")
_GENERATOR_NAME = "numpy.random.Generator(PCG64DXSM)"
_ORDERING = "sample:systematic-base-curve,statistical-final-coil"


def _canonical_array(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=_CANONICAL_DTYPE, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class MaterializedCoilPerturbations:
    """Read-only FP64 samples shared verbatim by every execution lane."""

    gamma: np.ndarray
    gammadash: np.ndarray
    seed: int
    sigma: float
    length_scale: float
    n_derivs: int
    generator: str = _GENERATOR_NAME
    ordering: str = _ORDERING
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        gamma = _canonical_array(self.gamma)
        gammadash = _canonical_array(self.gammadash)
        if gamma.ndim != 4 or gamma.shape[-1] != 3:
            raise ValueError("gamma must have shape (samples, coils, points, 3)")
        if gammadash.shape != gamma.shape:
            raise ValueError("gammadash must have the same shape as gamma")
        if gamma.shape[0] < 1:
            raise ValueError("at least one stochastic sample is required")
        if self.n_derivs < 1:
            raise ValueError("n_derivs must include the first derivative")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "gammadash", gammadash)
        object.__setattr__(self, "sha256", self._fingerprint())

    @property
    def byte_order(self) -> str:
        """Return the canonical payload byte order."""
        return "little"

    @property
    def sample_count(self) -> int:
        return int(self.gamma.shape[0])

    @property
    def coil_count(self) -> int:
        return int(self.gamma.shape[1])

    @property
    def point_count(self) -> int:
        return int(self.gamma.shape[2])

    def metadata(self) -> dict[str, object]:
        """Return the complete JSON-serializable scientific-input identity."""
        return {
            "byte_order": self.byte_order,
            "coil_count": self.coil_count,
            "dtype": self.gamma.dtype.str,
            "gamma_shape": self.gamma.shape,
            "gammadash_shape": self.gammadash.shape,
            "generator": self.generator,
            "length_scale": self.length_scale,
            "n_derivs": self.n_derivs,
            "ordering": self.ordering,
            "point_count": self.point_count,
            "sample_count": self.sample_count,
            "seed": self.seed,
            "sigma": self.sigma,
        }

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"simsopt-jax-stochastic-coil-perturbations-v1\0")
        digest.update(
            json.dumps(
                self.metadata(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(self.gamma.tobytes(order="C"))
        digest.update(self.gammadash.tobytes(order="C"))
        return digest.hexdigest()

    def write(self, path: Path) -> None:
        """Persist an immutable bundle without pickle-dependent object state."""
        np.savez(
            path,
            gamma=self.gamma,
            gammadash=self.gammadash,
            metadata=np.asarray(
                json.dumps(
                    {**self.metadata(), "sha256": self.sha256},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        )


def materialize_stochastic_coil_perturbations(
    base_curves: Sequence[Curve],
    coils: Sequence[Coil],
    sampler: GaussianSampler,
    *,
    sample_count: int,
    seed: int,
) -> MaterializedCoilPerturbations:
    """Replay native PCG64DXSM draw order into canonical combined samples."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sampler.n_derivs < 1:
        raise ValueError("sampler must include the first derivative")

    source_by_identity = {id(curve): index for index, curve in enumerate(base_curves)}
    if len(source_by_identity) != len(base_curves):
        raise ValueError("base_curves must contain distinct curve objects")
    unwrapped = []
    for coil in coils:
        curve, rotation, _current, _scale = _unwrap_coil_curve_and_current_objects(
            coil.curve,
            coil.current,
        )
        source_index = source_by_identity.get(id(curve))
        if source_index is None:
            raise ValueError("every final coil must descend from one base curve")
        unwrapped.append((source_index, rotation))

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
            if rotation is not None:
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

    return MaterializedCoilPerturbations(
        gamma=gamma,
        gammadash=gammadash,
        seed=int(seed),
        sigma=float(sampler.sigma),
        length_scale=float(sampler.length_scale),
        n_derivs=int(sampler.n_derivs),
    )


__all__ = (
    "MaterializedCoilPerturbations",
    "materialize_stochastic_coil_perturbations",
)
