"""Canonical native-compatible stochastic coil perturbation tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Protocol

import numpy as np
from numpy.random import PCG64DXSM, Generator
from numpy.typing import NDArray


class GaussianPerturbationSampler(Protocol):
    """Native Gaussian sampler surface needed by the materializer."""

    points: NDArray[np.float64]
    sigma: float
    length_scale: float
    n_derivs: int

    def draw_sample(
        self,
        randomgen: Generator | None = None,
    ) -> list[NDArray[np.float64]]: ...


def _readonly_float64(values: object) -> NDArray[np.float64]:
    result = np.array(values, dtype="<f8", order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class StochasticPerturbationBundle:
    """Ordered FP64 samples and all metadata needed to reproduce their bytes."""

    gamma: NDArray[np.float64]
    gammadash: NDArray[np.float64]
    seed: int
    base_curve_count: int
    source_indices: tuple[int, ...]
    rotations: NDArray[np.float64]
    sampler_points: NDArray[np.float64]
    sigma: float
    length_scale: float
    n_derivs: int
    generator: str = field(init=False, default="PCG64DXSM")
    dtype: str = field(init=False, default="<f8")
    byte_order: str = field(init=False, default="little")
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        gamma = _readonly_float64(self.gamma)
        gammadash = _readonly_float64(self.gammadash)
        rotations = _readonly_float64(self.rotations)
        sampler_points = _readonly_float64(self.sampler_points)
        if gamma.ndim != 4 or gamma.shape[-1] != 3:
            raise ValueError("gamma must have shape (samples, coils, points, 3)")
        if gammadash.shape != gamma.shape:
            raise ValueError("gammadash must have the same shape as gamma")
        if gamma.shape[0] < 1:
            raise ValueError("at least one stochastic sample is required")
        if gamma.shape[1] != len(self.source_indices):
            raise ValueError("one source index is required for each final coil")
        if rotations.shape != (gamma.shape[1], 3, 3):
            raise ValueError("rotations must have shape (coils, 3, 3)")
        if sampler_points.shape != (gamma.shape[2],):
            raise ValueError("sampler points must match the perturbation quadrature")
        if self.base_curve_count < 1:
            raise ValueError("base_curve_count must be positive")
        if any(
            source < 0 or source >= self.base_curve_count
            for source in self.source_indices
        ):
            raise ValueError("source index is outside the base-curve range")
        if self.n_derivs < 1:
            raise ValueError("the sampler must provide first derivatives")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "gammadash", gammadash)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "sampler_points", sampler_points)
        object.__setattr__(self, "sha256", self._fingerprint())

    @property
    def sample_count(self) -> int:
        """Number of ordered coil-set realizations."""
        return int(self.gamma.shape[0])

    def _fingerprint(self) -> str:
        metadata = {
            "base_curve_count": self.base_curve_count,
            "byte_order": self.byte_order,
            "dtype": self.dtype,
            "gamma_shape": self.gamma.shape,
            "generator": self.generator,
            "length_scale": self.length_scale,
            "n_derivs": self.n_derivs,
            "schema_version": 1,
            "seed": self.seed,
            "sigma": self.sigma,
            "source_indices": self.source_indices,
        }
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                metadata,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        for name, values in (
            ("rotations", self.rotations),
            ("sampler_points", self.sampler_points),
            ("gamma", self.gamma),
            ("gammadash", self.gammadash),
        ):
            digest.update(name.encode("ascii"))
            digest.update(values.tobytes(order="C"))
        return digest.hexdigest()


def materialize_stochastic_coil_perturbations(
    sampler: GaussianPerturbationSampler,
    *,
    source_indices: tuple[int, ...],
    rotations: NDArray[np.float64],
    base_curve_count: int,
    sample_count: int,
    seed: int,
) -> StochasticPerturbationBundle:
    """Draw native-ordered systematic then independent statistical errors."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sampler.n_derivs < 1:
        raise ValueError("the sampler must provide first derivatives")
    rotations_array = np.asarray(rotations, dtype="<f8")
    coil_count = len(source_indices)
    if rotations_array.shape != (coil_count, 3, 3):
        raise ValueError("rotations must have shape (coils, 3, 3)")
    if base_curve_count < 1:
        raise ValueError("base_curve_count must be positive")
    if any(source < 0 or source >= base_curve_count for source in source_indices):
        raise ValueError("source index is outside the base-curve range")
    point_count = int(np.asarray(sampler.points).shape[0])
    gamma = np.empty((sample_count, coil_count, point_count, 3), dtype="<f8")
    gammadash = np.empty_like(gamma)
    randomgen = Generator(PCG64DXSM(seed))
    for sample_index in range(sample_count):
        systematic = tuple(
            sampler.draw_sample(randomgen) for _ in range(base_curve_count)
        )
        for coil_index, (source_index, rotation) in enumerate(
            zip(source_indices, rotations_array, strict=True)
        ):
            statistical = sampler.draw_sample(randomgen)
            gamma[sample_index, coil_index] = (
                systematic[source_index][0] @ rotation + statistical[0]
            )
            gammadash[sample_index, coil_index] = (
                systematic[source_index][1] @ rotation + statistical[1]
            )
    return StochasticPerturbationBundle(
        gamma=gamma,
        gammadash=gammadash,
        seed=seed,
        base_curve_count=base_curve_count,
        source_indices=source_indices,
        rotations=rotations_array,
        sampler_points=np.asarray(sampler.points, dtype="<f8"),
        sigma=float(sampler.sigma),
        length_scale=float(sampler.length_scale),
        n_derivs=int(sampler.n_derivs),
    )


__all__ = (
    "GaussianPerturbationSampler",
    "StochasticPerturbationBundle",
    "materialize_stochastic_coil_perturbations",
)
