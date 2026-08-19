"""Coordinate layout of the genuine-675 flat single-stage formulation.

One outer vector carries every optimized coordinate in a fixed contiguous
order: 11 coil DOFs, then 3 vessel DOFs, then 661 surface DOFs.  This module
is the only place that spells those offsets; every other module in the port
slices through the constants published here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np
from numpy.typing import NDArray

FLAT675_COIL_DOF_COUNT: Final[int] = 11
FLAT675_VESSEL_DOF_COUNT: Final[int] = 3
FLAT675_SURFACE_DOF_COUNT: Final[int] = 661
FLAT675_OUTER_DOF_COUNT: Final[int] = (
    FLAT675_COIL_DOF_COUNT + FLAT675_VESSEL_DOF_COUNT + FLAT675_SURFACE_DOF_COUNT
)

FLAT675_COIL_SLICE: Final[slice] = slice(0, FLAT675_COIL_DOF_COUNT)
FLAT675_VESSEL_SLICE: Final[slice] = slice(
    FLAT675_COIL_SLICE.stop,
    FLAT675_COIL_SLICE.stop + FLAT675_VESSEL_DOF_COUNT,
)
FLAT675_SURFACE_SLICE: Final[slice] = slice(
    FLAT675_VESSEL_SLICE.stop,
    FLAT675_VESSEL_SLICE.stop + FLAT675_SURFACE_DOF_COUNT,
)


class Flat675ContractError(ValueError):
    """Raised when flat-675 material violates the formulation's contract."""


def flat675_finite_float(value: object, where: str) -> float:
    """Return ``value`` as a finite float or fail with a located message."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise Flat675ContractError(f"{where} must be a real scalar.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise Flat675ContractError(f"{where} must be finite.")
    return scalar


def _finite_tuple(
    values: object,
    *,
    expected_count: int,
    where: str,
) -> tuple[float, ...]:
    """Return exactly ``expected_count`` finite floats from a JSON sequence."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise Flat675ContractError(f"{where} must be a sequence of scalars.")
    normalized = tuple(
        flat675_finite_float(value, f"{where}[{index}]")
        for index, value in enumerate(values)
    )
    if len(normalized) != expected_count:
        raise Flat675ContractError(
            f"{where} must contain exactly {expected_count} coordinates."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class Flat675Candidate:
    """One outer point held in its three physical owner blocks.

    The blocks are host tuples, not device arrays: a candidate is an input
    record that outlives any single trace, and ``outer_vector`` is the only
    place it becomes the flat 675 vector the objective consumes.

    Untrusted coordinates enter through :meth:`from_payload`, which is where
    every value is proved finite; direct construction from typed floats is
    checked for the block sizes only.
    """

    coil_coordinates: tuple[float, ...]
    vessel_coordinates: tuple[float, ...]
    surface_coordinates: tuple[float, ...]

    def __post_init__(self) -> None:
        for name, expected_count in (
            ("coil_coordinates", FLAT675_COIL_DOF_COUNT),
            ("vessel_coordinates", FLAT675_VESSEL_DOF_COUNT),
            ("surface_coordinates", FLAT675_SURFACE_DOF_COUNT),
        ):
            block = getattr(self, name)
            if not isinstance(block, tuple) or len(block) != expected_count:
                raise Flat675ContractError(
                    f"candidate.{name} must be a tuple of exactly "
                    f"{expected_count} coordinates."
                )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Flat675Candidate:
        """Build one candidate from an untrusted coordinate-block mapping."""
        missing = sorted(
            {"coil_coordinates", "vessel_coordinates", "surface_coordinates"}
            - frozenset(payload)
        )
        if missing:
            raise Flat675ContractError(f"candidate is missing {missing!r}.")
        return cls(
            coil_coordinates=_finite_tuple(
                payload["coil_coordinates"],
                expected_count=FLAT675_COIL_DOF_COUNT,
                where="candidate.coil_coordinates",
            ),
            vessel_coordinates=_finite_tuple(
                payload["vessel_coordinates"],
                expected_count=FLAT675_VESSEL_DOF_COUNT,
                where="candidate.vessel_coordinates",
            ),
            surface_coordinates=_finite_tuple(
                payload["surface_coordinates"],
                expected_count=FLAT675_SURFACE_DOF_COUNT,
                where="candidate.surface_coordinates",
            ),
        )

    def outer_vector(self) -> NDArray[np.float64]:
        """Return the coil/vessel/surface blocks as one float64 675-vector."""
        return np.concatenate(
            (
                np.asarray(self.coil_coordinates, dtype=np.float64),
                np.asarray(self.vessel_coordinates, dtype=np.float64),
                np.asarray(self.surface_coordinates, dtype=np.float64),
            )
        )

    @classmethod
    def from_outer_vector(cls, values: object) -> Flat675Candidate:
        """Split one flat 675-vector back into its three owner blocks."""
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (FLAT675_OUTER_DOF_COUNT,):
            raise Flat675ContractError(
                "flat-675 outer vector must have shape "
                f"({FLAT675_OUTER_DOF_COUNT},); got {vector.shape}."
            )
        return cls(
            coil_coordinates=tuple(vector[FLAT675_COIL_SLICE].tolist()),
            vessel_coordinates=tuple(vector[FLAT675_VESSEL_SLICE].tolist()),
            surface_coordinates=tuple(vector[FLAT675_SURFACE_SLICE].tolist()),
        )


__all__ = [
    "FLAT675_COIL_DOF_COUNT",
    "FLAT675_COIL_SLICE",
    "FLAT675_OUTER_DOF_COUNT",
    "FLAT675_SURFACE_DOF_COUNT",
    "FLAT675_SURFACE_SLICE",
    "FLAT675_VESSEL_DOF_COUNT",
    "FLAT675_VESSEL_SLICE",
    "Flat675Candidate",
    "Flat675ContractError",
    "flat675_finite_float",
]
