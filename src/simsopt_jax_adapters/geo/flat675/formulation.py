"""Coordinate layout of the genuine-675 flat single-stage formulation.

One outer vector carries every optimized coordinate in a fixed contiguous
order: coil DOFs, then vessel DOFs, then boundary DOFs.  The widths are a
property of the problem, not of this module: :mod:`.layout` derives them from
the coil owner map and the surface resolution, and the constants published
here are that derivation evaluated at the certified configuration.

"675" names the certified configuration, not a constraint.  Every
``FLAT675_*`` name below is :data:`~.layout.CERTIFIED_FLAT_LAYOUT` read out
under the spelling the sealed receipts, the campaign children, the shipped
example and the tests already use, so those consumers see no change while the
core reads a per-problem record.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .layout import CERTIFIED_FLAT_LAYOUT, FlatSingleStageLayout

FLAT675_COIL_DOF_COUNT: Final[int] = CERTIFIED_FLAT_LAYOUT.coil_dof_count
FLAT675_VESSEL_DOF_COUNT: Final[int] = CERTIFIED_FLAT_LAYOUT.vessel_dof_count
FLAT675_SURFACE_DOF_COUNT: Final[int] = CERTIFIED_FLAT_LAYOUT.surface_dof_count
FLAT675_OUTER_DOF_COUNT: Final[int] = CERTIFIED_FLAT_LAYOUT.outer_dof_count

FLAT675_COIL_SLICE: Final[slice] = CERTIFIED_FLAT_LAYOUT.coil_slice
FLAT675_VESSEL_SLICE: Final[slice] = CERTIFIED_FLAT_LAYOUT.vessel_slice
FLAT675_SURFACE_SLICE: Final[slice] = CERTIFIED_FLAT_LAYOUT.surface_slice


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
    place it becomes the flat outer vector the objective consumes.  The
    ``layout`` names the block widths it must satisfy and defaults to the
    certified configuration, so existing callers construct exactly what they
    always did.

    Untrusted coordinates enter through :meth:`from_payload`, which is where
    every value is proved finite; direct construction from typed floats is
    checked for the block sizes only.
    """

    coil_coordinates: tuple[float, ...]
    vessel_coordinates: tuple[float, ...]
    surface_coordinates: tuple[float, ...]
    layout: FlatSingleStageLayout = CERTIFIED_FLAT_LAYOUT

    def __post_init__(self) -> None:
        for name, expected_count in self.layout.block_widths():
            block = getattr(self, name)
            if not isinstance(block, tuple) or len(block) != expected_count:
                raise Flat675ContractError(
                    f"candidate.{name} must be a tuple of exactly "
                    f"{expected_count} coordinates."
                )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        layout: FlatSingleStageLayout = CERTIFIED_FLAT_LAYOUT,
    ) -> Flat675Candidate:
        """Build one candidate from an untrusted coordinate-block mapping."""
        missing = sorted(
            {"coil_coordinates", "vessel_coordinates", "surface_coordinates"}
            - frozenset(payload)
        )
        if missing:
            raise Flat675ContractError(f"candidate is missing {missing!r}.")
        widths = dict(layout.block_widths())
        return cls(
            coil_coordinates=_finite_tuple(
                payload["coil_coordinates"],
                expected_count=widths["coil_coordinates"],
                where="candidate.coil_coordinates",
            ),
            vessel_coordinates=_finite_tuple(
                payload["vessel_coordinates"],
                expected_count=widths["vessel_coordinates"],
                where="candidate.vessel_coordinates",
            ),
            surface_coordinates=_finite_tuple(
                payload["surface_coordinates"],
                expected_count=widths["surface_coordinates"],
                where="candidate.surface_coordinates",
            ),
            layout=layout,
        )

    def outer_vector(self) -> NDArray[np.float64]:
        """Return the coil/vessel/surface blocks as one float64 outer vector."""
        return np.concatenate(
            (
                np.asarray(self.coil_coordinates, dtype=np.float64),
                np.asarray(self.vessel_coordinates, dtype=np.float64),
                np.asarray(self.surface_coordinates, dtype=np.float64),
            )
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
