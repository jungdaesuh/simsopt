"""The per-problem block layout of a flat coupled single-stage vector.

The flat formulation's outer vector is three contiguous blocks — coils,
vessel, boundary — and every width in it follows from two things: how many
owner DOFs the coil set exposes, and what surface resolution the problem
targets.  This module owns that derivation once, so that a problem's widths
and the slices that read them cannot disagree.

"675" names the certified configuration, not a constraint: the distinguished
:data:`CERTIFIED_FLAT_LAYOUT` is one instance of :class:`FlatSingleStageLayout`
among many, and it is the instance the sealed receipts, the campaign children
and the shipped example consume.  The historic ``FLAT675_*`` constants are its
widths and slices, re-exported unchanged from :mod:`.formulation`.

The surface width is **derived, never supplied**.  It is not a field of the
record: it is a property computed from ``(mpol, ntor, stellsym)``, so a caller
cannot construct a record whose declared width disagrees with its resolution.
The two symmetry modes have different counts because they parameterize
different coefficient sets — the same split the surface kernels take when they
choose between the compact stellarator-symmetric scatter and the full
three-block unpacking.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from simsopt_jax.core.surface_fourier_indices import stellsym_scatter_indices

# The vessel template is a three-free-DOF record by construction (major
# radius, and the circular cross-section's two poloidal amplitudes), so the
# vessel block is a constant of the formulation rather than a per-problem
# width.  Generalizing it is out of scope.
FLAT_VESSEL_DOF_COUNT: Final[int] = 3


class FlatLayoutError(ValueError):
    """Raised when a requested flat single-stage layout is not constructible."""


@lru_cache(maxsize=None)
def surface_block_dof_count(*, mpol: int, ntor: int, stellsym: bool) -> int:
    """The number of boundary DOFs a tensor-Fourier resolution exposes.

    This is the repository's own count, not a reimplementation of it: under
    stellarator symmetry it is the length of the compact scatter map the
    surface kernels index with, and without symmetry it is the full
    ``3 x (2*mpol+1) x (2*ntor+1)`` coefficient block those kernels unpack.
    Both agree with ``SurfaceXYZTensorFourier.get_dofs()`` at every
    resolution, which is the equality the layout tests pin.
    """
    if mpol < 1:
        raise FlatLayoutError(f"a boundary needs mpol >= 1; got {mpol}.")
    if ntor < 0:
        raise FlatLayoutError(f"a boundary needs ntor >= 0; got {ntor}.")
    if stellsym:
        return int(stellsym_scatter_indices(int(mpol), int(ntor)).shape[0])
    return 3 * (2 * int(mpol) + 1) * (2 * int(ntor) + 1)


@dataclass(frozen=True, slots=True)
class FlatSingleStageLayout:
    """One problem's block widths, derived from its own resolution and coils.

    The record carries the surface resolution triple it was built from beside
    the widths that triple produces, so a reader never has to look elsewhere
    to learn why a block is the size it is.
    """

    coil_dof_count: int
    surface_mpol: int
    surface_ntor: int
    surface_stellsym: bool

    def __post_init__(self) -> None:
        if self.coil_dof_count < 1:
            raise FlatLayoutError(
                "a flat single-stage problem needs at least one coil owner "
                f"DOF; got {self.coil_dof_count}."
            )
        # Constructing the width here fails an impossible resolution at record
        # construction rather than at the first evaluation.
        surface_block_dof_count(
            mpol=self.surface_mpol,
            ntor=self.surface_ntor,
            stellsym=self.surface_stellsym,
        )

    @property
    def vessel_dof_count(self) -> int:
        return FLAT_VESSEL_DOF_COUNT

    @property
    def surface_dof_count(self) -> int:
        """Derived from the resolution triple; never supplied independently."""
        return surface_block_dof_count(
            mpol=self.surface_mpol,
            ntor=self.surface_ntor,
            stellsym=self.surface_stellsym,
        )

    @property
    def outer_dof_count(self) -> int:
        return self.coil_dof_count + self.vessel_dof_count + self.surface_dof_count

    @property
    def coil_slice(self) -> slice:
        return slice(0, self.coil_dof_count)

    @property
    def vessel_slice(self) -> slice:
        return slice(self.coil_dof_count, self.coil_dof_count + self.vessel_dof_count)

    @property
    def surface_slice(self) -> slice:
        start = self.coil_dof_count + self.vessel_dof_count
        return slice(start, start + self.surface_dof_count)

    def block_widths(self) -> tuple[tuple[str, int], ...]:
        """The per-block width table, in the outer vector's own order."""
        return (
            ("coil_coordinates", self.coil_dof_count),
            ("vessel_coordinates", self.vessel_dof_count),
            ("surface_coordinates", self.surface_dof_count),
        )


# The distinguished instance: the configuration the sealed receipts, the
# campaign children, the shipped example and the frozen bundle all speak.
# Its triple pins the 661, which is why no width is written here.
CERTIFIED_FLAT_LAYOUT: Final[FlatSingleStageLayout] = FlatSingleStageLayout(
    coil_dof_count=11,
    surface_mpol=10,
    surface_ntor=10,
    surface_stellsym=True,
)


__all__ = [
    "CERTIFIED_FLAT_LAYOUT",
    "FLAT_VESSEL_DOF_COUNT",
    "FlatLayoutError",
    "FlatSingleStageLayout",
    "surface_block_dof_count",
]
