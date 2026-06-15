from dataclasses import dataclass

from simsopt.geo import SurfaceRZFourier

from banana_opt.hardware_contracts import (
    BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
    LCFS_CLEARANCE_REFERENCE_MAJOR_RADIUS_M,
    LCFS_CLEARANCE_REFERENCE_MINOR_RADIUS_M,
    VACUUM_VESSEL_MAJOR_RADIUS_M,
    VACUUM_VESSEL_MINOR_RADIUS_M,
)


@dataclass(frozen=True)
class BananaReferenceSurfaces:
    vessel: SurfaceRZFourier
    lcfs_clearance_reference: SurfaceRZFourier
    coil_winding_surface: SurfaceRZFourier


def build_banana_reference_surfaces(
    nfp: int,
    banana_surf_radius: float,
    banana_surf_major_radius: float = BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
    coil_winding_surface_mpol: int = 1,
    coil_winding_surface_ntor: int = 0,
) -> BananaReferenceSurfaces:
    """Build the vessel, LCFS-clearance, and coil-winding reference tori.

    banana_surf_radius: coil-winding surface MINOR radius (m).
    banana_surf_major_radius: coil-winding surface MAJOR radius (m); defaults to the
        on-spec sensor-array center ``BANANA_WINDING_SURFACE_MAJOR_RADIUS_M``. Only the
        coil-winding surface uses it — the vessel and the (fixed) LCFS-clearance band
        keep their hardware constants. Exposed so a winding-surface continuation can
        ramp the major radius alongside the minor.
    """
    vessel = SurfaceRZFourier(nfp=nfp, stellsym=True)
    vessel.set_rc(0, 0, VACUUM_VESSEL_MAJOR_RADIUS_M)
    vessel.set_rc(1, 0, VACUUM_VESSEL_MINOR_RADIUS_M)
    vessel.set_zs(1, 0, VACUUM_VESSEL_MINOR_RADIUS_M)

    lcfs_clearance_reference = SurfaceRZFourier(nfp=nfp, stellsym=True)
    lcfs_clearance_reference.set_rc(0, 0, LCFS_CLEARANCE_REFERENCE_MAJOR_RADIUS_M)
    lcfs_clearance_reference.set_rc(1, 0, LCFS_CLEARANCE_REFERENCE_MINOR_RADIUS_M)
    lcfs_clearance_reference.set_zs(1, 0, LCFS_CLEARANCE_REFERENCE_MINOR_RADIUS_M)

    if int(coil_winding_surface_mpol) < 1:
        raise ValueError("coil_winding_surface_mpol must be >= 1.")
    if int(coil_winding_surface_ntor) < 0:
        raise ValueError("coil_winding_surface_ntor must be >= 0.")

    coil_winding_surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=True,
        mpol=int(coil_winding_surface_mpol),
        ntor=int(coil_winding_surface_ntor),
    )
    coil_winding_surface.set_rc(0, 0, banana_surf_major_radius)
    coil_winding_surface.set_rc(1, 0, banana_surf_radius)
    coil_winding_surface.set_zs(1, 0, banana_surf_radius)

    return BananaReferenceSurfaces(
        vessel=vessel,
        lcfs_clearance_reference=lcfs_clearance_reference,
        coil_winding_surface=coil_winding_surface,
    )
