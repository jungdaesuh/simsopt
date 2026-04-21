from dataclasses import dataclass

from simsopt.geo import SurfaceRZFourier

from banana_opt.hardware_contracts import (
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
) -> BananaReferenceSurfaces:
    vessel = SurfaceRZFourier(nfp=nfp, stellsym=True)
    vessel.set_rc(0, 0, VACUUM_VESSEL_MAJOR_RADIUS_M)
    vessel.set_rc(1, 0, VACUUM_VESSEL_MINOR_RADIUS_M)
    vessel.set_zs(1, 0, VACUUM_VESSEL_MINOR_RADIUS_M)

    lcfs_clearance_reference = SurfaceRZFourier(nfp=nfp, stellsym=True)
    lcfs_clearance_reference.set_rc(0, 0, LCFS_CLEARANCE_REFERENCE_MAJOR_RADIUS_M)
    lcfs_clearance_reference.set_rc(1, 0, LCFS_CLEARANCE_REFERENCE_MINOR_RADIUS_M)
    lcfs_clearance_reference.set_zs(1, 0, LCFS_CLEARANCE_REFERENCE_MINOR_RADIUS_M)

    coil_winding_surface = SurfaceRZFourier(nfp=nfp, stellsym=True)
    coil_winding_surface.set_rc(0, 0, VACUUM_VESSEL_MAJOR_RADIUS_M)
    coil_winding_surface.set_rc(1, 0, banana_surf_radius)
    coil_winding_surface.set_zs(1, 0, banana_surf_radius)

    return BananaReferenceSurfaces(
        vessel=vessel,
        lcfs_clearance_reference=lcfs_clearance_reference,
        coil_winding_surface=coil_winding_surface,
    )
