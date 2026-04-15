from dataclasses import dataclass

from simsopt.geo import SurfaceRZFourier

from banana_opt.hardware_contracts import (
    VACUUM_VESSEL_MAJOR_RADIUS_M,
    VACUUM_VESSEL_MINOR_RADIUS_M,
)


@dataclass(frozen=True)
class BananaReferenceSurfaces:
    vessel: SurfaceRZFourier
    hbt: SurfaceRZFourier
    coil_winding_surface: SurfaceRZFourier


def build_banana_reference_surfaces(
    nfp: int,
    banana_surf_radius: float,
) -> BananaReferenceSurfaces:
    vessel = SurfaceRZFourier(nfp=nfp, stellsym=True)
    vessel.set_rc(0, 0, VACUUM_VESSEL_MAJOR_RADIUS_M)
    vessel.set_rc(1, 0, VACUUM_VESSEL_MINOR_RADIUS_M)
    vessel.set_zs(1, 0, VACUUM_VESSEL_MINOR_RADIUS_M)

    hbt = SurfaceRZFourier(nfp=nfp, stellsym=True)
    hbt.set_rc(0, 0, 0.9115)
    hbt.set_rc(1, 0, 0.1605)
    hbt.set_zs(1, 0, 0.152)

    coil_winding_surface = SurfaceRZFourier(nfp=nfp, stellsym=True)
    coil_winding_surface.set_rc(0, 0, VACUUM_VESSEL_MAJOR_RADIUS_M)
    coil_winding_surface.set_rc(1, 0, banana_surf_radius)
    coil_winding_surface.set_zs(1, 0, banana_surf_radius)

    return BananaReferenceSurfaces(
        vessel=vessel,
        hbt=hbt,
        coil_winding_surface=coil_winding_surface,
    )
