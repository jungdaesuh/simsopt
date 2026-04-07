from dataclasses import dataclass

from simsopt.geo import SurfaceRZFourier


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
    vessel.set_rc(0, 0, 0.976)
    vessel.set_rc(1, 0, 0.222)
    vessel.set_zs(1, 0, 0.222)

    hbt = SurfaceRZFourier(nfp=nfp, stellsym=True)
    hbt.set_rc(0, 0, 0.9115)
    hbt.set_rc(1, 0, 0.1605)
    hbt.set_zs(1, 0, 0.152)

    coil_winding_surface = SurfaceRZFourier(nfp=nfp, stellsym=True)
    coil_winding_surface.set_rc(0, 0, 0.976)
    coil_winding_surface.set_rc(1, 0, banana_surf_radius)
    coil_winding_surface.set_zs(1, 0, banana_surf_radius)

    return BananaReferenceSurfaces(
        vessel=vessel,
        hbt=hbt,
        coil_winding_surface=coil_winding_surface,
    )
