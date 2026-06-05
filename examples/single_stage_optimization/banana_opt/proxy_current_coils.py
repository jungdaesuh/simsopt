from __future__ import annotations

from typing import Literal, Protocol

from scipy.io import netcdf_file

from simsopt.field import Coil, Current
from simsopt.geo import CurveXYZFourier


ProxyCurrentPlacement = Literal["axis_fourier_zeroth", "surface_major_radius"]


class MajorRadiusSurface(Protocol):
    def major_radius(self) -> float:
        ...


def build_planar_proxy_plasma_current_coils(
    *,
    placement: ProxyCurrentPlacement,
    plasma_current_A: float,
    equilibrium_file: str | None = None,
    surface_scale_factor: float = 1.0,
    nphi: int = 0,
    ntheta: int = 0,
    toroidal_flux: float | None = None,
    surface: MajorRadiusSurface | None = None,
) -> list[Coil]:
    """Build the planar proxy line current for finite-current design fields.

    ``axis_fourier_zeroth`` is Wataru's scaled VMEC-axis zeroth coefficient.
    ``surface_major_radius`` is the historical jhalpern30 major-radius ring.
    Both placements are prescribed design fields and must be marked separately.
    """

    if placement == "axis_fourier_zeroth":
        if equilibrium_file is None:
            raise ValueError("axis_fourier_zeroth placement requires equilibrium_file.")
        if toroidal_flux is None:
            raise ValueError("axis_fourier_zeroth placement requires toroidal_flux.")
        _validate_normalized_toroidal_flux(
            toroidal_flux,
            field_name="proxy plasma-current toroidal_flux",
        )
        with netcdf_file(equilibrium_file, mmap=False) as equilibrium_netcdf:
            raxis_cc = equilibrium_netcdf.variables["raxis_cc"][:].copy()
            zaxis_cs = equilibrium_netcdf.variables["zaxis_cs"][:].copy()
        axis_scale = float(surface_scale_factor)
        return [
            _build_planar_ring_proxy_current_coil(
                major_radius_m=float(raxis_cc[0]) * axis_scale,
                midplane_z_m=float(zaxis_cs[0]) * axis_scale,
                plasma_current_A=float(plasma_current_A),
            )
        ]
    if placement == "surface_major_radius":
        if surface is None:
            raise ValueError("surface_major_radius placement requires surface.")
        return [
            _build_planar_ring_proxy_current_coil(
                major_radius_m=float(surface.major_radius()),
                midplane_z_m=0.0,
                plasma_current_A=float(plasma_current_A),
            )
        ]
    raise ValueError(f"Unsupported proxy-current placement {placement!r}.")


def _build_planar_ring_proxy_current_coil(
    *,
    major_radius_m: float,
    midplane_z_m: float,
    plasma_current_A: float,
) -> Coil:
    proxy_curve = CurveXYZFourier(128, 1)
    proxy_curve.set("xc(1)", float(major_radius_m))
    proxy_curve.set("ys(1)", float(major_radius_m))
    proxy_curve.set("zc(0)", float(midplane_z_m))
    proxy_curve.fix_all()
    proxy_current = Current(float(plasma_current_A))
    proxy_current.fix_all()
    return Coil(proxy_curve, proxy_current)


def _validate_normalized_toroidal_flux(
    value: float,
    *,
    field_name: str,
) -> float:
    flux = float(value)
    if not 0.0 <= flux <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1 inclusive, got {value!r}."
        )
    return flux
