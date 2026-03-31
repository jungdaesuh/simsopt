"""Pure JAX kernel-layer specs and helpers."""

from .field import (
    grouped_biot_savart_A_from_spec,
    grouped_biot_savart_B_and_dB_from_spec,
    grouped_biot_savart_B_from_spec,
    grouped_biot_savart_dB_by_dX_from_spec,
    grouped_coil_set_spec_from_lists,
)
from .objectives_flux import (
    build_fourier_basis,
    fixed_surface_flux_integral,
    fixed_surface_flux_integral_from_B,
)
from .surface_rzfourier import (
    surface_rz_fourier_area_from_spec,
    surface_rz_fourier_gamma_from_spec,
    surface_rz_fourier_gammadash1_from_spec,
    surface_rz_fourier_gammadash2_from_spec,
    surface_rz_fourier_normal_from_spec,
    surface_rz_fourier_unitnormal_from_spec,
    surface_rz_fourier_volume_from_spec,
)
from .specs import (
    CoilGroupSpec,
    FixedSurfaceFluxSpec,
    GroupedCoilSetSpec,
    SurfaceRZFourierSpec,
    make_coil_group_spec,
    make_fixed_surface_flux_spec,
    make_grouped_coil_set_spec,
    make_surface_rzfourier_spec,
)

__all__ = [
    "CoilGroupSpec",
    "FixedSurfaceFluxSpec",
    "GroupedCoilSetSpec",
    "SurfaceRZFourierSpec",
    "build_fourier_basis",
    "fixed_surface_flux_integral",
    "fixed_surface_flux_integral_from_B",
    "grouped_biot_savart_A_from_spec",
    "grouped_biot_savart_B_and_dB_from_spec",
    "grouped_biot_savart_B_from_spec",
    "grouped_biot_savart_dB_by_dX_from_spec",
    "make_coil_group_spec",
    "make_fixed_surface_flux_spec",
    "make_grouped_coil_set_spec",
    "make_surface_rzfourier_spec",
    "grouped_coil_set_spec_from_lists",
    "surface_rz_fourier_area_from_spec",
    "surface_rz_fourier_gamma_from_spec",
    "surface_rz_fourier_gammadash1_from_spec",
    "surface_rz_fourier_gammadash2_from_spec",
    "surface_rz_fourier_normal_from_spec",
    "surface_rz_fourier_unitnormal_from_spec",
    "surface_rz_fourier_volume_from_spec",
]
