"""Build the archived 255×64 nested-LS problem from the F3 frozen bundle.

Gate 1 of the reduced nested-LS track. Coils come from the bundle's native
Biot-Savart JSON; the surface comes from the flat-675 runtime spec. This
does not evaluate F3's fused objective and does not inherit the 7.70× claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from simsopt._core.json import GSONDecoder
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier, Volume

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.flat675.bundle import load_flat675_bundle
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_NEWTON_MAXITER,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_WEIGHT_INV_MODB,
)

DEFAULT_FLAT675_BUNDLE_ROOT = (
    Path.home() / "simsopt_mixed_artifacts" / "genuine675-r3-input-1c23f6c5-20260721-r1"
)
NATIVE_BIOT_SAVART_FILENAME = "native_biot_savart.json"

# Reconstruct §2.1 archived-start QR inner state (C++ Newton was a no-op).
ARCHIVED_START_QR_IOTA = 0.1500517839808274
ARCHIVED_START_QR_G = 2.010619295609829


def archived_flat675_bundle_available(bundle_root: Path | None = None) -> bool:
    """True when the host-local F3 bundle and native Biot-Savart JSON exist."""

    root = DEFAULT_FLAT675_BUNDLE_ROOT if bundle_root is None else Path(bundle_root)
    return root.is_dir() and (root / NATIVE_BIOT_SAVART_FILENAME).is_file()


def _clone_tensor_surface(surface: SurfaceXYZTensorFourier) -> SurfaceXYZTensorFourier:
    cloned = SurfaceXYZTensorFourier(
        mpol=int(surface.mpol),
        ntor=int(surface.ntor),
        nfp=int(surface.nfp),
        stellsym=bool(surface.stellsym),
        clamped_dims=list(surface.clamped_dims),
        quadpoints_phi=np.asarray(surface.quadpoints_phi, dtype=np.float64).copy(),
        quadpoints_theta=np.asarray(surface.quadpoints_theta, dtype=np.float64).copy(),
    )
    cloned.set_dofs(np.asarray(surface.get_dofs(), dtype=np.float64).copy())
    return cloned


def _surface_from_template(template, surface_coordinates: NDArray[np.float64]):
    surface = SurfaceXYZTensorFourier(
        mpol=int(template.mpol),
        ntor=int(template.ntor),
        nfp=int(template.nfp),
        stellsym=bool(template.stellsym),
        clamped_dims=list(template.clamped_dims),
        quadpoints_phi=np.asarray(template.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(template.quadpoints_theta, dtype=np.float64),
    )
    surface.set_dofs(np.asarray(surface_coordinates, dtype=np.float64))
    return surface


def load_archived_nested_ls_pair(
    bundle_root: Path | None = None,
    *,
    coil_coordinates: object | None = None,
    surface_coordinates: object | None = None,
    materialize_dense_linearization: bool = False,
) -> tuple[BoozerSurface, BoozerSurfaceJAX, float]:
    """Native C++ and JAX LS surfaces on frozen archived coils.

    Default coordinates are the bundle start. Pass the F3 B37 coil and
    surface blocks to evaluate that frozen-coil point instead.

    Dense Hessian materialization is off: assembling ``H_ss`` at 661
    through autodiff-of-QR is a Gate 3 measurement, not a Gate 1
    stationarity check.
    """

    root = DEFAULT_FLAT675_BUNDLE_ROOT if bundle_root is None else Path(bundle_root)
    problem = load_flat675_bundle(root)
    biot_savart = json.loads(
        (root / NATIVE_BIOT_SAVART_FILENAME).read_text(),
        cls=GSONDecoder,
    )
    if not isinstance(biot_savart, BiotSavart):
        raise TypeError(
            "archived native_biot_savart.json must decode to BiotSavart; "
            f"got {type(biot_savart).__name__}."
        )
    template = problem.material.boozer.surface_template
    start = problem.start_candidate
    coils = (
        np.asarray(start.coil_coordinates, dtype=np.float64)
        if coil_coordinates is None
        else np.asarray(coil_coordinates, dtype=np.float64)
    )
    surface_dofs = (
        np.asarray(start.surface_coordinates, dtype=np.float64)
        if surface_coordinates is None
        else np.asarray(surface_coordinates, dtype=np.float64)
    )
    if coils.shape != biot_savart.x.shape:
        raise ValueError(
            "coil coordinates shape "
            f"{coils.shape} does not match archived BiotSavart.x "
            f"{biot_savart.x.shape}."
        )
    biot_savart.x = coils
    surface = _surface_from_template(template, surface_dofs)
    native_surface = _clone_tensor_surface(surface)
    jax_surface = _clone_tensor_surface(surface)
    target = float(problem.objective_policy.boozer_target_label)
    newton_options = {
        "verbose": False,
        "newton_tol": NESTED_LS_NEWTON_TOL,
        "newton_maxiter": NESTED_LS_NEWTON_MAXITER,
        "weight_inv_modB": NESTED_LS_WEIGHT_INV_MODB,
    }
    native = BoozerSurface(
        biot_savart,
        native_surface,
        Volume(native_surface),
        target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        options=newton_options,
    )
    jax_boozer = BoozerSurfaceJAX(
        BiotSavartJAX(biot_savart.coils),
        jax_surface,
        Volume(jax_surface),
        target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        options={
            **newton_options,
            "optimizer_backend": "scipy",
            "materialize_dense_linearization": bool(materialize_dense_linearization),
        },
    )
    return native, jax_boozer, target


__all__ = [
    "ARCHIVED_START_QR_G",
    "ARCHIVED_START_QR_IOTA",
    "DEFAULT_FLAT675_BUNDLE_ROOT",
    "archived_flat675_bundle_available",
    "load_archived_nested_ls_pair",
]
