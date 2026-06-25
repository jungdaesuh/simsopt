"""Shared Boozer benchmark/parity problem builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunCodeProblem:
    coils: list
    surface: object
    volume: object
    vol_target: float
    iota0: float
    G0: float


def clone_tensor_surface(surface):
    """Return an independent SurfaceXYZTensorFourier copy with identical DOFs."""
    surface_copy = type(surface)(
        mpol=surface.mpol,
        ntor=surface.ntor,
        stellsym=surface.stellsym,
        nfp=surface.nfp,
        quadpoints_phi=np.array(surface.quadpoints_phi, copy=True),
        quadpoints_theta=np.array(surface.quadpoints_theta, copy=True),
    )
    surface_copy.set_dofs(np.array(surface.get_dofs(), copy=True))
    return surface_copy


def build_synthetic_boozer_problem(config) -> RunCodeProblem:
    """Build the synthetic Boozer problem used by benchmark lanes."""
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.geo import (
        SurfaceRZFourier,
        SurfaceXYZTensorFourier,
        Volume,
        create_equally_spaced_curves,
    )

    base_curves = create_equally_spaced_curves(
        config.ncoils,
        config.nfp,
        stellsym=False,
        R0=1.0,
        R1=0.5,
        order=3,
    )
    base_currents = [Current(1e5) for _ in range(config.ncoils)]
    for current in base_currents:
        current.fix_all()
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        config.nfp,
        stellsym=False,
    )

    quadpoints_phi = np.linspace(0.0, 1.0 / config.nfp, config.nphi, endpoint=False)
    quadpoints_theta = np.linspace(0.0, 1.0, config.ntheta, endpoint=False)

    surface = SurfaceXYZTensorFourier(
        mpol=config.mpol,
        ntor=config.ntor,
        stellsym=False,
        nfp=config.nfp,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    seed_surface = SurfaceRZFourier(
        nfp=config.nfp,
        stellsym=False,
        mpol=1,
        ntor=0,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    seed_surface.set_rc(0, 0, 1.0)
    seed_surface.set_rc(1, 0, 0.15)
    seed_surface.set_zs(1, 0, 0.15)
    surface.least_squares_fit(seed_surface.gamma())

    volume = Volume(surface)
    vol_target = volume.J()

    mu0 = 4 * np.pi * 1e-7
    G0 = mu0 * sum(abs(coil.current.get_value()) for coil in coils)
    iota0 = 0.3

    return RunCodeProblem(
        coils=coils,
        surface=surface,
        volume=volume,
        vol_target=vol_target,
        iota0=iota0,
        G0=G0,
    )


def build_ls_parity_problem(
    *,
    ncoils: int = 2,
    nfp: int = 2,
    mpol: int = 2,
    ntor: int = 2,
    nphi: int | None = None,
    ntheta: int | None = None,
) -> RunCodeProblem:
    """Build the known-good LS parity fixture used by integration tests."""
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.geo import (
        SurfaceRZFourier,
        SurfaceXYZTensorFourier,
        Volume,
        create_equally_spaced_curves,
    )

    resolved_nphi = 2 * ntor + 1 if nphi is None else int(nphi)
    resolved_ntheta = 2 * mpol + 1 if ntheta is None else int(ntheta)
    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=3,
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    for current in base_currents:
        current.fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=True)

    quadpoints_phi = np.linspace(0, 1.0 / nfp, resolved_nphi, endpoint=False)
    quadpoints_theta = np.linspace(0, 1.0, resolved_ntheta, endpoint=False)

    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    surface.set_dofs(np.zeros_like(surface.get_dofs()))
    seed_surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    seed_surface.set_rc(0, 0, 1.0)
    seed_surface.set_rc(1, 0, 0.15)
    seed_surface.set_zs(1, 0, 0.15)
    surface.least_squares_fit(seed_surface.gamma())

    volume = Volume(surface)
    vol_target = volume.J()

    mu0 = 4 * np.pi * 1e-7
    G0 = mu0 * sum(abs(coil.current.get_value()) for coil in coils)
    iota0 = 0.3

    return RunCodeProblem(
        coils=coils,
        surface=surface,
        volume=volume,
        vol_target=vol_target,
        iota0=iota0,
        G0=G0,
    )
