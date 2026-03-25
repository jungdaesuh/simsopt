"""Shared real single-stage smoke fixture for Tier 3 and Tier 4."""

from __future__ import annotations

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLASMA_SURF_FILENAME = "wout_nfp22ginsburg_000_014417_iota15.nc"
DEFAULT_STAGE2_SEED_DIR = (
    REPO_ROOT / "benchmarks" / "fixtures" / "single_stage_seed_iota15"
)
DEFAULT_STAGE2_BS_PATH = DEFAULT_STAGE2_SEED_DIR / "biot_savart_opt.json"
DEFAULT_EQUILIBRIA_DIR = (
    REPO_ROOT / "examples" / "single_stage_optimization" / "equilibria"
)
DEFAULT_SMOKE_NPHI = 31
DEFAULT_SMOKE_NTHETA = 16
DEFAULT_SMOKE_MPOL = 2
DEFAULT_SMOKE_NTOR = 2
DEFAULT_VOL_TARGET = 0.10
DEFAULT_IOTA_TARGET = 0.15
DEFAULT_OPTIMIZER_BACKEND = "scipy"
DEFAULT_CONSTRAINT_WEIGHT = 1.0
DEFAULT_NUM_TF_COILS = 20


def _emit_stage(
    on_stage,
    label: str,
    **extra: float | str | None,
) -> None:
    if on_stage is not None:
        on_stage(label, **extra)


def resolve_equilibrium_path(
    *,
    plasma_surf_filename: str = DEFAULT_PLASMA_SURF_FILENAME,
    equilibria_dir: str | Path = DEFAULT_EQUILIBRIA_DIR,
    equilibrium_path: str | Path | None = None,
) -> Path:
    """Resolve the equilibrium file for the real single-stage seed family."""
    if equilibrium_path is not None:
        return Path(equilibrium_path)

    candidate = Path(equilibria_dir) / plasma_surf_filename
    if candidate.exists():
        return candidate
    return candidate


def build_real_single_stage_init_fixture(
    *,
    backend: str,
    plasma_surf_filename: str = DEFAULT_PLASMA_SURF_FILENAME,
    equilibria_dir: str | Path = DEFAULT_EQUILIBRIA_DIR,
    equilibrium_path: str | Path | None = None,
    stage2_bs_path: str | Path = DEFAULT_STAGE2_BS_PATH,
    nphi: int = DEFAULT_SMOKE_NPHI,
    ntheta: int = DEFAULT_SMOKE_NTHETA,
    mpol: int = DEFAULT_SMOKE_MPOL,
    ntor: int = DEFAULT_SMOKE_NTOR,
    vol_target: float = DEFAULT_VOL_TARGET,
    iota_target: float = DEFAULT_IOTA_TARGET,
    optimizer_backend: str = DEFAULT_OPTIMIZER_BACKEND,
    boozer_optimizer_backend: str | None = None,
    boozer_limited_memory: bool = False,
    constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT,
    num_tf_coils: int = DEFAULT_NUM_TF_COILS,
    on_stage=None,
) -> dict[str, object]:
    """Build the real reduced-grid single-stage init fixture without diagnostics."""
    from simsopt._core.optimizable import load
    from simsopt.geo import SurfaceRZFourier

    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    resolved_boozer_optimizer_backend = (
        single_stage_example.resolve_boozer_optimizer_backend(
            backend,
            optimizer_backend,
            boozer_optimizer_backend,
        )
    )

    stage2_bs_path = Path(stage2_bs_path)
    _, stage2_results = single_stage_example.load_stage2_results(str(stage2_bs_path))
    major_radius = float(stage2_results["MAJOR_RADIUS"])
    toroidal_flux = float(stage2_results["TOROIDAL_FLUX"])
    _emit_stage(
        on_stage,
        "after_stage2_results_load",
        major_radius=major_radius,
        toroidal_flux=toroidal_flux,
    )
    equilibrium_file = resolve_equilibrium_path(
        plasma_surf_filename=plasma_surf_filename,
        equilibria_dir=equilibria_dir,
        equilibrium_path=equilibrium_path,
    )

    bs_loaded = load(str(stage2_bs_path))
    if backend == "jax":
        from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

        bs = BiotSavartJAX(bs_loaded.coils)
    else:
        bs = bs_loaded
    _emit_stage(on_stage, "after_biotsavart_load", backend=backend)

    surf = SurfaceRZFourier.from_wout(
        str(equilibrium_file),
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
        s=toroidal_flux,
    )
    surf.set_dofs(surf.get_dofs() * major_radius / surf.major_radius())
    _emit_stage(
        on_stage,
        "after_surface_seed_setup",
        nphi=int(nphi),
        ntheta=int(ntheta),
        mpol=int(mpol),
        ntor=int(ntor),
    )

    tf_coils = bs.coils[:num_tf_coils]
    current_sum = sum(abs(coil.current.get_value()) for coil in tf_coils)
    g0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))

    boozer_surface = single_stage_example.initialize_boozer_surface(
        surf,
        mpol,
        ntor,
        bs,
        vol_target,
        constraint_weight,
        iota_target,
        g0,
        backend=backend,
        optimizer_backend=resolved_boozer_optimizer_backend,
        boozer_limited_memory=boozer_limited_memory,
        on_stage=on_stage,
    )
    return {
        "bs": bs,
        "boozer_surface": boozer_surface,
        "boozer_optimizer_backend": resolved_boozer_optimizer_backend,
        "equilibrium_path": str(equilibrium_file),
        "stage2_bs_path": str(stage2_bs_path),
        "vol_target": float(vol_target),
        "iota_target": float(iota_target),
        "surface_shape": {
            "nphi": int(nphi),
            "ntheta": int(ntheta),
            "mpol": int(mpol),
            "ntor": int(ntor),
        },
    }
