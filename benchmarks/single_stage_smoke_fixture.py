"""Shared real single-stage smoke fixture for Tier 3 and Tier 4."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.single_stage_smoke_defaults import (
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_STAGE2_BS_PATH,
)
from examples.single_stage_optimization.equilibria_paths import (
    DEFAULT_EQUILIBRIA_DIR,
    resolve_equilibrium_path as _resolve_equilibrium_path,
)

DEFAULT_SMOKE_NPHI = 31
DEFAULT_SMOKE_NTHETA = 16
DEFAULT_SMOKE_MPOL = 2
DEFAULT_SMOKE_NTOR = 2
DEFAULT_VOL_TARGET = 0.10
DEFAULT_IOTA_TARGET = 0.15
DEFAULT_OPTIMIZER_BACKEND = "ondevice"
DEFAULT_REFERENCE_OPTIMIZER_BACKEND = "scipy"
DEFAULT_CONSTRAINT_WEIGHT = 1.0
DEFAULT_NUM_TF_COILS = 20


def default_optimizer_backend_for_backend(backend: str) -> str:
    return (
        DEFAULT_OPTIMIZER_BACKEND
        if backend == "jax"
        else DEFAULT_REFERENCE_OPTIMIZER_BACKEND
    )


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
    return _resolve_equilibrium_path(
        plasma_surf_filename=plasma_surf_filename,
        equilibria_dir=equilibria_dir,
        equilibrium_path=equilibrium_path,
    )


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
    optimizer_backend: str | None = None,
    boozer_optimizer_backend: str | None = None,
    boozer_least_squares_algorithm: str | None = None,
    boozer_limited_memory: bool | None = None,
    bs_dofs_override: np.ndarray | None = None,
    boozer_surface_dofs_override: np.ndarray | None = None,
    boozer_iota_override: float | None = None,
    boozer_G_override: float | None = None,
    constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT,
    num_tf_coils: int = DEFAULT_NUM_TF_COILS,
    on_stage=None,
) -> dict[str, object]:
    """Build the real reduced-grid single-stage init fixture without diagnostics.

    Optional Boozer override arguments replay the solve from a previously
    converged Boozer state instead of restarting from the raw Stage 2 seed.
    """
    from simsopt._core.optimizable import load
    from simsopt.geo import SurfaceRZFourier

    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    resolved_optimizer_backend = (
        default_optimizer_backend_for_backend(backend)
        if optimizer_backend is None
        else optimizer_backend
    )
    resolved_boozer_optimizer_backend = (
        single_stage_example.resolve_boozer_optimizer_backend(
            backend,
            resolved_optimizer_backend,
            boozer_optimizer_backend,
        )
    )
    resolved_boozer_limited_memory = (
        single_stage_example.resolve_single_stage_boozer_limited_memory(
            backend,
            resolved_optimizer_backend,
            resolved_boozer_optimizer_backend,
            boozer_limited_memory,
        )
    )
    resolved_boozer_init_base_overrides = (
        single_stage_example.resolve_target_lane_boozer_init_base_overrides(
            field_backend=backend,
            optimizer_backend=resolved_optimizer_backend,
            boozer_limited_memory=resolved_boozer_limited_memory,
            target_lane_boozer_bfgs_tol=(
                single_stage_example.resolve_target_lane_boozer_bfgs_tol(
                    backend,
                    resolved_optimizer_backend,
                    None,
                    benchmark_mode=False,
                )
            ),
            target_lane_boozer_bfgs_maxiter=(
                single_stage_example.resolve_target_lane_boozer_bfgs_maxiter(
                    backend,
                    resolved_optimizer_backend,
                    None,
                    benchmark_mode=False,
                )
            ),
            target_lane_boozer_newton_tol=(
                single_stage_example.resolve_target_lane_boozer_newton_tol(
                    backend,
                    resolved_optimizer_backend,
                    None,
                )
            ),
            target_lane_boozer_newton_maxiter=(
                single_stage_example.resolve_target_lane_boozer_newton_maxiter(
                    backend,
                    resolved_optimizer_backend,
                    None,
                )
            ),
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
    if bs_dofs_override is not None:
        bs.x = np.asarray(bs_dofs_override, dtype=float)
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
        boozer_least_squares_algorithm=boozer_least_squares_algorithm,
        boozer_limited_memory=resolved_boozer_limited_memory,
        bfgs_tol_override=resolved_boozer_init_base_overrides["bfgs_tol_override"],
        bfgs_maxiter_override=resolved_boozer_init_base_overrides[
            "bfgs_maxiter_override"
        ],
        newton_tol_override=resolved_boozer_init_base_overrides["newton_tol_override"],
        newton_maxiter_override=resolved_boozer_init_base_overrides[
            "newton_maxiter_override"
        ],
        surface_dofs_override=boozer_surface_dofs_override,
        iota_override=boozer_iota_override,
        G_override=boozer_G_override,
        on_stage=on_stage,
    )
    return {
        "bs": bs,
        "boozer_surface": boozer_surface,
        "boozer_optimizer_backend": resolved_boozer_optimizer_backend,
        "boozer_least_squares_algorithm": boozer_least_squares_algorithm,
        "boozer_limited_memory": resolved_boozer_limited_memory,
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
