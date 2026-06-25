from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, EXAMPLE_ROOT)

SIMSOPT_ROOT = os.path.abspath(os.path.join(EXAMPLE_ROOT, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
SRC_ROOT = os.path.join(SIMSOPT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, SIMSOPT_ROOT)
sys.path.insert(0, REPO_ROOT)

from repo_bootstrap import bootstrap_local_simsopt, configure_entrypoint_jax_runtime


def _default_platform_from_backend(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--backend" and index + 1 < len(argv) and argv[index + 1] == "cpu":
            return "cpu"
        if token == "--backend=cpu":
            return "cpu"
    return None


configure_entrypoint_jax_runtime(
    sys.argv[1:],
    default_platform=_default_platform_from_backend(sys.argv[1:]),
)
bootstrap_local_simsopt(SRC_ROOT)

import jax  # noqa: F401
from simsopt.geo import Surface, SurfaceRZFourier, SurfaceXYZFourier, SurfaceXYZTensorFourier
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

from banana_opt.jax_banana_drivers import (
    DriverLog,
    MU0,
    banana_geometry_diagnostics,
    build_jax_boozer_surface,
    build_single_stage_objective,
    build_surface,
    diagnostics_from_terms,
    ensure_writable_outputs,
    load_or_build_biotsavart,
    load_stage2_boozer_seed,
    minimize_single_stage_soft_penalty,
    output_paths,
    read_banana_dofs,
    require_successful_boozer_solve,
    result_payload,
    save_chainable_boozer_surface,
    tf_current_total_abs_A,
    write_json,
)
from banana_opt.jax_banana_types import (
    BoozerSolveState,
    DEFAULT_BANANA_ORDER,
    DEFAULT_PROXY_RZ,
    HBT_BANANA_WS,
    SingleStageWeights,
)


@dataclass(frozen=True)
class SeedPaths:
    biotsavart: Path | None
    surface: Path | None
    boozersurface: Path | None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-stage banana JAX soft-penalty optimization."
    )
    parser.add_argument("--backend", choices=("jax", "cpu"), default="jax")
    parser.add_argument("--platform", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--output-root", default="outputs_single_stage_jax")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")

    parser.add_argument("--warm-start-run-dir", default=None)
    parser.add_argument("--stage2-bs-path", default=None)
    parser.add_argument("--jax-runtime-seed-spec", default=None)
    parser.add_argument("--compile-jax-runtime-seed-spec", action="store_true")

    parser.add_argument("--surface-path", "--surface", dest="surface_path", default=None)
    parser.add_argument("--vmec-s", type=float, default=1.0)
    parser.add_argument("--surface-scale", type=float, default=None)
    parser.add_argument("--mpol", type=int, default=8)
    parser.add_argument("--ntor", type=int, default=8)
    parser.add_argument("--nphi", type=int, default=65)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument("--constraint-weight", type=float, default=None)

    parser.add_argument("--biotsavart-file", default=None)
    parser.add_argument("--banana-dofs", default=None)
    parser.add_argument(
        "--banana-order",
        "--order",
        dest="banana_order",
        type=int,
        default=DEFAULT_BANANA_ORDER,
    )
    parser.add_argument("--tf-current-ka", type=float, default=-80.0)
    parser.add_argument("--banana-current-ka", type=float, default=16.0)
    parser.add_argument("--proxy-current-ka", type=float, default=0.0)
    parser.add_argument("--proxy-r", type=float, default=DEFAULT_PROXY_RZ[0])
    parser.add_argument("--proxy-z", type=float, default=DEFAULT_PROXY_RZ[1])
    parser.add_argument("--vf-current-ka", type=float, default=0.0)
    parser.add_argument("--free-tf-current", action="store_true")
    parser.add_argument("--free-banana-current", action="store_true")
    parser.add_argument("--free-vf-current", action="store_true")

    parser.add_argument("--iota-target", "--iota", dest="iota_target", type=float, default=0.15)
    parser.add_argument("--sign-g", type=float, default=-1.0)
    parser.add_argument("--weight-non-quasisymmetric-ratio", type=float, default=None)
    parser.add_argument("--weight-boozer-residual", type=float, default=None)
    parser.add_argument("--weight-iota", type=float, default=None)
    parser.add_argument("--weight-coil-length", type=float, default=None)
    parser.add_argument("--weight-coil-coil-distance", type=float, default=None)
    parser.add_argument("--weight-coil-surface-distance", type=float, default=None)
    parser.add_argument("--weight-coil-curvature", type=float, default=None)
    parser.add_argument("--weight-poloidal-extent", type=float, default=None)
    parser.add_argument("--weight-ellipse-width", type=float, default=None)
    parser.add_argument("--weight-global-curvature-radius", type=float, default=None)
    parser.add_argument("--weight-currents", type=float, default=None)
    parser.add_argument("--include-boozer-residual", action="store_true")
    parser.add_argument("--no-current-penalties", action="store_true")
    parser.add_argument("--include-min-length", action="store_true")
    parser.add_argument("--no-width", action="store_true")
    return parser


def _single_stage_weights(args: argparse.Namespace) -> SingleStageWeights:
    updates: dict[str, float] = {}
    mapping = {
        "nonqs": "weight_non_quasisymmetric_ratio",
        "bres": "weight_boozer_residual",
        "iota": "weight_iota",
        "length": "weight_coil_length",
        "ccdist": "weight_coil_coil_distance",
        "csdist": "weight_coil_surface_distance",
        "curvature": "weight_coil_curvature",
        "poloidal": "weight_poloidal_extent",
        "width": "weight_ellipse_width",
        "global_curvature_radius": "weight_global_curvature_radius",
        "currents": "weight_currents",
    }
    for field_name, arg_name in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = float(value)
    return replace(SingleStageWeights(), **updates)


def _default_surface(
    *,
    mpol: int,
    ntor: int,
    nphi: int,
    ntheta: int,
) -> SurfaceRZFourier:
    quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
        nphi=nphi,
        ntheta=ntheta,
        nfp=HBT_BANANA_WS.nfp,
    )
    surface = SurfaceRZFourier(
        nfp=HBT_BANANA_WS.nfp,
        stellsym=HBT_BANANA_WS.stellsym,
        mpol=mpol,
        ntor=ntor,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    surface.set_rc(0, 0, 0.78)
    surface.set_rc(1, 0, 0.08)
    surface.set_zs(1, 0, 0.08)
    return surface


def _surface_from_args(
    args: argparse.Namespace,
) -> SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier:
    if args.surface_path is None:
        return _default_surface(
            mpol=args.mpol,
            ntor=args.ntor,
            nphi=args.nphi,
            ntheta=args.ntheta,
        )
    return build_surface(
        args.surface_path,
        vmec_s=args.vmec_s,
        scale=args.surface_scale,
        nphi=args.nphi,
        ntheta=args.ntheta,
    )


def _run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return Path(args.run_dir)
    return Path(args.output_root)


def _boozer_exact_grid(args: argparse.Namespace) -> tuple[int, int]:
    if HBT_BANANA_WS.stellsym:
        return 2 * int(args.ntor) + 1, 2 * int(args.mpol) + 1
    return int(args.nphi), int(args.ntheta)


def _seed_spec_paths(path: Path) -> SeedPaths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    biotsavart = payload.get("biotsavart")
    surface = payload.get("surface")
    boozersurface = payload.get("boozersurface")
    return SeedPaths(
        None if biotsavart is None else Path(str(biotsavart)),
        None if surface is None else Path(str(surface)),
        None if boozersurface is None else Path(str(boozersurface)),
    )


def _run_dir_seed_paths(run_dir: Path) -> SeedPaths:
    return SeedPaths(
        biotsavart=run_dir / "biot_savart_opt.json",
        surface=run_dir / "surf_opt.json",
        boozersurface=run_dir / "boozersurface_opt.json",
    )


def _validate_seed_artifacts(seed_paths: SeedPaths) -> SeedPaths:
    missing = [
        path
        for path in (
            seed_paths.biotsavart,
            seed_paths.surface,
            seed_paths.boozersurface,
        )
        if path is not None and not path.exists()
    ]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Cannot compile single-stage seed spec; missing Stage 2 artifact(s): {joined}"
        )
    return seed_paths


def _seed_paths_from_args(args: argparse.Namespace) -> SeedPaths | None:
    if args.stage2_bs_path is not None:
        return SeedPaths(None, None, Path(args.stage2_bs_path))
    if args.jax_runtime_seed_spec is not None and not args.compile_jax_runtime_seed_spec:
        return _seed_spec_paths(Path(args.jax_runtime_seed_spec))
    if args.warm_start_run_dir is not None:
        return _run_dir_seed_paths(Path(args.warm_start_run_dir))
    return None


def _write_seed_spec(args: argparse.Namespace) -> None:
    if args.jax_runtime_seed_spec is None:
        raise ValueError("--compile-jax-runtime-seed-spec requires --jax-runtime-seed-spec")
    if args.warm_start_run_dir is None and args.stage2_bs_path is None:
        raise ValueError(
            "--compile-jax-runtime-seed-spec requires --warm-start-run-dir or --stage2-bs-path"
        )
    seed_paths = (
        SeedPaths(None, None, Path(args.stage2_bs_path))
        if args.stage2_bs_path is not None
        else _run_dir_seed_paths(Path(args.warm_start_run_dir))
    )
    seed_paths = _validate_seed_artifacts(seed_paths)
    write_json(
        Path(args.jax_runtime_seed_spec),
        {
            "schema_version": 1,
            "driver": "single_stage_jax_runtime_seed_spec",
            "stage2_run_dir": args.warm_start_run_dir,
            "biotsavart": None if seed_paths.biotsavart is None else str(seed_paths.biotsavart),
            "surface": None if seed_paths.surface is None else str(seed_paths.surface),
            "boozersurface": None if seed_paths.boozersurface is None else str(seed_paths.boozersurface),
            "mpol": args.mpol,
            "ntor": args.ntor,
            "nphi": args.nphi,
            "ntheta": args.ntheta,
        },
    )


def _load_seed_inputs(
    args: argparse.Namespace,
) -> tuple[
    object,
    SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier,
]:
    seed_paths = _seed_paths_from_args(args)
    if seed_paths is not None and seed_paths.boozersurface is not None:
        return load_stage2_boozer_seed(seed_paths.boozersurface)
    if seed_paths is not None and seed_paths.biotsavart is not None:
        biotsavart = load_or_build_biotsavart(
            biotsavart_file=seed_paths.biotsavart,
            tf_current_ka=args.tf_current_ka,
            tf_fix_current=not args.free_tf_current,
            banana_current_ka=args.banana_current_ka,
            banana_fix_current=not args.free_banana_current,
            banana_order=args.banana_order,
            banana_dofs=read_banana_dofs(args.banana_dofs),
            proxy_current_ka=args.proxy_current_ka,
            proxy_rz=(args.proxy_r, args.proxy_z),
            vf_current_ka=args.vf_current_ka,
            vf_fix_current=not args.free_vf_current,
        )
        if seed_paths.surface is None:
            return biotsavart, _surface_from_args(args)
        surface = build_surface(
            seed_paths.surface,
            vmec_s=args.vmec_s,
            scale=args.surface_scale,
            nphi=args.nphi,
            ntheta=args.ntheta,
        )
        return biotsavart, surface
    return (
        load_or_build_biotsavart(
            biotsavart_file=args.biotsavart_file,
            tf_current_ka=args.tf_current_ka,
            tf_fix_current=not args.free_tf_current,
            banana_current_ka=args.banana_current_ka,
            banana_fix_current=not args.free_banana_current,
            banana_order=args.banana_order,
            banana_dofs=read_banana_dofs(args.banana_dofs),
            proxy_current_ka=args.proxy_current_ka,
            proxy_rz=(args.proxy_r, args.proxy_z),
            vf_current_ka=args.vf_current_ka,
            vf_fix_current=not args.free_vf_current,
        ),
        _surface_from_args(args),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.compile_jax_runtime_seed_spec:
        _write_seed_spec(args)
        return 0

    paths = output_paths(_run_dir(args), prefix="single_stage")
    ensure_writable_outputs(paths, overwrite=args.overwrite)
    log = DriverLog(paths.log)

    biotsavart, surface = _load_seed_inputs(args)
    biotsavart_jax = BiotSavartJAX(biotsavart.coils)
    boozer_nphi, boozer_ntheta = _boozer_exact_grid(args)
    boozersurface = build_jax_boozer_surface(
        biotsavart_jax=biotsavart_jax,
        surface=surface,
        mpol=args.mpol,
        ntor=args.ntor,
        nphi=boozer_nphi,
        ntheta=boozer_ntheta,
        constraint_weight=args.constraint_weight,
    )
    solve_state = require_successful_boozer_solve(
        boozersurface,
        BoozerSolveState(
            iota=args.iota_target,
            G=args.sign_g * tf_current_total_abs_A(biotsavart_jax) * MU0,
        ),
    )
    weights = _single_stage_weights(args)
    objective, terms = build_single_stage_objective(
        boozersurface=boozersurface,
        biotsavart_jax=biotsavart_jax,
        iota_target=args.iota_target,
        weights=weights,
        include_boozer_residual=(
            args.include_boozer_residual or args.constraint_weight is not None
        ),
        include_current_penalties=not args.no_current_penalties,
        include_min_length=args.include_min_length,
        include_width=not args.no_width,
    )

    log("Single-stage JAX soft-penalty optimization")
    result, tracker = minimize_single_stage_soft_penalty(
        objective=objective,
        boozersurface=boozersurface,
        initial_state=solve_state,
        maxiter=args.maxiter,
        log=log,
    )
    diagnostics = diagnostics_from_terms(objective, terms)
    diagnostics.update(banana_geometry_diagnostics(biotsavart_jax))

    biotsavart.save(str(paths.biotsavart))
    boozersurface.surface.save(str(paths.surface))
    save_chainable_boozer_surface(
        paths.boozersurface,
        biotsavart=biotsavart,
        surface=boozersurface.surface,
        constraint_weight=args.constraint_weight,
    )
    write_json(
        paths.results,
        result_payload(
            driver="single_stage_jax_soft_penalty",
            args=vars(args),
            optimizer_result=result,
            tracker=tracker,
            diagnostics=diagnostics,
            paths=paths,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
