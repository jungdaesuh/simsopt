from __future__ import annotations

import argparse
from dataclasses import replace
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
    banana_geometry_diagnostics,
    build_stage2_objective,
    build_surface,
    diagnostics_from_terms,
    ensure_writable_outputs,
    load_or_build_biotsavart,
    minimize_soft_penalty,
    output_paths,
    read_banana_dofs,
    result_payload,
    save_chainable_boozer_surface,
    write_json,
)
from banana_opt.jax_banana_types import (
    DEFAULT_BANANA_ORDER,
    DEFAULT_PROXY_RZ,
    HBT_BANANA_WS,
    Stage2Weights,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 2 banana-coil JAX soft-penalty optimization."
    )
    parser.add_argument("--backend", choices=("jax", "cpu"), default="jax")
    parser.add_argument("--platform", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--output-root", default="outputs_stage2_jax")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")

    parser.add_argument("--surface-path", "--surface", dest="surface_path", default=None)
    parser.add_argument("--vmec-s", type=float, default=1.0)
    parser.add_argument("--surface-scale", type=float, default=None)
    parser.add_argument("--mpol", type=int, default=8)
    parser.add_argument("--ntor", type=int, default=8)
    parser.add_argument("--nphi", type=int, default=65)
    parser.add_argument("--ntheta", type=int, default=64)

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

    parser.add_argument("--weight-sqflux", type=float, default=None)
    parser.add_argument("--weight-coil-length", type=float, default=None)
    parser.add_argument("--weight-coil-coil-distance", type=float, default=None)
    parser.add_argument("--weight-coil-surface-distance", type=float, default=None)
    parser.add_argument("--weight-coil-curvature", type=float, default=None)
    parser.add_argument("--weight-poloidal-extent", type=float, default=None)
    parser.add_argument("--weight-ellipse-width", type=float, default=None)
    parser.add_argument("--weight-global-curvature-radius", type=float, default=None)
    parser.add_argument("--weight-currents", type=float, default=None)
    parser.add_argument("--no-current-penalties", action="store_true")
    parser.add_argument("--include-min-length", action="store_true")
    parser.add_argument("--no-width", action="store_true")
    return parser


def _stage2_weights(args: argparse.Namespace) -> Stage2Weights:
    updates: dict[str, float] = {}
    mapping = {
        "sqflux": "weight_sqflux",
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
    return replace(Stage2Weights(), **updates)


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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = output_paths(_run_dir(args), prefix="stage2")
    ensure_writable_outputs(paths, overwrite=args.overwrite)
    log = DriverLog(paths.log)

    surface = _surface_from_args(args)
    biotsavart = load_or_build_biotsavart(
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
    )
    biotsavart_jax = BiotSavartJAX(biotsavart.coils)
    objective, terms = build_stage2_objective(
        biotsavart_jax=biotsavart_jax,
        surface=surface,
        weights=_stage2_weights(args),
        include_current_penalties=not args.no_current_penalties,
        include_min_length=args.include_min_length,
        include_width=not args.no_width,
    )

    log("Stage 2 JAX soft-penalty optimization")
    result, tracker = minimize_soft_penalty(
        objective=objective,
        maxiter=args.maxiter,
        log=log,
    )
    diagnostics = diagnostics_from_terms(objective, terms)
    diagnostics.update(banana_geometry_diagnostics(biotsavart_jax))

    biotsavart.save(str(paths.biotsavart))
    surface.save(str(paths.surface))
    save_chainable_boozer_surface(
        paths.boozersurface,
        biotsavart=biotsavart,
        surface=surface,
        constraint_weight=None,
    )
    write_json(
        paths.results,
        result_payload(
            driver="stage2_jax_soft_penalty",
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
