from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

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
import numpy as np
from simsopt._core.optimizable import Optimizable
from simsopt.geo import (
    Surface,
    SurfaceRZFourier,
    SurfaceXYZFourier,
    SurfaceXYZTensorFourier,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

from banana_opt.jax_banana_drivers import (
    DriverLog,
    MU0,
    banana_geometry_diagnostics,
    boozer_solver_grid_shape,
    build_jax_boozer_surface,
    build_single_stage_common_objective,
    build_single_stage_objective,
    build_surface,
    comparison_backend_label,
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
    COMMON_OBJECTIVE_CONTRACT_ID,
    COMMON_OBJECTIVE_TERM_NAMES,
    DEFAULT_BANANA_ORDER,
    DEFAULT_BOOZER_CONSTRAINT_WEIGHT,
    DEFAULT_PROXY_RZ,
    HBT_BANANA_WS,
    SingleStageWeights,
    WeightedTerm,
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
    parser.add_argument("--outer-maxcor", type=int, default=300)
    parser.add_argument("--outer-maxls", type=int, default=20)
    parser.add_argument("--outer-ftol", type=float, default=1.0e-15)
    parser.add_argument("--outer-gtol", type=float, default=1.0e-15)
    parser.add_argument("--output-root", default="outputs_single_stage_jax")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")

    parser.add_argument("--warm-start-run-dir", default=None)
    parser.add_argument("--stage2-bs-path", default=None)
    parser.add_argument("--jax-runtime-seed-spec", default=None)
    parser.add_argument("--compile-jax-runtime-seed-spec", action="store_true")
    parser.add_argument("--boozer-state-path", default=None)
    parser.add_argument("--run-config-sha256", default=None)
    parser.add_argument(
        "--objective-profile",
        choices=("default", "common-seven-term"),
        default="default",
    )

    parser.add_argument(
        "--surface-path", "--surface", dest="surface_path", default=None
    )
    parser.add_argument("--vmec-s", type=float, default=1.0)
    parser.add_argument("--surface-scale", type=float, default=None)
    parser.add_argument("--mpol", type=int, default=8)
    parser.add_argument("--ntor", type=int, default=8)
    parser.add_argument("--nphi", type=int, default=65)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument(
        "--constraint-weight",
        type=float,
        default=DEFAULT_BOOZER_CONSTRAINT_WEIGHT,
    )
    parser.add_argument("--boozer-bfgs-tol", type=float, default=1.0e-10)
    parser.add_argument("--boozer-bfgs-maxiter", type=int, default=1500)
    parser.add_argument("--boozer-newton-tol", type=float, default=1.0e-11)
    parser.add_argument("--boozer-newton-maxiter", type=int, default=40)
    parser.add_argument("--boozer-limited-memory", action="store_true")

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

    parser.add_argument(
        "--iota-target", "--iota", dest="iota_target", type=float, default=0.15
    )
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


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _float_array_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8").reshape(-1)
    return sha256(canonical.tobytes(order="C")).hexdigest()


def _string_sequence_sha256(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _initial_solve_state(
    args: argparse.Namespace,
    biotsavart_jax: BiotSavartJAX,
) -> BoozerSolveState:
    if args.boozer_state_path is not None:
        payload = json.loads(Path(args.boozer_state_path).read_text(encoding="utf-8"))
        return BoozerSolveState(iota=float(payload["iota"]), G=float(payload["G"]))
    return BoozerSolveState(
        iota=args.iota_target,
        G=args.sign_g * tf_current_total_abs_A(biotsavart_jax) * MU0,
    )


def _term_payload(terms: list[WeightedTerm]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for term in terms:
        raw_value = float(term.objective.J())
        payload[term.name] = {
            "raw": raw_value,
            "weight": float(term.weight),
            "weighted": float(term.weight) * raw_value,
        }
    return payload


def _comparison_state_payload(
    *,
    objective: Optimizable,
    terms: list[WeightedTerm],
    boozersurface: BoozerSurfaceJAX,
    solve_state: BoozerSolveState,
    objective_value: float | None = None,
    gradient: np.ndarray | None = None,
) -> dict[str, object]:
    dofs = np.asarray(objective.x, dtype=np.float64)
    resolved_objective_value = (
        float(objective.J()) if objective_value is None else float(objective_value)
    )
    term_values = _term_payload(terms)
    if term_values["coil_surface_distance"]["raw"] > 0.0:
        raise RuntimeError(
            "The matched seven-term benchmark requires the coil-surface-distance "
            "hinge to remain inactive because its implicit Boozer-surface derivative "
            "is not implemented"
        )
    resolved_gradient = (
        np.asarray(objective.dJ(), dtype=np.float64)
        if gradient is None
        else np.asarray(gradient, dtype=np.float64)
    )
    return {
        "objective": resolved_objective_value,
        "gradient_norm": float(np.linalg.norm(resolved_gradient)),
        "iota": solve_state.iota,
        "G": solve_state.G,
        "volume": float(boozersurface.surface.volume()),
        "dofs": dofs.tolist(),
        "dof_count": int(dofs.size),
        "dofs_sha256": _float_array_sha256(dofs),
        "gradient": resolved_gradient.tolist(),
        "gradient_count": int(resolved_gradient.size),
        "gradient_sha256": _float_array_sha256(resolved_gradient),
        "terms": term_values,
    }


def _common_contract_payload(
    objective: Optimizable,
    terms: list[WeightedTerm],
) -> dict[str, object]:
    dof_names = [str(name) for name in objective.dof_names]
    return {
        "id": COMMON_OBJECTIVE_CONTRACT_ID,
        "ordered_terms": list(COMMON_OBJECTIVE_TERM_NAMES),
        "weights": {term.name: float(term.weight) for term in terms},
        "optimizer_method": "L-BFGS-B",
        "constraint_method": "soft-penalty",
        "dtype": "float64",
        "mixed_precision": False,
        "adjoint_acceptance_policy": "checked-residual-and-condition",
        "dof_names": dof_names,
        "dof_count": len(dof_names),
        "dof_names_sha256": _string_sequence_sha256(dof_names),
    }


def _comparison_input_sha256(args: argparse.Namespace) -> dict[str, str]:
    return {
        "surface": _sha256_file(args.surface_path),
        "biotsavart": _sha256_file(args.biotsavart_file),
        "boozer_state": _sha256_file(args.boozer_state_path),
    }


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


def _boozer_grid(args: argparse.Namespace) -> tuple[int, int]:
    return boozer_solver_grid_shape(
        mpol=args.mpol,
        ntor=args.ntor,
        nphi=args.nphi,
        ntheta=args.ntheta,
        stellsym=HBT_BANANA_WS.stellsym,
        constraint_weight=args.constraint_weight,
    )


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
    if (
        args.jax_runtime_seed_spec is not None
        and not args.compile_jax_runtime_seed_spec
    ):
        return _seed_spec_paths(Path(args.jax_runtime_seed_spec))
    if args.warm_start_run_dir is not None:
        return _run_dir_seed_paths(Path(args.warm_start_run_dir))
    return None


def _write_seed_spec(args: argparse.Namespace) -> None:
    if args.jax_runtime_seed_spec is None:
        raise ValueError(
            "--compile-jax-runtime-seed-spec requires --jax-runtime-seed-spec"
        )
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
            "biotsavart": None
            if seed_paths.biotsavart is None
            else str(seed_paths.biotsavart),
            "surface": None if seed_paths.surface is None else str(seed_paths.surface),
            "boozersurface": None
            if seed_paths.boozersurface is None
            else str(seed_paths.boozersurface),
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
    run_started = perf_counter()
    args = _parser().parse_args(argv)
    if args.compile_jax_runtime_seed_spec:
        _write_seed_spec(args)
        return 0
    common_profile = args.objective_profile == "common-seven-term"
    if common_profile and os.environ.get("SIMSOPT_MIXED_PRECISION", "0") != "0":
        raise RuntimeError(
            "common-seven-term comparison requires SIMSOPT_MIXED_PRECISION=0"
        )
    if common_profile and (
        args.surface_path is None
        or args.biotsavart_file is None
        or args.boozer_state_path is None
        or args.run_config_sha256 is None
    ):
        raise ValueError(
            "common-seven-term comparison requires --surface-path, "
            "--biotsavart-file, --boozer-state-path, and --run-config-sha256"
        )
    if common_profile and not bool(jax.config.jax_enable_x64):
        raise RuntimeError("common-seven-term comparison requires JAX FP64")
    comparison_backend = (
        comparison_backend_label(jax.devices()[0].platform) if common_profile else None
    )

    setup_started = perf_counter()
    paths = output_paths(_run_dir(args), prefix="single_stage")
    ensure_writable_outputs(paths, overwrite=args.overwrite)
    log = DriverLog(paths.log, persist=not common_profile)

    biotsavart, surface = _load_seed_inputs(args)
    biotsavart_jax = BiotSavartJAX(biotsavart.coils)
    boozer_nphi, boozer_ntheta = _boozer_grid(args)
    boozersurface = build_jax_boozer_surface(
        biotsavart_jax=biotsavart_jax,
        surface=surface,
        mpol=args.mpol,
        ntor=args.ntor,
        nphi=boozer_nphi,
        ntheta=boozer_ntheta,
        constraint_weight=args.constraint_weight,
        bfgs_tol=args.boozer_bfgs_tol,
        bfgs_maxiter=args.boozer_bfgs_maxiter,
        newton_tol=args.boozer_newton_tol,
        newton_maxiter=args.boozer_newton_maxiter,
        limited_memory=args.boozer_limited_memory,
    )
    weights = _single_stage_weights(args)
    coil_surface_distance = None
    if common_profile:
        objective, terms, coil_surface_distance = build_single_stage_common_objective(
            boozersurface=boozersurface,
            biotsavart_jax=biotsavart_jax,
            iota_target=args.iota_target,
            weights=weights,
        )
    else:
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
    setup_s = perf_counter() - setup_started

    initial_solve_started = perf_counter()
    solve_state = require_successful_boozer_solve(
        boozersurface,
        _initial_solve_state(args, biotsavart_jax),
    )
    initial_inner_solve_s = perf_counter() - initial_solve_started
    initial_payload_started = perf_counter()
    initial_payload = (
        _comparison_state_payload(
            objective=objective,
            terms=terms,
            boozersurface=boozersurface,
            solve_state=solve_state,
        )
        if common_profile
        else None
    )
    initial_state_payload_s = perf_counter() - initial_payload_started

    log("Single-stage JAX soft-penalty optimization")
    result, tracker = minimize_single_stage_soft_penalty(
        objective=objective,
        boozersurface=boozersurface,
        initial_state=solve_state,
        maxiter=args.maxiter,
        log=log,
        maxcor=args.outer_maxcor,
        maxls=args.outer_maxls,
        ftol=args.outer_ftol,
        gtol=args.outer_gtol,
        coil_surface_distance=coil_surface_distance,
    )
    diagnostics = {} if common_profile else diagnostics_from_terms(objective, terms)
    if not common_profile:
        diagnostics.update(banana_geometry_diagnostics(biotsavart_jax))

    final_solve_state = BoozerSolveState(
        iota=float(boozersurface.res["iota"]),
        G=float(boozersurface.res["G"]),
    )
    final_payload = (
        _comparison_state_payload(
            objective=objective,
            terms=terms,
            boozersurface=boozersurface,
            solve_state=final_solve_state,
            objective_value=float(result.fun),
            gradient=np.asarray(result.jac, dtype=np.float64),
        )
        if common_profile
        else None
    )

    artifact_started = perf_counter()
    biotsavart.save(str(paths.biotsavart))
    boozersurface.surface.save(str(paths.surface))
    save_chainable_boozer_surface(
        paths.boozersurface,
        biotsavart=biotsavart,
        surface=boozersurface.surface,
        constraint_weight=args.constraint_weight,
    )
    artifact_write_s = perf_counter() - artifact_started
    payload = result_payload(
        driver="single_stage_jax_soft_penalty",
        args=vars(args),
        optimizer_result=result,
        tracker=tracker,
        diagnostics=diagnostics,
        paths=paths,
    )
    if common_profile:
        payload.update(
            {
                "backend": comparison_backend,
                "precision": "float64",
                "constraint_method": "soft-penalty",
                "mixed_precision": False,
                "comparison_schema_version": 1,
                "objective_contract": _common_contract_payload(objective, terms),
                "input_sha256": _comparison_input_sha256(args),
                "run_config_sha256": args.run_config_sha256,
                "initial_state": initial_payload,
                "final_state": final_payload,
                "timings": {
                    "setup_s": setup_s,
                    "initial_inner_solve_s": initial_inner_solve_s,
                    "initial_state_payload_s": initial_state_payload_s,
                    "outer_optimization_s": tracker.optimizer_s,
                    "final_inner_solve_s": tracker.final_inner_solve_s,
                    "final_objective_s": tracker.final_objective_s,
                    "outer_optimization_and_finalize_s": tracker.elapsed_s,
                    "artifact_write_s": artifact_write_s,
                    "measured_before_results_write_s": perf_counter() - run_started,
                },
            }
        )
        payload["objective_contract"]["inactive_term_requirements"] = {
            "coil_surface_distance": 0.0
        }
        payload["optimizer"].update(
            {
                "method": "L-BFGS-B",
                "status": int(result.status),
                "evaluations": tracker.evaluations,
                "rejected_evaluations": tracker.rejected_evaluations,
                "accepted_iterations": tracker.iterations,
                "maxiter": args.maxiter,
                "maxcor": args.outer_maxcor,
                "maxls": args.outer_maxls,
                "ftol": args.outer_ftol,
                "gtol": args.outer_gtol,
            }
        )
        assert paths.surface is not None
        payload["outputs"] = {
            "biotsavart": {
                "path": str(paths.biotsavart.resolve()),
                "sha256": _sha256_file(paths.biotsavart),
            },
            "surface": {
                "path": str(paths.surface.resolve()),
                "sha256": _sha256_file(paths.surface),
            },
            "boozersurface": {
                "path": str(paths.boozersurface.resolve()),
                "sha256": _sha256_file(paths.boozersurface),
            },
            "results": str(paths.results.resolve()),
        }
    write_json(paths.results, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
