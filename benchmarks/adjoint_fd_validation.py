"""Tier 4 adjoint pipeline validation on the real single-stage fixture."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.validation_ladder_common import (
    apply_compilation_cache_policy,
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    describe_compile_behavior,
    optimizer_drift_tolerances,
    preparse_platform,
    print_provenance,
    relative_error,
    require_x64_runtime,
    resolve_probe_lane,
    write_json,
)
from benchmarks.validation_ladder_contract import (
    TIER4_ADJOINT_FD_EPS_LADDER,
    TIER4_ADJOINT_FD_MIN_MPOL,
    TIER4_ADJOINT_FD_EPS_WINDOW,
)
from benchmarks.run_code_benchmark_common import summarize_result_fun
from benchmarks.adjoint_probe_common import (
    compute_adjoint_state,
    compute_implicit_gradient_correction,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_IOTA_TARGET,
    DEFAULT_NUM_TF_COILS,
    DEFAULT_OPTIMIZER_BACKEND,
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_MPOL,
    DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NPHI,
    DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTHETA,
    DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTOR,
    DEFAULT_VOL_TARGET,
    build_real_single_stage_init_fixture,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)
apply_compilation_cache_policy()

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context="Adjoint FD validation")

_TIER4_TOLERANCES = optimizer_drift_tolerances("tier4_adjoint_fd")
ADJOINT_RESIDUAL_REL_TOL = _TIER4_TOLERANCES["adjoint_residual_rel_tol"]
RECOMPOSED_TOTAL_REL_TOL = _TIER4_TOLERANCES["recomposed_total_rel_tol"]
FIXED_SURFACE_FD_REL_TOL = _TIER4_TOLERANCES["fixed_surface_fd_rel_tol"]
FIXED_SURFACE_FD_ABS_TOL = _TIER4_TOLERANCES["fixed_surface_fd_abs_tol"]
FULL_RESOLVE_FD_REL_TOL = _TIER4_TOLERANCES["full_resolve_fd_rel_tol"]
FULL_RESOLVE_FD_ABS_TOL = _TIER4_TOLERANCES["full_resolve_fd_abs_tol"]
TRACEABLE_SINGLE_STAGE_FD_REL_TOL = _TIER4_TOLERANCES[
    "traceable_single_stage_fd_rel_tol"
]
TRACEABLE_SINGLE_STAGE_FD_ABS_TOL = _TIER4_TOLERANCES[
    "traceable_single_stage_fd_abs_tol"
]
OPERATOR_DOT_PRODUCT_REL_TOL = 1e-10
OPERATOR_DOT_PRODUCT_ABS_TOL = 1e-10
_STABLE_IOTA_ABS_TOL = 5e-3
_STABLE_G_REL_TOL = 5e-3
_STABLE_FUN_REL_TOL = 0.25


def _fd_eps_window_label() -> str:
    low, high = TIER4_ADJOINT_FD_EPS_WINDOW
    return f"[{low:.1e}, {high:.1e}]"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _production_fd_eps(value: str) -> float:
    parsed = _positive_float(value)
    low, high = TIER4_ADJOINT_FD_EPS_WINDOW
    if parsed < low or parsed > high:
        raise argparse.ArgumentTypeError(
            f"production FD eps must be in {_fd_eps_window_label()}"
        )
    return parsed


def _production_fd_mpol(value: str) -> int:
    parsed = int(value)
    if parsed < TIER4_ADJOINT_FD_MIN_MPOL:
        raise argparse.ArgumentTypeError(
            "Phase-1 production FD certification requires "
            f"mpol >= {TIER4_ADJOINT_FD_MIN_MPOL}; got mpol={parsed}."
        )
    return parsed


def _validate_production_fd_eps_ladder(ladder: tuple[float, ...]) -> tuple[float, ...]:
    if len(ladder) < 2:
        raise ValueError(
            "Phase-1 FD certification requires --eps-ladder with at least two "
            "strictly decreasing values."
        )
    low, high = TIER4_ADJOINT_FD_EPS_WINDOW
    for eps in ladder:
        if eps < low or eps > high:
            raise ValueError(
                f"production FD eps values must be in {_fd_eps_window_label()}"
            )
    for coarse_eps, fine_eps in zip(ladder, ladder[1:]):
        if fine_eps >= coarse_eps:
            raise ValueError(
                "Phase-1 FD certification requires a strictly decreasing --eps-ladder."
            )
    return ladder


def _resolve_fd_eps_ladder(args: argparse.Namespace) -> tuple[float, ...]:
    if args.eps is not None:
        raise ValueError(
            "Phase-1 FD certification requires --eps-ladder; a single --eps "
            "cannot certify coarse-to-fine Taylor error decrease."
        )
    ladder = tuple(float(value) for value in args.eps_ladder)
    return _validate_production_fd_eps_ladder(ladder)


def validate_production_fd_scale(args: argparse.Namespace) -> None:
    """Require a production mpol lane for the Phase-1 FD certification gate."""
    _production_fd_mpol(str(args.mpol))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the stable adjoint/VJP pipeline, fixed-surface FD, "
            "and full re-solve FD on the real single-stage fixture."
        )
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before import/use.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write structured validation results.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=DEFAULT_PLASMA_SURF_FILENAME,
        help="VMEC equilibrium filename for the real single-stage fixture.",
    )
    parser.add_argument(
        "--equilibria-dir",
        default=str(DEFAULT_EQUILIBRIA_DIR),
        help="Directory that contains VMEC equilibrium files.",
    )
    parser.add_argument(
        "--equilibrium-path",
        default=None,
        help="Explicit equilibrium path override.",
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=None,
        help=(
            "Path to the Stage 2 seed biot_savart_opt.json to certify "
            "(required with --raw-stage2-seed). No default: a certificate must "
            "state which seed it certifies."
        ),
    )
    runtime_seed_group = parser.add_mutually_exclusive_group()
    runtime_seed_group.add_argument(
        "--jax-runtime-seed-spec",
        default=None,
        help=(
            "Canonical JAX runtime seed spec (single_stage_jax_runtime_spec.json) "
            "to certify. Installs this resolved seed state and validates the fused "
            "traceable single-stage value/grad path without replaying raw Stage 2 "
            "Boozer initialization. No default: the seed must be explicit."
        ),
    )
    runtime_seed_group.add_argument(
        "--raw-stage2-seed",
        action="store_true",
        help=(
            "Replay the raw Stage 2 seed from --stage2-bs-path instead of a "
            "resolved JAX runtime seed. Preserves the legacy full-adjoint "
            "diagnostic path."
        ),
    )
    parser.add_argument(
        "--nphi",
        type=int,
        default=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NPHI,
        help="Surface toroidal grid points.",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTHETA,
        help="Surface poloidal grid points.",
    )
    parser.add_argument(
        "--mpol",
        type=_production_fd_mpol,
        default=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_MPOL,
        help=(
            "Surface poloidal mode count. Defaults to the Phase-1 production "
            "runtime-seed resolution "
            f"({DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_MPOL})."
        ),
    )
    parser.add_argument(
        "--ntor",
        type=int,
        default=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTOR,
        help="Surface toroidal mode count.",
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=DEFAULT_VOL_TARGET,
        help="Single-stage target volume.",
    )
    parser.add_argument(
        "--iota-target",
        type=float,
        default=DEFAULT_IOTA_TARGET,
        help="Single-stage target iota.",
    )
    parser.add_argument(
        "--optimizer-backend",
        choices=(DEFAULT_OPTIMIZER_BACKEND,),
        default=DEFAULT_OPTIMIZER_BACKEND,
        help="JAX target-lane optimizer backend for the adjoint probe.",
    )
    parser.add_argument(
        "--samples",
        type=_positive_int,
        default=3,
        help="Random fixed-surface and full re-solve finite-difference samples to try.",
    )
    parser.add_argument(
        "--min-stable-samples",
        type=_positive_int,
        default=2,
        help="Minimum number of branch-stable full re-solve FD samples required.",
    )
    parser.add_argument(
        "--eps",
        type=_production_fd_eps,
        default=None,
        help=(
            "Deprecated single finite-difference perturbation magnitude. "
            "Supplying --eps is rejected for certification; use --eps-ladder."
        ),
    )
    parser.add_argument(
        "--eps-ladder",
        type=_production_fd_eps,
        nargs="+",
        default=TIER4_ADJOINT_FD_EPS_LADDER,
        help=(
            "Production finite-difference perturbation ladder. Default: "
            + " ".join(f"{value:.1e}" for value in TIER4_ADJOINT_FD_EPS_LADDER)
        ),
    )
    args = parser.parse_args()
    if args.raw_stage2_seed:
        args.jax_runtime_seed_spec = None
        if args.stage2_bs_path is None:
            parser.error(
                "--raw-stage2-seed requires an explicit --stage2-bs-path "
                "(the raw Stage 2 seed to certify)."
            )
    elif args.jax_runtime_seed_spec is None:
        parser.error(
            "adjoint FD certification requires an explicit seed source: pass "
            "--jax-runtime-seed-spec <single_stage_jax_runtime_spec.json>, or "
            "--raw-stage2-seed with --stage2-bs-path <biot_savart_opt.json>. "
            "There is no default seed -- a certificate must state what it certifies."
        )
    try:
        _resolve_fd_eps_ladder(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def compute_direct_and_total_gradients(
    jr_jax,
    bs_jax,
    implicit_correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the fixed-surface direct term and the full reduced gradient."""
    from simsopt_jax_adapters.geo.surface_objectives import (
        _value_and_direct_coil_gradient,
    )

    booz_jax = jr_jax.boozer_surface
    total_gradient = np.asarray(jr_jax.dJ(), dtype=float)
    iota = booz_jax.res["iota"]
    g_value = booz_jax.res["G"]
    weight_inv_modB = booz_jax.res.get("weight_inv_modB", True)
    surface_dofs = booz_jax._get_surface_dofs()
    x_inner, optimize_G = jr_jax._inner_objective_state(
        iota,
        g_value,
        sdofs=surface_dofs,
    )
    coil_dofs = np.asarray(bs_jax.x.copy(), dtype=float)
    _, direct_gradient = _value_and_direct_coil_gradient(
        jr_jax._direct_objective_value_and_grad,
        coil_dofs,
        x_inner,
        optimize_G,
        weight_inv_modB,
    )
    direct_gradient = np.asarray(jax.device_get(direct_gradient), dtype=float)
    recomposed_total = direct_gradient - implicit_correction
    recomposed_rel = float(
        np.linalg.norm(total_gradient - recomposed_total)
        / (np.linalg.norm(total_gradient) + 1e-30)
    )
    return direct_gradient, total_gradient, recomposed_rel


def compute_fixed_surface_fd_samples(
    bs_jax,
    booz_jax,
    direct_gradient: np.ndarray,
    *,
    samples: int,
    eps: float,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    """Compare the fixed-surface direct-field term against directional FD."""
    import jax.numpy as jnp

    from simsopt_jax.geo.boozer_residual import boozer_residual_vector

    gamma_fixed = booz_jax.surface.gamma().reshape(-1, 3)
    xphi = jnp.asarray(booz_jax.surface.gammadash1())
    xtheta = jnp.asarray(booz_jax.surface.gammadash2())
    nphi = booz_jax.surface.quadpoints_phi.size
    ntheta = booz_jax.surface.quadpoints_theta.size
    num_pts = 3 * nphi * ntheta
    iota_sol = booz_jax.res["iota"]
    g_sol = booz_jax.res["G"]

    def j_at_fixed_surface(coil_x: np.ndarray) -> float:
        bs_jax.x = coil_x
        bs_jax.set_points(gamma_fixed)
        b_field = bs_jax.B().reshape(nphi, ntheta, 3)
        residual = boozer_residual_vector(g_sol, iota_sol, b_field, xphi, xtheta, True)
        return 0.5 * float(jnp.sum(residual**2)) / num_pts

    x0 = bs_jax.x.copy()
    rng = np.random.RandomState(42)
    sample_records: list[dict[str, float | int | bool]] = []
    for sample_index in range(samples):
        direction = rng.randn(len(x0))
        direction /= np.linalg.norm(direction)
        directional_grad = float(np.dot(direct_gradient, direction))
        directional_fd = (
            j_at_fixed_surface(x0 + eps * direction)
            - j_at_fixed_surface(x0 - eps * direction)
        ) / (2.0 * eps)
        abs_err = abs(directional_grad - directional_fd)
        rel_err = abs_err / (abs(directional_fd) + 1e-30)
        accepted = (
            rel_err < FIXED_SURFACE_FD_REL_TOL or abs_err < FIXED_SURFACE_FD_ABS_TOL
        )
        sample_records.append(
            {
                "sample_index": sample_index,
                "accepted": accepted,
                "direct_directional": directional_grad,
                "fd_directional": directional_fd,
                "abs_err": abs_err,
                "rel_err": rel_err,
            }
        )

    bs_jax.x = x0
    bs_jax.set_points(gamma_fixed)
    return direct_gradient, sample_records


def _build_real_fixture_at(
    args: argparse.Namespace,
    *,
    coil_dofs: np.ndarray | None = None,
    surface_dofs: np.ndarray | None = None,
    iota: float | None = None,
    G: float | None = None,
):
    fixture = build_real_single_stage_init_fixture(
        backend="jax",
        plasma_surf_filename=args.plasma_surf_filename,
        equilibria_dir=args.equilibria_dir,
        equilibrium_path=args.equilibrium_path,
        stage2_bs_path=args.stage2_bs_path,
        nphi=args.nphi,
        ntheta=args.ntheta,
        mpol=args.mpol,
        ntor=args.ntor,
        vol_target=args.vol_target,
        iota_target=args.iota_target,
        optimizer_backend=args.optimizer_backend,
        bs_dofs_override=coil_dofs,
        boozer_surface_dofs_override=surface_dofs,
        boozer_iota_override=iota,
        boozer_G_override=G,
    )
    fixture["num_tf_coils"] = DEFAULT_NUM_TF_COILS
    fixture["banana_curve_index"] = DEFAULT_NUM_TF_COILS
    return fixture


def _runtime_seed_coil_layout(runtime_spec) -> tuple[int, int]:
    return int(runtime_spec.seed.num_tf_coils), int(
        runtime_spec.seed.banana_curve_index
    )


def _build_jax_runtime_seed_fixture_at(args: argparse.Namespace):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    runtime_seed_state = single_stage_example.load_single_stage_jax_runtime_seed_spec(
        args.jax_runtime_seed_spec,
        mpol=args.mpol,
        ntor=args.ntor,
        nphi=args.nphi,
        ntheta=args.ntheta,
    )
    runtime_spec = runtime_seed_state["runtime_spec"]
    num_tf_coils, banana_curve_index = _runtime_seed_coil_layout(runtime_spec)
    bs_jax = single_stage_example.SingleStageRuntimeSpecBiotSavartJAX(runtime_spec)
    surf = single_stage_example.build_single_stage_surface_from_jax_runtime_spec(
        runtime_spec
    )
    tf_coils = bs_jax.coils[:num_tf_coils]
    current_sum = sum(abs(coil.current.get_value()) for coil in tf_coils)
    g0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))
    boozer_surface = single_stage_example.initialize_boozer_surface(
        surf,
        args.mpol,
        args.ntor,
        bs_jax,
        args.vol_target,
        1.0,
        args.iota_target,
        g0,
        backend="jax",
        optimizer_backend=single_stage_example.resolve_boozer_optimizer_backend(
            "jax",
            args.optimizer_backend,
            None,
        ),
        surface_dofs_override=runtime_seed_state["surface_dofs"],
        iota_override=float(runtime_seed_state["iota_scalar"]),
        G_override=float(runtime_seed_state["G_scalar"]),
        reuse_resolved_warm_start_solve=True,
    )
    _install_traceable_runtime_seed_linearization(boozer_surface)
    return {
        "bs": bs_jax,
        "boozer_surface": boozer_surface,
        "boozer_optimizer_backend": args.optimizer_backend,
        "boozer_least_squares_algorithm": None,
        "boozer_limited_memory": False,
        "equilibrium_path": str(args.jax_runtime_seed_spec),
        "stage2_bs_path": str(args.jax_runtime_seed_spec),
        "vol_target": float(args.vol_target),
        "iota_target": float(args.iota_target),
        "surface_shape": {
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
        },
        "runtime_seed_spec_path": str(args.jax_runtime_seed_spec),
        "num_tf_coils": num_tf_coils,
        "banana_curve_index": banana_curve_index,
    }


def _install_traceable_runtime_seed_linearization(boozer_surface) -> None:
    install_linearization = getattr(
        boozer_surface,
        "install_traceable_hessian_linearization_for_value_only_state",
        None,
    )
    if not callable(install_linearization):
        raise TypeError(
            "Runtime-seed FD certification requires "
            "install_traceable_hessian_linearization_for_value_only_state()."
        )
    install_linearization()


def _build_validation_fixture_at(args: argparse.Namespace):
    if args.jax_runtime_seed_spec is not None:
        return _build_jax_runtime_seed_fixture_at(args)
    return _build_real_fixture_at(args)


def _build_real_resolve_overrides(
    base_state: dict[str, float | np.ndarray],
) -> dict[str, float | np.ndarray]:
    return {
        "surface_dofs": np.asarray(base_state["surface_dofs"], dtype=float),
        "iota": float(base_state["iota"]),
        "G": float(base_state["G"]),
    }


def _is_stable_resolve(
    base_state: dict[str, float | np.ndarray],
    *,
    iota_value: float,
    g_value: float,
    fun_value: float,
) -> bool:
    return (
        abs(iota_value - float(base_state["iota"])) < _STABLE_IOTA_ABS_TOL
        and relative_error(g_value, float(base_state["G"])) < _STABLE_G_REL_TOL
        and relative_error(fun_value, float(base_state["fun"])) < _STABLE_FUN_REL_TOL
    )


def _resolve_total_objective_at(
    args: argparse.Namespace,
    base_state: dict[str, float | np.ndarray],
    coil_dofs: np.ndarray,
) -> dict[str, float | bool | str]:
    from simsopt_jax_adapters.geo.surface_objectives import BoozerResidualJAX

    fixture = _build_real_fixture_at(
        args,
        coil_dofs=coil_dofs,
        **_build_real_resolve_overrides(base_state),
    )
    bs_jax = fixture["bs"]
    booz_jax = fixture["boozer_surface"]
    result = booz_jax.res
    if result is None or not result.get("success", False):
        return {"stable": False, "reason": "solve_failed"}
    if bool(booz_jax.surface.is_self_intersecting()):
        return {"stable": False, "reason": "self_intersecting"}

    iota_value = float(result["iota"])
    g_value = float(result["G"])
    fun_value = float(summarize_result_fun(result))
    stable = _is_stable_resolve(
        base_state,
        iota_value=iota_value,
        g_value=g_value,
        fun_value=fun_value,
    )
    if not stable:
        return {
            "stable": False,
            "reason": "branch_switch",
            "iota": iota_value,
            "G": g_value,
            "fun": fun_value,
        }

    objective_value = float(BoozerResidualJAX(booz_jax, bs_jax).J())
    return {
        "stable": True,
        "reason": "ok",
        "objective": objective_value,
        "iota": iota_value,
        "G": g_value,
        "fun": fun_value,
    }


def compute_full_resolve_fd_samples(
    args: argparse.Namespace,
    total_gradient: np.ndarray,
    base_state: dict[str, float | np.ndarray],
    *,
    samples: int,
    eps: float,
) -> tuple[int, list[dict[str, float | int | bool | str]]]:
    x0 = np.asarray(base_state["coil_dofs"], dtype=float)
    rng = np.random.RandomState(42)
    stable_samples = 0
    sample_records: list[dict[str, float | int | bool | str]] = []
    for sample_index in range(samples):
        direction = rng.randn(len(x0))
        direction /= np.linalg.norm(direction)

        plus = _resolve_total_objective_at(args, base_state, x0 + eps * direction)
        minus = _resolve_total_objective_at(args, base_state, x0 - eps * direction)

        if not bool(plus["stable"]) or not bool(minus["stable"]):
            sample_records.append(
                {
                    "sample_index": sample_index,
                    "stable": False,
                    "accepted": False,
                    "plus_reason": str(plus["reason"]),
                    "minus_reason": str(minus["reason"]),
                }
            )
            continue

        stable_samples += 1
        directional_grad = float(np.dot(total_gradient, direction))
        directional_fd = (float(plus["objective"]) - float(minus["objective"])) / (
            2.0 * eps
        )
        abs_err = abs(directional_grad - directional_fd)
        rel_err = abs_err / (abs(directional_fd) + 1e-30)
        accepted = (
            rel_err < FULL_RESOLVE_FD_REL_TOL or abs_err < FULL_RESOLVE_FD_ABS_TOL
        )
        sample_records.append(
            {
                "sample_index": sample_index,
                "stable": True,
                "accepted": accepted,
                "total_directional": directional_grad,
                "fd_directional": directional_fd,
                "abs_err": abs_err,
                "rel_err": rel_err,
                "plus_iota": float(plus["iota"]),
                "minus_iota": float(minus["iota"]),
                "plus_fun": float(plus["fun"]),
                "minus_fun": float(minus["fun"]),
            }
        )
    return stable_samples, sample_records


def _traceable_forward_result_host(
    forward_result,
    coil_dofs: np.ndarray,
) -> dict[str, float | bool]:
    import jax.numpy as jnp

    result = forward_result(jnp.asarray(coil_dofs, dtype=jnp.float64))
    (
        value,
        success,
        primal_success,
        adjoint_linear_solve_available,
        iota,
        g_value,
    ) = jax.block_until_ready(
        (
            result["value"],
            result["success"],
            result["primal_success"],
            result["adjoint_linear_solve_available"],
            result["iota"],
            result["G"],
        )
    )
    return {
        "value": float(jax.device_get(value)),
        "success": bool(jax.device_get(success)),
        "primal_success": bool(jax.device_get(primal_success)),
        "adjoint_linear_solve_available": bool(
            jax.device_get(adjoint_linear_solve_available)
        ),
        "iota": float(jax.device_get(iota)),
        "G": float(jax.device_get(g_value)),
    }


def _traceable_value_and_grad_host(
    value_and_grad,
    coil_dofs: np.ndarray,
) -> tuple[float, np.ndarray]:
    import jax.numpy as jnp

    value, gradient = value_and_grad(jnp.asarray(coil_dofs, dtype=jnp.float64))
    value, gradient = jax.block_until_ready((value, gradient))
    return (
        float(jax.device_get(value)),
        np.asarray(jax.device_get(gradient), dtype=np.float64),
    )


def _fd_error_record(
    *,
    eps: float,
    directional_gradient: float,
    plus: Mapping[str, float | bool],
    minus: Mapping[str, float | bool],
    rel_tol: float,
    abs_tol: float,
) -> dict[str, object]:
    directional_fd = (float(plus["value"]) - float(minus["value"])) / (2.0 * eps)
    abs_err = abs(directional_gradient - directional_fd)
    rel_err = abs_err / (abs(directional_fd) + 1e-30)
    raw_accepted = (
        bool(np.isfinite(directional_fd))
        and bool(np.isfinite(abs_err))
        and bool(np.isfinite(rel_err))
        and (rel_err <= rel_tol or abs_err <= abs_tol)
    )
    if not bool(plus["primal_success"]) or not bool(minus["primal_success"]):
        status = "inner_solve_failed"
        accepted = False
    elif not bool(plus["success"]) or not bool(minus["success"]):
        status = "invalid_fd_window"
        accepted = False
    elif raw_accepted:
        status = "accepted"
        accepted = True
    else:
        status = "raw_mismatch"
        accepted = False
    return {
        "eps": float(eps),
        "accepted": bool(accepted),
        "raw_accepted": bool(raw_accepted),
        "status": status,
        "directional_gradient": float(directional_gradient),
        "fd_directional": float(directional_fd),
        "abs_err": float(abs_err),
        "rel_err": float(rel_err),
        "plus": dict(plus),
        "minus": dict(minus),
    }


def _is_clean_fd_record(record: Mapping[str, object]) -> bool:
    return str(record["status"]) in {"accepted", "raw_mismatch"}


def _richardson_fd_record(
    clean_records: list[dict[str, object]],
    directional_gradient: float,
) -> dict[str, float | bool] | None:
    if len(clean_records) < 2:
        return None
    coarse_record = clean_records[-2]
    fine_record = clean_records[-1]
    coarse_eps = float(coarse_record["eps"])
    fine_eps = float(fine_record["eps"])
    if fine_eps <= 0.0 or coarse_eps <= fine_eps:
        return None
    ratio = coarse_eps / fine_eps
    ratio_squared = ratio * ratio
    if ratio_squared <= 1.0:
        return None
    coarse_fd = float(coarse_record["fd_directional"])
    fine_fd = float(fine_record["fd_directional"])
    extrapolated_fd = (
        ratio_squared * fine_fd - coarse_fd
    ) / (ratio_squared - 1.0)
    abs_err = abs(directional_gradient - extrapolated_fd)
    rel_err = abs_err / (abs(extrapolated_fd) + 1e-30)
    accepted = (
        bool(np.isfinite(extrapolated_fd))
        and bool(np.isfinite(abs_err))
        and bool(np.isfinite(rel_err))
        and (
            rel_err <= TRACEABLE_SINGLE_STAGE_FD_REL_TOL
            or abs_err <= TRACEABLE_SINGLE_STAGE_FD_ABS_TOL
        )
    )
    return {
        "coarse_eps": float(coarse_eps),
        "fine_eps": float(fine_eps),
        "extrapolated_fd_directional": float(extrapolated_fd),
        "abs_err": float(abs_err),
        "rel_err": float(rel_err),
        "accepted": bool(accepted),
    }


def _observed_fd_order(clean_records: list[dict[str, object]]) -> float | None:
    if len(clean_records) < 2:
        return None
    coarse_record = clean_records[-2]
    fine_record = clean_records[-1]
    coarse_eps = float(coarse_record["eps"])
    fine_eps = float(fine_record["eps"])
    coarse_abs_err = float(coarse_record["abs_err"])
    fine_abs_err = float(fine_record["abs_err"])
    if (
        fine_eps <= 0.0
        or coarse_eps <= fine_eps
        or coarse_abs_err <= 0.0
        or fine_abs_err <= 0.0
    ):
        return None
    ratio = coarse_eps / fine_eps
    error_ratio = coarse_abs_err / fine_abs_err
    if ratio <= 1.0 or error_ratio <= 0.0:
        return None
    observed_order = np.log(error_ratio) / np.log(ratio)
    if not np.isfinite(observed_order):
        return None
    return float(observed_order)


def _traceable_fd_direction_status(
    *,
    eps_records: list[dict[str, object]],
    clean_records: list[dict[str, object]],
    richardson: dict[str, float | bool] | None,
    observed_order: float | None,
    taylor_rate_decrease: bool,
) -> str:
    if clean_records:
        finest_clean_record = clean_records[-1]
        if bool(finest_clean_record["accepted"]):
            return "accepted"
        if bool(richardson and richardson["accepted"]) and (
            observed_order is None or observed_order >= 1.5
        ):
            return "accepted"
    statuses = {str(record["status"]) for record in eps_records}
    if "inner_solve_failed" in statuses:
        return "inner_solve_failed"
    if "invalid_fd_window" in statuses:
        return "invalid_fd_window"
    if taylor_rate_decrease:
        return "needs_smaller_eps"
    return "gradient_mismatch"


def compute_traceable_single_stage_fd_ladder(
    forward_result,
    value_and_grad,
    coil_dofs: np.ndarray,
    *,
    samples: int,
    eps_ladder: tuple[float, ...],
) -> dict[str, object]:
    """Compare fused single-stage adjoint directions against central FD."""
    eps_ladder = _validate_production_fd_eps_ladder(eps_ladder)
    x0 = np.asarray(coil_dofs, dtype=np.float64)
    base_value, base_gradient = _traceable_value_and_grad_host(value_and_grad, x0)
    rng = np.random.RandomState(42)
    direction_records: list[dict[str, object]] = []
    for sample_index in range(samples):
        direction = rng.randn(len(x0))
        direction /= np.linalg.norm(direction)
        directional_gradient = float(np.dot(base_gradient, direction))
        eps_records: list[dict[str, object]] = []
        for eps in eps_ladder:
            plus = _traceable_forward_result_host(
                forward_result,
                x0 + eps * direction,
            )
            minus = _traceable_forward_result_host(
                forward_result,
                x0 - eps * direction,
            )
            eps_records.append(
                _fd_error_record(
                    eps=eps,
                    directional_gradient=directional_gradient,
                    plus=plus,
                    minus=minus,
                    rel_tol=TRACEABLE_SINGLE_STAGE_FD_REL_TOL,
                    abs_tol=TRACEABLE_SINGLE_STAGE_FD_ABS_TOL,
                )
            )

        clean_records = [
            dict(record) for record in eps_records if _is_clean_fd_record(record)
        ]
        if clean_records:
            coarse_abs_err = float(clean_records[0]["abs_err"])
            finest_abs_err = float(clean_records[-1]["abs_err"])
        else:
            coarse_abs_err = float(eps_records[0]["abs_err"])
            finest_abs_err = float(eps_records[-1]["abs_err"])
        taylor_rate_decrease = (
            finest_abs_err <= coarse_abs_err
            or finest_abs_err <= TRACEABLE_SINGLE_STAGE_FD_ABS_TOL
        )
        richardson = _richardson_fd_record(clean_records, directional_gradient)
        observed_order = _observed_fd_order(clean_records)
        status = _traceable_fd_direction_status(
            eps_records=eps_records,
            clean_records=clean_records,
            richardson=richardson,
            observed_order=observed_order,
            taylor_rate_decrease=bool(taylor_rate_decrease),
        )
        direction_records.append(
            {
                "sample_index": int(sample_index),
                "status": status,
                "accepted": status == "accepted",
                "taylor_rate_decrease": bool(taylor_rate_decrease),
                "observed_order": observed_order,
                "richardson": richardson,
                "directional_gradient": float(directional_gradient),
                "eps_records": eps_records,
            }
        )

    return {
        "validated_quantity": "fused_traceable_single_stage_gradient_after_inner_resolve",
        "base_value": float(base_value),
        "objective_scale": float(max(abs(base_value), 1.0)),
        "coil_dof_norm": float(np.linalg.norm(x0)),
        "gradient_norm": float(np.linalg.norm(base_gradient)),
        "gradient_finite": bool(np.all(np.isfinite(base_gradient))),
        "roundoff_floor_eps": float(
            np.cbrt(np.finfo(np.float64).eps) * max(float(np.linalg.norm(x0)), 1.0)
        ),
        "roundoff_floor_model": "cbrt(float64_eps)*max(norm(coil_dofs),1)",
        "eps_ladder": [float(eps) for eps in eps_ladder],
        "directions": direction_records,
        "rel_tol": float(TRACEABLE_SINGLE_STAGE_FD_REL_TOL),
        "abs_tol": float(TRACEABLE_SINGLE_STAGE_FD_ABS_TOL),
    }


def compute_operator_dot_product_contract(
    adjoint_runtime_state,
    *,
    samples: int,
    seed: int = 20260627,
) -> dict[str, object]:
    """Check the runtime linearization transpose without finite differences."""
    import jax.numpy as jnp

    rng = np.random.RandomState(seed)
    decision_size = int(adjoint_runtime_state.decision_size)
    sample_records: list[dict[str, float | bool | int]] = []
    for sample_index in range(samples):
        vector = rng.randn(decision_size)
        covector = rng.randn(decision_size)
        vector_dev = jnp.asarray(vector, dtype=jnp.float64)
        covector_dev = jnp.asarray(covector, dtype=jnp.float64)
        forward_action = adjoint_runtime_state.apply_forward(vector_dev)
        transpose_action = adjoint_runtime_state.apply_transpose(covector_dev)
        forward_action, transpose_action = jax.block_until_ready(
            (forward_action, transpose_action)
        )
        forward_np = np.asarray(jax.device_get(forward_action), dtype=np.float64)
        transpose_np = np.asarray(jax.device_get(transpose_action), dtype=np.float64)
        lhs = float(np.dot(forward_np, covector))
        rhs = float(np.dot(vector, transpose_np))
        abs_err = abs(lhs - rhs)
        rel_err = abs_err / (max(abs(lhs), abs(rhs), 1.0))
        accepted = (
            bool(np.isfinite(lhs))
            and bool(np.isfinite(rhs))
            and (
                rel_err <= OPERATOR_DOT_PRODUCT_REL_TOL
                or abs_err <= OPERATOR_DOT_PRODUCT_ABS_TOL
            )
        )
        sample_records.append(
            {
                "sample_index": int(sample_index),
                "accepted": bool(accepted),
                "lhs_forward_dot_covector": float(lhs),
                "rhs_vector_dot_transpose": float(rhs),
                "abs_err": float(abs_err),
                "rel_err": float(rel_err),
            }
        )
    return {
        "validated_quantity": "runtime_linearization_transpose_dot_product",
        "linear_solve_backend": str(adjoint_runtime_state.linear_solve_backend),
        "decision_size": int(decision_size),
        "rel_tol": float(OPERATOR_DOT_PRODUCT_REL_TOL),
        "abs_tol": float(OPERATOR_DOT_PRODUCT_ABS_TOL),
        "samples": sample_records,
        "passed": all(bool(record["accepted"]) for record in sample_records),
    }


def compute_optional_operator_dot_product_contract(
    booz_jax,
    *,
    samples: int,
) -> dict[str, object]:
    """Return a transpose check when the Boozer runtime exposes that state."""
    try:
        adjoint_runtime_state = booz_jax.get_adjoint_runtime_state()
    except RuntimeError as exc:
        if "no valid adjoint state" not in str(exc):
            raise
        return {
            "validated_quantity": "runtime_linearization_transpose_dot_product",
            "status": "unsupported",
            "blocking": False,
            "reason": "boozer_surface_adjoint_runtime_state_unavailable",
            "message": str(exc),
            "samples": [],
            "passed": True,
        }
    contract = compute_operator_dot_product_contract(
        adjoint_runtime_state,
        samples=samples,
    )
    contract["status"] = "evaluated"
    contract["blocking"] = True
    return contract


def traceable_single_stage_jvp_vjp_feasibility() -> dict[str, object]:
    """Record why the public runtime objective is not an exact JVP/VJP oracle."""
    return {
        "status": "unsupported",
        "blocking": False,
        "reason": "public_objective_custom_vjp_forward_mode_unavailable",
    }


def _build_traceable_single_stage_objective_config(fixture: dict[str, object]):
    from benchmarks.single_stage_objective_eval import (
        objective_weights_from_example_defaults,
    )
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    bs_jax = fixture["bs"]
    booz_jax = fixture["boozer_surface"]
    weights = objective_weights_from_example_defaults()
    banana_curve = bs_jax.coils[int(fixture["banana_curve_index"])].curve
    return single_stage_example.build_traceable_single_stage_outer_objective_config(
        booz_jax,
        bs_jax,
        banana_curve,
        single_stage_example.build_single_stage_vessel_surface(),
        non_qs_weight=weights.non_qs,
        residual_weight=weights.res,
        iota_weight=weights.iotas,
        length_weight=weights.length,
        length_target=weights.length_target,
        curve_curve_weight=weights.cc,
        curve_curve_threshold=weights.cc_dist,
        curve_surface_weight=weights.cs,
        curve_surface_threshold=weights.cs_dist,
        surface_vessel_weight=weights.surf_dist,
        surface_vessel_threshold=weights.ss_dist,
        curvature_weight=weights.curvature,
        curvature_threshold=weights.curvature_threshold,
    )


def traceable_single_stage_forward_operator_contract(
    runtime_bundle,
    coil_dofs: np.ndarray,
) -> dict[str, bool | str]:
    """Return proof that the public forward-result path is operator-backed."""
    import jax.numpy as jnp

    forward_result = runtime_bundle["forward_result"](
        jnp.asarray(coil_dofs, dtype=jnp.float64)
    )
    success = bool(jax.device_get(forward_result["success"]))
    linear_solve_factors = forward_result["linear_solve_factors"]
    backend = str(forward_result["linear_solve_backend"])
    dense_factors_available = bool(
        forward_result["dense_linear_solve_factors_available"]
    )
    return {
        "forward_success": success,
        "linear_solve_factors_none": linear_solve_factors is None,
        "dense_linear_solve_factors_available": dense_factors_available,
        "linear_solve_backend": backend,
    }


def evaluate_adjoint_validation(metrics: dict[str, Any]) -> list[str]:
    """Return ladder failures for the stable adjoint validation contract."""
    failures: list[str] = []
    adjoint_residual_rel = float(metrics["adjoint_residual_rel"])
    if adjoint_residual_rel >= ADJOINT_RESIDUAL_REL_TOL:
        failures.append(f"Adjoint solve residual too large: {adjoint_residual_rel:.2e}")

    if not bool(metrics["implicit_gradient_finite"]):
        failures.append("Implicit correction produced NaN/inf.")

    implicit_gradient_norm = float(metrics["implicit_gradient_norm"])
    if implicit_gradient_norm <= 0.0:
        failures.append("Implicit correction produced zero gradient.")

    if not bool(metrics["total_gradient_finite"]):
        failures.append("Total reduced gradient produced NaN/inf.")

    total_gradient_norm = float(metrics["total_gradient_norm"])
    if total_gradient_norm <= 0.0:
        failures.append("Total reduced gradient is zero.")

    recomposed_total_rel = float(metrics["recomposed_total_rel"])
    if recomposed_total_rel >= RECOMPOSED_TOTAL_REL_TOL:
        failures.append(
            f"Direct-minus-implicit recomposition drift too large: {recomposed_total_rel:.2e}"
        )

    if not metrics["fd_samples"]:
        failures.append("No fixed-surface FD samples were evaluated.")

    for sample in metrics["fd_samples"]:
        sample_record = dict(sample)
        if bool(sample_record["accepted"]):
            continue
        sample_index = int(sample_record["sample_index"])
        rel_err = float(sample_record["rel_err"])
        abs_err = float(sample_record["abs_err"])
        failures.append(
            f"Fixed-surface FD sample {sample_index} exceeded tolerance: "
            f"rel_err={rel_err:.2e}, abs_err={abs_err:.2e}"
        )

    full_resolve_fd_samples = metrics["full_resolve_fd_samples"]
    if not full_resolve_fd_samples:
        failures.append("No full re-solve FD samples were evaluated.")

    stable_resolve_fd_samples = int(metrics["stable_resolve_fd_samples"])
    min_stable_resolve_fd_samples = int(metrics["min_stable_resolve_fd_samples"])
    if stable_resolve_fd_samples < min_stable_resolve_fd_samples:
        failures.append(
            "Only "
            f"{stable_resolve_fd_samples} stable full re-solve FD samples were found; "
            f"need at least {min_stable_resolve_fd_samples}."
        )

    for sample in full_resolve_fd_samples:
        sample_record = dict(sample)
        if not bool(sample_record.get("stable", True)):
            continue
        if bool(sample_record["accepted"]):
            continue
        sample_index = int(sample_record["sample_index"])
        rel_err = float(sample_record["rel_err"])
        abs_err = float(sample_record["abs_err"])
        failures.append(
            f"Full re-solve FD sample {sample_index} exceeded tolerance: "
            f"rel_err={rel_err:.2e}, abs_err={abs_err:.2e}"
        )

    failures.extend(evaluate_traceable_single_stage_validation(metrics))
    return failures


def evaluate_traceable_single_stage_validation(metrics: dict[str, Any]) -> list[str]:
    """Return failures for the fused traceable single-stage FD certificate."""
    failures: list[str] = []
    forward_operator_contract = dict(metrics["forward_operator_contract"])
    if not bool(forward_operator_contract["forward_success"]):
        failures.append("Traceable split forward solve did not succeed.")
    if not bool(forward_operator_contract["linear_solve_factors_none"]):
        failures.append(
            "Traceable forward-result path exposed dense linear_solve_factors; "
            "Phase-1 FD gate is not exercising the forward operator path."
        )
    if bool(forward_operator_contract["dense_linear_solve_factors_available"]):
        failures.append(
            "Traceable forward-result metadata reports dense linear solve factors."
        )
    if str(forward_operator_contract["linear_solve_backend"]) != "operator":
        failures.append(
            "Traceable forward-result metadata did not report the operator backend."
        )

    traceable_single_stage_fd = dict(metrics["traceable_single_stage_fd"])
    if not bool(traceable_single_stage_fd["gradient_finite"]):
        failures.append("Traceable single-stage gradient produced NaN/inf.")
    if float(traceable_single_stage_fd["gradient_norm"]) <= 0.0:
        failures.append("Traceable single-stage gradient is zero.")
    directions = traceable_single_stage_fd["directions"]
    if not directions:
        failures.append("No traceable single-stage FD directions were evaluated.")
    for direction in directions:
        direction_record = dict(direction)
        sample_index = int(direction_record["sample_index"])
        if bool(direction_record["accepted"]):
            continue
        eps_records = list(direction_record["eps_records"])
        clean_records = [
            dict(record) for record in eps_records if _is_clean_fd_record(dict(record))
        ]
        finest_record = dict(clean_records[-1] if clean_records else eps_records[-1])
        rel_err = float(finest_record["rel_err"])
        abs_err = float(finest_record["abs_err"])
        status = str(direction_record.get("status", "gradient_mismatch"))
        if status in {"inner_solve_failed", "invalid_fd_window"}:
            failures.append(
                f"Traceable single-stage FD direction {sample_index} has "
                f"status={status}; FD window is not a valid gradient oracle "
                f"(rel_err={rel_err:.2e}, abs_err={abs_err:.2e})."
            )
            continue
        if not bool(direction_record["taylor_rate_decrease"]):
            failures.append(
                f"Traceable single-stage FD direction {sample_index} did not "
                "show coarse-to-fine Taylor error decrease."
            )
            continue
        failures.append(
            f"Traceable single-stage FD direction {sample_index} exceeded "
            f"status={status} tolerance: rel_err={rel_err:.2e}, abs_err={abs_err:.2e}"
        )
    operator_dot_product = metrics.get("operator_dot_product_contract")
    if operator_dot_product is not None:
        operator_dot_product = dict(operator_dot_product)
        if (
            str(operator_dot_product.get("status", "evaluated")) != "unsupported"
            and not bool(operator_dot_product["passed"])
        ):
            failures.append(
                "Runtime adjoint linearization failed the dot-product transpose check."
            )
    return failures


def main() -> None:
    args = parse_args()
    validate_production_fd_scale(args)
    fd_eps_ladder = _resolve_fd_eps_ladder(args)
    bootstrap_local_simsopt()
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Adjoint pipeline validation",
        extra={
            "lane": resolve_probe_lane(optimizer_backend=args.optimizer_backend),
            "fixture": "real-single-stage-init",
            "platform_request": args.platform,
            "plasma_surf_filename": args.plasma_surf_filename,
            "stage2_seed_path": (
                None
                if args.stage2_bs_path is None
                else str(Path(args.stage2_bs_path))
            ),
            "jax_runtime_seed_spec": (
                None
                if args.jax_runtime_seed_spec is None
                else str(Path(args.jax_runtime_seed_spec))
            ),
            "validation_mode": (
                "raw_fixture_full_adjoint"
                if args.jax_runtime_seed_spec is None
                else "traceable_runtime_seed"
            ),
            "optimizer_backend": args.optimizer_backend,
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "samples": int(args.samples),
            "eps_ladder": [float(eps) for eps in fd_eps_ladder],
            "eps_window": [float(value) for value in TIER4_ADJOINT_FD_EPS_WINDOW],
            "compile_behavior": describe_compile_behavior(uses_subprocesses=False),
            "optimizer_drift_tolerances": dict(_TIER4_TOLERANCES),
        },
    )
    print_provenance(provenance)

    fixture = _build_validation_fixture_at(args)
    bs_jax = fixture["bs"]
    booz_jax = fixture["boozer_surface"]
    base_result = booz_jax.res
    if base_result is None or not base_result.get("success", False):
        raise RuntimeError(
            "Baseline Boozer solve failed; cannot run adjoint validation."
        )

    runtime_seed_mode = args.jax_runtime_seed_spec is not None
    if runtime_seed_mode:
        adjoint_residual_rel = None
        implicit_gradient = np.asarray([], dtype=np.float64)
        direct_gradient = np.asarray([], dtype=np.float64)
        total_gradient = np.asarray([], dtype=np.float64)
        recomposed_total_rel = None
        fd_samples = []
        stable_resolve_fd_samples = 0
        full_resolve_fd_samples = []
    else:
        from simsopt_jax_adapters.geo.surface_objectives import BoozerResidualJAX

        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
        adjoint, adjoint_residual_rel = compute_adjoint_state(jr_jax)
        implicit_gradient = compute_implicit_gradient_correction(
            jr_jax, bs_jax, adjoint
        )
        direct_gradient, total_gradient, recomposed_total_rel = (
            compute_direct_and_total_gradients(
                jr_jax,
                bs_jax,
                implicit_gradient,
            )
        )
        _, fd_samples = compute_fixed_surface_fd_samples(
            bs_jax,
            booz_jax,
            direct_gradient,
            samples=args.samples,
            eps=fd_eps_ladder[0],
        )
        base_state = {
            "coil_dofs": np.asarray(bs_jax.x, dtype=float).copy(),
            "surface_dofs": np.asarray(booz_jax.surface.get_dofs(), dtype=float).copy(),
            "iota": float(booz_jax.res["iota"]),
            "G": float(booz_jax.res["G"]),
            "fun": float(summarize_result_fun(booz_jax.res)),
        }
        stable_resolve_fd_samples, full_resolve_fd_samples = (
            compute_full_resolve_fd_samples(
                args,
                total_gradient,
                base_state,
                samples=args.samples,
                eps=fd_eps_ladder[0],
            )
        )

    from simsopt_jax_adapters.geo.surface_objectives import (
        make_traceable_objective_runtime_bundle,
    )
    import jax.numpy as jnp

    traceable_outer_config = _build_traceable_single_stage_objective_config(fixture)
    traceable_iota_target = jnp.asarray(args.iota_target, dtype=jnp.float64)
    traceable_coil_dofs = np.asarray(bs_jax.x.copy(), dtype=np.float64)
    traceable_runtime_bundle = make_traceable_objective_runtime_bundle(
        booz_jax,
        bs_jax,
        traceable_iota_target,
        include_profile_suite=False,
        include_host_wrappers=False,
        outer_objective_config=traceable_outer_config,
    )
    forward_operator_contract = traceable_single_stage_forward_operator_contract(
        traceable_runtime_bundle,
        traceable_coil_dofs,
    )
    operator_dot_product_contract = compute_optional_operator_dot_product_contract(
        booz_jax,
        samples=args.samples,
    )
    jvp_vjp_feasibility = traceable_single_stage_jvp_vjp_feasibility()
    traceable_single_stage_fd = compute_traceable_single_stage_fd_ladder(
        traceable_runtime_bundle["forward_result"],
        traceable_runtime_bundle["value_and_grad"],
        traceable_coil_dofs,
        samples=args.samples,
        eps_ladder=fd_eps_ladder,
    )

    if runtime_seed_mode:
        print("legacy adjoint sections: skipped (jax runtime seed mode)")
    else:
        print(f"adjoint residual: {adjoint_residual_rel:.2e}")
        print(f"implicit correction norm: {np.linalg.norm(implicit_gradient):.6e}")
        print(f"direct gradient norm: {np.linalg.norm(direct_gradient):.6e}")
        print(f"total gradient norm: {np.linalg.norm(total_gradient):.6e}")
    for sample in fd_samples:
        print(
            f"sample {sample['sample_index']}: direct={sample['direct_directional']:.6e} "
            f"fd={sample['fd_directional']:.6e} rel_err={sample['rel_err']:.2e}"
        )
    for sample in full_resolve_fd_samples:
        if not bool(sample.get("stable", True)):
            print(
                f"re-solve sample {sample['sample_index']}: rejected "
                f"(plus={sample['plus_reason']}, minus={sample['minus_reason']})"
            )
            continue
        print(
            f"re-solve sample {sample['sample_index']}: total={sample['total_directional']:.6e} "
            f"fd={sample['fd_directional']:.6e} rel_err={sample['rel_err']:.2e}"
        )
    print(
        "traceable forward-operator contract: "
        f"success={forward_operator_contract['forward_success']} "
        f"backend={forward_operator_contract['linear_solve_backend']} "
        f"factors_none={forward_operator_contract['linear_solve_factors_none']}"
    )
    print(
        "operator dot-product contract: "
        f"passed={operator_dot_product_contract['passed']} "
        f"samples={len(operator_dot_product_contract['samples'])}"
    )
    print(
        "exact JVP/VJP feasibility: "
        f"status={jvp_vjp_feasibility['status']} "
        f"reason={jvp_vjp_feasibility['reason']}"
    )
    for direction in traceable_single_stage_fd["directions"]:
        finest_record = direction["eps_records"][-1]
        print(
            f"traceable FD direction {direction['sample_index']}: "
            f"eps={finest_record['eps']:.2e} "
            f"grad={finest_record['directional_gradient']:.6e} "
            f"fd={finest_record['fd_directional']:.6e} "
            f"rel_err={finest_record['rel_err']:.2e}"
        )

    traceable_metrics = {
        "forward_operator_contract": forward_operator_contract,
        "operator_dot_product_contract": operator_dot_product_contract,
        "jvp_vjp_feasibility": jvp_vjp_feasibility,
        "traceable_single_stage_fd": traceable_single_stage_fd,
    }
    if runtime_seed_mode:
        metrics = traceable_metrics
        failures = evaluate_traceable_single_stage_validation(metrics)
    else:
        metrics = {
            "adjoint_residual_rel": adjoint_residual_rel,
            "implicit_gradient_finite": bool(np.all(np.isfinite(implicit_gradient))),
            "implicit_gradient_norm": float(np.linalg.norm(implicit_gradient)),
            "total_gradient_finite": bool(np.all(np.isfinite(total_gradient))),
            "total_gradient_norm": float(np.linalg.norm(total_gradient)),
            "recomposed_total_rel": recomposed_total_rel,
            "fd_samples": fd_samples,
            "stable_resolve_fd_samples": stable_resolve_fd_samples,
            "min_stable_resolve_fd_samples": int(args.min_stable_samples),
            "full_resolve_fd_samples": full_resolve_fd_samples,
            **traceable_metrics,
        }
        failures = evaluate_adjoint_validation(metrics)

    payload = {
        "provenance": provenance,
        "baseline": {
            "iota": float(booz_jax.res["iota"]),
            "G": float(booz_jax.res["G"]),
            "solve_success": bool(base_result.get("success", False)),
            "equilibrium_path": str(fixture["equilibrium_path"]),
            "stage2_bs_path": str(fixture["stage2_bs_path"]),
            "jax_runtime_seed_spec": (
                None
                if args.jax_runtime_seed_spec is None
                else str(args.jax_runtime_seed_spec)
            ),
            "validation_mode": (
                "traceable_runtime_seed"
                if runtime_seed_mode
                else "raw_fixture_full_adjoint"
            ),
        },
        "adjoint": {
            "residual_rel": adjoint_residual_rel,
            "implicit_gradient_norm": float(np.linalg.norm(implicit_gradient)),
            "implicit_gradient_finite": bool(np.all(np.isfinite(implicit_gradient))),
            "total_gradient_norm": float(np.linalg.norm(total_gradient)),
            "total_gradient_finite": bool(np.all(np.isfinite(total_gradient))),
            "recomposed_total_rel": recomposed_total_rel,
        },
        "fixed_surface_fd": {
            "validated_quantity": "direct_gradient_at_fixed_surface",
            "gradient_norm": float(np.linalg.norm(direct_gradient)),
            "rel_tol": FIXED_SURFACE_FD_REL_TOL,
            "abs_tol": FIXED_SURFACE_FD_ABS_TOL,
            "samples": fd_samples,
        },
        "full_resolve_fd": {
            "validated_quantity": "total_gradient_after_full_resolve",
            "gradient_norm": float(np.linalg.norm(total_gradient)),
            "stable_samples": stable_resolve_fd_samples,
            "min_stable_samples": int(args.min_stable_samples),
            "rel_tol": FULL_RESOLVE_FD_REL_TOL,
            "abs_tol": FULL_RESOLVE_FD_ABS_TOL,
            "samples": full_resolve_fd_samples,
        },
        "forward_operator_contract": forward_operator_contract,
        "operator_dot_product_contract": operator_dot_product_contract,
        "jvp_vjp_feasibility": jvp_vjp_feasibility,
        "traceable_single_stage_fd": traceable_single_stage_fd,
        "failures": failures,
        "passed": not failures,
    }
    write_json(args.output_json, payload)
    if failures:
        print("ADJOINT FD VALIDATION FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("ADJOINT FD VALIDATION PASSED")


if __name__ == "__main__":
    main()
