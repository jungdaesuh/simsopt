"""Tier 4 adjoint pipeline validation on the real single-stage fixture."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

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
from benchmarks.run_code_benchmark_common import summarize_result_fun
from benchmarks.adjoint_probe_common import (
    compute_adjoint_state,
    compute_implicit_gradient_correction,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_IOTA_TARGET,
    DEFAULT_OPTIMIZER_BACKEND,
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
    DEFAULT_STAGE2_BS_PATH,
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
_STABLE_IOTA_ABS_TOL = 5e-3
_STABLE_G_REL_TOL = 5e-3
_STABLE_FUN_REL_TOL = 0.25


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the stable adjoint/VJP pipeline, fixed-surface FD, "
            "and full re-solve FD on the real reduced fixture."
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
        default=str(DEFAULT_STAGE2_BS_PATH),
        help="Path to the fixed Stage 2 seed biot_savart_opt.json fixture.",
    )
    parser.add_argument(
        "--nphi",
        type=int,
        default=DEFAULT_SMOKE_NPHI,
        help="Surface toroidal grid points.",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=DEFAULT_SMOKE_NTHETA,
        help="Surface poloidal grid points.",
    )
    parser.add_argument(
        "--mpol",
        type=int,
        default=DEFAULT_SMOKE_MPOL,
        help="Surface poloidal mode count.",
    )
    parser.add_argument(
        "--ntor",
        type=int,
        default=DEFAULT_SMOKE_NTOR,
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
        type=_positive_float,
        default=1e-4,
        help="Finite-difference perturbation magnitude.",
    )
    return parser.parse_args()


def compute_direct_and_total_gradients(
    jr_jax,
    bs_jax,
    implicit_correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the fixed-surface direct term and the full reduced gradient."""
    from simsopt.geo.surfaceobjectives_jax import _value_and_direct_coil_derivative

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
    _, direct_derivative = _value_and_direct_coil_derivative(
        bs_jax,
        jr_jax._direct_objective_value_and_grad,
        coil_dofs,
        x_inner,
        optimize_G,
        weight_inv_modB,
    )
    direct_gradient = np.asarray(direct_derivative(bs_jax), dtype=float)
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

    from simsopt.geo.boozer_residual_jax import boozer_residual_vector

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
    return build_real_single_stage_init_fixture(
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
    from simsopt.geo.surfaceobjectives_jax import BoozerResidualJAX
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

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
    is_self_intersecting, check_available = (
        single_stage_example.evaluate_surface_self_intersection(booz_jax.surface)
    )
    if check_available and is_self_intersecting:
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
        directional_fd = (
            float(plus["objective"]) - float(minus["objective"])
        ) / (2.0 * eps)
        abs_err = abs(directional_grad - directional_fd)
        rel_err = abs_err / (abs(directional_fd) + 1e-30)
        accepted = rel_err < FULL_RESOLVE_FD_REL_TOL or abs_err < FULL_RESOLVE_FD_ABS_TOL
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


def evaluate_adjoint_validation(metrics: dict[str, Any]) -> list[str]:
    """Return ladder failures for the stable adjoint validation contract."""
    failures: list[str] = []
    adjoint_residual_rel = float(metrics["adjoint_residual_rel"])
    if adjoint_residual_rel >= ADJOINT_RESIDUAL_REL_TOL:
        failures.append(
            f"Adjoint solve residual too large: {adjoint_residual_rel:.2e}"
        )

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
    return failures


def main() -> None:
    args = parse_args()
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
            "stage2_seed_path": str(Path(args.stage2_bs_path)),
            "optimizer_backend": args.optimizer_backend,
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "samples": int(args.samples),
            "eps": float(args.eps),
            "compile_behavior": describe_compile_behavior(uses_subprocesses=False),
            "optimizer_drift_tolerances": dict(_TIER4_TOLERANCES),
        },
    )
    print_provenance(provenance)

    fixture = _build_real_fixture_at(args)
    bs_jax = fixture["bs"]
    booz_jax = fixture["boozer_surface"]
    base_result = booz_jax.res
    if base_result is None or not base_result.get("success", False):
        raise RuntimeError("Baseline Boozer solve failed; cannot run adjoint validation.")

    from simsopt.geo.surfaceobjectives_jax import BoozerResidualJAX

    jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
    adjoint, adjoint_residual_rel = compute_adjoint_state(jr_jax)
    implicit_gradient = compute_implicit_gradient_correction(jr_jax, bs_jax, adjoint)
    direct_gradient, total_gradient, recomposed_total_rel = compute_direct_and_total_gradients(
        jr_jax,
        bs_jax,
        implicit_gradient,
    )
    _, fd_samples = compute_fixed_surface_fd_samples(
        bs_jax,
        booz_jax,
        direct_gradient,
        samples=args.samples,
        eps=args.eps,
    )
    base_state = {
        "coil_dofs": np.asarray(bs_jax.x, dtype=float).copy(),
        "surface_dofs": np.asarray(booz_jax.surface.get_dofs(), dtype=float).copy(),
        "iota": float(booz_jax.res["iota"]),
        "G": float(booz_jax.res["G"]),
        "fun": float(summarize_result_fun(booz_jax.res)),
    }
    stable_resolve_fd_samples, full_resolve_fd_samples = compute_full_resolve_fd_samples(
        args,
        total_gradient,
        base_state,
        samples=args.samples,
        eps=args.eps,
    )

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
