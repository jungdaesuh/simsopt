"""Tier 4 adjoint pipeline validation on a stable public-lane fixture."""

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
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    preparse_platform,
    print_provenance,
    write_json,
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

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)


ADJOINT_RESIDUAL_REL_TOL = 1e-10
RECOMPOSED_TOTAL_REL_TOL = 1e-12
FIXED_SURFACE_FD_REL_TOL = 1e-3
FIXED_SURFACE_FD_ABS_TOL = 1e-8


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
        description="Validate the stable adjoint/VJP pipeline plus fixed-surface FD."
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
        choices=("scipy", "hybrid", "ondevice"),
        default=DEFAULT_OPTIMIZER_BACKEND,
        help="JAX Boozer optimizer backend for the adjoint probe.",
    )
    parser.add_argument(
        "--samples",
        type=_positive_int,
        default=3,
        help="Random fixed-surface finite-difference samples to try.",
    )
    parser.add_argument(
        "--eps",
        type=_positive_float,
        default=1e-4,
        help="Finite-difference perturbation magnitude.",
    )
    return parser.parse_args()


def compute_adjoint_state(jr_jax) -> tuple[np.ndarray, float]:
    """Return the objective-consistent adjoint vector and its residual."""
    from simsopt.objectives.utilities import forward_backward

    booz_jax = jr_jax.boozer_surface
    p_mat, l_mat, u_mat = booz_jax.res["PLU"]
    surface = jr_jax.surface
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size
    constraint_weight = (
        jr_jax.constraint_weight if jr_jax.constraint_weight is not None else 1.0
    )
    dJ_ds = jr_jax._compute_dJ_ds(
        booz_jax.res["iota"],
        booz_jax.res["G"],
        booz_jax.res.get("weight_inv_modB", True),
        constraint_weight,
        nphi,
        ntheta,
    )
    adj = forward_backward(p_mat, l_mat, u_mat, dJ_ds)
    hessian = p_mat @ l_mat @ u_mat
    residual = hessian.T @ adj - dJ_ds
    rel = float(np.linalg.norm(residual) / (np.linalg.norm(dJ_ds) + 1e-30))
    return adj, rel


def compute_implicit_gradient_correction(jr_jax, bs_jax, adj: np.ndarray) -> np.ndarray:
    """Project the adjoint cotangents back to coil DOFs."""
    from simsopt.geo.surfaceobjectives_jax import _coil_cotangents_to_derivative

    booz_jax = jr_jax.boozer_surface
    vjp_fn = booz_jax.res["vjp"]
    adj_cot = vjp_fn(adj, booz_jax, booz_jax.res["iota"], booz_jax.res["G"])
    adj_deriv = _coil_cotangents_to_derivative(bs_jax.coils, *adj_cot)
    return np.asarray(adj_deriv(bs_jax), dtype=float)


def compute_direct_and_total_gradients(
    jr_jax,
    bs_jax,
    implicit_correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the fixed-surface direct term and the full reduced gradient."""
    from simsopt.geo.surfaceobjectives_jax import _resolved_boozer_G

    booz_jax = jr_jax.boozer_surface
    total_gradient = np.asarray(jr_jax.dJ(), dtype=float)

    surface = jr_jax.surface
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size
    num_points = 3 * nphi * ntheta
    iota = booz_jax.res["iota"]
    g_value = booz_jax.res["G"]
    effective_g = _resolved_boozer_G(booz_jax)
    weight_inv_modB = booz_jax.res.get("weight_inv_modB", True)
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    b_field = bs_jax.B().reshape(nphi, ntheta, 3)
    dJ_dB = jr_jax._compute_dJ_by_dB(
        b_field,
        xphi,
        xtheta,
        iota,
        effective_g,
        weight_inv_modB,
        nphi,
        ntheta,
        num_points,
    )
    direct_derivative = bs_jax.B_vjp(dJ_dB)
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
    return failures


def main() -> None:
    args = parse_args()
    bootstrap_local_simsopt()
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Adjoint pipeline validation",
        extra={
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
        },
    )
    print_provenance(provenance)

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
    )
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

    print(f"adjoint residual: {adjoint_residual_rel:.2e}")
    print(f"implicit correction norm: {np.linalg.norm(implicit_gradient):.6e}")
    print(f"direct gradient norm: {np.linalg.norm(direct_gradient):.6e}")
    print(f"total gradient norm: {np.linalg.norm(total_gradient):.6e}")
    for sample in fd_samples:
        print(
            f"sample {sample['sample_index']}: direct={sample['direct_directional']:.6e} "
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
