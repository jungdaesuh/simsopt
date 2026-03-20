"""Tier 4 adjoint vs finite-difference validation with branch-stability checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.run_code_benchmark_common import summarize_result_fun
from benchmarks.validation_ladder_common import (
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    preparse_platform,
    print_provenance,
    relative_error,
    write_json,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an adjoint directional derivative against re-solve FD."
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
        "--samples",
        type=int,
        default=5,
        help="Random directional-derivative samples to try.",
    )
    parser.add_argument(
        "--min-stable-samples",
        type=int,
        default=2,
        help="Minimum number of branch-stable FD samples required for a PASS.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-4,
        help="Finite-difference perturbation magnitude.",
    )
    return parser.parse_args()


def build_probe_fixture():
    """Create a non-trivial public-lane Boozer solve for adjoint/FD probing."""
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, Volume, create_equally_spaced_curves
    from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX

    ncoils = 2
    nfp = 2
    stellsym = True
    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=stellsym,
        R0=1.0,
        R1=0.5,
        order=3,
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    for current in base_currents:
        current.fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    mpol = 3
    ntor = 3
    nphi = 2 * ntor + 1
    ntheta = 2 * mpol + 1
    surf = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, ntheta, endpoint=False),
    )
    seed_surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=0,
        quadpoints_phi=surf.quadpoints_phi,
        quadpoints_theta=surf.quadpoints_theta,
    )
    seed_surface.set_rc(0, 0, 1.0)
    seed_surface.set_rc(1, 0, 0.15)
    seed_surface.set_zs(1, 0, 0.15)
    surf.least_squares_fit(seed_surface.gamma())

    bs_jax = BiotSavartJAX(coils)
    volume = Volume(surf)
    vol_target = volume.J()
    mu0 = 4 * np.pi * 1e-7
    G0 = mu0 * sum(abs(coil.current.get_value()) for coil in coils)
    iota0 = 0.3
    booz = BoozerSurfaceJAX(
        bs_jax,
        surf,
        volume,
        vol_target,
        constraint_weight=1.0,
        options={
            "verbose": False,
            "bfgs_maxiter": 300,
            "bfgs_tol": 1e-8,
            "newton_maxiter": 20,
            "newton_tol": 1e-9,
            "optimizer_backend": "scipy",
        },
    )
    return bs_jax, booz, iota0, G0


def main() -> None:
    args = parse_args()
    bootstrap_local_simsopt()
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Adjoint FD validation",
        extra={
            "platform_request": args.platform,
            "samples": int(args.samples),
            "eps": float(args.eps),
        },
    )
    print_provenance(provenance)

    from simsopt.geo.surfaceobjectives_jax import IotasJAX

    bs_jax, booz_jax, iota0, G0 = build_probe_fixture()
    base_result = booz_jax.run_code(iota0, G0)
    if base_result is None or not base_result.get("success", False):
        raise RuntimeError("Baseline Boozer solve failed; cannot run adjoint FD validation.")

    iota_objective = IotasJAX(booz_jax)
    gradient = np.asarray(iota_objective.dJ(), dtype=float)

    base_state = {
        "surface_dofs": np.asarray(booz_jax.surface.get_dofs(), dtype=float).copy(),
        "iota": float(booz_jax.res["iota"]),
        "G": float(booz_jax.res["G"]),
        "fun": float(summarize_result_fun(booz_jax.res)),
    }
    x0 = bs_jax.x.copy()

    def resolve_iota_at(coil_dofs: np.ndarray) -> dict:
        bs_jax.x = coil_dofs
        booz_jax.surface.set_dofs(base_state["surface_dofs"].copy())
        booz_jax.res["iota"] = base_state["iota"]
        booz_jax.res["G"] = base_state["G"]
        result = booz_jax.run_code(base_state["iota"], base_state["G"])
        if result is None or not result.get("success", False):
            return {"stable": False, "reason": "solve_failed"}
        if booz_jax.surface.is_self_intersecting():
            return {"stable": False, "reason": "self_intersecting"}
        iota_value = float(result["iota"])
        g_value = float(result["G"])
        fun_value = float(summarize_result_fun(result))
        stable = (
            abs(iota_value - base_state["iota"]) < 5e-3
            and relative_error(g_value, base_state["G"]) < 5e-3
            and relative_error(fun_value, base_state["fun"]) < 0.25
        )
        return {
            "stable": stable,
            "reason": "ok" if stable else "branch_switch",
            "iota": iota_value,
            "G": g_value,
            "fun": fun_value,
        }

    samples: list[dict] = []
    stable_samples = 0
    failures: list[str] = []
    rng = np.random.RandomState(42)
    for sample_index in range(args.samples):
        direction = rng.randn(len(x0))
        direction /= np.linalg.norm(direction)

        plus = resolve_iota_at(x0 + args.eps * direction)
        minus = resolve_iota_at(x0 - args.eps * direction)

        if not plus["stable"] or not minus["stable"]:
            samples.append(
                {
                    "sample_index": sample_index,
                    "accepted": False,
                    "plus_reason": plus["reason"],
                    "minus_reason": minus["reason"],
                }
            )
            print(
                f"sample {sample_index}: rejected "
                f"(plus={plus['reason']}, minus={minus['reason']})"
            )
            continue

        stable_samples += 1
        adjoint_directional = float(np.dot(gradient, direction))
        fd_directional = (plus["iota"] - minus["iota"]) / (2.0 * args.eps)
        abs_err = abs(adjoint_directional - fd_directional)
        rel_err = abs_err / (abs(fd_directional) + 1e-30)
        accepted = rel_err < 1e-2
        samples.append(
            {
                "sample_index": sample_index,
                "accepted": accepted,
                "adjoint_directional": adjoint_directional,
                "fd_directional": fd_directional,
                "abs_err": abs_err,
                "rel_err": rel_err,
                "plus_iota": plus["iota"],
                "minus_iota": minus["iota"],
            }
        )
        print(
            f"sample {sample_index}: adjoint={adjoint_directional:.6e} "
            f"fd={fd_directional:.6e} rel_err={rel_err:.2e}"
        )
        if not accepted:
            failures.append(
                f"Stable sample {sample_index} exceeded rel_err tolerance: {rel_err:.2e}"
            )

    bs_jax.x = x0
    booz_jax.surface.set_dofs(base_state["surface_dofs"].copy())

    if stable_samples < args.min_stable_samples:
        failures.append(
            f"Only {stable_samples} stable FD samples were found; need at least {args.min_stable_samples}."
        )

    payload = {
        "provenance": provenance,
        "baseline": {
            "iota": base_state["iota"],
            "G": base_state["G"],
            "fun": base_state["fun"],
            "gradient_norm": float(np.linalg.norm(gradient)),
        },
        "stable_samples": stable_samples,
        "samples": samples,
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
