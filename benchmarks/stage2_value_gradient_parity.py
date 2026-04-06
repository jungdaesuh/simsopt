"""Tier 1 Stage 2 squared-flux value/gradient parity probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import time

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
    l2_relative_error,
    load_json,
    maybe_initialize_distributed_runtime,
    optimizer_drift_tolerances,
    max_relative_error,
    preparse_platform,
    print_provenance,
    require_x64_runtime,
    relative_error,
    resolve_probe_lane,
    repo_pythonpath_env,
    run_python_script,
    write_json,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_PLASMA_SURF_FILENAME,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)
apply_compilation_cache_policy()

import jax
import jaxlib

maybe_initialize_distributed_runtime()
jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context="Stage 2 value/gradient parity")


_TIER1_TOLERANCES = optimizer_drift_tolerances("tier1_stage2_value_gradient")
OBJECTIVE_REL_TOL = _TIER1_TOLERANCES["objective_rel_tol"]
GRADIENT_RTOL = _TIER1_TOLERANCES["gradient_rtol"]
GRADIENT_ATOL = _TIER1_TOLERANCES["gradient_atol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Stage 2 SquaredFlux value/gradient parity on CPU vs JAX."
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before import/use.",
    )
    parser.add_argument(
        "--fixture",
        choices=("procedural", "real"),
        default="procedural",
        help="Procedural production-grid probe (Tier 1a) or real Stage 2 setup (Tier 1b).",
    )
    parser.add_argument("--nphi", type=int, default=255, help="Surface toroidal grid points.")
    parser.add_argument("--ntheta", type=int, default=64, help="Surface poloidal grid points.")
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write structured parity results.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=DEFAULT_PLASMA_SURF_FILENAME,
        help="VMEC equilibrium filename for the real Stage 2 fixture.",
    )
    parser.add_argument(
        "--equilibria-dir",
        default=str(DEFAULT_EQUILIBRIA_DIR),
        help="Directory that contains VMEC equilibrium files.",
    )
    parser.add_argument(
        "--equilibrium-path",
        default=None,
        help="Explicit equilibrium path for the real Stage 2 fixture.",
    )
    return parser.parse_args()


def build_procedural_fixture(nphi: int, ntheta: int):
    """Create a production-grid procedural Stage 2 parity fixture."""
    from simsopt.field import BiotSavart, Coil, Current
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import CurveXYZFourier, SurfaceRZFourier, create_equally_spaced_curves
    from simsopt.objectives import SquaredFlux
    from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX

    tf_curves = create_equally_spaced_curves(
        20,
        1,
        stellsym=False,
        R0=0.976,
        R1=0.4,
        order=1,
    )
    tf_currents = [Current(1e5) for _ in range(20)]
    for tf_curve in tf_curves:
        tf_curve.fix_all()
    for tf_current in tf_currents:
        tf_current.fix_all()
    tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]

    banana_curve = CurveXYZFourier(
        np.linspace(0.0, 1.0, 128, endpoint=False),
        order=1,
    )
    banana_curve.full_x = tf_curves[0].full_x.copy()
    banana_coil = Coil(banana_curve, Current(1e5))
    banana_coil.current.fix_all()

    all_coils = tf_coils + [banana_coil]

    surf = SurfaceRZFourier(
        nfp=1,
        stellsym=False,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, ntheta, endpoint=False),
    )
    surf.set_rc(0, 0, 0.915)
    surf.set_rc(1, 0, 0.16)
    surf.set_zs(1, 0, 0.16)
    surf.fix_all()

    points = surf.gamma().reshape((-1, 3))

    bs_cpu = BiotSavart(all_coils)
    bs_cpu.set_points(points)
    flux_cpu = SquaredFlux(surf, bs_cpu)

    bs_jax = BiotSavartJAX(all_coils)
    flux_jax = SquaredFluxJAX(surf, bs_jax)
    return flux_cpu, flux_jax


def run_procedural_fixture(args: argparse.Namespace) -> dict:
    """Evaluate the procedural production-grid parity fixture."""
    flux_cpu, flux_jax = build_procedural_fixture(args.nphi, args.ntheta)

    cpu_start = time.perf_counter()
    j_cpu = float(flux_cpu.J())
    grad_cpu = np.asarray(flux_cpu.dJ(), dtype=float)
    cpu_elapsed_s = time.perf_counter() - cpu_start

    jax_start = time.perf_counter()
    j_jax = float(flux_jax.J())
    grad_jax = np.asarray(flux_jax.dJ(), dtype=float)
    jax_elapsed_s = time.perf_counter() - jax_start

    return {
        "cpu": {
            "J": j_cpu,
            "grad_norm": float(np.linalg.norm(grad_cpu)),
            "dof_count": int(grad_cpu.size),
            "elapsed_s": float(cpu_elapsed_s),
        },
        "jax": {
            "J": j_jax,
            "grad_norm": float(np.linalg.norm(grad_jax)),
            "dof_count": int(grad_jax.size),
            "elapsed_s": float(jax_elapsed_s),
        },
        "comparisons": {
            "j_rel_err": relative_error(j_jax, j_cpu),
            "grad_l2_rel_err": l2_relative_error(grad_jax, grad_cpu),
            "grad_max_rel_err": max_relative_error(grad_jax, grad_cpu),
            "grad_max_abs_err": float(np.max(np.abs(grad_jax - grad_cpu))),
            "grad_allclose": bool(
                np.allclose(
                    grad_jax,
                    grad_cpu,
                    rtol=GRADIENT_RTOL,
                    atol=GRADIENT_ATOL,
                )
            ),
        },
    }


def run_real_fixture(args: argparse.Namespace) -> dict:
    """Evaluate the real Stage 2 setup by reusing the example driver."""
    stage2_script = (
        REPO_ROOT
        / "examples"
        / "single_stage_optimization"
        / "STAGE_2"
        / "banana_coil_solver.py"
    )
    common_args = [
        "--probe-only",
        "--nphi",
        str(args.nphi),
        "--ntheta",
        str(args.ntheta),
    ]
    if args.equilibrium_path:
        common_args.extend(["--equilibrium-path", args.equilibrium_path])
    else:
        common_args.extend(
            [
                "--plasma-surf-filename",
                args.plasma_surf_filename,
                "--equilibria-dir",
                args.equilibria_dir,
            ]
        )

    with tempfile.TemporaryDirectory(prefix="stage2-tier1-") as temp_dir:
        cpu_json = str(Path(temp_dir) / "cpu_snapshot.json")
        jax_json = str(Path(temp_dir) / "jax_snapshot.json")

        cpu_start = time.perf_counter()
        run_python_script(
            stage2_script,
            ["--backend", "cpu", "--export-objective-json", cpu_json, *common_args],
            env=repo_pythonpath_env(
                platform="cpu",
                clear_backend_guardrails=True,
            ),
            cwd=REPO_ROOT,
            bootstrap_repo=True,
            stream_output=True,
        )
        cpu_elapsed_s = time.perf_counter() - cpu_start

        jax_start = time.perf_counter()
        run_python_script(
            stage2_script,
            ["--backend", "jax", "--export-objective-json", jax_json, *common_args],
            env=repo_pythonpath_env(platform=args.platform),
            cwd=REPO_ROOT,
            bootstrap_repo=True,
            stream_output=True,
        )
        jax_elapsed_s = time.perf_counter() - jax_start

        cpu_payload = load_json(cpu_json)
        jax_payload = load_json(jax_json)

    cpu_flux = cpu_payload["squared_flux"]
    jax_flux = jax_payload["squared_flux"]
    grad_cpu = np.asarray(cpu_flux["dJ"], dtype=float)
    grad_jax = np.asarray(jax_flux["dJ"], dtype=float)

    return {
        "cpu": {
            "J": float(cpu_flux["J"]),
            "grad_norm": float(cpu_flux["grad_norm"]),
            "dof_count": int(cpu_payload["dof_count"]),
            "equilibrium_path": cpu_payload["equilibrium_path"],
            "elapsed_s": float(cpu_elapsed_s),
            **(
                {"sharding_summaries": cpu_payload["sharding_summaries"]}
                if "sharding_summaries" in cpu_payload
                else {}
            ),
        },
        "jax": {
            "J": float(jax_flux["J"]),
            "grad_norm": float(jax_flux["grad_norm"]),
            "dof_count": int(jax_payload["dof_count"]),
            "equilibrium_path": jax_payload["equilibrium_path"],
            "elapsed_s": float(jax_elapsed_s),
            **(
                {"sharding_summaries": jax_payload["sharding_summaries"]}
                if "sharding_summaries" in jax_payload
                else {}
            ),
        },
        "comparisons": {
            "j_rel_err": relative_error(float(jax_flux["J"]), float(cpu_flux["J"])),
            "grad_l2_rel_err": l2_relative_error(grad_jax, grad_cpu),
            "grad_max_rel_err": max_relative_error(grad_jax, grad_cpu),
            "grad_max_abs_err": float(np.max(np.abs(grad_jax - grad_cpu))),
            "grad_allclose": bool(
                np.allclose(
                    grad_jax,
                    grad_cpu,
                    rtol=GRADIENT_RTOL,
                    atol=GRADIENT_ATOL,
                )
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.fixture == "procedural":
        bootstrap_local_simsopt()
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Stage 2 value/gradient parity",
        extra={
            "lane": resolve_probe_lane(),
            "fixture": args.fixture,
            "platform_request": args.platform,
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "compile_behavior": describe_compile_behavior(
                uses_subprocesses=args.fixture == "real"
            ),
            "optimizer_drift_tolerances": dict(_TIER1_TOLERANCES),
        },
    )
    print_provenance(provenance)

    results = (
        run_procedural_fixture(args)
        if args.fixture == "procedural"
        else run_real_fixture(args)
    )

    failures: list[str] = []
    if not np.isfinite(results["cpu"]["J"]):
        failures.append("CPU squared-flux value is non-finite.")
    if not np.isfinite(results["jax"]["J"]):
        failures.append("JAX squared-flux value is non-finite.")
    if not np.isfinite(results["comparisons"]["j_rel_err"]):
        failures.append("Squared-flux relative error is non-finite.")
    if not np.isfinite(results["comparisons"]["grad_l2_rel_err"]):
        failures.append("Squared-flux gradient L2 relative error is non-finite.")
    if results["comparisons"]["j_rel_err"] >= OBJECTIVE_REL_TOL:
        failures.append(
            f"Squared-flux value relative error too large: {results['comparisons']['j_rel_err']:.2e}"
        )
    if not results["comparisons"]["grad_allclose"]:
        failures.append(
            "Squared-flux gradient parity failed "
            f"np.allclose(rtol={GRADIENT_RTOL:.0e}, atol={GRADIENT_ATOL:.0e})."
        )

    print(
        "CPU vs JAX: "
        f"J rel_err={results['comparisons']['j_rel_err']:.2e}, "
        f"grad L2 rel_err={results['comparisons']['grad_l2_rel_err']:.2e}, "
        f"grad max rel_err={results['comparisons']['grad_max_rel_err']:.2e}, "
        f"grad max abs_err={results['comparisons']['grad_max_abs_err']:.2e}, "
        f"grad allclose={results['comparisons']['grad_allclose']}"
    )

    payload = {
        "provenance": provenance,
        "results": results,
        "failures": failures,
        "passed": not failures,
    }
    write_json(args.output_json, payload)
    if failures:
        print("STAGE 2 VALUE/GRADIENT PARITY FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("STAGE 2 VALUE/GRADIENT PARITY PASSED")


if __name__ == "__main__":
    main()
