"""Parity probe for CPU BoozerSurface vs JAX BoozerSurfaceJAX."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))


def _preparse_platform(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("auto", "cpu", "cuda"), default="auto")
    args, _ = parser.parse_known_args(argv)
    return args.platform


REQUESTED_PLATFORM = _preparse_platform(sys.argv[1:])
if REQUESTED_PLATFORM != "auto":
    os.environ["JAX_PLATFORMS"] = REQUESTED_PLATFORM

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)

from benchmarks.benchmark_problem import build_ls_parity_problem, clone_tensor_surface
from benchmarks.run_code_benchmark_common import summarize_result_fun


IOTA_TOL = 1e-3
LABEL_TOL = 1e-3
SOLVER_OPTIONS = {
    "verbose": False,
    "bfgs_maxiter": 300,
    "bfgs_tol": 1e-8,
    "newton_maxiter": 20,
    "newton_tol": 1e-9,
}


def get_git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def print_provenance() -> None:
    print(f"{'=' * 70}")
    print("run_code() parity probe")
    print(f"{'=' * 70}")
    print(f"repo sha:     {get_git_sha()}")
    print(f"jax:          {jax.__version__}")
    print(f"jaxlib:       {jaxlib.__version__}")
    print(f"backend:      {jax.default_backend()}")
    print(f"devices:      {jax.devices()}")
    print(f"x64 enabled:  {jax.numpy.zeros(1).dtype == jax.numpy.float64}")


def run_probe():
    from simsopt.field import BiotSavart
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import BoozerSurface, BoozerSurfaceJAX, Volume

    problem = build_ls_parity_problem()
    surf_cpu = clone_tensor_surface(problem.surface)
    surf_jax = clone_tensor_surface(problem.surface)
    vol_cpu = Volume(surf_cpu)
    vol_jax = Volume(surf_jax)
    bs_cpu = BiotSavart(problem.coils)
    bs_jax = BiotSavartJAX(problem.coils)

    booz_cpu = BoozerSurface(
        bs_cpu,
        surf_cpu,
        vol_cpu,
        problem.vol_target,
        constraint_weight=1.0,
        options=dict(SOLVER_OPTIONS),
    )
    booz_jax = BoozerSurfaceJAX(
        bs_jax,
        surf_jax,
        vol_jax,
        problem.vol_target,
        constraint_weight=1.0,
        options={**SOLVER_OPTIONS, "optimizer_backend": "scipy"},
    )

    res_cpu = booz_cpu.run_code(problem.iota0, problem.G0)
    res_jax = booz_jax.run_code(problem.iota0, problem.G0)
    if res_cpu is None or res_jax is None:
        raise RuntimeError("run_code() returned None during parity probe")

    cpu_label_err = abs(vol_cpu.J() - problem.vol_target)
    jax_label_err = abs(vol_jax.J() - problem.vol_target)
    iota_diff = abs(float(res_cpu["iota"]) - float(res_jax["iota"]))

    print(
        "CPU: "
        f"success={res_cpu['success']} iter={res_cpu['iter']} "
        f"iota={float(res_cpu['iota']):.6e} "
        f"label_err={cpu_label_err:.6e} "
        f"fun={summarize_result_fun(res_cpu):.6e}"
    )
    print(
        "JAX: "
        f"success={res_jax['success']} iter={res_jax['iter']} "
        f"iota={float(res_jax['iota']):.6e} "
        f"label_err={jax_label_err:.6e} "
        f"fun={summarize_result_fun(res_jax):.6e}"
    )
    print(f"|iota diff|={iota_diff:.6e}")

    failures = []
    if not res_cpu["success"]:
        failures.append("CPU solver did not converge")
    if not res_jax["success"]:
        failures.append("JAX solver did not converge")
    if abs(float(res_cpu["iota"])) >= IOTA_TOL:
        failures.append(f"CPU iota too large: {float(res_cpu['iota']):.6e}")
    if abs(float(res_jax["iota"])) >= IOTA_TOL:
        failures.append(f"JAX iota too large: {float(res_jax['iota']):.6e}")
    if cpu_label_err >= LABEL_TOL:
        failures.append(f"CPU label error too large: {cpu_label_err:.6e}")
    if jax_label_err >= LABEL_TOL:
        failures.append(f"JAX label error too large: {jax_label_err:.6e}")
    if iota_diff >= IOTA_TOL:
        failures.append(f"Iota disagreement too large: {iota_diff:.6e}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before importing JAX.",
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    print_provenance()
    failures = run_probe()
    if failures:
        print("PARITY PROBE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("PARITY PROBE PASSED")


if __name__ == "__main__":
    main()
