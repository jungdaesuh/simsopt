from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SIMSOPT_ROOT = os.path.abspath(os.path.join(EXAMPLE_ROOT, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
SRC_ROOT = os.path.join(SIMSOPT_ROOT, "src")
sys.path.insert(0, EXAMPLE_ROOT)
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, SIMSOPT_ROOT)
sys.path.insert(0, REPO_ROOT)

from repo_bootstrap import bootstrap_local_simsopt


bootstrap_local_simsopt(SRC_ROOT)

from banana_opt.jax_banana_types import SingleStageWeights
from banana_opt.native_banana_drivers import (
    NativeSingleStageConfig,
    common_only_weights,
    run_native_single_stage,
)


DEFAULT_WEIGHTS = SingleStageWeights()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the native SIMSOPT CPU seven-term banana full loop with "
            "host SciPy L-BFGS-B and BoozerLS."
        )
    )
    parser.add_argument("--biotsavart-file", type=Path, required=True)
    parser.add_argument("--surface-path", type=Path, required=True)
    parser.add_argument("--boozer-state-path", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs_single_stage_native")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-config-sha256", required=True)
    parser.add_argument(
        "--objective-profile",
        choices=("common-seven-term",),
        default="common-seven-term",
    )

    parser.add_argument("--vmec-s", type=float, default=1.0)
    parser.add_argument("--surface-scale", type=float, default=None)
    parser.add_argument("--mpol", type=int, default=8)
    parser.add_argument("--ntor", type=int, default=8)
    parser.add_argument("--nphi", type=int, default=65)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument(
        "--iota-target", "--iota", dest="iota_target", type=float, default=0.15
    )

    parser.add_argument("--boozer-bfgs-tol", type=float, default=1.0e-10)
    parser.add_argument("--boozer-bfgs-maxiter", type=int, default=1500)
    parser.add_argument("--boozer-newton-tol", type=float, default=1.0e-11)
    parser.add_argument("--boozer-newton-maxiter", type=int, default=40)
    parser.add_argument("--boozer-limited-memory", action="store_true")

    parser.add_argument(
        "--outer-maxiter", "--maxiter", dest="outer_maxiter", type=int, default=50
    )
    parser.add_argument("--outer-maxcor", type=int, default=300)
    parser.add_argument("--outer-maxls", type=int, default=20)
    parser.add_argument("--outer-ftol", type=float, default=1.0e-15)
    parser.add_argument("--outer-gtol", type=float, default=1.0e-15)

    parser.add_argument(
        "--weight-non-quasisymmetric-ratio",
        type=float,
        default=DEFAULT_WEIGHTS.nonqs,
    )
    parser.add_argument(
        "--weight-boozer-residual",
        type=float,
        default=DEFAULT_WEIGHTS.bres,
    )
    parser.add_argument("--weight-iota", type=float, default=DEFAULT_WEIGHTS.iota)
    parser.add_argument(
        "--weight-coil-length",
        type=float,
        default=DEFAULT_WEIGHTS.length,
    )
    parser.add_argument(
        "--weight-coil-coil-distance",
        type=float,
        default=DEFAULT_WEIGHTS.ccdist,
    )
    parser.add_argument(
        "--weight-coil-surface-distance",
        type=float,
        default=DEFAULT_WEIGHTS.csdist,
    )
    parser.add_argument(
        "--weight-coil-curvature",
        type=float,
        default=DEFAULT_WEIGHTS.curvature,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    weights = common_only_weights(
        nonqs=args.weight_non_quasisymmetric_ratio,
        bres=args.weight_boozer_residual,
        iota=args.weight_iota,
        length=args.weight_coil_length,
        ccdist=args.weight_coil_coil_distance,
        csdist=args.weight_coil_surface_distance,
        curvature=args.weight_coil_curvature,
    )
    config = NativeSingleStageConfig(
        biotsavart_file=args.biotsavart_file,
        surface_path=args.surface_path,
        boozer_state_path=args.boozer_state_path,
        output_root=args.output_root,
        overwrite=args.overwrite,
        run_config_sha256=args.run_config_sha256,
        objective_profile=args.objective_profile,
        vmec_s=args.vmec_s,
        surface_scale=args.surface_scale,
        mpol=args.mpol,
        ntor=args.ntor,
        nphi=args.nphi,
        ntheta=args.ntheta,
        constraint_weight=args.constraint_weight,
        iota_target=args.iota_target,
        boozer_bfgs_tol=args.boozer_bfgs_tol,
        boozer_bfgs_maxiter=args.boozer_bfgs_maxiter,
        boozer_newton_tol=args.boozer_newton_tol,
        boozer_newton_maxiter=args.boozer_newton_maxiter,
        boozer_limited_memory=args.boozer_limited_memory,
        outer_maxiter=args.outer_maxiter,
        outer_maxcor=args.outer_maxcor,
        outer_maxls=args.outer_maxls,
        outer_ftol=args.outer_ftol,
        outer_gtol=args.outer_gtol,
        weights=weights,
    )
    run_native_single_stage(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
