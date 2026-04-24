"""Measure traceable single-stage target-lane compiler shape.

This probe intentionally records structural lowering metrics only. CUDA runtime
signoff still belongs in the GPU benchmark lane.

Usage:
    PYTHONPATH=src python benchmarks/traceable_target_lane_compile_shape.py \
        --platform cpu --boozer-kind ls --output-json /tmp/compile_shape.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.validation_ladder_common import (
    apply_benchmark_compilation_cache_policy,
    apply_requested_platform,
    build_provenance,
    print_provenance,
    preparse_platform,
    require_x64_runtime,
    write_json,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)
apply_benchmark_compilation_cache_policy(
    "traceable_target_lane_compile_shape",
    requested_platform=REQUESTED_PLATFORM,
)

import jax
import jax.numpy as jnp
import jaxlib

jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context="Traceable target-lane compile-shape probe")

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
from benchmarks.traceable_compile_shape import summarize_lowered_callable
from examples.single_stage_optimization.SINGLE_STAGE import (
    single_stage_banana_example as single_stage_example,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lower seeded traceable single-stage value-and-grad callables and "
            "write control-flow shape counts."
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
        help="Path to write the structured compile-shape payload.",
    )
    parser.add_argument(
        "--boozer-kind",
        choices=("ls", "exact", "both"),
        default="ls",
        help="Boozer solve kind to lower.",
    )
    parser.add_argument(
        "--include-public",
        action="store_true",
        help="Also lower the public baseline-aware runtime-bundle value_and_grad.",
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
    parser.add_argument("--nphi", type=int, default=DEFAULT_SMOKE_NPHI)
    parser.add_argument("--ntheta", type=int, default=DEFAULT_SMOKE_NTHETA)
    parser.add_argument("--mpol", type=int, default=DEFAULT_SMOKE_MPOL)
    parser.add_argument("--ntor", type=int, default=DEFAULT_SMOKE_NTOR)
    parser.add_argument("--vol-target", type=float, default=DEFAULT_VOL_TARGET)
    parser.add_argument("--iota-target", type=float, default=DEFAULT_IOTA_TARGET)
    return parser.parse_args()


def _constraint_weight_for_kind(boozer_kind: str) -> float | None:
    return 1.0 if boozer_kind == "ls" else None


def _profile_coil_dofs(bs) -> jax.Array:
    return jnp.asarray(
        single_stage_example.build_target_lane_profile_coil_dofs(bs.x.copy()),
        dtype=jnp.float64,
    )


def _build_seeded_value_and_grad(boozer_surface, bs, iota_target: float):
    return single_stage_example.build_traceable_single_stage_seeded_value_and_grad(
        boozer_surface,
        bs,
        iota_target,
        outer_objective_config=None,
        success_filter=None,
    )


def _build_public_value_and_grad(boozer_surface, bs, iota_target: float):
    runtime_bundle = (
        single_stage_example.get_traceable_single_stage_runtime_bundle_builder()(
            boozer_surface,
            bs,
            iota_target,
            include_profile_suite=False,
            include_host_wrappers=False,
            outer_objective_config=None,
            success_filter=None,
        )
    )
    return runtime_bundle["value_and_grad"]


def _measure_kind(args: argparse.Namespace, boozer_kind: str) -> dict[str, object]:
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
        optimizer_backend=DEFAULT_OPTIMIZER_BACKEND,
        constraint_weight=_constraint_weight_for_kind(boozer_kind),
    )
    bs = fixture["bs"]
    boozer_surface = fixture["boozer_surface"]
    coil_dofs = _profile_coil_dofs(bs)

    seeded = _build_seeded_value_and_grad(boozer_surface, bs, args.iota_target)
    measurements = [
        summarize_lowered_callable(
            f"{boozer_kind}.seeded_value_and_grad",
            seeded.value_and_grad,
            coil_dofs,
        )
    ]
    if args.include_public:
        measurements.append(
            summarize_lowered_callable(
                f"{boozer_kind}.public_value_and_grad",
                _build_public_value_and_grad(
                    boozer_surface,
                    bs,
                    args.iota_target,
                ),
                coil_dofs,
            )
        )

    return {
        "boozer_kind": boozer_kind,
        "fixture": {
            "equilibrium_path": fixture["equilibrium_path"],
            "stage2_bs_path": fixture["stage2_bs_path"],
            "surface_shape": fixture["surface_shape"],
            "boozer_optimizer_backend": fixture["boozer_optimizer_backend"],
            "boozer_least_squares_algorithm": fixture[
                "boozer_least_squares_algorithm"
            ],
            "boozer_limited_memory": fixture["boozer_limited_memory"],
        },
        "coil_dof_count": int(np.asarray(bs.x).size),
        "measurements": measurements,
    }


def main() -> None:
    args = parse_args()
    kinds = ("ls", "exact") if args.boozer_kind == "both" else (args.boozer_kind,)
    payload = {
        "provenance": build_provenance(
            jax,
            jaxlib,
            title="Traceable target-lane compile-shape probe",
            extra={
                "platform_request": args.platform,
                "compile_shape_only": True,
            },
        ),
        "cases": [_measure_kind(args, boozer_kind) for boozer_kind in kinds],
    }
    print_provenance(payload["provenance"])
    write_json(args.output_json, payload)


if __name__ == "__main__":
    main()
