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
    bootstrap_local_simsopt,
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
bootstrap_local_simsopt()

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
        "--exact-fixture",
        choices=("simple-ls-warmstart", "real-single-stage"),
        default="simple-ls-warmstart",
        help=(
            "Fixture used for exact-mode lowering. The simple fixture follows "
            "the integration-test-proven LS warm-start exact Newton path."
        ),
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


def _build_optimizer_value_and_grad(boozer_surface, bs, iota_target: float):
    return single_stage_example.build_traceable_single_stage_value_and_grad(
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


def _build_real_single_stage_fixture(args: argparse.Namespace, boozer_kind: str):
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
    return {
        "bs": fixture["bs"],
        "boozer_surface": fixture["boozer_surface"],
        "iota_target": float(args.iota_target),
        "metadata": {
            "fixture_kind": "real-single-stage",
            "equilibrium_path": fixture["equilibrium_path"],
            "stage2_bs_path": fixture["stage2_bs_path"],
            "surface_shape": fixture["surface_shape"],
            "boozer_optimizer_backend": fixture["boozer_optimizer_backend"],
            "boozer_least_squares_algorithm": fixture["boozer_least_squares_algorithm"],
            "boozer_limited_memory": fixture["boozer_limited_memory"],
        },
    }


def _build_simple_exact_ls_warmstart_fixture():
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import (
        SurfaceRZFourier,
        SurfaceXYZTensorFourier,
        Volume,
        create_equally_spaced_curves,
    )
    from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX

    ncoils = 2
    nfp = 2
    mpol = 2
    ntor = 2
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
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=stellsym)
    bs = BiotSavartJAX(coils)

    quadpoints_phi = np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False)
    quadpoints_theta = np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False)
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    seed_surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=0,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    seed_surface.set_rc(0, 0, 1.0)
    seed_surface.set_rc(1, 0, 0.15)
    seed_surface.set_zs(1, 0, 0.15)
    surface.least_squares_fit(seed_surface.gamma())

    label = Volume(surface)
    target_label = label.J()
    mu0 = 4.0 * np.pi * 1e-7
    G0 = mu0 * sum(abs(coil.current.get_value()) for coil in coils)
    iota0 = 0.3
    ls_boozer = BoozerSurfaceJAX(
        bs,
        surface,
        label,
        target_label,
        constraint_weight=1.0,
        options={
            "verbose": False,
            "optimizer_backend": "ondevice",
            "bfgs_maxiter": 300,
            "bfgs_tol": 1e-10,
            "newton_maxiter": 20,
            "newton_tol": 1e-11,
        },
    )
    ls_result = ls_boozer.run_code(iota0, G0)
    if not bool(ls_result.get("success", False)):
        raise RuntimeError("Simple exact fixture LS warm-start did not converge.")

    exact_boozer = BoozerSurfaceJAX(
        bs,
        surface,
        label,
        target_label,
        constraint_weight=None,
        options={
            "verbose": False,
            "newton_maxiter": 40,
            "newton_tol": 1e-8,
        },
    )
    exact_result = exact_boozer.run_code(ls_result["iota"], ls_result["G"])
    if not bool(exact_result.get("success", False)):
        raise RuntimeError("Simple exact fixture exact Newton did not converge.")

    exact_boozer.constraint_weight = 0.0
    return {
        "bs": bs,
        "boozer_surface": exact_boozer,
        "iota_target": float(exact_result["iota"]),
        "metadata": {
            "fixture_kind": "simple-ls-warmstart",
            "surface_shape": {
                "nphi": int(quadpoints_phi.size),
                "ntheta": int(quadpoints_theta.size),
                "mpol": int(mpol),
                "ntor": int(ntor),
            },
            "ncoils": int(ncoils),
            "nfp": int(nfp),
            "ls_success": bool(ls_result["success"]),
            "exact_success": bool(exact_result["success"]),
            "iota_target": float(exact_result["iota"]),
            "G": float(exact_result["G"]),
            "traceable_objective_constraint_weight": 0.0,
        },
    }


def _build_measurement_fixture(args: argparse.Namespace, boozer_kind: str):
    if boozer_kind == "exact" and args.exact_fixture == "simple-ls-warmstart":
        return _build_simple_exact_ls_warmstart_fixture()
    return _build_real_single_stage_fixture(args, boozer_kind)


def _measure_kind(args: argparse.Namespace, boozer_kind: str) -> dict[str, object]:
    fixture = _build_measurement_fixture(args, boozer_kind)
    bs = fixture["bs"]
    boozer_surface = fixture["boozer_surface"]
    iota_target = fixture["iota_target"]
    coil_dofs = _profile_coil_dofs(bs)

    optimizer_value_and_grad = _build_optimizer_value_and_grad(
        boozer_surface,
        bs,
        iota_target,
    )
    measurements = [
        summarize_lowered_callable(
            f"{boozer_kind}.optimizer_value_and_grad",
            optimizer_value_and_grad,
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
                    iota_target,
                ),
                coil_dofs,
            )
        )

    return {
        "boozer_kind": boozer_kind,
        "fixture": fixture["metadata"],
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
