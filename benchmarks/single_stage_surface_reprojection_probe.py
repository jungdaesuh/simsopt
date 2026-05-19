"""Stagewise repro probe for the real single-stage SurfaceRZFourier path."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable, TypeVar

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
    DEFAULT_STAGE2_BS_PATH,
    resolve_equilibrium_path,
)
from benchmarks.validation_ladder_common import (
    apply_benchmark_compilation_cache_policy,
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    describe_compile_behavior,
    maybe_initialize_distributed_runtime,
    preparse_platform,
    print_provenance,
    require_x64_runtime,
    write_json,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)
apply_benchmark_compilation_cache_policy(
    "single_stage_surface_reprojection_probe",
    requested_platform=REQUESTED_PLATFORM,
)
bootstrap_local_simsopt()

import jax
import jaxlib

maybe_initialize_distributed_runtime()
jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context="Single-stage SurfaceRZFourier reprojection probe")

from examples.single_stage_optimization.SINGLE_STAGE import (
    single_stage_banana_example as single_stage_example,
)
from simsopt.geo import SurfaceRZFourier
from simsopt.jax_core._math_utils import as_jax_float64
from simsopt.jax_core.surface_rzfourier import (
    surface_rz_fourier_gamma_from_dofs,
    surface_rz_fourier_gamma_from_spec,
    surface_rz_fourier_spec_from_dofs,
)


T = TypeVar("T")


@dataclass(frozen=True)
class StageRecord:
    name: str
    status: str
    elapsed_s: float
    details: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolate the real single-stage SurfaceRZFourier reprojection path "
            "with staged diagnostics."
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
        help="Path to write the structured probe result.",
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
        help="Surface toroidal grid points for the source SurfaceRZFourier.",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=DEFAULT_SMOKE_NTHETA,
        help="Surface poloidal grid points for the source SurfaceRZFourier.",
    )
    parser.add_argument(
        "--target-mpol",
        type=int,
        default=DEFAULT_SMOKE_MPOL,
        help="Target SurfaceXYZTensorFourier poloidal mode count.",
    )
    parser.add_argument(
        "--target-ntor",
        type=int,
        default=DEFAULT_SMOKE_NTOR,
        help="Target SurfaceXYZTensorFourier toroidal mode count.",
    )
    parser.add_argument(
        "--stop-after-stage",
        choices=(
            "load_source_surface",
            "device_put_source_dofs",
            "surface_rz_fourier_spec_from_dofs",
            "surface_rz_fourier_gamma_from_spec",
            "surface_rz_fourier_gamma_from_dofs",
            "project_surface_dofs_to_resolution",
        ),
        default=None,
        help="Stop after the named stage and mark the probe successful.",
    )
    return parser.parse_args()


def _block_tree(value: object) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            jax.block_until_ready(leaf)


def _surface_details(surface: SurfaceRZFourier) -> dict[str, Any]:
    return {
        "surface_class": type(surface).__name__,
        "mpol": int(surface.mpol),
        "ntor": int(surface.ntor),
        "nfp": int(surface.nfp),
        "stellsym": bool(surface.stellsym),
        "nphi": int(len(surface.quadpoints_phi)),
        "ntheta": int(len(surface.quadpoints_theta)),
        "dofs_size": int(np.asarray(surface.get_dofs(), dtype=np.float64).size),
    }


def _array_details(array: object) -> dict[str, Any]:
    if isinstance(array, jax.Array):
        return {
            "shape": tuple(int(dim) for dim in array.shape),
            "dtype": str(array.dtype),
        }
    array_np = np.asarray(array)
    return {
        "shape": tuple(int(dim) for dim in array_np.shape),
        "dtype": str(array_np.dtype),
        "size": int(array_np.size),
    }


def _spec_details(spec: Any) -> dict[str, Any]:
    return {
        "mpol": int(spec.mpol),
        "ntor": int(spec.ntor),
        "nfp": int(spec.nfp),
        "stellsym": bool(spec.stellsym),
        "rc_shape": tuple(int(dim) for dim in spec.rc.shape),
        "zs_shape": tuple(int(dim) for dim in spec.zs.shape),
    }


def _source_bundle_details(
    item: tuple[SurfaceRZFourier, dict[str, Any]],
) -> dict[str, Any]:
    surface, context = item
    return {**context, **_surface_details(surface)}


def _run_stage(
    *,
    name: str,
    stages: list[StageRecord],
    callback: Callable[[], T],
    summarize: Callable[[T], dict[str, Any]] | None = None,
) -> T:
    started_at = time.perf_counter()
    try:
        value = callback()
        _block_tree(value)
    except Exception as exc:
        stages.append(
            StageRecord(
                name=name,
                status="failed",
                elapsed_s=float(time.perf_counter() - started_at),
                details={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )
        raise
    details = {} if summarize is None else summarize(value)
    stages.append(
        StageRecord(
            name=name,
            status="passed",
            elapsed_s=float(time.perf_counter() - started_at),
            details=details,
        )
    )
    return value


def _should_stop_after(args: argparse.Namespace, stage_name: str) -> bool:
    return str(args.stop_after_stage) == stage_name


def _build_source_surface(
    args: argparse.Namespace,
) -> tuple[SurfaceRZFourier, dict[str, Any]]:
    stage2_results_path, stage2_results = single_stage_example.load_stage2_results(
        args.stage2_bs_path
    )
    major_radius = float(stage2_results["MAJOR_RADIUS"])
    toroidal_flux = float(stage2_results["TOROIDAL_FLUX"])
    equilibrium_file = resolve_equilibrium_path(
        plasma_surf_filename=args.plasma_surf_filename,
        equilibria_dir=args.equilibria_dir,
        equilibrium_path=args.equilibrium_path,
    )
    surface = SurfaceRZFourier.from_wout(
        str(equilibrium_file),
        range="half period",
        nphi=int(args.nphi),
        ntheta=int(args.ntheta),
        s=toroidal_flux,
    )
    surface.set_dofs(surface.get_dofs() * major_radius / surface.major_radius())
    return surface, {
        "equilibrium_path": str(equilibrium_file),
        "stage2_results_path": str(stage2_results_path),
        "major_radius": major_radius,
        "toroidal_flux": toroidal_flux,
    }


def _unhandled_failure_record(exc: Exception) -> StageRecord:
    return StageRecord(
        name="probe_script",
        status="failed",
        elapsed_s=0.0,
        details={
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def main() -> int:
    args = parse_args()
    stages: list[StageRecord] = []
    source_context: dict[str, Any] = {}
    failure: StageRecord | None = None
    try:
        source_surface, source_context = _run_stage(
            name="load_source_surface",
            stages=stages,
            callback=lambda: _build_source_surface(args),
            summarize=_source_bundle_details,
        )
        if _should_stop_after(args, "load_source_surface"):
            _finish(
                args=args,
                stages=stages,
                source_context=source_context,
                failure=None,
            )
            return 0

        source_dofs_host = np.asarray(source_surface.get_dofs(), dtype=np.float64)

        source_dofs_device = _run_stage(
            name="device_put_source_dofs",
            stages=stages,
            callback=lambda: as_jax_float64(source_dofs_host),
            summarize=_array_details,
        )
        if _should_stop_after(args, "device_put_source_dofs"):
            _finish(
                args=args,
                stages=stages,
                source_context=source_context,
                failure=None,
            )
            return 0

        source_spec = _run_stage(
            name="surface_rz_fourier_spec_from_dofs",
            stages=stages,
            callback=lambda: surface_rz_fourier_spec_from_dofs(
                source_dofs_device,
                quadpoints_phi=as_jax_float64(source_surface.quadpoints_phi),
                quadpoints_theta=as_jax_float64(source_surface.quadpoints_theta),
                mpol=int(source_surface.mpol),
                ntor=int(source_surface.ntor),
                nfp=int(source_surface.nfp),
                stellsym=bool(source_surface.stellsym),
            ),
            summarize=_spec_details,
        )
        if _should_stop_after(args, "surface_rz_fourier_spec_from_dofs"):
            _finish(
                args=args,
                stages=stages,
                source_context=source_context,
                failure=None,
            )
            return 0

        _run_stage(
            name="surface_rz_fourier_gamma_from_spec",
            stages=stages,
            callback=lambda: surface_rz_fourier_gamma_from_spec(source_spec),
            summarize=_array_details,
        )
        if _should_stop_after(args, "surface_rz_fourier_gamma_from_spec"):
            _finish(
                args=args,
                stages=stages,
                source_context=source_context,
                failure=None,
            )
            return 0

        _run_stage(
            name="surface_rz_fourier_gamma_from_dofs",
            stages=stages,
            callback=lambda: surface_rz_fourier_gamma_from_dofs(
                source_spec, source_dofs_device
            ),
            summarize=_array_details,
        )
        if _should_stop_after(args, "surface_rz_fourier_gamma_from_dofs"):
            _finish(
                args=args,
                stages=stages,
                source_context=source_context,
                failure=None,
            )
            return 0

        _run_stage(
            name="project_surface_dofs_to_resolution",
            stages=stages,
            callback=lambda: single_stage_example.project_surface_dofs_to_resolution(
                source_surface,
                mpol=int(args.target_mpol),
                ntor=int(args.target_ntor),
                quadpoints_phi=source_surface.quadpoints_phi,
                quadpoints_theta=source_surface.quadpoints_theta,
            ),
            summarize=_array_details,
        )
    except Exception as exc:
        if stages and stages[-1].status == "failed":
            failure = stages[-1]
        else:
            failure = _unhandled_failure_record(exc)
            stages.append(failure)
        _finish(
            args=args,
            stages=stages,
            source_context=source_context,
            failure=failure,
        )
        return 1
    _finish(args=args, stages=stages, source_context=source_context, failure=None)
    return 0


def _finish(
    *,
    args: argparse.Namespace,
    stages: list[StageRecord],
    source_context: dict[str, Any],
    failure: StageRecord | None,
) -> None:
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Single-stage SurfaceRZFourier reprojection probe",
        extra={
            "fixture": "real-single-stage-surface-rzfourier-reprojection",
            "platform_request": args.platform,
            "plasma_surf_filename": args.plasma_surf_filename,
            "stage2_seed_path": str(Path(args.stage2_bs_path)),
            "target_mpol": int(args.target_mpol),
            "target_ntor": int(args.target_ntor),
            "compile_behavior": describe_compile_behavior(uses_subprocesses=False),
            **source_context,
        },
    )
    print_provenance(provenance)
    payload = {
        "provenance": provenance,
        "source_surface": {
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "target_mpol": int(args.target_mpol),
            "target_ntor": int(args.target_ntor),
        },
        "stages": [asdict(stage) for stage in stages],
        "failure_stage": None if failure is None else failure.name,
        "passed": failure is None,
    }
    write_json(args.output_json, payload)
    if failure is not None:
        print(f"REPROJECTION PROBE FAILED AT {failure.name}")
        raise SystemExit(1)
    print("REPROJECTION PROBE PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
