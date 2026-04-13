"""Probe outer-JIT buffer donation for synthetic and real Stage 2 field paths."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from repo_bootstrap import configure_entrypoint_jax_runtime


configure_entrypoint_jax_runtime(sys.argv[1:])

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from benchmarks.validation_ladder_common import (
    bootstrap_local_simsopt,
    build_provenance,
    peak_rss_mb,
    print_provenance,
    query_gpu_memory_mb,
    write_json,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_STAGE2_BS_PATH,
    resolve_equilibrium_path,
)

bootstrap_local_simsopt()

from simsopt._core.optimizable import load
from simsopt import backend as simsopt_backend
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.geo import SurfaceRZFourier
from simsopt.jax_core.biotsavart import biot_savart_B
from simsopt.jax_core.field import grouped_biot_savart_B_from_spec
from simsopt.jax_core.objectives_flux import fixed_surface_flux_specs_from_surface


@dataclass(frozen=True)
class DonationProbeShape:
    ncoils: int
    nquad: int
    npoints: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        choices=("procedural", "real-stage2"),
        default="procedural",
        help="Synthetic kernel-shape probe or the real Stage 2 grouped-field fixture.",
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before import/use.",
    )
    parser.add_argument(
        "--mode",
        choices=simsopt_backend.VALID_BACKEND_MODES,
        default="jax_cpu_parity",
        help="Backend mode used to evaluate the probe.",
    )
    parser.add_argument("--ncoils", type=int, default=16)
    parser.add_argument("--nquad", type=int, default=128)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stage2-bs-path",
        type=str,
        default=str(DEFAULT_STAGE2_BS_PATH),
        help="Path to the fixed Stage 2 seed biot_savart_opt.json fixture.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        type=str,
        default=DEFAULT_PLASMA_SURF_FILENAME,
        help="VMEC equilibrium filename for the real Stage 2 fixture.",
    )
    parser.add_argument(
        "--equilibria-dir",
        type=str,
        default=str(DEFAULT_EQUILIBRIA_DIR),
        help="Directory containing the VMEC equilibrium file.",
    )
    parser.add_argument(
        "--equilibrium-path",
        type=str,
        default=None,
        help="Optional explicit equilibrium path overriding --equilibria-dir/filename.",
    )
    parser.add_argument(
        "--stage2-nphi",
        type=int,
        default=31,
        help="Toroidal grid size for the real Stage 2 surface probe.",
    )
    parser.add_argument(
        "--stage2-ntheta",
        type=int,
        default=16,
        help="Poloidal grid size for the real Stage 2 surface probe.",
    )
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def _make_fixture(shape: DonationProbeShape, *, seed: int):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(shape.npoints, 3))
    points[:, 0] -= 2.0
    gammas = rng.normal(size=(shape.ncoils, shape.nquad, 3))
    gammas[:, :, 0] += 1.5
    gammadashs = rng.normal(size=(shape.ncoils, shape.nquad, 3))
    currents = rng.normal(loc=1.0e5, scale=2.0e4, size=(shape.ncoils,))
    return (
        np.asarray(points, dtype=np.float64),
        jnp.asarray(gammas, dtype=jnp.float64),
        jnp.asarray(gammadashs, dtype=jnp.float64),
        jnp.asarray(currents, dtype=jnp.float64),
    )


def _snapshot(label: str, started_at: float) -> dict[str, float | str | None]:
    return {
        "label": label,
        "elapsed_s": float(time.perf_counter() - started_at),
        "rss_mb": peak_rss_mb(),
        "gpu_memory_mb": query_gpu_memory_mb(),
    }


def _peak_snapshot_value(
    snapshots: list[dict[str, float | str | None]],
    key: str,
) -> float | None:
    values = [snapshot[key] for snapshot in snapshots if snapshot[key] is not None]
    if not values:
        return None
    return max(float(value) for value in values)


def _estimate_input_bytes(
    host_points: np.ndarray,
    gammas: jax.Array,
    gammadashs: jax.Array,
    currents: jax.Array,
) -> int:
    return int(host_points.nbytes + gammas.nbytes + gammadashs.nbytes + currents.nbytes)


def _fresh_points(host_points: np.ndarray) -> jax.Array:
    return jnp.asarray(host_points, dtype=jnp.float64)


def _estimate_tree_bytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        nbytes = getattr(leaf, "nbytes", None)
        if nbytes is not None:
            total += int(nbytes)
    return total


def _baseline_kernel():
    return jax.jit(
        lambda points, gammas, gammadashs, currents: biot_savart_B(
            points,
            gammas,
            gammadashs,
            currents,
        )
    )


def _donated_points_kernel():
    return jax.jit(
        lambda points, gammas, gammadashs, currents: biot_savart_B(
            points,
            gammas,
            gammadashs,
            currents,
        ),
        donate_argnums=(0,),
    )


def _baseline_grouped_kernel():
    return jax.jit(
        lambda points, coil_spec: grouped_biot_savart_B_from_spec(points, coil_spec)
    )


def _donated_grouped_points_kernel():
    return jax.jit(
        lambda points, coil_spec: grouped_biot_savart_B_from_spec(points, coil_spec),
        donate_argnums=(0,),
    )


def _measure_probe_case(
    *,
    kernel,
    host_points: np.ndarray,
    gammas: jax.Array,
    gammadashs: jax.Array,
    currents: jax.Array,
    warmup: int,
    repeat: int,
    donate_points: bool,
) -> tuple[dict[str, object], np.ndarray]:
    snapshots: list[dict[str, float | str | None]] = []
    durations: list[float] = []
    started_at = time.perf_counter()

    result = kernel(_fresh_points(host_points), gammas, gammadashs, currents)
    jax.block_until_ready(result)
    compile_s = float(time.perf_counter() - started_at)
    snapshots.append(_snapshot("after_compile", started_at))

    for _ in range(warmup):
        warm_result = kernel(_fresh_points(host_points), gammas, gammadashs, currents)
        jax.block_until_ready(warm_result)
    snapshots.append(_snapshot("after_warmup", started_at))

    final_result = np.asarray(jax.device_get(result), dtype=np.float64)
    for _ in range(repeat):
        t0 = time.perf_counter()
        call_result = kernel(_fresh_points(host_points), gammas, gammadashs, currents)
        jax.block_until_ready(call_result)
        durations.append(time.perf_counter() - t0)
        final_result = np.asarray(jax.device_get(call_result), dtype=np.float64)
    snapshots.append(_snapshot("after_repeats", started_at))

    return (
        {
            "donate_argnums": [0] if donate_points else [],
            "compile_s": compile_s,
            "median_ms": float(np.median(durations) * 1e3),
            "mean_ms": float(np.mean(durations) * 1e3),
            "repeat_count": int(repeat),
            "warmup_count": int(warmup),
            "peak_rss_mb": _peak_snapshot_value(snapshots, "rss_mb"),
            "max_sampled_gpu_memory_mb": _peak_snapshot_value(
                snapshots, "gpu_memory_mb"
            ),
            "snapshots": snapshots,
            "public_api_safe": (not donate_points),
            "contract_note": (
                "Outer-JIT donation on points is safe only when the caller treats "
                "the points buffer as disposable after the call."
                if donate_points
                else "Baseline public-contract behavior: inputs remain reusable."
            ),
        },
        final_result,
    )


def _measure_grouped_probe_case(
    *,
    kernel,
    host_points: np.ndarray,
    coil_spec,
    warmup: int,
    repeat: int,
    donate_points: bool,
) -> tuple[dict[str, object], np.ndarray]:
    snapshots: list[dict[str, float | str | None]] = []
    durations: list[float] = []
    started_at = time.perf_counter()

    result = kernel(_fresh_points(host_points), coil_spec)
    jax.block_until_ready(result)
    compile_s = float(time.perf_counter() - started_at)
    snapshots.append(_snapshot("after_compile", started_at))

    for _ in range(warmup):
        warm_result = kernel(_fresh_points(host_points), coil_spec)
        jax.block_until_ready(warm_result)
    snapshots.append(_snapshot("after_warmup", started_at))

    final_result = np.asarray(jax.device_get(result), dtype=np.float64)
    for _ in range(repeat):
        t0 = time.perf_counter()
        call_result = kernel(_fresh_points(host_points), coil_spec)
        jax.block_until_ready(call_result)
        durations.append(time.perf_counter() - t0)
        final_result = np.asarray(jax.device_get(call_result), dtype=np.float64)
    snapshots.append(_snapshot("after_repeats", started_at))

    return (
        {
            "donate_argnums": [0] if donate_points else [],
            "compile_s": compile_s,
            "median_ms": float(np.median(durations) * 1e3),
            "mean_ms": float(np.mean(durations) * 1e3),
            "repeat_count": int(repeat),
            "warmup_count": int(warmup),
            "peak_rss_mb": _peak_snapshot_value(snapshots, "rss_mb"),
            "max_sampled_gpu_memory_mb": _peak_snapshot_value(
                snapshots, "gpu_memory_mb"
            ),
            "snapshots": snapshots,
            "public_api_safe": (not donate_points),
            "contract_note": (
                "Outer-JIT donation on points is safe only when the caller treats "
                "the points buffer as disposable after the call."
                if donate_points
                else "Baseline public-contract behavior: inputs remain reusable."
            ),
        },
        final_result,
    )


def _restore_backend_config(config) -> None:
    simsopt_backend.set_backend(
        config.mode,
        strict=config.strict,
        debug_nans=config.debug_nans,
        transfer_guard=config.transfer_guard,
        compilation_cache_dir=config.compilation_cache_dir,
        configure_runtime=False,
    )


def _build_real_stage2_grouped_fixture(
    *,
    stage2_bs_path: str | Path,
    plasma_surf_filename: str,
    equilibria_dir: str | Path,
    equilibrium_path: str | Path | None,
    nphi: int,
    ntheta: int,
) -> tuple[np.ndarray, object, dict[str, object], int]:
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    stage2_bs_path = Path(stage2_bs_path).expanduser().resolve()
    equilibrium_file = resolve_equilibrium_path(
        plasma_surf_filename=plasma_surf_filename,
        equilibria_dir=equilibria_dir,
        equilibrium_path=equilibrium_path,
    )
    _, stage2_results = single_stage_example.load_stage2_results(str(stage2_bs_path))
    major_radius = float(stage2_results["MAJOR_RADIUS"])
    toroidal_flux = float(stage2_results["TOROIDAL_FLUX"])

    bs_loaded = load(str(stage2_bs_path))
    bs = BiotSavartJAX(bs_loaded.coils)
    surface = SurfaceRZFourier.from_wout(
        str(equilibrium_file),
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
        s=toroidal_flux,
    )
    surface.set_dofs(surface.get_dofs() * major_radius / surface.major_radius())
    _, flux_spec = fixed_surface_flux_specs_from_surface(
        surface,
        definition="quadratic flux",
    )
    host_points = np.asarray(flux_spec.points, dtype=np.float64)
    coil_spec = bs.coil_set_spec()
    metadata = {
        "kind": "real-stage2",
        "stage2_bs_path": str(stage2_bs_path),
        "equilibrium_path": str(equilibrium_file),
        "nphi": int(nphi),
        "ntheta": int(ntheta),
        "point_count": int(host_points.shape[0]),
    }
    input_bytes = int(host_points.nbytes + _estimate_tree_bytes(coil_spec))
    return host_points, coil_spec, metadata, input_bytes


def build_biotsavart_donation_probe_payload(
    *,
    title: str,
    mode: str,
    shape: DonationProbeShape,
    warmup: int,
    repeat: int,
    seed: int,
    fixture: str = "procedural",
    stage2_bs_path: str | Path = DEFAULT_STAGE2_BS_PATH,
    plasma_surf_filename: str = DEFAULT_PLASMA_SURF_FILENAME,
    equilibria_dir: str | Path = DEFAULT_EQUILIBRIA_DIR,
    equilibrium_path: str | Path | None = None,
    stage2_nphi: int = 31,
    stage2_ntheta: int = 16,
) -> dict[str, object]:
    previous = simsopt_backend.get_backend_config()
    simsopt_backend.set_backend(mode, configure_runtime=False)
    try:
        tuning = simsopt_backend.get_field_kernel_tuning(mode)
        fixture_payload = {"kind": fixture}
        if fixture == "real-stage2":
            host_points, coil_spec, fixture_payload, input_bytes = (
                _build_real_stage2_grouped_fixture(
                    stage2_bs_path=stage2_bs_path,
                    plasma_surf_filename=plasma_surf_filename,
                    equilibria_dir=equilibria_dir,
                    equilibrium_path=equilibrium_path,
                    nphi=stage2_nphi,
                    ntheta=stage2_ntheta,
                )
            )
            baseline_payload, baseline_result = _measure_grouped_probe_case(
                kernel=_baseline_grouped_kernel(),
                host_points=host_points,
                coil_spec=coil_spec,
                warmup=warmup,
                repeat=repeat,
                donate_points=False,
            )
            donated_payload, donated_result = _measure_grouped_probe_case(
                kernel=_donated_grouped_points_kernel(),
                host_points=host_points,
                coil_spec=coil_spec,
                warmup=warmup,
                repeat=repeat,
                donate_points=True,
            )
        else:
            host_points, gammas, gammadashs, currents = _make_fixture(shape, seed=seed)
            input_bytes = _estimate_input_bytes(
                host_points, gammas, gammadashs, currents
            )
            baseline_payload, baseline_result = _measure_probe_case(
                kernel=_baseline_kernel(),
                host_points=host_points,
                gammas=gammas,
                gammadashs=gammadashs,
                currents=currents,
                warmup=warmup,
                repeat=repeat,
                donate_points=False,
            )
            donated_payload, donated_result = _measure_probe_case(
                kernel=_donated_points_kernel(),
                host_points=host_points,
                gammas=gammas,
                gammadashs=gammadashs,
                currents=currents,
                warmup=warmup,
                repeat=repeat,
                donate_points=True,
            )

        abs_diff = np.abs(baseline_result - donated_result)
        baseline_scale = np.maximum(np.abs(baseline_result), 1e-300)
        provenance = build_provenance(
            jax,
            jaxlib,
            title=title,
            extra={
                "lane": "biotsavart-donation-probe",
                "backend_mode": mode,
                "chunk_policy": tuning.chunk_policy,
                "coil_chunk_size": tuning.coil_chunk_size,
                "quadrature_block_size": tuning.quadrature_block_size,
                "point_chunk_size": simsopt_backend.get_point_chunk_size(),
            },
        )
        return {
            "provenance": provenance,
            "fixture": fixture_payload,
            "shape": asdict(shape),
            "input_bytes": input_bytes,
            "cases": {
                "baseline": baseline_payload,
                "donate_points": donated_payload,
            },
            "comparison": {
                "output_shape": list(baseline_result.shape),
                "max_abs_diff": float(np.max(abs_diff)),
                "max_rel_diff": float(np.max(abs_diff / baseline_scale)),
            },
        }
    finally:
        _restore_backend_config(previous)


def main() -> None:
    args = _parse_args()
    payload = build_biotsavart_donation_probe_payload(
        title="Biot-Savart outer-JIT donation probe",
        mode=args.mode,
        shape=DonationProbeShape(
            ncoils=args.ncoils,
            nquad=args.nquad,
            npoints=args.npoints,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        fixture=args.fixture,
        stage2_bs_path=args.stage2_bs_path,
        plasma_surf_filename=args.plasma_surf_filename,
        equilibria_dir=args.equilibria_dir,
        equilibrium_path=args.equilibrium_path,
        stage2_nphi=args.stage2_nphi,
        stage2_ntheta=args.stage2_ntheta,
    )
    print_provenance(payload["provenance"])
    if payload["fixture"]["kind"] == "real-stage2":
        print(
            "Real Stage 2 fixture: "
            f"nphi={payload['fixture']['nphi']} "
            f"ntheta={payload['fixture']['ntheta']} "
            f"points={payload['fixture']['point_count']} "
            f"input={payload['input_bytes']}B"
        )
    else:
        print(
            "Probe shape: "
            f"coils={payload['shape']['ncoils']} "
            f"nquad={payload['shape']['nquad']} "
            f"points={payload['shape']['npoints']} "
            f"input={payload['input_bytes']}B"
        )
    for case_name in ("baseline", "donate_points"):
        case_payload = payload["cases"][case_name]
        print(
            f"{case_name}: compile={case_payload['compile_s']:.3f}s "
            f"median={case_payload['median_ms']:.3f}ms "
            f"mean={case_payload['mean_ms']:.3f}ms "
            f"peak_rss={case_payload['peak_rss_mb']:.2f}MB "
            f"max_sampled_gpu={case_payload['max_sampled_gpu_memory_mb']}"
        )
    print(
        "Comparison: "
        f"max_abs_diff={payload['comparison']['max_abs_diff']:.3e} "
        f"max_rel_diff={payload['comparison']['max_rel_diff']:.3e}"
    )
    write_json(args.output_json, payload)


if __name__ == "__main__":
    main()
