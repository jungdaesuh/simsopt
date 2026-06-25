"""Pole-2 compile-breadth probe for the single-stage GPU value/grad kernels.

This is the Phase-1 measurement harness for
``docs/pole2_compile_breadth_kernelization_implementation_plan_2026-06-23.md``.
The single-stage GPU value+gradient program has a "pole 2" compile wall: at
mpol10 the per-evaluation program takes tens-of-minutes-to-an-hour to *construct +
XLA-compile* with the GPU idle, before a single optimizer step runs. This probe
localizes WHICH compiled unit dominates and how the cost scales mpol6 -> mpol8 ->
mpol10, by compiling the two kernels SEPARATELY and recording, for each:

  * ``construct_wall_s`` -- the ``jax.jit(f).lower(*args)`` wall (trace + lower).
  * ``compile_wall_s``   -- the ``.compile()`` wall on the lowered object
    (= XLA compile), measured separately from the lower and from first execution.
  * ``hlo_op_count``     -- instruction count in ``compiled.as_text()``.
  * ``cost_analysis``    -- ``compiled.cost_analysis()`` (flops / bytes accessed),
    when XLA exposes it for the active backend.

This is a compile-only probe: no kernel is executed, so it isolates the
construct+compile cost that pole 2 is about.

The two kernels (the K1/K2 split the plan localizes) come from the production
SSOT traceable-runtime builders in
``src/simsopt_jax_adapters/geo/surface_objectives_traceable.py``:

  * K1 = ``_forward_result_for`` -- the device-traceable forward Boozer solve.
    Built inside ``_build_traceable_objective_compiled_bundle_from_state`` and
    exposed as the bundle's ``compiled_forward_result_for`` host boundary; we lower
    it under a thin ``jax.jit`` wrapper (the boundary itself is not a jit object).
    Argument: ``(coil_dofs,)``.
  * K2 = ``_solved_state_value_and_grad_for`` -- objective value + IFT/operator-GMRES
    adjoint gradient from an ALREADY-solved Boozer state (no forward solve). Built by
    ``_build_traceable_solved_state_value_and_grad_from_state``, which returns the
    ``jax.jit``-wrapped kernel directly. Arguments:
    ``(coil_dofs, solved_x, solved_linear_solve_factors)``.

Both kernels are built from one shared traceable-runtime ``state`` (the same
``_build_traceable_objective_state`` the production target lane uses), seeded with
the production ``outer_objective_config`` (all 8 single-stage terms) so the measured
graph breadth matches the production value/grad graph rather than a Boozer-only
subset.

Cold vs warm cache and HLO dumps are controlled ENTIRELY by the environment, the
same way the other benchmarks work -- this script does not override them:

  * Set ``JAX_COMPILATION_CACHE_DIR`` to a FRESH directory per cold-cache run (and
    reuse the same directory only when intentionally measuring persistent-cache
    reuse). On a cuda run with no explicit value, the benchmark cache policy picks a
    stable per-device dir, so a cold-cache measurement REQUIRES exporting a fresh
    ``JAX_COMPILATION_CACHE_DIR``. The resolved policy is recorded in the output.
  * Export ``XLA_FLAGS=--xla_dump_to=<dir>`` (BEFORE this process starts, i.e. before
    JAX backend init) to dump the per-kernel HLO for offline sub-computation
    attribution (BiotSavart eval vs residual/Jacobian vs adjoint).
  * Keep ``XLA_FLAGS`` fixed across the mpol sweep so the per-mpol compiles are
    comparable and share cache keys.

Pod command line (A40, paths from RUNBOOK.md sec 1.1 / 4.3) -- see this module's
``main`` and the closing comment for the exact invocation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import resource
import sys
import time

import numpy as np

from benchmarks.validation_ladder_common import (
    apply_benchmark_compilation_cache_policy,
    apply_requested_platform,
    bootstrap_local_simsopt,
    current_compilation_cache_metadata,
    isolate_parent_cuda_memory_allocator,
    preparse_platform,
    require_requested_platform_runtime,
    require_x64_runtime,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_IOTA_TARGET,
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_STAGE2_BS_PATH,
    DEFAULT_VOL_TARGET,
    build_real_single_stage_init_fixture,
)
from benchmarks.single_stage_objective_eval import (
    objective_weights_from_example_defaults,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
CHILD_CUDA_MEMORY_ENV = isolate_parent_cuda_memory_allocator(REQUESTED_PLATFORM)
apply_requested_platform(REQUESTED_PLATFORM)
_CACHE_POLICY_METADATA = apply_benchmark_compilation_cache_policy(
    "compile_breadth_probe",
    requested_platform=REQUESTED_PLATFORM,
)
bootstrap_local_simsopt()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

_RUNTIME_CONTEXT = "Pole-2 compile-breadth probe"
require_x64_runtime(jax, context=_RUNTIME_CONTEXT)
require_requested_platform_runtime(
    jax,
    requested_platform=REQUESTED_PLATFORM,
    context=_RUNTIME_CONTEXT,
)

from examples.single_stage_optimization.SINGLE_STAGE import (  # noqa: E402
    single_stage_banana_example as single_stage_example,
)
from simsopt_jax_adapters.geo import surface_objectives_traceable as traceable  # noqa: E402

# The seed family the RUNBOOK sec 4.3 april mpol10 recipe pins.  Each entry maps a
# resolution to the matching grid; the per-mpol grid is taken from this table unless
# the caller overrides --nphi / --ntheta / --ntor (which then apply to every mpol).
_DEFAULT_RESOLUTION_GRID = {
    6: {"ntor": 6, "nphi": 79, "ntheta": 24},
    8: {"ntor": 8, "nphi": 103, "ntheta": 28},
    10: {"ntor": 10, "nphi": 127, "ntheta": 32},
}
_DEFAULT_MPOL_LIST = (6, 8, 10)


@dataclass(frozen=True)
class _Resolution:
    mpol: int
    ntor: int
    nphi: int
    ntheta: int


@dataclass(frozen=True)
class _CompiledKernel:
    label: str
    fn: object
    args: tuple[object, ...]


def _maxrss_bytes() -> int:
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(maxrss)
    return int(maxrss) * 1024


def _device_summary() -> list[dict[str, int | str]]:
    return [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in jax.devices()
    ]


def _resolutions(args: argparse.Namespace) -> list[_Resolution]:
    resolutions: list[_Resolution] = []
    for mpol in args.mpol_list:
        grid = _DEFAULT_RESOLUTION_GRID.get(mpol)
        if grid is None and (
            args.ntor is None or args.nphi is None or args.ntheta is None
        ):
            raise ValueError(
                f"mpol={mpol} has no default grid; pass --ntor/--nphi/--ntheta "
                "to set the grid for every requested mpol."
            )
        base = grid if grid is not None else {}
        resolutions.append(
            _Resolution(
                mpol=int(mpol),
                ntor=int(args.ntor if args.ntor is not None else base["ntor"]),
                nphi=int(args.nphi if args.nphi is not None else base["nphi"]),
                ntheta=int(
                    args.ntheta if args.ntheta is not None else base["ntheta"]
                ),
            )
        )
    return resolutions


def _seed_resolved_surface(args: argparse.Namespace):
    """Load the warm-start donor's resolved surface state, or None when absent.

    Returns the ``SerializedSurfaceState`` (resolved DOFs + native mpol/ntor) and
    the donor's ``FINAL_IOTA`` so the matching resolution can install the production
    Boozer state value-only (no fresh solve, no self-intersection re-solve risk).
    """
    if args.warm_start_run_dir is None:
        return None
    warm_start = single_stage_example.load_single_stage_warm_start_state(
        args.warm_start_run_dir
    )
    return {
        "surface": warm_start["surface"],
        "iota": warm_start["iota"],
        "G": warm_start["G"],
    }


def _build_solved_boozer_fixture(
    args: argparse.Namespace,
    resolution: _Resolution,
    seed_resolved,
) -> tuple[dict[str, object], str]:
    """Build a solved JAX Boozer fixture at one resolution.

    When the requested resolution matches the warm-start donor's native resolution,
    install the donor's RESOLVED surface DOFs value-only (``reuse_resolved_warm_start_solve``)
    -- this is the production state and avoids a fresh solve that collapses for
    production seeds.  At other resolutions (used only for the compile-SHAPE scaling
    study) the donor surface does not fit, so the fixture does a fresh JAX Boozer
    solve; the compiled graph shape is resolution-driven, so this still yields a
    faithful K1/K2 graph at that resolution.
    """
    use_resolved = (
        seed_resolved is not None
        and int(seed_resolved["surface"].mpol) == resolution.mpol
        and int(seed_resolved["surface"].ntor) == resolution.ntor
    )
    if use_resolved:
        surface_dofs_override = np.asarray(
            seed_resolved["surface"].dofs, dtype=float
        )
        iota_override = float(seed_resolved["iota"])
        G_override = (
            None if seed_resolved["G"] is None else float(seed_resolved["G"])
        )
        seed_state = "resolved-warm-start"
    else:
        surface_dofs_override = None
        iota_override = None
        G_override = None
        seed_state = "fresh-solve"

    fixture = build_real_single_stage_init_fixture(
        backend="jax",
        optimizer_backend="ondevice",
        plasma_surf_filename=args.plasma_surf_filename,
        equilibria_dir=args.equilibria_dir,
        stage2_bs_path=args.stage2_bs_path,
        nphi=resolution.nphi,
        ntheta=resolution.ntheta,
        mpol=resolution.mpol,
        ntor=resolution.ntor,
        vol_target=args.vol_target,
        iota_target=args.iota_target,
        num_tf_coils=args.num_tf_coils,
        boozer_surface_dofs_override=surface_dofs_override,
        boozer_iota_override=iota_override,
        boozer_G_override=G_override,
        reuse_resolved_warm_start_solve=use_resolved,
    )
    if not fixture["boozer_surface"].res.get("success", False):
        raise RuntimeError(
            f"mpol={resolution.mpol}: JAX Boozer fixture did not reach a solved "
            "state; cannot build the compile-breadth kernels."
        )
    # A resolved warm-start state is installed value-only (no traceable Boozer
    # linearization), but K2's adjoint requires a hessian/exact_jacobian
    # linearization.  The production target lane materializes it right before
    # building the value/grad kernel (single_stage_banana_example.py:16284); mirror
    # that so the measured K1/K2 graph matches production.  Self-guards: a no-op when
    # the state already carries a traceable linearization (the fresh-solve path).
    single_stage_example._materialize_target_lane_boozer_linearization(
        fixture["boozer_surface"]
    )
    return fixture, seed_state


def _outer_objective_config(fixture: dict[str, object], *, num_tf_coils: int):
    """Build the production 8-term outer-objective config for the fixture.

    Mirrors ``main()``'s ``build_target_lane_outer_objective_config`` call so the
    measured K1/K2 graph carries the full single-stage breadth (nonqs / residual /
    iota / length / curve-curve / curve-surface / surface-vessel / curvature), not a
    Boozer-only subset.  Weights come from the example's argparse defaults (the same
    SSOT the parity matrix uses).  ``num_tf_coils`` slices the banana coils off the
    fixed TF set and MUST match the value the fixture was built with.
    """
    weights = objective_weights_from_example_defaults()
    bs = fixture["bs"]
    boozer_surface = fixture["boozer_surface"]
    banana_curve = [coil.curve for coil in bs.coils[num_tf_coils:]][0]
    vessel_surface = single_stage_example.build_single_stage_vessel_surface()
    return single_stage_example.build_target_lane_outer_objective_config(
        boozer_surface,
        bs,
        banana_curve,
        vessel_surface,
        non_qs_weight=weights.non_qs,
        residual_weight=weights.res,
        iota_weight=weights.iotas,
        length_weight=weights.length,
        length_target=weights.length_target,
        curve_curve_threshold=weights.cc_dist,
        curve_curve_weight=weights.cc,
        curve_surface_threshold=weights.cs_dist,
        curve_surface_weight=weights.cs,
        surface_vessel_threshold=weights.ss_dist,
        surface_vessel_weight=weights.surf_dist,
        curvature_threshold=weights.curvature_threshold,
        curvature_weight=weights.curvature,
    )


def _deviceify(value):
    """Place a hostified runtime constant tree back on the active device.

    ``state`` arrays are hostified (numpy) by the traceable-runtime builders, and the
    K1 boundary rejects host inputs; the production runtime re-places them with
    ``_traceable_runtime_deviceify_tree`` before calling the kernels, so the probe
    does the same for faithful tracing.
    """
    return traceable._traceable_runtime_deviceify_tree(value)


def _build_kernels(
    fixture: dict[str, object],
    *,
    num_tf_coils: int,
) -> list[_CompiledKernel]:
    """Build the K1 (forward solve) and K2 (value+adjoint) kernels for one fixture."""
    booz_jax = fixture["boozer_surface"]
    bs_jax = fixture["bs"]
    iota_target = fixture["iota_target"]
    outer_objective_config = _outer_objective_config(
        fixture, num_tf_coils=num_tf_coils
    )

    state = traceable._build_traceable_objective_state(
        booz_jax,
        bs_jax,
        iota_target,
        outer_objective_config=outer_objective_config,
    )

    # K1: forward Boozer solve.  The bundle exposes the forward kernel as the
    # ``compiled_forward_result_for`` host boundary (which wraps an internal
    # ``jax.jit(_forward_result_for)``); we lower it under a thin jit wrapper so it
    # exposes ``.lower()/.compile()``.  The inner jit is inlined during tracing, so
    # the measured construct+compile reflects the forward graph; the wrapper adds
    # only a trivial passthrough op (verify against the HLO dump on the pod).
    bundle = traceable._build_traceable_objective_compiled_bundle_from_state(
        booz_jax,
        state,
    )
    forward_boundary = bundle["compiled_forward_result_for"]

    def _k1_forward_result_for(coil_dofs):
        return forward_boundary(coil_dofs)

    # K2: value + adjoint gradient from an already-solved state.  The builder returns
    # the ``jax.jit``-wrapped kernel directly, so it is lowerable as-is.
    k2_value_and_grad = traceable._build_traceable_solved_state_value_and_grad_from_state(
        booz_jax,
        state,
    )

    # K0 (fused): the scipy-jax FULLGRAPH kernel -- forward solve + value + adjoint
    # joined by the success ``lax.cond`` in ONE jit (``_value_and_grad_for``, bundle
    # key ``compiled_value_and_grad_for``).  This is exactly the kernel the decomposed
    # lane splits into K1/K2; measuring it isolates the fusion's compile cost vs the
    # K1+K2 sum (the fused-vs-decomposed pole-2 reconciliation).  Wrapped in a thin
    # ``jax.jit`` so the host boundary exposes ``.lower()/.compile()`` (inner jit
    # inlined under trace), same as K1.
    fused_value_and_grad = bundle["compiled_value_and_grad_for"]

    def _k0_fused_value_and_grad_for(coil_dofs):
        return fused_value_and_grad(coil_dofs)

    coil_dofs = _deviceify(state["baseline_coil_dofs"])
    solved_x = _deviceify(state["baseline_x"])
    solved_linear_solve_factors = _deviceify(state["baseline_linear_solve_factors"])

    return [
        _CompiledKernel(
            label="K1_forward_result_for",
            fn=jax.jit(_k1_forward_result_for),
            args=(coil_dofs,),
        ),
        _CompiledKernel(
            label="K2_solved_state_value_and_grad_for",
            fn=k2_value_and_grad,
            args=(coil_dofs, solved_x, solved_linear_solve_factors),
        ),
        _CompiledKernel(
            label="K0_fused_value_and_grad_for",
            fn=jax.jit(_k0_fused_value_and_grad_for),
            args=(coil_dofs,),
        ),
    ]


def _hlo_op_count(compiled_text: str) -> int:
    """Count HLO instructions in a compiled executable's textual dump.

    Each non-empty, non-brace line inside the module body that contains an XLA
    assignment (``%name = ... op(...)``) is one instruction; counting ``= `` lines is
    a stable, backend-agnostic proxy for instruction count.
    """
    count = 0
    for line in compiled_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped in {"{", "}"}:
            continue
        if " = " in stripped:
            count += 1
    return count


def _cost_analysis(compiled) -> dict[str, float] | None:
    """Return ``compiled.cost_analysis()`` reduced to scalar float entries.

    XLA returns a dict (or list of dicts) whose entries are numpy scalars; some
    backends do not implement cost analysis and raise, in which case we record None.
    """
    try:
        raw = compiled.cost_analysis()
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        raw = raw[0]
    if not isinstance(raw, Mapping):
        return None
    reduced: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            reduced[str(key)] = float(value)
    return reduced


def _measure_kernel(kernel: _CompiledKernel) -> dict[str, object]:
    """Separately time construct (lower) and XLA compile for one kernel."""
    rss_before_bytes = _maxrss_bytes()

    construct_start = time.perf_counter()
    lowered = kernel.fn.lower(*kernel.args)
    construct_wall_s = time.perf_counter() - construct_start

    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_wall_s = time.perf_counter() - compile_start

    compiled_text = compiled.as_text()
    cost_analysis = _cost_analysis(compiled)
    rss_after_bytes = _maxrss_bytes()

    lowered_text = lowered.as_text()
    measurement = {
        "label": kernel.label,
        "construct_wall_s": construct_wall_s,
        "compile_wall_s": compile_wall_s,
        "hlo_op_count": _hlo_op_count(compiled_text),
        "compiled_text_bytes": len(compiled_text.encode("utf-8")),
        "lowered_text_bytes": len(lowered_text.encode("utf-8")),
        "lowered_text_lines": len(lowered_text.splitlines()),
        "cost_analysis": cost_analysis,
        "peak_host_rss_bytes": rss_after_bytes,
        "peak_host_rss_delta_bytes": max(0, rss_after_bytes - rss_before_bytes),
    }
    del compiled
    del lowered
    return measurement


def _gpu_peak_bytes() -> int | None:
    if jax.default_backend() == "cpu":
        return None
    stats = jax.devices()[0].memory_stats()
    if stats is None:
        return None
    return int(stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0)))


def _probe_resolution(
    args: argparse.Namespace,
    resolution: _Resolution,
    seed_resolved,
) -> dict[str, object]:
    fixture, seed_state = _build_solved_boozer_fixture(args, resolution, seed_resolved)
    kernels = _build_kernels(fixture, num_tf_coils=int(args.num_tf_coils))
    kernel_results = {kernel.label: _measure_kernel(kernel) for kernel in kernels}
    return {
        "mpol": resolution.mpol,
        "ntor": resolution.ntor,
        "nphi": resolution.nphi,
        "ntheta": resolution.ntheta,
        "seed_state": seed_state,
        "decision_vector_size": int(
            np.asarray(
                fixture["boozer_surface"]._pack_decision_vector(
                    fixture["boozer_surface"].res["iota"],
                    fixture["boozer_surface"].res["G"],
                )
            ).size
        ),
        "coil_dof_size": int(np.asarray(fixture["bs"].x, dtype=float).size),
        "kernels": kernel_results,
        "peak_gpu_bytes": _gpu_peak_bytes(),
    }


def _print_summary_table(results_by_mpol: dict[int, dict[str, object]]) -> None:
    kernel_labels = (
        "K1_forward_result_for",
        "K2_solved_state_value_and_grad_for",
        "K0_fused_value_and_grad_for",
    )
    short = {
        "K1_forward_result_for": "K1",
        "K2_solved_state_value_and_grad_for": "K2",
        "K0_fused_value_and_grad_for": "K0fused",
    }
    print("\n## Compile-breadth (construct=lower wall, compile=XLA wall, hlo=op count)\n")
    header = ["mpol", "grid (ntor/nphi/ntheta)", "seed"]
    for label in kernel_labels:
        tag = short[label]
        header.extend(
            [f"{tag} construct_s", f"{tag} compile_s", f"{tag} hlo_ops"]
        )
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for mpol in sorted(results_by_mpol):
        record = results_by_mpol[mpol]
        row = [
            str(mpol),
            f"{record['ntor']}/{record['nphi']}/{record['ntheta']}",
            str(record["seed_state"]),
        ]
        for label in kernel_labels:
            kernel = record["kernels"][label]
            row.extend(
                [
                    f"{kernel['construct_wall_s']:.2f}",
                    f"{kernel['compile_wall_s']:.2f}",
                    str(kernel["hlo_op_count"]),
                ]
            )
        print("| " + " | ".join(row) + " |")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before import/use.",
    )
    parser.add_argument(
        "--mpol-list",
        type=int,
        nargs="+",
        default=list(_DEFAULT_MPOL_LIST),
        help="Surface poloidal mode counts to probe (default: 6 8 10).",
    )
    parser.add_argument(
        "--ntor",
        type=int,
        default=None,
        help="Override the toroidal mode count for EVERY mpol (default: per-mpol grid).",
    )
    parser.add_argument(
        "--nphi",
        type=int,
        default=None,
        help="Override the surface toroidal grid for EVERY mpol (default: per-mpol grid).",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=None,
        help="Override the surface poloidal grid for EVERY mpol (default: per-mpol grid).",
    )
    parser.add_argument(
        "--warm-start-run-dir",
        default=None,
        help=(
            "Single-stage donor directory (surf_opt.json + results.json). When a "
            "probed mpol/ntor matches the donor's resolved surface, the production "
            "resolved Boozer state is installed value-only; other resolutions use a "
            "fresh JAX solve for the compile-shape scaling study."
        ),
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=str(DEFAULT_STAGE2_BS_PATH),
        help="Path to the Stage 2 seed biot_savart_opt.json fixture.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=DEFAULT_PLASMA_SURF_FILENAME,
        help="VMEC equilibrium filename for the single-stage fixture.",
    )
    parser.add_argument(
        "--equilibria-dir",
        default=str(DEFAULT_EQUILIBRIA_DIR),
        help="Directory that contains VMEC equilibrium files.",
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
        "--num-tf-coils",
        type=int,
        default=20,
        help="Number of fixed TF coils in the single-stage seed package.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write the structured compile-breadth results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    resolutions = _resolutions(args)
    seed_resolved = _seed_resolved_surface(args)

    results_by_mpol: dict[int, dict[str, object]] = {}
    for resolution in resolutions:
        print(
            f"[compile-breadth] mpol={resolution.mpol} "
            f"ntor={resolution.ntor} nphi={resolution.nphi} "
            f"ntheta={resolution.ntheta}: building solved fixture + kernels"
        )
        record = _probe_resolution(args, resolution, seed_resolved)
        results_by_mpol[resolution.mpol] = record
        for label, kernel in record["kernels"].items():
            print(
                f"[compile-breadth] mpol={resolution.mpol} {label}: "
                f"construct={kernel['construct_wall_s']:.2f}s "
                f"compile={kernel['compile_wall_s']:.2f}s "
                f"hlo_ops={kernel['hlo_op_count']}"
            )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "requested_platform": REQUESTED_PLATFORM,
        "platform": platform.platform(),
        "devices": _device_summary(),
        "compilation_cache_policy": current_compilation_cache_metadata(),
        "compilation_cache_policy_at_import": _CACHE_POLICY_METADATA,
        "child_cuda_memory_env": CHILD_CUDA_MEMORY_ENV,
        "case": {
            "mpol_list": list(args.mpol_list),
            "warm_start_run_dir": args.warm_start_run_dir,
            "stage2_bs_path": args.stage2_bs_path,
            "plasma_surf_filename": args.plasma_surf_filename,
            "equilibria_dir": args.equilibria_dir,
            "vol_target": float(args.vol_target),
            "iota_target": float(args.iota_target),
            "num_tf_coils": int(args.num_tf_coils),
        },
        "results_by_mpol": {str(mpol): record for mpol, record in results_by_mpol.items()},
    }
    output_path = Path(args.output_json)
    _write_json(output_path, payload)
    _print_summary_table(results_by_mpol)
    print(json.dumps({"output_json": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()


# Pod command line (A40 `qzs2fv1nokop0s`; paths from RUNBOOK.md sec 1.1 / 4.3):
#
#   REPO=/workspace/simsopt_lbfgs_fullcpu_a40_20260619T2155Z
#   PY=/workspace/venvs/simsopt-lbfgs-a40-py312/bin/python
#   SP=$("$PY" -c 'import site;print(site.getsitepackages()[0])')
#   TS=$(date +%s)
#   setsid env -u OPTIMIZER_BACKEND \
#     LD_LIBRARY_PATH="$(ls -d $SP/nvidia/*/lib|paste -sd: -)" \
#     JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu \
#     SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_JAX_PLATFORM=cuda \
#     SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled \
#     JAX_COMPILATION_CACHE_DIR=/workspace/cbp_cache_$TS \
#     XLA_FLAGS="--xla_dump_to=/workspace/cbp_hlo_$TS" \
#     PYTHONPATH="$REPO/src:$REPO" \
#     "$PY" "$REPO/benchmarks/compile_breadth_probe.py" --platform cuda \
#       --mpol-list 6 8 10 \
#       --warm-start-run-dir /workspace/april_iota285_seed \
#       --stage2-bs-path /workspace/april_iota285_seed/biot_savart_opt.json \
#       --plasma-surf-filename wout_nfp5ginsburg_000_014417_iota15.nc \
#       --equilibria-dir /workspace/equilibria \
#       --vol-target 0.04 --iota-target 0.285 --num-tf-coils 20 \
#       --output-json /workspace/cbp_$TS.json </dev/null >/workspace/cbp_$TS.log 2>&1 &
#
# Use a FRESH JAX_COMPILATION_CACHE_DIR per cold-cache run; reuse it only when
# intentionally measuring persistent-cache reuse.  Keep XLA_FLAGS fixed across the
# mpol sweep so the per-mpol compiles share cache keys.
