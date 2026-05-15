from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.single_stage_smoke_fixture import (  # noqa: E402
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_IOTA_TARGET,
    DEFAULT_NUM_TF_COILS,
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
    DEFAULT_STAGE2_BS_PATH,
    DEFAULT_VOL_TARGET,
    build_real_single_stage_init_fixture,
)
from benchmarks.validation_ladder_common import (  # noqa: E402
    bootstrap_local_simsopt,
    find_single_file,
    load_json,
    repo_pythonpath_env,
    run_python_script,
)
from benchmarks.validation_ladder_contract import (  # noqa: E402
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
    single_stage_proof_contract,
)

bootstrap_local_simsopt()

pytest.importorskip(
    "simsoptpp",
    reason=(
        "Single-stage integration tests require a simsoptpp-backed JAX runtime "
        "(for example the conda env `jax`)."
    ),
)

from simsopt._core.optimizable import load  # noqa: E402
from simsopt.geo import CurveLength  # noqa: E402
from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance  # noqa: E402
from conftest import ensure_gpu_determinism_xla_flag  # noqa: E402


@dataclass(frozen=True)
class SingleStagePhysicsSummary:
    solver_success: bool | None
    iterations: int
    self_intersecting: bool
    final_iota: float
    final_volume: float
    mean_abs_bdotn_over_b: float
    max_abs_bdotn_over_b: float
    banana_curve_length: float
    banana_curve_max_curvature: float
    curve_curve_distance: float
    curve_surface_distance: float


@dataclass(frozen=True)
class SingleStageOuterRun:
    results: dict[str, Any]
    summary: SingleStagePhysicsSummary


def _single_stage_script_path() -> Path:
    return (
        REPO_ROOT
        / "examples"
        / "single_stage_optimization"
        / "SINGLE_STAGE"
        / "single_stage_banana_example.py"
    )


def _single_stage_outer_loop_probe_path() -> Path:
    return REPO_ROOT / "benchmarks" / "single_stage_outer_loop_probe.py"


def _single_stage_parity_cache_key() -> str:
    """Namespace the persistent JAX cache by the live target-lane sources.

    JAX's persistent cache can reuse compiled executables across reruns, but
    these subprocess proofs exercise large traced closures whose CPU executables
    are brittle across in-flight source changes. Salt the cache directory with
    the relevant source contents so warm reruns stay fast without reviving stale
    executables from an older objective/optimizer contract.
    """
    digest = hashlib.sha256()
    for path in (
        _single_stage_script_path(),
        REPO_ROOT / "src" / "simsopt" / "geo" / "boozersurface_jax.py",
        REPO_ROOT / "src" / "simsopt" / "geo" / "optimizer_jax.py",
        REPO_ROOT / "src" / "simsopt" / "geo" / "surfaceobjectives_jax.py",
        REPO_ROOT / "src" / "simsopt" / "field" / "biotsavart.py",
    ):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _single_stage_subprocess_env(
    *,
    backend: str,
    platform: str,
    strict_backend_mode: str | None = None,
    transfer_guard: str | None = None,
) -> dict[str, str]:
    # Keep a stable cache across reruns because the real JAX outer-loop probe
    # can otherwise spend minutes in cold XLA compilation with no stage output.
    env = repo_pythonpath_env(
        platform=platform,
        disable_compilation_cache=(backend != "jax"),
        clear_backend_guardrails=(backend != "jax"),
    )
    if backend == "jax":
        cache_root = (
            REPO_ROOT
            / ".artifacts"
            / "jax_compilation_cache"
            / (
                "test_single_stage_physics_parity"
                f"-{platform}-{_single_stage_parity_cache_key()}"
            )
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(
            tempfile.mkdtemp(
                prefix="run-",
                dir=str(cache_root),
            )
        )
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
        env["SIMSOPT_JAX_COMPILATION_CACHE_POLICY"] = "explicit"
        env.pop("SIMSOPT_DISABLE_JAX_COMPILATION_CACHE", None)
    if strict_backend_mode is not None:
        env["SIMSOPT_BACKEND_MODE"] = str(strict_backend_mode)
        env["SIMSOPT_BACKEND_STRICT"] = "1"
        if strict_backend_mode == "jax_gpu_parity":
            ensure_gpu_determinism_xla_flag(env)
    if transfer_guard is not None:
        env["SIMSOPT_JAX_TRANSFER_GUARD"] = str(transfer_guard)
    return env


def _build_single_stage_script_command(
    *,
    backend: str,
    optimizer_backend: str,
    maxiter: int,
    stage2_bs_path: Path,
    jax_runtime_seed_spec: Path | None = None,
    benchmark_mode: bool = False,
    record_jax_compile_diagnostics: bool = False,
    disable_target_lane_success_filter: bool = False,
    target_lane_accepted_step_sync: str | None = None,
) -> list[str]:
    # Keep the parity module on one explicit outer-loop budget. The production
    # donor-aware auto initial phase is a search-policy heuristic, not a physics
    # invariant, and it now differs intentionally between the CPU/reference and
    # JAX/ondevice lanes.
    command = [
        "--backend",
        backend,
        "--plasma-surf-filename",
        DEFAULT_PLASMA_SURF_FILENAME,
        "--stage2-bs-path",
        str(stage2_bs_path),
        "--nphi",
        str(DEFAULT_SMOKE_NPHI),
        "--ntheta",
        str(DEFAULT_SMOKE_NTHETA),
        "--mpol",
        str(DEFAULT_SMOKE_MPOL),
        "--ntor",
        str(DEFAULT_SMOKE_NTOR),
        "--vol-target",
        str(DEFAULT_VOL_TARGET),
        "--iota-target",
        str(DEFAULT_IOTA_TARGET),
        "--maxiter",
        str(maxiter),
        "--initial-step-scale",
        "1.0",
        "--initial-step-maxiter",
        "0",
        "--equilibria-dir",
        str(DEFAULT_EQUILIBRIA_DIR),
    ]
    if backend == "jax":
        if jax_runtime_seed_spec is None:
            raise ValueError("JAX single-stage commands require a runtime seed spec.")
        command += [
            "--optimizer-backend",
            optimizer_backend,
            "--jax-runtime-seed-spec",
            str(jax_runtime_seed_spec),
        ]
    if benchmark_mode:
        command.append("--benchmark-mode")
    if record_jax_compile_diagnostics:
        command.append("--record-jax-compile-diagnostics")
    if disable_target_lane_success_filter:
        command.append("--disable-target-lane-success-filter")
    if target_lane_accepted_step_sync is not None:
        command += [
            "--target-lane-accepted-step-sync",
            str(target_lane_accepted_step_sync),
        ]
    return command


def _compile_jax_runtime_seed_spec(stage2_bs_path: Path, output_path: Path) -> Path:
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

    fixture = build_real_single_stage_init_fixture(
        backend="cpu",
        plasma_surf_filename=DEFAULT_PLASMA_SURF_FILENAME,
        equilibria_dir=DEFAULT_EQUILIBRIA_DIR,
        stage2_bs_path=stage2_bs_path,
        nphi=DEFAULT_SMOKE_NPHI,
        ntheta=DEFAULT_SMOKE_NTHETA,
        mpol=DEFAULT_SMOKE_MPOL,
        ntor=DEFAULT_SMOKE_NTOR,
        vol_target=DEFAULT_VOL_TARGET,
        iota_target=DEFAULT_IOTA_TARGET,
    )
    boozer_surface = fixture["boozer_surface"]
    biot_savart = fixture["bs"]
    _, stage2_results = single_stage_example.load_stage2_results(str(stage2_bs_path))
    stage2_seed = single_stage_example.build_single_stage_runtime_stage2_seed_payload(
        stage2_results,
        banana_surf_radius=float(stage2_results["banana_surf_radius"]),
    )
    runtime_seed_bs = BiotSavartJAX(biot_savart.coils)
    return Path(
        single_stage_example.write_single_stage_jax_runtime_seed_spec(
            str(output_path),
            surface=boozer_surface.surface,
            iota=float(boozer_surface.res["iota"]),
            G=float(boozer_surface.res["G"]),
            mpol=DEFAULT_SMOKE_MPOL,
            ntor=DEFAULT_SMOKE_NTOR,
            quadpoints_phi=boozer_surface.surface.quadpoints_phi,
            quadpoints_theta=boozer_surface.surface.quadpoints_theta,
            coil_dof_extraction_spec=runtime_seed_bs.coil_dof_extraction_spec(),
            coil_dofs=runtime_seed_bs.x.copy(),
            num_tf_coils=DEFAULT_NUM_TF_COILS,
            banana_curve_index=DEFAULT_NUM_TF_COILS,
            tf_current_A=single_stage_example.resolve_loaded_tf_current_A(
                stage2_results.get("TF_CURRENT_A"),
                biot_savart.coils[:DEFAULT_NUM_TF_COILS],
                enforce_limit=False,
            ),
            banana_current_A=float(
                biot_savart.coils[DEFAULT_NUM_TF_COILS].current.get_value()
            ),
            stage2_seed=stage2_seed,
        )
    )


def test_single_stage_subprocess_env_preserves_existing_xla_flags(monkeypatch):
    monkeypatch.setattr(
        sys.modules[__name__],
        "repo_pythonpath_env",
        lambda **_kwargs: {
            "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/tmp/cuda --other-flag=1"
        },
    )

    env = _single_stage_subprocess_env(
        backend="jax",
        platform="cuda",
        strict_backend_mode="jax_gpu_parity",
    )

    assert env["XLA_FLAGS"].split() == [
        "--xla_gpu_cuda_data_dir=/tmp/cuda",
        "--other-flag=1",
        "--xla_gpu_deterministic_ops=true",
    ]


def _run_single_stage_script(
    *,
    backend: str,
    optimizer_backend: str,
    maxiter: int,
    platform: str,
    stage2_bs_path: Path,
) -> SingleStageOuterRun:
    with tempfile.TemporaryDirectory(prefix=f"single-stage-{backend}-") as tmp_dir:
        output_root = Path(tmp_dir) / "outputs"
        jax_runtime_seed_spec = (
            _compile_jax_runtime_seed_spec(
                stage2_bs_path,
                Path(tmp_dir) / "single_stage_jax_runtime_spec.json",
            )
            if backend == "jax"
            else None
        )
        command = _build_single_stage_script_command(
            backend=backend,
            optimizer_backend=optimizer_backend,
            maxiter=maxiter,
            stage2_bs_path=stage2_bs_path,
            jax_runtime_seed_spec=jax_runtime_seed_spec,
        )
        command[0:0] = [
            "--output-root",
            str(output_root),
        ]
        run_python_script(
            _single_stage_script_path(),
            command,
            env=_single_stage_subprocess_env(backend=backend, platform=platform),
            cwd=REPO_ROOT,
            bootstrap_repo=True,
            stream_output=True,
        )
        results, surface, biot_savart = _load_single_stage_outputs(output_root)
        return SingleStageOuterRun(
            results=results,
            summary=_make_outer_smoke_summary(
                results,
                surface,
                biot_savart,
                iterations=int(results.get("iterations", 0)),
                self_intersecting=bool(results["SELF_INTERSECTING"]),
            ),
        )


def _run_single_stage_script_results(
    *,
    backend: str,
    optimizer_backend: str,
    maxiter: int,
    platform: str,
    stage2_bs_path: Path,
    benchmark_mode: bool = False,
    record_jax_compile_diagnostics: bool = False,
    disable_target_lane_success_filter: bool = False,
    target_lane_accepted_step_sync: str | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix=f"single-stage-{backend}-results-"
    ) as tmp_dir:
        output_root = Path(tmp_dir) / "outputs"
        jax_runtime_seed_spec = (
            _compile_jax_runtime_seed_spec(
                stage2_bs_path,
                Path(tmp_dir) / "single_stage_jax_runtime_spec.json",
            )
            if backend == "jax"
            else None
        )
        command = _build_single_stage_script_command(
            backend=backend,
            optimizer_backend=optimizer_backend,
            maxiter=maxiter,
            stage2_bs_path=stage2_bs_path,
            jax_runtime_seed_spec=jax_runtime_seed_spec,
            benchmark_mode=benchmark_mode,
            record_jax_compile_diagnostics=record_jax_compile_diagnostics,
            disable_target_lane_success_filter=disable_target_lane_success_filter,
            target_lane_accepted_step_sync=target_lane_accepted_step_sync,
        )
        command[0:0] = ["--output-root", str(output_root)]
        run_python_script(
            _single_stage_script_path(),
            command,
            env=_single_stage_subprocess_env(backend=backend, platform=platform),
            cwd=REPO_ROOT,
            bootstrap_repo=True,
            stream_output=True,
        )
        results_path = find_single_file(output_root, "results.json")
        return dict(load_json(results_path))


def _require_cuda_runtime_or_skip() -> None:
    jax = pytest.importorskip("jax")
    if not any(device.platform in {"cuda", "gpu"} for device in jax.devices()):
        pytest.skip("CUDA GPU not available")


def _run_single_stage_outer_loop_probe(
    *,
    platform: str,
    optimizer_backend: str,
    maxiter: int,
    strict_backend_mode: str | None = None,
    transfer_guard: str | None = None,
    enable_compile_diagnostics: bool = False,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="single-stage-outer-loop-probe-"
    ) as tmp_dir:
        output_json = Path(tmp_dir) / "probe.json"
        command = [
            "--platform",
            platform,
            "--optimizer-backend",
            optimizer_backend,
            "--maxiter",
            str(maxiter),
            "--output-json",
            str(output_json),
        ]
        if enable_compile_diagnostics:
            command.append("--enable-compile-diagnostics")
        run_python_script(
            _single_stage_outer_loop_probe_path(),
            command,
            env=_single_stage_subprocess_env(
                backend="jax",
                platform=platform,
                strict_backend_mode=strict_backend_mode,
                transfer_guard=transfer_guard,
            ),
            cwd=REPO_ROOT,
            bootstrap_repo=True,
            stream_output=True,
        )
        return dict(load_json(output_json))


def _load_single_stage_outputs(output_root: Path) -> tuple[dict[str, Any], Any, Any]:
    results_path = find_single_file(output_root, "results.json")
    results = dict(load_json(results_path))
    surface_paths = list(output_root.rglob("surf_opt.json"))
    biot_savart_paths = list(output_root.rglob("biot_savart_opt.json"))
    if surface_paths and biot_savart_paths:
        return results, load(surface_paths[0]), load(biot_savart_paths[0])

    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )
    from simsopt.field.biotsavart_jax_backend import SingleStageRuntimeSpecBiotSavartJAX

    runtime_spec_path = find_single_file(
        output_root, "single_stage_jax_runtime_spec.json"
    )
    runtime_state = single_stage_example.load_single_stage_jax_runtime_seed_spec(
        runtime_spec_path,
        mpol=DEFAULT_SMOKE_MPOL,
        ntor=DEFAULT_SMOKE_NTOR,
        nphi=DEFAULT_SMOKE_NPHI,
        ntheta=DEFAULT_SMOKE_NTHETA,
    )
    return (
        results,
        single_stage_example.build_single_stage_surface_from_jax_runtime_spec(
            runtime_state["runtime_spec"]
        ),
        SingleStageRuntimeSpecBiotSavartJAX(runtime_state["runtime_spec"]),
    )


def _physics_summary(
    surface,
    biot_savart,
) -> tuple[float, float, float, float, float, float]:
    unitn = surface.unitnormal()
    biot_savart.set_points(surface.gamma().reshape(-1, 3))
    b_field = np.asarray(biot_savart.B(), dtype=float).reshape(unitn.shape)
    b_norm = np.linalg.norm(b_field, axis=2)
    bdotn_over_b = np.abs(np.sum(b_field * unitn, axis=2)) / b_norm

    curves = [coil.curve for coil in biot_savart.coils]
    banana_curve = curves[DEFAULT_NUM_TF_COILS]
    return (
        float(np.mean(bdotn_over_b)),
        float(np.max(bdotn_over_b)),
        float(CurveLength(banana_curve).J()),
        float(np.max(banana_curve.kappa())),
        float(CurveCurveDistance(curves, 0.05).shortest_distance()),
        float(CurveSurfaceDistance(curves, surface, 0.02).shortest_distance()),
    )


def _make_summary(
    *,
    final_iota: float,
    final_volume: float,
    solver_success: bool | None,
    iterations: int,
    self_intersecting: bool,
    mean_abs_bdotn_over_b: float,
    max_abs_bdotn_over_b: float,
    banana_curve_length: float,
    banana_curve_max_curvature: float,
    curve_curve_distance: float,
    curve_surface_distance: float,
) -> SingleStagePhysicsSummary:
    return SingleStagePhysicsSummary(
        solver_success=None if solver_success is None else bool(solver_success),
        iterations=int(iterations),
        self_intersecting=bool(self_intersecting),
        final_iota=float(final_iota),
        final_volume=float(final_volume),
        mean_abs_bdotn_over_b=float(mean_abs_bdotn_over_b),
        max_abs_bdotn_over_b=float(max_abs_bdotn_over_b),
        banana_curve_length=float(banana_curve_length),
        banana_curve_max_curvature=float(banana_curve_max_curvature),
        curve_curve_distance=float(curve_curve_distance),
        curve_surface_distance=float(curve_surface_distance),
    )


def _make_outer_smoke_summary(
    results: dict[str, Any],
    surface,
    biot_savart,
    *,
    iterations: int,
    self_intersecting: bool,
) -> SingleStagePhysicsSummary:
    (
        mean_abs_bdotn_over_b,
        max_abs_bdotn_over_b,
        banana_curve_length,
        banana_curve_max_curvature,
        curve_curve_distance,
        curve_surface_distance,
    ) = _physics_summary(surface, biot_savart)
    return _make_summary(
        final_iota=float(results["FINAL_IOTA"]),
        final_volume=float(surface.volume()),
        solver_success=None,
        iterations=iterations,
        self_intersecting=self_intersecting,
        mean_abs_bdotn_over_b=float(results["FIELD_ERROR"]),
        max_abs_bdotn_over_b=max_abs_bdotn_over_b,
        banana_curve_length=banana_curve_length,
        banana_curve_max_curvature=banana_curve_max_curvature,
        curve_curve_distance=curve_curve_distance,
        curve_surface_distance=curve_surface_distance,
    )


def _make_init_summary(
    booz_jax,
    biot_savart,
    *,
    solver_success: bool,
    iterations: int,
    self_intersecting: bool,
) -> SingleStagePhysicsSummary:
    (
        mean_abs_bdotn_over_b,
        max_abs_bdotn_over_b,
        banana_curve_length,
        banana_curve_max_curvature,
        curve_curve_distance,
        curve_surface_distance,
    ) = _physics_summary(booz_jax.surface, biot_savart)
    return _make_summary(
        final_iota=float(booz_jax.res["iota"]),
        final_volume=float(booz_jax.surface.volume()),
        solver_success=solver_success,
        iterations=iterations,
        self_intersecting=self_intersecting,
        mean_abs_bdotn_over_b=mean_abs_bdotn_over_b,
        max_abs_bdotn_over_b=max_abs_bdotn_over_b,
        banana_curve_length=banana_curve_length,
        banana_curve_max_curvature=banana_curve_max_curvature,
        curve_curve_distance=curve_curve_distance,
        curve_surface_distance=curve_surface_distance,
    )


def _build_init_fixture(
    *,
    backend: str,
    bs_dofs_override: np.ndarray | None = None,
) -> tuple[dict[str, object], SingleStagePhysicsSummary]:
    fixture = build_real_single_stage_init_fixture(
        backend=backend,
        plasma_surf_filename=DEFAULT_PLASMA_SURF_FILENAME,
        equilibria_dir=DEFAULT_EQUILIBRIA_DIR,
        stage2_bs_path=DEFAULT_STAGE2_BS_PATH,
        nphi=DEFAULT_SMOKE_NPHI,
        ntheta=DEFAULT_SMOKE_NTHETA,
        mpol=DEFAULT_SMOKE_MPOL,
        ntor=DEFAULT_SMOKE_NTOR,
        vol_target=DEFAULT_VOL_TARGET,
        iota_target=DEFAULT_IOTA_TARGET,
        bs_dofs_override=bs_dofs_override,
    )
    boozer_surface = fixture["boozer_surface"]
    biot_savart = fixture["bs"]
    result = boozer_surface.res or {}
    summary = _make_init_summary(
        boozer_surface,
        biot_savart,
        solver_success=bool(result.get("success", False)),
        iterations=int(result.get("iter", 0)),
        self_intersecting=bool(boozer_surface.surface.is_self_intersecting()),
    )
    return fixture, summary


def _assert_physics_quantity_parity(
    cpu: SingleStagePhysicsSummary,
    jax: SingleStagePhysicsSummary,
    *,
    context: str,
    rtol: float = 2e-3,
    atol: float = 1e-6,
) -> None:
    if cpu.solver_success is not None or jax.solver_success is not None:
        assert cpu.solver_success is True and jax.solver_success is True, (
            f"{context}: solver failed"
        )
    assert cpu.self_intersecting == jax.self_intersecting, (
        f"{context}: self-intersection status diverged"
    )
    comparisons = {
        "final_iota": (cpu.final_iota, jax.final_iota),
        "final_volume": (cpu.final_volume, jax.final_volume),
        "mean_abs_bdotn_over_b": (
            cpu.mean_abs_bdotn_over_b,
            jax.mean_abs_bdotn_over_b,
        ),
        "max_abs_bdotn_over_b": (cpu.max_abs_bdotn_over_b, jax.max_abs_bdotn_over_b),
        "banana_curve_length": (cpu.banana_curve_length, jax.banana_curve_length),
        "banana_curve_max_curvature": (
            cpu.banana_curve_max_curvature,
            jax.banana_curve_max_curvature,
        ),
        "curve_curve_distance": (cpu.curve_curve_distance, jax.curve_curve_distance),
        "curve_surface_distance": (
            cpu.curve_surface_distance,
            jax.curve_surface_distance,
        ),
    }
    for field_name, (cpu_value, jax_value) in comparisons.items():
        np.testing.assert_allclose(
            jax_value,
            cpu_value,
            rtol=rtol,
            atol=atol,
            err_msg=f"{context}: {field_name} parity failed",
        )


def _assert_outer_loop_single_step_consistency(
    cpu_run: SingleStageOuterRun,
    jax_run: SingleStageOuterRun,
    *,
    context: str,
) -> None:
    """Check one-step outer-loop consistency without assuming path equality.

    The CPU/reference and ondevice target lanes use different L-BFGS
    implementations, so a one-iteration budget does not guarantee identical
    accepted line-search steps even when both optimize the same objective.
    Require the objective and core field quantities to remain close while hard
    geometry constraints stay satisfied on both lanes.

    Cross-lane vs ceilings-only metric classification:

    * Cross-lane (CPU and JAX must converge to the same value on the same
      field, modulo line-search differences):
        - ``final_objective``         (rtol=5e-4)
        - ``final_volume``            (rtol=2e-3)
        - ``final_iota``              (atol=3e-4)
        - ``mean_abs_bdotn_over_b``   (rtol=5e-3) — quality metric on the
          shared B-field and surface; if both lanes optimize the same
          objective they must agree on the resulting field-error norm.
    * Ceilings-only (hard geometry/coil-design constraints; both lanes must
      satisfy them independently, but the absolute value is path-dependent
      because each L-BFGS trajectory walks through different feasible
      interiors of the constraint set):
        - ``curve_curve_distance``        (>= 0.05)
        - ``curve_surface_distance``      (>= 0.02)
        - ``banana_curve_max_curvature``  (<= 40.0)
      A cross-lane equality assertion here would over-constrain the
      optimizer — two trajectories with different accepted step lengths
      can legitimately end at different points inside the feasible set.
    """
    cpu = cpu_run.summary
    jax = jax_run.summary
    assert cpu.self_intersecting is False, f"{context}: CPU step self-intersected"
    assert jax.self_intersecting is False, f"{context}: JAX step self-intersected"

    np.testing.assert_allclose(
        float(jax_run.results["FINAL_OBJECTIVE"]),
        float(cpu_run.results["FINAL_OBJECTIVE"]),
        rtol=5e-4,
        atol=1e-6,
        err_msg=f"{context}: final objective diverged",
    )
    np.testing.assert_allclose(
        jax.final_volume,
        cpu.final_volume,
        rtol=2e-3,
        atol=1e-6,
        err_msg=f"{context}: final volume parity failed",
    )
    np.testing.assert_allclose(
        jax.final_iota,
        cpu.final_iota,
        rtol=0.0,
        atol=3e-4,
        err_msg=f"{context}: final iota drift exceeded absolute tolerance",
    )
    for label, cpu_value, jax_value, ceiling in (
        (
            "mean_abs_bdotn_over_b",
            cpu.mean_abs_bdotn_over_b,
            jax.mean_abs_bdotn_over_b,
            5e-3,
        ),
    ):
        assert np.isfinite(cpu_value), f"{context}: CPU {label} was non-finite"
        assert np.isfinite(jax_value), f"{context}: JAX {label} was non-finite"
        assert cpu_value <= ceiling, (
            f"{context}: CPU {label} exceeded physical ceiling {ceiling}"
        )
        assert jax_value <= ceiling, (
            f"{context}: JAX {label} exceeded physical ceiling {ceiling}"
        )
        # Cross-lane assertion (audit #14): mean_abs_bdotn_over_b is a
        # quality metric on the same B-field and surface; if both lanes
        # truly optimize the same objective and reach the same accepted
        # step neighborhood, the resulting field-error norm must agree
        # within a loose-but-honest tolerance. The ceiling check above
        # only catches absolute violations; this catches two divergent
        # trajectories that both happen to stay under the ceiling.
        np.testing.assert_allclose(
            jax_value,
            cpu_value,
            rtol=5e-3,
            atol=0.0,
            err_msg=(
                f"{context}: {label} should converge cross-lane "
                "(quality metric on same field)"
            ),
        )

    for label, threshold, cpu_value, jax_value in (
        (
            "curve_curve_distance",
            0.05,
            cpu.curve_curve_distance,
            jax.curve_curve_distance,
        ),
        (
            "curve_surface_distance",
            0.02,
            cpu.curve_surface_distance,
            jax.curve_surface_distance,
        ),
    ):
        assert cpu_value >= threshold, (
            f"{context}: CPU {label} violated hard threshold {threshold}"
        )
        assert jax_value >= threshold, (
            f"{context}: JAX {label} violated hard threshold {threshold}"
        )
    for label, limit, cpu_value, jax_value in (
        (
            "banana_curve_max_curvature",
            40.0,
            cpu.banana_curve_max_curvature,
            jax.banana_curve_max_curvature,
        ),
    ):
        assert cpu_value <= limit, f"{context}: CPU {label} exceeded hard limit {limit}"
        assert jax_value <= limit, f"{context}: JAX {label} exceeded hard limit {limit}"


@pytest.fixture(scope="module")
def outer_baseline_runs() -> tuple[SingleStageOuterRun, SingleStageOuterRun]:
    cpu_run = _run_single_stage_script(
        backend="cpu",
        optimizer_backend="scipy",
        maxiter=1,
        platform="cpu",
        stage2_bs_path=DEFAULT_STAGE2_BS_PATH,
    )
    jax_run = _run_single_stage_script(
        backend="jax",
        optimizer_backend="ondevice",
        maxiter=1,
        platform="cpu",
        stage2_bs_path=DEFAULT_STAGE2_BS_PATH,
    )
    return cpu_run, jax_run


@pytest.fixture(scope="module")
def init_baseline_runs() -> tuple[
    dict[str, object],
    SingleStagePhysicsSummary,
    dict[str, object],
    SingleStagePhysicsSummary,
]:
    cpu_fixture, cpu_summary = _build_init_fixture(backend="cpu")
    jax_fixture, jax_summary = _build_init_fixture(backend="jax")
    return cpu_fixture, cpu_summary, jax_fixture, jax_summary


class TestSingleStagePhysicsSmokeParity:
    def test_outer_loop_physics_quantity_single_step_budget_smoke_parity(
        self,
        outer_baseline_runs,
    ):
        """One-step-budget outer loop stays objective-consistent and feasible."""
        cpu_run, jax_run = outer_baseline_runs
        _assert_outer_loop_single_step_consistency(
            cpu_run,
            jax_run,
            context="single-stage outer-loop smoke parity",
        )
        assert cpu_run.results["max_iterations"] == 1
        assert jax_run.results["max_iterations"] == 1
        assert cpu_run.results["TERMINATION_MESSAGE"] != "init_only"
        assert jax_run.results["TERMINATION_MESSAGE"] != "init_only"
        assert cpu_run.results["iterations"] == jax_run.results["iterations"]

    def test_init_state_sensitivity_smoke_parity_under_small_initial_coil_perturbation(
        self, init_baseline_runs
    ):
        """Small perturbations preserve init-state parity; this is not a full basin study."""
        cpu_fixture, cpu_summary, jax_fixture, jax_summary = init_baseline_runs
        base_dofs = np.asarray(cpu_fixture["bs"].x, dtype=float)
        rng = np.random.RandomState(7)
        perturbation = rng.standard_normal(base_dofs.shape)
        perturbation /= np.linalg.norm(perturbation)
        perturbed_dofs = (
            base_dofs + 1e-4 * max(np.linalg.norm(base_dofs), 1.0) * perturbation
        )

        _, cpu_perturbed_summary = _build_init_fixture(
            backend="cpu",
            bs_dofs_override=perturbed_dofs,
        )
        _, jax_perturbed_summary = _build_init_fixture(
            backend="jax",
            bs_dofs_override=perturbed_dofs,
        )

        _assert_physics_quantity_parity(
            cpu_summary,
            cpu_perturbed_summary,
            context="cpu basin stability",
            rtol=5e-3,
            atol=1e-3,
        )
        _assert_physics_quantity_parity(
            jax_summary,
            jax_perturbed_summary,
            context="jax basin stability",
            rtol=5e-3,
            atol=1e-3,
        )
        _assert_physics_quantity_parity(
            cpu_perturbed_summary,
            jax_perturbed_summary,
            context="perturbed CPU vs JAX basin parity",
            rtol=5e-3,
            atol=1e-3,
        )


# Audit #22 pin (the no-progress sentinel): require the strict-transfer-guard
# CUDA outer-loop probe to reduce the objective by at least 5% over its 10
# accepted L-BFGS iterations. This is a conservative floor — a healthy outer
# loop typically descends much further; the floor catches a "barely moved"
# regression that ``objective_decrease > 0`` alone would silently accept.
_CUDA_OUTER_LOOP_OBJECTIVE_DECREASE_RATIO_CEILING = 0.95


class TestSingleStageOuterLoopGpuProof:
    @pytest.mark.slow
    def test_cuda_outer_loop_probe_converges_under_strict_transfer_guard(self):
        _require_cuda_runtime_or_skip()
        contract = single_stage_proof_contract(TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG)
        payload = _run_single_stage_outer_loop_probe(
            platform="cuda",
            optimizer_backend="ondevice",
            maxiter=int(contract["default_maxiter"]),
            strict_backend_mode="jax_gpu_parity",
            transfer_guard="disallow",
        )

        # Audit #22: the probe driver self-reports ``payload["passed"]`` /
        # ``payload["failures"]`` — asserting on those would be circular.
        # Instead, assert the rung tag and the individual physics-content
        # fields that *compose* the probe verdict, then independently
        # re-evaluate the final objective from the recorded component
        # penalties (a code path that does not flow through the
        # optimizer's tracked ``fun`` value).
        assert payload["rung"] == TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG

        provenance = payload["provenance"]
        assert provenance["backend_mode"] == "jax_gpu_parity"
        assert provenance["backend_strict"] is True
        # SIMSOPT_JAX_TRANSFER_GUARD=disallow makes any host<->device
        # transfer raise inside the subprocess; reaching this assertion
        # means the subprocess returned normally, so the
        # transfer-guard violation count is exactly zero.
        assert provenance["transfer_guard"] == "disallow"

        probe = payload["probe"]
        assert probe["iterations"] >= int(contract["min_iterations"])
        assert (
            probe["outer_optimizer_method"]
            == contract["required_outer_optimizer_method"]
        )
        assert probe["boozer_optimizer_backend"] == "ondevice"
        assert probe["boozer_optimizer_method"] == "bfgs-ondevice"
        assert probe["initial_objective"] is not None
        assert probe["final_objective"] is not None
        assert probe["objective_decreased"] is True
        assert probe["objective_decrease"] is not None
        assert probe["objective_decrease"] > 0.0
        assert probe["self_intersecting"] is False
        assert all(probe["finite_result_keys"].values())

        # Audit #22 point 2: pin the objective-decrease ratio against a
        # recorded ceiling so a "moved by epsilon" regression fails loudly
        # instead of riding through on ``objective_decrease > 0`` alone.
        initial_objective = float(probe["initial_objective"])
        final_objective = float(probe["final_objective"])
        assert initial_objective > 0.0, (
            "initial single-stage outer-loop objective must be strictly "
            "positive (weighted-penalty sum); got "
            f"{initial_objective}."
        )
        objective_decrease_ratio = final_objective / initial_objective
        assert objective_decrease_ratio < 1.0, (
            "Final/initial objective ratio must be < 1 for a real "
            f"descent; got {objective_decrease_ratio}."
        )
        assert objective_decrease_ratio < (
            _CUDA_OUTER_LOOP_OBJECTIVE_DECREASE_RATIO_CEILING
        ), (
            "Outer-loop objective ratio (final/initial) regressed past "
            "the pinned 5%-decrease ceiling "
            f"{_CUDA_OUTER_LOOP_OBJECTIVE_DECREASE_RATIO_CEILING}; got "
            f"{objective_decrease_ratio} "
            f"(initial={initial_objective}, final={final_objective})."
        )

        # Audit #22 point 3: independent re-evaluation oracle. The probe
        # records ``FINAL_OBJECTIVE`` from the optimizer's tracked ``fun``
        # value, while ``FINAL_NON_QS`` / ``FINAL_*_PENALTY`` come from a
        # separate post-optimization JAX path (the runtime bundle's
        # ``reporting_metrics`` JIT, not ``value_and_grad``). Recomposing
        # the weighted sum and matching ``FINAL_OBJECTIVE`` at machine
        # precision validates the optimizer's reported value through a
        # code path that does not flow through the optimizer.
        results = payload["results"]
        recomputed_final_objective = (
            float(results["FINAL_NON_QS"])
            + float(results["RES_WEIGHT"]) * float(results["FINAL_BOOZER_RESIDUAL"])
            + float(results["IOTAS_WEIGHT"]) * float(results["FINAL_IOTA_PENALTY"])
            + float(results["LENGTH_WEIGHT"]) * float(results["FINAL_LENGTH_PENALTY"])
            + float(results["CC_WEIGHT"]) * float(results["FINAL_CURVE_CURVE_PENALTY"])
            + float(results["CS_WEIGHT"])
            * float(results["FINAL_CURVE_SURFACE_PENALTY"])
            + float(results["SURF_DIST_WEIGHT"])
            * float(results["FINAL_SURFACE_VESSEL_PENALTY"])
            + float(results["CURVATURE_WEIGHT"])
            * float(results["FINAL_CURVATURE_PENALTY"])
        )
        np.testing.assert_allclose(
            recomputed_final_objective,
            final_objective,
            rtol=1e-10,
            atol=0.0,
            err_msg=(
                "Independent recomputation of the final outer-loop "
                "objective from weighted penalty components disagreed "
                "with the optimizer's reported FINAL_OBJECTIVE; the "
                "probe verdict is no longer self-consistent."
            ),
        )


# Audit #23: ``TestSingleStageOuterLoopCompileSmoke`` moved to
# ``tests/test_jax_compile_diagnostics.py`` as
# ``TestJaxCompileDiagnosticParser``. That test verifies parser
# invariants of the ``JAX_COMPILE_DIAGNOSTICS`` recorder; it is an
# instrumentation/bookkeeping test, not physics parity, and so does not
# belong in a file named ``*_physics_parity.py``. The helper
# ``_run_single_stage_script_results`` defined above is intentionally
# kept here as the single source of truth — the new file imports it
# rather than duplicating ~200 lines of subprocess plumbing.
