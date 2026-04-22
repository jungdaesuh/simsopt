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
        "(for example columbia-jax-0.9.2)."
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
        command += ["--optimizer-backend", optimizer_backend]
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
        command = _build_single_stage_script_command(
            backend=backend,
            optimizer_backend=optimizer_backend,
            maxiter=maxiter,
            stage2_bs_path=stage2_bs_path,
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
    with tempfile.TemporaryDirectory(prefix=f"single-stage-{backend}-results-") as tmp_dir:
        output_root = Path(tmp_dir) / "outputs"
        command = _build_single_stage_script_command(
            backend=backend,
            optimizer_backend=optimizer_backend,
            maxiter=maxiter,
            stage2_bs_path=stage2_bs_path,
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
    surface_path = find_single_file(output_root, "surf_opt.json")
    biot_savart_path = find_single_file(output_root, "biot_savart_opt.json")
    return dict(load_json(results_path)), load(surface_path), load(biot_savart_path)


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
        ("mean_abs_bdotn_over_b", cpu.mean_abs_bdotn_over_b, jax.mean_abs_bdotn_over_b, 5e-3),
    ):
        assert np.isfinite(cpu_value), f"{context}: CPU {label} was non-finite"
        assert np.isfinite(jax_value), f"{context}: JAX {label} was non-finite"
        assert cpu_value <= ceiling, (
            f"{context}: CPU {label} exceeded physical ceiling {ceiling}"
        )
        assert jax_value <= ceiling, (
            f"{context}: JAX {label} exceeded physical ceiling {ceiling}"
        )

    for label, threshold, cpu_value, jax_value in (
        ("curve_curve_distance", 0.05, cpu.curve_curve_distance, jax.curve_curve_distance),
        ("curve_surface_distance", 0.02, cpu.curve_surface_distance, jax.curve_surface_distance),
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

        assert payload["rung"] == TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG
        assert payload["passed"] is True
        assert payload["failures"] == []

        provenance = payload["provenance"]
        assert provenance["backend_mode"] == "jax_gpu_parity"
        assert provenance["backend_strict"] is True
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


class TestSingleStageOuterLoopCompileSmoke:
    @pytest.mark.slow
    def test_cpu_target_lane_case_records_compile_diagnostic_accounting(self):
        results = _run_single_stage_script_results(
            backend="jax",
            optimizer_backend="ondevice",
            platform="cpu",
            stage2_bs_path=DEFAULT_STAGE2_BS_PATH,
            maxiter=2,
            benchmark_mode=True,
            record_jax_compile_diagnostics=True,
            target_lane_accepted_step_sync="final-only",
        )

        diagnostics = results.get("JAX_COMPILE_DIAGNOSTICS")
        assert isinstance(diagnostics, dict)
        compile_targets = diagnostics.get("compile_targets")
        cache_miss_sites = diagnostics.get("cache_miss_sites")
        assert isinstance(compile_targets, dict)
        assert isinstance(cache_miss_sites, dict)
        compile_event_count = int(diagnostics.get("compile_event_count", -1))
        cache_miss_count = int(diagnostics.get("cache_miss_count", -1))
        compile_target_parse_miss_count = int(
            diagnostics.get("compile_target_parse_miss_count", -1)
        )
        cache_miss_site_parse_miss_count = int(
            diagnostics.get("cache_miss_site_parse_miss_count", -1)
        )
        assert compile_event_count >= 0
        assert cache_miss_count >= 0
        assert compile_target_parse_miss_count >= 0
        assert cache_miss_site_parse_miss_count >= 0
        assert sum(int(value) for value in compile_targets.values()) == (
            compile_event_count - compile_target_parse_miss_count
        )
        assert sum(int(value) for value in cache_miss_sites.values()) == (
            cache_miss_count - cache_miss_site_parse_miss_count
        )
