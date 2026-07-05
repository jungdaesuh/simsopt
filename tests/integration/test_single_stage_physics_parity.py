from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sys
import tempfile
import types
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
    SMOKE_TEST_STAGE2_BS_PATH,
    DEFAULT_VOL_TARGET,
    build_real_single_stage_init_fixture,
)
from benchmarks.validation_ladder_common import (  # noqa: E402
    bootstrap_local_simsopt,
    find_single_file,
    load_single_stage_final_payload,
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
        REPO_ROOT / "src" / "simsopt_jax_adapters" / "geo" / "boozer_surface.py",
        REPO_ROOT / "src" / "simsopt_jax" / "geo" / "optimizers" / "optimizer.py",
        REPO_ROOT / "src" / "simsopt_jax_adapters" / "geo" / "surface_objectives.py",
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
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_root)
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
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

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
        "--xla_gpu_exclude_nondeterministic_ops=true",
    ]
    cache_dir = Path(env["JAX_COMPILATION_CACHE_DIR"])
    assert cache_dir.parent.name == "jax_compilation_cache"
    assert cache_dir.name.startswith("test_single_stage_physics_parity-cuda-")


def test_single_stage_subprocess_env_uses_cuda_only_by_default(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    env = _single_stage_subprocess_env(
        backend="jax",
        platform="cuda",
        strict_backend_mode="jax_gpu_parity",
    )

    assert env["JAX_PLATFORMS"] == "cuda"


def test_repo_pythonpath_env_preserves_explicit_cpu_callback_lane(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")

    env = repo_pythonpath_env(platform="cuda")

    assert env["JAX_PLATFORMS"] == "cuda,cpu"


_TRACEABLE_NEWTON_LINEAR_SOLVER_ENV = "SIMSOPT_TRACEABLE_NEWTON_LINEAR_SOLVER"


def test_repo_pythonpath_env_injects_reference_leg_newton_linear_solver(monkeypatch):
    """The reference leg forces the solver into its child env; the target leg does not.

    The two matrix legs differ only by whether the ``traceable_newton_linear_solver``
    override is supplied to the env builder. The reference call site passes the
    resolved value; the target call site omits the argument entirely, so a
    native-CPU target can never inherit the reference-only override.
    """
    monkeypatch.delenv(_TRACEABLE_NEWTON_LINEAR_SOLVER_ENV, raising=False)

    reference_env = repo_pythonpath_env(
        platform="cpu",
        traceable_newton_linear_solver="dense_lu",
    )
    target_env = repo_pythonpath_env(platform="cuda")

    assert reference_env[_TRACEABLE_NEWTON_LINEAR_SOLVER_ENV] == "dense_lu"
    assert _TRACEABLE_NEWTON_LINEAR_SOLVER_ENV not in target_env


def test_repo_pythonpath_env_newton_solver_override_none_preserves_inheritance(
    monkeypatch,
):
    """A ``None`` override injects nothing and never scrubs an inherited value."""
    monkeypatch.setenv(_TRACEABLE_NEWTON_LINEAR_SOLVER_ENV, "operator_gmres")
    inherited_env = repo_pythonpath_env(
        platform="cpu",
        traceable_newton_linear_solver=None,
    )
    assert inherited_env[_TRACEABLE_NEWTON_LINEAR_SOLVER_ENV] == "operator_gmres"

    monkeypatch.delenv(_TRACEABLE_NEWTON_LINEAR_SOLVER_ENV, raising=False)
    absent_env = repo_pythonpath_env(
        platform="cpu",
        traceable_newton_linear_solver=None,
    )
    assert _TRACEABLE_NEWTON_LINEAR_SOLVER_ENV not in absent_env


def test_reference_leg_newton_linear_solver_override_resolves_matrix_knob(monkeypatch):
    """The MATRIX knob maps to a pass-through value (non-empty) or ``None``.

    The resolver stays deliberately non-validating: the child process's solver
    resolver in ``simsopt_jax.geo.optimizers.optimizer`` is the single source of
    truth for the accepted vocabulary and fails loud on unknown codes, so the
    harness forwards the value verbatim.
    """
    from benchmarks.single_stage_init_parity import (
        _reference_leg_newton_linear_solver_override,
    )

    monkeypatch.setenv(
        "MATRIX_REFERENCE_NEWTON_LINEAR_SOLVER", "hybrid_final_dense_ir"
    )
    assert _reference_leg_newton_linear_solver_override() == "hybrid_final_dense_ir"

    monkeypatch.delenv(_TRACEABLE_NEWTON_LINEAR_SOLVER_ENV, raising=False)
    reference_env = repo_pythonpath_env(
        platform="cpu",
        traceable_newton_linear_solver=_reference_leg_newton_linear_solver_override(),
    )
    target_env = repo_pythonpath_env(platform="cuda")
    assert reference_env[_TRACEABLE_NEWTON_LINEAR_SOLVER_ENV] == "hybrid_final_dense_ir"
    assert _TRACEABLE_NEWTON_LINEAR_SOLVER_ENV not in target_env

    monkeypatch.delenv("MATRIX_REFERENCE_NEWTON_LINEAR_SOLVER", raising=False)
    assert _reference_leg_newton_linear_solver_override() is None

    monkeypatch.setenv("MATRIX_REFERENCE_NEWTON_LINEAR_SOLVER", "")
    assert _reference_leg_newton_linear_solver_override() is None


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
        jax_runtime_seed_spec = _compile_jax_runtime_seed_spec(
            SMOKE_TEST_STAGE2_BS_PATH,
            Path(tmp_dir) / "single_stage_jax_runtime_spec.json",
        )
        command = [
            "--platform",
            platform,
            "--optimizer-backend",
            optimizer_backend,
            "--maxiter",
            str(maxiter),
            "--jax-runtime-seed-spec",
            str(jax_runtime_seed_spec),
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
    results, _ = load_single_stage_final_payload(output_root)
    surface_paths = list(output_root.rglob("surf_opt.json"))
    biot_savart_paths = list(output_root.rglob("biot_savart_opt.json"))
    if surface_paths and biot_savart_paths:
        return results, load(surface_paths[0]), load(biot_savart_paths[0])

    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )
    from simsopt_jax_adapters.field.biotsavart_backend import (
        SingleStageRuntimeSpecBiotSavartJAX,
    )

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
        stage2_bs_path=SMOKE_TEST_STAGE2_BS_PATH,
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

    Metric classification:

    * Lane-local descent (CPU and JAX must each make objective progress from
      their own starting objective):
        - ``FINAL_OBJECTIVE`` must be finite and below ``INITIAL_OBJECTIVE``.
        - the weighted final objective must recompose from its result payload.
    * Lane-local physics ceilings (both lanes must satisfy the same physical
      acceptability bounds independently):
        - ``final_volume`` near the target volume.
        - ``mean_abs_bdotn_over_b`` below the one-step field-error ceiling.
        - ``curve_curve_distance``        (>= 0.05)
        - ``curve_surface_distance``      (>= 0.02)
        - ``banana_curve_max_curvature``  (<= 40.0)
      Cross-lane equality would over-constrain this smoke: two valid L-BFGS
      implementations can take different accepted line-search steps on the
      same first-iteration budget.
    """
    cpu = cpu_run.summary
    jax = jax_run.summary
    assert cpu.self_intersecting is False, f"{context}: CPU step self-intersected"
    assert jax.self_intersecting is False, f"{context}: JAX step self-intersected"

    for lane, run in (("CPU", cpu_run), ("JAX", jax_run)):
        initial_objective = float(run.results["INITIAL_OBJECTIVE"])
        final_objective = float(run.results["FINAL_OBJECTIVE"])
        assert np.isfinite(initial_objective), (
            f"{context}: {lane} INITIAL_OBJECTIVE was non-finite"
        )
        assert np.isfinite(final_objective), (
            f"{context}: {lane} FINAL_OBJECTIVE was non-finite"
        )
        assert initial_objective > 0.0, (
            f"{context}: {lane} INITIAL_OBJECTIVE must be positive"
        )
        assert final_objective < initial_objective, (
            f"{context}: {lane} objective did not decrease "
            f"(initial={initial_objective}, final={final_objective})"
        )
        recomposed_final_objective = (
            float(run.results["FINAL_NON_QS"])
            + float(run.results["RES_WEIGHT"])
            * float(run.results["FINAL_BOOZER_RESIDUAL"])
            + float(run.results["IOTAS_WEIGHT"])
            * float(run.results["FINAL_IOTA_PENALTY"])
            + float(run.results["LENGTH_WEIGHT"])
            * float(run.results["FINAL_LENGTH_PENALTY"])
            + float(run.results["CC_WEIGHT"])
            * float(run.results["FINAL_CURVE_CURVE_PENALTY"])
            + float(run.results["CS_WEIGHT"])
            * float(run.results["FINAL_CURVE_SURFACE_PENALTY"])
            + float(run.results["SURF_DIST_WEIGHT"])
            * float(run.results["FINAL_SURFACE_VESSEL_PENALTY"])
            + float(run.results["CURVATURE_WEIGHT"])
            * float(run.results["FINAL_CURVATURE_PENALTY"])
        )
        np.testing.assert_allclose(
            recomposed_final_objective,
            final_objective,
            rtol=1e-10,
            atol=0.0,
            err_msg=(
                f"{context}: {lane} weighted penalty components disagreed "
                "with FINAL_OBJECTIVE"
            ),
        )

    for lane, summary in (("CPU", cpu), ("JAX", jax)):
        np.testing.assert_allclose(
            summary.final_volume,
            DEFAULT_VOL_TARGET,
            rtol=2e-3,
            atol=1e-6,
            err_msg=f"{context}: {lane} final volume left target envelope",
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
        stage2_bs_path=SMOKE_TEST_STAGE2_BS_PATH,
    )
    jax_run = _run_single_stage_script(
        backend="jax",
        optimizer_backend="ondevice",
        maxiter=1,
        platform="cpu",
        stage2_bs_path=SMOKE_TEST_STAGE2_BS_PATH,
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
            optimizer_backend="scipy-jax",
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


def test_host_jax_compile_diagnostics_condition_includes_host_outer():
    from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
        should_record_single_stage_jax_compile_diagnostics,
    )

    args = types.SimpleNamespace(record_jax_compile_diagnostics=True)

    assert should_record_single_stage_jax_compile_diagnostics(
        args,
        use_target_lane=False,
        use_host_jax_outer_objective=True,
    )
    assert not should_record_single_stage_jax_compile_diagnostics(
        args,
        use_target_lane=False,
        use_host_jax_outer_objective=False,
    )


def test_single_stage_cli_accepts_reuse_resolved_warm_start_solve(monkeypatch):
    from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
        parse_args,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_banana_example.py",
            "--backend",
            "jax",
            "--reuse-resolved-warm-start-solve",
        ],
    )

    args = parse_args()

    assert args.backend == "jax"
    assert args.reuse_resolved_warm_start_solve is True


def test_single_stage_cli_accepts_compact_objective_evaluation_trace(monkeypatch):
    from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
        parse_args,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_banana_example.py",
            "--record-objective-evaluation-trace",
            "--compact-objective-evaluation-trace",
        ],
    )

    args = parse_args()

    assert args.record_objective_evaluation_trace is True
    assert args.compact_objective_evaluation_trace is True


def test_compact_single_stage_progress_payload_omits_replay_values():
    from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
        compact_single_stage_progress_payload,
    )

    payload = {
        "objective": {"value": 1.25, "finite": True},
        "native_gradient": {
            "values": [float(index) for index in range(100)],
            "inf_norm": 99.0,
            "size": 100,
            "all_finite": True,
        },
        "boozer_solver_metadata": {"linearization_kind": "hessian"},
        "dense_hessian": {
            "values": [[1.0, 0.0], [0.0, 1.0]],
            "shape": [2, 2],
        },
    }

    compact = compact_single_stage_progress_payload(payload)

    assert compact["objective"]["value"] == 1.25
    assert compact["native_gradient"]["inf_norm"] == 99.0
    assert compact["native_gradient"]["values_omitted"] is True
    assert compact["native_gradient"]["values_length"] == 100
    assert "values" not in compact["native_gradient"]
    assert compact["boozer_solver_metadata"]["linearization_kind"] == "hessian"
    assert compact["dense_hessian"]["shape"] == [2, 2]
    assert compact["dense_hessian"]["values_omitted"] is True


def test_outer_loop_probe_cli_accepts_host_jax_memory_gate(monkeypatch, tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        _expected_outer_optimizer_method,
        parse_args,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_outer_loop_probe.py",
            "--output-json",
            str(tmp_path / "probe.json"),
            "--optimizer-backend",
            "host-jax",
            "--boozer-least-squares-algorithm",
            "lm",
            "--target-lane-boozer-newton-tol",
            "1e-13",
            "--target-lane-boozer-newton-maxiter",
            "80",
            "--single-stage-case-timeout-seconds",
            "60",
            "--enable-host-jax-memory-gate",
        ],
    )

    args = parse_args()

    assert args.optimizer_backend == "host-jax"
    assert args.boozer_least_squares_algorithm == "lm"
    assert args.target_lane_boozer_newton_tol == pytest.approx(1.0e-13)
    assert args.target_lane_boozer_newton_maxiter == 80
    assert args.single_stage_case_timeout_seconds == pytest.approx(60.0)
    assert args.enable_host_jax_memory_gate is True
    assert _expected_outer_optimizer_method(args.optimizer_backend) == "lbfgs"


def test_outer_loop_probe_cli_accepts_phase3_gradient_proof_gate(monkeypatch, tmp_path):
    from benchmarks.single_stage_outer_loop_probe import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_outer_loop_probe.py",
            "--output-json",
            str(tmp_path / "probe.json"),
            "--record-objective-evaluation-trace",
            "--enable-phase3-gradient-proof-gate",
            "--phase3-constraint-margin-abs-tol",
            "1e-6",
        ],
    )

    args = parse_args()

    assert args.record_objective_evaluation_trace is True
    assert args.enable_phase3_gradient_proof_gate is True
    assert args.phase3_constraint_margin_abs_tol == pytest.approx(1.0e-6)


def test_outer_loop_probe_cli_accepts_phase0_noise_calibration_replay_gate(
    monkeypatch,
    tmp_path,
):
    from benchmarks.single_stage_outer_loop_probe import parse_args

    baseline_progress_json = tmp_path / "baseline_outer_optimizer_progress.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_outer_loop_probe.py",
            "--output-json",
            str(tmp_path / "probe.json"),
            "--replay-objective-evaluation-trace",
            str(baseline_progress_json),
            "--enable-phase0-noise-calibration-gate",
            "--phase0-noise-baseline-newton-tol",
            "1e-11",
            "--phase0-noise-tightened-newton-tol",
            "1e-13",
            "--phase0-noise-objective-evaluation-index",
            "2",
        ],
    )

    args = parse_args()

    assert args.replay_objective_evaluation_trace == str(baseline_progress_json)
    assert args.enable_phase0_noise_calibration_gate is True
    assert args.phase0_noise_baseline_progress_json is None
    assert args.phase0_noise_baseline_newton_tol == pytest.approx(1.0e-11)
    assert args.phase0_noise_tightened_newton_tol == pytest.approx(1.0e-13)
    assert args.phase0_noise_objective_evaluation_index == 2


def test_outer_loop_probe_defaults_to_runtime_seed_resolved_state_reuse(monkeypatch):
    from benchmarks.single_stage_outer_loop_probe import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_outer_loop_probe.py",
            "--output-json",
            "/tmp/probe.json",
            "--jax-runtime-seed-spec",
            "/tmp/single_stage_jax_runtime_spec.json",
        ],
    )

    args = parse_args()

    assert args.reuse_jax_runtime_seed_solve is True
    assert args.outer_maxls == 20


def test_outer_loop_probe_can_replay_runtime_seed_setup_solve(monkeypatch):
    from benchmarks.single_stage_outer_loop_probe import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_outer_loop_probe.py",
            "--output-json",
            "/tmp/probe.json",
            "--jax-runtime-seed-spec",
            "/tmp/single_stage_jax_runtime_spec.json",
            "--no-reuse-jax-runtime-seed-solve",
            "--outer-maxls",
            "8",
        ],
    )

    args = parse_args()

    assert args.reuse_jax_runtime_seed_solve is False
    assert args.outer_maxls == 8


def test_outer_loop_probe_phase0_noise_gate_compares_replay_trace(
    monkeypatch,
    tmp_path,
):
    import benchmarks.single_stage_outer_loop_probe as probe
    from benchmarks.validation_ladder_common import load_json, write_json

    baseline_progress_json = tmp_path / "baseline_outer_optimizer_progress.json"
    tightened_progress_json = tmp_path / "tightened_outer_optimizer_progress.json"
    output_json = tmp_path / "probe.json"
    candidate = [0.5, -1.0, 0.25]
    write_json(
        baseline_progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 0.18000000010, "finite": True},
                    "optimizer_gradient": {
                        "values": [0.1, -0.2, 0.3],
                        "all_finite": True,
                        "inf_norm": 0.3,
                    },
                    "candidate_optimizer_dofs": {"values": candidate},
                    "boozer_solver_metadata": {"newton_tol": 1.0e-11},
                }
            ]
        },
    )
    write_json(
        tightened_progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 0.18000000001, "finite": True},
                    "optimizer_gradient": {
                        "values": [0.1, -0.2, 0.3],
                        "all_finite": True,
                        "inf_norm": 0.3,
                    },
                    "candidate_optimizer_dofs": {"values": candidate},
                    "boozer_solver_metadata": {"newton_tol": 1.0e-13},
                }
            ]
        },
    )
    captured: dict[str, Path | None] = {}

    def fake_run_single_stage_case(
        args,
        backend,
        *,
        platform,
        benchmark_mode,
        load_surface_gamma,
        profile_target_lane,
        profile_target_lane_only,
        diagnose_target_lane_scaled_phase1,
        record_target_lane_invalid_state_events,
        experimental_target_lane_value_and_grad,
        enable_compile_diagnostics,
        deterministic_gpu_reductions,
        jax_runtime_seed_spec,
        replay_objective_evaluation_trace,
        output_root,
    ):
        captured["replay_objective_evaluation_trace"] = (
            replay_objective_evaluation_trace
        )
        resolved_boozer_backend = probe.resolve_boozer_optimizer_backend(
            args.optimizer_backend,
            args.boozer_optimizer_backend,
        )
        resolved_boozer_algorithm = probe.resolve_boozer_least_squares_algorithm(
            resolved_boozer_backend,
            args.boozer_least_squares_algorithm,
        )
        return {
            "results": {
                "iterations": 0,
                "boozer_optimizer_backend": resolved_boozer_backend,
                "boozer_optimizer_method": probe.resolve_boozer_optimizer_method(
                    resolved_boozer_backend,
                    least_squares_algorithm=resolved_boozer_algorithm,
                ),
                "outer_optimizer_method": probe._expected_outer_optimizer_method(
                    args.optimizer_backend
                ),
                "SELF_INTERSECTING": False,
                "SELF_INTERSECTION_CHECK_AVAILABLE": True,
                "FINAL_IOTA": 0.15,
                "FINAL_VOLUME": 0.45,
                "FIELD_ERROR": 1.0e-4,
                "MAX_CURVATURE": 2.0,
                "INITIAL_OBJECTIVE": 0.18000000010,
                "FINAL_OBJECTIVE": 0.18000000001,
            },
            "outer_optimizer_progress_json": str(tightened_progress_json),
            "elapsed_s": 0.0,
            "phase_timings": {},
        }

    monkeypatch.setattr(probe, "bootstrap_local_simsopt", lambda: None)
    monkeypatch.setattr(probe, "print_provenance", lambda provenance: None)
    monkeypatch.setattr(probe, "_run_single_stage_case", fake_run_single_stage_case)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_outer_loop_probe.py",
            "--output-json",
            str(output_json),
            "--enable-phase0-noise-calibration-gate",
            "--phase0-noise-baseline-progress-json",
            str(baseline_progress_json),
            "--target-lane-boozer-newton-tol",
            "1e-13",
        ],
    )

    probe.main()
    payload = load_json(output_json)

    assert captured["replay_objective_evaluation_trace"] == baseline_progress_json
    assert payload["status"] == "replay-measurement-passed"
    assert payload["measurement_passed"] is True
    assert payload["convergence_passed"] is False
    assert payload["passed"] is False
    assert payload["replay_measurement_run"] is True
    assert payload["probe"]["iterations"] == 0
    assert payload["phase0_noise_calibration"]["passed"] is True
    assert payload["phase0_noise_calibration"]["candidate_dofs_match"] is True
    assert payload["phase0_noise_calibration"]["objective_abs_delta"] == pytest.approx(
        9.0e-11
    )
    assert (
        payload["phase0_line_search_trace"]["classification"] == "finite_flat_plateau"
    )


def test_target_lane_replay_trace_event_records_boozer_solver_metadata():
    from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
        build_single_stage_target_lane_objective_evaluation_trace_event,
    )

    event = build_single_stage_target_lane_objective_evaluation_trace_event(
        source_event={
            "accepted_iteration_target": 0,
            "line_search_evaluation": 1,
            "accepted_iterations": 0,
        },
        candidate_optimizer_dofs=np.array([0.5, -1.0, 0.25], dtype=np.float64),
        objective_value=0.18000000001,
        optimizer_gradient=np.array([0.1, -0.2, 0.3], dtype=np.float64),
        forward_result={
            "success": True,
            "primal_success": True,
            "iota": 0.15,
            "G": 1.0,
            "sdofs": np.array([0.0, 1.0], dtype=np.float64),
        },
        boozer_solver_metadata={
            "boozer_optimizer_backend": "ondevice",
            "newton_tol": 1.0e-13,
            "newton_maxiter": 80,
        },
    )

    assert event["target_native_replay"] is True
    assert event["boozer_solver_metadata"]["boozer_optimizer_backend"] == "ondevice"
    assert event["boozer_solver_metadata"]["newton_tol"] == pytest.approx(1.0e-13)
    assert event["boozer_solver_metadata"]["newton_maxiter"] == 80
    assert event["candidate_optimizer_dofs"]["values"] == [0.5, -1.0, 0.25]


def test_outer_loop_probe_failure_json_preserves_partial_phase0_trace(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        write_outer_loop_case_execution_failure_json,
    )
    from benchmarks.validation_ladder_common import write_json

    output_json = tmp_path / "probe.json"
    run_dir = tmp_path / "case_outputs" / "mpol=10-ntor=10-deadbeef"
    run_dir.mkdir(parents=True)
    progress_json = run_dir / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "current_event": "boozer_init_started",
            "event_count": 2,
            "events": [
                {"label": "pre_optimizer_startup_ready"},
                {"label": "boozer_init_started"},
            ],
        },
    )

    write_outer_loop_case_execution_failure_json(
        output_json,
        provenance={"platform_request": "cpu"},
        case_output_root=tmp_path / "case_outputs",
        error=RuntimeError("Subprocess timed out after 60.000s"),
    )

    payload = dict(load_json(output_json))

    assert payload["status"] == "case-execution-failed"
    assert payload["passed"] is False
    assert payload["artifacts"]["outer_optimizer_progress_json"] == [str(progress_json)]
    assert (
        payload["phase0_line_search_trace"]["classification"]
        == "missing_objective_evaluations"
    )
    assert any(
        "Progress trace contains no objective_evaluation events" in failure
        for failure in payload["failures"]
    )


def test_host_jax_memory_gate_skips_warmup_and_checks_growth(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import evaluate_host_jax_memory_gate
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "memory_snapshot": {"rss_mb": 100.0, "gpu_memory_mb": 1000.0},
                },
                {
                    "label": "objective_evaluation",
                    "memory_snapshot": {"rss_mb": 120.0, "gpu_memory_mb": 1100.0},
                },
                {
                    "label": "objective_evaluation",
                    "memory_snapshot": {"rss_mb": 120.5, "gpu_memory_mb": 1100.5},
                },
            ]
        },
    )

    summary, failures = evaluate_host_jax_memory_gate(
        progress_json=progress_json,
        warmup_evaluations=1,
        max_steady_rss_growth_mb=1.0,
        max_steady_gpu_growth_mb=1.0,
        require_gpu_memory=True,
    )

    assert failures == []
    assert summary["steady_snapshot_count"] == 2
    assert summary["peak_steady_rss_growth_mb"] == pytest.approx(0.5)
    assert summary["peak_steady_gpu_memory_growth_mb"] == pytest.approx(0.5)


def test_host_jax_memory_gate_rejects_transient_post_warm_spike(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import evaluate_host_jax_memory_gate
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "memory_snapshot": {"rss_mb": 100.0, "gpu_memory_mb": 1000.0},
                },
                {
                    "label": "objective_evaluation",
                    "memory_snapshot": {"rss_mb": 120.0, "gpu_memory_mb": 1100.0},
                },
                {
                    "label": "objective_evaluation",
                    "memory_snapshot": {"rss_mb": 220.0, "gpu_memory_mb": 1300.0},
                },
                {
                    "label": "objective_evaluation",
                    "memory_snapshot": {"rss_mb": 120.5, "gpu_memory_mb": 1100.5},
                },
            ]
        },
    )

    summary, failures = evaluate_host_jax_memory_gate(
        progress_json=progress_json,
        warmup_evaluations=1,
        max_steady_rss_growth_mb=64.0,
        max_steady_gpu_growth_mb=64.0,
        require_gpu_memory=True,
    )

    assert summary["peak_steady_rss_growth_mb"] == pytest.approx(100.0)
    assert summary["peak_steady_gpu_memory_growth_mb"] == pytest.approx(200.0)
    assert any("peak RSS growth" in failure for failure in failures)
    assert any("peak GPU memory growth" in failure for failure in failures)


def test_host_jax_memory_gate_reports_missing_progress_json(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import evaluate_host_jax_memory_gate

    summary, failures = evaluate_host_jax_memory_gate(
        progress_json=tmp_path / "missing_outer_optimizer_progress.json",
        warmup_evaluations=1,
        max_steady_rss_growth_mb=64.0,
        max_steady_gpu_growth_mb=64.0,
        require_gpu_memory=True,
    )

    assert not summary["passed"]
    assert summary["event_count"] == 0
    assert any("progress trace" in failure for failure in failures)


def test_host_jax_compile_gate_rejects_post_warm_counter_growth(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import evaluate_host_jax_compile_gate
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "compile_diagnostics_snapshot": {
                        "compile_event_count": 3,
                        "cache_miss_count": 3,
                    },
                },
                {
                    "label": "objective_evaluation",
                    "compile_diagnostics_snapshot": {
                        "compile_event_count": 5,
                        "cache_miss_count": 5,
                    },
                },
                {
                    "label": "objective_evaluation",
                    "compile_diagnostics_snapshot": {
                        "compile_event_count": 6,
                        "cache_miss_count": 5,
                    },
                },
            ]
        },
    )

    summary, failures = evaluate_host_jax_compile_gate(
        progress_json=progress_json,
        warmup_evaluations=1,
    )

    assert summary["steady_compile_event_count_growth"] == 1
    assert summary["steady_cache_miss_count_growth"] == 0
    assert any("compile event growth" in failure for failure in failures)


def test_host_jax_compile_gate_reports_missing_progress_json(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import evaluate_host_jax_compile_gate

    summary, failures = evaluate_host_jax_compile_gate(
        progress_json=tmp_path / "missing_outer_optimizer_progress.json",
        warmup_evaluations=1,
    )

    assert not summary["passed"]
    assert summary["event_count"] == 0
    assert any("progress trace" in failure for failure in failures)


def test_host_jax_compile_gate_accepts_steady_post_warm_counters(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import evaluate_host_jax_compile_gate
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "compile_diagnostics_snapshot": {
                        "compile_event_count": 3,
                        "cache_miss_count": 3,
                    },
                },
                {
                    "label": "objective_evaluation",
                    "compile_diagnostics_snapshot": {
                        "compile_event_count": 5,
                        "cache_miss_count": 5,
                    },
                },
                {
                    "label": "objective_evaluation",
                    "compile_diagnostics_snapshot": {
                        "compile_event_count": 5,
                        "cache_miss_count": 5,
                    },
                },
            ]
        },
    )

    summary, failures = evaluate_host_jax_compile_gate(
        progress_json=progress_json,
        warmup_evaluations=1,
    )

    assert failures == []
    assert summary["passed"] is True
    assert summary["steady_compile_event_count_growth"] == 0
    assert summary["steady_cache_miss_count_growth"] == 0


def test_phase0_line_search_trace_classifies_finite_plateau_failure(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase0_line_search_trace,
    )
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "event_count": 4,
            "events": [
                {
                    "label": "objective_evaluation",
                    "event_index": 0,
                    "objective": {"value": 1.0, "finite": True},
                    "optimizer_gradient": {"all_finite": True, "inf_norm": 3.0},
                },
                {
                    "label": "objective_evaluation",
                    "event_index": 1,
                    "objective": {"value": 1.0 + 2.0e-12, "finite": True},
                    "optimizer_gradient": {"all_finite": True, "inf_norm": 2.9},
                },
                {
                    "label": "objective_evaluation",
                    "event_index": 2,
                    "objective": {"value": 1.0 - 3.0e-12, "finite": True},
                    "optimizer_gradient": {"all_finite": True, "inf_norm": 2.8},
                },
                {
                    "label": "phase2_returned",
                    "event_index": 3,
                    "result": {
                        "message": "ABNORMAL_TERMINATION_IN_LNSRCH",
                        "status": 2,
                        "ls_status": -9,
                        "nfev": 21,
                        "success": False,
                    },
                },
            ],
        },
    )

    summary, failures = evaluate_phase0_line_search_trace(progress_json=progress_json)

    assert failures == []
    assert summary["classification"] == "finite_plateau_line_search_failure"
    assert summary["line_search_failure_evidence"] is True
    assert summary["termination_evidence"]["message"] == (
        "ABNORMAL_TERMINATION_IN_LNSRCH"
    )
    assert summary["termination_evidence"]["nfev"] == 21
    assert summary["nonfinite_event_indices"] == []


def test_phase0_line_search_trace_does_not_treat_status_as_proof(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase0_line_search_trace,
    )
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 1.0, "finite": True},
                    "optimizer_gradient": {"all_finite": True, "inf_norm": 3.0},
                },
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 1.0 + 2.0e-12, "finite": True},
                    "optimizer_gradient": {"all_finite": True, "inf_norm": 2.9},
                },
                {
                    "label": "phase2_returned",
                    "result": {
                        "message": "",
                        "status": 2,
                        "ls_status": -9,
                        "nfev": 21,
                        "success": False,
                    },
                },
            ],
        },
    )

    summary, failures = evaluate_phase0_line_search_trace(progress_json=progress_json)

    assert failures == []
    assert summary["line_search_failure_evidence"] is False
    assert summary["classification"] == "finite_flat_plateau"
    assert summary["termination_evidence"]["status"] == 2


def test_phase0_line_search_trace_preserves_task_evidence(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase0_line_search_trace,
    )
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 1.0, "finite": True},
                    "optimizer_gradient": {"all_finite": True, "inf_norm": 3.0},
                },
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 1.0 + 1.0e-12, "finite": True},
                    "optimizer_gradient": {"all_finite": True, "inf_norm": 2.9},
                },
                {
                    "label": "phase2_returned",
                    "result": {
                        "message": "",
                        "status": 2,
                        "task": "ABNORMAL_TERMINATION_IN_LNSRCH",
                    },
                },
            ],
        },
    )

    summary, failures = evaluate_phase0_line_search_trace(progress_json=progress_json)

    assert failures == []
    assert summary["classification"] == "finite_plateau_line_search_failure"
    assert summary["line_search_failure_evidence"] is True
    assert summary["termination_evidence"]["task"] == "ABNORMAL_TERMINATION_IN_LNSRCH"


def test_phase0_line_search_trace_classifies_nonfinite_event(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase0_line_search_trace,
    )
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "event_index": 0,
                    "objective": {"value": 1.0, "finite": True},
                    "optimizer_gradient": {"all_finite": True, "inf_norm": 3.0},
                },
                {
                    "label": "objective_evaluation",
                    "event_index": 1,
                    "objective": {"value": None, "finite": False},
                    "optimizer_gradient": {"all_finite": False, "inf_norm": None},
                },
            ],
        },
    )

    summary, failures = evaluate_phase0_line_search_trace(progress_json=progress_json)

    assert failures == [
        "Progress trace contains nonfinite objective or optimizer gradient events."
    ]
    assert summary["classification"] == "nonfinite_objective_or_gradient"
    assert summary["passed"] is False
    assert summary["nonfinite_event_indices"] == [1]


def test_phase3_gradient_trace_accepts_point_dependent_accepted_step(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase3_accepted_step_gradient_trace,
    )
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "event_count": 3,
            "events": [
                {
                    "label": "objective_evaluation",
                    "event_index": 0,
                    "accepted_iteration_target": 0,
                    "line_search_evaluation": 1,
                    "native_gradient_used": True,
                    "candidate_optimizer_dofs": {"values": [0.0, 0.0]},
                    "objective": {"value": 2.0, "finite": True},
                    "optimizer_gradient": {
                        "values": [1.0, -0.25],
                        "all_finite": True,
                        "inf_norm": 1.0,
                    },
                    "hardware_status": {
                        "threshold_margins": {
                            "curve_curve_min_dist": 1.0,
                            "curve_surface_min_dist": 1.0,
                            "surface_vessel_min_dist": 1.0,
                            "max_curvature": 1.0,
                        }
                    },
                },
                {
                    "label": "objective_evaluation",
                    "event_index": 1,
                    "accepted_iteration_target": 1,
                    "line_search_evaluation": 1,
                    "native_gradient_used": True,
                    "candidate_optimizer_dofs": {"values": [0.2, -0.1]},
                    "objective": {"value": 1.5, "finite": True},
                    "optimizer_gradient": {
                        "values": [0.5, -0.125],
                        "all_finite": True,
                        "inf_norm": 0.5,
                    },
                    "hardware_status": {
                        "threshold_margins": {
                            "curve_curve_min_dist": 2.0e-7,
                            "curve_surface_min_dist": 0.5,
                            "surface_vessel_min_dist": 0.5,
                            "max_curvature": 0.5,
                        }
                    },
                },
                {"label": "phase2_returned", "result": {"status": 0, "success": True}},
            ],
        },
    )

    summary, failures = evaluate_phase3_accepted_step_gradient_trace(
        progress_json=progress_json,
        constraint_margin_abs_tol=1.0e-6,
    )

    assert failures == []
    assert summary["passed"] is True
    assert summary["accepted_step_group_count"] == 2
    assert summary["replay_grade_group_count"] == 2
    assert summary["point_dependent_gradient_evidence"] is True
    assert summary["constraint_marginal_evidence"] is True
    assert summary["constraint_marginal_point_dependent_gradient_evidence"] is True
    assert summary["comparisons"][0]["candidate_moved_from_baseline"] is True
    assert summary["comparisons"][0]["gradient_differs_from_baseline"] is True
    assert summary["comparisons"][0]["hardware_margin_abs_min"] == pytest.approx(2.0e-7)


def test_phase3_gradient_trace_rejects_split_margin_and_gradient_evidence(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase3_accepted_step_gradient_trace,
    )
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "accepted_iteration_target": 0,
                    "native_gradient_used": True,
                    "candidate_optimizer_dofs": {"values": [0.0, 0.0]},
                    "objective": {"value": 2.0, "finite": True},
                    "optimizer_gradient": {
                        "values": [1.0, -0.25],
                        "all_finite": True,
                        "inf_norm": 1.0,
                    },
                },
                {
                    "label": "objective_evaluation",
                    "accepted_iteration_target": 1,
                    "native_gradient_used": True,
                    "candidate_optimizer_dofs": {"values": [0.2, -0.1]},
                    "objective": {"value": 1.5, "finite": True},
                    "optimizer_gradient": {
                        "values": [0.5, -0.125],
                        "all_finite": True,
                        "inf_norm": 0.5,
                    },
                    "hardware_status": {
                        "threshold_margins": {
                            "curve_curve_min_dist": 1.0,
                            "curve_surface_min_dist": 1.0,
                            "surface_vessel_min_dist": 1.0,
                            "max_curvature": 1.0,
                        }
                    },
                },
                {
                    "label": "objective_evaluation",
                    "accepted_iteration_target": 2,
                    "native_gradient_used": True,
                    "candidate_optimizer_dofs": {"values": [0.25, -0.15]},
                    "objective": {"value": 1.25, "finite": True},
                    "optimizer_gradient": {
                        "values": [1.0, -0.25],
                        "all_finite": True,
                        "inf_norm": 1.0,
                    },
                    "hardware_status": {
                        "threshold_margins": {
                            "curve_curve_min_dist": 2.0e-7,
                            "curve_surface_min_dist": 0.5,
                            "surface_vessel_min_dist": 0.5,
                            "max_curvature": 0.5,
                        }
                    },
                },
            ],
        },
    )

    summary, failures = evaluate_phase3_accepted_step_gradient_trace(
        progress_json=progress_json,
        constraint_margin_abs_tol=1.0e-6,
    )

    assert summary["passed"] is False
    assert summary["point_dependent_gradient_evidence"] is True
    assert summary["constraint_marginal_evidence"] is True
    assert summary["constraint_marginal_point_dependent_gradient_evidence"] is False
    assert any(
        "point-dependent accepted gradient at the requested hardware constraint margin"
        in failure
        for failure in failures
    )


def test_phase3_gradient_trace_rejects_frozen_accepted_step_gradient(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase3_accepted_step_gradient_trace,
    )
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "accepted_iteration_target": 0,
                    "native_gradient_used": True,
                    "candidate_optimizer_dofs": {"values": [0.0, 0.0]},
                    "objective": {"value": 2.0, "finite": True},
                    "optimizer_gradient": {
                        "values": [1.0, -0.25],
                        "all_finite": True,
                        "inf_norm": 1.0,
                    },
                },
                {
                    "label": "objective_evaluation",
                    "accepted_iteration_target": 1,
                    "native_gradient_used": True,
                    "candidate_optimizer_dofs": {"values": [0.2, -0.1]},
                    "objective": {"value": 1.5, "finite": True},
                    "optimizer_gradient": {
                        "values": [1.0, -0.25],
                        "all_finite": True,
                        "inf_norm": 1.0,
                    },
                },
            ],
        },
    )

    summary, failures = evaluate_phase3_accepted_step_gradient_trace(
        progress_json=progress_json
    )

    assert summary["passed"] is False
    assert summary["point_dependent_gradient_evidence"] is False
    assert summary["comparisons"][0]["candidate_moved_from_baseline"] is True
    assert summary["comparisons"][0]["gradient_differs_from_baseline"] is False
    assert any("frozen baseline gradient" in failure for failure in failures)


def test_phase3_gradient_trace_requires_replay_grade_accepted_step_values(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase3_accepted_step_gradient_trace,
    )
    from benchmarks.validation_ladder_common import write_json

    progress_json = tmp_path / "outer_optimizer_progress.json"
    write_json(
        progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "accepted_iteration_target": 0,
                    "native_gradient_used": True,
                    "objective": {"value": 2.0, "finite": True},
                    "optimizer_gradient": {
                        "inf_norm": 1.0,
                        "all_finite": True,
                        "values_omitted": True,
                    },
                },
                {
                    "label": "objective_evaluation",
                    "accepted_iteration_target": 1,
                    "native_gradient_used": True,
                    "objective": {"value": 1.5, "finite": True},
                    "optimizer_gradient": {
                        "inf_norm": 0.5,
                        "all_finite": True,
                        "values_omitted": True,
                    },
                },
            ],
        },
    )

    summary, failures = evaluate_phase3_accepted_step_gradient_trace(
        progress_json=progress_json
    )

    assert summary["passed"] is False
    assert summary["replay_grade_group_count"] == 0
    assert summary["missing_replay_grade_targets"] == [0, 1]
    assert any("replay-grade accepted-step groups" in failure for failure in failures)


def test_phase0_noise_calibration_requires_fixed_candidate_trace(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase0_noise_calibration_pair,
    )
    from benchmarks.validation_ladder_common import write_json

    baseline_progress_json = tmp_path / "baseline_outer_optimizer_progress.json"
    tightened_progress_json = tmp_path / "tight_outer_optimizer_progress.json"
    candidate = [0.5, -1.0, 0.25]
    write_json(
        baseline_progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 0.18000000010, "finite": True},
                    "candidate_optimizer_dofs": {"values": candidate},
                    "boozer_solver_metadata": {"newton_tol": 1.0e-11},
                }
            ]
        },
    )
    write_json(
        tightened_progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 0.18000000001, "finite": True},
                    "candidate_optimizer_dofs": {"values": candidate},
                    "boozer_solver_metadata": {"newton_tol": 1.0e-13},
                }
            ]
        },
    )

    summary, failures = evaluate_phase0_noise_calibration_pair(
        baseline_progress_json=baseline_progress_json,
        tightened_progress_json=tightened_progress_json,
        baseline_newton_tol=1.0e-11,
        tightened_newton_tol=1.0e-13,
    )

    assert failures == []
    assert summary["candidate_dofs_match"] is True
    assert summary["objective_abs_delta"] == pytest.approx(9.0e-11)
    assert summary["baseline_newton_tol"] == pytest.approx(1.0e-11)
    assert summary["tightened_newton_tol"] == pytest.approx(1.0e-13)
    assert summary["baseline_recorded_newton_tol"] == pytest.approx(1.0e-11)
    assert summary["tightened_recorded_newton_tol"] == pytest.approx(1.0e-13)
    assert summary["baseline_newton_tol_matches"] is True
    assert summary["tightened_newton_tol_matches"] is True


def test_phase0_noise_calibration_rejects_candidate_mismatch(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase0_noise_calibration_pair,
    )
    from benchmarks.validation_ladder_common import write_json

    baseline_progress_json = tmp_path / "baseline_outer_optimizer_progress.json"
    tightened_progress_json = tmp_path / "tight_outer_optimizer_progress.json"
    write_json(
        baseline_progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 0.18, "finite": True},
                    "candidate_optimizer_dofs": {"values": [0.5, -1.0, 0.25]},
                    "boozer_solver_metadata": {"newton_tol": 1.0e-11},
                }
            ]
        },
    )
    write_json(
        tightened_progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 0.17, "finite": True},
                    "candidate_optimizer_dofs": {"values": [0.5, -1.0, 0.30]},
                    "boozer_solver_metadata": {"newton_tol": 1.0e-13},
                }
            ]
        },
    )

    summary, failures = evaluate_phase0_noise_calibration_pair(
        baseline_progress_json=baseline_progress_json,
        tightened_progress_json=tightened_progress_json,
        baseline_newton_tol=1.0e-11,
        tightened_newton_tol=1.0e-13,
    )

    assert summary["passed"] is False
    assert summary["candidate_dofs_match"] is False
    assert any("candidate DOFs fixed" in failure for failure in failures)


def test_phase0_noise_calibration_rejects_missing_candidate_values(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase0_noise_calibration_pair,
    )
    from benchmarks.validation_ladder_common import write_json

    baseline_progress_json = tmp_path / "baseline_outer_optimizer_progress.json"
    tightened_progress_json = tmp_path / "tight_outer_optimizer_progress.json"
    for path, objective_value, newton_tol in (
        (baseline_progress_json, 0.18000000010, 1.0e-11),
        (tightened_progress_json, 0.18000000001, 1.0e-13),
    ):
        write_json(
            path,
            {
                "events": [
                    {
                        "label": "objective_evaluation",
                        "objective": {"value": objective_value, "finite": True},
                        "candidate_optimizer_dofs": {
                            "values_omitted": True,
                            "values_length": 3,
                        },
                        "boozer_solver_metadata": {"newton_tol": newton_tol},
                    }
                ]
            },
        )

    summary, failures = evaluate_phase0_noise_calibration_pair(
        baseline_progress_json=baseline_progress_json,
        tightened_progress_json=tightened_progress_json,
        baseline_newton_tol=1.0e-11,
        tightened_newton_tol=1.0e-13,
    )

    assert summary["passed"] is False
    assert summary["candidate_dofs_match"] is None
    assert any("replay-grade candidate values" in failure for failure in failures)


def test_phase0_noise_calibration_rejects_newton_tol_mismatch(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase0_noise_calibration_pair,
    )
    from benchmarks.validation_ladder_common import write_json

    baseline_progress_json = tmp_path / "baseline_outer_optimizer_progress.json"
    tightened_progress_json = tmp_path / "tight_outer_optimizer_progress.json"
    candidate = [0.5, -1.0, 0.25]
    write_json(
        baseline_progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 0.18000000010, "finite": True},
                    "candidate_optimizer_dofs": {"values": candidate},
                    "boozer_solver_metadata": {"newton_tol": 1.0e-10},
                }
            ]
        },
    )
    write_json(
        tightened_progress_json,
        {
            "events": [
                {
                    "label": "objective_evaluation",
                    "objective": {"value": 0.18000000001, "finite": True},
                    "candidate_optimizer_dofs": {"values": candidate},
                    "boozer_solver_metadata": {"newton_tol": 1.0e-13},
                }
            ]
        },
    )

    summary, failures = evaluate_phase0_noise_calibration_pair(
        baseline_progress_json=baseline_progress_json,
        tightened_progress_json=tightened_progress_json,
        baseline_newton_tol=1.0e-11,
        tightened_newton_tol=1.0e-13,
    )

    assert summary["passed"] is False
    assert summary["baseline_newton_tol_matches"] is False
    assert any("Baseline trace newton_tol" in failure for failure in failures)


def test_phase0_noise_calibration_rejects_negative_event_index(tmp_path):
    from benchmarks.single_stage_outer_loop_probe import (
        evaluate_phase0_noise_calibration_pair,
    )
    from benchmarks.validation_ladder_common import write_json

    baseline_progress_json = tmp_path / "baseline_outer_optimizer_progress.json"
    tightened_progress_json = tmp_path / "tight_outer_optimizer_progress.json"
    candidate = [0.5, -1.0, 0.25]
    for path, objective_value, newton_tol in (
        (baseline_progress_json, 0.18000000010, 1.0e-11),
        (tightened_progress_json, 0.18000000001, 1.0e-13),
    ):
        write_json(
            path,
            {
                "events": [
                    {
                        "label": "objective_evaluation",
                        "objective": {"value": objective_value, "finite": True},
                        "candidate_optimizer_dofs": {"values": candidate},
                        "boozer_solver_metadata": {"newton_tol": newton_tol},
                    }
                ]
            },
        )

    summary, failures = evaluate_phase0_noise_calibration_pair(
        baseline_progress_json=baseline_progress_json,
        tightened_progress_json=tightened_progress_json,
        baseline_newton_tol=1.0e-11,
        tightened_newton_tol=1.0e-13,
        objective_evaluation_index=-1,
    )

    assert summary["passed"] is False
    assert summary["objective_evaluation_index"] == -1
    assert any("must be non-negative" in failure for failure in failures)


# Audit #23: ``TestSingleStageOuterLoopCompileSmoke`` moved to
# ``tests/test_jax_compile_diagnostics.py`` as
# ``TestJaxCompileDiagnosticParser``. That test verifies parser
# invariants of the ``JAX_COMPILE_DIAGNOSTICS`` recorder; it is an
# instrumentation/bookkeeping test, not physics parity, and so does not
# belong in a file named ``*_physics_parity.py``. The helper
# ``_run_single_stage_script_results`` defined above is intentionally
# kept here as the single source of truth — the new file imports it
# rather than duplicating ~200 lines of subprocess plumbing.


class TestDeferredSurfaceNativeDofLineage:
    """Pin the native Optimizable dof-lineage contract of the deferred surface.

    The full-state target lane (ondevice/optax/optimistix with the penalty
    constraint method) consumes CPU-order ``JF.x``, whose getter reads
    ``opt._dofs.free_x`` and whose setter writes ``opt.local_x`` on every
    dof-lineage member, including the deferred surface proxy. Production job
    54335305 crashed with ``AttributeError: 'jaxlib._jax.ArrayImpl' object
    has no attribute 'free_x'`` because the proxy's runtime dof attribute
    shadowed that delegation.
    """

    @staticmethod
    def _build_deferred_surface():
        from examples.single_stage_optimization.SINGLE_STAGE import (
            single_stage_banana_example as single_stage_example,
        )
        from simsopt.geo import SurfaceXYZTensorFourier

        quadpoints_phi = np.linspace(0.0, 1.0, 8, endpoint=False)
        quadpoints_theta = np.linspace(0.0, 1.0, 8, endpoint=False)
        template = SurfaceXYZTensorFourier(
            mpol=1,
            ntor=1,
            nfp=2,
            stellsym=True,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        return single_stage_example.DeferredSurfaceXYZTensorFourier(
            mpol=1,
            ntor=1,
            nfp=2,
            stellsym=True,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            dofs=np.asarray(template.get_dofs(), dtype=np.float64),
        )

    def test_native_lineage_getter_reads_materialized_dofs(self):
        deferred = self._build_deferred_surface()
        free_x = deferred._dofs.free_x
        np.testing.assert_allclose(
            np.asarray(free_x, dtype=np.float64),
            np.asarray(deferred.local_x, dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_native_lineage_round_trip_through_composite_x(self):
        from simsopt._core.optimizable import Optimizable

        deferred = self._build_deferred_surface()

        class _LineageConsumer(Optimizable):
            def __init__(self, surface):
                super().__init__(depends_on=[surface])

            def J(self):
                return 0.0

        consumer = _LineageConsumer(deferred)
        baseline = np.asarray(consumer.x, dtype=np.float64).copy()
        assert baseline.size > 0

        perturbed = baseline + 0.125
        consumer.x = perturbed
        np.testing.assert_allclose(
            np.asarray(consumer.x, dtype=np.float64),
            perturbed,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(deferred.get_dofs(), dtype=np.float64),
            np.asarray(deferred._materialize_surface().get_dofs(), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_native_lineage_full_x_round_trip(self):
        from simsopt._core.optimizable import Optimizable

        deferred = self._build_deferred_surface()

        class _LineageConsumer(Optimizable):
            def __init__(self, surface):
                super().__init__(depends_on=[surface])

            def J(self):
                return 0.0

        consumer = _LineageConsumer(deferred)
        baseline = np.asarray(consumer.full_x, dtype=np.float64).copy()
        assert baseline.size > 0

        perturbed = baseline - 0.0625
        consumer.full_x = perturbed
        np.testing.assert_allclose(
            np.asarray(consumer.full_x, dtype=np.float64),
            perturbed,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(deferred.get_dofs(), dtype=np.float64),
            np.asarray(deferred._materialize_surface().get_dofs(), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )


def test_target_lane_optimizer_seed_contract_matches_target_minimize_guard():
    """supports-seed contracts must resolve to the one seed-capable method.

    target_minimize() only consumes initial_value_and_grad with
    method='lbfgs-ondevice' (the ondevice private L-BFGS); every other
    target route rejects it. Tier 5's outer-loop probe crashed because the
    example's seed-support contract also claimed the scipy-jax route.
    """
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    seed_supported_methods = set()
    for optimizer_backend in (
        "ondevice",
        "scipy-jax",
        "scipy-jax-decomposed",
        "host-jax",
        "scipy-jax-fullgraph",
        "optax-lbfgs",
        "optimistix-lbfgs",
    ):
        contract = single_stage_example.resolve_single_stage_optimizer_contract(
            "jax", optimizer_backend
        )
        if single_stage_example.target_lane_contract_supports_optimizer_seed(contract):
            seed_supported_methods.add(
                single_stage_example.single_stage_optimizer_contract_method(contract)
            )
    assert seed_supported_methods == {"lbfgs-ondevice"}


def test_single_stage_cli_accepts_scipy_jax_decomposed(monkeypatch):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_banana_example.py",
            "--backend",
            "jax",
            "--optimizer-backend",
            "scipy-jax-decomposed",
            "--target-lane-boozer-newton-stab",
            "1e-4",
        ],
    )

    args = single_stage_example.parse_args()

    assert args.backend == "jax"
    assert args.optimizer_backend == "scipy-jax-decomposed"
    assert args.target_lane_boozer_newton_stab == pytest.approx(1.0e-4)


def test_single_stage_cli_warns_for_deprecated_scipy_jax(monkeypatch):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_banana_example.py",
            "--backend",
            "jax",
            "--optimizer-backend",
            "scipy-jax",
        ],
    )

    with pytest.warns(FutureWarning, match="deprecated.*scipy-jax-decomposed"):
        args = single_stage_example.parse_args()

    assert args.backend == "jax"
    assert args.optimizer_backend == "scipy-jax"


def test_single_stage_cli_defaults_to_scipy_jax_decomposed_for_jax_backend(monkeypatch):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_banana_example.py",
            "--backend",
            "jax",
        ],
    )

    args = single_stage_example.parse_args()

    assert args.backend == "jax"
    assert args.optimizer_backend == "scipy-jax-decomposed"


def test_host_jax_single_stage_contract_uses_host_control():
    """host-jax must not enter the fused target-lane objective bundle."""
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    contract = single_stage_example.resolve_single_stage_optimizer_contract(
        "jax",
        "host-jax",
    )

    assert isinstance(contract, single_stage_example.ReferenceOptimizerContract)
    assert (
        single_stage_example.single_stage_optimizer_contract_method(contract) == "lbfgs"
    )
    assert not single_stage_example.single_stage_optimizer_contract_uses_array_native_target_lane(
        contract,
        constraint_method="penalty",
    )
    single_stage_example.require_single_stage_jax_target_lane(
        use_jax=True,
        use_target_lane=False,
        optimizer_method="lbfgs",
        optimizer_backend="host-jax",
    )


def test_lbfgs_ondevice_production_warning_is_route_specific():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    warning = single_stage_example.single_stage_lbfgs_ondevice_production_warning(
        optimizer_method="lbfgs-ondevice",
        optimizer_backend="ondevice",
        use_target_lane=True,
    )

    assert warning is not None
    assert "lbfgs-ondevice remains experimental" in warning
    assert (
        single_stage_example.single_stage_lbfgs_ondevice_production_warning(
            optimizer_method="lbfgs",
            optimizer_backend="host-jax",
            use_target_lane=False,
        )
        is None
    )


def _target_lane_invalid_step_entry(
    *,
    iteration=0,
    requested_initial_step=1.0,
    first_tested_alpha=1.0,
    ls_status=-9,
):
    return {
        "iteration": iteration,
        "step_scale": requested_initial_step,
        "line_search_failed": True,
        "nonfinite_step": False,
        "stalled_step": False,
        "valid_curvature": True,
        "trial_converged": False,
        "ls_status": ls_status,
        "requested_initial_step": requested_initial_step,
        "first_tested_alpha": first_tested_alpha,
        "best_finite_alpha": 0.0,
        "returned_alpha": 0.0,
        "failure_reason": "line_search_failed",
        "armijo_margin": 0.0,
        "curvature_margin": 0.0,
    }


def test_target_lane_zero_iteration_line_search_failure_anchor_is_specific():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    event = single_stage_example._build_target_lane_invalid_state_event(
        phase="phase2",
        **_target_lane_invalid_step_entry(),
    )
    anchor = single_stage_example.target_lane_zero_iteration_line_search_failure_anchor(
        event
    )

    assert anchor == {
        "iteration": 0,
        "ls_status": -9,
        "requested_initial_step": {
            "value": 1.0,
            "finite": True,
            "classification": None,
        },
        "first_tested_alpha": {"value": 1.0, "finite": True, "classification": None},
        "failure_reason": "line_search_failed",
    }

    later_event = dict(event)
    later_event["iteration"] = 1
    assert (
        single_stage_example.target_lane_zero_iteration_line_search_failure_anchor(
            later_event
        )
        is None
    )

    anchor_counts = {}
    assert (
        single_stage_example.record_repeated_target_lane_line_search_failure_anchor(
            anchor_counts,
            [event, event],
        )
        is None
    )
    trigger = (
        single_stage_example.record_repeated_target_lane_line_search_failure_anchor(
            anchor_counts,
            [event],
        )
    )
    assert trigger["reason"] == "repeated_zero_iteration_line_search_failure"
    assert trigger["repeated_count"] == 2


def test_target_lane_retry_fail_fast_stops_repeated_zero_iteration_anchor(monkeypatch):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    calls = []
    anchor_state = {
        "coil_dofs": np.asarray([1.0, 2.0], dtype=np.float64),
        "sdofs": np.asarray([0.1, 0.2], dtype=np.float64),
        "iota": 0.3,
        "G": 1.4,
        "J": 9.0,
        "dJ": np.asarray([0.5, 0.25], dtype=np.float64),
        "intersecting": False,
        "self_intersection_check_available": True,
        "hardware_constraint_status": {"success": True},
    }
    run_dict = {
        "x_prev": np.asarray([1.0, 2.0], dtype=np.float64),
        "sdofs": np.asarray([0.1, 0.2], dtype=np.float64),
        "iota": 0.3,
        "G": 1.4,
        "J": 9.0,
        "dJ": np.asarray([0.5, 0.25], dtype=np.float64),
        "intersecting": False,
        "self_intersection_check_available": True,
        "hardware_constraint_status": {"success": True},
        "latest_local_incumbent": anchor_state,
        "latest_local_stage": "initial",
        "best_local_incumbent": None,
        "best_local_stage": None,
    }

    def fake_run_single_stage_optimizer(*_args, **_kwargs):
        calls.append(np.asarray(_args[1], dtype=np.float64).copy())
        return types.SimpleNamespace(
            success=False,
            nit=0,
            nfev=9,
            njev=9,
            x=np.asarray(_args[1], dtype=np.float64).copy(),
            fun=12.0,
            jac=np.asarray([3.0, 4.0], dtype=np.float64),
            status=2,
            ls_status=-9,
            message="line search failed",
            invalid_step_log=[_target_lane_invalid_step_entry()],
        )

    events = []
    invalid_state_events = []
    monkeypatch.setattr(
        single_stage_example,
        "run_single_stage_optimizer",
        fake_run_single_stage_optimizer,
    )

    result, retry_summary = (
        single_stage_example.run_single_stage_target_lane_optimizer_with_retries(
            lambda x: (0.0, np.zeros_like(x)),
            np.asarray([4.0, 5.0], dtype=np.float64),
            phase="phase2",
            callback=None,
            retry_callback=None,
            result_state_sync=None,
            contract=object(),
            maxiter=10,
            ftol=1.0e-9,
            gtol=1.0e-9,
            maxcor=10,
            outer_maxls=20,
            scalar_fun=None,
            target_lane_initial_step_size=None,
            failure_callback=None,
            invalid_state_events=invalid_state_events,
            run_dict=run_dict,
            single_stage_search_policy=single_stage_example.SingleStageSearchPolicy(
                donor_class="stage2_seed_only",
                search_policy="repair_first",
                adaptive_failure_penalty_weight=1.0,
                invalid_step_retry_budget=4,
            ),
            progress_event_callback=lambda label, **fields: events.append(
                (label, fields)
            ),
        )
    )

    assert len(calls) == 2
    np.testing.assert_allclose(calls[0], [4.0, 5.0])
    np.testing.assert_allclose(calls[1], [1.0, 2.0])
    assert retry_summary["attempt_count"] == 1
    assert retry_summary["fail_fast_triggered"] is True
    assert retry_summary["fail_fast_reason"] == (
        "repeated_zero_iteration_line_search_failure"
    )
    assert retry_summary["fail_fast_repeated_count"] == 2
    assert result.line_search_fail_fast["repeated_count"] == 2
    assert "line_search_fail_fast=repeated_zero_iteration_line_search_failure" in (
        result.message
    )
    assert [
        event[0] for event in events if event[0].endswith("fail_fast_triggered")
    ] == ["phase2_line_search_fail_fast_triggered"]
    assert len(invalid_state_events) == 2


def test_single_stage_jax_boozer_options_preserve_host_jax_backend():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    options = {"verbose": True}

    resolved_backend = (
        single_stage_example._configure_single_stage_jax_boozer_solver_options(
            options,
            "jax",
            "ondevice",
            boozer_optimizer_backend="host-jax",
            boozer_least_squares_algorithm="lm",
            boozer_limited_memory=False,
        )
    )

    assert resolved_backend == "host-jax"
    assert options == {
        "verbose": True,
        "optimizer_backend": "host-jax",
        "limited_memory": False,
        "least_squares_algorithm": "lm",
    }


def test_host_jax_boozer_default_algorithm_is_lm():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )
    from simsopt_jax.geo.optimizers.single_stage_routing import (
        resolve_boozer_least_squares_algorithm,
    )

    assert (
        single_stage_example.resolve_single_stage_default_boozer_least_squares_algorithm(
            "jax",
            "host-jax",
            boozer_optimizer_backend="host-jax",
            boozer_least_squares_algorithm=None,
        )
        == "lm"
    )
    assert resolve_boozer_least_squares_algorithm("host-jax") == "lm"
    assert resolve_boozer_least_squares_algorithm("ondevice") == "quasi-newton"


def test_host_jax_boozer_budget_flags_drive_kernelized_init_overrides():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    assert single_stage_example.single_stage_boozer_budget_overrides_apply(
        "jax",
        "host-jax",
    )

    overrides = single_stage_example.resolve_target_lane_boozer_init_base_overrides(
        field_backend="jax",
        optimizer_backend="host-jax",
        boozer_optimizer_backend="host-jax",
        boozer_limited_memory=False,
        target_lane_boozer_bfgs_tol=None,
        target_lane_boozer_bfgs_maxiter=1500,
        target_lane_boozer_newton_tol=None,
        target_lane_boozer_newton_maxiter=50,
        target_lane_boozer_newton_stab=1.0e-4,
    )

    assert overrides["bfgs_maxiter_override"] == 1500
    assert overrides["newton_maxiter_override"] == 50
    assert overrides["newton_stab_override"] == pytest.approx(1.0e-4)
    assert overrides["newton_tol_override"] is None


def test_target_lane_newton_stab_rejects_parity_mode():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    with pytest.raises(ValueError, match="parity mode requires newton_stab=0.0"):
        single_stage_example.validate_target_lane_boozer_newton_stab_backend_mode(
            1.0e-4,
            parity_mode=True,
        )


def test_target_lane_newton_stab_allows_nonparity_and_zero_parity():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    single_stage_example.validate_target_lane_boozer_newton_stab_backend_mode(
        1.0e-4,
        parity_mode=False,
    )
    single_stage_example.validate_target_lane_boozer_newton_stab_backend_mode(
        0.0,
        parity_mode=True,
    )


def test_host_jax_single_stage_optimizer_forwards_host_control_permission(monkeypatch):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    captured_kwargs = {}

    def fake_reference_minimize(fun, dofs, **kwargs):
        captured_kwargs.update(kwargs)
        value, gradient = fun(dofs)
        return types.SimpleNamespace(
            success=True,
            nit=0,
            nfev=1,
            njev=1,
            message="ok",
            x=np.asarray(dofs, dtype=np.float64),
            fun=value,
            jac=gradient,
        )

    monkeypatch.setattr(
        single_stage_example,
        "reference_minimize",
        fake_reference_minimize,
    )

    def value_and_grad(dofs):
        dofs_array = np.asarray(dofs, dtype=np.float64)
        return float(np.sum(dofs_array * dofs_array)), 2.0 * dofs_array

    result = single_stage_example.run_single_stage_optimizer(
        value_and_grad,
        np.asarray([1.0, 2.0], dtype=np.float64),
        contract=single_stage_example.ReferenceOptimizerContract(
            driver=single_stage_example.Driver.SCIPY_LBFGSB,
        ),
        maxiter=3,
        ftol=1e-9,
        gtol=1e-9,
        maxcor=10,
        outer_maxls=20,
        callback=None,
        allow_jax_host_control=True,
    )

    assert result.success
    assert captured_kwargs["allow_jax_host_control"] is True


def test_host_jax_adapter_uses_solved_state_kernel_without_legacy_objective(tmp_path):
    """host-jax outer evaluation must not call the mutable graph value/grad."""
    import jax.numpy as jnp
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    class _Surface:
        x = np.asarray([0.1, 0.2], dtype=np.float64)

        def is_self_intersecting(self):
            return False

        def volume(self):
            return 1.25

    class _BoozerSurface:
        supports_explicit_surface_warm_start = True

        def __init__(self):
            self.surface = _Surface()
            self.run_calls = []
            self.res = {
                "success": True,
                "primal_success": True,
                "iota": jnp.asarray(0.3, dtype=jnp.float64),
                "G": jnp.asarray(1.7, dtype=jnp.float64),
            }

        def run_code(self, iota, G, *, sdofs):
            self.run_calls.append((float(iota), float(G), np.asarray(sdofs).copy()))

        def get_adjoint_runtime_state(self):
            solved_state = types.SimpleNamespace(
                sdofs=jnp.asarray([0.4, 0.5], dtype=jnp.float64),
                iota=jnp.asarray(0.6, dtype=jnp.float64),
                G=jnp.asarray(1.8, dtype=jnp.float64),
            )
            factors = (
                jnp.eye(3, dtype=jnp.float64),
                jnp.eye(3, dtype=jnp.float64),
                jnp.eye(3, dtype=jnp.float64),
            )
            return types.SimpleNamespace(
                solved_state=solved_state,
                linear_solve_factors=factors,
            )

        def _pack_decision_vector(self, iota, G, sdofs=None):
            return jnp.concatenate(
                [
                    jnp.asarray(sdofs, dtype=jnp.float64),
                    jnp.asarray([iota, G], dtype=jnp.float64),
                ]
            )

    class _ForbiddenObjective:
        def J(self):
            raise AssertionError("host-jax adapter must not call JF.J()")

        def dJ(self):
            raise AssertionError("host-jax adapter must not call JF.dJ()")

    kernel_calls = []

    def solved_state_value_and_grad(coil_dofs, solved_x, linear_solve_factors):
        kernel_calls.append(
            {
                "coil_dofs": np.asarray(coil_dofs),
                "solved_x": np.asarray(solved_x),
                "factor_count": len(linear_solve_factors),
            }
        )
        return jnp.asarray(7.0, dtype=jnp.float64), 3.0 * coil_dofs

    applied_dofs = []

    def apply_coil_dofs(x):
        applied_dofs.append(np.asarray(x, dtype=np.float64).copy())

    run_dict = {
        "sdofs": np.asarray([0.1, 0.2], dtype=np.float64),
        "iota": 0.3,
        "G": 1.7,
        "J": float("nan"),
        "dJ": np.zeros(2, dtype=np.float64),
        "initial_objective": float("nan"),
        "initial_objective_pending": True,
        "it": 1,
        "lscount": 0,
        "failure_count": 0,
        "x_prev": np.asarray([1.0, 2.0], dtype=np.float64),
        "intersecting": False,
        "self_intersection_check_available": True,
        "latest_local_incumbent": None,
        "latest_local_metric": None,
        "latest_local_stage": None,
        "best_local_incumbent": None,
        "best_local_metric": None,
        "best_local_stage": None,
    }
    adapter = single_stage_example.HostJaxSingleStageAdapter(
        run_dict=run_dict,
        boozer_surface=_BoozerSurface(),
        JF=_ForbiddenObjective(),
        bs=object(),
        objectives={},
        objective_weights={},
        diagnostics={},
        log_path=str(tmp_path / "host_jax.log"),
        apply_coil_dofs=apply_coil_dofs,
        solved_state_value_and_grad=solved_state_value_and_grad,
    )

    value, grad = adapter(np.asarray([1.0, 2.0], dtype=np.float64))

    assert value == pytest.approx(7.0)
    np.testing.assert_allclose(grad, np.asarray([3.0, 6.0], dtype=np.float64))
    assert len(kernel_calls) == 1
    np.testing.assert_allclose(kernel_calls[0]["coil_dofs"], [1.0, 2.0])
    np.testing.assert_allclose(kernel_calls[0]["solved_x"], [0.4, 0.5, 0.6, 1.8])
    assert kernel_calls[0]["factor_count"] == 3
    assert len(applied_dofs) == 1
    assert run_dict["initial_objective_pending"] is False
    assert run_dict["J"] == pytest.approx(7.0)

    adapter.callback(np.asarray([1.0, 2.0], dtype=np.float64))

    np.testing.assert_allclose(run_dict["sdofs"], [0.4, 0.5])
    assert run_dict["iota"] == pytest.approx(0.6)
    assert run_dict["G"] == pytest.approx(1.8)
    assert run_dict["lscount"] == 0
    assert run_dict["it"] == 2


def test_host_jax_adapter_builds_solved_state_kernel_after_boozer_solve(tmp_path):
    """host-jax must not capture startup value-only linearization state."""
    import jax.numpy as jnp
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    class _Surface:
        x = np.asarray([0.1, 0.2], dtype=np.float64)

        def is_self_intersecting(self):
            return False

        def volume(self):
            return 1.25

    class _BoozerSurface:
        supports_explicit_surface_warm_start = True

        def __init__(self):
            self.surface = _Surface()
            self.run_calls = []
            self.res = {
                "success": True,
                "primal_success": True,
                "iota": jnp.asarray(0.3, dtype=jnp.float64),
                "G": jnp.asarray(1.7, dtype=jnp.float64),
                "linearization_kind": "value_only",
                "adjoint_linear_solve_available": False,
            }

        def run_code(self, iota, G, *, sdofs):
            self.run_calls.append((float(iota), float(G), np.asarray(sdofs).copy()))
            self.res["linearization_kind"] = "hessian"
            self.res["adjoint_linear_solve_available"] = True

        def get_adjoint_runtime_state(self):
            assert self.run_calls, "kernel built before Boozer run_code"
            assert self.res["linearization_kind"] == "hessian"
            solved_state = types.SimpleNamespace(
                sdofs=jnp.asarray([0.4, 0.5], dtype=jnp.float64),
                iota=jnp.asarray(0.6, dtype=jnp.float64),
                G=jnp.asarray(1.8, dtype=jnp.float64),
            )
            factors = (
                jnp.eye(3, dtype=jnp.float64),
                jnp.eye(3, dtype=jnp.float64),
                jnp.eye(3, dtype=jnp.float64),
            )
            return types.SimpleNamespace(
                solved_state=solved_state,
                linear_solve_factors=factors,
            )

        def _pack_decision_vector(self, iota, G, sdofs=None):
            return jnp.concatenate(
                [
                    jnp.asarray(sdofs, dtype=jnp.float64),
                    jnp.asarray([iota, G], dtype=jnp.float64),
                ]
            )

    class _ForbiddenObjective:
        def J(self):
            raise AssertionError("host-jax adapter must not call JF.J()")

        def dJ(self):
            raise AssertionError("host-jax adapter must not call JF.dJ()")

    factory_calls = []
    kernel_calls = []

    def solved_state_value_and_grad_factory():
        factory_calls.append("built")

        def solved_state_value_and_grad(coil_dofs, solved_x, linear_solve_factors):
            kernel_calls.append(
                {
                    "coil_dofs": np.asarray(coil_dofs),
                    "solved_x": np.asarray(solved_x),
                    "factor_count": len(linear_solve_factors),
                }
            )
            return jnp.asarray(7.0, dtype=jnp.float64), 3.0 * coil_dofs

        return solved_state_value_and_grad

    run_dict = {
        "sdofs": np.asarray([0.1, 0.2], dtype=np.float64),
        "iota": 0.3,
        "G": 1.7,
        "J": float("nan"),
        "dJ": np.zeros(2, dtype=np.float64),
        "initial_objective": float("nan"),
        "initial_objective_pending": True,
        "it": 1,
        "lscount": 0,
        "failure_count": 0,
        "x_prev": np.asarray([1.0, 2.0], dtype=np.float64),
        "intersecting": False,
        "self_intersection_check_available": True,
        "latest_local_incumbent": None,
        "latest_local_metric": None,
        "latest_local_stage": None,
        "best_local_incumbent": None,
        "best_local_metric": None,
        "best_local_stage": None,
    }
    boozer_surface = _BoozerSurface()
    adapter = single_stage_example.HostJaxSingleStageAdapter(
        run_dict=run_dict,
        boozer_surface=boozer_surface,
        JF=_ForbiddenObjective(),
        bs=object(),
        objectives={},
        objective_weights={},
        diagnostics={},
        log_path=str(tmp_path / "host_jax.log"),
        solved_state_value_and_grad_factory=solved_state_value_and_grad_factory,
    )

    value, grad = adapter(np.asarray([1.0, 2.0], dtype=np.float64))
    second_value, second_grad = adapter(np.asarray([1.0, 2.0], dtype=np.float64))

    assert value == pytest.approx(7.0)
    assert second_value == pytest.approx(7.0)
    np.testing.assert_allclose(grad, np.asarray([3.0, 6.0], dtype=np.float64))
    np.testing.assert_allclose(second_grad, np.asarray([3.0, 6.0], dtype=np.float64))
    assert len(factory_calls) == 1
    assert len(boozer_surface.run_calls) == 2
    assert len(kernel_calls) == 2
    assert kernel_calls[0]["factor_count"] == 3


def test_single_stage_failure_penalty_is_finite_before_initial_objective_seed():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    run_dict = {
        "adaptive_failure_penalty_weight": 1.5,
        "J": float("nan"),
        "dJ": np.zeros(2, dtype=np.float64),
        "initial_objective": float("nan"),
        "initial_objective_pending": True,
        "failure_count": 0,
        "x_prev": np.asarray([1.0, 2.0], dtype=np.float64),
        "search_policy": "repair_first",
        "donor_class": "stage2_seed_only",
    }
    penalty, summary = (
        single_stage_example._compute_single_stage_failure_penalty_from_residual_inf(
            np.asarray([1.0, 2.0], dtype=np.float64),
            run_dict,
            success_solve=True,
            is_intersecting=False,
            hardware_status={"success": False, "violation_keys": ["curvature"]},
            residual_inf=0.0,
        )
    )

    assert np.isfinite(penalty)
    assert penalty == pytest.approx(1.75)
    failure_objective = single_stage_example._single_stage_failure_objective_value(
        run_dict,
        penalty,
    )
    assert failure_objective == pytest.approx(2.75)
    single_stage_example.seed_single_stage_initial_objective_from_values(
        run_dict,
        objective_value=failure_objective,
        objective_grad=run_dict["dJ"],
    )
    assert run_dict["initial_objective_pending"] is False
    assert run_dict["initial_objective"] == pytest.approx(2.75)
    assert summary["penalty"] == pytest.approx(penalty)
    assert summary["reject_class"] == "hardware"


def test_target_lane_initial_objective_records_host_sync_boundaries():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    run_dict = {
        "initial_objective_pending": True,
        "it": 1,
        "accepted_iterations": 0,
    }
    events = []
    trace_events = []

    def target_value_and_grad_objective(objective_dofs):
        return np.asarray(1.25, dtype=np.float64), 2.0 * np.asarray(objective_dofs)

    value, grad = (
        single_stage_example.seed_pending_single_stage_target_lane_initial_objective(
            run_dict,
            target_value_and_grad_objective=target_value_and_grad_objective,
            optimizer_dofs=np.asarray([3.0, 4.0], dtype=np.float64),
            objective_input_dofs=lambda x: np.asarray(x, dtype=np.float64),
            record_outer_optimizer_event=lambda label, **fields: events.append(
                (label, fields)
            ),
            objective_evaluation_trace_callback=trace_events.append,
        )
    )

    labels = [label for label, _fields in events]
    assert labels.index("target_lane_initial_objective_value_and_grad_returned") < (
        labels.index("target_lane_initial_objective_finite_check_started")
    )
    assert labels.index("target_lane_initial_objective_finite_check_started") < (
        labels.index("target_lane_initial_objective_finite_check_returned")
    )
    assert labels.index("target_lane_initial_objective_finite_check_returned") < (
        labels.index("target_lane_initial_objective_trace_event_started")
    )
    assert labels.index("target_lane_initial_objective_trace_event_started") < (
        labels.index("target_lane_initial_objective_trace_event_returned")
    )
    finite_fields = dict(events)["target_lane_initial_objective_finite_check_returned"]
    assert finite_fields["phase"] == "initial"
    assert finite_fields["finite"] is True
    assert len(trace_events) == 1
    assert run_dict["initial_objective_pending"] is False
    assert value == pytest.approx(1.25)
    np.testing.assert_allclose(grad, [6.0, 8.0])


def test_cpu_resolved_warm_start_install_is_value_only_and_stales_next_solve():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    class _Surface:
        def __init__(self):
            self.x = np.zeros(2, dtype=np.float64)

        def set_dofs(self, dofs):
            self.x = np.asarray(dofs, dtype=np.float64).copy()

    class _BoozerSurface:
        def __init__(self):
            self.surface = _Surface()
            self.options = {"weight_inv_modB": True}
            self.boozer_type = "ls"
            self.need_to_run_code = True
            self.res = None

    boozer_surface = _BoozerSurface()

    res = single_stage_example.install_cpu_value_only_solved_boozer_state(
        boozer_surface,
        sdofs=np.asarray([0.3, 0.4], dtype=np.float64),
        iota=0.5,
        G=1.5,
    )

    np.testing.assert_allclose(boozer_surface.surface.x, [0.3, 0.4])
    assert boozer_surface.need_to_run_code is False
    assert res["success"] is True
    assert res["linearization_kind"] == "value_only"
    assert res["adjoint_linear_solve_available"] is False
    assert "PLU" not in res
    assert "vjp" not in res

    class _IotaShouldNotRun:
        def __init__(self, boozer_surface):
            self.boozer_surface = boozer_surface

        def J(self):
            raise AssertionError("value-only donor iota should come from res")

    assert single_stage_example.resolve_single_stage_iota_metric(
        boozer_surface,
        _IotaShouldNotRun,
        benchmark_mode=False,
    ) == pytest.approx(0.5)

    single_stage_example._restore_cpu_boozer_state(
        boozer_surface,
        {
            "sdofs": np.asarray([0.7, 0.8], dtype=np.float64),
            "iota": 0.9,
            "G": 2.5,
        },
    )

    np.testing.assert_allclose(boozer_surface.surface.x, [0.7, 0.8])
    assert boozer_surface.res["iota"] == pytest.approx(0.9)
    assert boozer_surface.res["G"] == pytest.approx(2.5)
    assert boozer_surface.need_to_run_code is True


def test_target_lane_materializes_value_only_boozer_linearization():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    class _BoozerSurface:
        def __init__(self):
            self.need_to_run_code = False
            self.traceable_install_called = False
            self.res = {
                "success": True,
                "primal_success": True,
                "linearization_kind": "value_only",
                "adjoint_linear_solve_available": False,
                "sdofs": np.asarray([0.3, 0.4], dtype=np.float64),
                "iota": np.asarray(0.5, dtype=np.float64),
                "G": np.asarray(1.5, dtype=np.float64),
                "weight_inv_modB": True,
            }

        def get_solved_runtime_state(self):
            return types.SimpleNamespace(
                sdofs=np.asarray(self.res["sdofs"], dtype=np.float64),
                iota=np.asarray(self.res["iota"], dtype=np.float64),
                G=np.asarray(self.res["G"], dtype=np.float64),
            )

        def install_traceable_hessian_linearization_for_value_only_state(self):
            self.traceable_install_called = True
            self.res = {
                "success": True,
                "primal_success": True,
                "linearization_kind": "hessian",
                "adjoint_linear_solve_available": False,
                "sdofs": np.asarray([0.7, 0.8], dtype=np.float64),
                "iota": np.asarray(0.9, dtype=np.float64),
                "G": np.asarray(2.5, dtype=np.float64),
                "weight_inv_modB": True,
            }
            return self.res

        def run_code(self, iota, G, *, sdofs=None):
            raise AssertionError("traceable materialization must not rerun Boozer")

    boozer_surface = _BoozerSurface()
    run_dict = {
        "sdofs": np.asarray([0.3, 0.4], dtype=np.float64),
        "iota": 0.5,
        "G": 1.5,
        "J": 11.0,
        "dJ": np.asarray([1.0, 2.0], dtype=np.float64),
        "lscount": 9,
        "target_lane_reporting_metrics": {"stale": True},
        "target_lane_reporting_coil_dofs": np.asarray([3.0], dtype=np.float64),
        "target_lane_reporting_include_distance_metrics": True,
    }
    events = []

    materialized = single_stage_example._materialize_target_lane_boozer_linearization(
        boozer_surface,
        run_dict,
        record_outer_optimizer_event=lambda label, **fields: events.append(
            (label, fields)
        ),
    )

    assert materialized is True
    assert boozer_surface.traceable_install_called is True
    np.testing.assert_allclose(run_dict["sdofs"], [0.7, 0.8])
    assert run_dict["iota"] == pytest.approx(0.9)
    assert run_dict["G"] == pytest.approx(2.5)
    assert run_dict["J"] == pytest.approx(11.0)
    np.testing.assert_allclose(run_dict["dJ"], [1.0, 2.0])
    assert run_dict["lscount"] == 0
    assert "target_lane_reporting_metrics" not in run_dict
    assert "target_lane_reporting_coil_dofs" not in run_dict
    assert "target_lane_reporting_include_distance_metrics" not in run_dict
    assert [event[0] for event in events] == [
        "target_lane_boozer_linearization_materialization_started",
        "target_lane_boozer_linearization_materialization_returned",
    ]
    assert events[-1][1]["linearization_kind"] == "hessian"
    assert events[-1][1]["adjoint_linear_solve_available"] is False


def test_initial_target_lane_reporting_skips_pending_objective(monkeypatch):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    calls = {"objective": 0}

    def _fake_reporting_metrics_from_solution(
        coil_dofs,
        solved_state,
        success,
        *,
        include_distance_metrics,
    ):
        assert include_distance_metrics is True
        np.testing.assert_allclose(coil_dofs, [1.0, 2.0])
        np.testing.assert_allclose(solved_state, [0.1, 0.2])
        assert bool(success) is True
        return {
            "solver_success": True,
            "curve_curve_min_dist": 1.0,
            "curve_surface_min_dist": 1.0,
            "surface_vessel_min_dist": 1.0,
            "max_curvature": 1.0,
        }

    def _runtime_bundle_builder(*_args, **_kwargs):
        def _objective(_coil_dofs):
            calls["objective"] += 1
            raise AssertionError(
                "initial reporting must not evaluate pending objective"
            )

        return {
            "reporting_metrics": lambda _coil_dofs, **_kwargs: {},
            "reporting_metrics_from_solution": (_fake_reporting_metrics_from_solution),
            "objective": _objective,
            "value_and_grad": lambda _coil_dofs: (0.0, np.zeros(2)),
            "forward_result": lambda _coil_dofs: {
                "success": True,
                "finite": True,
                "sdofs": np.asarray([0.3, 0.4], dtype=np.float64),
                "iota": np.asarray(0.5, dtype=np.float64),
                "G": np.asarray(1.5, dtype=np.float64),
                "x": np.asarray([0.1, 0.2], dtype=np.float64),
            },
        }

    monkeypatch.setattr(
        single_stage_example,
        "get_traceable_single_stage_runtime_bundle_builder",
        lambda: _runtime_bundle_builder,
    )
    monkeypatch.setattr(single_stage_example, "CC_DIST", 0.05, raising=False)
    monkeypatch.setattr(single_stage_example, "CS_DIST", 0.02, raising=False)
    monkeypatch.setattr(single_stage_example, "SS_DIST", 0.04, raising=False)
    monkeypatch.setattr(
        single_stage_example,
        "CURVATURE_THRESHOLD",
        40.0,
        raising=False,
    )

    events = []
    sync = single_stage_example.build_single_stage_target_lane_accepted_step_sync(
        object(),
        object(),
        0.5,
        outer_objective_config=object(),
        success_filter=None,
        record_outer_optimizer_event=lambda label, **fields: events.append(
            (label, fields)
        ),
    )
    run_dict = {"initial_objective_pending": True}

    accepted_step_summary = sync(
        run_dict,
        np.asarray([1.0, 2.0], dtype=np.float64),
        benchmark_mode=False,
        update_run_state=False,
    )

    assert calls["objective"] == 0
    assert accepted_step_summary["objective_value"] is None
    assert run_dict["target_lane_reporting_metrics"]["solver_success"] is True
    assert run_dict["target_lane_reporting_metrics"]["max_curvature"] == pytest.approx(
        1.0
    )
    objective_events = [
        fields
        for label, fields in events
        if label == "target_lane_reporting_objective_started"
    ]
    assert objective_events == [{"reused": False, "skipped": True}]


def test_target_lane_finite_nonconverged_sync_does_not_advance_seed(monkeypatch):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    def _fake_reporting_metrics_from_solution(
        coil_dofs,
        solved_state,
        success,
        *,
        include_distance_metrics,
    ):
        np.testing.assert_allclose(coil_dofs, [1.0, 2.0])
        np.testing.assert_allclose(solved_state, [9.0, 10.0, 11.0])
        assert bool(success) is False
        assert include_distance_metrics is True
        return {
            "solver_success": False,
            "curve_curve_min_dist": 1.0,
            "curve_surface_min_dist": 1.0,
            "surface_vessel_min_dist": 1.0,
            "max_curvature": 1.0,
        }

    def _runtime_bundle_builder(*_args, **_kwargs):
        return {
            "reporting_metrics": lambda _coil_dofs, **_kwargs: {},
            "reporting_metrics_from_solution": (_fake_reporting_metrics_from_solution),
            "objective": lambda _coil_dofs: 0.0,
            "value_and_grad": lambda _coil_dofs: (
                0.0,
                np.zeros(2, dtype=np.float64),
            ),
            "forward_result": lambda _coil_dofs: {},
        }

    monkeypatch.setattr(
        single_stage_example,
        "get_traceable_single_stage_runtime_bundle_builder",
        lambda: _runtime_bundle_builder,
    )
    monkeypatch.setattr(single_stage_example, "CC_DIST", 0.05, raising=False)
    monkeypatch.setattr(single_stage_example, "CS_DIST", 0.02, raising=False)
    monkeypatch.setattr(single_stage_example, "SS_DIST", 0.04, raising=False)
    monkeypatch.setattr(
        single_stage_example,
        "CURVATURE_THRESHOLD",
        40.0,
        raising=False,
    )

    sync = single_stage_example.build_single_stage_target_lane_accepted_step_sync(
        object(),
        object(),
        0.5,
        outer_objective_config=object(),
        success_filter=None,
    )
    run_dict = {
        "sdofs": np.asarray([0.3, 0.4], dtype=np.float64),
        "iota": 0.5,
        "G": 1.5,
        "J": 2.0,
        "dJ": np.asarray([7.0, 8.0], dtype=np.float64),
        "lscount": 4,
        "x_prev": np.asarray([1.0, 2.0], dtype=np.float64),
        "intersecting": False,
        "self_intersection_check_available": True,
    }
    failed_solve_result = {
        "success": False,
        "finite": True,
        "sdofs": np.asarray([99.0, 101.0], dtype=np.float64),
        "iota": np.asarray(9161.0, dtype=np.float64),
        "G": np.asarray(3.599, dtype=np.float64),
        "x": np.asarray([9.0, 10.0, 11.0], dtype=np.float64),
        "objective_value": np.asarray(12.0, dtype=np.float64),
        "objective_grad": np.asarray([13.0, 14.0], dtype=np.float64),
    }

    accepted_step_summary = sync(
        run_dict,
        np.asarray([1.0, 2.0], dtype=np.float64),
        benchmark_mode=False,
        update_run_state=True,
        target_lane_solve_result=failed_solve_result,
    )

    np.testing.assert_allclose(run_dict["sdofs"], [0.3, 0.4])
    assert run_dict["iota"] == pytest.approx(0.5)
    assert run_dict["G"] == pytest.approx(1.5)
    assert run_dict["J"] == pytest.approx(12.0)
    np.testing.assert_allclose(run_dict["dJ"], [13.0, 14.0])
    assert run_dict["lscount"] == 0
    assert accepted_step_summary["objective_value"] == pytest.approx(12.0)
    assert run_dict["target_lane_reporting_metrics"]["solver_success"] is False
    np.testing.assert_allclose(run_dict["latest_local_incumbent"]["sdofs"], [0.3, 0.4])


def test_successful_cpu_candidate_seeds_pending_initial_objective():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    class _Surface:
        def __init__(self):
            self.x = np.zeros(2, dtype=np.float64)

        def set_dofs(self, dofs):
            self.x = np.asarray(dofs, dtype=np.float64).copy()

        def is_self_intersecting(self):
            return False

        def volume(self):
            return 1.25

    class _BoozerSurface:
        def __init__(self):
            self.surface = _Surface()
            self.need_to_run_code = True
            self.run_calls = []
            self.res = {
                "success": True,
                "iota": 0.1,
                "G": 0.2,
            }

        def run_code(self, iota, G):
            self.run_calls.append((float(iota), float(G)))
            self.need_to_run_code = False
            self.res = {
                "success": True,
                "iota": float(iota),
                "G": float(G),
                "residual": np.zeros(1, dtype=np.float64),
            }
            return self.res

    class _Objective:
        def J(self):
            return 5.0

        def dJ(self):
            return np.asarray([7.0, 8.0], dtype=np.float64)

    run_dict = {
        "sdofs": np.asarray([0.3, 0.4], dtype=np.float64),
        "iota": 0.5,
        "G": 1.5,
        "J": float("nan"),
        "dJ": np.zeros(2, dtype=np.float64),
        "initial_objective": float("nan"),
        "initial_objective_pending": True,
        "failure_count": 0,
        "x_prev": np.asarray([1.0, 2.0], dtype=np.float64),
        "intersecting": False,
        "self_intersection_check_available": True,
    }
    boozer_surface = _BoozerSurface()

    value, grad = single_stage_example._evaluate_candidate_impl(
        np.asarray([1.0, 2.0], dtype=np.float64),
        run_dict,
        boozer_surface,
        _Objective(),
    )

    assert value == pytest.approx(5.0)
    np.testing.assert_allclose(grad, [7.0, 8.0])
    assert boozer_surface.run_calls == [(0.5, 1.5)]
    assert run_dict["initial_objective_pending"] is False
    assert run_dict["initial_objective"] == pytest.approx(5.0)
    np.testing.assert_allclose(run_dict["dJ"], [7.0, 8.0])


def test_failed_explicit_warm_start_candidate_restores_accepted_state():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    class _Surface:
        def __init__(self):
            self.x = np.zeros(2, dtype=np.float64)

        def set_dofs(self, dofs):
            self.x = np.asarray(dofs, dtype=np.float64).copy()

        def is_self_intersecting(self):
            return False

    class _BoozerSurface:
        supports_explicit_surface_warm_start = True

        def __init__(self):
            self.surface = _Surface()
            self.need_to_run_code = True
            self.restore_calls = []
            self.res = {
                "success": True,
                "primal_success": True,
                "iota": 0.5,
                "G": 1.5,
                "residual": np.zeros(1, dtype=np.float64),
            }

        def run_code(self, iota, G, *, sdofs):
            np.testing.assert_allclose(sdofs, [0.3, 0.4])
            self.surface.set_dofs([99.0, 101.0])
            self.res = {
                "success": False,
                "primal_success": False,
                "iota": 9161.0,
                "G": 3.599,
                "residual": np.asarray([8.0], dtype=np.float64),
                "fun": 32.0,
            }
            self.need_to_run_code = False
            return self.res

        def install_value_only_solved_runtime_state(
            self,
            *,
            sdofs,
            iota,
            G,
            optimizer_method,
        ):
            self.restore_calls.append(str(optimizer_method))
            self.surface.set_dofs(sdofs)
            self.res = {
                "success": True,
                "primal_success": True,
                "sdofs": np.asarray(sdofs, dtype=np.float64).copy(),
                "iota": float(iota),
                "G": float(G),
                "residual": np.zeros(1, dtype=np.float64),
            }
            self.need_to_run_code = False
            return self.res

    class _Objective:
        def J(self):
            raise AssertionError("failed candidates must not evaluate J")

        def dJ(self):
            raise AssertionError("failed candidates must reuse prior gradient")

    run_dict = {
        "sdofs": np.asarray([0.3, 0.4], dtype=np.float64),
        "iota": 0.5,
        "G": 1.5,
        "J": 2.0,
        "dJ": np.asarray([7.0, 8.0], dtype=np.float64),
        "initial_objective": 2.0,
        "initial_objective_pending": False,
        "failure_count": 0,
        "x_prev": np.asarray([1.0, 2.0], dtype=np.float64),
        "search_policy": "repair_first",
        "donor_class": "runtime_seed",
        "intersecting": False,
        "self_intersection_check_available": True,
    }
    boozer_surface = _BoozerSurface()

    value, grad = single_stage_example._evaluate_candidate_impl(
        np.asarray([4.0, 6.0], dtype=np.float64),
        run_dict,
        boozer_surface,
        _Objective(),
    )

    assert value > 2.0
    np.testing.assert_allclose(grad, [7.0, 8.0])
    assert run_dict["failure_count"] == 1
    assert run_dict["last_candidate_failure"]["reject_class"] == "solver"
    assert boozer_surface.restore_calls == ["accepted-state-rollback"]
    np.testing.assert_allclose(boozer_surface.surface.x, [0.3, 0.4])
    assert boozer_surface.res["success"] is True
    assert boozer_surface.res["iota"] == pytest.approx(0.5)
    assert boozer_surface.res["G"] == pytest.approx(1.5)
    assert boozer_surface.need_to_run_code is True


def test_single_stage_failure_penalty_uses_pre_trial_line_search_point():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    run_dict = {
        "adaptive_failure_penalty_weight": 1.0,
        "J": 2.0,
        "dJ": np.zeros(2, dtype=np.float64),
        "initial_objective": 2.0,
        "initial_objective_pending": False,
        "failure_count": 0,
        "lscount": 0,
        "x_prev": np.asarray([1.0, 2.0], dtype=np.float64),
        "search_policy": "repair_first",
        "donor_class": "stage2_seed_only",
    }
    trial_x = np.asarray([4.0, 6.0], dtype=np.float64)

    single_stage_example._update_line_search_state(trial_x, run_dict)
    penalty, summary = (
        single_stage_example._compute_single_stage_failure_penalty_from_residual_inf(
            trial_x,
            run_dict,
            success_solve=True,
            is_intersecting=True,
            hardware_status=None,
            residual_inf=0.0,
        )
    )

    np.testing.assert_allclose(run_dict["x_prev"], trial_x)
    assert run_dict["lscount"] == 1
    assert summary["step_norm"] == pytest.approx(5.0)
    assert summary["step_ratio"] == pytest.approx(5.0 / np.sqrt(5.0))
    assert summary["reject_class"] == "self_intersection"
    assert penalty == pytest.approx(2.0 * (1.0 + 5.0 / np.sqrt(5.0) + 0.5))


def test_final_penalty_metrics_failed_host_jax_state_skips_solved_objectives(
    monkeypatch,
):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    class _Surface:
        def volume(self):
            return 0.039

    class BoozerSurfaceJAX:
        surface = _Surface()
        res = {"success": False, "G": -2.0, "iota": -1.0e-4}

    class _ForbiddenSolvedStateObjective:
        def J(self):
            raise AssertionError("failed solved state objective must not run")

    class _Penalty:
        def __init__(self, value):
            self.value = float(value)

        def J(self):
            return self.value

    class _DistancePenalty(_Penalty):
        def shortest_distance(self):
            return self.value

    class _CurvaturePenalty(_Penalty):
        class _Curve:
            def kappa(self):
                return np.asarray([1.0, 3.0], dtype=np.float64)

        curve = _Curve()

    monkeypatch.setattr(
        single_stage_example,
        "norm_field_summary",
        lambda _surface, _bs: (0.123, None, None, None, None, None),
    )

    metrics = single_stage_example.resolve_single_stage_final_penalty_metrics(
        use_target_lane=False,
        benchmark_mode=False,
        skip_outer_optimizer=False,
        boozer_surface=BoozerSurfaceJAX(),
        bs=object(),
        iota_target=0.3,
        coil_dofs=np.zeros(2, dtype=np.float64),
        outer_objective_config=None,
        success_filter=None,
        curvelength=_Penalty(4.0),
        j_non_qs=_ForbiddenSolvedStateObjective(),
        j_boozer_residual=_ForbiddenSolvedStateObjective(),
        j_iota=_ForbiddenSolvedStateObjective(),
        j_curve_length=_Penalty(5.0),
        j_curve_curve=_DistancePenalty(0.2),
        j_curve_surface=_DistancePenalty(0.3),
        j_surface_surface=_DistancePenalty(0.4),
        j_curvature=_CurvaturePenalty(6.0),
        cc_dist=0.1,
        cs_dist=0.1,
        ss_dist=0.1,
        curvature_threshold=10.0,
        run_dict={"last_candidate_failure": {"reject_class": "self_intersection"}},
    )

    assert metrics["solved_state_metrics_available"] is False
    assert metrics["solved_state_metrics_unavailable_reason"] == (
        "failed_boozer_solved_state"
    )
    assert np.isnan(metrics["final_non_qs"])
    assert np.isnan(metrics["final_boozer_residual"])
    assert np.isnan(metrics["final_iota_penalty"])
    assert metrics["field_error"] == pytest.approx(0.123)
    assert metrics["final_iota"] == pytest.approx(-1.0e-4)


def test_target_benchmark_initial_status_skips_exact_hardware_evaluation(
    monkeypatch,
):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    def exact_hardware_status_forbidden(_objectives, _diagnostics):
        raise AssertionError("benchmark target startup must not evaluate hardware")

    monkeypatch.setattr(
        single_stage_example,
        "_evaluate_single_stage_hardware_status_reporting_boundary",
        exact_hardware_status_forbidden,
    )

    status = single_stage_example._resolve_initial_single_stage_hardware_status(
        use_target_lane=True,
        benchmark_mode=True,
        objectives=object(),
        diagnostics=object(),
    )

    assert status["success"] is None
    assert status["violations"] == ["skipped_in_benchmark_mode"]


def test_benchmark_final_penalty_metrics_skip_exact_distance_status(monkeypatch):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    class _Surface:
        def volume(self):
            return 0.039

    class _BoozerSurface:
        surface = _Surface()
        res = {"success": True, "G": -2.0, "iota": 0.15}

    class _Penalty:
        def __init__(self, value):
            self.value = float(value)

        def J(self):
            return self.value

    class _DistancePenalty(_Penalty):
        def shortest_distance(self):
            raise AssertionError("benchmark final status must not replay distance")

    class _CurvaturePenalty(_Penalty):
        class _Curve:
            def kappa(self):
                return np.asarray([1.0, 3.0], dtype=np.float64)

        curve = _Curve()

    monkeypatch.setattr(
        single_stage_example,
        "norm_field_summary",
        lambda _surface, _bs: (0.123, None, None, None, None, None),
    )

    metrics = single_stage_example.resolve_single_stage_final_penalty_metrics(
        use_target_lane=False,
        benchmark_mode=True,
        skip_outer_optimizer=False,
        boozer_surface=_BoozerSurface(),
        bs=object(),
        iota_target=0.3,
        coil_dofs=np.zeros(2, dtype=np.float64),
        outer_objective_config=None,
        success_filter=None,
        curvelength=_Penalty(4.0),
        j_non_qs=_Penalty(1.0),
        j_boozer_residual=_Penalty(2.0),
        j_iota=_Penalty(3.0),
        j_curve_length=_Penalty(5.0),
        j_curve_curve=_DistancePenalty(0.2),
        j_curve_surface=_DistancePenalty(0.3),
        j_surface_surface=_DistancePenalty(0.4),
        j_curvature=_CurvaturePenalty(6.0),
        cc_dist=0.1,
        cs_dist=0.1,
        ss_dist=0.1,
        curvature_threshold=10.0,
        run_dict={},
    )

    assert metrics["hardware_status"]["success"] is None
    assert metrics["hardware_status"]["violations"] == ["skipped_in_benchmark_mode"]
    assert metrics["curve_curve_min_dist"] is None
    assert metrics["curve_surface_min_dist"] is None
    assert metrics["surface_vessel_min_dist"] is None


def test_benchmark_target_reporting_sync_skips_exact_hardware_status(monkeypatch):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    def exact_hardware_status_forbidden(*_args, **_kwargs):
        raise AssertionError("benchmark target reporting must not evaluate hardware")

    def fake_bundle_builder():
        def build_runtime_bundle(*_args, **_kwargs):
            def reporting_metrics(coil_dofs, *, include_distance_metrics):
                assert include_distance_metrics is False
                return {
                    "solver_success": np.asarray(True),
                    "final_G": np.asarray(1.0),
                    "final_non_qs": np.asarray(0.0),
                    "final_boozer_residual": np.asarray(0.0),
                    "final_iota_penalty": np.asarray(0.0),
                    "final_length_penalty": np.asarray(0.0),
                    "final_curve_curve_penalty": np.asarray(0.0),
                    "final_curve_surface_penalty": np.asarray(0.0),
                    "final_surface_vessel_penalty": np.asarray(0.0),
                    "final_curvature_penalty": np.asarray(0.0),
                    "coil_length": np.asarray(1.0),
                    "max_curvature": np.asarray(2.0),
                    "banana_current_A": np.asarray(3.0),
                    "field_error": np.asarray(4.0),
                    "final_volume": np.asarray(5.0),
                    "final_iota": np.asarray(6.0),
                }

            def forward_result(coil_dofs):
                coil_dofs = np.asarray(coil_dofs, dtype=np.float64)
                return {
                    "success": np.asarray(True),
                    "finite": np.asarray(True),
                    "sdofs": np.asarray([0.0], dtype=np.float64),
                    "iota": np.asarray(0.15),
                    "G": np.asarray(1.0),
                    "x": np.asarray([0.0], dtype=np.float64),
                    "objective_value": np.asarray(7.0),
                    "objective_grad": np.ones_like(coil_dofs),
                }

            return {
                "reporting_metrics": reporting_metrics,
                "objective": lambda _coil_dofs: np.asarray(7.0),
                "value_and_grad": lambda coil_dofs: (
                    np.asarray(7.0),
                    np.ones_like(np.asarray(coil_dofs, dtype=np.float64)),
                ),
                "forward_result": forward_result,
            }

        return build_runtime_bundle

    monkeypatch.setattr(
        single_stage_example,
        "get_traceable_single_stage_runtime_bundle_builder",
        fake_bundle_builder,
    )
    monkeypatch.setattr(
        single_stage_example,
        "evaluate_single_stage_hardware_constraints_pure",
        exact_hardware_status_forbidden,
    )

    sync = single_stage_example.build_single_stage_target_lane_accepted_step_sync(
        object(),
        object(),
        0.15,
        outer_objective_config=None,
        success_filter=None,
    )
    summary = sync(
        {},
        np.asarray([1.0, 2.0], dtype=np.float64),
        benchmark_mode=True,
        update_run_state=False,
    )

    assert summary["reporting_metrics"]["hardware_status"]["success"] is None
    assert summary["reporting_metrics"]["hardware_status"]["violations"] == [
        "skipped_in_benchmark_mode"
    ]


def test_final_penalty_metrics_failed_native_state_skips_solved_objectives(
    monkeypatch,
):
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    class _Surface:
        def volume(self):
            return 0.049

    class _NativeBoozerSurface:
        surface = _Surface()
        res = {"success": True, "G": -1.5, "iota": 0.12}

    class _ForbiddenSolvedStateObjective:
        def J(self):
            raise AssertionError("failed final candidate objective must not run")

    class _Penalty:
        def __init__(self, value):
            self.value = float(value)

        def J(self):
            return self.value

    class _DistancePenalty(_Penalty):
        def shortest_distance(self):
            return self.value

    class _CurvaturePenalty(_Penalty):
        class _Curve:
            def kappa(self):
                return np.asarray([2.0, 4.0], dtype=np.float64)

        curve = _Curve()

    monkeypatch.setattr(
        single_stage_example,
        "norm_field_summary",
        lambda _surface, _bs: (0.321, None, None, None, None, None),
    )

    metrics = single_stage_example.resolve_single_stage_final_penalty_metrics(
        use_target_lane=False,
        benchmark_mode=False,
        skip_outer_optimizer=False,
        boozer_surface=_NativeBoozerSurface(),
        bs=object(),
        iota_target=0.3,
        coil_dofs=np.zeros(2, dtype=np.float64),
        outer_objective_config=None,
        success_filter=None,
        curvelength=_Penalty(4.0),
        j_non_qs=_ForbiddenSolvedStateObjective(),
        j_boozer_residual=_ForbiddenSolvedStateObjective(),
        j_iota=_ForbiddenSolvedStateObjective(),
        j_curve_length=_Penalty(5.0),
        j_curve_curve=_DistancePenalty(0.2),
        j_curve_surface=_DistancePenalty(0.3),
        j_surface_surface=_DistancePenalty(0.4),
        j_curvature=_CurvaturePenalty(6.0),
        cc_dist=0.1,
        cs_dist=0.1,
        ss_dist=0.1,
        curvature_threshold=10.0,
        run_dict={"last_candidate_failure": {"reject_class": "self_intersection"}},
    )

    assert metrics["solved_state_metrics_available"] is False
    assert metrics["solved_state_metrics_unavailable_reason"] == (
        "failed_boozer_solved_state"
    )
    assert np.isnan(metrics["final_non_qs"])
    assert np.isnan(metrics["final_boozer_residual"])
    assert np.isnan(metrics["final_iota_penalty"])
    assert metrics["field_error"] == pytest.approx(0.321)
    assert metrics["final_volume"] == pytest.approx(0.049)
    assert metrics["final_iota"] == pytest.approx(0.12)
    assert metrics["final_G"] == pytest.approx(-1.5)


def test_final_candidate_failure_forces_optimizer_failure_verdict():
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    success, message = single_stage_example.apply_final_candidate_failure_verdict(
        True,
        "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
        {"last_candidate_failure": {"reject_class": "self_intersection"}},
    )

    assert success is False
    assert "final_candidate_failure=self_intersection" in message
