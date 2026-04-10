from __future__ import annotations

from dataclasses import dataclass
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

bootstrap_local_simsopt()

pytest.importorskip(
    "simsoptpp",
    reason="Single-stage integration tests require simsoptpp (use candidate-fixed env)",
)

from simsopt._core.optimizable import load  # noqa: E402
from simsopt.geo import CurveLength  # noqa: E402
from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance  # noqa: E402


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
        command = [
            "--backend",
            backend,
            "--output-root",
            str(output_root),
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
            "--equilibria-dir",
            str(DEFAULT_EQUILIBRIA_DIR),
        ]
        if backend == "jax":
            command += ["--optimizer-backend", optimizer_backend]
        run_python_script(
            _single_stage_script_path(),
            command,
            env=repo_pythonpath_env(
                platform=platform,
                disable_compilation_cache=True,
                clear_backend_guardrails=(backend != "jax"),
            ),
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
        """One-step-budget outer-loop smoke parity for key physics quantities."""
        cpu_run, jax_run = outer_baseline_runs
        _assert_physics_quantity_parity(
            cpu_run.summary,
            jax_run.summary,
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
