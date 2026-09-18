"""Contracts for the serial nested-LS example's public transaction helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from simsopt._examples_runtime import ExampleResult, run_example
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt_jax_adapters.geo import nested_ls_example as example
from simsopt_jax_adapters.geo.nested_ls_ncsx import NcsxNestedLsProblem


def test_nested_ls_example_resolution_documents_smoke_and_full_grids():
    assert example.nested_ls_example_resolution(
        True
    ) == example.NestedLsExampleResolution(
        mpol=2,
        ntor=2,
        nphi=7,
        ntheta=7,
    )
    assert example.nested_ls_example_resolution(
        False
    ) == example.NestedLsExampleResolution(
        mpol=6,
        ntor=6,
        nphi=48,
        ntheta=48,
    )


def test_outer_commits_exact_callback_candidate_not_last_feasible_trial(monkeypatch):
    problem = object.__new__(NcsxNestedLsProblem)
    start = np.array([0.0], dtype=np.float64)
    trial = np.array([0.25], dtype=np.float64)
    installed: list[float] = []

    monkeypatch.setattr(example, "_outer_coil_dofs", lambda _problem: start.copy())
    monkeypatch.setattr(
        example,
        "_snapshot_candidate",
        lambda _problem, *, objective, gradient, coil_dofs: (
            example.NestedLsExampleCandidate(
                objective=float(objective),
                gradient=np.array(gradient, dtype=np.float64, copy=True),
                coil_dofs=np.array(coil_dofs, dtype=np.float64, copy=True),
                surface_dofs=(),
                iotas=(0.1,),
                g_values=(2.0,),
                inner_success=True,
                inner_reduced_gradient_l2=None,
                inner_full_penalty_jacobian_l2=1.0e-12,
            )
        ),
    )
    monkeypatch.setattr(
        example,
        "_install_candidate",
        lambda _problem, candidate: installed.append(float(candidate.objective)),
    )

    def evaluate(_problem, coil_dofs):
        value = 1.0 + float(coil_dofs[0])
        return value, np.array([1.0], dtype=np.float64)

    def fake_minimize(objective, x0, **kwargs):
        callback = kwargs["callback"]
        objective(x0)
        objective(trial)
        callback(trial.copy())
        return SimpleNamespace(
            nit=1,
            nfev=2,
            success=False,
            status=1,
            message="iteration limit",
        )

    monkeypatch.setattr(example, "minimize", fake_minimize)
    run = example.run_nested_ls_example_outer(
        problem,
        backend="native",
        inner_policy="banana_bfgs_then_newton",
        resolution=example.nested_ls_example_resolution(True),
        max_steps=1,
        evaluate=evaluate,
    )

    assert installed == [1.0, 1.0, 1.25, 1.25]
    assert run.final_objective == 1.25
    assert run.accepted_nonzero_coil_movement
    assert run.accepted_coil_delta_l2 == 0.25
    assert not run.optimizer_success
    assert run.optimizer_status == 1
    assert run.stopping_reason == "iteration_limit"
    assert run.budget_exhausted
    assert not run.completed_feasible_step


@pytest.mark.parametrize(
    "exit_status",
    (
        "failed",
        "runtime_install_took_newton_step",
        "runtime_install_changed_surface",
        "runtime_install_changed_y",
        "runtime_install_failed",
    ),
)
def test_outer_rejection_uses_committed_anchor_and_keeps_install_failures_loud(
    monkeypatch, exit_status: str
):
    problem = object.__new__(NcsxNestedLsProblem)
    start = np.array([0.0], dtype=np.float64)
    rejected = np.array([0.5], dtype=np.float64)
    installed: list[float] = []
    restored: list[NcsxNestedLsProblem] = []

    monkeypatch.setattr(example, "_outer_coil_dofs", lambda _problem: start.copy())
    monkeypatch.setattr(example, "restore_ncsx_anchor", restored.append)
    monkeypatch.setattr(
        example,
        "_snapshot_candidate",
        lambda _problem, *, objective, gradient, coil_dofs: (
            example.NestedLsExampleCandidate(
                objective=float(objective),
                gradient=np.array(gradient, dtype=np.float64, copy=True),
                coil_dofs=np.array(coil_dofs, dtype=np.float64, copy=True),
                surface_dofs=(),
                iotas=(0.1,),
                g_values=(2.0,),
                inner_success=True,
                inner_reduced_gradient_l2=None,
                inner_full_penalty_jacobian_l2=1.0e-12,
            )
        ),
    )
    monkeypatch.setattr(
        example,
        "_install_candidate",
        lambda _problem, candidate: installed.append(float(candidate.objective)),
    )

    def evaluate(_problem, coil_dofs):
        if float(coil_dofs[0]) == 0.0:
            return 1.0, np.array([1.0], dtype=np.float64)
        raise example.NcsxNestedLsInnerSolveFailed(
            iteration_count=40,
            grad_l2=1.0,
            exit_status=exit_status,
        )

    def fake_minimize(objective, x0, **kwargs):
        objective(x0)
        objective(rejected)
        return SimpleNamespace(
            nit=0,
            nfev=2,
            success=False,
            status=1,
            message="rejected trial",
        )

    monkeypatch.setattr(example, "minimize", fake_minimize)
    if exit_status in (
        "runtime_install_changed_surface",
        "runtime_install_changed_y",
        "runtime_install_failed",
    ):
        with pytest.raises(example.NcsxNestedLsInnerSolveFailed) as raised:
            example.run_nested_ls_example_outer(
                problem,
                backend="native",
                inner_policy="banana_bfgs_then_newton",
                resolution=example.nested_ls_example_resolution(True),
                max_steps=1,
                evaluate=evaluate,
            )
        assert raised.value.exit_status == exit_status
        assert restored == [problem]
        assert installed == [1.0]
        return
    run = example.run_nested_ls_example_outer(
        problem,
        backend="native",
        inner_policy="banana_bfgs_then_newton",
        resolution=example.nested_ls_example_resolution(True),
        max_steps=1,
        evaluate=evaluate,
    )

    assert installed == [1.0, 1.0]
    assert restored == [problem]
    assert run.final_objective == 1.0
    assert run.rejected_evaluations == 1
    assert not run.accepted_nonzero_coil_movement
    assert not run.completed_feasible_step


@pytest.mark.boozer
def test_native_smoke_accepts_a_nonzero_moving_coil_step():
    run = example.run_native_nested_ls_example(smoke=True, max_steps=1)

    assert run.completed_feasible_step
    assert run.inner_success
    assert run.inner_reduced_gradient_l2 is None
    assert run.inner_full_penalty_jacobian_l2 is not None
    assert np.isfinite(run.inner_full_penalty_jacobian_l2)
    assert run.accepted_nonzero_coil_movement
    assert run.accepted_coil_delta_l2 > 0.0


@pytest.mark.boozer
@pytest.mark.parametrize(("mpol", "ntor"), ((2, 2), (8, 2), (2, 8)))
def test_resample_surface_preserves_shared_tensor_fourier_modes_and_zero_fills_growth(
    mpol: int,
    ntor: int,
):
    source = SurfaceXYZTensorFourier(
        nfp=3,
        stellsym=True,
        mpol=6,
        ntor=6,
        quadpoints_phi=np.linspace(0.0, 1.0 / 3.0, 15, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 15, endpoint=False),
    )
    for index, coefficients in enumerate((source.xcs, source.ycs, source.zcs)):
        coefficients[:, :] = index * 10_000 + np.arange(coefficients.size).reshape(
            coefficients.shape
        )
    resolution = example.NestedLsExampleResolution(
        mpol,
        ntor,
        2 * mpol + 1,
        2 * ntor + 1,
    )

    resampled = example._resample_surface(source, resolution)

    for destination_i in range(2 * resolution.mpol + 1):
        poloidal_mode = (
            destination_i
            if destination_i <= resolution.mpol
            else destination_i - resolution.mpol
        )
        for destination_j in range(2 * resolution.ntor + 1):
            toroidal_mode = (
                destination_j
                if destination_j <= resolution.ntor
                else destination_j - resolution.ntor
            )
            if poloidal_mode > source.mpol or toroidal_mode > source.ntor:
                expected = (0.0, 0.0, 0.0)
            else:
                source_i = (
                    poloidal_mode
                    if destination_i <= resolution.mpol
                    else source.mpol + poloidal_mode
                )
                source_j = (
                    toroidal_mode
                    if destination_j <= resolution.ntor
                    else source.ntor + toroidal_mode
                )
                expected = (
                    source.xcs[source_i, source_j],
                    source.ycs[source_i, source_j],
                    source.zcs[source_i, source_j],
                )
            assert (
                resampled.xcs[destination_i, destination_j],
                resampled.ycs[destination_i, destination_j],
                resampled.zcs[destination_i, destination_j],
            ) == expected


def test_common_example_runner_rejects_zero_steps_before_solving(tmp_path: Path):
    solve_calls: list[int] = []

    def solve(_directory: Path, max_steps: int, _scale: str) -> ExampleResult:
        solve_calls.append(max_steps)
        return ExampleResult("unreachable", {}, "ok")

    with pytest.raises(SystemExit, match="2"):
        run_example(
            ["--max-steps", "0", "--output-dir", str(tmp_path)],
            description=None,
            temporary_prefix="unused-",
            bounded_steps=1,
            native_default_steps=3,
            solve=solve,
            runtime_metadata=lambda _scale: {
                "backend_mode": "native_cpu",
                "platform": "cpu",
                "precision": "fp64",
            },
        )

    assert solve_calls == []


def test_runtime_metadata_failure_prevents_solver_and_output_creation(tmp_path: Path):
    output_directory = tmp_path / "not-created"
    solve_calls: list[Path] = []

    def solve(directory: Path, _max_steps: int, _scale: str) -> ExampleResult:
        solve_calls.append(directory)
        return ExampleResult("unreachable", {}, "ok")

    def runtime_metadata(_scale: str) -> dict[str, str]:
        raise RuntimeError("runtime policy rejected")

    with pytest.raises(RuntimeError, match="runtime policy rejected"):
        run_example(
            ["--smoke", "--output-dir", str(output_directory)],
            description=None,
            temporary_prefix="unused-",
            bounded_steps=1,
            native_default_steps=3,
            solve=solve,
            runtime_metadata=runtime_metadata,
        )
    assert solve_calls == []
    assert not output_directory.exists()


@pytest.mark.boozer
@pytest.mark.parametrize(
    ("script_path", "backend", "result_filename"),
    (
        ("examples/jax/2_Intermediate/boozerQA_ls.py", "jax", "boozerQA_ls.json"),
        (
            "examples/jax/native_reference/boozerQA_ls.py",
            "native",
            "native-boozerQA_ls.json",
        ),
    ),
)
def test_public_smoke_lane_writes_the_common_output_contract(
    tmp_path: Path, script_path: str, backend: str, result_filename: str
):
    script = Path(__file__).resolve().parents[2] / script_path
    completed = subprocess.run(
        (
            sys.executable,
            str(script),
            "--smoke",
            "--json",
            "--output-dir",
            str(tmp_path),
        ),
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["scale"] == "bounded"
    assert payload["observables"]["backend"] == backend
    if backend == "jax":
        assert payload["observables"]["inner_reduced_gradient_l2"] is not None
    else:
        assert payload["observables"]["inner_reduced_gradient_l2"] is None
    assert payload["observables"]["inner_full_penalty_jacobian_l2"] is not None
    assert json.loads((tmp_path / result_filename).read_text()) == payload
