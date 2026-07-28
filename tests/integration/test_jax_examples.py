from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import examples.jax.run_examples as example_runner
from examples.jax._lane_environment import (
    build_execution_environment,
    build_lane_environment,
)
from examples.jax._manifest import (
    JaxExampleRecord,
    JaxExamplesManifest,
)
from examples.jax.manifest_runtime import (
    RuntimeContractPair,
    RuntimeExample,
    load_runtime_contract_pair,
)
from examples.jax.parity._manifest import ParityManifest
from examples.jax.run_examples import (
    _parse_arguments,
    build_child_command,
    run_lane,
    run_profile,
)
from simsopt_jax.config import ExecutionIntent, JaxDevice


def _record(
    path: str,
    *,
    lanes: tuple[str, ...] = ("cpu-smoke",),
    smoke_args: tuple[str, ...] = (),
) -> JaxExampleRecord:
    return JaxExampleRecord(
        id="test-example",
        path=path,
        status="ready",
        tier="1_Simple",
        inspired_by=("1_Simple/just_a_quadratic.py",),
        execution_kind="pure",
        jax_surfaces=("simsopt_jax.solve.least_squares_serial_solve_jax",),
        host_boundaries=(),
        extras=("JAX",),
        smoke_args=smoke_args,
        correctness_tests=("tests/integration/test_jax_examples.py",),
        lanes=lanes,
    )


def _manifest(record: JaxExampleRecord) -> JaxExamplesManifest:
    return JaxExamplesManifest(source_catalog=(), jax_examples=(record,))


def _repository_examples(repo_root: Path) -> tuple[RuntimeExample, ...]:
    return load_runtime_contract_pair(
        repo_root / "examples" / "jax" / "manifest.json",
        repo_root / "examples" / "jax" / "parity_manifest.json",
        repo_root=repo_root,
    ).examples


def test_every_ready_repository_example_runs_on_cpu_and_gpu() -> None:
    """Ready JAX lessons use the same implementation on both real platforms."""
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)

    for example in examples:
        if example.status == "ready":
            assert example.lanes == ("cpu-smoke", "gpu-strict"), example.id


def test_every_ready_example_runs_with_global_strict_transfer_guard_on_cpu() -> None:
    """Exercise the global JAX guard without requiring CUDA in the CPU test job."""
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)
    environment = build_lane_environment(
        "cpu-smoke",
        os.environ,
        repo_root=repo_root,
    )
    environment["SIMSOPT_JAX_TRANSFER_GUARD"] = "disallow"
    environment["JAX_TRANSFER_GUARD"] = "disallow"

    for example in examples:
        if example.status != "ready":
            continue
        completed = subprocess.run(
            build_child_command(example, repo_root=repo_root),
            cwd=repo_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, f"{example.id}: {completed.stderr}"


def test_gpu_examples_do_not_import_host_scipy_optimizers() -> None:
    """A GPU-strict example cannot hide a CPU optimizer behind JAX metrics."""
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)

    for example in examples:
        if example.status != "ready" or "gpu-strict" not in example.lanes:
            continue
        example_path = repo_root / "examples" / "jax" / example.path
        syntax_tree = ast.parse(example_path.read_text(encoding="utf-8"))
        imported_modules = {
            alias.name
            for node in ast.walk(syntax_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(syntax_tree)
            if isinstance(node, ast.ImportFrom)
        }
        scipy_modules = {
            module
            for module in imported_modules
            if module == "scipy" or module.startswith("scipy.")
        }
        assert not scipy_modules, example.id


def _write_child(repo_root: Path, filename: str, source: str) -> str:
    relative_path = f"1_Simple/{filename}"
    child_path = repo_root / "examples" / "jax" / relative_path
    child_path.parent.mkdir(parents=True)
    child_path.write_text(source, encoding="utf-8")
    return relative_path


def _result_payload(
    *,
    backend_mode: str = "jax_gpu_parity",
    platform: str = "gpu",
    precision: str = "fp64",
    status: str = "ok",
) -> dict[str, object]:
    return {
        "example_id": "test-example",
        "backend_mode": backend_mode,
        "platform": platform,
        "precision": precision,
        "status": status,
        "observables": {"value": 1.0},
    }


def test_child_command_has_exact_bounded_argument_order(tmp_path: Path) -> None:
    record = _record("1_Simple/example.py", smoke_args=("--steps", "2"))

    assert build_child_command(record, repo_root=tmp_path) == (
        sys.executable,
        str(tmp_path / "examples" / "jax" / "1_Simple" / "example.py"),
        "--smoke",
        "--json",
        "--steps",
        "2",
    )


def test_lane_environment_owns_runtime_selection_before_child_startup() -> None:
    cpu = build_lane_environment("cpu-smoke", {"PRESERVED": "yes"})
    gpu = build_lane_environment("gpu-strict", {"PRESERVED": "yes"})

    assert cpu["PRESERVED"] == "yes"
    assert cpu["SIMSOPT_BACKEND_MODE"] == "jax_cpu_parity"
    assert cpu["SIMSOPT_PRECISION"] == "fp64"
    assert cpu["JAX_PLATFORMS"] == "cpu"
    assert cpu["JAX_ENABLE_X64"] == "1"
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu["MPI4PY_RC_INITIALIZE"] == "false"
    assert gpu["MPI4PY_RC_INITIALIZE"] == "false"
    assert gpu["JAX_TRANSFER_GUARD"] == "disallow"
    assert (
        gpu
        | {
            "PRESERVED": "yes",
            "SIMSOPT_BACKEND_MODE": "jax_gpu_parity",
            "SIMSOPT_BACKEND_STRICT": "1",
            "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
            "SIMSOPT_PRECISION": "fp64",
            "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true",
            "JAX_PLATFORMS": "cuda",
            "JAX_ENABLE_X64": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
        == gpu
    )


@pytest.mark.parametrize(
    ("device", "intent", "expected_mode", "expected_guard"),
    (
        ("cpu", "fast", "jax_cpu_fast", "log"),
        ("cpu", "parity", "jax_cpu_parity", "allow"),
        ("gpu", "fast", "jax_gpu_fast", "log"),
        ("gpu", "parity", "jax_gpu_parity", "disallow"),
    ),
)
def test_execution_environment_scrubs_inherited_runtime_selectors(
    device: JaxDevice,
    intent: ExecutionIntent,
    expected_mode: str,
    expected_guard: str,
) -> None:
    inherited = {
        "PRESERVED": "yes",
        "SIMSOPT_BACKEND": "cpu",
        "STAGE2_BACKEND": "cpu",
        "SIMSOPT_BACKEND_MODE": "native_cpu",
        "SIMSOPT_JAX_PLATFORM": "cpu",
        "SIMSOPT_JAX_BACKEND": "cpu",
        "SIMSOPT_BACKEND_STRICT": "0",
        "SIMSOPT_PRECISION": "mixed",
        "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
        "JAX_TRANSFER_GUARD": "disallow",
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_ENABLE_X64": "0",
    }

    profile, environment = build_execution_environment(
        device,
        intent,
        inherited,
    )

    assert profile.mode == expected_mode
    assert environment["PRESERVED"] == "yes"
    assert environment["SIMSOPT_BACKEND_MODE"] == expected_mode
    assert environment["SIMSOPT_BACKEND_STRICT"] == "1"
    assert environment["SIMSOPT_PRECISION"] == "fp64"
    assert environment["JAX_PLATFORMS"] == ("cpu" if device == "cpu" else "cuda")
    assert environment["JAX_TRANSFER_GUARD"] == expected_guard
    assert "SIMSOPT_BACKEND" not in environment
    assert "STAGE2_BACKEND" not in environment
    assert "SIMSOPT_JAX_PLATFORM" not in environment
    assert "SIMSOPT_JAX_BACKEND" not in environment
    assert "JAX_PLATFORM_NAME" not in environment


@pytest.mark.parametrize(
    ("device", "intent", "backend_mode", "platform"),
    (
        ("cpu", "fast", "jax_cpu_fast", "cpu"),
        ("cpu", "parity", "jax_cpu_parity", "cpu"),
        ("gpu", "fast", "jax_gpu_fast", "gpu"),
        ("gpu", "parity", "jax_gpu_parity", "gpu"),
    ),
)
def test_runner_accepts_only_the_selected_profile_result(
    tmp_path: Path,
    device: JaxDevice,
    intent: ExecutionIntent,
    backend_mode: str,
    platform: str,
) -> None:
    payload = _result_payload(backend_mode=backend_mode, platform=platform)
    child = _write_child(
        tmp_path,
        "selected_profile.py",
        f"import json\nprint(json.dumps({payload!r}, sort_keys=True))\n",
    )
    stdout = io.StringIO()

    exit_code = run_profile(
        _manifest(_record(child, lanes=("cpu-smoke", "gpu-strict"))),
        device,
        intent,
        repo_root=tmp_path,
        base_environment={},
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    assert "PASS test-example" in stdout.getvalue()


@pytest.mark.parametrize("intent", ("fast", "parity"))
def test_gpu_profiles_reject_cpu_fallback_result(
    tmp_path: Path,
    intent: ExecutionIntent,
) -> None:
    payload = _result_payload(
        backend_mode=f"jax_gpu_{intent}",
        platform="cpu",
    )
    child = _write_child(
        tmp_path,
        "cpu_fallback.py",
        f"import json\nprint(json.dumps({payload!r}, sort_keys=True))\n",
    )
    stderr = io.StringIO()

    exit_code = run_profile(
        _manifest(_record(child, lanes=("gpu-strict",))),
        "gpu",
        intent,
        repo_root=tmp_path,
        base_environment={},
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == 1
    assert "platform must be gpu" in stderr.getvalue()


@pytest.mark.parametrize(
    ("arguments", "expected_device", "expected_intent"),
    (
        (("--device", "cpu"), "cpu", "fast"),
        (("--device", "gpu"), "gpu", "fast"),
        (("--device", "cpu", "--intent", "parity"), "cpu", "parity"),
        (("--device", "gpu", "--intent", "parity"), "gpu", "parity"),
    ),
)
def test_runner_parser_supports_device_and_execution_intent(
    arguments: tuple[str, ...],
    expected_device: str,
    expected_intent: str,
) -> None:
    parsed = _parse_arguments(arguments)

    assert parsed.device == expected_device
    assert parsed.intent == expected_intent


def test_runner_parser_rejects_mixed_legacy_and_new_selectors() -> None:
    with pytest.raises(SystemExit):
        _parse_arguments(("--lane", "cpu-smoke", "--device", "cpu"))

    with pytest.raises(SystemExit):
        _parse_arguments(("--lane", "cpu-smoke", "--intent", "fast"))


def test_runner_manifest_observability_distinguishes_v1_adapter() -> None:
    manifest = JaxExamplesManifest(
        source_catalog=(),
        jax_examples=(),
        schema_version=1,
        used_legacy_manifest_adapter=True,
    )

    assert example_runner.manifest_observability_payload(manifest) == {
        "manifest_schema_version": 1,
        "used_legacy_manifest_adapter": True,
    }


def test_runner_emits_manifest_observability_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = RuntimeContractPair(
        version_pair=(3, 2),
        used_legacy_adapter=False,
        examples=(),
        parity=ParityManifest(schema_version=2, relationships=()),
    )
    monkeypatch.setattr(
        example_runner,
        "load_runtime_contract_pair",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(example_runner, "run_profile", lambda *_args, **_kwargs: 0)

    exit_code = example_runner.main(
        ["--device", "cpu", "--manifest", str(tmp_path / "manifest.json")]
    )

    assert exit_code == 0
    assert capsys.readouterr().err == (
        '{"examples_manifest_schema_version":3,'
        '"parity_manifest_schema_version":2,'
        '"used_legacy_manifest_adapter":false}\n'
    )


def test_runner_executes_real_child_with_exact_smoke_arguments(tmp_path: Path) -> None:
    payload = _result_payload(
        backend_mode="jax_cpu_parity", platform="cpu", precision="fp64"
    )
    child = _write_child(
        tmp_path,
        "success.py",
        "import json\n"
        "import sys\n"
        "if sys.argv[1:] != ['--smoke', '--json', '--steps', '2']:\n"
        "    raise SystemExit(9)\n"
        f"print(json.dumps({payload!r}, sort_keys=True))\n",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    with pytest.warns(DeprecationWarning, match="--lane is deprecated"):
        exit_code = run_lane(
            _manifest(_record(child, smoke_args=("--steps", "2"))),
            "cpu-smoke",
            repo_root=tmp_path,
            base_environment={},
            stdout=stdout,
            stderr=stderr,
        )

    assert exit_code == 0
    assert "PASS test-example" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_runner_propagates_nonzero_child_with_full_context(tmp_path: Path) -> None:
    child = _write_child(
        tmp_path,
        "failure.py",
        "import sys\n"
        "print('child stdout')\n"
        "print('child stderr', file=sys.stderr)\n"
        "raise SystemExit(7)\n",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_lane(
        _manifest(_record(child)),
        "cpu-smoke",
        repo_root=tmp_path,
        base_environment={},
        stdout=stdout,
        stderr=stderr,
    )

    failure = stderr.getvalue()
    assert exit_code == 1
    assert "test-example" in failure
    assert "1_Simple/just_a_quadratic.py" in failure
    assert "--smoke --json" in failure
    assert "child stdout" in failure
    assert "child stderr" in failure


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        (_result_payload(status="skipped"), "status must be ok"),
        (_result_payload(status="unsupported"), "status must be ok"),
        (_result_payload(platform="cpu"), "platform must be gpu"),
        (_result_payload(backend_mode="jax_cpu_parity"), "backend_mode must be"),
        (_result_payload(precision="mixed"), "precision must be fp64"),
    ],
)
def test_gpu_strict_rejects_skip_fallback_and_wrong_runtime(
    tmp_path: Path, payload: dict[str, object], expected_message: str
) -> None:
    child = _write_child(
        tmp_path,
        "invalid_gpu.py",
        f"import json\nprint(json.dumps({payload!r}, sort_keys=True))\n",
    )
    stderr = io.StringIO()

    exit_code = run_lane(
        _manifest(_record(child, lanes=("gpu-strict",))),
        "gpu-strict",
        repo_root=tmp_path,
        base_environment={},
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == 1
    assert expected_message in stderr.getvalue()


def test_gpu_strict_rejects_malformed_result(tmp_path: Path) -> None:
    child = _write_child(tmp_path, "malformed.py", "print('not json')\n")
    stderr = io.StringIO()

    exit_code = run_lane(
        _manifest(_record(child, lanes=("gpu-strict",))),
        "gpu-strict",
        repo_root=tmp_path,
        base_environment={},
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == 1
    assert "final stdout line is not valid JSON" in stderr.getvalue()


def test_traceable_least_squares_example_matches_analytic_optimum() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)
    example = next(
        record for record in examples if record.id == "traceable-least-squares"
    )

    assert example.status == "ready"
    existing_logs = set(repo_root.glob("simsopt_*.dat"))
    completed = subprocess.run(
        build_child_command(example, repo_root=repo_root),
        cwd=repo_root,
        env=build_lane_environment("cpu-smoke", os.environ, repo_root=repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["example_id"] == "traceable-least-squares"
    assert result["backend_mode"] == "jax_cpu_parity"
    assert result["platform"] == "cpu"
    assert result["precision"] == "fp64"
    assert result["status"] == "ok"
    assert result["observables"]["solution"] == pytest.approx([1.0, 2.0, 3.0])
    assert result["observables"]["objective"] <= 1.0e-16
    assert result["observables"]["residual_norm"] <= 1.0e-12
    assert result["observables"]["gradient_inf_norm"] <= 1.0e-12
    assert result["observables"]["solver_driver"] == "simsopt_lm_gmres"
    assert result["observables"]["solver_success"] is True
    assert result["observables"]["solver_status"] in (0, 1, 2)
    assert 0 < result["observables"]["iterations"] <= 32
    assert result["observables"]["function_evaluations"] > 0
    assert result["observables"]["jacobian_evaluations"] > 0
    assert set(repo_root.glob("simsopt_*.dat")) == existing_logs


def test_curve_length_example_matches_circle_oracle_and_directional_fd() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)
    example = next(
        record for record in examples if record.id == "curve-length-optimization"
    )

    assert example.status == "ready"
    completed = subprocess.run(
        build_child_command(example, repo_root=repo_root),
        cwd=repo_root,
        env=build_lane_environment("cpu-smoke", os.environ, repo_root=repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["example_id"] == "curve-length-optimization"
    assert result["status"] == "ok"
    assert result["observables"]["final_length"] == pytest.approx(
        result["observables"]["circle_oracle"], rel=1.0e-10, abs=1.0e-10
    )
    assert result["observables"]["gradient_fd_error"] <= 1.0e-6


def test_surface_geometry_example_matches_axisymmetric_torus_oracle() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)
    example = next(
        record for record in examples if record.id == "surface-geometry-optimization"
    )

    assert example.status == "ready"
    completed = subprocess.run(
        build_child_command(example, repo_root=repo_root),
        cwd=repo_root,
        env=build_lane_environment("cpu-smoke", os.environ, repo_root=repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    observables = result["observables"]
    assert result["status"] == "ok"
    assert observables["area"] == pytest.approx(
        observables["area_oracle"], rel=1.0e-9, abs=1.0e-10
    )
    assert observables["volume"] == pytest.approx(
        observables["volume_oracle"], rel=1.0e-9, abs=1.0e-10
    )
    assert observables["residual_norm"] <= 1.0e-9


def test_permanent_magnet_example_matches_greedy_coordinate_oracle() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)
    example = next(
        record for record in examples if record.id == "permanent-magnet-optimization"
    )

    assert example.status == "ready"
    completed = subprocess.run(
        build_child_command(example, repo_root=repo_root),
        cwd=repo_root,
        env=build_lane_environment("cpu-smoke", os.environ, repo_root=repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    observables = result["observables"]
    assert result["status"] == "ok"
    assert observables["moments"] == [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    assert observables["residual_norm"] == pytest.approx(5.0**0.5 / 5.0)
    assert observables["selected_dipoles"] == [0, 1]


def test_fieldline_example_matches_pure_toroidal_orbit_oracle() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)
    example = next(
        record for record in examples if record.id == "fieldline-and-particle-tracing"
    )

    assert example.status == "ready"
    completed = subprocess.run(
        build_child_command(example, repo_root=repo_root),
        cwd=repo_root,
        env=build_lane_environment("cpu-smoke", os.environ, repo_root=repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    observables = result["observables"]
    assert result["status"] == "ok"
    assert observables["integrator_status"] == 0
    assert observables["event_count"] == 0
    assert observables["final_state"] == pytest.approx(
        observables["analytic_final_state"], rel=1.0e-9, abs=1.0e-10
    )


def test_coil_flux_example_has_independent_gradient_oracle_and_reduces_flux() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)
    example = next(
        record for record in examples if record.id == "coil-flux-optimization"
    )

    assert example.status == "ready"
    assert example.classification == "adapter"
    completed = subprocess.run(
        build_child_command(example, repo_root=repo_root),
        cwd=repo_root,
        env=build_lane_environment("cpu-smoke", os.environ, repo_root=repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    observables = result["observables"]
    assert result["status"] == "ok"
    assert observables["gradient_fd_error"] <= 1.0e-8
    assert observables["final_flux"] <= 1.0e-6 * observables["initial_flux"]
    assert observables["coil_length"] == pytest.approx(
        observables["coil_length_oracle"], rel=1.0e-12, abs=1.0e-12
    )


def test_qfm_example_reduces_penalty_and_publishes_final_surface() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)
    example = next(
        record for record in examples if record.id == "qfm-surface-optimization"
    )

    assert example.status == "ready"
    completed = subprocess.run(
        build_child_command(example, repo_root=repo_root),
        cwd=repo_root,
        env=build_lane_environment("cpu-smoke", os.environ, repo_root=repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    observables = result["observables"]
    assert result["status"] == "ok"
    assert observables["solver_success"] is True
    assert observables["final_penalty"] < observables["initial_penalty"]
    assert observables["surface_update_norm"] > 0.0
    assert observables["gradient_norm"] < observables["initial_gradient_norm"]
    assert observables["gradient_norm"] <= 1.0e-8
    assert 0 < observables["iterations"] <= 75


def test_boozer_example_reports_solver_certificate_and_final_surface() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)
    example = next(
        record for record in examples if record.id == "boozer-surface-optimization"
    )

    assert example.status == "ready"
    completed = subprocess.run(
        build_child_command(example, repo_root=repo_root),
        cwd=repo_root,
        env=build_lane_environment("cpu-smoke", os.environ, repo_root=repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    observables = result["observables"]
    assert result["status"] == "ok"
    assert observables["solver_success"] is True
    assert observables["surface_update_norm"] > 0.0
    assert observables["residual_norm"] < 1.0
    assert observables["final_gradient_inf_norm"] <= 1.0e-8


def test_wireframe_example_matches_constrained_oracle_and_publishes_state() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)
    example = next(
        record for record in examples if record.id == "wireframe-optimization"
    )

    assert example.status == "ready"
    completed = subprocess.run(
        build_child_command(example, repo_root=repo_root),
        cwd=repo_root,
        env=build_lane_environment("cpu-smoke", os.environ, repo_root=repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    observables = result["observables"]
    assert result["status"] == "ok"
    assert observables["solution_oracle_error"] <= 1.0e-10
    assert observables["constraint_residual_norm"] <= 1.0e-10
    assert observables["published_current_error"] == 0.0
    assert observables["gsco_nonfinal_steps"] == 1
    assert observables["gsco_enclosed_segments"] == 4


def test_force_finite_build_example_matches_force_and_frame_oracles() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples = _repository_examples(repo_root)
    example = next(
        record for record in examples if record.id == "coil-force-and-finite-build"
    )

    assert example.status == "ready"
    completed = subprocess.run(
        build_child_command(example, repo_root=repo_root),
        cwd=repo_root,
        env=build_lane_environment("cpu-smoke", os.environ, repo_root=repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    observables = result["observables"]
    assert result["status"] == "ok"
    assert observables["force_objective_oracle_error"] <= 1.0e-12
    assert observables["gradient_fd_error"] <= 1.0e-6
    assert observables["frame_orthonormality_error"] <= 1.0e-12
    assert observables["planar_torsion_max"] <= 1.0e-12
