from __future__ import annotations

from argparse import Namespace
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from benchmarks.single_stage_full_loop_compare import (
    ComparisonTolerances,
    ContractError,
    OBJECTIVE_CONTRACT_ID,
    ORDERED_TERMS,
    TERM_WEIGHTS,
    _parser,
    _shared_configuration,
    _tolerances,
    build_lane_command,
    compare_lane_results,
    lane_environment,
    parse_gnu_time_verbose,
    parse_lane_result,
    resolve_external_output_root,
    sha256_float64_sequence,
    sha256_json,
    source_relevant_git_status,
)


INPUT_SHA256 = {
    "surface": "a" * 64,
    "biotsavart": "b" * 64,
    "boozer_state": "c" * 64,
}
RUN_CONFIG_SHA256 = "d" * 64
DOF_NAMES = ["surface:rc(0,0)", "curve:xc(0)"]
INITIAL_DOFS = (0.25, -0.5)
FINAL_DOFS = (0.2, -0.4)
LAUNCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "perlmutter"
    / "single_stage_full_loop_cpu_gpu.slurm"
)
INTERPRETER_PROBE_PATH = (
    Path(__file__).resolve().parent / "subprocess" / "full_loop_interpreter_probe.py"
)


def _run_interpreter_probe(
    *,
    runner: Path,
    requested_python: Path,
    tmp_path: Path,
    case_name: str,
) -> subprocess.CompletedProcess[str]:
    input_paths = [
        tmp_path / f"{case_name}-{name}" for name in ("surface", "biotsavart", "boozer")
    ]
    for input_path in input_paths:
        input_path.write_text("{}\n", encoding="utf-8")
    return subprocess.run(
        [
            str(runner),
            str(INTERPRETER_PROBE_PATH),
            "--python",
            str(requested_python),
            "--surface-path",
            str(input_paths[0]),
            "--biotsavart-file",
            str(input_paths[1]),
            "--boozer-state-path",
            str(input_paths[2]),
            "--output-root",
            str(tmp_path / f"{case_name}-artifacts"),
            "--environment-lock-sha256",
            "a" * 64,
            "--iota-target",
            "0.11",
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=LAUNCHER_PATH.parents[2],
    )


def _term_payload(
    objective: float,
    *,
    boozer_residual_shift: float = 0.0,
    coil_surface_distance_shift: float = 0.0,
) -> dict[str, dict[str, float]]:
    weighted_values = {name: 0.0 for name in ORDERED_TERMS}
    weighted_values["non_quasisymmetric_ratio"] = (
        objective - boozer_residual_shift - coil_surface_distance_shift
    )
    weighted_values["boozer_residual"] = boozer_residual_shift
    weighted_values["coil_surface_distance"] = coil_surface_distance_shift
    return {
        name: {
            "raw": weighted_values[name] / TERM_WEIGHTS[name],
            "weight": TERM_WEIGHTS[name],
            "weighted": TERM_WEIGHTS[name]
            * (weighted_values[name] / TERM_WEIGHTS[name]),
        }
        for name in ORDERED_TERMS
    }


def _state_payload(
    *,
    dofs: tuple[float, ...],
    objective: float,
    gradient: tuple[float, ...],
    iota: float,
    G: float,
    volume: float,
    boozer_residual_shift: float = 0.0,
    coil_surface_distance_shift: float = 0.0,
) -> dict[str, object]:
    return {
        "dofs": list(dofs),
        "dof_count": len(dofs),
        "dofs_sha256": sha256_float64_sequence(dofs),
        "objective": objective,
        "gradient": list(gradient),
        "gradient_count": len(gradient),
        "gradient_sha256": sha256_float64_sequence(gradient),
        "gradient_norm": math.hypot(*gradient),
        "iota": iota,
        "G": G,
        "volume": volume,
        "terms": _term_payload(
            objective,
            boozer_residual_shift=boozer_residual_shift,
            coil_surface_distance_shift=coil_surface_distance_shift,
        ),
    }


def _lane_payload(
    *,
    backend: str = "native-simsopt-cpu",
    initial_objective: float = 4.0,
    initial_gradient: tuple[float, ...] = (2.0, 0.0),
    initial_iota: float = 0.15,
    initial_G: float = -2.0,
    initial_volume: float = 0.04,
    final_objective: float = 1.0,
    final_dofs: tuple[float, ...] = FINAL_DOFS,
    final_gradient: tuple[float, ...] = (0.02, 0.0),
    final_iota: float = 0.15001,
    final_G: float = -2.000001,
    final_volume: float = 0.04,
    final_boozer_residual_shift: float = 0.0,
    final_coil_surface_distance_shift: float = 0.0,
    optimizer_success: bool = True,
    rejected_evaluations: int = 0,
    input_sha256: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "comparison_schema_version": 1,
        "backend": backend,
        "precision": "float64",
        "constraint_method": "soft-penalty",
        "mixed_precision": False,
        "objective_contract": {
            "id": OBJECTIVE_CONTRACT_ID,
            "ordered_terms": list(ORDERED_TERMS),
            "weights": dict(TERM_WEIGHTS),
            "optimizer_method": "L-BFGS-B",
            "constraint_method": "soft-penalty",
            "dtype": "float64",
            "mixed_precision": False,
            "adjoint_acceptance_policy": (
                "checked-residual-and-condition"
                if backend == "jax-cuda"
                else "native-plu-finite-gradient"
            ),
            "inactive_term_requirements": {"coil_surface_distance": 0.0},
            "dof_names": DOF_NAMES,
            "dof_count": len(DOF_NAMES),
            "dof_names_sha256": sha256_json(DOF_NAMES),
        },
        "input_sha256": dict(INPUT_SHA256 if input_sha256 is None else input_sha256),
        "run_config_sha256": RUN_CONFIG_SHA256,
        "initial_state": _state_payload(
            dofs=INITIAL_DOFS,
            objective=initial_objective,
            gradient=initial_gradient,
            iota=initial_iota,
            G=initial_G,
            volume=initial_volume,
        ),
        "final_state": _state_payload(
            dofs=final_dofs,
            objective=final_objective,
            gradient=final_gradient,
            iota=final_iota,
            G=final_G,
            volume=final_volume,
            boozer_residual_shift=final_boozer_residual_shift,
            coil_surface_distance_shift=final_coil_surface_distance_shift,
        ),
        "optimizer": {
            "method": "L-BFGS-B",
            "success": optimizer_success,
            "nit": 9,
            "nfev": 12,
            "rejected_evaluations": rejected_evaluations,
        },
    }


def test_parse_lane_result_accepts_complete_fp64_contract() -> None:
    parsed = parse_lane_result(_lane_payload())

    assert parsed.contract.contract_id == OBJECTIVE_CONTRACT_ID
    assert parsed.contract.ordered_terms == ORDERED_TERMS
    assert parsed.contract.dof_names == tuple(DOF_NAMES)
    assert parsed.contract.dtype == "float64"
    assert parsed.contract.mixed_precision is False
    assert parsed.input_sha256 == INPUT_SHA256
    assert parsed.initial_state.objective == 4.0
    assert parsed.initial_state.dofs == INITIAL_DOFS
    assert parsed.initial_state.gradient == (2.0, 0.0)
    assert tuple(term.name for term in parsed.final_state.terms) == ORDERED_TERMS
    assert parsed.final_state.iota == 0.15001


def test_parse_lane_result_rejects_missing_parity_evidence() -> None:
    payload = _lane_payload()
    del payload["initial_state"]

    with pytest.raises(ContractError, match="initial_state must be a JSON object"):
        parse_lane_result(payload)


def test_parse_lane_result_rejects_dof_name_hash_not_derived_from_names() -> None:
    payload = _lane_payload()
    objective_contract = payload["objective_contract"]
    assert isinstance(objective_contract, dict)
    objective_contract["dof_names_sha256"] = "0" * 64

    with pytest.raises(ContractError, match="does not identify dof_names"):
        parse_lane_result(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dofs_sha256", "0" * 64, "dofs_sha256 does not identify"),
        ("gradient_sha256", "0" * 64, "gradient_sha256 does not identify"),
        ("dof_count", 3, "dof_count must match"),
        ("gradient_count", 3, "gradient_count must match"),
        ("gradient_norm", 99.0, "gradient_norm does not match"),
    ),
)
def test_parse_lane_result_recomputes_state_counts_hashes_and_norm(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _lane_payload()
    initial_state = payload["initial_state"]
    assert isinstance(initial_state, dict)
    initial_state[field] = value

    with pytest.raises(ContractError, match=message):
        parse_lane_result(payload)


def test_compare_lane_results_passes_contract_seed_and_accuracy_gates() -> None:
    cpu = parse_lane_result(_lane_payload())
    jax = parse_lane_result(
        _lane_payload(
            backend="jax-cuda",
            initial_objective=4.0 + 1.0e-9,
            initial_gradient=(2.0 + 1.0e-7, 0.0),
            initial_iota=0.15 + 1.0e-11,
            initial_G=-2.0 + 1.0e-11,
            initial_volume=0.04 + 1.0e-12,
            final_objective=1.0005,
            final_iota=0.15005,
            final_G=-2.000002,
            final_volume=0.040001,
        )
    )

    comparison = compare_lane_results(
        cpu,
        jax,
        expected_input_sha256=INPUT_SHA256,
        expected_run_config_sha256=RUN_CONFIG_SHA256,
        tolerances=ComparisonTolerances(),
    )

    assert comparison["passed"] is True
    assert comparison["failures"] == []


def test_compare_lane_results_fails_closed_on_input_and_accuracy_drift() -> None:
    wrong_inputs = dict(INPUT_SHA256)
    wrong_inputs["surface"] = "9" * 64
    cpu = parse_lane_result(_lane_payload())
    jax = parse_lane_result(
        _lane_payload(
            backend="jax-cuda",
            final_iota=0.151,
            input_sha256=wrong_inputs,
        )
    )

    comparison = compare_lane_results(
        cpu,
        jax,
        expected_input_sha256=INPUT_SHA256,
        expected_run_config_sha256=RUN_CONFIG_SHA256,
        tolerances=ComparisonTolerances(),
    )

    assert comparison["passed"] is False
    assert "inputs.jax" in comparison["failures"]
    assert "final_state.iota" in comparison["failures"]


def test_production_requires_successful_optimizer_and_zero_rejections() -> None:
    cpu = parse_lane_result(
        _lane_payload(optimizer_success=False, rejected_evaluations=1)
    )
    jax = parse_lane_result(_lane_payload(backend="jax-cuda"))

    comparison = compare_lane_results(
        cpu,
        jax,
        expected_input_sha256=INPUT_SHA256,
        expected_run_config_sha256=RUN_CONFIG_SHA256,
        tolerances=ComparisonTolerances(),
        mode="production",
    )

    assert comparison["passed"] is False
    assert comparison["claim_ready"] is False
    assert "optimizer.success.cpu" in comparison["failures"]
    assert "optimizer.rejected_evaluations.cpu" in comparison["failures"]


def test_production_rejects_huge_final_gradient_without_progress() -> None:
    cpu = parse_lane_result(_lane_payload())
    jax = parse_lane_result(
        _lane_payload(backend="jax-cuda", final_gradient=(1.0e100, 0.0))
    )

    comparison = compare_lane_results(
        cpu,
        jax,
        expected_input_sha256=INPUT_SHA256,
        expected_run_config_sha256=RUN_CONFIG_SHA256,
        tolerances=ComparisonTolerances(),
        mode="production",
    )

    assert comparison["passed"] is False
    assert "final_state.gradient_progress.jax" in comparison["failures"]
    assert "final_state.gradient_norm" in comparison["failures"]


def test_production_rejects_final_dof_drift_despite_matching_observables() -> None:
    cpu = parse_lane_result(_lane_payload())
    jax = parse_lane_result(_lane_payload(backend="jax-cuda", final_dofs=(0.21, -0.4)))

    comparison = compare_lane_results(
        cpu,
        jax,
        expected_input_sha256=INPUT_SHA256,
        expected_run_config_sha256=RUN_CONFIG_SHA256,
        tolerances=ComparisonTolerances(),
        mode="production",
    )

    assert comparison["passed"] is False
    assert "final_state.dofs" in comparison["failures"]


def test_diagnostic_mode_keeps_final_dof_drift_advisory() -> None:
    cpu = parse_lane_result(_lane_payload())
    jax = parse_lane_result(_lane_payload(backend="jax-cuda", final_dofs=(0.21, -0.4)))

    comparison = compare_lane_results(
        cpu,
        jax,
        expected_input_sha256=INPUT_SHA256,
        expected_run_config_sha256=RUN_CONFIG_SHA256,
        tolerances=ComparisonTolerances(),
        mode="diagnostic",
    )

    assert comparison["passed"] is True
    assert comparison["claim_ready"] is False
    assert "final_state.dofs" in comparison["advisory_failures"]
    assert comparison["tolerances"]["final_dofs_rtol"] == 1.0e-3
    assert comparison["tolerances"]["final_dofs_atol"] == 1.0e-6


def test_final_dof_tolerance_cli_flags_override_defaults() -> None:
    args = _parser().parse_args(
        [
            "--surface-path",
            "/seed/surface.json",
            "--biotsavart-file",
            "/seed/biotsavart.json",
            "--boozer-state-path",
            "/seed/boozer-state.json",
            "--output-root",
            "/output",
            "--environment-lock-sha256",
            "e" * 64,
            "--iota-target",
            "0.15",
            "--final-dofs-rtol",
            "2e-3",
            "--final-dofs-atol",
            "3e-6",
        ]
    )

    tolerances = _tolerances(args)

    assert tolerances.final_dofs_rtol == 2.0e-3
    assert tolerances.final_dofs_atol == 3.0e-6


def test_environment_lock_digest_is_part_of_shared_run_identity() -> None:
    base_arguments = [
        "--surface-path",
        "/seed/surface.json",
        "--biotsavart-file",
        "/seed/biotsavart.json",
        "--boozer-state-path",
        "/seed/boozer-state.json",
        "--output-root",
        "/output",
        "--iota-target",
        "0.15",
        "--environment-lock-sha256",
    ]
    first = _shared_configuration(
        _parser().parse_args([*base_arguments, "e" * 64]), INPUT_SHA256
    )
    second = _shared_configuration(
        _parser().parse_args([*base_arguments, "f" * 64]), INPUT_SHA256
    )

    assert first["environment_lock_sha256"] == "e" * 64
    assert sha256_json(first) != sha256_json(second)


def test_environment_lock_digest_must_be_canonical_sha256() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--surface-path",
                "/seed/surface.json",
                "--biotsavart-file",
                "/seed/biotsavart.json",
                "--boozer-state-path",
                "/seed/boozer-state.json",
                "--output-root",
                "/output",
                "--iota-target",
                "0.15",
                "--environment-lock-sha256",
                "E" * 64,
            ]
        )


def test_diagnostic_mode_keeps_outcome_failures_advisory() -> None:
    cpu = parse_lane_result(_lane_payload(optimizer_success=False))
    jax = parse_lane_result(
        _lane_payload(backend="jax-cuda", final_gradient=(1.0e100, 0.0))
    )

    comparison = compare_lane_results(
        cpu,
        jax,
        expected_input_sha256=INPUT_SHA256,
        expected_run_config_sha256=RUN_CONFIG_SHA256,
        tolerances=ComparisonTolerances(),
        mode="diagnostic",
    )

    assert comparison["passed"] is True
    assert comparison["claim_ready"] is False
    assert comparison["failures"] == []
    assert "optimizer.success.cpu" in comparison["advisory_failures"]
    assert "final_state.gradient_progress.jax" in comparison["advisory_failures"]


def test_production_detects_compensating_final_term_drift() -> None:
    cpu = parse_lane_result(_lane_payload())
    jax = parse_lane_result(
        _lane_payload(backend="jax-cuda", final_boozer_residual_shift=1.0e-2)
    )

    comparison = compare_lane_results(
        cpu,
        jax,
        expected_input_sha256=INPUT_SHA256,
        expected_run_config_sha256=RUN_CONFIG_SHA256,
        tolerances=ComparisonTolerances(),
        mode="production",
    )

    assert comparison["passed"] is False
    assert "final_state.terms.non_quasisymmetric_ratio.raw" in comparison["failures"]
    assert "final_state.terms.boozer_residual.weighted" in comparison["failures"]


def test_production_requires_inactive_coil_surface_distance_hinge() -> None:
    cpu = parse_lane_result(_lane_payload())
    jax = parse_lane_result(
        _lane_payload(backend="jax-cuda", final_coil_surface_distance_shift=1.0e-3)
    )

    comparison = compare_lane_results(
        cpu,
        jax,
        expected_input_sha256=INPUT_SHA256,
        expected_run_config_sha256=RUN_CONFIG_SHA256,
        tolerances=ComparisonTolerances(),
        mode="production",
    )

    assert comparison["passed"] is False
    assert (
        "final_state.terms.coil_surface_distance.inactive.jax" in comparison["failures"]
    )


def test_parse_gnu_time_verbose_requires_one_positive_max_rss_record() -> None:
    report = """
        Command being timed: "python child.py"
        User time (seconds): 1.20
        Maximum resident set size (kbytes): 123456
        Exit status: 0
    """

    assert parse_gnu_time_verbose(report) == 123456

    with pytest.raises(ContractError, match="exactly one"):
        parse_gnu_time_verbose("Exit status: 0")


def test_lane_commands_share_profile_but_only_jax_gets_exclusion_flags() -> None:
    args = Namespace(
        python=Path("/python"),
        surface_path=Path("/seed/surface.json"),
        biotsavart_file=Path("/seed/biotsavart.json"),
        boozer_state_path=Path("/seed/boozer-state.json"),
        vmec_s=1.0,
        surface_scale=None,
        mpol=10,
        ntor=10,
        nphi=255,
        ntheta=64,
        constraint_weight=1.0,
        iota_target=0.15,
        maxiter=1500,
        outer_maxcor=300,
        outer_maxls=20,
        outer_ftol=1.0e-15,
        outer_gtol=1.0e-8,
        boozer_bfgs_tol=1.0e-10,
        boozer_bfgs_maxiter=1500,
        boozer_newton_tol=1.0e-13,
        boozer_newton_maxiter=50,
    )
    cpu = build_lane_command(
        args,
        lane="cpu",
        run_dir=Path("/output/cpu"),
        run_config_sha256=RUN_CONFIG_SHA256,
    )
    jax = build_lane_command(
        args,
        lane="jax",
        run_dir=Path("/output/jax"),
        run_config_sha256=RUN_CONFIG_SHA256,
    )

    assert cpu[cpu.index("--objective-profile") + 1] == "common-seven-term"
    assert jax[jax.index("--objective-profile") + 1] == "common-seven-term"
    assert "--output-root" in cpu
    assert "--run-dir" not in cpu
    assert "--run-dir" in jax
    assert "--weight-poloidal-extent" not in cpu
    assert "--weight-poloidal-extent" in jax
    assert "--no-current-penalties" in jax
    assert "--no-width" in jax
    assert "--sign-g" not in cpu
    assert "--sign-g" not in jax


@pytest.mark.parametrize("interpreter_path_kind", ("absolute", "relative"))
def test_normalization_preserves_virtual_environment_interpreter(
    tmp_path: Path,
    interpreter_path_kind: str,
) -> None:
    venv_root = tmp_path / "venv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            "--without-pip",
            "--symlinks",
            str(venv_root),
        ],
        check=True,
    )
    venv_python = venv_root / "bin" / "python"
    assert venv_python.is_symlink()
    assert venv_python.resolve() == Path(sys.executable).resolve()
    repo_root = LAUNCHER_PATH.parents[2]
    if interpreter_path_kind == "absolute":
        requested_python = venv_python
    else:
        requested_python = Path(os.path.relpath(venv_python, repo_root))
    expected_python = (repo_root / requested_python).absolute()
    probe = _run_interpreter_probe(
        runner=venv_python,
        requested_python=requested_python,
        tmp_path=tmp_path,
        case_name=f"venv-entry-{interpreter_path_kind}",
    )
    payload = json.loads(probe.stdout)

    assert payload["normalized_python"] == str(expected_python)
    assert payload["commands"] == {
        "cpu": str(expected_python),
        "jax": str(expected_python),
    }
    assert Path(payload["sys_prefix"]) == venv_root
    command_probe = subprocess.run(
        [
            payload["commands"]["cpu"],
            str(INTERPRETER_PROBE_PATH),
            "--print-prefix",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert Path(command_probe.stdout.strip()) == venv_root


def test_normalization_preserves_parent_symlink_dotdot_traversal(
    tmp_path: Path,
) -> None:
    correct_root = tmp_path / "correct"
    nested_dir = correct_root / "nested"
    correct_bin = correct_root / "bin"
    nested_dir.mkdir(parents=True)
    correct_bin.mkdir()
    (correct_bin / "python").symlink_to("/usr/bin/true")
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(nested_dir, target_is_directory=True)
    requested_python = parent_alias / ".." / "bin" / "python"

    wrong_bin = tmp_path / "bin"
    wrong_bin.mkdir()
    (wrong_bin / "python").symlink_to("/usr/bin/false")
    assert requested_python.is_file()
    expected_python = requested_python.absolute()

    probe = _run_interpreter_probe(
        runner=Path(sys.executable),
        requested_python=requested_python,
        tmp_path=tmp_path,
        case_name="parent-dotdot",
    )
    payload = json.loads(probe.stdout)

    assert payload["normalized_python"] == str(expected_python)
    assert payload["commands"] == {
        "cpu": str(expected_python),
        "jax": str(expected_python),
    }
    for command in payload["commands"].values():
        completed = subprocess.run([command], check=False)
        assert completed.returncode == 0


@pytest.mark.parametrize(
    "visible_devices",
    ("", "0,1", "GPU-first,GPU-second", "0 1", " 0"),
)
def test_jax_lane_requires_exactly_one_cuda_selector(
    visible_devices: str,
) -> None:
    with pytest.raises(RuntimeError, match="exactly one Slurm-assigned"):
        lane_environment("jax", {"CUDA_VISIBLE_DEVICES": visible_devices})

    environment = lane_environment("jax", {"CUDA_VISIBLE_DEVICES": "0"})
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["JAX_PLATFORMS"] == "cuda"


def test_source_status_ignores_only_top_level_slurm_scheduler_logs() -> None:
    status = "\n".join(
        (
            "?? slurm-123.out",
            "?? slurm-123.err",
            "?? nested/slurm-123.out",
            " M benchmarks/runner.py",
        )
    )

    assert source_relevant_git_status(status) == "\n".join(
        ("?? nested/slurm-123.out", " M benchmarks/runner.py")
    )


def test_launcher_venv_root_respects_tmpdir_and_uid_scope(
    tmp_path: Path,
) -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assignment = next(
        line for line in source.splitlines() if line.startswith('VENV_ROOT="')
    )

    def resolve_venv_root(environment: dict[str, str]) -> str:
        completed = subprocess.run(
            ["bash", "-c", f'{assignment}\nprintf "%s" "$VENV_ROOT"'],
            check=True,
            text=True,
            capture_output=True,
            env=environment,
        )
        return completed.stdout

    environment = os.environ.copy()
    environment.pop("VENV_ROOT", None)
    environment["TMPDIR"] = str(tmp_path / "node-local")
    assert resolve_venv_root(environment) == str(
        tmp_path / "node-local" / f"simsopt-full-loop-envs-{os.getuid()}"
    )

    environment.pop("TMPDIR")
    assert resolve_venv_root(environment) == (
        f"/tmp/simsopt-full-loop-envs-{os.getuid()}"
    )

    environment["VENV_ROOT"] = str(tmp_path / "explicit-override")
    assert resolve_venv_root(environment) == environment["VENV_ROOT"]


def test_launcher_pins_expected_bootstrap_tool_versions() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    bootstrap = re.search(
        r"pip install --upgrade \\\n"
        r"\s+'pip==([^']+)' 'setuptools==([^']+)' 'wheel==([^']+)'",
        source,
    )

    assert bootstrap is not None
    assert bootstrap.groups() == ("26.1.2", "83.0.0", "0.47.0")


def test_launcher_installs_only_benchmark_runtime_extras() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    editable_specs = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith('-e "${REPO_ROOT}'):
            editable_specs.append(stripped.removeprefix('-e "').split('"', 1)[0])

    assert editable_specs == ["${REPO_ROOT}[JAX_GPU]", "${REPO_ROOT}"]
    assert (
        re.search(
            r"\[[^]]*\btest\b[^]]*\]",
            source,
            re.IGNORECASE,
        )
        is None
    )
    assert (
        re.search(
            r"\b(?:algs|pytest|ground|bentley[-_.]ottmann|qsc)\b",
            source,
            re.IGNORECASE,
        )
        is None
    )
    assert source.count("'shapely==2.1.2'") == 1
    assert source.count("'numba==0.65.1'") == 1


def test_launcher_isolates_python_environment_before_python_use(
    tmp_path: Path,
) -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    lines = source.splitlines()
    isolation = "unset PYTHONPATH\nexport PYTHONNOUSERSITE=1"
    module_load = "module load python/3.13-26.1.0"
    module_index = lines.index(module_load)

    assert source.count(module_load) == 1
    assert source.count("PYTHONPATH") == 1
    assert source.count("unset PYTHONPATH") == 1
    assert source.count("export PYTHONNOUSERSITE=1") == 1
    assert lines[module_index : module_index + 3] == [
        module_load,
        "unset PYTHONPATH",
        "export PYTHONNOUSERSITE=1",
    ]
    executable_prefix = [
        line
        for line in lines[: module_index + 3]
        if line.strip() and not line.startswith("#")
    ]
    assert executable_prefix == [
        "set -euo pipefail",
        module_load,
        "unset PYTHONPATH",
        "export PYTHONNOUSERSITE=1",
    ]

    fake_site = tmp_path / "pymon"
    fake_metadata = fake_site / "nersc_pymon-0.5.0.dist-info"
    fake_metadata.mkdir(parents=True)
    (fake_metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: nersc-pymon\nVersion: 0.5.0\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(fake_site),
        "PYTHONNOUSERSITE": "0",
    }
    probe = (
        "from importlib.metadata import PackageNotFoundError, version\n"
        "try:\n"
        "    print(version('nersc-pymon'))\n"
        "except PackageNotFoundError:\n"
        "    print('ABSENT')\n"
    )
    leaked = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert leaked.returncode == 0, leaked.stderr
    assert leaked.stdout.strip() == "0.5.0"

    isolated = subprocess.run(
        [
            "bash",
            "-c",
            f'{isolation}\nexec "$1" -c "$2"',
            "isolate-python-environment-test",
            sys.executable,
            probe,
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert isolated.returncode == 0, isolated.stderr
    assert isolated.stdout.strip() == "ABSENT"


def test_launcher_requires_a_private_venv_root(tmp_path: Path) -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    function = re.search(
        r"(?ms)^prepare_private_venv_root\(\) \{.*?^\}\n",
        source,
    )
    assert function is not None
    prepare_call = 'prepare_private_venv_root "${VENV_ROOT}"'
    canonical_assignment = 'VENV_ROOT="$(realpath -- "${VENV_ROOT}")"'
    assert source.count(prepare_call) == 1
    assert source.count(canonical_assignment) == 1
    assert source.index(prepare_call) < source.index(canonical_assignment)

    def prepare(root_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                f'{function.group(0)}\nprepare_private_venv_root "$1"',
                "prepare-private-venv-root-test",
                str(root_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )

    private_root = tmp_path / "private"
    assert prepare(private_root).returncode == 0
    assert private_root.stat().st_mode & 0o777 == 0o700
    assert prepare(private_root).returncode == 0

    permissive_root = tmp_path / "permissive"
    permissive_root.mkdir(mode=0o755)
    permissive_root.chmod(0o755)
    completed = prepare(permissive_root)
    assert completed.returncode == 2
    assert "owned by the current user with mode 700" in completed.stderr
    assert permissive_root.stat().st_mode & 0o777 == 0o755

    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir(mode=0o700)
    symlink_target.chmod(0o700)
    symlink_root = tmp_path / "symlink-root"
    symlink_root.symlink_to(symlink_target, target_is_directory=True)
    completed = prepare(symlink_root)
    assert completed.returncode == 2
    assert "must not be a symbolic link" in completed.stderr
    assert symlink_root.is_symlink()
    assert symlink_target.stat().st_mode & 0o777 == 0o700
    assert tuple(symlink_target.iterdir()) == ()


@pytest.mark.parametrize(
    (
        "profile",
        "execution_mode",
        "slurm_step_id",
        "visible_devices",
        "expected_returncode",
    ),
    (
        ("pilot", "slurm-step", None, None, 0),
        ("full", "slurm-step", None, None, 0),
        ("pilot", "direct", "12", "0", 0),
        ("pilot", "direct", None, "0", 2),
        ("full", "direct", "12", "0", 2),
        ("pilot", "direct", "12", None, 2),
        ("pilot", "direct", "12", "0,1", 2),
        ("pilot", "direct", "12", "0 1", 2),
        ("pilot", "unknown", "12", "0", 2),
    ),
)
def test_launcher_pair_execution_mode_contract(
    profile: str,
    execution_mode: str,
    slurm_step_id: str | None,
    visible_devices: str | None,
    expected_returncode: int,
) -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    function = re.search(
        r"(?ms)^validate_pair_execution_mode\(\) \{.*?^\}\n",
        source,
    )
    assert function is not None
    environment = os.environ.copy()
    environment.pop("SLURM_STEP_ID", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if slurm_step_id is not None:
        environment["SLURM_STEP_ID"] = slurm_step_id
    if visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'{function.group(0)}\nvalidate_pair_execution_mode "$1" "$2"',
            "validate-pair-execution-mode-test",
            profile,
            execution_mode,
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )

    assert completed.returncode == expected_returncode


@pytest.mark.parametrize(
    ("execution_mode", "expected_command"),
    (
        (
            "slurm-step",
            (
                "srun",
                "--ntasks=1",
                "--cpus-per-task=32",
                "--gpus-per-task=1",
                "--cpu-bind=cores",
            ),
        ),
        ("direct", ()),
    ),
)
def test_launcher_pair_execution_mode_selects_expected_command(
    execution_mode: str,
    expected_command: tuple[str, ...],
) -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    function = re.search(
        r"(?ms)^configure_pair_launcher\(\) \{.*?^\}\n",
        source,
    )
    assert function is not None
    completed = subprocess.run(
        [
            "bash",
            "-c",
            (
                f"{function.group(0)}\n"
                'configure_pair_launcher "$1"\n'
                "if ((${#PAIR_LAUNCHER[@]})); then\n"
                '    printf "%s\\n" "${PAIR_LAUNCHER[@]}"\n'
                "fi"
            ),
            "configure-pair-launcher-test",
            execution_mode,
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "SLURM_CPUS_PER_TASK": "32"},
    )

    assert tuple(completed.stdout.splitlines()) == expected_command
    comparator_invocation = (
        'if "${PAIR_LAUNCHER[@]}" "${VENV_DIR}/bin/python" \\\n'
        '        "${REPO_ROOT}/benchmarks/single_stage_full_loop_compare.py" \\\n'
        '        --python "${VENV_DIR}/bin/python" \\'
    )
    assert source.count(comparator_invocation) == 1
    assert source.count("single_stage_full_loop_compare.py") == 1
    assert source.count("--python") == 1


def test_comparator_output_root_must_be_outside_source_checkout(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside_root = tmp_path / "artifacts"
    sibling_prefix = tmp_path / "repo-results"
    symlink_to_repo = tmp_path / "repo-link"
    symlink_to_repo.symlink_to(repo_root, target_is_directory=True)

    assert resolve_external_output_root(repo_root, outside_root) == outside_root
    assert resolve_external_output_root(repo_root, sibling_prefix) == sibling_prefix
    for invalid_root in (
        repo_root,
        repo_root / ".artifacts" / "run",
        repo_root / "nested" / ".." / ".artifacts" / "run",
        symlink_to_repo / ".artifacts" / "run",
    ):
        with pytest.raises(ValueError, match="outside the source checkout"):
            resolve_external_output_root(repo_root, invalid_root)


@pytest.mark.parametrize(
    ("root_label", "root_case", "expected_returncode"),
    (
        ("ARTIFACT_ROOT", "outside", 0),
        ("ARTIFACT_ROOT", "sibling-prefix", 0),
        ("ARTIFACT_ROOT", "equal", 2),
        ("ARTIFACT_ROOT", "nested", 2),
        ("ARTIFACT_ROOT", "dotdot", 2),
        ("ARTIFACT_ROOT", "symlink-nested", 2),
        ("VENV_ROOT", "outside", 0),
        ("VENV_ROOT", "ignored-build", 2),
    ),
)
def test_launcher_write_roots_must_be_outside_source_checkout(
    tmp_path: Path,
    root_label: str,
    root_case: str,
    expected_returncode: int,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    symlink_to_repo = tmp_path / "repo-link"
    symlink_to_repo.symlink_to(repo_root, target_is_directory=True)
    write_roots = {
        "outside": tmp_path / "artifacts",
        "sibling-prefix": tmp_path / "repo-results",
        "equal": repo_root,
        "nested": repo_root / ".artifacts" / "run",
        "dotdot": repo_root / "nested" / ".." / ".artifacts" / "run",
        "symlink-nested": symlink_to_repo / ".artifacts" / "run",
        "ignored-build": repo_root / "build" / "full-loop-envs",
    }
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^validate_external_write_root\(\) \{.*?^\}\n",
        source,
    )
    assert match is not None
    validation_call = 'validate_external_write_root "$1" "$2" "$3"'
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f"{match.group(0)}\n{validation_call}",
            "validate-write-root-test",
            root_label,
            str(repo_root),
            str(write_roots[root_case]),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == expected_returncode
    first_write = source.index('mkdir -p "${MACHINE_ROOT}"')
    assert source.index('validate_external_write_root "ARTIFACT_ROOT"') < first_write
    assert source.index('validate_external_write_root "VENV_ROOT"') < first_write


@pytest.mark.parametrize(
    ("root_case", "expected_returncode"),
    (
        ("siblings", 0),
        ("sibling-prefix", 0),
        ("equal", 2),
        ("artifact-under-venv", 2),
        ("venv-under-artifact", 2),
    ),
)
def test_launcher_artifact_and_venv_roots_must_be_disjoint(
    tmp_path: Path,
    root_case: str,
    expected_returncode: int,
) -> None:
    root_pairs = {
        "siblings": (
            tmp_path / "artifacts" / "job",
            tmp_path / "environments" / "job",
        ),
        "sibling-prefix": (
            tmp_path / "run",
            tmp_path / "run-other",
        ),
        "equal": (
            tmp_path / "shared",
            tmp_path / "shared",
        ),
        "artifact-under-venv": (
            tmp_path / "environment" / "job" / "artifacts",
            tmp_path / "environment" / "job",
        ),
        "venv-under-artifact": (
            tmp_path / "artifacts",
            tmp_path / "artifacts" / "environment" / "job",
        ),
    }
    artifact_root, venv_dir = root_pairs[root_case]
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^validate_disjoint_write_roots\(\) \{.*?^\}\n",
        source,
    )
    assert match is not None
    validation_call = 'validate_disjoint_write_roots "$1" "$2" "$3" "$4"'
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f"{match.group(0)}\n{validation_call}",
            "validate-disjoint-write-roots-test",
            "ARTIFACT_ROOT",
            str(artifact_root),
            "VENV_DIR",
            str(venv_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == expected_returncode
    assert source.index("validate_disjoint_write_roots \\") < source.index(
        'mkdir -p "${MACHINE_ROOT}"'
    )


@pytest.mark.parametrize("root_label", ("ARTIFACT_ROOT", "VENV_DIR"))
def test_launcher_reserves_fresh_write_roots_without_mutating_existing_data(
    tmp_path: Path,
    root_label: str,
) -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^reserve_fresh_directory\(\) \{.*?^\}\n", source)
    assert match is not None
    validation_call = 'reserve_fresh_directory "$1" "$2"'
    root_path = tmp_path / root_label.lower()
    first = subprocess.run(
        [
            "bash",
            "-c",
            f"{match.group(0)}\n{validation_call}",
            "reserve-fresh-directory-test",
            root_label,
            str(root_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert first.returncode == 0
    assert root_path.is_dir()

    sentinel = root_path / "sentinel.txt"
    sentinel.write_text("preserve-me\n", encoding="utf-8")
    second = subprocess.run(
        [
            "bash",
            "-c",
            f"{match.group(0)}\n{validation_call}",
            "reserve-fresh-directory-test",
            root_label,
            str(root_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert second.returncode == 2
    assert sentinel.read_text(encoding="utf-8") == "preserve-me\n"
    assert f"{root_label} must not exist before launch" in second.stderr


@pytest.mark.parametrize("existing_kind", ("empty", "symlink"))
def test_launcher_rejects_other_existing_artifact_root_forms(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^reserve_fresh_directory\(\) \{.*?^\}\n", source)
    assert match is not None
    artifact_root = tmp_path / "artifacts"
    preserved_path = artifact_root
    if existing_kind == "empty":
        artifact_root.mkdir()
    else:
        preserved_path = tmp_path / "existing-target"
        preserved_path.mkdir()
        (preserved_path / "sentinel.txt").write_text(
            "preserve-me\n",
            encoding="utf-8",
        )
        artifact_root.symlink_to(preserved_path, target_is_directory=True)

    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'{match.group(0)}\nreserve_fresh_directory "$1" "$2"',
            "reserve-fresh-directory-test",
            "ARTIFACT_ROOT",
            str(artifact_root),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert artifact_root.exists()
    if existing_kind == "empty":
        assert list(preserved_path.iterdir()) == []
    else:
        assert (preserved_path / "sentinel.txt").read_text(encoding="utf-8") == (
            "preserve-me\n"
        )


def test_launcher_venv_creation_is_non_destructive_and_follows_reservation() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    artifact_reservation = source.index(
        'reserve_fresh_directory "ARTIFACT_ROOT" "${ARTIFACT_ROOT}"'
    )
    venv_reservation = source.index('reserve_fresh_directory "VENV_DIR" "${VENV_DIR}"')
    first_artifact_write = source.index('mkdir -p "${MACHINE_ROOT}"')
    venv_creation = source.index('python -m venv "${VENV_DIR}"')

    assert artifact_reservation < first_artifact_write
    assert venv_reservation < first_artifact_write
    assert venv_reservation < venv_creation
    assert "python -m venv --clear" not in source


@pytest.mark.parametrize(
    ("profile", "pair_count", "expected_returncode"),
    (
        ("pilot", "1", 0),
        ("pilot", "2", 0),
        ("pilot", "0", 2),
        ("pilot", "invalid", 2),
        ("full", "2", 0),
        ("full", "4", 0),
        ("full", "0", 2),
        ("full", "1", 2),
        ("full", "3", 2),
    ),
)
def test_launcher_pair_count_validation(
    profile: str,
    pair_count: str,
    expected_returncode: int,
) -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^validate_pair_count\(\) \{.*?^\}\n",
        source,
    )
    assert match is not None
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'{match.group(0)}\nvalidate_pair_count "$1" "$2"',
            "validate-pair-count-test",
            profile,
            pair_count,
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == expected_returncode
