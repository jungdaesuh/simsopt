from __future__ import annotations

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

import benchmarks.single_stage_full_loop_compare as full_loop_compare
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
FAKE_PYTHON_PATH = (
    Path(__file__).resolve().parent / "subprocess" / "full_loop_fake_python.py"
)
COMPARATOR_PATH = LAUNCHER_PATH.parents[1] / "single_stage_full_loop_compare.py"
THREE_LANES = ("native_cpu", "jax_cpu", "jax_gpu")
THREE_NODES = {
    "native_cpu": "nid000101",
    "jax_cpu": "nid000202",
    "jax_gpu": "nid000303",
}


def _install_fake_python(tmp_path: Path) -> Path:
    fake_python = tmp_path / "shared" / "venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_bytes(FAKE_PYTHON_PATH.read_bytes())
    fake_python.chmod(0o755)
    return fake_python


def _launcher_python_heredoc(containing: str) -> str:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"(?ms)<<'PY'[^\n]*\n(.*?)^PY$", source)
    matches = [block for block in blocks if containing in block.splitlines()]
    assert len(matches) == 1
    return matches[0]


def _prepare_arguments(
    *,
    tmp_path: Path,
    output_root: Path,
    python: Path,
    nodes: dict[str, str] | None = None,
) -> list[str]:
    seed_root = tmp_path / "seed"
    seed_root.mkdir(exist_ok=True)
    input_paths = {
        "surface": seed_root / "surface.json",
        "biotsavart": seed_root / "biotsavart.json",
        "boozer": seed_root / "boozer.json",
    }
    for input_path in input_paths.values():
        input_path.write_text("{}\n", encoding="utf-8")
    assigned_nodes = THREE_NODES if nodes is None else nodes
    return [
        "prepare",
        "--python",
        str(python),
        "--surface-path",
        str(input_paths["surface"]),
        "--biotsavart-file",
        str(input_paths["biotsavart"]),
        "--boozer-state-path",
        str(input_paths["boozer"]),
        "--output-root",
        str(output_root),
        "--environment-lock-sha256",
        "e" * 64,
        "--iota-target",
        "0.15",
        "--maxiter",
        "1",
        "--native-cpu-node",
        assigned_nodes["native_cpu"],
        "--jax-cpu-node",
        assigned_nodes["jax_cpu"],
        "--jax-gpu-node",
        assigned_nodes["jax_gpu"],
    ]


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
                if backend in {"jax-cpu", "jax-cuda"}
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


def _prepare_three_lane_probe(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nodes: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    output_root = tmp_path / "pair"
    fake_python = _install_fake_python(tmp_path)
    clean_identity = full_loop_compare.SourceIdentity(
        commit_sha="1" * 40,
        tree_sha="2" * 40,
        status_porcelain="",
    )
    monkeypatch.setattr(
        full_loop_compare,
        "source_identity",
        lambda _repo_root: clean_identity,
    )
    returncode = full_loop_compare.main(
        _prepare_arguments(
            tmp_path=tmp_path,
            output_root=output_root,
            python=fake_python,
            nodes=nodes,
        )
    )
    assert returncode == 0
    return output_root, fake_python


def _write_three_lane_result_templates(output_root: Path, tmp_path: Path) -> Path:
    manifest = json.loads(
        (output_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    results_root = tmp_path / "probe-results"
    results_root.mkdir()
    backends = {
        "native_cpu": "native-simsopt-cpu",
        "jax_cpu": "jax-cpu",
        "jax_gpu": "jax-cuda",
    }
    for lane, backend in backends.items():
        payload = _lane_payload(
            backend=backend,
            input_sha256=manifest["input_sha256"],
        )
        payload["run_config_sha256"] = manifest["run_config_sha256"]
        (results_root / f"{lane}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return results_root


def _run_three_lane_probe(
    *,
    output_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    results_root = _write_three_lane_result_templates(output_root, tmp_path)

    def fake_step_identity(assigned_node: str) -> full_loop_compare.StepIdentity:
        return full_loop_compare.StepIdentity(
            assigned_node=assigned_node,
            actual_node=assigned_node,
            slurm_job_id="12345",
            slurm_step_id=f"step-{assigned_node}",
            slurm_step_nodelist=assigned_node,
            slurm_step_num_nodes="1",
            slurm_node_id="0",
            slurm_process_id="0",
        )

    monkeypatch.setattr(full_loop_compare, "_raw_step_identity", fake_step_identity)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-7")
    monkeypatch.setenv("FULL_LOOP_PROBE_RESULTS_ROOT", str(results_root))
    monkeypatch.setenv("FULL_LOOP_PROBE_HOLD_SECONDS", "0.3")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            lane: executor.submit(
                full_loop_compare.main,
                ["run-lane", "--output-root", str(output_root), "--lane", lane],
            )
            for lane in THREE_LANES
        }
        return {lane: future.result(timeout=15) for lane, future in futures.items()}


def _adjudicate_three_lane_probe(output_root: Path) -> int:
    return full_loop_compare.main(
        [
            "adjudicate",
            "--output-root",
            str(output_root),
            "--native-cpu-returncode",
            "0",
            "--jax-cpu-returncode",
            "0",
            "--jax-gpu-returncode",
            "0",
        ]
    )


def _valid_gpu_memory_hook(output_root: Path) -> dict[str, object]:
    execution = json.loads(
        (output_root / "jax_gpu" / "execution.json").read_text(encoding="utf-8")
    )
    environment = execution["environment"]
    assert isinstance(environment, dict)
    cuda_visible_devices = environment["CUDA_VISIBLE_DEVICES"]
    assert isinstance(cuda_visible_devices, str)
    return {
        "schema_version": 1,
        "metric": "nvidia-smi compute-process used_memory",
        "unit": "MiB",
        "pair": output_root.name,
        "node": THREE_NODES["jax_gpu"],
        "slurm_step_id": execution["slurm_step_id"],
        "cuda_visible_devices": cuda_visible_devices,
        "gpu_inventory": [
            {
                "index": 0,
                "name": "NVIDIA H100 80GB HBM3",
                "uuid": "GPU-probe",
                "memory_total_mib": 81920,
                "driver_version": "575.57.08",
            }
        ],
        "sample_count": 3,
        "gpu_uuids": ["GPU-probe"],
        "process_ids": [4242],
        "first_sample_at_utc": execution["started_at_utc"],
        "last_sample_at_utc": execution["ended_at_utc"],
        "maximum_used_memory_mib": 512,
        "sampler_queries": {
            "query_count": 3,
            "successful_query_count": 3,
            "failure_count": 0,
            "all_succeeded": True,
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


def test_three_lane_commands_share_interpreter_and_profile_with_exact_backends() -> (
    None
):
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
    native_cpu = build_lane_command(
        args,
        lane="native_cpu",
        run_dir=Path("/output/native_cpu"),
        run_config_sha256=RUN_CONFIG_SHA256,
    )
    jax_cpu = build_lane_command(
        args,
        lane="jax_cpu",
        run_dir=Path("/output/jax_cpu"),
        run_config_sha256=RUN_CONFIG_SHA256,
    )
    jax_gpu = build_lane_command(
        args,
        lane="jax_gpu",
        run_dir=Path("/output/jax_gpu"),
        run_config_sha256=RUN_CONFIG_SHA256,
    )

    commands = (native_cpu, jax_cpu, jax_gpu)
    assert {command[0] for command in commands} == {"/python"}
    assert {
        command[command.index("--objective-profile") + 1] for command in commands
    } == {"common-seven-term"}
    assert "--output-root" in native_cpu
    assert "--run-dir" not in native_cpu
    assert "--backend" not in native_cpu
    assert jax_cpu[jax_cpu.index("--backend") + 1] == "cpu"
    assert jax_cpu[jax_cpu.index("--platform") + 1] == "cpu"
    assert jax_gpu[jax_gpu.index("--backend") + 1] == "jax"
    assert jax_gpu[jax_gpu.index("--platform") + 1] == "cuda"
    for jax_command in (jax_cpu, jax_gpu):
        assert "--run-dir" in jax_command
        assert "--weight-poloidal-extent" in jax_command
        assert "--no-current-penalties" in jax_command
        assert "--no-width" in jax_command
    assert "--weight-poloidal-extent" not in native_cpu
    assert all("--sign-g" not in command for command in commands)


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
def test_jax_gpu_lane_requires_exactly_one_cuda_selector(
    visible_devices: str,
) -> None:
    with pytest.raises(RuntimeError, match="exactly one Slurm-assigned"):
        lane_environment("jax_gpu", {"CUDA_VISIBLE_DEVICES": visible_devices})

    environment = lane_environment("jax_gpu", {"CUDA_VISIBLE_DEVICES": "0"})
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["JAX_PLATFORMS"] == "cuda"


@pytest.mark.parametrize("lane", ("native_cpu", "jax_cpu"))
def test_cpu_lane_hides_parent_gpu_visibility(lane: str) -> None:
    environment = lane_environment(
        lane,
        {
            "CUDA_VISIBLE_DEVICES": "GPU-parent-0,GPU-parent-1",
            "JAX_PLATFORMS": "cuda",
            "SIMSOPT_JAX_PLATFORM": "cuda",
            "SIMSOPT_JAX_BACKEND": "cuda",
        },
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["JAX_PLATFORMS"] == "cpu"
    assert environment["SIMSOPT_JAX_PLATFORM"] == "cpu"
    assert environment["SIMSOPT_JAX_BACKEND"] == "cpu"


def test_comparator_cli_exposes_only_three_canonical_operational_lanes() -> None:
    parser = full_loop_compare._cli_parser()
    for lane in THREE_LANES:
        arguments = parser.parse_args(
            ["run-lane", "--output-root", "/pair", "--lane", lane]
        )
        assert arguments.command == "run-lane"
        assert arguments.lane == lane

    for legacy_lane in ("cpu", "jax"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["run-lane", "--output-root", "/pair", "--lane", legacy_lane]
            )

    adjudicate = parser.parse_args(
        [
            "adjudicate",
            "--output-root",
            "/pair",
            "--native-cpu-returncode",
            "3",
            "--jax-cpu-returncode",
            "5",
            "--jax-gpu-returncode",
            "7",
        ]
    )
    assert (
        adjudicate.native_cpu_returncode,
        adjudicate.jax_cpu_returncode,
        adjudicate.jax_gpu_returncode,
    ) == (3, 5, 7)


def test_prepare_manifest_freezes_exact_three_node_barrier_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root, _fake_python = _prepare_three_lane_probe(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    manifest = json.loads(
        (output_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["execution_topology"] == {
        "policy": "concurrent-different-nodes",
        "assigned_nodes": THREE_NODES,
        "barrier": {
            "protocol": "shared-ready-files-v1",
            "participants": list(THREE_LANES),
        },
    }
    assert (output_root / "run_manifest.sha256").is_file()
    assert (output_root / "barrier").is_dir()


def test_prepare_rejects_colocated_nodes_before_writing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    colocated = dict(THREE_NODES)
    colocated["jax_cpu"] = colocated["native_cpu"]

    with pytest.raises(ContractError, match="distinct"):
        _prepare_three_lane_probe(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            nodes=colocated,
        )

    assert not (tmp_path / "pair" / "run_manifest.json").exists()


def test_run_lane_preflight_failure_writes_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root, _fake_python = _prepare_three_lane_probe(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    manifest = json.loads(
        (output_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    Path(manifest["input_paths"]["surface"]).unlink()

    with pytest.raises(ContractError, match="Prepared surface input is missing"):
        full_loop_compare.main(
            [
                "run-lane",
                "--output-root",
                str(output_root),
                "--lane",
                "native_cpu",
            ]
        )

    invocation = json.loads(
        (output_root / "native_cpu" / "invocation.json").read_text(encoding="utf-8")
    )
    execution = json.loads(
        (output_root / "native_cpu" / "execution.json").read_text(encoding="utf-8")
    )
    assert invocation["failure"]["type"] == "ContractError"
    assert execution["status"] == "failed"
    assert execution["failure"] == invocation["failure"]
    assert execution["runner_ended_at_utc"] == invocation["runner_ended_at_utc"]


def test_adjudicate_missing_prepared_root_writes_failed_comparison(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "missing-pair"

    returncode = full_loop_compare.main(
        [
            "adjudicate",
            "--output-root",
            str(output_root),
            "--native-cpu-returncode",
            "125",
            "--jax-cpu-returncode",
            "125",
            "--jax-gpu-returncode",
            "125",
        ]
    )

    assert returncode == 1
    comparison = json.loads(
        (output_root / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["status"] == "failed"
    assert comparison["claim_ready"] is False
    assert comparison["performance"] is None
    assert "Prepared run manifest or digest is missing" in comparison["failures"][0]


def test_run_lane_cli_uses_real_three_party_barrier_and_isolates_environments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root, _fake_python = _prepare_three_lane_probe(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    returncodes = _run_three_lane_probe(
        output_root=output_root,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert returncodes == {lane: 0 for lane in THREE_LANES}
    for lane, assigned_node in THREE_NODES.items():
        invocation = json.loads(
            (output_root / lane / "invocation.json").read_text(encoding="utf-8")
        )
        execution = json.loads(
            (output_root / lane / "execution.json").read_text(encoding="utf-8")
        )
        expected_peers = set(THREE_LANES) - {lane}
        assert invocation["assigned_node"] == assigned_node
        assert invocation["actual_node"] == assigned_node
        assert invocation["slurm_job_id"] == "12345"
        assert execution["slurm_job_id"] == "12345"
        assert set(invocation["barrier_peer_observed_at_utc"]) == expected_peers
        assert set(execution["barrier_peer_observed_at_utc"]) == expected_peers
        assert invocation["barrier_peer_slurm_job_ids"] == {
            peer_lane: "12345" for peer_lane in expected_peers
        }
        assert execution["barrier_peer_slurm_job_ids"] == {
            peer_lane: "12345" for peer_lane in expected_peers
        }
        driver_probe = json.loads(
            (output_root / lane / "driver_probe.json").read_text(encoding="utf-8")
        )
        environment = driver_probe["environment"]
        if lane == "jax_gpu":
            assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-7"
            assert environment["JAX_PLATFORMS"] == "cuda"
        else:
            assert environment["CUDA_VISIBLE_DEVICES"] == ""
            assert environment["JAX_PLATFORMS"] == "cpu"

    gpu_memory_hook = _valid_gpu_memory_hook(output_root)
    (output_root / "jax_gpu" / "gpu_process_memory.json").write_text(
        json.dumps(gpu_memory_hook, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    adjudicate_returncode = _adjudicate_three_lane_probe(output_root)
    assert adjudicate_returncode == 0
    comparison = json.loads(
        (output_root / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["claim_ready"] is True
    assert comparison["performance"]["gpu_process_memory"] == gpu_memory_hook
    for lane in THREE_LANES:
        lane_evidence = comparison["lanes"][lane]
        assert lane_evidence["slurm_job_id"] == "12345"
        assert lane_evidence["barrier_peer_slurm_job_ids"] == {
            peer_lane: "12345" for peer_lane in set(THREE_LANES) - {lane}
        }
    assert comparison["execution_topology"] == {
        "policy": "concurrent-different-nodes",
        "assigned_nodes": THREE_NODES,
        "barrier": {
            "protocol": "shared-ready-files-v1",
            "participants": list(THREE_LANES),
        },
    }


@pytest.mark.parametrize(
    ("failure_case", "expected_failure"),
    (
        ("missing", "JAX-GPU process-memory evidence is missing"),
        ("zero-sample", "sample_count must be positive"),
        ("failed-query", "sampler queries did not all succeed"),
        ("wrong-node", "node differs from execution"),
    ),
)
def test_adjudication_requires_claim_grade_gpu_memory_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_case: str,
    expected_failure: str,
) -> None:
    output_root, _fake_python = _prepare_three_lane_probe(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert _run_three_lane_probe(
        output_root=output_root,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    ) == {lane: 0 for lane in THREE_LANES}
    hook_path = output_root / "jax_gpu" / "gpu_process_memory.json"
    if failure_case != "missing":
        hook = _valid_gpu_memory_hook(output_root)
        if failure_case == "zero-sample":
            hook["sample_count"] = 0
        elif failure_case == "failed-query":
            sampler_queries = hook["sampler_queries"]
            assert isinstance(sampler_queries, dict)
            sampler_queries.update(
                {
                    "successful_query_count": 2,
                    "failure_count": 1,
                    "all_succeeded": False,
                }
            )
        else:
            hook["node"] = "nid999999"
        hook_path.write_text(
            json.dumps(hook, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    returncode = _adjudicate_three_lane_probe(output_root)

    assert returncode == 1
    comparison = json.loads(
        (output_root / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["status"] == "failed"
    assert comparison["claim_ready"] is False
    assert comparison["performance"] is None
    assert expected_failure in comparison["failures"][0]


def test_adjudication_rejects_lanes_from_different_slurm_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root, _fake_python = _prepare_three_lane_probe(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert _run_three_lane_probe(
        output_root=output_root,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    ) == {lane: 0 for lane in THREE_LANES}
    for artifact_name in ("invocation.json", "execution.json"):
        artifact_path = output_root / "jax_cpu" / artifact_name
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["slurm_job_id"] = "54321"
        artifact["barrier_peer_slurm_job_ids"] = {
            peer_lane: "54321" for peer_lane in set(THREE_LANES) - {"jax_cpu"}
        }
        artifact_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output_root / "jax_gpu" / "gpu_process_memory.json").write_text(
        json.dumps(_valid_gpu_memory_hook(output_root), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    returncode = _adjudicate_three_lane_probe(output_root)

    assert returncode == 1
    comparison = json.loads(
        (output_root / "comparison.json").read_text(encoding="utf-8")
    )
    assert "execution_topology.slurm_job_ids_not_equal" in comparison["failures"]


@pytest.mark.parametrize(
    ("failure_case", "expected_failure"),
    (
        ("missing", "lane_evidence.jax_cpu"),
        ("mismatched", "lane_evidence.jax_cpu"),
        ("colocated", "lane_evidence.jax_cpu"),
        ("nonzero", "step_returncode.jax_gpu"),
    ),
)
def test_adjudication_fails_closed_on_incomplete_or_invalid_lane_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_case: str,
    expected_failure: str,
) -> None:
    output_root, _fake_python = _prepare_three_lane_probe(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert _run_three_lane_probe(
        output_root=output_root,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    ) == {lane: 0 for lane in THREE_LANES}

    jax_cpu_execution_path = output_root / "jax_cpu" / "execution.json"
    if failure_case == "missing":
        jax_cpu_execution_path.unlink()
    elif failure_case in {"mismatched", "colocated"}:
        execution = json.loads(jax_cpu_execution_path.read_text(encoding="utf-8"))
        if failure_case == "mismatched":
            execution["actual_node"] = "nid999999"
        else:
            execution["assigned_node"] = THREE_NODES["native_cpu"]
            execution["actual_node"] = THREE_NODES["native_cpu"]
            execution["slurm_step_nodelist"] = THREE_NODES["native_cpu"]
        jax_cpu_execution_path.write_text(
            json.dumps(execution, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    jax_gpu_returncode = "9" if failure_case == "nonzero" else "0"
    adjudicate_returncode = full_loop_compare.main(
        [
            "adjudicate",
            "--output-root",
            str(output_root),
            "--native-cpu-returncode",
            "0",
            "--jax-cpu-returncode",
            "0",
            "--jax-gpu-returncode",
            jax_gpu_returncode,
        ]
    )

    assert adjudicate_returncode == 1
    comparison = json.loads(
        (output_root / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["status"] == "failed"
    assert comparison["claim_ready"] is False
    assert any(expected_failure in failure for failure in comparison["failures"])


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


def test_launcher_venv_root_uses_shared_scratch_and_uid_scope(
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
    environment["SCRATCH"] = str(tmp_path / "shared-scratch")
    environment["TMPDIR"] = str(tmp_path / "node-local")
    assert resolve_venv_root(environment) == str(
        tmp_path / "shared-scratch" / f"simsopt-full-loop-envs-{os.getuid()}"
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
    isolation_index = source.index(isolation)
    assert source.index('if [[ "${1:-}" == "__run-lane-step" ]]') < isolation_index
    assert isolation_index < source.index('REPO_ROOT="${REPO_ROOT:-')
    assert isolation_index < source.index('python -m venv "${VENV_DIR}"')

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
    ("profile", "execution_mode", "expected_returncode"),
    (
        ("pilot", "slurm-step", 0),
        ("full", "slurm-step", 0),
        ("pilot", "direct", 2),
        ("full", "direct", 2),
        ("full", "sequential", 2),
        ("pilot", "unknown", 2),
    ),
)
def test_launcher_has_no_direct_or_sequential_fallback(
    profile: str,
    execution_mode: str,
    expected_returncode: int,
) -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    function = re.search(
        r"(?ms)^validate_pair_execution_mode\(\) \{.*?^\}\n",
        source,
    )
    assert function is not None
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
    )

    assert completed.returncode == expected_returncode


def test_launcher_execution_mode_guidance_names_only_supported_mode() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    function = re.search(
        r"(?ms)^validate_pair_execution_mode\(\) \{.*?^\}\n",
        source,
    )
    assert function is not None
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'{function.group(0)}\nvalidate_pair_execution_mode "$1" "$2"',
            "validate-pair-execution-mode-guidance-test",
            "full",
            "unknown",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "PAIR_EXECUTION_MODE must be slurm-step; got unknown" in completed.stderr
    assert "or direct" not in completed.stderr


@pytest.mark.parametrize(
    ("allocated_nodes", "expected_returncode"),
    (
        (("nid001", "nid002", "nid003"), 0),
        (("nid001", "nid002"), 2),
        (("nid001", "nid002", "nid002"), 2),
        (("nid001", "nid002", "nid003", "nid004"), 2),
    ),
)
def test_launcher_requires_exactly_three_distinct_allocated_nodes(
    allocated_nodes: tuple[str, ...],
    expected_returncode: int,
) -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    function = re.search(
        r"(?ms)^validate_allocated_nodes\(\) \{.*?^\}\n",
        source,
    )
    assert function is not None
    completed = subprocess.run(
        [
            "bash",
            "-c",
            (f'{function.group(0)}\nALLOCATED_NODES=("$@")\nvalidate_allocated_nodes'),
            "validate-allocated-nodes-test",
            *allocated_nodes,
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == expected_returncode
    assert "#SBATCH -N 3" in source
    assert "#SBATCH --ntasks=3" in source


def test_launcher_rotates_all_three_node_roles_between_pairs() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    function = re.search(r"(?ms)^assign_pair_nodes\(\) \{.*?^\}\n", source)
    assert function is not None
    completed = subprocess.run(
        [
            "bash",
            "-c",
            (
                f"{function.group(0)}\n"
                "ALLOCATED_NODES=(nid001 nid002 nid003)\n"
                "for pair_index in 1 2 3 4; do\n"
                '    assign_pair_nodes "${pair_index}"\n'
                "    printf '%s,%s,%s\\n' \"${native_cpu_node}\" "
                '"${jax_cpu_node}" "${jax_gpu_node}"\n'
                "done"
            ),
            "assign-pair-nodes-test",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert completed.stdout.splitlines() == [
        "nid001,nid002,nid003",
        "nid002,nid003,nid001",
        "nid003,nid001,nid002",
        "nid001,nid002,nid003",
    ]


def test_launcher_starts_three_exactly_pinned_steps_before_waiting(
    tmp_path: Path,
) -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    function = re.search(r"(?ms)^launch_pair_lanes\(\) \{.*?^\}\n", source)
    assert function is not None
    pair_dir = tmp_path / "pair"
    barrier_root = tmp_path / "fake-srun-barrier"
    pair_dir.mkdir()
    barrier_root.mkdir()
    completed = subprocess.run(
        [
            "bash",
            "-c",
            (
                "set -euo pipefail\n"
                "shopt -s nullglob\n"
                f"{function.group(0)}\n"
                "srun() {\n"
                "    local lane=''\n"
                "    local argument\n"
                '    for argument in "$@"; do\n'
                '        case "${argument}" in\n'
                '            native_cpu|jax_cpu|jax_gpu) lane="${argument}" ;;\n'
                "        esac\n"
                "    done\n"
                '    [[ -n "${lane}" ]]\n'
                '    printf \'%s\\n\' "$@" > "${BARRIER_ROOT}/${lane}.argv"\n'
                '    touch "${BARRIER_ROOT}/${lane}.ready"\n'
                "    while true; do\n"
                '        local -a ready=("${BARRIER_ROOT}"/*.ready)\n'
                "        ((${#ready[@]} == 3)) && break\n"
                "        sleep 0.01\n"
                "    done\n"
                "}\n"
                f'launch_pair_lanes "$PAIR_DIR" {THREE_NODES["native_cpu"]} '
                f"{THREE_NODES['jax_cpu']} {THREE_NODES['jax_gpu']}\n"
                'wait "${NATIVE_CPU_STEP_PID}"\n'
                'wait "${JAX_CPU_STEP_PID}"\n'
                'wait "${JAX_GPU_STEP_PID}"\n'
            ),
            "launch-three-lanes-test",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
        env={
            **os.environ,
            "PAIR_DIR": str(pair_dir),
            "BARRIER_ROOT": str(barrier_root),
            "SLURM_CPUS_PER_TASK": "32",
            "LAUNCHER_PATH": "/shared/launcher.slurm",
            "VENV_DIR": "/shared/venv",
            "COMPARATOR_PATH": "/shared/comparator.py",
            "GPU_PROCESS_MEMORY_CSV": "/shared/process.csv",
            "GPU_SAMPLER_QUERY_CSV": "/shared/queries.csv",
            "GPU_SAMPLER_ERROR_LOG": "/shared/errors.log",
        },
    )
    assert completed.returncode == 0

    lane_arguments = {
        lane: (barrier_root / f"{lane}.argv").read_text(encoding="utf-8").splitlines()
        for lane in THREE_LANES
    }
    for lane, node in THREE_NODES.items():
        arguments = lane_arguments[lane]
        assert arguments.count("--nodes=1") == 1
        assert arguments.count("--ntasks=1") == 1
        assert arguments.count(f"--nodelist={node}") == 1
        assert lane in arguments
    for lane in ("native_cpu", "jax_cpu"):
        assert "--gres=none" in lane_arguments[lane]
        assert "CUDA_VISIBLE_DEVICES=" in lane_arguments[lane]
        assert "--gpus-per-task=1" not in lane_arguments[lane]
    assert "--gpus-per-task=1" in lane_arguments["jax_gpu"]
    assert "--gres=none" not in lane_arguments["jax_gpu"]


def test_launcher_collects_all_three_lane_returncodes_independently() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    function = re.search(r"(?ms)^wait_for_pair_lanes\(\) \{.*?^\}\n", source)
    assert function is not None
    completed = subprocess.run(
        [
            "bash",
            "-c",
            (
                f"{function.group(0)}\n"
                "(exit 3) & NATIVE_CPU_STEP_PID=$!\n"
                "(exit 5) & JAX_CPU_STEP_PID=$!\n"
                "(exit 7) & JAX_GPU_STEP_PID=$!\n"
                "wait_for_pair_lanes\n"
                "printf '%s,%s,%s\\n' \"${native_cpu_returncode}\" "
                '"${jax_cpu_returncode}" "${jax_gpu_returncode}"'
            ),
            "wait-three-lanes-test",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert completed.stdout.strip() == "3,5,7"
    adjudication = source[source.index('"${COMPARATOR_PATH}" adjudicate') :]
    assert '--native-cpu-returncode "${native_cpu_returncode}"' in adjudication
    assert '--jax-cpu-returncode "${jax_cpu_returncode}"' in adjudication
    assert '--jax-gpu-returncode "${jax_gpu_returncode}"' in adjudication


def test_launcher_finishes_lane_local_gpu_hook_before_adjudication() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    wrapper = re.search(r"(?ms)^run_lane_step\(\) \{.*?^\}\n", source)
    assert wrapper is not None
    wrapper_source = wrapper.group(0)
    assert wrapper_source.index('wait "${gpu_monitor_pid}"') < wrapper_source.index(
        "if write_gpu_memory_hook"
    )
    assert 'pair_dir / "jax_gpu" / "gpu_process_memory.json"' in source

    pair_loop = source[source.index("FAILED_PAIRS=0") :]
    assert pair_loop.index("wait_for_pair_lanes") < pair_loop.index(
        '"${COMPARATOR_PATH}" adjudicate'
    )


def test_launcher_gpu_summary_null_timestamps_fail_evidence_without_crashing(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    pair_root = artifact_root / "pairs"
    execution_root = pair_root / "pair-01" / "jax_gpu"
    execution_root.mkdir(parents=True)
    (execution_root / "execution.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "started_at_utc": None,
                "ended_at_utc": None,
            }
        ),
        encoding="utf-8",
    )
    machine_root = artifact_root / "machine"
    machine_root.mkdir()
    process_csv = machine_root / "gpu_process_memory.csv"
    process_csv.write_text(
        "timestamp_utc,pair,node,gpu_uuid,pid,used_memory_mib\n"
        "2026-07-16T00:00:00+00:00,pair-01,nid1,GPU-1,123,512\n",
        encoding="utf-8",
    )
    query_csv = machine_root / "gpu_sampler_queries.csv"
    query_csv.write_text(
        "timestamp_utc,pair,node,status,exit_code\n"
        "2026-07-16T00:00:00+00:00,pair-01,nid1,ok,0\n",
        encoding="utf-8",
    )
    error_log = machine_root / "gpu_sampler_errors.log"
    error_log.write_text("", encoding="utf-8")
    summary_path = machine_root / "gpu_process_memory_summary.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _launcher_python_heredoc(
                'process_csv_path = Path(os.environ["GPU_PROCESS_MEMORY_CSV"])'
            ),
        ],
        check=False,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "GPU_PROCESS_MEMORY_CSV": str(process_csv),
            "GPU_SAMPLER_QUERY_CSV": str(query_csv),
            "GPU_SAMPLER_ERROR_LOG": str(error_log),
            "GPU_PROCESS_MEMORY_SUMMARY": str(summary_path),
            "PAIR_ROOT": str(pair_root),
            "PROFILE": "full",
            "PAIR_COUNT": "1",
        },
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pair_summary = summary["by_pair"]["pair-01"]
    assert summary["schema_version"] == 2
    assert summary["overall"]["sample_count"] == 1
    assert pair_summary["jax_gpu_execution_window_recorded"] is False
    assert pair_summary["jax_gpu_process_sample_count"] == 0
    assert summary["evidence_gate"] == {
        "profile": "full",
        "expected_pair_count": 1,
        "all_queries_succeeded": True,
        "every_pair_has_jax_gpu_process_sample": False,
        "passed": False,
        "required_for_claim": True,
    }


def test_launcher_overlap_probe_treats_null_timestamps_as_incomplete(
    tmp_path: Path,
) -> None:
    pair_dir = tmp_path / "pair-01"
    for lane in THREE_LANES:
        lane_dir = pair_dir / lane
        lane_dir.mkdir(parents=True)
        (lane_dir / "execution.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "started_at_utc": None,
                    "ended_at_utc": None,
                }
            ),
            encoding="utf-8",
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _launcher_python_heredoc('    print("false false 0.0")'),
        ],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "PAIR_DIR": str(pair_dir)},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "false false 0.0"


def test_launcher_aggregate_persists_failed_null_performance_comparison(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    pair_dir = artifact_root / "pairs" / "pair-01"
    pair_dir.mkdir(parents=True)
    (pair_dir / "pair_status.tsv").write_text(
        "\n".join(
            (
                "pair_execution_mode\tconcurrent-srun-steps",
                "adjudicated_status\tfailed",
                "native_cpu_node\tnid1",
                "jax_cpu_node\tnid2",
                "jax_gpu_node\tnid3",
                "all_lanes_overlapped\tfalse",
                "comparison_status\tfailed",
                "comparison_claim_ready\tfalse",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    failure = "lane_evidence.jax_gpu: execution status is failed"
    (pair_dir / "comparison.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "claim_ready": False,
                "performance": None,
                "parity": {
                    "required_failures": [failure],
                    "advisory_failures": [],
                },
                "failures": [failure],
            }
        ),
        encoding="utf-8",
    )
    machine_root = artifact_root / "machine"
    machine_root.mkdir()
    gpu_summary_path = machine_root / "gpu_process_memory_summary.json"
    gpu_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "by_pair": {},
                "evidence_gate": {
                    "profile": "full",
                    "expected_pair_count": 1,
                    "all_queries_succeeded": False,
                    "every_pair_has_jax_gpu_process_sample": False,
                    "passed": False,
                    "required_for_claim": True,
                },
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _launcher_python_heredoc(
                "successful_native_cpu_over_jax_cpu_speedups: list[float] = []"
            ),
        ],
        check=False,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "ARTIFACT_ROOT": str(artifact_root),
            "PROFILE": "full",
            "EVIDENCE_CLASS": "production",
            "PAIR_EXECUTION_MODE": "concurrent-srun-steps",
            "PAIR_COUNT": "1",
            "GPU_PROCESS_MEMORY_SUMMARY": str(gpu_summary_path),
            "ENVIRONMENT_LOCK_MANIFEST": str(
                artifact_root / "environment" / "manifest.json"
            ),
        },
    )

    assert completed.returncode == 1
    aggregate = json.loads(
        (artifact_root / "comparison.json").read_text(encoding="utf-8")
    )
    pair = aggregate["pairs"][0]
    assert aggregate["schema_version"] == 2
    assert aggregate["status"] == "failed"
    assert aggregate["claim_ready"] is False
    assert aggregate["measurement_design_valid"] is True
    assert aggregate["repeat_design_valid"] is False
    assert pair["performance"] is None
    assert pair["host_rss_kib"] is None
    assert pair["failures"] == [failure]
    assert pair["required_failures"] == [failure]


def test_launcher_single_triplet_is_claim_ready_without_claiming_repeats(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    pair_dir = artifact_root / "pairs" / "pair-01"
    pair_dir.mkdir(parents=True)
    (pair_dir / "pair_status.tsv").write_text(
        "\n".join(
            (
                "pair_execution_mode\tconcurrent-srun-steps",
                "adjudicated_status\tpassed",
                "native_cpu_node\tnid1",
                "jax_cpu_node\tnid2",
                "jax_gpu_node\tnid3",
                "all_lanes_overlapped\ttrue",
                "comparison_status\tpassed",
                "comparison_claim_ready\ttrue",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (pair_dir / "comparison.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "claim_ready": True,
                "performance": {
                    "speedups": {
                        "native_cpu_over_jax_cpu": 2.0,
                        "native_cpu_over_jax_gpu": 4.0,
                        "jax_cpu_over_jax_gpu": 2.0,
                    },
                    "wall_seconds": {
                        "native_cpu": 40.0,
                        "jax_cpu": 20.0,
                        "jax_gpu": 10.0,
                    },
                    "host_max_rss_kib": {
                        "native_cpu": 100,
                        "jax_cpu": 200,
                        "jax_gpu": 300,
                    },
                },
                "parity": {
                    "required_failures": [],
                    "advisory_failures": [],
                },
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    machine_root = artifact_root / "machine"
    machine_root.mkdir()
    gpu_summary_path = machine_root / "gpu_process_memory_summary.json"
    gpu_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "by_pair": {},
                "evidence_gate": {
                    "profile": "full",
                    "expected_pair_count": 1,
                    "all_queries_succeeded": True,
                    "every_pair_has_jax_gpu_process_sample": True,
                    "passed": True,
                    "required_for_claim": True,
                },
            }
        ),
        encoding="utf-8",
    )
    environment_root = artifact_root / "environment"
    environment_root.mkdir()
    lock_path = environment_root / "requirements.lock"
    lock_path.write_text("locked-dependency\n", encoding="utf-8")
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    environment_manifest_path = environment_root / "manifest.json"
    environment_manifest_path.write_text(
        json.dumps(
            {
                "lock_sha256": lock_sha256,
                "secure_install_verified": True,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _launcher_python_heredoc(
                "successful_native_cpu_over_jax_cpu_speedups: list[float] = []"
            ),
        ],
        check=False,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "ARTIFACT_ROOT": str(artifact_root),
            "PROFILE": "full",
            "EVIDENCE_CLASS": "production",
            "PAIR_EXECUTION_MODE": "concurrent-srun-steps",
            "PAIR_COUNT": "1",
            "GPU_PROCESS_MEMORY_SUMMARY": str(gpu_summary_path),
            "ENVIRONMENT_LOCK_MANIFEST": str(environment_manifest_path),
        },
    )

    assert completed.returncode == 0, completed.stderr
    aggregate = json.loads(
        (artifact_root / "comparison.json").read_text(encoding="utf-8")
    )
    assert aggregate["status"] == "passed"
    assert aggregate["claim_ready"] is True
    assert aggregate["measurement_design_valid"] is True
    assert aggregate["repeat_design_valid"] is False
    assert aggregate["measurement_class"] == "single_matched_triplet"
    assert (
        aggregate["measurement_interpretation"]
        == "one matched measurement; no statistical inference"
    )


def test_launcher_creates_pair_directory_before_adjudication() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    pair_loop = source[source.index("FAILED_PAIRS=0") :]
    prepare_index = pair_loop.index('"${COMPARATOR_PATH}" prepare')
    mkdir_index = pair_loop.index('mkdir -p "${pair_dir}"')
    adjudicate_index = pair_loop.index('"${COMPARATOR_PATH}" adjudicate')

    assert prepare_index < mkdir_index < adjudicate_index


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
        ("full", "1", 0),
        ("full", "2", 0),
        ("full", "3", 0),
        ("full", "4", 0),
        ("full", "0", 2),
        ("full", "invalid", 2),
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


def test_launcher_full_profile_uses_requested_solver_budgets() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    full_profile = re.search(r"(?ms)^    full\)\n(?P<body>.*?)^        ;;", source)

    assert full_profile is not None
    body = full_profile.group("body")
    assert 'PAIR_COUNT="${PAIR_COUNT:-1}"' in body
    assert 'OUTER_MAXITER="${OUTER_MAXITER:-1500}"' in body
    assert 'BOOZER_NEWTON_MAXITER="${BOOZER_NEWTON_MAXITER:-50}"' in body
