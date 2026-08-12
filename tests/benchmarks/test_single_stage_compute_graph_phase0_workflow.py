from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from benchmarks.landau_a100_qualification import (
    QUALIFICATION_SCHEMA_ID,
)
from benchmarks.landau_a100_qualification import (
    _canonical_json_bytes as landau_canonical_json_bytes,
)
from benchmarks.single_stage_compute_graph_isolated_launch import (
    normalize_static_timing_environment,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    canonical_json_bytes,
    canonical_sha256,
)
from benchmarks.single_stage_compute_graph_phase0_workflow import (
    _LOCAL_PROBE_SOURCE,
    NATIVE_REFERENCE_MODULE,
    CommandResult,
    Phase0WorkflowError,
    Phase0WorkflowInputs,
    _landau_qualification,
    build_phase0_workflow,
    write_phase0_workflow_plan,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    RoleRoot,
    load_snapshot_manifest,
    publish_immutable_snapshot,
)
from benchmarks.single_stage_compute_graph_snapshot_publish import PUBLICATION_SCHEMA_ID
from benchmarks.single_stage_compute_graph_specimen import (
    SCHEMA_ID as SPECIMEN_SCHEMA_ID,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _json(path: Path, document: object) -> Path:
    return _write(path, canonical_json_bytes(document))


def _chain(tmp_path: Path) -> tuple[Phase0WorkflowInputs, dict[str, object]]:
    source = tmp_path / "source"
    package_paths = {}
    for package in ("simsopt", "simsopt_jax", "simsopt_jax_adapters"):
        package_paths[package] = _write(
            source / "src" / package / "__init__.py", f"NAME={package!r}\n".encode()
        )
    native = _write(source / "src" / "simsoptpp.py", b"NATIVE=True\n")
    benchmark_relative_path = f"{NATIVE_REFERENCE_MODULE.replace('.', '/')}.py"
    benchmark = _write(source / benchmark_relative_path, b"VALUE=1\n")
    test = _write(source / "tests" / "test_runner.py", b"def test_value(): pass\n")
    config = _write(source / "config" / "policy.txt", b"lineax==0.1.1\n")

    specimen_root = source / "phase0-specimen"
    arrays = {}
    for name, values in (
        ("axis_dofs", np.asarray([1.0], dtype=np.float64)),
        ("coil_dofs", np.asarray([2.0], dtype=np.float64)),
        ("surface_dofs", np.asarray([3.0], dtype=np.float64)),
    ):
        array_path = specimen_root / "input_bundle" / "inputs" / f"{name}.npy"
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(array_path, values, allow_pickle=False)
        arrays[name] = {
            "path": f"inputs/{name}.npy",
            "sha256": _sha256(array_path.read_bytes()),
        }
    input_bundle = {
        "case_id": "synthetic",
        "scale": "native_default",
        "input_fingerprint": "1" * 64,
        "configuration_fingerprint": "2" * 64,
        "configuration": {},
        "arrays": arrays,
    }
    bundle_path = _json(
        specimen_root / "input_bundle" / "input_bundle.json", input_bundle
    )
    candidate_path = specimen_root / "changed_state_candidate.npy"
    candidate = np.arange(461, dtype=np.float64)
    np.save(candidate_path, candidate, allow_pickle=False)
    parameter_sha256 = _sha256(
        np.ascontiguousarray(candidate, dtype=np.dtype("<f8")).tobytes()
    )
    specimen = {
        "specimen_id": "native-single-stage-changed-state-c0-v1",
        "input_bundle_sha256": _sha256(bundle_path.read_bytes()),
        "parameter_sha256": parameter_sha256,
        "state_dimension": 255,
        "coil_dof_count": 461,
        "grids": {
            "inner_surface_points": 169,
            "non_qs_surface_points": 1600,
            "physical_coil_contributions": 18,
            "quadrature_nodes": 250,
        },
        "weights": {"iota": 1.0},
        "tolerances": {"inner": 1e-12},
        "solver_graph_id": "c0-current-jvp-incremental-gmres",
        "solver_graph_sha256": "3" * 64,
    }
    specimen_document = {
        "schema_id": SPECIMEN_SCHEMA_ID,
        "specimen": specimen,
        "specimen_sha256": canonical_sha256(specimen),
        "input_bundle": {
            "relative_path": "input_bundle",
            "input_fingerprint": "1" * 64,
            "configuration_fingerprint": "2" * 64,
        },
        "candidate": {
            "relative_path": "changed_state_candidate.npy",
            "file_sha256": _sha256(candidate_path.read_bytes()),
            "dtype": "float64",
            "shape": [461],
            "parameter_sha256": parameter_sha256,
            "baseline_parameter_sha256": "4" * 64,
            "differs_from_baseline": True,
            "generator": "fixture",
        },
        "solver_graph": {"variant": "C0"},
        "effective_policies": {
            "dense_batch_width": 8,
            "point_chunk_size": None,
            "coil_chunk_size": None,
            "quadrature_block_sizes": [128, 122],
        },
    }
    _json(specimen_root / "specimen.json", specimen_document)

    snapshot = tmp_path / "snapshot"
    publication = publish_immutable_snapshot(
        snapshot,
        (
            RoleRoot("execution_source", source / "src" / "simsopt", "src/simsopt"),
            RoleRoot(
                "execution_source", source / "src" / "simsopt_jax", "src/simsopt_jax"
            ),
            RoleRoot(
                "execution_source",
                source / "src" / "simsopt_jax_adapters",
                "src/simsopt_jax_adapters",
            ),
            RoleRoot("configuration", config, "config/policy.txt"),
            RoleRoot("configuration", specimen_root, "phase0-specimen"),
            RoleRoot("benchmark", benchmark, benchmark_relative_path),
            RoleRoot("test", test, "tests/test_runner.py"),
            RoleRoot("native_extension", native, "src/simsoptpp.py"),
        ),
    )
    state = {
        "repository_commit": "5" * 40,
        "git_status_short": ["?? synthetic"],
        "tracked_diff_sha256": "6" * 64,
        "untracked_manifest_sha256": "7" * 64,
    }
    worktree = {**state, "source_state_sha256": canonical_sha256(state)}
    native_entry = next(
        entry for entry in publication.entries if entry.role == "native_extension"
    )
    publication_document = {
        "schema_id": PUBLICATION_SCHEMA_ID,
        "repository_root": str(tmp_path / "repository"),
        "snapshot_root": str(snapshot.absolute()),
        "snapshot_manifest_sha256": publication.manifest_sha256,
        "cross_host_source_sha256": canonical_sha256(
            [
                entry.to_json()
                for entry in publication.entries
                if entry.role != "native_extension"
                and entry.relative_path != "benchmarks/landau_a100_overlay_lock.txt"
            ]
        ),
        "native_extension": native_entry.to_json(),
        "worktree": worktree,
    }
    publication_path = _json(tmp_path / "publication.json", publication_document)
    entry_by_path = {entry.relative_path: entry for entry in publication.entries}
    bindings = []
    for module, relative_path in (
        ("simsopt", "src/simsopt/__init__.py"),
        ("simsopt_jax", "src/simsopt_jax/__init__.py"),
        ("simsopt_jax_adapters", "src/simsopt_jax_adapters/__init__.py"),
        ("simsoptpp", "src/simsoptpp.py"),
    ):
        entry = entry_by_path[relative_path]
        bindings.append(
            {
                "module": module,
                "relative_path": relative_path,
                "size_bytes": entry.size_bytes,
                "sha256": entry.sha256,
            }
        )
    attestation = {
        "schema_id": "single-stage-compute-graph-import-attestation-v1",
        "state": "pass",
        "snapshot_manifest_sha256": publication.manifest_sha256,
        "interpreter_path": str(Path(os.path.abspath(sys.executable))),
        "python_version": sys.version,
        "bindings": bindings,
    }
    attestation_path = _json(tmp_path / "attestation.json", attestation)
    inputs = Phase0WorkflowInputs(
        snapshot_root=snapshot,
        publication_path=publication_path,
        import_attestation_path=attestation_path,
        interpreter=Path(sys.executable).absolute(),
        output_root=tmp_path / "c0-output",
        compilation_cache_directory=tmp_path / "jax-cache",
        native_reference_path=tmp_path / "native-reference.json",
        lane_id="rtx5090",
    )
    return inputs, {
        "bindings": bindings,
        "snapshot": snapshot,
        "publication": publication_document,
    }


def _runner(context: dict[str, object]):
    native_path = next(
        binding["relative_path"]
        for binding in context["bindings"]
        if binding["module"] == "simsoptpp"
    )

    def run(argv, environment):
        if argv[0] == "nvidia-smi":
            return CommandResult(
                0, "GPU-test, NVIDIA GeForce RTX 5090, 32607, 575.0\n", ""
            )
        probe = {
            "schema_id": "single-stage-compute-graph-local-probe-v1",
            "interpreter": {
                "entrypoint_path": str(Path(os.path.abspath(sys.executable))),
                "target_path": str(Path(sys.executable).resolve()),
                "target_size": Path(sys.executable).resolve().stat().st_size,
                "target_sha256": _sha256(Path(sys.executable).resolve().read_bytes()),
                "prefix": sys.prefix,
                "base_prefix": sys.base_prefix,
            },
            "jax_version": "0.10.0",
            "jaxlib_version": "0.10.0",
            "jax_backend": "gpu",
            "x64_enabled": True,
            "devices": [{"kind": "NVIDIA GeForce RTX 5090", "platform": "gpu"}],
            "cuda_runtime": "CUDA 12.9",
            "resolved_cuda_libraries": ["/usr/lib/libcuda.so.1"],
            "native_extension_path": str(context["snapshot"] / native_path),
            "strict_transfer_smoke": {
                "guard": "disallow",
                "dtype": "float64",
                "shape": [],
                "finite": True,
            },
            "packages": {"jax": "0.10.0", "jaxlib": "0.10.0", "lineax": "0.1.1"},
            "static_timing_environment": normalize_static_timing_environment(
                environment
            ),
            "effective_numerical_policies": {
                "dense_batch_width": 8,
                "point_chunk_size": None,
                "coil_chunk_size": None,
                "quadrature_block_sizes": [128, 122],
            },
        }
        return CommandResult(0, json.dumps(probe), "")

    return run


def test_builds_canonical_native_invocation_receipt_and_runner_spec(
    tmp_path: Path,
) -> None:
    inputs, context = _chain(tmp_path)

    plan = build_phase0_workflow(
        inputs, environment={"PATH": os.environ["PATH"]}, runner=_runner(context)
    )
    write_phase0_workflow_plan(tmp_path / "plan", plan)

    assert plan.runner_spec["receipt_template"] == plan.receipt_template
    assert plan.native_reference_launch.argv[:4] == (
        str(Path(sys.executable).absolute()),
        "-P",
        "-s",
        "-c",
    )
    assert (
        "benchmarks.single_stage_compute_graph_native_reference"
        in plan.native_reference_launch.argv
    )
    runtime_contract_index = plan.native_reference_launch.argv.index(
        "--runtime-contract-json"
    )
    runtime_contract = json.loads(
        plan.native_reference_launch.argv[runtime_contract_index + 1]
    )
    assert runtime_contract == {
        "runtime": plan.runner_spec["provenance"]["runtime"],
        "static_environment": plan.runner_spec["provenance"]["environment"],
        "route_environment": {
            "JAX_COMPILATION_CACHE_DIR": str(
                inputs.compilation_cache_directory.resolve()
            )
        },
        "policies": plan.runner_spec["provenance"]["policies"],
        "expected_runtime_identity_sha256": plan.document["runtime_identity_sha256"],
    }
    assert plan.receipt_template["specimen_sha256"] == plan.document["specimen_sha256"]
    assert plan.runner_spec["native_reference_path"] == str(
        inputs.native_reference_path.absolute()
    )
    assert plan.qualification == plan.receipt_template["lanes"][0]["qualification"]
    assert plan.runtime_provenance == plan.runner_spec["provenance"]
    assert plan.device_probe["gpu"]["uuid"] == "GPU-test"
    standalone = plan.document["standalone_evidence"]
    assert standalone == {
        "qualification": {
            "relative_path": "qualification.json",
            "sha256": canonical_sha256(plan.qualification),
        },
        "device_probe": {
            "relative_path": "device-probe.json",
            "sha256": canonical_sha256(plan.device_probe),
        },
        "runtime_provenance": {
            "relative_path": "runtime-provenance.json",
            "sha256": canonical_sha256(plan.runtime_provenance),
        },
    }
    for name in (
        "workflow.json",
        "phase0-receipt-template.json",
        "c0-runner-spec.json",
        "qualification.json",
        "device-probe.json",
        "runtime-provenance.json",
    ):
        path = tmp_path / "plan" / name
        assert path.read_bytes() == canonical_json_bytes(json.loads(path.read_bytes()))


def test_a100_requires_standalone_qualification_input(tmp_path: Path) -> None:
    inputs, context = _chain(tmp_path)
    a100_inputs = replace(inputs, lane_id="a100")

    with pytest.raises(Phase0WorkflowError, match="requires Landau qualification"):
        build_phase0_workflow(a100_inputs, runner=_runner(context))


@pytest.mark.parametrize(
    "artifact",
    ("qualification", "device_probe", "runtime_provenance"),
)
def test_writer_rejects_tampered_standalone_evidence_before_creation(
    tmp_path: Path,
    artifact: str,
) -> None:
    inputs, context = _chain(tmp_path)
    plan = build_phase0_workflow(inputs, runner=_runner(context))
    destination = tmp_path / "tampered-plan"
    if artifact == "qualification":
        tampered = dict(plan.qualification)
        attempted = dict(tampered["attempted_identity"])
        attempted["gpu_uuid"] = "GPU-other"
        tampered["attempted_identity"] = attempted
        plan = replace(plan, qualification=tampered)
    elif artifact == "device_probe":
        tampered = dict(plan.device_probe)
        native = dict(tampered["native_binary"])
        native["sha256"] = "f" * 64
        tampered["native_binary"] = native
        plan = replace(plan, device_probe=tampered)
    else:
        tampered = dict(plan.runtime_provenance)
        allocation = dict(tampered["allocation"])
        allocation["gpu_uuid"] = "GPU-other"
        tampered["allocation"] = allocation
        plan = replace(plan, runtime_provenance=tampered)

    with pytest.raises(Phase0WorkflowError, match="standalone"):
        write_phase0_workflow_plan(destination, plan)
    assert not destination.exists()


def test_writer_rejects_mismatched_standalone_reference_before_creation(
    tmp_path: Path,
) -> None:
    inputs, context = _chain(tmp_path)
    plan = build_phase0_workflow(inputs, runner=_runner(context))
    document = dict(plan.document)
    references = dict(document["standalone_evidence"])
    qualification = dict(references["qualification"])
    qualification["sha256"] = "f" * 64
    references["qualification"] = qualification
    document["standalone_evidence"] = references
    plan = replace(plan, document=document)
    destination = tmp_path / "mismatched-plan"

    with pytest.raises(Phase0WorkflowError, match="references drifted"):
        write_phase0_workflow_plan(destination, plan)
    assert not destination.exists()


def test_writer_rejects_output_root_collision(tmp_path: Path) -> None:
    inputs, context = _chain(tmp_path)
    plan = build_phase0_workflow(inputs, runner=_runner(context))
    destination = tmp_path / "existing-plan"
    destination.mkdir()
    sentinel = _write(destination / "sentinel", b"preserve")

    with pytest.raises(FileExistsError):
        write_phase0_workflow_plan(destination, plan)
    assert sentinel.read_bytes() == b"preserve"


def test_local_probe_removes_editable_import_finders_before_snapshot_imports() -> None:
    filter_position = _LOCAL_PROBE_SOURCE.index("sys.meta_path[:] =")

    assert "importlib.machinery.PathFinder" in _LOCAL_PROBE_SOURCE
    assert filter_position < _LOCAL_PROBE_SOURCE.index("import simsoptpp")


def test_publication_tamper_fails_closed(tmp_path: Path) -> None:
    inputs, context = _chain(tmp_path)
    publication = dict(context["publication"])
    publication["snapshot_manifest_sha256"] = "f" * 64
    inputs.publication_path.write_bytes(canonical_json_bytes(publication))

    with pytest.raises(Phase0WorkflowError, match="manifest identity mismatch"):
        build_phase0_workflow(inputs, runner=_runner(context))


def test_specimen_candidate_tamper_fails_closed(tmp_path: Path) -> None:
    inputs, context = _chain(tmp_path)
    candidate = context["snapshot"] / "phase0-specimen" / "changed_state_candidate.npy"
    candidate.chmod(0o644)
    candidate.write_bytes(b"tampered")

    with pytest.raises(
        (Phase0WorkflowError, ValueError), match="manifest file|candidate"
    ):
        build_phase0_workflow(inputs, runner=_runner(context))


def _landau_receipt(
    inputs: Phase0WorkflowInputs, context: dict[str, object]
) -> dict[str, object]:
    interpreter = Path(os.path.abspath(inputs.interpreter))
    target = interpreter.resolve()
    imports = {
        binding["module"]: {
            "path": str(context["snapshot"] / binding["relative_path"]),
            "size": binding["size_bytes"],
            "sha256": binding["sha256"],
        }
        for binding in context["bindings"]
        if binding["module"] != "simsoptpp"
    }
    native_binding = next(
        binding for binding in context["bindings"] if binding["module"] == "simsoptpp"
    )
    _, manifest_sha256 = load_snapshot_manifest(context["snapshot"])
    receipt = {
        "schema_id": QUALIFICATION_SCHEMA_ID,
        "state": "PASS",
        "blockers": [],
        "slurm": {
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_NODELIST": "landau",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        "hostname": "landau.example.edu",
        "interpreter": {
            "entrypoint_path": str(interpreter),
            "target_path": str(target),
            "target_size": target.stat().st_size,
            "target_sha256": _sha256(target.read_bytes()),
            "prefix": "/qualified/venv",
            "base_prefix": "/usr",
        },
        "physical_gpus": [
            {
                "uuid": "GPU-a100",
                "name": "NVIDIA A100",
                "memory_mib": 40536,
                "driver_version": "470.0",
            }
        ],
        "cuda": {
            "runtime_platform_version": "CUDA 12.6.3",
            "compatibility_paths": ["/opt/cuda-12.6/compat"],
            "resolved_libraries": [
                {
                    "path": "/opt/cuda-12.6/compat/libcuda.so.1",
                    "size": 1,
                    "sha256": "9" * 64,
                }
            ],
        },
        "jax": {
            "version": "0.10.0",
            "jaxlib_version": "0.10.0",
            "backend": "gpu",
            "x64_enabled": True,
            "platform_version": "CUDA 12.6.3",
        },
        "smoke": {"finite": True},
        "overlay": {
            "actual_packages": {
                "jax": "0.10.0",
                "jaxlib": "0.10.0",
                "lineax": "0.1.1",
            }
        },
        "source": {
            "snapshot_root": str(context["snapshot"]),
            "snapshot_manifest_sha256": manifest_sha256,
            "external_provenance": context["publication"],
        },
        "imports": imports,
        "native_binary": {
            "path": str(context["snapshot"] / native_binding["relative_path"]),
            "size": native_binding["size_bytes"],
            "sha256": native_binding["sha256"],
        },
        "environment": {
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_NODELIST": "landau",
            "CUDA_VISIBLE_DEVICES": "0",
            "LD_LIBRARY_PATH": "/opt/cuda-12.6/compat:/overlay/lib",
            "JAX_TRANSFER_GUARD": "disallow",
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
        },
        "static_timing_environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "LD_LIBRARY_PATH": "/opt/cuda-12.6/compat:/overlay/lib",
            "JAX_TRANSFER_GUARD": "disallow",
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
        },
        "numerical_policies": {
            "source_path": str(context["snapshot"] / "phase0-specimen/specimen.json"),
            "source_sha256": next(
                entry.sha256
                for entry in load_snapshot_manifest(context["snapshot"])[0]
                if entry.relative_path == "phase0-specimen/specimen.json"
            ),
            "source_manifest_entry": next(
                entry.to_json()
                for entry in load_snapshot_manifest(context["snapshot"])[0]
                if entry.relative_path == "phase0-specimen/specimen.json"
            ),
            "effective": {
                "dense_batch_width": 8,
                "point_chunk_size": None,
                "coil_chunk_size": None,
                "quadrature_block_sizes": [128, 122],
            },
            "declared": {
                "dense_batch_width": 8,
                "point_chunk_size": None,
                "coil_chunk_size": None,
                "quadrature_block_sizes": [128, 122],
            },
        },
        "cpu_affinity": [0, 1],
        "failed_commands": [],
    }
    receipt["receipt_sha256"] = _sha256(landau_canonical_json_bytes(receipt))
    return receipt


def test_landau_adapter_binds_interpreter_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    inputs, context = _chain(tmp_path)
    receipt = _landau_receipt(inputs, context)
    receipt_path = _json(tmp_path / "landau.json", receipt)
    entries, manifest_sha256 = load_snapshot_manifest(context["snapshot"])
    attestation = json.loads(inputs.import_attestation_path.read_bytes())

    qualification, provenance = _landau_qualification(
        receipt_path,
        snapshot_root=context["snapshot"],
        interpreter=inputs.interpreter,
        attestation_document=attestation,
        publication=context["publication"],
        entries=entries,
        manifest_sha256=manifest_sha256,
        cache_directory=inputs.compilation_cache_directory,
        execution_environment=receipt["environment"],
    )

    assert qualification["outcome"] == "qualified"
    assert provenance["interpreter_path"] == str(inputs.interpreter)
    assert provenance["runtime"]["cuda_runtime"] == "CUDA 12.6.3"
    assert provenance["runtime"]["jax_backend"] == "gpu"
    assert provenance["policies"] == receipt["numerical_policies"]["effective"]
    assert provenance["environment"] == receipt["static_timing_environment"]

    receipt["interpreter"]["entrypoint_path"] = "/wrong/python"
    receipt_without_hash = dict(receipt)
    receipt_without_hash.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha256(
        landau_canonical_json_bytes(receipt_without_hash)
    )
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    with pytest.raises(Phase0WorkflowError, match="interpreter identity"):
        _landau_qualification(
            receipt_path,
            snapshot_root=context["snapshot"],
            interpreter=inputs.interpreter,
            attestation_document=attestation,
            publication=context["publication"],
            entries=entries,
            manifest_sha256=manifest_sha256,
            cache_directory=inputs.compilation_cache_directory,
            execution_environment=receipt["environment"],
        )


@pytest.mark.parametrize("drift_side", ("workflow", "receipt"))
def test_landau_static_environment_requires_exact_two_sided_equality(
    tmp_path: Path, drift_side: str
) -> None:
    inputs, context = _chain(tmp_path)
    receipt = _landau_receipt(inputs, context)
    execution_environment = dict(receipt["static_timing_environment"])
    if drift_side == "workflow":
        execution_environment["XLA_FLAGS"] = "--workflow-only"
    else:
        receipt["static_timing_environment"]["XLA_FLAGS"] = "--receipt-only"
        receipt_without_hash = dict(receipt)
        receipt_without_hash.pop("receipt_sha256")
        receipt["receipt_sha256"] = _sha256(
            landau_canonical_json_bytes(receipt_without_hash)
        )
    receipt_path = _json(tmp_path / f"landau-{drift_side}.json", receipt)
    entries, manifest_sha256 = load_snapshot_manifest(context["snapshot"])
    attestation = json.loads(inputs.import_attestation_path.read_bytes())

    with pytest.raises(Phase0WorkflowError, match="static timing environment"):
        _landau_qualification(
            receipt_path,
            snapshot_root=context["snapshot"],
            interpreter=inputs.interpreter,
            attestation_document=attestation,
            publication=context["publication"],
            entries=entries,
            manifest_sha256=manifest_sha256,
            cache_directory=inputs.compilation_cache_directory,
            execution_environment=execution_environment,
        )
