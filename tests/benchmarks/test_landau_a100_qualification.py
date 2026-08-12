from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.landau_a100_qualification import (
    OBSERVATION_SCHEMA_ID,
    QUALIFICATION_SCHEMA_ID,
    CommandResult,
    _canonical_json_bytes,
    _parse_args,
    build_landau_qualification_receipt,
    collect_landau_observations,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    SOURCE_MANIFEST_SCHEMA_ID,
    RoleRoot,
    publish_immutable_snapshot,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    canonical_json_bytes as snapshot_canonical_json_bytes,
)

_HASH = "a" * 64
_HEAD = "b" * 40


def _file(path: str, *, sha256: str = _HASH) -> dict[str, object]:
    return {"path": path, "size": 128, "sha256": sha256}


def _manifest_entries(snapshot_root: str = "/snapshot") -> list[dict[str, object]]:
    del snapshot_root
    return [
        {
            "role": "benchmark",
            "relative_path": "benchmarks/run.py",
            "size_bytes": 128,
            "sha256": _HASH,
        },
        {
            "role": "configuration",
            "relative_path": "config/landau.lock.txt",
            "size_bytes": 128,
            "sha256": _HASH,
        },
        {
            "role": "configuration",
            "relative_path": "phase0-specimen/specimen.json",
            "size_bytes": 128,
            "sha256": _HASH,
        },
        *[
            {
                "role": "execution_source",
                "relative_path": path,
                "size_bytes": 128,
                "sha256": _HASH,
            }
            for path in (
                "src/simsopt/__init__.py",
                "src/simsopt_jax/__init__.py",
                "src/simsopt_jax_adapters/__init__.py",
            )
        ],
        {
            "role": "native_extension",
            "relative_path": "src/simsoptpp.cpython-311-x86_64-linux-gnu.so",
            "size_bytes": 128,
            "sha256": _HASH,
        },
        {
            "role": "test",
            "relative_path": "tests/test_run.py",
            "size_bytes": 128,
            "sha256": _HASH,
        },
    ]


def _probe(snapshot_root: str = "/snapshot") -> dict[str, object]:
    return {
        "interpreter": {
            "entrypoint_path": "/qualified/venv/bin/python",
            "target_path": "/qualified/base/python3.11",
            "target_size": 128,
            "target_sha256": _HASH,
            "prefix": "/qualified/venv",
            "base_prefix": "/qualified/base",
        },
        "jax": {
            "version": "0.10.0",
            "jaxlib_version": "0.10.0",
            "x64_enabled": True,
            "backend": "gpu",
            "platform_version": "CUDA 12.6.3",
            "devices": [{"id": 0, "platform": "gpu", "kind": "NVIDIA A100-PCIE-40GB"}],
        },
        "packages": {"jax": "0.10.0", "jaxlib": "0.10.0", "lineax": "0.1.1"},
        "imports": {
            "simsopt": _file(f"{snapshot_root}/src/simsopt/__init__.py"),
            "simsopt_jax": _file(f"{snapshot_root}/src/simsopt_jax/__init__.py"),
            "simsopt_jax_adapters": _file(
                f"{snapshot_root}/src/simsopt_jax_adapters/__init__.py"
            ),
        },
        "native_binary": _file(
            f"{snapshot_root}/src/simsoptpp.cpython-311-x86_64-linux-gnu.so"
        ),
        "resolved_cuda_libraries": [
            _file("/opt/cuda-12.6/compat/libcuda.so.1"),
            _file("/overlay/nvidia/cuda_runtime/lib/libcudart.so.12"),
        ],
        "smoke": {
            "transfer_guard": "disallow",
            "input_dtype": "float64",
            "output_dtype": "float64",
            "output_shape": [],
            "finite": True,
            "value": 17.8125,
        },
        "static_timing_environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "LD_LIBRARY_PATH": "/opt/cuda-12.6/compat:/overlay/lib",
            "JAX_TRANSFER_GUARD": "disallow",
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
        },
        "effective_numerical_policies": {
            "dense_batch_width": 8,
            "point_chunk_size": None,
            "coil_chunk_size": None,
            "quadrature_block_sizes": [128, 122],
        },
    }


def _command(stdout: str = "", *, returncode: int = 0) -> dict[str, object]:
    return {
        "argv": ["synthetic"],
        "returncode": returncode,
        "stdout": stdout,
        "stderr": "",
    }


def _observations(snapshot_root: str = "/snapshot") -> dict[str, object]:
    packages = {"jax": "0.10.0", "jaxlib": "0.10.0", "lineax": "0.1.1"}
    freeze = (
        "\n".join(f"{name}=={version}" for name, version in packages.items()) + "\n"
    )
    manifest_entries = _manifest_entries(snapshot_root)
    manifest_sha256 = hashlib.sha256(
        snapshot_canonical_json_bytes(
            {
                "schema_id": SOURCE_MANIFEST_SCHEMA_ID,
                "entries": manifest_entries,
            }
        )
    ).hexdigest()
    lock_entry = next(
        entry
        for entry in manifest_entries
        if entry["relative_path"] == "config/landau.lock.txt"
    )
    return {
        "schema_id": OBSERVATION_SCHEMA_ID,
        "interpreter": {
            "entrypoint_path": "/qualified/venv/bin/python",
            "target_path": "/qualified/base/python3.11",
            "target_size": 128,
            "target_sha256": _HASH,
        },
        "environment": {
            "SLURM_JOB_ID": "48151623",
            "SLURM_JOB_NODELIST": "landau",
            "CUDA_VISIBLE_DEVICES": "0",
            "SLURM_STEP_ID": "0",
            "LD_LIBRARY_PATH": "/opt/cuda-12.6/compat:/overlay/lib",
            "JAX_TRANSFER_GUARD": "disallow",
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cuda",
        },
        "commands": {
            "hostname": _command("landau.example.edu\n"),
            "gpu": _command(
                "GPU-250014ca-8cb3-bdcd-ad1d-2f6f64529b8d, "
                "NVIDIA A100-PCIE-40GB, 40536, 470.256.02\n"
            ),
            "pip_freeze": _command(freeze),
            "jax_probe": _command(json.dumps(_probe(snapshot_root), sort_keys=True)),
        },
        "cpu_affinity": [0, 1, 2, 3],
        "source": {
            "snapshot_root": snapshot_root,
            "snapshot_manifest_path": f"{snapshot_root}/phase0-source-manifest.json",
            "snapshot_manifest_schema_id": SOURCE_MANIFEST_SCHEMA_ID,
            "snapshot_manifest_sha256": manifest_sha256,
            "snapshot_manifest_entries": manifest_entries,
            "external_provenance": {"head": _HEAD},
        },
        "overlay": {
            "lock_path": f"{snapshot_root}/config/landau.lock.txt",
            "lock_sha256": _HASH,
            "lock_manifest_entry": lock_entry,
            "expected_packages": packages,
            "expected_native_binary_sha256": _HASH,
            "actual_freeze": freeze,
        },
        "numerical_policies": {
            "source_path": f"{snapshot_root}/phase0-specimen/specimen.json",
            "source_sha256": _HASH,
            "source_manifest_entry": next(
                entry
                for entry in manifest_entries
                if entry["relative_path"] == "phase0-specimen/specimen.json"
            ),
            "declared": {
                "dense_batch_width": 8,
                "point_chunk_size": None,
                "coil_chunk_size": None,
                "quadrature_block_sizes": [128, 122],
            },
        },
        "probe": json.dumps(_probe(snapshot_root), sort_keys=True),
    }


def test_complete_current_landau_observation_passes() -> None:
    receipt = build_landau_qualification_receipt(_observations())

    assert receipt["schema_id"] == QUALIFICATION_SCHEMA_ID
    assert receipt["state"] == "PASS"
    assert receipt["blockers"] == []
    assert receipt["cuda"]["compatibility_policy"] == "cuda-12.6-forward-compat-only"
    assert receipt["cuda"]["runtime_platform_version"] == "CUDA 12.6.3"
    assert receipt["numerical_policies"]["effective"] == {
        "dense_batch_width": 8,
        "point_chunk_size": None,
        "coil_chunk_size": None,
        "quadrature_block_sizes": [128, 122],
    }
    assert receipt["physical_gpus"][0]["uuid"].startswith("GPU-")
    assert receipt["slurm"]["scheduler_gpu_accounting"] == {
        "environment_key": "SLURM_JOB_GPUS",
        "state": "unavailable_not_configured",
        "value": None,
    }
    assert receipt["slurm"]["exclusive_node_preferred"] is True
    assert len(receipt["receipt_sha256"]) == 64


def test_nondefault_imported_policy_drift_is_blocked() -> None:
    observations = _observations()
    probe = json.loads(observations["commands"]["jax_probe"]["stdout"])
    probe["effective_numerical_policies"]["dense_batch_width"] = 16
    observations["commands"]["jax_probe"]["stdout"] = json.dumps(probe)

    receipt = build_landau_qualification_receipt(observations)

    assert receipt["state"] == "BLOCKED"
    assert "NUMERICAL_POLICY_DRIFT" in {
        blocker["code"] for blocker in receipt["blockers"]
    }


def test_interpreter_probe_mismatch_and_nonvenv_runtime_block() -> None:
    observations = _observations()
    probe = json.loads(observations["commands"]["jax_probe"]["stdout"])
    probe["interpreter"]["target_sha256"] = "b" * 64
    probe["interpreter"]["base_prefix"] = probe["interpreter"]["prefix"]
    observations["commands"]["jax_probe"]["stdout"] = json.dumps(probe, sort_keys=True)

    receipt = build_landau_qualification_receipt(observations)
    codes = {blocker["code"] for blocker in receipt["blockers"]}

    assert receipt["state"] == "BLOCKED"
    assert "INTERPRETER_PROBE_IDENTITY_MISMATCH" in codes
    assert "VIRTUALENV_IDENTITY_INVALID" in codes


def test_unsupported_or_mixed_compatibility_stack_blocks() -> None:
    observations = _observations()
    observations["environment"]["LD_LIBRARY_PATH"] = (
        "/opt/cuda-12.6/compat:/opt/cuda-12.9/compat"
    )
    receipt = build_landau_qualification_receipt(observations)

    codes = {row["code"] for row in receipt["blockers"]}
    assert receipt["state"] == "BLOCKED"
    assert "UNSUPPORTED_CUDA_COMPAT_PATH" in codes
    assert "MIXED_CUDA_COMPAT_PATHS" in codes


def test_malformed_probe_output_is_a_machine_readable_blocker() -> None:
    observations = _observations()
    observations["commands"]["jax_probe"]["stdout"] = "not-json"
    receipt = build_landau_qualification_receipt(observations)

    assert receipt["state"] == "BLOCKED"
    assert "JAX_PROBE_OUTPUT_INVALID" in {row["code"] for row in receipt["blockers"]}


def test_libcuda_must_resolve_from_selected_12_6_compatibility_path() -> None:
    observations = _observations()
    probe = json.loads(observations["commands"]["jax_probe"]["stdout"])
    probe["resolved_cuda_libraries"][0]["path"] = "/usr/lib/libcuda.so.1"
    observations["commands"]["jax_probe"]["stdout"] = json.dumps(probe)
    receipt = build_landau_qualification_receipt(observations)

    assert receipt["state"] == "BLOCKED"
    assert "CUDA_DRIVER_LIBRARY_NOT_FROM_12_6_COMPAT" in {
        row["code"] for row in receipt["blockers"]
    }


def test_overlay_is_exact_and_requires_lineax_0_1_1() -> None:
    observations = _observations()
    observations["overlay"]["actual_freeze"] = (
        "jax==0.10.0\njaxlib==0.10.0\nlineax==0.1.0\n"
    )
    receipt = build_landau_qualification_receipt(observations)

    codes = {row["code"] for row in receipt["blockers"]}
    assert receipt["state"] == "BLOCKED"
    assert codes >= {"DEPENDENCY_OVERLAY_MISMATCH", "LINEAX_VERSION_INVALID"}


def test_overlay_uses_canonical_distribution_names() -> None:
    observations = _observations()
    observations["overlay"]["expected_packages"]["ruamel-yaml"] = "0.19.1"
    observations["overlay"]["actual_freeze"] += "ruamel.yaml==0.19.1\n"
    probe = json.loads(observations["commands"]["jax_probe"]["stdout"])
    probe["packages"]["ruamel-yaml"] = "0.19.1"
    observations["commands"]["jax_probe"]["stdout"] = json.dumps(probe)

    receipt = build_landau_qualification_receipt(observations)

    assert receipt["state"] == "PASS"


def test_strict_transfer_finite_fp64_smoke_is_mandatory() -> None:
    observations = _observations()
    probe = json.loads(observations["commands"]["jax_probe"]["stdout"])
    probe["smoke"]["output_dtype"] = "float32"
    observations["commands"]["jax_probe"]["stdout"] = json.dumps(probe)
    receipt = build_landau_qualification_receipt(observations)

    assert receipt["state"] == "BLOCKED"
    assert "STRICT_TRANSFER_FP64_SMOKE_FAILED" in {
        row["code"] for row in receipt["blockers"]
    }


def test_runtime_platform_and_policy_source_are_mandatory() -> None:
    observations = _observations()
    probe = json.loads(observations["commands"]["jax_probe"]["stdout"])
    probe["jax"]["platform_version"] = ""
    observations["commands"]["jax_probe"]["stdout"] = json.dumps(probe)
    observations["numerical_policies"]["source_sha256"] = "b" * 64

    receipt = build_landau_qualification_receipt(observations)

    assert receipt["state"] == "BLOCKED"
    assert {row["code"] for row in receipt["blockers"]} >= {
        "CUDA_RUNTIME_PLATFORM_IDENTITY_MISSING",
        "NUMERICAL_POLICY_SOURCE_BINDING_INVALID",
    }


def test_imports_must_resolve_below_receipt_source_root() -> None:
    observations = _observations()
    probe = json.loads(observations["commands"]["jax_probe"]["stdout"])
    probe["imports"]["simsopt_jax"]["path"] = "/stale/src/simsopt_jax/__init__.py"
    observations["commands"]["jax_probe"]["stdout"] = json.dumps(probe)
    receipt = build_landau_qualification_receipt(observations)

    assert receipt["state"] == "BLOCKED"
    assert "IMPORT_OUTSIDE_IMMUTABLE_SNAPSHOT" in {
        row["code"] for row in receipt["blockers"]
    }


def test_snapshot_manifest_hash_and_import_binding_are_mandatory() -> None:
    observations = _observations()
    observations["source"]["snapshot_manifest_sha256"] = "f" * 64
    probe = json.loads(observations["commands"]["jax_probe"]["stdout"])
    probe["imports"]["simsopt_jax"]["sha256"] = "e" * 64
    observations["commands"]["jax_probe"]["stdout"] = json.dumps(probe)

    receipt = build_landau_qualification_receipt(observations)

    codes = {row["code"] for row in receipt["blockers"]}
    assert receipt["state"] == "BLOCKED"
    assert codes >= {
        "SNAPSHOT_MANIFEST_HASH_MISMATCH",
        "IMPORT_SNAPSHOT_IDENTITY_MISMATCH",
    }


def test_failed_collection_command_is_a_machine_readable_blocker() -> None:
    observations = _observations()
    observations["commands"]["jax_probe"] = _command(returncode=1)
    receipt = build_landau_qualification_receipt(observations)

    assert receipt["state"] == "BLOCKED"
    assert "COLLECTION_COMMAND_FAILED" in {row["code"] for row in receipt["blockers"]}


def test_collector_accepts_injected_commands_without_gpu(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    native_binary = source_root / "native" / "simsoptpp.cpython-311-x86_64-linux-gnu.so"
    native_binary.parent.mkdir(parents=True)
    native_binary.write_bytes(b"synthetic native extension\n")
    native_hash = hashlib.sha256(native_binary.read_bytes()).hexdigest()
    overlay_lock_source = source_root / "config" / "landau.lock.txt"
    overlay_lock_source.parent.mkdir(parents=True)
    overlay_lock_source.write_text(
        "# sha256(simsoptpp.cpython-311-x86_64-linux-gnu.so) =\n"
        f"# {native_hash}\n"
        "jax==0.10.0\n"
        "jaxlib==0.10.0\n",
        encoding="utf-8",
    )
    packages = source_root / "packages"
    for package in ("simsopt", "simsopt_jax", "simsopt_jax_adapters"):
        path = packages / package / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"PACKAGE = {package!r}\n", encoding="utf-8")
    benchmark = source_root / "benchmark" / "run.py"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("VALUE = 1\n", encoding="utf-8")
    test_source = source_root / "test" / "test_run.py"
    test_source.parent.mkdir(parents=True)
    test_source.write_text("def test_value(): pass\n", encoding="utf-8")
    specimen_source = source_root / "phase0-specimen" / "specimen.json"
    specimen_source.parent.mkdir(parents=True)
    specimen_source.write_bytes(
        _canonical_json_bytes(
            {
                "effective_policies": {
                    "dense_batch_width": 8,
                    "point_chunk_size": None,
                    "coil_chunk_size": None,
                    "quadrature_block_sizes": [128, 122],
                },
                "specimen": {"grids": {"quadrature_nodes": 250}},
            }
        )
    )
    snapshot_root = tmp_path / "snapshot"
    publish_immutable_snapshot(
        snapshot_root,
        (
            RoleRoot("execution_source", packages, "src"),
            RoleRoot("configuration", overlay_lock_source, "config/landau.lock.txt"),
            RoleRoot(
                "configuration",
                specimen_source,
                "phase0-specimen/specimen.json",
            ),
            RoleRoot("benchmark", benchmark, "benchmarks/run.py"),
            RoleRoot("test", test_source, "tests/test_run.py"),
            RoleRoot(
                "native_extension",
                native_binary,
                "src/simsoptpp.cpython-311-x86_64-linux-gnu.so",
            ),
        ),
    )
    overlay_lock = snapshot_root / "config" / "landau.lock.txt"
    freeze = "jax==0.10.0\njaxlib==0.10.0\nlineax==0.1.1\n"
    entries = json.loads((snapshot_root / "phase0-source-manifest.json").read_bytes())[
        "entries"
    ]
    identities = {entry["relative_path"]: entry for entry in entries}
    python_target = tmp_path / "base" / "python3.11"
    python_target.parent.mkdir()
    python_target.write_bytes(b"synthetic python executable\n")
    python_target.chmod(0o755)
    python_entrypoint = tmp_path / "venv" / "bin" / "python"
    python_entrypoint.parent.mkdir(parents=True)
    python_entrypoint.symlink_to(python_target)
    python_target_sha256 = hashlib.sha256(python_target.read_bytes()).hexdigest()

    def snapshot_file(relative_path: str) -> dict[str, object]:
        entry = identities[relative_path]
        return {
            "path": str(snapshot_root / relative_path),
            "size": entry["size_bytes"],
            "sha256": entry["sha256"],
        }

    probe_document = _probe(str(snapshot_root.resolve()))
    probe_document["interpreter"] = {
        "entrypoint_path": str(python_entrypoint),
        "target_path": str(python_target),
        "target_size": python_target.stat().st_size,
        "target_sha256": python_target_sha256,
        "prefix": str(tmp_path / "venv"),
        "base_prefix": str(tmp_path / "base"),
    }
    probe_document["imports"] = {
        package: snapshot_file(f"src/{package}/__init__.py")
        for package in ("simsopt", "simsopt_jax", "simsopt_jax_adapters")
    }
    probe_document["native_binary"] = snapshot_file(
        "src/simsoptpp.cpython-311-x86_64-linux-gnu.so"
    )
    probe = json.dumps(probe_document, sort_keys=True)

    def runner(argv: Sequence[str], _environment: Mapping[str, str]) -> CommandResult:
        if argv[0] == "hostname":
            return CommandResult(0, "landau.example.edu\n", "")
        if argv[0] == "nvidia-smi":
            return CommandResult(
                0,
                "GPU-250014ca-8cb3-bdcd-ad1d-2f6f64529b8d, "
                "NVIDIA A100-PCIE-40GB, 40536, 470.256.02\n",
                "",
            )
        if tuple(argv[-2:]) == ("pip", "freeze"):
            return CommandResult(0, freeze, "")
        if "-c" in argv:
            return CommandResult(0, probe, "")
        raise AssertionError(f"unexpected command: {argv}")

    environment = {
        "SLURM_JOB_ID": "48151623",
        "SLURM_JOB_NODELIST": "landau",
        "SLURM_JOB_GPUS": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/opt/cuda-12.6/compat:/overlay/lib",
    }
    observations = collect_landau_observations(
        snapshot_root=snapshot_root,
        python_executable=python_entrypoint,
        overlay_lock=overlay_lock,
        environment=environment,
        runner=runner,
        cpu_affinity=(0, 1),
    )
    receipt = build_landau_qualification_receipt(observations)

    assert receipt["state"] == "PASS"
    assert observations["environment"]["JAX_TRANSFER_GUARD"] == "disallow"
    assert observations["environment"]["JAX_ENABLE_X64"] == "true"
    assert observations["cpu_affinity"] == [0, 1]
    assert observations["interpreter"]["entrypoint_path"] == str(python_entrypoint)
    assert observations["interpreter"]["target_path"] == str(python_target)
    assert set(observations["commands"]) == {
        "hostname",
        "gpu",
        "pip_freeze",
        "jax_probe",
    }


def test_receipt_is_deterministic_and_input_is_not_mutated() -> None:
    observations = _observations()
    original = copy.deepcopy(observations)

    first = build_landau_qualification_receipt(observations)
    second = build_landau_qualification_receipt(observations)

    assert observations == original
    assert _canonical_json_bytes(first) == _canonical_json_bytes(second)
    receipt_without_hash = dict(first)
    expected_hash = receipt_without_hash.pop("receipt_sha256")
    assert (
        hashlib.sha256(_canonical_json_bytes(receipt_without_hash)).hexdigest()
        == expected_hash
    )


def test_cli_requires_immutable_snapshot_root() -> None:
    options = _parse_args(
        (
            "--snapshot-root",
            "/immutable/snapshot",
            "--python",
            "/qualified/python",
            "--overlay-lock",
            "/immutable/snapshot/config/landau.lock.txt",
        )
    )

    assert options.snapshot_root == Path("/immutable/snapshot")
    assert not hasattr(options, "repo_root")
