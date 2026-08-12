from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import benchmarks.single_stage_compute_graph_isolated_launch as isolated_launch
import pytest
from benchmarks.single_stage_compute_graph_isolated_launch import (
    SnapshotModuleLaunchError,
    build_snapshot_module_launch,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    MANIFEST_FILENAME,
    RoleRoot,
    load_snapshot_manifest,
    publish_immutable_snapshot,
)

_MODULE = "benchmarks.single_stage_compute_graph_command_buffer_control"
_ORCHESTRATION_MODULES = (
    "benchmarks.landau_a100_qualification",
    "benchmarks.single_stage_compute_graph_canary_evaluator",
    "benchmarks.single_stage_compute_graph_c0_runner",
    "benchmarks.single_stage_compute_graph_native_trajectory",
    "benchmarks.single_stage_compute_graph_native_trajectory_runner",
    "benchmarks.single_stage_compute_graph_phase0_post_gate",
    "benchmarks.single_stage_compute_graph_phase0_workflow",
    "benchmarks.single_stage_compute_graph_variant_trajectory",
    "benchmarks.single_stage_compute_graph_variant_trajectory_runner",
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _snapshot(root: Path, *, module_marker: str = "SNAPSHOT") -> Path:
    source = root / "source"
    execution_source = _write(
        source / "execution" / "simsopt_jax" / "__init__.py",
        "PACKAGE = 'simsopt_jax'\n",
    ).parent
    parity_child = _write(
        source / "parity" / "child.py",
        f"print({module_marker!r})\n",
    )
    configuration = _write(source / "configuration" / "input.json", "{}\n")
    benchmark_root = source / "benchmark"
    _write(benchmark_root / "__init__.py", "\n")
    _write(
        benchmark_root / "single_stage_compute_graph_command_buffer_control.py",
        f"print({module_marker!r})\n",
    )
    for module in _ORCHESTRATION_MODULES:
        _write(
            benchmark_root / f"{module.removeprefix('benchmarks.')}.py",
            f"print({module_marker!r})\n",
        )
    test = _write(source / "test" / "test_control.py", "def test_control(): pass\n")
    native_extension = _write(source / "native" / "simsoptpp.py", "NATIVE = True\n")
    destination = root / "snapshot"
    publish_immutable_snapshot(
        destination,
        (
            RoleRoot("execution_source", execution_source, "src/simsopt_jax"),
            RoleRoot("execution_source", parity_child, "examples/jax/parity/child.py"),
            RoleRoot("configuration", configuration, "inputs/input.json"),
            RoleRoot("benchmark", benchmark_root, "benchmarks"),
            RoleRoot("test", test, "tests/test_control.py"),
            RoleRoot("native_extension", native_extension, "src/simsoptpp.py"),
        ),
    )
    return destination


def test_valid_manifest_builds_exact_isolated_launch(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)

    launch = build_snapshot_module_launch(
        Path(sys.executable),
        snapshot,
        _MODULE,
        ("probe", "--parameter-sha256", "a" * 64),
        {
            "PATH": "/usr/bin",
            "CUDA_VISIBLE_DEVICES": "0",
            "SIMSOPT_EXACT_ADJOINT_DENSE_LU": "1",
            "PYTHONHOME": "/ambient/home",
            "PYTHONPATH": "/ambient/editable/src",
            "PYTHONUSERBASE": "/ambient/user",
        },
    )

    assert launch.argv == (
        str(Path(sys.executable).absolute()),
        "-P",
        "-s",
        "-c",
        isolated_launch.ISOLATED_MODULE_BOOTSTRAP,
        _MODULE,
        "probe",
        "--parameter-sha256",
        "a" * 64,
    )
    assert launch.cwd == snapshot.resolve()
    assert launch.environment["PYTHONPATH"] == f"{snapshot / 'src'}:{snapshot}"
    assert launch.environment["PYTHONNOUSERSITE"] == "1"
    assert launch.environment["PYTHONSAFEPATH"] == "1"
    assert launch.environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert launch.environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert launch.environment["SIMSOPT_EXACT_ADJOINT_DENSE_LU"] == "1"
    assert "PYTHONHOME" not in launch.environment
    assert "PYTHONUSERBASE" not in launch.environment


def test_virtualenv_entry_point_is_not_resolved_to_base_python(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    launch = build_snapshot_module_launch(
        Path(sys.executable),
        snapshot,
        _MODULE,
        (),
        {"PATH": "/usr/bin"},
    )

    completed = subprocess.run(
        (
            launch.argv[0],
            "-P",
            "-s",
            "-c",
            "import sys; print(sys.prefix)",
        ),
        cwd=launch.cwd,
        env=dict(launch.environment),
        check=True,
        capture_output=True,
        text=True,
    )

    assert launch.argv[0] == str(Path(sys.executable).absolute())
    assert completed.stdout.strip() == sys.prefix


def test_module_allowlist_rejects_unapproved_module(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)

    with pytest.raises(SnapshotModuleLaunchError, match="not allowed"):
        build_snapshot_module_launch(
            Path(sys.executable), snapshot, "benchmarks.unapproved", (), {}
        )


def test_manifested_execution_source_module_is_allowed(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    module = "examples.jax.parity.child"

    launch = build_snapshot_module_launch(
        Path(sys.executable), snapshot, module, ("--help",), {"PATH": "/usr/bin"}
    )

    completed = subprocess.run(
        launch.argv,
        cwd=launch.cwd,
        env=dict(launch.environment),
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "SNAPSHOT"


@pytest.mark.parametrize("module", _ORCHESTRATION_MODULES)
def test_snapshot_orchestration_module_is_allowed(tmp_path: Path, module: str) -> None:
    snapshot = _snapshot(tmp_path)

    launch = build_snapshot_module_launch(
        Path(sys.executable), snapshot, module, ("--help",), {"PATH": "/usr/bin"}
    )

    assert launch.argv[-2:] == (module, "--help")
    assert launch.cwd == snapshot.resolve()


def test_manifested_canary_evaluator_help_executes_from_new_snapshot(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    module = "benchmarks.single_stage_compute_graph_canary_evaluator"
    launch = build_snapshot_module_launch(
        Path(sys.executable), snapshot, module, ("--help",), {"PATH": "/usr/bin"}
    )

    completed = subprocess.run(
        launch.argv,
        cwd=launch.cwd,
        env=dict(launch.environment),
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "SNAPSHOT"


def test_tampered_manifested_bytes_are_rejected(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    module_path = (
        snapshot / "benchmarks" / "single_stage_compute_graph_command_buffer_control.py"
    )
    module_path.chmod(0o644)
    module_path.write_text("print('TAMPERED')\n", encoding="utf-8")

    with pytest.raises(
        SnapshotModuleLaunchError, match="immutable snapshot is invalid"
    ):
        build_snapshot_module_launch(Path(sys.executable), snapshot, _MODULE, (), {})


def test_invalid_manifest_document_is_rejected(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    manifest = snapshot / MANIFEST_FILENAME
    manifest.chmod(0o644)
    manifest.write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(
        SnapshotModuleLaunchError, match="immutable snapshot is invalid"
    ):
        build_snapshot_module_launch(Path(sys.executable), snapshot, _MODULE, (), {})


def test_missing_native_extension_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(tmp_path)
    entries, manifest_sha256 = load_snapshot_manifest(snapshot)
    without_native = tuple(
        entry for entry in entries if entry.role != "native_extension"
    )
    monkeypatch.setattr(
        isolated_launch,
        "load_snapshot_manifest",
        lambda snapshot_root: (without_native, manifest_sha256),
    )

    with pytest.raises(SnapshotModuleLaunchError, match="lacks a native extension"):
        build_snapshot_module_launch(Path(sys.executable), snapshot, _MODULE, (), {})


def test_ambient_editable_module_cannot_win_import_resolution(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "frozen", module_marker="SNAPSHOT")
    ambient = tmp_path / "ambient"
    _write(ambient / "benchmarks" / "__init__.py", "\n")
    _write(
        ambient / "benchmarks" / "single_stage_compute_graph_command_buffer_control.py",
        "print('AMBIENT')\n",
    )
    launch = build_snapshot_module_launch(
        Path(sys.executable),
        snapshot,
        _MODULE,
        (),
        {
            "PATH": "/usr/bin",
            "PYTHONPATH": str(ambient),
            "PYTHONHOME": str(ambient / "python-home"),
        },
    )

    completed = subprocess.run(
        launch.argv,
        cwd=launch.cwd,
        env=dict(launch.environment),
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "SNAPSHOT"
    assert "AMBIENT" not in completed.stdout


def test_installed_editable_finder_cannot_override_snapshot_package(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-with-import"
    execution_source = _write(
        source / "execution" / "simsopt_jax" / "__init__.py",
        "PACKAGE = 'snapshot'\n",
    ).parent
    configuration = _write(source / "configuration" / "input.json", "{}\n")
    benchmark_root = source / "benchmark"
    _write(benchmark_root / "__init__.py", "\n")
    _write(
        benchmark_root / "single_stage_compute_graph_command_buffer_control.py",
        "import simsopt_jax; print(simsopt_jax.__file__)\n",
    )
    test = _write(source / "test" / "test_control.py", "def test_control(): pass\n")
    native_extension = _write(source / "native" / "simsoptpp.py", "NATIVE = True\n")
    clean_snapshot = tmp_path / "snapshot-with-import"
    publish_immutable_snapshot(
        clean_snapshot,
        (
            RoleRoot("execution_source", execution_source, "src/simsopt_jax"),
            RoleRoot("configuration", configuration, "inputs/input.json"),
            RoleRoot("benchmark", benchmark_root, "benchmarks"),
            RoleRoot("test", test, "tests/test_control.py"),
            RoleRoot("native_extension", native_extension, "src/simsoptpp.py"),
        ),
    )
    launch = build_snapshot_module_launch(
        Path(sys.executable), clean_snapshot, _MODULE, (), {"PATH": "/usr/bin"}
    )

    completed = subprocess.run(
        launch.argv,
        cwd=launch.cwd,
        env=dict(launch.environment),
        check=True,
        capture_output=True,
        text=True,
    )

    assert Path(completed.stdout.strip()).resolve().is_relative_to(clean_snapshot)
