"""Cross-runner contract for bounded and native-default example execution."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax  # noqa: F401
import numpy as np
import pytest
from examples.jax.manifest_runtime import RuntimeExample
from examples.jax.parity import child as parity_child
from examples.jax.parity import provenance as parity_provenance
from examples.jax.parity.cases import get_case
from examples.jax.parity.child import main as run_parity_child
from examples.jax.parity.input_bundle import (
    create_input_bundle,
    read_input_bundle,
)
from examples.jax.parity.provenance import SnapshotLaneIdentity, SnapshotProfileId
from examples.jax.parity.receipts import load_lane_observation
from examples.jax.parity.runner import build_child_command as build_parity_command
from examples.jax.run_examples import (
    _parse_arguments as parse_example_arguments,
)
from examples.jax.run_examples import (
    build_child_command as build_example_command,
)
from examples.jax.run_parity import _parse_arguments as parse_parity_arguments
from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example


def test_input_bundle_import_does_not_load_eager_example_package() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import sys; import examples.jax.parity.input_bundle; assert "
                "'simsopt_jax.examples' not in sys.modules; assert "
                "'jax' not in sys.modules; assert 'jaxlib' not in sys.modules"
            ),
        ),
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _snapshot_identity(profile_id: SnapshotProfileId) -> SnapshotLaneIdentity:
    return SnapshotLaneIdentity(
        profile_id=profile_id,
        lane="native-cpu" if profile_id == "native_cpu" else "jax-gpu",
        backend_mode="native_cpu",
        driver="driver",
        execution_platform="cpu" if profile_id == "native_cpu" else "gpu",
        runtime_identity_sha256="1" * 64,
        source_sha256="2" * 64,
        gpu_uuid="GPU-test",
        snapshot_root=parity_child._REPO_ROOT,
        repository_commit="a" * 40,
        repository_dirty=False,
        tracked_diff_sha256="3" * 64,
        untracked_files=(),
        manifest_entries={},
        native_extension_path=parity_child._REPO_ROOT / "simsoptpp.so",
        native_extension_sha256="4" * 64,
        interpreter_path=Path(sys.executable),
        python_version=sys.version.split()[0],
        jax_version="test",
        jaxlib_version="test",
        bound_environment={},
        static_environment={},
    )


def _example() -> RuntimeExample:
    return RuntimeExample(
        id="native-just-a-quadratic",
        path="1_Simple/just_a_quadratic.py",
        status="ready",
        lanes=("cpu-smoke", "gpu-strict"),
        smoke_args=("--max-steps", "2"),
        classification="mirror",
        teaching_kind="one_to_one",
        source="1_Simple/just_a_quadratic.py",
        compatibility=None,
    )


def test_example_scale_defaults_to_bounded_and_accepts_native_default() -> None:
    bounded = parse_example_arguments(("--device", "cpu"))
    native_default = parse_example_arguments(
        ("--device", "cpu", "--scale", "native_default")
    )

    assert bounded.scale == "bounded"
    assert native_default.scale == "native_default"


def test_example_child_argv_derives_smoke_only_from_typed_scale(
    tmp_path: Path,
) -> None:
    bounded = build_example_command(
        _example(),
        repo_root=tmp_path,
        scale="bounded",
    )
    native_default = build_example_command(
        _example(),
        repo_root=tmp_path,
        scale="native_default",
    )

    prefix = (
        sys.executable,
        "-S",
        str(tmp_path / "examples" / "jax" / "1_Simple" / "just_a_quadratic.py"),
    )
    assert bounded == (
        *prefix,
        "--smoke",
        "--json",
        "--max-steps",
        "2",
    )
    assert native_default == (
        *prefix,
        "--json",
        "--max-steps",
        "2",
    )


@pytest.mark.parametrize(
    ("arguments", "expected_scale", "expected_steps"),
    (
        (("--json", "--max-steps", "1"), "native_default", 1),
        (("--smoke", "--json", "--max-steps", "999"), "bounded", 999),
    ),
)
def test_shared_example_runner_passes_scale_independently_of_step_budget(
    tmp_path: Path,
    arguments: tuple[str, ...],
    expected_scale: ExecutionScale,
    expected_steps: int,
) -> None:
    observed: list[tuple[int, ExecutionScale]] = []

    def solve(
        _output_directory: Path,
        max_steps: int,
        scale: ExecutionScale,
    ) -> ExampleResult:
        observed.append((max_steps, scale))
        return ExampleResult(example_id="scale-probe", observables={}, status="ok")

    exit_code = run_example(
        [*arguments, "--output-dir", str(tmp_path)],
        description=None,
        temporary_prefix="unused-",
        bounded_steps=2,
        native_default_steps=20,
        solve=solve,
    )

    assert exit_code == 0
    assert observed == [(expected_steps, expected_scale)]


def test_shared_examples_never_infer_scale_from_step_thresholds() -> None:
    examples_root = Path(__file__).resolve().parents[2] / "examples" / "jax"
    offenders: list[str] = []
    for path in sorted(examples_root.glob("[123]_*/*.py")):
        source = path.read_text(encoding="utf-8")
        if "run_example(" not in source:
            continue
        module = ast.parse(source)
        for node in ast.walk(module):
            if not isinstance(node, ast.Compare):
                continue
            names = {
                child.id for child in ast.walk(node) if isinstance(child, ast.Name)
            }
            if "max_steps" in names and any(
                name.startswith("NATIVE_") for name in names
            ):
                offenders.append(str(path.relative_to(examples_root)))

    assert offenders == []


def test_parity_child_argv_carries_scale_without_boolean_inference(
    tmp_path: Path,
) -> None:
    command = build_parity_command(
        python_executable="/venv/bin/python",
        case_id="quadratic",
        lane="jax-gpu",
        input_bundle_path=tmp_path / "input_bundle.json",
        result_directory=tmp_path / "result",
        scale="native_default",
    )

    assert command[-2:] == ("--scale", "native_default")
    assert "--smoke" not in command


def test_parity_scale_defaults_to_bounded_and_rejects_smoke_conflict(
    tmp_path: Path,
) -> None:
    required = (
        "--case",
        "quadratic",
        "--lanes",
        "native-cpu",
        "--artifact-root",
        str(tmp_path),
    )

    assert parse_parity_arguments(list(required)).scale == "bounded"
    assert (
        parse_parity_arguments([*required, "--scale", "native_default"]).scale
        == "native_default"
    )
    with pytest.raises(SystemExit):
        parse_parity_arguments([*required, "--scale", "native_default", "--smoke"])


def test_input_bundle_persists_scale_inside_its_fingerprint(
    tmp_path: Path,
) -> None:
    bundle = create_input_bundle(
        tmp_path,
        case_id="quadratic",
        random_seed=0,
        arrays={"parameters": np.asarray([1.0], dtype=np.float64)},
        configuration={"max_steps": 2},
        scale="native_default",
    )
    loaded, _arrays = read_input_bundle(tmp_path)

    assert bundle.scale == "native_default"
    assert loaded == bundle


def test_input_bundle_rejects_scale_changed_without_fingerprint(
    tmp_path: Path,
) -> None:
    create_input_bundle(
        tmp_path,
        case_id="quadratic",
        random_seed=0,
        arrays={"parameters": np.asarray([1.0], dtype=np.float64)},
        configuration={"max_steps": 2},
        scale="bounded",
    )
    path = tmp_path / "input_bundle.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scale"] = "native_default"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        read_input_bundle(tmp_path)


def test_parity_child_rejects_requested_scale_different_from_input(
    tmp_path: Path,
) -> None:
    create_input_bundle(
        tmp_path / "inputs",
        case_id="traceable-least-squares",
        random_seed=0,
        arrays={"parameters": np.asarray([1.0], dtype=np.float64)},
        configuration={"max_steps": 2},
        scale="bounded",
    )

    with pytest.raises(ValueError, match="does not match requested"):
        run_parity_child(
            [
                "--case",
                "traceable-least-squares",
                "--lane",
                "native-cpu",
                "--input-bundle",
                str(tmp_path / "inputs" / "input_bundle.json"),
                "--result-directory",
                str(tmp_path / "result"),
                "--scale",
                "native_default",
            ]
        )


def test_snapshot_child_rejects_cross_profile_before_case_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parity_child,
        "load_snapshot_lane_identity",
        lambda _path: _snapshot_identity("jax_gpu_fast"),
    )
    monkeypatch.setattr(
        parity_child,
        "get_case",
        lambda _case_id: pytest.fail("case executed before profile validation"),
    )

    with pytest.raises(ValueError, match="profile does not match"):
        run_parity_child(
            [
                "--case",
                "traceable-least-squares",
                "--lane",
                "native-cpu",
                "--input-bundle",
                "/unused/input_bundle.json",
                "--result-directory",
                "/unused/result",
                "--immutable-snapshot-provenance",
                "/unused/identity.json",
                "--scale",
                "bounded",
            ]
        )


def test_snapshot_child_does_not_publish_when_dynamic_provenance_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_bundle = get_case("traceable-least-squares").create_input(
        tmp_path / "inputs", "bounded"
    )
    identity = _snapshot_identity("native_cpu")
    identity = replace(
        identity,
        backend_mode="native_cpu",
        driver="scipy_least_squares",
    )
    monkeypatch.setattr(
        parity_child, "load_snapshot_lane_identity", lambda _path: identity
    )
    monkeypatch.setattr(
        parity_child,
        "collect_snapshot_lane_provenance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("dynamic provenance unavailable")
        ),
    )

    with pytest.raises(ValueError, match="dynamic provenance unavailable"):
        run_parity_child(
            [
                "--case",
                "traceable-least-squares",
                "--lane",
                "native-cpu",
                "--input-bundle",
                str(tmp_path / "inputs" / "input_bundle.json"),
                "--result-directory",
                str(tmp_path / "result"),
                "--immutable-snapshot-provenance",
                str(tmp_path / "identity.json"),
                "--scale",
                input_bundle.scale,
            ]
        )
    assert not (tmp_path / "result" / "lane_result.json").exists()


def test_snapshot_dynamic_source_collection_rejects_ambient_project_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    escaped = tmp_path / "escaped.py"
    escaped.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "simsopt.escaped_snapshot_test",
        SimpleNamespace(__file__=str(escaped)),
    )

    with pytest.raises(ValueError, match="escaped immutable root"):
        parity_provenance._snapshot_executed_sources(_snapshot_identity("native_cpu"))


def test_native_child_records_native_synchronization_when_jax_is_loaded(
    tmp_path: Path,
) -> None:
    input_bundle = get_case("traceable-least-squares").create_input(
        tmp_path / "inputs", "bounded"
    )

    assert (
        run_parity_child(
            [
                "--case",
                "traceable-least-squares",
                "--lane",
                "native-cpu",
                "--input-bundle",
                str(tmp_path / "inputs" / "input_bundle.json"),
                "--result-directory",
                str(tmp_path / "result"),
                "--scale",
                input_bundle.scale,
            ]
        )
        == 0
    )

    receipt = load_lane_observation(tmp_path / "result")
    assert receipt.provenance is not None
    assert (
        receipt.provenance.measurement_synchronization == "native synchronous execution"
    )
