"""Cross-runner contract for bounded and native-default example execution."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import jax  # noqa: F401
import numpy as np
import pytest
from examples.jax._manifest import resolve_example_implementation
from examples.jax.manifest_runtime import RuntimeExample, load_runtime_contract_pair
from examples.jax.outer_optimizer_policy import parse_outer_optimizer_policy
from examples.jax.parity.cases import get_case
from examples.jax.parity.child import main as run_parity_child
from examples.jax.parity.input_bundle import (
    create_input_bundle,
    read_input_bundle,
)
from examples.jax.parity.receipts import load_lane_observation
from examples.jax.parity.runner import build_child_command as build_parity_command
from examples.jax.run_examples import (
    _parse_arguments as parse_example_arguments,
)
from examples.jax.run_examples import (
    build_child_command as build_example_command,
)
from examples.jax.run_parity import _parse_arguments as parse_parity_arguments
from simsopt_contracts import examples_runtime as shared_example_runtime
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


def _example() -> RuntimeExample:
    return RuntimeExample(
        id="native-just-a-quadratic",
        path="1_Simple/just_a_quadratic.py",
        status="ready",
        lanes=("cpu-smoke", "gpu-strict"),
        smoke_args=("--max-steps", "2"),
        classification="adapter",
        teaching_kind="one_to_one",
        source="1_Simple/just_a_quadratic.py",
        outer_optimizer_policy=parse_outer_optimizer_policy(
            "scipy-trf-over-jax-quadratic",
            example_id="native-just-a-quadratic",
            example_path="1_Simple/just_a_quadratic.py",
            ready=True,
        ),
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
    """No example decides its scale by comparing steps against a native budget.

    The sweep reads each tier script AND every module it reaches in
    ``simsopt_jax_adapters.examples``, the sanctioned home of example
    implementations, so relocating an example's code out of ``examples/jax``
    cannot shrink the gate.  Measured at this commit: 40 scripts on disk under
    ``examples/jax/[123]_*``, of which 39 define ``main`` themselves and one
    forwards into that package; the manifest declares 41 examples (39 ready and
    2 planned, one planned row having no file yet).  The scanned set is bound to
    DISK, every on-disk script must be a declared example, and the manifest's
    ready rows are a second bound -- so neither a glob change, a manifest row
    deletion, nor a relocation can shrink coverage silently.
    """

    repo_root = Path(__file__).resolve().parents[2]
    examples_root = repo_root / "examples" / "jax"
    on_disk = {
        f"{tier}/{path.name}"
        for tier in ("1_Simple", "2_Intermediate", "3_Advanced")
        for path in (examples_root / tier).iterdir()
        if path.is_file() and path.suffix == ".py"
    }
    examples = load_runtime_contract_pair(
        examples_root / "manifest.json",
        examples_root / "parity_manifest.json",
        repo_root=repo_root,
    ).examples
    declared_paths = {example.path for example in examples}
    ready_paths = {example.path for example in examples if example.status == "ready"}

    scanned: set[str] = set()
    offenders: list[str] = []
    for path in sorted(examples_root.glob("[123]_*/*.py")):
        implementation = resolve_example_implementation(path, repo_root=repo_root)
        example_path = path.relative_to(examples_root).as_posix()
        scanned.add(example_path)
        for source in implementation.sources:
            for node in ast.walk(ast.parse(source.text)):
                if not isinstance(node, ast.Compare):
                    continue
                names = {
                    child.id for child in ast.walk(node) if isinstance(child, ast.Name)
                }
                if "max_steps" in names and any(
                    name.startswith("NATIVE_") for name in names
                ):
                    offenders.append(
                        f"{example_path} -> "
                        f"{source.path.relative_to(repo_root)}:{node.lineno}"
                    )

    assert scanned == on_disk, (
        "the scale-inference sweep does not cover every tier script on disk: "
        f"unscanned={sorted(on_disk - scanned)}, unexpected={sorted(scanned - on_disk)}"
    )
    undeclared = sorted(on_disk - declared_paths)
    assert not undeclared, (
        f"tier scripts on disk that no manifest row declares: {undeclared}"
    )
    uncovered = sorted(ready_paths - scanned)
    assert not uncovered, (
        f"ready manifest examples left the scale-inference sweep: {uncovered}"
    )
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
        case_id="native-just-a-quadratic",
        random_seed=0,
        arrays={"parameters": np.asarray([1.0], dtype=np.float64)},
        configuration={"max_steps": 2},
        scale="bounded",
    )

    with pytest.raises(ValueError, match="does not match requested"):
        run_parity_child(
            [
                "--case",
                "native-just-a-quadratic",
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


def test_native_child_records_native_synchronization_when_jax_is_loaded(
    tmp_path: Path,
) -> None:
    input_bundle = get_case("native-just-a-quadratic").create_input(
        tmp_path / "inputs", "bounded"
    )

    assert (
        run_parity_child(
            [
                "--case",
                "native-just-a-quadratic",
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


def test_jax_example_result_extends_the_shared_runtime() -> None:
    assert issubclass(ExampleResult, shared_example_runtime.ExampleResult)


def test_shared_example_runtime_imports_without_native_or_jax() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            "-c",
            "import sys; from simsopt_contracts import examples_runtime; "
            "assert examples_runtime.EXECUTION_SCALES == ('bounded', 'native_default'); "
            "assert not {'simsopt', 'simsoptpp', 'jax', 'jaxlib'} & sys.modules.keys()",
        ),
        cwd=Path(__file__).resolve().parents[2],
        # ``-S`` drops site-packages; the repository sources are the child's
        # only import root, so the import cannot lean on any installed package.
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
