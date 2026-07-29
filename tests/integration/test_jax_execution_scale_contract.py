"""Cross-runner contract for bounded and native-default example execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.jax.manifest_runtime import RuntimeExample
from examples.jax.parity.child import main as run_parity_child
from examples.jax.parity.input_bundle import (
    create_input_bundle,
    read_input_bundle,
)
from examples.jax.parity.runner import build_child_command as build_parity_command
from examples.jax.run_parity import _parse_arguments as parse_parity_arguments
from examples.jax.run_examples import (
    _parse_arguments as parse_example_arguments,
)
from examples.jax.run_examples import (
    build_child_command as build_example_command,
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
