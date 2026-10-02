"""Runner contract for the example manifest."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from pathlib import Path

import pytest
import examples.jax.run_examples as runner
from examples.jax.manifest_runtime import (
    RuntimeExample,
    RuntimeManifest,
)


def _runtime_manifest(schema_version: int) -> RuntimeManifest:
    return RuntimeManifest(
        schema_version=schema_version,
        examples=(
            RuntimeExample(
                id="native-just-a-quadratic",
                path="1_Simple/just_a_quadratic.py",
                status="ready",
                lanes=("cpu-smoke", "gpu-strict"),
                smoke_args=(),
                classification="mirror",
                teaching_kind="one_to_one",
                source="1_Simple/just_a_quadratic.py",
            ),
        ),
    )


def test_runner_observability_binds_the_manifest_version() -> None:
    assert runner.manifest_observability_payload(_runtime_manifest(3)) == {
        "examples_manifest_schema_version": 3,
        "used_legacy_manifest_adapter": False,
    }


def test_runner_main_loads_one_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples_path = tmp_path / "manifest.json"
    observed_paths: list[Path] = []

    def load_manifest(selected_examples: Path, *, repo_root: Path) -> RuntimeManifest:
        assert repo_root == runner._REPO_ROOT
        observed_paths.append(selected_examples)
        return _runtime_manifest(3)

    monkeypatch.setattr(runner, "load_runtime_manifest", load_manifest)
    monkeypatch.setattr(runner, "run_profile", lambda *_args, **_kwargs: 0)

    exit_code = runner.main(["--device", "cpu", "--manifest", str(examples_path)])

    assert exit_code == 0
    assert observed_paths == [examples_path]
