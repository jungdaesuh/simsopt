"""Runner contract for atomic legacy and canonical manifest pairs."""

from __future__ import annotations

from pathlib import Path

import pytest
import examples.jax.run_examples as runner
from examples.jax.manifest_runtime import (
    RuntimeContractPair,
    RuntimeExample,
)
from examples.jax.parity._manifest import ParityManifest


def _runtime_pair(version_pair: tuple[int, int]) -> RuntimeContractPair:
    return RuntimeContractPair(
        version_pair=version_pair,
        used_legacy_adapter=version_pair == (2, 1),
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
                compatibility=None,
            ),
        ),
        parity=ParityManifest(schema_version=version_pair[1], relationships=()),
    )


def test_runner_observability_binds_both_manifest_versions() -> None:
    assert runner.manifest_observability_payload(_runtime_pair((3, 2))) == {
        "examples_manifest_schema_version": 3,
        "parity_manifest_schema_version": 2,
        "used_legacy_manifest_adapter": False,
    }


def test_runner_main_loads_one_atomic_manifest_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples_path = tmp_path / "manifest.json"
    parity_path = tmp_path / "parity_manifest.json"
    observed_paths: list[tuple[Path, Path]] = []

    def load_pair(
        selected_examples: Path,
        selected_parity: Path,
        *,
        repo_root: Path,
    ) -> RuntimeContractPair:
        assert repo_root == runner._REPO_ROOT
        observed_paths.append((selected_examples, selected_parity))
        return _runtime_pair((3, 2))

    monkeypatch.setattr(runner, "load_runtime_contract_pair", load_pair)
    monkeypatch.setattr(runner, "run_profile", lambda *_args, **_kwargs: 0)

    exit_code = runner.main(
        [
            "--device",
            "cpu",
            "--manifest",
            str(examples_path),
            "--parity-manifest",
            str(parity_path),
        ]
    )

    assert exit_code == 0
    assert observed_paths == [(examples_path, parity_path)]
