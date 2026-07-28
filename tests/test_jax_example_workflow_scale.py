"""Workflow scale selection must be explicit and fail closed."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / ".github" / "workflows" / "jax_smoke.yml"
AUTHORITY = ROOT / ".github" / "workflows" / "jax_gpu_parity.yml"


def test_pr_example_commands_select_bounded_scale_explicitly() -> None:
    source = SMOKE.read_text(encoding="utf-8")

    assert "run_examples.py --device cpu --scale bounded" in source
    assert (
        "run_examples.py --device cpu --intent parity --scale bounded" in source
    )
    assert "run_examples.py --device gpu --scale bounded" in source
    assert (
        "run_examples.py --device gpu --intent parity --scale bounded" in source
    )
    parity_commands = source.split("python examples/jax/run_parity.py")[1:]
    assert len(parity_commands) == 2
    assert all("--scale bounded" in command for command in parity_commands)


def test_native_default_authority_is_manual_and_explicit() -> None:
    source = AUTHORITY.read_text(encoding="utf-8")

    assert "run_native_default:" in source
    assert "if: github.event_name == 'workflow_dispatch' && inputs.run_native_default" in (
        source
    )
    assert "--scale bounded" in source
    assert "--scale native_default" in source
