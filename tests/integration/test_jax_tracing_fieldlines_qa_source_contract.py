"""Source-fidelity checks for the QA field-line mirror."""

from __future__ import annotations

from pathlib import Path


def test_qa_interpolation_preserves_the_native_skip_domain() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/jax/1_Simple/tracing_fieldlines_QA.py"
    ).read_text(encoding="utf-8")

    assert "def skip(" in source
    assert "skip=skip" in source
