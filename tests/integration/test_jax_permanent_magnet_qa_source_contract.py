"""Source-fidelity checks for the permanent-magnet QA mirror."""

from __future__ import annotations

from pathlib import Path


def test_permanent_magnet_qa_uses_jax_coil_and_relax_split_solves() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/jax/2_Intermediate/permanent_magnet_QA.py"
    ).read_text(encoding="utf-8")

    assert "make_stage_two_objective" in source
    assert "serial_solve_jax" in source
    assert "PermanentMagnetGridJAX" in source
    assert source.count("relax_and_split_jax(") == 2
    assert "require_success=False" in source
    assert "scipy" not in source.lower()
