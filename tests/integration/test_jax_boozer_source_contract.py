"""Source-fidelity checks for the Boozer-surface mirror."""

from __future__ import annotations

from pathlib import Path


def test_boozer_mirror_preserves_three_stage_label_workflow() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/jax/2_Intermediate/boozer.py"
    ).read_text(encoding="utf-8")

    assert "BoozerSurfaceJAX" in source
    assert "minimize_boozer_penalty_constraints_LBFGS" in source
    assert "minimize_boozer_penalty_constraints_ls" in source
    assert "target_flux = 3.0 * float(toroidal_flux.J())" in source
    assert "scipy" not in source.lower()
