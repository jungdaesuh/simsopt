"""Source-fidelity checks for the Boozer-surface mirror."""

from __future__ import annotations

from pathlib import Path


def test_boozer_mirror_preserves_three_stage_label_workflow() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/jax/2_Intermediate/boozer.py"
    ).read_text(encoding="utf-8")

    assert "BoozerSurfaceJAX" in source
    # The three official stages are driven through their single owner,
    # simsopt_jax.examples.boozer_official, which the parity case
    # examples/jax/parity/cases/native_boozer.py drives too: one L-BFGS-B
    # reduction, one manual Levenberg-Marquardt polish, and one more after the
    # label is switched to three times the converged toroidal flux (upstream
    # examples/2_Intermediate/boozer.py:45-63 at 9e027eac3).
    assert "run_boozer_lbfgs_stage(" in source
    assert source.count("run_boozer_manual_stage(") == 2
    assert "target_flux = OFFICIAL_FLUX_MULTIPLIER * float(toroidal_flux.J())" in source
    assert "scipy" not in source.lower()
