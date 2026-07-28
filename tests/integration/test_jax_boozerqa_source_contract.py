"""Source-fidelity checks for the Boozer-QA mirror."""

from __future__ import annotations

from pathlib import Path


def test_boozerqa_mirror_uses_traceable_jax_inner_and_outer_solves() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/jax/2_Intermediate/boozerQA.py"
    ).read_text(encoding="utf-8")

    assert "BoozerSurfaceJAX" in source
    assert "make_traceable_objective_runtime_bundle" in source
    assert "serial_solve_jax" in source
    assert '"non_qs_weight": 1.0' in source
    assert '"iota_weight": 1.0' in source
    assert '"major_radius_weight": 1.0' in source
    assert '"length_weight": 1.0' in source
    assert '"newton_maxiter": 20' in source
    assert "qs_resolution = 20" in source
    assert "rtol=0.0" in source
    assert "require_success=False" in source
    assert "scipy" not in source.lower()
