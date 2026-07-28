"""Source-fidelity checks for the Boozer-QA mirror."""

from __future__ import annotations

from pathlib import Path


def test_boozerqa_mirror_uses_memory_bounded_host_outer_solve() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/jax/2_Intermediate/boozerQA.py"
    ).read_text(encoding="utf-8")

    assert "BoozerSurfaceJAX" in source
    assert "make_traceable_objective_runtime_bundle" in source
    assert "scalar_example_driver" in source
    assert "minimize_lbfgs_host_core" in source
    assert "minimize_bfgs_host_core" in source
    assert "line_search_value_and_grad_more_thuente_host" in source
    assert "serial_solve_jax" not in source
    assert '"non_qs_weight": 1.0' in source
    assert '"iota_weight": 1.0' in source
    assert '"major_radius_weight": 1.0' in source
    assert '"length_weight": 1.0' in source
    assert '"newton_maxiter": 20' in source
    assert "qs_resolution = 20" in source
    assert "bounded_steps=2" in source
    assert "scipy" not in source.lower()
