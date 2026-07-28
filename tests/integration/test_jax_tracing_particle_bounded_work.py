"""Bounded-work contract for the particle-tracing tutorial."""

from __future__ import annotations

from pathlib import Path


def test_particle_smoke_bounds_adaptive_work_without_changing_tolerance() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/jax/1_Simple/tracing_particle.py"
    ).read_text(encoding="utf-8")

    assert "BOUNDED_MAX_STEPS = 512" in source
    assert "tol=1.0e-9" in source
    assert "max_steps=trace_max_steps" in source
