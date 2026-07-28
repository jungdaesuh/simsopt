"""Source-fidelity checks for the particle-tracing mirror."""

from __future__ import annotations

from pathlib import Path


def test_particle_mirror_preserves_native_sampling_and_trace_contract() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/jax/1_Simple/tracing_particle.py"
    ).read_text(encoding="utf-8")

    assert "RandomState(1)" in source
    assert "KINETIC_ENERGY = 5_000.0 * ONE_EV" in source
    assert 'mode="gc_vac"' in source
    assert "forget_exact_path=True" in source
    assert "range(4)" in source
