"""Fail-closed artifact contract for VMEC-hybrid authority."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "jax_vmec_hybrid_authority.yml"


def test_vmec_hybrid_authority_publishes_immutable_build_receipt() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "vmec-build.json" in source
    assert '"vmec_sha256"' in source
    assert '"mpi_world_size"' in source
    assert '"repository_commit"' in source
    assert '"repository_dirty"' in source
    assert "json.dump" in source
