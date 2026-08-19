"""Assemble a frozen flat-675 input bundle into an evaluable problem.

One directory holds the runtime seed spec, the vessel material, and the
manifest; this module is where those three become the material, policies, and
start candidate that :mod:`.objective` evaluates.  It is the only module in
the port that both reads files and builds device material.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .boozer_material import Flat675BoozerMaterial
from .formulation import Flat675Candidate
from .manifest import (
    load_flat675_input_manifest,
    load_flat675_vessel_template,
)
from .objective import Flat675Material
from .policy import Flat675BoozerSystemPolicy, Flat675ObjectivePolicy
from .runtime_spec_loader import load_flat675_runtime_spec


@dataclass(frozen=True, slots=True)
class Flat675Bundle:
    """One frozen bundle resolved into everything an evaluation needs."""

    material: Flat675Material
    objective_policy: Flat675ObjectivePolicy
    boozer_policy: Flat675BoozerSystemPolicy
    start_candidate: Flat675Candidate


def load_flat675_bundle(bundle_root: Path | str) -> Flat675Bundle:
    """Read one frozen flat-675 input bundle directory."""
    root = Path(bundle_root)
    manifest = load_flat675_input_manifest(root)
    return Flat675Bundle(
        material=Flat675Material(
            boozer=Flat675BoozerMaterial.from_runtime_spec(
                load_flat675_runtime_spec(root)
            ),
            vessel_template=load_flat675_vessel_template(root),
        ),
        objective_policy=manifest.objective_policy,
        boozer_policy=manifest.boozer_construction_policy,
        start_candidate=manifest.candidate,
    )


__all__ = [
    "Flat675Bundle",
    "load_flat675_bundle",
]
