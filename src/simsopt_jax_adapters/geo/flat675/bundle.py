"""Read a frozen flat-675 input bundle as one caller of the general path.

The bundle is the archived campaign input: a runtime seed spec, a vessel
material, and a manifest.  This module turns those files into the specs the
constructor consumes and then hands them to
:func:`~.construction.assemble_flat675_problem` — the same assembly every
other caller uses.  There is deliberately no bundle-only construction route:
a divergence between the archived path and the general one is exactly the
class of drift the SSOT gate exists to forbid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .construction import Flat675Problem, assemble_flat675_problem
from .formulation import Flat675Candidate, Flat675ContractError
from .manifest import (
    load_flat675_input_manifest,
    load_flat675_vessel_template,
)
from .runtime_spec_loader import load_flat675_runtime_spec

# The frozen-bundle spelling of the constructed record.  Kept because the
# campaign harness, its tests, and the sealed run directories all name it.
Flat675Bundle = Flat675Problem


def load_flat675_bundle(bundle_root: Path | str) -> Flat675Problem:
    """Read one frozen flat-675 input bundle directory."""
    root = Path(bundle_root)
    manifest = load_flat675_input_manifest(root)
    runtime_spec = load_flat675_runtime_spec(root)
    seed = runtime_spec.seed
    problem = assemble_flat675_problem(
        surface_template=seed.surface,
        coil_dof_extraction=seed.coil_dof_extraction,
        coil_dofs=seed.coil_dofs,
        vessel_template=load_flat675_vessel_template(root),
        objective_policy=manifest.objective_policy,
        boozer_policy=manifest.boozer_construction_policy,
        nphi=runtime_spec.nphi,
        ntheta=runtime_spec.ntheta,
    )
    _require_manifest_candidate_agreement(manifest.candidate, problem.start_candidate)
    return problem


def _require_manifest_candidate_agreement(
    recorded: Flat675Candidate,
    derived: Flat675Candidate,
) -> None:
    """Fail closed unless the manifest's candidate is the geometry's own.

    The manifest publishes the archived start independently of the seed spec
    and the vessel material.  Deriving the start from the geometry keeps one
    construction path, and comparing it against the manifest keeps the
    archive's authority: a bundle whose two records disagree is not the
    archived problem, whichever one is right.
    """
    for block, recorded_block, derived_block in (
        ("coil", recorded.coil_coordinates, derived.coil_coordinates),
        ("vessel", recorded.vessel_coordinates, derived.vessel_coordinates),
        ("surface", recorded.surface_coordinates, derived.surface_coordinates),
    ):
        if not np.array_equal(
            np.asarray(recorded_block, dtype=np.float64),
            np.asarray(derived_block, dtype=np.float64),
        ):
            raise Flat675ContractError(
                f"the bundle manifest's {block} block differs from the block its "
                "own runtime spec and vessel material produce."
            )


__all__ = [
    "Flat675Bundle",
    "Flat675Problem",
    "load_flat675_bundle",
]
