"""Pure-JAX flat-675 single-stage objective.

The flat formulation optimizes all 675 outer coordinates at once — 11 coil, 3
vessel, 661 surface — with only the two Boozer scalars ``(iota, G)`` left in
an inner state that a two-column least-squares solve closes in the forward
pass.  The result is one differentiable device program with no nested solver,
which is what lets the fused on-device L-BFGS recipe run it end to end.

The physics reuses this repo's shared single-stage math; what is vendored here
is the 675 layout, its typed policy, the Boozer system construction, and the
frozen-bundle readers.
"""

from __future__ import annotations

from .boozer_material import (
    Flat675BoozerMaterial,
    Flat675BoozerSystem,
    Flat675CandidateGeometry,
    build_flat675_boozer_system,
    flat675_candidate_geometry,
)
from .bundle import Flat675Bundle, load_flat675_bundle
from .construction import (
    CERTIFIED_MPOL,
    CERTIFIED_NPHI,
    CERTIFIED_NTHETA,
    CERTIFIED_NTOR,
    CERTIFIED_STELLSYM,
    CERTIFIED_SURFACE_RANGE,
    DEFAULT_FLAT675_BOOZER_POLICY,
    DEFAULT_VESSEL_CLEARANCE_FACTOR,
    CoilDofExtractionProvider,
    Flat675Problem,
    assemble_flat675_problem,
    build_flat675_problem,
    default_flat675_objective_policy,
    fit_flat675_boundary,
    require_certified_coil_layout,
    require_certified_surface_layout,
    synthesize_flat675_vessel,
)
from .formulation import (
    FLAT675_COIL_DOF_COUNT,
    FLAT675_COIL_SLICE,
    FLAT675_OUTER_DOF_COUNT,
    FLAT675_SURFACE_DOF_COUNT,
    FLAT675_SURFACE_SLICE,
    FLAT675_VESSEL_DOF_COUNT,
    FLAT675_VESSEL_SLICE,
    Flat675Candidate,
    Flat675ContractError,
)
from .layout import (
    CERTIFIED_FLAT_LAYOUT,
    FLAT_VESSEL_DOF_COUNT,
    FlatLayoutError,
    FlatSingleStageLayout,
    surface_block_dof_count,
)
from .manifest import (
    Flat675InputManifest,
    load_flat675_input_manifest,
    load_flat675_vessel_template,
)
from .objective import (
    Flat675Material,
    Flat675Programs,
    bind_flat675_programs,
    flat675_objective,
    flat675_weighted_terms,
)
from .policy import (
    FLAT675_OBJECTIVE_TERM_KEYS,
    Flat675BoozerLabelType,
    Flat675BoozerSystemPolicy,
    Flat675HardwareSoftPenaltyPolicy,
    Flat675ObjectivePolicy,
)
from .runtime_spec_loader import load_flat675_runtime_spec
from .y_solve import Flat675YSolution, solve_flat675_y_qr

__all__ = [
    "CERTIFIED_FLAT_LAYOUT",
    "CERTIFIED_MPOL",
    "CERTIFIED_NPHI",
    "CERTIFIED_NTHETA",
    "CERTIFIED_NTOR",
    "CERTIFIED_STELLSYM",
    "CERTIFIED_SURFACE_RANGE",
    "DEFAULT_FLAT675_BOOZER_POLICY",
    "DEFAULT_VESSEL_CLEARANCE_FACTOR",
    "FLAT675_COIL_DOF_COUNT",
    "FLAT675_COIL_SLICE",
    "FLAT675_OBJECTIVE_TERM_KEYS",
    "FLAT675_OUTER_DOF_COUNT",
    "FLAT675_SURFACE_DOF_COUNT",
    "FLAT675_SURFACE_SLICE",
    "FLAT675_VESSEL_DOF_COUNT",
    "FLAT675_VESSEL_SLICE",
    "FLAT_VESSEL_DOF_COUNT",
    "CoilDofExtractionProvider",
    "Flat675BoozerLabelType",
    "Flat675BoozerMaterial",
    "Flat675BoozerSystem",
    "Flat675BoozerSystemPolicy",
    "Flat675Bundle",
    "Flat675Candidate",
    "Flat675CandidateGeometry",
    "Flat675ContractError",
    "Flat675HardwareSoftPenaltyPolicy",
    "Flat675InputManifest",
    "Flat675Material",
    "Flat675ObjectivePolicy",
    "Flat675Problem",
    "Flat675Programs",
    "Flat675YSolution",
    "FlatLayoutError",
    "FlatSingleStageLayout",
    "assemble_flat675_problem",
    "bind_flat675_programs",
    "build_flat675_boozer_system",
    "build_flat675_problem",
    "default_flat675_objective_policy",
    "fit_flat675_boundary",
    "flat675_candidate_geometry",
    "flat675_objective",
    "flat675_weighted_terms",
    "load_flat675_bundle",
    "load_flat675_input_manifest",
    "load_flat675_runtime_spec",
    "load_flat675_vessel_template",
    "require_certified_coil_layout",
    "require_certified_surface_layout",
    "solve_flat675_y_qr",
    "surface_block_dof_count",
    "synthesize_flat675_vessel",
]
