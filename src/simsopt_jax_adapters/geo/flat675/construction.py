"""Build a flat-675 problem from simsopt objects.

This is the single assembly path for the promoted flat coupled single-stage
formulation.  Two front doors reach it: :func:`build_flat675_problem`, which
fits user geometry onto the certified layout, and the frozen-bundle loader in
:mod:`.bundle`, which reads the archived campaign inputs.  Neither keeps a
private construction route — the bundle loader is one caller of
:func:`assemble_flat675_problem` like any other.  The shipped lesson that
drives both is ``examples/jax/3_Advanced/single_stage_flat675.py``.

The layout is always 11 coil + 3 vessel + 661 surface.  "No vessel" therefore
means "an inactive vessel term", never a 672-DOF vector: when the caller omits
a vessel this module synthesizes one placed far enough out that the
surface-to-vessel hinge and its gradient are exactly zero at the start.

Rung-1 scope (charter ``docs/jax_flat675_promotion_plan.md``): the boundary
must be stellarator-symmetric and is fitted at the campaign resolution, and
the coil set must be the certified owner layout.  All three restrictions are
enforced fail-closed, and each names the rung-2 change that would lift it.
Nothing here silently reshapes an input to make it fit: a boundary the
certified layout cannot represent is refused, never coerced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray
from simsopt.geo import Surface, SurfaceXYZTensorFourier
from simsopt_jax.core.specs import (
    CoilSetDofExtractionSpec,
    CurveCWSFourierRZSpec,
    CurveXYZFourierSpec,
    SurfaceRZFourierSpec,
    SurfaceXYZTensorFourierSpec,
    make_surface_rzfourier_spec,
    make_surface_xyz_tensor_fourier_spec,
)
from simsopt_jax.core.surface_dofs import surface_gamma_tangents_from_dofs
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_dofs_from_spec

from .boozer_material import Flat675BoozerMaterial
from .formulation import (
    FLAT675_COIL_DOF_COUNT,
    FLAT675_SURFACE_DOF_COUNT,
    Flat675Candidate,
    Flat675ContractError,
)
from .objective import Flat675Material
from .policy import (
    Flat675BoozerLabelType,
    Flat675BoozerSystemPolicy,
    Flat675HardwareSoftPenaltyPolicy,
    Flat675ObjectivePolicy,
)

# The certified surface layout.  mpol = ntor = 10 with stellarator symmetry is
# exactly 661 DOFs; dropping the symmetry doubles it to 1323 and the layout
# validators refuse that, so the pair is forced at rung 1.  nfp is free.
CERTIFIED_MPOL: Final[int] = 10
CERTIFIED_NTOR: Final[int] = 10
CERTIFIED_STELLSYM: Final[bool] = True
# The campaign's quadrature: a half field period in phi, a full period in
# theta.  Callers may resample, but this is the resolution the receipts speak
# to and the default the constructor uses.
CERTIFIED_NPHI: Final[int] = 255
CERTIFIED_NTHETA: Final[int] = 64
CERTIFIED_SURFACE_RANGE: Final[str] = "half period"

# The synthesized vessel sits this many hinge thresholds outside the boundary's
# own extent.  Any factor above one leaves the hinge exactly inactive; the
# margin keeps it inactive after a few optimizer steps rather than only at the
# start.
DEFAULT_VESSEL_CLEARANCE_FACTOR: Final[float] = 3.0

_RUNG_TWO_SURFACE_MESSAGE: Final[str] = (
    "Rung 1 fits onto the stellarator-symmetric 661-DOF layout, so a boundary "
    "that is not stellarator-symmetric cannot be represented: its rs/zc "
    "content has no home in the target layout at any resolution, and "
    "returning the symmetrized shape would silently change the plasma. If the "
    "boundary is in fact symmetric and only its flag says otherwise, rebuild "
    "it with stellsym=True. Carrying genuine asymmetry through the "
    "formulation is the rung-2 chartered change in "
    "docs/jax_flat675_promotion_plan.md, reported rather than attempted."
)

_RUNG_TWO_COIL_MESSAGE: Final[str] = (
    "Rung 1 accepts only the certified coil owner layout: one free current at "
    "owner DOF 0, a CurveCWSFourierRZ family carrying owner DOFs 1-10, and "
    "fixed CurveXYZFourier TF coils with empty owner maps. Generic coil sets "
    "(for example create_equally_spaced_curves with per-coil Current objects) "
    "cannot satisfy the 11-owner validator. Relaxing it is the rung-2 "
    "chartered change in docs/jax_flat675_promotion_plan.md, reported rather "
    "than attempted."
)


class CoilDofExtractionProvider(Protocol):
    """The narrow field interface this module consumes.

    Only the owner-DOF contract and the current owner values are needed, so
    the constructor depends on those two members rather than on a concrete
    Biot-Savart class.
    """

    def coil_dof_extraction_spec(self) -> CoilSetDofExtractionSpec: ...

    @property
    def x(self) -> object: ...


@dataclass(frozen=True, slots=True)
class Flat675Problem:
    """Everything an evaluation needs: material, both policies, and a start."""

    material: Flat675Material
    objective_policy: Flat675ObjectivePolicy
    boozer_policy: Flat675BoozerSystemPolicy
    start_candidate: Flat675Candidate


# --------------------------------------------------------------------------
# Frozen campaign defaults (charter: documented defaults, not knobs)
# --------------------------------------------------------------------------

DEFAULT_FLAT675_BOOZER_POLICY: Final[Flat675BoozerSystemPolicy] = (
    Flat675BoozerSystemPolicy(weight_by_inverse_field_magnitude=True)
)


def default_flat675_objective_policy(
    *,
    optimized_coil_index: int,
) -> Flat675ObjectivePolicy:
    """The campaign's frozen weights, targets, and hardware penalties.

    These are the values the sealed F3 receipts were measured under.  They are
    defaults rather than constants because a caller may legitimately optimize a
    different problem; a caller who changes them is no longer running the
    configuration those receipts certify.
    """
    return Flat675ObjectivePolicy(
        iota_target=0.15,
        boozer_constraint_weight=1.0,
        boozer_target_label=0.1,
        boozer_label_type=Flat675BoozerLabelType.VOLUME,
        boozer_label_phi_index=0,
        non_qs_weight=1.0,
        non_qs_grid_size=40,
        residual_weight=1000.0,
        iota_weight=100.0,
        length_weight=1.0,
        length_target_m=1.7,
        curve_curve_weight=1000.0,
        curve_curve_threshold_m=0.05,
        curve_surface_weight=1.0,
        curve_surface_threshold_m=0.02,
        surface_vessel_weight=1000.0,
        surface_vessel_threshold_m=0.04,
        curvature_weight=1.0,
        curvature_threshold_inverse_m=40.0,
        optimized_coil_index=optimized_coil_index,
        hardware_soft_penalty_policy=Flat675HardwareSoftPenaltyPolicy(
            common_scale=10.0,
            curve_curve_base_weight=100.0,
            curvature_base_weight=0.1,
            curve_curve_threshold_m=0.05,
            curvature_threshold_inverse_m=40.0,
            penalty_exponent=2.0,
        ),
    )


# --------------------------------------------------------------------------
# Boundary: fit any compatible surface onto the certified layout
# --------------------------------------------------------------------------


def fit_flat675_boundary(
    boundary: Surface,
    *,
    nphi: int = CERTIFIED_NPHI,
    ntheta: int = CERTIFIED_NTHETA,
) -> SurfaceXYZTensorFourierSpec:
    """Fit a boundary onto the certified 661-DOF surface layout.

    The fit is simsopt's own ``least_squares_fit`` against the boundary's
    surface points sampled on the certified quadrature, so the result is the
    boundary this repo's surface machinery would produce, not a reimplemented
    projection.

    Two things the caller must know, because a fit onto a fixed layout cannot
    be lossless in general:

    * **Stellarator symmetry is required, not imposed.**  The target layout is
      stellarator-symmetric, so a boundary carrying ``rs``/``zc`` content has
      no representation in it at any resolution.  Such a boundary is refused
      rather than symmetrized: quietly dropping that content would return a
      different plasma shape under the caller's own variable name.
    * **Poloidal and toroidal resolution above the certified layout is
      truncated**, and that is the documented meaning of "fit".  Unlike the
      symmetry case this is an approximation *within* the target layout — it
      converges to the boundary as the layout's resolution grows, and the
      layout in question is stated here (``mpol = ntor = 10``).  A boundary
      with content above those modes comes back smoothed; measure the
      deviation if it matters to the study.
    """
    if not isinstance(boundary, Surface):
        raise Flat675ContractError(
            f"boundary must be a simsopt Surface; got {type(boundary).__name__}."
        )
    source = boundary.to_RZFourier()
    if not source.stellsym:
        raise Flat675ContractError(_RUNG_TWO_SURFACE_MESSAGE)
    nfp = int(boundary.nfp)
    quadpoints = Surface.get_quadpoints(
        nfp=nfp,
        range=CERTIFIED_SURFACE_RANGE,
        nphi=int(nphi),
        ntheta=int(ntheta),
    )
    fitted = SurfaceXYZTensorFourier(
        mpol=CERTIFIED_MPOL,
        ntor=CERTIFIED_NTOR,
        nfp=nfp,
        stellsym=CERTIFIED_STELLSYM,
        quadpoints_phi=quadpoints[0],
        quadpoints_theta=quadpoints[1],
    )
    resampled = type(source)(
        nfp=source.nfp,
        stellsym=source.stellsym,
        mpol=source.mpol,
        ntor=source.ntor,
        quadpoints_phi=fitted.quadpoints_phi,
        quadpoints_theta=fitted.quadpoints_theta,
    )
    resampled.x = source.x
    fitted.least_squares_fit(resampled.gamma())
    # The width is not re-asserted here: ``fitted`` is built two statements
    # above at the certified mpol/ntor/stellsym, so its DOF count is fixed by
    # construction.  The material checks the 661 it actually depends on.
    dofs = np.asarray(fitted.get_dofs(), dtype=np.float64)
    return make_surface_xyz_tensor_fourier_spec(
        dofs=dofs,
        quadpoints_phi=np.asarray(fitted.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(fitted.quadpoints_theta, dtype=np.float64),
        nfp=nfp,
        stellsym=CERTIFIED_STELLSYM,
        mpol=CERTIFIED_MPOL,
        ntor=CERTIFIED_NTOR,
    )


def require_certified_surface_layout(
    surface_template: SurfaceXYZTensorFourierSpec,
) -> None:
    """Refuse a surface that is not the certified 661-DOF layout."""
    if (
        surface_template.mpol != CERTIFIED_MPOL
        or surface_template.ntor != CERTIFIED_NTOR
    ):
        raise Flat675ContractError(
            f"rung 1 forces mpol=ntor={CERTIFIED_MPOL}; got "
            f"mpol={surface_template.mpol}, ntor={surface_template.ntor}. "
            "Arbitrary resolutions are the rung-2 chartered change."
        )
    if not surface_template.stellsym:
        raise Flat675ContractError(
            "rung 1 forces stellsym=True: a non-symmetric surface at "
            f"mpol=ntor={CERTIFIED_MPOL} carries 1323 DOFs, and the flat-675 "
            f"layout validators require exactly {FLAT675_SURFACE_DOF_COUNT}."
        )


# --------------------------------------------------------------------------
# Coils: the certified owner layout, or a named refusal
# --------------------------------------------------------------------------


def require_certified_coil_layout(
    extraction: CoilSetDofExtractionSpec,
) -> int:
    """Refuse anything but the certified owner layout; return the free index.

    The returned index is the first free winding-surface coil, which is the
    coil the length and curvature penalties are written against.
    """
    if not isinstance(extraction, CoilSetDofExtractionSpec):
        raise Flat675ContractError(
            "coil extraction must be a CoilSetDofExtractionSpec; got "
            f"{type(extraction).__name__}."
        )
    free_indices: list[int] = []
    expected_curve_map = tuple(
        (owner, owner + 1, owner - 1, owner)
        for owner in range(1, FLAT675_COIL_DOF_COUNT)
    )
    for index, coil in enumerate(extraction.coils):
        curve_segments = tuple(coil.curve_map.owner_segments)
        current_segments = tuple(coil.current_map.owner_segments)
        if isinstance(coil.curve, CurveXYZFourierSpec):
            if curve_segments or current_segments:
                raise Flat675ContractError(
                    f"coil {index} is a CurveXYZFourier TF coil but claims owner "
                    f"DOFs (curve={curve_segments}, current={current_segments}); "
                    "the certified layout fixes them. " + _RUNG_TWO_COIL_MESSAGE
                )
            continue
        if not isinstance(coil.curve, CurveCWSFourierRZSpec):
            raise Flat675ContractError(
                f"coil {index} has curve type {type(coil.curve).__name__}. "
                + _RUNG_TWO_COIL_MESSAGE
            )
        if curve_segments != expected_curve_map:
            raise Flat675ContractError(
                f"coil {index} does not carry owner DOFs 1-"
                f"{FLAT675_COIL_DOF_COUNT - 1} contiguously; got "
                f"{curve_segments}. " + _RUNG_TWO_COIL_MESSAGE
            )
        if current_segments != ((0, 1, 0, 1),):
            raise Flat675ContractError(
                f"coil {index} does not share the single free current at owner "
                f"DOF 0; got {current_segments}. " + _RUNG_TWO_COIL_MESSAGE
            )
        free_indices.append(index)
    if not free_indices:
        raise Flat675ContractError(
            "the coil set declares no free winding-surface coil. "
            + _RUNG_TWO_COIL_MESSAGE
        )
    return free_indices[0]


# --------------------------------------------------------------------------
# Vessel: supplied, or synthesized inactive
# --------------------------------------------------------------------------


def synthesize_flat675_vessel(
    surface_template: SurfaceXYZTensorFourierSpec,
    *,
    hinge_threshold_m: float,
    clearance_factor: float = DEFAULT_VESSEL_CLEARANCE_FACTOR,
) -> SurfaceRZFourierSpec:
    """Build the default 3-DOF vessel, placed so the hinge is exactly inactive.

    The vessel is the circular-cross-section torus that encloses the boundary
    with a uniform clearance.  Because every boundary point lies at most
    ``a_max`` from the centre circle and the vessel sits at
    ``a_max + clearance``, the closest surface-to-vessel distance is at least
    ``clearance``; with ``clearance`` above the hinge threshold the penalty is
    identically zero, so its gradient onto the three vessel DOFs is too.
    """
    points = _surface_points(surface_template)
    radius = np.hypot(points[:, 0], points[:, 1])
    height = points[:, 2]
    major_radius = 0.5 * (float(radius.max()) + float(radius.min()))
    minor_extent = float(
        np.max(np.hypot(radius - major_radius, height)),
    )
    clearance = float(hinge_threshold_m) * float(clearance_factor)
    vessel_minor = minor_extent + clearance
    block = np.zeros((2, 1), dtype=np.float64)
    block[0, 0] = major_radius
    block[1, 0] = vessel_minor
    zero_block = np.zeros((2, 1), dtype=np.float64)
    poloidal_block = np.zeros((2, 1), dtype=np.float64)
    poloidal_block[1, 0] = vessel_minor
    return make_surface_rzfourier_spec(
        rc=block,
        zs=poloidal_block,
        rs=zero_block,
        zc=zero_block,
        quadpoints_phi=np.asarray(surface_template.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(
            surface_template.quadpoints_theta, dtype=np.float64
        ),
        nfp=int(surface_template.nfp),
        stellsym=True,
    )


def _surface_points(
    surface_template: SurfaceXYZTensorFourierSpec,
) -> NDArray[np.float64]:
    """The certified surface's own Cartesian quadrature points."""
    gamma, _toroidal, _poloidal = surface_gamma_tangents_from_dofs(
        surface_template, surface_template.dofs
    )
    return np.asarray(gamma, dtype=np.float64).reshape((-1, 3))


# --------------------------------------------------------------------------
# Assembly: the one path both front doors use
# --------------------------------------------------------------------------


def assemble_flat675_problem(
    *,
    surface_template: SurfaceXYZTensorFourierSpec,
    coil_dof_extraction: CoilSetDofExtractionSpec,
    coil_dofs: object,
    vessel_template: SurfaceRZFourierSpec,
    objective_policy: Flat675ObjectivePolicy,
    boozer_policy: Flat675BoozerSystemPolicy,
    nphi: int,
    ntheta: int,
) -> Flat675Problem:
    """Bind validated material and policy into an evaluable problem.

    The start candidate is derived from the same specs the material is built
    from, so a problem cannot carry a start that disagrees with its own
    geometry.  ``coil_dofs`` must already be float64; this is the one place
    every caller's coil block passes through, so it is where the port's fp64
    contract is enforced.
    """
    require_certified_surface_layout(surface_template)
    boozer = Flat675BoozerMaterial(
        surface_template=surface_template,
        coil_dof_extraction=coil_dof_extraction,
        mpol=surface_template.mpol,
        ntor=surface_template.ntor,
        nfp=surface_template.nfp,
        nphi=int(nphi),
        ntheta=int(ntheta),
    )
    material = Flat675Material(boozer=boozer, vessel_template=vessel_template)
    coil_block = np.asarray(coil_dofs)
    if coil_block.shape != (FLAT675_COIL_DOF_COUNT,):
        raise Flat675ContractError(
            f"the coil block must carry exactly {FLAT675_COIL_DOF_COUNT} owner "
            f"DOFs; got {coil_block.shape}."
        )
    # Refused rather than promoted: a single-precision block reaching here is a
    # precision defect upstream, and silently widening it would hide the defect
    # behind an fp64 result.
    if coil_block.dtype != np.float64:
        raise Flat675ContractError(
            "the coil block must already be float64; got "
            f"{coil_block.dtype}. This port is fp64 only."
        )
    vessel_block = np.asarray(
        surface_rz_fourier_dofs_from_spec(vessel_template), dtype=np.float64
    )
    return Flat675Problem(
        material=material,
        objective_policy=objective_policy,
        boozer_policy=boozer_policy,
        start_candidate=Flat675Candidate(
            coil_coordinates=tuple(coil_block.tolist()),
            vessel_coordinates=tuple(vessel_block.tolist()),
            surface_coordinates=tuple(
                np.asarray(surface_template.dofs, dtype=np.float64).tolist()
            ),
        ),
    )


def build_flat675_problem(
    *,
    boundary: Surface,
    field: CoilDofExtractionProvider,
    vessel: SurfaceRZFourierSpec | None = None,
    nphi: int = CERTIFIED_NPHI,
    ntheta: int = CERTIFIED_NTHETA,
    objective_policy: Flat675ObjectivePolicy | None = None,
    boozer_policy: Flat675BoozerSystemPolicy | None = None,
    vessel_clearance_factor: float = DEFAULT_VESSEL_CLEARANCE_FACTOR,
) -> Flat675Problem:
    """Build a flat-675 problem from simsopt geometry.

    ``boundary`` is any simsopt surface; it is fitted onto the certified
    661-DOF layout.  ``field`` must expose the certified coil owner layout.
    ``vessel`` is optional: omitting it synthesizes one whose hinge term is
    exactly inactive at the start, so the 11+3+661 layout always holds.

    The default policy is the campaign's frozen one; the sealed receipts speak
    to that configuration, and a caller who overrides it is running a problem
    those receipts do not certify.
    """
    surface_template = fit_flat675_boundary(boundary, nphi=nphi, ntheta=ntheta)
    extraction = field.coil_dof_extraction_spec()
    free_coil_index = require_certified_coil_layout(extraction)
    policy = (
        default_flat675_objective_policy(optimized_coil_index=free_coil_index)
        if objective_policy is None
        else objective_policy
    )
    vessel_template = (
        synthesize_flat675_vessel(
            surface_template,
            hinge_threshold_m=policy.surface_vessel_threshold_m,
            clearance_factor=vessel_clearance_factor,
        )
        if vessel is None
        else vessel
    )
    return assemble_flat675_problem(
        surface_template=surface_template,
        coil_dof_extraction=extraction,
        coil_dofs=np.asarray(field.x),
        vessel_template=vessel_template,
        objective_policy=policy,
        boozer_policy=(
            DEFAULT_FLAT675_BOOZER_POLICY if boozer_policy is None else boozer_policy
        ),
        nphi=int(nphi),
        ntheta=int(ntheta),
    )


__all__ = [
    "CERTIFIED_MPOL",
    "CERTIFIED_NPHI",
    "CERTIFIED_NTHETA",
    "CERTIFIED_NTOR",
    "CERTIFIED_STELLSYM",
    "CERTIFIED_SURFACE_RANGE",
    "DEFAULT_FLAT675_BOOZER_POLICY",
    "DEFAULT_VESSEL_CLEARANCE_FACTOR",
    "CoilDofExtractionProvider",
    "Flat675Problem",
    "assemble_flat675_problem",
    "build_flat675_problem",
    "default_flat675_objective_policy",
    "fit_flat675_boundary",
    "require_certified_coil_layout",
    "require_certified_surface_layout",
    "synthesize_flat675_vessel",
]
