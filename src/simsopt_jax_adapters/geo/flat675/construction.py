"""Build a flat-675 problem from simsopt objects.

This is the single assembly path for the promoted flat coupled single-stage
formulation.  Two front doors reach it: :func:`build_flat675_problem`, which
fits user geometry onto the certified layout, and the frozen-bundle loader in
:mod:`.bundle`, which reads the archived campaign inputs.  Neither keeps a
private construction route — the bundle loader is one caller of
:func:`assemble_flat675_problem` like any other.  The shipped lesson that
drives both is ``examples/jax/3_Advanced/single_stage_flat675.py``.

"675" names the certified configuration, not a constraint.  The surface
resolution is chosen by explicit ``mpol``/``ntor``/``stellsym`` parameters
that default to the certified triple, and the coil block is however many
owner DOFs the coil set exposes.  The layout is never inferred from the
boundary handed in: a caller asks for a target and the boundary is fitted
onto it, which is what lets a refusal mean "your boundary does not fit the
target you asked for" rather than "your boundary is wrong".

The vessel block stays three DOFs.  "No vessel" means "an inactive vessel
term", never a shorter vector: when the caller omits a vessel this module
synthesizes one placed far enough out that the surface-to-vessel hinge and
its gradient are exactly zero at the start.

Nothing here silently reshapes an input to make it fit.  A boundary that the
requested layout cannot represent is refused, never coerced — and the refusal
names the parameter that would accept it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray
from simsopt.geo import Surface, SurfaceXYZTensorFourier
from simsopt_jax.core.specs import (
    CoilSetDofExtractionSpec,
    SurfaceRZFourierSpec,
    SurfaceXYZTensorFourierSpec,
    make_surface_rzfourier_spec,
    make_surface_xyz_tensor_fourier_spec,
)
from simsopt_jax.core.surface_dofs import surface_gamma_tangents_from_dofs
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_dofs_from_spec

from .boozer_material import Flat675BoozerMaterial
from .formulation import Flat675Candidate, Flat675ContractError
from .layout import CERTIFIED_FLAT_LAYOUT, FlatSingleStageLayout
from .objective import Flat675Material
from .policy import (
    Flat675BoozerLabelType,
    Flat675BoozerSystemPolicy,
    Flat675HardwareSoftPenaltyPolicy,
    Flat675ObjectivePolicy,
)

# The certified surface layout, read off the distinguished record rather than
# re-declared.  These are the constructor's DEFAULTS, not its limits, and the
# constructor builds its layout from whatever these resolve to — so a literal
# here would be a second source for the triple that already defines 661, and
# drifting it would quietly hand the default path a non-certified problem
# while the record still said 675.  nfp is free.
CERTIFIED_MPOL: Final[int] = CERTIFIED_FLAT_LAYOUT.surface_mpol
CERTIFIED_NTOR: Final[int] = CERTIFIED_FLAT_LAYOUT.surface_ntor
CERTIFIED_STELLSYM: Final[bool] = CERTIFIED_FLAT_LAYOUT.surface_stellsym
# The campaign's quadrature: a half field period in phi, a full period in
# theta.  Grid density is not layout — it changes how finely the surface is
# sampled, not how many coordinates describe it — so these stay literals and
# the record has nothing to say about them.
CERTIFIED_NPHI: Final[int] = 255
CERTIFIED_NTHETA: Final[int] = 64
CERTIFIED_SURFACE_RANGE: Final[str] = "half period"

# The phi range whose grid mean equals the whole-torus mean at each symmetry.
# Every surface integral in this formulation — the Boozer residual, the label,
# the non-QS ratio — is a mean over the supplied quadrature, so the range is
# not a resolution knob: it is the assumption that the sampled patch stands
# for the torus.  A stellarator-symmetric boundary is represented by a half
# field period; an asymmetric one is not, and needs a full field period.  Both
# tile the torus exactly, so both means are the torus mean.
STELLSYM_SURFACE_RANGE: Final[str] = "half period"
ASYMMETRIC_SURFACE_RANGE: Final[str] = "field period"


def surface_quadrature_range(*, stellsym: bool) -> str:
    """The phi range a boundary of this symmetry must be sampled over."""
    return STELLSYM_SURFACE_RANGE if stellsym else ASYMMETRIC_SURFACE_RANGE


# The synthesized vessel sits this many hinge thresholds outside the boundary's
# own extent.  Any factor above one leaves the hinge exactly inactive; the
# margin keeps it inactive after a few optimizer steps rather than only at the
# start.
DEFAULT_VESSEL_CLEARANCE_FACTOR: Final[float] = 3.0

# The refusal survives generality: an asymmetric boundary aimed at a
# stellarator-symmetric target is a projection onto a proper subspace, and
# returning the symmetrized shape would silently change the plasma.  What
# changed is the remedy the message names — the formulation now carries
# asymmetry, so the caller asks for it rather than waiting for a charter.
_ASYMMETRIC_BOUNDARY_REFUSAL: Final[str] = (
    "this boundary is not stellarator-symmetric, and a stellarator-symmetric "
    "layout was requested (stellsym=True). Its rs/zc content has no "
    "representation in that layout at any resolution, so fitting anyway would "
    "return a different plasma shape under your own variable name. Pass "
    "stellsym=False to build the problem on the asymmetric layout instead; if "
    "the boundary is in fact symmetric and only its flag says otherwise, "
    "rebuild it with stellsym=True."
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
    mpol: int = CERTIFIED_MPOL,
    ntor: int = CERTIFIED_NTOR,
    stellsym: bool = CERTIFIED_STELLSYM,
    nphi: int = CERTIFIED_NPHI,
    ntheta: int = CERTIFIED_NTHETA,
) -> SurfaceXYZTensorFourierSpec:
    """Fit a boundary onto the requested tensor-Fourier surface layout.

    The target is chosen by ``mpol``, ``ntor`` and ``stellsym``, which default
    to the certified triple.  It is never inferred from ``boundary``: the
    caller says what layout the problem is posed on, and this function reports
    whether the boundary fits it.  The fit itself is simsopt's own
    ``least_squares_fit`` against the boundary's surface points sampled on the
    target's quadrature, so the result is the boundary this repo's surface
    machinery would produce, not a reimplemented projection.

    The phi range follows the requested symmetry
    (:func:`surface_quadrature_range`), because the formulation's integrals
    are means over the supplied grid and only a range that tiles the torus
    makes that mean the torus mean.

    Two loss modes, unchanged in kind by generality:

    * **Symmetry is required of the boundary, never imposed on it.**  A
      boundary carrying ``rs``/``zc`` content has no representation in a
      stellarator-symmetric target at any resolution, so it is refused rather
      than symmetrized.  The refusal names ``stellsym=False``, which builds
      the problem on the asymmetric layout and keeps that content.
    * **Resolution above the target is truncated**, and that is the documented
      meaning of "fit".  Unlike the symmetry case this is an approximation
      *within* the target — it converges to the boundary as ``mpol``/``ntor``
      grow, and the target is the caller's own argument.  A boundary with
      content above those modes comes back smoothed; measure the deviation if
      it matters to the study.
    """
    if not isinstance(boundary, Surface):
        raise Flat675ContractError(
            f"boundary must be a simsopt Surface; got {type(boundary).__name__}."
        )
    source = boundary.to_RZFourier()
    if stellsym and not source.stellsym:
        raise Flat675ContractError(_ASYMMETRIC_BOUNDARY_REFUSAL)
    nfp = int(boundary.nfp)
    quadpoints = Surface.get_quadpoints(
        nfp=nfp,
        range=surface_quadrature_range(stellsym=bool(stellsym)),
        nphi=int(nphi),
        ntheta=int(ntheta),
    )
    fitted = SurfaceXYZTensorFourier(
        mpol=int(mpol),
        ntor=int(ntor),
        nfp=nfp,
        stellsym=bool(stellsym),
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
    dofs = np.asarray(fitted.get_dofs(), dtype=np.float64)
    return make_surface_xyz_tensor_fourier_spec(
        dofs=dofs,
        quadpoints_phi=np.asarray(fitted.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(fitted.quadpoints_theta, dtype=np.float64),
        nfp=nfp,
        stellsym=bool(stellsym),
        mpol=int(mpol),
        ntor=int(ntor),
    )


# --------------------------------------------------------------------------
# Coils: any contiguous owner map
# --------------------------------------------------------------------------


def coil_owner_dof_count(extraction: CoilSetDofExtractionSpec) -> int:
    """Return the coil block's width; refuse a map that is not contiguous.

    The formulation's only requirement on a coil set is that its owner DOFs
    are exactly ``0 .. N-1`` with nothing missing and nothing claimed twice
    over — that is what makes a contiguous coil block meaningful.  Owner
    count, per-segment widths and curve family are all free: a
    curve-on-winding-surface family sharing one current, a set of plain
    ``CurveXYZFourier`` curves with a ``Current`` each, or any mixture the
    field machinery can evaluate.

    A gap or an out-of-range index is refused because it would silently
    change which coordinate drives which coil, which is a different problem
    wearing the same shape.
    """
    if not isinstance(extraction, CoilSetDofExtractionSpec):
        raise Flat675ContractError(
            "coil extraction must be a CoilSetDofExtractionSpec; got "
            f"{type(extraction).__name__}."
        )
    owners: set[int] = set()
    for index, coil in enumerate(extraction.coils):
        for mapping in (coil.curve_map, coil.current_map, coil.surface_map):
            if mapping is None:
                continue
            for segment in mapping.owner_segments:
                owner_start, owner_end, target_start, target_end = segment
                if (
                    owner_start < 0
                    or owner_end < owner_start
                    or target_start < 0
                    or target_end < target_start
                    or owner_end - owner_start != target_end - target_start
                ):
                    raise Flat675ContractError(
                        f"coil {index} carries a malformed owner segment "
                        f"{segment}: owner and target spans must be "
                        "non-negative and equal in length."
                    )
                owners.update(range(owner_start, owner_end))
    if not owners:
        raise Flat675ContractError(
            "the coil set claims no owner DOFs, so it has nothing to "
            "optimize. At least one coil must expose a free curve or current."
        )
    owner_count = max(owners) + 1
    missing = sorted(frozenset(range(owner_count)) - owners)
    if missing:
        raise Flat675ContractError(
            f"the coil owner map must cover 0..{owner_count - 1} contiguously; "
            f"it claims {owner_count} as its highest owner DOF but never "
            f"claims {missing}. A gap means some coordinate of the coil block "
            "drives nothing."
        )
    return owner_count


def default_optimized_coil_index(extraction: CoilSetDofExtractionSpec) -> int:
    """The first coil that owns any DOF — the one the shape penalties address.

    The length and curvature penalties are written against a single coil.
    Fixed coils (empty owner maps) are skipped because a penalty on a coil
    nothing can move is a constant, and a constant in the objective is a
    silent way to make a gate look satisfied.
    """
    for index, coil in enumerate(extraction.coils):
        for mapping in (coil.curve_map, coil.current_map, coil.surface_map):
            if mapping is not None and tuple(mapping.owner_segments):
                return index
    raise Flat675ContractError(
        "the coil set declares no free coil, so no coil can carry the length "
        "and curvature penalties."
    )


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
    layout: FlatSingleStageLayout = CERTIFIED_FLAT_LAYOUT,
) -> Flat675Problem:
    """Bind validated material and policy into an evaluable problem.

    The start candidate is derived from the same specs the material is built
    from, so a problem cannot carry a start that disagrees with its own
    geometry.  ``layout`` says which block widths this problem has and
    defaults to the certified one; the material refuses a surface template
    whose resolution is not the layout's, so the two cannot drift apart.
    ``coil_dofs`` must already be float64; this is the one place every
    caller's coil block passes through, so it is where the port's fp64
    contract is enforced.
    """
    boozer = Flat675BoozerMaterial(
        surface_template=surface_template,
        coil_dof_extraction=coil_dof_extraction,
        mpol=surface_template.mpol,
        ntor=surface_template.ntor,
        nfp=surface_template.nfp,
        nphi=int(nphi),
        ntheta=int(ntheta),
        layout=layout,
    )
    material = Flat675Material(boozer=boozer, vessel_template=vessel_template)
    coil_block = np.asarray(coil_dofs)
    if coil_block.shape != (layout.coil_dof_count,):
        raise Flat675ContractError(
            f"the coil block must carry exactly {layout.coil_dof_count} owner "
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
            layout=layout,
        ),
    )


def build_flat675_problem(
    *,
    boundary: Surface,
    field: CoilDofExtractionProvider,
    vessel: SurfaceRZFourierSpec | None = None,
    mpol: int = CERTIFIED_MPOL,
    ntor: int = CERTIFIED_NTOR,
    stellsym: bool = CERTIFIED_STELLSYM,
    nphi: int = CERTIFIED_NPHI,
    ntheta: int = CERTIFIED_NTHETA,
    objective_policy: Flat675ObjectivePolicy | None = None,
    boozer_policy: Flat675BoozerSystemPolicy | None = None,
    vessel_clearance_factor: float = DEFAULT_VESSEL_CLEARANCE_FACTOR,
) -> Flat675Problem:
    """Build a flat coupled single-stage problem from simsopt geometry.

    The surface layout is REQUESTED, not inferred: ``mpol``, ``ntor`` and
    ``stellsym`` name the target and default to the certified triple
    ``(10, 10, True)``, which is the 661-DOF boundary block the sealed
    receipts speak to.  ``boundary`` is then fitted onto that target — see
    :func:`fit_flat675_boundary` for the two ways a fit can lose information,
    and read them before handing this function a boundary whose shape you have
    not checked at the layout you asked for.

    ``field`` may expose any coil set whose owner DOFs cover ``0 .. N-1``
    contiguously; ``N`` becomes the coil block's width.  ``vessel`` is
    optional: omitting it synthesizes one whose hinge term is exactly inactive
    at the start, so the coil + 3 + surface layout always holds.

    The default policy is the campaign's frozen one, with its shape penalties
    pointed at the first free coil.  The sealed receipts speak to that
    configuration at the certified layout; a caller who changes either is
    running a problem those receipts do not certify.
    """
    surface_template = fit_flat675_boundary(
        boundary,
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nphi=nphi,
        ntheta=ntheta,
    )
    extraction = field.coil_dof_extraction_spec()
    layout = FlatSingleStageLayout(
        coil_dof_count=coil_owner_dof_count(extraction),
        surface_mpol=int(mpol),
        surface_ntor=int(ntor),
        surface_stellsym=bool(stellsym),
    )
    policy = (
        default_flat675_objective_policy(
            optimized_coil_index=default_optimized_coil_index(extraction)
        )
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
        layout=layout,
    )


__all__ = [
    "ASYMMETRIC_SURFACE_RANGE",
    "CERTIFIED_MPOL",
    "CERTIFIED_NPHI",
    "CERTIFIED_NTHETA",
    "CERTIFIED_NTOR",
    "CERTIFIED_STELLSYM",
    "CERTIFIED_SURFACE_RANGE",
    "DEFAULT_FLAT675_BOOZER_POLICY",
    "DEFAULT_VESSEL_CLEARANCE_FACTOR",
    "STELLSYM_SURFACE_RANGE",
    "CoilDofExtractionProvider",
    "Flat675Problem",
    "assemble_flat675_problem",
    "build_flat675_problem",
    "coil_owner_dof_count",
    "default_flat675_objective_policy",
    "default_optimized_coil_index",
    "fit_flat675_boundary",
    "surface_quadrature_range",
    "synthesize_flat675_vessel",
]
