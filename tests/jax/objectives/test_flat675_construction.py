"""Artifact-free coverage for the flat-675 constructor.

The parity gates in ``test_flat675_objective.py`` need the archived campaign
bundle, which exists only on the campaign host.  The constructor is the
surface every other user reaches the formulation through, so it is exercised
here from geometry this file builds: a small stellarator-symmetric boundary,
two fixed TF coils, and one curve-on-winding-surface family under the
certified owner layout.  No bundle, no device-sized work.

The synthetic problem is deliberately coarse (16x16 quadrature, order-2
curves).  Nothing here asserts a physics value — the numbers a physicist
would check live in the archived certificate.  What is asserted is the
contract: the layout the constructor must always produce, the vessel branch's
exact inactivity, and each refusal that survives generality.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.field import Coil, Current, coils_via_symmetries
from simsopt.geo import (
    CurveXYZFourier,
    Surface,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt_jax.core.specs import (
    CoilSetDofExtractionSpec,
    SurfaceRZFourierSpec,
    SurfaceXYZTensorFourierSpec,
    make_coil_dof_extraction_spec,
    make_coil_set_dof_extraction_spec,
    make_curve_cwsfourier_rz_spec,
    make_curve_xyzfourier_spec,
    make_optimizable_dof_map_spec,
    make_surface_rzfourier_spec,
)
from simsopt_jax.core.surface_dofs import surface_gamma_tangents_from_dofs
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo import CurveCWSFourier
from simsopt_jax_adapters.geo.flat675 import (
    ASYMMETRIC_SURFACE_RANGE,
    CERTIFIED_MPOL,
    CERTIFIED_NPHI,
    CERTIFIED_NTHETA,
    CERTIFIED_NTOR,
    CERTIFIED_STELLSYM,
    CERTIFIED_SURFACE_RANGE,
    DEFAULT_FLAT675_BOOZER_POLICY,
    DEFAULT_VESSEL_CLEARANCE_FACTOR,
    FLAT675_COIL_DOF_COUNT,
    FLAT675_OBJECTIVE_TERM_KEYS,
    FLAT675_OUTER_DOF_COUNT,
    FLAT675_SURFACE_DOF_COUNT,
    FLAT675_VESSEL_DOF_COUNT,
    FLAT675_VESSEL_SLICE,
    STELLSYM_SURFACE_RANGE,
    Flat675Bundle,
    Flat675ContractError,
    Flat675Problem,
    assemble_flat675_problem,
    bind_flat675_programs,
    build_flat675_problem,
    coil_owner_dof_count,
    default_flat675_objective_policy,
    default_optimized_coil_index,
    fit_flat675_boundary,
    flat675_objective,
    flat675_weighted_terms,
    surface_block_dof_count,
    surface_quadrature_range,
    synthesize_flat675_vessel,
)

# The synthetic problem's scale.  Small enough to build and differentiate in a
# CI process, large enough that every layout width the port pins is real.
_NFP = 2
_QUADRATURE = 16
_CURVE_QUADPOINTS = 32
_CURVE_ORDER = 2
_BOUNDARY_MAJOR_RADIUS = 1.0
_BOUNDARY_MINOR_RADIUS = 0.18
_WINDING_MINOR_RADIUS = 0.45
_TF_COIL_COUNT = 2

# The certified layout puts the free winding-surface coils after the fixed TF
# coils, so the first free index is the TF count.
_EXPECTED_FREE_COIL_INDEX = _TF_COIL_COUNT

# ``surface_vessel`` in the shared term order.
_SURFACE_VESSEL_TERM_INDEX = FLAT675_OBJECTIVE_TERM_KEYS.index("surface_vessel")

_CURVE_DOF_COUNT = 3 * (2 * _CURVE_ORDER + 1)


def _grid(count: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, count, endpoint=False, dtype=np.float64)


def _boundary() -> SurfaceRZFourier:
    """A small stellarator-symmetric boundary with a non-axisymmetric mode."""
    surface = SurfaceRZFourier(
        nfp=_NFP,
        stellsym=True,
        mpol=2,
        ntor=2,
        quadpoints_phi=_grid(32),
        quadpoints_theta=_grid(32),
    )
    surface.set_rc(0, 0, _BOUNDARY_MAJOR_RADIUS)
    surface.set_rc(1, 0, _BOUNDARY_MINOR_RADIUS)
    surface.set_zs(1, 0, _BOUNDARY_MINOR_RADIUS)
    surface.set_rc(1, 1, 0.01)
    return surface


def _winding_surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier(
        nfp=_NFP,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=_grid(_QUADRATURE),
        quadpoints_theta=_grid(_QUADRATURE),
    )
    surface.set_rc(0, 0, _BOUNDARY_MAJOR_RADIUS)
    surface.set_rc(1, 0, _WINDING_MINOR_RADIUS)
    surface.set_zs(1, 0, _WINDING_MINOR_RADIUS)
    return surface


def _field() -> BiotSavartJAX:
    """The certified owner layout: fixed TF coils plus one free CWS family.

    The base curve's ``(phi, theta)`` path is an ellipse rather than a line
    segment.  A degenerate back-and-forth path has vanishing tangents at its
    turning points, where the curvature penalty is not defined — the certified
    curve is a closed loop for the same reason.
    """
    base = CurveCWSFourier(
        quadpoints=_CURVE_QUADPOINTS, order=_CURVE_ORDER, surf=_winding_surface()
    )
    # modes: phic(0..2), phis(1..2), thetac(0..2), thetas(1..2)
    base.x = np.array(
        [0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float64
    )
    free_coils = coils_via_symmetries([base], [Current(1.0e5)], _NFP, True)

    fixed_coils = []
    for index in range(_TF_COIL_COUNT):
        curve = CurveXYZFourier(16, 1)
        curve.x = np.array(
            [0.0, 0.0, 1.0 + 0.1 * index, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            dtype=np.float64,
        )
        curve.fix_all()
        current = Current(3.0e4)
        current.fix_all()
        fixed_coils.append(Coil(curve, current))

    return BiotSavartJAX(fixed_coils + free_coils)


def _problem(vessel: SurfaceRZFourierSpec | None = None) -> Flat675Problem:
    return build_flat675_problem(
        boundary=_boundary(),
        field=_field(),
        vessel=vessel,
        nphi=_QUADRATURE,
        ntheta=_QUADRATURE,
    )


def _fitted_boundary() -> SurfaceXYZTensorFourierSpec:
    return fit_flat675_boundary(_boundary(), nphi=_QUADRATURE, ntheta=_QUADRATURE)


def _cartesian_points(surface: SurfaceXYZTensorFourierSpec) -> np.ndarray:
    gamma, _toroidal, _poloidal = surface_gamma_tangents_from_dofs(
        surface, surface.dofs
    )
    return np.asarray(gamma, dtype=np.float64).reshape((-1, 3))


def _owner_map(
    owner_segments: tuple[tuple[int, int, int, int], ...],
    *,
    template_dof_count: int,
) -> object:
    return make_optimizable_dof_map_spec(
        template_full_dofs=np.zeros(template_dof_count, dtype=np.float64),
        owner_segments=owner_segments,
        input_mode="slice",
        input_start=0,
        input_end=template_dof_count,
    )


_CERTIFIED_CURVE_SEGMENTS = tuple(
    (owner, owner + 1, owner - 1, owner) for owner in range(1, FLAT675_COIL_DOF_COUNT)
)
_CERTIFIED_CURRENT_SEGMENTS = ((0, 1, 0, 1),)


def _cws_curve_spec() -> object:
    return make_curve_cwsfourier_rz_spec(
        # phic(0..2), phis(1..2), thetac(0..2), thetas(1..2)
        dofs=np.zeros(4 * _CURVE_ORDER + 2, dtype=np.float64),
        quadpoints=_grid(8),
        surface=make_surface_rzfourier_spec(
            rc=np.array([[1.0]], dtype=np.float64),
            zs=np.array([[0.0]], dtype=np.float64),
            rs=np.array([[0.0]], dtype=np.float64),
            zc=np.array([[0.0]], dtype=np.float64),
            quadpoints_phi=_grid(4),
            quadpoints_theta=_grid(4),
            nfp=_NFP,
            stellsym=True,
        ),
        order=_CURVE_ORDER,
    )


def _xyz_curve_spec() -> object:
    return make_curve_xyzfourier_spec(
        dofs=np.zeros(_CURVE_DOF_COUNT, dtype=np.float64),
        quadpoints=_grid(4),
        order=_CURVE_ORDER,
    )


def _coil_set(
    *,
    curve: object,
    curve_segments: tuple[tuple[int, int, int, int], ...],
    current_segments: tuple[tuple[int, int, int, int], ...],
) -> CoilSetDofExtractionSpec:
    """One synthetic coil, wired to whatever owner layout a test needs."""
    return make_coil_set_dof_extraction_spec(
        coils=(
            make_coil_dof_extraction_spec(
                curve=curve,  # type: ignore[arg-type]
                curve_map=_owner_map(  # type: ignore[arg-type]
                    curve_segments, template_dof_count=_CURVE_DOF_COUNT
                ),
                current_map=_owner_map(  # type: ignore[arg-type]
                    current_segments, template_dof_count=1
                ),
            ),
        )
    )


# --- the layout the constructor must always produce ------------------------


def test_constructor_builds_the_certified_675_layout() -> None:
    """11 coil + 3 vessel + 661 surface, from ordinary simsopt objects."""
    problem = _problem()

    candidate = problem.start_candidate
    assert candidate.outer_vector().shape == (FLAT675_OUTER_DOF_COUNT,)
    assert len(candidate.coil_coordinates) == FLAT675_COIL_DOF_COUNT
    assert len(candidate.vessel_coordinates) == FLAT675_VESSEL_DOF_COUNT
    assert len(candidate.surface_coordinates) == FLAT675_SURFACE_DOF_COUNT
    assert problem.material.boozer.surface_template.dofs.shape == (
        FLAT675_SURFACE_DOF_COUNT,
    )


def test_constructor_points_the_shape_penalties_at_the_free_coil() -> None:
    """The optimized coil is the first free CWS coil, not a fixed TF coil."""
    problem = _problem()

    assert problem.objective_policy.optimized_coil_index == _EXPECTED_FREE_COIL_INDEX


def test_constructor_output_is_differentiable_and_finite() -> None:
    """The whole point of the constructor: an evaluable, differentiable problem."""
    problem = _problem()
    start = jnp.asarray(problem.start_candidate.outer_vector())

    value, gradient = jax.value_and_grad(
        lambda x: flat675_objective(
            x,
            material=problem.material,
            objective_policy=problem.objective_policy,
            boozer_policy=problem.boozer_policy,
        )
    )(start)

    assert np.isfinite(float(value))
    gradient_array = np.asarray(gradient, dtype=np.float64)
    assert gradient_array.shape == (FLAT675_OUTER_DOF_COUNT,)
    assert np.all(np.isfinite(gradient_array))
    assert np.max(np.abs(gradient_array)) > 0.0


def test_constructor_output_binds_into_the_fused_programs() -> None:
    """A constructed problem is what the fused device lane consumes."""
    problem = _problem()
    start = jnp.asarray(problem.start_candidate.outer_vector())

    programs = bind_flat675_programs(
        material=problem.material,
        objective_policy=problem.objective_policy,
        boozer_policy=problem.boozer_policy,
    )

    bound = float(programs.objective_fn(start))
    direct = float(
        flat675_objective(
            start,
            material=problem.material,
            objective_policy=problem.objective_policy,
            boozer_policy=problem.boozer_policy,
        )
    )
    assert bound == direct


# --- the vessel-optional branch --------------------------------------------


def test_omitted_vessel_leaves_the_hinge_penalty_exactly_inactive() -> None:
    """ "No vessel" means an inactive vessel term, never a 672-DOF vector.

    Exact zeros, not a tolerance: the synthesized vessel is placed outside the
    hinge, so the penalty is identically zero and its gradient onto the three
    vessel DOFs is structurally zero rather than merely small.
    """
    problem = _problem()
    start = jnp.asarray(problem.start_candidate.outer_vector())

    terms = np.asarray(
        flat675_weighted_terms(
            start,
            material=problem.material,
            objective_policy=problem.objective_policy,
            boozer_policy=problem.boozer_policy,
        ),
        dtype=np.float64,
    )
    gradient = np.asarray(
        jax.grad(
            lambda x: flat675_objective(
                x,
                material=problem.material,
                objective_policy=problem.objective_policy,
                boozer_policy=problem.boozer_policy,
            )
        )(start),
        dtype=np.float64,
    )

    assert float(terms[_SURFACE_VESSEL_TERM_INDEX]) == 0.0
    assert np.all(gradient[FLAT675_VESSEL_SLICE] == 0.0)


def test_synthesized_vessel_clears_the_boundary_by_the_stated_margin() -> None:
    """The clearance is the hinge threshold times the documented factor."""
    surface = _fitted_boundary()
    threshold = 0.04

    vessel = synthesize_flat675_vessel(surface, hinge_threshold_m=threshold)

    points = _cartesian_points(surface)
    radius = np.hypot(points[:, 0], points[:, 1])
    major_radius = 0.5 * (float(radius.max()) + float(radius.min()))
    minor_extent = float(np.max(np.hypot(radius - major_radius, points[:, 2])))

    vessel_major = float(np.asarray(vessel.rc)[0, 0])
    vessel_minor = float(np.asarray(vessel.rc)[1, 0])
    assert vessel_major == pytest.approx(major_radius)
    assert vessel_minor - minor_extent == pytest.approx(
        threshold * DEFAULT_VESSEL_CLEARANCE_FACTOR
    )
    # A circular cross-section: the poloidal amplitudes match.
    assert float(np.asarray(vessel.zs)[1, 0]) == vessel_minor
    assert vessel.mpol == 1
    assert vessel.ntor == 0


def test_vessel_clearance_factor_moves_the_synthesized_vessel() -> None:
    """The margin is a documented knob, not a hidden constant."""
    surface = _fitted_boundary()

    close = synthesize_flat675_vessel(
        surface, hinge_threshold_m=0.04, clearance_factor=1.5
    )
    far = synthesize_flat675_vessel(
        surface, hinge_threshold_m=0.04, clearance_factor=6.0
    )

    assert float(np.asarray(far.rc)[1, 0]) > float(np.asarray(close.rc)[1, 0])


def test_supplied_vessel_is_used_verbatim() -> None:
    """A caller who brings a vessel gets that vessel, not a synthesized one."""
    supplied = make_surface_rzfourier_spec(
        rc=np.array([[1.0], [0.6]], dtype=np.float64),
        zs=np.array([[0.0], [0.55]], dtype=np.float64),
        rs=np.zeros((2, 1), dtype=np.float64),
        zc=np.zeros((2, 1), dtype=np.float64),
        quadpoints_phi=_grid(_QUADRATURE),
        quadpoints_theta=_grid(_QUADRATURE),
        nfp=_NFP,
        stellsym=True,
    )

    problem = _problem(vessel=supplied)

    assert problem.material.vessel_template is supplied
    assert problem.start_candidate.vessel_coordinates == (1.0, 0.6, 0.55)


# --- the boundary fit -------------------------------------------------------


def test_fit_produces_the_certified_661_dof_surface_layout() -> None:
    surface = fit_flat675_boundary(_boundary(), nphi=_QUADRATURE, ntheta=_QUADRATURE)

    assert surface.dofs.shape == (FLAT675_SURFACE_DOF_COUNT,)
    assert surface.mpol == CERTIFIED_MPOL
    assert surface.ntor == CERTIFIED_NTOR
    assert bool(surface.stellsym) == CERTIFIED_STELLSYM
    assert surface.nfp == _NFP


def test_fit_reproduces_the_boundary_it_was_given() -> None:
    """The fit is a change of representation, not a change of shape."""
    boundary = _boundary()
    quadpoints = Surface.get_quadpoints(
        nfp=_NFP,
        range=CERTIFIED_SURFACE_RANGE,
        nphi=_QUADRATURE,
        ntheta=_QUADRATURE,
    )
    reference = SurfaceRZFourier(
        nfp=_NFP,
        stellsym=True,
        mpol=boundary.mpol,
        ntor=boundary.ntor,
        quadpoints_phi=quadpoints[0],
        quadpoints_theta=quadpoints[1],
    )
    reference.x = boundary.x

    fitted = fit_flat675_boundary(boundary, nphi=_QUADRATURE, ntheta=_QUADRATURE)

    deviation = np.max(
        np.abs(
            _cartesian_points(fitted) - reference.gamma().reshape((-1, 3)),
        )
    )
    assert deviation < 1.0e-12


def test_fit_refuses_something_that_is_not_a_surface() -> None:
    with pytest.raises(Flat675ContractError, match="must be a simsopt Surface"):
        fit_flat675_boundary(object())  # type: ignore[arg-type]


def _asymmetric_boundary(*, rs: float, zc: float) -> SurfaceRZFourier:
    """The synthetic boundary with stellarator symmetry deliberately broken."""
    surface = SurfaceRZFourier(
        nfp=_NFP,
        stellsym=False,
        mpol=2,
        ntor=2,
        quadpoints_phi=_grid(32),
        quadpoints_theta=_grid(32),
    )
    surface.set_rc(0, 0, _BOUNDARY_MAJOR_RADIUS)
    surface.set_rc(1, 0, _BOUNDARY_MINOR_RADIUS)
    surface.set_zs(1, 0, _BOUNDARY_MINOR_RADIUS)
    surface.set_rs(1, 0, rs)
    surface.set_zc(1, 0, zc)
    return surface


def test_fit_refuses_a_boundary_that_is_not_stellarator_symmetric() -> None:
    """Refused, not symmetrized — the coercion this repo bans.

    The rs/zc content has no representation in a stellarator-symmetric target
    at any resolution, so fitting anyway would hand back a different plasma
    shape under the caller's own variable name.  The refusal survives rung-2
    generality; what changed is the remedy it names.
    """
    with pytest.raises(Flat675ContractError) as excinfo:
        fit_flat675_boundary(
            _asymmetric_boundary(rs=0.02, zc=0.015),
            nphi=_QUADRATURE,
            ntheta=_QUADRATURE,
        )

    message = str(excinfo.value)
    assert "not stellarator-symmetric" in message
    assert "stellsym=False" in message
    # The message must not defer to a charter that no longer owns the change.
    assert "rung" not in message.lower()


def test_fit_refuses_an_asymmetric_flag_even_with_no_asymmetric_content() -> None:
    """The line is the symmetry declaration, not a smallness threshold.

    A boundary whose rs/zc happen to be zero is representable, but deciding
    that from the coefficients needs a "close enough to zero" tolerance, and a
    tolerance is exactly the fudge that lets a real asymmetry through. The
    flag is exact, and the refusal says how to clear it.
    """
    with pytest.raises(Flat675ContractError, match="rebuild it with stellsym=True"):
        fit_flat675_boundary(
            _asymmetric_boundary(rs=0.0, zc=0.0),
            stellsym=True,
            nphi=_QUADRATURE,
            ntheta=_QUADRATURE,
        )


def test_fit_refuses_asymmetry_before_the_constructor_can_use_it() -> None:
    """The refusal holds at the front door, not only in the fit helper."""
    with pytest.raises(Flat675ContractError, match="stellsym=False"):
        build_flat675_problem(
            boundary=_asymmetric_boundary(rs=0.02, zc=0.015),
            field=_field(),
            stellsym=True,
            nphi=_QUADRATURE,
            ntheta=_QUADRATURE,
        )


def test_fit_truncates_resolution_above_the_certified_layout() -> None:
    """Accepted and smoothed — stated behavior, adjudicated apart from symmetry.

    Content above ``mpol = ntor = 10`` is an approximation *within* the target
    layout: it converges to the boundary as the layout's resolution grows, and
    the layout is documented.  That is what "fit" means, unlike the symmetry
    case where no resolution ever recovers the input.  This test pins the
    distinction so that collapsing the two is a deliberate act.
    """
    high_mpol, high_ntor, amplitude = 15, 12, 1.0e-2
    boundary = SurfaceRZFourier(
        nfp=_NFP,
        stellsym=True,
        mpol=high_mpol,
        ntor=high_ntor,
        quadpoints_phi=_grid(64),
        quadpoints_theta=_grid(64),
    )
    boundary.set_rc(0, 0, _BOUNDARY_MAJOR_RADIUS)
    boundary.set_rc(1, 0, _BOUNDARY_MINOR_RADIUS)
    boundary.set_zs(1, 0, _BOUNDARY_MINOR_RADIUS)
    boundary.set_rc(high_mpol, high_ntor, amplitude)
    boundary.set_zs(high_mpol, high_ntor, amplitude)

    fitted = fit_flat675_boundary(boundary, nphi=32, ntheta=32)

    quadpoints = Surface.get_quadpoints(
        nfp=_NFP, range=CERTIFIED_SURFACE_RANGE, nphi=32, ntheta=32
    )
    reference = SurfaceRZFourier(
        nfp=_NFP,
        stellsym=True,
        mpol=high_mpol,
        ntor=high_ntor,
        quadpoints_phi=quadpoints[0],
        quadpoints_theta=quadpoints[1],
    )
    reference.x = boundary.x
    target = reference.gamma().reshape((-1, 3))
    deviation = float(
        np.linalg.norm(_cartesian_points(fitted) - target) / np.linalg.norm(target)
    )

    # Smoothed by roughly the amplitude that sits above the layout, not by an
    # unbounded amount, and not to machine precision either.
    assert 1.0e-4 < deviation < 1.0e-1


def test_certified_quadrature_defaults_are_the_campaign_resolution() -> None:
    assert (CERTIFIED_NPHI, CERTIFIED_NTHETA) == (255, 64)
    assert CERTIFIED_SURFACE_RANGE == "half period"


# --- the surface layout is requested, not inferred --------------------------


def test_quadrature_range_follows_the_requested_symmetry() -> None:
    """The range is the assumption that the sampled patch stands for the torus.

    Every surface integral in this formulation is a mean over the supplied
    grid, so only a phi range that tiles the torus makes that mean the torus
    mean.  A stellarator-symmetric boundary is tiled by a half field period;
    an asymmetric one is not, and needs the full period.
    """
    assert surface_quadrature_range(stellsym=True) == STELLSYM_SURFACE_RANGE
    assert surface_quadrature_range(stellsym=False) == ASYMMETRIC_SURFACE_RANGE
    assert STELLSYM_SURFACE_RANGE == "half period"
    assert ASYMMETRIC_SURFACE_RANGE == "field period"
    # The certified configuration is the symmetric one, so its range is that.
    assert CERTIFIED_SURFACE_RANGE == STELLSYM_SURFACE_RANGE


@pytest.mark.parametrize(("mpol", "ntor"), [(4, 4), (6, 2), (10, 10)])
def test_fit_targets_the_requested_resolution(mpol: int, ntor: int) -> None:
    """The target is the caller's argument, never read off the boundary."""
    fitted = fit_flat675_boundary(
        _boundary(), mpol=mpol, ntor=ntor, nphi=_QUADRATURE, ntheta=_QUADRATURE
    )

    assert (fitted.mpol, fitted.ntor) == (mpol, ntor)
    assert fitted.dofs.shape == (
        surface_block_dof_count(mpol=mpol, ntor=ntor, stellsym=True),
    )


def test_fit_defaults_to_the_certified_triple() -> None:
    """A caller who names no layout gets the configuration the receipts speak."""
    fitted = fit_flat675_boundary(_boundary(), nphi=_QUADRATURE, ntheta=_QUADRATURE)

    assert (fitted.mpol, fitted.ntor, bool(fitted.stellsym)) == (
        CERTIFIED_MPOL,
        CERTIFIED_NTOR,
        CERTIFIED_STELLSYM,
    )
    assert fitted.dofs.shape == (FLAT675_SURFACE_DOF_COUNT,)


def test_fit_builds_the_asymmetric_layout_when_it_is_asked_to() -> None:
    """stellsym=False is now a layout the formulation carries, not a refusal."""
    fitted = fit_flat675_boundary(
        _asymmetric_boundary(rs=0.02, zc=0.015),
        mpol=4,
        ntor=4,
        stellsym=False,
        nphi=_QUADRATURE,
        ntheta=_QUADRATURE,
    )

    assert bool(fitted.stellsym) is False
    assert fitted.dofs.shape == (
        surface_block_dof_count(mpol=4, ntor=4, stellsym=False),
    )


# --- the coil owner map: any contiguous cover -------------------------------


def test_certified_coil_set_reports_its_own_owner_width() -> None:
    """The control, and the source of the index the default policy uses."""
    extraction = _field().coil_dof_extraction_spec()

    assert coil_owner_dof_count(extraction) == FLAT675_COIL_DOF_COUNT
    assert default_optimized_coil_index(extraction) == _EXPECTED_FREE_COIL_INDEX


def test_generic_coil_sets_are_accepted() -> None:
    """The case rung 1 refused: plain curves, one Current each.

    ``create_equally_spaced_curves`` + ``Current`` produces per-coil owner
    segments of arbitrary width — the layout the old validator could not
    express.  It is now simply a coil block of whatever width it covers.
    """
    curves = create_equally_spaced_curves(
        3, _NFP, stellsym=True, R0=1.0, R1=0.5, order=3
    )
    field = BiotSavartJAX(
        coils_via_symmetries(curves, [Current(1.0e5) for _ in curves], _NFP, True)
    )
    extraction = field.coil_dof_extraction_spec()

    owner_count = coil_owner_dof_count(extraction)
    assert owner_count == len(np.asarray(field.x))
    assert default_optimized_coil_index(extraction) == 0


def test_coil_owner_map_refuses_a_gap() -> None:
    """A gap means some coil coordinate drives nothing."""
    extraction = _coil_set(
        curve=_cws_curve_spec(),
        curve_segments=((0, 2, 0, 2), (3, 5, 2, 4)),
        current_segments=(),
    )

    with pytest.raises(Flat675ContractError, match=r"never claims \[2\]"):
        coil_owner_dof_count(extraction)


def test_coil_owner_map_refuses_a_malformed_segment() -> None:
    """Owner and target spans must be non-negative and equal in length."""
    extraction = _coil_set(
        curve=_cws_curve_spec(),
        curve_segments=((0, 4, 0, 2),),
        current_segments=(),
    )

    with pytest.raises(Flat675ContractError, match="malformed owner segment"):
        coil_owner_dof_count(extraction)


def test_coil_owner_map_refuses_a_set_that_owns_nothing() -> None:
    """All-fixed coils are a field, not an optimizable coil set."""
    extraction = _coil_set(
        curve=_xyz_curve_spec(), curve_segments=(), current_segments=()
    )

    with pytest.raises(Flat675ContractError, match="claims no owner DOFs"):
        coil_owner_dof_count(extraction)
    with pytest.raises(Flat675ContractError, match="declares no free coil"):
        default_optimized_coil_index(extraction)


def test_coil_owner_map_refuses_something_that_is_not_an_extraction_spec() -> None:
    with pytest.raises(
        Flat675ContractError, match="must be a CoilSetDofExtractionSpec"
    ):
        coil_owner_dof_count(object())  # type: ignore[arg-type]


# --- assembly and policy defaults ------------------------------------------


def _assemble_with(coil_dofs: np.ndarray) -> Flat675Problem:
    """The shared assembly on valid material, varying only the coil block."""
    surface = _fitted_boundary()
    return assemble_flat675_problem(
        surface_template=surface,
        coil_dof_extraction=_field().coil_dof_extraction_spec(),
        coil_dofs=coil_dofs,
        vessel_template=synthesize_flat675_vessel(surface, hinge_threshold_m=0.04),
        objective_policy=default_flat675_objective_policy(
            optimized_coil_index=_EXPECTED_FREE_COIL_INDEX
        ),
        boozer_policy=DEFAULT_FLAT675_BOOZER_POLICY,
        nphi=_QUADRATURE,
        ntheta=_QUADRATURE,
    )


def test_assembly_refuses_a_coil_block_of_the_wrong_width() -> None:
    """The start candidate must match the layout the material was built for."""
    with pytest.raises(
        Flat675ContractError,
        match=f"exactly {FLAT675_COIL_DOF_COUNT} owner",
    ):
        _assemble_with(np.zeros(FLAT675_COIL_DOF_COUNT - 1, dtype=np.float64))


def test_assembly_refuses_a_single_precision_coil_block() -> None:
    """Refused, not widened: this port is fp64 only.

    Silently promoting a float32 block would produce an fp64 answer that hides
    a single-precision defect upstream of the constructor.
    """
    with pytest.raises(Flat675ContractError, match="must already be float64"):
        _assemble_with(np.zeros(FLAT675_COIL_DOF_COUNT, dtype=np.float32))


def test_assembly_accepts_the_float64_control() -> None:
    """Without this, a guard that refused every block would look identical."""
    problem = _assemble_with(np.zeros(FLAT675_COIL_DOF_COUNT, dtype=np.float64))

    assert problem.start_candidate.coil_coordinates == (0.0,) * FLAT675_COIL_DOF_COUNT


def test_default_policy_is_the_frozen_campaign_configuration() -> None:
    """These are the weights the sealed receipts were measured under.

    A silent change here would make every citation of those receipts wrong
    while every other test still passed, so the values are pinned literally.
    """
    policy = default_flat675_objective_policy(optimized_coil_index=3)

    assert policy.iota_target == 0.15
    assert policy.residual_weight == 1000.0
    assert policy.iota_weight == 100.0
    assert policy.length_weight == 1.0
    assert policy.length_target_m == 1.7
    assert policy.curve_curve_weight == 1000.0
    assert policy.curve_curve_threshold_m == 0.05
    assert policy.curve_surface_weight == 1.0
    assert policy.curve_surface_threshold_m == 0.02
    assert policy.surface_vessel_weight == 1000.0
    assert policy.surface_vessel_threshold_m == 0.04
    assert policy.curvature_weight == 1.0
    assert policy.curvature_threshold_inverse_m == 40.0
    assert policy.non_qs_weight == 1.0
    assert policy.non_qs_grid_size == 40
    assert policy.boozer_constraint_weight == 1.0
    assert policy.boozer_target_label == 0.1
    assert policy.optimized_coil_index == 3

    penalty = policy.hardware_soft_penalty_policy
    assert penalty.common_scale == 10.0
    assert penalty.curve_curve_base_weight == 100.0
    assert penalty.curvature_base_weight == 0.1
    assert penalty.penalty_exponent == 2.0


def test_default_boozer_policy_weights_by_inverse_field_magnitude() -> None:
    assert DEFAULT_FLAT675_BOOZER_POLICY.weight_by_inverse_field_magnitude is True


def test_constructor_uses_the_frozen_defaults_when_none_are_given() -> None:
    problem = _problem()

    assert problem.boozer_policy == DEFAULT_FLAT675_BOOZER_POLICY
    assert problem.objective_policy == default_flat675_objective_policy(
        optimized_coil_index=_EXPECTED_FREE_COIL_INDEX
    )


# --- SSOT: one construction path --------------------------------------------


def test_the_bundle_record_is_the_constructed_record() -> None:
    """The frozen-bundle loader kept its spelling, not a separate type.

    If the bundle path ever grew a private construction route again, it would
    need its own record type to carry the difference.
    """
    assert Flat675Bundle is Flat675Problem


def test_the_constructor_is_a_caller_of_the_shared_assembly() -> None:
    """Assembling by hand from the same inputs reproduces the constructor bitwise."""
    surface = _fitted_boundary()
    field = _field()
    extraction = field.coil_dof_extraction_spec()
    policy = default_flat675_objective_policy(
        optimized_coil_index=default_optimized_coil_index(extraction)
    )
    assembled = assemble_flat675_problem(
        surface_template=surface,
        coil_dof_extraction=extraction,
        coil_dofs=np.asarray(field.x, dtype=np.float64),
        vessel_template=synthesize_flat675_vessel(
            surface, hinge_threshold_m=policy.surface_vessel_threshold_m
        ),
        objective_policy=policy,
        boozer_policy=DEFAULT_FLAT675_BOOZER_POLICY,
        nphi=_QUADRATURE,
        ntheta=_QUADRATURE,
    )
    constructed = _problem()

    assert (
        np.asarray(assembled.start_candidate.outer_vector()).tobytes()
        == np.asarray(constructed.start_candidate.outer_vector()).tobytes()
    )
    start = jnp.asarray(constructed.start_candidate.outer_vector())
    values = [
        float(
            flat675_objective(
                start,
                material=problem.material,
                objective_policy=problem.objective_policy,
                boozer_policy=problem.boozer_policy,
            )
        )
        for problem in (assembled, constructed)
    ]
    assert values[0] == values[1]
