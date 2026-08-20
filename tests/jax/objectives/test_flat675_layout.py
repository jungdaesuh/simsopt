"""The per-problem layout record, and the first solve that leaves 675.

Two things are held here.  First, the record's arithmetic: block widths must
be *derived* from the surface resolution and the coil owner count, and the
derivation must agree with simsopt's own DOF count at every resolution rather
than with a formula this repository invented.  Second, and the point of the
whole exercise, that the generalized core actually runs off the certified
layout: a full fused solve on an ``mpol = ntor = 4`` problem — a 121-DOF
boundary block, 135 outer coordinates instead of 675.

The solve is artifact-free and runs on CPU.  Its gates are the ones a
correctness run can honestly carry: a finite endpoint, an objective that
improved, and an endpoint gradient smaller than the start's.  No timing claim
is made and none could be: this is a correctness test, not a measurement.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from simsopt.field import Coil, Current, coils_via_symmetries
from simsopt.geo import (
    CurveXYZFourier,
    Surface,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
)
from simsopt_jax.core.specs import (
    make_surface_rzfourier_spec,
    make_surface_xyz_tensor_fourier_spec,
)
from simsopt_jax.examples.single_stage_flat675 import (
    prepare_single_stage_flat675,
    solve_single_stage_flat675,
)
from simsopt_jax.runtime.host_boundary import host_transfer_audit
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo import CurveCWSFourier
from simsopt_jax_adapters.geo.flat675 import (
    CERTIFIED_FLAT_LAYOUT,
    DEFAULT_FLAT675_BOOZER_POLICY,
    FLAT675_COIL_DOF_COUNT,
    FLAT675_COIL_SLICE,
    FLAT675_OUTER_DOF_COUNT,
    FLAT675_SURFACE_DOF_COUNT,
    FLAT675_SURFACE_SLICE,
    FLAT675_VESSEL_DOF_COUNT,
    FLAT675_VESSEL_SLICE,
    Flat675BoozerMaterial,
    Flat675Candidate,
    Flat675ContractError,
    Flat675Material,
    FlatLayoutError,
    FlatSingleStageLayout,
    bind_flat675_programs,
    default_flat675_objective_policy,
    surface_block_dof_count,
)

# The small layout: a quarter of the certified poloidal/toroidal resolution.
SMALL_MPOL = SMALL_NTOR = 4
SMALL_SURFACE_DOF_COUNT = 121
SMALL_GRID = 8
SMALL_NFP = 2
SMALL_STEPS = 5
OPTIMIZED_COIL_INDEX = 2

# The winding-surface coil shape the certified campaign uses, reused here so
# the small problem starts from a physically-shaped coil rather than noise.
WINDING_COIL_DOFS = (
    0.119251,
    0.012469,
    -0.017700,
    -0.068677,
    -0.014250,
    0.430936,
    0.082708,
    -0.015905,
    -0.113852,
    -0.025142,
)


# --- the record's arithmetic ------------------------------------------------


@pytest.mark.parametrize(
    ("mpol", "ntor"),
    [(10, 10), (4, 4), (1, 0), (2, 3), (6, 2)],
)
@pytest.mark.parametrize("stellsym", [True, False])
def test_surface_width_matches_simsopts_own_dof_count(
    mpol: int, ntor: int, stellsym: bool
) -> None:
    """The derivation is simsopt's count, not a formula invented here.

    This is the equality the record's docstring claims, checked against the
    class the boundary is actually built from — including the asymmetric mode,
    whose count the layout must get right before rung 2 can offer it.
    """
    quadpoints = np.linspace(0.0, 1.0, 4, endpoint=False)
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        nfp=2,
        stellsym=stellsym,
        quadpoints_phi=quadpoints,
        quadpoints_theta=quadpoints,
    )

    assert surface_block_dof_count(mpol=mpol, ntor=ntor, stellsym=stellsym) == len(
        surface.get_dofs()
    )


def test_certified_layout_reproduces_the_historic_constants() -> None:
    """The distinguished instance IS the 675 the receipts speak.

    If this drifts, every sealed citation silently changes meaning, so the
    numbers are written literally rather than derived from the record.
    """
    assert CERTIFIED_FLAT_LAYOUT.coil_dof_count == 11
    assert CERTIFIED_FLAT_LAYOUT.vessel_dof_count == 3
    assert CERTIFIED_FLAT_LAYOUT.surface_dof_count == 661
    assert CERTIFIED_FLAT_LAYOUT.outer_dof_count == 675
    assert (
        CERTIFIED_FLAT_LAYOUT.surface_mpol,
        CERTIFIED_FLAT_LAYOUT.surface_ntor,
        CERTIFIED_FLAT_LAYOUT.surface_stellsym,
    ) == (10, 10, True)


def test_historic_constants_are_the_certified_record_read_out() -> None:
    """The re-export linchpin: existing consumers see the record's values."""
    assert FLAT675_COIL_DOF_COUNT == CERTIFIED_FLAT_LAYOUT.coil_dof_count
    assert FLAT675_VESSEL_DOF_COUNT == CERTIFIED_FLAT_LAYOUT.vessel_dof_count
    assert FLAT675_SURFACE_DOF_COUNT == CERTIFIED_FLAT_LAYOUT.surface_dof_count
    assert FLAT675_OUTER_DOF_COUNT == CERTIFIED_FLAT_LAYOUT.outer_dof_count
    assert FLAT675_COIL_SLICE == CERTIFIED_FLAT_LAYOUT.coil_slice
    assert FLAT675_VESSEL_SLICE == CERTIFIED_FLAT_LAYOUT.vessel_slice
    assert FLAT675_SURFACE_SLICE == CERTIFIED_FLAT_LAYOUT.surface_slice


def test_the_surface_width_cannot_be_supplied_independently() -> None:
    """No dual source, structurally: the width is not a field to disagree with."""
    assert "surface_dof_count" not in FlatSingleStageLayout.__dataclass_fields__
    # The type checker rejects this call for the same reason the runtime does,
    # which is the property under test; the suppression keeps the runtime proof
    # without pretending the static one does not also hold.
    with pytest.raises(TypeError):
        FlatSingleStageLayout(
            coil_dof_count=11,
            surface_mpol=4,
            surface_ntor=4,
            surface_stellsym=True,
            surface_dof_count=999,  # pyright: ignore[reportCallIssue]
        )


def test_blocks_partition_the_outer_vector_without_gap_or_overlap() -> None:
    """Three slices, contiguous, covering exactly the outer width."""
    layout = FlatSingleStageLayout(
        coil_dof_count=7,
        surface_mpol=SMALL_MPOL,
        surface_ntor=SMALL_NTOR,
        surface_stellsym=True,
    )

    assert layout.coil_slice.start == 0
    assert layout.coil_slice.stop == layout.vessel_slice.start
    assert layout.vessel_slice.stop == layout.surface_slice.start
    assert layout.surface_slice.stop == layout.outer_dof_count
    covered = (
        len(range(*layout.coil_slice.indices(layout.outer_dof_count)))
        + len(range(*layout.vessel_slice.indices(layout.outer_dof_count)))
        + len(range(*layout.surface_slice.indices(layout.outer_dof_count)))
    )
    assert covered == layout.outer_dof_count


@pytest.mark.parametrize(
    ("keywords", "expected"),
    [
        ({"coil_dof_count": 0}, "at least one coil owner"),
        ({"surface_mpol": 0}, "mpol >= 1"),
        ({"surface_ntor": -1}, "ntor >= 0"),
    ],
)
def test_impossible_layouts_are_refused_at_construction(
    keywords: dict[str, int], expected: str
) -> None:
    """A layout that cannot exist fails when built, not at first evaluation."""
    valid = {
        "coil_dof_count": 11,
        "surface_mpol": SMALL_MPOL,
        "surface_ntor": SMALL_NTOR,
        "surface_stellsym": True,
    }

    with pytest.raises(FlatLayoutError, match=expected):
        FlatSingleStageLayout(**{**valid, **keywords})


def test_candidate_blocks_follow_the_layout_it_is_given() -> None:
    """The per-block width table is the record's, not a module constant's."""
    layout = FlatSingleStageLayout(
        coil_dof_count=5,
        surface_mpol=SMALL_MPOL,
        surface_ntor=SMALL_NTOR,
        surface_stellsym=True,
    )

    candidate = Flat675Candidate(
        coil_coordinates=(0.0,) * 5,
        vessel_coordinates=(1.0, 0.5, 0.5),
        surface_coordinates=(0.0,) * SMALL_SURFACE_DOF_COUNT,
        layout=layout,
    )

    assert candidate.outer_vector().shape == (layout.outer_dof_count,)
    with pytest.raises(Flat675ContractError, match="coil_coordinates"):
        Flat675Candidate(
            coil_coordinates=(0.0,) * FLAT675_COIL_DOF_COUNT,
            vessel_coordinates=(1.0, 0.5, 0.5),
            surface_coordinates=(0.0,) * SMALL_SURFACE_DOF_COUNT,
            layout=layout,
        )


def test_candidate_defaults_to_the_certified_layout() -> None:
    """Existing callers construct exactly what they always did."""
    candidate = Flat675Candidate(
        coil_coordinates=(0.0,) * FLAT675_COIL_DOF_COUNT,
        vessel_coordinates=(1.0, 0.5, 0.5),
        surface_coordinates=(0.0,) * FLAT675_SURFACE_DOF_COUNT,
    )

    assert candidate.layout == CERTIFIED_FLAT_LAYOUT
    assert candidate.outer_vector().shape == (FLAT675_OUTER_DOF_COUNT,)


# --- the first solve that leaves 675 ---------------------------------------


def _small_layout_problem() -> tuple[
    Flat675Material, FlatSingleStageLayout, np.ndarray
]:
    """Build an ``mpol = ntor = 4`` problem straight from the core API.

    Deliberately not through ``build_flat675_problem``: generalizing the
    constructor is G2's subject, and this test exists to show the *core* runs
    off-675 once it reads a layout record.
    """
    layout = FlatSingleStageLayout(
        coil_dof_count=FLAT675_COIL_DOF_COUNT,
        surface_mpol=SMALL_MPOL,
        surface_ntor=SMALL_NTOR,
        surface_stellsym=True,
    )
    quadpoints = Surface.get_quadpoints(
        nfp=SMALL_NFP, range="half period", nphi=SMALL_GRID, ntheta=SMALL_GRID
    )
    fitted = SurfaceXYZTensorFourier(
        mpol=SMALL_MPOL,
        ntor=SMALL_NTOR,
        nfp=SMALL_NFP,
        stellsym=True,
        quadpoints_phi=quadpoints[0],
        quadpoints_theta=quadpoints[1],
    )
    source = SurfaceRZFourier(
        nfp=SMALL_NFP,
        stellsym=True,
        mpol=2,
        ntor=2,
        quadpoints_phi=quadpoints[0],
        quadpoints_theta=quadpoints[1],
    )
    source.set_rc(0, 0, 1.0)
    source.set_rc(1, 0, 0.18)
    source.set_zs(1, 0, 0.18)
    source.set_rc(1, 1, 0.01)
    fitted.least_squares_fit(source.gamma())
    surface_dofs = np.asarray(fitted.get_dofs(), dtype=np.float64)

    surface_spec = make_surface_xyz_tensor_fourier_spec(
        dofs=surface_dofs,
        quadpoints_phi=np.asarray(fitted.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(fitted.quadpoints_theta, dtype=np.float64),
        nfp=SMALL_NFP,
        stellsym=True,
        mpol=SMALL_MPOL,
        ntor=SMALL_NTOR,
    )

    winding = SurfaceRZFourier(
        nfp=SMALL_NFP,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 16, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 16, endpoint=False),
    )
    winding.set_rc(0, 0, 1.0)
    winding.set_rc(1, 0, 0.42)
    winding.set_zs(1, 0, 0.42)
    base_curve = CurveCWSFourier(quadpoints=24, order=2, surf=winding)
    base_curve.x = np.asarray(WINDING_COIL_DOFS, dtype=np.float64)
    free_coils = coils_via_symmetries([base_curve], [Current(1.5e5)], SMALL_NFP, True)
    fixed_coils = []
    for index in range(2):
        curve = CurveXYZFourier(16, 1)
        curve.x = np.array(
            [0.0, 0.0, 1.0 + 0.1 * index, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            dtype=np.float64,
        )
        curve.fix_all()
        current = Current(1.0e5)
        current.fix_all()
        fixed_coils.append(Coil(curve, current))
    field = BiotSavartJAX(fixed_coils + list(free_coils))

    vessel = make_surface_rzfourier_spec(
        rc=np.array([[1.0], [0.55]], dtype=np.float64),
        zs=np.array([[0.0], [0.55]], dtype=np.float64),
        rs=np.zeros((2, 1), dtype=np.float64),
        zc=np.zeros((2, 1), dtype=np.float64),
        quadpoints_phi=np.asarray(fitted.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(fitted.quadpoints_theta, dtype=np.float64),
        nfp=SMALL_NFP,
        stellsym=True,
    )

    material = Flat675Material(
        boozer=Flat675BoozerMaterial(
            surface_template=surface_spec,
            coil_dof_extraction=field.coil_dof_extraction_spec(),
            mpol=SMALL_MPOL,
            ntor=SMALL_NTOR,
            nfp=SMALL_NFP,
            nphi=SMALL_GRID,
            ntheta=SMALL_GRID,
            layout=layout,
        ),
        vessel_template=vessel,
    )
    start = np.concatenate(
        (
            np.asarray(field.x, dtype=np.float64),
            np.array([1.0, 0.55, 0.55], dtype=np.float64),
            surface_dofs,
        )
    )
    return material, layout, start


def test_small_layout_material_carries_its_own_widths() -> None:
    """135 outer coordinates, not 675 — the core stopped assuming."""
    material, layout, start = _small_layout_problem()

    assert layout.surface_dof_count == SMALL_SURFACE_DOF_COUNT
    assert layout.outer_dof_count == FLAT675_COIL_DOF_COUNT + 3 + 121
    assert material.layout == layout
    assert start.shape == (layout.outer_dof_count,)
    assert start.shape != (FLAT675_OUTER_DOF_COUNT,)


def test_small_layout_runs_a_full_fused_solve() -> None:
    """The generalized core solves a problem that is not the certified one.

    Correctness gates only: finite endpoint, objective improved, endpoint
    gradient infinity norm strictly below the start's.  Any seconds this run
    takes are incidental and carry no claim.
    """
    material, layout, start = _small_layout_problem()
    policy = default_flat675_objective_policy(optimized_coil_index=OPTIMIZED_COIL_INDEX)
    programs = bind_flat675_programs(
        material=material,
        objective_policy=policy,
        boozer_policy=DEFAULT_FLAT675_BOOZER_POLICY,
    )
    gradient_of = jax.grad(programs.objective_fn)

    start_device = jax.device_put(start)
    start_objective = float(programs.objective_fn(start_device))
    start_gradient = np.asarray(
        jax.device_get(gradient_of(start_device)), dtype=np.float64
    )

    prepared = prepare_single_stage_flat675(
        objective_fn=programs.objective_fn,
        diagnostics_fn=programs.diagnostics_fn,
        initial_parameters=start_device,
        objective_scale=jax.device_put(np.asarray(1.0, dtype=np.float64)),
    )
    with host_transfer_audit() as audit, jax.transfer_guard("disallow"):
        result = solve_single_stage_flat675(
            prepared,
            driver=Driver.SIMSOPT_LBFGSB,
            max_steps=SMALL_STEPS,
            rtol=0.0,
            atol=1.0e-12,
        )
    endpoint = np.asarray(jax.device_get(result.x), dtype=np.float64)
    endpoint_gradient = np.asarray(
        jax.device_get(gradient_of(jax.device_put(endpoint))), dtype=np.float64
    )
    ledger = {entry.phase: entry.calls for entry in audit.summary()}

    assert endpoint.shape == (layout.outer_dof_count,)
    assert np.all(np.isfinite(endpoint))
    assert np.isfinite(start_objective)
    assert float(result.fun) < start_objective
    assert np.max(np.abs(endpoint_gradient)) < np.max(np.abs(start_gradient))
    # The fused lane's discipline is layout-independent: still no per-step
    # host round trip once the widths stop being 675.
    assert ledger.get("advance", 0) == 0
    assert ledger.get("callback", 0) == 0
    assert ledger.get("final_result", 0) > 0
