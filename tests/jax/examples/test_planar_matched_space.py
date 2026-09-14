"""The native planar script, its parity twin, and the mirror share one space.

All three optimize the coil set alone: 4x18 ``CurvePlanarFourier`` coordinates
plus the three free currents, ordered as ``BiotSavart(coils).x``.

History.  ``28e470f43`` gave ``CurveSurfaceDistance`` ownership of the surface
and added ``s.fix_all()`` to ``stage_two_optimization.py`` and
``coil_forces.py`` but not to the planar script, which therefore optimized
**196** coordinates -- the 75 above plus 121 free ``SurfaceRZFourier`` ones.
Those 121 were never an optimizable space: ``SquaredFlux`` depends on the field
alone and never re-sets the Biot-Savart evaluation points, so a Taylor test in
surface-only directions plateaued at 2.9911e-2 instead of vanishing while
coil-only directions converged quadratically.  The conversion is completed now,
and ``test_native_script_optimizes_only_the_coil_coordinates`` fails with 196
if the ``s.fix_all()`` line is ever dropped again.
"""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    LinkingNumber,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    create_equally_spaced_planar_curves,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux

ROOT = Path(__file__).resolve().parents[3]
MIRROR = (
    ROOT / "examples" / "jax" / "2_Intermediate" / "stage_two_optimization_planar_coils.py"
)
NATIVE_SCRIPT = (
    ROOT / "examples" / "2_Intermediate" / "stage_two_optimization_planar_coils.py"
)
NATIVE_SURFACE = ROOT / "tests" / "test_files" / "input.LandremanPaul2021_QA"
# Everything the native script does before this marker builds the problem; what
# follows is the Taylor test and the two optimizer calls.
NATIVE_OPTIMIZER_SECTION = 'print("""'

NCOILS = 4
CURVE_ORDER = 5
CURVE_QUADPOINTS = 75
SURFACE_RESOLUTION = 32
LENGTH_TARGET = 10.4
FIRST_LENGTH_WEIGHT = 10.0
COIL_COORDINATES = 75


def _geometry():
    """The twin's construction: fixed surface, four planar base coils, one fixed current."""
    surface = SurfaceRZFourier.from_vmec_input(
        NATIVE_SURFACE,
        range="half period",
        nphi=SURFACE_RESOLUTION,
        ntheta=SURFACE_RESOLUTION,
    )
    surface.fix_all()
    base_curves = create_equally_spaced_planar_curves(
        NCOILS,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=CURVE_ORDER,
        numquadpoints=CURVE_QUADPOINTS,
    )
    base_currents = [Current(1.0e5) for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    return surface, base_curves, coils


def _native_twin_objective(surface, base_curves, coils, field):
    """``examples/2_Intermediate/stage_two_optimization_planar_coils.py`` stage one."""
    curves = [coil.curve for coil in coils]
    return (
        SquaredFlux(surface, field)
        + FIRST_LENGTH_WEIGHT
        * QuadraticPenalty(sum(CurveLength(c) for c in base_curves), LENGTH_TARGET)
        + 1000.0 * CurveCurveDistance(curves, 0.08, num_basecurves=NCOILS)
        + 10.0 * CurveSurfaceDistance(curves, surface, 0.12)
        + 1.0e-6 * sum(LpCurveCurvature(c, 2, 10.0) for c in base_curves)
        + 1.0e-6
        * sum(QuadraticPenalty(MeanSquaredCurvature(c), 10.0) for c in base_curves)
        + 1.0 * LinkingNumber(curves)
    )


def _native_script_problem(working_directory: Path) -> dict[str, object]:
    """Run the shipped script's construction section and hand back its namespace.

    The script is a module-level program, not an importable builder, so the only
    way to read the space it actually optimizes is to execute the source it
    ships, truncated just before its first optimizer call.  ``__file__`` carries
    the real path so the script still resolves ``tests/test_files``; the working
    directory is redirected so its VTK output lands in a temporary tree.
    """
    source = NATIVE_SCRIPT.read_text(encoding="utf-8")
    construction = source[: source.index(NATIVE_OPTIMIZER_SECTION)]
    namespace: dict[str, object] = {
        "__file__": str(NATIVE_SCRIPT),
        "__name__": "native_planar_construction",
    }
    with chdir(working_directory):
        exec(compile(construction, str(NATIVE_SCRIPT), "exec"), namespace)
    return namespace


@pytest.mark.native_cpu_reference
def test_native_script_optimizes_only_the_coil_coordinates(tmp_path: Path) -> None:
    namespace = _native_script_problem(tmp_path)
    objective = namespace["JF"]
    field = namespace["bs"]
    surface = namespace["s"]

    assert surface.x.size == 0, (
        f"the shipped native script left {surface.x.size} surface coordinates "
        "free; CurveSurfaceDistance pulls them into JF.x, where SquaredFlux "
        "owns no partial for them"
    )
    assert objective.x.size == COIL_COORDINATES, (
        f"the shipped native script optimizes {objective.x.size} coordinates, "
        f"not {COIL_COORDINATES}"
    )
    assert list(objective.dof_names) == list(field.dof_names), (
        "the shipped native script's objective orders the coil coordinates "
        "differently from BiotSavart(coils).x, which the mirror indexes"
    )


@pytest.mark.native_cpu_reference
def test_parity_twin_fixes_the_surface_and_exposes_only_coil_coordinates() -> None:
    from examples.jax.parity.cases.native_stage_two_optimization_planar_coils import (
        _build_geometry,
        _scale_configuration,
    )

    configuration = _scale_configuration("native_default")
    surface, _base_curves, coils = _build_geometry(configuration)

    assert surface.x.size == 0, (
        f"the parity twin left {surface.x.size} surface coordinates free; the "
        "matched space is the coil set alone"
    )
    assert BiotSavart(coils).x.size == COIL_COORDINATES


@pytest.mark.native_cpu_reference
def test_twin_objective_and_mirror_field_expose_the_same_coordinate_order() -> None:
    surface, base_curves, coils = _geometry()
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    objective = _native_twin_objective(surface, base_curves, coils, field)

    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

    mirror_field = BiotSavartJAX(coils)

    assert objective.x.size == COIL_COORDINATES, (
        f"the fixed-surface twin optimizes {objective.x.size} coordinates; the "
        "shipped script's 196 come from the surface it forgets to fix"
    )
    assert list(objective.dof_names) == list(field.dof_names), (
        "the twin objective reorders the coil coordinates relative to "
        "BiotSavart(coils).x, so the mirror's parameter vector no longer indexes "
        "the same physical quantities"
    )
    assert list(mirror_field.dof_names) == list(field.dof_names)
    assert np.array_equal(
        np.asarray(mirror_field.x, dtype=np.float64),
        np.asarray(field.x, dtype=np.float64),
    )


@pytest.mark.native_cpu_reference
def test_mirror_selects_natives_optimizer_route_and_tolerances() -> None:
    """What the mirror file itself chooses: SciPy L-BFGS-B, tol 1e-15, 400 steps."""
    source = MIRROR.read_text(encoding="utf-8")

    assert "driver=Driver.SCIPY_LBFGSB" in source
    assert "rtol=1.0e-15" in source and "atol=1.0e-15" in source, (
        "the planar mirror sets its own tolerances; both must equal native's "
        "single tol=1e-15, which SciPy expands into ftol and gtol"
    )
    assert "NATIVE_ITERATIONS = 400" in source


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.native_cpu_reference
def test_mirror_and_native_twin_agree_at_the_matched_initial_state() -> None:
    import jax
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.examples import (
        solve_standard_stage_two,
        standard_stage_two_optimizer_observables,
    )
    from simsopt_jax.objectives import StageTwoObjectiveConfig
    from simsopt_jax.solve.driver import Driver
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    surface, base_curves, coils = _geometry()
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    objective = _native_twin_objective(surface, base_curves, coils, field)
    initial = np.asarray(field.x, dtype=np.float64)
    objective.x = initial
    native_value = float(objective.J())
    native_gradient = np.asarray(objective.dJ(), dtype=np.float64)

    mirror_surface, _mirror_curves, mirror_coils = _geometry()
    mirror_field = BiotSavartJAX(mirror_coils)
    mirror_flux = SquaredFluxJAX(mirror_surface, mirror_field)
    device = get_runtime_jax_device()
    result = solve_standard_stage_two(
        field=mirror_field,
        flux_spec=mirror_flux.fixed_surface_flux_spec(),
        surface_gamma=jax.device_put(
            np.asarray(mirror_surface.gamma(), dtype=np.float64).reshape((-1, 3)),
            device,
        ),
        surface_normal=jax.device_put(
            np.asarray(mirror_surface.normal(), dtype=np.float64).reshape((-1, 3)),
            device,
        ),
        initial_parameters=jax.device_put(initial, device),
        taylor_direction=jax.device_put(
            np.random.RandomState(1).uniform(size=initial.shape), device
        ),
        regularization_config=StageTwoObjectiveConfig(
            num_base_curves=NCOILS,
            length_target=LENGTH_TARGET,
            length_target_mode="identity",
            curve_curve_minimum_distance=0.08,
            curve_curve_weight=1000.0,
            curve_surface_minimum_distance=0.12,
            curve_surface_weight=10.0,
            curvature_threshold=10.0,
            curvature_weight=1.0e-6,
            mean_squared_curvature_threshold=10.0,
            mean_squared_curvature_target_mode="identity",
            mean_squared_curvature_weight=1.0e-6,
            linking_number_weight=1.0,
        ),
        first_length_weight=jax.device_put(
            np.asarray(FIRST_LENGTH_WEIGHT, dtype=np.float64), device
        ),
        second_length_weight=jax.device_put(np.asarray(1.0, dtype=np.float64), device),
        max_steps=1,
        rtol=1.0e-15,
        atol=1.0e-15,
        driver=Driver.SCIPY_LBFGSB,
    )
    mirror_value = float(np.asarray(jax.device_get(result.initial.objective)))
    mirror_gradient = np.asarray(
        jax.device_get(result.initial.objective_gradient), dtype=np.float64
    )

    # Inherited from the shared SciPy route, not chosen by the mirror: native's
    # maxcor=300 and SciPy's own maxfun/maxls defaults, read off the run.
    stage_options = standard_stage_two_optimizer_observables(result)["solver_options"]
    for options in stage_options:
        assert options["maxcor"] == 300
        assert options["maxfun"] == 15000, (
            "the SciPy route stops the mirror at a different evaluation budget "
            f"than native's 15000: {options['maxfun']}"
        )
        assert options["maxls"] == 20

    assert mirror_gradient.shape == native_gradient.shape == (COIL_COORDINATES,)
    relative_value_error = abs(mirror_value - native_value) / abs(native_value)
    relative_gradient_error = float(
        np.linalg.norm(mirror_gradient - native_gradient)
        / np.linalg.norm(native_gradient)
    )
    assert relative_value_error < 1.0e-14, (
        f"objective at the matched initial state differs relatively by "
        f"{relative_value_error:.3e} (mirror {mirror_value!r}, native {native_value!r})"
    )
    assert relative_gradient_error < 1.0e-13, (
        "gradient at the matched initial state differs by relative L2 "
        f"{relative_gradient_error:.3e}; the two lanes no longer evaluate the "
        "same objective on the same 75 coordinates"
    )
