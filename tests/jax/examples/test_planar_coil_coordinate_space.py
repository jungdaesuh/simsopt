"""The native planar script, its fixed-surface twin, and the mirror share one space.

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
and the static construction check fails if either the surface or first current
is left free, or if construction changes the coordinate order.
The native script is read, never executed: its construction is checked against
the twin's term by term, and the twin carries the numeric claims.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
from collections.abc import Mapping
from pathlib import Path

import jax
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
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import (
    solve_standard_stage_two,
    standard_stage_two_optimizer_observables,
)
from simsopt_jax.objectives import StageTwoObjectiveConfig
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

ROOT = Path(__file__).resolve().parents[3]
MIRROR = (
    ROOT
    / "examples"
    / "jax"
    / "2_Intermediate"
    / "stage_two_optimization_planar_coils.py"
)
NATIVE_SCRIPT = (
    ROOT / "examples" / "2_Intermediate" / "stage_two_optimization_planar_coils.py"
)
NATIVE_SURFACE = ROOT / "tests" / "test_files" / "input.LandremanPaul2021_QA"

NCOILS = 4
CURVE_ORDER = 5
CURVE_QUADPOINTS = 75
SURFACE_RESOLUTION = 32
LENGTH_TARGET = 10.4
FIRST_LENGTH_WEIGHT = 10.0
CC_THRESHOLD = 0.08
CC_WEIGHT = 1000.0
CS_THRESHOLD = 0.12
CS_WEIGHT = 10.0
CURVATURE_THRESHOLD = 10.0
CURVATURE_WEIGHT = 1.0e-6
MSC_THRESHOLD = 10.0
MSC_WEIGHT = 1.0e-6
LINKING_NUMBER_WEIGHT = 1.0
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
        + CC_WEIGHT * CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=NCOILS)
        + CS_WEIGHT * CurveSurfaceDistance(curves, surface, CS_THRESHOLD)
        + CURVATURE_WEIGHT
        * sum(LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves)
        + MSC_WEIGHT
        * sum(
            QuadraticPenalty(MeanSquaredCurvature(c), MSC_THRESHOLD)
            for c in base_curves
        )
        + LINKING_NUMBER_WEIGHT * LinkingNumber(curves)
    )


# --- Reading the native script without executing it ---------------------------
# A term is described structurally: calls by callee name and described arguments,
# numbers by their value (module constants resolved), comprehension loop
# variables by what they range over, and every other name as an opaque variable,
# so the script's ``s``/``bs`` and the twin's ``surface``/``field`` compare equal.

_VARIABLE = ("variable",)


def _number(node: ast.expr, constants: Mapping[str, float]) -> float | None:
    """The value of a numeric literal expression, or ``None`` if it is not one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _number(node.operand, constants)
        return None if operand is None else -operand
    if isinstance(node, ast.BinOp):
        left = _number(node.left, constants)
        right = _number(node.right, constants)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Div):
            return left / right
        return None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Weight"
        and len(node.args) == 1
    ):
        return _number(node.args[0], constants)
    return None


def _assignments(statements: list[ast.stmt]) -> dict[str, ast.expr]:
    """Single-name assignments, the last one before the end of ``statements``."""
    return {
        statement.targets[0].id: statement.value
        for statement in statements
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    }


def _constants(bindings: Mapping[str, ast.expr]) -> dict[str, float]:
    constants: dict[str, float] = {}
    for name, value in bindings.items():
        number = _number(value, constants)
        if number is not None:
            constants[name] = number
    return constants


def _describe(
    node: ast.expr,
    bindings: Mapping[str, ast.expr],
    constants: Mapping[str, float],
    loop: Mapping[str, tuple[object, ...]],
    *,
    term: bool = False,
) -> tuple[object, ...]:
    """Describe ``node``; ``term`` resolves a name bound to an objective call."""
    number = _number(node, constants)
    if number is not None:
        return ("number", number)
    if isinstance(node, ast.Name):
        if node.id in loop:
            return loop[node.id]
        bound = bindings.get(node.id)
        if isinstance(bound, ast.ListComp | ast.GeneratorExp) or (
            term and isinstance(bound, ast.Call)
        ):
            return _describe(bound, bindings, constants, loop)
        return _VARIABLE
    if isinstance(node, ast.Attribute):
        return (
            "attribute",
            node.attr,
            _describe(node.value, bindings, constants, loop),
        )
    if isinstance(node, ast.ListComp | ast.GeneratorExp):
        (generator,) = node.generators
        assert isinstance(generator.target, ast.Name)
        element = _describe(generator.iter, bindings, constants, loop)
        if element[0] == "each":
            element = element[1]
        else:
            element = _VARIABLE
        inner = {**loop, generator.target.id: element}
        return ("each", _describe(node.elt, bindings, constants, inner))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return (
            "call",
            node.func.id,
            tuple(_describe(arg, bindings, constants, loop) for arg in node.args),
            tuple(
                sorted(
                    (
                        str(keyword.arg),
                        _describe(keyword.value, bindings, constants, loop),
                    )
                    for keyword in node.keywords
                )
            ),
        )
    raise AssertionError(f"undescribed objective term: {ast.dump(node)}")


def _terms(
    expression: ast.expr,
    bindings: Mapping[str, ast.expr],
    constants: Mapping[str, float],
) -> list[tuple[float, tuple[object, ...]]]:
    """``(weight, description)`` of each ``+``-separated term of an objective."""
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        return _terms(expression.left, bindings, constants) + _terms(
            expression.right, bindings, constants
        )
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mult):
        weight = _number(expression.left, constants)
        if weight is not None:
            return [
                (
                    weight,
                    _describe(expression.right, bindings, constants, {}, term=True),
                )
            ]
    return [(1.0, _describe(expression, bindings, constants, {}, term=True))]


def _native_script_statements() -> list[ast.stmt]:
    source = NATIVE_SCRIPT.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(NATIVE_SCRIPT)).body


def _statement_index(statements: list[ast.stmt], matches) -> int:
    return next(
        index for index, statement in enumerate(statements) if matches(statement)
    )


def _assigns(name: str):
    return lambda statement: (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == name
    )


def _native_objective_terms() -> list[tuple[float, tuple[object, ...]]]:
    statements = _native_script_statements()
    objective_at = _statement_index(statements, _assigns("JF"))
    bindings = _assignments(statements[:objective_at])
    return _terms(
        _assignments(statements[: objective_at + 1])["JF"],
        bindings,
        _constants(bindings),
    )


def _twin_objective_terms() -> list[tuple[float, tuple[object, ...]]]:
    module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    twin = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_native_twin_objective"
    )
    returned = next(node for node in twin.body if isinstance(node, ast.Return))
    assert returned.value is not None
    bindings = _assignments(twin.body)
    return _terms(returned.value, bindings, _constants(_assignments(module.body)))


class _InlineNumericLiterals(ast.NodeTransformer):
    """Replace names bound to ``int`` or ``float`` literals by those literals.

    The literal keeps its type: ``ncoils = 4.0`` must not match ``4``.
    """

    def __init__(self, bindings: Mapping[str, ast.expr]) -> None:
        self.bindings = bindings

    def visit_Name(self, node: ast.Name) -> ast.expr:
        bound = self.bindings.get(node.id)
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(bound, ast.Constant)
            and type(bound.value) in (int, float)
        ):
            return ast.Constant(value=bound.value)
        return node



def _assert_native_coordinate_construction(source: str) -> None:
    """Pin geometry, fixed DOFs and object creation order before the first JF.

    Optimizable orders parents by creation, so term order alone cannot prove
    that the script and twin index the same physical coordinates. Every
    statement from the surface to ``JF``, VTK output included, must match, so
    an added call such as ``s.unfix_all()`` is drift too.
    """
    statements = ast.parse(source).body
    objective_at = _statement_index(statements, _assigns("JF"))
    expected = ast.parse(
        f"""
s = SurfaceRZFourier.from_vmec_input(filename, range="half period",
                                    nphi={SURFACE_RESOLUTION}, ntheta={SURFACE_RESOLUTION})
s.fix_all()
base_curves = create_equally_spaced_planar_curves({NCOILS}, s.nfp, stellsym=True,
                                                R0=1.0, R1=0.5, order={CURVE_ORDER})
base_currents = [Current(1e5) for i in range({NCOILS})]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {{"B_N": np.sum(bs.B().reshape(({SURFACE_RESOLUTION}, {SURFACE_RESOLUTION}, 3)) * s.unitnormal(), axis=2)[:, :, None]}}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, {CC_THRESHOLD}, num_basecurves={NCOILS})
Jcsdist = CurveSurfaceDistance(curves, s, {CS_THRESHOLD})
Jcs = [LpCurveCurvature(c, 2, {CURVATURE_THRESHOLD}) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
linkNum = LinkingNumber(curves)
"""
    )
    surface_at = _statement_index(statements, _assigns("s"))
    construction = statements[surface_at:objective_at]
    bindings = _assignments(statements[:surface_at])
    actual = _InlineNumericLiterals(bindings).visit(ast.Module(construction, []))
    assert ast.dump(actual) == ast.dump(expected), (
        "native coordinate construction drift"
    )


def test_native_script_preserves_geometry_fixed_dofs_and_construction_order() -> None:
    _assert_native_coordinate_construction(NATIVE_SCRIPT.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("s.fix_all()", ""),
        ("s.fix_all()", "s.fix_all()\ns.unfix_all()"),
        ("base_currents[0].fix_all()", ""),
        ("ncoils = 4", "ncoils = 5"),
        ("ncoils = 4", "ncoils = 4.0"),
        (
            'curves_to_vtk(curves, OUT_DIR + "curves_init")\npointData = {"B_N": '
            "np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}",
            'curves_to_vtk(curves, OUT_DIR + "curves_init")\npointData = s.unfix_all()',
        ),
        ("R1 = 0.5", "R1 = 0.6"),
        ("order = 5", "order = 6"),
        (
            "Jf = SquaredFlux(s, bs)\nJls = [CurveLength(c) for c in base_curves]",
            "Jls = [CurveLength(c) for c in base_curves]\nJf = SquaredFlux(s, bs)",
        ),
        (
            "Jls = [CurveLength(c) for c in base_curves]",
            "Jls = [CurveLength(c) for c in reversed(base_curves)]",
        ),
    ],
)
def test_coordinate_construction_check_rejects_script_drift(before, after) -> None:
    source = NATIVE_SCRIPT.read_text(encoding="utf-8")
    assert source.count(before) == 1
    with pytest.raises(AssertionError, match="native coordinate construction drift"):
        _assert_native_coordinate_construction(source.replace(before, after))


def test_native_script_objective_is_the_twins_term_by_term() -> None:
    """The script's ``JF`` has the twin's terms, arguments and weights, in order."""
    assert _native_objective_terms() == _twin_objective_terms()


@pytest.mark.native_cpu_reference
def test_twin_objective_and_mirror_field_expose_the_same_coordinate_order() -> None:
    surface, base_curves, coils = _geometry()
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    objective = _native_twin_objective(surface, base_curves, coils, field)

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
