"""Stage-II example objectives: device-operand weights publish unchanged values.

The bounded problems of ``examples/jax/3_Advanced/stage_two_optimization_finitebuild.py``
and ``coil_forces.py`` are rebuilt here from the same public constructors and
constants (the scripts are entry points, not importable modules; their complete
bounded construction is compared by AST below), and the shipped coil
forces run is executed as the runner executes it, as a fresh child process.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
import itertools
import json
import os
import subprocess
import sys
import sysconfig
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
import simsoptpp
from examples.jax._lane_environment import build_execution_environment
from simsopt.field import (
    Coil,
    Current,
    apply_symmetries_to_currents,
    apply_symmetries_to_curves,
    coils_via_symmetries,
)
from simsopt.geo import (
    CurveLength,
    SurfaceRZFourier,
    create_equally_spaced_curves,
    create_multifilament_grid,
)
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.core import compute_filament_offsets
from simsopt_jax.objectives import StageTwoObjectiveConfig
from simsopt_jax.solve.serial import (
    TraceableParametricScalarProblem,
    TraceableScalarProblem,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives import (
    FiniteBuildStageTwoConfig,
    ForceStageTwoConfig,
    make_finite_build_stage_two_objective,
    make_force_stage_two_length_penalty,
    make_force_stage_two_objective,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "examples" / "jax" / "3_Advanced"
TEST_DATA = REPO_ROOT / "tests" / "test_files"
# ``main()`` runs both examples with this budget in bounded mode.
_BOUNDED_STEPS = 3

#: The scripts' own constants, pinned against their source by the first test.
FINITE_BUILD_CONSTANTS = {
    "SOLVE_OBJECTIVE_SCALE": 1.0e-4,
    "NUM_BASE_CURVES": 4,
    "NUM_FILAMENTS_N": 2,
    "NUM_FILAMENTS_B": 3,
    "GAP_SIZE_N": 0.02,
    "GAP_SIZE_B": 0.04,
    "ROTATION_ORDER": 1,
}
COIL_FORCES_CONSTANTS = {
    "FIRST_LENGTH_WEIGHT": 1.0e-3,
    "SECOND_LENGTH_WEIGHT": 1.0e-4,
}
FIRST_LENGTH_WEIGHT = COIL_FORCES_CONSTANTS["FIRST_LENGTH_WEIGHT"]
SECOND_LENGTH_WEIGHT = COIL_FORCES_CONSTANTS["SECOND_LENGTH_WEIGHT"]


def _qa_surface() -> SurfaceRZFourier:
    return SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        range="half period",
        nphi=4,
        ntheta=4,
    )


def _finite_build_problem():
    """``stage_two_optimization_finitebuild._build_problem("bounded")``."""
    constants = FINITE_BUILD_CONSTANTS
    count = int(constants["NUM_BASE_CURVES"])
    filaments_n = int(constants["NUM_FILAMENTS_N"])
    filaments_b = int(constants["NUM_FILAMENTS_B"])
    gap_n = float(constants["GAP_SIZE_N"])
    gap_b = float(constants["GAP_SIZE_B"])
    surface = _qa_surface()
    base_curves = create_equally_spaced_curves(
        count,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.7,
        order=2,
        numquadpoints=8,
        use_jax_curve=False,
    )
    filament_count = filaments_n * filaments_b
    base_currents = []
    for index in range(count):
        current = Current(1.0)
        if index == 0:
            current.fix_all()
        base_currents.append(current * (1.0e5 / filament_count))
    base_filaments = list(
        itertools.chain.from_iterable(
            create_multifilament_grid(
                curve,
                filaments_n,
                filaments_b,
                gap_n,
                gap_b,
                rotation_order=int(constants["ROTATION_ORDER"]),
            )
            for curve in base_curves
        )
    )
    filament_currents = list(
        itertools.chain.from_iterable(
            [current] * filament_count for current in base_currents
        )
    )
    filament_curves = apply_symmetries_to_curves(base_filaments, surface.nfp, True)
    currents = apply_symmetries_to_currents(filament_currents, surface.nfp, True)
    coils = [
        Coil(curve, current)
        for curve, current in zip(filament_curves, currents, strict=True)
    ]
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    initial_lengths = tuple(float(CurveLength(curve).J()) for curve in base_curves)
    config = FiniteBuildStageTwoConfig(
        num_base_curves=count,
        filament_offsets=compute_filament_offsets(
            numfilaments_n=filaments_n,
            numfilaments_b=filaments_b,
            gapsize_n=gap_n,
            gapsize_b=gap_b,
        ),
        symmetry_copies=surface.nfp * 2,
        length_targets=initial_lengths,
        length_weight=1.0e-2,
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=10.0,
    )
    return field, flux, config


def _force_stage_config() -> StageTwoObjectiveConfig:
    """``coil_forces._stage_config()``."""
    return StageTwoObjectiveConfig(
        num_base_curves=3,
        length_target=17.4,
        length_target_mode="max",
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=1000.0,
        curve_surface_minimum_distance=0.3,
        curve_surface_weight=10.0,
        curvature_threshold=5.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=5.0,
        mean_squared_curvature_weight=1.0e-6,
    )


def _force_terms_config() -> ForceStageTwoConfig:
    """``coil_forces._force_config()``."""
    return ForceStageTwoConfig(
        num_force_coils=3,
        force_weight=1.0e-2,
        vacuum_energy_weight=1.0e-4,
        force_power=4.0,
        force_threshold=0.0,
        downsample=1,
    )


def _force_problem():
    """``coil_forces._build_problem("bounded")``, on the runtime policy's device."""
    surface = _qa_surface()
    base_curves = create_equally_spaced_curves(
        3,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=2,
        numquadpoints=8,
        use_jax_curve=False,
    )
    base_currents = [Current(1.0e5) for _ in base_curves]
    base_currents[0].fix_all()
    regularization = 0.05**2 / np.sqrt(np.e)
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        surface.stellsym,
        [regularization for _ in base_curves],
    )
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    device = get_runtime_jax_device()
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)), device
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)), device
    )
    target_quadpoints = jax.device_put(
        np.stack(
            tuple(
                np.asarray(curve.quadpoints, dtype=np.float64) for curve in base_curves
            )
        ),
        device,
    )
    regularizations = jax.device_put(
        np.full(len(coils), regularization, dtype=np.float64), device
    )
    return (
        field,
        flux,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
    )


def _child_environment() -> dict[str, str]:
    _, environment = build_execution_environment(
        "cpu", "fast", os.environ, repo_root=REPO_ROOT
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(Path(simsoptpp.__file__).resolve().parent),
            str(REPO_ROOT / "src"),
            str(sysconfig.get_paths()["purelib"]),
            str(Path(jax.__file__).resolve().parents[1]),
        )
    )
    return environment


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _literal_bindings(module: ast.Module) -> dict[str, ast.expr]:
    return {
        node.targets[0].id: node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant | ast.Dict)
    }


def _call(scope: ast.AST, callee: str) -> ast.Call:
    return next(
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == callee
    )


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


class _BoundedConstruction(ast.NodeTransformer):
    """Resolve literal aliases and bounded settings; preserve every other AST node.

    Comparing entire bodies retains constructor arguments, wiring, loops,
    current fixing and call order rather than a selected list of constants.
    """

    def __init__(self, module: ast.Module, surface: ast.expr) -> None:
        self.bindings = _literal_bindings(module)
        self.helpers = {"_qa_surface": surface}
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "_stage_config",
                "_force_config",
                "_force_stage_config",
                "_force_terms_config",
            }:
                returned = node.body[-1]
                assert isinstance(returned, ast.Return) and returned.value is not None
                self.helpers[node.name] = returned.value

    def describe(self, node: ast.AST) -> str:
        return ast.dump(self.visit(deepcopy(node)))

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if isinstance(node.ctx, ast.Load):
            bound = self.bindings.get(node.id)
            if bound is not None:
                return self.visit(deepcopy(bound))
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.value, ast.Dict):
            return next(
                value
                for key, value in zip(node.value.keys, node.value.values, strict=True)
                if ast.dump(key) == ast.dump(node.slice)
            )
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in self.helpers:
                assert not node.args and not node.keywords
                return self.visit(deepcopy(self.helpers[node.func.id]))
            if node.func.id in {"int", "float"} and isinstance(
                node.args[0], ast.Constant
            ):
                value = node.args[0].value
                assert isinstance(value, int | float)
                return ast.Constant(
                    int(value) if node.func.id == "int" else float(value)
                )
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "native_scale":
                assert ast.dump(node.value) == ast.dump(
                    ast.parse('scale == "native_default"', mode="eval").body
                ), "bounded scale selection drift"
                self.bindings[name] = ast.Constant(False)
                return None
            self.generic_visit(node)
            if isinstance(node.value, ast.Constant | ast.Dict):
                self.bindings[name] = node.value
                return None
            return node
        self.generic_visit(node)
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        node.test = self.visit(node.test)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            return self.visit(node.body if node.test.value else node.orelse)
        self.generic_visit(node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        self.generic_visit(node)
        return node


def _assert_complete_construction(name: str, source: str) -> None:
    script = ast.parse(source)
    builders = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    surface_return = _function(builders, "_qa_surface").body[0]
    assert isinstance(surface_return, ast.Return) and surface_return.value is not None
    finite_build = name == "stage_two_optimization_finitebuild"
    constants_name = (
        "FINITE_BUILD_CONSTANTS" if finite_build else "COIL_FORCES_CONSTANTS"
    )
    constants = _literal_bindings(builders)[constants_name]
    assert isinstance(constants, ast.Dict)
    constant_values: dict[str, ast.expr] = {}
    script_constants = _literal_bindings(script)
    for key, value in zip(constants.keys, constants.values, strict=True):
        assert isinstance(key, ast.Constant) and isinstance(key.value, str)
        constant_values[key.value] = value
        assert ast.dump(script_constants[key.value]) == ast.dump(value), (
            f"{name}.{key.value} construction drift"
        )
    pairs = (
        [("_build_problem", "_finite_build_problem")]
        if finite_build
        else [
            ("_build_problem", "_force_problem"),
            ("_stage_config", "_force_stage_config"),
            ("_force_config", "_force_terms_config"),
        ]
    )
    for script_function, builder_function in pairs:
        actual = _BoundedConstruction(script, surface_return.value).describe(
            ast.Module(_function(script, script_function).body, [])
        )
        expected = _BoundedConstruction(builders, surface_return.value).describe(
            ast.Module(_function(builders, builder_function).body, [])
        )
        assert actual == expected, f"{name}.{script_function} construction drift"

    factories = (
        ["make_finite_build_stage_two_objective"]
        if finite_build
        else ["make_force_stage_two_objective", "make_force_stage_two_length_penalty"]
    )
    script_solve = _function(script, "solve")
    builder_state = _function(
        builders, "_finite_build_state" if finite_build else "_force_state"
    )
    for factory in factories:
        script_call = _call(script_solve, factory)
        builder_call = _call(builder_state, factory)
        actual = _BoundedConstruction(script, surface_return.value)
        expected = _BoundedConstruction(builders, surface_return.value)
        for scope, normalizer in ((script_solve, actual), (builder_state, expected)):
            for statement in scope.body:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id
                    in {"flux_objective", "stage_config", "force_config"}
                ):
                    normalizer.bindings[statement.targets[0].id] = statement.value
        if not finite_build:
            parameter = _function(builders, "build").args.args[0].arg
            expected.bindings[parameter] = expected.bindings["stage_config"]
        assert actual.describe(script_call) == expected.describe(builder_call), (
            f"{name}.{factory} construction drift"
        )

    weight_inputs = (
        [
            (
                _keyword(
                    _call(script_solve, "prepare_finite_build_stage_two"),
                    "objective_scale",
                ),
                _call(builder_state, "_device_scalar").args[0],
            )
        ]
        if finite_build
        else [
            (
                _keyword(
                    _call(script_solve, "TraceableParametricScalarProblem"),
                    "objective_parameter",
                ),
                constant_values["FIRST_LENGTH_WEIGHT"],
            ),
            (
                _call(script_solve, "problem.set_objective_parameter").args[0],
                constant_values["SECOND_LENGTH_WEIGHT"],
            ),
        ]
    )
    for script_input, builder_input in weight_inputs:
        actual = _BoundedConstruction(script, surface_return.value)
        expected = _BoundedConstruction(builders, surface_return.value)
        expected.bindings["value"] = builder_input
        scalar = _call(_function(builders, "_device_scalar"), "np.asarray")
        assert actual.describe(_call(script_input, "np.asarray")) == expected.describe(
            scalar
        ), f"{name} device weight construction drift"


@pytest.mark.parametrize("name", ["stage_two_optimization_finitebuild", "coil_forces"])
def test_rebuilt_problems_match_complete_script_construction(name: str) -> None:
    _assert_complete_construction(
        name, (EXAMPLES / f"{name}.py").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("name", "before", "after"),
    [
        ("stage_two_optimization_finitebuild", "R1=0.7", "R1=0.8"),
        (
            "stage_two_optimization_finitebuild",
            "length_weight=1.0e-2",
            "length_weight=2.0e-2",
        ),
        (
            "stage_two_optimization_finitebuild",
            "rotation_order=ROTATION_ORDER",
            "rotation_order=2",
        ),
        (
            "stage_two_optimization_finitebuild",
            "SOLVE_OBJECTIVE_SCALE = 1.0e-4",
            "SOLVE_OBJECTIVE_SCALE = 2.0e-4",
        ),
        (
            "stage_two_optimization_finitebuild",
            "np.asarray(SOLVE_OBJECTIVE_SCALE, dtype=np.float64)",
            "np.asarray(PUBLISHED_OBJECTIVE_SCALE, dtype=np.float64)",
        ),
        ("coil_forces", "R1=0.5", "R1=0.6"),
        ("coil_forces", "length_target=17.4", "length_target=18.4"),
        ("coil_forces", "force_weight=1.0e-2", "force_weight=2.0e-2"),
        ("coil_forces", "base_currents[0].fix_all()", ""),
        ("coil_forces", "FIRST_LENGTH_WEIGHT = 1.0e-3", "FIRST_LENGTH_WEIGHT = 2.0e-3"),
        (
            "coil_forces",
            "np.asarray(FIRST_LENGTH_WEIGHT, dtype=np.float64)",
            "np.asarray(SECOND_LENGTH_WEIGHT, dtype=np.float64)",
        ),
        (
            "coil_forces",
            "flux_objective = flux.traceable_objective()",
            "flux_objective = flux.fixed_surface_flux_spec()",
        ),
    ],
)
def test_complete_construction_check_rejects_script_drift(name, before, after) -> None:
    source = (EXAMPLES / f"{name}.py").read_text(encoding="utf-8")
    assert source.count(before) == 1
    with pytest.raises(AssertionError, match="construction drift"):
        _assert_complete_construction(name, source.replace(before, after))


def _bitwise_equal(first: jax.Array, second: jax.Array) -> bool:
    return bool(
        np.array_equal(
            np.asarray(first, dtype=np.float64).view(np.uint64),
            np.asarray(second, dtype=np.float64).view(np.uint64),
        )
    )


def _device_scalar(value: float) -> jax.Array:
    return jax.device_put(np.asarray(value, dtype=np.float64))


def _finite_build_state() -> tuple[jax.Array, TraceableParametricScalarProblem, object]:
    field, flux, config = _finite_build_problem()
    objective = make_finite_build_stage_two_objective(
        field,
        flux.fixed_surface_flux_spec(),
        config,
    )
    parameters = jax.device_put(np.asarray(field.x, dtype=np.float64))
    problem = TraceableParametricScalarProblem(
        objective_fn=lambda current, objective_scale: (
            objective_scale * objective(current)
        ),
        objective_parameter=_device_scalar(
            float(FINITE_BUILD_CONSTANTS["SOLVE_OBJECTIVE_SCALE"])
        ),
        x=parameters,
    )
    return parameters, problem, objective


def _force_state():
    (
        field,
        flux,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
    ) = _force_problem()
    stage_config = _force_stage_config()
    force_config = _force_terms_config()

    def build(config) -> object:
        return make_force_stage_two_objective(
            field,
            flux.traceable_objective(),
            surface_gamma,
            surface_normal,
            target_quadpoints,
            regularizations,
            config,
            force_config,
        )

    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    zero_weight_objective = build(stage_config)

    def weighted_objective(current: jax.Array, length_weight: jax.Array) -> jax.Array:
        return zero_weight_objective(current) + length_penalty(current, length_weight)

    parameters = jax.device_put(np.asarray(field.x, dtype=np.float64))
    return field, stage_config, build, weighted_objective, parameters


def test_finitebuild_publishes_the_fresh_unscaled_gradient_bit_for_bit() -> None:
    parameters, problem, objective = _finite_build_state()
    problem.value_and_grad(parameters)
    problem.set_objective_parameter(_device_scalar(1.0))
    published_value, published_gradient = problem.value_and_grad(parameters)

    reference = TraceableScalarProblem(objective, parameters)
    reference_value, reference_gradient = reference.value_and_grad(parameters)

    assert _bitwise_equal(published_value, reference_value)
    assert _bitwise_equal(published_gradient, reference_gradient)


def test_coil_forces_weighted_objective_is_bitwise_static_below_the_length_target() -> (
    None
):
    """The receipt guard: at the shipped start the split objective is bit-exact.

    The example's base curves start under the 17.4 m length target, so the
    penalty is exactly zero and ``objective(x) + 0.0`` reassociates nothing.
    That is the only regime where the refactored formulation can claim bitwise
    equality with the pre-change one, so the equality is asserted here and only
    here; the active regime is the sibling test below.
    """
    field, stage_config, build, weighted_objective, parameters = _force_state()
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    weighted_program = jax.jit(jax.value_and_grad(weighted_objective, argnums=0))

    for weight in (FIRST_LENGTH_WEIGHT, SECOND_LENGTH_WEIGHT):
        static_program = jax.jit(
            jax.value_and_grad(build(replace(stage_config, length_weight=weight)))
        )
        static_value, static_gradient = static_program(parameters)
        value, gradient = weighted_program(parameters, _device_scalar(weight))

        assert float(length_penalty(parameters, _device_scalar(weight))) == 0.0, (
            "the total base-curve length crossed the length target, so this "
            "point no longer certifies the bitwise claim"
        )
        assert _bitwise_equal(value, static_value)
        assert _bitwise_equal(gradient, static_gradient)


def test_coil_forces_weighted_objective_matches_each_static_length_weight() -> None:
    """Where the penalty is active the device weight must price it correctly.

    Below the length target the weight multiplies an exactly-zero excess, so a
    comparison there cannot fail on the weight at all; this loop runs at an
    extended point where each weight moves the objective.  The refactor
    reassociates the penalty sum, so the agreement is at the ~1 ULP level rather
    than bitwise.
    """
    field, stage_config, build, weighted_objective, parameters = _force_state()
    extended = parameters * 2.0
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    weighted_program = jax.jit(jax.value_and_grad(weighted_objective, argnums=0))

    for weight in (FIRST_LENGTH_WEIGHT, SECOND_LENGTH_WEIGHT):
        static_program = jax.jit(
            jax.value_and_grad(build(replace(stage_config, length_weight=weight)))
        )
        static_value, static_gradient = static_program(extended)
        value, gradient = weighted_program(extended, _device_scalar(weight))

        assert float(length_penalty(extended, _device_scalar(weight))) > 0.0
        np.testing.assert_allclose(
            np.asarray(value),
            np.asarray(static_value),
            rtol=1.0e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(gradient),
            np.asarray(static_gradient),
            rtol=1.0e-13,
            atol=0.0,
        )


def test_coil_forces_active_length_penalty_separates_the_two_stage_weights() -> None:
    """Where the penalty bites, the stage weight moves the objective it prices.

    The excess over the length target is read back from the penalty at the first
    weight, so the assertion is on the whole weighted objective: switching the
    device weight may only shift it by the length penalty it buys.
    """
    field, stage_config, _build, weighted_objective, parameters = _force_state()
    extended = parameters * 2.0
    first_weight = FIRST_LENGTH_WEIGHT
    second_weight = SECOND_LENGTH_WEIGHT
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    weighted_program = jax.jit(weighted_objective)
    first_value = float(weighted_program(extended, _device_scalar(first_weight)))
    second_value = float(weighted_program(extended, _device_scalar(second_weight)))
    excess_squared = (
        2.0 * float(length_penalty(extended, _device_scalar(first_weight)))
    ) / first_weight

    assert first_value != second_value, (
        "the length weight left the objective unchanged at a point where the "
        f"penalty is active (excess^2={excess_squared:.6e})"
    )
    np.testing.assert_allclose(
        first_value - second_value,
        0.5 * (first_weight - second_weight) * excess_squared,
        rtol=1.0e-9,
        atol=0.0,
    )


# The published values below are regression pins taken from one live bounded
# run on this branch, not independent oracles: they fail when the example's
# numbers move, and say nothing about whether the physics is right.
#
# Re-anchored 2026-09-20 from one bounded device-mode solve on this machine
# (CPU, fp64, OMP_NUM_THREADS=4).
#
# "No behaviour change" HOLDS ONLY AGAINST THE PRE-WAVE-3 SOURCE: the values
# below are bitwise equal to what that source publishes on this machine.
# Against the campaign baseline the device endpoint DID move, and the campaign
# change that moved it is FINITE_BUILD_LBFGS_HISTORY 10 -> 400 in
# src/simsopt_jax/examples/stage_two_finitebuild.py, still passed as
# ``lbfgs_history`` to the fused device solve this test drives (and as
# ``maxcor`` to the official SciPy route).  400 is upstream's own number
# (official 3_Advanced/stage_two_optimization_finitebuild.py line 173,
# ``'maxcor': 400``), so the change is a fidelity correction -- but it IS a
# device-lane behaviour change, and the certified finite-build device speed-up
# was measured at history 10.
#
def test_coil_forces_example_bounded_solve_lands_on_its_endpoint(tmp_path) -> None:
    """Both shipped stages run, the second after swapping the device length weight."""
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            str(EXAMPLES / "coil_forces.py"),
            "--smoke",
            "--json",
            "--output-dir",
            str(tmp_path),
        ),
        cwd=REPO_ROOT,
        env=_child_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    observables = result["observables"]

    assert result["status"] == "ok"
    assert observables["solver_iterations"] == [_BOUNDED_STEPS, _BOUNDED_STEPS]
    # Regression pin retaken 2026-09-13 after the mirror adopted native's
    # post-Taylor start state and SciPy L-BFGS-B at native policy (ftol=gtol=1e-15).
    np.testing.assert_allclose(
        observables["final_objective"],
        0.003424195336484035,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        observables["squared_flux"],
        0.0033617810994737273,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        observables["vacuum_energy"],
        0.6148225859961913,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_force_stage_two_length_penalty_rejects_a_static_length_weight() -> None:
    field, stage_config, _build, _objective, _parameters = _force_state()

    with pytest.raises(ValueError, match="length_weight must be zero"):
        make_force_stage_two_length_penalty(
            field,
            replace(stage_config, length_weight=1.0e-3),
        )
