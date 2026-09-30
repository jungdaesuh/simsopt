"""One compiled graph per Stage-II example objective, with unchanged values."""

from __future__ import annotations

import importlib.util
import logging
import re
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import jax
import numpy as np
import pytest
from simsopt_jax.solve.serial import (
    TraceableParametricScalarProblem,
    TraceableScalarProblem,
)
from simsopt_jax_adapters.objectives import (
    make_finite_build_stage_two_objective,
    make_force_stage_two_length_penalty,
    make_force_stage_two_objective,
)

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "jax" / "3_Advanced"
_COMPILE_LOGGER = "jax._src.interpreters.pxla"
_COMPILE_PREFIX = "Compiling "
_COMPILED_PROGRAM = re.compile(r"^Compiling jit\(([^)]*)\)")
# ``TraceableScalarProblem``/``TraceableParametricScalarProblem`` name every
# objective value/gradient executable they build after the jaxpr they close
# over, so the compile log tells stage graphs apart from device plumbing.
_OBJECTIVE_PROGRAM = "value_and_grad_from_jaxpr"
# ``main()`` runs both examples with this budget in bounded mode.
_BOUNDED_STEPS = 3
# The finite-build claim is ONE DISTINCT objective program, asserted below as
# ``len(set(objective_graphs)) == 1``: the shipped solve republishes at
# ``PUBLISHED_OBJECTIVE_SCALE`` through the solve's own graph, with the scale as
# a device operand, rather than tracing a second graph with the scale baked in.
#
# Re-anchored 2026-09-20 for the finite-build example, from 2 to 1.  What was
# measured, on this machine (CPU, fp64, OMP_NUM_THREADS=4), one bounded solve
# per process:
#   * the shipped solve compiles exactly ONE ``value_and_grad_from_jaxpr``,
#     and it compiles AFTER the fused programs -- the ordered log ends
#     ... lbfgs_private_initial_state_solver, lbfgsb_fused_stepwise,
#     value_and_grad_from_jaxpr, function_from_jaxpr (the diagnostics
#     republication);
#   * the same is true of the PRE-WAVE example source driven through the
#     pre-wave call site, both alone and with the whole file running
#     (1 failed, 8 passed, same ``assert 1 == 2``), and
#     ``src/simsopt_jax/solve`` and ``src/simsopt_jax/geo/optimizers`` are
#     byte-identical to 85051efe9.  So the retired value 2 was already stale
#     at that commit; it is not a change of this example or of its solver.
#
# RETIRED 2026-09-20: a second pin, ``_FINITEBUILD_OBJECTIVE_GRAPHS = 1`` on the
# EXECUTABLE count ``len(objective_graphs)``, carried an in-file admission that
# no mutation could make it fail.  It was measured twice, by injecting one extra
# ``jax.jit`` wrapper over the solve's own computation -- once before the solve,
# once after it over a fresh copy of the initial parameters.  Both injections
# logged executables=2 AND distinct=2: JAX's compile line carries the
# per-argument sharding annotation, which differs between the first and any
# later compile of the same program, so a second executable always moves the
# distinct-signature assertion as well.  The count assertion could therefore
# only fail where the assertion above it had already failed -- a duplicate, not
# a guard -- and is gone.  The distinct-signature assertion IS falsifiable, and
# those same two injections are what falsify it.  Measurements:
# .artifacts/official-mirror-closure-20260919/fix-wave-4/index-bands/proofs/
# old_behaviour_objective_graph_count.py and ..._v2.py.
#
# The retired executable counts 2 (the private BFGS solver's own jitted copy for
# its entry evaluation) and, before it, 3 (the ``serial_solve_jax`` routing this
# example left in ead83eaef) belonged to call paths the example no longer takes.
#
# What carries the fused-device-lane evidence, with a demonstrated failure mode,
# is tests/integration/test_jax_stage_two_finitebuild_strict_transfer.py: zero
# ``advance`` and zero ``callback`` host observations against a positive
# ``final_result`` control, and a ledger that does not move when the step budget
# doubles.
# a1356ceb7 switched coil-forces ``STAGE_DRIVER`` to SciPy L-BFGS-B (no
# fused-solver copy of the objective) and added native's 5-epsilon Taylor
# vmap.  The unbatched float64[47] graph is still one program reused across
# both length-weight stages; the float64[5,47] graph is the Taylor check,
# not a stage-specific objective.  JAX's compile log now includes shapes, so
# those two signatures are distinct strings.  Manual bisect of this test:
# only a1356ceb7 and 2b14e7d02 (maxls) touch coil_forces.py after the pin
# was introduced in 5b50d6390; 2b14e7d02 does not add a graph.
_COIL_FORCES_OBJECTIVE_GRAPHS = 2


def _example(name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        f"jax_example_{name}",
        EXAMPLES / f"{name}.py",
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class _CompilationRecorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.compilations: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith(_COMPILE_PREFIX):
            self.compilations.append(message)


@contextmanager
def _recorded_compilations():
    recorder = _CompilationRecorder()
    logger = logging.getLogger(_COMPILE_LOGGER)
    logger.addHandler(recorder)
    try:
        with jax.log_compiles():
            yield recorder.compilations
    finally:
        logger.removeHandler(recorder)


def _objective_graph_compilations(compilations: list[str]) -> list[str]:
    """Return the objective value/gradient graphs XLA compiled, in log order."""
    programs = []
    for message in compilations:
        match = _COMPILED_PROGRAM.match(message)
        if match is not None and match.group(1) == _OBJECTIVE_PROGRAM:
            programs.append(message)
    return programs


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
    example = _example("stage_two_optimization_finitebuild")
    field, flux, config = example._build_problem("bounded")
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
        objective_parameter=_device_scalar(example.SOLVE_OBJECTIVE_SCALE),
        x=parameters,
    )
    return parameters, problem, objective


def _force_state():
    example = _example("coil_forces")
    (
        field,
        flux,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
    ) = example._build_problem("bounded")
    stage_config = example._stage_config()
    force_config = example._force_config()

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
    return example, field, stage_config, build, weighted_objective, parameters


def test_finitebuild_publishes_the_fresh_unscaled_gradient_bit_for_bit() -> None:
    parameters, problem, objective = _finite_build_state()
    problem.value_and_grad(parameters)
    problem.set_objective_parameter(_device_scalar(1.0))
    published_value, published_gradient = problem.value_and_grad(parameters)

    reference = TraceableScalarProblem(objective, parameters)
    reference_value, reference_gradient = reference.value_and_grad(parameters)

    assert _bitwise_equal(published_value, reference_value)
    assert _bitwise_equal(published_gradient, reference_gradient)


def test_finitebuild_unscaled_publication_compiles_no_second_graph() -> None:
    parameters, problem, objective = _finite_build_state()
    problem.value_and_grad(parameters)

    with _recorded_compilations() as compilations:
        replaced_publication = TraceableScalarProblem(objective, parameters)
        jax.block_until_ready(replaced_publication.value_and_grad(parameters))
        replaced_compilations = len(compilations)
        problem.set_objective_parameter(_device_scalar(1.0))
        jax.block_until_ready(problem.value_and_grad(parameters))

    assert replaced_compilations > 0
    assert len(compilations) == replaced_compilations


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
    example, field, stage_config, build, weighted_objective, parameters = _force_state()
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    weighted_program = jax.jit(jax.value_and_grad(weighted_objective, argnums=0))

    for weight in (example.FIRST_LENGTH_WEIGHT, example.SECOND_LENGTH_WEIGHT):
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
    example, field, stage_config, build, weighted_objective, parameters = _force_state()
    extended = parameters * 2.0
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    weighted_program = jax.jit(jax.value_and_grad(weighted_objective, argnums=0))

    for weight in (example.FIRST_LENGTH_WEIGHT, example.SECOND_LENGTH_WEIGHT):
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
    example, field, stage_config, _build, weighted_objective, parameters = (
        _force_state()
    )
    extended = parameters * 2.0
    first_weight = example.FIRST_LENGTH_WEIGHT
    second_weight = example.SECOND_LENGTH_WEIGHT
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


def test_coil_forces_second_stage_compiles_no_second_graph() -> None:
    example, _field, stage_config, build, weighted_objective, parameters = (
        _force_state()
    )
    problem = TraceableParametricScalarProblem(
        objective_fn=weighted_objective,
        objective_parameter=_device_scalar(example.FIRST_LENGTH_WEIGHT),
        x=parameters,
    )
    problem.value_and_grad(parameters)

    with _recorded_compilations() as compilations:
        replaced_stage = TraceableScalarProblem(
            build(
                replace(
                    stage_config,
                    length_weight=example.SECOND_LENGTH_WEIGHT,
                )
            ),
            parameters,
        )
        jax.block_until_ready(replaced_stage.value_and_grad(parameters))
        replaced_compilations = len(compilations)
        problem.set_objective_parameter(_device_scalar(example.SECOND_LENGTH_WEIGHT))
        jax.block_until_ready(problem.value_and_grad(parameters))

    assert replaced_compilations > 0
    assert len(compilations) == replaced_compilations


# The published values below are regression pins taken from one live bounded
# run on this branch, not independent oracles: they fail when the example's
# numbers move, and say nothing about whether the physics is right.
#
# Re-anchored 2026-09-20 from one bounded device-mode solve on this machine
# (CPU, fp64, OMP_NUM_THREADS=4).  The retired triple
# (0.021277039189432682, 0.018102344118240837, 0.08248454829861396) was
# already unreachable at 85051efe9 -- the compile assertion above fails first,
# so no run had compared them for some time.
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
# Re-baselined 2026-09-29 (PLAN.md amendment 6 cleanups): the parity package
# moved the device endpoint by 3.4e-12 relative. Measured per commit with the
# same test: the L-BFGS-B fix (the unconstrained fast path searches along
# SciPy's d = z - x) moved the objective 3.30e-12 relative, and device_one
# carrying no derivative path then moved it a further 6.9e-17 absolute; the
# CurveLength mean and the arclength VJP restores left it unchanged.


def test_finitebuild_example_solve_publishes_one_objective_graph(tmp_path) -> None:
    """The shipped solve reuses one objective graph and lands where it did.

    ``solve()`` drives the optimizer and then republishes at
    ``PUBLISHED_OBJECTIVE_SCALE``.  The scale is a device operand of the solve's
    graph, so every objective executable the solve compiles carries the same
    signature; baking the scale in would need a second problem, whose graph
    carries one operand fewer and so shows up as a second distinct signature.
    That distinct-signature count is the claim.  The executable-count pin that
    used to sit beside it was retired as a duplicate of it; see the note above.
    """
    example = _example("stage_two_optimization_finitebuild")

    with _recorded_compilations() as compilations:
        # The claim below -- and the executable budget and the endpoint pins --
        # are about the fused device solver, which is now the example's opt-in
        # performance mode rather than its default.
        result = example.solve(
            tmp_path,
            _BOUNDED_STEPS,
            "bounded",
            driver=example.FINITE_BUILD_DEVICE_DRIVER,
        )

    objective_graphs = _objective_graph_compilations(compilations)
    assert len(set(objective_graphs)) == 1, (
        "the shipped solve compiled objective graphs at "
        f"{len(set(objective_graphs))} distinct signatures; the unscaled "
        "republication must re-enter the solve's graph with the scale as a "
        "device operand rather than trace one with the scale baked in"
    )
    assert result.status == "ok"
    assert result.observables["solver_iterations"] == _BOUNDED_STEPS
    np.testing.assert_allclose(
        result.observables["final_objective"],
        0.005313081455091724,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.observables["squared_flux"],
        0.0030583480385486506,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.observables["minimum_clearance"],
        0.0946712348814373,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_coil_forces_example_solve_publishes_one_objective_graph(tmp_path) -> None:
    """Both shipped stages share one unbatched objective graph.

    ``solve()`` runs the first stage, swaps the device length weight, and runs
    the second; a stage-specific objective would compile a further unbatched
    graph. a1356ceb7's SciPy driver dropped the fused-solver copy, and its
    native Taylor vmap compiles the same jaxpr at float64[5,47].
    """
    example = _example("coil_forces")

    with _recorded_compilations() as compilations:
        result = example.solve(tmp_path, _BOUNDED_STEPS, "bounded")

    objective_graphs = _objective_graph_compilations(compilations)
    assert len(objective_graphs) == _COIL_FORCES_OBJECTIVE_GRAPHS, (
        "the shipped solve compiled a different number of objective graphs "
        f"({len(objective_graphs)}, {len(set(objective_graphs))} distinct); "
        "SciPy L-BFGS-B plus the 5-epsilon Taylor vmap is two signatures, "
        "and both length-weight stages must share the unbatched one"
    )
    assert result.status == "ok"
    assert result.observables["solver_iterations"] == (_BOUNDED_STEPS, _BOUNDED_STEPS)
    # Regression pin retaken 2026-09-13 after the mirror adopted native's
    # post-Taylor start state and SciPy L-BFGS-B at native policy (ftol=gtol=1e-15).
    np.testing.assert_allclose(
        result.observables["final_objective"],
        0.003424195336484035,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.observables["squared_flux"],
        0.0033617810994737273,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.observables["vacuum_energy"],
        0.6148225859961913,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_force_stage_two_length_penalty_rejects_a_static_length_weight() -> None:
    _example_module, field, stage_config, _build, _objective, _parameters = (
        _force_state()
    )

    with pytest.raises(ValueError, match="length_weight must be zero"):
        make_force_stage_two_length_penalty(
            field,
            replace(stage_config, length_weight=1.0e-3),
        )
