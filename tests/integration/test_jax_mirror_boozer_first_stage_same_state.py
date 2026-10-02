"""Same-state proof for the ``2_Intermediate/boozer.py`` FIRST stage: objective and gradient at identical x.

The first stage's end point is path-dependent and informational; what decides is whether the
native library and the JAX mirror evaluate the function that stage minimizes -- upstream's
first-stage penalty, its value AND its gradient -- to within rounding at identical states: the
fitted start ``x0`` and ``first``, the end of a live native first stage (``BoozerSurface`` and
SciPy L-BFGS-B through ``run_boozer_lbfgs_stage``, upstream's algorithm, run for
:data:`FIRST_STAGE_MAXITER` of upstream's 300 iterations to keep the test small), at the reduced
``mpol = ntor = 2`` and the official ``mpol = ntor = 5`` resolutions.  Native:
``BoozerSurface.boozer_penalty_constraints_vectorized(x, 1, cw, optimize_G=True,
weight_inv_modB=True)``.  JAX: the value-and-gradient program ``_scalar_value_and_grad`` of
``BoozerSurfaceJAX._make_penalty_objective_with(True, True, cw)``, the function the mirror's
L-BFGS-B minimizes, compiled and run on the parity lane's device.

"Within rounding" means twice the DERIVED bound of ``tests/boozer_first_stage_roundoff.py``
(over ``tests/forward_roundoff_bound.py``), componentwise for the gradient; the model's own
float64 evaluation must also sit within one bound of the native lane.  Refusals R1-R7 are
errors, never a wider tolerance, and the negative controls NC1-NC3 must FAIL the same check.
The CPU pair is a derived first-order bound.  The GPU pair is an ENGINEERING ERROR ENVELOPE in
one constant: XLA compiles the lane's ``1/sqrt`` to ``rsqrt``, charged 3 u from CUDA 12.9's
1-ulp table entry, which NVIDIA states is not guaranteed.
"""

from __future__ import annotations

from jax_test_support import (
    fixture_jax_runtime_guard,  # noqa: F401
    fixture_parity_lane,  # noqa: F401
    enable_strict_parity_backend,
    parity_default_device,
    parity_device,
)

import hashlib
import math
import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import mpmath
import numpy as np
import pytest
import simsoptpp
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.field.coil import Coil, ScaledCurrent
from simsopt.geo import Area, BoozerSurface, SurfaceXYZTensorFourier
from simsopt_jax.backend.runtime import get_backend_policy
from simsopt_jax.core._device_scalars import two_pi
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_CONSTRAINT_WEIGHT,
    OFFICIAL_INITIAL_IOTA,
    OFFICIAL_LBFGS_MAXITER,
    OFFICIAL_LS_MAXITER,
    OFFICIAL_SOLVER_TOLERANCE,
    OFFICIAL_SURFACE_DISTANCE,
    OFFICIAL_SURFACE_RESOLUTION,
    BoozerStageState,
    boozer_official_options,
    run_boozer_lbfgs_stage,
)
from simsopt_jax.geo.optimizers.private._common import _scalar_value_and_grad
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import (
    BoozerSurfaceJAX,
    _geometry_from_surface_dofs,
    _surface_sample_z,
)

import forward_roundoff_bound as fb
from boozer_first_stage_roundoff import (
    TRIG_ULP,
    TWO_PI,
    CoilSet,
    FirstStageBound,
    SurfaceGrid,
    biot_savart_field,
    coil_set,
    enveloped_inverse_cube,
    first_stage_objective_bound,
    flat_points,
    local_seed,
    native_inverse_cube,
    surface_elements,
    unwrapped_coil_curve,
    used_trig_arguments,
)

SCALES: tuple[ExecutionScale, ...] = ("bounded", "native_default")
#: Surface resolution ``mpol = ntor`` per scale: the reduced one and upstream's.
SURFACE_RESOLUTION: dict[ExecutionScale, int] = {
    "bounded": 2,
    "native_default": OFFICIAL_SURFACE_RESOLUTION,
}
STATES = ("x0", "first")
CONTROL_STATES = STATES
#: Iterations of the live native first stage behind the ``first`` state: a tenth of upstream's
#: ``OFFICIAL_LBFGS_MAXITER``, which keeps the state off the start without a 30 s solve.
FIRST_STAGE_MAXITER = OFFICIAL_LBFGS_MAXITER // 10
CONTROLS = ("NC1", "NC2", "NC3")
#: The build whose library constants B2 established. Checked only when a campaign reproduction is
#: requested (PLAN.md amendment 8, F3): any other build still has its trig constants re-established
#: by R4 and its precision and fast-math assumptions checked by R5.
CAMPAIGN_REPRODUCTION_ENV = "SIMSOPT_PARITY_CAMPAIGN_REPRODUCTION"
SIMSOPTPP_SHA256 = "0149cc25cefa1d90d01fc307e7a70975cd13c424ff7a35171e1ef8d945741453"
JAX_VERSION = "0.10.0"
#: HLO opcodes that evaluate a transcendental or a root; only the four B2 established may run (R2).
_TRANSCENDENTAL_OPCODES = frozenset(
    {
        "sqrt",
        "rsqrt",
        "cbrt",
        "sine",
        "cosine",
        "tan",
        "atan2",
        "exponential",
        "exponential-minus-one",
        "log",
        "log-plus-one",
        "logistic",
        "power",
        "tanh",
        "erf",
    }
)
_ESTABLISHED_OPCODES = frozenset({"sqrt", "rsqrt", "sine", "cosine"})
#: A negative control plants ten times the doubled bound (NC1, NC2).
_PLANT_FACTOR = 20.0
#: NC3: every JAX coil current scaled by this factor.
_CURRENT_PLANT = 1.0 + 1.0e-8
U = float(np.finfo(np.float64).eps) / 2.0


# ---------------------------------------------------------------------------
# The two lanes at one state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeState:
    """The construction and the native library's objective at one state."""

    configuration: dict[str, object]
    native_field: BiotSavart
    surface: object
    area: Area
    target_label: float
    constraint_weight: float
    x: np.ndarray
    value: float
    gradient: np.ndarray


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    """The official ``boozer.py`` settings at one surface resolution."""
    return {
        "mpol": SURFACE_RESOLUTION[scale],
        "ntor": SURFACE_RESOLUTION[scale],
        "jax_bfgs_maxiter": OFFICIAL_LBFGS_MAXITER,
        "jax_ls_maxiter": OFFICIAL_LS_MAXITER,
        "solver_tolerance": OFFICIAL_SOLVER_TOLERANCE,
        "constraint_weight": OFFICIAL_CONSTRAINT_WEIGHT,
        "initial_iota": OFFICIAL_INITIAL_IOTA,
        "surface_distance": OFFICIAL_SURFACE_DISTANCE,
    }


def _problem(configuration: dict[str, object]):
    """NCSX coils and the fitted tensor-Fourier start surface, as ``boozer.py`` builds them."""
    _, base_currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    field = BiotSavart(native_field.coils)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    mpol = int(configuration["mpol"])
    ntor = int(configuration["ntor"])
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False),
    )
    surface.fit_to_curve(
        magnetic_axis, float(configuration["surface_distance"]), flip_theta=True
    )
    return magnetic_axis, native_field, field, surface, float(G0)


@lru_cache(maxsize=None)
def _first_stage_end(scale: ExecutionScale) -> tuple[float, ...]:
    """The live native first stage's end state ``(surface dofs, iota, G)``, on fresh objects."""
    configuration = _scale_configuration(scale)
    _axis, native_field, _field, surface, G0 = _problem(configuration)
    area = Area(surface)
    outcome = run_boozer_lbfgs_stage(
        BoozerSurface(native_field, surface, area, float(area.J())),
        BoozerStageState(
            surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
            iota=float(configuration["initial_iota"]),
            G=G0,
        ),
        tol=float(configuration["solver_tolerance"]),
        maxiter=FIRST_STAGE_MAXITER,
        constraint_weight=float(configuration["constraint_weight"]),
    )
    return tuple(
        np.concatenate(
            (outcome.state.surface_dofs, [outcome.state.iota, outcome.state.G])
        ).tolist()
    )


def _state_vector(scale: ExecutionScale, state: str, surface, G0: float) -> np.ndarray:
    if state == "x0":
        initial_iota = float(_scale_configuration(scale)["initial_iota"])
        return np.concatenate((surface.get_dofs(), [initial_iota, G0]))
    return np.asarray(_first_stage_end(scale), dtype=np.float64)


def _native_objective(native: NativeState, x: np.ndarray) -> tuple[float, np.ndarray]:
    value, gradient = BoozerSurface(
        native.native_field, native.surface, native.area, native.target_label
    ).boozer_penalty_constraints_vectorized(
        np.asarray(x, dtype=np.float64),
        derivatives=1,
        constraint_weight=native.constraint_weight,
        optimize_G=True,
        weight_inv_modB=True,
    )
    return float(value), np.asarray(gradient, dtype=np.float64)


def native_state(scale: ExecutionScale, state: str) -> NativeState:
    """A fresh construction; the label target is the start surface's area A0, as the workflow sets it."""
    configuration = _scale_configuration(scale)
    _axis, native_field, _field, surface, G0 = _problem(configuration)
    area = Area(surface)
    partial = NativeState(
        configuration,
        native_field,
        surface,
        area,
        float(area.J()),
        float(configuration["constraint_weight"]),
        _state_vector(scale, state, surface, G0),
        math.nan,
        np.empty(0),
    )
    value, gradient = _native_objective(partial, partial.x)
    return replace(partial, value=value, gradient=gradient)


@lru_cache(maxsize=None)
def _bound(scale: ExecutionScale, state: str, lane: str) -> FirstStageBound:
    """The derived bound at one state (the lane only selects its trig constant)."""
    native = native_state(scale, state)
    return first_stage_objective_bound(
        SurfaceGrid.from_surface(native.surface),
        coil_set(native.native_field.coils, TRIG_ULP[lane]),
        native.x,
        native.target_label,
        native.constraint_weight,
        TRIG_ULP[lane],
    )


@dataclass(frozen=True)
class JaxEvaluation:
    field: BiotSavartJAX
    solver: BoozerSurfaceJAX
    hlo: str
    value: float
    gradient: np.ndarray


def jax_objective(
    native: NativeState,
    lane: str,
    x: np.ndarray,
    *,
    coils=None,
    target_label: float | None = None,
) -> JaxEvaluation:
    """The lane's first-stage value-and-gradient program, compiled and run on the lane device."""
    configuration = native.configuration
    field = BiotSavartJAX(native.native_field.coils if coils is None else coils)
    solver = BoozerSurfaceJAX(
        field,
        native.surface,
        native.area,
        native.target_label if target_label is None else target_label,
        constraint_weight=native.constraint_weight,
        options=boozer_official_options(
            rough_maxiter=int(configuration["jax_bfgs_maxiter"]),
            ls_maxiter=int(configuration["jax_ls_maxiter"]),
            tolerance=float(configuration["solver_tolerance"]),
        ),
    )
    program = _scalar_value_and_grad(
        solver._make_penalty_objective_with(True, True, native.constraint_weight)
    )
    x_device = jax.device_put(np.asarray(x, dtype=np.float64), parity_device(lane))
    compiled = program.lower(x_device).compile()
    hlo = compiled.as_text()
    _refuse_untrusted_program(hlo)
    value, gradient = jax.device_get(compiled(x_device))
    return JaxEvaluation(
        field, solver, hlo, float(value), np.asarray(gradient, dtype=np.float64)
    )


def _refuse_untrusted_program(hlo: str) -> None:
    """R2: only the transcendental ops whose constants B2 established may run."""
    opcodes = set(re.findall(r"\s([a-z][a-z0-9-]*)\(", hlo))
    unexpected = (opcodes & _TRANSCENDENTAL_OPCODES) - _ESTABLISHED_OPCODES
    if unexpected:
        raise ValueError(f"R2: the compiled JAX program evaluates {sorted(unexpected)}")
    for target in re.findall(r'custom_call_target="([^"]+)"', hlo):
        if re.search(r"sin|cos|exp|log|pow|tan|erf|sqrt", target, re.IGNORECASE):
            raise ValueError(f"R2: the compiled JAX program calls {target!r}")


def _check(
    model: fb.Bounded, native: NativeState, value: float, gradient: np.ndarray
) -> tuple[bool, float, float]:
    """The B2 comparison: within twice the bound, componentwise; and the worst ratios."""
    value_bound, gradient_bound = fb.cross_implementation_bounds(model)
    value_ratio = abs(value - native.value) / float(value_bound)
    gradient_ratio = float(np.max(np.abs(gradient - native.gradient) / gradient_bound))
    return value_ratio <= 1.0 and gradient_ratio <= 1.0, value_ratio, gradient_ratio


# ---------------------------------------------------------------------------
# Refusals that concern the shared inputs (R6, R7) and the model's elements
# ---------------------------------------------------------------------------


def _assert_axis_z_is_exactly_zero(
    native: NativeState, evaluation: JaxEvaluation
) -> None:
    """R6: z(phi=0, theta=0) and its gradient row vanish in both lanes (stellarator symmetry)."""
    assert native.surface.gamma()[0, 0, 2] == 0.0
    assert not np.any(native.surface.dgamma_by_dcoeff()[0, 0, 2, :])
    arguments = evaluation.solver._traceable_surface_runtime_args()

    def axis_z(surface_dofs):
        geometry = _geometry_from_surface_dofs(
            surface_dofs,
            quadpoints_phi=arguments["quadpoints_phi"],
            quadpoints_theta=arguments["quadpoints_theta"],
            mpol=arguments["mpol"],
            ntor=arguments["ntor"],
            nfp=arguments["nfp"],
            stellsym=arguments["stellsym"],
            scatter_indices=arguments["scatter_indices"],
            surface_kind=arguments["surface_kind"],
            clamped_dims=arguments["clamped_dims"],
        )
        return _surface_sample_z(geometry.gamma)

    value, row = jax.device_get(jax.value_and_grad(axis_z)(jnp.asarray(native.x[:-2])))
    assert float(value) == 0.0
    assert not np.any(np.asarray(row))


def _assert_shared_constants(native: NativeState, evaluation: JaxEvaluation) -> None:
    """R7: coil dofs, currents, rotmat, quadpoints, A0 and cw are the same float64 in both lanes."""
    solver = evaluation.solver
    assert solver.targetlabel == native.target_label
    assert float(solver.constraint_weight) == native.constraint_weight
    np.testing.assert_array_equal(solver.quadpoints_phi, native.surface.quadpoints_phi)
    np.testing.assert_array_equal(
        solver.quadpoints_theta, native.surface.quadpoints_theta
    )
    coils = native.native_field.coils
    specs = evaluation.field.coil_specs()
    assert len(specs) == len(coils)
    for coil, spec in zip(coils, specs, strict=True):
        base, rotmat = unwrapped_coil_curve(coil.curve)
        np.testing.assert_array_equal(np.asarray(spec.curve.dofs), base.local_full_x)
        np.testing.assert_array_equal(
            np.asarray(spec.curve.quadpoints), base.quadpoints
        )
        assert spec.symmetry.has_rotation == (rotmat is not None)
        if rotmat is not None:
            np.testing.assert_array_equal(np.asarray(spec.symmetry.rotmat), rotmat)
    currents = np.empty(len(coils))
    for (_g, _gd, group_currents), indices in zip(
        solver.coil_set_spec.field_inputs(),
        solver.coil_set_spec.coil_index_lists(),
        strict=True,
    ):
        currents[list(indices)] = np.asarray(group_currents)
    np.testing.assert_array_equal(
        currents, [coil.current.get_value() for coil in coils]
    )


def _assert_within(label: str, computed, model, error_units) -> None:
    """A lane's element lies within the model element's derived error."""
    excess = np.abs(np.asarray(computed) - model) - np.asarray(error_units) * U
    assert np.all(excess <= 0.0), (
        f"{label}: a lane's element leaves the derived element error by "
        f"{float(np.max(excess)):.3e}"
    )


def _assert_elements_bound_the_lanes(
    native: NativeState, bound: FirstStageBound, coils: CoilSet, evaluation
) -> None:
    """The injected elements are transcribed right: each lane's arrays lie inside them."""
    surface = native.surface
    size = native.x.size - 2
    for name, native_value, native_jacobian in (
        ("gamma", surface.gamma(), surface.dgamma_by_dcoeff()),
        ("gammadash1", surface.gammadash1(), surface.dgammadash1_by_dcoeff()),
        ("gammadash2", surface.gammadash2(), surface.dgammadash2_by_dcoeff()),
    ):
        element = getattr(bound.surface, name)
        _assert_within(f"native {name}", native_value, element.v, element.e)
        _assert_within(
            f"native d{name}/dc",
            native_jacobian,
            element.d[..., :size],
            element.Ed[..., :size],
        )
    for index, coil in enumerate(native.native_field.coils):
        _assert_within(
            "native coil gamma",
            coil.curve.gamma(),
            coils.gamma[index],
            coils.gamma_error[index],
        )
        _assert_within(
            "native coil gammadash",
            coil.curve.gammadash(),
            coils.gammadash[index],
            coils.gammadash_error[index],
        )
    spec = evaluation.solver.coil_set_spec
    for (gammas, gammadashs, _currents), indices in zip(
        spec.field_inputs(), spec.coil_index_lists(), strict=True
    ):
        rows = list(indices)
        _assert_within(
            "JAX coil gamma", gammas, coils.gamma[rows], coils.gamma_error[rows]
        )
        _assert_within(
            "JAX coil gammadash",
            gammadashs,
            coils.gammadash[rows],
            coils.gammadash_error[rows],
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compose_equals_the_direct_trace_at_one_bounded_point() -> None:
    """``fb.compose`` of the per-point Biot-Savart trace equals the direct trace (to 1e-12)."""
    native = native_state("bounded", "x0")
    ulp = TRIG_ULP["cpu"]
    grid = SurfaceGrid.from_surface(native.surface)
    coils = coil_set(native.native_field.coils, ulp)
    elements = surface_elements(grid, native.x[:-2], native.x.size, ulp)
    point = flat_points(elements.gamma)[7:8]
    seed = local_seed(point.v, point.e)
    for inverse_cube in (native_inverse_cube, enveloped_inverse_cube):
        direct = biot_savart_field(point, coils, inverse_cube)
        composed = fb.compose(biot_savart_field(seed, coils, inverse_cube), point)
        np.testing.assert_array_equal(composed.v, direct.v)
        np.testing.assert_allclose(composed.e, direct.e, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(composed.d, direct.d, rtol=1e-12, atol=1e-300)
        for name in ("D", "Ed", "paths"):
            composed_array = getattr(composed, name)
            direct_array = getattr(direct, name)
            if inverse_cube is native_inverse_cube:
                np.testing.assert_allclose(
                    composed_array, direct_array, rtol=1e-12, atol=0.0
                )
            else:
                # After an envelope the chained maximum can only exceed the direct one.
                assert np.all(composed_array >= direct_array * (1.0 - 1e-12)), name


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("scale", SCALES)
def test_first_stage_value_and_gradient_agree_within_the_derived_bound(
    scale: ExecutionScale,
    state: str,
    parity_lane: str,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, parity_lane, precision="fp64")
    native = native_state(scale, state)
    bound = _bound(scale, state, parity_lane)
    model = bound.total
    value_bound, gradient_bound = fb.cross_implementation_bounds(model)

    # The bound's own float64 evaluation is a third implementation: it must sit within one
    # bound of the native lane, or the bound would be a bound on some other formula.
    assert abs(float(model.v) - native.value) <= float(model.value_bound())
    assert np.all(np.abs(model.d - native.gradient) <= model.derivative_bound())

    with parity_default_device(parity_lane):
        evaluation = jax_objective(native, parity_lane, native.x)
        _assert_axis_z_is_exactly_zero(native, evaluation)
    _assert_shared_constants(native, evaluation)
    _assert_elements_bound_the_lanes(
        native,
        bound,
        coil_set(native.native_field.coils, TRIG_ULP[parity_lane]),
        evaluation,
    )

    passed, value_ratio, gradient_ratio = _check(
        model, native, evaluation.value, evaluation.gradient
    )
    print(
        f"B2 {scale} {state} {parity_lane}: value ratio {value_ratio:.3e}, "
        f"gradient ratio {gradient_ratio:.3e}, doubled value bound "
        f"{float(value_bound):.3e} (F {native.value:.6e}), max doubled gradient bound "
        f"{float(np.max(gradient_bound)):.3e} (|g|_inf {np.max(np.abs(native.gradient)):.3e})"
    )
    value_difference = abs(evaluation.value - native.value)
    assert value_difference <= float(value_bound), (
        f"{scale} {state}: JAX first-stage objective {evaluation.value!r} against native "
        f"{native.value!r}: {value_difference:.3e} over twice the derived bound "
        f"{float(value_bound):.3e}"
    )
    gradient_difference = np.abs(evaluation.gradient - native.gradient)
    worst = int(np.argmax(gradient_difference / gradient_bound))
    assert passed, (
        f"{scale} {state}: JAX first-stage gradient component {worst} differs from native "
        f"by {gradient_difference[worst]:.3e}, over twice the derived bound "
        f"{gradient_bound[worst]:.3e}"
    )


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("state", CONTROL_STATES)
@pytest.mark.parametrize("scale", SCALES)
def test_negative_control_fails_the_same_check(
    scale: ExecutionScale,
    state: str,
    control: str,
    parity_lane: str,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """NC1 (value), NC2 (gradient), NC3 (physical): a planted discrepancy must be caught."""
    enable_strict_parity_backend(monkeypatch, request, parity_lane, precision="fp64")
    native = native_state(scale, state)
    model = _bound(scale, state, parity_lane).total
    value_bound = float(model.value_bound())
    gradient_bound = model.derivative_bound()

    with parity_default_device(parity_lane):
        if control == "NC1":
            # JAX at x + h g/|g|: a planted value change of ten times the doubled bound.
            norm = float(np.linalg.norm(native.gradient))
            step = _PLANT_FACTOR * value_bound / norm
            moved = native.x + step * native.gradient / norm
            planted = _native_objective(native, moved)[0] - native.value
            if abs(planted - step * norm) > 0.1 * step * norm:
                raise RuntimeError(
                    f"NC1 refused: the native lane moves by {planted:.3e}, not the planted "
                    f"{step * norm:.3e}"
                )
            evaluation = jax_objective(native, parity_lane, moved)
            assert abs(evaluation.value - native.value) > 2.0 * value_bound
        elif control == "NC2":
            # JAX with target A0 + D: ten times the doubled bound at the most sensitive component.
            native.surface.set_dofs(native.x[:-2])
            label_gradient = np.asarray(
                native.area.dJ_by_dsurfacecoefficients(), dtype=np.float64
            )
            sensitivity = np.abs(label_gradient) / gradient_bound[: label_gradient.size]
            component = int(np.argmax(sensitivity))
            shift = (
                _PLANT_FACTOR
                * gradient_bound[component]
                / (native.constraint_weight * abs(label_gradient[component]))
            )
            target = native.target_label + shift
            planted = (
                native.constraint_weight
                * (target - native.target_label)
                * abs(label_gradient[component])
            )
            if abs(planted - _PLANT_FACTOR * gradient_bound[component]) > 0.1 * planted:
                raise RuntimeError(f"NC2 refused: the realized plant is {planted:.3e}")
            evaluation = jax_objective(
                native, parity_lane, native.x, target_label=target
            )
            difference = abs(
                evaluation.gradient[component] - native.gradient[component]
            )
            assert difference > 2.0 * gradient_bound[component]
        else:
            # JAX coil currents scaled by 1 + 1e-8.
            coils = [
                Coil(coil.curve, ScaledCurrent(coil.current, _CURRENT_PLANT))
                for coil in native.native_field.coils
            ]
            evaluation = jax_objective(native, parity_lane, native.x, coils=coils)
            passed, value_ratio, gradient_ratio = _check(
                model, native, evaluation.value, evaluation.gradient
            )
            assert not passed, (value_ratio, gradient_ratio)
    print(f"B2 {control} {scale} {state} {parity_lane}: caught")


def test_two_pi_is_the_formula_constant_on_the_lane_device(parity_lane: str) -> None:
    """R3: the JAX lane's ``two_pi()`` is ``2*np.pi`` bitwise, eager and compiled."""
    reference = jax.device_put(np.zeros(3), parity_device(parity_lane))
    assert float(two_pi(reference)) == TWO_PI
    assert float(jax.jit(two_pi)(reference)) == TWO_PI


def _worst_library_ulp(args: np.ndarray, values: np.ndarray, cosine: bool) -> float:
    function = mpmath.cos if cosine else mpmath.sin
    worst = 0.0
    for arg, value in zip(args.ravel(), np.asarray(values).ravel(), strict=True):
        exact = function(mpmath.mpf(float(arg)))
        rounded = float(exact)
        if rounded == 0.0:
            assert value == 0.0
            continue
        ulp = float(np.spacing(np.float64(abs(rounded))))
        worst = max(worst, float(abs(mpmath.mpf(float(value)) - exact)) / ulp)
    return worst


def test_trig_libraries_meet_the_registered_constants_at_the_used_arguments(
    parity_lane: str,
) -> None:
    """R4: glibc (the native .so) and the lane device's sin/cos, against mpmath at every used argument."""
    mpmath.mp.dps = 60
    device = parity_device(parity_lane)
    cosine = jax.jit(jnp.cos)
    sine = jax.jit(jnp.sin)
    for scale in SCALES:
        _axis, native_field, _field, surface, _G0 = _problem(
            _scale_configuration(scale)
        )
        base, _rotmat = unwrapped_coil_curve(native_field.coils[0].curve)
        arguments = used_trig_arguments(
            SurfaceGrid.from_surface(surface),
            np.asarray(base.quadpoints, dtype=np.float64),
            int(base.order),
        )
        for name, args in arguments.items():
            for is_cosine, library, device_function in (
                (True, math.cos, cosine),
                (False, math.sin, sine),
            ):
                glibc = np.vectorize(library)(args)
                on_device = np.asarray(device_function(jax.device_put(args, device)))
                assert _worst_library_ulp(args, glibc, is_cosine) <= TRIG_ULP["cpu"], (
                    name
                )
                assert (
                    _worst_library_ulp(args, on_device, is_cosine)
                    <= TRIG_ULP[parity_lane]
                ), name


def test_pinned_builds_and_precision(
    parity_lane: str,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """R5: FP64 and no fast-math XLA flag; the campaign's own builds only when reproducing it.

    With ``SIMSOPT_PARITY_CAMPAIGN_REPRODUCTION=1`` the run claims to reproduce
    the recorded campaign, so it must load the exact simsoptpp and jax/jaxlib
    builds whose constants B2 established, and refuses otherwise. Without it any
    build is tested on its own terms: R4 re-establishes the trig constants for
    the build that runs, and this test checks the numerical assumptions.
    """
    enable_strict_parity_backend(monkeypatch, request, parity_lane, precision="fp64")
    if os.environ.get(CAMPAIGN_REPRODUCTION_ENV) == "1":
        library = Path(simsoptpp.__file__)
        loaded = hashlib.sha256(library.read_bytes()).hexdigest()
        assert loaded == SIMSOPTPP_SHA256, (
            f"{CAMPAIGN_REPRODUCTION_ENV}=1 reproduces the campaign build "
            f"{SIMSOPTPP_SHA256}; {library} is {loaded}"
        )
        assert (jax.__version__, jaxlib.__version__) == (JAX_VERSION, JAX_VERSION), (
            f"{CAMPAIGN_REPRODUCTION_ENV}=1 reproduces jax/jaxlib {JAX_VERSION}"
        )
    assert bool(jax.config.read("jax_enable_x64"))
    policy = get_backend_policy()
    assert policy.resolved_precision == "fp64"
    assert policy.compute_dtype == "float64"
    for token in os.environ.get("XLA_FLAGS", "").split():
        assert "fast_math" not in token or token.endswith("=false"), token
    # The GPU lane's f64 matmuls are cuBLAS DGEMM in IEEE fp64, not its opt-in emulation.
    assert "CUBLAS_EMULATION_STRATEGY" not in os.environ
