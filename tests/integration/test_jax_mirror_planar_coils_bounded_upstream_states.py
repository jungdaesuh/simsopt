"""Same-state proof for the planar-coils lanes at UPSTREAM's recorded BOUNDED states.

At the bounded scale the planar-coils end objective is judged by a band from
upstream's own nine one-ulp draws (``examples/jax/parity/official_scatter_contracts.py``).
This file asks the question beneath that band: at every state upstream recorded
in the bounded scatter record -- each of the nine draws' start, stage-1 end and
stage-2 end, read from the tracked record, none written here -- do the native
lane's objective (``NativePlanarEvaluator``) and the JAX lane's objective
program (``standard_stage_two_state``, the entry the JAX lane's own solve
evaluates) take the same VALUE and the same GRADIENT, each at the stage's own
length weight?  Upstream's recorded gradients after the start state are stale
(the persistent ``CurvePlanarFourier`` Jacobian cache) and cannot be a
reference, so the two lanes of this branch are compared with each other.

"The same" means: within the DERIVED rounding bound of the formula both lanes
evaluate (``planar_stage_two_roundoff.planar_objective_bound``), doubled for two
implementations -- worst-order sums, per-rounding charges, the chain rule's path
sums for the gradient, per-element geometry counts taken from the longer of the
two lanes' forms.  No tolerance here is fitted to a measured gap.

That bound replaces a scalar relative budget whose scale was the objective's
value and the gradient's largest entry, which fails at these states for a
reason the scale cannot see: the length penalty ``0.5 w (L - L0)^2`` sits at
``L - L0 ~ 4e-3`` on ``L ~ 10.4``, so one ulp of ``L`` moves its value by
``2.5e3`` ulps of itself, and at the stage-2 ends the flux and length gradients
cancel to a total gradient far below either term's.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import jax
import numpy as np
import pytest
from examples.jax.parity.cases.native_stage_two_optimization_planar_coils import (
    NativePlanarEvaluator,
    _build_geometry,
    _mapping_float,
    _mapping_int,
    _scale_configuration,
    build_native_evaluator,
)
from examples.jax.parity.official_reference import load_upstream_scatter
from simsopt_jax.examples.stage_two_standard import standard_stage_two_state
from simsopt_jax.objectives import StageTwoObjectiveConfig
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

# venv site-packages/tests shadows the repo tests package, so the helpers are
# imported as top-level modules from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from forward_roundoff_bound import cross_implementation_bounds
from planar_stage_two_roundoff import planar_objective_bound

CASE_ID = "native-stage-two-optimization-planar-coils"
SCALE = "bounded"
CONFIGURATION = _scale_configuration(SCALE)
SCATTER = load_upstream_scatter(CASE_ID, SCALE)
#: (record parameter key, stage whose length weight the objective carries there)
RECORDED_STATES = (
    ("initial:parameters", "first_length_weight"),
    ("first:parameters", "first_length_weight"),
    ("final:parameters", "second_length_weight"),
)
CASES = tuple(
    (run.k, parameter_key, weight_name)
    for run in SCATTER.runs
    for parameter_key, weight_name in RECORDED_STATES
)


def _regularization_config(
    configuration: Mapping[str, object],
) -> StageTwoObjectiveConfig:
    """The regularization the JAX lane builds from the same configuration (length weight explicit)."""
    return StageTwoObjectiveConfig(
        num_base_curves=_mapping_int(configuration, "num_base_curves"),
        length_target=_mapping_float(configuration, "length_target"),
        length_target_mode="identity",
        curve_curve_minimum_distance=_mapping_float(
            configuration, "curve_curve_threshold"
        ),
        curve_curve_weight=_mapping_float(configuration, "curve_curve_weight"),
        curve_surface_minimum_distance=_mapping_float(
            configuration, "curve_surface_threshold"
        ),
        curve_surface_weight=_mapping_float(configuration, "curve_surface_weight"),
        curvature_threshold=_mapping_float(configuration, "curvature_threshold"),
        curvature_weight=_mapping_float(configuration, "curvature_weight"),
        mean_squared_curvature_threshold=_mapping_float(
            configuration, "mean_squared_curvature_threshold"
        ),
        mean_squared_curvature_target_mode="identity",
        mean_squared_curvature_weight=_mapping_float(
            configuration, "mean_squared_curvature_weight"
        ),
        linking_number_weight=_mapping_float(configuration, "linking_number_weight"),
    )


@pytest.fixture(scope="module")
def evaluator() -> NativePlanarEvaluator:
    """The native lane's own objective assembly at the bounded scale upstream's record ran."""
    return build_native_evaluator(CONFIGURATION)


def test_the_bounded_record_starts_at_the_native_lanes_start_state(
    evaluator: NativePlanarEvaluator,
) -> None:
    """Upstream's unperturbed start is the lane's construction, bit for bit."""
    np.testing.assert_array_equal(
        SCATTER.run(0).value("initial:parameters"),
        np.asarray(evaluator.field.x, dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("k", "parameter_key", "weight_name"),
    CASES,
    ids=[f"k{k}-{key.split(':')[0]}" for k, key, _weight in CASES],
)
def test_native_and_jax_objective_and_gradient_agree_at_upstreams_bounded_state(
    evaluator: NativePlanarEvaluator,
    k: int,
    parameter_key: str,
    weight_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    parameters = SCATTER.run(k).value(parameter_key)
    length_weight = _mapping_float(CONFIGURATION, weight_name)

    objective = evaluator.weighted(length_weight)
    objective.x = parameters
    native_value = float(objective.J())
    native_gradient = np.asarray(objective.dJ(), dtype=np.float64)

    surface, _base_curves, coils = _build_geometry(CONFIGURATION)
    field = BiotSavartJAX(coils)
    state = standard_stage_two_state(
        field=field,
        flux_spec=SquaredFluxJAX(surface, field).fixed_surface_flux_spec(),
        surface_gamma=np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)),
        surface_normal=np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)),
        parameters=parameters,
        regularization_config=_regularization_config(CONFIGURATION),
        length_weight=np.asarray(length_weight, dtype=np.float64),
    )
    jax_value = float(np.asarray(jax.device_get(state.objective)))
    jax_gradient = np.asarray(
        jax.device_get(state.objective_gradient), dtype=np.float64
    )

    model = planar_objective_bound(evaluator, parameters, length_weight).total
    value_bound, gradient_bound = cross_implementation_bounds(model)
    value_bound = float(value_bound)

    # The bound's own float64 evaluation of the formula is a third
    # implementation: if it strayed from the native lane, the bound would be a
    # bound on some other formula.
    assert abs(float(model.v) - native_value) <= value_bound
    assert np.all(np.abs(model.d - native_gradient) <= gradient_bound)

    value_difference = abs(jax_value - native_value)
    assert value_difference <= value_bound, (
        f"k={k} {parameter_key}: JAX objective {jax_value!r} against native "
        f"{native_value!r}: {value_difference:.3e} over the derived bound "
        f"{value_bound:.3e}"
    )
    assert jax_gradient.shape == native_gradient.shape
    gradient_difference = np.abs(jax_gradient - native_gradient)
    worst = int(np.argmax(gradient_difference / gradient_bound))
    assert np.all(gradient_difference <= gradient_bound), (
        f"k={k} {parameter_key}: JAX gradient component {worst} differs from "
        f"native by {gradient_difference[worst]:.3e}, over the derived bound "
        f"{gradient_bound[worst]:.3e}"
    )
