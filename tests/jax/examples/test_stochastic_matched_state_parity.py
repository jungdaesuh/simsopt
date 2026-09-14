"""Matched-state evaluator parity for the stochastic stage-two lanes.

Endpoint coordinates of an iteration-capped L-BFGS-B run are not a parity
criterion — the native lane's own endpoint moves under a change of
``OMP_NUM_THREADS`` alone.  What *is* a criterion, and what this file pins, is
the evaluator: handed the same samples and the same coordinates, the native
objective cloned from
``examples/jax/parity/cases/native_stage_two_optimization_stochastic.py`` and
the JAX mirror's ``make_stochastic_stage_two_objective`` must return the same
value and the same gradient to fp64 round-off.

Both lanes are built through ``benchmarks/stochastic_stage_two_probe.py`` so the
tested construction is the published one rather than a second copy of it.  The
scale is ``bounded`` (2 samples, 4x4 surface, order-2 curves) and every solve is
one iteration: this is an evaluator test, not a trajectory test.

Run one file per process (JAX x64 is process-global)::

    PYTHONPATH=src:build/cp311-cp311-linux_x86_64 JAX_ENABLE_X64=1 \
    JAX_PLATFORMS=cpu MPI4PY_RC_INITIALIZE=false OMP_NUM_THREADS=4 \
    .venv/bin/python -m pytest tests/jax/examples/test_stochastic_matched_state_parity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.stochastic_stage_two_probe import (
    build_shared_inputs,
    run_jax_leg,
    run_native_leg,
)
from examples.jax.parity.cases.native_stage_two_optimization_stochastic import (
    _symmetry_layout,
)
from simsopt.geo import GaussianSampler
from simsopt_jax.examples import (
    materialize_stochastic_coil_perturbations,
    stochastic_stage_two_configuration,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import _scalar_options


#: The lane the mirror mirrors; its optimizer call is the policy SSOT.
NATIVE_EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "2_Intermediate"
    / "stage_two_optimization_stochastic.py"
)


#: fp64 round-off over a 2-sample bounded objective, not a physics tolerance:
#: the observed agreement at this scale is ~5e-17 absolute on both quantities,
#: and at the shipped native_default scale ~5e-16 on the gradient.
EVALUATOR_ATOL = 1.0e-12
EVALUATOR_RTOL = 1.0e-9


@pytest.fixture(scope="module")
def matched_lanes():
    """Both lanes at ``bounded`` scale, the JAX lane probed at native's endpoint.

    The native leg's ``final`` state and the JAX leg's ``probe`` state are the
    same 96 coordinates by construction: the JAX leg is handed the array the
    native leg ended on.  That, plus the shared ``initial`` state, is two
    matched states for the price of two legs.
    """
    shared = build_shared_inputs("bounded")
    _, native = run_native_leg(
        shared, budget=1, maxcor=10, repeat=1, probe_parameters=None
    )
    _, jax_leg = run_jax_leg(
        shared,
        budget=1,
        maxcor=10,
        repeat=1,
        sample_tile=None,
        probe_parameters=native["final_parameters"],
    )
    return shared, native, jax_leg


def test_mirror_convergence_tolerances_equal_the_native_example_call() -> None:
    """``tol=1e-15`` in the native call is *two* tolerances, and both must mirror.

    ``scipy.optimize.minimize(..., method="L-BFGS-B", tol=x)`` sets ``ftol`` and
    ``gtol`` to ``x``, so the native example runs at ``ftol = gtol = 1e-15``.
    ``serial_solve_jax`` spells the same pair ``rtol -> ftol``, ``atol -> gtol``
    (``src/simsopt_jax/solve/serial.py::_scalar_options``); a mirror that left
    ``atol`` at the 1e-8 library default would stop on a gradient test seven
    orders looser than the lane it mirrors.
    """
    native_source = NATIVE_EXAMPLE.read_text(encoding="utf-8")
    assert "tol=1e-15" in native_source
    assert "method='L-BFGS-B'" in native_source

    for scale in ("bounded", "native_default"):
        configuration = stochastic_stage_two_configuration(scale)
        assert configuration.rtol == 1.0e-15
        assert configuration.atol == 1.0e-15, (
            f"{scale} mirror would pass gtol={configuration.atol!r} where the "
            "native example passes 1e-15"
        )

    options = _scalar_options(
        Driver.SIMSOPT_LBFGSB,
        rtol=1.0e-15,
        atol=1.0e-15,
        max_steps=400,
        maxcor=10,
        line_search_max_steps=None,
    )
    assert (options.ftol, options.gtol) == (1.0e-15, 1.0e-15)


def test_training_samples_are_reproducible_from_seed_rng_and_ordering(
    matched_lanes,
) -> None:
    shared, _, _ = matched_lanes
    sampler = GaussianSampler(
        shared.base_curves[0].quadpoints,
        sigma=shared.configuration.perturbation_sigma,
        length_scale=shared.configuration.perturbation_length_scale,
        n_derivs=1,
    )
    source_indices, rotations = _symmetry_layout(shared.coils, shared.base_curves)
    redrawn = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=source_indices,
        rotations=rotations,
        base_curve_count=len(shared.base_curves),
        sample_count=shared.configuration.training_sample_count,
        seed=shared.configuration.training_seed,
    )

    assert redrawn.sha256 == shared.training.sha256, (
        "redrawing the training bundle from the same seed produced a different "
        "fingerprint: the PCG64DXSM stream, the draw ordering "
        "(systematic-per-base-curve then statistical-per-coil) or the dtype "
        "moved, and the two lanes are no longer perturbed by the same bytes"
    )
    np.testing.assert_array_equal(redrawn.gamma, shared.training.gamma)
    np.testing.assert_array_equal(redrawn.gammadash, shared.training.gammadash)


def test_both_lanes_start_from_one_initial_state(matched_lanes) -> None:
    _, native, jax_leg = matched_lanes

    # The initial DOFs are read independently per lane (simsopt BiotSavart vs
    # BiotSavartJAX) and must still agree bit for bit.
    assert native["dof_count"] == jax_leg["dof_count"]
    np.testing.assert_array_equal(
        native["initial_parameters"],
        jax_leg["initial_parameters"],
        err_msg=(
            "the two lanes started from different coordinates, so nothing "
            "downstream of this is a parity measurement"
        ),
    )


def test_evaluator_agrees_at_the_matched_initial_state(matched_lanes) -> None:
    _, native, jax_leg = matched_lanes
    native_state = native["evaluations"]["initial"]
    jax_state = jax_leg["evaluations"]["initial"]

    assert native_state["objective"] == pytest.approx(
        jax_state["objective"], rel=EVALUATOR_RTOL, abs=EVALUATOR_ATOL
    ), (
        "stochastic objective value disagrees at the shared initial state: "
        f"native {native_state['objective']!r} vs jax {jax_state['objective']!r}"
    )
    gradient_gap = float(
        np.max(np.abs(native_state["gradient"] - jax_state["gradient"]))
    )
    assert gradient_gap <= EVALUATOR_ATOL, (
        "stochastic gradient disagrees at the shared initial state by "
        f"{gradient_gap:.3e}, above fp64 round-off; an evaluator defect, not "
        "an optimizer trajectory difference"
    )


def test_evaluator_agrees_at_the_native_endpoint_handed_to_the_jax_lane(
    matched_lanes,
) -> None:
    _, native, jax_leg = matched_lanes
    native_state = native["evaluations"]["final"]
    jax_state = jax_leg["evaluations"]["probe"]

    assert native_state["objective"] == pytest.approx(
        jax_state["objective"], rel=EVALUATOR_RTOL, abs=EVALUATOR_ATOL
    ), (
        "stochastic objective value disagrees at the native endpoint: "
        f"native {native_state['objective']!r} vs jax {jax_state['objective']!r}"
    )
    gradient_gap = float(
        np.max(np.abs(native_state["gradient"] - jax_state["gradient"]))
    )
    assert gradient_gap <= EVALUATOR_ATOL, (
        "stochastic gradient disagrees at the native endpoint by "
        f"{gradient_gap:.3e}, above fp64 round-off"
    )


def test_every_leg_retains_an_endpoint_gradient(matched_lanes) -> None:
    """Endpoints are retained with their gradients — and compared by nobody.

    A capped run's endpoint is not reproducible even native-to-native across
    thread counts, so no test here gates on ``final_parameters`` agreement; what
    the driver owes a reader is the gradient at the state it stopped on, which
    says how far from stationary that state is.
    """
    _, native, jax_leg = matched_lanes

    for lane, leg in (("native", native), ("jax", jax_leg)):
        gradient = leg["evaluations"]["final"]["gradient"]
        assert gradient.shape == (leg["dof_count"],), (
            f"{lane} leg retained no endpoint gradient of the right shape: "
            f"{gradient.shape}"
        )
        assert np.all(np.isfinite(gradient))
