"""Budget accounting and line-search-anchor contracts for lbfgs-ondevice.

The private L-BFGS driver brackets each phase of a solve with a
``<phase>_started`` / ``<phase>_returned`` diagnostic label pair, and publishes
a rejected-step record for abnormal terminations.  Downstream gates read both:
they subtract label timestamps to charge a solve against a seed budget and a
main-optimizer budget, and they key repeated failures on the anchor fields the
rejected-step record carries.  Label *ordering* is pinned elsewhere
(``tests/geo/test_boozersurface_jax_private.py``); what these tests pin is the
arithmetic underneath the budgets -- that the phases are closed and disjoint,
that seeding repartitions work instead of adding it, that each budget line
appears exactly when its work does -- and the semantics of the anchor a gate
would key on.

The anchor contract is that every published field is read from the terminal
L-BFGS-B workspace: what the workspace cannot answer for is absent, not
defaulted, so a gate keyed on any field is keyed on a measurement.
"""

from __future__ import annotations

import json
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax.geo.optimizers.optimizer as _opt
import simsopt_jax.geo.optimizers.private._lbfgs as _lbfgs
import simsopt_jax.geo.optimizers.private._lbfgsb_scipy as lbfgsb
import simsopt_jax.geo.optimizers.private._result_converters as _converters
from simsopt_jax.geo.optimizer_host_lbfgs import (
    LINE_SEARCH_FAILURE_REASON_MAXITER,
    LINE_SEARCH_FAILURE_REASON_NOT_DESCENT,
)
from simsopt_jax.geo.optimizers.private._common import (
    private_optimizer_runtime_is_supported,
)

pytestmark = [
    pytest.mark.private_optimizer_runtime,
    pytest.mark.skipif(
        not private_optimizer_runtime_is_supported(jax.__version__),
        reason=(
            "lbfgs-ondevice budget and anchor accounting is validated on the "
            "pinned JAX runtime."
        ),
    ),
]

_SEED_PHASE = "lbfgs_initial_value_and_grad_seed"
_INITIAL_STATE_PHASE = "lbfgs_initial_state"
_MAIN_KERNEL_PHASE = "lbfgs_main_kernel"
_EFFECTS_BARRIER_PHASE = "lbfgs_effects_barrier"
_RESULT_CONVERSION_PHASE = "lbfgs_result_conversion"

_ANCHOR_FIELDS_ALWAYS = frozenset(
    {
        "iteration",
        "step_scale",
        "line_search_failed",
        "nonfinite_step",
        "ls_status",
        "failure_reason",
    }
)
# dcsrch's curvature residual exists only once dcsrch has run, which at ABNORMAL
# means the search spent its backtrack budget (info == MAXLS).  A non-descent
# direction is rejected before dcsrch is called at all.
_ANCHOR_FIELDS_DCSRCH_RAN = _ANCHOR_FIELDS_ALWAYS | {"curvature_margin"}
# Names the L-BFGS-B workspace cannot answer for, so the anchor must not carry
# them.  ``lnsrlb`` keeps one ``stp`` slot that ``dcsrch`` overwrites on every
# entry, which destroys the requested initial step, the first tested alpha, and
# the alpha the last trial was evaluated at; ``stalled_step`` /
# ``valid_curvature`` / ``trial_converged`` are outer-iteration measurements
# this lane never takes.
_FIELDS_THE_WORKSPACE_CANNOT_ANSWER = (
    "requested_initial_step",
    "first_tested_alpha",
    "best_finite_alpha",
    "returned_alpha",
    "armijo_margin",
    "stalled_step",
    "valid_curvature",
    "trial_converged",
)


class _DiagnosticRun(NamedTuple):
    result: object
    events: tuple[tuple[str, float, dict[str, object]], ...]
    wall_s: float


def _shifted_quadratic(x):
    shifted = x - 0.25
    return 0.5 * jnp.dot(shifted, shifted), shifted


def _flat_objective_with_constant_slope(x):
    # Constant value against a gradient that claims a descent direction: every
    # trial step fails Armijo, so dcsrch gives up before any iteration is
    # accepted and the driver terminates ABNORMAL with a finite state.
    return jnp.asarray(0.0, dtype=x.dtype), jnp.ones_like(x)


def _i32(value):
    return jnp.asarray(value, dtype=jnp.int32)


def _f64(value):
    return jnp.asarray(value, dtype=jnp.float64)


def _split_phase_label(label):
    for suffix in ("_started", "_returned"):
        if label.endswith(suffix):
            return label[: -len(suffix)], suffix
    return label, ""


def _phase_spans(events):
    spans = {}
    faults = []
    open_phase = None
    for label, timestamp, _fields in events:
        phase, suffix = _split_phase_label(label)
        if suffix == "_started":
            if open_phase is not None:
                faults.append(f"{label} opened while {open_phase} was still open")
            elif phase in spans:
                faults.append(f"{label} reopened an already accounted phase")
            else:
                open_phase = phase
                spans[phase] = [timestamp, None]
        elif suffix == "_returned":
            if open_phase != phase:
                faults.append(f"{label} closed no matching open phase")
            else:
                spans[phase][1] = timestamp
                open_phase = None
        else:
            faults.append(f"{label} is not a paired budget label")
    if open_phase is not None:
        faults.append(f"{open_phase} never returned")
    return {phase: tuple(bounds) for phase, bounds in spans.items()}, faults


def _phase_fields(events, label):
    return [fields for emitted, _timestamp, fields in events if emitted == label]


def _run_lbfgs(objective, x0, *, initial_value_and_grad, observed):
    events = []

    def record(label, **fields):
        events.append((label, time.perf_counter(), fields))

    with _opt.target_optimizer_diagnostic_events(record):
        entered = time.perf_counter()
        result = _opt.target_minimize(
            objective,
            x0,
            method="lbfgs-ondevice",
            tol=1.0e-10,
            maxiter=8,
            value_and_grad=True,
            initial_value_and_grad=initial_value_and_grad,
            progress_callback=(lambda *_args: None) if observed else None,
        )
        returned = time.perf_counter()
    return _DiagnosticRun(
        result=result, events=tuple(events), wall_s=returned - entered
    )


@pytest.fixture(scope="module")
def lbfgs_runs():
    x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    seed = _shifted_quadratic(x0)
    return {
        "seeded": _run_lbfgs(
            _shifted_quadratic, x0, initial_value_and_grad=seed, observed=True
        ),
        "unseeded": _run_lbfgs(
            _shifted_quadratic, x0, initial_value_and_grad=None, observed=True
        ),
        "seeded_unobserved": _run_lbfgs(
            _shifted_quadratic, x0, initial_value_and_grad=seed, observed=False
        ),
        "line_search_failure": _run_lbfgs(
            _flat_objective_with_constant_slope,
            x0,
            initial_value_and_grad=None,
            observed=False,
        ),
    }


@pytest.fixture(scope="module")
def line_search_failure_anchor(lbfgs_runs):
    # The record is a single terminal observation, not a history.  ``setulb``
    # reports ABNORMAL only when a line search fails with no correction pairs
    # left to refresh from, which ends the solve, so a run publishes either zero
    # or one record and ``rejected_step_count`` is an abnormal-termination
    # indicator rather than a tally of rejected trials.
    log = lbfgs_runs["line_search_failure"].result.invalid_step_log
    assert len(log) == 1
    return log[0]


def test_seeding_repartitions_the_budget_without_adding_optimizer_work(lbfgs_runs):
    seeded = lbfgs_runs["seeded"].result
    unseeded = lbfgs_runs["unseeded"].result

    assert seeded.success is True
    assert unseeded.success is True
    assert seeded.nfev == unseeded.nfev
    assert seeded.njev == unseeded.njev
    assert seeded.nit == unseeded.nit
    np.testing.assert_allclose(seeded.x, unseeded.x, rtol=0.0, atol=1.0e-12)
    assert float(seeded.fun) == pytest.approx(float(unseeded.fun), abs=1.0e-12)


def test_seed_budget_line_exists_only_when_a_seed_is_supplied(lbfgs_runs):
    seeded_spans, _ = _phase_spans(lbfgs_runs["seeded"].events)
    unseeded_spans, _ = _phase_spans(lbfgs_runs["unseeded"].events)

    assert _SEED_PHASE in seeded_spans
    assert _SEED_PHASE not in unseeded_spans
    assert _MAIN_KERNEL_PHASE in seeded_spans
    assert _MAIN_KERNEL_PHASE in unseeded_spans
    assert seeded_spans[_SEED_PHASE][1] <= seeded_spans[_MAIN_KERNEL_PHASE][0]


def test_budget_phase_labels_form_disjoint_closed_spans(lbfgs_runs):
    for name, run in lbfgs_runs.items():
        spans, faults = _phase_spans(run.events)

        assert faults == [], name
        assert spans, name
        durations = [end - start for start, end in spans.values()]
        assert all(duration >= 0.0 for duration in durations), name
        assert sum(durations) <= run.wall_s, name


def test_main_kernel_budget_closes_before_host_result_conversion(lbfgs_runs):
    for name, run in lbfgs_runs.items():
        spans, _ = _phase_spans(run.events)

        assert _RESULT_CONVERSION_PHASE in spans, name
        assert spans[_INITIAL_STATE_PHASE][1] <= spans[_MAIN_KERNEL_PHASE][0], name
        assert spans[_MAIN_KERNEL_PHASE][1] <= spans[_RESULT_CONVERSION_PHASE][0], name


def test_effects_barrier_budget_line_tracks_host_observation(lbfgs_runs):
    # Same objective, same seed: ``observed`` is the only variable between the
    # two runs, so the barrier line can only be tracking host observation.
    observed = lbfgs_runs["seeded"]
    unobserved = lbfgs_runs["seeded_unobserved"]
    observed_spans, _ = _phase_spans(observed.events)
    unobserved_spans, _ = _phase_spans(unobserved.events)
    started = f"{_MAIN_KERNEL_PHASE}_started"

    assert [
        fields["accepted_step_callback"]
        for fields in _phase_fields(observed.events, started)
    ] == [True]
    assert [
        fields["accepted_step_callback"]
        for fields in _phase_fields(unobserved.events, started)
    ] == [False]
    assert _EFFECTS_BARRIER_PHASE in observed_spans
    assert (
        observed_spans[_MAIN_KERNEL_PHASE][1]
        <= observed_spans[_EFFECTS_BARRIER_PHASE][0]
    )
    assert _EFFECTS_BARRIER_PHASE not in unobserved_spans


def test_line_search_failure_anchor_pins_the_terminal_rejected_step(
    lbfgs_runs, line_search_failure_anchor
):
    result = lbfgs_runs["line_search_failure"].result

    assert result.success is False
    assert result.rejected_step_count == 1
    assert line_search_failure_anchor["line_search_failed"] is True
    assert line_search_failure_anchor["iteration"] == result.nit == 0
    assert line_search_failure_anchor["ls_status"] == result.ls_status
    assert line_search_failure_anchor["ls_status"] < 0


def test_line_search_failure_anchor_names_the_terminal_failure_mode(
    line_search_failure_anchor,
):
    # ``failure_reason`` is read from ``isave[34]``, whose only two terminal
    # values are NOT_DESCENT (the first directional derivative of a line search
    # was not negative) and MAXLS (the search spent its backtrack budget).  This
    # objective keeps every trial above the Armijo line, so the search exhausts
    # ``maxls`` -- and the anchor must say so rather than fall back on the
    # unqualified "line search failed".
    assert line_search_failure_anchor["ls_status"] == (
        lbfgsb.LBFGSB_LINE_SEARCH_INFO_MAXLS
    )
    assert line_search_failure_anchor["failure_reason"] == (
        LINE_SEARCH_FAILURE_REASON_MAXITER
    )


def test_line_search_failure_anchor_reports_its_step_under_one_name(
    line_search_failure_anchor,
):
    # ``dsave[13]`` is published once, as ``step_scale``.  It is the line
    # search's step scale at the moment it was abandoned -- dcsrch's next
    # proposal, not a step any trial was evaluated at -- so the anchor must not
    # also offer it under a name that claims a position in the trial sequence.
    step_scale = line_search_failure_anchor["step_scale"]

    assert np.isfinite(step_scale)
    assert step_scale > 0.0
    assert set(line_search_failure_anchor).isdisjoint(
        _FIELDS_THE_WORKSPACE_CANNOT_ANSWER
    )


def test_anchor_step_scale_is_the_abandoned_step_not_the_initial_one(
    line_search_failure_anchor,
):
    # ``lnsrlb`` opens an unbounded first line search at ``min(1/|d|, stpmx)``,
    # which for this objective is ``1/sqrt(2)``.  The published step scale is
    # what is left after the search has backtracked itself into the ground, so
    # it must be orders of magnitude below that opening step -- the check that
    # would have failed while the anchor published ``dsave[13]`` under
    # ``requested_initial_step``.
    opening_step = 1.0 / np.sqrt(2.0)

    assert line_search_failure_anchor["step_scale"] < 1.0e-6 * opening_step


def test_anchor_curvature_margin_is_dcsrch_own_curvature_residual(
    line_search_failure_anchor,
):
    # dcsrch accepts a step when ``abs(g) <= gtol*(-ginit)``; the margin is that
    # test's residual, and it needs no alpha, so the terminal workspace can still
    # answer for it.  The objective reports a constant gradient of ones against
    # ``d = -g``, so ``phi'`` is -2 at alpha=0 and at every trial, giving
    # ``|-2| - gtol*2``.
    expected = 2.0 - lbfgsb.LBFGSB_LINE_SEARCH_GTOL * 2.0

    margin = line_search_failure_anchor["curvature_margin"]
    assert np.isfinite(margin)
    assert margin == pytest.approx(expected, rel=1.0e-12)
    # Positive residual: the curvature condition was never met either, which a
    # hard-coded ``valid_curvature=True`` used to deny.
    assert margin > 0.0


def test_finite_line_search_failure_is_not_reported_as_a_nonfinite_step(
    lbfgs_runs, line_search_failure_anchor
):
    result = lbfgs_runs["line_search_failure"].result

    assert np.isfinite(float(result.fun))
    assert np.all(np.isfinite(np.asarray(result.jac)))
    assert line_search_failure_anchor["nonfinite_step"] is False


def _abnormal_state_with_line_search_info(info):
    """An otherwise-pristine ABNORMAL state carrying one terminal ``info``.

    The dcsrch save area stays as ``lbfgsb_empty_workspace`` left it -- all
    zeros -- which is exactly the state ``lnsrlb`` hands back when it rejects a
    non-descent direction on the first line search, since it returns ``dsave``
    untouched without ever calling dcsrch.
    """
    state = lbfgsb.lbfgsb_initial_state(
        jnp.asarray([1.0, -2.0], dtype=jnp.float64), m=5
    )
    workspace = state.workspace
    return state._replace(
        workspace=workspace._replace(
            task=workspace.task.at[0].set(lbfgsb.ABNORMAL),
            isave=workspace.isave.at[34].set(info),
            # A directional derivative survives from an earlier search even when
            # this one never ran a curvature test.
            dsave=workspace.dsave.at[lbfgsb.LBFGSB_DSAVE_GD].set(2.0),
        )
    )


def _anchor_for_line_search_info(info):
    record = _lbfgs._lbfgsb_invalid_step_record(
        _abnormal_state_with_line_search_info(info)
    )
    events = _converters._private_lbfgs_invalid_step_record_to_host(record)
    assert len(events) == 1
    return events[0]


def test_lnsrlb_rejects_a_non_descent_direction_without_running_dcsrch():
    # Drive lnsrlb straight at the non-descent branch: ifun=0 makes this the
    # first function value of the search and d=+g makes the directional
    # derivative +2, so the search is abandoned before dcsrch is reached.  The
    # branch must report NOT_DESCENT and hand the dcsrch save area back
    # untouched -- the reason a curvature residual cannot be computed here.
    n = 2
    ones = jnp.ones((n,), dtype=jnp.float64)
    zeros = jnp.zeros((n,), dtype=jnp.float64)
    isave_in = jnp.zeros((2,), dtype=jnp.int32)
    dsave_in = jnp.zeros((13,), dtype=jnp.float64)

    result = lbfgsb.lbfgsb_lnsrlb(
        zeros,  # l
        zeros,  # u
        jnp.zeros((n,), dtype=jnp.int32),  # nbd: unbounded
        ones,  # x
        _f64(0.0),  # f
        _f64(0.0),  # fold
        _f64(0.0),  # gd
        _f64(0.0),  # gdold
        ones,  # g
        ones,  # d -- points along +g, so g.d = +2 > 0
        zeros,  # r
        zeros,  # t
        zeros,  # z
        _f64(1.0),  # stp
        _f64(0.0),  # dnorm
        _f64(0.0),  # dtd
        _f64(0.0),  # xstep
        _f64(0.0),  # stpmx
        _i32(0),  # iteration
        _i32(0),  # ifun -- first function value of this line search
        _i32(0),  # iback
        _i32(0),  # nfgv
        _i32(0),  # info
        _i32(lbfgsb.START),  # task
        _i32(lbfgsb.NO_MSG),  # task_msg
        False,  # boxed
        False,  # cnstnd
        isave_in,
        dsave_in,
        _i32(lbfgsb.START),  # temp_task
        _i32(lbfgsb.NO_MSG),  # temp_task_msg
    )

    assert int(result.gd) == 2
    assert int(result.info) == lbfgsb.LBFGSB_LINE_SEARCH_INFO_NOT_DESCENT
    # dcsrch never ran, so its save area -- ginit included -- is exactly what
    # went in, not a description of this search.
    np.testing.assert_array_equal(np.asarray(result.dsave), np.asarray(dsave_in))
    np.testing.assert_array_equal(np.asarray(result.isave), np.asarray(isave_in))


def test_not_descent_anchor_omits_the_curvature_margin():
    # The gate: on NOT_DESCENT the save area's ginit belongs to a previous
    # search (zero here, as on a first search), so publishing
    # abs(gd) - gtol*(-ginit) would yield abs(gd) -- a plausible number that is
    # the residual of no test that was ever evaluated.  The key must be absent.
    #
    # Driven through the record builder to isolate the gate; a full solve
    # reaches this state too -- see the subnormal-gradient test below.
    anchor = _anchor_for_line_search_info(lbfgsb.LBFGSB_LINE_SEARCH_INFO_NOT_DESCENT)

    assert anchor["failure_reason"] == LINE_SEARCH_FAILURE_REASON_NOT_DESCENT
    assert set(anchor) == _ANCHOR_FIELDS_ALWAYS
    assert "curvature_margin" not in anchor


def test_maxls_anchor_publishes_the_curvature_margin_from_the_same_state():
    # Same synthetic state, same stale zero ginit, only info differs: MAXLS
    # means dcsrch ran, so the residual is a measurement and is published.
    # Pinning both branches against one state is what makes the gate -- rather
    # than any incidental difference between two solves -- the thing under test.
    anchor = _anchor_for_line_search_info(lbfgsb.LBFGSB_LINE_SEARCH_INFO_MAXLS)

    assert anchor["failure_reason"] == LINE_SEARCH_FAILURE_REASON_MAXITER
    assert set(anchor) == _ANCHOR_FIELDS_DCSRCH_RAN
    assert anchor["curvature_margin"] == pytest.approx(2.0, rel=1.0e-12)


def _constant_gradient_objective(scale):
    def objective(x):
        return jnp.asarray(0.0, dtype=x.dtype), jnp.full_like(x, scale)

    return objective


def _solve_with_constant_gradient(scale, tol):
    return _opt.target_minimize(
        _constant_gradient_objective(scale),
        jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        method="lbfgs-ondevice",
        tol=tol,
        maxiter=8,
        value_and_grad=True,
    )


def test_a_solve_reaches_not_descent_when_the_squared_gradient_is_subnormal():
    # NOT_DESCENT is reachable from a real solve, so the anchor's gate is load
    # bearing rather than defensive.  This lane is always unbounded
    # (_lbfgs.py `_LBFGSB_PRIVATE_BOUNDS = None` is the only value any call site
    # passes), so ABNORMAL's col == 0 precondition means the direction is
    # `lbfgsb_two_loop_direction`, which with no live pair is `-g/theta`
    # (`lbfgsb_apply_inverse_hessian` docstring) at theta = 1.  Mathematically
    # `g.d = -|g|^2/theta <= 0` and the non-descent branch could never fire --
    # but XLA flushes subnormal float64 products to -0.0, so once |g| is small
    # enough that |g|^2 is subnormal (|g| < ~1.5e-154) every term of the dot
    # product becomes -0.0, `gd` is exactly -0.0, and `gd >= 0.0` is true.
    # A pgtol below |g|_inf keeps the projected-gradient test from converging
    # first.  tol is nonzero here: reaching this does not need a degenerate tol.
    result = _solve_with_constant_gradient(1.0e-180, 1.0e-200)

    assert result.success is False
    assert result.ls_status == lbfgsb.LBFGSB_LINE_SEARCH_INFO_NOT_DESCENT
    assert len(result.invalid_step_log) == 1

    anchor = result.invalid_step_log[0]
    assert anchor["failure_reason"] == LINE_SEARCH_FAILURE_REASON_NOT_DESCENT
    # The gate, on the real path: dcsrch never ran, so no curvature residual.
    assert "curvature_margin" not in anchor
    assert set(anchor) == _ANCHOR_FIELDS_ALWAYS


def test_a_normal_magnitude_gradient_does_not_reach_not_descent():
    # The discriminating half: |g| = 1e-150 makes |g|^2 = 1e-300 a normal
    # float64, nothing is flushed, `gd` stays strictly negative and the same
    # objective and tol take the ordinary maxls path instead.  Subnormal
    # flushing -- not the flat objective or the tiny tol -- is what reaches
    # NOT_DESCENT.
    result = _solve_with_constant_gradient(1.0e-150, 1.0e-200)

    assert result.ls_status == lbfgsb.LBFGSB_LINE_SEARCH_INFO_MAXLS
    assert result.invalid_step_log[0]["failure_reason"] == (
        LINE_SEARCH_FAILURE_REASON_MAXITER
    )
    assert "curvature_margin" in result.invalid_step_log[0]


def test_lbfgsb_anchor_publishes_no_unmeasured_field(line_search_failure_anchor):
    # Every field the anchor carries is read from the terminal workspace, so a
    # consumer may key on any of them.  Stall, L-BFGS curvature-pair validity and
    # trial convergence are outer-iteration measurements this lane never takes;
    # they used to be published as unconditional literals and are now absent.
    assert set(line_search_failure_anchor) == _ANCHOR_FIELDS_DCSRCH_RAN


def test_line_search_failure_anchor_survives_a_json_round_trip(
    line_search_failure_anchor,
):
    round_tripped = json.loads(json.dumps(line_search_failure_anchor))

    assert set(line_search_failure_anchor) == _ANCHOR_FIELDS_DCSRCH_RAN
    assert round_tripped == line_search_failure_anchor
    assert isinstance(line_search_failure_anchor["failure_reason"], str)
    assert isinstance(line_search_failure_anchor["iteration"], int)
    assert isinstance(line_search_failure_anchor["ls_status"], int)
    assert isinstance(line_search_failure_anchor["step_scale"], float)
    assert isinstance(line_search_failure_anchor["curvature_margin"], float)
    assert isinstance(line_search_failure_anchor["line_search_failed"], bool)


def test_converged_runs_record_no_line_search_failure_anchor(lbfgs_runs):
    for name in ("seeded", "unseeded"):
        result = lbfgs_runs[name].result

        assert result.success is True
        assert result.invalid_step_log == [], name
        assert result.rejected_step_count == 0, name
