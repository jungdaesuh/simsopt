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
from simsopt_jax.geo.optimizer_host_lbfgs import LINE_SEARCH_FAILURE_REASON_FAILED
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

_ANCHOR_FIELDS = frozenset(
    {
        "iteration",
        "step_scale",
        "line_search_failed",
        "nonfinite_step",
        "stalled_step",
        "valid_curvature",
        "trial_converged",
        "ls_status",
        "requested_initial_step",
        "first_tested_alpha",
        "best_finite_alpha",
        "returned_alpha",
        "failure_reason",
        "armijo_margin",
        "curvature_margin",
    }
)
_UNPOPULATED_ANCHOR_SLOTS = (
    "best_finite_alpha",
    "returned_alpha",
    "armijo_margin",
    "curvature_margin",
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
    # The record is a single terminal slot, not a history: the buffers are cut
    # from ``isave[:1]`` / ``dsave[:1]`` (_lbfgs.py:837-838), so ``count`` is a
    # 0/1 abnormal-termination indicator and the ring-buffer reader in
    # _result_converters.py:111-159 can never wrap.  ``rejected_step_count``
    # therefore indicates an abnormal termination; it does not count rejections.
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
    assert line_search_failure_anchor["failure_reason"] == (
        LINE_SEARCH_FAILURE_REASON_FAILED
    )
    assert line_search_failure_anchor["iteration"] == result.nit == 0
    assert line_search_failure_anchor["ls_status"] == result.ls_status
    assert line_search_failure_anchor["ls_status"] < 0


def test_line_search_failure_anchor_reports_one_step_under_three_names(
    line_search_failure_anchor,
):
    # One value fans out to three names: ``step_slot`` is built once from
    # ``dsave[13]`` (_lbfgs.py:847-848) and published as ``step_scale``,
    # ``requested_initial_step``, and ``first_tested_alpha``
    # (_lbfgs.py:858,865-866), so an anchor keyed on any of the three is the
    # same anchor.
    step_scale = line_search_failure_anchor["step_scale"]

    assert np.isfinite(step_scale)
    assert step_scale > 0.0
    assert line_search_failure_anchor["requested_initial_step"] == step_scale
    assert line_search_failure_anchor["first_tested_alpha"] == step_scale


def test_lbfgsb_anchor_leaves_alpha_and_margin_slots_unpopulated(
    line_search_failure_anchor,
):
    # These four slots are hard zeros on this lane (``float_slot``,
    # _lbfgs.py:867-868,870-871), so keying an anchor on a returned alpha or on an
    # Armijo/curvature margin collapses every anchor onto the same identity.
    for slot in _UNPOPULATED_ANCHOR_SLOTS:
        assert line_search_failure_anchor[slot] == 0.0


def test_finite_line_search_failure_is_not_reported_as_a_nonfinite_step(
    lbfgs_runs, line_search_failure_anchor
):
    result = lbfgs_runs["line_search_failure"].result

    assert np.isfinite(float(result.fun))
    assert np.all(np.isfinite(np.asarray(result.jac)))
    assert line_search_failure_anchor["nonfinite_step"] is False


def test_lbfgsb_anchor_stall_curvature_and_trial_flags_are_path_constants(
    line_search_failure_anchor,
):
    # These three flags are unconditional literals on this lane
    # (``false_slot`` / ``true_slot`` / ``false_slot``, _lbfgs.py:861-863):
    # nothing computes stall, curvature validity, or trial convergence for an
    # L-BFGS-B anchor, so a consumer must not read them as measurements.
    assert line_search_failure_anchor["stalled_step"] is False
    assert line_search_failure_anchor["valid_curvature"] is True
    assert line_search_failure_anchor["trial_converged"] is False


def test_line_search_failure_anchor_survives_a_json_round_trip(
    line_search_failure_anchor,
):
    round_tripped = json.loads(json.dumps(line_search_failure_anchor))

    assert set(line_search_failure_anchor) == _ANCHOR_FIELDS
    assert round_tripped == line_search_failure_anchor
    assert isinstance(line_search_failure_anchor["failure_reason"], str)
    assert isinstance(line_search_failure_anchor["iteration"], int)
    assert isinstance(line_search_failure_anchor["ls_status"], int)
    assert isinstance(line_search_failure_anchor["step_scale"], float)
    assert isinstance(line_search_failure_anchor["line_search_failed"], bool)


def test_converged_runs_record_no_line_search_failure_anchor(lbfgs_runs):
    for name in ("seeded", "unseeded"):
        result = lbfgs_runs[name].result

        assert result.success is True
        assert result.invalid_step_log == [], name
        assert result.rejected_step_count == 0, name
