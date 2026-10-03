"""Exact continuation checks for Cartesian fieldline and vacuum-GC tracing."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.core.tracing import (
    FieldlineTracingSpec,
    GuidingCenterTracingSpec,
    IterStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    _trace_fieldline_chunk,
    _trace_guiding_center_chunk,
    trace_fieldline,
    trace_guiding_center,
)


def _rotating_field(position):
    return jnp.stack((-position[1], position[0], jnp.asarray(0.0)))


def _constant_field_with_derivative(_position):
    return jnp.array((0.0, 0.0, 1.0)), jnp.zeros((3, 3))


def _live_rows(result):
    return np.asarray(result.trajectory)[np.asarray(result.mask)]


def _live_hits(result):
    return np.asarray(result.phi_hits)[: int(result.phi_hits_count)]


def _assemble(call, spec, *, first_continuation=None):
    continuation = first_continuation
    paths = []
    hits = []
    for chunk_index in range(200):
        result, continuation = call(spec, continuation)
        rows = _live_rows(result)
        paths.append(rows if chunk_index == 0 else rows[1:])
        hits.append(_live_hits(result))
        if int(result.status) != 1:
            break
    else:
        raise AssertionError("trace did not reach a terminal reason")
    return result, continuation, np.concatenate(paths), np.concatenate(hits)


def test_fieldline_chunks_preserve_rejected_step_controller_and_plane_events():
    y0 = jnp.array((1.0, 0.0, 0.0))
    phis = jnp.array((0.3, 0.7))
    base = FieldlineTracingSpec(
        tmax=1.5, rtol=1e-11, atol=1e-11, max_steps=400, max_phi_hits=16,
    )
    _, seed = _trace_fieldline_chunk(
        replace(base, max_steps=1), y0, _rotating_field, phis=phis
    )
    forced = replace(seed, h=jnp.asarray(1.0, dtype=jnp.float64))
    rejected, after_reject = _trace_fieldline_chunk(
        replace(base, max_steps=1), y0, _rotating_field,
        phis=phis, continuation=forced,
    )
    assert int(rejected.steps_taken) == 0
    assert float(after_reject.t) == float(forced.t)
    assert float(after_reject.h) < float(forced.h)
    assert int(after_reject.trial_count) == int(forced.trial_count) + 1

    def call(spec, continuation):
        return _trace_fieldline_chunk(
            spec, y0, _rotating_field, phis=phis, continuation=continuation
        )

    reference, _ = call(base, forced)
    terminal, state, path, hits = _assemble(
        call, replace(base, max_steps=3), first_continuation=forced
    )
    assert int(reference.status) == int(terminal.status) == 0
    assert int(state.trial_count) > int(state.accepted_count)
    np.testing.assert_allclose(path, _live_rows(reference), rtol=0, atol=2e-13)
    np.testing.assert_allclose(hits, _live_hits(reference), rtol=0, atol=2e-13)


def test_fieldline_chunks_preserve_global_iteration_and_transit_criteria():
    y0 = jnp.array((1.0, 0.0, 0.0))
    for criterion in (
        IterStoppingCriterion(max_iter=7),
        ToroidalTransitStoppingCriterion(max_transits=0.08),
    ):
        base = FieldlineTracingSpec(
            tmax=2.0, rtol=1e-8, atol=1e-8, max_steps=300,
            max_phi_hits=16,
        )

        def call(spec, continuation):
            return _trace_fieldline_chunk(
                spec, y0, _rotating_field,
                phis=jnp.array((0.2,)), stopping_criteria=(criterion,),
                continuation=continuation,
            )

        reference, _ = call(base, None)
        terminal, state, path, hits = _assemble(call, replace(base, max_steps=2))
        assert int(reference.status) == int(terminal.status) == -1
        assert int(state.trial_count) > 2
        assert hits.shape[0] == int(reference.phi_hits_count)
        np.testing.assert_allclose(path, _live_rows(reference), rtol=0, atol=2e-13)
        np.testing.assert_allclose(hits, _live_hits(reference), rtol=0, atol=2e-13)


def test_guiding_center_chunks_preserve_iteration_event_and_public_result():
    y0 = jnp.array((1.0, 0.0, 0.0, 1.0))
    base = GuidingCenterTracingSpec(
        tmax=100.0, rtol=1e-9, atol=1e-9, max_steps=100,
        dtmax=1.0,
        max_phi_hits=8,
    )
    stopping = (IterStoppingCriterion(max_iter=2),)

    def call(spec, continuation):
        return _trace_guiding_center_chunk(
            spec, y0, _constant_field_with_derivative,
            m=1.0, q=1.0, mu=0.0, stopping_criteria=stopping,
            continuation=continuation,
        )

    reference = trace_guiding_center(
        base, y0, _constant_field_with_derivative,
        m=1.0, q=1.0, mu=0.0, stopping_criteria=stopping,
    )
    terminal, state, path, hits = _assemble(call, replace(base, max_steps=2))
    assert int(reference.status) == int(terminal.status) == -1
    assert int(state.trial_count) == 3
    np.testing.assert_allclose(path, _live_rows(reference), rtol=0, atol=2e-13)
    np.testing.assert_allclose(hits, _live_hits(reference), rtol=0, atol=2e-13)

    fieldline_public = trace_fieldline(
        FieldlineTracingSpec(tmax=0.1, rtol=1e-8, atol=1e-8, max_steps=100),
        jnp.array((1.0, 0.0, 0.0)), _rotating_field,
    )
    assert int(fieldline_public.status) == 0


def test_continuation_state_is_jittable_pytree():
    y0 = jnp.array((1.0, 0.0, 0.0))
    spec = FieldlineTracingSpec(
        tmax=1.0, rtol=1e-9, atol=1e-9, max_steps=2,
    )
    _, state = _trace_fieldline_chunk(spec, y0, _rotating_field)

    def resume(continuation):
        return _trace_fieldline_chunk(
            spec, y0, _rotating_field, continuation=continuation
        )

    eager_result, eager_state = resume(state)
    compiled_result, compiled_state = jax.jit(resume)(state)
    np.testing.assert_allclose(
        _live_rows(compiled_result), _live_rows(eager_result), rtol=0, atol=2e-13
    )
    assert int(compiled_state.trial_count) == int(eager_state.trial_count)
