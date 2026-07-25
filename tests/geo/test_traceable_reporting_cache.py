"""Focused contracts for reusing traceable outer-objective reporting terms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt_jax_adapters.geo import surface_objectives_traceable as _traceable


def _raw_terms(scale=1.0):
    return {
        term_name: jnp.asarray(scale * (index + 1), dtype=jnp.float64)
        for index, (term_name, _weight_key) in enumerate(
            _traceable._TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS
        )
    }


def _packed_result(*, outer_raw_terms=None):
    scalar = jnp.asarray(1.0, dtype=jnp.float64)
    return _traceable._pack_traceable_forward_result(
        value=scalar,
        x=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        sdofs=jnp.asarray([1.0], dtype=jnp.float64),
        iota=scalar,
        G=scalar,
        linear_solve_factors=None,
        success=jnp.asarray(True),
        primal_success=jnp.asarray(True),
        adjoint_linear_solve_available=jnp.asarray(True),
        outer_raw_terms=outer_raw_terms,
    )


def test_forward_result_exposes_fixed_shape_raw_term_payload():
    expected_terms = _raw_terms()
    present, actual_terms = _traceable.traceable_forward_result_outer_raw_terms(
        _packed_result(outer_raw_terms=expected_terms)
    )

    assert bool(np.asarray(present))
    assert actual_terms.keys() == expected_terms.keys()
    for term_name, expected in expected_terms.items():
        np.testing.assert_array_equal(np.asarray(actual_terms[term_name]), expected)


def test_forward_result_marks_absent_raw_terms_without_changing_tree_shape():
    present, raw_terms = _traceable.traceable_forward_result_outer_raw_terms(
        _packed_result()
    )

    assert not bool(np.asarray(present))
    assert raw_terms.keys() == _raw_terms().keys()
    assert all(np.isnan(np.asarray(value)) for value in raw_terms.values())


def test_total_objective_aux_reuses_the_canonical_weighted_terms(monkeypatch):
    raw_terms = _raw_terms()
    weighted_terms = {
        term_name: value * jnp.asarray(2.0, dtype=jnp.float64)
        for term_name, value in raw_terms.items()
    }
    monkeypatch.setattr(
        _traceable,
        "_traceable_single_stage_outer_term_values",
        lambda *_args, **_kwargs: raw_terms,
    )
    monkeypatch.setattr(
        _traceable,
        "_traceable_weighted_single_stage_outer_term_values",
        lambda *_args, **_kwargs: weighted_terms,
    )
    monkeypatch.setattr(
        _traceable,
        "_traceable_total_objective_kwargs",
        lambda _objective_kwargs: {},
    )

    total, actual_raw_terms = (
        _traceable._evaluate_traceable_total_objective_with_raw_terms(
            jnp.asarray([0.0], dtype=jnp.float64),
            jnp.asarray([0.0], dtype=jnp.float64),
            object(),
            {"outer_objective_config": {}},
        )
    )

    expected_total = sum(float(np.asarray(value)) for value in weighted_terms.values())
    assert float(np.asarray(total)) == expected_total
    assert actual_raw_terms is raw_terms


def test_reporting_bundle_selects_cached_terms_only_when_supplied(monkeypatch):
    build_calls = []

    def make_uncached(_compiled_bundle, *, include_distance_metrics):
        build_calls.append(("uncached", include_distance_metrics))
        return lambda coil_dofs, solved_x, solver_success: (
            "uncached",
            include_distance_metrics,
            coil_dofs,
            solved_x,
            solver_success,
        )

    def make_cached(_compiled_bundle, *, include_distance_metrics):
        build_calls.append(("cached", include_distance_metrics))
        return lambda coil_dofs, solved_x, solver_success, present, raw_terms: (
            "cached",
            include_distance_metrics,
            coil_dofs,
            solved_x,
            solver_success,
            present,
            raw_terms,
        )

    monkeypatch.setattr(
        _traceable,
        "_make_traceable_reporting_metrics_from_solution",
        make_uncached,
    )
    monkeypatch.setattr(
        _traceable,
        "_make_traceable_reporting_metrics_from_solution_with_raw_terms",
        make_cached,
    )
    reporting = _traceable._make_traceable_reporting_metrics_from_solution_bundle(
        object()
    )

    assert build_calls == [
        ("uncached", True),
        ("uncached", False),
        ("cached", True),
        ("cached", False),
    ]
    assert reporting("coils", "state", True, include_distance_metrics=False) == (
        "uncached",
        False,
        "coils",
        "state",
        True,
    )
    raw_terms = _raw_terms()
    cached = reporting(
        "coils",
        "state",
        True,
        outer_raw_terms=(True, raw_terms),
    )
    assert cached[:6] == ("cached", True, "coils", "state", True, True)
    assert cached[6] is raw_terms


def test_public_reporting_boundary_stages_cached_terms_on_solution_device():
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu_devices:
        pytest.skip("CUDA device required for strict placement coverage")
    device = gpu_devices[0]
    observed = {}

    def reporting(
        coil_dofs,
        solved_x,
        solver_success,
        *,
        include_distance_metrics,
        outer_raw_terms,
    ):
        observed.update(
            coil_dofs=coil_dofs,
            solved_x=solved_x,
            solver_success=solver_success,
            include_distance_metrics=include_distance_metrics,
            outer_raw_terms=outer_raw_terms,
        )
        return solved_x

    boundary = _traceable._make_traceable_lazy_reporting_metrics_from_solution_boundary(
        {"reporting_metrics_from_solution": reporting}
    )
    solved_x = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64), device)
    raw_terms = {
        term_name: np.asarray(index + 1.0, dtype=np.float64)
        for index, (term_name, _weight_key) in enumerate(
            _traceable._TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS
        )
    }

    with jax.transfer_guard("disallow"):
        result = boundary(
            np.asarray([0.5], dtype=np.float64),
            solved_x,
            True,
            include_distance_metrics=False,
            outer_raw_terms=(True, raw_terms),
        )
        result.block_until_ready()

    assert observed["include_distance_metrics"] is False
    present, staged_terms = observed["outer_raw_terms"]
    assert present.devices() == {device}
    assert all(term.devices() == {device} for term in staged_terms.values())
