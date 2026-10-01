"""Each BoozerQA parity lane runs its own workflow owner, end to end."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import enable_non_strict_jax_backend
from examples.jax.parity.cases import get_case, native_boozerqa
from examples.jax.parity.input_bundle import load_input_bundle
from simsopt_jax.examples.single_stage_boozer_vacuum import JAX_PARITY_DRIVER_ID

_PUBLISHED_OBSERVABLES = frozenset(
    {
        "construction:surface_dofs",
        "construction:coil_dofs",
        "initial:parameters",
        "initial:objective",
        "initial:gradient",
        "initial:iota",
        "initial:volume",
        "final:parameters",
        "final:objective",
        "final:gradient",
        "final:non_qs_ratio",
        "final:iota",
        "final:volume",
        "final:major_radius_penalty",
        "final:length_penalty",
        "final:inner_solver_success",
        "final:outer_solver_success",
    }
)


def _forbidden(message: str):
    def forbidden(*_arguments: object, **_keywords: object) -> object:
        raise AssertionError(message)

    return forbidden


def _bounded_input(tmp_path: Path):
    case = get_case("native-boozerqa")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _stored, arrays = load_input_bundle(input_root, bundle)
    return case, bundle, arrays


def test_official_jax_lane_runs_the_owner_and_never_the_traceable_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """The official JAX lane must not reach the traceable-session route at all.

    The session route stays in this module for ``execute_variant``; this run
    fails if the official lane builds the session runtime or calls ``_jax``.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")
    case, bundle, arrays = _bounded_input(tmp_path)
    message = "the official JAX lane must run the analytic owner"
    monkeypatch.setattr(
        native_boozerqa, "_prepare_jax_variant_runtime", _forbidden(message)
    )
    monkeypatch.setattr(native_boozerqa, "_jax", _forbidden(message))

    observation = case.execute("jax-cpu", bundle, arrays)

    assert observation.driver == JAX_PARITY_DRIVER_ID
    assert observation.normalized_status == "budget_exhausted"
    assert bool(observation.values["final:inner_solver_success"])
    assert set(observation.values) >= _PUBLISHED_OBSERVABLES


def test_native_lane_runs_the_native_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The native lane is the parity reference and never the analytic owner.

    A mis-route of ``native-cpu`` into the JAX owner would compare JAX against
    JAX and still pass the mirror parity test.
    """
    case, bundle, arrays = _bounded_input(tmp_path)
    monkeypatch.setattr(
        native_boozerqa,
        "_execute_official_jax",
        _forbidden("the native lane must not run the analytic owner"),
    )

    observation = case.execute("native-cpu", bundle, arrays)

    assert observation.backend_mode == "native_cpu"
    assert observation.driver == "simsopt_scipy_bfgs_with_boozer_newton"
    assert observation.normalized_status == "budget_exhausted"
    assert set(observation.values) >= _PUBLISHED_OBSERVABLES
