"""Source-fidelity checks for the Boozer-QA mirror."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from conftest import ast_names_used, enable_non_strict_jax_backend
from examples.jax.parity.cases import get_case, native_boozerqa
from examples.jax.parity.cases.native_boozerqa import BoozerSingleStageSpec
from examples.jax.parity.input_bundle import load_input_bundle
from simsopt_jax_adapters.geo import boozer_qa_problem
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_QA_NEWTON_MAXITER,
    OFFICIAL_QA_NON_QS_RESOLUTION,
    OFFICIAL_QA_PENALTY_WEIGHT,
    boozer_qa_outer_objective_config,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
#: The public script is not an importable module, so it is the one path that stays
#: repository-relative; the owner and the case come from the modules actually imported.
_MIRROR = _REPOSITORY_ROOT / "examples/jax/2_Intermediate/boozerQA.py"
_OWNER = Path(boozer_qa_problem.__file__)
_PARITY_CASE = Path(native_boozerqa.__file__)


def test_boozerqa_mirror_uses_memory_bounded_host_outer_solve() -> None:
    source = _MIRROR.read_text(encoding="utf-8")
    owner = _OWNER.read_text(encoding="utf-8")
    parity_case = _PARITY_CASE.read_text(encoding="utf-8")

    # Upstream re-solves the Boozer surface with the dense analytic Newton route
    # on every objective evaluation (``solve_residual_equation_exactly_newton``),
    # so the mirror runs the certified analytic evaluator over the NumPy-baking
    # Boozer adapter, and it runs it through the single owner of that workflow --
    # the same owner the matched parity case drives.
    assert "BoozerQAProblem" in source
    assert "from simsopt_jax_adapters.geo.boozer_qa_problem import" in source
    assert "BoozerQAProblem" in parity_case
    assert "from simsopt_jax_adapters.geo.boozer_qa_problem import" in parity_case
    assert "HostConstructionBoozerSurfaceJAX" in owner
    assert "ExactAnalyticSingleStage" in owner
    # The official script drives one host BFGS solve over the device objective
    # (upstream examples/2_Intermediate/boozerQA.py:143-145,
    # `minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': 1e3},
    # tol=1e-15)`), so the mirror's outer method is fixed by the official
    # workflow and must not follow the execution mode's default driver.
    assert "scalar_example_driver" not in source
    assert "minimize_lbfgs_host_core" not in source
    assert "minimize_bfgs_host_core" in source
    assert "line_search_value_and_grad_more_thuente_host" in source
    assert "serial_solve_jax" not in source
    # The four official penalty weights and the whole outer objective dictionary
    # have one owner (simsopt_jax.examples.boozer_official), which the workflow
    # owner reads once for both callers; neither the script nor the case may
    # restate them.
    assert "boozer_qa_outer_objective_config" in owner
    assert not any(
        "non_qs_weight" in name for name in ast_names_used(ast.parse(source))
    )
    assert not any("non_qs_weight" in name for name in ast_names_used(ast.parse(owner)))
    configuration = boozer_qa_outer_objective_config(
        nfp=3,
        non_qs_resolution=OFFICIAL_QA_NON_QS_RESOLUTION,
        length_target=1.0,
        major_radius_target=1.0,
        vessel_gamma=[[[0.0, 0.0, 0.0]]],
    )
    assert configuration["non_qs_weight"] == OFFICIAL_QA_PENALTY_WEIGHT == 1.0
    assert configuration["iota_weight"] == OFFICIAL_QA_PENALTY_WEIGHT
    assert configuration["major_radius_weight"] == OFFICIAL_QA_PENALTY_WEIGHT
    assert configuration["length_weight"] == OFFICIAL_QA_PENALTY_WEIGHT
    assert configuration["residual_weight"] == 0.0
    assert '"newton_maxiter": OFFICIAL_QA_NEWTON_MAXITER' in source
    assert OFFICIAL_QA_NEWTON_MAXITER == 20
    assert (
        'OFFICIAL_QA_NON_QS_RESOLUTION if scale == "native_default" else 4'
    ) in source
    assert OFFICIAL_QA_NON_QS_RESOLUTION == 20
    assert "bounded_steps=2" in source
    assert "scipy" not in source.lower()


def test_official_jax_lane_runs_the_owner_and_never_the_traceable_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """The official JAX lane must not reach the traceable-session route at all.

    Source text is not coverage: the session route stays in this module for
    ``execute_variant``, so a test that only greps for it would pass while the
    official lane ran something else.  This one fails the run if the official lane builds the
    session runtime or calls ``_jax``.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")
    case = get_case("native-boozerqa")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _stored, arrays = load_input_bundle(input_root, bundle)

    def forbidden(*_arguments: object, **_keywords: object) -> object:
        raise AssertionError("the official JAX lane must run the analytic owner")

    monkeypatch.setattr(native_boozerqa, "_prepare_jax_variant_runtime", forbidden)
    monkeypatch.setattr(native_boozerqa, "_jax", forbidden)

    observation = case.execute("jax-cpu", bundle, arrays)

    assert observation.normalized_status == "budget_exhausted"
    assert bool(observation.values["final:inner_solver_success"])
    assert set(observation.values) >= {
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


def test_execute_variant_still_reaches_the_session_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every caller other than the official lane keeps the session route.

    ``execute_variant`` must still dispatch JAX lanes to ``_jax``.
    """
    calls: list[tuple[str, str]] = []

    def recording_jax(
        lane: str,
        bundle: object,
        arrays: object,
        spec: object,
    ) -> str:
        calls.append((lane, str(getattr(spec, "case_id", spec))))
        return "session-observation"

    monkeypatch.setattr(native_boozerqa, "_jax", recording_jax)

    assert (
        native_boozerqa.execute_variant(
            "jax-cpu", None, {}, native_boozerqa.BOOZER_QA_SPEC
        )
        == "session-observation"
    )
    assert calls == [("jax-cpu", native_boozerqa.BOOZER_QA_SPEC.case_id)]


def test_native_lane_still_reaches_the_variant_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native lane is the parity reference: it must keep ``execute_variant``.

    ``execute`` now dispatches on the lane; a mis-route of ``native-cpu`` into
    the analytic owner would compare JAX against JAX and still pass the mirror
    parity test, so the native branch is pinned here as well.
    """
    calls: list[tuple[str, str]] = []

    def recording_variant(
        lane: str, bundle: object, arrays: object, spec: BoozerSingleStageSpec
    ) -> str:
        calls.append((lane, spec.case_id))
        return "variant-observation"

    def forbidden(*_arguments: object, **_keywords: object) -> object:
        raise AssertionError("the native lane must not run the analytic owner")

    monkeypatch.setattr(native_boozerqa, "execute_variant", recording_variant)
    monkeypatch.setattr(native_boozerqa, "_execute_official_jax", forbidden)

    assert native_boozerqa.execute("native-cpu", None, {}) == "variant-observation"
    assert calls == [("native-cpu", native_boozerqa.BOOZER_QA_SPEC.case_id)]


@pytest.mark.parametrize("lane", ("jax-cpu", "jax-gpu"))
def test_both_jax_lanes_reach_the_analytic_owner(
    monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """Both JAX lanes take the official route; neither falls back to the session."""
    calls: list[str] = []

    def recording_official(lane: str, bundle: object, arrays: object) -> str:
        calls.append(lane)
        return "official-observation"

    def forbidden(*_arguments: object, **_keywords: object) -> object:
        raise AssertionError("a JAX lane must not run the variant route")

    monkeypatch.setattr(native_boozerqa, "_execute_official_jax", recording_official)
    monkeypatch.setattr(native_boozerqa, "execute_variant", forbidden)

    assert native_boozerqa.execute(lane, None, {}) == "official-observation"
    assert calls == [lane]
