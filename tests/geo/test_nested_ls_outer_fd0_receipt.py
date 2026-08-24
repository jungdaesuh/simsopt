"""Receipt identity for the stock and predictor FD-0 lanes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import nested_ls_outer_fd0 as fd0
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_OUTER_FD0_SCHEMA,
    nested_ls_jax_inner_policy,
)


def _probe() -> SimpleNamespace:
    return SimpleNamespace(
        adjoint_live_eta=0.0,
        adjoint_live_eta_tol=1.0,
        as_payload=lambda: {"directions": 11},
        base_objective_scatter=0.0,
        base_objectives=(1.0,),
        directions=11,
        directions_passed=11,
        fail_closed_reason=None,
        inner_predictor_enabled=False,
        min_step_rule="scatter_bounded",
        mixed_form_max_abs_difference=0.0,
        objective=1.0,
        outer_gradient_l2=1.0,
        rows=(),
        step_rule="descent_ladder",
        worst_rel_error=1.0e-8,
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    predictor: bool,
) -> tuple[Path, dict[str, object]]:
    writes: list[tuple[Path, dict[str, object]]] = []
    heads = iter(("a" * 40, "a" * 40))
    monkeypatch.setattr(fd0, "REPO", tmp_path)
    monkeypatch.setattr(fd0, "EVIDENCE", tmp_path)
    monkeypatch.setattr(fd0, "_require_clean_tree", lambda: next(heads))
    monkeypatch.setattr(
        fd0, "load_flat675_lane_blocks", lambda _lane: (object(), object(), "lane")
    )
    monkeypatch.setattr(
        fd0,
        "load_archived_nested_ls_pair",
        lambda **_kwargs: (object(), object(), object()),
    )
    monkeypatch.setattr(
        fd0, "evaluate_f3_b37_outer_fd0_probe", lambda *_a, **_k: _probe()
    )
    monkeypatch.setattr(
        fd0,
        "write_strict_json",
        lambda path, payload: writes.append((path, payload)),
    )
    argv = ["--inner-predictor"] if predictor else []
    fd0.main(argv)
    assert len(writes) == 1
    return writes[0]


def test_predictor_and_stock_receipts_have_distinct_sealed_identities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stock_path, stock = _run(monkeypatch, tmp_path, predictor=False)
    predictor_path, predictor = _run(monkeypatch, tmp_path, predictor=True)

    assert stock_path.name == "nested_ls_outer_fd0_20260823.json"
    assert predictor_path.name == "nested_ls_outer_fd0_20260823.predictor.json"
    assert stock["schema"] == predictor["schema"] == NESTED_LS_OUTER_FD0_SCHEMA
    assert "--inner-predictor" not in str(stock["command"])
    assert "--inner-predictor" in str(predictor["command"])
    assert stock["inner_policy"] == nested_ls_jax_inner_policy(
        ift_stab=fd0.F3_B37_IFT_STAB,
        inner_substep_legs=(1,),
        inner_predictor=False,
    )
    assert predictor["inner_policy"] == nested_ls_jax_inner_policy(
        ift_stab=fd0.F3_B37_IFT_STAB,
        inner_substep_legs=(1,),
        inner_predictor=True,
    )
    assert stock["claim_boundary"]["inner_predictor_enabled"] is False
    assert predictor["claim_boundary"]["inner_predictor_enabled"] is True


def test_fd0_refuses_to_publish_if_head_changes_during_the_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    heads = iter(("a" * 40, "b" * 40))
    monkeypatch.setattr(fd0, "REPO", tmp_path)
    monkeypatch.setattr(fd0, "EVIDENCE", tmp_path)
    monkeypatch.setattr(fd0, "_require_clean_tree", lambda: next(heads))
    monkeypatch.setattr(
        fd0, "load_flat675_lane_blocks", lambda _lane: (object(), object(), "lane")
    )
    monkeypatch.setattr(
        fd0,
        "load_archived_nested_ls_pair",
        lambda **_kwargs: (object(), object(), object()),
    )
    monkeypatch.setattr(
        fd0, "evaluate_f3_b37_outer_fd0_probe", lambda *_a, **_k: _probe()
    )
    writes: list[Path] = []
    monkeypatch.setattr(
        fd0, "write_strict_json", lambda path, _payload: writes.append(path)
    )

    with pytest.raises(SystemExit, match="HEAD changed during Gate FD-0"):
        fd0.main([])
    assert writes == []
