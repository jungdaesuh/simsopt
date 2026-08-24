"""The verdict renderer reads receipts; it never recalls or invents a number."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.nested_ls_outer_verdict import VerdictInputError, render

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "docs" / "receipts" / "evidence"


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sweep(head: str = "abc") -> dict:
    return {
        "git_head": head,
        "claim_boundary": {},
        "best_omp_num_threads": 14,
        "per_omp_min_process_wall_seconds": {
            "8": 1117.36,
            "14": 1044.71,
            "16": 1060.15,
        },
    }


def _b3(head: str = "abc") -> dict:
    return {
        "git_head": head,
        "speedup_min_over_min": 2.2922,
        "native_min_process_wall_seconds": 1606.5,
        "native_median_process_wall_seconds": 1702.71,
        "native_max_process_wall_seconds": 2563.82,
        "jax_min_process_wall_seconds": 700.86,
        "jax_median_process_wall_seconds": 748.93,
        "jax_max_process_wall_seconds": 761.38,
        "claim_boundary": {
            "measured_j_rel_gap_max": 0.0,
            "nested_speed_claim": True,
            "native_omp_num_threads": 14,
            "omp_provenance": "swept_artifact",
        },
    }


def test_a_missing_receipt_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(VerdictInputError, match="sweep receipt does not exist"):
        render(tmp_path / "absent.json", _write(tmp_path / "b3.json", _b3()), None)


def test_an_unreadable_receipt_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    bad = tmp_path / "sweep.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(VerdictInputError, match="unreadable"):
        render(bad, _write(tmp_path / "b3.json", _b3()), None)


def test_receipts_from_different_commits_are_refused(tmp_path: Path) -> None:
    """The failure this check exists for.

    Each gate binds to the one above it pairwise, but this is the one place a
    human reads all three at once. A verdict assembled across commits would
    describe a program that never existed, and every individual number in it
    would be real -- which is exactly what makes it dangerous.
    """

    with pytest.raises(VerdictInputError, match="disagree on git_head"):
        render(
            _write(tmp_path / "sweep.json", _sweep(head="aaa")),
            _write(tmp_path / "b3.json", _b3(head="bbb")),
            None,
        )


def test_a_missing_b37_says_not_run_rather_than_placeholding(tmp_path: Path) -> None:
    out = render(
        _write(tmp_path / "sweep.json", _sweep()),
        _write(tmp_path / "b3.json", _b3()),
        None,
    )
    assert "**Not run.**" in out
    assert "0.0" not in out.split("### B37")[1]


def test_every_printed_number_comes_from_the_receipt(tmp_path: Path) -> None:
    out = render(
        _write(tmp_path / "sweep.json", _sweep()),
        _write(tmp_path / "b3.json", _b3()),
        None,
    )
    assert "2.2922" in out
    assert "1044.7" in out and "**(best)**" in out
    assert "swept_artifact" in out
    # The scope disclaimers are not optional garnish: a nested number quoted
    # without them reads as an F3 claim.
    assert "single-state" in out
    assert "trajectory_is_stock" in out


def test_a_receipt_missing_a_required_field_is_refused(tmp_path: Path) -> None:
    b3 = _b3()
    del b3["claim_boundary"]["nested_speed_claim"]
    with pytest.raises(VerdictInputError, match="nested_speed_claim"):
        render(
            _write(tmp_path / "sweep.json", _sweep()),
            _write(tmp_path / "b3.json", b3),
            None,
        )
