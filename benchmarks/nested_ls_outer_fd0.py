"""Gate FD-0 of the eight-term nested-LS outer charter.

Freezes the dense-LU walk endpoint, evaluates the outer gradient there,
and central-differences the eight-term ``J`` along all 11 coil unit
directions at the frozen per-direction step and at half of it. Every
perturbed inner re-solve is rejudged by the C++ LS Newton judge. Physics
gate only: no timing content, no speed claim, not F3 7.70x.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    DEFAULT_F3_B37_GPU_LANE,
    NestedLsOuterFd0Probe,
    evaluate_f3_b37_outer_fd0_probe,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
)

from benchmarks.nested_ls_shamanskii_attribution import (
    git_implementation_dirty,
    write_strict_json,
)

EVIDENCE = REPO / "docs" / "receipts" / "evidence"
PUBLICATION = (
    "Gate FD-0: all 11 coil-direction central differences of the eight-term "
    "outer J against the Schur adjoint gradient at the dense-LU walk "
    "endpoint, step halved. Physics gate, no timing content, not F3 7.70x."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate FD-0 outer gradient probe.")
    parser.add_argument(
        "--tag",
        default="",
        help="Receipt suffix, e.g. a100 -> nested_ls_outer_fd0_20260823.a100.json",
    )
    return parser.parse_args(argv)


def _require_clean_tree() -> str:
    dirty = git_implementation_dirty().strip()
    if dirty:
        raise SystemExit(
            f"Gate FD-0 requires a clean tree (implementation, not evidence):\n{dirty}"
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()


def _log_lines(probe: NestedLsOuterFd0Probe, git_head: str) -> list[str]:
    """A header, then one line per coil direction and per perturbed solve."""

    lines = [
        f"driver benchmarks.nested_ls_outer_fd0 git_head {git_head}",
        f"step_rule {probe.step_rule}",
        (
            f"base J={probe.objective!r}"
            f" ||g||_2={probe.outer_gradient_l2!r}"
            f" live_eta={probe.adjoint_live_eta!r}"
            f" tol={probe.adjoint_live_eta_tol!r}"
            f" mixed_form_max_abs={probe.mixed_form_max_abs_difference!r}"
        ),
    ]
    for row in probe.rows:
        lines.append(
            f"dir={row['index']} coil={row['coil_value']!r}"
            f" eps={row['epsilon']!r} eps_half={row['epsilon_half']!r}"
            f" pred={row['predicted_dot']!r} fd={row['fd_dot']!r}"
            f" fd_half={row['fd_dot_half']!r} rel={row['rel_error']!r}"
            f" rel_half={row['rel_error_half']!r}"
            f" halved_better={row['halving_reduced_error']}"
            f" pass={row['direction_pass']}"
        )
        evaluations = row["evaluations"]
        for evaluation in evaluations if isinstance(evaluations, tuple) else ():
            lines.append(
                f"  dir={evaluation['index']} step={evaluation['requested_step']!r}"
                f" realized={evaluation['realized_step']!r}"
                f" J={evaluation['objective']!r}"
                f" inner_iter={evaluation['inner_iterations']}"
                f" inner_grad={evaluation['inner_grad_l2']!r}"
                f" rejudge_iter={evaluation['rejudge_iter']}"
                f" rejudge_grad={evaluation['rejudge_grad_l2']!r}"
                f" step_exact={evaluation['rejudge_coil_step_exact']}"
            )
    lines.append(
        f"directions_passed {probe.directions_passed}/{probe.directions}"
        f" worst_rel={probe.worst_rel_error!r}"
        f" fail_closed_reason {probe.fail_closed_reason!r}"
    )
    return lines


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tag = str(args.tag).strip()
    suffix = f".{tag}" if tag else ""
    out_json = EVIDENCE / f"nested_ls_outer_fd0_20260823{suffix}.json"
    out_log = EVIDENCE / f"nested_ls_outer_fd0_20260823{suffix}.log"
    sha = _require_clean_tree()
    coils, surface, lane = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    del _target
    probe = evaluate_f3_b37_outer_fd0_probe(jax_boozer, native)
    payload: dict[str, object] = {
        "claim_boundary": {
            "cap_2048_attempted": False,
            "comparable_operators": False,
            "explicit_inverse_m_production": False,
            "f3_sealed": True,
            "gate": "fd0",
            "inherits_f3_7_70x": False,
            "moving_coil_outer_loop": False,
            "nested_speed_claim": False,
            "physics_gate_only": True,
            "tag": tag or None,
            "timing_content": False,
        },
        "command": (
            "SIMSOPT_BACKEND_MODE=jax_gpu_fast JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 "
            ".venv-qn-gpu/bin/python benchmarks/nested_ls_outer_fd0.py"
            + (f" --tag {tag}" if tag else "")
        ),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "driver": "benchmarks.nested_ls_outer_fd0",
        "execution_log": str(out_log.relative_to(REPO)),
        "fail_closed_reason": probe.fail_closed_reason,
        "git_head": sha,
        "lane": lane,
        "probe": probe.as_payload(),
        "publication": PUBLICATION,
        "schema": "nested-ls-outer-fd0.v1",
        "written_by_pytest": False,
    }
    write_strict_json(out_json, payload)
    out_log.write_text("\n".join(_log_lines(probe, sha)) + "\n", encoding="utf-8")
    print("wrote", out_json, flush=True)
    print("wrote", out_log, flush=True)
    print(
        "fd0_ok",
        probe.fail_closed_reason is None,
        "passed",
        probe.directions_passed,
        "worst_rel",
        probe.worst_rel_error,
        flush=True,
    )
    if probe.fail_closed_reason is not None:
        raise SystemExit(f"Gate FD-0 failed closed: {probe.fail_closed_reason}")


if __name__ == "__main__":
    main()
