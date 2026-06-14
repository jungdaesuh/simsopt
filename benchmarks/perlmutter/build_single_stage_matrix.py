#!/usr/bin/env python3
"""Generate the single-stage 11-vs-51 production test matrix manifest.

The single-stage formulation dimension is hard-coupled to the outer optimizer
backend (no flag decouples them), so the achievable cells are:

- 11 reduced (coils only, surface solved each iteration): ``scipy-jax``
  (host SciPy L-BFGS-B over the differentiable inner Boozer solve).
- 51 full-space (51-dim coils+surface vector via the full-graph DOF map):
  ``ondevice`` (on-device JAX L-BFGS). Only the coil block carries outer
  gradient; the surface is re-solved each iteration by the same inner Boozer
  solve as the 11 reduced lane (``run_code_traceable``), not a residual penalty.

Every parity run also produces the native cpp/CPU reference child, which the
harness always drives at 51 full-space DOFs (there is no coil-only native
reference in this benchmark). So an 11-dim JAX lane is compared against a
51-dim cpp reference unless an external 11-dim native reference is supplied.

The inner Boozer least-squares algorithm (``quasi-newton`` / ``lm`` /
``lm-minpack``) reaches the child via the ``BOOZER_LEAST_SQUARES_ALGORITHM``
env var; the launchers export it from ``PROD_BOOZER_LS_ALGORITHM``. It is
documented to apply to the on-device (ondevice) lane; its effect on the
host-SciPy scipy-jax reduced lane is unvalidated (may fall back to
quasi-newton), so those cells are marked extended.

This script emits the manifest as JSON and a readable Markdown table. It does
not submit anything; ``submit_single_stage_matrix.py`` consumes the JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = "benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json"

BUDGETS = {
    "outer_lbfgs_maxiter": 1500,
    "target_lane_boozer_bfgs_maxiter": 1500,
    "target_lane_boozer_newton_maxiter": 50,
    "target_lane_boozer_newton_polish_policy": "run",
}

# User naming -> child --boozer-least-squares-algorithm value.
INNER_LS = {
    "quasinewton": "quasi-newton",
    "LM": "lm",
    "LS": "lm-minpack",
}

# Formulation dim is fixed by the optimizer backend (verified against the
# contract predicates in single_stage_banana_example.py).
FORMULATIONS = {
    11: {
        "optimizer_backend": "scipy-jax",
        "outer_optimizer": "host-scipy",
        "description": "reduced coil-only, Boozer surface solved each iteration",
        "inner_ls_applies": False,  # scipy-jax reduced lane: inner-LS unvalidated
    },
    51: {
        "optimizer_backend": "ondevice",
        "outer_optimizer": "ondevice-jax",
        "description": "51-dim coils+surface vector (full-graph DOF map); outer gradient on the coil block only, surface re-solved each iteration by the same inner Boozer solve as the reduced lane (not a residual penalty)",
        "inner_ls_applies": True,
    },
}

PLATFORMS = {
    "cpu": {"launcher": "single_stage_production_cpu.slurm", "account": "m4680",
            "sbatch_extra": "-q regular --exclusive --mem=0"},
    "gpu": {"launcher": "single_stage_production_gpu.slurm", "account": "m4680_g",
            "sbatch_extra": "--mem-per-cpu=3592M"},
}

TIERS = {
    "mpol2": {
        "mpol": 2, "ntor": 2, "nphi": 31, "ntheta": 16,
        "warm_start": None, "binds_budget": False, "runnable_now": True,
        "note": "smoke resolution: loose tolerances (gtol 1e-2), budgets do not "
                "bind, optimizers stop in ~5-219 steps; cheap diagnostics tier",
    },
    "mpol10": {
        "mpol": 10, "ntor": 10, "nphi": 64, "ntheta": 32,
        "warm_start": "<continuation-donor>", "binds_budget": True,
        "runnable_now": False,
        "note": "production resolution: tolerances tighten (gtol 1e-7), 1500/50 "
                "budgets bind; requires --warm-start-run-dir from the donor "
                "build (2->4->6->8->10 ladder); smaller than mpol12 so the "
                "ondevice-51 dense-Newton graph is more likely to fit, though it "
                "may still need polish-policy skip",
    },
}


def build_cells() -> list[dict]:
    cells: list[dict] = []
    for dim, fdef in FORMULATIONS.items():
        for ls_name, ls_value in INNER_LS.items():
            for platform in PLATFORMS:
                for tier in TIERS:
                    # quasi-newton is the child default; only set the env for
                    # the explicit lm / lm-minpack opt-ins.
                    prod_ls = "" if ls_value == "quasi-newton" else ls_value
                    if dim == 51:
                        status = "core"
                    else:
                        status = "core" if ls_value == "quasi-newton" else "extended"
                    reason = None
                    if status == "extended":
                        reason = ("inner-LS effect on the scipy-jax reduced lane "
                                  "is unvalidated; may fall back to quasi-newton")
                    cell_id = f"ss_{dim}_{fdef['optimizer_backend'].replace('-', '')}_{platform}_{ls_name}_{tier}"
                    cells.append({
                        "id": cell_id,
                        "formulation_dim": dim,
                        "formulation": fdef["description"],
                        "optimizer_backend": fdef["optimizer_backend"],
                        "outer_optimizer": fdef["outer_optimizer"],
                        "platform": platform,
                        "lane": f"jax-{platform}",
                        "inner_ls_name": ls_name,
                        "inner_ls_value": ls_value,
                        "tier": tier,
                        "reference_dim": 51,
                        "reference_lane": "cpp-cpu",
                        "dim_matched_reference": dim == 51,
                        "status": status,
                        "status_reason": reason,
                        "launcher": PLATFORMS[platform]["launcher"],
                        "account": PLATFORMS[platform]["account"],
                        "sbatch_extra": PLATFORMS[platform]["sbatch_extra"],
                        "env": {
                            "PROD_OPTIMIZER_BACKEND": fdef["optimizer_backend"],
                            "PROD_BOOZER_LS_ALGORITHM": prod_ls,
                            "PROD_MAXITER": str(BUDGETS["outer_lbfgs_maxiter"]),
                            "PROD_BOOZER_BFGS_MAXITER": str(BUDGETS["target_lane_boozer_bfgs_maxiter"]),
                            "PROD_NEWTON_MAXITER": str(BUDGETS["target_lane_boozer_newton_maxiter"]),
                            "PROD_NEWTON_POLISH_POLICY": BUDGETS["target_lane_boozer_newton_polish_policy"],
                            "PROD_MPOL": str(TIERS[tier]["mpol"]),
                            "PROD_NTOR": str(TIERS[tier]["ntor"]),
                            "PROD_NPHI": str(TIERS[tier]["nphi"]),
                            "PROD_NTHETA": str(TIERS[tier]["ntheta"]),
                        },
                        "depends_on": None if TIERS[tier]["runnable_now"]
                        else "continuation donor build (job 54363243)",
                    })
    return cells


def build_manifest(source_sha: str) -> dict:
    return {
        "matrix_id": "single_stage_11_51_2026-06-13",
        "source_sha": source_sha,
        "seed": SEED,
        "budgets": BUDGETS,
        "inner_ls_naming": INNER_LS,
        "formulation_backend_coupling": {
            str(dim): FORMULATIONS[dim]["optimizer_backend"] for dim in FORMULATIONS
        },
        "notes": [
            "11=scipy-jax (reduced), 51=ondevice (full); coupling is not "
            "overridable by any flag.",
            "Every run also yields the cpp/CPU reference at 51 full-space; "
            "there is no coil-only native reference in this harness, so the "
            "11-dim JAX lane has a dim-mismatched cpp reference.",
            "optax-lbfgs and optimistix-lbfgs are intentionally excluded.",
            "lm/lm-minpack export BOOZER_LEAST_SQUARES_ALGORITHM to BOTH the "
            "target and reference children; the cpp reference uses the native "
            "C++ Boozer solver and is expected to ignore it, but the first "
            "lm/lm-minpack run must confirm the reference child tolerates it.",
        ],
        "tiers": TIERS,
        "cells": build_cells(),
    }


def render_markdown(manifest: dict) -> str:
    lines = [
        "# Single-stage 11-vs-51 production test matrix",
        "",
        f"- Matrix id: `{manifest['matrix_id']}`",
        f"- Source SHA: `{manifest['source_sha']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Budgets: outer L-BFGS {manifest['budgets']['outer_lbfgs_maxiter']}, "
        f"Boozer BFGS {manifest['budgets']['target_lane_boozer_bfgs_maxiter']}, "
        f"Boozer Newton {manifest['budgets']['target_lane_boozer_newton_maxiter']}, "
        f"polish `{manifest['budgets']['target_lane_boozer_newton_polish_policy']}`",
        "- Inner-LS naming: "
        + ", ".join(f"{k}=`{v}`" for k, v in manifest["inner_ls_naming"].items()),
        "",
        "## Formulation/backend coupling",
        "",
        "- `11` reduced (coils only, surface solved) ⇒ `scipy-jax` (host SciPy).",
        "- `51` 51-dim coils+surface vector (full-graph DOF map) ⇒ `ondevice` (on-device JAX; outer gradient on the coil block only, surface re-solved by the same inner Boozer solve as the 11 lane, not a residual penalty).",
        "- The cpp/CPU reference is always 51; the 11-dim JAX lane is compared "
        "against a dim-mismatched 51 reference.",
        "",
        "## Cells",
        "",
        "| id | dim | backend | platform | inner-LS | tier | status | ref dim match | runnable now |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in manifest["cells"]:
        runnable = "yes" if manifest["tiers"][c["tier"]]["runnable_now"] else "no (donor)"
        lines.append(
            f"| `{c['id']}` | {c['formulation_dim']} | {c['optimizer_backend']} | "
            f"{c['platform']} | {c['inner_ls_name']}=`{c['inner_ls_value']}` | "
            f"{c['tier']} | {c['status']} | "
            f"{'yes' if c['dim_matched_reference'] else 'NO (51 vs 11)'} | {runnable} |"
        )
    core = sum(1 for c in manifest["cells"] if c["status"] == "core")
    ext = sum(1 for c in manifest["cells"] if c["status"] == "extended")
    lines += [
        "",
        f"Total cells: {len(manifest['cells'])} ({core} core, {ext} extended).",
        "",
        "## Notes",
        "",
        *[f"- {n}" for n in manifest["notes"]],
        "",
        "## Tier detail",
        "",
    ]
    for tier, t in manifest["tiers"].items():
        lines.append(
            f"- `{tier}`: mpol={t['mpol']} ntor={t['ntor']} nphi={t['nphi']} "
            f"ntheta={t['ntheta']}, binds_budget={t['binds_budget']}, "
            f"runnable_now={t['runnable_now']}. {t['note']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True,
                        help="Clean source SHA the matrix targets.")
    parser.add_argument("--out-json", default="docs/single_stage_11_51_matrix_2026-06-13.json")
    parser.add_argument("--out-md", default="docs/single_stage_11_51_matrix_2026-06-13.md")
    args = parser.parse_args()

    manifest = build_manifest(args.source_sha)
    json_path = REPO_ROOT / args.out_json
    md_path = REPO_ROOT / args.out_md
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(manifest), encoding="utf-8")
    print(f"wrote {json_path} ({len(manifest['cells'])} cells)")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
