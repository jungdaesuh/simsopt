#!/usr/bin/env python3
"""Generate the single-stage 11-vs-51 production test matrix manifest.

The single-stage formulation dimension is hard-coupled to the outer optimizer
backend (no flag decouples them). Both achievable cells drive the outer loop on
the host (SciPy L-BFGS-B), so neither compiles the whole optimization into one
XLA graph -- the monolithic ``ondevice`` lane that did so OOMs at production
resolution (422 GiB compile-time section memory) and is intentionally excluded:

- 11 reduced (coils only, surface solved each iteration by the inner Boozer
  solve): ``scipy-jax`` (host SciPy L-BFGS-B over the differentiable inner
  Boozer solve).
- 51 full-space (51-dim coils+surface vector via the full-graph ``JF.x`` DOF
  map): ``scipy-jax-fullgraph`` (host SciPy L-BFGS-B over the full DOF vector;
  outer gradient on the coil block only, the surface re-solved each iteration by
  the inner Boozer solve, not a residual penalty).

Every parity run also produces the native cpp/CPU reference child, which the
harness always drives at 51 full-space DOFs (there is no coil-only native
reference in this benchmark). So an 11-dim ``scipy-jax`` lane is compared against
a 51-dim cpp reference (performance/feasibility), while the 51-dim
``scipy-jax-fullgraph`` lane is dim-matched and replay-capable (exact
same-candidate bit-parity).

The inner Boozer least-squares solve is quasi-newton for both lanes; the LM /
lm-minpack variants are dropped (unvalidated on the host-SciPy lanes, and the
two lanes use different inner backends). quasi-newton is the child default, so
no ``BOOZER_LEAST_SQUARES_ALGORITHM`` override is exported.

Inner Boozer *optimizer* backend: under the jax_*_parity backend modes the inner
Boozer solve must run on-device (JAX). The reduced ``scipy-jax`` lane already
selects ``ondevice`` by the child default, but ``scipy-jax-fullgraph`` defaults
its inner Boozer to ``scipy`` when ``--boozer-optimizer-backend`` is omitted, and
the harness only auto-supplies ``ondevice`` on cuda -- not cpu
(``single_stage_init_parity.py`` ``_resolve_target_boozer_optimizer_backend``).
So the fullgraph cells set ``PROD_BOOZER_OPTIMIZER_BACKEND=ondevice`` explicitly;
without it the fullgraph CPU lane raises at ``boozer_surface.py:5659`` under
``jax_cpu_parity`` (confirmed by the b5f97fdf9 gate smoke, where the GPU lane --
which got ``ondevice`` via the cuda auto-default -- ran past the same point).

This script emits the manifest as JSON and a readable Markdown table. It does
not submit anything; ``submit_single_stage_matrix.py`` consumes the JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.benchmark_timing_labels import (
    MIXED_PARITY_REFERENCE,
    benchmark_timing_label,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = "benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json"

# Inner Boozer least-squares solve: quasi-newton for both lanes (no LM axis).
# quasi-newton is the child default, so PROD_BOOZER_LS_ALGORITHM stays empty and
# the launcher does not export BOOZER_LEAST_SQUARES_ALGORITHM.
INNER_BOOZER_LEAST_SQUARES = "quasi-newton"

BUDGETS = {
    "outer_lbfgs_maxiter": 1500,
    "target_lane_boozer_bfgs_maxiter": 1500,
    "target_lane_boozer_newton_maxiter": 50,
    "target_lane_boozer_newton_polish_policy": "run",
}

MIXED_PARITY_TIMING_NOTE = (
    "single_stage_init_parity.py co-produces target-lane and cpp/CPU "
    "reference timings; do not use the whole job wall time as a clean GPU "
    "headline."
)

# Formulation dim is fixed by the optimizer backend (verified against the
# contract predicates in single_stage_banana_example.py and the valid-backend
# set TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS in
# src/simsopt_jax/geo/_optimizer_backend_choices.py). Both lanes are host-driven
# SciPy control over a JAX-evaluated value/grad.
#
# inner_boozer_optimizer_backend: forces --boozer-optimizer-backend for the lane.
# Empty = rely on the child/harness default (the reduced scipy-jax lane already
# resolves to ondevice). "ondevice" = set it explicitly, required for the
# fullgraph lane on CPU (its inner Boozer otherwise defaults to scipy, which
# jax_cpu_parity rejects at boozer_surface.py:5659).
FORMULATIONS = {
    11: {
        "optimizer_backend": "scipy-jax",
        "outer_optimizer": "host-scipy",
        "inner_boozer_optimizer_backend": "",
        "description": "reduced coil-only; Boozer surface solved each iteration "
                       "by the inner solve; host SciPy L-BFGS-B outer loop",
    },
    51: {
        "optimizer_backend": "scipy-jax-fullgraph",
        "outer_optimizer": "host-scipy",
        "inner_boozer_optimizer_backend": "ondevice",
        "description": "51-dim coils+surface vector over the full-graph JF.x DOF "
                       "map; host SciPy L-BFGS-B outer loop; outer gradient on "
                       "the coil block only, surface re-solved each iteration by "
                       "the inner Boozer solve (not a residual penalty)",
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
                "bind, optimizers stop in ~5-219 steps; cheap diagnostics tier. "
                "The scipy-jax-fullgraph mpol2 cells double as the Phase 0 GATE "
                "smoke (inspect their result JSON for passed=true).",
    },
    "mpol10": {
        "mpol": 10, "ntor": 10, "nphi": 64, "ntheta": 32,
        "warm_start": "<continuation-donor>", "binds_budget": True,
        "runnable_now": False,
        "note": "production resolution: tolerances tighten (gtol 1e-7), 1500/50 "
                "budgets bind; requires --warm-start-run-dir from the "
                "continuation donor build (2->4->6->8->10 ladder via "
                "--intermediate-rungs). Both host-SciPy lanes compile only one "
                "value/grad eval per step, so node memory is bounded by a single "
                "eval rather than the whole optimization (no monolithic jit).",
    },
}


def build_cells() -> list[dict]:
    cells: list[dict] = []
    for dim, fdef in FORMULATIONS.items():
        for platform in PLATFORMS:
            for tier in TIERS:
                backend_slug = fdef["optimizer_backend"].replace("-", "")
                cell_id = f"ss_{dim}_{backend_slug}_{platform}_{tier}"
                timing_label = benchmark_timing_label(
                    MIXED_PARITY_REFERENCE,
                    includes_gpu_target=platform == "gpu",
                    includes_cpu_reference=True,
                    supports_performance_headline=False,
                    headline_timing_classification=None,
                    note=MIXED_PARITY_TIMING_NOTE,
                )
                cells.append({
                    "id": cell_id,
                    **timing_label,
                    "formulation_dim": dim,
                    "formulation": fdef["description"],
                    "optimizer_backend": fdef["optimizer_backend"],
                    "outer_optimizer": fdef["outer_optimizer"],
                    "inner_boozer_optimizer_backend": fdef["inner_boozer_optimizer_backend"],
                    "platform": platform,
                    "lane": f"jax-{platform}",
                    "inner_boozer_least_squares": INNER_BOOZER_LEAST_SQUARES,
                    "tier": tier,
                    "reference_dim": 51,
                    "reference_lane": "cpp-cpu",
                    "dim_matched_reference": dim == 51,
                    "status": "core",
                    "launcher": PLATFORMS[platform]["launcher"],
                    "account": PLATFORMS[platform]["account"],
                    "sbatch_extra": PLATFORMS[platform]["sbatch_extra"],
                    "env": {
                        "PROD_OPTIMIZER_BACKEND": fdef["optimizer_backend"],
                        # Inner Boozer optimizer backend: empty for the reduced
                        # lane (child default ondevice), "ondevice" forced for
                        # the fullgraph lane (see module docstring).
                        "PROD_BOOZER_OPTIMIZER_BACKEND": fdef["inner_boozer_optimizer_backend"],
                        # quasi-newton inner: leave empty so the launcher does not
                        # export BOOZER_LEAST_SQUARES_ALGORITHM (child default).
                        "PROD_BOOZER_LS_ALGORITHM": "",
                        "PROD_MAXITER": str(BUDGETS["outer_lbfgs_maxiter"]),
                        "PROD_BOOZER_BFGS_MAXITER": str(BUDGETS["target_lane_boozer_bfgs_maxiter"]),
                        "PROD_NEWTON_MAXITER": str(BUDGETS["target_lane_boozer_newton_maxiter"]),
                        "PROD_NEWTON_POLISH_POLICY": BUDGETS["target_lane_boozer_newton_polish_policy"],
                        "PROD_TIMING_CLASSIFICATION": timing_label["timing_classification"],
                        "PROD_SUPPORTS_PERFORMANCE_HEADLINE": "0",
                        "PROD_MPOL": str(TIERS[tier]["mpol"]),
                        "PROD_NTOR": str(TIERS[tier]["ntor"]),
                        "PROD_NPHI": str(TIERS[tier]["nphi"]),
                        "PROD_NTHETA": str(TIERS[tier]["ntheta"]),
                    },
                    "depends_on": None if TIERS[tier]["runnable_now"]
                    else "continuation donor warm-start build (native_cpu donor)",
                })
    return cells


def build_manifest(source_sha: str) -> dict:
    return {
        "matrix_id": "single_stage_11_51_2026-06-13",
        "source_sha": source_sha,
        "seed": SEED,
        "budgets": BUDGETS,
        "inner_boozer_least_squares": INNER_BOOZER_LEAST_SQUARES,
        "formulation_backend_coupling": {
            str(dim): FORMULATIONS[dim]["optimizer_backend"] for dim in FORMULATIONS
        },
        "notes": [
            "Both lanes run the outer loop on the host (SciPy L-BFGS-B); only "
            "one value/grad eval is compiled per step, so neither builds the "
            "monolithic on-device graph that OOMs. 11=scipy-jax (reduced), "
            "51=scipy-jax-fullgraph (full JF.x). The ondevice lane is removed.",
            "Formulation dim is hard-coupled to the optimizer backend; no flag "
            "decouples them.",
            "Every run also yields the cpp/CPU reference at 51 full-space; there "
            "is no coil-only native reference in this harness, so the 11-dim "
            "scipy-jax lane has a dim-mismatched 51 reference "
            "(performance/feasibility), while the 51-dim scipy-jax-fullgraph "
            "lane is dim-matched and replay-capable (exact same-candidate "
            "bit-parity).",
            "Inner Boozer least-squares is quasi-newton for both lanes; "
            "LM/lm-minpack and optax-lbfgs/optimistix-lbfgs are intentionally "
            "excluded.",
            "scipy-jax-fullgraph cells force --boozer-optimizer-backend ondevice "
            "(via PROD_BOOZER_OPTIMIZER_BACKEND): the fullgraph inner Boozer "
            "otherwise defaults to scipy on CPU, which jax_cpu_parity rejects at "
            "boozer_surface.py:5659. The GPU auto-default already supplies "
            "ondevice. Confirmed by the b5f97fdf9 gate smoke.",
            "GATE: scipy-jax-fullgraph (51) has no passing artifact yet (its "
            "only completed run pre-fix failed; the b5f97fdf9 smoke cleared the "
            "free_x bug and ran 5 outer iterations on GPU but did not reach "
            "passed=true -- CPU needed the boozer-backend fix above; GPU is "
            "compile-bound). A mpol=2 smoke must show rc=0/passed=true at the "
            "source SHA before any mpol=10 fullgraph cell is submitted.",
            "Timing label: all matrix cells are mixed-parity-reference because "
            "the parity harness co-produces the JAX target lane and the cpp/CPU "
            "reference. Whole-job wall time is not headline-eligible clean GPU "
            "timing.",
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
        f"- Inner Boozer least-squares: `{manifest['inner_boozer_least_squares']}` "
        "(both lanes; LM/lm-minpack dropped)",
        "",
        "## Formulation/backend coupling",
        "",
        "- `11` reduced (coils only, surface solved each iteration by the inner "
        "Boozer solve) ⇒ `scipy-jax` (host SciPy L-BFGS-B).",
        "- `51` 51-dim coils+surface vector (full-graph `JF.x` DOF map) ⇒ "
        "`scipy-jax-fullgraph` (host SciPy L-BFGS-B over the full DOF vector; "
        "outer gradient on the coil block only, surface re-solved each iteration "
        "by the inner Boozer solve, not a residual penalty).",
        "- Both lanes are host-driven (one value/grad eval compiled per step); "
        "the monolithic `ondevice` lane is removed (it OOMs at production "
        "resolution).",
        "- The inner Boozer solve runs on-device (JAX) on both lanes; the "
        "`scipy-jax-fullgraph` cells force `--boozer-optimizer-backend ondevice` "
        "(its CPU default is otherwise `scipy`, rejected under `jax_cpu_parity`).",
        "- The cpp/CPU reference is always 51; the 11-dim `scipy-jax` lane is "
        "compared against a dim-mismatched 51 reference, while the 51-dim "
        "`scipy-jax-fullgraph` lane is dim-matched.",
        "- Timing classification: `mixed-parity-reference`; whole-job wall time "
        "is not a clean GPU headline.",
        "",
        "## Cells",
        "",
        "| id | dim | backend | timing | headline? | inner-boozer | platform | tier | status | ref dim match | runnable now |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in manifest["cells"]:
        runnable = "yes" if manifest["tiers"][c["tier"]]["runnable_now"] else "no (donor)"
        inner = c["inner_boozer_optimizer_backend"] or "default"
        lines.append(
            f"| `{c['id']}` | {c['formulation_dim']} | {c['optimizer_backend']} | "
            f"{c['timing_classification']} | "
            f"{'yes' if c['supports_performance_headline'] else 'no'} | "
            f"{inner} | {c['platform']} | {c['tier']} | {c['status']} | "
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
