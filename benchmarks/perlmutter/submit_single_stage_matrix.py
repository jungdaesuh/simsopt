#!/usr/bin/env python3
"""Emit (or run) sbatch commands for single-stage 11-vs-51 matrix cells.

Reads the manifest produced by ``build_single_stage_matrix.py`` and, for each
selected cell, prints the ``sbatch`` command with the cell's ``PROD_*`` env
exported into the parameterized production launcher. Staging (one fresh
bundle-cloned checkout per job, run root outside the checkout) is left to the
operator: pass ``--repo-root`` and ``--run-root-base`` and this prints the
exact command to run on the login node.

This script does not SSH or submit on its own; it prints commands (or, with
``--exec`` on a Slurm login node, runs them). That keeps the dangerous part
(actual submission) explicit and reviewable.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select(cells: list[dict], *, status: str | None, tier: str | None,
           dim: int | None, platform: str | None, ids: list[str] | None) -> list[dict]:
    out = []
    for c in cells:
        if ids is not None and c["id"] not in ids:
            continue
        if status is not None and c["status"] != status:
            continue
        if tier is not None and c["tier"] != tier:
            continue
        if dim is not None and c["formulation_dim"] != dim:
            continue
        if platform is not None and c["platform"] != platform:
            continue
        out.append(c)
    return out


def sbatch_command(cell: dict, *, tiers: dict, repo_root: str, run_root_base: str,
                   warm_start: str | None) -> list[str]:
    env_items = dict(cell["env"])
    # The warm-start requirement is defined per tier in the manifest (the
    # generator's SSOT); read it rather than hardcoding a tier name.
    if tiers[cell["tier"]]["warm_start"] is not None:
        if warm_start is None:
            raise SystemExit(
                f"cell {cell['id']} is tier {cell['tier']} and requires "
                "--warm-start-run-dir (the continuation donor)"
            )
        env_items["PROD_WARM_START_RUN_DIR"] = warm_start
    run_root = f"{run_root_base.rstrip('/')}/{cell['id']}"
    export = "ALL,REPO_ROOT={repo},RUN_ROOT={run},{kv}".format(
        repo=repo_root,
        run=run_root,
        kv=",".join(f"{k}={v}" for k, v in env_items.items() if v != ""),
    )
    launcher = f"{repo_root.rstrip('/')}/benchmarks/perlmutter/{cell['launcher']}"
    cmd = ["sbatch", "-A", cell["account"], "-J", cell["id"]]
    cmd += shlex.split(cell["sbatch_extra"])
    cmd += [f"--export={export}", launcher]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",
                        default="docs/single_stage_11_51_matrix_2026-06-13.json")
    parser.add_argument("--repo-root", required=True,
                        help="Clean source checkout on the cluster (per-job).")
    parser.add_argument("--run-root-base", required=True,
                        help="Scratch run-root base OUTSIDE the checkout.")
    parser.add_argument("--warm-start-run-dir", default=None,
                        help="Continuation donor dir, required for production-tier "
                             "cells (any tier whose manifest warm_start is set).")
    parser.add_argument("--status", choices=["core", "extended"], default=None)
    parser.add_argument("--tier", choices=["mpol2", "mpol10"], default=None)
    parser.add_argument("--dim", type=int, choices=[11, 51], default=None)
    parser.add_argument("--platform", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--ids", nargs="*", default=None,
                        help="Explicit cell ids to select.")
    parser.add_argument("--exec", action="store_true",
                        help="Actually run sbatch (login node only). Default prints.")
    args = parser.parse_args()

    manifest = load_manifest(REPO_ROOT / args.manifest
                             if not Path(args.manifest).is_absolute()
                             else Path(args.manifest))
    cells = select(manifest["cells"], status=args.status, tier=args.tier,
                   dim=args.dim, platform=args.platform, ids=args.ids)
    if not cells:
        raise SystemExit("no cells matched the selection")

    # The launchers enforce RUN_ROOT-outside-REPO_ROOT themselves and fail
    # closed; pre-warn here to save a wasted queue round-trip.
    repo_real = os.path.realpath(args.repo_root)
    run_base_real = os.path.realpath(args.run_root_base)
    if run_base_real == repo_real or run_base_real.startswith(repo_real + os.sep):
        print(f"# WARNING: --run-root-base ({run_base_real}) is inside "
              f"--repo-root ({repo_real}); the launcher clean-source gate will "
              "reject these jobs.", file=sys.stderr)

    print(f"# {len(cells)} cell(s) selected from {manifest['matrix_id']} "
          f"(source_sha {manifest['source_sha']})")
    for c in cells:
        cmd = sbatch_command(c, tiers=manifest["tiers"], repo_root=args.repo_root,
                             run_root_base=args.run_root_base,
                             warm_start=args.warm_start_run_dir)
        print(
            f"# {c['id']}  dim={c['formulation_dim']} status={c['status']} "
            f"timing={c['timing_classification']} "
            f"headline={c['supports_performance_headline']}"
        )
        printable = " ".join(shlex.quote(p) for p in cmd)
        print(printable)
        if args.exec:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
