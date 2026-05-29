#!/usr/bin/env python3
"""Select replayable single-stage seeds from an autoresearch results.jsonl.

Given an autoresearch ``results_surrogate.jsonl`` (one optimization run per
line), this emits a replay manifest of candidate seeds for re-running in the
simsopt-jax single-stage banana example, classified along every axis that
governs whether a seed can actually be replayed *here*:

  * convergence   -- optimizer succeeded AND iota reached its target (not a
                     frozen 1-iteration re-score of an already-good seed)
  * hardware      -- the run passed the banana-coil engineering constraints
  * vacuum-clean  -- no finite Boozer I / net plasma current / confinement
                     objective; simsopt-jax single-stage is a *vacuum* path
                     and silently ignores finite current, so a finite-current
                     seed is format-replayable but NOT physics-faithful here
  * format        -- the warm-start contract artifacts exist
                     (biot_savart_opt.json + surf_opt.json)
  * equilibrium   -- the target equilibrium file resolves on this machine

The manifest is the single source of truth a launcher consumes; it carries the
raw surrogate params, the joined artifact physics, the resolved file paths, and
a launch-ready simsopt-jax CLI flag set for each ready candidate.

This selector is read-only: it inspects files, it never mutates a seed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

# Warm-start contract artifacts the simsopt-jax single-stage example requires
# (single_stage_banana_example.py: restart-artifact contract).
REQUIRED_SEED_ARTIFACTS = ("biot_savart_opt.json", "surf_opt.json")

DEFAULT_RESULTS = "/Users/suhjungdae/code/columbia/autoresearch/results_surrogate.jsonl"
DEFAULT_EQUILIBRIA_DIR = "/Users/suhjungdae/code/columbia/DATABASE/EQUILIBRIA/desc"

# Surrogate param -> simsopt-jax CLI flag (objective weights, targets, geometry
# the repo accepts at runtime). Params NOT listed here are seed-derived
# (order/major_radius/toroidal_flux come from --stage2-bs-path's results.json)
# or surrogate-only (confinement/ALM/basin/finite-current) with no jax flag.
PARAM_TO_FLAG = {
    "cc_weight": "--cc-weight",
    "cc_dist": "--cc-dist",
    "cs_weight": "--cs-weight",
    "cs_dist": "--cs-dist",
    "ss_dist": "--ss-dist",
    "ss_length_weight": "--length-weight",
    "curvature_weight": "--curvature-weight",
    "curvature_threshold": "--curvature-threshold",
    "constraint_weight": "--constraint-weight",
    "res_weight": "--res-weight",
    "iotas_weight": "--iotas-weight",
    "surf_dist_weight": "--surf-dist-weight",
    "banana_surf_radius": "--banana-surf-radius",
    "iota_target": "--iota-target",
    "vol_target": "--vol-target",
    "num_tf_coils": "--num-tf-coils",
    "mpol": "--mpol",
    "ntor": "--ntor",
    "nphi": "--nphi",
    "ntheta": "--ntheta",
}

# Convergence tolerance: |final_iota - iota_target| at/below this (with >1
# iteration) counts as "genuinely converged to target", separating real optima
# from frozen seeds re-scored against a target they never moved toward.
IOTA_TARGET_TOL = 5e-3

# simsopt-jax coil-current hard limits (SSOT:
# examples/single_stage_optimization/banana_opt/hardware_contracts.py). A seed
# whose per-coil TF / banana current exceeds these is outside this repo's
# hardware envelope: the run only proceeds by bypassing the hardware-valid gate
# (explicit seed spec), so it is NOT "compatible hw limits" here.
REPO_TF_CURRENT_HARD_LIMIT_A = 8.0e4
REPO_BANANA_CURRENT_HARD_LIMIT_A = 1.6e4


def _num(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _is_zeroish(x: Any) -> bool:
    v = _num(x)
    return v is None or abs(v) < 1e-12


@dataclass
class Candidate:
    equilibrium: str
    score: Optional[float]
    artifact_dir: Optional[str]
    seed_bs_path: Optional[str]  # the biot_savart_opt.json to seed from
    surf_path: Optional[str]
    results_json_path: Optional[str]
    equilibrium_path: Optional[str]  # resolved local wout file
    # physics / convergence
    iterations: Optional[int]
    final_iota: Optional[float]
    iota_target: Optional[float]
    iota_err: Optional[float]
    field_error: Optional[float]
    nonqs_ratio: Optional[float]
    boozer_residual: Optional[float]
    max_curvature: Optional[float]
    # finite-current axis
    boozer_I: Optional[float]
    plasma_current_a: Optional[float]
    confinement_weight: Optional[float]
    finite_current_mode: Optional[str]
    # coil-current hardware envelope (this repo's hard limits)
    tf_current_a: Optional[float]
    banana_current_a: Optional[float]
    current_within_repo_limits: bool
    # resolution
    mpol: Optional[int]
    ntor: Optional[int]
    nphi: Optional[int]
    ntheta: Optional[int]
    order: Optional[int]
    # readiness flags
    hw_ok: bool
    converged_to_target: bool
    vacuum_clean: bool
    format_ready: bool
    equilibrium_found: bool
    replay_ready_parity: bool  # ingestible CPU==GPU parity input
    replay_faithful_physics: bool  # also a valid physics oracle here
    blockers: list[str] = field(default_factory=list)
    launch_flags: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Optional[str]) -> dict:
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def _resolve_equilibrium(res: dict, equilibria_dir: str) -> Optional[str]:
    """Resolve the target equilibrium to a local file.

    Prefer the absolute path recorded in the artifact; otherwise resolve its
    basename under equilibria_dir.
    """
    for key in ("PLASMA_SURF_PATH", "PLASMA_SURF_FILENAME"):
        val = res.get(key)
        if not val:
            continue
        if os.path.isabs(val) and os.path.exists(val):
            return val
        cand = os.path.join(equilibria_dir, os.path.basename(val))
        if os.path.exists(cand):
            return cand
    return None


def build_candidate(row: dict, equilibria_dir: str) -> Candidate:
    params = row.get("params", {}) or {}
    artifact_dir = row.get("single_stage_artifact_dir") or row.get("artifact_dir")
    res = _read_json(os.path.join(artifact_dir, "results.json")) if artifact_dir else {}

    seed_bs = (
        os.path.join(artifact_dir, "biot_savart_opt.json") if artifact_dir else None
    )
    surf = os.path.join(artifact_dir, "surf_opt.json") if artifact_dir else None
    format_ready = bool(
        artifact_dir
        and all(
            os.path.exists(os.path.join(artifact_dir, a))
            for a in REQUIRED_SEED_ARTIFACTS
        )
    )

    equilibrium_path = _resolve_equilibrium(res, equilibria_dir)

    final_iota = _num(row.get("final_iota"))
    iota_target = _num(params.get("iota_target"))
    iota_err = (
        abs(final_iota - iota_target)
        if (final_iota is not None and iota_target is not None)
        else None
    )
    iters = row.get("iterations")
    iters = (
        int(iters)
        if isinstance(iters, (int, float)) and not isinstance(iters, bool)
        else None
    )

    boozer_I = _num(res.get("BOOZER_I"))
    plasma_current = _num(res.get("PLASMA_CURRENT_A"))
    confinement_w = _num(res.get("CONFINEMENT_OBJECTIVE_WEIGHT"))
    finite_mode = res.get("FINITE_CURRENT_MODE")

    # Per-coil TF / banana current vs this repo's hard limits. results.json
    # carries the TF current; banana current is not always recorded (lives on
    # the coil object), so an absent banana value is treated as unknown.
    tf_current = _num(res.get("STAGE2_TF_CURRENT_A")) or _num(res.get("TF_CURRENT_A"))
    banana_current = _num(res.get("STAGE2_BANANA_CURRENT_A")) or _num(
        res.get("BANANA_CURRENT_A")
    )
    tf_over = tf_current is not None and tf_current > REPO_TF_CURRENT_HARD_LIMIT_A * (
        1 + 1e-9
    )
    banana_over = (
        banana_current is not None
        and banana_current > REPO_BANANA_CURRENT_HARD_LIMIT_A * (1 + 1e-9)
    )
    current_within_repo_limits = (
        (tf_current is not None) and not tf_over and not banana_over
    )

    hw_ok = row.get("hardware_constraints_ok") is True
    converged = bool(
        row.get("optimizer_success") is True
        and iters is not None
        and iters > 1
        and iota_err is not None
        and iota_err <= IOTA_TARGET_TOL
    )
    vacuum_clean = (
        _is_zeroish(boozer_I)
        and _is_zeroish(plasma_current)
        and _is_zeroish(confinement_w)
    )

    blockers: list[str] = []
    if not hw_ok:
        blockers.append("hardware_constraints_failed")
    if not format_ready:
        blockers.append("missing_warm_start_artifacts")
    if not equilibrium_path:
        blockers.append("equilibrium_file_unresolved")
    if not vacuum_clean:
        bits = []
        if not _is_zeroish(boozer_I):
            bits.append(f"BOOZER_I={boozer_I}")
        if not _is_zeroish(plasma_current):
            bits.append(f"plasma_current_A={plasma_current}")
        if not _is_zeroish(confinement_w):
            bits.append(f"confinement_weight={confinement_w}")
        blockers.append("finite_current_not_vacuum(" + ",".join(bits) + ")")
    if not current_within_repo_limits:
        if tf_over:
            blockers.append(
                f"tf_current_exceeds_repo_limit({tf_current:.0f}>{REPO_TF_CURRENT_HARD_LIMIT_A:.0f}A)"
            )
        if banana_over:
            blockers.append(
                f"banana_current_exceeds_repo_limit({banana_current:.0f}>{REPO_BANANA_CURRENT_HARD_LIMIT_A:.0f}A)"
            )
        if tf_current is None:
            blockers.append("tf_current_unknown")
    if not converged:
        blockers.append("not_converged_to_iota_target")

    replay_ready_parity = format_ready and bool(equilibrium_path) and hw_ok
    replay_faithful_physics = (
        replay_ready_parity
        and vacuum_clean
        and converged
        and current_within_repo_limits
    )

    launch_flags: dict[str, Any] = {}
    if replay_ready_parity:
        launch_flags["--equilibrium-path"] = equilibrium_path
        launch_flags["--stage2-bs-path"] = (
            seed_bs  # overrides all derived seed settings
        )
        for pkey, flag in PARAM_TO_FLAG.items():
            if pkey in params and params[pkey] is not None:
                launch_flags[flag] = params[pkey]

    return Candidate(
        equilibrium=str(row.get("equilibrium")),
        score=_num(row.get("score")),
        artifact_dir=artifact_dir,
        seed_bs_path=seed_bs,
        surf_path=surf,
        results_json_path=os.path.join(artifact_dir, "results.json")
        if artifact_dir
        else None,
        equilibrium_path=equilibrium_path,
        iterations=iters,
        final_iota=final_iota,
        iota_target=iota_target,
        iota_err=iota_err,
        field_error=_num(row.get("field_error")),
        nonqs_ratio=_num(row.get("nonqs_ratio")),
        boozer_residual=_num(row.get("boozer_residual")),
        max_curvature=_num(row.get("max_curvature")),
        boozer_I=boozer_I,
        plasma_current_a=plasma_current,
        confinement_weight=confinement_w,
        finite_current_mode=finite_mode,
        tf_current_a=tf_current,
        banana_current_a=banana_current,
        current_within_repo_limits=current_within_repo_limits,
        mpol=params.get("mpol"),
        ntor=params.get("ntor"),
        nphi=params.get("nphi"),
        ntheta=params.get("ntheta"),
        order=params.get("order"),
        hw_ok=hw_ok,
        converged_to_target=converged,
        vacuum_clean=vacuum_clean,
        format_ready=format_ready,
        equilibrium_found=bool(equilibrium_path),
        replay_ready_parity=replay_ready_parity,
        replay_faithful_physics=replay_faithful_physics,
        blockers=blockers,
        launch_flags=launch_flags,
    )


def select(results_path: str, equilibria_dir: str, solver: str) -> list[Candidate]:
    cands: list[Candidate] = []
    with open(results_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("solver") != solver:
                continue
            if row.get("status") != "pass":
                continue
            if row.get("self_intersecting") is True:
                continue
            cands.append(build_candidate(row, equilibria_dir))
    cands.sort(key=lambda c: -(c.score or 0.0))
    return cands


# Map a single-stage results.json (UPPERCASE keys) into the jsonl-row shape
# build_candidate() consumes. Lets the selector discover converged optima that
# live only as artifact dirs on disk (autoresearch/runs/...), never indexed in a
# results jsonl -- which is where the best in-envelope vacuum seeds turned out
# to be.
def _row_from_results(results_json_path: str, artifact_dir: str) -> dict:
    r = _read_json(results_json_path)
    params = {
        "mpol": r.get("mpol"),
        "ntor": r.get("ntor"),
        "order": r.get("order"),
        "nphi": r.get("nphi"),
        "ntheta": r.get("ntheta"),
        "iota_target": r.get("TARGET_IOTA"),
        "vol_target": r.get("TARGET_VOLUME"),
        "num_tf_coils": r.get("NUM_TF_COILS") or r.get("num_tf_coils"),
        "banana_surf_radius": r.get("banana_surf_radius")
        or r.get("BANANA_SURF_RADIUS"),
        "cc_weight": r.get("CC_WEIGHT"),
        "cc_dist": r.get("CC_DIST"),
        "cs_weight": r.get("CS_WEIGHT"),
        "cs_dist": r.get("CS_DIST"),
        "ss_dist": r.get("SS_DIST"),
        "ss_length_weight": r.get("LENGTH_WEIGHT"),
        "curvature_weight": r.get("CURVATURE_WEIGHT"),
        "curvature_threshold": r.get("CURVATURE_THRESHOLD"),
        "constraint_weight": r.get("CONSTRAINT_WEIGHT"),
        "res_weight": r.get("RES_WEIGHT"),
        "iotas_weight": r.get("IOTAS_WEIGHT"),
        "surf_dist_weight": r.get("SURF_DIST_WEIGHT"),
    }
    return {
        "solver": "single-stage",
        "status": "pass" if r.get("OPTIMIZER_SUCCESS") is True else "fail",
        "equilibrium": r.get("PLASMA_SURF_FILENAME") or os.path.basename(artifact_dir),
        "score": None,
        "final_iota": r.get("FINAL_IOTA"),
        "iterations": r.get("iterations") or r.get("ITERATIONS"),
        "optimizer_success": r.get("OPTIMIZER_SUCCESS"),
        "hardware_constraints_ok": r.get("HARDWARE_CONSTRAINTS_OK"),
        "self_intersecting": r.get("SELF_INTERSECTING"),
        "field_error": r.get("FIELD_ERROR"),
        "nonqs_ratio": r.get("NONQS_RATIO"),
        "boozer_residual": r.get("BOOZER_RESIDUAL"),
        "max_curvature": r.get("MAX_CURVATURE"),
        "single_stage_artifact_dir": artifact_dir,
        "params": params,
    }


def scan_tree(roots: list[str], equilibria_dir: str) -> list[Candidate]:
    """Walk artifact-dir trees for single-stage seeds (bs_opt+surf_opt+results)."""
    cands: list[Candidate] = []
    seen: set[str] = set()
    skipped = 0
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            if not (
                "biot_savart_opt.json" in filenames
                and "surf_opt.json" in filenames
                and "results.json" in filenames
            ):
                continue
            if dirpath in seen:
                continue
            seen.add(dirpath)
            # Trees contain partial/corrupt results.json from interrupted runs;
            # skip unreadable ones rather than abort the whole scan.
            try:
                row = _row_from_results(os.path.join(dirpath, "results.json"), dirpath)
            except (json.JSONDecodeError, OSError, ValueError):
                skipped += 1
                continue
            if row.get("self_intersecting") is True:
                continue
            cands.append(build_candidate(row, equilibria_dir))
    if skipped:
        print(f"# tree-scan skipped {skipped} unreadable results.json", file=sys.stderr)
    # physics-faithful first, then lowest field error.
    cands.sort(
        key=lambda c: (
            not c.replay_faithful_physics,
            c.field_error if c.field_error is not None else float("inf"),
        )
    )
    return cands


def format_launch_command(c: Candidate, optimizer_backend: str, maxiter: int) -> str:
    parts = [
        "python single_stage_banana_example.py",
        "--backend jax",
        f"--optimizer-backend {optimizer_backend}",
        f"--maxiter {maxiter}",
    ]
    for flag, val in c.launch_flags.items():
        parts.append(f"{flag} {val}")
    return " \\\n    ".join(parts)


def _seed_coil_currents(bs_path: str, num_tf: int) -> tuple[float, float]:
    """Authoritative per-coil currents from the serialized seed.

    results.json's STAGE2_TF_CURRENT_A is the seed INPUT current, not the final
    optimized coil current -- the single-stage optimizer drives coil currents as
    DOFs, so they can end up over the hardware envelope while results.json still
    reports the seed value. Only the serialized coils (biot_savart_opt.json) are
    authoritative. Returns (max_tf_A, max_banana_A); banana is NaN if absent.
    Lazy simsopt import keeps the jsonl path free of the heavy dependency.
    """
    from simsopt._core import load

    bs = load(bs_path)
    cur = [abs(float(c.current.get_value())) for c in bs.coils]
    ntf = num_tf if (num_tf and num_tf < len(cur)) else min(20, len(cur))
    max_tf = max(cur[:ntf]) if ntf else float("nan")
    max_ban = max(cur[ntf:]) if len(cur) > ntf else float("nan")
    return max_tf, max_ban


def verify_coil_currents(cands: list[Candidate]) -> int:
    """Override the results.json current check with actual seed coil currents,
    for format-ready candidates. Mutates candidates in place; returns count
    verified. Scoped to format-ready seeds so the load cost stays small.
    """
    verified = 0
    for c in cands:
        if not (c.format_ready and c.seed_bs_path and os.path.exists(c.seed_bs_path)):
            continue
        ntf = c.launch_flags.get("--num-tf-coils")
        ntf = int(ntf) if ntf else 20
        try:
            max_tf, max_ban = _seed_coil_currents(c.seed_bs_path, ntf)
        except Exception as exc:  # corrupt/unloadable seed: flag, do not crash
            c.blockers.append(f"coil_current_unverified({type(exc).__name__})")
            continue
        verified += 1
        c.tf_current_a = max_tf
        c.banana_current_a = None if max_ban != max_ban else max_ban
        tf_over = max_tf > REPO_TF_CURRENT_HARD_LIMIT_A * (1 + 1e-9)
        ban_over = (
            max_ban == max_ban
        ) and max_ban > REPO_BANANA_CURRENT_HARD_LIMIT_A * (1 + 1e-9)
        c.current_within_repo_limits = not tf_over and not ban_over
        c.blockers = [
            b for b in c.blockers if not b.startswith(("tf_current", "banana_current"))
        ]
        if tf_over:
            c.blockers.append(
                f"tf_current_exceeds_repo_limit({max_tf:.0f}>{REPO_TF_CURRENT_HARD_LIMIT_A:.0f}A)"
            )
        if ban_over:
            c.blockers.append(
                f"banana_current_exceeds_repo_limit({max_ban:.0f}>{REPO_BANANA_CURRENT_HARD_LIMIT_A:.0f}A)"
            )
        c.replay_faithful_physics = (
            c.replay_ready_parity
            and c.vacuum_clean
            and c.converged_to_target
            and c.current_within_repo_limits
        )
    return verified


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results", default=DEFAULT_RESULTS, help="autoresearch results jsonl"
    )
    ap.add_argument(
        "--equilibria-dir",
        default=DEFAULT_EQUILIBRIA_DIR,
        help="dir holding wout_*.nc equilibria",
    )
    ap.add_argument(
        "--solver", default="single-stage", choices=["single-stage", "stage2"]
    )
    ap.add_argument(
        "--scan-tree",
        action="append",
        default=None,
        metavar="ROOT",
        help="scan artifact-dir trees for seeds instead of a results jsonl "
        "(repeatable). Finds converged optima that are not indexed in any jsonl.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="write manifest JSONL here (default: stdout summary only)",
    )
    ap.add_argument(
        "--ready",
        choices=["parity", "physics", "any"],
        default="any",
        help="filter manifest: parity-ingestible, physics-faithful, or any passing run",
    )
    ap.add_argument(
        "--emit-commands",
        action="store_true",
        help="print launch commands for ready candidates",
    )
    ap.add_argument("--optimizer-backend", default="scipy-jax")
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument(
        "--verify-coil-currents",
        action="store_true",
        help="load biot_savart_opt.json for format-ready candidates and check "
        "ACTUAL coil currents vs the repo envelope (authoritative; the "
        "results.json current is the seed input, not the final optimized coils).",
    )
    args = ap.parse_args()

    if args.scan_tree:
        cands = scan_tree(args.scan_tree, args.equilibria_dir)
        source_label = "tree-scan " + ",".join(args.scan_tree)
    else:
        cands = select(args.results, args.equilibria_dir, args.solver)
        source_label = args.solver

    if args.verify_coil_currents:
        n = verify_coil_currents(cands)
        print(
            f"# verified actual coil currents for {n} format-ready seeds",
            file=sys.stderr,
        )

    def keep(c: Candidate) -> bool:
        if args.ready == "parity":
            return c.replay_ready_parity
        if args.ready == "physics":
            return c.replay_faithful_physics
        return True

    kept = [c for c in cands if keep(c)]

    print(
        f"# {source_label}: {len(cands)} candidates | "
        f"parity-ready={sum(c.replay_ready_parity for c in cands)} "
        f"physics-faithful={sum(c.replay_faithful_physics for c in cands)} "
        f"vacuum-clean={sum(c.vacuum_clean for c in cands)} "
        f"hw-current-ok={sum(c.current_within_repo_limits for c in cands)} "
        f"converged-to-target={sum(c.converged_to_target for c in cands)}",
        file=sys.stderr,
    )
    hdr = (
        f"{'equil':<13}{'score':>6}{'conv':>5}{'vac':>4}{'cur':>4}{'fmt':>4}{'eq':>4}"
        f"{'fiota/tgt':>13}{'res':>14}  blockers"
    )
    print(hdr, file=sys.stderr)
    print("-" * 100, file=sys.stderr)
    for c in kept:
        fi = f"{c.final_iota:.4f}/{c.iota_target}" if c.final_iota is not None else "NA"
        res = f"{c.mpol}/{c.ntor}/{c.nphi}/{c.ntheta}"
        print(
            f"{c.equilibrium:<13}{(c.score or 0):6.3f}"
            f"{'Y' if c.converged_to_target else '.':>5}{'Y' if c.vacuum_clean else '.':>4}"
            f"{'Y' if c.current_within_repo_limits else '.':>4}"
            f"{'Y' if c.format_ready else '.':>4}{'Y' if c.equilibrium_found else '.':>4}"
            f"{fi:>13}{res:>14}  {';'.join(c.blockers) or 'NONE'}",
            file=sys.stderr,
        )

    if args.out:
        with open(args.out, "w") as fh:
            for c in kept:
                fh.write(json.dumps(asdict(c)) + "\n")
        print(f"\n# wrote {len(kept)} candidates -> {args.out}", file=sys.stderr)

    if args.emit_commands:
        for c in kept:
            if not c.replay_ready_parity:
                continue
            if c.replay_faithful_physics:
                tag = "PHYSICS-FAITHFUL"
            else:
                reasons = [
                    b for b in c.blockers if b not in ("not_converged_to_iota_target",)
                ]
                tag = (
                    "PARITY-INPUT-ONLY (" + ("; ".join(reasons) or "see blockers") + ")"
                )
            score_str = (
                f"score {c.score:.3f}"
                if c.score is not None
                else (
                    f"field_err {c.field_error:.2e}"
                    if c.field_error is not None
                    else "n/a"
                )
            )
            print(f"\n# === {c.equilibrium} ({score_str}) -- {tag} ===")
            print(format_launch_command(c, args.optimizer_backend, args.maxiter))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
