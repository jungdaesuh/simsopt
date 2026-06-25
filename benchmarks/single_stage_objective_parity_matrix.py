"""Multi-seed, multi-backend FIXED-POINT parity matrix for the single-stage banana
objective.

For each autoresearch seed this evaluates the full 8-term single-stage objective
``J`` and its gradient ``dJ`` -- built through the example's SSOT factory
(``benchmarks.single_stage_objective_eval``) -- in the native-C++ lane and the JAX
lane at the *same* resolved Boozer state, and records per-lane wall time, peak host
RSS and (on GPU) peak device memory plus the cross-lane ``J``/``dJ`` differences.

The three reported columns -- ``cpp``, ``jax-cpu``, ``jax-gpu`` -- come from TWO
code lanes evaluated under two JAX platforms:

  * ``JAX_PLATFORMS=cpu``  run -> records ``cpp`` + ``jax-cpu`` (and cpp-vs-jax-cpu parity)
  * ``JAX_PLATFORMS=cuda`` run -> records ``cpp`` + ``jax-gpu`` (and cpp-vs-jax-gpu parity)

The native C++ lane is platform-independent, so its ``J``/``dJ`` must match between
the two runs; ``--collate`` cross-checks that and assembles the final table.

This is a fixed-point (single-evaluation) parity matrix, NOT a converged-trajectory
comparison: per-evaluation ``J``/``dJ`` at a matched resolved state is the
well-posed cross-backend parity claim (different outer optimizers diverge after
step 1 even when both are correct).  For each seed the native-CPU lane warm-starts
from the seed's *resolved* Boozer surface at its NATIVE resolution: it installs the
resolved surface DOFs, recovers the consistent ``(iota*, G*)`` and full-solves from
that Boozer fixed point (a fresh solve at a coarser resolution collapses to a
degenerate, self-intersecting surface for production seeds).  The JAX lane is then
seeded from the CPU lane's converged surface / iota / G so both lanes evaluate at the
identical state.  The seed supplies the diverse coil set + config + its native
resolution; ``--mpol`` / ``--ntor`` / ``--nphi`` / ``--ntheta`` are diagnostic
overrides only and must match the seed's resolved surface (its DOFs are read at that
resolution).

Run (on the pod, per platform), then collate::

    # cpu platform: cpp + jax-cpu
    env -u OPTIMIZER_BACKEND JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
        SIMSOPT_REPO_SHA=x SIMSOPT_GIT_STATUS_SHORT=x \\
        PYTHONPATH="$REPO/src:$REPO" python -m benchmarks.single_stage_objective_parity_matrix \\
        --seeds <seed_dir> ... --lane-label jax-cpu --out /tmp/matrix_cpu.json

    # cuda platform: cpp + jax-gpu  (see HANDOFF for the full cuda env / ptxas note)
    ... JAX_PLATFORMS=cuda ... --lane-label jax-gpu --out /tmp/matrix_gpu.json

    python -m benchmarks.single_stage_objective_parity_matrix \\
        --collate /tmp/matrix_cpu.json /tmp/matrix_gpu.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.single_stage_objective_eval import (  # noqa: E402
    build_lane_objective,
    build_matched_same_state_fixture_pair,
    objective_weights_from_example_defaults,
)
from benchmarks.single_stage_smoke_fixture import (  # noqa: E402
    DEFAULT_CONSTRAINT_WEIGHT,
)
from benchmarks.validation_ladder_contract import (  # noqa: E402
    parity_ladder_tolerances,
)
from examples.single_stage_optimization.equilibria_paths import (  # noqa: E402
    DEFAULT_EQUILIBRIA_DIR,
)

# ``ru_maxrss`` is reported in KiB on Linux and in bytes on macOS.  The matrix runs
# on the Linux pod; convert to bytes consistently for the table.
_RSS_TO_BYTES = 1 if sys.platform == "darwin" else 1024


@dataclass(frozen=True)
class SeedSpec:
    """Resolved per-seed configuration read from an autoresearch ``results.json``."""

    name: str
    run_dir: str
    stage2_bs_path: str
    plasma_surf_filename: str
    mpol: int
    ntor: int
    nphi: int
    ntheta: int
    vol_target: float
    iota_target: float
    num_tf_coils: int
    final_iota: float
    constraint_weight: float


def _seed_name(run_dir: str) -> str:
    parts = os.path.abspath(run_dir).rstrip(os.sep).split(os.sep)
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def load_seed_spec(
    run_dir: str,
    *,
    mpol_override: int | None = None,
    ntor_override: int | None = None,
    nphi_override: int | None = None,
    ntheta_override: int | None = None,
) -> SeedSpec:
    """Read a seed's ``results.json`` into a :class:`SeedSpec`.

    The matrix evaluates each seed at its NATIVE Boozer resolution (``mpol`` /
    ``ntor`` / ``nphi`` / ``ntheta`` from ``results.json``) and consumes the seed's
    resolved surface DOFs (see :func:`load_resolved_surface_dofs`) -- a fresh solve
    at a coarser resolution collapses to a degenerate surface for production seeds.
    The resolution overrides exist only for diagnostics and default to the seed's
    production values when None.  ``CONSTRAINT_WEIGHT`` is read so the matched-state
    solve uses the seed's own Boozer penalty weighting (defaults to
    ``DEFAULT_CONSTRAINT_WEIGHT`` for legacy seeds that predate the field).
    """
    run_dir = os.path.abspath(run_dir)
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as f:
        r = json.load(f)
    plasma_surf = os.path.basename(str(r["PLASMA_SURF_FILENAME"]))

    def _resolved(override, key):
        return int(override if override is not None else r[key])

    return SeedSpec(
        name=_seed_name(run_dir),
        run_dir=run_dir,
        stage2_bs_path=os.path.join(run_dir, "biot_savart_opt.json"),
        plasma_surf_filename=plasma_surf,
        mpol=_resolved(mpol_override, "mpol"),
        ntor=_resolved(ntor_override, "ntor"),
        nphi=_resolved(nphi_override, "nphi"),
        ntheta=_resolved(ntheta_override, "ntheta"),
        vol_target=float(r["TARGET_VOLUME"]),
        iota_target=float(r["TARGET_IOTA"]),
        num_tf_coils=int(r["NUM_TF_COILS"]),
        # ``.get(key, default)`` returns None for a present-but-null key, so guard
        # explicitly (some seeds persist FINAL_IOTA / CONSTRAINT_WEIGHT as JSON null).
        final_iota=(
            float(r["FINAL_IOTA"]) if r.get("FINAL_IOTA") is not None
            else float(r["TARGET_IOTA"])
        ),
        constraint_weight=(
            float(r["CONSTRAINT_WEIGHT"]) if r.get("CONSTRAINT_WEIGHT") is not None
            else DEFAULT_CONSTRAINT_WEIGHT
        ),
    )


def load_resolved_surface_dofs(run_dir: str, *, mpol: int, ntor: int) -> np.ndarray:
    """Extract the resolved Boozer surface free-DOFs from the GSON object graph.

    A fresh Boozer solve at a tractable resolution collapses to a degenerate
    surface for production seeds, so the matrix warm-starts the CPU lane from the
    seed's *resolved* surface at its native ``mpol``/``ntor``.  The serialized
    ``BoozerSurface`` container can reference an autoresearch pipeline module (e.g.
    ``banana_opt``) that is not importable here, so ``simsopt.load`` of the whole
    object fails with ``ModuleNotFoundError``.  The resolved surface itself is a
    standard simsopt ``SurfaceXYZTensorFourier`` whose DOFs live in the
    ``simsopt_objs`` graph, so read its FULL dof vector directly -- the exact vector
    ``surface.set_dofs`` (which assigns ``local_full_x``) /
    ``initialize_boozer_surface``'s ``surface_dofs_override`` expects.  Reading the
    graph rather than reconstructing the object keeps this drift-proof against the
    serializing pipeline's module layout.  The resolved Boozer surface is all-free
    (no fixed DOFs); this is asserted, so a future fixed-DOF seed fails loud instead
    of silently returning a short free-only vector to ``set_dofs``.  ``mpol``/``ntor``
    select the surface and assert the file matches the ``results.json`` resolution.
    """
    path = os.path.join(os.path.abspath(run_dir), "surf_opt_boozer_surface.json")
    with open(path, encoding="utf-8") as f:
        objs = json.load(f)["simsopt_objs"]

    matches = []
    for obj in objs.values():
        if not (isinstance(obj, dict) and obj.get("@class") == "SurfaceXYZTensorFourier"):
            continue
        if int(obj["mpol"]) != int(mpol) or int(obj["ntor"]) != int(ntor):
            continue
        dofs_node = obj["dofs"]
        if isinstance(dofs_node, dict) and dofs_node.get("$type") == "ref":
            dofs_node = objs[dofs_node["value"]]
        x_full = np.asarray(dofs_node["x"]["data"], dtype=float)
        free = dofs_node.get("free")
        if isinstance(free, dict) and "data" in free:
            free_mask = np.asarray(free["data"], dtype=bool)
            if not bool(free_mask.all()):
                raise ValueError(
                    f"{run_dir}: resolved SurfaceXYZTensorFourier has "
                    f"{int((~free_mask).sum())}/{free_mask.size} fixed DOF(s); the "
                    "value-only install assigns this vector to set_dofs "
                    "(local_full_x), which expects the FULL dof vector -- an "
                    "all-free resolved surface is required."
                )
        matches.append(x_full)
    if len(matches) != 1:
        raise ValueError(
            f"{run_dir}: expected exactly one SurfaceXYZTensorFourier at "
            f"mpol={mpol} ntor={ntor} in surf_opt_boozer_surface.json, "
            f"found {len(matches)}"
        )
    return matches[0]


def _peak_host_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * _RSS_TO_BYTES


def _gpu_peak_bytes() -> int | None:
    """Peak device bytes-in-use on the active GPU, or None on a CPU backend."""
    import jax

    if jax.default_backend() == "cpu":
        return None
    stats = jax.devices()[0].memory_stats()
    if stats is None:
        return None
    return int(stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0)))


def _evaluate(JF, x) -> tuple[float, np.ndarray, float]:
    """Drive ``JF`` to coil-DOF point ``x`` and time a single ``J`` + ``dJ`` eval."""
    JF.x = x
    t0 = perf_counter()
    J = float(np.asarray(JF.J(), dtype=float))
    dJ = np.asarray(JF.dJ(), dtype=float)
    wall = perf_counter() - t0
    return J, dJ, wall


def _grad_stats(dJ: np.ndarray) -> dict[str, float]:
    return {
        "inf_norm": float(np.max(np.abs(dJ))),
        "l2_norm": float(np.linalg.norm(dJ)),
        "ndof": int(dJ.size),
    }


# Per-term sub-objectives of the SSOT SingleStageObjective.  Recording each term's J
# localizes any cross-lane gap.  Only the field/surface-coupled terms (nonqs /
# boozer_residual / iota) are backend-selected (simsopt vs simsopt_jax_adapters) and
# are thus the genuine cross-boundary parity signal.  The five geometric terms
# (curve_length / curve_curve / curvature / curve_surface / surf_surf) use the SAME
# class in both lanes (curve objectives from simsopt.geo; SurfaceSurfaceDistance is a
# single JAX-backed class shared by both lanes), so they agree by construction at the
# matched coil/surface state -- a self-consistency check, not a cross-kernel one.
_TERM_FIELDS = (
    ("nonqs", "JnonQSRatio"),
    ("boozer_residual", "JBoozerResidual"),
    ("iota", "Jiota"),
    ("curve_length", "JCurveLength"),
    ("curve_curve", "JCurveCurve"),
    ("curve_surface", "JCurveSurface"),
    ("surf_surf", "JSurfSurf"),
    ("curvature", "JCurvature"),
)


def _term_values(obj) -> dict[str, float]:
    """Per-term J of the single-stage objective (to localize any parity gap)."""
    return {
        name: float(np.asarray(getattr(obj, attr).J(), dtype=float))
        for name, attr in _TERM_FIELDS
    }


def run_seed(
    seed: SeedSpec,
    *,
    equilibria_dir: str,
    jax_optimizer_backend: str,
    jax_lane_label: str,
    warm_eval: bool,
) -> dict:
    """Evaluate one seed in the native-C++ lane and the JAX lane at a matched state.

    Returns a record with per-lane ``J``/``dJ`` stats, wall time (cold incl. any
    JAX compile, and optionally a warm/compiled re-eval at a perturbed point) and
    memory, plus the cross-lane parity diffs.
    """
    # Single weights instance shared by both lanes -> the only per-lane difference
    # is the backend, never the weights.
    weights = objective_weights_from_example_defaults()

    # Warm-start the native-CPU Boozer solve at the seed's RESOLVED surface (at the
    # seed's native mpol/ntor) -- a fresh solve at a tractable resolution collapses
    # to a degenerate (near-zero-iota, self-intersecting) surface for production
    # seeds.  The JAX lane is then seeded from the CPU lane's converged
    # surface/iota/G so both lanes evaluate at the IDENTICAL resolved state
    # (same-state parity).
    resolved_dofs = load_resolved_surface_dofs(
        seed.run_dir, mpol=seed.mpol, ntor=seed.ntor
    )
    cpu_fixture, jax_fixture = build_matched_same_state_fixture_pair(
        plasma_surf_filename=seed.plasma_surf_filename,
        equilibria_dir=equilibria_dir,
        stage2_bs_path=seed.stage2_bs_path,
        nphi=seed.nphi,
        ntheta=seed.ntheta,
        mpol=seed.mpol,
        ntor=seed.ntor,
        vol_target=seed.vol_target,
        iota_target=seed.iota_target,
        num_tf_coils=seed.num_tf_coils,
        constraint_weight=seed.constraint_weight,
        jax_optimizer_backend=jax_optimizer_backend,
        cpu_boozer_surface_dofs_override=resolved_dofs,
        cpu_boozer_iota_override=seed.final_iota,
        recover_iota_G=True,
    )
    assert cpu_fixture["boozer_surface"].res.get("success", False)
    assert jax_fixture["boozer_surface"].res.get("success", False)

    cpu_obj = build_lane_objective(
        cpu_fixture, use_jax=False, num_tf_coils=seed.num_tf_coils, weights=weights
    )
    jax_obj = build_lane_objective(
        jax_fixture, use_jax=True, num_tf_coils=seed.num_tf_coils, weights=weights
    )
    cpu_JF, jax_JF = cpu_obj.JF, jax_obj.JF

    cpu_x = np.asarray(cpu_JF.x, dtype=float)
    jax_x = np.asarray(jax_JF.x, dtype=float)
    assert cpu_x.shape == jax_x.shape, (
        f"{seed.name}: native/JAX JF.x layouts differ "
        f"(cpu={cpu_x.shape}, jax={jax_x.shape})"
    )
    # Both lanes load the identical Stage-2 coils, so their objective free-DOFs
    # (the coil DOFs) must agree before evaluation; a mismatch means the two lanes
    # were seeded from different coils and the parity comparison is meaningless.
    assert np.allclose(jax_x, cpu_x, rtol=0.0, atol=1e-10), (
        f"{seed.name}: native/JAX seed coil DOFs disagree before evaluation "
        f"(max|Δ|={float(np.max(np.abs(jax_x - cpu_x))):.3e})"
    )
    x0 = cpu_x.copy()

    rss0 = _peak_host_rss_bytes()
    cpu_J, cpu_dJ, cpu_wall = _evaluate(cpu_JF, x0)
    rss_cpu = _peak_host_rss_bytes()
    jax_J, jax_dJ, jax_wall = _evaluate(jax_JF, x0)
    rss_jax = _peak_host_rss_bytes()
    gpu_peak = _gpu_peak_bytes()

    # Per-term breakdown at x0 (cached) -- localizes any cross-lane gap to specific
    # objective terms (cross-boundary nonqs/boozer_residual/iota vs geometric).
    cpp_terms = _term_values(cpu_obj)
    jax_terms = _term_values(jax_obj)

    cpu_warm = jax_warm = None
    if warm_eval:
        # Perturb to force a genuine recompute (native) / compiled device re-exec
        # (JAX), not an Optimizable cache hit at the unchanged x0.
        x1 = x0 + 1e-8
        _, _, cpu_warm = _evaluate(cpu_JF, x1)
        _, _, jax_warm = _evaluate(jax_JF, x1)

    assert np.all(np.isfinite(cpu_dJ)) and np.linalg.norm(cpu_dJ) > 0
    assert np.all(np.isfinite(jax_dJ)) and np.linalg.norm(jax_dJ) > 0

    value_abs_diff = abs(jax_J - cpu_J)
    grad_abs_diff = float(np.max(np.abs(jax_dJ - cpu_dJ)))
    grad_rel_diff = grad_abs_diff / (float(np.max(np.abs(cpu_dJ))) + 1e-30)

    return {
        "seed": seed.name,
        "spec": asdict(seed),
        "jax_lane_label": jax_lane_label,
        "jax_optimizer_backend": jax_optimizer_backend,
        "cpp": {
            "J": cpu_J,
            "grad": _grad_stats(cpu_dJ),
            "wall_cold_s": cpu_wall,
            "wall_warm_s": cpu_warm,
            "peak_host_rss_bytes": rss_cpu,
        },
        "jax": {
            "J": jax_J,
            "grad": _grad_stats(jax_dJ),
            "wall_cold_s": jax_wall,
            "wall_warm_s": jax_warm,
            "peak_host_rss_bytes": rss_jax,
            "peak_gpu_bytes": gpu_peak,
        },
        "parity_cpp_vs_jax": {
            "value_abs_diff": value_abs_diff,
            "grad_abs_diff": grad_abs_diff,
            "grad_rel_diff": grad_rel_diff,
        },
        "terms": {"cpp": cpp_terms, "jax": jax_terms},
        "host_rss_baseline_bytes": rss0,
    }


def _jax_platform_label() -> str:
    import jax

    return jax.default_backend()


def _gib(n) -> str:
    return "-" if n is None else f"{n / 1024**3:.3f}"


def _run(args: argparse.Namespace) -> None:
    platform = _jax_platform_label()
    print(f"[matrix] jax backend = {platform!r}; jax lane label = {args.lane_label!r}")
    records: list[dict] = []
    for run_dir in args.seeds:
        seed = load_seed_spec(
            run_dir,
            mpol_override=args.mpol,
            ntor_override=args.ntor,
            nphi_override=args.nphi,
            ntheta_override=args.ntheta,
        )
        print(
            f"[matrix] {seed.name}: mpol={seed.mpol} ntor={seed.ntor} "
            f"nphi={seed.nphi} ntheta={seed.ntheta} iota*={seed.iota_target} "
            f"vol*={seed.vol_target}"
        )
        rec = run_seed(
            seed,
            equilibria_dir=args.equilibria_dir,
            jax_optimizer_backend=args.jax_optimizer_backend,
            jax_lane_label=args.lane_label,
            warm_eval=not args.no_warm_eval,
        )
        records.append(rec)
        p = rec["parity_cpp_vs_jax"]
        print(
            f"[matrix] {seed.name}: cpp J={rec['cpp']['J']:.12e} "
            f"jax J={rec['jax']['J']:.12e} |dJ|diff={p['grad_abs_diff']:.3e} "
            f"cpp_wall={rec['cpp']['wall_cold_s']:.2f}s "
            f"jax_wall={rec['jax']['wall_cold_s']:.2f}s "
            f"gpu={_gib(rec['jax']['peak_gpu_bytes'])}GiB"
        )
        # Incremental persistence: a later-seed failure never loses earlier results.
        payload = {"platform": platform, "lane_label": args.lane_label, "records": records}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print(f"[matrix] wrote {len(records)} record(s) -> {args.out}")


def _wall_cell(rec: dict | None, key: str) -> str:
    """Format a lane's wall time as ``cold[/warm]`` seconds, or ``-`` if absent."""
    if rec is None:
        return "-"
    cold = rec[key]["wall_cold_s"]
    warm = rec[key]["wall_warm_s"]
    return f"{cold:.2f}" + (f"/{warm:.2f}" if warm is not None else "")


def _parity_relative(rec: dict) -> float:
    """Worst relative cross-lane diff for one seed/lane (value vs gradient)."""
    p = rec["parity_cpp_vs_jax"]
    value_rel = p["value_abs_diff"] / (abs(rec["cpp"]["J"]) + 1e-30)
    return max(value_rel, p["grad_rel_diff"])


def _parity_cell(rec: dict | None, bar: float) -> str:
    """``max|ΔdJ|`` plus a PASS(✓)/FAIL(✗) verdict against the relative ``bar``."""
    if rec is None:
        return "-"
    grad_abs = rec["parity_cpp_vs_jax"]["grad_abs_diff"]
    mark = "✓" if _parity_relative(rec) <= bar else "✗"
    return f"{grad_abs:.2e} {mark}"


def _collate(result_files: list[str], grad_tol: float) -> None:
    by_seed: dict[str, dict] = {}
    cpp_by_seed: dict[str, dict] = {}
    for path in result_files:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        for rec in payload["records"]:
            seed = rec["seed"]
            entry = by_seed.setdefault(seed, {"spec": rec["spec"]})
            entry[rec["jax_lane_label"]] = rec
            # Cross-check the platform-independent native lane across files.
            prev = cpp_by_seed.get(seed)
            if prev is not None:
                dj = abs(prev["J"] - rec["cpp"]["J"])
                if dj > grad_tol:
                    print(
                        f"[collate] WARNING {seed}: cpp J differs across files "
                        f"({prev['J']:.12e} vs {rec['cpp']['J']:.12e}, |Δ|={dj:.3e})"
                    )
            cpp_by_seed[seed] = rec["cpp"]

    bar = parity_ladder_tolerances("ls-wrapper-gradient")["rtol"]
    print("\n## Parity (native-C++ vs JAX, J + max|ΔdJ| + verdict)\n")
    print(
        f"Verdict bar: max(value_rel, grad_rel) <= {bar:.0e} "
        "(parity-ladder ls-wrapper-gradient rtol SSOT).\n"
    )
    print("| seed | mpol/ntor | cpp J | jax-cpu J | jax-gpu J | "
          "cpp·jax-cpu max|ΔdJ| | cpp·jax-gpu max|ΔdJ| |")
    print("|---|---|---|---|---|---|---|")
    verdicts: list[bool] = []
    for seed, entry in by_seed.items():
        spec = entry["spec"]
        cpu = entry.get("jax-cpu")
        gpu = entry.get("jax-gpu")
        cpp_J = (cpu or gpu)["cpp"]["J"]
        cpu_J = f"{cpu['jax']['J']:.10e}" if cpu else "-"
        gpu_J = f"{gpu['jax']['J']:.10e}" if gpu else "-"
        verdicts.extend(
            _parity_relative(r) <= bar for r in (cpu, gpu) if r is not None
        )
        print(f"| {seed} | {spec['mpol']}/{spec['ntor']} | {cpp_J:.10e} | "
              f"{cpu_J} | {gpu_J} | {_parity_cell(cpu, bar)} | {_parity_cell(gpu, bar)} |")
    print(
        f"\n**Parity verdict: {sum(verdicts)}/{len(verdicts)} lane-checks PASS** "
        f"at relative bar {bar:.0e}."
    )

    print("\n## Performance (wall cold[/warm] s · peak host RSS GiB · peak GPU GiB)\n")
    print("| seed | cpp wall | cpp RSS | jax-cpu wall | jax-cpu RSS | "
          "jax-gpu wall | jax-gpu RSS | jax-gpu GPU |")
    print("|---|---|---|---|---|---|---|---|")
    for seed, entry in by_seed.items():
        cpu = entry.get("jax-cpu")
        gpu = entry.get("jax-gpu")
        src = cpu or gpu

        cpp_wall = _wall_cell(src, "cpp")
        cpp_rss = _gib(src["cpp"]["peak_host_rss_bytes"]) if src else "-"
        cpu_wall = _wall_cell(cpu, "jax")
        cpu_rss = _gib(cpu["jax"]["peak_host_rss_bytes"]) if cpu else "-"
        gpu_wall = _wall_cell(gpu, "jax")
        gpu_rss = _gib(gpu["jax"]["peak_host_rss_bytes"]) if gpu else "-"
        gpu_mem = _gib(gpu["jax"]["peak_gpu_bytes"]) if gpu else "-"
        print(f"| {seed} | {cpp_wall} | {cpp_rss} | {cpu_wall} | {cpu_rss} | "
              f"{gpu_wall} | {gpu_rss} | {gpu_mem} |")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", nargs="+", default=[],
        help="autoresearch seed run directories (each with results.json, "
        "biot_savart_opt.json, surf_opt_boozer_surface.json)",
    )
    parser.add_argument("--out", default="/tmp/single_stage_parity_matrix.json")
    parser.add_argument(
        "--equilibria-dir", default=str(DEFAULT_EQUILIBRIA_DIR),
        help="directory holding the seeds' wout_*.nc equilibria (e.g. the pod's "
        "/workspace/equilibria); defaults to the repo's example equilibria dir",
    )
    parser.add_argument(
        "--lane-label", default=None,
        help="label for the JAX column ('jax-cpu' or 'jax-gpu'); defaults to "
        "'jax-<backend>'",
    )
    parser.add_argument(
        "--mpol", type=int, default=None,
        help="diagnostic override of the seed's native mpol; the matrix consumes the "
        "seed's resolved surface DOFs, so this must equal the value in results.json "
        "(load_resolved_surface_dofs raises if no resolved surface at this mpol/ntor)",
    )
    parser.add_argument("--ntor", type=int, default=None, help="override seed ntor")
    parser.add_argument("--nphi", type=int, default=None, help="override seed nphi")
    parser.add_argument("--ntheta", type=int, default=None, help="override seed ntheta")
    parser.add_argument("--jax-optimizer-backend", default="ondevice")
    parser.add_argument(
        "--no-warm-eval", action="store_true",
        help="skip the warm/compiled re-eval (halves cost; reports cold wall only)",
    )
    parser.add_argument(
        "--collate", nargs="+", default=None,
        help="collate mode: result JSON files to merge into the matrix tables",
    )
    parser.add_argument(
        "--collate-grad-tol", type=float, default=1e-9,
        help="tolerance for the cross-file native-lane J consistency check",
    )
    args = parser.parse_args(argv)

    if args.collate:
        _collate(args.collate, args.collate_grad_tol)
        return
    if not args.seeds:
        parser.error("either --seeds (run mode) or --collate (collate mode) is required")
    if args.lane_label is None:
        args.lane_label = f"jax-{_jax_platform_label()}"
    _run(args)


if __name__ == "__main__":
    main()
