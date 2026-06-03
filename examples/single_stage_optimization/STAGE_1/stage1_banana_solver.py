"""
01_stage1_driver.py
───────────────────
Stage 1 VMEC fixed-boundary optimization for the banana coil
stellarator-tokamak hybrid targeting quasi-axisymmetry (QA, M=1 N=0).

Optimizes the VMEC boundary shape to improve QA using Boozer coordinates
(booz_xform) with a resolution ramp.  Supports warm start from an existing
wout file and cold start from a programmatic boundary for Pareto scans.

Pipeline:  01_stage1 (this) -> 02_stage2 -> 03_singlestage

At completion, saves the optimized VMEC input/wout pair and builds the
BoozerSurface (coils + surface) via utils/init_boozersurface for stage 2
warm-start.

Usage (MPI required):
    srun -n 16 python 01_stage1_driver.py
"""
import atexit
import numpy as np
import os
import shutil
import sys
import time
import netCDF4 as nc4

from datetime import datetime, timedelta

import argparse
import json
from pathlib import Path

# CO-LOCATED COPY of banana_drivers/01_stage1_driver.py, living in
# simsopt-surrogate alongside STAGE_2/ and SINGLE_STAGE/. It REUSES the shared
# Stage-1 utils + config.yaml from the banana_drivers repo (SSOT — not
# duplicated) off banana_drivers/utils, and replaces the banana_drivers sqlite
# RunRegistry with a results.json writer (the autoresearch runner owns
# provenance via --output-root). COLUMBIA_ROOT overrides the sibling-repo path.
_COLUMBIA_ROOT = os.environ.get(
    "COLUMBIA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")),
)
sys.path.insert(0, os.path.join(_COLUMBIA_ROOT, "banana_drivers", "utils"))
from output_dir import resolve_output_dir
from config_loader import load_config
from init_boozersurface import build_and_save, build_banana_coils
from hbt_parameters import compute_phiedge, compute_tf_rbtor, rescale_phiedge_to_rbtor
from near_axis_seed import near_axis_seed
from wout_shear import (
    interpolate_iota_at_s,
    working_layer_bounds,
    working_layer_shear_slope,
)

from simsopt import make_optimizable
from simsopt._core.util import ObjectiveFailure
from simsopt.geo import CurveSurfaceDistance
from simsopt.geo.surfaceobjectives import PrincipalCurvature
from simsopt.mhd import Vmec, Boozer, Quasisymmetry
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition


# ──────────────────────────────────────────────────────────────────────────────
# MPI setup
# ──────────────────────────────────────────────────────────────────────────────
mpi = MpiPartition()


def proc0_print(*args, **kwargs):
    if mpi.proc0_world:
        kwargs.setdefault('flush', True)
        print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Output root + run_registry shim (co-located solver: results.json, no sqlite)
# ──────────────────────────────────────────────────────────────────────────────
# The autoresearch thin runner owns the run directory and passes it as
# --output-root; we write all artifacts + a results.json there instead of into
# the banana_drivers sqlite registry. `artifact_path`/`run_dir`/
# `install_atexit_handler`/`RunRegistry` are local no-op shims of the
# banana_drivers run_registry API so the optimization body below is unchanged.
def _parse_output_root(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output-root", type=str, default=None)
    known, _ = p.parse_known_args(argv)
    return known.output_root


_OUTPUT_ROOT = _parse_output_root(sys.argv[1:])

_ARTIFACT_NAMES = {
    "diagnostics": "diagnostics.txt",
    "wout_init": "wout_init.nc",
    "boozmn_init": "boozmn_init.nc",
    "vmec_input_opt": "input_opt",
    "wout_opt": "wout_opt.nc",
    "boozmn_opt": "boozmn_opt.nc",
    "bsurf_opt": "bsurf_opt.json",
}


def run_dir(stage, run_id, out_dir):
    """Runner owns the run directory: artifacts live directly under --output-root."""
    return out_dir


def artifact_path(stage, run_id, out_dir, name):
    return os.path.join(out_dir, _ARTIFACT_NAMES[name])


def install_atexit_handler(registry, stage, run_id):
    return None


def _write_stage1_results(*, status, reason, metrics):
    """Write the runner-contract results.json (replaces the sqlite registry).

    Maps the Stage-1 metric dict to the SolverSpec required keys. proc0 only.
    Artifact-path globals (wout_opt_path, bsurf_out_path, ...) are read via
    globals() because an early failure (e.g. rejected seed) calls this before
    those names are bound — they resolve to None in that case.
    """
    metrics = dict(metrics or {})
    record = {
        "status": status,
        "status_reason": reason,
        "FINAL_IOTA": metrics.get("final_iota_working_s"),
        "FINAL_VOLUME": metrics.get("final_volume"),
        "FINAL_QS_METRIC": metrics.get("final_qs_metric"),
        "FINAL_LAYER_SHEAR_SLOPE": metrics.get("final_layer_shear_slope"),
        "FINAL_LAYER_SHEAR_SHORTFALL": metrics.get("final_layer_shear_shortfall"),
        "FINAL_IOTA_AXIS": metrics.get("final_iota_axis"),
        "FINAL_IOTA_EDGE": metrics.get("final_iota_edge"),
        "FINAL_WORKING_S": metrics.get("final_working_s"),
        "FINAL_ASPECT": metrics.get("final_aspect"),
        "IOTA_TARGET": globals().get("IOTA_TARGET"),
        "VOLUME_TARGET": globals().get("VOLUME_TARGET"),
        "ASPECT_TARGET": globals().get("ASPECT_TARGET"),
        "NFEV": metrics.get("nfev"),
        "RUNTIME_S": metrics.get("runtime_s"),
        "WOUT_PATH": globals().get("wout_opt_path"),
        "BSURF_PATH": globals().get("bsurf_out_path"),
        "BOOZMN_PATH": globals().get("boozmn_path"),
        "VMEC_INPUT_PATH": globals().get("input_opt_path"),
    }
    results_path = os.path.join(RUN_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(record, f, sort_keys=True, indent=2)
        f.write("\n")
    proc0_print(f"results.json written to {results_path}")


class RunRegistry:
    """No-op shim of the banana_drivers RunRegistry; emits results.json on finalize."""

    def register_stage1(self, cfg, slurm_meta=None):
        return ("stage1_run", True)

    def mark_running(self, *args, **kwargs):
        return None

    def mark_failed(self, stage, run_id, *, error_code=None, error_message=None,
                    metrics=None, slurm_wall_s=None, **kwargs):
        _write_stage1_results(
            status="fail",
            reason=str(error_message or error_code or "fail"),
            metrics=metrics,
        )

    def mark_success(self, stage, run_id, *, metrics=None, slurm_wall_s=None, **kwargs):
        _write_stage1_results(status="success", reason=None, metrics=metrics)


# ──────────────────────────────────────────────────────────────────────────────
# Load configuration
# ──────────────────────────────────────────────────────────────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))
cfg, _cfg_path = load_config()

# Device geometry
NFP      = cfg['device']['nfp']
STELLSYM = cfg['device']['stellsym']
VMEC_RBTOR = compute_tf_rbtor(cfg['tf_coils']['num'], cfg['tf_coils']['current'])

# Stage 1 settings
s1 = cfg['stage1']

# Seed mode override: BANANA_SEED=warm|cold supersedes stage1.cold_start in
# config.yaml. Lets submit.sh flip warm/cold without touching the config.
_seed_env = os.environ.get('BANANA_SEED')
if _seed_env is None:
    COLD_START = s1['cold_start']
elif _seed_env == 'warm':
    COLD_START = False
elif _seed_env == 'cold':
    COLD_START = True
else:
    raise ValueError(
        f"BANANA_SEED must be 'warm' or 'cold', got {_seed_env!r}"
    )
MAX_MODE_STEPS  = s1['max_mode_steps']
VMEC_MPOL       = s1['vmec_mpol']
VMEC_NTOR       = s1['vmec_ntor']
BOOZER_MPOL     = s1['boozer_mpol']
BOOZER_NTOR     = s1['boozer_ntor']
MAX_NFEV        = s1['max_nfev']
STEP_TRUST_RADIUS = [float(radius) for radius in s1['step_trust_radius']]
QS_SURFACES     = s1['qs_surfaces']
NS_ARRAY        = s1['ns_array']          # per failure mode #9 — radial resolution ramp
NITER_ARRAY     = s1['niter_array']
FTOL_ARRAY      = s1['ftol_array']

if len(STEP_TRUST_RADIUS) != len(MAX_MODE_STEPS):
    raise ValueError(
        f"stage1.step_trust_radius (len={len(STEP_TRUST_RADIUS)}) must match "
        f"stage1.max_mode_steps (len={len(MAX_MODE_STEPS)})"
    )
if len(NITER_ARRAY) != len(NS_ARRAY):
    raise ValueError(
        f"stage1.niter_array (len={len(NITER_ARRAY)}) must match stage1.ns_array "
        f"(len={len(NS_ARRAY)})"
    )
if len(FTOL_ARRAY) != len(NS_ARRAY):
    raise ValueError(
        f"stage1.ftol_array (len={len(FTOL_ARRAY)}) must match stage1.ns_array "
        f"(len={len(NS_ARRAY)})"
    )

# Physics targets (env var overrides for Pareto scans)
IOTA_TARGET   = float(os.environ.get('BANANA_IOTA', s1['iota_target']))
ASPECT_TARGET = float(os.environ.get('BANANA_ASPECT', s1['aspect_target']))
VOLUME_TARGET = float(os.environ.get('BANANA_VOLUME', s1['volume_target']))
WORKING_S = float(os.environ.get('BANANA_WORKING_S', s1.get('working_s', 0.90)))
WORKING_LAYER_INNER_S, WORKING_LAYER_OUTER_S = working_layer_bounds(WORKING_S)
WORKING_LAYER_SHEAR_MIN = float(
    os.environ.get(
        'BANANA_WORKING_LAYER_SHEAR_MIN',
        s1.get('working_layer_shear_min', 0.12),
    )
)

# Objective weights
ASPECT_WEIGHT = s1['aspect_weight']
IOTA_WEIGHT   = s1['iota_weight']
VOLUME_WEIGHT = s1['volume_weight']
QS_WEIGHT     = s1['qs_weight']
WORKING_LAYER_SHEAR_WEIGHT = float(
    os.environ.get(
        'BANANA_WORKING_LAYER_SHEAR_WEIGHT',
        s1.get('working_layer_shear_weight', IOTA_WEIGHT),
    )
)

# Weight mode: 'user' applies the weights above as-is; 'relative' divides the
# scalar targets' weights by |target| so each residual is a relative error (the
# user weight then acts as a relative-error coefficient). QS target is 0, so QS
# weight is always applied as-is.
WEIGHT_MODE = s1.get('weight_mode', 'user')
if WEIGHT_MODE not in ('user', 'relative'):
    raise ValueError(f"stage1.weight_mode must be 'user' or 'relative', got {WEIGHT_MODE!r}")

# Cold start boundary: user supplies (R0, V, iota); driver derives (a, phiedge,
# helical seed) from device constants. See cold_start_stage1_prompt.md.
COLD_R0     = float(os.environ.get('BANANA_COLD_R0', s1['cold_start_R0']))
COLD_VOLUME = float(os.environ.get('BANANA_VOLUME', s1['cold_start_volume']))
# Cold-start iota target is the same knob as the stage 1 iota objective target,
# so reuse IOTA_TARGET (already set above, with BANANA_IOTA env override).
COLD_A = float(np.sqrt(COLD_VOLUME / (2.0 * np.pi**2 * COLD_R0)))
COLD_PHIEDGE = compute_phiedge(VMEC_RBTOR, COLD_A, COLD_R0)

# Warm-start wout path. BANANA_WARM_WOUT (absolute, or relative to _base_dir)
# supersedes config.yaml warm_start.wout_filepath, so a caller (e.g. the closed
# loop) can warm-start each lineage from an arbitrary parent/donor wout without
# editing the config — mirrors the BANANA_SEED / BANANA_IOTA env-override pattern.
_warm_wout_env = os.environ.get('BANANA_WARM_WOUT')
if _warm_wout_env:
    WOUT_FILE = (
        _warm_wout_env if os.path.isabs(_warm_wout_env)
        else os.path.join(_base_dir, _warm_wout_env)
    )
else:
    WOUT_FILE = os.path.join(_base_dir, cfg['warm_start']['wout_filepath'])

# Write env-resolved values back into cfg so content-addressed hashing sees
# the effective inputs, not the raw config.yaml values. Pareto sweeps vary
# these via env vars; without this write-back, runs collide on the same run_id.
s1['cold_start']         = bool(COLD_START)
s1['iota_target']        = IOTA_TARGET
s1['aspect_target']      = ASPECT_TARGET
s1['volume_target']      = VOLUME_TARGET
s1['working_s']          = WORKING_S
s1['working_layer_shear_min'] = WORKING_LAYER_SHEAR_MIN
s1['working_layer_shear_weight'] = WORKING_LAYER_SHEAR_WEIGHT
s1['step_trust_radius'] = STEP_TRUST_RADIUS
s1['cold_start_R0']      = COLD_R0
s1['cold_start_volume']  = COLD_VOLUME
# Warm-start only: fold the resolved donor wout into the hashed inputs so two
# warm lineages off different parents do not collide on the same run_id. Cold
# runs leave warm_start untouched (the donor is not an effective input there).
if not COLD_START:
    cfg['warm_start']['wout_filepath'] = WOUT_FILE

# Output directory root. The autoresearch runner passes --output-root and owns
# the run directory; fall back to the banana_drivers BANANA_OUT_DIR resolver
# only when run standalone without --output-root.
OUT_DIR = str(Path(_OUTPUT_ROOT).resolve()) if _OUTPUT_ROOT else resolve_output_dir()


# ──────────────────────────────────────────────────────────────────────────────
# Registry: register this stage 1 run (rank 0 only; broadcast run_id to peers)
# ──────────────────────────────────────────────────────────────────────────────
# Content-addressed ID derived from STAGE1_INPUT_KEYS + git commit. Same
# inputs + same code → same ID. build_input_blob will KeyError if the
# whitelist has drifted away from config.yaml, so typos surface here instead
# of producing stable-but-meaningless hashes.
_slurm_meta = {
    "slurm_qos":           os.environ.get("SLURM_JOB_QOS"),
    "slurm_partition":     os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_ntasks":        int(os.environ["SLURM_NTASKS"]) if os.environ.get("SLURM_NTASKS") else None,
    "slurm_cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]) if os.environ.get("SLURM_CPUS_PER_TASK") else None,
    "slurm_time_limit_s":  None,   # SLURM doesn't expose TIME in a structured way; leave null
}
_slurm_job_id = os.environ.get("SLURM_JOB_ID")

registry: "RunRegistry | None" = None
_reg_payload: "tuple[str, bool] | None" = None
if mpi.proc0_world:
    registry = RunRegistry()
    _reg_payload = registry.register_stage1(cfg, slurm_meta=_slurm_meta)

# Broadcast (run_id, is_new) so every rank can resolve per-run paths and
# print a consistent header. compute_run_id is deterministic, but routing
# through rank 0 means only one process hits git + sqlite.
_reg_payload = mpi.comm_world.bcast(_reg_payload, root=0)
RUN_ID, _is_new = _reg_payload

RUN_DIR = run_dir("stage1", RUN_ID, OUT_DIR)
# All ranks create (exist_ok=True is idempotent) — every rank later needs
# RUN_DIR to exist for its per-group tempfile.mkdtemp and for opening
# diagnostics / wout snapshot files.
os.makedirs(RUN_DIR, exist_ok=True)

DIAGNOSTICS_FILE = artifact_path("stage1", RUN_ID, OUT_DIR, "diagnostics")


# ──────────────────────────────────────────────────────────────────────────────
# Output atexit handler
# ──────────────────────────────────────────────────────────────────────────────
def _emit_out_dir_on_exit():
    """Print the per-run directory so the shell script can move the log file
    into it (run_driver.sh reads this line)."""
    proc0_print(f"OUT_DIR={RUN_DIR}")


atexit.register(_emit_out_dir_on_exit)

# Mark the row 'failed' with error_code=unclean_exit if the interpreter exits
# while still in 'running' (uncaught exception, non-zero sys.exit, etc.).
# Rank-0 only; SIGKILL/OOM is caught by sweep() via sacct.
if mpi.proc0_world and registry is not None:
    install_atexit_handler(registry, "stage1", RUN_ID)


# ──────────────────────────────────────────────────────────────────────────────
# Print input parameters
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
INPUT PARAMETERS ─────────────────────────────
    Config:          {_cfg_path}
    Date:            {datetime.now()}
    Run ID:          {RUN_ID}  ({'new' if _is_new else 'rerun'})
    Run dir:         {RUN_DIR}
    MPI ranks:       {mpi.nprocs_world}

    Start mode:      {'COLD start (near-axis seed via pyQSC)' if COLD_START else 'WARM start from wout'}  {'(env BANANA_SEED)' if 'BANANA_SEED' in os.environ else ''}
    {'Boundary:        R0=' + f'{COLD_R0:.4f}' + ' m, V=' + f'{COLD_VOLUME:.4f}' + ' m^3 (a=' + f'{COLD_A:.4f}' + ' m), phiedge=' + f'{COLD_PHIEDGE:.6e}' + ' Wb (rbtor=' + f'{VMEC_RBTOR:.4f}' + ' T*m)' if COLD_START else 'Wout:            ' + WOUT_FILE}

    Physics targets:
        iota(s={WORKING_S:.3f}) = {IOTA_TARGET}  {'(env BANANA_IOTA)' if 'BANANA_IOTA' in os.environ else ''}
        layer shear >= {WORKING_LAYER_SHEAR_MIN} over s=[{WORKING_LAYER_INNER_S:.3f}, {WORKING_LAYER_OUTER_S:.3f}]
        aspect      = {ASPECT_TARGET}  {'(env BANANA_ASPECT)' if 'BANANA_ASPECT' in os.environ else ''}
        volume      = {VOLUME_TARGET}  {'(env BANANA_VOLUME)' if 'BANANA_VOLUME' in os.environ else ''}

    QA target: M=1, N=0
        surfaces    = {QS_SURFACES}

    Objective weights:
        aspect      = {ASPECT_WEIGHT:.3e}
        iota        = {IOTA_WEIGHT:.3e}
        shear       = {WORKING_LAYER_SHEAR_WEIGHT:.3e}
        volume      = {VOLUME_WEIGHT:.3e}
        qs          = {QS_WEIGHT:.3e}

    Resolution ramp ({len(MAX_MODE_STEPS)} steps):
        max_mode    = {MAX_MODE_STEPS}
        vmec mpol   = {VMEC_MPOL}
        vmec ntor   = {VMEC_NTOR}
        boozer mpol = {BOOZER_MPOL}
        boozer ntor = {BOOZER_NTOR}
        max_nfev    = {MAX_NFEV}
        trust radius = {STEP_TRUST_RADIUS}

    Output directory: {OUT_DIR}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Initialize VMEC
# ──────────────────────────────────────────────────────────────────────────────
if COLD_START:
    proc0_print('Cold start: building VMEC boundary from a near-axis seed (pyQSC)...')
    # Derive the seed boundary + axis guess from (R0, a, iota, nfp) via the
    # first-order near-axis expansion. This produces nonzero helical content
    # (modes with n != 0), which is required to escape the zero-beta zero-iota
    # trap (failure mode #3). The axis guess is self-consistent with the
    # boundary by construction, sidestepping ARNORM degeneracy (failure mode #1).
    try:
        seed = near_axis_seed(R0=COLD_R0, a=COLD_A, iota_target=IOTA_TARGET, nfp=NFP)
    except RuntimeError as e:
        # pyQSC near-axis walker could not produce a seed bracketing the
        # iota target under the elongation cap (or could not resolve the
        # minor-radius root). Exit with a distinct status so sweep
        # bookkeeping can separate "rejected seed" (2) from "VMEC/driver
        # crash" (1) and "walltime" (143).
        proc0_print(f'  REJECTED SEED: {e}')
        proc0_print(
            f'  (R0={COLD_R0}, a={COLD_A}, iota_target={IOTA_TARGET}, V={COLD_VOLUME}, nfp={NFP})'
        )
        if mpi.proc0_world and registry is not None:
            registry.mark_failed(
                "stage1", RUN_ID,
                error_code="bad_input",
                error_message=f"near_axis_seed rejected: {e}",
            )
        sys.exit(2)
    proc0_print(
        f'  pyQSC near-axis seed: etabar={seed["etabar"]:.4f}, '
        f'r={seed["r"]:.4f}, iota={seed["iota"]:.6f}, '
        f'max_elongation={seed["max_elongation"]:.4f}'
    )

    vmec = Vmec(mpi=mpi)
    vmec.indata.nfp = NFP
    vmec.indata.mpol = VMEC_MPOL[0]
    vmec.indata.ntor = VMEC_NTOR[0]
    vmec.indata.lasym = False
    vmec.indata.phiedge = COLD_PHIEDGE
    # Radial ramp: the driver applies NS_ARRAY[step] per resolution step in the
    # main loop below. Set the initial values here so the first vmec.run() at
    # INITIAL STATE time sees a well-resolved grid (failure mode #9).
    vmec.indata.ns_array[:]    = 0
    vmec.indata.niter_array[:] = 0
    vmec.indata.ftol_array[:]  = 0.0
    vmec.indata.ns_array[:len(NS_ARRAY)]    = NS_ARRAY
    vmec.indata.niter_array[:len(NS_ARRAY)] = NITER_ARRAY
    vmec.indata.ftol_array[:len(NS_ARRAY)]  = FTOL_ARRAY

    # Zero-beta, zero-current, shape-derived-iota profile.
    # ncurr=1 prescribes toroidal current (ac polynomial + curtor) and lets
    # VMEC solve for iota as an output of the boundary shape. With ac[:]=0 and
    # curtor=0, the plasma carries no net current and iota is determined purely
    # by the boundary geometry — this makes iota(s) a meaningful residual
    # with nonzero gradient w.r.t. boundary DOFs. (The ncurr=0 / ai[0]=target
    # formulation pinned iota_edge to ai[0] regardless of shape, giving a
    # constant residual that the optimizer exploited by collapsing to an
    # axisymmetric boundary with trivial QS=0 at I=0.)
    vmec.indata.ncurr = 1
    vmec.indata.ac[:] = 0.0
    vmec.indata.curtor = 0.0
    vmec.indata.ai[:] = 0.0
    vmec.indata.pres_scale = 0.0
    vmec.indata.am[:] = 0.0

    # Self-consistent magnetic axis guess — avoids ARNORM degeneracy.
    vmec.indata.raxis_cc[:] = 0.0
    vmec.indata.zaxis_cs[:] = 0.0
    raxis = seed['raxis_cc']
    zaxis = seed['zaxis_cs']
    vmec.indata.raxis_cc[:len(raxis)] = raxis
    vmec.indata.zaxis_cs[:len(zaxis)] = zaxis

    # Build a fresh SurfaceRZFourier at the starting VMEC resolution and
    # populate it from the near-axis seed Fourier coefficients. Assign it to
    # vmec.boundary — this is the same pattern the warm-start branch uses to
    # replace the input.default boundary with the seed geometry.
    from simsopt.geo import SurfaceRZFourier
    seed_surf = SurfaceRZFourier(
        nfp=NFP, stellsym=True,
        mpol=VMEC_MPOL[0], ntor=VMEC_NTOR[0],
    )
    # Start from zero, then overwrite the modes returned by near_axis_seed.
    # Any mode outside (mpol0, ntor0) is silently truncated here — the seed's
    # first-order boundary concentrates amplitude in low-m/low-n modes anyway,
    # and the ramp loop below frees higher modes step by step.
    seed_surf.x = np.zeros_like(seed_surf.x)
    for (m, n), val in seed['rbc'].items():
        if m <= VMEC_MPOL[0] and abs(n) <= VMEC_NTOR[0]:
            seed_surf.set_rc(m, n, float(val))
    for (m, n), val in seed['zbs'].items():
        if m <= VMEC_MPOL[0] and abs(n) <= VMEC_NTOR[0]:
            seed_surf.set_zs(m, n, float(val))
    vmec.boundary = seed_surf

    # CRITICAL: re-sync the Optimizable DOF cache after writing indata.phiedge
    # and replacing vmec.boundary. Failure mode #6: without this, the next
    # least_squares_mpi_solve broadcast restores the stale input.default
    # phiedge=1.0 and produces |B|~12 T.
    vmec.local_full_x = np.asarray(vmec.get_dofs())

    proc0_print(
        f'  Boundary: R0={COLD_R0:.4f} m, a={COLD_A:.4f} m, '
        f'phiedge={COLD_PHIEDGE:.6e} Wb'
    )
    proc0_print(
        f'  Axis: raxis_cc={raxis}, zaxis_cs={zaxis}'
    )
else:
    # Warm start: the seed wout has been pre-processed by utils/vmec_resize.py
    # to have LCFS (s=1) == target plasma boundary, at the correct major radius
    # and enclosed toroidal flux. Stage 1 imports its geometry, then applies
    # the signed TF-current rbtor scale used by the rest of the pipeline.
    proc0_print(f'Warm start: seeding boundary from {WOUT_FILE}')
    ds = nc4.Dataset(WOUT_FILE)
    wout_nfp = int(ds.variables['nfp'][:])
    wout_mpol = int(ds.variables['mpol'][:])
    wout_ntor = int(ds.variables['ntor'][:])
    wout_phiedge = float(ds.variables['phi'][:][-1])
    wout_rbtor = float(ds.variables['rbtor'][:])
    ds.close()
    warm_phiedge = rescale_phiedge_to_rbtor(wout_phiedge, wout_rbtor, VMEC_RBTOR)

    vmec = Vmec(mpi=mpi)
    vmec.indata.nfp = wout_nfp
    vmec.indata.mpol = max(wout_mpol, VMEC_MPOL[0])
    vmec.indata.ntor = max(wout_ntor, VMEC_NTOR[0])
    vmec.indata.phiedge = warm_phiedge
    # Bump NITER for the high-mpol resolution steps. input.default ships with
    # niter_array[:]=3000, insufficient at mpol=5 for this equilibrium — see
    # job 51257661 where FSQR plateaued at 1.56e-10 above FTOLV=1e-10.
    # Use config-driven NITER_ARRAY here; max(NITER_ARRAY) preserves the
    # bump-to-largest-budget intent for the warm-start broadcast initializer
    # while keeping niter as a single SSOT in config.yaml.
    vmec.indata.niter_array[:] = max(NITER_ARRAY)

    # CRITICAL: Vmec's Optimizable DOF cache was populated from input.default
    # (phiedge=1.0) during __init__. Overriding vmec.indata.phiedge alone does
    # NOT update that cache — so when the least-squares optimizer later calls
    # `prob.x = x`, Vmec.set_dofs is invoked with the stale [1.0, 0.0, 1.0]
    # vector and silently resets indata.phiedge back to 1.0. Re-syncing
    # local_full_x from indata here locks the cache to the signed TF phiedge.
    vmec.local_full_x = np.asarray(vmec.get_dofs())

    # Load LCFS directly — the resized seed has LCFS == target boundary.
    from simsopt.geo import SurfaceRZFourier
    wout_surf = SurfaceRZFourier.from_wout(WOUT_FILE, range='full torus',
                                           nphi=50, ntheta=50)
    vmec.boundary = wout_surf
    proc0_print(f'  nfp={wout_nfp}, mpol={vmec.indata.mpol}, ntor={vmec.indata.ntor}, '
                f'seed_rbtor={wout_rbtor:.6f}, phiedge={vmec.indata.phiedge:.6f}, '
                f'R0={wout_surf.major_radius():.4f} m')

vmec.verbose = mpi.proc0_world
surf = vmec.boundary


# ──────────────────────────────────────────────────────────────────────────────
# Configure Boozer + Quasisymmetry objectives
# ──────────────────────────────────────────────────────────────────────────────
proc0_print('Configuring Boozer + Quasisymmetry (M=1, N=0) objectives...')
boozer = Boozer(vmec)
boozer.bx.verbose = mpi.proc0_world

# QA objective on multiple flux surfaces
qs_list = [Quasisymmetry(boozer, s, 1, 0) for s in QS_SURFACES]


def _vmec_iotaf(vmec_instance: Vmec) -> np.ndarray:
    vmec_instance.run()
    return np.asarray(vmec_instance.wout.iotaf, dtype=float)


def _working_iota_from_vmec(vmec_instance: Vmec) -> float:
    return interpolate_iota_at_s(_vmec_iotaf(vmec_instance), WORKING_S)


def _working_layer_shear_slope_from_vmec(vmec_instance: Vmec) -> float:
    return working_layer_shear_slope(_vmec_iotaf(vmec_instance), WORKING_S)


def _working_layer_shear_shortfall_from_vmec(vmec_instance: Vmec) -> float:
    return max(
        0.0,
        WORKING_LAYER_SHEAR_MIN - _working_layer_shear_slope_from_vmec(vmec_instance),
    )


def _working_iota() -> float:
    return _working_iota_from_vmec(vmec)


def _working_layer_shear_slope() -> float:
    return _working_layer_shear_slope_from_vmec(vmec)


def _working_layer_shear_shortfall() -> float:
    return _working_layer_shear_shortfall_from_vmec(vmec)


working_iota_objective = make_optimizable(_working_iota_from_vmec, vmec)
working_layer_shear_shortfall_objective = make_optimizable(
    _working_layer_shear_shortfall_from_vmec,
    vmec,
)


# ──────────────────────────────────────────────────────────────────────────────
# Coil-aware DIRECTIONAL objective (closed-loop reframe, 2026-05-28)
# ──────────────────────────────────────────────────────────────────────────────
# The loop's goal is DIRECTIONAL: MAXIMIZE iota and volume, MINIMIZE QA /
# complexity, subject to coil realizability. Instead of (x - target)^2 penalties
# (which pin iota/volume to a set point), iota and volume enter as
# deficit-from-aspiration shortfall residuals (mirroring the working-layer shear
# shortfall): minimizing max(0, aspiration - x)^2 pushes x UP and never penalizes
# exceeding the aspiration. Aspirations are set high enough that the deficit does
# not saturate in the feasible range ("pure maximize"). A resonance-notch term
# pushes iota off the low-order rationals 1/5=0.20 and 1/4=0.25.
#
# Coil realizability enters via CurveSurfaceDistance against the FIXED banana
# coils on the 0.903/0.142 winding surface (penalizes boundaries that crowd the
# coils) — the surface-side fix for the boundary-agnostic realizability wall.
# PrincipalCurvature (surface smoothness) is opt-in (env, default 0) pending
# HBT-scale calibration of its curvature bound.
STAGE1_IOTA_ASPIRATION = float(os.environ.get("STAGE1_IOTA_ASPIRATION", "1.0"))
# Lower edge of the iota objective: max(0, ASPIRATION - iota) pushes iota UP
# toward this floor. Default 1.0 = pure-maximize (push iota as high as possible).
STAGE1_IOTA_CEILING = float(os.environ.get("STAGE1_IOTA_CEILING", "inf"))
# Upper edge of the iota band: max(0, iota - CEILING) pushes iota DOWN once it
# exceeds the ceiling. Default +inf = no ceiling (one-sided pure-maximize, the
# historical behavior). iota then enters as a TWO-SIDED BAND penalty (see
# _iota_band_penalty_from_vmec) driving iota into [ASPIRATION, CEILING]. The
# closed loop sets a finite band so iota is a STEERABLE target, not a runaway
# push: a real-resolution run with no ceiling drove iota to 0.79 (unrealizable),
# and a bare low aspiration UNDERSHOOTS (0.15 -> 0.071) because lowering it only
# weakens the push, letting QA win. The upper edge kills the runaway; the lower
# edge keeps iota off the floor. NOTE: realizability is best bounded by the
# L_grad_B term (Kappel, STAGE1_LGRADB_WEIGHT) once calibrated -- this band is
# the cheaper steering/safety rail.
# Volume band [ASPIRATION, CEILING], mirroring the iota band: max(0, ASPIRATION - V)
# pushes volume UP toward the aspiration, max(0, V - CEILING) pushes it DOWN above the
# ceiling, zero inside. The closed loop sets a finite band so volume is steered into
# its operating box rather than running past it. Default aspiration 0.20 / ceiling
# +inf = the historical one-sided push-to-0.20, so standalone runs are byte-unchanged.
STAGE1_VOLUME_ASPIRATION = float(os.environ.get("STAGE1_VOLUME_ASPIRATION", "0.20"))
STAGE1_VOLUME_CEILING = float(os.environ.get("STAGE1_VOLUME_CEILING", "inf"))
STAGE1_IOTA_RESONANCES = (0.20, 0.25)  # 1/5, 1/4 at NFP=5 — avoid
STAGE1_IOTA_NOTCH_EPS = 0.01
STAGE1_IOTA_NOTCH_WEIGHT = 50.0
STAGE1_CS_THRESHOLD = 0.010           # 1.0 cm coil-plasma clearance (HW spec)
STAGE1_CS_WEIGHT = float(os.environ.get("STAGE1_CS_WEIGHT", "1.0"))
STAGE1_PRINC_CURV_WEIGHT = float(os.environ.get("STAGE1_PRINC_CURV_WEIGHT", "0.0"))
# min(L_grad_B) coil-realizability metric (Kappel et al., PPCF 66 025018 (2024)):
# the magnetic-field scale length on the plasma boundary, sqrt(2)*|B|/||grad B||_F.
# Larger min(L_grad_B) -> gentler field structure -> more coil-realizable. Entered
# as a shortfall below a floor (target 0 -> minimizing pushes the boundary min UP).
# Opt-in via STAGE1_LGRADB_WEIGHT; the floor (meters) needs HBT-scale calibration
# (prior probes bracket good ~0.34 m vs hard ~0.13 m), so it defaults near the
# good regime and stays inert until a weight is set.
STAGE1_LGRADB_WEIGHT = float(os.environ.get("STAGE1_LGRADB_WEIGHT", "0.0"))
STAGE1_LGRADB_FLOOR = float(os.environ.get("STAGE1_LGRADB_FLOOR", "0.30"))
STAGE1_LGRADB_NTHETA = int(os.environ.get("STAGE1_LGRADB_NTHETA", "32"))
STAGE1_LGRADB_NPHI = int(os.environ.get("STAGE1_LGRADB_NPHI", "32"))


def _iota_band_penalty_from_vmec(vmec_instance: Vmec) -> float:
    # Two-sided band: push iota UP below the floor (ASPIRATION) and DOWN above
    # the ceiling, zero inside [ASPIRATION, CEILING]. With the default ceiling
    # (+inf) this reduces exactly to the one-sided pure-maximize shortfall
    # max(0, ASPIRATION - iota), so default-run output is unchanged.
    iota = _working_iota_from_vmec(vmec_instance)
    return (
        max(0.0, STAGE1_IOTA_ASPIRATION - iota)
        + max(0.0, iota - STAGE1_IOTA_CEILING)
    )


def _volume_deficit_from_vmec(vmec_instance: Vmec) -> float:
    # Two-sided volume band (mirrors _iota_band_penalty_from_vmec): push UP below the
    # aspiration and DOWN above the ceiling, zero inside [ASPIRATION, CEILING]. The
    # default ceiling (+inf) reduces exactly to the historical one-sided push.
    vmec_instance.run()
    vol = float(vmec_instance.volume())
    return (
        max(0.0, STAGE1_VOLUME_ASPIRATION - vol)
        + max(0.0, vol - STAGE1_VOLUME_CEILING)
    )


def _iota_resonance_notch_from_vmec(vmec_instance: Vmec) -> float:
    iota = abs(_working_iota_from_vmec(vmec_instance))
    return sum(
        max(0.0, STAGE1_IOTA_NOTCH_EPS - abs(iota - r))
        for r in STAGE1_IOTA_RESONANCES
    )


def _min_lgradB_from_vmec(vmec_instance: Vmec) -> float:
    """Minimum field scale length sqrt(2)*|B|/||grad B||_F over the boundary.

    Uses SIMSOPT's vetted ``vmec_compute_geometry`` (the Kappel et al. L_grad_B)
    on a theta x phi grid at s=1 over one field period. Units: meters.
    """
    vmec_instance.run()
    theta = np.linspace(0.0, 2.0 * np.pi, STAGE1_LGRADB_NTHETA, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / NFP, STAGE1_LGRADB_NPHI, endpoint=False)
    geometry = vmec_compute_geometry(vmec_instance, 1.0, theta, phi)
    return float(np.min(geometry.L_grad_B))


def _lgradB_shortfall_from_vmec(vmec_instance: Vmec) -> float:
    return max(0.0, STAGE1_LGRADB_FLOOR - _min_lgradB_from_vmec(vmec_instance))


iota_band_objective = make_optimizable(_iota_band_penalty_from_vmec, vmec)
volume_deficit_objective = make_optimizable(_volume_deficit_from_vmec, vmec)
iota_resonance_notch_objective = make_optimizable(_iota_resonance_notch_from_vmec, vmec)
lgradB_shortfall_objective = make_optimizable(_lgradB_shortfall_from_vmec, vmec)

# Fixed banana coils on the winding surface, for the CurveSurfaceDistance
# realizability term. Coils are held fixed (we vary only the plasma boundary),
# so this acts as a one-sided clearance penalty on the boundary.
_banana_coils_cs, _winding_surface_cs = build_banana_coils(cfg)
for _coil in _banana_coils_cs:
    _coil.curve.fix_all()
    _coil.current.fix_all()
banana_curves_cs = [_coil.curve for _coil in _banana_coils_cs]


def _build_prob():
    """Build LeastSquaresProblem from current objectives.

    Must be called after updating boozer.mpol/ntor — Quasisymmetry.J() returns
    one residual per Boozer mode, so the residual vector length changes with
    boozer resolution.  LeastSquaresProblem caches nvals on first eval and
    cannot handle a size change, so we rebuild it at each resolution step.
    """
    # Target iota on the same working surface family used by the Stage 1 gate.
    # Magnetic shear enters as a shortfall residual, so sufficient shear is not
    # penalized while flat profiles are pushed out of the topology basin.
    #
    # Weight mode:
    #   'user'     — weights used as-is (absolute residuals)
    #   'relative' — effective weight = user_weight / |target|, so the scalar
    #                residuals become relative errors. QS target is 0, so the
    #                QS weight is always applied as-is.
    if WEIGHT_MODE == 'relative':
        aw = ASPECT_WEIGHT / abs(ASPECT_TARGET)
    else:
        aw = ASPECT_WEIGHT
    # DIRECTIONAL: iota and volume enter as deficit-from-aspiration shortfalls
    # (target 0 -> minimizing pushes them UP), not as (x - target)^2 set-points.
    # `aw` keeps aspect as a loose sanity anchor; iota/volume use their raw
    # weights (the relative-mode |target| division is meaningless for a deficit).
    tuples = [
        (vmec.aspect, ASPECT_TARGET, aw),
        (iota_band_objective.J, 0.0, IOTA_WEIGHT),
        (iota_resonance_notch_objective.J, 0.0, STAGE1_IOTA_NOTCH_WEIGHT),
        (working_layer_shear_shortfall_objective.J, 0.0, WORKING_LAYER_SHEAR_WEIGHT),
        (volume_deficit_objective.J, 0.0, VOLUME_WEIGHT),
    ]
    # Coil realizability: penalize boundaries that crowd the fixed banana coils.
    # Built here (not once) because `surf` is replaced at each resolution step.
    cs_dist = CurveSurfaceDistance(banana_curves_cs, surf, STAGE1_CS_THRESHOLD)
    tuples.append((cs_dist.J, 0.0, STAGE1_CS_WEIGHT))
    if STAGE1_PRINC_CURV_WEIGHT > 0.0:
        # Surface-smoothness analog of the 100 m^-1 coil-curvature cap. kappamax
        # values need HBT-scale calibration; opt-in via STAGE1_PRINC_CURV_WEIGHT.
        pc = PrincipalCurvature(
            surf, kappamax1=-1.0, kappamax2=6.67, weight1=1.0, weight2=-0.5,
        )
        tuples.append((pc.J, 0.0, STAGE1_PRINC_CURV_WEIGHT))
    if STAGE1_LGRADB_WEIGHT > 0.0:
        # Coil realizability via the boundary field scale length (Kappel metric).
        # vmec-based (surface-independent), so reused across resolution steps;
        # appended only when enabled to keep default runs byte-identical.
        tuples.append((lgradB_shortfall_objective.J, 0.0, STAGE1_LGRADB_WEIGHT))
    for qs in qs_list:
        tuples.append((qs.J, 0, QS_WEIGHT))
    return LeastSquaresProblem.from_tuples(tuples)


prob = _build_prob()
proc0_print(
    f'  {4 + len(qs_list)} objective terms (aspect + iota(s={WORKING_S:.3f}) '
    f'+ shear shortfall + volume + {len(qs_list)} QS surfaces)'
)


# ──────────────────────────────────────────────────────────────────────────────
# Per-group working directories (avoids VMEC fort.9 file conflicts across MPI groups)
# ──────────────────────────────────────────────────────────────────────────────
import tempfile
_orig_dir = os.getcwd()
_group_dir = tempfile.mkdtemp(prefix=f'vmec_g{mpi.group:03d}_', dir=RUN_DIR)
os.chdir(_group_dir)

@atexit.register
def _cleanup_group_dir():
    os.chdir(_orig_dir)
    shutil.rmtree(_group_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Print initial state
# ──────────────────────────────────────────────────────────────────────────────
# Run VMEC once to get initial equilibrium values
vmec.run()
initial_working_iota = _working_iota()
initial_working_shear = _working_layer_shear_slope()
proc0_print(
    f"""
INITIAL STATE ─────────────────────────────────
    Aspect ratio:        {float(vmec.aspect()):.6f}  (target: {ASPECT_TARGET})
    Iota axis:           {float(vmec.iota_axis()):.6f}  (target: {IOTA_TARGET})
    Iota edge:           {float(vmec.iota_edge()):.6f}  (target: {IOTA_TARGET})
    Iota s={WORKING_S:.3f}:      {initial_working_iota:.6f}  (target: {IOTA_TARGET})
    Layer shear slope:   {initial_working_shear:.6f}  (min: {WORKING_LAYER_SHEAR_MIN})
    Volume:              {float(vmec.volume()):.6f} m^3  (target: {VOLUME_TARGET})
    VMEC mpol:           {vmec.indata.mpol}
    VMEC ntor:           {vmec.indata.ntor}
    Boundary ndofs:      {len(surf.get_dofs())}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Save initial wout + boozmn (for comparison plots vs the optimized result)
# ──────────────────────────────────────────────────────────────────────────────
# Snapshot the pre-optimization equilibrium so downstream analysis can diff
# the cold/warm-start seed against the final optimized surface. Uses the
# first ramp step's Boozer resolution; the optimizer will reset these inside
# the ramp loop before the first least_squares_mpi_solve call.
_init_wout_path   = artifact_path("stage1", RUN_ID, OUT_DIR, "wout_init")
_init_boozmn_path = artifact_path("stage1", RUN_ID, OUT_DIR, "boozmn_init")

boozer.mpol = BOOZER_MPOL[0]
boozer.ntor = BOOZER_NTOR[0]
boozer.run()

if mpi.proc0_world:
    if vmec.output_file and os.path.exists(vmec.output_file):
        shutil.copy2(vmec.output_file, _init_wout_path)
        proc0_print(f'Initial wout saved to {_init_wout_path}')
    else:
        raise RuntimeError('initial VMEC run did not produce a wout snapshot')
    boozer.bx.write_boozmn(_init_boozmn_path)
    proc0_print(f'Initial boozmn saved to {_init_boozmn_path}')

# Workaround for upstream SIMSOPT bug: Boozer.run() assigns
# self.bx.compute_surfs AFTER calling self.bx.init_from_vmec(wout.ns, ...),
# so when the ramp loop drops VMEC to a smaller ns than the initial snapshot
# ran at, booz_xform validates the NEW ns against the STALE compute_surfs
# from the previous (larger-ns) run and throws
# "compute_surfs has an entry that is too large for the given ns".
# Clearing compute_surfs here makes init_from_vmec see an empty list, so the
# first ramp-step run() succeeds and writes its own compute_surfs.
# Proper fix belongs upstream in simsopt/mhd/boozer.py (see PLAN.md).
boozer.bx.compute_surfs = np.array([], dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Initialize diagnostics file
# ──────────────────────────────────────────────────────────────────────────────
t0 = time.time()

if mpi.proc0_world:
    with open(DIAGNOSTICS_FILE, 'w') as f:
        f.write('# Stage 1 VMEC QA Optimization Diagnostics\n')
        f.write(f'# Date: {datetime.now()}\n')
        f.write(f'# Start mode: {"COLD" if COLD_START else "WARM"}\n')
        f.write(f'# Targets: iota={IOTA_TARGET}, aspect={ASPECT_TARGET}, volume={VOLUME_TARGET}\n')
        f.write(f'# QA surfaces: {QS_SURFACES}\n')
        f.write(
            'step,max_mode,vmec_mpol,vmec_ntor,boozer_mpol,boozer_ntor,'
            'max_nfev,vmec_iter,'
            'aspect,iota_axis,iota_edge,iota_working_s,'
            'layer_shear_slope,layer_shear_shortfall,volume,'
            'qs_total,objective,'
            'runtime\n'
        )


# ──────────────────────────────────────────────────────────────────────────────
# Resolution ramp optimization
# ──────────────────────────────────────────────────────────────────────────────
if mpi.proc0_world and registry is not None:
    registry.mark_running("stage1", RUN_ID, slurm_job_id=_slurm_job_id)

n_steps = len(MAX_MODE_STEPS)
for step in range(n_steps):
    step_t0 = time.time()

    max_mode  = MAX_MODE_STEPS[step]
    v_mpol    = VMEC_MPOL[step]
    v_ntor    = VMEC_NTOR[step]
    b_mpol    = BOOZER_MPOL[step]
    b_ntor    = BOOZER_NTOR[step]
    max_nfev  = MAX_NFEV[step]

    # Update VMEC resolution
    vmec.indata.mpol = v_mpol
    vmec.indata.ntor = v_ntor

    # Grow the boundary SurfaceRZFourier basis to (v_mpol, v_ntor) if needed.
    # SurfaceRZFourier.change_resolution() mutates in place and returns None.
    # copy(mpol=..., ntor=...) is the API that returns a resized surface while
    # preserving existing coefficients and padding higher modes with zero.
    # Without this, cold-start seed_surf (built at VMEC_MPOL[0]/VMEC_NTOR[0])
    # and warm-start wout_surf (at wout_mpol/ntor) lack the dof names for
    # high-m/high-n modes, so the fixed_range call below raises "'rc(0,4)' is
    # not in list" when max_mode exceeds the initial basis.
    if surf.mpol < v_mpol or surf.ntor < v_ntor:
        new_surf = surf.copy(
            mpol=max(surf.mpol, v_mpol),
            ntor=max(surf.ntor, v_ntor),
        )
        vmec.boundary = new_surf
        surf = new_surf
        # Re-sync the Vmec Optimizable DOF cache after replacing boundary
        # (same failure mode #6 fix as the cold-start branch).
        vmec.local_full_x = np.asarray(vmec.get_dofs())

    # Radial resolution ramp. Use all NS stages up to and including the current
    # step's target, so the multi-grid solver climbs to ns = NS_ARRAY[step] on
    # each optimizer eval. Failure mode #9: without this, a fixed ns can't
    # resolve the helical content's gradient and iota drifts from the prescribed
    # profile.
    _ns_up_to_step = NS_ARRAY[: step + 1] if step + 1 <= len(NS_ARRAY) else NS_ARRAY
    vmec.indata.ns_array[:]    = 0
    vmec.indata.niter_array[:] = 0
    vmec.indata.ftol_array[:]  = 0.0
    _niter_up_to_step = NITER_ARRAY[: step + 1] if step + 1 <= len(NITER_ARRAY) else NITER_ARRAY
    _ftol_up_to_step  = FTOL_ARRAY[: step + 1]  if step + 1 <= len(FTOL_ARRAY)  else FTOL_ARRAY
    vmec.indata.ns_array[:len(_ns_up_to_step)]    = _ns_up_to_step
    vmec.indata.niter_array[:len(_ns_up_to_step)] = _niter_up_to_step
    vmec.indata.ftol_array[:len(_ns_up_to_step)]  = _ftol_up_to_step

    # Update booz_xform resolution
    boozer.mpol = b_mpol
    boozer.ntor = b_ntor

    # Rebuild LeastSquaresProblem — Quasisymmetry.J() residual length changes
    # with boozer mpol/ntor (one residual per symmetry-breaking Boozer mode).
    prob = _build_prob()

    # Free boundary modes up to max_mode
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")  # Keep major radius fixed

    n_free = len([d for d, f in zip(surf.dof_names, surf.dofs_free_status) if f])
    trust_radius = STEP_TRUST_RADIUS[step]
    trust_center = np.copy(prob.x)
    trust_lower = trust_center - trust_radius
    trust_upper = trust_center + trust_radius

    proc0_print(
        f"""
STEP {step+1}/{n_steps} ───────────────────────────────────
    max_mode    = {max_mode}
    vmec mpol   = {v_mpol}, ntor = {v_ntor}
    boozer mpol = {b_mpol}, ntor = {b_ntor}
    max_nfev    = {max_nfev}
    trust radius = {trust_radius:.6e}
    free DOFs   = {n_free}
    optimizer DOFs = {prob.dof_size}
    vmec iter   = {vmec.iter} (cumulative)
"""
    )

    # Optimize
    least_squares_mpi_solve(
        prob,
        mpi,
        grad=True,
        max_nfev=max_nfev,
        bounds=(trust_lower, trust_upper),
    )

    # Preserve wout from this step
    vmec.files_to_delete = []

    # Post-step diagnostics re-trigger vmec.run() on the optimizer's current
    # DOF state. If that state no longer converges, the stage-1 candidate is
    # invalid and must not seed the next resolution step.
    step_runtime = time.time() - step_t0
    try:
        qs_total = float(sum(np.sum(qs.J()**2) for qs in qs_list))
        obj = prob.objective()
        aspect = float(vmec.aspect())
        iota_ax = float(vmec.iota_axis())
        iota_ed = float(vmec.iota_edge())
        iota_working = _working_iota()
        layer_shear = _working_layer_shear_slope()
        layer_shear_shortfall = max(0.0, WORKING_LAYER_SHEAR_MIN - layer_shear)
        vol = float(vmec.volume())
    except ObjectiveFailure as e:
        if mpi.proc0_world and registry is not None:
            registry.mark_failed(
                "stage1", RUN_ID,
                error_code="solver_diverged",
                error_message=f"post-step VMEC diagnostics failed at step {step+1}: {e}",
                slurm_wall_s=float(time.time() - t0),
            )
        raise SystemExit(1) from e

    proc0_print(
        f"""
    STEP {step+1} RESULTS (runtime: {timedelta(seconds=step_runtime)}):
        Aspect ratio:    {aspect:.6f}  (target: {ASPECT_TARGET})
        Iota axis:       {iota_ax:.6f}  (target: {IOTA_TARGET})
        Iota edge:       {iota_ed:.6f}  (target: {IOTA_TARGET})
        Iota s={WORKING_S:.3f}:  {iota_working:.6f}  (target: {IOTA_TARGET})
        Layer shear:     {layer_shear:.6f}  (shortfall: {layer_shear_shortfall:.6f})
        Volume:          {vol:.6f} m^3
        QS metric total: {qs_total:.6e}
        Objective:       {obj:.6e}
        VMEC iter:       {vmec.iter}
"""
    )

    # Write diagnostics row
    if mpi.proc0_world:
        with open(DIAGNOSTICS_FILE, 'a') as f:
            f.write(
                f'{step+1},{max_mode},{v_mpol},{v_ntor},{b_mpol},{b_ntor},'
                f'{max_nfev},{vmec.iter},'
                f'{aspect:.6e},{iota_ax:.6e},{iota_ed:.6e},{iota_working:.6e},'
                f'{layer_shear:.6e},{layer_shear_shortfall:.6e},{vol:.6e},'
                f'{qs_total:.6e},{obj:.6e},'
                f'{step_runtime:.2f}\n'
            )


# ──────────────────────────────────────────────────────────────────────────────
# Termination summary
# ──────────────────────────────────────────────────────────────────────────────
total_runtime = time.time() - t0

# Boundary optimization changes the LCFS area while phiedge stays fixed. Re-match
# phiedge to the signed TF-current rbtor before the final wout is written.
# simsopt's Vmec.run() is a no-op while need_to_run_code is False (vmec.py:154-157,
# 667-669) and the flag is not auto-flipped by mutating indata, so set it
# explicitly before re-running.
pre_match_rbtor = float(vmec.wout.rbtor)
vmec.indata.phiedge = rescale_phiedge_to_rbtor(
    vmec.indata.phiedge,
    pre_match_rbtor,
    VMEC_RBTOR,
)
vmec.need_to_run_code = True
vmec.run()
post_match_rbtor = float(vmec.wout.rbtor)
if (post_match_rbtor > 0) != (VMEC_RBTOR > 0):
    raise RuntimeError(
        f"VMEC rbtor rematch produced wrong sign: target={VMEC_RBTOR:.6e}, "
        f"pre={pre_match_rbtor:.6e}, got={post_match_rbtor:.6e}"
    )
proc0_print(
    f"Final TF rbtor match: {pre_match_rbtor:.6e} -> "
    f"{post_match_rbtor:.6e} T*m (target {VMEC_RBTOR:.6e})"
)

# Re-evaluate final state. A non-converging final state is not a valid stage-1
# artifact, even if an earlier resolution step produced a wout.
try:
    qs_total = float(sum(np.sum(qs.J()**2) for qs in qs_list))
    obj = prob.objective()
    final_aspect = float(vmec.aspect())
    final_iota_ax = float(vmec.iota_axis())
    final_iota_ed = float(vmec.iota_edge())
    final_iota_working = _working_iota()
    final_layer_shear = _working_layer_shear_slope()
    final_layer_shear_shortfall = max(
        0.0,
        WORKING_LAYER_SHEAR_MIN - final_layer_shear,
    )
    final_vol = float(vmec.volume())
except ObjectiveFailure as e:
    if mpi.proc0_world and registry is not None:
        registry.mark_failed(
            "stage1", RUN_ID,
            error_code="solver_diverged",
            error_message=f"final VMEC diagnostics failed: {e}",
            slurm_wall_s=float(total_runtime),
        )
    raise SystemExit(1) from e

aspect_err = abs(final_aspect - ASPECT_TARGET)
iota_axis_err = abs(final_iota_ax - IOTA_TARGET)
iota_edge_err = abs(final_iota_ed - IOTA_TARGET)
iota_working_err = abs(final_iota_working - IOTA_TARGET)

# Success criterion (cold-start update, 2026-04-12): gate only on VMEC
# convergence. Physics-metric checks (iota, aspect, volume) are
# evaluated externally by a post-processing script over the diagnostics CSV
# so that Pareto-scan runs always complete and produce outputs regardless
# of whether a specific (R0, V, iota) target was achievable — see the
# "Success metrics" section of local/cold_start_stage1_prompt.md. Errors are
# still computed below for the FINAL STATE report.
success = True

proc0_print(
    f"""
[{datetime.now()}] ...optimization complete
Total runtime: {timedelta(seconds=total_runtime)}

{'SUCCESS' if success else 'FAILURE'} ─────────────────────────────────────────
    Aspect ratio:    {final_aspect:.6f}  (target: {ASPECT_TARGET}, err: {aspect_err:.6f})
    Iota axis:       {final_iota_ax:.6f}  (target: {IOTA_TARGET}, err: {iota_axis_err:.6f})
    Iota edge:       {final_iota_ed:.6f}  (target: {IOTA_TARGET}, err: {iota_edge_err:.6f})
    Iota s={WORKING_S:.3f}:  {final_iota_working:.6f}  (target: {IOTA_TARGET}, err: {iota_working_err:.6f})
    Layer shear:     {final_layer_shear:.6f}  (min: {WORKING_LAYER_SHEAR_MIN}, shortfall: {final_layer_shear_shortfall:.6f})
    Volume:          {final_vol:.6f} m^3
    QS metric total: {qs_total:.6e}
    Final objective: {obj:.6e}
    VMEC iterations: {vmec.iter}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Print final state
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
FINAL STATE ───────────────────────────────────
    Aspect ratio:        {final_aspect:.6f}  (target: {ASPECT_TARGET})
    Iota axis:           {final_iota_ax:.6f}  (target: {IOTA_TARGET})
    Iota edge:           {final_iota_ed:.6f}  (target: {IOTA_TARGET})
    Iota s={WORKING_S:.3f}:      {final_iota_working:.6f}  (target: {IOTA_TARGET})
    Layer shear slope:   {final_layer_shear:.6f}  (min: {WORKING_LAYER_SHEAR_MIN})
    Volume:              {final_vol:.6f} m^3  (target: {VOLUME_TARGET})
    VMEC mpol:           {vmec.indata.mpol}
    VMEC ntor:           {vmec.indata.ntor}
    Boundary ndofs:      {len(surf.get_dofs())}

    Per-surface QS metrics:"""
)
for i, (s, qs) in enumerate(zip(QS_SURFACES, qs_list)):
    _qs_val = float(np.sum(qs.J()**2))
    proc0_print(f'        s={s:.2f}:  {_qs_val:.6e}')

proc0_print(f'    QS total:            {qs_total:.6e}')


# ──────────────────────────────────────────────────────────────────────────────
# Save optimized input + wout + boozmn
# ──────────────────────────────────────────────────────────────────────────────
input_opt_path = artifact_path("stage1", RUN_ID, OUT_DIR, "vmec_input_opt")
wout_opt_path = artifact_path("stage1", RUN_ID, OUT_DIR, "wout_opt")

# Save only the final converged VMEC state. Earlier intermediate wouts are
# diagnostic breadcrumbs, not valid downstream Stage 1 artifacts.
wout_src = vmec.output_file if vmec.output_file and os.path.exists(vmec.output_file) else None
_wout_saved = False
if mpi.proc0_world:
    if wout_src and os.path.exists(wout_src):
        vmec.write_input(input_opt_path)
        proc0_print(f'Optimized VMEC input saved to {input_opt_path}')
        shutil.copy2(wout_src, wout_opt_path)
        proc0_print(f'Optimized wout saved to {wout_opt_path}')
        _wout_saved = True
    else:
        proc0_print(
            'ERROR: Final converged VMEC state did not produce a wout. '
            'Skipping boozmn and BoozerSurface writes.'
        )
        if registry is not None:
            registry.mark_failed(
                "stage1", RUN_ID,
                error_code="file_save_failed",
                error_message="final converged VMEC state did not produce a wout",
                slurm_wall_s=float(total_runtime),
            )
_wout_saved = mpi.comm_world.bcast(_wout_saved, root=0)
if not _wout_saved:
    raise SystemExit(1)

# Write classic boozmn_*.nc via booz_xform. boozer.bx is a direct reference to
# the native booz_xform.Booz_xform C++ object (SIMSOPT's Boozer.save() only
# writes the generic Optimizable JSON, not the standard boozmn NetCDF).
# The FINAL STATE block above calls qs.J() which triggers boozer.bx.run() on
# the final equilibrium, so bx results are current. Naming convention mirrors
# VMEC's wout_<ext>.nc → boozmn_<ext>.nc.
boozmn_path = artifact_path("stage1", RUN_ID, OUT_DIR, "boozmn_opt")
# Only write boozmn when the final-state eval succeeded. boozer.bx holds
# state from the final booz_xform run.
if mpi.proc0_world:
    boozer.bx.write_boozmn(boozmn_path)
    proc0_print(f'Boozer transform saved to {boozmn_path}')


# ──────────────────────────────────────────────────────────────────────────────
# Return to original directory and clean up per-group temp dirs
# ──────────────────────────────────────────────────────────────────────────────
atexit.unregister(_cleanup_group_dir)
_cleanup_group_dir()


# ──────────────────────────────────────────────────────────────────────────────
# Build BoozerSurface (coils + surface) for stage 2 warm-start
# ──────────────────────────────────────────────────────────────────────────────
_bsurf_saved = False
if mpi.proc0_world and _wout_saved:
    proc0_print('\nBuilding BoozerSurface from optimized wout for stage 2...')
    bsurf_out_path = artifact_path("stage1", RUN_ID, OUT_DIR, "bsurf_opt")
    build_and_save(cfg, wout_path=wout_opt_path, out_path=bsurf_out_path,
                   save_vtk=True, print_fn=proc0_print)
    proc0_print(f'BoozerSurface saved to {bsurf_out_path}')
    _bsurf_saved = True
elif mpi.proc0_world:
    raise RuntimeError('final VMEC state did not produce a wout for BoozerSurface build')

proc0_print(f'\nDiagnostics saved to {DIAGNOSTICS_FILE}')
proc0_print(f'Outputs saved to {RUN_DIR}')


# ──────────────────────────────────────────────────────────────────────────────
# Registry: finalize the row with metrics or an error code
# ──────────────────────────────────────────────────────────────────────────────
# Success criterion mirrors the termination banner above but also
# requires that we actually saved the wout and built the BoozerSurface — a
# stage 1 row without those artifacts is useless to downstream stages.
if mpi.proc0_world and registry is not None:
    _metrics = {
        "final_iota_axis": final_iota_ax,
        "final_iota_edge": final_iota_ed,
        "final_iota_working_s": final_iota_working,
        "final_working_s": WORKING_S,
        "final_layer_shear_slope": final_layer_shear,
        "final_layer_shear_shortfall": final_layer_shear_shortfall,
        "final_aspect":    final_aspect,
        "final_volume":    final_vol,
        "final_qs_metric": qs_total,
        "nfev":            int(vmec.iter),
        "runtime_s":       float(total_runtime),
    }
    if _wout_saved and _bsurf_saved:
        registry.mark_success("stage1", RUN_ID, metrics=_metrics,
                              slurm_wall_s=float(total_runtime))
    else:
        if not _wout_saved:
            err_code = "solver_diverged"
            err_msg  = "no converging VMEC run across resolution ramp"
        else:
            err_code = "file_save_failed"
            err_msg  = "wout saved but BoozerSurface build failed"
        registry.mark_failed("stage1", RUN_ID,
                             error_code=err_code, error_message=err_msg,
                             slurm_wall_s=float(total_runtime),
                             metrics=_metrics)
