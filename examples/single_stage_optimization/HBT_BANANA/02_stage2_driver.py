"""
02_stage2_driver.py
───────────────────
Stage 2 coil-only optimization for the banana coil stellarator-tokamak hybrid.

This experimental HBT_BANANA copy is weighted-only: fixed-weight L-BFGS-B on
a single scalar objective
    JF = w_sqf*Jsqf + w_l*Jl + w_cc*Jcc + w_curv*Jcurv
         [+ w_curr*Jcurr when current_mode_stage2='penalized'].

Pipeline:  01_stage1 -> 02_stage2 (this) -> 03_singlestage

Usage:
    python 02_stage2_driver.py
"""
import atexit
import re
import numpy as np
import os
import sys
import time
import yaml

from datetime import datetime, timedelta
from scipy.optimize import minimize

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'utils'))
sys.path.insert(0, os.path.join(_THIS_DIR, 'new_objectives'))
from output_dir import resolve_output_dir
from current_penalty import CurrentPenaltyWrapper
from hardware_metrics import (
    boozer_json_vacuum_lineage,
    curve_poloidal_half_extent,
)
from run_registry import RunRegistry, artifact_path, install_atexit_handler, run_dir
from poloidal_extent import PoloidalExtent
from ellipse_width import ProjectedEllipseWidth
from self_intersect import CurveSelfIntersect
from global_curvature_radius import GlobalRadiusCurvature

from simsopt._core import load
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    LpCurveCurvature,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux


def proc0_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Load configuration
# ──────────────────────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(_cfg_path) as _f:
    cfg = yaml.safe_load(_f)

# TF coils
TF_NUM = cfg['tf_coils']['num']

# Banana coils
BANANA_CURV_P              = cfg['banana_coils']['curv_p']
BANANA_CURRENT_MAX         = cfg['banana_coils']['current_max']
BANANA_CURRENT_FIXED_S2    = float(os.environ.get(
    'BANANA_I_FIXED_S2',
    cfg['banana_coils']['current_fixed_stage2']
))
BANANA_CURRENT_CAP         = cfg['banana_coils'].get('current_cap_stage2', True)
WINDING_R0                 = float(cfg['winding_surface']['R0'])
WINDING_A                  = float(cfg['winding_surface']['a'])
# TF current penalty bound: |I_tf| penalized above its operating magnitude
# (80 kA), matching the reference hardware limit. TF coils are the fixed
# background field, so this term is normally inactive; it guards against an
# unfixed TF current drifting past hardware.
TF_CURRENT_MAX             = abs(float(cfg['tf_coils']['current']))

# Stage 2 current handling: 'free' | 'penalized' | 'fixed'
STAGE2_CURRENT_MODE = os.environ.get(
    'BANANA_CURRENT_MODE_S2',
    cfg['banana_coils'].get('current_mode_stage2', 'fixed')
).lower()
if STAGE2_CURRENT_MODE not in ('free', 'penalized', 'fixed'):
    raise ValueError(
        f"current_mode_stage2 must be 'free', 'penalized', or 'fixed', "
        f"got {STAGE2_CURRENT_MODE!r}"
    )

# Hardware engineering tolerances (enforced unmodified by singlestage).
LENGTH_MAX_HW = float(cfg['thresholds']['length_max'])
LENGTH_TARGET_HW = float(cfg['thresholds']['length_target'])
CC_MIN_HW     = float(cfg['thresholds']['coil_coil_min'])
CURV_MAX_HW   = float(cfg['thresholds']['curvature_max'])
# Width and self-intersection targets are enforced unmodified (no stage 2
# relaxation), matching the banana_drivers reference objective set.
WIDTH_MAX_HW   = float(cfg['thresholds']['width_max'])
WIDTH_MIN_HW   = float(cfg['thresholds']['width_min'])
SELFINT_MIN_HW = float(cfg['thresholds']['self_intersect_min'])
GCR_MIN_HW     = float(cfg['thresholds']['global_curvature_radius_min'])
GCR_EXP_WEIGHT = float(cfg['thresholds']['global_curvature_exp_weight'])

# Stage 2 per-threshold relaxation factors (env var > config > 1.0).
# Stage 2 only needs coils good enough for singlestage to polish; relaxing
# its thresholds lets L-BFGS-B drive sqflx lower and shrinks axis drift.
LENGTH_RELAX = float(os.environ.get(
    'BANANA_STAGE2_LENGTH_RELAX', cfg['stage2_relaxation']['length']))
CC_RELAX     = float(os.environ.get(
    'BANANA_STAGE2_CC_RELAX',     cfg['stage2_relaxation']['coil_coil']))
CURV_RELAX   = float(os.environ.get(
    'BANANA_STAGE2_CURV_RELAX',   cfg['stage2_relaxation']['curvature']))
POLOIDAL_RELAX = float(cfg['stage2_relaxation']['poloidal_extent'])

# Effective thresholds seen by the stage 2 objective.
LENGTH_THRESHOLD = LENGTH_TARGET_HW * LENGTH_RELAX
CC_THRESHOLD     = CC_MIN_HW     / CC_RELAX
CURV_THRESHOLD   = CURV_MAX_HW   * CURV_RELAX
POLOIDAL_THRESHOLD_RAD = np.deg2rad(
    float(cfg['thresholds']['poloidal_half_width_max_deg']) * POLOIDAL_RELAX
)

# Weighted-mode params
STAGE2_MODE = 'weighted'
SQF_WEIGHT  = float(cfg['stage2_weights']['squared_flux'])
LEN_WEIGHT  = float(cfg['stage2_weights']['length'])
CC_WEIGHT   = float(cfg['stage2_weights']['coil_coil'])
CURV_WEIGHT = float(cfg['stage2_weights']['curvature'])
POL_WEIGHT  = float(cfg['stage2_weights']['poloidal_extent'])
CURR_WEIGHT = float(cfg['stage2_weights']['current'])
WIDTH_WEIGHT   = float(cfg['stage2_weights']['width'])
SELFINT_WEIGHT = float(cfg['stage2_weights']['selfint'])
GCR_WEIGHT     = float(cfg['stage2_weights']['global_curvature'])

MAXITER = int(cfg['stage2_optimizer']['maxiter'])
MAXCOR  = int(cfg['stage2_optimizer']['maxcor'])
MAXFUN  = int(cfg['stage2_optimizer']['maxfun'])
TOL     = float(cfg['stage2_optimizer']['tol'])
FTOL    = float(cfg['stage2_optimizer']['ftol'])
GTOL    = float(cfg['stage2_optimizer']['gtol'])


# ──────────────────────────────────────────────────────────────────────────────
# Output directory, registry registration, and atexit handler
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR = resolve_output_dir()

# Parent stage 1 run id — hard-required. Reproducibility over ergonomics:
# the user must pin the parent explicitly in config.yaml (no "latest run"
# shortcut).
STAGE1_ID = cfg['warm_start'].get('stage1_id')
if not STAGE1_ID:
    raise ValueError(
        "02_stage2: cfg['warm_start']['stage1_id'] is required "
        "(set to the s01_xxxxxx id of the stage 1 run to warm-start from)."
    )

_slurm_meta = {
    "slurm_qos":           os.environ.get("SLURM_JOB_QOS"),
    "slurm_partition":     os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_ntasks":        int(os.environ["SLURM_NTASKS"]) if os.environ.get("SLURM_NTASKS") else None,
    "slurm_cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]) if os.environ.get("SLURM_CPUS_PER_TASK") else None,
    "slurm_time_limit_s":  None,
}
_slurm_job_id = os.environ.get("SLURM_JOB_ID")

# Write env-resolved values back into cfg so content-addressed hashing sees
# the effective inputs. Pareto sweeps vary these via env vars; without this
# write-back, runs collide on the same run_id.
cfg['stage2_mode']                                = STAGE2_MODE
cfg['banana_coils']['current_mode_stage2']        = STAGE2_CURRENT_MODE
cfg['banana_coils']['current_fixed_stage2']       = BANANA_CURRENT_FIXED_S2
cfg['stage2_relaxation']['length']                = LENGTH_RELAX
cfg['stage2_relaxation']['coil_coil']             = CC_RELAX
cfg['stage2_relaxation']['curvature']             = CURV_RELAX
cfg['stage2_relaxation']['poloidal_extent']       = POLOIDAL_RELAX

registry = RunRegistry()
RUN_ID, _is_new = registry.register_stage2(cfg, stage1_id=STAGE1_ID,
                                           slurm_meta=_slurm_meta)
install_atexit_handler(registry, "stage2", RUN_ID)

RUN_DIR = run_dir("stage2", RUN_ID, OUT_DIR)
os.makedirs(RUN_DIR, exist_ok=True)

# Warm-start stage 1 bsurf is determined exclusively by the parent stage 1
# run id. No override: the hashed stage1_id must match the file actually
# loaded, or content-addressing is a lie.
INIT_BSURF_FILE = os.path.abspath(
    artifact_path("stage1", STAGE1_ID, OUT_DIR, "bsurf_opt")
)

DIAGNOSTICS_FILE = artifact_path("stage2", RUN_ID, OUT_DIR, "diagnostics")


def _emit_out_dir_on_exit():
    """Print per-run directory so the shell script can move the log file."""
    proc0_print(f"OUT_DIR={RUN_DIR}")


atexit.register(_emit_out_dir_on_exit)


# ──────────────────────────────────────────────────────────────────────────────
# Print input parameters
# ──────────────────────────────────────────────────────────────────────────────
_header = f"""
INPUT PARAMETERS ─────────────────────────────
    Config:          {_cfg_path}
    Date:            {datetime.now()}
    Mode:            weighted
    Run ID:          {RUN_ID} ({'new' if _is_new else 'rerun'})
    Run dir:         {RUN_DIR}

    Warm-start:
        stage1_id   = {STAGE1_ID}
        bsurf       = {INIT_BSURF_FILE}

    Banana coils:
        curv p-norm      = {BANANA_CURV_P}
        current_max (HW) = {BANANA_CURRENT_MAX/1e3:.1f} kA  (enforced in singlestage)
        current_mode_s2  = {STAGE2_CURRENT_MODE}
        current_fixed_s2 = {BANANA_CURRENT_FIXED_S2/1e3:.1f} kA  (used when mode='fixed')
        current_pen_max  = {BANANA_CURRENT_MAX/1e3:.1f} kA banana / {TF_CURRENT_MAX/1e3:.1f} kA TF  (penalty thresholds when mode='penalized')
        current_cap_hard = {BANANA_CURRENT_CAP} (L-BFGS-B bound)
        winding_surface  = R0 {WINDING_R0:.3f} m, a {WINDING_A:.3f} m

    Thresholds (HW tolerance × stage 2 relaxation = effective):
        length_abs  = {LENGTH_MAX_HW} m
        length_tgt  = {LENGTH_TARGET_HW} m × {LENGTH_RELAX} = {LENGTH_THRESHOLD} m
        length_min  = {LENGTH_THRESHOLD/2} m  (half the max-length threshold)
        cc_min      = {CC_MIN_HW} m        / {CC_RELAX}     = {CC_THRESHOLD} m
        curv_max    = {CURV_MAX_HW} m^-1   × {CURV_RELAX}   = {CURV_THRESHOLD} m^-1
        poloidal    = {cfg['thresholds']['poloidal_half_width_max_deg']} deg × {POLOIDAL_RELAX} = {POLOIDAL_THRESHOLD_RAD:.6f} rad
        width       = [{WIDTH_MIN_HW}, {WIDTH_MAX_HW}] m
        selfint     = {SELFINT_MIN_HW} m
        gcr_min     = {GCR_MIN_HW} m  (barrier softness {GCR_EXP_WEIGHT} m)

    Objective weights:
        squared_flux = {SQF_WEIGHT:.3e}
        length       = {LEN_WEIGHT:.3e}
        coil_coil    = {CC_WEIGHT:.3e}
        curvature    = {CURV_WEIGHT:.3e}
        poloidal     = {POL_WEIGHT:.3e}
        width        = {WIDTH_WEIGHT:.3e}
        selfint      = {SELFINT_WEIGHT:.3e}{'' if SELFINT_WEIGHT else '  (diagnostic only)'}
        global_curv  = {GCR_WEIGHT:.3e}{'' if GCR_WEIGHT else '  (diagnostic only)'}
        current      = {CURR_WEIGHT:.3e}  (used when current_mode_s2='penalized')

    Optimizer (L-BFGS-B):
        maxiter = {MAXITER}
        maxcor  = {MAXCOR}
        maxfun  = {MAXFUN}
        tol     = {TOL:.3e}
        ftol    = {FTOL:.3e}
        gtol    = {GTOL:.3e}
"""
proc0_print(_header)


# ──────────────────────────────────────────────────────────────────────────────
# Load warm-start BoozerSurface and extract coils
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(f'Loading BoozerSurface from {INIT_BSURF_FILE}')
boozersurface = load(INIT_BSURF_FILE)
surface = boozersurface.surface
biotsavart = boozersurface.biotsavart
coils = biotsavart.coils

tf_coils = coils[:TF_NUM]
banana_coils = coils[TF_NUM:]
banana_curves = [coil.curve for coil in banana_coils]
banana_curve = banana_curves[0]
banana_current = banana_coils[0].current
tf_current_0 = tf_coils[0].current

# ─────────────────────────────────────────────────────────────────────────────
# Apply current mode: pin + fix the DOF when mode='fixed'.
#
# banana_current is ScaledCurrent(Current(1), scale).  get_value() is linear
# in the single underlying Current DOF, so scaling .x by (target / current)
# sets the physical value exactly.  fix_all() walks the tree and fixes the
# child DOF so it is removed from the free-DOF set before JF is built.
# ─────────────────────────────────────────────────────────────────────────────
if STAGE2_CURRENT_MODE == 'fixed':
    inner = banana_current.current_to_scale
    current_now = banana_current.get_value()
    inner.x = inner.x * (BANANA_CURRENT_FIXED_S2 / current_now)
    banana_current.fix_all()
    proc0_print(
        f'  Banana current pinned at {banana_current.get_value()/1e3:.1f} kA '
        f'and fixed (mode=fixed).'
    )
else:
    proc0_print(f'  Banana current free (mode={STAGE2_CURRENT_MODE}).')

# Use the BoozerSurface's own surface for SquaredFlux evaluation.
# With stellsym coils, one field period is sufficient (no full-torus needed).
biotsavart.set_points(surface.gamma().reshape((-1, 3)))
proc0_print(f'  {len(tf_coils)} TF coils + {len(banana_coils)} banana coils loaded')

Bbs = biotsavart.B().reshape(surface.gamma().shape)
Bdotn_surf = np.sum(Bbs * surface.unitnormal(), axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Define objective function
# ──────────────────────────────────────────────────────────────────────────────
Jsqf  = SquaredFlux(surface, biotsavart, definition="normalized")
_Jl   = CurveLength(banana_curve)
Jl    = QuadraticPenalty(_Jl, LENGTH_THRESHOLD, "max")
# Lower length bound at half the max-length threshold (reference: max_length/2)
# keeps the coil from collapsing while the other penalties pull it inward.
Jl_min = QuadraticPenalty(_Jl, LENGTH_THRESHOLD / 2, "min")
# Coil-coil distance over banana coils only — TF coils are fixed background,
# so TF-TF and TF-banana pairs add no gradient (matches the reference).
Jcc   = CurveCurveDistance(banana_curves, CC_THRESHOLD)
Jcurv = LpCurveCurvature(banana_curve, BANANA_CURV_P, CURV_THRESHOLD)
Jpol  = PoloidalExtent(
    banana_curve, WINDING_R0, POLOIDAL_THRESHOLD_RAD, p=BANANA_CURV_P
)
# Projected-ellipse width, bounded above (port-fit) and below (anti-collapse).
_width = ProjectedEllipseWidth(banana_curve, WINDING_R0, WINDING_A)
Jwmax  = QuadraticPenalty(_width, WIDTH_MAX_HW, "max")
Jwmin  = QuadraticPenalty(_width, WIDTH_MIN_HW, "min")
# Self-intersection hinge (figure-8 prevention); neighbor_skip = int(1.5*order)
# mirrors the reference.
Jself  = CurveSelfIntersect(
    banana_curve, SELFINT_MIN_HW, int(1.5 * banana_curve.order)
)
# Gonzalez-Maddocks global radius of curvature: the same self-contact
# constraint as Jself, but smooth on its noncoincident/nonparallel branch and
# summed over every near-contact pair rather than hinged on the single worst
# one. Its exponential penalty is finite; exact sampled coincidences receive
# radius zero. Only one of the two belongs in JF; both are reported.
Jgcr   = GlobalRadiusCurvature(banana_curve, GCR_MIN_HW, GCR_EXP_WEIGHT)
# Current penalty (only when mode='penalized'): penalize |I_banana| above the
# 16 kA hardware limit and |I_tf| above 80 kA, matching the reference. In
# 'fixed' mode the banana current is pinned and dropped from the DOF set; in
# 'free' mode it is left unconstrained — both omit this term.
if STAGE2_CURRENT_MODE == 'penalized':
    _Jbananacurr = CurrentPenaltyWrapper(banana_current)
    Jbananacurr  = QuadraticPenalty(_Jbananacurr, BANANA_CURRENT_MAX, "max")
    _Jtfcurr     = CurrentPenaltyWrapper(tf_current_0)
    Jtfcurr      = QuadraticPenalty(_Jtfcurr, TF_CURRENT_MAX, "max")
    Jcurr        = Jtfcurr + Jbananacurr
else:
    Jcurr = None

JF = (SQF_WEIGHT * Jsqf) + (LEN_WEIGHT * Jl) + (LEN_WEIGHT * Jl_min) \
   + (CC_WEIGHT * Jcc) + (CURV_WEIGHT * Jcurv) + (POL_WEIGHT * Jpol) \
   + (WIDTH_WEIGHT * (Jwmax + Jwmin))

# Self-contact: Jgcr and Jself enforce the same constraint by different
# means, so whichever carries a nonzero weight owns it and the other stays a
# reported diagnostic. Zero-weight terms are left out of JF entirely rather
# than added as 0*J — CurveSelfIntersect is O(N^2) per evaluation and there
# is no reason to pay for it on the critical path.
if GCR_WEIGHT:
    JF = JF + (GCR_WEIGHT * Jgcr)
if SELFINT_WEIGHT:
    JF = JF + (SELFINT_WEIGHT * Jself)

if Jcurr is not None:
    JF = JF + (CURR_WEIGHT * Jcurr)


def _objective_lines():
    """Return weighted objective and gradient diagnostics."""
    return [
        ('Objective J (weighted)', f'{JF.J():.6e}'),
        ('||grad J||',             f'{np.linalg.norm(JF.dJ()):.6e}'),
    ]


def _format_objective_block(indent='        '):
    lines = _objective_lines()
    width = max(len(label) for label, _ in lines) + 2
    return '\n'.join(f'{indent}{label + ":":<{width}}             {val}' for label, val in lines)


# ──────────────────────────────────────────────────────────────────────────────
# Print initial state
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
INITIAL STATE ─────────────────────────────────
    Parameter values:
        Banana coil current:             {banana_current.get_value()/1e3:.6e} kA
        Mean |B.N|:                      {np.mean(np.abs(Bdotn_surf)):.6e}
        Squared flux (SquaredFlux.J):    {Jsqf.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1
        Poloidal half extent:            {curve_poloidal_half_extent(banana_curve, WINDING_R0):.6e} rad
        Min global curv radius:          {Jgcr.shortest_radius():.6e} m (limit: {GCR_MIN_HW:.3e} m)
        Shortest self-distance:          {Jself.shortest_self_distance():.6e} m (limit: {SELFINT_MIN_HW:.3e} m)

    Objective ({STAGE2_MODE}):
{_format_objective_block()}

    Penalty values:
        Squared flux penalty:            {Jsqf.J():.6e}
        Length penalty (QuadPen.J):      {Jl.J():.6e}
        CC distance penalty:             {Jcc.J():.6e}
        Curvature penalty (LpCurvCurv):  {Jcurv.J():.6e}
        Poloidal extent penalty:         {Jpol.J():.6e}

    n_dofs = {len(JF.x)}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Optimization tracking and diagnostics
# ──────────────────────────────────────────────────────────────────────────────
track = dict(
    eval=0,
    iter=0,
    f_prev=None,
    f_curr=None,
)


def _write_diagnostics_row(J, dJ, t0):
    """Append a single diagnostics row to the CSV file (inner-loop tracking)."""
    t_elapsed = time.time() - t0
    dJ_norm = np.linalg.norm(dJ)

    track['eval'] += 1
    row = (
        f"{track['iter']},{track['eval']},{t_elapsed:.2f},"
        f"{J:.6e},{dJ_norm:.6e},"
        f"{Jsqf.J():.6e},"
        f"{_Jl.J():.6e},"
        f"{Jcc.shortest_distance():.6e},"
        f"{banana_curve.kappa().max():.6e},"
        f"{curve_poloidal_half_extent(banana_curve, WINDING_R0):.6e},"
        f"{Jgcr.shortest_radius():.6e},"
        f"{Jself.shortest_self_distance():.6e}"
    )
    proc0_print(row)
    with open(DIAGNOSTICS_FILE, 'a') as f:
        f.write(row + "\n")


def fun(x):
    """Weighted-mode objective for L-BFGS-B (inner-loop evaluation)."""
    JF.x = x
    J = JF.J()
    dJ = JF.dJ()
    _write_diagnostics_row(J, dJ, t0)
    return J, dJ


def _print_state(iter_label):
    runtime = time.time() - t0
    Bdotn = np.mean(np.abs(np.sum(
        biotsavart.B().reshape(surface.gamma().shape) * surface.unitnormal(),
        axis=-1
    )))
    proc0_print(
        f"""
[{datetime.now()}; {timedelta(seconds=runtime)} elapsed] {iter_label}
    Parameter values:
        Banana coil current:             {banana_current.get_value()/1e3:.6e} kA
        Mean |B.N|:                      {Bdotn:.6e}
        Squared flux (SquaredFlux.J):    {Jsqf.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1
        Poloidal half extent:            {curve_poloidal_half_extent(banana_curve, WINDING_R0):.6e} rad
        Min global curv radius:          {Jgcr.shortest_radius():.6e} m (limit: {GCR_MIN_HW:.3e} m)
        Shortest self-distance:          {Jself.shortest_self_distance():.6e} m (limit: {SELFINT_MIN_HW:.3e} m)

    Objective ({STAGE2_MODE}):
{_format_objective_block()}

    Penalty values:
        Squared flux penalty:            {Jsqf.J():.6e}
        Length penalty (QuadPen.J):      {Jl.J():.6e}
        CC distance penalty:             {Jcc.J():.6e}
        Curvature penalty (LpCurvCurv):  {Jcurv.J():.6e}
        Poloidal extent penalty:         {Jpol.J():.6e}
"""
    )


def callback_weighted(x):
    """L-BFGS-B iteration callback (weighted mode)."""
    J = JF.J()
    track['f_prev'] = track['f_curr']
    track['f_curr'] = J
    track['iter'] += 1
    track['eval'] = 0
    _print_state(f"ITERATION {track['iter']:03d}/{MAXITER}")


# ──────────────────────────────────────────────────────────────────────────────
# Initialize diagnostics file
# ──────────────────────────────────────────────────────────────────────────────
t0 = time.time()

with open(DIAGNOSTICS_FILE, 'w') as f:
    f.write('# Stage 2 Diagnostics\n')
    f.write(f'# Date: {datetime.now()}\n')
    f.write(f'# Mode: {STAGE2_MODE}\n')
    f.write(f'# TF: {len(tf_coils)} coils, Banana: {banana_current.get_value()/1e3:.0f} kA (init)\n')
    f.write(f'# LENGTH_THRESHOLD={LENGTH_THRESHOLD}, CC_THRESHOLD={CC_THRESHOLD}, CURV_THRESHOLD={CURV_THRESHOLD}, POLOIDAL_THRESHOLD_RAD={POLOIDAL_THRESHOLD_RAD}\n')
    f.write(f'# MAXITER={MAXITER}, FTOL={FTOL:.3e}, GTOL={GTOL:.3e}\n')
    f.write(
        'iter,eval,runtime,'
        'objective,grad_norm,'
        'sqflx,'
        'coil_length,'
        'ccdist,'
        'max_kappa,'
        'poloidal_extent_rad,'
        'min_global_curv_radius,'
        'shortest_self_distance\n'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Run optimization
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(f'[{datetime.now()}] Starting stage 2 optimization...')
registry.mark_running("stage2", RUN_ID, slurm_job_id=_slurm_job_id)
x0 = JF.x

# L-BFGS-B bounds: optionally cap banana current DOF at BANANA_CURRENT_MAX
bounds = None
if BANANA_CURRENT_CAP and banana_current.dof_names:
    dof_names = JF.dof_names
    banana_current_dof = banana_current.dof_names[0]
    if banana_current_dof in dof_names:
        current_dof_idx = dof_names.index(banana_current_dof)
        bounds = [(None, None)] * len(x0)
        banana_dof_val = x0[current_dof_idx]
        banana_phys_val = banana_current.get_value()
        bound_abs = abs(BANANA_CURRENT_MAX * banana_dof_val / banana_phys_val)
        bounds[current_dof_idx] = (-bound_abs, bound_abs)
        proc0_print(f'    Bound on DOF[{current_dof_idx}] ({dof_names[current_dof_idx]}): '
                    f'abs <= {bound_abs:.4f}'
                    f' (physical: |I| <= {BANANA_CURRENT_MAX/1e3:.0f} kA)')
    else:
        raise RuntimeError('banana current DOF not found in stage 2 objective')
elif BANANA_CURRENT_CAP:
    proc0_print('    No current bound (banana current is fixed)')
else:
    proc0_print('    No current bound (current_cap_stage2=false)')

res = minimize(
    fun, JF.x, jac=True, method='L-BFGS-B', tol=TOL,
    bounds=bounds,
    callback=callback_weighted,
    options=dict(maxiter=MAXITER, maxcor=MAXCOR, maxfun=MAXFUN,
                 ftol=FTOL, gtol=GTOL),
)


# ──────────────────────────────────────────────────────────────────────────────
# Termination summary
# ──────────────────────────────────────────────────────────────────────────────
end_date = datetime.now()
opt_runtime = time.time() - t0

hit_maxiter = res.nit >= MAXITER
hit_maxfun  = res.nfev >= MAXFUN
grad_inf    = np.max(np.abs(res.jac)) if hasattr(res, 'jac') and res.jac is not None else float('nan')
hit_gtol    = grad_inf <= GTOL

EPSMCH  = np.finfo(float).eps
FACTR   = FTOL / EPSMCH
f_curr  = track['f_curr']
f_prev  = track['f_prev']
if f_prev is None:
    rel_red_str = 'nan'
    f_cond_str = f"F={f_curr}, F_prev={f_prev}"
else:
    rel_red = (f_prev - f_curr) / max(1.0, abs(f_prev), abs(f_curr))
    rel_red_str = f"{rel_red:.3e}"
    f_cond_str = f"F={f_curr:.6e}, F_prev={f_prev:.6e}"
hit_ftol = bool(re.search(
    r'REL[_\s]REDUCTION[_\s]OF[_\s]F|RELATIVE\s+REDUCTION\s+OF\s+F',
    res.message, re.IGNORECASE
))

if res.success:
    verdict = 'CONVERGED' if hit_gtol else 'WARNING'
else:
    verdict = 'BUDGET_EXHAUSTED' if hit_maxiter else 'FAILURE'
stage2_ok = verdict == 'CONVERGED'

verdict_explanations = {
    'CONVERGED':        'gtol satisfied',
    'BUDGET_EXHAUSTED': 'maxiter reached before convergence',
    'WARNING':          'early exit before maxiter, gradient not small',
    'FAILURE':          'scipy internal failure',
}

proc0_print(
    f"""
[{end_date}] ...optimization complete
Total runtime: {timedelta(seconds=opt_runtime)}

{verdict} ─────────────────────────────────────────
    Banana coil current : {banana_current.get_value()/1e3:.5f} kA
    scipy message       : {res.message}
    scipy success       : {res.success}
    iterations          : {res.nit} / {MAXITER}  (maxiter {'REACHED' if hit_maxiter else 'not reached'})
    fun evals           : {res.nfev} / {MAXFUN}  (maxfun  {'REACHED' if hit_maxfun  else 'not reached'})
    grad inf-norm       : {grad_inf:.3e}  (gtol={GTOL:.3e}, {'SATISFIED' if hit_gtol else 'NOT satisfied'})
    ftol condition      : {'SATISFIED' if hit_ftol else 'NOT satisfied'}
        {f_cond_str}
        rel reduction = (F_prev-F)/max(1,|F_prev|,|F|) = {rel_red_str}
        threshold = FACTR*EPSMCH = ({FACTR:.3e})*({EPSMCH:.3e}) = {FTOL:.3e}
    final objective     : {res.fun:.6e}
    verdict             : {verdict}  ({verdict_explanations[verdict]})
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Print final state
# ──────────────────────────────────────────────────────────────────────────────
biotsavart.set_points(surface.gamma().reshape((-1, 3)))
Bbs = biotsavart.B().reshape(surface.gamma().shape)
Bdotn_surf = np.sum(Bbs * surface.unitnormal(), axis=-1)

proc0_print(
    f"""
FINAL STATE ───────────────────────────────────
    Parameter values:
        Banana coil current:             {banana_current.get_value()/1e3:.6e} kA
        Mean |B.N|:                      {np.mean(np.abs(Bdotn_surf)):.6e}
        Squared flux (SquaredFlux.J):    {Jsqf.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1
        Poloidal half extent:            {curve_poloidal_half_extent(banana_curve, WINDING_R0):.6e} rad
        Min global curv radius:          {Jgcr.shortest_radius():.6e} m (limit: {GCR_MIN_HW:.3e} m)
        Shortest self-distance:          {Jself.shortest_self_distance():.6e} m (limit: {SELFINT_MIN_HW:.3e} m)

    Objective ({STAGE2_MODE}):
{_format_objective_block()}

    Penalty values:
        Squared flux penalty:            {Jsqf.J():.6e}
        Length penalty (QuadPen.J):      {Jl.J():.6e}
        CC distance penalty:             {Jcc.J():.6e}
        Curvature penalty (LpCurvCurv):  {Jcurv.J():.6e}
        Poloidal extent penalty:         {Jpol.J():.6e}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Save final outputs
# ──────────────────────────────────────────────────────────────────────────────
# Save BoozerSurface (canonical opt output only on optimizer success).
_bsurf_kind = "bsurf_opt" if stage2_ok else "bsurf_failed"
_bsurf_out_path = artifact_path("stage2", RUN_ID, OUT_DIR, _bsurf_kind)
boozersurface.save(_bsurf_out_path)
_lineage_metrics = boozer_json_vacuum_lineage(_bsurf_out_path)

proc0_print(f'Diagnostics saved to {DIAGNOSTICS_FILE}')
proc0_print(f'Outputs saved to {RUN_DIR}')


# ──────────────────────────────────────────────────────────────────────────────
# Registry finalization
# ──────────────────────────────────────────────────────────────────────────────
_metrics = {
    "final_sqflx":          float(Jsqf.J()),
    "final_max_curvature":  float(banana_curve.kappa().max()),
    "final_max_length":     float(_Jl.J()),
    "final_min_cc_dist":    float(Jcc.shortest_distance()),
    "final_min_cs_dist":    None,
    "final_banana_current": float(banana_current.get_value()),
    "runtime_s":            float(opt_runtime),
    "tf_current_A":          float(cfg['tf_coils']['current']),
    "banana_current_max_abs_A": float(abs(banana_current.get_value())),
    "banana_current_limit_A": float(BANANA_CURRENT_MAX),
    "length_target_m":       float(LENGTH_TARGET_HW),
    "length_abs_max_m":      float(LENGTH_MAX_HW),
    "coil_coil_min_threshold_m": float(CC_MIN_HW),
    "curvature_threshold_inv_m": float(CURV_MAX_HW),
    "poloidal_extent_rad":   float(curve_poloidal_half_extent(banana_curve, WINDING_R0)),
    "poloidal_extent_threshold_rad": float(POLOIDAL_THRESHOLD_RAD),
    "min_global_curv_radius_m": float(Jgcr.shortest_radius()),
    "global_curv_radius_threshold_m": float(GCR_MIN_HW),
    "shortest_self_distance_m": float(Jself.shortest_self_distance()),
    "self_intersect_threshold_m": float(SELFINT_MIN_HW),
    "winding_R0_m":          float(WINDING_R0),
    "winding_a_m":           float(WINDING_A),
    **_lineage_metrics,
}
if stage2_ok and not _lineage_metrics["vacuum_lineage_ok"]:
    _err_code = "file_save_failed"
    _err_msg = "saved BoozerSurface failed vacuum lineage validation"
    registry.mark_failed("stage2", RUN_ID, error_code=_err_code,
                         error_message=_err_msg, slurm_wall_s=float(opt_runtime),
                         metrics=_metrics)
    raise RuntimeError(_err_msg)
if stage2_ok:
    registry.mark_success("stage2", RUN_ID, metrics=_metrics,
                          slurm_wall_s=float(opt_runtime))
else:
    _err_code = "timeout" if hit_maxiter or hit_maxfun else "solver_diverged"
    _err_msg = "stage 2 optimizer did not converge"
    registry.mark_failed("stage2", RUN_ID, error_code=_err_code,
                         error_message=_err_msg, slurm_wall_s=float(opt_runtime),
                         metrics=_metrics)
    raise RuntimeError(_err_msg)
