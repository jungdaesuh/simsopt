"""
03_singlestage_driver.py
────────────────────────
Stage 3 (singlestage) joint coil + surface optimization for the banana coil
stellarator-tokamak hybrid using BoozerLS.

Jointly optimizes banana coil DOFs and plasma surface Fourier coefficients
to minimize NonQuasiSymmetricRatio + BoozerResidual + geometric penalties
using L-BFGS-B.

Pipeline:  01_stage1 -> 02_stage2 -> 03_singlestage (this)

Usage:
    python 03_singlestage_driver.py
"""
import atexit
import numpy as np
import os
import re
import sys
import time
import yaml

from datetime import datetime, timedelta
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from output_dir import resolve_output_dir
from current_penalty import CurrentPenaltyWrapper
from run_registry import RunRegistry, artifact_path, install_atexit_handler, run_dir

from simsopt._core import load
from simsopt.geo import (
    BoozerResidual,
    BoozerSurface,
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    Iotas,
    LpCurveCurvature,
    NonQuasiSymmetricRatio,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    Volume,
    boozer_surface_residual,
)
from simsopt.objectives import QuadraticPenalty


def proc0_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Load configuration
# ──────────────────────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(_cfg_path) as _f:
    cfg = yaml.safe_load(_f)

# Output directory (resolve early — other paths depend on it)
OUT_DIR = resolve_output_dir()

# Parent run IDs — both hard-required. Stage 2 id is the direct parent and
# supplies the warm-start BoozerSurface; stage 1 id supplies the optimized
# wout needed to fit the initial plasma surface.
STAGE1_ID = cfg['warm_start'].get('stage1_id')
STAGE2_ID = cfg['warm_start'].get('stage2_id')
if not STAGE1_ID:
    raise ValueError(
        "03_singlestage: cfg['warm_start']['stage1_id'] is required "
        "(set to the s01_xxxxxx id of the stage 1 run that produced the wout)."
    )
if not STAGE2_ID:
    raise ValueError(
        "03_singlestage: cfg['warm_start']['stage2_id'] is required "
        "(set to the s02_xxxxxx id of the stage 2 run to warm-start from)."
    )

_slurm_meta = {
    "slurm_qos":           os.environ.get("SLURM_JOB_QOS"),
    "slurm_partition":     os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_ntasks":        int(os.environ["SLURM_NTASKS"]) if os.environ.get("SLURM_NTASKS") else None,
    "slurm_cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]) if os.environ.get("SLURM_CPUS_PER_TASK") else None,
    "slurm_time_limit_s":  None,
}
_slurm_job_id = os.environ.get("SLURM_JOB_ID")

registry = RunRegistry()
RUN_ID, _is_new = registry.register_singlestage(cfg, stage2_id=STAGE2_ID,
                                                slurm_meta=_slurm_meta)

RUN_DIR = run_dir("singlestage", RUN_ID, OUT_DIR)
os.makedirs(RUN_DIR, exist_ok=True)

# Device geometry
NFP      = cfg['device']['nfp']
STELLSYM = cfg['device']['stellsym']

# TF coil layout
TF_NUM = cfg['tf_coils']['num']

# Banana coil constraints
BANANA_CURV_P      = cfg['banana_coils']['curv_p']
BANANA_CURRENT_MAX = cfg['banana_coils']['current_max']

# Physics targets
TARGET_VOLUME = cfg['targets']['volume']
TARGET_IOTA   = cfg['targets']['iota']

# Warm-start
STAGE2_BSURF_FILE = artifact_path("stage2", STAGE2_ID, OUT_DIR, "bsurf_opt")

# Boozer surface
CONSTRAINT_WEIGHT = cfg['boozer']['constraint_weight']
MPOL   = cfg['boozer']['mpol']
NTOR   = cfg['boozer']['ntor']

# Plasma surface from VMEC (for initialization)
NPHI   = cfg['plasma_surface']['nphi']
NTHETA = cfg['plasma_surface']['ntheta']
VMEC_S = cfg['plasma_surface']['vmec_s']
WOUT_FILE = artifact_path("stage1", STAGE1_ID, OUT_DIR, "wout_opt")

# Objective thresholds (hardware constraints — not relaxable)
LENGTH_MAX_HW    = cfg['thresholds']['length_max']
LENGTH_THRESHOLD = cfg['thresholds']['length_target']
CC_THRESHOLD     = cfg['thresholds']['coil_coil_min']
CS_THRESHOLD     = cfg['thresholds']['coil_surface_min']
CURV_THRESHOLD   = cfg['thresholds']['curvature_max']

# Objective weights
NONQS_WEIGHT = cfg['singlestage_weights']['nonqs']
BRES_WEIGHT  = cfg['singlestage_weights']['boozer_residual']
IOTA_WEIGHT  = cfg['singlestage_weights']['iota']
LEN_WEIGHT   = cfg['singlestage_weights']['length']
CC_WEIGHT    = cfg['singlestage_weights']['coil_coil']
CS_WEIGHT    = cfg['singlestage_weights']['coil_surface']
CURV_WEIGHT  = cfg['singlestage_weights']['curvature']
CURR_WEIGHT  = cfg['singlestage_weights']['current']

# Optimizer (L-BFGS-B)
MAXITER = cfg['singlestage_optimizer']['maxiter']
MAXCOR  = cfg['singlestage_optimizer']['maxcor']
MAXFUN  = cfg['singlestage_optimizer']['maxfun']
TOL     = cfg['singlestage_optimizer']['tol']
FTOL    = cfg['singlestage_optimizer']['ftol']
GTOL    = cfg['singlestage_optimizer']['gtol']

# ──────────────────────────────────────────────────────────────────────────────
# Output atexit handler
# ──────────────────────────────────────────────────────────────────────────────
DIAGNOSTICS_FILE = artifact_path("singlestage", RUN_ID, OUT_DIR, "diagnostics")


def _emit_out_dir_on_exit():
    """Print per-run directory so the shell script can move the log file."""
    proc0_print(f"OUT_DIR={RUN_DIR}")


atexit.register(_emit_out_dir_on_exit)
install_atexit_handler(registry, "singlestage", RUN_ID)


# ──────────────────────────────────────────────────────────────────────────────
# Print input parameters
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
INPUT PARAMETERS ─────────────────────────────
    Config:          {_cfg_path}
    Date:            {datetime.now()}
    Run ID:          {RUN_ID} ({'new' if _is_new else 'rerun'})
    Run dir:         {RUN_DIR}
    Warm-start ids:  stage1={STAGE1_ID}, stage2={STAGE2_ID}

    Physics targets:
        volume      = {TARGET_VOLUME}
        iota        = {TARGET_IOTA}

    Boozer surface:
        method           = BoozerLS
        constraint_weight = {CONSTRAINT_WEIGHT:.3e}
        mpol             = {MPOL}
        ntor             = {NTOR}

    Banana coil curvature p-norm = {BANANA_CURV_P}

    Warm-start:
        bsurf       = {STAGE2_BSURF_FILE}
        wout        = {WOUT_FILE}

    Thresholds:
        length_abs  = {LENGTH_MAX_HW} m
        length_tgt  = {LENGTH_THRESHOLD} m
        cc_min      = {CC_THRESHOLD} m
        cs_min      = {CS_THRESHOLD} m
        curv_max    = {CURV_THRESHOLD} m^-1
        current_max = {BANANA_CURRENT_MAX/1e3:.0f} kA

    Objective weights:
        nonqs       = {NONQS_WEIGHT:.3e}
        boozer_res  = {BRES_WEIGHT:.3e}
        iota        = {IOTA_WEIGHT:.3e}
        length      = {LEN_WEIGHT:.3e}
        coil_coil   = {CC_WEIGHT:.3e}
        coil_surf   = {CS_WEIGHT:.3e}
        curvature   = {CURV_WEIGHT:.3e}
        current     = {CURR_WEIGHT:.3e}

    Optimizer (L-BFGS-B):
        maxiter = {MAXITER}
        maxcor  = {MAXCOR}
        maxfun  = {MAXFUN}
        tol     = {TOL:.3e}
        ftol    = {FTOL:.3e}
        gtol    = {GTOL:.3e}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Load warm-start data and build surface
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(f'Loading BoozerSurface from {STAGE2_BSURF_FILE}')

surface = SurfaceRZFourier.from_wout(
    WOUT_FILE, range="field period", nphi=NPHI, ntheta=NTHETA, s=VMEC_S,
)
# The stage 1 seed (produced by utils/vmec_resize.py) has LCFS == target
# plasma boundary, and stage 1 preserves this. No rescaling is needed.
gamma = surface.gamma().copy()
quadpoints_theta = surface.quadpoints_theta.copy()
quadpoints_phi = surface.quadpoints_phi.copy()

boozersurface_loaded = load(STAGE2_BSURF_FILE)
biotsavart = boozersurface_loaded.biotsavart
coils = biotsavart.coils
curves = [coil.curve for coil in coils]

tf_coils = coils[:TF_NUM]
tf_currents = [coil.current for coil in tf_coils]

banana_coils = coils[TF_NUM:]
banana_curves = [coil.curve for coil in banana_coils]
banana_curve = banana_curves[0]
# Stage 2 can pin the banana current DOF before saving. Singlestage optimizes
# that current, so restore it to the free-DOF set before building the objective.
banana_coils[0].current.unfix_all()

current_tot = sum(c.get_value() for c in tf_currents)
G0 = 4e-7 * np.pi * current_tot

surface = SurfaceXYZTensorFourier(
    mpol=MPOL, ntor=NTOR, nfp=NFP, stellsym=STELLSYM,
    quadpoints_theta=quadpoints_theta,
    quadpoints_phi=quadpoints_phi,
)
surface.least_squares_fit(gamma)


# ──────────────────────────────────────────────────────────────────────────────
# Build Boozer surface and solve initial equilibrium
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(f'Solving initial Boozer surface (BoozerLS, MPOL={MPOL}, NTOR={NTOR})...')

Jvol = Volume(surface)
boozersurface = BoozerSurface(
    biotsavart, surface, Jvol, TARGET_VOLUME, CONSTRAINT_WEIGHT,
    options=dict(verbose=True),
)
res = boozersurface.run_code(TARGET_IOTA, G0)

solve_success = res["success"]
not_intersecting = not boozersurface.surface.is_self_intersecting()
success = solve_success and not_intersecting
proc0_print(f'  Solve success: {solve_success}, Not self-intersecting: {not_intersecting}')
if not success:
    raise RuntimeError("Initial Boozer surface solve failed")

biotsavart.set_points(surface.gamma().reshape((-1, 3)))


# ──────────────────────────────────────────────────────────────────────────────
# Define objective function
# ──────────────────────────────────────────────────────────────────────────────
_Jnonqs = [NonQuasiSymmetricRatio(boozersurface, biotsavart)]
Jnonqs  = sum(_Jnonqs)
_Jbres  = [BoozerResidual(boozersurface, biotsavart)]
Jbres   = sum(_Jbres)
_Jiota  = Iotas(boozersurface)
Jiota   = QuadraticPenalty(_Jiota, TARGET_IOTA)
_Jl     = CurveLength(banana_curve)
Jl      = QuadraticPenalty(_Jl, LENGTH_THRESHOLD, "max")
Jcs     = CurveSurfaceDistance(curves, surface, CS_THRESHOLD)
Jcc     = CurveCurveDistance(curves, CC_THRESHOLD)
Jcurv   = LpCurveCurvature(banana_curve, BANANA_CURV_P, CURV_THRESHOLD)
_Jcurr  = CurrentPenaltyWrapper(banana_coils[0].current)
Jcurr   = QuadraticPenalty(_Jcurr, BANANA_CURRENT_MAX, "max")

# Auto-detect current enforcement mode:
# If initial current already within limit → hard L-BFGS-B bound (no penalty needed)
# If initial current exceeds limit → soft penalty to drive it down
CURRENT_VIOLATES = abs(banana_coils[0].current.get_value()) > BANANA_CURRENT_MAX
if CURRENT_VIOLATES:
    proc0_print(f'  Banana current {abs(banana_coils[0].current.get_value())/1e3:.3f} kA '
                f'exceeds limit {BANANA_CURRENT_MAX/1e3:.0f} kA → using soft penalty (weight={CURR_WEIGHT:.3e})')
    JF = (NONQS_WEIGHT * Jnonqs) + (BRES_WEIGHT * Jbres) + (IOTA_WEIGHT * Jiota) \
       + (LEN_WEIGHT * Jl) + (CS_WEIGHT * Jcs) + (CC_WEIGHT * Jcc) + (CURV_WEIGHT * Jcurv) \
       + (CURR_WEIGHT * Jcurr)
else:
    proc0_print(f'  Banana current {abs(banana_coils[0].current.get_value())/1e3:.3f} kA '
                f'within limit {BANANA_CURRENT_MAX/1e3:.0f} kA → using hard L-BFGS-B bound')
    JF = (NONQS_WEIGHT * Jnonqs) + (BRES_WEIGHT * Jbres) + (IOTA_WEIGHT * Jiota) \
       + (LEN_WEIGHT * Jl) + (CS_WEIGHT * Jcs) + (CC_WEIGHT * Jcc) + (CURV_WEIGHT * Jcurv)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: compute Boozer residual norm
# ──────────────────────────────────────────────────────────────────────────────
def _boozer_residual_norm():
    """Compute the normalized Boozer residual from the current surface state."""
    bsr = boozersurface.res
    num_pts = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
    r, = boozer_surface_residual(
        surface, bsr['iota'], bsr['G'], biotsavart, derivatives=0,
        weight_inv_modB=boozersurface.options.get("weight_inv_modB", True),
    )
    return 0.5 * np.sum((r / np.sqrt(num_pts))**2)


# ──────────────────────────────────────────────────────────────────────────────
# Print initial state
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
INITIAL STATE (MPOL={MPOL}) ───────────────────
    Parameter values:
        Non-QS ratio:                    {Jnonqs.J():.6e}
        Boozer residual:                 {_boozer_residual_norm():.6e}
        Iota:                            {_Jiota.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m (limit: {LENGTH_THRESHOLD:.6e} m)
        Banana coil current:             {_Jcurr.J()/1e3:.3f} kA (limit: {BANANA_CURRENT_MAX/1e3:.0f} kA)
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        CS separation (shortest_dist):   {Jcs.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    Penalty values:
        Objective J:                     {JF.J():.6e}
        ||grad J||:                      {np.linalg.norm(JF.dJ()):.6e}
        Non-QS ratio penalty:            ({NONQS_WEIGHT:.3e}){Jnonqs.J():.6e} = {NONQS_WEIGHT * Jnonqs.J():.6e}
        Boozer residual penalty:         ({BRES_WEIGHT:.3e}){Jbres.J():.6e} = {BRES_WEIGHT * Jbres.J():.6e}
        Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
        Length penalty (QuadPen.J):      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
        CC distance penalty:             ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
        CS distance penalty:             ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
        Curvature penalty (LpCurvCurv):  ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
        Current penalty (QuadPen.J):     ({CURR_WEIGHT:.3e}){Jcurr.J():.6e} = {CURR_WEIGHT * Jcurr.J():.6e}

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
    sdofs_prev=surface.x.copy(),
    iota_prev=boozersurface.res["iota"],
    G_prev=boozersurface.res["G"],
)


def _write_diagnostics_row(J, dJ, t0):
    """Append a single diagnostics row to the CSV file (inner-loop tracking)."""
    t_elapsed = time.time() - t0
    dJ_norm = np.linalg.norm(dJ)

    track['eval'] += 1
    row = (
        f"{track['iter']},{track['eval']},{t_elapsed:.2f},"
        f"{J:.6e},{dJ_norm:.6e},"
        f"{Jnonqs.J():.6e},"
        f"{_Jiota.J():.6e},"
        f"{_Jl.J():.6e},"
        f"{Jcc.shortest_distance():.6e},"
        f"{Jcs.shortest_distance():.6e},"
        f"{banana_curve.kappa().max():.6e},"
        f"{_Jcurr.J():.6e},"
        "1"
    )
    proc0_print(row)
    with open(DIAGNOSTICS_FILE, 'a') as f:
        f.write(row + "\n")


def fun(dofs):
    """Objective function for L-BFGS-B with a required Boozer solve."""
    # Restore surface state for warm-start
    surface.x                 = track["sdofs_prev"]
    boozersurface.res["iota"] = track["iota_prev"]
    boozersurface.res["G"]    = track["G_prev"]

    JF.x = dofs
    res = boozersurface.run_code(track["iota_prev"], track["G_prev"])

    solve_success = res["success"]
    not_intersecting = not surface.is_self_intersecting()
    success = solve_success and not_intersecting

    if success:
        J  = JF.J()
        dJ = JF.dJ()
    else:
        reasons = []
        if not solve_success:
            reasons.append("Boozer solve failed")
        if not not_intersecting:
            reasons.append("surface is self-intersecting")
        raise RuntimeError("; ".join(reasons))

    _write_diagnostics_row(J, dJ, t0)
    return J, dJ


def callback(x):
    """Callback called after each L-BFGS-B iteration (outer-loop tracking)."""
    J  = JF.J()
    dJ = JF.dJ()
    res = boozersurface.res
    track["sdofs_prev"] = surface.x.copy()
    track["iota_prev"]  = res["iota"]
    track["G_prev"]     = res["G"]
    track['f_prev'] = track['f_curr']
    track['f_curr'] = J
    track['iter'] += 1
    track['eval'] = 0
    runtime = time.time() - t0

    proc0_print(
        f"""
[{datetime.now()}; {timedelta(seconds=runtime)} elapsed] ITERATION {track['iter']:03d}/{MAXITER}
    Parameter values:
        Non-QS ratio:                    {Jnonqs.J():.6e}
        Boozer residual:                 {_boozer_residual_norm():.6e}
        Iota:                            {_Jiota.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        Banana coil current:             {_Jcurr.J()/1e3:.3f} kA (limit: {BANANA_CURRENT_MAX/1e3:.0f} kA)
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        CS separation (shortest_dist):   {Jcs.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    Penalty values:
        Objective J:                     {JF.J():.6e}
        ||grad J||:                      {np.linalg.norm(dJ):.6e}
        Non-QS ratio penalty:            ({NONQS_WEIGHT:.3e}){Jnonqs.J():.6e} = {NONQS_WEIGHT * Jnonqs.J():.6e}
        Boozer residual penalty:         ({BRES_WEIGHT:.3e}){Jbres.J():.6e} = {BRES_WEIGHT * Jbres.J():.6e}
        Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
        Length penalty (QuadPen.J):      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
        CC distance penalty:             ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
        CS distance penalty:             ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
        Curvature penalty (LpCurvCurv):  ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
        Current penalty (QuadPen.J):     ({CURR_WEIGHT:.3e}){Jcurr.J():.6e} = {CURR_WEIGHT * Jcurr.J():.6e}
"""
    )


# ──────────────────────────────────────────────────────────────────────────────
# Initialize diagnostics file
# ──────────────────────────────────────────────────────────────────────────────
t0 = time.time()

with open(DIAGNOSTICS_FILE, 'w') as f:
    f.write(f'# Singlestage Diagnostics\n')
    f.write(f'# Date: {datetime.now()}\n')
    f.write(f'# BoozerSurface: {STAGE2_BSURF_FILE}\n')
    f.write(f'# MPOL={MPOL}, NTOR={NTOR}, CONSTRAINT_WEIGHT={CONSTRAINT_WEIGHT:.3e}\n')
    f.write(f'# TARGET_VOLUME={TARGET_VOLUME}, TARGET_IOTA={TARGET_IOTA}\n')
    f.write(f'# LENGTH_THRESHOLD={LENGTH_THRESHOLD}, CC_THRESHOLD={CC_THRESHOLD}, CS_THRESHOLD={CS_THRESHOLD}, CURV_THRESHOLD={CURV_THRESHOLD}\n')
    f.write(f'# MAXITER={MAXITER}, FTOL={FTOL:.3e}, GTOL={GTOL:.3e}\n')
    f.write(
        'iter,eval,runtime,'
        'objective,grad_norm,'
        'nonqs,'
        'iota,'
        'coil_length,'
        'ccdist,csdist,'
        'max_kappa,'
        'banana_current,'
        'solve_ok\n'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Run optimization
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(f'[{datetime.now()}] Starting singlestage optimization (MPOL={MPOL})...')
registry.mark_running("singlestage", RUN_ID, slurm_job_id=_slurm_job_id)
x0 = JF.x

# Hard L-BFGS-B bound when current is within limit (no penalty in objective)
bounds = None
if not CURRENT_VIOLATES and banana_coils[0].current.dof_names:
    dof_names = JF.dof_names
    banana_current_dof = banana_coils[0].current.dof_names[0]
    if banana_current_dof in dof_names:
        current_dof_idx = dof_names.index(banana_current_dof)
        bounds = [(None, None)] * len(x0)
        banana_dof_val = x0[current_dof_idx]
        banana_phys_val = banana_coils[0].current.get_value()
        bound_abs = abs(BANANA_CURRENT_MAX * banana_dof_val / banana_phys_val)
        bounds[current_dof_idx] = (-bound_abs, bound_abs)
        proc0_print(f'    Bound on DOF[{current_dof_idx}] ({dof_names[current_dof_idx]}): '
                    f'abs <= {bound_abs:.4f}'
                    f' (physical: |I| <= {BANANA_CURRENT_MAX/1e3:.0f} kA)')
    else:
        raise RuntimeError('banana current DOF not found in singlestage objective')

res = minimize(
    fun, x0, jac=True, method='L-BFGS-B', tol=TOL,
    bounds=bounds,
    callback=callback,
    options=dict(maxiter=MAXITER, maxcor=MAXCOR, maxfun=MAXFUN,
                 ftol=FTOL, gtol=GTOL),
)


# ──────────────────────────────────────────────────────────────────────────────
# Termination summary
# ──────────────────────────────────────────────────────────────────────────────
end_date = datetime.now()
opt_runtime = time.time() - t0

grad_inf = np.max(np.abs(res.jac)) if hasattr(res, 'jac') and res.jac is not None else float('nan')
hit_maxiter = res.nit >= MAXITER
hit_maxfun  = res.nfev >= MAXFUN
hit_gtol    = grad_inf <= GTOL

EPSMCH  = np.finfo(float).eps
FACTR   = FTOL / EPSMCH
f_curr  = track['f_curr']
f_prev  = track['f_prev']
if f_prev is None:
    rel_red = float('nan')
    rel_red_str = f"{rel_red}"
    f_cond_str = f"F={f_curr}, F_prev={f_prev}"
else:
    rel_red = (f_prev - f_curr) / max(1.0, abs(f_prev), abs(f_curr))
    rel_red_str = f"{rel_red:.3e}"
    f_cond_str = f"F={f_curr:.6e}, F_prev={f_prev:.6e}"
hit_ftol = bool(re.search(
    r'REL[_\s]REDUCTION[_\s]OF[_\s]F|RELATIVE\s+REDUCTION\s+OF\s+F',
    res.message, re.IGNORECASE
))

success = res.success

proc0_print(
    f"""
[{end_date}] ...optimization complete
Total runtime: {timedelta(seconds=opt_runtime)}

{'SUCCESS' if success else 'FAILURE'} ─────────────────────────────────────────
    Banana coil current : {banana_coils[0].current.get_value()/1e3:.5f} kA
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
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Print final state
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
FINAL STATE (MPOL={MPOL}) ─────────────────────
    Parameter values:
        Non-QS ratio:                    {Jnonqs.J():.6e}
        Boozer residual:                 {_boozer_residual_norm():.6e}
        Iota:                            {_Jiota.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        Banana coil current:             {_Jcurr.J()/1e3:.3f} kA (limit: {BANANA_CURRENT_MAX/1e3:.0f} kA)
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        CS separation (shortest_dist):   {Jcs.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    Penalty values:
        Objective J:                     {JF.J():.6e}
        ||grad J||:                      {np.linalg.norm(JF.dJ()):.6e}
        Non-QS ratio penalty:            ({NONQS_WEIGHT:.3e}){Jnonqs.J():.6e} = {NONQS_WEIGHT * Jnonqs.J():.6e}
        Boozer residual penalty:         ({BRES_WEIGHT:.3e}){Jbres.J():.6e} = {BRES_WEIGHT * Jbres.J():.6e}
        Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
        Length penalty (QuadPen.J):      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
        CC distance penalty:             ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
        CS distance penalty:             ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
        Curvature penalty (LpCurvCurv):  ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
        Current penalty (QuadPen.J):     ({CURR_WEIGHT:.3e}){Jcurr.J():.6e} = {CURR_WEIGHT * Jcurr.J():.6e}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Save final outputs
# ──────────────────────────────────────────────────────────────────────────────
# Save BoozerSurface (canonical opt output only on optimizer success).
_bsurf_kind = "bsurf_opt" if success else "bsurf_failed"
_bsurf_out_path = artifact_path("singlestage", RUN_ID, OUT_DIR, _bsurf_kind)
boozersurface.save(_bsurf_out_path)
if success:
    _state_out_path = artifact_path("singlestage", RUN_ID, OUT_DIR, "state_opt")
    np.savez(_state_out_path,
             iota=boozersurface.res["iota"], G=boozersurface.res["G"])

proc0_print(f'Diagnostics saved to {DIAGNOSTICS_FILE}')
proc0_print(f'Outputs saved to {RUN_DIR}')


# ──────────────────────────────────────────────────────────────────────────────
# Registry finalization
# ──────────────────────────────────────────────────────────────────────────────
_metrics = {
    "final_iota":            float(boozersurface.res["iota"]),
    "final_qs_metric":       float(Jnonqs.J()),
    "final_boozer_residual": float(_boozer_residual_norm()),
    "final_sqflx":           None,
    "final_max_curvature":   float(banana_curve.kappa().max()),
    "final_min_cc_dist":     float(Jcc.shortest_distance()),
    "final_min_cs_dist":     float(Jcs.shortest_distance()),
    "final_max_length":      float(_Jl.J()),
    "final_banana_current":  float(banana_coils[0].current.get_value()),
    "runtime_s":             float(opt_runtime),
}
if success:
    registry.mark_success("singlestage", RUN_ID, metrics=_metrics,
                          slurm_wall_s=float(opt_runtime))
else:
    _err_code = "timeout" if hit_maxiter or hit_maxfun else "solver_diverged"
    _err_msg = "singlestage optimizer did not converge"
    registry.mark_failed("singlestage", RUN_ID, error_code=_err_code,
                         error_message=_err_msg, slurm_wall_s=float(opt_runtime),
                         metrics=_metrics)
    raise RuntimeError(_err_msg)
