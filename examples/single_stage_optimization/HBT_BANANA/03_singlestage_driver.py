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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'utils'))
sys.path.insert(0, os.path.join(_THIS_DIR, 'new_objectives'))
from output_dir import resolve_output_dir
from current_penalty import CurrentPenaltyWrapper
from hardware_metrics import (
    boozer_json_vacuum_lineage,
    curve_poloidal_half_extent,
    surface_shape_metrics,
    surface_vessel_clearance,
)
from run_registry import RunRegistry, artifact_path, install_atexit_handler, run_dir
from solver_log import make_solver_iter_tap, stdout_to_log
from poloidal_extent import PoloidalExtent
from ellipse_width import ProjectedEllipseWidth
from self_intersect import CurveSelfIntersect
from global_curvature_radius import GlobalRadiusCurvature
from vessel_clearance import CircularVesselClearance

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
install_atexit_handler(registry, "singlestage", RUN_ID)

RUN_DIR = run_dir("singlestage", RUN_ID, OUT_DIR)
os.makedirs(RUN_DIR, exist_ok=True)

# Device geometry
NFP      = cfg['device']['nfp']
STELLSYM = cfg['device']['stellsym']
VESSEL_MAJOR_R = float(cfg['device']['major_radius'])
VESSEL_MINOR_R = float(cfg['device']['vessel_minor_radius'])
TARGET_LCFS_MAJOR_R = float(cfg['device']['plasma_radius'])
TARGET_LCFS_MINOR_R = float(cfg['device']['plasma_minor_radius'])

# TF coil layout
TF_NUM = cfg['tf_coils']['num']

# Banana coil constraints
BANANA_CURV_P      = cfg['banana_coils']['curv_p']
BANANA_CURRENT_MAX = cfg['banana_coils']['current_max']
WINDING_R0         = float(cfg['winding_surface']['R0'])
WINDING_A          = float(cfg['winding_surface']['a'])
# TF current penalty bound: |I_tf| penalized above its operating magnitude
# (80 kA), matching the reference hardware limit. TF current is fixed
# background, so this term is normally inactive; it guards against drift.
TF_CURRENT_MAX     = abs(float(cfg['tf_coils']['current']))

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
PV_THRESHOLD     = cfg['thresholds']['plasma_vessel_min']
CURV_THRESHOLD   = cfg['thresholds']['curvature_max']
POLOIDAL_THRESHOLD_RAD = np.deg2rad(
    float(cfg['thresholds']['poloidal_half_width_max_deg'])
)
WIDTH_MAX_HW    = float(cfg['thresholds']['width_max'])
WIDTH_MIN_HW    = float(cfg['thresholds']['width_min'])
SELFINT_MIN_HW  = float(cfg['thresholds']['self_intersect_min'])
GCR_MIN_HW      = float(cfg['thresholds']['global_curvature_radius_min'])
GCR_EXP_WEIGHT  = float(cfg['thresholds']['global_curvature_exp_weight'])

# Objective weights
NONQS_WEIGHT = cfg['singlestage_weights']['nonqs']
BRES_WEIGHT  = cfg['singlestage_weights']['boozer_residual']
IOTA_WEIGHT  = cfg['singlestage_weights']['iota']
LEN_WEIGHT   = cfg['singlestage_weights']['length']
CC_WEIGHT    = cfg['singlestage_weights']['coil_coil']
CS_WEIGHT    = cfg['singlestage_weights']['coil_surface']
CURV_WEIGHT  = cfg['singlestage_weights']['curvature']
POL_WEIGHT   = cfg['singlestage_weights']['poloidal_extent']
PV_WEIGHT    = cfg['singlestage_weights']['plasma_vessel']
CURR_WEIGHT  = cfg['singlestage_weights']['current']
WIDTH_WEIGHT   = cfg['singlestage_weights']['width']
SELFINT_WEIGHT = cfg['singlestage_weights']['selfint']
GCR_WEIGHT     = cfg['singlestage_weights']['global_curvature']

# Optimizer (L-BFGS-B)
MAXITER = cfg['singlestage_optimizer']['maxiter']
MAXCOR  = cfg['singlestage_optimizer']['maxcor']
MAXFUN  = cfg['singlestage_optimizer']['maxfun']
TOL     = cfg['singlestage_optimizer']['tol']


def _tolerance_for_mpol(table, mpol):
    """Look up a resolution-keyed L-BFGS-B tolerance, clamped at both ends.

    `table` maps poloidal resolution to tolerance and is contiguous over its
    key range. Resolutions below the lowest key take the lowest key's value
    and resolutions above the highest take the highest key's, so an mpol
    outside the tabulated range degrades to the nearest calibrated row
    instead of failing.
    """
    resolutions = sorted(table)
    return float(table[min(max(mpol, resolutions[0]), resolutions[-1])])


FTOL = _tolerance_for_mpol(cfg['singlestage_optimizer']['ftol_per_mpol'], MPOL)
GTOL = _tolerance_for_mpol(cfg['singlestage_optimizer']['gtol_per_mpol'], MPOL)

# ──────────────────────────────────────────────────────────────────────────────
# Output atexit handler
# ──────────────────────────────────────────────────────────────────────────────
DIAGNOSTICS_FILE = artifact_path("singlestage", RUN_ID, OUT_DIR, "diagnostics")


def _emit_out_dir_on_exit():
    """Print per-run directory so the shell script can move the log file."""
    proc0_print(f"OUT_DIR={RUN_DIR}")


atexit.register(_emit_out_dir_on_exit)


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
    Winding surface:
        R0         = {WINDING_R0:.3f} m
        a          = {WINDING_A:.3f} m

    Warm-start:
        bsurf       = {STAGE2_BSURF_FILE}
        wout        = {WOUT_FILE}

    Thresholds:
        length_abs  = {LENGTH_MAX_HW} m
        length_tgt  = {LENGTH_THRESHOLD} m
        length_min  = {LENGTH_THRESHOLD/2} m  (half the max-length threshold)
        cc_min      = {CC_THRESHOLD} m
        cs_min      = {CS_THRESHOLD} m
        pv_min      = {PV_THRESHOLD} m
        curv_max    = {CURV_THRESHOLD} m^-1
        poloidal    = {POLOIDAL_THRESHOLD_RAD:.6f} rad
        width       = [{WIDTH_MIN_HW}, {WIDTH_MAX_HW}] m
        selfint     = {SELFINT_MIN_HW} m
        gcr_min     = {GCR_MIN_HW} m  (barrier softness {GCR_EXP_WEIGHT} m)
        current_max = {BANANA_CURRENT_MAX/1e3:.0f} kA banana / {TF_CURRENT_MAX/1e3:.0f} kA TF

    Objective weights:
        nonqs       = {NONQS_WEIGHT:.3e}
        boozer_res  = {BRES_WEIGHT:.3e}
        iota        = {IOTA_WEIGHT:.3e}
        length      = {LEN_WEIGHT:.3e}
        coil_coil   = {CC_WEIGHT:.3e}
        coil_surf   = {CS_WEIGHT:.3e}
        curvature   = {CURV_WEIGHT:.3e}
        poloidal    = {POL_WEIGHT:.3e}
        plasma_vess = {PV_WEIGHT:.3e}
        width       = {WIDTH_WEIGHT:.3e}
        selfint     = {SELFINT_WEIGHT:.3e}{'' if SELFINT_WEIGHT else '  (diagnostic only)'}
        global_curv = {GCR_WEIGHT:.3e}{'' if GCR_WEIGHT else '  (diagnostic only)'}
        current     = {CURR_WEIGHT:.3e}

    Optimizer (L-BFGS-B):
        maxiter = {MAXITER}
        maxcor  = {MAXCOR}
        maxfun  = {MAXFUN}
        tol     = {TOL:.3e}
        ftol    = {FTOL:.3e}  (from ftol_per_mpol[mpol={MPOL}])
        gtol    = {GTOL:.3e}  (from gtol_per_mpol[mpol={MPOL}])
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

# The Boozer solvers print their own convergence trace to stdout. Folding it
# through the driver's log keeps it interleaved with the driver's own output
# in file order, and the tap scrapes the inner solver iteration counts back
# out of those lines so each outer evaluation can report how much work the
# inner solve cost. BoozerLS runs BFGS then Newton; BoozerExact runs Newton
# only, and `make_solver_iter_tap` leaves absent keys untouched.
solver_iters = (dict(bfgs_nit=0, newton_nit=0)
                if boozersurface.constraint_weight is not None
                else dict(newton_nit=0))
_solver_iter_tap = make_solver_iter_tap(solver_iters)

with stdout_to_log(proc0_print, tap=_solver_iter_tap):
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
# Lower length bound at half the max-length threshold (reference: max_length/2).
Jl_min  = QuadraticPenalty(_Jl, LENGTH_THRESHOLD / 2, "min")
# Coil-surface distance spans all coils; coil-coil distance is banana-only
# (TF coils are fixed background), matching the reference.
Jcs     = CurveSurfaceDistance(curves, surface, CS_THRESHOLD)
Jcc     = CurveCurveDistance(banana_curves, CC_THRESHOLD)
Jcurv   = LpCurveCurvature(banana_curve, BANANA_CURV_P, CURV_THRESHOLD)
Jpol    = PoloidalExtent(
    banana_curve, WINDING_R0, POLOIDAL_THRESHOLD_RAD, p=BANANA_CURV_P
)
Jpv     = CircularVesselClearance(
    surface, VESSEL_MAJOR_R, VESSEL_MINOR_R, PV_THRESHOLD, p=BANANA_CURV_P
)
# Projected-ellipse width (port-fit max + anti-collapse min) and
# self-intersection hinge, matching the reference objective set.
_width  = ProjectedEllipseWidth(banana_curve, WINDING_R0, WINDING_A)
Jwmax   = QuadraticPenalty(_width, WIDTH_MAX_HW, "max")
Jwmin   = QuadraticPenalty(_width, WIDTH_MIN_HW, "min")
Jself   = CurveSelfIntersect(
    banana_curve, SELFINT_MIN_HW, int(1.5 * banana_curve.order)
)
# Gonzalez-Maddocks global radius of curvature: the same self-contact
# constraint as Jself, but smooth on its noncoincident/nonparallel branch and
# summed over every near-contact pair rather than hinged on the single worst
# one, so L-BFGS-B gets a gradient before the constraint is violated rather
# than after. Its exponential penalty is finite; exact sampled coincidences
# receive radius zero. Only one of the two belongs in JF; both are reported.
Jgcr    = GlobalRadiusCurvature(banana_curve, GCR_MIN_HW, GCR_EXP_WEIGHT)
# Current penalties: |I_banana| above 16 kA (added below only when the warm
# start already violates the limit) and |I_tf| above 80 kA (always present;
# inactive while TF stays at its fixed operating point).
_Jcurr  = CurrentPenaltyWrapper(banana_coils[0].current)
Jcurr   = QuadraticPenalty(_Jcurr, BANANA_CURRENT_MAX, "max")
_Jtfcurr = CurrentPenaltyWrapper(tf_currents[0])
Jtfcurr  = QuadraticPenalty(_Jtfcurr, TF_CURRENT_MAX, "max")

# Always-on objective: physics (nonqs, boozer residual, iota) + the full
# banana_drivers reference geometric set (length max/min, coil-coil,
# curvature, poloidal extent, width max/min) + the local surface-coupling
# clearances (coil-surface, plasma-vessel) + the TF current guard. The
# self-contact and banana-current penalties are added conditionally below.
JF = (NONQS_WEIGHT * Jnonqs) + (BRES_WEIGHT * Jbres) + (IOTA_WEIGHT * Jiota) \
   + (LEN_WEIGHT * Jl) + (LEN_WEIGHT * Jl_min) + (CS_WEIGHT * Jcs) \
   + (CC_WEIGHT * Jcc) + (CURV_WEIGHT * Jcurv) + (POL_WEIGHT * Jpol) \
   + (PV_WEIGHT * Jpv) + (WIDTH_WEIGHT * (Jwmax + Jwmin)) \
   + (CURR_WEIGHT * Jtfcurr)

# Self-contact: Jgcr and Jself enforce the same constraint by different
# means, so whichever carries a nonzero weight owns it and the other stays a
# reported diagnostic. Zero-weight terms are left out of JF entirely rather
# than added as 0*J — CurveSelfIntersect is O(N^2) per evaluation and there
# is no reason to pay for it on the critical path.
if GCR_WEIGHT:
    JF = JF + (GCR_WEIGHT * Jgcr)
if SELFINT_WEIGHT:
    JF = JF + (SELFINT_WEIGHT * Jself)

# Auto-detect banana-current enforcement:
#   within limit  → hard L-BFGS-B bound (no penalty term)
#   exceeds limit → soft QuadraticPenalty to drive it down
CURRENT_VIOLATES = abs(banana_coils[0].current.get_value()) > BANANA_CURRENT_MAX
if CURRENT_VIOLATES:
    proc0_print(f'  Banana current {abs(banana_coils[0].current.get_value())/1e3:.3f} kA '
                f'exceeds limit {BANANA_CURRENT_MAX/1e3:.0f} kA → using soft penalty (weight={CURR_WEIGHT:.3e})')
    JF = JF + (CURR_WEIGHT * Jcurr)
else:
    proc0_print(f'  Banana current {abs(banana_coils[0].current.get_value())/1e3:.3f} kA '
                f'within limit {BANANA_CURRENT_MAX/1e3:.0f} kA → using hard L-BFGS-B bound')


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
        Poloidal half extent:            {curve_poloidal_half_extent(banana_curve, WINDING_R0):.6e} rad
        Plasma-vessel clearance:         {surface_vessel_clearance(surface, VESSEL_MAJOR_R, VESSEL_MINOR_R):.6e} m
        Min global curv radius:          {Jgcr.shortest_radius():.6e} m (limit: {GCR_MIN_HW:.3e} m)
        Shortest self-distance:          {Jself.shortest_self_distance():.6e} m (limit: {SELFINT_MIN_HW:.3e} m)

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
        Poloidal extent penalty:         ({POL_WEIGHT:.3e}){Jpol.J():.6e} = {POL_WEIGHT * Jpol.J():.6e}
        Plasma-vessel penalty:           ({PV_WEIGHT:.3e}){Jpv.J():.6e} = {PV_WEIGHT * Jpv.J():.6e}
        Current penalty (QuadPen.J):     ({CURR_WEIGHT:.3e}){Jcurr.J():.6e} = {CURR_WEIGHT * Jcurr.J():.6e}

    n_dofs = {len(JF.x)}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Optimization tracking and diagnostics
# ──────────────────────────────────────────────────────────────────────────────
# `*_prev` entries are the last state L-BFGS-B accepted, i.e. the last one
# whose Boozer solve converged on a non-self-intersecting surface. They are
# both the warm-start for the next solve and the fallback a rejected step
# returns to; `callback` advances them only on an accepted iterate. Keeping
# the accepted coil DOFs with the other rollback state is important because
# the failed Boozer trial has already mutated the shared objective graph.
track = dict(
    eval=0,
    iter=0,
    f_prev=None,
    f_curr=None,
    sdofs_prev=surface.x.copy(),
    iota_prev=boozersurface.res["iota"],
    G_prev=boozersurface.res["G"],
    x_prev=JF.x.copy(),
    J_prev=JF.J(),
)


def _write_diagnostics_row(J, dJ, t0, solve_ok):
    """Append a single diagnostics row to the CSV file (inner-loop tracking).

    `solve_ok` records whether this evaluation's Boozer solve converged on a
    non-self-intersecting surface. Rejected evaluations are written too, with
    the fallback J/dJ they returned, so the CSV shows where the line search
    probed and was pushed back rather than silently skipping those rows.
    """
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
        f"{curve_poloidal_half_extent(banana_curve, WINDING_R0):.6e},"
        f"{surface_vessel_clearance(surface, VESSEL_MAJOR_R, VESSEL_MINOR_R):.6e},"
        f"{Jgcr.shortest_radius():.6e},"
        f"{Jself.shortest_self_distance():.6e},"
        f"{','.join(str(n) for n in solver_iters.values())},"
        f"{int(solve_ok)}"
    )
    proc0_print(row)
    with open(DIAGNOSTICS_FILE, 'a') as f:
        f.write(row + "\n")


def fun(dofs):
    """Objective function for L-BFGS-B with a required Boozer solve.

    A Boozer solve that fails to converge, or converges onto a
    self-intersecting surface, is a property of the trial coil DOFs rather
    than a fatal condition: L-BFGS-B is mid-line-search and only needs to be
    told the step is bad. This returns a smooth quadratic reject penalty
    centered on the last accepted DOFs, so the returned ``(J, dJ)`` is a
    valid trial-point pair whose descent direction contracts the line search.
    The complete accepted objective state, including the coil DOFs, is
    restored because `run_code` mutates shared state even when it fails.
    Raising here instead would abandon an otherwise healthy optimization at
    the first bad trial.
    """
    # Restore surface state for warm-start
    surface.x                 = track["sdofs_prev"]
    boozersurface.res["iota"] = track["iota_prev"]
    boozersurface.res["G"]    = track["G_prev"]

    JF.x = dofs
    for nit_key in solver_iters:
        solver_iters[nit_key] = 0
    with stdout_to_log(proc0_print, tap=_solver_iter_tap):
        res = boozersurface.run_code(track["iota_prev"], track["G_prev"])

    solve_success = res["success"]
    not_intersecting = not surface.is_self_intersecting()
    success = solve_success and not_intersecting

    if success:
        J  = JF.J()
        dJ = JF.dJ()
    else:
        delta = np.asarray(dofs) - track["x_prev"]
        reject_scale = max(1.0, abs(track["J_prev"]))
        J  = track["J_prev"] + reject_scale * (1.0 + np.dot(delta, delta))
        dJ = 2.0 * reject_scale * delta
        surface.x                 = track["sdofs_prev"]
        boozersurface.res["iota"] = track["iota_prev"]
        boozersurface.res["G"]    = track["G_prev"]
        JF.x                       = track["x_prev"]
        reasons = []
        if not solve_success:
            reasons.append("Boozer solve failed")
        if not not_intersecting:
            reasons.append("surface is self-intersecting")
        proc0_print(f"    step rejected ({'; '.join(reasons)}); "
                    f"returning a quadratic reject penalty")

    _write_diagnostics_row(J, dJ, t0, solve_ok=success)
    return J, dJ


def callback(x):
    """Callback called after each L-BFGS-B iteration (outer-loop tracking)."""
    accepted_x = np.asarray(x).copy()
    JF.x = accepted_x
    J  = JF.J()
    dJ = JF.dJ()
    res = boozersurface.res
    track["x_prev"] = accepted_x
    track["sdofs_prev"] = surface.x.copy()
    track["iota_prev"]  = res["iota"]
    track["G_prev"]     = res["G"]
    track["J_prev"]     = J
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
        Poloidal half extent:            {curve_poloidal_half_extent(banana_curve, WINDING_R0):.6e} rad
        Plasma-vessel clearance:         {surface_vessel_clearance(surface, VESSEL_MAJOR_R, VESSEL_MINOR_R):.6e} m
        Min global curv radius:          {Jgcr.shortest_radius():.6e} m (limit: {GCR_MIN_HW:.3e} m)
        Shortest self-distance:          {Jself.shortest_self_distance():.6e} m (limit: {SELFINT_MIN_HW:.3e} m)

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
        Poloidal extent penalty:         ({POL_WEIGHT:.3e}){Jpol.J():.6e} = {POL_WEIGHT * Jpol.J():.6e}
        Plasma-vessel penalty:           ({PV_WEIGHT:.3e}){Jpv.J():.6e} = {PV_WEIGHT * Jpv.J():.6e}
        Current penalty (QuadPen.J):     ({CURR_WEIGHT:.3e}){Jcurr.J():.6e} = {CURR_WEIGHT * Jcurr.J():.6e}
"""
    )


# ──────────────────────────────────────────────────────────────────────────────
# Initialize diagnostics file
# ──────────────────────────────────────────────────────────────────────────────
t0 = time.time()

with open(DIAGNOSTICS_FILE, 'w') as f:
    f.write('# Singlestage Diagnostics\n')
    f.write(f'# Date: {datetime.now()}\n')
    f.write(f'# BoozerSurface: {STAGE2_BSURF_FILE}\n')
    f.write(f'# MPOL={MPOL}, NTOR={NTOR}, CONSTRAINT_WEIGHT={CONSTRAINT_WEIGHT:.3e}\n')
    f.write(f'# TARGET_VOLUME={TARGET_VOLUME}, TARGET_IOTA={TARGET_IOTA}\n')
    f.write(f'# LENGTH_THRESHOLD={LENGTH_THRESHOLD}, CC_THRESHOLD={CC_THRESHOLD}, CS_THRESHOLD={CS_THRESHOLD}, PV_THRESHOLD={PV_THRESHOLD}, CURV_THRESHOLD={CURV_THRESHOLD}, POLOIDAL_THRESHOLD_RAD={POLOIDAL_THRESHOLD_RAD}\n')
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
        'poloidal_extent_rad,'
        'plasma_vessel_clearance,'
        'min_global_curv_radius,'
        'shortest_self_distance,'
        f"{','.join(solver_iters)},"
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
        Poloidal half extent:            {curve_poloidal_half_extent(banana_curve, WINDING_R0):.6e} rad
        Plasma-vessel clearance:         {surface_vessel_clearance(surface, VESSEL_MAJOR_R, VESSEL_MINOR_R):.6e} m
        Min global curv radius:          {Jgcr.shortest_radius():.6e} m (limit: {GCR_MIN_HW:.3e} m)
        Shortest self-distance:          {Jself.shortest_self_distance():.6e} m (limit: {SELFINT_MIN_HW:.3e} m)

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
        Poloidal extent penalty:         ({POL_WEIGHT:.3e}){Jpol.J():.6e} = {POL_WEIGHT * Jpol.J():.6e}
        Plasma-vessel penalty:           ({PV_WEIGHT:.3e}){Jpv.J():.6e} = {PV_WEIGHT * Jpv.J():.6e}
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
_lineage_metrics = boozer_json_vacuum_lineage(_bsurf_out_path)
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
    "tf_current_A":           float(cfg['tf_coils']['current']),
    "banana_current_max_abs_A": float(abs(banana_coils[0].current.get_value())),
    "banana_current_limit_A": float(BANANA_CURRENT_MAX),
    "length_target_m":        float(LENGTH_THRESHOLD),
    "length_abs_max_m":       float(LENGTH_MAX_HW),
    "coil_coil_min_threshold_m": float(CC_THRESHOLD),
    "coil_plasma_min_threshold_m": float(CS_THRESHOLD),
    "plasma_vessel_min_m":    float(surface_vessel_clearance(surface, VESSEL_MAJOR_R, VESSEL_MINOR_R)),
    "plasma_vessel_min_threshold_m": float(PV_THRESHOLD),
    "curvature_threshold_inv_m": float(CURV_THRESHOLD),
    "poloidal_extent_rad":    float(curve_poloidal_half_extent(banana_curve, WINDING_R0)),
    "poloidal_extent_threshold_rad": float(POLOIDAL_THRESHOLD_RAD),
    "min_global_curv_radius_m": float(Jgcr.shortest_radius()),
    "global_curv_radius_threshold_m": float(GCR_MIN_HW),
    "shortest_self_distance_m": float(Jself.shortest_self_distance()),
    "self_intersect_threshold_m": float(SELFINT_MIN_HW),
    "boozer_bfgs_nit":        int(solver_iters.get("bfgs_nit", 0)),
    "boozer_newton_nit":      int(solver_iters.get("newton_nit", 0)),
    "winding_R0_m":           float(WINDING_R0),
    "winding_a_m":            float(WINDING_A),
    "target_lcfs_major_radius_max_m": float(TARGET_LCFS_MAJOR_R),
    "target_lcfs_minor_radius_max_m": float(TARGET_LCFS_MINOR_R),
    **surface_shape_metrics(surface),
    **_lineage_metrics,
}
if success and not _lineage_metrics["vacuum_lineage_ok"]:
    _err_code = "file_save_failed"
    _err_msg = "saved BoozerSurface failed vacuum lineage validation"
    registry.mark_failed("singlestage", RUN_ID, error_code=_err_code,
                         error_message=_err_msg, slurm_wall_s=float(opt_runtime),
                         metrics=_metrics)
    raise RuntimeError(_err_msg)
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
