import argparse
from dataclasses import dataclass
from functools import lru_cache
import logging
import os
import sys
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, EXAMPLE_ROOT)

SIMSOPT_ROOT = os.path.abspath(os.path.join(EXAMPLE_ROOT, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
SRC_ROOT = os.path.join(SIMSOPT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, SIMSOPT_ROOT)
sys.path.insert(0, REPO_ROOT)

from repo_bootstrap import bootstrap_local_simsopt, configure_entrypoint_jax_runtime


configure_entrypoint_jax_runtime(sys.argv[1:])

# bootstrap_local_simsopt must run BEFORE any simsopt.* imports to avoid
# pybind11 double-registration when both site-packages and build-directory
# simsoptpp .so files exist.
bootstrap_local_simsopt(SRC_ROOT)

import jax
import jaxlib
import numpy as np

from jax_host_boundary import host_array, host_bool, host_float
from equilibria_paths import (
    DEFAULT_EQUILIBRIA_DIR,
    WORKSPACE_EQUILIBRIA_DIR,
    resolve_equilibrium_path,
)
from hardware_constraints import (
    apply_hardware_constraint_verdict,
    sanitize_json_payload,
)
from run_metadata import build_artifact_manifest, build_runtime_provenance
from alm_utils import (
    minimize_alm,
    run_directional_taylor_test,
    validate_alm_cli_args,
)
from banana_opt.current_contracts import (
    apply_penalty_traversal_forbidden_box_bounds,
)
from banana_opt.hardware_constraint_schema import (
    hardware_constraint_alm_names,
)
from banana_opt.hardware_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
    BANANA_WINDING_MINOR_RADIUS_M,
    COIL_COIL_MIN_DIST_M,
    COIL_LENGTH_TARGET_M,
    COIL_PLASMA_MIN_DIST_M,
    COIL_VESSEL_MIN_DIST_M,
    MAX_CURVATURE_INV_M,
    PLASMA_VESSEL_MIN_DIST_M,
    TF_CURRENT_HARD_LIMIT_A,
    VACUUM_VESSEL_MAJOR_RADIUS_M,
    validate_banana_winding_surface_radius,
    validate_tf_current_limit,
)
from banana_opt.reference_surfaces import build_banana_reference_surfaces
from banana_opt.stage2_objectives import (
    build_stage2_alm_settings as _build_stage2_alm_settings_impl,
    build_stage2_results as _build_stage2_results_impl,
    evaluate_stage2_alm_problem as _evaluate_stage2_alm_problem_impl,
    evaluate_stage2_hardware_constraints as _evaluate_stage2_hardware_constraints_impl,
    smooth_min_curve_surface_signed_constraint as _smooth_min_curve_surface_signed_constraint_impl,
    stage2_constraint_activity_tolerances as _stage2_constraint_activity_tolerances_impl,
)
from simsopt.config import maybe_initialize_distributed_jax
from simsopt.jax_core import (
    closed_curve_self_intersection_summary,
    curve_gamma_and_dash_from_spec,
    curve_spec_from_curve,
    curve_spec_with_quadpoints,
)

maybe_initialize_distributed_jax()
LOGGER = logging.getLogger(__name__)
DATABASE_EQUILIBRIA_DIR = str(WORKSPACE_EQUILIBRIA_DIR)
STAGE2_TARGET_OBJECTIVE_DOF_LAYOUT_ERROR = (
    "Stage 2 target objective DOF layout does not match the composite objective."
)
_STAGE2_RESULTS_SCHEMA_VERSION = 1
_STAGE2_REQUIRED_ARTIFACT_FILENAMES = (
    "results.json",
    "biot_savart_opt.json",
    "surf_opt.json",
)
_SMOOTHING_EPS = float(np.finfo(float).eps)


@lru_cache(maxsize=32)
def _cached_raw_terms_jacobian(raw_terms):
    return jax.jit(jax.jacrev(raw_terms))


def _zero_gradient_like(values):
    return np.zeros_like(np.asarray(values, dtype=float))


def _optional_host_float(value):
    return None if value is None else host_float(value)


def _host_curve_max_curvature(curve):
    return float(np.max(host_array(curve.kappa(), dtype=np.float64)))


def _host_dofs_copy(dofs):
    return host_array(dofs, dtype=np.float64).copy()


def resolve_curvature_threshold(value: float) -> float:
    """Clamp Stage 2 curvature thresholds to the shared hardware ceiling."""
    return min(float(value), MAX_CURVATURE_INV_M)


def stage2_alm_constraint_names(*, include_coil_surface: bool) -> tuple[str, ...]:
    requested_names = [
        "coil_coil_spacing",
        "max_curvature",
        "coil_length",
        "banana_current",
    ]
    if include_coil_surface:
        requested_names.insert(1, "coil_surface_spacing")
    return hardware_constraint_alm_names(names=tuple(requested_names))


def validate_banana_current_cli_args(args) -> None:
    banana_init_current_A = float(args.banana_init_current_A)
    banana_current_max_A = float(args.banana_current_max_A)
    if not (0.0 < banana_init_current_A <= BANANA_CURRENT_HARD_LIMIT_A):
        raise ValueError(
            f"--banana-init-current-A must be in the interval (0, {BANANA_CURRENT_HARD_LIMIT_A:.0f}]."
        )
    if not (0.0 < banana_current_max_A <= BANANA_CURRENT_HARD_LIMIT_A):
        raise ValueError(
            f"--banana-current-max-A must be in the interval (0, {BANANA_CURRENT_HARD_LIMIT_A:.0f}]."
        )
    if banana_init_current_A > banana_current_max_A:
        raise ValueError(
            "--banana-init-current-A cannot exceed --banana-current-max-A."
        )
    validate_tf_current_limit(args.tf_current_A)


def build_hbt_reference_surfaces(nfp: int, banana_surf_radius: float):
    surfaces = build_banana_reference_surfaces(nfp, banana_surf_radius)
    return surfaces.hbt, surfaces.coil_winding_surface, surfaces.vessel


def evaluate_stage2_hardware_constraints(
    coil_length,
    length_target,
    curve_curve_min_dist,
    cc_threshold,
    max_curvature,
    curvature_threshold,
    curve_surface_min_dist=None,
    coil_surface_threshold=None,
    plasma_vessel_min_dist=None,
    plasma_vessel_threshold=None,
    banana_current_A=None,
    banana_current_threshold=None,
    tf_current_A=None,
    tf_current_threshold=None,
    *,
    self_intersecting=False,
):
    """Evaluate hard Stage 2 hardware constraints against realized geometry."""
    status = _evaluate_stage2_hardware_constraints_impl(
        coil_length,
        length_target,
        curve_curve_min_dist,
        cc_threshold,
        max_curvature,
        curvature_threshold,
        curve_surface_min_dist=curve_surface_min_dist,
        coil_surface_threshold=coil_surface_threshold,
        plasma_vessel_min_dist=plasma_vessel_min_dist,
        plasma_vessel_threshold=plasma_vessel_threshold,
        banana_current_A=banana_current_A,
        banana_current_threshold=banana_current_threshold,
        tf_current_A=tf_current_A,
        tf_current_threshold=tf_current_threshold,
    )
    if self_intersecting:
        status = {
            **status,
            "success": False,
            "violations": [*status["violations"], "banana_curve is self-intersecting"],
        }
    status["self_intersecting"] = bool(self_intersecting)
    return status


def _evaluate_stage2_artifact_hardware_status(
    *,
    banana_curve,
    coil_length,
    length_target,
    curve_curve_min_dist,
    cc_threshold,
    max_curvature,
    curvature_threshold,
    curve_surface_min_dist,
    coil_surface_threshold,
    plasma_vessel_min_dist,
    plasma_vessel_threshold,
    banana_current_A,
    banana_current_threshold,
    tf_current_A,
    tf_current_threshold,
):
    hardware_status = evaluate_stage2_hardware_constraints(
        coil_length,
        length_target,
        curve_curve_min_dist,
        cc_threshold,
        max_curvature,
        curvature_threshold,
        curve_surface_min_dist=curve_surface_min_dist,
        coil_surface_threshold=coil_surface_threshold,
        plasma_vessel_min_dist=plasma_vessel_min_dist,
        plasma_vessel_threshold=plasma_vessel_threshold,
        banana_current_A=banana_current_A,
        banana_current_threshold=banana_current_threshold,
        tf_current_A=tf_current_A,
        tf_current_threshold=tf_current_threshold,
    )
    if not hardware_status["success"]:
        return hardware_status
    if not build_curve_self_intersection_summary(banana_curve)["intersecting"]:
        return hardware_status
    return evaluate_stage2_hardware_constraints(
        coil_length,
        length_target,
        curve_curve_min_dist,
        cc_threshold,
        max_curvature,
        curvature_threshold,
        curve_surface_min_dist=curve_surface_min_dist,
        coil_surface_threshold=coil_surface_threshold,
        plasma_vessel_min_dist=plasma_vessel_min_dist,
        plasma_vessel_threshold=plasma_vessel_threshold,
        banana_current_A=banana_current_A,
        banana_current_threshold=banana_current_threshold,
        tf_current_A=tf_current_A,
        tf_current_threshold=tf_current_threshold,
        self_intersecting=True,
    )


def _capture_stage2_artifact_state(
    *,
    dofs,
    JF,
    BASE_OBJECTIVE,
    Jf,
    Jls,
    Jccdist,
    Jcsdist,
    new_banana_curve,
    new_banana_coils,
    new_tf_coils,
    length_target,
    cc_threshold,
    curvature_threshold,
    coil_surface_threshold,
    plasma_vessel_min_dist,
    plasma_vessel_threshold,
    banana_current_max_A,
):
    candidate_x = _host_dofs_copy(dofs)
    JF.x = candidate_x
    BASE_OBJECTIVE.x = candidate_x
    coil_length = host_float(Jls.J())
    curve_curve_min_dist = host_float(Jccdist.shortest_distance())
    curve_surface_min_dist = _optional_host_float(
        None if Jcsdist is None else Jcsdist.shortest_distance()
    )
    max_curvature = _host_curve_max_curvature(new_banana_curve)
    banana_current_A = host_float(new_banana_coils[0].current.get_value())
    tf_current_A = host_float(new_tf_coils[0].current.get_value())
    hardware_status = _evaluate_stage2_artifact_hardware_status(
        banana_curve=new_banana_curve,
        coil_length=coil_length,
        length_target=length_target,
        curve_curve_min_dist=curve_curve_min_dist,
        cc_threshold=cc_threshold,
        max_curvature=max_curvature,
        curvature_threshold=curvature_threshold,
        curve_surface_min_dist=curve_surface_min_dist,
        coil_surface_threshold=coil_surface_threshold,
        plasma_vessel_min_dist=plasma_vessel_min_dist,
        plasma_vessel_threshold=plasma_vessel_threshold,
        banana_current_A=banana_current_A,
        banana_current_threshold=banana_current_max_A,
        tf_current_A=tf_current_A,
        tf_current_threshold=TF_CURRENT_HARD_LIMIT_A,
    )
    return {
        "x": candidate_x,
        "field_objective": host_float(Jf.J()),
        "coil_length": coil_length,
        "curve_curve_min_dist": curve_curve_min_dist,
        "curve_surface_min_dist": curve_surface_min_dist,
        "max_curvature": max_curvature,
        "banana_current_A": banana_current_A,
        "tf_current_A": tf_current_A,
        "hardware_status": hardware_status,
    }


def parse_args():
    from simsopt.geo.optimizer_jax import VALID_LEAST_SQUARES_ALGORITHMS

    parser = argparse.ArgumentParser(
        description="Run Stage 2 banana coil optimization against a fixed plasma surface.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=os.environ.get(
            "PLASMA_SURF_FILENAME", "wout_nfp22ginsburg_000_014417_iota15.nc"
        ),
        help="VMEC wout filename under the equilibria directory.",
    )
    parser.add_argument(
        "--equilibria-dir",
        default=os.environ.get("EQUILIBRIA_DIR", str(DEFAULT_EQUILIBRIA_DIR)),
        help="Directory that contains the equilibrium wout files.",
    )
    parser.add_argument(
        "--equilibrium-path",
        default=os.environ.get("EQUILIBRIUM_PATH"),
        help="Explicit path to the equilibrium file. Overrides --equilibria-dir.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("STAGE2_OUTPUT_ROOT", SCRIPT_DIR),
        help="Directory where outputs-[plasma] will be written.",
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=os.environ.get("STAGE2_BS_PATH"),
        help="Optional path to a saved Stage 2 biot_savart_opt.json seed to restart from.",
    )
    parser.add_argument(
        "--export-objective-json",
        default=os.environ.get("STAGE2_EXPORT_OBJECTIVE_JSON"),
        help="Optional path to write the initialized squared-flux/composite objective snapshot as JSON.",
    )
    parser.add_argument(
        "--override-dofs-json",
        default=os.environ.get("STAGE2_OVERRIDE_DOFS_JSON"),
        help=(
            "Optional JSON file containing an explicit DOF vector to load into the "
            "Stage 2 composite objective before probe/init/optimization."
        ),
    )
    parser.add_argument(
        "--trajectory-json",
        default=os.environ.get("STAGE2_TRAJECTORY_JSON"),
        help="Optional path to write per-evaluation objective diagnostics as JSON.",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Skip heavy VTK/plot/save artifact generation while still writing results.json.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Build the initialized Stage 2 objective snapshot, export JSON if requested, and exit before optimization/postprocess.",
    )
    parser.add_argument("--nphi", type=int, default=int(os.environ.get("NPHI", "255")))
    parser.add_argument(
        "--ntheta", type=int, default=int(os.environ.get("NTHETA", "64"))
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Build and save the initialized configuration without running the optimizer.",
    )
    parser.add_argument(
        "--banana-surf-radius",
        type=float,
        default=float(
            os.environ.get("BANANA_SURF_RADIUS", str(BANANA_WINDING_MINOR_RADIUS_M))
        ),
        help="Coil surface minor radius (default 0.21 m, concentric with HBT vacuum vessel).",
    )
    parser.add_argument(
        "--tf-current-A",
        type=float,
        default=float(os.environ.get("TF_CURRENT_A", str(TF_CURRENT_HARD_LIMIT_A))),
        help="Per-TF-coil current in physical SI amperes (default 8e4 = 80 kA).",
    )
    parser.add_argument(
        "--banana-init-current-A",
        type=float,
        default=float(os.environ.get("BANANA_INIT_CURRENT_A", "1e4")),
        help="Fresh-initialization banana-coil current in SI amperes.",
    )
    parser.add_argument(
        "--banana-current-max-A",
        type=float,
        default=float(
            os.environ.get(
                "BANANA_CURRENT_MAX_A",
                str(BANANA_CURRENT_HARD_LIMIT_A),
            )
        ),
        help="Hard upper bound on the realized banana-coil current in SI amperes.",
    )
    parser.add_argument(
        "--major-radius",
        type=float,
        default=float(os.environ.get("MAJOR_RADIUS", "0.915")),
        help="Target major radius used to rescale the plasma surface.",
    )
    parser.add_argument(
        "--toroidal-flux",
        type=float,
        default=float(os.environ.get("TOROIDAL_FLUX", "0.24")),
        help="Flux-surface label s used when loading the VMEC surface.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=int(os.environ.get("COIL_ORDER", "2")),
        help="Fourier order for the banana coil.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=int(os.environ.get("MAXITER", "300")),
        help="Maximum optimizer iterations.",
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=float(os.environ.get("FTOL", "1e-15")),
        help=(
            "L-BFGS-B function change tolerance. Ignored by the LM lane. "
            "Default 1e-15 (factr~4.5) effectively lets maxiter control termination."
        ),
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=float(os.environ.get("GTOL", "1e-15")),
        help=(
            "Optimizer termination tolerance. On the LM lane this is the "
            "least-squares gradient infinity-norm tolerance."
        ),
    )
    parser.add_argument(
        "--constraint-method",
        choices=["penalty", "alm"],
        default=os.environ.get("CONSTRAINT_METHOD", "penalty"),
        help="Use the weighted-penalty objective or the augmented Lagrangian outer loop.",
    )
    parser.add_argument(
        "--alm-max-outer-iters",
        type=int,
        default=int(os.environ.get("ALM_MAX_OUTER_ITERS", "10")),
        help="Maximum number of ALM outer iterations (default 10).",
    )
    parser.add_argument(
        "--alm-penalty-init",
        type=float,
        default=float(os.environ.get("ALM_PENALTY_INIT", "1.0")),
        help="Initial ALM penalty parameter (default 1.0).",
    )
    parser.add_argument(
        "--alm-penalty-scale",
        type=float,
        default=float(os.environ.get("ALM_PENALTY_SCALE", "10.0")),
        help="Multiplicative ALM penalty growth factor (default 10.0).",
    )
    parser.add_argument(
        "--alm-penalty-max",
        type=float,
        default=float(os.environ.get("ALM_PENALTY_MAX", "1e8")),
        help="Maximum ALM penalty parameter before capped termination (default 1e8).",
    )
    parser.add_argument(
        "--alm-feas-tol",
        type=float,
        default=float(os.environ.get("ALM_FEAS_TOL", "1e-6")),
        help="ALM max-violation stopping tolerance (default 1e-6).",
    )
    parser.add_argument(
        "--alm-stationarity-tol",
        type=float,
        default=float(os.environ.get("ALM_STATIONARITY_TOL", "1e-6")),
        help="ALM augmented-gradient stopping tolerance (default 1e-6).",
    )
    parser.add_argument(
        "--alm-trust-radius-init",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_INIT", "0.05")),
        help="Initial relative trust radius for bounded ALM inner solves (0 disables bounds).",
    )
    parser.add_argument(
        "--alm-trust-radius-min",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_MIN", "1e-4")),
        help="Minimum relative trust radius for bounded ALM inner solves.",
    )
    parser.add_argument(
        "--alm-trust-radius-shrink",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_SHRINK", "0.5")),
        help="Multiplicative shrink factor for the ALM inner trust radius.",
    )
    parser.add_argument(
        "--alm-trust-radius-grow",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_GROW", "1.5")),
        help="Multiplicative growth factor for the ALM inner trust radius after good steps.",
    )
    parser.add_argument(
        "--alm-max-inner-attempts",
        type=int,
        default=int(os.environ.get("ALM_MAX_INNER_ATTEMPTS", "4")),
        help="Maximum number of trust-radius retries per ALM outer iteration.",
    )
    parser.add_argument(
        "--alm-max-subproblem-continuations",
        type=int,
        default=int(os.environ.get("ALM_MAX_SUBPROBLEM_CONTINUATIONS", "20")),
        help="Maximum accepted-feasible continuation solves before forcing an ALM return.",
    )
    parser.add_argument(
        "--alm-distance-smoothing",
        type=float,
        default=float(os.environ.get("ALM_DISTANCE_SMOOTHING", "0.005")),
        help="Distance soft-min temperature for Stage 2 ALM spacing constraints.",
    )
    parser.add_argument(
        "--alm-curvature-smoothing",
        type=float,
        default=float(os.environ.get("ALM_CURVATURE_SMOOTHING", "0.25")),
        help="Curvature soft-max temperature for Stage 2 ALM curvature constraints.",
    )
    parser.add_argument(
        "--alm-taylor-test",
        action="store_true",
        help="Run a directional Taylor test on the initialized Stage 2 ALM subproblem before optimization.",
    )
    parser.add_argument(
        "--alm-taylor-test-seed",
        type=int,
        default=int(os.environ.get("ALM_TAYLOR_TEST_SEED", "1")),
        help="Random seed used to build the Stage 2 ALM Taylor-test direction.",
    )
    parser.add_argument(
        "--length-weight",
        type=float,
        default=float(os.environ.get("LENGTH_WEIGHT", "0.0005")),
        help="Curve-length penalty weight.",
    )
    parser.add_argument(
        "--length-target",
        type=float,
        default=float(os.environ.get("LENGTH_TARGET", str(COIL_LENGTH_TARGET_M))),
        help="Curve-length target in meters.",
    )
    parser.add_argument(
        "--cc-threshold",
        type=float,
        default=float(os.environ.get("CC_THRESHOLD", str(COIL_COIL_MIN_DIST_M))),
        help="Coil-coil distance threshold in meters.",
    )
    parser.add_argument(
        "--cc-weight",
        type=float,
        default=float(os.environ.get("CC_WEIGHT", "100")),
        help="Coil-coil distance penalty weight.",
    )
    parser.add_argument(
        "--curvature-weight",
        type=float,
        default=float(os.environ.get("CURVATURE_WEIGHT", "0.0001")),
        help="Curvature penalty weight.",
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=float(os.environ.get("CURVATURE_THRESHOLD", str(MAX_CURVATURE_INV_M))),
        help="Curvature threshold in m^-1 (default 100, matching the hardware ceiling).",
    )
    parser.add_argument(
        "--theta-center",
        type=float,
        default=float(os.environ.get("THETA_CENTER", "0.5")),
        help="Initial banana-coil poloidal center in normalized angle coordinates.",
    )
    parser.add_argument(
        "--phi-center",
        type=float,
        default=float(os.environ.get("PHI_CENTER", "0.06")),
        help="Initial banana-coil toroidal center in normalized angle coordinates.",
    )
    parser.add_argument(
        "--theta-width",
        type=float,
        default=float(os.environ.get("THETA_WIDTH", "0.1")),
        help="Initial banana-coil poloidal width in normalized angle coordinates.",
    )
    parser.add_argument(
        "--phi-width",
        type=float,
        default=float(os.environ.get("PHI_WIDTH", "0.03")),
        help="Initial banana-coil toroidal width in normalized angle coordinates.",
    )
    parser.add_argument(
        "--num-quadpoints",
        type=int,
        default=int(os.environ.get("NUM_QUADPOINTS", "128")),
        help="Number of quadrature points per coil (default 128).",
    )
    parser.add_argument(
        "--curvature-p-norm",
        type=int,
        default=int(os.environ.get("CURVATURE_P_NORM", "4")),
        help="Lp norm exponent for curvature penalty (default 4).",
    )
    parser.add_argument(
        "--squared-flux-weight",
        type=float,
        default=float(os.environ.get("SQUARED_FLUX_WEIGHT", "1.0")),
        help="Squared flux objective weight (default 1.0).",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "jax"],
        default=os.environ.get("SIMSOPT_BACKEND")
        or os.environ.get("STAGE2_BACKEND", "cpu"),
        help="Field backend: 'cpu' (simsoptpp) or 'jax' (JAX Biot-Savart). "
        "Env: SIMSOPT_BACKEND (or legacy STAGE2_BACKEND).",
    )
    parser.add_argument(
        "--optimizer-backend",
        choices=["scipy", "ondevice"],
        default=os.environ.get("STAGE2_OPTIMIZER_BACKEND")
        or os.environ.get("OPTIMIZER_BACKEND"),
        help=(
            "Stage 2 optimizer backend. "
            "'scipy' is the trusted reference lane; "
            "'ondevice' is the JAX target optimizer lane. Defaults to 'ondevice' on "
            "the JAX backend and 'scipy' on the CPU/reference backend when no "
            "explicit override is provided."
        ),
    )
    parser.add_argument(
        "--least-squares-algorithm",
        choices=sorted(VALID_LEAST_SQUARES_ALGORITHMS),
        default=os.environ.get("STAGE2_LEAST_SQUARES_ALGORITHM"),
        help=(
            "Stage 2 least-squares algorithm. 'lm' routes the JAX ondevice "
            "lane through the pure JAX LM residual solver. Defaults to "
            "'quasi-newton' when no explicit override is provided; 'lm' is "
            "explicit opt-in only."
        ),
    )
    parser.add_argument(
        "--record-warm-timings",
        action="store_true",
        help=(
            "After the primary JAX ondevice Stage 2 optimization run, reset to "
            "the same initial DOFs and rerun once in-process to capture a warm "
            "timing."
        ),
    )
    parser.add_argument(
        "--field-diagnostic-stride",
        type=int,
        default=int(os.environ.get("FIELD_DIAGNOSTIC_STRIDE", "0")),
        help=(
            "Refresh the expensive squared-flux-only and mean-|B.n|/|B| "
            "diagnostics every N optimizer evaluations on explicit value/grad "
            "lanes. Use 0 to select the lane default."
        ),
    )
    parser.add_argument(
        "--profile-step-json",
        default=None,
        help=(
            "Write a one-step explicit Stage 2 objective breakdown JSON. "
            "Supported on reference/value-and-gradient lanes only."
        ),
    )
    args = parser.parse_args()
    args.optimizer_backend = resolve_stage2_default_optimizer_backend(
        args.backend,
        args.optimizer_backend,
    )
    args.least_squares_algorithm = resolve_stage2_default_least_squares_algorithm(
        args.backend,
        args.optimizer_backend,
        args.least_squares_algorithm,
    )
    validate_stage2_constraint_method_args(args)
    return args


def resolve_stage2_default_optimizer_backend(field_backend, optimizer_backend=None):
    """Resolve the implicit Stage 2 optimizer lane from the selected backend."""
    if optimizer_backend is not None:
        return optimizer_backend
    if field_backend == "jax":
        return "ondevice"
    return "scipy"


def resolve_stage2_default_least_squares_algorithm(
    field_backend,
    optimizer_backend,
    least_squares_algorithm=None,
):
    """Resolve the implicit Stage 2 least-squares algorithm for the active lane."""
    if least_squares_algorithm is not None:
        return least_squares_algorithm
    return "quasi-newton"


def validate_stage2_constraint_method_args(args) -> None:
    args.banana_surf_radius = validate_banana_winding_surface_radius(
        args.banana_surf_radius
    )
    validate_banana_current_cli_args(args)
    if args.constraint_method == "alm":
        validate_alm_cli_args(args)


def build_equilibrium_path(args):
    return str(
        resolve_equilibrium_path(
            plasma_surf_filename=args.plasma_surf_filename,
            equilibria_dir=args.equilibria_dir,
            equilibrium_path=args.equilibrium_path,
            fallback_dirs=(DEFAULT_EQUILIBRIA_DIR, WORKSPACE_EQUILIBRIA_DIR),
        )
    )


def load_stage2_seed_configuration(seed_bs_path, surf, num_tf_coils, out_dir):
    bs = load(seed_bs_path)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    coils = bs.coils
    curves = [coil.curve for coil in coils]
    curves_to_vtk(curves, out_dir + "curves_init", close=True)
    unitn = surf.unitnormal()
    point_data = {
        "B_N": np.sum(bs.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]
    }
    surf.to_vtk(out_dir + "surf_init", extra_data=point_data)

    banana_coils = coils[num_tf_coils:]
    banana_curve = banana_coils[0].curve
    tf_coils = coils[:num_tf_coils]
    return bs, curves, banana_curve, banana_coils, tf_coils


def initSurface(R0, s, file_loc, nphi, ntheta):
    # Initialize the boundary magnetic surface and scale it to the target major radius
    surf = SurfaceRZFourier.from_wout(
        file_loc, range="full torus", nphi=nphi, ntheta=ntheta, s=s
    )
    # scale the surface down to the target appropriate major radius
    surf.set_dofs(surf.get_dofs() * R0 / surf.major_radius())
    print("Major radius target: ", R0)
    print("Major radius actual: ", surf.major_radius())
    print("Minor radius: ", surf.minor_radius())
    return surf


def initializeCoils(
    surf,
    surf_coils,
    tf_coils,
    num_quadpoints,
    order,
    banana_init_current_A,
    phi_center,
    theta_center,
    phi_width,
    theta_width,
    OUT_DIR,
):
    # Initialize banana coils on the coil winding surface
    # Keep the production banana-coil class shared across CPU and JAX lanes.
    # CurveCWSFourierCPP now exposes the JAX geometry/VJP surface needed by the
    # target lane, and reusing it here avoids backend-specific curve-derivative
    # drift in near-threshold Stage 2 states.
    banana_curve = CurveCWSFourierCPP(
        np.linspace(0, 1, num_quadpoints, endpoint=False), order=order, surf=surf_coils
    )
    banana_curve.set("phic(0)", phi_center)
    banana_curve.set("thetac(0)", theta_center)
    banana_curve.set("phic(1)", phi_width)
    banana_curve.set("thetas(1)", theta_width)

    # Apply symmetries - if stellsym = False, only one per half field period (and two if true)
    banana_coils = coils_via_symmetries(
        [banana_curve],
        [ScaledCurrent(Current(1), float(banana_init_current_A))],
        surf_coils.nfp,
        surf_coils.stellsym,
    )

    # Combined coil set to evaluate magnetic field
    coils = tf_coils + banana_coils
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    # Save initialization state
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, OUT_DIR + "curves_init", close=True)
    unitn = surf.unitnormal()
    pointData = {"B_N": np.sum(bs.B().reshape(unitn.shape) * unitn, axis=2)[:, :, None]}
    surf.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
    return bs, curves, banana_curve, banana_coils


def build_curve_self_intersection_summary(
    curve,
    *,
    npts=2000,
    tol_factor=0.1,
    neighbor_skip=3,
):
    """Return Stage 2 closed-curve self-intersection diagnostics from the JAX SSOT."""
    quadpoints = np.linspace(0.0, 1.0, int(npts), endpoint=False, dtype=np.float64)
    curve_spec = curve_spec_with_quadpoints(curve_spec_from_curve(curve), quadpoints)
    gamma, _ = curve_gamma_and_dash_from_spec(curve_spec)
    min_distance, tolerance, penalty, intersecting = (
        closed_curve_self_intersection_summary(
            gamma,
            tolerance_factor=tol_factor,
            neighbor_skip=neighbor_skip,
        )
    )
    return {
        "min_distance": host_float(min_distance),
        "tolerance": host_float(tolerance),
        "penalty": host_float(penalty),
        "intersecting": host_bool(intersecting),
        "npts": int(npts),
        "tolerance_factor": float(tol_factor),
        "neighbor_skip": int(neighbor_skip),
    }


def magneticFieldPlots(surf, bs, OUT_DIR_ITER):
    """Generate normal-field and magnitude-field diagnostic plots."""
    mean_abs_relBfinal_norm, modBfinal, surf_area, phi, theta = norm_field_plot(
        surf, bs, OUT_DIR_ITER + "NormFieldPlot"
    )
    magnitude_field_plot(
        modBfinal, surf_area, phi, theta, OUT_DIR_ITER + "MagFieldPlot"
    )
    return mean_abs_relBfinal_norm


def compute_mean_abs_relbn(surf, bs):
    """Return the area-weighted mean |B·n|/|B| without plotting artifacts."""
    theta = surf.quadpoints_theta
    phi = surf.quadpoints_phi
    del theta, phi  # keep logic aligned with plotting utility inputs
    n = surf.normal()
    absn = np.linalg.norm(n, axis=2)
    unitn = n * (1.0 / absn)[:, :, None]
    sqrt_area = np.sqrt(absn.reshape((-1, 1)) / float(absn.size))
    surf_area = sqrt_area**2
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bfinal = host_array(bs.B()).reshape(n.shape)
    Bfinal_norm = np.sum(Bfinal * unitn, axis=2)[:, :, None]
    modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))[:, :, None]
    relBfinal_norm = Bfinal_norm / modBfinal
    abs_relBfinal_norm_dA = np.abs(relBfinal_norm.reshape((-1, 1))) * surf_area
    return host_float(np.sum(abs_relBfinal_norm_dA) / np.sum(surf_area))


def resolve_stage2_field_bs(new_bs, bs_jax):
    return bs_jax if bs_jax is not None else new_bs


def _block_until_ready_tree(value):
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def time_stage2_callback(callback):
    start = time.perf_counter()
    _block_until_ready_tree(callback())
    return float(time.perf_counter() - start)


def time_stage2_callback_result(callback):
    start = time.perf_counter()
    result = callback()
    _block_until_ready_tree(result)
    return float(time.perf_counter() - start), result


def profile_stage2_named_callbacks(callbacks):
    return {
        name: time_stage2_callback(callback) for name, callback in callbacks.items()
    }


def build_stage2_profile_breakdown(timings):
    total_s = float(sum(timings.values()))
    dominant = sorted(
        (
            {
                "name": name,
                "elapsed_s": elapsed_s,
                "share": (elapsed_s / total_s) if total_s > 0.0 else 0.0,
            }
            for name, elapsed_s in timings.items()
        ),
        key=lambda item: item["elapsed_s"],
        reverse=True,
    )
    return total_s, dominant


def build_stage2_objective_term_callbacks(
    context,
):
    return {
        "squared_flux": {
            "J": context.Jf.J,
            "dJ": lambda: compute_stage2_term_gradient(context.Jf, context.JF),
        },
        "length_penalty": {
            "J": lambda: compute_stage2_length_penalty_value(
                context.Jls.J(),
                context.length_target,
            ),
            "dJ": lambda: compute_stage2_length_penalty_gradient(
                host_float(context.Jls.J()),
                context.length_target,
                context.Jls,
                context.JF,
            ),
        },
        "coil_distance": {
            "J": context.Jccdist.J,
            "dJ": lambda: compute_stage2_term_gradient(context.Jccdist, context.JF),
        },
        "curvature": {
            "J": context.Jc.J,
            "dJ": lambda: compute_stage2_term_gradient(context.Jc, context.JF),
        },
    }


def profile_stage2_objective_terms(context):
    term_callbacks = build_stage2_objective_term_callbacks(context)
    value_timings = profile_stage2_named_callbacks(
        {name: callbacks["J"] for name, callbacks in term_callbacks.items()}
    )
    gradient_timings = profile_stage2_named_callbacks(
        {name: callbacks["dJ"] for name, callbacks in term_callbacks.items()}
    )
    value_total_s, dominant_value_terms = build_stage2_profile_breakdown(value_timings)
    gradient_total_s, dominant_gradient_terms = build_stage2_profile_breakdown(
        gradient_timings
    )
    return {
        "value_timings_s": value_timings,
        "value_total_s": value_total_s,
        "dominant_value_terms": dominant_value_terms,
        "gradient_timings_s": gradient_timings,
        "gradient_total_s": gradient_total_s,
        "dominant_gradient_terms": dominant_gradient_terms,
    }


def profile_stage2_squared_flux_internal_components(Jf):
    if getattr(Jf, "_use_jax_native", True):
        return {}, 0.0, [], {}, [], []

    field_B_for_J_s, field_B_for_J = time_stage2_callback_result(Jf.field.B)
    integral_only_s = time_stage2_callback(lambda: Jf._jit_integral(field_B_for_J))
    field_B_for_dJ_s, field_B_for_dJ = time_stage2_callback_result(Jf.field.B)
    integral_value_grad_s, (_, dJ_dB) = time_stage2_callback_result(
        lambda: Jf._jit_integral_value_grad(field_B_for_dJ)
    )
    field_B_vjp_component_timings = {}
    dominant_field_B_vjp_components = []
    dominant_field_B_vjp_coils = []
    if hasattr(Jf.field, "profile_B_vjp"):
        field_B_vjp_profile = Jf.field.profile_B_vjp(host_array(dJ_dB))
        field_B_vjp_s = float(field_B_vjp_profile["wall_time_s"])
        field_B_vjp_component_timings = {
            name: float(elapsed_s)
            for name, elapsed_s in field_B_vjp_profile["component_timings_s"].items()
        }
        dominant_field_B_vjp_components = list(
            field_B_vjp_profile["dominant_components"]
        )
        dominant_field_B_vjp_coils = list(field_B_vjp_profile["dominant_coils"])
    else:
        field_B_vjp_s, _ = time_stage2_callback_result(
            lambda: Jf.field.B_vjp(host_array(dJ_dB))
        )
    timings = {
        "field_B_for_J_s": field_B_for_J_s,
        "integral_only_s": integral_only_s,
        "field_B_for_dJ_s": field_B_for_dJ_s,
        "integral_value_grad_s": integral_value_grad_s,
        "field_B_vjp_s": field_B_vjp_s,
    }
    total_s, dominant = build_stage2_profile_breakdown(timings)
    return (
        timings,
        total_s,
        dominant,
        field_B_vjp_component_timings,
        dominant_field_B_vjp_components,
        dominant_field_B_vjp_coils,
    )


@dataclass(frozen=True)
class Stage2ObjectiveContext:
    """Stage 2 objective state plus the composite root used for DOF projection."""

    # `JF` is the composite/root Optimizable whose DOF layout defines the
    # gradient space for all explicit Stage 2 term projections.
    JF: object
    new_bs: object
    new_surf: object
    Jf: object
    Jls: object
    Jccdist: object
    Jc: object
    squared_flux_weight: float
    length_weight: float
    length_target: float
    cc_weight: float
    cc_threshold: float
    curvature_weight: float
    bs_jax: object | None = None


def make_stage2_objective_context(
    JF,
    new_bs,
    new_surf,
    Jf,
    Jls,
    Jccdist,
    Jc,
    squared_flux_weight,
    length_weight,
    length_target,
    cc_weight,
    cc_threshold,
    curvature_weight,
    *,
    bs_jax=None,
):
    return Stage2ObjectiveContext(
        JF,
        new_bs,
        new_surf,
        Jf,
        Jls,
        Jccdist,
        Jc,
        float(squared_flux_weight),
        float(length_weight),
        float(length_target),
        float(cc_weight),
        float(cc_threshold),
        float(curvature_weight),
        bs_jax,
    )


def compute_stage2_field_diagnostics(
    new_bs,
    new_surf,
    *,
    bs_jax=None,
):
    field_bs = resolve_stage2_field_bs(new_bs, bs_jax)
    return {
        "mean_abs_relBfinal_norm": compute_mean_abs_relbn(new_surf, field_bs),
    }


def compute_stage2_length_penalty_value(curve_length, length_target):
    return 0.5 * max(host_float(curve_length) - host_float(length_target), 0.0) ** 2


def compute_stage2_max_curvature_value(Jc):
    banana_curve = getattr(Jc, "curve", None)
    if banana_curve is not None:
        return _host_curve_max_curvature(banana_curve)
    return host_float(Jc.J())


def stage2_curvature_within_threshold(curvature: float, threshold: float) -> bool:
    return host_float(curvature) <= host_float(threshold)


def compute_stage2_term_gradient(term, root_objective):
    return host_array(term.dJ(partials=True)(root_objective))


def _stage2_term_payload_entry(weight, raw_value, raw_grad):
    raw_grad_array = host_array(raw_grad)
    weight_float = host_float(weight)
    raw_value_float = host_float(raw_value)
    weighted_grad = weight_float * raw_grad_array
    return {
        "weight": weight_float,
        "raw_J": raw_value_float,
        "J": weight_float * raw_value_float,
        "dJ": weighted_grad,
        "grad_norm": float(np.linalg.norm(weighted_grad)),
    }


def _serialize_stage2_term_payload(entries):
    return {
        name: {
            "weight": float(entry["weight"]),
            "raw_J": float(entry["raw_J"]),
            "J": float(entry["J"]),
            "dJ": np.asarray(entry["dJ"], dtype=float).tolist(),
            "grad_norm": float(entry["grad_norm"]),
        }
        for name, entry in entries.items()
    }


def _build_stage2_explicit_term_payload(context):
    squared_flux_grad = compute_stage2_term_gradient(context.Jf, context.JF)
    squared_flux = host_float(context.Jf.J())
    curve_length = host_float(context.Jls.J())
    length_penalty = compute_stage2_length_penalty_value(
        curve_length,
        context.length_target,
    )
    length_grad = compute_stage2_length_penalty_gradient(
        curve_length,
        context.length_target,
        context.Jls,
        context.JF,
    )
    coil_distance_penalty = host_float(context.Jccdist.J())
    coil_distance_grad = compute_stage2_term_gradient(context.Jccdist, context.JF)
    curvature_penalty = host_float(context.Jc.J())
    curvature = compute_stage2_max_curvature_value(context.Jc)
    curvature_grad = compute_stage2_term_gradient(context.Jc, context.JF)
    entries = {
        "squared_flux": _stage2_term_payload_entry(
            context.squared_flux_weight,
            squared_flux,
            squared_flux_grad,
        ),
        "length_penalty": _stage2_term_payload_entry(
            context.length_weight,
            length_penalty,
            length_grad,
        ),
        "coil_distance_penalty": _stage2_term_payload_entry(
            context.cc_weight,
            coil_distance_penalty,
            coil_distance_grad,
        ),
        "curvature_penalty": _stage2_term_payload_entry(
            context.curvature_weight,
            curvature_penalty,
            curvature_grad,
        ),
    }
    return entries, curve_length, curvature


def _build_stage2_target_term_payload(target_objective_bundle, dofs):
    raw_terms = getattr(target_objective_bundle, "raw_terms", None)
    terms = getattr(target_objective_bundle, "terms", ())
    if raw_terms is None or not terms:
        return None
    dofs64 = host_array(dofs)
    dofs_jax = jax.device_put(dofs64)
    raw_values = host_array(raw_terms(dofs_jax))
    raw_gradients = host_array(_cached_raw_terms_jacobian(raw_terms)(dofs_jax))
    entries = {}
    for index, term in enumerate(terms):
        entries[term.name] = _stage2_term_payload_entry(
            term.weight,
            raw_values[index],
            raw_gradients[index],
        )
    return _serialize_stage2_term_payload(entries)


def _build_stage2_target_sharding_payload(target_objective_bundle, dofs):
    field_summary_fn = getattr(target_objective_bundle, "field_sharding_summary", None)
    pairwise_summary_fn = getattr(
        target_objective_bundle,
        "pairwise_penalty_sharding_summary",
        None,
    )
    summaries = {}
    if callable(field_summary_fn):
        summaries["field"] = field_summary_fn(dofs)
    if callable(pairwise_summary_fn):
        summaries["pairwise_penalty"] = pairwise_summary_fn(dofs)
    return summaries or None


def compute_stage2_length_penalty_gradient(
    curve_length,
    length_target,
    Jls,
    root_objective,
):
    active_diff = max(host_float(curve_length) - host_float(length_target), 0.0)
    return active_diff * compute_stage2_term_gradient(Jls, root_objective)


@dataclass(frozen=True)
class Stage2DistanceConstraintState:
    coil_coil_distance: float
    coil_distance_penalty: float
    cc_threshold: float
    violated: bool


def compute_stage2_distance_constraint_state(
    context,
    *,
    term_entries,
):
    """Return the Stage 2 coil-distance state for the current objective eval.

    This state is recomputed on every trajectory/objective snapshot, regardless
    of whether field diagnostics were reused or skipped by stride policy. The
    hard minimum-distance gate must never depend on diagnostic caching.
    """
    coil_coil_distance = host_float(context.Jccdist.shortest_distance())
    coil_distance_penalty = host_float(term_entries["coil_distance_penalty"]["raw_J"])
    cc_threshold = host_float(context.cc_threshold)
    violated = coil_coil_distance <= cc_threshold or not np.isfinite(
        coil_distance_penalty
    )
    return Stage2DistanceConstraintState(
        coil_coil_distance=coil_coil_distance,
        coil_distance_penalty=coil_distance_penalty,
        cc_threshold=cc_threshold,
        violated=bool(violated),
    )


@dataclass(frozen=True)
class Stage2FeasiblePartial:
    dofs: np.ndarray
    objective: float
    curve_length: float
    coil_coil_distance: float
    max_curvature: float
    accepted_index: int


@dataclass(frozen=True)
class Stage2ExactHardwarePass:
    dofs: np.ndarray
    field_objective: float
    source: str


def _stage2_feasible_partial_sort_key(partial: Stage2FeasiblePartial):
    return (
        float(partial.objective),
        float(partial.curve_length),
        -float(partial.coil_coil_distance),
        float(partial.max_curvature),
        int(partial.accepted_index),
    )


def select_better_stage2_exact_hardware_pass(
    current: Stage2ExactHardwarePass | None,
    candidate: Stage2ExactHardwarePass,
) -> Stage2ExactHardwarePass:
    if current is None or float(candidate.field_objective) < float(
        current.field_objective
    ):
        return candidate
    return current


def capture_stage2_exact_hardware_pass_candidate(
    artifact_state: dict[str, object],
    *,
    source: str,
) -> Stage2ExactHardwarePass | None:
    hardware_status = artifact_state["hardware_status"]
    assert isinstance(hardware_status, dict)
    if not hardware_status["success"]:
        return None
    return Stage2ExactHardwarePass(
        dofs=np.asarray(artifact_state["x"], dtype=float),
        field_objective=float(artifact_state["field_objective"]),
        source=str(source),
    )


def select_better_stage2_feasible_partial(
    current: Stage2FeasiblePartial | None,
    candidate: Stage2FeasiblePartial,
) -> Stage2FeasiblePartial:
    if current is None:
        return candidate
    if _stage2_feasible_partial_sort_key(candidate) < _stage2_feasible_partial_sort_key(
        current
    ):
        return candidate
    return current


def capture_stage2_feasible_partial_candidate(
    JF,
    Jls,
    Jccdist,
    banana_curve,
    length_target,
    cc_threshold,
    curvature_threshold,
    *,
    Jcsdist=None,
    coil_surface_threshold=None,
    plasma_vessel_min_dist=None,
    plasma_vessel_threshold=None,
    banana_current_A=None,
    banana_current_threshold=None,
    tf_current_A=None,
    tf_current_threshold=None,
    accepted_index,
):
    objective = host_float(JF.J())
    curve_length = host_float(Jls.J())
    coil_coil_distance = host_float(Jccdist.shortest_distance())
    max_curvature = _host_curve_max_curvature(banana_curve)
    curve_surface_min_dist = _optional_host_float(
        None if Jcsdist is None else Jcsdist.shortest_distance()
    )
    banana_current_A_host = _optional_host_float(banana_current_A)
    tf_current_A_host = _optional_host_float(tf_current_A)
    hardware_status = _evaluate_stage2_artifact_hardware_status(
        banana_curve=banana_curve,
        coil_length=curve_length,
        length_target=length_target,
        curve_curve_min_dist=coil_coil_distance,
        cc_threshold=cc_threshold,
        max_curvature=max_curvature,
        curvature_threshold=curvature_threshold,
        curve_surface_min_dist=curve_surface_min_dist,
        coil_surface_threshold=coil_surface_threshold,
        plasma_vessel_min_dist=plasma_vessel_min_dist,
        plasma_vessel_threshold=plasma_vessel_threshold,
        banana_current_A=banana_current_A_host,
        banana_current_threshold=banana_current_threshold,
        tf_current_A=tf_current_A_host,
        tf_current_threshold=tf_current_threshold,
    )
    if not hardware_status["success"]:
        return None, hardware_status
    return (
        Stage2FeasiblePartial(
            dofs=_host_dofs_copy(JF.x),
            objective=objective,
            curve_length=curve_length,
            coil_coil_distance=coil_coil_distance,
            max_curvature=max_curvature,
            accepted_index=int(accepted_index),
        ),
        hardware_status,
    )


def restore_stage2_exact_hardware_pass_for_artifact_output(
    best_exact_hardware_pass: Stage2ExactHardwarePass | None,
    final_artifact_state: dict[str, object],
    *,
    optimizer_success: bool,
    termination_message: str | None,
) -> tuple[np.ndarray | None, bool, str | None]:
    hardware_status = final_artifact_state["hardware_status"]
    assert isinstance(hardware_status, dict)
    if best_exact_hardware_pass is None or hardware_status["success"]:
        return None, bool(optimizer_success), termination_message
    if termination_message:
        termination_message = (
            f"{termination_message}; restored_best_exact_hardware_pass"
        )
    else:
        termination_message = "restored_best_exact_hardware_pass"
    return (
        np.asarray(best_exact_hardware_pass.dofs, dtype=float),
        False,
        termination_message,
    )


def should_restore_stage2_feasible_partial(
    best_partial: Stage2FeasiblePartial | None,
    final_partial: Stage2FeasiblePartial | None,
    *,
    optimizer_success: bool,
) -> bool:
    if optimizer_success or best_partial is None:
        return False
    if final_partial is None:
        return True
    return _stage2_feasible_partial_sort_key(best_partial) < _stage2_feasible_partial_sort_key(
        final_partial
    )


def evaluate_stage2_objective(
    context,
    *,
    diagnostics=None,
    recompute_diagnostics=True,
):
    """Return composite objective diagnostics using the currently loaded DOFs."""
    term_entries, curve_length, curvature = _build_stage2_explicit_term_payload(context)
    grad = sum(np.asarray(entry["dJ"], dtype=float) for entry in term_entries.values())
    J = sum(host_float(entry["J"]) for entry in term_entries.values())
    squared_flux = host_float(term_entries["squared_flux"]["raw_J"])
    if recompute_diagnostics or diagnostics is None:
        diagnostics = compute_stage2_field_diagnostics(
            context.new_bs,
            context.new_surf,
            bs_jax=context.bs_jax,
        )
    distance_state = compute_stage2_distance_constraint_state(
        context,
        term_entries=term_entries,
    )
    diagnostics["coil_coil_distance"] = distance_state.coil_coil_distance
    snapshot = {
        "J": J,
        "Jf": squared_flux,
        "mean_abs_relBfinal_norm": host_float(diagnostics["mean_abs_relBfinal_norm"]),
        "curve_length": curve_length,
        "coil_coil_distance": distance_state.coil_coil_distance,
        "curvature": curvature,
        "grad_norm": float(np.linalg.norm(grad)),
        "distance_constraint_violated": distance_state.violated,
    }
    return snapshot, grad, diagnostics


def profile_stage2_explicit_step(
    context,
):
    """Return a one-step timing breakdown for the explicit Stage 2 objective."""
    objective_path_timings = {
        "JF_J_s": time_stage2_callback(context.JF.J),
        "JF_dJ_s": time_stage2_callback(context.JF.dJ),
    }

    start = time.perf_counter()
    snapshot, gradient, _ = evaluate_stage2_objective(
        context,
    )
    observed_step_total_s = float(time.perf_counter() - start)

    field_bs = resolve_stage2_field_bs(context.new_bs, context.bs_jax)
    extra_diagnostic_callbacks = {
        "Jf_J_s": lambda: host_float(context.Jf.J()),
        "mean_abs_relBfinal_norm_s": lambda: compute_mean_abs_relbn(
            context.new_surf, field_bs
        ),
        "curve_length_s": lambda: host_float(context.Jls.J()),
        "coil_coil_distance_s": lambda: host_float(context.Jccdist.shortest_distance()),
        "curvature_s": lambda: compute_stage2_max_curvature_value(context.Jc),
    }
    extra_diagnostic_timings = profile_stage2_named_callbacks(
        extra_diagnostic_callbacks
    )
    term_profile = profile_stage2_objective_terms(context)

    extra_diagnostic_total_s, dominant_extra_diagnostics = (
        build_stage2_profile_breakdown(extra_diagnostic_timings)
    )
    (
        squared_flux_internal_timings,
        squared_flux_internal_total_s,
        dominant_squared_flux_internal_components,
        squared_flux_field_b_vjp_component_timings,
        dominant_squared_flux_field_b_vjp_components,
        dominant_squared_flux_field_b_vjp_coils,
    ) = profile_stage2_squared_flux_internal_components(context.Jf)
    return {
        "objective_path_timings_s": objective_path_timings,
        "observed_step_total_s": observed_step_total_s,
        "extra_diagnostic_timings_s": extra_diagnostic_timings,
        "extra_diagnostic_total_s": extra_diagnostic_total_s,
        "dominant_extra_diagnostics": dominant_extra_diagnostics,
        "objective_term_value_timings_s": term_profile["value_timings_s"],
        "objective_term_value_total_s": term_profile["value_total_s"],
        "dominant_objective_value_terms": term_profile["dominant_value_terms"],
        "objective_term_gradient_timings_s": term_profile["gradient_timings_s"],
        "objective_term_gradient_total_s": term_profile["gradient_total_s"],
        "dominant_objective_gradient_terms": term_profile["dominant_gradient_terms"],
        "squared_flux_internal_timings_s": squared_flux_internal_timings,
        "squared_flux_internal_total_s": squared_flux_internal_total_s,
        "dominant_squared_flux_internal_components": dominant_squared_flux_internal_components,
        "squared_flux_field_b_vjp_component_timings_s": squared_flux_field_b_vjp_component_timings,
        "dominant_squared_flux_field_b_vjp_components": dominant_squared_flux_field_b_vjp_components,
        "dominant_squared_flux_field_b_vjp_coils": dominant_squared_flux_field_b_vjp_coils,
        "snapshot": snapshot,
    }


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def _smoothmax_selected(
    values: np.ndarray,
    temperature: float,
) -> tuple[float, np.ndarray]:
    temperature = max(float(temperature), _SMOOTHING_EPS)
    max_value = float(np.max(values))
    shifted = (np.asarray(values, dtype=float) - max_value) / temperature
    weights = _stable_softmax(shifted)
    smooth_value = max_value + temperature * float(np.log(np.sum(np.exp(shifted))))
    return smooth_value, weights


def _smoothmin_selected(
    values: np.ndarray,
    temperature: float,
) -> tuple[float, np.ndarray]:
    temperature = max(float(temperature), _SMOOTHING_EPS)
    min_value = float(np.min(values))
    shifted = -(np.asarray(values, dtype=float) - min_value) / temperature
    weights = _stable_softmax(shifted)
    smooth_value = min_value - temperature * float(np.log(np.sum(np.exp(shifted))))
    return smooth_value, weights


def stage2_constraint_activity_tolerances(
    distance_smoothing: float,
    curvature_smoothing: float,
) -> list[float]:
    return [
        1e-3,
        1.0 / max(float(distance_smoothing), _SMOOTHING_EPS),
        1.0 / max(float(curvature_smoothing), _SMOOTHING_EPS),
    ]


def smooth_max_curvature_signed_constraint(
    curve,
    threshold: float,
    temperature: float,
    base_objective_optimizable,
):
    kappa = np.asarray(curve.kappa(), dtype=float)
    hard_max = float(np.max(kappa))
    active_mask = kappa >= (hard_max - 4.0 * float(temperature))
    if not np.any(active_mask):
        active_mask[np.argmax(kappa)] = True
    smooth_max, active_weights = _smoothmax_selected(kappa[active_mask], temperature)
    full_weights = np.zeros_like(kappa)
    full_weights[active_mask] = active_weights
    grad = np.asarray(
        curve.dkappa_by_dcoeff_vjp(full_weights)(base_objective_optimizable),
        dtype=float,
    )
    return smooth_max - float(threshold), grad


def smooth_min_distance_signed_constraint(
    curves,
    minimum_distance: float,
    temperature: float,
    base_objective_optimizable,
):
    from simsopt._core.derivative import Derivative

    pair_blocks = []
    hard_min = np.inf
    for curve_index, curve_i in enumerate(curves):
        gamma_i = np.asarray(curve_i.gamma(), dtype=float)
        for other_index in range(curve_index):
            curve_j = curves[other_index]
            gamma_j = np.asarray(curve_j.gamma(), dtype=float)
            diffs = gamma_i[:, None, :] - gamma_j[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            hard_min = min(hard_min, float(np.min(dists)))
            pair_blocks.append((curve_index, other_index, diffs, dists))

    if not pair_blocks:
        return float(minimum_distance), _zero_gradient_like(
            base_objective_optimizable.x
        )

    selection_window = 4.0 * float(temperature)
    selected_distances = []
    selected_entries = []
    for curve_index, other_index, diffs, dists in pair_blocks:
        mask = dists <= (hard_min + selection_window)
        if not np.any(mask):
            mask[np.unravel_index(np.argmin(dists), dists.shape)] = True
        rows, cols = np.nonzero(mask)
        selected_distances.append(dists[rows, cols])
        selected_entries.append(
            (
                curve_index,
                other_index,
                rows,
                cols,
                diffs[rows, cols],
                dists[rows, cols],
            )
        )

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = _smoothmin_selected(flat_distances, temperature)

    point_gradients = [
        np.zeros_like(np.asarray(curve.gamma(), dtype=float))
        for curve in curves
    ]
    offset = 0
    for curve_index, other_index, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset : offset + count]
        offset += count
        directions = diffs / np.maximum(distances[:, None], _SMOOTHING_EPS)
        np.add.at(point_gradients[curve_index], rows, local_weights[:, None] * directions)
        np.add.at(
            point_gradients[other_index],
            cols,
            -local_weights[:, None] * directions,
        )

    derivative = Derivative({})
    for curve, point_gradient in zip(curves, point_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    grad = np.asarray(derivative(base_objective_optimizable), dtype=float)
    # Negate: constraint value is (minimum_distance - smooth_min), so its
    # gradient w.r.t. DOFs is -∂smooth_min/∂DOFs.
    return float(minimum_distance) - smooth_min, -grad


# ALM WIP parking lot:
# The helper functions below are kept as a local recovery point, but the
# parser/runtime no longer wires them into the live Stage 2 entry path.
def build_stage2_alm_settings(args):
    return ALMSettings(  # noqa: F821 — parked ALM import
        max_outer_iterations=args.alm_max_outer_iters,
        max_subproblem_continuations=args.alm_max_subproblem_continuations,
        penalty_init=args.alm_penalty_init,
        penalty_scale=args.alm_penalty_scale,
        feasibility_tol=args.alm_feas_tol,
        stationarity_tol=args.alm_stationarity_tol,
        trust_radius_init=(
            None
            if float(args.alm_trust_radius_init) == 0.0
            else float(args.alm_trust_radius_init)
        ),
        trust_radius_min=args.alm_trust_radius_min,
        trust_radius_shrink=args.alm_trust_radius_shrink,
        trust_radius_grow=args.alm_trust_radius_grow,
        max_inner_attempts=args.alm_max_inner_attempts,
    )


def evaluate_stage2_alm_problem(
    dofs,
    base_objective,
    new_bs,
    new_surf,
    Jf,
    Jls,
    length_target,
    Jccdist,
    Jc,
    distance_smoothing,
    curvature_smoothing,
    multipliers,
    penalty,
):
    base_objective.x = dofs
    base_value = float(base_objective.J())
    base_grad = np.asarray(base_objective.dJ(), dtype=float)

    coil_length = float(Jls.J())
    length_violation = upper_bound_residual(coil_length, length_target)  # noqa: F821
    length_grad = np.asarray(
        Jls.dJ(partials=True)(base_objective),
        dtype=float,
    )

    curve_curve_min_dist = float(Jccdist.shortest_distance())
    curve_curve_violation = lower_bound_residual(  # noqa: F821
        curve_curve_min_dist,
        Jccdist.minimum_distance,
    )
    curve_curve_signed_value, curve_curve_grad = smooth_min_distance_signed_constraint(
        Jccdist.curves,
        Jccdist.minimum_distance,
        distance_smoothing,
        base_objective,
    )

    max_curvature = float(np.max(Jc.curve.kappa()))
    curvature_violation = upper_bound_residual(max_curvature, Jc.threshold)  # noqa: F821
    curvature_signed_value, curvature_grad = smooth_max_curvature_signed_constraint(
        Jc.curve,
        Jc.threshold,
        curvature_smoothing,
        base_objective,
    )

    evaluation = augmented_inequality_objective(  # noqa: F821
        base_value,
        base_grad,
        [
            coil_length - length_target,
            curve_curve_signed_value,
            curvature_signed_value,
        ],
        [length_grad, curve_curve_grad, curvature_grad],
        multipliers,
        penalty,
    )
    evaluation.update(
        {
            "base_value": base_value,
            "constraint_names": list(STAGE2_ALM_CONSTRAINT_NAMES),  # noqa: F821
            "dual_update_values": [
                coil_length - length_target,
                curve_curve_signed_value,
                curvature_signed_value,
            ],
            "constraint_grads": [length_grad, curve_curve_grad, curvature_grad],
            "constraint_activity_tolerances": stage2_constraint_activity_tolerances(
                distance_smoothing,
                curvature_smoothing,
            ),
            "feasibility_values": [
                length_violation,
                curve_curve_violation,
                curvature_violation,
            ],
            "max_feasibility_violation": max(
                length_violation,
                curve_curve_violation,
                curvature_violation,
            ),
            "metric_grad": base_grad,
            "metric_stationarity_norm": float(np.linalg.norm(base_grad)),
        }
    )

    unitn = new_surf.unitnormal()
    BdotN = np.mean(
        np.abs(np.sum(new_bs.B().reshape(unitn.shape) * unitn, axis=2))
    )
    outstr = (
        f"ALM J={evaluation['total']:.1e}, Jflux={base_value:.1e}, "
        f"Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}"
    )
    outstr += (
        f", Len={coil_length:.1f}m, Len+={length_violation:.2e}, "
        f"Lens={coil_length - length_target:.2e}"
    )
    outstr += (
        f", C-C-Sep={curve_curve_min_dist:.2f}m, CC+={curve_curve_violation:.2e}, "
        f"CCs={curve_curve_signed_value:.2e}"
    )
    outstr += (
        f", Curvature={max_curvature:.2f}, Curv+={curvature_violation:.2e}, "
        f"Curvs={curvature_signed_value:.2e}"
    )
    outstr += (
        f", ║∇L_A║={evaluation['stationarity_norm']:.1e}, "
        f"║∇Jflux║={np.linalg.norm(base_grad):.1e}, mu={penalty:.1e}"
    )
    print(outstr)
    return evaluation


_STAGE2_COMPONENT_LABEL = "the Stage 2 outer loop"


def resolve_stage2_optimizer_contract(
    field_backend,
    optimizer_backend,
    *,
    least_squares_algorithm="quasi-newton",
):
    """Resolve the optimizer contract for the Stage 2 outer loop."""
    from simsopt.geo.optimizer_jax import (
        resolve_reference_outer_loop_optimizer_contract,
        resolve_target_outer_loop_optimizer_contract,
    )

    if least_squares_algorithm == "lm":
        return resolve_target_outer_loop_optimizer_contract(
            field_backend,
            optimizer_backend,
            component_label=_STAGE2_COMPONENT_LABEL,
            least_squares_algorithm=least_squares_algorithm,
        )

    if field_backend == "jax":
        return resolve_target_outer_loop_optimizer_contract(
            field_backend,
            optimizer_backend,
            component_label=_STAGE2_COMPONENT_LABEL,
            least_squares_algorithm=least_squares_algorithm,
        )
    return resolve_reference_outer_loop_optimizer_contract(
        field_backend,
        optimizer_backend,
        component_label=_STAGE2_COMPONENT_LABEL,
    )


def resolve_stage2_optimizer_method(
    field_backend,
    optimizer_backend,
    *,
    least_squares_algorithm="quasi-newton",
):
    """Resolve the shared optimizer substrate for the Stage 2 outer loop."""
    return resolve_stage2_optimizer_contract(
        field_backend,
        optimizer_backend,
        least_squares_algorithm=least_squares_algorithm,
    ).method


def should_build_stage2_target_objective(
    field_backend,
    optimizer_backend,
    *,
    least_squares_algorithm="quasi-newton",
):
    """Return whether the JAX Stage 2 target objective should drive optimization."""
    from simsopt.geo.optimizer_jax import TargetOptimizerContract

    contract = resolve_stage2_optimizer_contract(
        field_backend,
        optimizer_backend,
        least_squares_algorithm=least_squares_algorithm,
    )
    return isinstance(contract, TargetOptimizerContract)


def resolve_stage2_target_lane_requirements(
    field_backend,
    optimizer_backend,
    *,
    least_squares_algorithm,
    probe_only,
    export_objective_json,
):
    """Resolve target-lane requirements before building explicit objectives."""
    needs_target_probe_payload = (
        export_objective_json is not None and optimizer_backend == "ondevice"
    )
    probe_only_target_payload = (
        probe_only and needs_target_probe_payload and field_backend != "jax"
    )
    outer_contract = None
    use_target_objective_lane = False
    if not probe_only_target_payload:
        outer_contract = resolve_stage2_optimizer_contract(
            field_backend,
            optimizer_backend,
            least_squares_algorithm=least_squares_algorithm,
        )
        use_target_objective_lane = should_build_stage2_target_objective(
            field_backend,
            optimizer_backend,
            least_squares_algorithm=least_squares_algorithm,
        )
    return (
        outer_contract,
        use_target_objective_lane,
        needs_target_probe_payload,
        probe_only_target_payload,
    )


def validate_stage2_target_objective_dof_layout(
    target_objective_bundle,
    dofs,
):
    if target_objective_bundle.expected_dof_count != dofs.size:
        raise RuntimeError(STAGE2_TARGET_OBJECTIVE_DOF_LAYOUT_ERROR)


def resolve_stage2_target_value_and_grad(target_objective_bundle):
    if target_objective_bundle is None:
        return None
    if target_objective_bundle.value_and_grad is not None:
        return target_objective_bundle.value_and_grad
    return jax.jit(jax.value_and_grad(target_objective_bundle.objective))


def resolve_stage2_target_least_squares_residual(target_objective_bundle):
    if target_objective_bundle is None:
        return None
    return getattr(target_objective_bundle, "least_squares_residual", None)


def build_stage2_target_optimizer_state(target_objective_bundle, dofs):
    from simsopt.objectives.stage2_target_objective_jax import (
        stage2_target_optimizer_state_from_dofs,
    )

    curve_dof_count = int(target_objective_bundle.expected_dof_count) - 1
    return stage2_target_optimizer_state_from_dofs(
        dofs,
        curve_dof_count=curve_dof_count,
    )


def flatten_stage2_target_optimizer_state(dofs):
    from simsopt.objectives.stage2_target_objective_jax import (
        Stage2TargetOptimizerState,
        stage2_target_optimizer_state_to_dofs,
    )

    if isinstance(dofs, Stage2TargetOptimizerState):
        return host_array(stage2_target_optimizer_state_to_dofs(dofs))
    return host_array(dofs)


def resolve_stage2_field_diagnostic_stride(args):
    """Return the effective field-diagnostic refresh stride for the active lane."""
    requested_stride = int(args.field_diagnostic_stride)
    if requested_stride > 0:
        return requested_stride
    return 1


def should_recompute_stage2_field_diagnostics(
    diagnostics,
    *,
    eval_index,
    stride,
):
    return (
        diagnostics is None
        or stride <= 1
        or eval_index == 1
        or eval_index % stride == 0
    )


def plan_stage2_field_diagnostic_evaluation(
    field_diagnostic_state,
    *,
    stride,
):
    next_eval = field_diagnostic_state["eval_count"] + 1
    diagnostics = field_diagnostic_state["diagnostics"]
    return (
        next_eval,
        diagnostics,
        should_recompute_stage2_field_diagnostics(
            diagnostics,
            eval_index=next_eval,
            stride=stride,
        ),
    )


def store_stage2_field_diagnostics(
    field_diagnostic_state,
    *,
    eval_index,
    diagnostics,
):
    field_diagnostic_state["eval_count"] = int(eval_index)
    field_diagnostic_state["diagnostics"] = (
        None if diagnostics is None else dict(diagnostics)
    )


def run_stage2_optimizer(
    value_and_grad_fun=None,
    dofs=None,
    *,
    contract,
    maxiter,
    ftol,
    gtol,
    maxcor=300,
    scalar_fun=None,
    residual_fun=None,
    callback=None,
    progress_callback=None,
    failure_callback=None,
):
    """Run the Stage 2 outer optimization through the lane-specific substrate."""
    from simsopt.geo.optimizer_jax import (
        ReferenceOptimizerContract,
        TargetOptimizerContract,
        reference_minimize,
        target_least_squares,
        target_minimize,
    )

    use_explicit_value_and_grad = value_and_grad_fun is not None
    if isinstance(contract, TargetOptimizerContract) and contract.use_least_squares_objective:
        if residual_fun is None:
            raise RuntimeError(
                "Stage 2 LM optimization requires an explicit residual-vector objective."
            )
        if failure_callback is not None:
            raise ValueError(
                "Stage 2 target-lane LM optimization does not support "
                "failure_callback; line-search failure diagnostics are only "
                "available for the lbfgs-ondevice target lane."
            )
        return target_least_squares(
            residual_fun,
            dofs,
            method=contract.method,
            tol=gtol,
            maxiter=maxiter,
            callback=callback,
            progress_callback=progress_callback,
        )
    if isinstance(contract, TargetOptimizerContract):
        if scalar_fun is None and not use_explicit_value_and_grad:
            raise RuntimeError(
                "Stage 2 target-lane optimization requires a JAX target objective."
            )
        objective_fun = (
            value_and_grad_fun if use_explicit_value_and_grad else scalar_fun
        )
        return target_minimize(
            objective_fun,
            dofs,
            method=contract.method,
            tol=gtol,
            maxiter=maxiter,
            options={
                "maxcor": int(maxcor),
                "ftol": float(ftol),
            },
            value_and_grad=use_explicit_value_and_grad,
            callback=callback,
            progress_callback=progress_callback,
            failure_callback=failure_callback,
        )
    if not isinstance(contract, ReferenceOptimizerContract):
        raise RuntimeError(
            f"Unsupported Stage 2 optimizer contract {type(contract)!r}."
        )
    if not use_explicit_value_and_grad:
        raise RuntimeError(
            "Stage 2 reference-lane optimization requires an explicit "
            "value-and-gradient objective."
        )
    if failure_callback is not None:
        raise ValueError(
            "Stage 2 reference-lane optimization does not support "
            "failure_callback; use the target ondevice lane for "
            "line-search failure diagnostics."
        )
    return reference_minimize(
        value_and_grad_fun,
        dofs,
        method=contract.method,
        tol=gtol,
        maxiter=maxiter,
        options={
            "maxcor": int(maxcor),
            "ftol": float(ftol),
        },
        value_and_grad=True,
        callback=callback,
        progress_callback=progress_callback,
    )


def run_stage2_optimizer_timed(
    value_and_grad_fun=None,
    dofs=None,
    *,
    contract,
    maxiter,
    ftol,
    gtol,
    maxcor=300,
    scalar_fun=None,
    residual_fun=None,
    callback=None,
    progress_callback=None,
    failure_callback=None,
):
    """Run the Stage 2 optimizer and return both result and elapsed time."""
    start = time.perf_counter()
    result = run_stage2_optimizer(
        value_and_grad_fun,
        dofs,
        contract=contract,
        maxiter=maxiter,
        ftol=ftol,
        gtol=gtol,
        maxcor=maxcor,
        scalar_fun=scalar_fun,
        residual_fun=residual_fun,
        callback=callback,
        progress_callback=progress_callback,
        failure_callback=failure_callback,
    )
    return result, float(time.perf_counter() - start)


def run_stage2_alm_optimizer_timed(
    dofs,
    *,
    constraint_names,
    settings,
    evaluate_problem,
    maxiter,
    ftol,
    gtol,
    maxcor=300,
    inner_optimizer_contract=None,
    target_inner_value_and_grad=None,
    accepted_callback=None,
    outer_state_callback=None,
):
    start = time.perf_counter()
    result = minimize_alm(
        dofs,
        constraint_names,
        evaluate_problem,
        settings,
        {
            "maxiter": int(maxiter),
            "maxcor": int(maxcor),
            "ftol": float(ftol),
            "gtol": float(gtol),
        },
        inner_optimizer_contract=inner_optimizer_contract,
        target_inner_value_and_grad=target_inner_value_and_grad,
        accepted_callback=accepted_callback,
        outer_state_callback=outer_state_callback,
    )
    return result, float(time.perf_counter() - start)


def _build_stage2_probe_composite_payload(
    context,
    *,
    target_objective_bundle=None,
):
    """Return the probe composite snapshot/gradient using the active optimizer SSOT."""
    explicit_snapshot, explicit_grad, _ = evaluate_stage2_objective(
        context,
    )
    objective_source = "explicit-composite"
    composite_value = host_float(explicit_snapshot["J"])
    composite_grad = host_array(explicit_grad)
    composite_terms = None
    if target_objective_bundle is not None:
        dofs_jax = jax.device_put(host_array(context.JF.x))
        target_value_and_grad = resolve_stage2_target_value_and_grad(
            target_objective_bundle
        )
        assert target_value_and_grad is not None
        composite_value, composite_grad = target_value_and_grad(dofs_jax)
        composite_value = host_float(composite_value)
        composite_grad = host_array(composite_grad)
        objective_source = "target-objective"
        composite_terms = _build_stage2_target_term_payload(
            target_objective_bundle,
            host_array(context.JF.x),
        )
    return (
        {
            **explicit_snapshot,
            "J": composite_value,
            "grad_norm": float(np.linalg.norm(composite_grad)),
            "objective_source": objective_source,
        },
        composite_grad,
        composite_terms,
    )


def build_stage2_probe_payload(
    JF,
    new_bs,
    new_surf,
    banana_curve,
    Jf,
    Jls,
    Jccdist,
    Jc,
    bs_jax=None,
    *,
    backend,
    optimizer_backend,
    equilibrium_path,
    nphi,
    ntheta,
    squared_flux_weight,
    length_weight,
    length_target,
    cc_weight,
    cc_threshold,
    curvature_weight,
    self_intersection_summary=None,
    target_objective_bundle=None,
):
    """Serialize the initialized Stage 2 objective state for parity probes."""
    dofs = host_array(JF.x)
    context = make_stage2_objective_context(
        JF,
        new_bs,
        new_surf,
        Jf,
        Jls,
        Jccdist,
        Jc,
        squared_flux_weight,
        length_weight,
        length_target,
        cc_weight,
        cc_threshold,
        curvature_weight,
        bs_jax=bs_jax,
    )
    composite_snapshot, composite_grad, composite_terms = (
        _build_stage2_probe_composite_payload(
            context,
            target_objective_bundle=target_objective_bundle,
        )
    )
    sharding_summaries = None
    if target_objective_bundle is not None:
        sharding_summaries = _build_stage2_target_sharding_payload(
            target_objective_bundle,
            dofs,
        )
    flux_grad = host_array(Jf.dJ())
    curvature_threshold = host_float(context.Jc.threshold)
    curvature = host_float(composite_snapshot["curvature"])
    payload = {
        "backend": backend,
        "optimizer_backend": optimizer_backend,
        "banana_curve_class": type(banana_curve).__name__,
        "equilibrium_path": equilibrium_path,
        "nphi": int(nphi),
        "ntheta": int(ntheta),
        "dof_count": int(composite_grad.size),
        "curvature_threshold": curvature_threshold,
        "curvature_within_threshold": stage2_curvature_within_threshold(
            curvature,
            curvature_threshold,
        ),
        "curvature_margin": curvature_threshold - curvature,
        "squared_flux": {
            "J": host_float(Jf.J()),
            "dJ": flux_grad.tolist(),
            "grad_norm": float(np.linalg.norm(flux_grad)),
        },
        "composite": {
            **composite_snapshot,
            "dJ": composite_grad.tolist(),
            **({"terms": composite_terms} if composite_terms is not None else {}),
        },
    }
    if self_intersection_summary is not None:
        payload["self_intersection"] = dict(self_intersection_summary)
    if sharding_summaries is not None:
        payload["sharding_summaries"] = sharding_summaries
    return payload


def write_json_file(path, payload):
    """Write JSON payloads for probe/export workflows."""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(sanitize_json_payload(payload), outfile, indent=2, allow_nan=False)


def _log_taylor_test_summary(label, result):
    LOGGER.info(
        "%s: passed=%s, directional_derivative=%.3e, max_ratio=%s",
        label,
        result["passed"],
        result["directional_derivative"],
        result["max_ratio"],
    )


def build_stage2_problem_contract(
    *,
    plasma_surf_filename,
    file_loc,
    nphi,
    ntheta,
    num_quadpoints,
    order,
    field_diagnostic_stride,
    R0,
    s,
    banana_surf_radius,
    theta_center,
    phi_center,
    theta_width,
    phi_width,
    LENGTH_WEIGHT,
    CC_WEIGHT,
    CURVATURE_WEIGHT,
    SQUARED_FLUX_WEIGHT,
    LENGTH_TARGET,
    CC_THRESHOLD,
    CURVATURE_THRESHOLD,
    args,
    MAXITER,
):
    return {
        "workflow": "stage2-banana-coil-optimization",
        "equilibrium": {
            "filename": plasma_surf_filename,
            "path": file_loc,
        },
        "resolution": {
            "nphi": int(nphi),
            "ntheta": int(ntheta),
            "num_quadpoints": int(num_quadpoints),
            "order": int(order),
            "field_diagnostic_stride": int(field_diagnostic_stride),
        },
        "seed_parameters": {
            "major_radius": float(R0),
            "toroidal_flux": float(s),
            "banana_surface_radius": float(banana_surf_radius),
            "theta_center": float(theta_center),
            "phi_center": float(phi_center),
            "theta_width": float(theta_width),
            "phi_width": float(phi_width),
        },
        "objective_weights": {
            "squared_flux": float(SQUARED_FLUX_WEIGHT),
            "length": float(LENGTH_WEIGHT),
            "coil_coil_distance": float(CC_WEIGHT),
            "curvature": float(CURVATURE_WEIGHT),
        },
        "hardware_thresholds": {
            "length_target": float(LENGTH_TARGET),
            "coil_coil_distance": float(CC_THRESHOLD),
            "coil_plasma_distance": float(COIL_PLASMA_MIN_DIST_M),
            "coil_vessel_clearance": float(COIL_VESSEL_MIN_DIST_M),
            "plasma_vessel_distance": float(PLASMA_VESSEL_MIN_DIST_M),
            "curvature": float(CURVATURE_THRESHOLD),
        },
        "runtime_contract": {
            "field_backend": args.backend,
            "optimizer_backend": args.optimizer_backend,
            "least_squares_algorithm": args.least_squares_algorithm,
            "constraint_method": args.constraint_method,
            "max_iterations": int(MAXITER),
            "init_only": bool(args.init_only),
            "skip_postprocess": bool(args.skip_postprocess),
        },
    }


def build_stage2_results_envelope(
    *,
    output_root,
    plasma_surf_filename,
    file_loc,
    nphi,
    ntheta,
    num_quadpoints,
    order,
    field_diagnostic_stride,
    R0,
    s,
    banana_surf_radius,
    theta_center,
    phi_center,
    theta_width,
    phi_width,
    LENGTH_WEIGHT,
    CC_WEIGHT,
    CURVATURE_WEIGHT,
    SQUARED_FLUX_WEIGHT,
    LENGTH_TARGET,
    CC_THRESHOLD,
    CURVATURE_THRESHOLD,
    args,
    MAXITER,
):
    artifacts = build_artifact_manifest(
        output_root,
        required_files=_STAGE2_REQUIRED_ARTIFACT_FILENAMES,
        planned_files=("results.json",),
    )
    artifacts["policy"] = {
        "skip_postprocess": bool(args.skip_postprocess),
    }
    return {
        "schema_version": _STAGE2_RESULTS_SCHEMA_VERSION,
        "provenance": build_runtime_provenance(
            title="Stage 2 banana coil optimization",
            repo_root=REPO_ROOT,
            script_path=__file__,
            output_root=output_root,
            argv=sys.argv,
            jax_module=jax,
            jaxlib_version=jaxlib.__version__,
        ),
        "artifacts": artifacts,
        "problem_contract": build_stage2_problem_contract(
            plasma_surf_filename=plasma_surf_filename,
            file_loc=file_loc,
            nphi=nphi,
            ntheta=ntheta,
            num_quadpoints=num_quadpoints,
            order=order,
            field_diagnostic_stride=field_diagnostic_stride,
            R0=R0,
            s=s,
            banana_surf_radius=banana_surf_radius,
            theta_center=theta_center,
            phi_center=phi_center,
            theta_width=theta_width,
            phi_width=phi_width,
            LENGTH_WEIGHT=LENGTH_WEIGHT,
            CC_WEIGHT=CC_WEIGHT,
            CURVATURE_WEIGHT=CURVATURE_WEIGHT,
            SQUARED_FLUX_WEIGHT=SQUARED_FLUX_WEIGHT,
            LENGTH_TARGET=LENGTH_TARGET,
            CC_THRESHOLD=CC_THRESHOLD,
            CURVATURE_THRESHOLD=CURVATURE_THRESHOLD,
            args=args,
            MAXITER=MAXITER,
        ),
    }


def load_stage2_override_dofs(path, expected_shape):
    """Load a Stage 2 DOF override vector from JSON and validate its shape."""
    with open(path, "r", encoding="utf-8") as infile:
        payload = json.load(infile)
    dofs = np.asarray(payload, dtype=float)
    if dofs.shape != expected_shape:
        raise ValueError(
            "Override DOF vector shape mismatch: "
            f"expected {expected_shape}, got {dofs.shape}."
        )
    return dofs


def append_stage2_trajectory_snapshot(trajectory_sink, snapshot, *, eval_index=None):
    """Append a Stage 2 diagnostic snapshot when trajectory capture is enabled."""
    if trajectory_sink is None:
        return
    trajectory_sink.append(
        {
            "eval_index": len(trajectory_sink) + 1
            if eval_index is None
            else int(eval_index),
            **snapshot,
        }
    )


def capture_stage2_trajectory_snapshot(
    trajectory_sink,
    JF,
    new_bs,
    new_surf,
    Jf,
    Jls,
    Jccdist,
    Jc,
    squared_flux_weight,
    length_weight,
    length_target,
    cc_weight,
    cc_threshold,
    curvature_weight,
    *,
    bs_jax=None,
):
    """Evaluate and append a Stage 2 diagnostic snapshot when requested."""
    context = make_stage2_objective_context(
        JF,
        new_bs,
        new_surf,
        Jf,
        Jls,
        Jccdist,
        Jc,
        squared_flux_weight,
        length_weight,
        length_target,
        cc_weight,
        cc_threshold,
        curvature_weight,
        bs_jax=bs_jax,
    )
    snapshot, _, _ = evaluate_stage2_objective(
        context,
    )
    append_stage2_trajectory_snapshot(trajectory_sink, snapshot)
    return snapshot


def make_fun(
    JF,
    new_bs,
    new_surf,
    Jf,
    Jls,
    Jccdist,
    Jc,
    squared_flux_weight,
    length_weight,
    length_target,
    cc_weight,
    cc_threshold,
    curvature_weight,
    bs_jax=None,
    trajectory_sink=None,
    field_diagnostic_stride=1,
):
    """Factory for the Stage 2 objective function.

    Returns a closure compatible with ``simsopt.geo.optimizer_jax.jax_minimize``
    that captures all required state explicitly rather than from module scope.

    When *bs_jax* is provided the logging ⟨B·n⟩ diagnostic is computed
    from the JAX field (avoids stale CPU cache and redundant CPU field
    evaluation inside the optimizer loop).
    """
    field_diagnostic_state = {
        "eval_count": 0,
        "diagnostics": None,
    }
    context = make_stage2_objective_context(
        JF,
        new_bs,
        new_surf,
        Jf,
        Jls,
        Jccdist,
        Jc,
        squared_flux_weight,
        length_weight,
        length_target,
        cc_weight,
        cc_threshold,
        curvature_weight,
        bs_jax=bs_jax,
    )

    def fun(dofs):
        next_eval, diagnostics, recompute_diagnostics = (
            plan_stage2_field_diagnostic_evaluation(
                field_diagnostic_state,
                stride=field_diagnostic_stride,
            )
        )
        JF.x = dofs
        snapshot, grad, diagnostics = evaluate_stage2_objective(
            context,
            diagnostics=diagnostics,
            recompute_diagnostics=recompute_diagnostics,
        )
        store_stage2_field_diagnostics(
            field_diagnostic_state,
            eval_index=next_eval,
            diagnostics=diagnostics,
        )
        append_stage2_trajectory_snapshot(
            trajectory_sink,
            snapshot,
            eval_index=next_eval,
        )
        outstr = f"J={snapshot['J']:.1e}, Jf={snapshot['Jf']:.1e}, ⟨B·n⟩={snapshot['mean_abs_relBfinal_norm']:.1e}"
        outstr += f", Len={snapshot['curve_length']:.1f}m"
        outstr += f", C-C-Sep={snapshot['coil_coil_distance']:.2f}m"
        outstr += f", Curvature={snapshot['curvature']:.2f}"
        outstr += f", ║∇J║={snapshot['grad_norm']:.1e}"
        print(outstr)
        return snapshot["J"], grad

    return fun


if __name__ == "__main__":
    # PRE-INITIALIZATION
    # ---------------------------------------------------------------------------------------
    args = parse_args()
    args.curvature_threshold = resolve_curvature_threshold(args.curvature_threshold)

    # Deferred imports — simsopt modules require simsoptpp (C++ extension)
    # for coil/surface setup.  Placing them after arg parsing lets --help work
    # even when simsoptpp is not installed.
    from simsopt._core.optimizable import load
    from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
    from simsopt.field.coil import ScaledCurrent
    from simsopt.geo import (
        SurfaceRZFourier,
        curves_to_vtk,
        create_equally_spaced_curves,
        CurveLength,
        CurveCurveDistance,
        CurveSurfaceDistance,
        LpCurveCurvature,
        CurveCWSFourierCPP,
    )
    from simsopt.objectives import SquaredFlux, QuadraticPenalty
    from simsopt.objectives.stage2_target_objective_jax import (
        Stage2PenaltyConfig,
        build_stage2_target_objective,
    )
    from plotting_utils import (
        norm_field_plot,
        magnitude_field_plot,
        cross_section_plot,
    )

    # File for the desired boundary magnetic surface:
    plasma_surf_filename = args.plasma_surf_filename
    file_loc = build_equilibrium_path(args)

    # Make Directory for output
    OUT_DIR = os.path.join(args.output_root, f"outputs-{plasma_surf_filename}") + "/"
    os.makedirs(OUT_DIR, exist_ok=True)

    nphi = args.nphi
    ntheta = args.ntheta

    # Create the TF coils in HBT - these will be fixed but create background toroidal field:
    tf_curves = create_equally_spaced_curves(
        20,
        1,
        stellsym=False,
        R0=VACUUM_VESSEL_MAJOR_RADIUS_M,
        R1=0.4,
        order=1,
    )
    tf_current_A = validate_tf_current_limit(args.tf_current_A)
    tf_currents = [Current(1.0) * tf_current_A for _ in range(20)]

    # All the TF degrees of freedom are fixed
    for tf_curve in tf_curves:
        tf_curve.fix_all()
    for tf_current in tf_currents:
        tf_current.fix_all()

    tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]

    # INITIALIZATION FOR BANANA COILS
    # ---------------------------------------------------------------------------------------
    # Initialize at inboard midplane (theta_center = 0.5) and mirrored over plane of symmetry
    theta_center = args.theta_center
    phi_center = args.phi_center
    theta_width = args.theta_width
    phi_width = args.phi_width

    num_quadpoints = args.num_quadpoints  # number of quadrature points for coils
    order = args.order  # number of Fourier modes for coils

    R0 = args.major_radius  # major radius
    s = args.toroidal_flux  # VMEC flux-surface label

    new_surf = initSurface(R0, s, file_loc, nphi, ntheta)
    banana_surf_radius = args.banana_surf_radius
    hbt, surf_coils, VV = build_hbt_reference_surfaces(
        new_surf.nfp,
        banana_surf_radius,
    )
    plasma_vessel_min_dist = float(
        np.min(
            np.linalg.norm(
                new_surf.gamma().reshape((-1, 1, 3)) - VV.gamma().reshape((1, -1, 3)),
                axis=2,
            )
        )
    )
    if plasma_vessel_min_dist < PLASMA_VESSEL_MIN_DIST_M:
        raise ValueError(
            "Fixed Stage 2 plasma surface violates the plasma-vessel clearance contract: "
            f"{plasma_vessel_min_dist:.6f} m < {PLASMA_VESSEL_MIN_DIST_M:.6f} m."
        )

    if args.stage2_bs_path:
        print(f"Loading Stage 2 seed from {args.stage2_bs_path}")
        init_coil_array = load_stage2_seed_configuration(
            args.stage2_bs_path,
            new_surf,
            len(tf_coils),
            OUT_DIR,
        )
        new_bs = init_coil_array[0]
        new_curves = init_coil_array[1]
        new_banana_curve = init_coil_array[2]
        new_banana_coils = init_coil_array[3]
        new_tf_coils = init_coil_array[4]
        tf_current_A = validate_tf_current_limit(
            float(new_tf_coils[0].current.get_value())
        )
    else:
        init_coil_array = initializeCoils(
            new_surf,
            surf_coils,
            tf_coils,
            num_quadpoints,
            order,
            args.banana_init_current_A,
            phi_center,
            theta_center,
            phi_width,
            theta_width,
            OUT_DIR,
        )
        new_bs = init_coil_array[0]
        new_curves = init_coil_array[1]
        new_banana_curve = init_coil_array[2]
        new_banana_coils = init_coil_array[3]
        new_tf_coils = tf_coils
    new_surf_coils = surf_coils
    initial_banana_current_A = float(new_banana_coils[0].current.get_value())

    # MAIN OPTIMIZATION
    # ---------------------------------------------------------------------------------------
    # Number of iterations to perform:
    MAXITER = args.maxiter
    # boolean for determining whether coil self-intersects
    intersecting = False

    LENGTH_WEIGHT = args.length_weight
    LENGTH_TARGET = min(args.length_target, COIL_LENGTH_TARGET_M)

    CC_THRESHOLD = max(args.cc_threshold, COIL_COIL_MIN_DIST_M)
    CC_WEIGHT = args.cc_weight
    CS_THRESHOLD = COIL_PLASMA_MIN_DIST_M

    CURVATURE_WEIGHT = args.curvature_weight
    CURVATURE_THRESHOLD = args.curvature_threshold
    SQUARED_FLUX_WEIGHT = args.squared_flux_weight
    CONSTRAINT_METHOD = args.constraint_method

    if CONSTRAINT_METHOD == "alm":
        outer_contract = None
        use_target_objective_lane = False
        needs_target_probe_payload = False
        probe_only_target_payload = False
        alm_inner_optimizer_contract = (
            resolve_stage2_optimizer_contract(
                args.backend,
                args.optimizer_backend,
            )
            if args.backend == "jax" and args.optimizer_backend == "ondevice"
            else None
        )
    else:
        (
            outer_contract,
            use_target_objective_lane,
            needs_target_probe_payload,
            probe_only_target_payload,
        ) = resolve_stage2_target_lane_requirements(
            args.backend,
            args.optimizer_backend,
            least_squares_algorithm=args.least_squares_algorithm,
            probe_only=args.probe_only,
            export_objective_json=args.export_objective_json,
        )
        alm_inner_optimizer_contract = None

    target_objective_bundle = None
    needs_target_objective_bundle = (
        use_target_objective_lane
        or needs_target_probe_payload
        or alm_inner_optimizer_contract is not None
    )
    if needs_target_objective_bundle:
        target_objective_bundle = build_stage2_target_objective(
            surface=new_surf,
            tf_coils=new_tf_coils,
            banana_coils=new_banana_coils,
            banana_curve=new_banana_curve,
            penalty_config=Stage2PenaltyConfig(
                squared_flux_weight=SQUARED_FLUX_WEIGHT,
                length_weight=LENGTH_WEIGHT,
                length_target=LENGTH_TARGET,
                cc_weight=CC_WEIGHT,
                cc_threshold=CC_THRESHOLD,
                curvature_weight=CURVATURE_WEIGHT,
                curvature_threshold=CURVATURE_THRESHOLD,
                curvature_p_norm=args.curvature_p_norm,
            ),
        )

    # Define the individual terms objective function:
    new_bs_jax = None
    if args.backend == "jax":
        from simsopt.field import BiotSavartJAX
        from simsopt.objectives import SquaredFluxJAX

        all_coils = list(new_tf_coils) + list(new_banana_coils)
        new_bs_jax = BiotSavartJAX(all_coils)
        Jf = SquaredFluxJAX(new_surf, new_bs_jax)  # JAX forward + autodiff gradient
        print("Stage 2 backend: JAX")
    else:
        Jf = SquaredFlux(new_surf, new_bs)  # penalty on B dot n
        print("Stage 2 backend: CPU (simsoptpp)")
    Jls = CurveLength(new_banana_curve)  # penalty on curve length
    Jccdist = CurveCurveDistance(
        new_curves, CC_THRESHOLD
    )  # penalty on coil-to-coil distance
    Jcsdist = CurveSurfaceDistance(new_curves, new_surf, CS_THRESHOLD)

    # Lp-norm curvature penalty (configurable via --curvature-p-norm)
    Jc = LpCurveCurvature(
        new_banana_curve,
        args.curvature_p_norm,
        CURVATURE_THRESHOLD,
    )
    print(f"Initial coil length: {host_float(Jls.J()):.2f} [m]")
    Jls_penalty = QuadraticPenalty(Jls, LENGTH_TARGET, "max")

    # TOTAL OBJECTIVE FUNCTION -
    # we'll penalize the coil length, coil-coil distance, and curvature while minimizing the normal field
    BASE_OBJECTIVE = SQUARED_FLUX_WEIGHT * Jf
    JF = (
        BASE_OBJECTIVE
        + LENGTH_WEIGHT * Jls_penalty
        + CC_WEIGHT * Jccdist
        + CC_WEIGHT * Jcsdist
        + CURVATURE_WEIGHT * Jc
    )

    OUT_DIR_ITER = (
        f"{OUT_DIR}R0={R0:g}-s={s:g}-LW={LENGTH_WEIGHT:g}-CCW={CC_WEIGHT:g}"
        f"-CCT={CC_THRESHOLD:g}-CW={CURVATURE_WEIGHT:g}-CT={CURVATURE_THRESHOLD:g}"
        f"-SR={banana_surf_radius:0.3f}-INITC={initial_banana_current_A:g}"
        f"-MAXC={args.banana_current_max_A:g}-TFC={tf_current_A:g}"
        f"-Order={order}-NQ={num_quadpoints}"
        f"-CP={args.curvature_p_norm}-SFW={SQUARED_FLUX_WEIGHT:g}"
        f"-backend={args.backend}-cm={CONSTRAINT_METHOD}"
        f"{'' if CONSTRAINT_METHOD != 'alm' else f'-AO={args.alm_max_outer_iters}-API={args.alm_penalty_init:g}-APS={args.alm_penalty_scale:g}'}"
        "/"
    )
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # Penalty mode optimizes the weighted composite objective directly; ALM keeps
    # the outer loop host-driven but shares the same root DOF state.
    dofs = BASE_OBJECTIVE.x if CONSTRAINT_METHOD == "alm" else JF.x
    if args.override_dofs_json is not None:
        dofs = load_stage2_override_dofs(
            args.override_dofs_json, np.asarray(dofs).shape
        )
        JF.x = np.asarray(dofs, dtype=float)
        BASE_OBJECTIVE.x = np.asarray(dofs, dtype=float)
    if CONSTRAINT_METHOD != "alm":
        apply_penalty_traversal_forbidden_box_bounds(
            bound_targets={"banana_current": new_banana_coils[0].current},
            requested_thresholds={"banana_current": args.banana_current_max_A},
            seed_values={"banana_current": initial_banana_current_A},
            validate_seed=bool(args.stage2_bs_path),
            seed_context="Loaded Stage 2 seed",
        )
    if target_objective_bundle is not None:
        validate_stage2_target_objective_dof_layout(
            target_objective_bundle,
            dofs,
        )
    trajectory: list[dict[str, object]] | None = [] if args.trajectory_json else None
    final_snapshot = None
    optimizer_timings = None
    alm_result = None
    best_feasible_partial = None
    accepted_iteration_count = 0
    restored_best_feasible_partial = False
    selected_result_x = None
    final_artifact_state = None
    if args.record_warm_timings and not use_target_objective_lane:
        raise ValueError(
            "--record-warm-timings is only supported on the JAX Stage 2 ondevice lane."
        )
    if args.profile_step_json is not None and use_target_objective_lane:
        raise ValueError(
            "--profile-step-json is only supported on explicit Stage 2 reference lanes."
        )
    if args.record_warm_timings and (args.probe_only or args.init_only):
        raise ValueError(
            "--record-warm-timings requires an actual Stage 2 optimization run and "
            "cannot be combined with --probe-only or --init-only."
        )
    field_diagnostic_stride = resolve_stage2_field_diagnostic_stride(args)
    fun = None
    if not use_target_objective_lane:
        fun = make_fun(
            JF,
            new_bs,
            new_surf,
            Jf,
            Jls,
            Jccdist,
            Jc,
            SQUARED_FLUX_WEIGHT,
            LENGTH_WEIGHT,
            LENGTH_TARGET,
            CC_WEIGHT,
            CC_THRESHOLD,
            CURVATURE_WEIGHT,
            bs_jax=new_bs_jax,
            trajectory_sink=trajectory,
            field_diagnostic_stride=field_diagnostic_stride,
        )

    def capture_artifact_state(candidate_x):
        return _capture_stage2_artifact_state(
            dofs=candidate_x,
            JF=JF,
            BASE_OBJECTIVE=BASE_OBJECTIVE,
            Jf=Jf,
            Jls=Jls,
            Jccdist=Jccdist,
            Jcsdist=Jcsdist,
            new_banana_curve=new_banana_curve,
            new_banana_coils=new_banana_coils,
            new_tf_coils=new_tf_coils,
            length_target=LENGTH_TARGET,
            cc_threshold=CC_THRESHOLD,
            curvature_threshold=CURVATURE_THRESHOLD,
            coil_surface_threshold=CS_THRESHOLD,
            plasma_vessel_min_dist=plasma_vessel_min_dist,
            plasma_vessel_threshold=PLASMA_VESSEL_MIN_DIST_M,
            banana_current_max_A=float(args.banana_current_max_A),
        )

    alm_settings = None
    alm_constraint_names = None
    alm_taylor_result = None
    evaluate_problem = None
    if CONSTRAINT_METHOD == "alm":
        alm_settings = _build_stage2_alm_settings_impl(args)
        alm_constraint_names = stage2_alm_constraint_names(
            include_coil_surface=Jcsdist is not None,
        )
        alm_field_diagnostic_state = {
            "eval_count": 0,
            "diagnostics": None,
        }

        def evaluate_problem(inner_dofs, multipliers, penalty):
            next_eval, diagnostics, recompute_diagnostics = (
                plan_stage2_field_diagnostic_evaluation(
                    alm_field_diagnostic_state,
                    stride=field_diagnostic_stride,
                )
            )
            evaluation = _evaluate_stage2_alm_problem_impl(
                inner_dofs,
                BASE_OBJECTIVE,
                new_bs,
                new_surf,
                Jf,
                Jls,
                LENGTH_TARGET,
                Jccdist,
                Jc,
                new_banana_coils[0].current,
                args.banana_current_max_A,
                args.alm_distance_smoothing,
                args.alm_curvature_smoothing,
                multipliers,
                penalty,
                _stage2_constraint_activity_tolerances_impl,
                smooth_min_distance_signed_constraint,
                smooth_max_curvature_signed_constraint,
                Jcsdist=Jcsdist,
                smooth_min_curve_surface_signed_constraint=(
                    _smooth_min_curve_surface_signed_constraint_impl
                ),
                diagnostics=diagnostics,
                recompute_diagnostics=recompute_diagnostics,
            )
            store_stage2_field_diagnostics(
                alm_field_diagnostic_state,
                eval_index=next_eval,
                diagnostics=evaluation.get(
                    "field_diagnostics",
                    diagnostics,
                ),
            )
            return evaluation

        if args.alm_taylor_test:
            alm_taylor_result = run_directional_taylor_test(
                evaluate_problem,
                dofs,
                np.zeros(len(alm_constraint_names), dtype=float),
                alm_settings.penalty_init,
                seed=args.alm_taylor_test_seed,
            )
            _log_taylor_test_summary("ALM Taylor", alm_taylor_result)
    if args.export_objective_json:
        initial_self_intersection_summary = build_curve_self_intersection_summary(
            new_banana_curve
        )
        probe_payload = build_stage2_probe_payload(
            JF,
            new_bs,
            new_surf,
            new_banana_curve,
            Jf,
            Jls,
            Jccdist,
            Jc,
            bs_jax=new_bs_jax,
            backend=args.backend,
            optimizer_backend=args.optimizer_backend,
            equilibrium_path=file_loc,
            nphi=nphi,
            ntheta=ntheta,
            squared_flux_weight=SQUARED_FLUX_WEIGHT,
            length_weight=LENGTH_WEIGHT,
            length_target=LENGTH_TARGET,
            cc_weight=CC_WEIGHT,
            cc_threshold=CC_THRESHOLD,
            curvature_weight=CURVATURE_WEIGHT,
            self_intersection_summary=initial_self_intersection_summary,
            target_objective_bundle=(
                target_objective_bundle
                if args.optimizer_backend == "ondevice"
                else None
            ),
        )
        write_json_file(args.export_objective_json, probe_payload)
        print(f"Wrote Stage 2 objective snapshot to {args.export_objective_json}")
    if args.profile_step_json is not None:
        profile_context = make_stage2_objective_context(
            JF,
            new_bs,
            new_surf,
            Jf,
            Jls,
            Jccdist,
            Jc,
            SQUARED_FLUX_WEIGHT,
            LENGTH_WEIGHT,
            LENGTH_TARGET,
            CC_WEIGHT,
            CC_THRESHOLD,
            CURVATURE_WEIGHT,
            bs_jax=new_bs_jax,
        )
        profile_payload = profile_stage2_explicit_step(profile_context)
        write_json_file(args.profile_step_json, profile_payload)
        print(f"Wrote Stage 2 step profile to {args.profile_step_json}")
    if args.probe_only:
        if args.trajectory_json:
            write_json_file(
                args.trajectory_json,
                {"backend": args.backend, "evaluations": trajectory or []},
            )
        print(
            "Probe-only mode requested; exiting before optimization and post-processing."
        )
        sys.exit(0)
    if args.init_only:
        res_nit = 0
        optimizer_success = True
        termination_message = "init_only"
        print("Skipping Stage 2 optimizer because --init-only was provided.")
    elif CONSTRAINT_METHOD == "alm":
        accepted_state = {
            "count": 0,
            "best_feasible_partial": None,
            "best_exact_hardware_pass": None,
        }
        initial_artifact_source = (
            "override_dofs"
            if args.override_dofs_json is not None
            else ("loaded_seed" if args.stage2_bs_path else "initial_state")
        )

        def maybe_record_exact_hardware_pass(candidate_x, *, source):
            candidate_state = capture_artifact_state(candidate_x)
            candidate = capture_stage2_exact_hardware_pass_candidate(
                candidate_state,
                source=source,
            )
            if candidate is None:
                return
            accepted_state["best_exact_hardware_pass"] = (
                select_better_stage2_exact_hardware_pass(
                    accepted_state["best_exact_hardware_pass"],
                    candidate,
                )
            )
            if accepted_state["best_exact_hardware_pass"] is candidate:
                print(
                    "[ALM] exact hardware-pass incumbent "
                    f"source={candidate.source}, "
                    f"field_objective={candidate.field_objective:.6e}, "
                    f"coil_length={candidate_state['coil_length']:.6f}"
                )

        def accepted_callback(current_dofs):
            flat_dofs = flatten_stage2_target_optimizer_state(current_dofs)
            accepted_state["count"] += 1
            JF.x = np.asarray(flat_dofs, dtype=float)
            BASE_OBJECTIVE.x = np.asarray(flat_dofs, dtype=float)
            candidate, _ = capture_stage2_feasible_partial_candidate(
                JF,
                Jls,
                Jccdist,
                new_banana_curve,
                LENGTH_TARGET,
                CC_THRESHOLD,
                CURVATURE_THRESHOLD,
                Jcsdist=Jcsdist,
                coil_surface_threshold=CS_THRESHOLD,
                plasma_vessel_min_dist=plasma_vessel_min_dist,
                plasma_vessel_threshold=PLASMA_VESSEL_MIN_DIST_M,
                banana_current_A=float(new_banana_coils[0].current.get_value()),
                banana_current_threshold=args.banana_current_max_A,
                tf_current_A=tf_current_A,
                tf_current_threshold=TF_CURRENT_HARD_LIMIT_A,
                accepted_index=accepted_state["count"],
            )
            if candidate is not None:
                accepted_state["best_feasible_partial"] = (
                    select_better_stage2_feasible_partial(
                        accepted_state["best_feasible_partial"],
                        candidate,
                    )
                )
            maybe_record_exact_hardware_pass(
                flat_dofs,
                source=f"accepted_iterate_{accepted_state['count']}",
            )

        def outer_state_callback(outer_iteration, multipliers, penalty):
            print(
                f"[ALM] outer_iteration={outer_iteration}, "
                f"multipliers={multipliers.tolist()}, penalty={penalty:.3e}"
            )

        assert alm_settings is not None
        assert alm_constraint_names is not None
        assert evaluate_problem is not None
        maybe_record_exact_hardware_pass(
            dofs,
            source=initial_artifact_source,
        )
        alm_target_value_and_grad = None
        if alm_inner_optimizer_contract is not None:
            assert target_objective_bundle is not None
            assert target_objective_bundle.alm_value_and_grad_builder is not None
            alm_target_value_and_grad = target_objective_bundle.alm_value_and_grad_builder(
                distance_smoothing=float(args.alm_distance_smoothing),
                curvature_smoothing=float(args.alm_curvature_smoothing),
                curve_surface_threshold=(
                    float(CS_THRESHOLD) if Jcsdist is not None else None
                ),
                banana_current_threshold=float(args.banana_current_max_A),
            )
        res, _ = run_stage2_alm_optimizer_timed(
            dofs,
            constraint_names=alm_constraint_names,
            settings=alm_settings,
            evaluate_problem=evaluate_problem,
            maxiter=MAXITER,
            maxcor=300,
            ftol=args.ftol,
            gtol=args.gtol,
            inner_optimizer_contract=alm_inner_optimizer_contract,
            target_inner_value_and_grad=alm_target_value_and_grad,
            accepted_callback=accepted_callback,
            outer_state_callback=outer_state_callback,
        )
        alm_result = res
        selected_result_x = _host_dofs_copy(res.x)
        JF.x = selected_result_x
        BASE_OBJECTIVE.x = selected_result_x
        res_nit = res.nit
        termination_message = str(res.message)
        optimizer_success = bool(res.success)
        print(res.message)
        best_feasible_partial = accepted_state["best_feasible_partial"]
        accepted_iteration_count = int(accepted_state["count"])
        restored_best_feasible_partial = bool(
            getattr(res, "restored_best_feasible", False)
        )
        best_exact_hardware_pass = accepted_state["best_exact_hardware_pass"]
        final_artifact_state = capture_artifact_state(selected_result_x)
        restored_result_x, optimizer_success, termination_message = (
            restore_stage2_exact_hardware_pass_for_artifact_output(
                best_exact_hardware_pass,
                final_artifact_state,
                optimizer_success=optimizer_success,
                termination_message=termination_message,
            )
        )
        if restored_result_x is not None:
            assert best_exact_hardware_pass is not None
            selected_result_x = restored_result_x
            final_artifact_state = capture_artifact_state(selected_result_x)
            JF.x = selected_result_x
            BASE_OBJECTIVE.x = selected_result_x
            final_snapshot = None
            if trajectory is not None:
                capture_stage2_trajectory_snapshot(
                    trajectory,
                    JF,
                    new_bs,
                    new_surf,
                    Jf,
                    Jls,
                    Jccdist,
                    Jc,
                    SQUARED_FLUX_WEIGHT,
                    LENGTH_WEIGHT,
                    LENGTH_TARGET,
                    CC_WEIGHT,
                    CC_THRESHOLD,
                    CURVATURE_WEIGHT,
                    bs_jax=new_bs_jax,
                )
            print(
                "[ALM] restoring best exact hardware-pass incumbent "
                f"from {best_exact_hardware_pass.source}"
            )
    else:
        initial_dofs = host_array(dofs).copy()
        optimizer_dofs = dofs
        accepted_state = {
            "count": 0,
            "best_feasible_partial": None,
        }

        def accepted_callback(current_dofs):
            flat_dofs = flatten_stage2_target_optimizer_state(current_dofs)
            accepted_state["count"] += 1
            JF.x = np.asarray(flat_dofs, dtype=float)
            candidate, _ = capture_stage2_feasible_partial_candidate(
                JF,
                Jls,
                Jccdist,
                new_banana_curve,
                LENGTH_TARGET,
                CC_THRESHOLD,
                CURVATURE_THRESHOLD,
                Jcsdist=Jcsdist,
                coil_surface_threshold=CS_THRESHOLD,
                plasma_vessel_min_dist=plasma_vessel_min_dist,
                plasma_vessel_threshold=PLASMA_VESSEL_MIN_DIST_M,
                banana_current_A=float(new_banana_coils[0].current.get_value()),
                banana_current_threshold=args.banana_current_max_A,
                tf_current_A=tf_current_A,
                tf_current_threshold=TF_CURRENT_HARD_LIMIT_A,
                accepted_index=accepted_state["count"],
            )
            if candidate is not None:
                accepted_state["best_feasible_partial"] = (
                    select_better_stage2_feasible_partial(
                        accepted_state["best_feasible_partial"],
                        candidate,
                    )
                )

        if use_target_objective_lane:
            assert target_objective_bundle is not None
            optimizer_dofs = build_stage2_target_optimizer_state(
                target_objective_bundle,
                initial_dofs,
            )
        assert outer_contract is not None
        if use_target_objective_lane:
            capture_stage2_trajectory_snapshot(
                trajectory,
                JF,
                new_bs,
                new_surf,
                Jf,
                Jls,
                Jccdist,
                Jc,
                SQUARED_FLUX_WEIGHT,
                LENGTH_WEIGHT,
                LENGTH_TARGET,
                CC_WEIGHT,
                CC_THRESHOLD,
                CURVATURE_WEIGHT,
                bs_jax=new_bs_jax,
            )
        target_value_and_grad = resolve_stage2_target_value_and_grad(
            target_objective_bundle
        )
        target_residual = resolve_stage2_target_least_squares_residual(
            target_objective_bundle
        )
        res, cold_elapsed_s = run_stage2_optimizer_timed(
            target_value_and_grad if target_value_and_grad is not None else fun,
            optimizer_dofs,
            contract=outer_contract,
            maxiter=MAXITER,
            maxcor=300,
            ftol=args.ftol,
            gtol=args.gtol,
            scalar_fun=(
                None
                if target_objective_bundle is None
                else target_objective_bundle.objective
            ),
            residual_fun=target_residual,
            callback=accepted_callback,
        )
        JF.x = flatten_stage2_target_optimizer_state(res.x)
        res_nit = res.nit
        termination_message = str(res.message)
        optimizer_success = bool(res.success)
        if use_target_objective_lane:
            assert target_objective_bundle is not None
            final_snapshot = capture_stage2_trajectory_snapshot(
                trajectory,
                JF,
                new_bs,
                new_surf,
                Jf,
                Jls,
                Jccdist,
                Jc,
                SQUARED_FLUX_WEIGHT,
                LENGTH_WEIGHT,
                LENGTH_TARGET,
                CC_WEIGHT,
                CC_THRESHOLD,
                CURVATURE_WEIGHT,
                bs_jax=new_bs_jax,
            )
            optimizer_timings = {
                "cold_run_s": float(cold_elapsed_s),
            }
            if args.record_warm_timings:
                JF.x = initial_dofs
                target_value_and_grad = resolve_stage2_target_value_and_grad(
                    target_objective_bundle
                )
                target_residual = resolve_stage2_target_least_squares_residual(
                    target_objective_bundle
                )
                from simsopt.geo.optimizer_jax import TargetOptimizerContract

                if (
                    isinstance(outer_contract, TargetOptimizerContract)
                    and outer_contract.use_least_squares_objective
                ):
                    assert target_residual is not None
                else:
                    assert target_value_and_grad is not None
                warm_optimizer_dofs = build_stage2_target_optimizer_state(
                    target_objective_bundle,
                    initial_dofs,
                )
                _, warm_elapsed_s = run_stage2_optimizer_timed(
                    target_value_and_grad,
                    warm_optimizer_dofs,
                    contract=outer_contract,
                    maxiter=MAXITER,
                    maxcor=300,
                    ftol=args.ftol,
                    gtol=args.gtol,
                    scalar_fun=target_objective_bundle.objective,
                    residual_fun=target_residual,
                )
                optimizer_timings["warm_run_s"] = float(warm_elapsed_s)
                optimizer_timings["compile_overhead_s"] = max(
                    float(cold_elapsed_s) - float(warm_elapsed_s),
                    0.0,
                )
                JF.x = flatten_stage2_target_optimizer_state(res.x)
        print(res.message)
        final_feasible_partial, _ = capture_stage2_feasible_partial_candidate(
            JF,
            Jls,
            Jccdist,
            new_banana_curve,
            LENGTH_TARGET,
            CC_THRESHOLD,
            CURVATURE_THRESHOLD,
            Jcsdist=Jcsdist,
            coil_surface_threshold=CS_THRESHOLD,
            plasma_vessel_min_dist=plasma_vessel_min_dist,
            plasma_vessel_threshold=PLASMA_VESSEL_MIN_DIST_M,
            banana_current_A=float(new_banana_coils[0].current.get_value()),
            banana_current_threshold=args.banana_current_max_A,
            tf_current_A=tf_current_A,
            tf_current_threshold=TF_CURRENT_HARD_LIMIT_A,
            accepted_index=accepted_state["count"],
        )
        restored_best_feasible_partial = should_restore_stage2_feasible_partial(
            accepted_state["best_feasible_partial"],
            final_feasible_partial,
            optimizer_success=optimizer_success,
        )
        if restored_best_feasible_partial:
            assert accepted_state["best_feasible_partial"] is not None
            JF.x = np.asarray(
                accepted_state["best_feasible_partial"].dofs,
                dtype=float,
            )
            final_snapshot = None
            if termination_message:
                termination_message = (
                    f"{termination_message}; restored_best_feasible_partial"
                )
            else:
                termination_message = "restored_best_feasible_partial"
            if trajectory is not None:
                capture_stage2_trajectory_snapshot(
                    trajectory,
                    JF,
                    new_bs,
                    new_surf,
                    Jf,
                    Jls,
                    Jccdist,
                    Jc,
                    SQUARED_FLUX_WEIGHT,
                    LENGTH_WEIGHT,
                    LENGTH_TARGET,
                    CC_WEIGHT,
                    CC_THRESHOLD,
                    CURVATURE_WEIGHT,
                    bs_jax=new_bs_jax,
                )
        best_feasible_partial = accepted_state["best_feasible_partial"]
        accepted_iteration_count = int(accepted_state["count"])
    if not args.init_only and final_artifact_state is None:
        if selected_result_x is None:
            selected_result_x = _host_dofs_copy(JF.x)
        final_artifact_state = capture_artifact_state(selected_result_x)
    if args.trajectory_json:
        write_json_file(
            args.trajectory_json,
            {"backend": args.backend, "evaluations": trajectory or []},
        )
        print(f"Wrote Stage 2 trajectory to {args.trajectory_json}")

    # POST-OPTIMIZATION PROCESSING AND OUTPUTS
    # Uses CPU new_bs intentionally: VTK, JSON save, and matplotlib all
    # require numpy arrays.  set_points() below forces fresh evaluation
    # with the optimized coil DOFs (shared coil objects).
    # ---------------------------------------------------------------------------------------
    final_self_intersection_summary = build_curve_self_intersection_summary(
        new_banana_curve
    )
    intersecting = final_self_intersection_summary["intersecting"]
    if intersecting:
        print("BANANA COIL IS SELF-INTERSECTING!")

    new_bs.set_points(new_surf.gamma().reshape((-1, 3)))
    unitn = new_surf.unitnormal()
    B_field = new_bs.B().reshape(unitn.shape)
    pointData = {
        "B_N/B": np.sum(B_field * unitn, axis=2)[:, :, None]
        / np.sqrt(np.sum(B_field**2, axis=2))[:, :, None]
    }
    if args.skip_postprocess and final_snapshot is None:
        final_context = make_stage2_objective_context(
            JF,
            new_bs,
            new_surf,
            Jf,
            Jls,
            Jccdist,
            Jc,
            SQUARED_FLUX_WEIGHT,
            LENGTH_WEIGHT,
            LENGTH_TARGET,
            CC_WEIGHT,
            CC_THRESHOLD,
            CURVATURE_WEIGHT,
            bs_jax=new_bs_jax,
        )
        final_snapshot, _, _ = evaluate_stage2_objective(
            final_context,
        )
    stage2_bs_output_path = os.path.join(OUT_DIR_ITER, "biot_savart_opt.json")
    stage2_surface_output_path = os.path.join(OUT_DIR_ITER, "surf_opt.json")
    new_bs.save(stage2_bs_output_path)
    new_surf.save(stage2_surface_output_path)
    fieldError = compute_mean_abs_relbn(new_surf, new_bs)
    if args.skip_postprocess:
        print(
            "Skipping Stage 2 post-processing artifacts because --skip-postprocess was provided."
        )
    else:
        curves_to_vtk(new_curves, OUT_DIR_ITER + "curves_opt", close=True)
        new_surf.to_vtk(OUT_DIR_ITER + "surf_opt", extra_data=pointData)
        VV.to_vtk(OUT_DIR_ITER + "VV")

        # Create toroidal cross section plot
        cross_section_plot(
            new_surf_coils,
            new_surf,
            new_banana_curve,
            OUT_DIR_ITER + "CrossSectionPlot",
            hbt,
            VV,
        )
        # Create field error plot
        fieldError = magneticFieldPlots(new_surf, new_bs, OUT_DIR_ITER)
    print(
        f"Banana Coil Current / TF Current = {new_banana_coils[0].current.get_value() / new_tf_coils[0].current.get_value():.3f}\n"
    )

    # Save the results of optimization to a separate file
    if final_snapshot is None:
        final_context = make_stage2_objective_context(
            JF,
            new_bs,
            new_surf,
            Jf,
            Jls,
            Jccdist,
            Jc,
            SQUARED_FLUX_WEIGHT,
            LENGTH_WEIGHT,
            LENGTH_TARGET,
            CC_WEIGHT,
            CC_THRESHOLD,
            CURVATURE_WEIGHT,
            bs_jax=new_bs_jax,
        )
        final_snapshot, _, _ = evaluate_stage2_objective(
            final_context,
        )
    final_curve_surface_min_dist = (
        float(final_artifact_state["curve_surface_min_dist"])
        if final_artifact_state is not None
        else float(Jcsdist.shortest_distance())
    )
    final_max_curvature = (
        float(final_artifact_state["max_curvature"])
        if final_artifact_state is not None
        else float(np.max(new_banana_curve.kappa()))
    )
    final_banana_current_A = (
        float(final_artifact_state["banana_current_A"])
        if final_artifact_state is not None
        else float(new_banana_coils[0].current.get_value())
    )
    banana_to_tf_current_ratio = float(final_banana_current_A / tf_current_A)
    hardware_status = evaluate_stage2_hardware_constraints(
        final_snapshot["curve_length"],
        LENGTH_TARGET,
        final_snapshot["coil_coil_distance"],
        CC_THRESHOLD,
        final_max_curvature,
        CURVATURE_THRESHOLD,
        curve_surface_min_dist=final_curve_surface_min_dist,
        coil_surface_threshold=CS_THRESHOLD,
        plasma_vessel_min_dist=plasma_vessel_min_dist,
        plasma_vessel_threshold=PLASMA_VESSEL_MIN_DIST_M,
        banana_current_A=final_banana_current_A,
        banana_current_threshold=args.banana_current_max_A,
        tf_current_A=tf_current_A,
        tf_current_threshold=TF_CURRENT_HARD_LIMIT_A,
        self_intersecting=intersecting,
    )
    optimizer_success, termination_message = apply_hardware_constraint_verdict(
        optimizer_success,
        termination_message,
        hardware_status,
        init_only=args.init_only,
    )
    results = {
        **build_stage2_results_envelope(
            output_root=OUT_DIR_ITER,
            plasma_surf_filename=plasma_surf_filename,
            file_loc=file_loc,
            nphi=nphi,
            ntheta=ntheta,
            num_quadpoints=num_quadpoints,
            order=order,
            field_diagnostic_stride=field_diagnostic_stride,
            R0=R0,
            s=s,
            banana_surf_radius=banana_surf_radius,
            theta_center=theta_center,
            phi_center=phi_center,
            theta_width=theta_width,
            phi_width=phi_width,
            LENGTH_WEIGHT=LENGTH_WEIGHT,
            CC_WEIGHT=CC_WEIGHT,
            CURVATURE_WEIGHT=CURVATURE_WEIGHT,
            SQUARED_FLUX_WEIGHT=SQUARED_FLUX_WEIGHT,
            LENGTH_TARGET=LENGTH_TARGET,
            CC_THRESHOLD=CC_THRESHOLD,
            CURVATURE_THRESHOLD=CURVATURE_THRESHOLD,
            args=args,
            MAXITER=MAXITER,
        ),
        **_build_stage2_results_impl(
            args=args,
            plasma_surf_filename=plasma_surf_filename,
            file_loc=file_loc,
            stage2_bs_path=args.stage2_bs_path,
            tf_current_A=tf_current_A,
            tf_current_sum_abs_A=sum(
                abs(coil.current.get_value()) for coil in new_tf_coils
            ),
            num_tf_coils=len(new_tf_coils),
            initial_banana_current_A=initial_banana_current_A,
            banana_current_A=final_banana_current_A,
            banana_to_tf_current_ratio=banana_to_tf_current_ratio,
            cc_threshold=CC_THRESHOLD,
            cc_weight=CC_WEIGHT,
            curvature_weight=CURVATURE_WEIGHT,
            curvature_threshold=CURVATURE_THRESHOLD,
            length_weight=LENGTH_WEIGHT,
            constraint_method=CONSTRAINT_METHOD,
            theta_center=theta_center,
            phi_center=phi_center,
            theta_width=theta_width,
            phi_width=phi_width,
            length_target=LENGTH_TARGET,
            major_radius=R0,
            toroidal_flux=s,
            nfp=new_surf.nfp,
            banana_surf_radius=banana_surf_radius,
            order=order,
            max_iterations=MAXITER,
            iterations=res_nit,
            termination_message=termination_message,
            optimizer_success=optimizer_success,
            basin_seed=None,
            basin_iterations=None,
            basin_minimization_failures=None,
            basin_accepted_hops=None,
            basin_rejected_hops=None,
            basin_best_objective=None,
            basin_accept_test_rejections=None,
            basin_accept_test_triggered=None,
            basin_nonfinite_rejections=None,
            basin_normalized_step_rejections=None,
            basin_completed_hops=None,
            basin_initial_objective=None,
            basin_best_hop_objective=None,
            basin_best_hop_index=None,
            basin_best_result_source=None,
            basin_objective_improvement=None,
            alm_result=alm_result,
            alm_taylor_result=alm_taylor_result,
            final_volume=float(new_surf.volume()),
            field_error=float(fieldError),
            intersecting=intersecting,
            final_max_curvature=final_max_curvature,
            final_coil_length=float(final_snapshot["curve_length"]),
            final_curve_curve_min_dist=float(final_snapshot["coil_coil_distance"]),
            hardware_status=hardware_status,
            final_curve_surface_min_dist=final_curve_surface_min_dist,
            plasma_vessel_min_dist=plasma_vessel_min_dist,
        ),
        "backend": args.backend,
        "optimizer_backend": args.optimizer_backend,
        "least_squares_algorithm": args.least_squares_algorithm,
        "field_diagnostic_stride": int(field_diagnostic_stride),
        "banana_curve_class": type(new_banana_curve).__name__,
        "OPTIMIZER_ACCEPTED_ITERATIONS": accepted_iteration_count,
        "BEST_FEASIBLE_PARTIAL_AVAILABLE": best_feasible_partial is not None,
        "BEST_FEASIBLE_PARTIAL_RESTORED": restored_best_feasible_partial,
        "BEST_FEASIBLE_PARTIAL_ACCEPTED_INDEX": (
            None
            if best_feasible_partial is None
            else int(best_feasible_partial.accepted_index)
        ),
        "BEST_FEASIBLE_PARTIAL_OBJECTIVE": (
            None
            if best_feasible_partial is None
            else float(best_feasible_partial.objective)
        ),
        "FINAL_DOFS": host_array(JF.x).tolist(),
        "FINAL_OBJECTIVE": final_snapshot["J"],
        "OBJECTIVE_J": final_snapshot["J"],
        "FINAL_SQUARED_FLUX": final_snapshot["Jf"],
        "FINAL_CURVE_LENGTH": final_snapshot["curve_length"],
        "FINAL_CC_DISTANCE": final_snapshot["coil_coil_distance"],
        "FINAL_COIL_SURFACE_DISTANCE": final_curve_surface_min_dist,
        "FINAL_MEAN_ABS_RELBN": final_snapshot["mean_abs_relBfinal_norm"],
        "FINAL_BANANA_GAMMA": np.asarray(
            new_banana_curve.gamma(), dtype=float
        ).tolist(),
        "SELF_INTERSECTION_MIN_DISTANCE": final_self_intersection_summary[
            "min_distance"
        ],
        "SELF_INTERSECTION_TOLERANCE": final_self_intersection_summary["tolerance"],
        "SELF_INTERSECTION_PENALTY": final_self_intersection_summary["penalty"],
    }
    if optimizer_timings is not None:
        results["OPTIMIZER_TIMINGS"] = optimizer_timings
    write_json_file(os.path.join(OUT_DIR_ITER, "results.json"), results)
